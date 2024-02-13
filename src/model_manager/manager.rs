use super::{
    dvb_wrapper::DvbWrapper,
    material,
    material::{PbrMaterial, TexMaterial},
    mesh,
    model::{JointTransforms, Model},
    pipeline::Pipeline,
};
use crate::{
    file_import::{Batch, DeviceVertexBuffers, Material, MeshLoaded},
    pbr_lights::PbrLightTrait,
    pbr_pipeline::{PbrPipeline, PushConstantData, UniformM},
    rh_error::RhError,
    texture::Manager as TextureManager,
    types::{CameraTrait, RenderFormat},
    util,
    vertex::InterVertexTrait,
};
use log::{error, info};
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::vertex_input::Vertex as VertexTrait, PipelineBindPoint,
    },
};

const M_SET: u32 = 1;
const TEX_SET: u32 = 2;
const VERTEX_BINDING: u32 = 0;
const M_BINDING: u32 = 0;

pub struct Manager {
    models: Vec<Model>,
    pipelines: Vec<Pipeline>,
    dvbs: Vec<DvbWrapper>,
    materials: Vec<PbrMaterial>,
    texture_manager: Arc<TextureManager>,
    device: Arc<Device>,
    mem_allocator: Arc<StandardMemoryAllocator>,
    set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_format: RenderFormat,
}

impl Manager {
    /// Creates a `ModelManager`
    ///
    /// # Errors
    /// May return `RhError`
    pub fn new(
        device: Arc<Device>,
        mem_allocator: Arc<StandardMemoryAllocator>,
        set_allocator: Arc<StandardDescriptorSetAllocator>,
        render_format: &RenderFormat,
    ) -> Result<Self, RhError> {
        let texture_manager =
            Arc::new(TextureManager::new(mem_allocator.clone()));
        Ok(Self {
            models: Vec::new(),
            pipelines: Vec::new(),
            dvbs: Vec::new(),
            materials: Vec::new(),
            texture_manager,
            device,
            mem_allocator,
            set_allocator,
            render_format: *render_format,
        })
    }

    /// Access the `TextureManager` owned by this `ModelManager`
    ///
    /// # Errors
    /// May return `RhError`
    #[must_use]
    pub const fn texture_manager(&self) -> &Arc<TextureManager> {
        &self.texture_manager
    }

    fn create_pipeline<T: VertexTrait>(&self) -> Result<PbrPipeline, RhError> {
        PbrPipeline::new::<T>(
            self.device.clone(),
            self.mem_allocator.clone(),
            &self.render_format,
        )
    }

    // Helper function generic for vertex formats provided the associated
    // `DeviceVertexBuffer` can be `.into()` the DVB enum. That should be
    // all Rhodora interleaved formats but the conditions were a bit tricky
    // to write. Then they have to propogate up to all the pub functions.
    // Additionally the command buffer builder could be primary or secondary
    // in theory, but secondary has probably never been tested.
    fn load_batch_impl<T, U>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch<U>,                   // Not a reference
        tex_opt: Option<Vec<TexMaterial>>, // Not a reference
    ) -> Result<Vec<usize>, RhError>
    where
        U: VertexTrait + InterVertexTrait,
        DvbWrapper: From<DeviceVertexBuffers<U>>,
    {
        // Look for an existing pipeline with the right configuration. If none
        // exists, create one.
        // FIXME: What does that mean? How do we do it?
        let pp_index = self.pipelines.len();
        if pp_index == 0 {
            let pp = self.create_pipeline::<U>()?;
            self.pipelines.push(Pipeline { pipeline: pp });
        }

        // Successful return will be a vector of the mesh indices
        let mut ret = Vec::new();

        // Create and store the shared DeviceVertexBuffer
        let dvb_index = self.dvbs.len();
        let dvb = DeviceVertexBuffers::<U>::new(
            cbb,
            self.mem_allocator.clone(),
            batch.vb_index,
            batch.vb_inter,
        )?;
        self.dvbs.push(dvb.into());

        // Convert the meshes and store in Model Manager
        let meshes = mesh::convert_batch_meshes(&batch.meshes)?;
        let material_offset = self.materials.len();
        for mesh in meshes {
            ret.push(self.models.len());
            self.models.push(Model {
                mesh,
                pipeline_index: pp_index,
                dvb_index,
                material_offset,
                visible: false, // To prevent model from popping up at origin
                matrix: glm::Mat4::identity(),
                joints: JointTransforms::default(),
            });
        }

        // Process the materials, loading all textures if required
        let tex_materials = {
            if let Some(tex) = tex_opt {
                tex
            } else {
                self.load_batch_materials(&batch.meshes, cbb)?
            }
        };

        // Process the materials, creating descriptors
        let layout = util::get_layout(
            &self.pipelines[pp_index].pipeline.graphics,
            TEX_SET as usize,
        )?;
        for m in tex_materials {
            self.materials.push(material::tex_to_pbr(
                &m,
                &self.set_allocator,
                &self.pipelines[pp_index].pipeline.sampler,
                layout,
            )?);
        }

        Ok(ret)
    }

    /// Load a batch of meshes and their materials
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_batch<T, U>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch<U>, // Not a reference
    ) -> Result<Vec<usize>, RhError>
    where
        U: VertexTrait + InterVertexTrait,
        DvbWrapper: From<DeviceVertexBuffers<U>>,
    {
        self.load_batch_impl(cbb, batch, None)
    }

    /// Load a batch of meshes with provided materials. Useful if you have
    /// created the materials seperately, for example by loading texture files
    /// in a separate thread.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_batch_meshes<T, U>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch<U>,                 // Not a reference
        tex_materials: Vec<TexMaterial>, // Not a reference
    ) -> Result<Vec<usize>, RhError>
    where
        U: VertexTrait + InterVertexTrait,
        DvbWrapper: From<DeviceVertexBuffers<U>>,
    {
        self.load_batch_impl(cbb, batch, Some(tex_materials))
    }

    fn load_material<T>(
        &mut self,
        m: &Material,
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<TexMaterial, RhError> {
        let filename = &m.colour_filename;
        info!("Loading {filename}");
        Ok(TexMaterial {
            texture: self.texture_manager.load(filename, cbb)?,
            diffuse: m.diffuse,
            roughness: m.roughness,
            metalness: m.metalness,
        })
    }

    /// Loads the materials associated with a batch. Called by `load_batch`.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_batch_materials<T>(
        &mut self,
        meshes: &[MeshLoaded],
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<Vec<TexMaterial>, RhError> {
        let mut tex_materials = Vec::new();
        for mesh_loaded in meshes {
            for m in &mesh_loaded.materials {
                tex_materials.push(self.load_material(m, cbb)?);
            }
        }
        Ok(tex_materials)
    }

    pub fn update(
        &mut self,
        index: usize,
        matrix: Option<glm::Mat4>,
        visible: Option<bool>,
    ) {
        if let Some(m) = matrix {
            self.models[index].matrix = m;
        }
        if let Some(v) = visible {
            self.models[index].visible = v;
        }
    }

    #[must_use]
    pub fn matrix_ref(&self, index: usize) -> &glm::Mat4 {
        &self.models[index].matrix
    }

    /// # Errors
    /// May return `RhError`
    pub fn draw<T>(
        &self,
        index: usize,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
    ) -> Result<(), RhError> {
        if !self.models[index].visible {
            return Ok(());
        }
        let pp_index = self.models[index].pipeline_index;

        // Uniform buffer for model matrix and texture is in descriptor set 1
        // since this will change for each model and updating a descriptor set
        // unbinds all of those with higher numbers
        let m_buffer = {
            // `model_view` is for converting positions to view space.
            // `norm_view` is for converting normals to view space. It could
            // be a `glm::Mat3` but std140 pads these and there is no compatible
            // `into()` method so a `glm::Mat4` is currently used insted. This
            // should be calculated from the transpose of the inverse of
            // `model_view` but unless there is non-uniform scaling this should
            // work. Of course this means there is no reason to actually send
            // it separately, so I guess FIXME.
            let mv = camera.view_matrix() * self.models[index].matrix;
            let data = UniformM {
                model_view: mv.into(),
                norm_view: mv.into(),
                joints: self.models[index].joints.into(),
            };
            let buffer =
                self.pipelines[pp_index].pipeline.m_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let desc_set = PersistentDescriptorSet::new(
            // Set 1
            desc_set_allocator,
            util::get_layout(
                &self.pipelines[pp_index].pipeline.graphics,
                M_SET as usize,
            )?
            .clone(),
            [WriteDescriptorSet::buffer(M_BINDING, m_buffer)],
        )?;

        // CommandBufferBuilder should have a render pass started
        cbb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipelines[pp_index].pipeline.layout().clone(),
            M_SET, // starting set, higher values also changed
            desc_set,
        );

        // Bind the vertex buffers, whichever vertex format they contain. The
        // enum is unrolled and feed to this helper function that can take
        // generics (closures can't).
        #[allow(clippy::items_after_statements)]
        fn bind_dvbs<T, U: VertexTrait>(
            cbb: &mut AutoCommandBufferBuilder<T>,
            dvb: &DeviceVertexBuffers<U>,
        ) {
            cbb.bind_vertex_buffers(VERTEX_BINDING, dvb.interleaved.clone())
                .bind_index_buffer(dvb.indices.clone());
        }
        let dvb_index = self.models[index].dvb_index;
        match &self.dvbs[dvb_index] {
            DvbWrapper::Skinned(dvb) => {
                bind_dvbs(cbb, dvb);
            }
            DvbWrapper::Rigid(dvb) => {
                bind_dvbs(cbb, dvb);
            } // Adding more vertex formats will requre more duplicate code here
              // but that will cause the build to fail so we will know that.
        }

        // All of the submeshes are in the buffers and share a model matrix
        // but they do need separate textures. They also have a rendering
        // order that should be used instead of as stored.
        for i in &self.models[index].mesh.order {
            let sub = self.models[index].mesh.submeshes[*i];

            // If the submesh has a material then use it
            if let Some(mat_id) = sub.material_id {
                let lib_mat_id = mat_id + self.models[index].material_offset;
                let material = &self.materials[lib_mat_id];
                let push_constants = PushConstantData {
                    diffuse: [
                        material.diffuse[0],
                        material.diffuse[1],
                        material.diffuse[2],
                        1.0,
                    ],
                    roughness: material.roughness,
                    metalness: material.metalness,
                };
                cbb.bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipelines[pp_index].pipeline.layout().clone(),
                    TEX_SET,
                    material.texture_set.clone(),
                )
                .push_constants(
                    self.pipelines[pp_index].pipeline.layout().clone(),
                    0,              // offset (must be multiple of 4)
                    push_constants, // (size must be multiple of 4)
                );
            }
            // Draw
            cbb.draw_indexed(
                sub.index_count,
                1, // instance_count
                sub.first_index,
                sub.vertex_offset,
                0, // first_instance
            )?;
        }
        Ok(())
    }

    /// Renders all the models. Provide a command buffer which has had
    /// `begin_rendering` and `set_viewport` called on it.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn draw_all<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        let mut bound_ppi: Option<usize> = None; // Index of bound pipeline

        for model_index in 0..self.models.len() {
            // Check if we need to bind a new pipeline for this model
            let ppi = self.models[model_index].pipeline_index;
            let need_bind = bound_ppi.map_or(true, |i| i != ppi);
            if need_bind {
                // Bind pipeline
                if ppi < self.pipelines.len() {
                    self.pipelines[ppi].pipeline.start_pass(
                        cbb,
                        desc_set_allocator,
                        camera,
                        lights,
                    )?;
                    bound_ppi = Some(ppi);
                } else {
                    error!(
                        "Model {} contains invalid pipeline_index {}",
                        model_index, ppi
                    );
                    return Err(RhError::PipelineError);
                }
            }

            // Draw
            self.draw(model_index, cbb, desc_set_allocator, camera)?;
        }
        Ok(())
    }
}
