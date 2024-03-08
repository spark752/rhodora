use super::{
    dvb_wrapper::DvbWrapper,
    layout,
    layout::{
        PushConstantData, LAYOUT_MODEL_BINDING, LAYOUT_MODEL_SET,
        LAYOUT_TEX_SET,
    },
    material,
    material::{PbrMaterial, TexMaterial},
    mesh,
    model::{JointTransforms, Model},
    pipeline::{Pipeline, UniformM},
};
use crate::{
    dvb::DeviceVertexBuffers,
    mesh_import::{Batch, ImportMaterial, MeshLoaded, Style},
    pbr_lights::PbrLightTrait,
    rh_error::RhError,
    texture::TextureManager,
    types::{CameraTrait, DeviceAccess, RenderFormat},
    util,
    vertex::{Format, RigidFormat, SkinnedFormat},
};
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::Device,
    image::sampler::{
        Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode, LOD_CLAMP_NONE,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::vertex_input::Vertex as VertexTrait, PipelineBindPoint,
    },
    Validated,
};

#[allow(unused_imports)]
use log::{debug, error, info, trace};

const VERTEX_BINDING: u32 = 0;

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
    sampler: Arc<Sampler>,
}

impl Manager {
    /// Creates a `ModelManager`
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    pub fn new<T>(
        device_access: DeviceAccess<T>,
        mem_allocator: Arc<StandardMemoryAllocator>,
        render_format: RenderFormat,
    ) -> Result<Self, RhError> {
        // TextureManager lets us share textures to avoid memory waste.
        // Why an `Arc` instead of direct ownership? An application can
        // create a multithreaded loading routine that gets this and
        // uses it with fewer lifetime complications.
        let texture_manager =
            Arc::new(TextureManager::new(mem_allocator.clone()));

        // A default sampler for handling albedo textures
        let sampler = Sampler::new(
            device_access.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;

        // Create the ModelManager now so we can use a member method to modify
        // it before returning
        let mut manager = Self {
            models: Vec::new(),
            pipelines: Vec::new(),
            dvbs: Vec::new(),
            materials: Vec::new(),
            texture_manager,
            device: device_access.device.clone(),
            mem_allocator,
            set_allocator: device_access.set_allocator,
            render_format,
            sampler: sampler.clone(),
        };

        // Create a default material in index 0 to always have available.
        // Material flow is complicated:
        // `ImportMaterial` is a material without a texture loaded.
        // `TexMaterial` has a texture loaded but is not in a descriptor set.
        // `PbrMaterial` is in a descriptor set and is what we need.
        // This requires loading a default texture which requires writing
        // commands to a buffer and then having somebody execute it.
        // Fortunately the system is already doing that for postprocess
        // initialization so we can put commands in there too.
        let material = {
            let import_material = ImportMaterial::default();
            let tex_material =
                manager.load_material(&import_material, device_access.cbb)?;

            // Creating the `PbrMaterial` needs a `Sampler` so one was added to
            // `Self`. Sharing is good. It also needs a `DescriptorSetLayout`
            // binding the sampler plus texture. It would be nice to share that
            // too eventually, but for now it is only here. As long as the
            // created pipeline is compatible with it that should be ok.
            let set_layout =
                layout::create_tex_set_layout(device_access.device)?;
            let pbr_material = material::tex_to_pbr(
                &tex_material,
                &manager.set_allocator,
                &sampler,
                &set_layout,
            );
            pbr_material?
        };
        manager.materials.push(material);

        Ok(manager)
    }

    /// Access the `TextureManager` owned by this `ModelManager`
    ///
    /// # Errors
    /// May return `RhError`
    #[must_use]
    pub const fn texture_manager(&self) -> &Arc<TextureManager> {
        &self.texture_manager
    }

    /// Helper function to create a compatible pipeline
    ///
    /// # Errors
    /// May return `RhError`
    fn create_pipeline(&self, style: Style) -> Result<Pipeline, RhError> {
        Pipeline::new(
            style,
            self.device.clone(),
            self.mem_allocator.clone(),
            &self.render_format,
            self.sampler.clone(),
        )
    }

    /// Helper function to search for a compatible pipeline, creating one if
    /// needed, and returning the index
    ///
    /// # Errors
    /// May return `RhError`
    fn find_create_pipeline(&mut self, style: Style) -> Result<usize, RhError> {
        let pp_search = self.pipelines.iter().position(|p| p.style == style);

        // The question mark operator in the else clause keeps us from
        // using an `or_else` type method here
        if let Some(i) = pp_search {
            Ok(i)
        } else {
            // Add new pipeline and return its index
            let i = self.pipelines.len();
            let pp = self.create_pipeline(style)?;
            self.pipelines.push(pp);
            Ok(i)
        }
    }

    /// Private helper function to process the batch.
    ///
    /// the command buffer builder could be primary or secondary
    /// in theory, but secondary has probably never been tested.
    ///
    /// # Errors
    /// May return `RhError`
    fn process_batch_impl<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch,                      // Not a reference
        tex_opt: Option<Vec<TexMaterial>>, // Not a reference
    ) -> Result<Vec<usize>, RhError> {
        // Look for an existing pipeline with the right configuration. If none
        // exists, create one.
        let pp_index = self.find_create_pipeline(batch.style)?;

        // Successful return will be a vector of the mesh indices
        let mut ret = Vec::new();

        // Create and store the shared DeviceVertexBuffer
        let dvb_index = self.dvbs.len();
        // FIXME Do a much better job with this
        if batch.style == Style::Skinned {
            let Format::Skinned(data) = batch.vb_inter.interleaved else {
                error!("Mismatch between batch style and data");
                return Err(RhError::UnsupportedFormat);
            };
            let dvb = DeviceVertexBuffers::<SkinnedFormat>::new(
                cbb,
                self.mem_allocator.clone(),
                batch.vb_index.indices,
                data,
            )?;
            self.dvbs.push(dvb.into());
        } else {
            let Format::Rigid(data) = batch.vb_inter.interleaved else {
                error!("Mismatch between batch style and data");
                return Err(RhError::UnsupportedFormat);
            };
            let dvb = DeviceVertexBuffers::<RigidFormat>::new(
                cbb,
                self.mem_allocator.clone(),
                batch.vb_index.indices,
                data,
            )?;
            self.dvbs.push(dvb.into());
        };

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
            &self.pipelines[pp_index].graphics,
            LAYOUT_TEX_SET,
        )?;
        for m in tex_materials {
            self.materials.push(material::tex_to_pbr(
                &m,
                &self.set_allocator,
                &self.pipelines[pp_index].sampler,
                layout,
            )?);
        }

        Ok(ret)
    }

    /// Processes a batch of meshes and their materials. The meshes should have
    /// already been loaded from files into the `batch` using the `Batch::load`
    /// method. That will have also collected material data but does not load
    /// the texture files so that will be done here. Therefore this function
    /// may take several seconds to return. If loading the textures separately,
    /// use `process_meshes` instead.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn process_batch<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch, // Not a reference
    ) -> Result<Vec<usize>, RhError> {
        self.process_batch_impl(cbb, batch, None)
    }

    /// Processes a batch of meshes and their materials. The meshes should have
    /// already been loaded from files into the `batch` using the `Batch::load`
    /// method. Materials should be provided with texture files already loaded.
    /// This function should be faster than `process_batch` which includes
    /// loading the textures.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn process_meshes<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch,                    // Not a reference
        tex_materials: Vec<TexMaterial>, // Not a reference
    ) -> Result<Vec<usize>, RhError> {
        self.process_batch_impl(cbb, batch, Some(tex_materials))
    }

    /// Private helper for creating the material by loading the texture
    ///
    /// # Errors
    /// May return `RhError`
    fn load_material<T>(
        &mut self,
        m: &ImportMaterial,
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
        let mut bound_dvb: Option<usize> = None; // Index of bound DVBs

        // There is no sorting here but because we load models in batches,
        // a series of models will use the same DVBs and pipelines.
        for model_index in 0..self.models.len() {
            // If not visible, draw nothing. This is a user controllable flag.
            // In the future additial conditions may be added for culling.
            if !self.models[model_index].visible {
                return Ok(());
            }

            // Check if we need to bind a new pipeline for this model
            let pp_index = self.models[model_index].pipeline_index;
            let need_bind = bound_ppi.map_or(true, |i| i != pp_index);
            if need_bind {
                // Bind pipeline
                //trace!("Binding the pipeline");
                if pp_index < self.pipelines.len() {
                    self.pipelines[pp_index].start_pass(
                        cbb,
                        desc_set_allocator,
                        camera,
                        lights,
                    )?;
                    bound_ppi = Some(pp_index);
                } else {
                    error!(
                        "Model {} contains invalid pipeline_index {}",
                        model_index, pp_index
                    );
                    return Err(RhError::PipelineError);
                }
            }

            // Check if we need to bind new DVBs for this model
            let dvb_index = self.models[model_index].dvb_index;
            let need_bind = bound_dvb.map_or(true, |i| i != dvb_index);
            if need_bind {
                // Bind DVBs
                //trace!("Binding DVBs");
                if dvb_index < self.dvbs.len() {
                    self.bind_dvbs(dvb_index, cbb);
                    bound_dvb = Some(dvb_index);
                } else {
                    error!(
                        "Model {} contains invalid dvb_index {}",
                        model_index, dvb_index
                    );
                    return Err(RhError::RenderPassError);
                }
            }

            // Draw
            //trace!("Drawing model_index={model_index}");
            self.draw_impl(
                model_index,
                pp_index,
                cbb,
                desc_set_allocator,
                camera,
            )?;
        }
        Ok(())
    }

    /// Helper function for binding dvbs
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    fn bind_dvbs<T>(
        &self,
        dvb_index: usize, // Caller checks if it is valid
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) {
        // Bind the vertex buffers, whichever vertex format they contain. The
        // enum is unrolled and feed to this helper function that can take
        // generics (closures can't).
        fn bind<T, U: VertexTrait>(
            cbb: &mut AutoCommandBufferBuilder<T>,
            dvb: &DeviceVertexBuffers<U>,
        ) {
            cbb.bind_vertex_buffers(VERTEX_BINDING, dvb.interleaved.clone())
                .unwrap() // `Box<ValidationError>`
                .bind_index_buffer(dvb.indices.clone())
                .unwrap(); // `Box<ValidationError>`
        }
        match &self.dvbs[dvb_index] {
            DvbWrapper::Skinned(dvb) => {
                bind(cbb, dvb);
            }
            DvbWrapper::Rigid(dvb) => {
                bind(cbb, dvb);
            } // Adding more vertex formats will requre more duplicate code here
              // but that will cause the build to fail so we will know that.
        }
    }

    /// Helper function used for drawing models
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    #[allow(clippy::too_many_lines)]
    fn draw_impl<T>(
        &self,
        model_index: usize,
        pp_index: usize,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
    ) -> Result<(), RhError> {
        // We know that model_index and pipeline_index are valid because
        // the caller checks them and also binds appropriate stuff.

        // Uniform buffer for model matrix should be in a descriptor set
        // numbered between pass specific stuff and submesh specific stuff since
        // binding will unbind higher numbered sets.
        let m_buffer = {
            // `model_view` is for converting positions to view space.
            // It is also used for normals because non-uniform scaling is not
            // supported.
            let mv = camera.view_matrix() * self.models[model_index].matrix;
            let data = UniformM {
                model_view: mv.into(),
                joints: self.models[model_index].joints.into(),
            };
            let buffer =
                self.pipelines[pp_index].subbuffer_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let desc_set = PersistentDescriptorSet::new(
            // Set 1
            desc_set_allocator,
            util::get_layout(
                &self.pipelines[pp_index].graphics,
                LAYOUT_MODEL_SET,
            )?
            .clone(),
            [WriteDescriptorSet::buffer(LAYOUT_MODEL_BINDING, m_buffer)],
            [],
        )
        .map_err(Validated::unwrap)?;

        // CommandBufferBuilder should have a render pass started
        cbb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pipelines[pp_index].layout().clone(),
            LAYOUT_MODEL_SET, // starting set, higher values also changed
            desc_set,
        )
        .unwrap(); // This is a Box<ValidationError>

        // All of the submeshes are in the buffers and share a model matrix
        // but they do need separate textures. They also have a rendering
        // order that should be used instead of as stored.
        for i in &self.models[model_index].mesh.order {
            // FIXME Can we do this validation at loading time and be sure
            // it is always valid?
            if *i >= self.models[model_index].mesh.submeshes.len() {
                error!(
                    "model_index={} uses submesh={} with len={}",
                    model_index,
                    *i,
                    self.models[model_index].mesh.submeshes.len()
                );
                return Err(RhError::IndexTooLarge);
            }
            let sub = self.models[model_index].mesh.submeshes[*i];

            // Get the material
            let material = {
                let lib_mat_id =
                    self.models[model_index].material_offset + sub.material_id;
                if lib_mat_id < self.materials.len() {
                    &self.materials[lib_mat_id]
                } else {
                    &self.materials[0] // Default material
                }
            };

            // Record material commands
            cbb.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipelines[pp_index].layout().clone(),
                LAYOUT_TEX_SET,
                material.texture_set.clone(),
            )
            .unwrap() // `Box<ValidationError>`
            .push_constants(
                self.pipelines[pp_index].layout().clone(),
                0, // offset (must be multiple of 4)
                push_constants(material),
            )
            .unwrap(); // `Box<ValidationError>`

            // Record drawing command
            cbb.draw_indexed(
                sub.index_count,
                1, // instance_count
                sub.first_index,
                sub.vertex_offset,
                0, // first_instance
            )
            .unwrap(); // `Box<ValidationError>`
        }
        Ok(())
    }
}

/// Private helper to assemble push constants
fn push_constants(material: &PbrMaterial) -> PushConstantData {
    PushConstantData {
        diffuse: [
            material.diffuse[0],
            material.diffuse[1],
            material.diffuse[2],
            1.0,
        ],
        roughness: material.roughness,
        metalness: material.metalness,
        // TODO: Add visualization stuff here somehow
        ..Default::default()
    }
}
