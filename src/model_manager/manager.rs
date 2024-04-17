use super::{
    dvb_wrapper::DvbWrapper,
    layout::{self, PushFragData, PushVertData, Writer},
    material::{self, PbrMaterial, TexMaterial},
    mesh,
    model::{JointTransforms, Model},
    pipeline::Pipeline,
};
use crate::{
    dvb::DeviceVertexBuffers,
    mesh_import::{Batch, ImportMaterial, MeshLoaded, Style},
    pbr_lights::PbrLightTrait,
    rh_error::RhError,
    texture::TextureManager,
    types::{CameraTrait, DeviceAccess, RenderFormat},
    vertex::{Format, RigidFormat, SkinnedFormat},
};
use ahash::{HashMap, HashMapExt};
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Device,
    image::sampler::{
        Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode, LOD_CLAMP_NONE,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::vertex_input::Vertex as VertexTrait, PipelineBindPoint,
        PipelineLayout,
    },
    Validated,
};

#[allow(unused_imports)]
use log::{debug, error, info, trace};

const VERTEX_BINDING: u32 = 0;

type ModelIndex = usize;
type MatrixIndex = u32;

/// Combines the pipeline with the buffers and a list of models that use them
struct Conduit {
    pipeline: Pipeline,
    dvb: DvbWrapper,
    model_indices: Vec<usize>,
}

pub struct Manager {
    models: Vec<Model>,
    conduits: Vec<Conduit>,
    materials: Vec<PbrMaterial>,
    texture_manager: Arc<TextureManager>,
    device: Arc<Device>,
    mem_allocator: Arc<StandardMemoryAllocator>,
    set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_format: RenderFormat,
    sampler: Arc<Sampler>,
    pipeline_layout: Arc<PipelineLayout>,
    writer: Writer,
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
            Arc::new(TextureManager::new(Arc::clone(&mem_allocator)));

        // A default sampler for handling albedo textures
        let sampler = Sampler::new(
            Arc::clone(&device_access.device),
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

        // Layout used by the shaders
        let pipeline_layout = {
            let set_info = layout::pipeline_create_info();
            PipelineLayout::new(
                Arc::clone(&device_access.device),
                set_info.into_pipeline_layout_create_info(Arc::clone(
                    &device_access.device,
                ))?,
            )
            .map_err(Validated::unwrap)?
        };
        let writer = Writer::new(&mem_allocator);

        // Create the ModelManager now so a member method can be used to modify
        // it before returning
        let mut manager = Self {
            models: Vec::new(),
            conduits: Vec::new(),
            materials: Vec::new(),
            texture_manager,
            device: Arc::clone(&device_access.device),
            mem_allocator,
            set_allocator: device_access.set_allocator,
            render_format,
            sampler,
            pipeline_layout,
            writer,
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
            // binding the sampler plus texture so that is shared as part of
            // the pipeline layout.
            let set_layout = layout::descriptor_set_layout(
                &manager.pipeline_layout,
                layout::SUBMESH_SET,
            )?;
            let pbr_material = material::tex_to_pbr(
                &tex_material,
                &manager.set_allocator,
                &manager.sampler,
                set_layout,
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
            Arc::clone(&self.device),
            &self.render_format,
            Arc::clone(&self.sampler),
        )
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
        // PREVIOUSLY: Look for an existing pipeline with the right
        // configuration. If none exists, create one.
        //let pp_index = self.find_create_pipeline(batch.style)?;
        // But now pipeline and DVBs are being combined into one thing so
        // there will be a pipeline per batch (but hopefully not many batches)
        let conduit_index = self.conduits.len();
        let pipeline = self.create_pipeline(batch.style)?;

        // Successful return will be a vector of the mesh indices
        let mut ret = Vec::new();

        // Create and store the shared DeviceVertexBuffer
        // FIXME Do a much better job with this
        let dvb = {
            if batch.style == Style::Skinned {
                let Format::Skinned(data) = batch.vb_inter.interleaved else {
                    error!("Mismatch between batch style and data");
                    return Err(RhError::UnsupportedFormat);
                };
                let dvb = DeviceVertexBuffers::<SkinnedFormat>::new(
                    cbb,
                    Arc::clone(&self.mem_allocator),
                    batch.vb_index.indices,
                    data,
                )?;
                dvb.into()
            } else {
                let Format::Rigid(data) = batch.vb_inter.interleaved else {
                    error!("Mismatch between batch style and data");
                    return Err(RhError::UnsupportedFormat);
                };
                let dvb = DeviceVertexBuffers::<RigidFormat>::new(
                    cbb,
                    Arc::clone(&self.mem_allocator),
                    batch.vb_index.indices,
                    data,
                )?;
                dvb.into()
            }
        };

        // Convert the meshes and store in Model Manager
        let meshes = mesh::convert_batch_meshes(&batch.meshes)?;
        let material_offset = self.materials.len();
        for mesh in meshes {
            ret.push(self.models.len());
            self.models.push(Model {
                mesh,
                conduit_index,
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
        let set_layout = layout::descriptor_set_layout(
            &self.pipeline_layout,
            layout::SUBMESH_SET,
        )?;
        for m in tex_materials {
            self.materials.push(material::tex_to_pbr(
                &m,
                &self.set_allocator,
                &pipeline.sampler,
                set_layout,
            )?);
        }

        // Store the conduit including the model indices which are conveniently
        // in the return vec
        self.conduits.push(Conduit {
            pipeline,
            dvb,
            model_indices: ret.clone(), // probably a short Vec
        });

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

    /// Temporary thing until we figure out the real API
    ///
    /// # Panics
    /// Will panic if the index is out of range
    pub fn update_joints(&mut self, index: usize, joints: JointTransforms) {
        self.models[index].joints = joints;
    }

    /// Iterates through all models. If the model should be drawn, its
    /// model-view matrix is calculated and stored. The index of that matrix is
    /// then added to a map with the `model_index` as the key. This allows a
    /// later drawing step to:
    ///
    /// 1. Check if a particular `model_index` should be drawn by checking if
    /// it is in the map.
    ///
    /// 2. If in the map, get the corresponding `matrix_index` and pass it
    /// to the vertex shader.
    ///
    /// 3. The shader can then use the `matrix_index` to get the model-view
    /// matrix from an array within a SSBO. The shader doesn't need to know
    /// about the `model_index` and the SSBO only needs enough data to cover
    /// models actually being drawn.
    ///
    /// Note that the matrices are returned in a `Vec` with index 0 containing
    /// the projection matrix. This is followed by the model-view matrices.
    /// This data can be copied directly to the SSBO. The shader will see it
    /// as two separate elements: a projection matrix, followed by an array of
    /// model-view matrices. The `model_index` values stored in the map are
    /// relative to this array, so will be correct in the shader, but off by
    /// one if trying to index the return `Vec`.
    ///
    /// Currently the determination if a model should be drawn is simply by
    /// checking its `visible` flag but some kind of culling can be added in
    /// the future.
    fn create_map(
        &mut self,
        camera: &impl CameraTrait,
    ) -> (HashMap<ModelIndex, MatrixIndex>, Vec<glm::Mat4>) {
        let mut map = HashMap::new();
        let mut matrices = vec![camera.proj_matrix()];
        let mut matrix_index: u32 = 0;
        let view_matrix = camera.view_matrix();
        for (i, model) in self.models.iter().enumerate() {
            if model.visible {
                matrices.push(view_matrix * model.matrix);
                map.insert(i, matrix_index);
                matrix_index += 1;
            }
        }
        (map, matrices)
    }

    /// Renders all the models. Provide a command buffer which has had
    /// `begin_rendering` and `set_viewport` called on it.
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan.
    pub fn draw_all<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        // Do culling and update matrices
        let (map, matrices) = self.create_map(camera);

        // Bind things that are common to the entire pass.
        // This includes a SSBO containing the projection matrix and an
        // array of model view matrices for the vertex shader and a UBO
        // with lighting information for the fragment shader.
        let desc_set = self.writer.pass_set(
            desc_set_allocator,
            &self.pipeline_layout,
            &matrices, // SSBO compatible Projection + Model-View array
            &layout::Lighting {
                ambient: lights.ambient_array(),
                lights: lights.light_array(&camera.view_matrix()),
            },
        )?;
        cbb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            Arc::clone(&self.pipeline_layout),
            layout::PASS_SET,
            desc_set,
        )
        .unwrap(); // `Box<ValidationError>`

        // Draw each conduit
        for conduit in &self.conduits {
            self.draw_conduit(conduit, cbb, desc_set_allocator, &map)?;
        }
        Ok(())
    }

    /// Draw the models in a conduit, binding as necessary and checking
    /// for visibility
    ///
    /// # Panics
    /// Will panic if `map` contains a model index which is out of range but
    /// this should have been checked somewhere.
    fn draw_conduit<T>(
        &self,
        conduit: &Conduit,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        map: &HashMap<ModelIndex, MatrixIndex>,
    ) -> Result<(), RhError> {
        // Avoid binding if there is nothing to do
        if conduit.model_indices.is_empty() {
            return Ok(());
        }

        // Bind the pipeline and device vertex buffers
        cbb.bind_pipeline_graphics(Arc::clone(&conduit.pipeline.graphics))
            .unwrap(); // `Box<ValidationError>`
        bind_dvb(&conduit.dvb, cbb);

        // Draw. Assume indices are in range because they should have been
        // checked somewhere.
        for model_index in &conduit.model_indices {
            // If this model is on the list of things to draw, do so
            if let Some(matrix_index) = map.get(model_index) {
                self.draw_impl(
                    &self.models[*model_index],
                    *matrix_index,
                    cbb,
                    desc_set_allocator,
                )?;
            }
        }
        Ok(())
    }

    /// Helper function used for drawing models
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    fn draw_impl<T>(
        &self,
        model: &Model,
        matrix_index: MatrixIndex,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
    ) -> Result<(), RhError> {
        // FIXME
        // Model matrix was moved into an array but joints have not been.
        // And they are being set even if this model doesn't have joints.
        let desc_set = self.writer.model_set(
            desc_set_allocator,
            &self.pipeline_layout,
            &layout::Joints {
                joints: model.joints.into(),
            },
        )?;

        // CommandBufferBuilder should have a render pass started
        cbb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            Arc::clone(&self.pipeline_layout),
            layout::MODEL_SET, // starting set, higher values also changed
            desc_set,
        )
        .unwrap(); // This is a Box<ValidationError>

        // All of the submeshes are in the buffers and share a model matrix
        // but they do need separate textures.
        for sub in &model.mesh.submeshes {
            // Get the material
            let material = {
                let lib_mat_id = model.material_offset + sub.material_id;
                if lib_mat_id < self.materials.len() {
                    &self.materials[lib_mat_id]
                } else {
                    &self.materials[0] // Default material
                }
            };

            // Record material commands
            cbb.bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                Arc::clone(&self.pipeline_layout),
                layout::SUBMESH_SET,
                Arc::clone(&material.texture_set),
            )
            .unwrap() // `Box<ValidationError>`
            .push_constants::<PushVertData>(
                Arc::clone(&self.pipeline_layout),
                layout::VERT_PUSH_OFFSET,
                matrix_index.into(),
            )
            .unwrap() // `Box<ValidationError>`
            .push_constants::<PushFragData>(
                Arc::clone(&self.pipeline_layout),
                layout::FRAG_PUSH_OFFSET,
                material.into(),
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

/// Private helper for binding DVBs
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
fn bind_dvb<T>(
    dvb_wrapper: &DvbWrapper,
    cbb: &mut AutoCommandBufferBuilder<T>,
) {
    // Bind the vertex buffers, whichever vertex format they contain. The
    // enum is unrolled and feed to this helper function that can take
    // generics (closures can't).
    fn bind<T, U: VertexTrait>(
        cbb: &mut AutoCommandBufferBuilder<T>,
        dvb: &DeviceVertexBuffers<U>,
    ) {
        cbb.bind_vertex_buffers(
            VERTEX_BINDING,          // probably 0
            dvb.interleaved.clone(), // struct containing Arcs
        )
        .unwrap() // `Box<ValidationError>`
        .bind_index_buffer(dvb.indices.clone()) // struct containing Arcs
        .unwrap(); // `Box<ValidationError>`
    }
    match dvb_wrapper {
        DvbWrapper::Skinned(dvb) => {
            bind(cbb, dvb);
        }
        DvbWrapper::Rigid(dvb) => {
            bind(cbb, dvb);
        } // Adding more vertex formats will requre more duplicate code here
          // but that will cause the build to fail so we will know that.
    }
}
