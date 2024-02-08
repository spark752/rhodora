use crate::{
    dualquat::DualQuat,
    file_import::{
        Batch, DeviceVertexBuffers, FileToLoad, Material, MeshLoaded, Submesh,
    },
    pbr_lights::PbrLightTrait,
    pbr_pipeline::{PbrPipeline, PushConstantData, UniformM},
    rh_error::RhError,
    texture::Manager as TextureManager,
    types::{CameraTrait, DeviceAccess, TextureView, MAX_JOINTS},
    util,
    vertex::{InterVertexTrait, SkinnedFormat, UnskinnedFormat},
};
use log::info;
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::vertex_input::Vertex as VertexTrait, PipelineBindPoint,
    },
    sampler::Sampler,
};

const M_SET: u32 = 1;
const TEX_SET: u32 = 2;
const VERTEX_BINDING: u32 = 0;
const M_BINDING: u32 = 0;
const DIFFUSE_TEX_BINDING: u32 = 0;

// Material with loaded texture not yet in a descriptor set
pub struct TexMaterial {
    pub texture: TextureView,
    pub diffuse: [f32; 3], // Multiplier for diffuse
    pub roughness: f32,
    pub metalness: f32,
}

// Material with loaded texture in a descriptor set
struct PbrMaterial {
    texture_set: Arc<PersistentDescriptorSet>, // Descriptor set for diffuse
    diffuse: [f32; 3],                         // Multiplier for diffuse
    roughness: f32,
    metalness: f32,
}

pub struct Mesh {
    submeshes: Vec<Submesh>,
    order: Vec<usize>,
}

#[derive(Clone, Copy)]
struct JointTransforms([DualQuat; MAX_JOINTS]);

impl Default for JointTransforms {
    fn default() -> Self {
        Self(std::array::from_fn(|_| DualQuat::default()))
        /*  FIXME Test by stretching some joints
        Self(std::array::from_fn(|i| {
            if i == 18 {
                // Head
                DualQuat {
                    real: glm::quat(0.0, 0.0, 0.0, 1.0),
                    dual: glm::quat(0.0, 0.0, 0.06, 0.0),
                }
            } else if i == 23 {
                // Right thumb
                DualQuat {
                    real: glm::quat(0.0, 0.0, 0.0, 1.0),
                    dual: glm::quat(0.0, -0.05, -0.05, 0.0),
                }
            } else {
                DualQuat::default()
            }
        })) */
    }
}

impl From<JointTransforms> for [[[f32; 4]; 2]; MAX_JOINTS] {
    fn from(jt: JointTransforms) -> [[[f32; 4]; 2]; MAX_JOINTS] {
        std::array::from_fn(|i| jt.0[i].into())
    }
}

/// Enum of supported vertex formats. Perhaps not all are constructed.
#[allow(dead_code)]
enum DvbWrapper {
    Unskinned(DeviceVertexBuffers<UnskinnedFormat>),
    Skinned(DeviceVertexBuffers<SkinnedFormat>),
}

impl From<DeviceVertexBuffers<UnskinnedFormat>> for DvbWrapper {
    fn from(f: DeviceVertexBuffers<UnskinnedFormat>) -> Self {
        Self::Unskinned(f)
    }
}

impl From<DeviceVertexBuffers<SkinnedFormat>> for DvbWrapper {
    fn from(f: DeviceVertexBuffers<SkinnedFormat>) -> Self {
        Self::Skinned(f)
    }
}

struct Model {
    mesh: Mesh,
    dvb_index: usize,
    material_offset: usize,
    visible: bool,
    matrix: glm::Mat4,
    joints: JointTransforms,
}

pub struct ModelManager {
    models: Vec<Model>,
    dvbs: Vec<DvbWrapper>,
    materials: Vec<PbrMaterial>,
    texture_manager: Arc<TextureManager>,
    mem_allocator: Arc<StandardMemoryAllocator>,
    pbr_pipeline: PbrPipeline,
}

impl ModelManager {
    /// # Errors
    /// May return `RhError`
    pub fn new(
        texture_manager: Arc<TextureManager>,
        mem_allocator: Arc<StandardMemoryAllocator>,
        pbr_pipeline: PbrPipeline, // Not a reference
    ) -> Result<Self, RhError> {
        Ok(Self {
            models: Vec::new(),
            dvbs: Vec::new(),
            materials: Vec::new(),
            texture_manager,
            mem_allocator,
            pbr_pipeline,
        })
    }

    /// # Errors
    /// May return `RhError`
    #[must_use]
    pub const fn texture_manager(&self) -> &Arc<TextureManager> {
        &self.texture_manager
    }

    /// Helper function generic for vertex formats provided the associated
    /// `DeviceVertexBuffer` can be `.into()` the DVB enum. That should be
    /// all Rhodora interleaved formats but the conditions were a bit tricky
    /// to write.
    fn load_batch_impl<T, U>(
        &mut self,
        device_access: &mut DeviceAccess<T>,
        batch: Batch<U>,                   // Not a reference
        tex_opt: Option<Vec<TexMaterial>>, // Not a reference
    ) -> Result<Vec<usize>, RhError>
    where
        U: VertexTrait + InterVertexTrait,
        DvbWrapper: From<DeviceVertexBuffers<U>>,
    {
        // Successful return will be a vector of the mesh indices
        let mut ret = Vec::new();

        // Create and store the shared DeviceVertexBuffer
        let dvb_index = self.dvbs.len();
        let dvb = DeviceVertexBuffers::<U>::new(
            device_access.cbb,
            self.mem_allocator.clone(),
            batch.vb_base,
            batch.vb_inter,
        )?;
        self.dvbs.push(dvb.into());

        // Convert the meshes and store in Model Manager
        let meshes = convert_batch_meshes(&batch.meshes)?;
        let material_offset = self.materials.len();
        for mesh in meshes {
            ret.push(self.models.len());
            self.models.push(Model {
                mesh,
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
                self.load_batch_materials(&batch.meshes, device_access.cbb)?
            }
        };

        // Process the materials, creating descriptors
        let layout =
            util::get_layout(&self.pbr_pipeline.pipeline, TEX_SET as usize)?;
        for m in tex_materials {
            self.materials.push(tex_to_pbr(
                &m,
                device_access.set_allocator,
                &self.pbr_pipeline.sampler,
                layout,
            )?);
        }

        Ok(ret)
    }

    /// Load a batch of meshes and their materials
    ///
    /// # Errors
    /// May return `RhError`
    //
    // Clippy doesn't like U being restricted by the private trait:
    // "Having private types or traits in item bounds makes it less clear what
    // interface the item actually provides."
    // But the restriction is needed and `DvbWrapper` not ready to be public
    // yet.
    #[allow(private_bounds)]
    pub fn load_batch<T, U>(
        &mut self,
        device_access: &mut DeviceAccess<T>,
        batch: Batch<U>, // Not a reference
    ) -> Result<Vec<usize>, RhError>
    where
        U: VertexTrait + InterVertexTrait,
        DvbWrapper: From<DeviceVertexBuffers<U>>,
    {
        self.load_batch_impl(device_access, batch, None)
    }

    /// Load a batch of meshes with provided materials. Useful if you have
    /// created the materials seperately, for example by loading texture files
    /// in a separate thread.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_batch_meshes<T>(
        &mut self,
        device_access: &mut DeviceAccess<T>,
        batch: Batch<SkinnedFormat>, // Not a reference
        tex_materials: Vec<TexMaterial>, // Not a reference
    ) -> Result<Vec<usize>, RhError> {
        self.load_batch_impl(device_access, batch, Some(tex_materials))
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

    /// Convenience function to load a single .obj file as a batch. Creates
    /// the batch, loads a single file to it, then processes it by calling
    /// `load_batch`.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load<T>(
        &mut self,
        device_access: &mut DeviceAccess<T>,
        file: &FileToLoad,
    ) -> Result<usize, RhError> {
        // Vertex format was once hard coded in lowest levels but now has made
        // it all the way up to here. Still some more to go.
        let mut batch = Batch::<SkinnedFormat>::new();
        batch.load(file)?;
        let ret = self.load_batch(device_access, batch)?;
        Ok(ret[0])
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
    pub fn draw_all<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
    ) -> Result<(), RhError> {
        for index in 0..self.models.len() {
            self.draw(index, cbb, descriptor_set_allocator, camera)?;
        }
        Ok(())
    }

    /// # Errors
    /// May return `RhError`
    pub fn draw<T>(
        &self,
        index: usize,
        cbb: &mut AutoCommandBufferBuilder<T>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
    ) -> Result<(), RhError> {
        if !self.models[index].visible {
            return Ok(());
        }

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
            let buffer = self.pbr_pipeline.m_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let desc_set = PersistentDescriptorSet::new(
            // Set 1
            descriptor_set_allocator,
            util::get_layout(&self.pbr_pipeline.pipeline, M_SET as usize)?
                .clone(),
            [WriteDescriptorSet::buffer(M_BINDING, m_buffer)],
        )?;

        // CommandBufferBuilder should have a render pass started
        cbb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            self.pbr_pipeline.layout().clone(),
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
            cbb.bind_vertex_buffers(
                VERTEX_BINDING,
                (dvb.positions.clone(), dvb.interleaved.clone()),
            )
            .bind_index_buffer(dvb.indices.clone());
        }
        let dvb_index = self.models[index].dvb_index;
        match &self.dvbs[dvb_index] {
            DvbWrapper::Skinned(dvb) => {
                bind_dvbs(cbb, dvb);
            }
            DvbWrapper::Unskinned(dvb) => {
                bind_dvbs(cbb, dvb);
            } // Adding more vertex formats will requre more duplicate code here
              // but the build will fail to tell us that.
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
                    self.pbr_pipeline.layout().clone(),
                    TEX_SET,
                    material.texture_set.clone(),
                )
                .push_constants(
                    self.pbr_pipeline.layout().clone(),
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

    /// Start a rendering pass using the owned PBR pipeline.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn start_pass<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        self.pbr_pipeline.start_pass(
            cbb,
            descriptor_set_allocator,
            camera,
            lights,
        )
    }
}

/// Convert a batch of meshes into the format needed by the Model Manager
///
/// # Errors
/// May return `RhError`
pub fn convert_batch_meshes(
    meshes_loaded: &[MeshLoaded],
) -> Result<Vec<Mesh>, RhError> {
    let mut meshes = Vec::new();

    // Adjust submesh structures. The material_id from the .obj file is
    // relative to that .mtl, so is changed to be relative to the whole
    // batch of materials. It is NOT adjusted here for an overall material
    // library to avoid dependency.
    // The first_index and vertex_offset are changed to reflect multiple .obj
    // files being loaded into one DVB. The Mesh does NOT contain an reference
    // to the particular DVB to avoid dependency.
    let mut next_index = 0;
    let mut next_vertex = 0;
    let mut next_material_id = 0;
    for mesh_loaded in meshes_loaded {
        // Collect submeshes for this mesh
        let mut submeshes = Vec::new();
        for sub in &mesh_loaded.submeshes {
            submeshes.push(Submesh {
                index_count: sub.index_count,
                first_index: sub.first_index + next_index,
                vertex_offset: sub.vertex_offset + next_vertex,
                vertex_count: sub.vertex_count,
                material_id: sub.material_id.map(|id| id + next_material_id),
            });
        }

        // Update offsets so if there is another mesh in the buffer
        if !submeshes.is_empty() {
            let last_sub = &submeshes[submeshes.len() - 1];
            next_index = last_sub.first_index + last_sub.index_count;
            next_vertex = last_sub.vertex_offset + last_sub.vertex_count;
        }

        // Create the Mesh and add to the return vec
        let submeshes_len = submeshes.len(); // To satisfy borrow checker
        meshes.push(Mesh {
            submeshes,
            order: mesh_loaded.order_option.as_ref().map_or_else(
                || (0..submeshes_len).collect::<Vec<_>>(),
                Clone::clone,
            ),
        });

        // Update material id offset. Normally there will be one per
        // submesh, but use the count from the material file in case it
        // is different.
        next_material_id += mesh_loaded.materials.len();
    }
    Ok(meshes)
}

fn tex_to_pbr(
    tex_material: &TexMaterial,
    set_allocator: &StandardDescriptorSetAllocator,
    sampler: &Arc<Sampler>,
    layout: &Arc<DescriptorSetLayout>,
) -> Result<PbrMaterial, RhError> {
    Ok(PbrMaterial {
        texture_set: PersistentDescriptorSet::new(
            set_allocator,
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                DIFFUSE_TEX_BINDING,
                tex_material.texture.clone(),
                sampler.clone(),
            )],
        )?,
        diffuse: tex_material.diffuse,
        roughness: tex_material.roughness,
        metalness: tex_material.metalness,
    })
}
