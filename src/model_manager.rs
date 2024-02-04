use crate::{
    dualquat::DualQuat,
    obj_format::{ObjBatch, ObjLoaded, ObjMaterial, ObjToLoad},
    pbr_pipeline::{PbrPipeline, PushConstantData, UniformM},
    rh_error::RhError,
    texture::Manager as TextureManager,
    types::{CameraTrait, DeviceAccess, Submesh, TextureView, MAX_JOINTS},
    util,
    vertex::{Buffers as VertexBuffers, Interleaved, Position},
};
use log::info;
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::PipelineBindPoint,
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

pub struct DeviceVertexBuffers {
    positions: Subbuffer<[Position]>,
    interleaved: Subbuffer<[Interleaved]>,
    indices: Subbuffer<[u16]>,
}

impl DeviceVertexBuffers {
    /// # Errors
    /// May return `RhError`
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<T>(
        _cbb: &mut AutoCommandBufferBuilder<T>,
        mem_allocator: Arc<StandardMemoryAllocator>, // Not a reference
        vb: VertexBuffers,                           // Not a reference
    ) -> Result<Self, RhError> {
        // Create commands to send the buffers to the GPU
        // FIXME Temporarily change all these lovely DeviceLocalBuffers into
        // something supported by vulkano
        let positions = Buffer::from_iter(
            &mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vb.positions,
        )?;
        let interleaved = Buffer::from_iter(
            &mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vb.interleaved,
        )?;
        let indices = Buffer::from_iter(
            &mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vb.indices,
        )?;
        Ok(Self {
            positions,
            interleaved,
            indices,
        })
    }
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

/*
fn from_mat3(m: &glm::Mat3) -> [[f32; 4]; 3] {
    [
        [m.m11, m.m21, m.m31, 0.0],
        [m.m21, m.m22, m.m32, 0.0],
        [m.m31, m.m32, m.m33, 0.0],
    ]
}
*/

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
    dvbs: Vec<DeviceVertexBuffers>,
    materials: Vec<PbrMaterial>,
    texture_manager: Arc<TextureManager>,
    mem_allocator: Arc<StandardMemoryAllocator>,
}

impl ModelManager {
    /// # Errors
    /// May return `RhError`
    pub fn new(
        texture_manager: Arc<TextureManager>,
        mem_allocator: Arc<StandardMemoryAllocator>,
    ) -> Result<Self, RhError> {
        Ok(Self {
            models: Vec::new(),
            dvbs: Vec::new(),
            materials: Vec::new(),
            texture_manager,
            mem_allocator,
        })
    }

    /// # Errors
    /// May return `RhError`
    #[must_use]
    pub const fn texture_manager(&self) -> &Arc<TextureManager> {
        &self.texture_manager
    }

    /// # Errors
    /// May return `RhError`
    pub fn load_batch<T>(
        &mut self,
        device_access: DeviceAccess<T>,
        pbr_pipeline: &PbrPipeline,
        batch: ObjBatch, // Not a reference
    ) -> Result<Vec<usize>, RhError> {
        self.load_batch_option(device_access, pbr_pipeline, batch, None)
    }

    /// # Errors
    /// May return `RhError`
    #[allow(clippy::needless_pass_by_value)]
    pub fn load_batch_option<T>(
        &mut self,
        device_access: DeviceAccess<T>, // Not a reference
        pbr_pipeline: &PbrPipeline,
        batch: ObjBatch,                      // Not a reference
        tex_option: Option<Vec<TexMaterial>>, // Not a reference
    ) -> Result<Vec<usize>, RhError> {
        // Successful return will be a vector of the mesh indices
        let mut ret = Vec::new();

        // Create and store the shared DeviceVertexBuffer
        let dvb_index = self.dvbs.len();
        let dvb = DeviceVertexBuffers::new(
            device_access.cbb,
            self.mem_allocator.clone(),
            batch.vb,
        )?;
        self.dvbs.push(dvb);

        // Convert the meshes and store in Object Manager
        let meshes = convert_batch_meshes(&batch.obj)?;
        let material_offset = self.materials.len();
        for mesh in meshes {
            ret.push(self.models.len());
            self.models.push(Model {
                mesh,
                dvb_index,
                material_offset,
                visible: false, // To prevent object from popping up at origin
                matrix: glm::Mat4::identity(),
                joints: JointTransforms::default(),
            });
        }

        // Process the materials, loading all textures if required
        let tex_materials = if let Some(tex_param) = tex_option {
            tex_param
        } else {
            self.load_batch_materials(&batch.obj, device_access.cbb)?
        };

        // Process the materials, creating descriptors
        let layout =
            util::get_layout(&pbr_pipeline.pipeline, TEX_SET as usize)?;
        for m in tex_materials {
            self.materials.push(tex_to_pbr(
                &m,
                device_access.set_allocator,
                &pbr_pipeline.sampler,
                layout,
            )?);
        }

        Ok(ret)
    }

    fn load_material<T>(
        &mut self,
        m: &ObjMaterial,
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

    /// # Errors
    /// May return `RhError`
    pub fn load_batch_materials<T>(
        &mut self,
        objects: &[ObjLoaded],
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<Vec<TexMaterial>, RhError> {
        let mut tex_materials = Vec::new();
        for obj_loaded in objects {
            for m in &obj_loaded.materials {
                tex_materials.push(self.load_material(m, cbb)?);
            }
        }
        Ok(tex_materials)
    }

    /// # Errors
    /// May return `RhError`
    pub fn load<T>(
        &mut self,
        device_access: DeviceAccess<T>,
        pbr_pipeline: &PbrPipeline,
        obj: &ObjToLoad,
    ) -> Result<usize, RhError> {
        // Convenience function to load a one .obj file batch
        let mut batch = ObjBatch::new();
        batch.load(obj)?;
        let ret = self.load_batch(device_access, pbr_pipeline, batch)?;
        Ok(ret[0])
    }

    /// # Errors
    /// May return `RhError`
    pub fn process<T>(
        &mut self,
        device_access: DeviceAccess<T>,
        pbr_pipeline: &PbrPipeline,
        obj: &ObjToLoad,
        load_result: tobj::LoadResult,
    ) -> Result<usize, RhError> {
        // Convenience function to load a one .obj file batch
        let mut batch = ObjBatch::new();
        batch.process(obj, load_result)?;
        let ret = self.load_batch(device_access, pbr_pipeline, batch)?;
        Ok(ret[0])
    }

    /// # Errors
    /// May return `RhError`
    pub fn load_gltf<T>(
        &mut self,
        device_access: DeviceAccess<T>,
        pbr_pipeline: &PbrPipeline,
        obj: &ObjToLoad,
    ) -> Result<usize, RhError> {
        // Convenience function to load a one gltf file batch
        let mut batch = ObjBatch::new();
        batch.load_gltf(obj)?;
        let ret = self.load_batch(device_access, pbr_pipeline, batch)?;
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
        pbr_pipeline: &PbrPipeline,
        camera: &impl CameraTrait,
    ) -> Result<(), RhError> {
        for index in 0..self.models.len() {
            self.draw(
                index,
                cbb,
                descriptor_set_allocator,
                pbr_pipeline,
                camera,
            )?;
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
        pbr_pipeline: &PbrPipeline,
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
            let buffer = pbr_pipeline.m_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let desc_set = PersistentDescriptorSet::new(
            // Set 1
            descriptor_set_allocator,
            util::get_layout(&pbr_pipeline.pipeline, M_SET as usize)?.clone(),
            [WriteDescriptorSet::buffer(M_BINDING, m_buffer)],
        )?;

        // CommandBufferBuilder should have a render pass started
        cbb.bind_descriptor_sets(
            PipelineBindPoint::Graphics,
            pbr_pipeline.layout().clone(),
            M_SET, // starting set, higher values also changed
            desc_set,
        );
        let dvb_index = self.models[index].dvb_index;
        let dvb = &self.dvbs[dvb_index];
        cbb.bind_vertex_buffers(
            VERTEX_BINDING,
            (dvb.positions.clone(), dvb.interleaved.clone()),
        )
        .bind_index_buffer(dvb.indices.clone());

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
                    pbr_pipeline.layout().clone(),
                    TEX_SET,
                    material.texture_set.clone(),
                )
                .push_constants(
                    pbr_pipeline.layout().clone(),
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
}

/// Convert a batch of meshes into the format needed by the Object Manager
///
/// # Errors
/// May return `RhError`
pub fn convert_batch_meshes(
    objs_loaded: &[ObjLoaded],
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
    for obj_loaded in objs_loaded {
        // Collect submeshes for this mesh
        let mut submeshes = Vec::new();
        for sub in &obj_loaded.submeshes {
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
            order: obj_loaded.order_option.as_ref().map_or_else(
                || (0..submeshes_len).collect::<Vec<_>>(),
                Clone::clone,
            ),
        });

        // Update material id offset. Normally there will be one per
        // submesh, but use the count from the material file in case it
        // is different.
        next_material_id += obj_loaded.materials.len();
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
