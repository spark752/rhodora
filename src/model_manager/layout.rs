use std::mem::size_of;
use std::{collections::BTreeMap, sync::Arc};

use nalgebra::base::SMatrix;
use vulkano::buffer::allocator::SubbufferAllocatorCreateInfo;
use vulkano::buffer::BufferUsage;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::PipelineLayout;
use vulkano::Validated;
use vulkano::{
    buffer::allocator::SubbufferAllocator,
    descriptor_set::layout::{
        DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
        DescriptorType,
    },
    pipeline::layout::{
        PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateFlags,
        PushConstantRange,
    },
    shader::ShaderStages,
};

#[allow(unused_imports)]
use log::debug;

use crate::rh_error::RhError;

use super::material::PbrMaterial;

/// Push constant data for the main rendering pass fragment shader
///
/// This data is placed after the vertex shader data. The fragment shader
/// accesses it by using the range starting at `FRAG_PUSH_OFFSET`.
///
/// Push constant values get embedded into the command buffer so they are very
/// fast but can't hold a lot of data. Vulkan implementations are required to
/// allow at least 128 bytes. Alignment rules should not be too tricky as long
/// as vec3 is avoided.
#[derive(vulkano::buffer::BufferContents, Clone, Copy, Default)]
#[repr(C)]
pub struct PushFragData {
    pub diffuse: [f32; 4], // Diffuse colour will be multiplied by texture
    pub roughness: f32,    // Material roughness, 0 to 1
    pub metalness: f32,    // Material metalness, 0 to 1
    pub ambient_mode: u32, // Used only for visualization shader
    pub specular_mode: u32, // Used only for visualization shader
    pub override_mode: u32, // Used only for visualization shader
} // size must be a multiple of 4 and less than 128, alignment must be 4

impl From<&PbrMaterial> for PushFragData {
    fn from(m: &PbrMaterial) -> Self {
        Self {
            diffuse: [m.diffuse[0], m.diffuse[1], m.diffuse[2], 0.0_f32],
            roughness: m.roughness,
            metalness: m.metalness,
            ..Default::default()
        }
    }
}

/// Push constant data for the main rendering pass vertex shader
///
/// This data is placed at the start of the constants before the fragment
/// shader data. The vertex data accesses it by using the range starting at 0.
///
/// Push constant values get embedded into the command buffer so they are very
/// fast but can't hold a lot of data. Vulkan implementations are required to
/// allow at least 128 bytes. Alignment rules should not be too tricky as long
/// as vec3 is avoided.
#[derive(vulkano::buffer::BufferContents, Clone, Copy, Default)]
#[repr(C)]
pub struct PushVertData {
    pub model_index: u32,
    pub pad: [u32; 3], // Pad out to 16 bytes
}

impl From<u32> for PushVertData {
    fn from(i: u32) -> Self {
        Self {
            model_index: i,
            ..Default::default()
        }
    }
}

// Simple compile time check to make sure things fit
const _: () =
    assert!(size_of::<PushFragData>() + size_of::<PushVertData>() <= 128);

// Will fit in a u32 since it was just checked to be <= 128
#[allow(clippy::cast_possible_truncation)]
pub const VERT_PUSH_SIZE: u32 = size_of::<PushVertData>() as u32;
pub const VERT_PUSH_OFFSET: u32 = 0;
pub const FRAG_PUSH_OFFSET: u32 = VERT_PUSH_SIZE;

pub const JOINTS_ELEMENTS: usize = 32;
pub const LIGHTS_ELEMENTS: usize = 4;

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub struct Joints {
    pub joints: [[[f32; 4]; 2]; JOINTS_ELEMENTS], // Not set from a GLM type
}

#[derive(vulkano::buffer::BufferContents, Clone, Copy)]
#[repr(C)]
pub struct Lighting {
    // Data is not currenty stored in GLM types
    //pub ambient: nalgebra::base::SVector<f32, 4>,
    //pub lights: [nalgebra::base::SVector<f32, 4>; 4],
    pub ambient: [f32; 4],
    pub lights: [[f32; 4]; LIGHTS_ELEMENTS],
}

// Use these values for layout in the shaders. For example:
//
// `layout(set = 2, binding = 0) uniform sampler2D tex;
//
// NOTE: Set values can not be changed without changing `pipeline_create_info`
// because the actual assignment is based on a vec created there.
pub const PASS_SET: u32 = 0; // Common to the entire rendering pass
pub const MATRIX_BINDING: u32 = 0;
pub const LIGHTS_BINDING: u32 = 1;

pub const MODEL_SET: u32 = 1; // Common to one model in the pass
pub const MODEL_BINDING: u32 = 0;

pub const SUBMESH_SET: u32 = 2; // Common to one submesh in the pass
pub const TEX_BINDING: u32 = 0;

/// Returns a struct for creating a descriptor set layout for things that
/// vary per submesh. Currently this is only the textures.
fn create_submesh_set_info() -> DescriptorSetLayoutCreateInfo {
    let mut tree = BTreeMap::new();
    let mut bind = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::CombinedImageSampler,
    );
    bind.stages = ShaderStages::FRAGMENT;
    tree.insert(TEX_BINDING, bind);
    DescriptorSetLayoutCreateInfo {
        bindings: tree,
        ..Default::default()
    }
}

/// Returns a struct for creating a descriptor set layout of items that
/// are constant for the entire rendering pass. The set number is determined
/// by the caller but they should make it equal to `LAYOUT_PASS_SET` which
/// should probably be 0. This allows the descriptor set to be bound for
/// the entire duration of the rendering pass.
///
/// Items in this set:
///
/// Projection matrix will be in a uniform buffer at bind point
/// `MATRIX_BINDING` (probably 0)
///
/// Lights will be in a uniform buffer at bind point `LAYOUT_LIGHTS_BINDING`
/// (probably 1)
fn create_pass_set_info() -> DescriptorSetLayoutCreateInfo {
    let mut tree = BTreeMap::new();

    // Matrices for vertex shader in a SSBO
    let mut vertex = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::StorageBuffer,
    );
    vertex.stages = ShaderStages::VERTEX;
    tree.insert(MATRIX_BINDING, vertex);

    // Lighting for fragment shader in a UBO
    let mut fragment = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::UniformBuffer,
    );
    fragment.stages = ShaderStages::FRAGMENT;
    tree.insert(LIGHTS_BINDING, fragment);

    DescriptorSetLayoutCreateInfo {
        bindings: tree,
        ..Default::default()
    }
}

/// Returns a struct for creating a descriptor set layout of model specific
/// data
fn create_model_set_info() -> DescriptorSetLayoutCreateInfo {
    let mut tree = BTreeMap::new();
    let mut bind = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::UniformBuffer,
    );
    bind.stages = ShaderStages::VERTEX;
    tree.insert(MODEL_BINDING, bind);
    DescriptorSetLayoutCreateInfo {
        bindings: tree,
        ..Default::default()
    }
}

/// Returns a vulkano `PipelineDescriptorSetLayoutCreateInfo` struct for
/// creating a pipeline descriptor set layout. This contains the buffer and
/// push constant configuration for shaders in the main rendering pass. It is
/// called when the pipeline is being created instead of relying on the shader.
/// It should not be necessary to call this function elsewhere.
///
/// It is necessary to reference the correct sets and binding points in the
/// shader to be compatible with this layout. The set points are determined
/// by the order the structure is built but some constants are created to match.
///
/// Set 0 = `PASS_SET`: Things that are common to all objects for the
/// entire rendering pass such as the projection matrix, an array of model
/// view matrices, and lighting information.
///
/// Set 1 = `MODEL_SET`: Things that change on a per model basis but are
/// constant across the model's submeshes. This is currently joints.
///
/// Set 2 = `SUBMESH_SET`: Things that change on a per submesh basis which
/// is currently textures.
///
/// Push Constants: Things that change per submesh that can be embedded into
/// the command buffer as push constants. Handled by the `PushConstantData`
/// type.
pub fn pipeline_create_info() -> PipelineDescriptorSetLayoutCreateInfo {
    #[allow(clippy::cast_possible_truncation)]
    let set_info = PipelineDescriptorSetLayoutCreateInfo {
        flags: PipelineLayoutCreateFlags::empty(),
        set_layouts: vec![
            create_pass_set_info(),
            //create_conduit_set_info(),
            create_model_set_info(),
            create_submesh_set_info(),
        ],
        push_constant_ranges: vec![
            PushConstantRange {
                stages: ShaderStages::VERTEX,
                offset: 0,
                size: VERT_PUSH_SIZE,
            },
            PushConstantRange {
                stages: ShaderStages::FRAGMENT,
                offset: VERT_PUSH_SIZE,
                size: std::mem::size_of::<PushFragData>() as u32,
            },
        ],
    };
    debug!("PipelineDescriptorSetLayoutCreateInfo={:?}", set_info);
    set_info
}

/// Gets the descriptor set layout for a particular set of the standard
/// pipeline layout.
///
/// # Errors
/// May return `RhError`
#[allow(clippy::module_name_repetitions)]
pub fn descriptor_set_layout(
    pipeline_layout: &Arc<PipelineLayout>,
    set: u32,
) -> Result<&Arc<DescriptorSetLayout>, RhError> {
    pipeline_layout
        .set_layouts()
        .get(set as usize)
        .ok_or(RhError::PipelineError)
}

/// For writing data to the UBOs/SSBOs defined by the layout
pub struct Writer {
    ubo_allocator: SubbufferAllocator,
    ssbo_allocator: SubbufferAllocator,
}

impl Writer {
    /// Creates a new `Writer`
    pub fn new(mem_allocator: &Arc<StandardMemoryAllocator>) -> Self {
        let ubo_allocator = SubbufferAllocator::new(
            Arc::clone(mem_allocator),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let ssbo_allocator = SubbufferAllocator::new(
            Arc::clone(mem_allocator),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        Self {
            ubo_allocator,
            ssbo_allocator,
        }
    }

    /// Writes to the buffers for the `PASS_SET`
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan.
    pub fn pass_set(
        &self,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        pipeline_layout: &Arc<PipelineLayout>,
        matrices: &[SMatrix<f32, 4, 4>],
        lighting: &Lighting,
    ) -> Result<Arc<PersistentDescriptorSet>, RhError> {
        let layout = descriptor_set_layout(pipeline_layout, PASS_SET)?;

        // Matrices look something like this to the shader:
        //      layout(set = 0, binding = 0) buffer Matrices {
        //          mat4 proj;
        //          mat4 model_view[];
        //      };
        // but can be combined into one slice of matrices for writing to
        // the SSBO. Therefore there is no sized type for the layout.
        let matrix_buffer = {
            let len = matrices.len() as u64;
            let buffer = self.ssbo_allocator.allocate_slice(len)?;
            buffer.write()?.copy_from_slice(matrices);
            buffer
        };

        let lights_buffer = {
            let buffer = self.ubo_allocator.allocate_sized()?;
            *buffer.write()? = *lighting;
            buffer
        };
        Ok(PersistentDescriptorSet::new(
            desc_set_allocator,
            Arc::clone(layout),
            [
                WriteDescriptorSet::buffer(MATRIX_BINDING, matrix_buffer),
                WriteDescriptorSet::buffer(LIGHTS_BINDING, lights_buffer),
            ],
            [],
        )
        .map_err(Validated::unwrap)?)
    }

    /// Writes to the buffers for the `MODEL_SET`
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan.
    pub fn model_set(
        &self,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        pipeline_layout: &Arc<PipelineLayout>,
        joints: &Joints,
    ) -> Result<Arc<PersistentDescriptorSet>, RhError> {
        let layout = descriptor_set_layout(pipeline_layout, MODEL_SET)?;
        let joints_buffer = {
            let buffer = self.ubo_allocator.allocate_sized()?;
            *buffer.write()? = *joints;
            buffer
        };
        Ok(PersistentDescriptorSet::new(
            desc_set_allocator,
            Arc::clone(layout),
            [WriteDescriptorSet::buffer(MODEL_BINDING, joints_buffer)],
            [],
        )
        .map_err(Validated::unwrap)?)
    }
}
