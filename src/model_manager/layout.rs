use std::{collections::BTreeMap, sync::Arc};

use crate::rh_error::RhError;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    descriptor_set::layout::{
        DescriptorSetLayout, DescriptorSetLayoutBinding,
        DescriptorSetLayoutCreateInfo, DescriptorType,
    },
    device::Device,
    shader::ShaderStages,
    Validated,
};

#[allow(unused_imports)]
use log::debug;

/// Push constant data for the main rendering pass fragment shader
///
/// Some shaders may not need all of these items but should declare their
/// `push_constant` block the same way and just ignore unneeded values.
/// The `Default` trait can be used to ignore these in rust code.
///
/// Push constant values get embedded into the command buffer so they are very
/// fast but can't hold a lot of data. Vulkan implementations are required to
/// allow at least 128 bytes. Alignment rules are not too tricky as long as
/// vec3 is avoided.
///
/// The range is shared across everything in the command buffer, even if there
/// are multiple pipelines recorded. In this project the `egui` integration
/// also uses push constants but is recorded in a separate rendering pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct PushConstantData {
    pub diffuse: [f32; 4], // Diffuse colour will be multiplied by texture
    pub roughness: f32,    // Material roughness, 0 to 1
    pub metalness: f32,    // Material metalness, 0 to 1
    pub ambient_mode: u32, // Used only for visualization shader
    pub specular_mode: u32, // Used only for visualization shader
    pub override_mode: u32, // Used only for visualization shader
} // size must be a multiple of 4 and less than 128, alignment must be 4

// Use these values for layout in the shaders. For example:
//
// `layout(set = 2, binding = 0) uniform sampler2D tex;
pub const LAYOUT_PASS_SET: u32 = 0;
pub const LAYOUT_PROJ_BINDING: u32 = 0;
pub const LAYOUT_LIGHTS_BINDING: u32 = 1;
pub const LAYOUT_MODEL_SET: u32 = 1;
pub const LAYOUT_MODEL_BINDING: u32 = 0;
pub const LAYOUT_TEX_SET: u32 = 2;
pub const LAYOUT_TEX_BINDING: u32 = 0;

// Note on function names:
// `create_set_layout` is a module name repetition, but calling it "create_set"
// seems more confusing because it is NOT creating a descriptor set, just the
// layout for one. Which is confusing enough as it is.

/// Creates a descriptor set layout for sampling the albedo texture in the
/// fragment shader.
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
///
#[allow(clippy::module_name_repetitions)]
pub fn create_tex_set_layout(
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, RhError> {
    let mut tree = BTreeMap::new();
    let mut bind = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::CombinedImageSampler,
    );
    bind.stages = ShaderStages::FRAGMENT;
    tree.insert(LAYOUT_TEX_BINDING, bind);

    let layout = DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: tree,
            ..Default::default()
        },
    )
    .map_err(Validated::unwrap)?;

    // Debug output should be something like
    /* 0: DescriptorSetLayoutBinding { binding_flags: empty(),
                    descriptor_type: CombinedImageSampler,
                    descriptor_count: 1,
                    stages: FRAGMENT,
                    immutable_samplers: [],
                    _ne: NonExhaustive(()) }
    */
    debug!("Generated descriptor set layout with bindings:");
    debug!("{:?}", layout.bindings());

    Ok(layout)
}

/// Creates a descriptor set layout for items that are constant for the entire
/// rendering pass
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
///
#[allow(clippy::module_name_repetitions)]
#[allow(dead_code)]
pub fn create_pass_set_layout(
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, RhError> {
    let mut tree = BTreeMap::new();
    let mut bind_proj = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::UniformBuffer,
    );
    bind_proj.stages = ShaderStages::VERTEX;
    tree.insert(LAYOUT_PROJ_BINDING, bind_proj);
    let mut bind_lights = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::UniformBuffer,
    );
    bind_lights.stages = ShaderStages::FRAGMENT;
    tree.insert(LAYOUT_LIGHTS_BINDING, bind_lights);
    let layout = DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: tree,
            ..Default::default()
        },
    )
    .map_err(Validated::unwrap)?;

    // Debug output should be something like
    /* 0: DescriptorSetLayoutBinding { binding_flags: empty(),
                    descriptor_type: UniformBuffer,
                    descriptor_count: 1,
                    stages: VERTEX,
                    immutable_samplers: [],
                    _ne: NonExhaustive(()) },
        1: DescriptorSetLayoutBinding { binding_flags: empty(),
                    descriptor_type: UniformBuffer,
                    descriptor_count: 1,
                    stages: FRAGMENT,
                    immutable_samplers: [],
                    _ne: NonExhaustive(()) }
    */
    debug!("Generated descriptor set layout with bindings:");
    debug!("{:?}", layout.bindings());

    Ok(layout)
}

/// Creates a descriptor set layout for items that are model specific
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
///
#[allow(clippy::module_name_repetitions)]
#[allow(dead_code)]
pub fn create_model_set_layout(
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, RhError> {
    let mut tree = BTreeMap::new();
    let mut bind = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::UniformBuffer,
    );
    bind.stages = ShaderStages::VERTEX;
    tree.insert(LAYOUT_MODEL_BINDING, bind);

    let layout = DescriptorSetLayout::new(
        device,
        DescriptorSetLayoutCreateInfo {
            bindings: tree,
            ..Default::default()
        },
    )
    .map_err(Validated::unwrap)?;

    // Debug output should be something like
    /* 0: DescriptorSetLayoutBinding { binding_flags: empty(),
                    descriptor_type: UniformBuffer,
                    descriptor_count: 1,
                    stages: VERTEX,
                    immutable_samplers: [],
                    _ne: NonExhaustive(()) }
    */
    debug!("Generated descriptor set layout with bindings:");
    debug!("{:?}", layout.bindings());

    Ok(layout)
}
