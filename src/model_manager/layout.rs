use std::{collections::BTreeMap, sync::Arc};

use crate::rh_error::RhError;

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

/// Use this value for `binding` in the fragment shader. For example:
///
/// `layout(set = 2, binding = 0) uniform sampler2D tex;`
///
/// Note that the `set` value depends on how the descriptor sets are combined.
const BIND_LOCATION: u32 = 0;

/// Creates a descriptor set layout for sampling the albedo texture in the
/// fragment shader. Eventually this might do more and/or have a better name.
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn create_set(
    device: Arc<Device>,
) -> Result<Arc<DescriptorSetLayout>, RhError> {
    let mut tree = BTreeMap::new();
    let mut bind = DescriptorSetLayoutBinding::descriptor_type(
        DescriptorType::CombinedImageSampler,
    );
    bind.stages = ShaderStages::FRAGMENT;
    tree.insert(BIND_LOCATION, bind);

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
                    _ne: NonExhaustive(()) }}
    */
    debug!("Generated descriptor set layout with bindings:");
    debug!("{:?}", layout.bindings());

    Ok(layout)
}
