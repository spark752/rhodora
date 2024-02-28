use crate::rh_error::RhError;
use std::sync::Arc;
use vulkano::{
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    image::{sampler::Sampler, view::ImageView},
    Validated,
};

// FIXME find a better place for this
/// Binds diffuse texture to first binding slot
const DIFFUSE_TEX_BINDING: u32 = 0;

/// Material with loaded texture not yet in a descriptor set
#[allow(clippy::module_name_repetitions)]
pub struct TexMaterial {
    pub texture: Arc<ImageView>,
    pub diffuse: [f32; 3], // Multiplier for diffuse
    pub roughness: f32,
    pub metalness: f32,
}

/// Material with loaded texture in a descriptor set
#[allow(clippy::module_name_repetitions)]
pub struct PbrMaterial {
    pub texture_set: Arc<PersistentDescriptorSet>, // Descriptor set for diffuse
    pub diffuse: [f32; 3],                         // Multiplier for diffuse
    pub roughness: f32,
    pub metalness: f32,
}

/// Convert from `TexMaterial` to a `PbrMaterial` by creating a descriptor set
///
/// # Errors
/// Returns `RhError` if the descriptor set creation fails
pub fn tex_to_pbr(
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
            [],
        )
        .map_err(Validated::unwrap)?,
        diffuse: tex_material.diffuse,
        roughness: tex_material.roughness,
        metalness: tex_material.metalness,
    })
}
