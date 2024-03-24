use serde::{Deserialize, Serialize};

/// Properties of the submesh
///
/// `material_id` is a relative index into the material list for this submesh.
/// It can be added to the `material_offset` of the model to get the absolute
/// index into the material list. Due to the way models are loaded in batches,
/// `material_id` does not necessarily start at 0 for each model. Multiple
/// models may have the same `material_offset` value but point to different
/// materials because their submeshes have different `material_id` values.
///
/// The default value of `material_id` is 0, which will be valid
/// even if `material_offset` is 0, because the material list contains a
/// default material in index 0.
#[derive(Copy, Clone, Default)]
pub struct Submesh {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub vertex_count: i32,
    pub material_id: usize,
}

/// Holds an imported material. This format is used when reading from a file or
/// creating manually before the texture has been loaded.
pub struct ImportMaterial {
    pub colour_filename: String, // Colour texture for diffuse
    pub diffuse: [f32; 3],       // Multiplier for diffuse
    pub roughness: f32,
    pub metalness: f32,
}

impl Default for ImportMaterial {
    fn default() -> Self {
        Self {
            colour_filename: String::new(),
            diffuse: [1.0, 1.0, 1.0],
            roughness: 0.5,
            metalness: 0.0,
        }
    }
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct ImportOptions {
    pub scale: f32,
    pub swizzle: bool,
    pub order_option: Option<Vec<usize>>,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            scale: 1.0f32,
            swizzle: true,
            order_option: None,
        }
    }
}

#[derive(Default)]
pub struct MeshLoaded {
    pub submeshes: Vec<Submesh>,
    pub materials: Vec<ImportMaterial>,
    pub order_option: Option<Vec<usize>>,
}

/// Intermediate vertex format for interleaved data
#[derive(Default)]
pub struct ImportVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
    pub joint_ids: [u8; 4],
    pub weights: [f32; 4],
}
