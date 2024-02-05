use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default)]
pub struct Submesh {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub vertex_count: i32,
    pub material_id: Option<usize>,
}

#[derive(Default)]
pub struct Material {
    pub colour_filename: String, // Colour texture for diffuse
    pub diffuse: [f32; 3],       // Multiplier for diffuse
    pub roughness: f32,
    pub metalness: f32,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct FileToLoad {
    pub filename: String, // Used for load_obj but not process_obj
    pub scale: f32,
    pub swizzle: bool,
    pub order_option: Option<Vec<usize>>,
}

impl Default for FileToLoad {
    fn default() -> Self {
        Self {
            filename: String::new(),
            scale: 1.0f32,
            swizzle: true,
            order_option: None,
        }
    }
}

#[derive(Default)]
pub struct MeshLoaded {
    pub submeshes: Vec<Submesh>,
    pub materials: Vec<Material>,
    pub order_option: Option<Vec<usize>>,
}

/// Intermediate vertex format for interleaved data
#[derive(Default)]
pub struct ImportVertex {
    pub normal: [f32; 3],
    pub tex_coord: [f32; 2],
    pub joint_ids: [u8; 4],
    pub weights: [f32; 4],
}
