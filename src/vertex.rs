// Standard vertex format is to have two streams:
// Position = positions only
// Interleaved = all other data, interleaved
use crate::file_import::ImportVertex;
use bytemuck::{Pod, Zeroable};
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// Vertex format for the position buffer of all meshes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

/// Holds the position and indices for all standard vertex formats
#[derive(Default)]
pub struct BaseBuffers {
    pub positions: Vec<Position>,
    pub indices: Vec<u16>,
}

impl BaseBuffers {
    pub fn push_position(&mut self, pos: &[f32; 3]) {
        self.positions.push(Position { position: *pos });
    }

    pub fn push_index(&mut self, idx: u16) {
        self.indices.push(idx);
    }
}

/// Trait required for all vertex formats used by the interleaved buffer.
/// Note this trait is in addition to the Vulkano `Vertex` trait required for
/// all vertex formats.
pub trait InterVertexTrait: Copy + Default + From<ImportVertex> {}

/// Vertex format for the interleaved buffer of unskinned meshes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
pub struct UnskinnedFormat {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub tex_coord: [f32; 2],
}

impl InterVertexTrait for UnskinnedFormat {}

impl From<ImportVertex> for UnskinnedFormat {
    fn from(f: ImportVertex) -> Self {
        Self {
            normal: f.normal,
            tex_coord: f.tex_coord,
        }
    }
}

/// Vertex format for the interleaved buffer of skinned meshes
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
pub struct SkinnedFormat {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub tex_coord: [f32; 2],
    #[format(R32_UINT)]
    pub joint_ids: u32,
    #[format(R32G32B32A32_SFLOAT)]
    pub weights: [f32; 4],
}

impl InterVertexTrait for SkinnedFormat {}

impl From<ImportVertex> for SkinnedFormat {
    fn from(f: ImportVertex) -> Self {
        Self {
            normal: f.normal,
            tex_coord: f.tex_coord,
            joint_ids: (u32::from(f.joint_ids[0]) << 24)
                + (u32::from(f.joint_ids[1]) << 16)
                + (u32::from(f.joint_ids[2]) << 8)
                + u32::from(f.joint_ids[3]),
            weights: f.weights,
        }
    }
}

/// Interleaved vertex buffer generic over different formats
#[derive(Default)]
pub struct InterBuffers<T: InterVertexTrait> {
    pub interleaved: Vec<T>,
}

impl<T: InterVertexTrait> InterBuffers<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, f: &T) {
        self.interleaved.push(*f);
    }
}