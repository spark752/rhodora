// Standard vertex format is to have two streams:
// Position = positions only
// Interleaved = all other data, interleaved
use crate::file_import::ImportVertex;
use bytemuck::{Pod, Zeroable};
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// Indices for all standard vertex formats
#[derive(Default)]
pub struct IndexBuffer {
    pub indices: Vec<u16>,
}

impl IndexBuffer {
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
pub struct RigidFormat {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub tex_coord: [f32; 2],
}

impl InterVertexTrait for RigidFormat {}

impl From<ImportVertex> for RigidFormat {
    fn from(f: ImportVertex) -> Self {
        Self {
            position: f.position,
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
    pub position: [f32; 3],
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
            position: f.position,
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
pub struct InterBuffer<T: InterVertexTrait> {
    pub interleaved: Vec<T>,
}

impl<T: InterVertexTrait> InterBuffer<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, f: &T) {
        self.interleaved.push(*f);
    }
}
