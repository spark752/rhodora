// Standard vertex format is to have two streams:
// Position = positions only
// Interleaved = all other data, interleaved
use bytemuck::{Pod, Zeroable};
use vulkano::pipeline::graphics::vertex_input::Vertex;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
pub struct Interleaved {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
    #[format(R32G32_SFLOAT)]
    pub tex_coord: [f32; 2],
    #[format(R32_UINT)]
    pub joint_ids: u32,
    #[format(R32G32B32A32_SFLOAT)]
    pub weights: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
pub struct Position {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}

pub struct Buffers {
    pub positions: Vec<Position>,
    pub interleaved: Vec<Interleaved>,
    pub indices: Vec<u16>,
}

impl Buffers {
    #[must_use]
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            interleaved: Vec::new(),
            indices: Vec::new(),
        }
    }
}

impl Default for Buffers {
    fn default() -> Self {
        Self::new()
    }
}
