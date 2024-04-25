// Standard vertex format is to have two streams:
// Position = positions only
// Interleaved = all other data, interleaved
use crate::mesh_import::{ImportVertex, Style};
use bytemuck::{Pod, Zeroable};
use vulkano::pipeline::graphics::vertex_input::Vertex;

/// Indices for all standard vertex formats
#[derive(Default)]
pub struct IndexBuffer {
    pub indices: Vec<u16>,
}

impl IndexBuffer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_index(&mut self, idx: u16) {
        self.indices.push(idx);
    }
}

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

impl From<ImportVertex> for RigidFormat {
    fn from(f: ImportVertex) -> Self {
        Self {
            position: f.position.into(),
            normal: f.normal.into(),
            tex_coord: f.tex_coord,
        }
    }
}

impl From<&ImportVertex> for RigidFormat {
    fn from(f: &ImportVertex) -> Self {
        Self {
            position: f.position.into(),
            normal: f.normal.into(),
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

impl From<ImportVertex> for SkinnedFormat {
    fn from(f: ImportVertex) -> Self {
        Self {
            position: f.position.into(),
            normal: f.normal.into(),
            tex_coord: f.tex_coord,
            joint_ids: (u32::from(f.joint_ids[0]) << 24)
                + (u32::from(f.joint_ids[1]) << 16)
                + (u32::from(f.joint_ids[2]) << 8)
                + u32::from(f.joint_ids[3]),
            weights: f.weights,
        }
    }
}

impl From<&ImportVertex> for SkinnedFormat {
    fn from(f: &ImportVertex) -> Self {
        Self {
            position: f.position.into(),
            normal: f.normal.into(),
            tex_coord: f.tex_coord,
            joint_ids: (u32::from(f.joint_ids[0]) << 24)
                + (u32::from(f.joint_ids[1]) << 16)
                + (u32::from(f.joint_ids[2]) << 8)
                + u32::from(f.joint_ids[3]),
            weights: f.weights,
        }
    }
}

pub enum Format {
    Rigid(Vec<RigidFormat>),
    Skinned(Vec<SkinnedFormat>),
}

pub struct InterBuffer {
    pub interleaved: Format,
}

impl InterBuffer {
    /// Creates a new `InterBuffer` of the desired `Style`
    #[must_use]
    pub const fn new(style: Style) -> Self {
        let interleaved = match style {
            Style::Rigid => Format::Rigid(Vec::new()),
            Style::Skinned => Format::Skinned(Vec::new()),
        };
        Self { interleaved }
    }

    /// Converts a single `ImportVertex` into the appropriate format and
    /// appends it to the buffer
    pub fn push(&mut self, f: ImportVertex) {
        match &mut self.interleaved {
            Format::Rigid(ref mut x) => x.push(f.into()),
            Format::Skinned(ref mut x) => x.push(f.into()),
        }
    }

    /// Converts a `ImportVertex` slice into the appropriate format and appends
    /// the vertices to the buffer
    pub fn append(&mut self, v: &[ImportVertex]) {
        match &mut self.interleaved {
            Format::Rigid(ref mut x) => {
                x.extend(
                    v.iter().map(<&ImportVertex as Into<RigidFormat>>::into),
                );
            }
            Format::Skinned(ref mut x) => {
                x.extend(
                    v.iter().map(<&ImportVertex as Into<SkinnedFormat>>::into),
                );
            }
        }
    }
}
