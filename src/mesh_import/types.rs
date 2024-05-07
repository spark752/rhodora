use nalgebra_glm as glm;
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
    pub position: glm::Vec3,
    pub normal: glm::Vec3,
    pub tex_coord: [f32; 2],
    pub mask: u32,
    pub joint_ids: [u8; 4],
    pub weights: [f32; 4],
}

/// Errors specific to importing data. `RhError` has a `From` trait to
/// handle these.
#[derive(Debug)]
pub enum ImportError {
    General,
    NoTriangles,
    NoIndices,
    NoPositions,
    NoNormals,
    NoWeights,
    CountMismatch,
    SparseMesh,
    BigJointIndices,
    NoInverseBind,
    ScaledJoints(usize),
    SparseAnimation,
    NoSampler,
    Morphing,
    NoNodeInfo(usize),
    NoRootNode(usize),
    ConflictingRootNodes(usize),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::General => write!(f, "general import error"),
            Self::NoTriangles => {
                write!(f, "only triangulated meshes are supported")
            }
            Self::NoIndices => {
                write!(f, "only indexed meshes are supported")
            }
            Self::NoPositions => {
                write!(f, "vertex positions are required")
            }
            Self::NoNormals => {
                write!(f, "vertex normals are required")
            }
            Self::NoWeights => {
                write!(f, "vertex weights are required for a skinned mesh")
            }
            Self::CountMismatch => {
                write!(f, "there is a mismatch in the count of vertices")
            }
            Self::SparseMesh => {
                write!(f, "sparse mesh data is not supported")
            }
            Self::BigJointIndices => {
                write!(f, "joint indices must be 8 bits for a skinned mesh")
            }
            Self::NoInverseBind => {
                write!(
                    f,
                    "inverse bind matrices are required for a skinned mesh"
                )
            }
            Self::ScaledJoints(a) => {
                write!(f, "node {a} is an unsupported scaled joint")
            }
            Self::SparseAnimation => {
                write!(f, "sparse animation data is not supported")
            }
            Self::NoSampler => {
                write!(f, "a sampler is required for animation")
            }
            Self::Morphing => {
                write!(f, "morphing animation is not supported")
            }
            Self::NoNodeInfo(a) => write!(f, "node {a} has missing info"),
            Self::NoRootNode(a) => write!(f, "skin {a} has no root node"),
            Self::ConflictingRootNodes(a) => {
                write!(f, "skin {a} has conflicting root nodes")
            }
        }
    }
}
