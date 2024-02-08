use super::mesh::Mesh;
use crate::{dualquat::DualQuat, types::MAX_JOINTS};
use nalgebra_glm as glm;

pub struct Model {
    pub mesh: Mesh,
    pub pipeline_index: usize,
    pub dvb_index: usize,
    pub material_offset: usize,
    pub visible: bool,
    pub matrix: glm::Mat4,
    pub joints: JointTransforms,
}

#[derive(Clone, Copy)]
pub struct JointTransforms([DualQuat; MAX_JOINTS]);

impl Default for JointTransforms {
    fn default() -> Self {
        Self(std::array::from_fn(|_| DualQuat::default()))
        /*  FIXME Test by stretching some joints
        Self(std::array::from_fn(|i| {
            if i == 18 {
                // Head
                DualQuat {
                    real: glm::quat(0.0, 0.0, 0.0, 1.0),
                    dual: glm::quat(0.0, 0.0, 0.06, 0.0),
                }
            } else if i == 23 {
                // Right thumb
                DualQuat {
                    real: glm::quat(0.0, 0.0, 0.0, 1.0),
                    dual: glm::quat(0.0, -0.05, -0.05, 0.0),
                }
            } else {
                DualQuat::default()
            }
        })) */
    }
}

impl From<JointTransforms> for [[[f32; 4]; 2]; MAX_JOINTS] {
    fn from(jt: JointTransforms) -> [[[f32; 4]; 2]; MAX_JOINTS] {
        std::array::from_fn(|i| jt.0[i].into())
    }
}
