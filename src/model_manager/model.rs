use super::mesh::Mesh;
use crate::{dualquat::DualQuat, types::MAX_JOINTS};
use nalgebra_glm as glm;

pub struct Model {
    pub mesh: Mesh,
    pub conduit_index: usize,
    pub material_offset: usize,
    pub visible: bool,
    pub matrix: glm::Mat4,
    pub joints: JointTransforms,
}

#[derive(Clone, Copy, Debug)]
pub struct JointTransforms(pub [DualQuat; MAX_JOINTS]);

impl Default for JointTransforms {
    fn default() -> Self {
        Self(std::array::from_fn(|_| DualQuat::default()))
    }
}

impl From<JointTransforms> for [[[f32; 4]; 2]; MAX_JOINTS] {
    fn from(jt: JointTransforms) -> [[[f32; 4]; 2]; MAX_JOINTS] {
        std::array::from_fn(|i| jt.0[i].into())
    }
}
