use super::mesh::Mesh;
use nalgebra_glm as glm;

pub struct Model {
    pub mesh: Mesh,
    pub conduit_index: usize,
    pub material_offset: usize,
    pub visible: bool,
    pub matrix: glm::Mat4,
    pub joint_data_offset: usize,
    pub joint_count: usize,
}
