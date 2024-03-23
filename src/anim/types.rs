use crate::dualquat::DualQuat;
use ahash::HashMap;
use nalgebra_glm as glm;

#[derive(Clone, Debug)]
pub struct JointInfo {
    pub name: String,
    pub parent: usize,
    pub children: Vec<usize>,
    pub inv_bind: DualQuat,
    pub bind_translation: glm::Vec3,
    pub bind_rotation: glm::Quat,
}

#[derive(Clone, Debug)]
pub struct Skeleton {
    pub name: String,
    pub root: usize,
    pub joint_to_node: Vec<usize>,
    pub tree: HashMap<usize, JointInfo>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Interpolation {
    Linear,
    Step,
    CubicSpline,
}

#[derive(Clone, Debug, Default)]
pub struct Rotation {
    pub time: f32,
    pub data: glm::Quat,
}

#[derive(Clone, Debug)]
pub struct RotationChannel {
    pub interpolation: Interpolation,
    pub channel: Vec<Rotation>,
}

#[derive(Clone, Debug, Default)]
pub struct Translation {
    pub time: f32,
    pub data: glm::Vec3,
}

#[derive(Clone, Debug)]
pub struct TranslationChannel {
    pub interpolation: Interpolation,
    pub channel: Vec<Translation>,
}

#[derive(Debug)]
pub struct RawAnimation {
    pub name: String,
    pub r_channels: HashMap<usize, RotationChannel>,
    pub t_channels: HashMap<usize, TranslationChannel>,
}
