use crate::dualquat::DualQuat;
use ahash::HashMap;

#[derive(Clone, Debug)]
pub struct JointInfo {
    pub name: String,
    pub parent: usize,
    pub children: Vec<usize>,
    pub inv_bind: DualQuat,
    pub bind: DualQuat,
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
pub struct Keyframe {
    pub time: f32,
    pub data: DualQuat,
}

#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub interpolation: Interpolation,
    pub data: Vec<Keyframe>,
}

#[derive(Clone, Debug)]
pub struct Animation {
    pub name: String,
    pub channels: HashMap<usize, AnimationChannel>,
}
