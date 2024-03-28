use super::{
    types::{Animation, Interpolation, Skeleton},
    AnimationChannel, JointInfo, Keyframe,
};
use crate::dualquat::{self, DualQuat};
use ahash::{HashMap, HashMapExt};
use log::debug;

/// Helper to calculate the parameter used for interpolation
fn weight(start: f32, end: f32, current: f32) -> f32 {
    const EPSILON: f32 = 0.0005;
    ((current - start) / (end - start).max(EPSILON)).clamp(0.0f32, 1.0f32)
}

/// Helper to calculate transforms
fn calculate(
    channel: &AnimationChannel,
    initial_frame: &Keyframe,
    current_time: f32,
) -> DualQuat {
    let mut frame = initial_frame;
    for f in &channel.data {
        if f.time < current_time {
            // This frame has a time before the current time, so
            // make it the new candidate frame. (Note that `frame`
            // and `f` are both references.)
            frame = f;
        } else {
            // This frame has a time equal or greater than the
            // desired time, so stop looping.
            if channel.interpolation == Interpolation::Step {
                // Step interpolation uses the candidate frame
                return frame.data;
            }
            // Other interpolation options are linear and cubic spline. Cubic
            // spline isn't supported so if it wasn't filtered out already it
            // will be treated as linear.
            // This currently uses dual quaternion linear blending but may
            // change to ScLERP when that option is available.
            return dualquat::dlb(
                &frame.data,
                &f.data,
                weight(frame.time, f.time, current_time),
            );
        }
    }
    // Fall through past the end of the channel so return data
    // from candidate frame
    frame.data
}

/// Calculates transform dual quaternion for given joint at arbitrary timestamp
fn transform(
    joint_info: &JointInfo,
    animation: &Animation,
    node_index: usize,
    current_time: f32,
) -> DualQuat {
    // The animation channel may not exist or the current time may be before
    // its first frame. The binding pose is used for the initial data at time
    // 0. It may be possible to do something more with this in the future to
    // blend from one animation to the next.
    let initial_frame = Keyframe {
        time: 0.0_f32,
        data: joint_info.bind,
    };
    animation
        .channels
        .get(&node_index)
        .map_or(joint_info.bind, |channel| {
            calculate(channel, &initial_frame, current_time)
        })
}

// Call with the root node to recursively calculate node transform
// including the inverse binding
fn traverse(
    skeleton: &Skeleton,
    animation: &Animation,
    node_index: usize,
    dq_in: DualQuat,
    output: &mut HashMap<usize, DualQuat>,
    current_time: f32,
) {
    let Some(joint_info) = skeleton.tree.get(&node_index) else {
        debug!("node_index={} not in tree", node_index);
        return;
    };

    // Calculate dual quaternion for this node
    let dq_anim = transform(joint_info, animation, node_index, current_time);
    let node_dq = dualquat::mul(&dq_in, &dq_anim);

    // Recurse through the children
    for child_index in &joint_info.children {
        traverse(
            skeleton,
            animation,
            *child_index,
            node_dq,
            output,
            current_time,
        );
    }

    // Combine with the inverse binding and write it to the output
    output.insert(
        node_index, // Keyed by node index (not joint index)
        dualquat::mul(&node_dq, &joint_info.inv_bind),
    );
}

/// Returns the joint transforms for an animation at an arbitrary timestamp.
/// Data is written to the provided mutable slice and only joints with valid
/// data will be written.
pub fn animate(
    skeleton: &Skeleton,
    animation: &Animation,
    output: &mut [DualQuat],
    current_time: f32,
) {
    // Walk the nodes
    let mut transforms = HashMap::<usize, DualQuat>::new();
    traverse(
        skeleton,
        animation,
        skeleton.root,
        DualQuat::default(),
        &mut transforms,
        current_time,
    );

    // Walk the joints. Stops upon reaching the end of the skeleton's
    // `joint_to_node` mapping or the end of the provided `output` slice.
    for (out, node_index) in output.iter_mut().zip(&skeleton.joint_to_node) {
        if let Some(dq) = transforms.get(node_index) {
            *out = *dq;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        animation::{
            Animation, AnimationChannel, Interpolation, JointInfo, Keyframe,
            Skeleton,
        },
        dualquat::{self, DualQuat},
    };
    use ahash::{HashMap, HashMapExt};
    use nalgebra_glm as glm;

    const EPSILON: f32 = 0.0005_f32;

    fn approx_eq(a: f32, b: f32) {
        assert!((b - a).abs() < EPSILON);
    }

    fn granary() -> (usize, JointInfo, AnimationChannel, Animation) {
        let node_index: usize = 5;
        let joint_info = JointInfo {
            name: "test".to_string(),
            parent: node_index,
            children: Vec::new(),
            inv_bind: DualQuat::default(),
            bind: DualQuat::default(),
        };
        let channel = AnimationChannel {
            interpolation: Interpolation::Linear,
            data: vec![Keyframe {
                time: 1.0_f32,
                data: DualQuat::new(
                    &glm::Quat::identity(),
                    &glm::vec3(10.0_f32, 4.0_f32, 0.0_f32),
                ),
            }],
        };
        let mut channels: HashMap<usize, AnimationChannel> = HashMap::new();
        channels.insert(node_index, channel.clone());
        let animation = Animation {
            name: "test".to_string(),
            max_time: 1.0_f32,
            channels,
        };
        (node_index, joint_info, channel, animation)
    }

    #[test]
    fn weight() {
        let x = super::weight(0.0, 10.0, 7.0);
        approx_eq(x, 0.7_f32);
        let x = super::weight(0.0, 10.0, 12.0);
        approx_eq(x, 1.0_f32);
        let x = super::weight(0.0, 10.0, -2.0);
        approx_eq(x, 0.0_f32);
        let x = super::weight(-2.0, 8.0, 3.0);
        approx_eq(x, 0.5_f32);
        let x = super::weight(1.0, 1.0, 1.0);
        assert!(x >= 0.0f32 && x <= 1.0f32);
    }

    #[test]
    fn calculate() {
        let (_node_index, _joint_info, channel, _animation) = granary();

        let keyframe = Keyframe {
            time: 0.0_f32,
            data: DualQuat::default(),
        };
        let current_time = 0.4_f32;
        let res = super::calculate(&channel, &keyframe, current_time);

        let (_, res_t) = dualquat::decompose(&res);
        let c = glm::equal_eps(
            &res_t,
            &glm::vec3(4.0_f32, 1.6_f32, 0.0_f32),
            EPSILON,
        );
        assert!(c.x && c.y && c.z);
    }

    #[test]
    fn transform() {
        let (node_index, joint_info, _channel, animation) = granary();

        let current_time = 0.7_f32;
        let res =
            super::transform(&joint_info, &animation, node_index, current_time);

        let (_, res_t) = dualquat::decompose(&res);
        let c = glm::equal_eps(
            &res_t,
            &glm::vec3(7.0_f32, 2.8_f32, 0.0_f32),
            EPSILON,
        );
        assert!(c.x && c.y && c.z);
    }

    #[test]
    fn traverse() {
        let (node_index, joint_info, _channel, animation) = granary();

        let mut tree: HashMap<usize, JointInfo> = HashMap::new();
        tree.insert(node_index, joint_info);
        let skeleton = Skeleton {
            name: "test".to_string(),
            root: node_index,
            joint_to_node: vec![node_index],
            tree,
        };
        let current_time = 0.3_f32;
        let mut output: HashMap<usize, DualQuat> = HashMap::new();
        super::traverse(
            &skeleton,
            &animation,
            node_index,
            DualQuat::default(),
            &mut output,
            current_time,
        );
        let res = output.get(&node_index).unwrap();

        let (_, res_t) = dualquat::decompose(&res);
        let c = glm::equal_eps(
            &res_t,
            &glm::vec3(3.0_f32, 1.2_f32, 0.0_f32),
            EPSILON,
        );
        assert!(c.x && c.y && c.z);
    }

    #[test]
    fn animate() {
        let (node_index, joint_info, _channel, animation) = granary();

        let mut tree: HashMap<usize, JointInfo> = HashMap::new();
        tree.insert(node_index, joint_info);
        let skeleton = Skeleton {
            name: "test".to_string(),
            root: node_index,
            joint_to_node: vec![node_index],
            tree,
        };
        let current_time = 0.9_f32;
        let mut output = [DualQuat::default()];
        super::animate(&skeleton, &animation, &mut output, current_time);

        let (_, res_t) = dualquat::decompose(&output[0]);
        let c = glm::equal_eps(
            &res_t,
            &glm::vec3(9.0_f32, 3.6_f32, 0.0_f32),
            EPSILON,
        );
        assert!(c.x && c.y && c.z);
    }
}
