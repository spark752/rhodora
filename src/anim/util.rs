use super::{
    types::{Animation, Interpolation, Skeleton},
    AnimationChannel, JointInfo, Keyframe,
};
use crate::{
    dualquat::{self, DualQuat},
    model_manager::JointTransforms,
};
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
    current_time: f32,
    dq_in: DualQuat,
    output: &mut HashMap<usize, DualQuat>,
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
            current_time,
            node_dq,
            output,
        );
    }

    // Combine with the inverse binding and write it to the output
    output.insert(
        node_index, // Keyed by node index (not joint index)
        dualquat::mul(&node_dq, &joint_info.inv_bind),
    );
}

/// Returns the joint transforms for an animation at an arbitrary timestamp
#[must_use]
pub fn animate(
    skeleton: &Skeleton,
    animation: &Animation,
    current_time: f32,
) -> JointTransforms {
    let mut output = JointTransforms::default();

    // Walk the nodes to calculate transforms
    let mut transforms = HashMap::<usize, DualQuat>::new();
    traverse(
        skeleton,
        animation,
        skeleton.root,
        current_time,
        DualQuat::default(),
        &mut transforms,
    );

    // Walk the joints to create the output
    for (joint_index, node_index) in skeleton.joint_to_node.iter().enumerate() {
        if let Some(dq) = transforms.get(node_index) {
            output.0[joint_index] = *dq;
        } // else output buffer is already set to default value
    }

    output
}

#[cfg(test)]
mod tests {
    const EPSILON: f32 = 0.0005;

    /// This does NOT do interpolation, just checks the weight function
    #[test]
    fn interpolation_weight() {
        let x = super::weight(0.0, 10.0, 7.0);
        assert!((x - 0.7f32).abs() < EPSILON);
        let x = super::weight(0.0, 10.0, 12.0);
        assert!((x - 1.0f32).abs() < EPSILON);
        let x = super::weight(0.0, 10.0, -2.0);
        assert!((x - 0.0f32).abs() < EPSILON);
        let x = super::weight(-2.0, 8.0, 3.0);
        assert!((x - 0.5f32).abs() < EPSILON);
        let x = super::weight(1.0, 1.0, 1.0);
        assert!(x >= 0.0f32 && x <= 1.0f32);
    }
}
