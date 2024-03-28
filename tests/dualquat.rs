//! Tests for dualquat module
//!
//! Note that nalgebra supports f32 and f64 quaternions but `DualQuat` is f32
//! only.
//!
//! With code like:
//! ```
//! let dq = DualQuat {
//!     real: glm::quat(1.0, 2.0, 3.0, 4.0),
//!     dual: glm::quat(5.0, 6.0, 7.0, 8.0),
//! };
//! ```
//! the compiler would normally treat values like "1.0" as f64 and create a f64
//! quaternion, so they should actually be written like "1.0f32". However since
//! the `DualQuat` needs f32, the compiler figures that out and creates f32
//! quaternions and everything is fine.
//!
//! Since users are likely to forget to use the f32 syntax, that is sometimes
//! done here intentionally for testing.
//!
//! For practical `DualQuat` use, a unit dual quaternion is often desired, but
//! this is not a requirement for maths, so the tests often use easy to enter
//! and easy to compare values such as the ones above.

use log::info;
use nalgebra_glm as glm;
use rhodora::dualquat::{self, DualQuat};
use std::sync::Once;

const EPSILON: f32 = 0.0001_f32; // Small value for float comparisons
static INIT: Once = Once::new();

/// Initializes logging in a "once per test run" manner. Call at the start of
/// each test that needs logging.
fn init_tests() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

/// Verifies a result is unit by multiplying by its conjugate
fn check_unit(dq: &DualQuat) {
    let unit = dualquat::mul(dq, &dualquat::conjugate(dq));
    let c = glm::quat_equal_eps(&unit.real, &glm::Quat::identity(), EPSILON);
    assert!(c.x && c.y && c.z && c.w);
    let c = glm::quat_equal_eps(
        &unit.dual,
        &glm::quat(0.0f32, 0.0f32, 0.0f32, 0.0f32),
        EPSILON,
    );
    assert!(c.x && c.y && c.z && c.w);
}

/// Debugs `check_unit`
#[allow(dead_code)]
fn debug_check_unit(dq: &DualQuat) {
    init_tests();
    info!(
        "debug_check_unit p^2={} pq={}",
        dq.real.i * dq.real.i
            + dq.real.j * dq.real.j
            + dq.real.k * dq.real.k
            + dq.real.w * dq.real.w,
        dq.real.i * dq.dual.i
            + dq.real.j * dq.dual.j
            + dq.real.k * dq.dual.k
            + dq.real.w * dq.dual.w,
    );
    let unit = dualquat::mul(dq, &dualquat::conjugate(dq));
    info!("debug_check_unit={:?}", unit);
}

/// Compares two dual quaternions for approximate equality
fn compare(dq1: &DualQuat, &dq2: &DualQuat) {
    let c = glm::quat_equal_eps(&dq1.real, &dq2.real, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
    let c = glm::quat_equal_eps(&dq1.dual, &dq2.dual, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}

/// Tests `DualQuat::default`
#[test]
fn default() {
    // Identity dual quaternion has 1 for real.w with everything else 0
    let dq = DualQuat::default();
    assert_eq!(dq.real, glm::Quat::identity());
    assert_eq!(dq.dual, glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32));
}

/// Tests `DualQuat::new`
#[test]
fn new() {
    init_tests();

    // Create the dual quaternion for testing
    let rot = glm::quat_angle_axis(
        -1.491_f32,
        &glm::vec3(0.620174_f32, -0.248069_f32, 0.744208_f32),
    );
    let trans = glm::vec3(-12.6_f32, 1204.0_f32, 0.004_f32);
    let dq1 = DualQuat::new(&rot, &trans);

    // Check for unit length
    check_unit(&dq1);

    // Build another dual quaternion manually to compare
    // The dual quaternion real part is just the `rot` quaternion defined
    // above. The dual part is based on:
    //  d = 1/2 * t * r
    // where t is the "pure" quaternion version of `trans` (a quaternion with
    // scalar part = 0) and r is `rot`.
    let pure = glm::quat(trans.x, trans.y, trans.z, 0.0_f32);
    let dual = 0.5_f32 * pure * rot;
    let dq2 = DualQuat { real: rot, dual };

    // Compare
    let c = glm::quat_equal_eps(&dq1.real, &dq2.real, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
    let c = glm::quat_equal_eps(&dq1.dual, &dq2.dual, EPSILON);
    assert!(c.x && c.y && c.z && c.w);

    // Convert to matrices to compare. This test is basically the same as
    // the test for `to_mat4`.
    let m1 = glm::Mat4::identity();
    let m1 = glm::translate(&m1, &trans);
    let m1 = m1 * glm::quat_to_mat4(&rot);
    //info!("new m1={:?}", m1);
    let m2 = dualquat::to_mat4(&dq1);
    //info!("new m2={:?}", m2);

    // Compare
    let c = glm::equal_columns_eps(&m1, &m2, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}

/// Tests `DualQuat::new_swizzle`
#[test]
fn new_swizzle() {
    let rot = glm::quat_angle_axis(
        0.352_f32,
        &glm::vec3(-0.67_f32, 0.33_f32, 0.67_f32),
    );
    let trans = glm::vec3(41.6_f32, -88.5_f32, 107.9_f32);

    let dq1 = DualQuat::new(&rot, &trans);
    let dq1 = dualquat::swizzle(&dq1);

    let dq2 = DualQuat::new_swizzle(&rot, &trans);

    compare(&dq1, &dq2);
}

/// Tests DualQuat `From` trait for converting to a GLSL shader friendly array
#[test]
fn from_for_glsl() {
    let dq = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(5.0, 6.0, 7.0, 8.0),
    };

    // Array that acts as a GLSL friendly mat2x4 for sending to shaders
    let m: [[f32; 4]; 2] = dq.into();

    // Real in one column, dual in the other, w last for each
    assert!(
        m[0][0] == dq.real.i
            && m[0][1] == dq.real.j
            && m[0][2] == dq.real.k
            && m[0][3] == dq.real.w
    );
    assert!(
        m[1][0] == dq.dual.i
            && m[1][1] == dq.dual.j
            && m[1][2] == dq.dual.k
            && m[1][3] == dq.dual.w
    );
}

/// Tests DualQuat `From` trait for converting from a 4x4 matrix array
#[test]
fn from_for_array() {
    init_tests();

    // This should be a valid rotation and translation matrix
    let arr: [[f32; 4]; 4] = {
        [
            [1.0_f32, 0.0_f32, 0.0f32, 0.0_f32], // column 0
            [0.0_f32, 0.3584_f32, -0.9336_f32, 0.0_f32], // column 1
            [0.0_f32, 0.9336_f32, 0.3584_f32, 0.0_f32], // column 2
            [5.0_f32, 7.0_f32, 9.0_f32, 1.0_f32], // column 3
        ]
    };

    // Comparison conversion
    let m1: glm::Mat4 = arr.into();
    let dq1 = dualquat::from_mat4(&m1);
    //info!("from_for_array dq1={:?}", dq1);

    // Value to test
    let dq2: DualQuat = arr.into();
    //info!("from for array dq2={:?}", dq2);

    compare(&dq1, &dq2);

    // This should not actually work as a unit dual quaternion but nothing
    // dangerous should happen
    let arr: [[f32; 4]; 4] = {
        [
            [3.0_f32, 0.0_f32, 8.0_f32, 0.0_f32], // column 0
            [0.0_f32, 4.0_f32, 5.0_f32, 0.0_f32], // column 1
            [9.0_f32, 6.0_f32, 7.0_f32, 0.0_f32], // column 2
            [0.0_f32, 1.0_f32, 2.0_f32, 1.0_f32], // column 3
        ]
    };

    // Comparison conversion
    let m1: glm::Mat4 = arr.into();
    let dq1 = dualquat::from_mat4(&m1);
    //info!("from_for_array dq1={:?}", dq1);

    // Value to test
    let dq2: DualQuat = arr.into();
    //info!("from for array dq2={:?}", dq2);

    compare(&dq1, &dq2);
}

/// Tests `DualQuat::add` and the `Add` and `Sub` traits
#[test]
fn add_sub() {
    // Addition of dual quaternions is component wise
    // [r1 + r2] + [d1 + d2]ϵ
    let compare = |res: &DualQuat| {
        assert!(
            res.real.i == 5.0_f32
                && res.real.j == 5.0_f32
                && res.real.k == 5.0_f32
                && res.real.w == 5.0_f32
        );
        assert!(
            res.dual.i == -5.0_f32
                && res.dual.j == -5.0_f32
                && res.dual.k == -5.0_f32
                && res.dual.w == -5.0_f32
        );
    };

    let dq1 = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(-1.0, -2.0, -3.0, -4.0),
    };
    let dq2 = DualQuat {
        real: glm::quat(4.0, 3.0, 2.0, 1.0),
        dual: glm::quat(-4.0, -3.0, -2.0, -1.0),
    };

    let res = dualquat::add(&dq1, &dq2);
    compare(&res);

    let res = dq1 + dq2;
    compare(&res);

    let res = res - dq2; // Starts with previous result
    assert_eq!(res, dq1); // Values with no rounding error so exact compare ok
}

/// Tests `DualQuat::mul` and the `Mul` trait
#[test]
fn mul() {
    // Multiplication has some terms cancel out because ϵ squared = 0
    // [r1 * r2] + [r1 * d2 + r2 * d1]ϵ
    let compare = |res: &DualQuat| {
        assert!(
            res.real.i == 12.0_f32
                && res.real.j == 24.0_f32
                && res.real.k == 6.0_f32
                && res.real.w == -12.0_f32
        );
        assert!(
            res.dual.i == -24.0_f32
                && res.dual.j == -48.0_f32
                && res.dual.k == -12.0_f32
                && res.dual.w == 24.0_f32
        );
    };

    let dq1 = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(-1.0, -2.0, -3.0, -4.0),
    };
    let dq2 = DualQuat {
        real: glm::quat(4.0, 3.0, 2.0, 1.0),
        dual: glm::quat(-4.0, -3.0, -2.0, -1.0),
    };

    let res = dualquat::mul(&dq1, &dq2);
    compare(&res);

    let res = dq1 * dq2;
    compare(&res);

    let res = dq1 * 3.0; // Rust should figure out this is 3.0_f32
    assert!(
        res.real.i == 3.0_f32
            && res.real.j == 6.0_f32
            && res.real.k == 9.0_f32
            && res.real.w == 12.0_f32
    );
    assert!(
        res.dual.i == -3.0_f32
            && res.dual.j == -6.0_f32
            && res.dual.k == -9.0_f32
            && res.dual.w == -12.0_f32
    );

    let res = -2.0 * dq2; // Rust should figure out this is -2.0_f32
    assert!(
        res.real.i == -8.0_f32
            && res.real.j == -6.0_f32
            && res.real.k == -4.0_f32
            && res.real.w == -2.0_f32
    );
    assert!(
        res.dual.i == 8.0_f32
            && res.dual.j == 6.0_f32
            && res.dual.k == 4.0_f32
            && res.dual.w == 2.0_f32
    );
}

/// Tests `DualQuat::conjugate`
#[test]
fn conjugate() {
    // The conjugate of a regular quaternion is the negation of the vector part:
    // q* = q0 - q1i - q2j - q3k
    //
    // Multiplying a quaternion by its conjugate will make the vector part
    // zero (note the lack of i,j, and k below) so producing a scalar,
    // which makes the conjugate a useful property:
    // qq* = q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3
    //
    // Of course nalgebra will still return this in a quaternion.
    // Example:
    // Note parameters are in xyzw order and f32 is being forced
    let q = glm::quat(1.0_f32, 2.0_f32, 3.0_f32, 4.0_f32);
    let conj = glm::quat_conjugate(&q);
    let res = q * conj;
    assert_eq!(res.scalar(), 30.0_f32); // Not testing anything in DualQuat

    // Dual quaternions have three possible conjugates.
    //
    // One possible conjugate is based on the rule for quaternions and so both
    // the real part and the dual part are independently conjugated:
    // [r0 - r1i - r2j - r3k] + [d0 - d1i - d2j - d3k]ϵ
    // This conjugate of Â is sometimes written Â* (usual conjugate symbol).
    // Multiplying a dual quaternion by this conjugate gives a dual scalar
    // with a ϵ part. However if the real and dual parts of the dual quaternion
    // are orthogonal, then the dual part of this conjugate becomes zero. If the
    // real part of the dual quaternion is also of unit length, then the real
    // part of this conjugate becomes 1. A dual quaternion that satisfies both
    // these conditions is called a "unit dual quaternion". This type of dual
    // quaternion is quite useful for graphics so this is the conjugate type
    // used by this library.
    //
    // Another possible conjugate is based on the rule for dual numbers. This
    // conjugate of Â is sometimes written Â• (solid dot symbol). Multiplying a
    // dual quaternion by this conjugate gives a less useful answer.
    //
    // The third possible conjugate combines the other two. This conjugate of Â
    // is sometimes written Â⋄ (diamond symbol). Mutiplying a dual quaternion by
    // this conjugate gives a dual quaternion with the real part a scalar and
    // the dual part is a vector.

    // Finally some tests
    let dq = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(5.0, 6.0, 7.0, 8.0),
    };
    let conj = dualquat::conjugate(&dq);
    assert!(
        conj.real.w == 4.0_f32
            && conj.real.i == -1.0_f32
            && conj.real.j == -2.0_f32
            && conj.real.k == -3.0_f32
    );
    assert!(
        conj.dual.w == 8.0_f32
            && conj.dual.i == -5.0_f32
            && conj.dual.j == -6.0_f32
            && conj.dual.k == -7.0_f32
    );
    let nconj = glm::quat_conjugate(&dq.real);
    let c = glm::quat_equal_eps(&nconj, &conj.real, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
    let nconj = glm::quat_conjugate(&dq.dual);
    let c = glm::quat_equal_eps(&nconj, &conj.dual, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}

/// Tests `DualQuat::to_mat4`
#[test]
fn to_mat4() {
    fn rotation(dq: &DualQuat) {
        let m1 = dualquat::to_mat4(&dq); // Function being tested
        let m2 = glm::quat_to_mat4(&dq.real); // Rotation only

        let c = glm::equal_columns_eps(&m1, &m2, EPSILON);
        assert!(c.x && c.y && c.z && c.w);
    }

    init_tests();

    // Identity dual quaternion should produce identity matrix
    let dq = DualQuat {
        real: glm::quat(0.0, 0.0, 0.0, 1.0),
        dual: glm::quat(0.0, 0.0, 0.0, 0.0),
    };
    let m = dualquat::to_mat4(&dq);
    assert_eq!(m, glm::Mat4::identity());

    // Rotation only dual quaternions should produce the same result as
    // a standard quaternion converted to matrix by glm. Comparisons are
    // done by the `rotation` function above. Debug output is provided to
    // visually inspect the matrix layout if desired.

    // Z axis rotation should have matrix values in upper left 4
    let dq = DualQuat {
        real: glm::quat_angle_axis(
            0.752_f32,
            &glm::vec3(0.0_f32, 0.0_f32, 1.0_f32),
        ),
        dual: glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
    };
    //info!("Z axis rotation test dq={:?}", dq);
    rotation(&dq);

    // X axis rotation should have matrix values in centre 4
    let dq = DualQuat {
        real: glm::quat_angle_axis(
            -0.314_f32,
            &glm::vec3(1.0_f32, 0.0_f32, 0.0_f32),
        ),
        dual: glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
    };
    //info!("X axis rotation test dq={:?}", dq);
    rotation(&dq);

    // Y axis rotation should have matrix values in first & third row
    let dq = DualQuat {
        real: glm::quat_angle_axis(
            0.0808_f32,
            &glm::vec3(0.0_f32, 1.0_f32, 0.0_f32),
        ),
        dual: glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
    };
    //info!("Y axis rotation test dq={:?}", dq);
    rotation(&dq);

    // Build up a matrix. glm has functions for quaternions, but doesn't
    // seem to have an equivalent to `translate` that would modify an existing
    // matrix using a quaternion rotation. Instead the quaternion can be
    // converted to a matrix and multiplied. The order of all this has to be
    // correct. Debug output is provided for visual inspection.
    let rot = glm::quat_angle_axis(
        std::f32::consts::FRAC_PI_3,
        &glm::vec3(0.811107_f32, 0.486664_f32, 0.324443_f32),
    );
    let trans = glm::vec3(14.2_f32, -3.36_f32, 18.9_f32);
    let m1 = glm::Mat4::identity();
    let m1 = glm::translate(&m1, &trans);
    let m1 = m1 * glm::quat_to_mat4(&rot);
    //info!("to_mat4 m1={:?}", m1);

    // Create a unit dual quaternion
    let dq = DualQuat::new(&rot, &trans);

    // Finally the actual function to test can be called
    let m2 = dualquat::to_mat4(&dq);
    //info!("to_mat4 m2={:?}", m2);

    // Compare
    let c = glm::equal_columns_eps(&m1, &m2, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}

/// Tests `DualQuat::from_mat4`
#[test]
fn from_mat4() {
    // Arbitrary input matrix
    let m1 = glm::Mat4::identity();
    let m1 = glm::translate(
        &m1, //
        &glm::vec3(31.0_f32, -192.52_f32, -0.34_f32),
    );
    let m1 = glm::rotate_z(&m1, -0.261_f32);

    // Function to test
    let dq = dualquat::from_mat4(&m1);

    // Verify result is a unit dual quaternion
    check_unit(&dq);

    // Convert back to matrix and compare with original
    let m2 = dualquat::to_mat4(&dq);
    let c = glm::equal_columns_eps(&m1, &m2, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}

/// Tests `DualQuat::from_mat4_swizzle`
#[test]
fn from_mat4_swizzle() {
    // Arbitrary input matrix
    let m1 = glm::Mat4::identity();
    let m1 = glm::translate(
        &m1, //
        &glm::vec3(42.2_f32, -3.0_f32, 0.5_f32),
    );
    let m1 = glm::rotate_x(&m1, 1.42_f32);

    // Manual swizzle
    let dq1 = dualquat::from_mat4(&m1);
    let dq1 = DualQuat {
        real: glm::quat(dq1.real.i, -dq1.real.k, dq1.real.j, dq1.real.w),
        dual: glm::quat(dq1.dual.i, -dq1.dual.k, dq1.dual.j, dq1.dual.w),
    };

    // Swizzle
    let dq2 = dualquat::from_mat4_swizzle(&m1);

    // Verify result is a unit dual quaternion
    check_unit(&dq2);

    // Compare
    compare(&dq1, &dq2);
}

/// Tests 'DualQuat::swizzle`
#[test]
fn swizzle() {
    // Arbitrary input matrix
    let m1 = glm::Mat4::identity();
    let m1 = glm::translate(
        &m1, //
        &glm::vec3(-5.33_f32, 0.00326_f32, 9.31_f32),
    );
    let m1 = glm::rotate_x(&m1, -0.777_f32);

    // Swizzle different ways
    let dq1 = dualquat::from_mat4_swizzle(&m1);
    let dq2 = dualquat::swizzle(&dualquat::from_mat4(&m1));

    // Verify result is a unit dual quaternion
    check_unit(&dq2);

    // Compare
    compare(&dq1, &dq2);
}

/// Tests `DualQuat::decompose`
#[test]
fn decompose() {
    // Create the dual quaternion for testing
    let rot1 = glm::quat_angle_axis(
        0.874_f32,
        &glm::vec3(0.620174_f32, -0.248069_f32, 0.744208_f32),
    );
    let trans1 = glm::vec3(14.9_f32, 12.3_f32, -1.5_f32);
    let dq = DualQuat::new(&rot1, &trans1);

    // Test
    let (rot2, trans2) = dualquat::decompose(&dq);
    let c = glm::quat_equal_eps(&rot1, &rot2, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
    let c = glm::equal_eps(&trans1, &trans2, EPSILON);
    assert!(c.x && c.y && c.z);
}

/// Tests `DualQuat::normalize`
#[test]
fn normalize() {
    init_tests();
    let dq1 = DualQuat::new(
        &glm::quat(8.0_f32, 9.0_f32, 10.0_f32, 11.0_f32),
        &glm::vec3(41.0_f32, 42.0_f32, 43.0_f32),
    );

    // This would panic, showing this is not a unit dual quaternion:
    //check_unit(&dq1);

    // But this should pass:
    let dq2 = dualquat::normalize(&dq1);
    check_unit(&dq2);

    // Some arbitrary non-orthogonal dual quaternion should work too:
    let dq1 = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(5.0, 6.0, 7.0, 8.0),
    };
    let dq2 = dualquat::normalize(&dq1);
    check_unit(&dq2);

    // Zero length will not normalize but shouldn't panic
    let dq1 = DualQuat::new(
        &glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
        &glm::vec3(0.0_f32, 0.0_f32, 0.0_f32),
    );
    let dq2 = dualquat::normalize(&dq1);
    info!("normalize zero={:?}", dq2);
}

/// Helper for interpolation tests which creates some dual quaternions
/// for testing and returns them and an interpolation for comparison
fn setup_interpolate(weight: f32) -> (DualQuat, DualQuat, DualQuat) {
    let r1 = glm::quat_angle_axis(
        0.752_f32, //
        &glm::vec3(0.0_f32, 0.0_f32, 1.0_f32),
    );
    let r2 = glm::quat_angle_axis(
        1.123_f32, //
        &glm::vec3(0.0_f32, 0.0_f32, 1.0_f32),
    );

    let t1 = glm::vec3(0.0_f32, 0.0_f32, 0.0_f32);
    let t2 = glm::vec3(4.0_f32, 2.0_f32, 1.0_f32);

    let dq1 = DualQuat::new(&r1, &t1);
    let dq2 = DualQuat::new(&r2, &t2);

    let r_res = glm::quat_slerp(&r1, &r2, weight);
    let t_res = glm::lerp(&t1, &t2, weight);

    (dq1, dq2, DualQuat::new(&r_res, &t_res))
}

/// Tests `DualQuat::sep`
#[test]
fn sep() {
    let func = {
        |weight| {
            let (dq1, dq2, dq_comp) = setup_interpolate(weight);
            let dq_res = dualquat::sep(&dq1, &dq2, weight);
            check_unit(&dq_res); // Important to make sure actually unit
            compare(&dq_res, &dq_comp);
        }
    };
    for i in 0..=10 {
        func((i as f32) * 0.1f32);
    }
}

/// Tests `DualQuat::dlb`
#[test]
fn dlb() {
    init_tests();
    let func = {
        |weight| {
            let (dq1, dq2, dq_comp) = setup_interpolate(weight);
            let dq_res = dualquat::dlb(&dq1, &dq2, weight);
            check_unit(&dq_res);

            // Different results are expected versus `setup_interpolate` so how
            // to compare?
            // Current solution is to look at rotation and translation
            // individually with some cherry picked epsilon values.
            let (r1, t1) = dualquat::decompose(&dq_res);
            let (r2, t2) = dualquat::decompose(&dq_comp);

            // Rotation values should be similar
            info!("weight: {}, dlb rot: {:?} vs sep rot: {:?}", weight, r1, r2);
            let c = glm::quat_equal_eps(&r1, &r2, 0.002_f32);
            assert!(c.x && c.y && c.z && c.w);

            // Translation may not be very similar at all
            info!(
                "weight: {}, dlb trans: {:?} vs sep trans: {:?}",
                weight, t1, t2
            );
            let c = glm::equal_eps(&t1, &t2, 0.2_f32);
            assert!(c.x && c.y && c.z);
        }
    };
    for i in 0..=10 {
        func((i as f32) * 0.1f32);
    }
}
