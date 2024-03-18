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
//! Since users are likely to forget to use the f32 syntax, that is usually
//! done here intentionally for testing.
//!
//! For practical `DualQuat` use, a unit dual quaternion is often desired, but
//! this is not a requirement for maths, so the tests often use easy to enter
//! and easy to compare values such as the ones above.

use log::info;
use nalgebra_glm as glm;
use rhodora::dualquat::{self, DualQuat};
use std::sync::Once;

const EPSILON: f32 = 0.0001f32; // Small value for float comparisons
static INIT: Once = Once::new();

/// Initializes logging in a "once per test run" manner. Call at the start of
/// each test that needs logging.
fn init_tests() {
    INIT.call_once(|| {
        env_logger::init();
    });
}

/// Verify a result is unit by multiplying by its conjugate
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

/// Tests `DualQuat::default`
#[test]
fn default() {
    // Identity dual quaternion has 1 for real.w with everything else 0
    let dq = DualQuat::default();
    assert_eq!(dq.real, glm::Quat::identity());
    assert_eq!(dq.dual, glm::quat(0.0f32, 0.0f32, 0.0f32, 0.0f32));
}

/// Tests DualQuat `From` trait for converting to a GLSL shader friendly array
#[test]
fn from() {
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

/// Tests `DualQuat::add`
#[test]
fn add() {
    // Addition of dual quaternions is component wise
    // [r1 + r2] + [d1 + d2]ϵ

    let dq1 = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(-1.0, -2.0, -3.0, -4.0),
    };
    let dq2 = DualQuat {
        real: glm::quat(4.0, 3.0, 2.0, 1.0),
        dual: glm::quat(-4.0, -3.0, -2.0, -1.0),
    };
    let res = dualquat::add(&dq1, &dq2);
    assert!(
        res.real.i == 5.0f32
            && res.real.j == 5.0f32
            && res.real.k == 5.0f32
            && res.real.w == 5.0f32
    );
    assert!(
        res.dual.i == -5.0f32
            && res.dual.j == -5.0f32
            && res.dual.k == -5.0f32
            && res.dual.w == -5.0f32
    );
}

/// Tests `DualQuat::mul`
#[test]
fn mul() {
    // Multiplication has some terms cancel out because ϵ squared = 0
    // [r1 * r2] + [r1 * d2 + r2 * d1]ϵ

    let dq1 = DualQuat {
        real: glm::quat(1.0, 2.0, 3.0, 4.0),
        dual: glm::quat(-1.0, -2.0, -3.0, -4.0),
    };
    let dq2 = DualQuat {
        real: glm::quat(4.0, 3.0, 2.0, 1.0),
        dual: glm::quat(-4.0, -3.0, -2.0, -1.0),
    };
    let res = dualquat::mul(&dq1, &dq2);
    assert!(
        res.real.i == 12.0f32
            && res.real.j == 24.0f32
            && res.real.k == 6.0f32
            && res.real.w == -12.0f32
    );
    assert!(
        res.dual.i == -24.0f32
            && res.dual.j == -48.0f32
            && res.dual.k == -12.0f32
            && res.dual.w == 24.0f32
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
    let q = glm::quat(1.0f32, 2.0f32, 3.0f32, 4.0f32);
    let conj = glm::quat_conjugate(&q);
    let res = q * conj;
    assert_eq!(res.scalar(), 30.0f32); // Not testing anything in DualQuat

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
        conj.real.w == 4.0f32
            && conj.real.i == -1.0f32
            && conj.real.j == -2.0f32
            && conj.real.k == -3.0f32
    );
    assert!(
        conj.dual.w == 8.0f32
            && conj.dual.i == -5.0f32
            && conj.dual.j == -6.0f32
            && conj.dual.k == -7.0f32
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
                                              //info!("result m1={:?}", m1);
                                              //info!("result m2={:?}", m2);
        let compare = glm::equal_columns_eps(&m1, &m2, EPSILON);
        assert!(compare.x && compare.y && compare.z && compare.w);
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
    // a standard quaternion converted to matrix by glm.

    // Z axis rotation should have matrix values in upper left 4
    let dq = DualQuat {
        real: glm::quat_angle_axis(
            0.752, //
            &glm::vec3(0.0, 0.0, 1.0),
        ),
        dual: glm::quat(0.0, 0.0, 0.0, 0.0),
    };
    info!("Z axis rotation test dq={:?}", dq);
    rotation(&dq);

    // X axis rotation should have matrix values in centre 4
    let dq = DualQuat {
        real: glm::quat_angle_axis(
            -0.314, //
            &glm::vec3(1.0, 0.0, 0.0),
        ),
        dual: glm::quat(0.0, 0.0, 0.0, 0.0),
    };
    info!("X axis rotation test dq={:?}", dq);
    rotation(&dq);

    // Y axis rotation should have matrix values in first & third row
    let dq = DualQuat {
        real: glm::quat_angle_axis(
            0.0808, //
            &glm::vec3(0.0, 1.0, 0.0),
        ),
        dual: glm::quat(0.0, 0.0, 0.0, 0.0),
    };
    info!("Y axis rotation test dq={:?}", dq);
    rotation(&dq);

    // TODO: Add some more testing including translations
}

/// Tests `DualQuat::from_mat4`
#[test]
fn from_mat4() {
    // Arbitrary input matrix
    let m1 = glm::Mat4::identity();
    let m1 = glm::rotate_z(&m1, -0.261);
    let m1 = glm::translate(
        &m1, //
        &glm::vec3(31.0f32, -192.52f32, -0.34f32),
    );

    // Function to test
    let dq = dualquat::from_mat4(&m1);

    // Verify result is a unit dual quaternion
    check_unit(&dq);

    // Convert back to matrix and compare with original
    let m2 = dualquat::to_mat4(&dq);
    let c = glm::equal_columns_eps(&m1, &m2, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}

/// Tests `DualQuat::from_mat4_swiz`
#[test]
fn from_mat4_swiz() {
    // Arbitrary input matrix
    let m1 = glm::Mat4::identity();
    let m1 = glm::rotate_x(&m1, 1.42);
    let m1 = glm::translate(
        &m1, //
        &glm::vec3(42.2f32, -3.0f32, 0.5f32),
    );

    // Manual swizzle
    let dq1 = dualquat::from_mat4(&m1);
    let dq1 = DualQuat {
        real: glm::quat(dq1.real.i, -dq1.real.k, dq1.real.j, dq1.real.w),
        dual: glm::quat(dq1.dual.i, -dq1.dual.k, dq1.dual.j, dq1.dual.w),
    };

    // Swizzle
    let dq2 = dualquat::from_mat4_swiz(&m1);

    // Verify result is a unit dual quaternion
    check_unit(&dq2);

    // Compare
    let c = glm::quat_equal_eps(&dq1.real, &dq2.real, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
    let c = glm::quat_equal_eps(&dq1.dual, &dq2.dual, EPSILON);
    assert!(c.x && c.y && c.z && c.w);
}
