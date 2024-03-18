//! Support for dual quaternions
//!
//! GLM has support in the `GLM_GTX_dual_quaternion` extension but this does not
//! seem to be available in the `nalgebra_glm` implementation. Basic
//! functionality is implemented here using `glm::Quat`.
//!
//! There is some dual quaternion support in `nalgebra` itself so this may
//! be updated to use that in the future.
//!
//! More information about how dual quaternions work may be found in the test
//! module.
use nalgebra_glm as glm;

/// A dual quaternion containing rotation in the `real` part and translation
/// in the `dual` part.
#[derive(Clone, Copy, Debug)]
pub struct DualQuat {
    pub real: glm::Quat,
    pub dual: glm::Quat,
}

/// Default is the dual quaternion identity value
impl Default for DualQuat {
    fn default() -> Self {
        Self {
            real: glm::quat(0.0f32, 0.0f32, 0.0f32, 1.0f32),
            dual: glm::quat(0.0f32, 0.0f32, 0.0f32, 0.0f32),
        }
    }
}

/// Conversion to GLSL shader ready mat2x4
impl From<DualQuat> for [[f32; 4]; 2] {
    fn from(dq: DualQuat) -> [[f32; 4]; 2] {
        [
            [
                dq.real.coords.x,
                dq.real.coords.y,
                dq.real.coords.z,
                dq.real.coords.w,
            ],
            [
                dq.dual.coords.x,
                dq.dual.coords.y,
                dq.dual.coords.z,
                dq.dual.coords.w,
            ],
        ]
    }
}

/// Adds two dual quaternions
#[must_use]
pub fn add(q1: &DualQuat, q2: &DualQuat) -> DualQuat {
    DualQuat {
        real: q1.real + q2.real,
        dual: q1.dual + q2.dual,
    }
}

/// Multiplies two dual quaternions
#[must_use]
pub fn mul(q1: &DualQuat, q2: &DualQuat) -> DualQuat {
    DualQuat {
        real: q1.real * q2.real,
        dual: q1.real * q2.dual + q1.dual * q2.real,
    }
}

/// Returns the conjugate of a dual quaternion. There are three possible ways
/// to conjugate a dual quaternion. This is the version most useful for unit
/// quaternions where both the real and dual part are conjugated separately.
#[must_use]
pub fn conjugate(q: &DualQuat) -> DualQuat {
    DualQuat {
        real: q.real.conjugate(),
        dual: q.dual.conjugate(),
    }
}

/// Creates a `glm::Mat4` from a dual quaternion
#[must_use]
pub fn to_mat4(dq: &DualQuat) -> glm::Mat4 {
    let r = dq.real.normalize();
    let rw = r.scalar();
    let rx = r.vector().x;
    let ry = r.vector().y;
    let rz = r.vector().z;
    let t = (dq.dual * 2.0f32) * dq.real.conjugate();

    // Without this "allow", clippy nursery will suggest some horrible to read
    // nested "mul_add" functions that may not be helpful anyway.
    // See https://github.com/rust-lang/rust-clippy/issues/6867
    #[allow(clippy::suboptimal_flops)]
    glm::mat4(
        rw * rw + rx * rx - ry * ry - rz * rz,
        2.0f32 * rx * ry - 2.0f32 * rw * rz,
        2.0f32 * rx * rz + 2.0f32 * rw * ry,
        t.vector().x,
        2.0f32 * rx * ry + 2.0f32 * rw * rz,
        rw * rw + ry * ry - rx * rx - rz * rz,
        2.0f32 * ry * rz - 2.0f32 * rw * rx,
        t.vector().y,
        2.0f32 * rx * rz - 2.0f32 * rw * ry,
        2.0f32 * ry * rz + 2.0f32 * rw * rx,
        rw * rw + rz * rz - rx * rx - ry * ry,
        t.vector().z,
        0.0f32,
        0.0f32,
        0.0f32,
        1.0f32,
    )
}

/// Creates a dual quaternion from a `glm::Mat4` which does not contain scaling
#[must_use]
pub fn from_mat4(m: &glm::Mat4) -> DualQuat {
    let real = glm::to_quat(m); // Ignores translation but scale is no good
    let t = glm::column(m, 3); // Translation column
    let from_t = glm::quat(t.x, t.y, t.z, 0.0f32);
    let dual = 0.5 * from_t * real;
    DualQuat { real, dual }
}

/// Creates a dual quaternion from a `glm::Mat4` which does not contain scaling
/// and swizzles it from Y axis up to Z axis up
#[must_use]
pub fn from_mat4_swiz(m: &glm::Mat4) -> DualQuat {
    let real = glm::to_quat(m); // Ignores translation but scale is no good
    let real = glm::quat(real.i, -real.k, real.j, real.w);
    let t = glm::column(m, 3); // Translation column
    let from_t = glm::quat(t.x, -t.z, t.y, 0.0f32);
    let dual = 0.5f32 * from_t * real;
    DualQuat { real, dual }
}
