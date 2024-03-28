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

// Without this "allow", clippy nursery suggests nested "mul_add" functions
// that are hard to read. Benchmarking has shown they are slower on default
// builds and not faster when FMA is enabled.
//
// See also https://github.com/rust-lang/rust-clippy/issues/6867
#![allow(clippy::suboptimal_flops)]

use std::ops::{Add, Mul, Sub};

use nalgebra_glm as glm;

/// A dual quaternion containing rotation in the `real` part and translation
/// in the `dual` part.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DualQuat {
    pub real: glm::Quat,
    pub dual: glm::Quat,
}

impl Default for DualQuat {
    /// Default is the dual quaternion identity value
    fn default() -> Self {
        Self {
            real: glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 1.0_f32),
            dual: glm::quat(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32),
        }
    }
}

impl DualQuat {
    /// Creates a unit dual quaternion from a unit rotation quaternion and a
    /// translation vector
    #[must_use]
    pub fn new(rot: &glm::Quat, trans: &glm::Vec3) -> Self {
        let pure = glm::quat(trans.x, trans.y, trans.z, 0.0_f32);
        let dual = 0.5_f32 * pure * rot; // 1/2 * t * r
        Self { real: *rot, dual }
    }

    /// Creates a unit dual quaternion from a unit rotation quaternion and a
    /// translation vector which are both swizzled from Y axis up to Z axis up
    #[must_use]
    pub fn new_swizzle(rot: &glm::Quat, trans: &glm::Vec3) -> Self {
        let real = glm::quat(rot.i, -rot.k, rot.j, rot.w);
        let pure = glm::quat(trans.x, -trans.z, trans.y, 0.0_f32);
        let dual = 0.5_f32 * pure * real; // 1/2 * t * r
        Self { real, dual }
    }
}

impl From<DualQuat> for [[f32; 4]; 2] {
    /// Converts from a dual quaternion to GLSL shader ready mat2x4. This is a
    /// matrix for transfer data, not for doing matrix math.
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

impl From<[[f32; 4]; 4]> for DualQuat {
    /// Converts from a 4x4 matrix array (not glm) into a dual quaternion
    fn from(m: [[f32; 4]; 4]) -> Self {
        // It may be more efficient to do this without going through glm,
        // yet somehow this benchmarks a bit faster than `from_mat4` itself.
        from_mat4(&m.into())
    }
}

impl Add for DualQuat {
    type Output = Self;
    /// Performs the + operation to add two dual quaternions
    fn add(self, rhs: Self) -> Self {
        Self {
            real: self.real + rhs.real,
            dual: self.dual + rhs.dual,
        }
    }
}

impl Sub for DualQuat {
    type Output = Self;
    /// Performs the - operation to subtract two dual quaternions
    fn sub(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            dual: self.dual - rhs.dual,
        }
    }
}

impl Mul<Self> for DualQuat {
    type Output = Self;
    /// Performs the * operation to multiply two dual quaternions
    fn mul(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real,
            dual: self.real * rhs.dual + self.dual * rhs.real,
        }
    }
}

impl Mul<f32> for DualQuat {
    type Output = Self;
    /// Performs the * operation to multiply a dual quaternion by a scalar
    fn mul(self, rhs: f32) -> Self {
        Self {
            real: self.real * rhs,
            dual: self.dual * rhs,
        }
    }
}

impl Mul<DualQuat> for f32 {
    type Output = DualQuat;
    /// Performs the * operation to multiply a scalar by a dual quaternion
    fn mul(self, rhs: DualQuat) -> DualQuat {
        DualQuat {
            real: self * rhs.real,
            dual: self * rhs.dual,
        }
    }
}

/// Adds two dual quaternions
#[must_use]
pub fn add(dq1: &DualQuat, dq2: &DualQuat) -> DualQuat {
    DualQuat {
        real: dq1.real + dq2.real,
        dual: dq1.dual + dq2.dual,
    }
}

/// Multiplies two dual quaternions
#[must_use]
pub fn mul(dq1: &DualQuat, dq2: &DualQuat) -> DualQuat {
    DualQuat {
        real: dq1.real * dq2.real,
        dual: dq1.real * dq2.dual + dq1.dual * dq2.real,
    }
}

/// Returns the conjugate of a dual quaternion. There are three possible ways
/// to conjugate a dual quaternion. This is the version most useful for unit
/// quaternions where both the real and dual part are conjugated separately.
#[must_use]
pub fn conjugate(dq: &DualQuat) -> DualQuat {
    DualQuat {
        real: dq.real.conjugate(),
        dual: dq.dual.conjugate(),
    }
}

/// Creates a `glm::Mat4` from a dual quaternion
#[must_use]
pub fn to_mat4(dq: &DualQuat) -> glm::Mat4 {
    // The input is probably already normalized and benchmarking shows this is
    // about half the execution time for the function... but that is like 5 nS
    // so not currently a concern.
    let r = dq.real.normalize();

    let rw = r.scalar();
    let rx = r.vector().x;
    let ry = r.vector().y;
    let rz = r.vector().z;
    let t = (dq.dual * 2.0_f32) * dq.real.conjugate();

    // Some minor benchmarking was done showing presquaring individual
    // terms like rw2 = rw * rw was not faster.

    glm::mat4(
        rw * rw + rx * rx - ry * ry - rz * rz,
        2.0_f32 * rx * ry - 2.0_f32 * rw * rz,
        2.0_f32 * rx * rz + 2.0_f32 * rw * ry,
        t.vector().x,
        2.0_f32 * rx * ry + 2.0_f32 * rw * rz,
        rw * rw + ry * ry - rx * rx - rz * rz,
        2.0_f32 * ry * rz - 2.0_f32 * rw * rx,
        t.vector().y,
        2.0_f32 * rx * rz - 2.0_f32 * rw * ry,
        2.0_f32 * ry * rz + 2.0_f32 * rw * rx,
        rw * rw + rz * rz - rx * rx - ry * ry,
        t.vector().z,
        0.0_f32,
        0.0_f32,
        0.0_f32,
        1.0_f32,
    )
}

/// Creates a dual quaternion from a `glm::Mat4` which does not contain scaling
#[must_use]
pub fn from_mat4(m: &glm::Mat4) -> DualQuat {
    let real = glm::to_quat(m); // Ignores translation but scale is no good
    let t = glm::column(m, 3); // Translation column
    let from_t = glm::quat(t.x, t.y, t.z, 0.0_f32);
    let dual = 0.5_f32 * from_t * real;
    DualQuat { real, dual }
}

/// Creates a dual quaternion from a `glm::Mat4` which does not contain scaling
/// and swizzles it from Y axis up to Z axis up
#[must_use]
pub fn from_mat4_swizzle(m: &glm::Mat4) -> DualQuat {
    let real = glm::to_quat(m); // Ignores translation but scale is no good
    let real = glm::quat(real.i, -real.k, real.j, real.w);
    let t = glm::column(m, 3); // Translation column
    let from_t = glm::quat(t.x, -t.z, t.y, 0.0_f32);
    let dual = 0.5_f32 * from_t * real;
    DualQuat { real, dual }
}

/// Swizzles a dual quaternion from Y axis up to Z axis up
#[must_use]
pub fn swizzle(dq: &DualQuat) -> DualQuat {
    DualQuat {
        real: glm::quat(dq.real.i, -dq.real.k, dq.real.j, dq.real.w),
        dual: glm::quat(dq.dual.i, -dq.dual.k, dq.dual.j, dq.dual.w),
    }
}

/// Decomposes a dual quaternion into a rotation quaternion and translation
/// vector
#[must_use]
pub fn decompose(dq: &DualQuat) -> (glm::Quat, glm::Vec3) {
    let t: glm::Vec3 =
        ((dq.dual * 2.0_f32) * dq.real.conjugate()).vector().into();
    (dq.real, t)
}

/// Normalizes a dual quaternion to produce a unit dual quaternion. If the
/// real part of the input has a magnitude of zero the result will contain
/// NaN.
#[must_use]
pub fn normalize(dq: &DualQuat) -> DualQuat {
    // Many normalize implementations just divide both parts by the magnitude
    // of the real part. That gives the correct result for the real part.
    // However the result for the dual part will be wrong unless the two parts
    // were already orthogonal.
    let r = dq.real;
    let d = dq.dual;
    let rn = r.norm();
    let rn_rec = 1.0_f32 / rn;
    let dn = (r.i * d.i + r.j * d.j + r.k * d.k + r.w * d.w) * rn_rec;
    let r_out = r * rn_rec;
    let d_out = (rn * d - r * dn) * rn_rec * rn_rec;
    DualQuat {
        real: r_out,
        dual: d_out,
    }
}

/// Interpolates between dual quaternions using separate interpolation
#[must_use]
pub fn sep(dq1: &DualQuat, dq2: &DualQuat, weight: f32) -> DualQuat {
    let (r1, t1) = decompose(dq1);
    let (r2, t2) = decompose(dq2);

    // `quat_slerp` should return a unit quaternion so the resulting dual
    // quaternion doesn't have to be normalized
    DualQuat::new(
        &glm::quat_slerp(&r1, &r2, weight),
        &glm::lerp(&t1, &t2, weight),
    )
}

/// Interpolates between dual quaternions using dual linear blending
#[must_use]
pub fn dlb(dq1: &DualQuat, dq2: &DualQuat, weight: f32) -> DualQuat {
    normalize(&DualQuat {
        real: (1.0_f32 - weight) * dq1.real + weight * dq2.real,
        dual: (1.0_f32 - weight) * dq1.dual + weight * dq2.dual,
    })
}
