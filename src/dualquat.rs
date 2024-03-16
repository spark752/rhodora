use nalgebra_glm as glm;

#[derive(Clone, Copy, Debug)]
pub struct DualQuat {
    pub real: glm::Quat,
    pub dual: glm::Quat,
}

/// Dual quaternion
/// GLM has support in the `GLM_GTX_dual_quaternion` extension but this does not
/// seem to be available in the `nalgebra_glm` implementation. So some
/// functionality is implemented here using `glm::Quat`.
impl Default for DualQuat {
    fn default() -> Self {
        Self {
            // Real part contains the rotation
            real: glm::quat(0.0, 0.0, 0.0, 1.0),
            // Dual part contains the translation but is also effected by
            // the rotation
            dual: glm::quat(0.0, 0.0, 0.0, 0.0),
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

#[must_use]
pub fn add(q1: &DualQuat, q2: &DualQuat) -> DualQuat {
    DualQuat {
        real: q1.real + q2.real,
        dual: q1.dual + q2.dual,
    }
}

#[must_use]
pub fn mul(q1: &DualQuat, q2: &DualQuat) -> DualQuat {
    DualQuat {
        real: q1.real * q2.real,
        dual: q1.real * q2.dual + q1.dual * q2.real,
    }
}

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
    // We can almost get there with glm conversions
    let r = dq.real.normalize();
    let rw = r.scalar();
    let rx = r.vector().x;
    let ry = r.vector().y;
    let rz = r.vector().z;
    let t = (dq.dual * 2.0) * dq.real.conjugate();

    // Without this "allow", clippy nursery will suggest some horrible to read
    // nested "mul_add" functions that may not be helpful anyway.
    // See https://github.com/rust-lang/rust-clippy/issues/6867
    #[allow(clippy::suboptimal_flops)]
    glm::mat4(
        rw * rw + rx * rx - ry * ry - rz * rz,
        2.0 * rx * ry - 2.0 * rw * rz,
        2.0 * rx * rz + 2.0 * rw * ry,
        t.vector().x,
        2.0 * rx * ry + 2.0 * rw * rz,
        rw * rw + ry * ry - rx * rx - rz * rz,
        2.0 * ry * rz - 2.0 * rw * rx,
        t.vector().y,
        2.0 * rx * rz - 2.0 * rw * ry,
        2.0 * ry * rz + 2.0 * rw * rx,
        rw * rw + rz * rz - rx * rx - ry * ry,
        t.vector().z,
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

/// Creates a dual quaternion from a `glm::Mat4` which does not contain scaling
#[must_use]
pub fn from_mat4(m: &glm::Mat4) -> DualQuat {
    let real = glm::to_quat(m); // Ignores translation but scale is no good
    let t = glm::column(m, 3); // Translation column
    let from_t = glm::quat(t.x, t.y, t.z, 0.0);
    let dual = 0.5 * from_t * real;
    DualQuat { real, dual }
    // FIXME Make sure of length
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::info;
    use std::sync::Once;

    const EPSILON: f32 = 0.0001; // Small value for float comparisons
    static INIT: Once = Once::new();

    fn init_tests() {
        INIT.call_once(|| {
            env_logger::init();
        });
    }

    fn rotation(dq: &DualQuat) {
        let m1 = to_mat4(&dq); // Function being tested
        let m2 = glm::quat_to_mat4(&dq.real); // Rotation only
                                              //info!("result m1={:?}", m1);
                                              //info!("result m2={:?}", m2);
        let compare = glm::equal_columns_eps(&m1, &m2, EPSILON);
        assert!(compare.x && compare.y && compare.z && compare.w);
    }

    #[test]
    fn test_to_mat4() {
        init_tests();
        info!("test_to_mat4");

        // Identity dual quaternion should produce identity matrix
        let dq = DualQuat {
            real: glm::quat(0.0, 0.0, 0.0, 1.0),
            dual: glm::quat(0.0, 0.0, 0.0, 0.0),
        };
        let m = to_mat4(&dq);
        assert!(m == glm::Mat4::identity());

        // Rotation only dual quaternions should produce the same result as
        // a standard quaternion converted to matrix by glm.

        // Z axis rotation should have matrix values in upper left 4
        let dq = DualQuat {
            real: glm::quat_angle_axis(0.752, &glm::vec3(0.0, 0.0, 1.0)),
            dual: glm::quat(0.0, 0.0, 0.0, 0.0),
        };
        info!("Z axis rotation test dq={:?}", dq);
        rotation(&dq);

        // X axis rotation should have matrix values in centre 4
        let dq = DualQuat {
            real: glm::quat_angle_axis(-0.314, &glm::vec3(1.0, 0.0, 0.0)),
            dual: glm::quat(0.0, 0.0, 0.0, 0.0),
        };
        info!("X axis rotation test dq={:?}", dq);
        rotation(&dq);

        // Y axis rotation should have matrix values in first & third row
        let dq = DualQuat {
            real: glm::quat_angle_axis(0.0808, &glm::vec3(0.0, 1.0, 0.0)),
            dual: glm::quat(0.0, 0.0, 0.0, 0.0),
        };
        info!("Y axis rotation test dq={:?}", dq);
        rotation(&dq);

        // TODO: Add some more testing including translations
    }

    #[test]
    fn test_from_mat4() {
        init_tests();
        info!("test_from_mat4");

        let m1 = glm::Mat4::identity();
        let m1 = glm::rotate_z(&m1, -0.261);
        let m1 = glm::translate(&m1, &glm::vec3(1.0, 0.0, -0.3));
        info!("m1={:?}", m1);
        let dq = from_mat4(&m1);
        info!("dq={:?}", dq);
        let m2 = to_mat4(&dq);
        info!("m2={:?}", m2);
        let compare = glm::equal_columns_eps(&m1, &m2, EPSILON);
        assert!(compare.x && compare.y && compare.z && compare.w);
    }
}
