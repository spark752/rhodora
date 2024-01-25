use nalgebra_glm as glm;

#[derive(Clone, Copy)]
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
            // FIXME: Values for testing
            //dual: glm::quat(0.5, 0.0, -0.2, 0.0),
            //    }
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
