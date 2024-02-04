use nalgebra_glm as glm;

pub type LightArray = [[f32; 4]; 4];

/// Trait used for PBR compatible lights. You can use `PbrLights` which
/// implement this trait or use your own custom implementation.
pub trait PbrLightTrait {
    fn ambient_array(&self) -> [f32; 4];
    fn light_array(&self, view_matrix: &glm::Mat4) -> LightArray;
}

/// Simple struct for lighting that implements `PbrLightTrait`.
pub struct PbrLights {
    pub ambient: [f32; 4],
    pub array: LightArray,
}

impl Default for PbrLights {
    fn default() -> Self {
        Self {
            // Ambient has a colour with intensity in the last element. This
            // is just multiplied by the colour so it is not in the same scale
            // as the intensity used for the other lights.
            ambient: [1.0, 1.0, 1.0, 0.1],

            // Lights are in VIEW space with intensity in last element.
            // Note that Vulkan view space has z = 0 at the near plane so these
            // defaults give a light above the camera at the near plane.
            array: [
                [0.0, -5.0, 0.0, 150.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
        }
    }
}

impl PbrLightTrait for PbrLights {
    fn ambient_array(&self) -> [f32; 4] {
        self.ambient
    }

    fn light_array(&self, _view_matrix: &glm::Mat4) -> LightArray {
        self.array
    }
}
