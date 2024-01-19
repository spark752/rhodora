use nalgebra_glm as glm;

pub type LightArray = [[f32; 4]; 4];

pub trait PbrLightTrait {
    fn ambient_array(&self) -> [f32; 4];
    fn light_array(&self, view_matrix: &glm::Mat4) -> LightArray;
}

pub struct PbrLights {
    pub ambient: f32,
    pub array: LightArray,
}

impl Default for PbrLights {
    fn default() -> Self {
        Self {
            ambient: 0.1f32,

            // Lights are in VIEW space with intensity in last element
            array: [
                [0.0f32, -5.0f32, 0.0f32, 150.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32],
                [0.0f32, 0.0f32, 0.0f32, 0.0f32],
            ],
        }
    }
}

impl PbrLightTrait for PbrLights {
    fn ambient_array(&self) -> [f32; 4] {
        [0.0f32, 0.0f32, 0.0f32, self.ambient]
    }

    fn light_array(&self, _view_matrix: &glm::Mat4) -> LightArray {
        self.array
    }
}
