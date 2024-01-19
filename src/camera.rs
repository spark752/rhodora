use crate::types::CameraTrait;
use nalgebra_glm as glm;

const NEAR_CLIP_METERS: f32 = 0.1;
const FAR_CLIP_METERS: f32 = 100.0;

#[derive(Debug, Copy, Clone)]
pub struct CameraProperties {
    pub aspect_ratio: f32,
    pub fovy: f32,
    pub position: glm::Vec3,
    pub target: glm::Vec3,
}

impl Default for CameraProperties {
    fn default() -> Self {
        Self {
            aspect_ratio: 16.0f32 / 9.0f32,
            fovy: 0.471f32,
            position: glm::vec3(0.0f32, 0.0f32, 0.0f32),
            target: glm::vec3(0.0f32, 1.0f32, 0.0f32),
        }
    }
}

/// The projection matrix depends on both fovy and aspect ratio, so both are
/// stored so that a caller can change one without having to know the other.
/// The view matrix depends on both position and target, so both are stored
/// so that a caller can change one without having to know the other.
#[derive(Debug, Copy, Clone)]
pub struct Camera {
    aspect_ratio: f32,
    fovy: f32,
    position: glm::Vec3,
    target: glm::Vec3,
    view: glm::Mat4,
    proj: glm::Mat4,
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(CameraProperties::default())
    }
}

/// Rendering requires access to the matrices as arrays, implemented as this
/// trait.
impl CameraTrait for Camera {
    fn view_matrix(&self) -> glm::Mat4 {
        self.view
    }

    fn proj_matrix(&self) -> glm::Mat4 {
        self.proj
    }
}

impl Camera {
    pub fn new(properties: CameraProperties) -> Self {
        Self {
            aspect_ratio: properties.aspect_ratio,
            fovy: properties.fovy,
            position: properties.position,
            target: properties.target,
            view: Self::build_view(&properties.position, &properties.target),
            proj: Self::build_proj(properties.aspect_ratio, properties.fovy),
        }
    }

    pub fn aspect_ratio(&mut self, aspect_ratio: f32) {
        self.proj = Self::build_proj(aspect_ratio, self.fovy);
        self.aspect_ratio = aspect_ratio;
    }

    pub fn zoom(&mut self, fovy: f32) {
        self.proj = Self::build_proj(self.aspect_ratio, fovy);
        self.fovy = fovy;
    }

    pub fn position(&mut self, position: &glm::Vec3) {
        self.view = Self::build_view(position, &self.target);
        self.position = *position;
    }

    pub fn target(&mut self, target: &glm::Vec3) {
        self.view = Self::build_view(&self.position, target);
        self.target = *target;
    }

    pub fn update_view(&mut self, position: glm::Vec3, target: glm::Vec3) {
        self.view = Self::build_view(&position, &target);
        self.position = position;
        self.target = target;
    }

    /// Calculate Model View matrix for a given Model matrix
    pub fn mv(&self, m: &glm::Mat4) -> glm::Mat4 {
        self.view * m
    }

    fn build_proj(aspect_ratio: f32, fovy: f32) -> glm::Mat4 {
        glm::perspective_lh_zo(
            aspect_ratio,
            fovy,
            NEAR_CLIP_METERS,
            FAR_CLIP_METERS,
        )
    }

    fn build_view(position: &glm::Vec3, target: &glm::Vec3) -> glm::Mat4 {
        // Need combination of left hand and down vector to get correct
        // transformation for Vulkan
        glm::look_at_lh(position, target, &glm::vec3(0.0, 0.0, -1.0))
    }
}
