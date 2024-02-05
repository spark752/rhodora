use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferExecFuture},
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Device,
    image::{view::ImageView, AttachmentImage, ImmutableImage, SwapchainImage},
    sync::future::{FenceSignalFuture, NowFuture},
};
use winit::event::{ElementState, VirtualKeyCode};

pub type TextureView = Arc<ImageView<ImmutableImage>>;
pub type AttachmentView = Arc<ImageView<AttachmentImage>>;
pub type SwapchainView = Arc<ImageView<SwapchainImage>>;
pub type TransferFuture = FenceSignalFuture<CommandBufferExecFuture<NowFuture>>;

pub struct RenderFormat {
    pub colour_format: vulkano::format::Format,
    pub depth_format: vulkano::format::Format,
    pub sample_count: vulkano::image::SampleCount,
}

/// A way to pass the Vulkan device and things that are often needed with it.
/// Does not include a generic memory allocator so that either Standard or Fast
/// allocators can be used.
pub struct DeviceAccess<'a, T> {
    pub device: Arc<Device>,
    pub set_allocator: &'a StandardDescriptorSetAllocator,
    pub cbb: &'a mut AutoCommandBufferBuilder<T>,
}

/// Trait for something that handles keyboard input events
pub trait KeyboardHandler {
    fn input(&mut self, keycode: VirtualKeyCode, state: ElementState);
}

/// Trait for camera matrices as arrays, needed for rendering
pub trait CameraTrait {
    fn view_matrix(&self) -> glm::Mat4;
    fn proj_matrix(&self) -> glm::Mat4;
}

/// Maximum joints for a skinned mesh. You can't actually change this constant
/// without also changing the value in the shader macro.
pub const MAX_JOINTS: usize = 32;
