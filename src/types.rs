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

#[derive(Copy, Clone)]
pub struct Submesh {
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub vertex_count: i32,
    pub material_id: Option<usize>,
}

pub mod vertex {
    // Standard vertex format is to have two streams:
    // Position = positions only
    // Interleaved = all other data, interleaved
    use bytemuck::{Pod, Zeroable};
    use vulkano::pipeline::graphics::vertex_input::Vertex;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
    pub struct Interleaved {
        #[format(R32G32B32_SFLOAT)]
        pub normal: [f32; 3],
        #[format(R32G32_SFLOAT)]
        pub tex_coord: [f32; 2],
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
    pub struct Position {
        #[format(R32G32B32_SFLOAT)]
        pub position: [f32; 3],
    }

    pub struct VertexBuffers {
        pub positions: Vec<Position>,
        pub interleaved: Vec<Interleaved>,
        pub indices: Vec<u16>,
    }

    impl VertexBuffers {
        #[must_use]
        pub fn new() -> Self {
            Self {
                positions: Vec::new(),
                interleaved: Vec::new(),
                indices: Vec::new(),
            }
        }
    }

    impl Default for VertexBuffers {
        fn default() -> Self {
            Self::new()
        }
    }
}

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
