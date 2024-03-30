// vulkano_win is deprecated, but if we get an error from it we still have to
// handle it
#![allow(deprecated)]

use std::{error, fmt};
use vulkano::{
    buffer::AllocateBufferError, command_buffer::CommandBufferExecError,
    image::AllocateImageError, library::LoadingError,
    memory::allocator::MemoryAllocatorError,
    pipeline::layout::IntoPipelineLayoutCreateInfoError, sync::HostAccessError,
    VulkanError,
};
use winit::error::OsError;

/// Unified error type
///
/// As of vulkano 0.34, many vulkano functions may return `ValidatedError` which
/// contains either some vulkano error type or a boxed `ValidationError` from
/// Vulkan. These are currently handled by mapping first through
/// `Validated::unwrap` which will panic on the validation errors.
/// The other errors will be converted to `RhError`.
///
/// Some of the lower level vulkano functions may return a boxed
/// `ValidationError` directly. These are currently handled by a standard
/// `unwrap` so will panic.
///
/// Some vulkano error types are very large so are boxed.
#[derive(Debug)]
pub enum RhError {
    RedoFromStart, // Placeholder inspired by Commodore BASIC
    WindowNotFound,
    QueueNotFound,
    DeviceNotFound,
    RetryRecreate,
    RecreateFailed,
    SwapchainOutOfDate,
    AcquireFailed,
    RenderPassError,
    PipelineError,
    FutureFlush(VulkanError),
    FenceError,
    InvalidFile,
    FileTooShort,
    DataNotConverted,
    UnsupportedFormat,
    UnsupportedDepthFormat,
    UnsupportedColourFormat,
    UnsupportedSwapchainFormat,
    VertexShaderError,
    FragmentShaderError,
    IndexTooLarge,
    VertexCountTooLarge,
    IndexCountTooLarge,
    WinitOsError(OsError),
    SerdeYamlError(Box<serde_yaml::Error>),
    StdIoError(std::io::Error),
    TObjLoadError(tobj::LoadError),
    ImageImageError(Box<image::error::ImageError>),
    VkVulkanError(VulkanError),
    VkLoadingError(Box<LoadingError>),
    VkCommandBufferExecError(Box<CommandBufferExecError>),
    VkMemoryAllocatorError(MemoryAllocatorError),
    VkAllocateBufferError(AllocateBufferError),
    VkAllocateImageError(AllocateImageError),
    VkHostAccessError(HostAccessError),
    VkIntoPipelineLayoutCreateInfoError(IntoPipelineLayoutCreateInfoError),
    GltfError(Box<gltf::Error>),
    ImportError(crate::mesh_import::ImportError),
}

impl error::Error for RhError {}

impl fmt::Display for RhError {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::RedoFromStart => {
                write!(f, "?REDO FROM START (don't know what happened)")
            }
            Self::WindowNotFound => write!(f, "winit window not found"),
            Self::QueueNotFound => write!(f, "Vulkan queue not found"),
            Self::DeviceNotFound => {
                write!(f, "no Vulkan 1.3 compatible device found")
            }
            Self::RetryRecreate => write!(f, "retry recreating the swapchain"),
            Self::RecreateFailed => write!(f, "swapchain recreation failed"),
            Self::SwapchainOutOfDate => {
                write!(f, "swapchain out of date, try recreating")
            }
            Self::AcquireFailed => write!(f, "acquire next image failed"),
            Self::RenderPassError => write!(f, "RenderPass error"),
            Self::PipelineError => write!(f, "Pipeline error"),
            Self::FutureFlush(e) => write!(f, "could not flush future: {e}"),
            Self::FenceError => write!(f, "could not get fence"),
            Self::InvalidFile => write!(f, "invalid file"),
            Self::FileTooShort => write!(f, "file too short"),
            Self::DataNotConverted => {
                write!(f, "data could not convert to a valid value")
            }
            Self::UnsupportedFormat => write!(f, "format is not supported"),
            Self::UnsupportedDepthFormat => {
                write!(f, "depth format is not supported")
            }
            Self::UnsupportedColourFormat => {
                write!(f, "colour format is not supported")
            }
            Self::UnsupportedSwapchainFormat => {
                write!(f, "swapchain format is not supported")
            }
            Self::VertexShaderError => write!(f, "vertex shader error"),
            Self::FragmentShaderError => write!(f, "fragment shader error"),
            Self::IndexTooLarge => write!(f, "index does not fit in 16 bits"),
            Self::IndexCountTooLarge => {
                write!(f, "index count does not fit in 32 bits")
            }
            Self::VertexCountTooLarge => {
                write!(f, "vertex count does not fit in 32 bits")
            }
            Self::WinitOsError(e) => write!(f, "OsError {e}"),
            Self::SerdeYamlError(e) => {
                write!(f, "serde_yaml::Error: {e}")
            }
            Self::StdIoError(e) => write!(f, "std::io::Error: {}", e.kind()),
            Self::TObjLoadError(e) => write!(f, "tobj crate LoadError: {e}"),
            Self::ImageImageError(e) => {
                write!(f, "image crate ImageError: {e}")
            }
            Self::VkVulkanError(e) => write!(f, "vulkano VulkanError: {e}"),
            Self::VkLoadingError(e) => write!(f, "vulkano LoadingError: {e}"),
            Self::VkCommandBufferExecError(e) => {
                write!(f, "vulkano CommandBufferExecError: {e}")
            }
            Self::VkMemoryAllocatorError(e) => {
                write!(f, "vulkano MemoryAllocatorError: {e}")
            }
            Self::VkAllocateBufferError(e) => {
                write!(f, "vulkano AllocateBufferError: {e}")
            }
            Self::VkAllocateImageError(e) => {
                write!(f, "vulkano AllocateImageError: {e}")
            }
            Self::VkHostAccessError(e) => {
                write!(f, "vulkano HostAccessError: {e}")
            }
            Self::VkIntoPipelineLayoutCreateInfoError(e) => {
                write!(f, "vulkano IntoPipelineLayoutCreateInfoError: {e}")
            }
            Self::GltfError(e) => {
                write!(f, "gltf Error: {e}")
            }
            Self::ImportError(e) => write!(f, "import error: {e}"),
        }
    }
}

impl From<serde_yaml::Error> for RhError {
    fn from(e: serde_yaml::Error) -> Self {
        Self::SerdeYamlError(Box::new(e))
    }
}

impl From<std::io::Error> for RhError {
    fn from(e: std::io::Error) -> Self {
        Self::StdIoError(e)
    }
}

impl From<tobj::LoadError> for RhError {
    fn from(e: tobj::LoadError) -> Self {
        Self::TObjLoadError(e)
    }
}

impl From<image::error::ImageError> for RhError {
    fn from(e: image::error::ImageError) -> Self {
        Self::ImageImageError(Box::new(e))
    }
}

impl From<LoadingError> for RhError {
    fn from(e: LoadingError) -> Self {
        Self::VkLoadingError(Box::new(e))
    }
}

impl From<VulkanError> for RhError {
    fn from(e: VulkanError) -> Self {
        Self::VkVulkanError(e)
    }
}

impl From<CommandBufferExecError> for RhError {
    fn from(e: CommandBufferExecError) -> Self {
        Self::VkCommandBufferExecError(Box::new(e))
    }
}

impl From<MemoryAllocatorError> for RhError {
    fn from(e: MemoryAllocatorError) -> Self {
        Self::VkMemoryAllocatorError(e)
    }
}

impl From<AllocateBufferError> for RhError {
    fn from(e: AllocateBufferError) -> Self {
        Self::VkAllocateBufferError(e)
    }
}

impl From<HostAccessError> for RhError {
    fn from(e: HostAccessError) -> Self {
        Self::VkHostAccessError(e)
    }
}

impl From<AllocateImageError> for RhError {
    fn from(e: AllocateImageError) -> Self {
        Self::VkAllocateImageError(e)
    }
}

impl From<IntoPipelineLayoutCreateInfoError> for RhError {
    fn from(e: IntoPipelineLayoutCreateInfoError) -> Self {
        Self::VkIntoPipelineLayoutCreateInfoError(e)
    }
}

impl From<OsError> for RhError {
    fn from(e: OsError) -> Self {
        Self::WinitOsError(e)
    }
}

impl From<crate::mesh_import::ImportError> for RhError {
    fn from(e: crate::mesh_import::ImportError) -> Self {
        Self::ImportError(e)
    }
}
