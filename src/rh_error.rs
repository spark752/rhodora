use std::{error, fmt};
use vulkano::{
    buffer::BufferError,
    command_buffer::{
        BuildError, CommandBufferBeginError, CommandBufferExecError, CopyError,
        PipelineExecutionError, RenderPassError,
    },
    descriptor_set::DescriptorSetCreationError,
    device::{physical::PhysicalDeviceError, DeviceCreationError},
    image::immutable::ImmutableImageCreationError,
    image::sys::ImageError,
    image::view::ImageViewCreationError,
    instance::InstanceCreationError,
    library::LoadingError,
    memory::allocator::AllocationCreationError,
    pipeline::graphics::GraphicsPipelineCreationError,
    sampler::SamplerCreationError,
    shader::ShaderCreationError,
    swapchain::SwapchainCreationError,
    sync::FlushError,
    VulkanError,
};

// Some vulkano error types are very large so are boxed
#[derive(Debug)]
pub enum RhError {
    WindowNotFound,
    QueueNotFound,
    DeviceNotFound,
    RetryRecreate,
    RecreateFailed,
    SwapchainOutOfDate,
    AcquireFailed,
    RenderPassError,
    PipelineError,
    FutureFlush,
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
    SerdeYamlError(Box<serde_yaml::Error>),
    StdIoError(std::io::Error),
    TObjLoadError(tobj::LoadError),
    ImageImageError(Box<image::error::ImageError>),
    VkWinCreationError(Box<vulkano_win::CreationError>),
    VkVulkanError(VulkanError),
    VkLoadingError(Box<LoadingError>),
    VkInstanceCreationError(Box<InstanceCreationError>),
    VkDeviceCreationError(Box<DeviceCreationError>),
    VkImageError(Box<ImageError>),
    VkImageViewCreationError(Box<ImageViewCreationError>),
    VkImmutableImageCreationError(Box<ImmutableImageCreationError>),
    VkDescriptorSetCreationError(Box<DescriptorSetCreationError>),
    VkFlushError(Box<FlushError>),
    VkBuildError(BuildError),
    VkCommandBufferExecError(Box<CommandBufferExecError>),
    VkAllocationCreationError(Box<AllocationCreationError>),
    VkPhysicalDeviceError(Box<PhysicalDeviceError>),
    VkSwapchainCreationError(Box<SwapchainCreationError>),
    VkCommandBufferBeginError(Box<CommandBufferBeginError>),
    VkShaderCreationError(Box<ShaderCreationError>),
    VkGraphicsPipelineCreationError(Box<GraphicsPipelineCreationError>),
    VkSamplerCreationError(Box<SamplerCreationError>),
    VkPipelineExecutionError(Box<PipelineExecutionError>),
    VkRenderPassError(Box<RenderPassError>),
    VkCopyError(Box<CopyError>),
    VkBufferError(Box<BufferError>),
}

impl error::Error for RhError {}

impl fmt::Display for RhError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
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
            Self::FutureFlush => write!(f, "could not flush future"),
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
                write!(f, "colopur format is not supported")
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
            Self::SerdeYamlError(e) => {
                write!(f, "serde_yaml::Error: {e}")
            }
            Self::StdIoError(e) => write!(f, "std::io::Error: {}", e.kind()),
            Self::TObjLoadError(e) => write!(f, "tobj crate LoadError: {e}"),
            Self::ImageImageError(e) => {
                write!(f, "image crate ImageError: {e}")
            }
            Self::VkWinCreationError(e) => {
                write!(f, "vulkano_win CreationError: {e}")
            }
            Self::VkVulkanError(e) => write!(f, "vulkano VulkanError: {e}"),
            Self::VkLoadingError(e) => write!(f, "vulkano LoadingError: {e}"),
            Self::VkInstanceCreationError(e) => {
                write!(f, "vulkano InstanceCreationError: {e}")
            }
            Self::VkDeviceCreationError(e) => {
                write!(f, "vulkano DeviceCreationError: {e}")
            }
            Self::VkImageError(e) => write!(f, "vulkano ImageError: {e}"),
            Self::VkImageViewCreationError(e) => {
                write!(f, "vulkano ImageViewCreationError: {e}")
            }
            Self::VkImmutableImageCreationError(e) => {
                write!(f, "vulkano ImmutableImageCreationError: {e}")
            }
            Self::VkDescriptorSetCreationError(e) => {
                write!(f, "vulkano DescriptorSetCreationError: {e}")
            }
            Self::VkFlushError(e) => write!(f, "vulkano FlushError: {e}"),
            Self::VkBuildError(e) => write!(f, "vulkano BuildError: {e}"),
            Self::VkCommandBufferExecError(e) => {
                write!(f, "vulkano CommandBufferExecError: {e}")
            }
            Self::VkAllocationCreationError(e) => {
                write!(f, "vulkano AllocationCreationError: {e}")
            }
            Self::VkPhysicalDeviceError(e) => {
                write!(f, "vulkano PhysicalDeviceError: {e}")
            }
            Self::VkSwapchainCreationError(e) => {
                write!(f, "vulkano SwapchainCreationError: {e}")
            }
            Self::VkCommandBufferBeginError(e) => {
                write!(f, "vulkano CommandBufferBeginError: {e}")
            }
            Self::VkShaderCreationError(e) => {
                write!(f, "vulkano ShaderCreationError: {e}")
            }
            Self::VkGraphicsPipelineCreationError(e) => {
                write!(f, "vulkano GraphicsPipelineCreationError: {e}")
            }
            Self::VkSamplerCreationError(e) => {
                write!(f, "vulkano SamplerCreationError: {e}")
            }
            Self::VkPipelineExecutionError(e) => {
                write!(f, "vulkano PipelineExecutionError: {e}")
            }
            Self::VkRenderPassError(e) => {
                write!(f, "vulkano RenderPassError: {e}")
            }
            Self::VkCopyError(e) => {
                write!(f, "vulkano CopyError: {e}")
            }
            Self::VkBufferError(e) => {
                write!(f, "vulkano BufferError: {e}")
            }
        }
    }
}

impl From<serde_yaml::Error> for RhError {
    fn from(e: serde_yaml::Error) -> Self {
        RhError::SerdeYamlError(Box::new(e))
    }
}

impl From<std::io::Error> for RhError {
    fn from(e: std::io::Error) -> Self {
        RhError::StdIoError(e)
    }
}

impl From<tobj::LoadError> for RhError {
    fn from(e: tobj::LoadError) -> Self {
        RhError::TObjLoadError(e)
    }
}

impl From<image::error::ImageError> for RhError {
    fn from(e: image::error::ImageError) -> Self {
        RhError::ImageImageError(Box::new(e))
    }
}

impl From<LoadingError> for RhError {
    fn from(e: LoadingError) -> Self {
        RhError::VkLoadingError(Box::new(e))
    }
}

impl From<InstanceCreationError> for RhError {
    fn from(e: InstanceCreationError) -> Self {
        RhError::VkInstanceCreationError(Box::new(e))
    }
}

impl From<DeviceCreationError> for RhError {
    fn from(e: DeviceCreationError) -> Self {
        RhError::VkDeviceCreationError(Box::new(e))
    }
}

impl From<ImageError> for RhError {
    fn from(e: ImageError) -> Self {
        RhError::VkImageError(Box::new(e))
    }
}

impl From<vulkano_win::CreationError> for RhError {
    fn from(e: vulkano_win::CreationError) -> Self {
        RhError::VkWinCreationError(Box::new(e))
    }
}

impl From<ImageViewCreationError> for RhError {
    fn from(e: ImageViewCreationError) -> Self {
        RhError::VkImageViewCreationError(Box::new(e))
    }
}

impl From<VulkanError> for RhError {
    fn from(e: VulkanError) -> Self {
        RhError::VkVulkanError(e)
    }
}

impl From<ImmutableImageCreationError> for RhError {
    fn from(e: ImmutableImageCreationError) -> Self {
        RhError::VkImmutableImageCreationError(Box::new(e))
    }
}

impl From<DescriptorSetCreationError> for RhError {
    fn from(e: DescriptorSetCreationError) -> Self {
        RhError::VkDescriptorSetCreationError(Box::new(e))
    }
}

impl From<FlushError> for RhError {
    fn from(e: FlushError) -> Self {
        RhError::VkFlushError(Box::new(e))
    }
}

impl From<BuildError> for RhError {
    fn from(e: BuildError) -> Self {
        RhError::VkBuildError(e)
    }
}

impl From<CommandBufferExecError> for RhError {
    fn from(e: CommandBufferExecError) -> Self {
        RhError::VkCommandBufferExecError(Box::new(e))
    }
}

impl From<AllocationCreationError> for RhError {
    fn from(e: AllocationCreationError) -> Self {
        RhError::VkAllocationCreationError(Box::new(e))
    }
}

impl From<PhysicalDeviceError> for RhError {
    fn from(e: PhysicalDeviceError) -> Self {
        RhError::VkPhysicalDeviceError(Box::new(e))
    }
}

impl From<SwapchainCreationError> for RhError {
    fn from(e: SwapchainCreationError) -> Self {
        RhError::VkSwapchainCreationError(Box::new(e))
    }
}

impl From<CommandBufferBeginError> for RhError {
    fn from(e: CommandBufferBeginError) -> Self {
        RhError::VkCommandBufferBeginError(Box::new(e))
    }
}

impl From<ShaderCreationError> for RhError {
    fn from(e: ShaderCreationError) -> Self {
        RhError::VkShaderCreationError(Box::new(e))
    }
}

impl From<GraphicsPipelineCreationError> for RhError {
    fn from(e: GraphicsPipelineCreationError) -> Self {
        RhError::VkGraphicsPipelineCreationError(Box::new(e))
    }
}

impl From<SamplerCreationError> for RhError {
    fn from(e: SamplerCreationError) -> Self {
        RhError::VkSamplerCreationError(Box::new(e))
    }
}

impl From<PipelineExecutionError> for RhError {
    fn from(e: PipelineExecutionError) -> Self {
        RhError::VkPipelineExecutionError(Box::new(e))
    }
}

impl From<RenderPassError> for RhError {
    fn from(e: RenderPassError) -> Self {
        RhError::VkRenderPassError(Box::new(e))
    }
}

impl From<CopyError> for RhError {
    fn from(e: CopyError) -> Self {
        RhError::VkCopyError(Box::new(e))
    }
}

impl From<BufferError> for RhError {
    fn from(e: BufferError) -> Self {
        RhError::VkBufferError(Box::new(e))
    }
}
