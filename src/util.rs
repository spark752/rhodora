/// A module of utility functions
use crate::vk_window::VkWindow;
use crate::{rh_error::RhError, types::TransferFuture};
use nalgebra_glm as glm;
use std::sync::Arc;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderingAttachmentInfo,
        RenderingAttachmentResolveInfo,
    },
    descriptor_set::layout::DescriptorSetLayout,
    device::{physical::PhysicalDevice, Queue},
    format::{Format, FormatFeatures},
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage, SampleCount},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator},
    pipeline::{
        graphics::color_blend::{
            AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState,
            ColorBlendState, ColorComponents,
        },
        GraphicsPipeline, Pipeline,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        ColorSpace, PresentMode, Surface, SurfaceInfo, Swapchain,
        SwapchainCreateInfo,
    },
    sync::GpuFuture,
};
use vulkano::{Validated, VulkanError};

/// # Errors
/// May return `RhError`
pub fn get_layout(
    pipeline: &Arc<GraphicsPipeline>,
    set_number: u32,
) -> Result<&Arc<DescriptorSetLayout>, RhError> {
    pipeline
        .layout()
        .set_layouts()
        .get(set_number as usize)
        .ok_or(RhError::PipelineError)
}

#[must_use]
pub fn alpha_blend_enable() -> ColorBlendState {
    ColorBlendState {
        // VkPipelineColorBlendAttachmentState
        attachments: vec![ColorBlendAttachmentState {
            blend: Some(
                // Set equivalents to srcColorBlendFactor etc. here
                AttachmentBlend {
                    // Disabled is source = 1, destination = 0,
                    // op = Add (but just set blend to None instead)
                    // This is for typical alpha blending:
                    src_color_blend_factor: BlendFactor::SrcAlpha,
                    dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                    color_blend_op: BlendOp::Add,
                    src_alpha_blend_factor: BlendFactor::One,
                    dst_alpha_blend_factor: BlendFactor::Zero,
                    alpha_blend_op: BlendOp::Add,
                },
            ),
            color_write_mask: ColorComponents::all(),
            color_write_enable: true,
        }],
        ..Default::default()
    }
}

/// Creates a rendering target
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn create_target(
    allocator: Arc<dyn MemoryAllocator>,
    dimensions: [u32; 2],
    format: Format,
) -> Result<Arc<ImageView>, RhError> {
    Ok(ImageView::new_default(
        Image::new(
            allocator,
            ImageCreateInfo {
                format,
                extent: [dimensions[0], dimensions[1], 1],
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::SAMPLED
                    | ImageUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .map_err(Validated::unwrap)?,
    )
    .map_err(Validated::unwrap)?)
}

/// Creates a depth buffer
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn create_depth(
    allocator: Arc<dyn MemoryAllocator>,
    dimensions: [u32; 2],
    format: Format,
    samples: SampleCount,
) -> Result<Arc<ImageView>, RhError> {
    Ok(ImageView::new_default(
        Image::new(
            allocator,
            ImageCreateInfo {
                format,
                extent: [dimensions[0], dimensions[1], 1],
                samples,
                usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .map_err(Validated::unwrap)?,
    )
    .map_err(Validated::unwrap)?)
}

/// Creates an MSAA rendering target
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn create_msaa(
    allocator: Arc<dyn MemoryAllocator>,
    dimensions: [u32; 2],
    format: Format,
    samples: SampleCount,
) -> Result<Arc<ImageView>, RhError> {
    Ok(ImageView::new_default(
        Image::new(
            allocator,
            ImageCreateInfo {
                format,
                extent: [dimensions[0], dimensions[1], 1],
                samples,
                usage: ImageUsage::COLOR_ATTACHMENT
                    | ImageUsage::TRANSIENT_ATTACHMENT,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )
        .map_err(Validated::unwrap)?,
    )
    .map_err(Validated::unwrap)?)
}

/// Selects first supported format for depth attachment from an array of
/// candidates
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn find_depth_format(
    physical: &PhysicalDevice,
    candidates: &[Format],
) -> Result<Format, RhError> {
    for candidate in candidates {
        if physical
            .format_properties(*candidate)
            .unwrap() // `Box<ValidationError>`
            .optimal_tiling_features
            .intersects(FormatFeatures::DEPTH_STENCIL_ATTACHMENT)
        {
            return Ok(*candidate);
        }
    }
    Err(RhError::UnsupportedDepthFormat)
}

/// Selects first supported format for colour attachment and sampling from
/// an array of candidates
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn find_colour_format(
    physical: &PhysicalDevice,
    candidates: &[Format],
) -> Result<Format, RhError> {
    for candidate in candidates {
        let features = physical
            .format_properties(*candidate)
            .unwrap() // `Box<ValidationError>`
            .optimal_tiling_features;
        if features.intersects(FormatFeatures::COLOR_ATTACHMENT)
            && features.intersects(FormatFeatures::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            return Ok(*candidate);
        }
    }
    Err(RhError::UnsupportedColourFormat)
}

/// Selects first supported format for swapchain use (compatible with the
/// surface) from an array of candidates
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn find_swapchain_format(
    physical: &PhysicalDevice,
    surface: &Surface,
    candidates: &[Format],
) -> Result<Format, RhError> {
    for candidate in candidates {
        let search = physical
            .surface_formats(surface, SurfaceInfo::default())
            .map_err(Validated::unwrap)?
            .into_iter()
            .find(|&f| f.0 == *candidate && f.1 == ColorSpace::SrgbNonLinear);
        if let Some(f) = search {
            return Ok(f.0);
        }
    }
    Err(RhError::UnsupportedSwapchainFormat)
}

/// # Errors
/// May return `RhError`
pub fn create_primary_cbb(
    cmd_allocator: &StandardCommandBufferAllocator,
    queue: &Queue,
) -> Result<
    AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    Validated<VulkanError>,
> {
    AutoCommandBufferBuilder::primary(
        cmd_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
}

#[must_use]
pub fn choose_present_mode(window: &VkWindow, vsync: bool) -> PresentMode {
    let mode_result = window
        .physical()
        .surface_present_modes(window.surface(), SurfaceInfo::default());
    mode_result.map_or(PresentMode::Fifo, |mut modes| {
        if vsync {
            // Might have to search through the list twice which
            // doesn't work well with iterators and find, but it is
            // a short list so can be collected into a vector.
            /*let mv: Vec<PresentMode> = modes.collect();
            for m in mv.iter() {
                if *m == PresentMode::Mailbox {
                    return *m;
                }
            }
            for m in mv.iter() {
                if *m == PresentMode::FifoRelaxed {
                    return *m;
                }
            }*/
            PresentMode::Fifo
        } else {
            // Immediate mode is the only vsync off mode. It is likely
            // supported but Vulkan doesn't require it.
            modes
                .find(|&m| m == PresentMode::Immediate)
                .unwrap_or(PresentMode::Fifo)
        }
    })
}

/// # Errors
/// May return `RhError`
pub fn create_swapchain(
    window: &VkWindow,
    format: Format,
    present_mode: PresentMode,
) -> Result<(Arc<Swapchain>, Vec<Arc<Image>>), RhError> {
    let (swapchain, images) = {
        let surface_caps = window.surface_caps().map_err(Validated::unwrap)?;
        Swapchain::new(
            window.device().clone(),
            window.surface().clone(),
            SwapchainCreateInfo {
                min_image_count: surface_caps.min_image_count + 1,
                image_format: format,
                image_extent: window.dimensions()?.into(),
                image_color_space: ColorSpace::SrgbNonLinear,
                present_mode,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?
    };
    Ok((swapchain, images))
}

#[must_use]
pub fn attachment_info(
    background: [f32; 4],
    target_image_view: Arc<ImageView>,
    msaa_option: Option<Arc<ImageView>>,
) -> RenderingAttachmentInfo {
    let clear_value = Some(background.into());
    if let Some(msaa_image_view) = msaa_option {
        RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::DontCare,
            clear_value,
            resolve_info: Some(RenderingAttachmentResolveInfo::image_view(
                target_image_view,
            )),
            ..RenderingAttachmentInfo::image_view(msaa_image_view)
        }
    } else {
        RenderingAttachmentInfo {
            load_op: AttachmentLoadOp::Clear,
            store_op: AttachmentStoreOp::Store,
            clear_value,
            ..RenderingAttachmentInfo::image_view(target_image_view)
        }
    }
}

/// Convert a focal length in mm (for a 35mm sensor) to a field of view
#[must_use]
pub fn focal_length_to_fovy(focal_length: f32) -> f32 {
    let f = focal_length.max(1.0f32);
    (12.0f32 / f).atan() * 2.0f32
}

/// Starts a transfer to the GPU by building and executing a command buffer
/// and returning a future to wait on
///
/// # Errors
/// May return `RhError`
pub fn start_transfer(
    cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    queue: Arc<Queue>,
) -> Result<TransferFuture, RhError> {
    Ok(cbb
        .build()
        .map_err(Validated::unwrap)?
        .execute(queue)?
        .then_signal_fence_and_flush()
        .map_err(Validated::unwrap)?)
}

/// Transforms a 3D position using a 4x4 matrix and return as a `glm::Vec3`
#[must_use]
pub fn transform(position: &glm::Vec3, matrix: &glm::Mat4) -> glm::Vec3 {
    let ws = glm::vec4(position.x, position.y, position.z, 1.0f32);
    let vs = matrix * ws;
    glm::vec3(vs.x, vs.y, vs.z)
}

/// Transforms a 3D vector using a 4x4 matrix and return as a `glm::Vec4`
#[must_use]
pub fn transform4(vector: &glm::Vec4, matrix: &glm::Mat4) -> glm::Vec4 {
    matrix * vector
}

/// Creates a set of `ImageView` objects from a set of `Image` objects
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan///
pub fn create_image_views(
    images: &[Arc<Image>],
) -> Result<Vec<Arc<ImageView>>, RhError> {
    let mut ret = Vec::new();
    for image in images {
        ret.push(
            ImageView::new_default(image.clone()).map_err(Validated::unwrap)?,
        );
    }
    Ok(ret)
}
