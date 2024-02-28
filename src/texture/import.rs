use std::sync::Arc;

use super::dds;
use crate::rh_error::RhError;
use image::{io::Reader, RgbaImage};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CopyBufferToImageInfo},
    format::Format,
    image::{view::ImageView, Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter,
    },
    Validated,
};

// Free functions which can be called in place of a caching `TextureManager`

/// Loads a png or dds file into an `ImageView`
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn load<T>(
    filename: &str,
    mem_allocator: Arc<(dyn MemoryAllocator)>,
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<Arc<ImageView>, RhError> {
    let extension = std::path::Path::new(filename)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .unwrap_or_default();
    if extension == "dds" {
        dds::load(filename, mem_allocator, cbb)
    } else {
        load_png(filename, mem_allocator, cbb)
    }
}

/// Loads a png file into an `ImageView`
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn load_png<T>(
    filename: &str,
    mem_allocator: Arc<(dyn MemoryAllocator)>,
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<Arc<ImageView>, RhError> {
    let image_data = Reader::open(filename)?.decode()?.into_rgba8();
    load_png_impl(image_data, mem_allocator, cbb)
}

/// Loads a dds file into an `ImageView`
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn load_default<T>(
    mem_allocator: Arc<(dyn MemoryAllocator)>,
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<Arc<ImageView>, RhError> {
    let bytes = include_bytes!("default.png");
    let image_data = image::load_from_memory(bytes)?.into_rgba8();
    load_png_impl(image_data, mem_allocator, cbb)
}

/// Private helper for doing the png load
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
fn load_png_impl<T>(
    image_data: RgbaImage,
    mem_allocator: Arc<(dyn MemoryAllocator)>,
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<Arc<ImageView>, RhError> {
    let (width, height) = image_data.dimensions();
    let extent = [width, height, 1];
    let format = Format::R8G8B8A8_SRGB;

    // Create vulkano data buffer with the image
    let data_buffer = Buffer::from_iter(
        mem_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        image_data.into_raw(),
    )
    .map_err(Validated::unwrap)?;

    // Create a vulkano image
    let image = Image::new(
        mem_allocator,
        ImageCreateInfo {
            format,
            extent,
            array_layers: 1, // Default but listed here for clarity
            mip_levels: 1,   // Default but listed here for clarity
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .map_err(Validated::unwrap)?;

    // Record commands to transfer data from the buffer to the image
    cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
        data_buffer,
        image.clone(),
    ))
    .unwrap(); // This is a Box<ValidationError>

    // Create a vulkano image view
    let image_view =
        ImageView::new_default(image).map_err(Validated::unwrap)?;

    Ok(image_view)
}
