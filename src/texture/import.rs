use super::dds;
use crate::{rh_error::RhError, types::TextureView};
use image::io::Reader;
use log::info;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    memory::allocator::MemoryAllocator,
};

// Free functions which can be called in place of a caching `TextureManager`

/// # Errors
/// May return `RhError`
pub fn load<T>(
    filename: &str,
    mem_allocator: &(impl MemoryAllocator + ?Sized),
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<TextureView, RhError> {
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

/// # Errors
/// May return `RhError`
pub fn load_png<T>(
    filename: &str,
    mem_allocator: &(impl MemoryAllocator + ?Sized),
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<TextureView, RhError> {
    let image = Reader::open(filename)?.decode()?.into_rgba8();
    let (width, height) = image.dimensions();
    info!("{filename} texture loaded w: {width}, h: {height}");
    let dimensions = ImageDimensions::Dim2d {
        width,
        height,
        array_layers: 1,
    };
    let texture_format = Format::R8G8B8A8_SRGB;
    let imm_image = ImmutableImage::from_iter(
        mem_allocator,
        image.into_raw(),
        dimensions,
        MipmapsCount::One,
        texture_format,
        cbb,
    )?;
    Ok(ImageView::new_default(imm_image)?)
}

/// # Errors
/// May return `RhError`
pub fn load_default<T>(
    mem_allocator: &(impl MemoryAllocator + ?Sized),
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<TextureView, RhError> {
    let bytes = include_bytes!("default.png");
    let image = image::load_from_memory(bytes)?.into_rgba8();
    let (width, height) = image.dimensions();
    let dimensions = ImageDimensions::Dim2d {
        width,
        height,
        array_layers: 1,
    };
    let texture_format = Format::R8G8B8A8_SRGB;
    let imm_image = ImmutableImage::from_iter(
        mem_allocator,
        image.into_raw(),
        dimensions,
        MipmapsCount::One,
        texture_format,
        cbb,
    )?;
    Ok(ImageView::new_default(imm_image)?)
}
