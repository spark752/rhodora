// This file is based on egui_winit_vulkano which is
// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::memory::Memory;
use crate::rh_error::RhError;
use image::RgbaImage;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract,
    },
    device::Queue,
    image::{Image, ImageCreateInfo, ImageUsage},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    Validated,
};

/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn immutable_texture_from_bytes(
    memory: &Memory,
    queue: Arc<Queue>,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<Arc<Image>, RhError> {
    let extent = [dimensions[0], dimensions[1], 1];

    // Create command buffer builder
    let mut cbb = AutoCommandBufferBuilder::primary(
        &memory.command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .map_err(Validated::unwrap)?;

    // Create vulkano data buffer from the image data
    let data_buffer = Buffer::from_iter(
        memory.memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        byte_data.iter().copied(),
    )
    .map_err(Validated::unwrap)?;

    // Create a vulkano image
    let image = Image::new(
        memory.memory_allocator.clone(),
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

    // Build and execute the command queue
    let _fut = cbb.build().map_err(Validated::unwrap)?.execute(queue)?;

    // Return as an Image, not an ImageView
    Ok(image)
}

/// # Errors
/// May return `RhError`
///
/// # Panics
/// Panics if the image can not be created
pub fn immutable_texture_from_file(
    memory: &Memory,
    queue: Arc<Queue>,
    file_bytes: &[u8],
    format: vulkano::format::Format,
) -> Result<Arc<Image>, RhError> {
    use image::GenericImageView;

    let img = image::load_from_memory(file_bytes)
        .expect("Failed to load image from bytes");
    let rgba = img.as_rgba8().map_or_else(
        || {
            // Convert rgb to rgba
            let rgb =
                img.as_rgb8().ok_or(RhError::UnsupportedFormat)?.to_owned();
            let mut raw_data = vec![];
            for val in rgb.chunks(3) {
                raw_data.push(val[0]);
                raw_data.push(val[1]);
                raw_data.push(val[2]);
                raw_data.push(255);
            }
            let new_rgba =
                RgbaImage::from_raw(rgb.width(), rgb.height(), raw_data)
                    .ok_or(RhError::UnsupportedFormat)?;
            Ok::<Vec<u8>, RhError>(new_rgba.to_vec())
        },
        |rgba| Ok(rgba.to_owned().to_vec()),
    )?;

    let dimensions = img.dimensions();
    immutable_texture_from_bytes(
        memory,
        queue,
        &rgba,
        dimensions.into(),
        format,
    )
}
