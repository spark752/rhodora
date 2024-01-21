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
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    device::Queue,
    image::{
        view::ImageView, ImageDimensions, ImageViewAbstract, ImmutableImage,
        MipmapsCount,
    },
};

/// # Errors
/// May return `RhError`
pub fn immutable_texture_from_bytes(
    memory: &Memory,
    queue: Arc<Queue>,
    byte_data: &[u8],
    dimensions: [u32; 2],
    format: vulkano::format::Format,
) -> Result<Arc<dyn ImageViewAbstract + Send + Sync + 'static>, RhError> {
    let vko_dims = ImageDimensions::Dim2d {
        width: dimensions[0],
        height: dimensions[1],
        array_layers: 1,
    };

    let mut cbb = AutoCommandBufferBuilder::primary(
        &memory.command_buffer_allocator,
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )?;
    let texture = ImmutableImage::from_iter(
        &memory.memory_allocator,
        byte_data.iter().copied(),
        vko_dims,
        MipmapsCount::One,
        format,
        &mut cbb,
    )?;
    let _fut = cbb.build()?.execute(queue)?;

    Ok(ImageView::new_default(texture)?)
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
) -> Result<Arc<dyn ImageViewAbstract + Send + Sync + 'static>, RhError> {
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
