// This file is based on egui_winit_vulkano which is
// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use crate::{memory::Memory, rh_error::RhError, texture, util};
use std::sync::Arc;
use vulkano::{
    command_buffer::PrimaryCommandBufferAbstract, device::Queue, image::Image,
    Validated,
};

/// Creates a vulkano `Image` from some byte data in RGBA format. This creates
/// AND executes a command buffer for this single texture, which may be
/// inefficient.
///
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
) -> Result<Arc<Image>, RhError> {
    // Create command buffer builder
    let mut cbb =
        util::create_primary_cbb(&memory.command_buffer_allocator, &queue)
            .map_err(Validated::unwrap)?;

    // Use a rhodora texture function for most of the work
    let image = texture::image_from_bytes(
        byte_data,
        dimensions,
        memory.memory_allocator.clone(),
        &mut cbb,
    )?;

    // Build and execute the command queue
    let _fut = cbb.build().map_err(Validated::unwrap)?.execute(queue)?;

    // Return an `Image` (not an `ImageView`)
    Ok(image)
}

/// Creates a vulkano `Image` from a .png file which is already in memory.
/// Converts the image data to RGBA if necessary, then calls
/// `immutable_texture_from_bytes` to do the rest.
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
pub fn immutable_texture_from_file(
    memory: &Memory,
    queue: Arc<Queue>,
    file_bytes: &[u8],
) -> Result<Arc<Image>, RhError> {
    let image_data = image::load_from_memory(file_bytes)?.into_rgba8();
    immutable_texture_from_bytes(
        memory,
        queue,
        &image_data,
        image_data.dimensions().into(),
    )
}
