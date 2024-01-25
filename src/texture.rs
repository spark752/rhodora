use crate::dds;
use crate::rh_error::RhError;
use crate::types::TextureView;
use ahash::AHashMap;
use image::io::Reader;
#[allow(unused_imports)]
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    memory::allocator::MemoryAllocator,
    memory::allocator::StandardMemoryAllocator,
};

/// Texture caching in a multithread friendly way.
/// Uses `ahash::AHashMap` in place of `std::collections::HashMap` only because
/// ahash is already a dependency for egui integration.
/// This struct may be sent across threads, so the cache is wrapped in a
/// `parking_lot::Mutex`. This was selected over `std::Mutex` only because
/// Vulkano and other packages already depend on it.
pub struct Manager {
    mem_allocator: Arc<StandardMemoryAllocator>,
    cache: Mutex<AHashMap<String, TextureView>>,
}

impl Manager {
    pub fn new(mem_allocator: Arc<StandardMemoryAllocator>) -> Self {
        Self {
            mem_allocator,
            // Reserve space to perhaps avoid some realloc/rehash.
            cache: Mutex::new(AHashMap::with_capacity(16)),
        }
    }

    /// Caches textures for improved performance. Loads default texture if
    /// filename is empty.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load<T>(
        &self,
        filename: &str,
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<TextureView, RhError> {
        // This is a potentially long critical section, but probably only one
        // thread is actually loading anyway and the Mutex is just for safety.
        let mut cache = self.cache.lock();

        if let Some(texture_view) = cache.get(filename) {
            info!("Texture cache hit: {}", filename);
            Ok(texture_view.clone())
        } else {
            info!("Texture cache miss: {}", filename);
            let texture_view = if filename.is_empty() {
                load_default(&self.mem_allocator, cbb)
            } else {
                load(filename, &self.mem_allocator, cbb)
            }?;
            cache.insert(filename.to_string(), texture_view.clone());
            drop(cache); // Probably makes no difference but makes clippy happy
            Ok(texture_view)
        }
    }
}

// Free functions which can be called in place of a caching `TextureManager`.

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
