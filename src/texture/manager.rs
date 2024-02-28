use super::import;
use crate::rh_error::RhError;
use ahash::AHashMap;
use log::info;
use parking_lot::Mutex;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder, image::view::ImageView,
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
    cache: Mutex<AHashMap<String, Arc<ImageView>>>,
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
    ) -> Result<Arc<ImageView>, RhError> {
        // This is a potentially long critical section, but probably only one
        // thread is actually loading anyway and the Mutex is just for safety.
        let mut cache = self.cache.lock();

        if let Some(texture_view) = cache.get(filename) {
            info!("Texture cache hit: {}", filename);
            Ok(texture_view.clone())
        } else {
            info!("Texture cache miss: {}", filename);
            let texture_view = if filename.is_empty() {
                import::load_default(self.mem_allocator.clone(), cbb)
            } else {
                import::load(filename, self.mem_allocator.clone(), cbb)
            }?;
            cache.insert(filename.to_string(), texture_view.clone());
            drop(cache); // Probably makes no difference but makes clippy happy
            Ok(texture_view)
        }
    }
}
