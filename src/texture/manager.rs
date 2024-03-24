use super::import;
use crate::rh_error::RhError;
use ahash::{HashMap, HashMapExt};
use log::info;
use parking_lot::Mutex;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder, image::view::ImageView,
    memory::allocator::StandardMemoryAllocator,
};

/// Texture caching in a multithread friendly way
pub struct Manager {
    mem_allocator: Arc<StandardMemoryAllocator>,
    cache: Mutex<HashMap<String, Arc<ImageView>>>,
}

impl Manager {
    pub fn new(mem_allocator: Arc<StandardMemoryAllocator>) -> Self {
        Self {
            mem_allocator,
            // Reserve space to perhaps avoid some realloc/rehash.
            cache: Mutex::new(HashMap::with_capacity(16)),
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
