use crate::rh_error::RhError;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::AutoCommandBufferBuilder,
    memory::allocator::{
        AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::graphics::vertex_input::Vertex,
    Validated,
};

/// Buffers that pass vertex data to the GPU. Note that `DeviceVertexBuffers`
/// uses 16 bit indices but is generic for any interleaved vertex format.
pub struct DeviceVertexBuffers<T> {
    pub indices: Subbuffer<[u16]>,
    pub interleaved: Subbuffer<[T]>,
}

impl<T: Vertex> DeviceVertexBuffers<T> {
    /// Creates a new `DeviceVertexBuffers` object storing the specified
    /// vertex format which must implement the Vulkano `Vertex` trait
    ///
    /// # Errors
    /// May return `RhError`
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<U>(
        _cbb: &mut AutoCommandBufferBuilder<U>, // See below
        mem_allocator: Arc<StandardMemoryAllocator>, // Not a reference
        index_buff: Vec<u16>,                   // Not a reference
        inter_buff: Vec<T>,                     // Not a reference
    ) -> Result<Self, RhError> {
        // Create commands to send the buffers to the GPU
        // FIXME The first parameter gave access to a command queue that
        // was used when creating DeviceLocalBuffers that would then be sent
        // over to the GPU. But vulkano removed the function that did that
        // so this temporary implementation is used instead.
        let indices = Buffer::from_iter(
            mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            index_buff,
        )
        .map_err(Validated::unwrap)?;
        let interleaved = Buffer::from_iter(
            mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            inter_buff,
        )
        .map_err(Validated::unwrap)?;
        Ok(Self {
            indices,
            interleaved,
        })
    }
}
