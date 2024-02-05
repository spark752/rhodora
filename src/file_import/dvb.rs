use crate::{
    rh_error::RhError,
    vertex::{BaseBuffers, InterBuffers, InterVertexTrait, Position},
};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::AutoCommandBufferBuilder,
    memory::allocator::{
        AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::graphics::vertex_input::Vertex,
};

/// Buffers that pass vertex data to the GPU. Note that `DeviceVertexBuffers`
/// is generic over any sized type for any sort of vertices but the `new`
/// method is restricted to the Rhodora vertex formats.
pub struct DeviceVertexBuffers<T> {
    pub positions: Subbuffer<[Position]>,
    pub interleaved: Subbuffer<[T]>,
    pub indices: Subbuffer<[u16]>,
}

impl<T: Vertex + InterVertexTrait> DeviceVertexBuffers<T> {
    /// Note that `DeviceVertexBuffers` is generic over any sized type
    /// for any sort of vertices but this method requires a type with the
    /// Vulkano `Vertex` trait and the Rhodora `InterVertexTrait` used by other
    /// buffer types.
    ///
    /// # Errors
    /// May return `RhError`
    #[allow(clippy::needless_pass_by_value)]
    pub fn new<U>(
        _cbb: &mut AutoCommandBufferBuilder<U>, // See below
        mem_allocator: Arc<StandardMemoryAllocator>, // Not a reference
        vb_base: BaseBuffers,                   // Not a reference
        vb_inter: InterBuffers<T>,              // Not a reference
    ) -> Result<Self, RhError> {
        // Create commands to send the buffers to the GPU
        // FIXME The first parameter gave access to a command queue that
        // was used when creating DeviceLocalBuffers that would then be sent
        // over to the GPU. But vulkano removed the function that did that
        // so this temporary implementation is used instead.
        let positions = Buffer::from_iter(
            &mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vb_base.positions,
        )?;
        let interleaved = Buffer::from_iter(
            &mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vb_inter.interleaved,
        )?;
        let indices = Buffer::from_iter(
            &mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vb_base.indices,
        )?;
        Ok(Self {
            positions,
            interleaved,
            indices,
        })
    }
}
