use std::sync::Arc;
use vulkano::{
    command_buffer::allocator::{
        StandardCommandBufferAllocator,
        StandardCommandBufferAllocatorCreateInfo,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::Device,
    memory::allocator::StandardMemoryAllocator,
};

// From Vulkano docs
// StandardMemoryAllocator: "Standard memory allocator intended as a global and
// general-puprose allocator... not to be used when allocations need to be made
// very frequently (say, once or more per frame.) For that purpose, use
// FastMemoryAllocator"
// StandardDescriptorSetAllocator: "The intended way to use this allocator is
// to have one that is used globally for the duration of the program, in order
// to avoid creating and destroying DescriptorPools... Alternatively, you can
// have one locally on a thread for the duration of the thread."
// StandardCommandBufferAllocator: "The intended way to use this allocator is
// to have one that is used globally for the duration of the program, in order
// to avoid creating and destroying CommandPools... Alternatively, you can have
// one locally on a thread for the duration of the thread."

pub struct Memory {
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub set_allocator: StandardDescriptorSetAllocator,
    pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Memory {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            memory_allocator: Arc::new(StandardMemoryAllocator::new_default(
                device.clone(),
            )),
            set_allocator: StandardDescriptorSetAllocator::new(device.clone()),
            command_buffer_allocator: Arc::new(
                StandardCommandBufferAllocator::new(
                    device.clone(),
                    StandardCommandBufferAllocatorCreateInfo::default(),
                ),
            ),
        }
    }
}
