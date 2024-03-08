use log::{info, trace};
use vulkano::sync::GpuFuture;

pub struct FrameControl {
    pub fences: Vec<Option<Box<dyn GpuFuture>>>,
    pub fence_index: usize,
    pub previous_fence_index: usize,
    pub frame_count: usize,
    pub frames_in_flight: usize,
}

impl FrameControl {
    #[must_use]
    pub fn new(frames_in_flight: usize) -> Self {
        if frames_in_flight != 1 {
            info!("frames_in_flight not currently supported, setting to 1");
        }
        let frames_in_flight = 1;

        // The vec! macro doesn't work because the type doesn't implement
        // clone
        let mut fences = Vec::new();
        for _ in 0..frames_in_flight {
            fences.push(None);
        }
        info!("Frames in flight set to {}", frames_in_flight);

        Self {
            fences,
            fence_index: 0,
            previous_fence_index: 0,
            frame_count: 0,
            frames_in_flight,
        }
    }

    #[allow(clippy::modulo_one)] // FRAMES_IN_FLIGHT might equal 1
    pub fn update(&mut self, new_fence: Option<Box<dyn GpuFuture>>) {
        self.fences[self.fence_index] = new_fence;
        self.previous_fence_index = self.fence_index;
        self.fence_index = (self.fence_index + 1) % self.frames_in_flight;
        self.frame_count += 1;
        trace!(
            "update fence_index={} previous_fence_index={} frame_count={}",
            self.previous_fence_index,
            self.fence_index,
            self.frame_count
        );
    }

    /// Takes the value of the current fence, returing it and replacing it with
    /// `None`
    pub fn take(&mut self) -> Option<Box<dyn GpuFuture>> {
        self.fences[self.fence_index].take()
    }
}
