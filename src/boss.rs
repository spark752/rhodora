use crate::{
    egui_integration::GuiTrait,
    frame_control::FrameControl,
    memory::Memory,
    mesh_import::{Batch, FileToLoad, Style},
    model_manager::ModelManager,
    pbr_lights::PbrLightTrait,
    postprocess::PostProcess,
    rh_error::RhError,
    texture::TextureManager,
    types::{
        CameraTrait, DeviceAccess, KeyboardHandler, RenderFormat,
        TransferFuture,
    },
    util,
    vk_window::{Properties, VkWindow},
};
use log::{debug, error, info, trace};
use std::time::Instant;
use std::{sync::Arc, time::Duration};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, SampleCount},
    pipeline::graphics::viewport::Viewport,
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::GpuFuture,
};
use vulkano::{swapchain::SwapchainAcquireFuture, Validated, VulkanError};
use winit::event_loop::ControlFlow;
use winit::{
    dpi::PhysicalSize, event::Event, event::KeyboardInput, event::WindowEvent,
    event_loop::EventLoop,
};

#[derive(Debug)]
pub enum Action {
    Continue,
    Return,
    Resize(PhysicalSize<u32>),
}

pub struct Timing {
    tick_interval_us: i64,
    time_acc_us: i64,
    previous_tick: Instant,
}

pub struct Limiter {
    target_duration: Duration,
    previous_instant: Instant,
}

pub struct Boss {
    window: VkWindow,
    render_format: RenderFormat,
    pub memory: Memory,
    pub swapchain: Arc<Swapchain>,
    swapchain_views: Vec<Arc<ImageView>>,
    postprocess: PostProcess,
    viewport: Viewport,
    depth_image_view: Arc<ImageView>,
    msaa_option: Option<Arc<ImageView>>,
    window_resized: bool,
    recreate_swapchain: bool,
    frame_control: FrameControl,
    pub model_manager: ModelManager,
    timing: Option<Timing>,
    limiter: Option<Limiter>,
    loop_start: Option<Instant>,
    vsync: bool,
}

impl Boss {
    /// # Errors
    /// May return `RhError`
    #[allow(clippy::too_many_lines)]
    pub fn new(
        event_loop: &EventLoop<()>,
        properties: Option<Properties>,
        sample_count: SampleCount,
        vsync: bool,
        frames_in_flight: usize,
    ) -> Result<Self, RhError> {
        // Create Vulkan interface, window, and surface
        let window = VkWindow::new(
            properties, event_loop, true, // Debut layer enable
        )?;

        // Create memory allocators
        let memory = Memory::new(window.device());

        // Create swapchain and images
        // The output will be interpreted as containing sRGB data. Therefore if
        // the postprocess pass outputs linear data, the swapchain must be set
        // to an sRGB format to do the conversion. If postprocess does the
        // conversion itself, the swapchain must be set to a UNORM format.
        // The choice of B or R first depends on what the hardware supports.
        let swapchain_candidates =
            [Format::B8G8R8A8_SRGB, Format::R8G8B8A8_SRGB];
        let swapchain_req_format = util::find_swapchain_format(
            window.physical(),
            window.surface(),
            &swapchain_candidates,
        )?;
        let (swapchain, images) = util::create_swapchain(
            &window,
            swapchain_req_format,
            util::choose_present_mode(&window, vsync),
        )?;
        let swapchain_views = util::create_image_views(&images)?;

        // Set output format for rendering pass
        let colour_candidates = [
            Format::R16G16B16A16_SFLOAT,
            Format::R32G32B32A32_SFLOAT,
            Format::B8G8R8A8_UNORM,
        ];
        let colour_format =
            util::find_colour_format(window.physical(), &colour_candidates)?;
        let depth_candidates = [
            Format::D32_SFLOAT,
            Format::D24_UNORM_S8_UINT,
            Format::D16_UNORM,
        ];
        let depth_format =
            util::find_depth_format(window.physical(), &depth_candidates)?;

        // Create a dynamic viewport
        let dimensions = window.dimensions()?;
        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: dimensions.into(),
            depth_range: 0.0..=1.0, // standard for Vulkan
        };

        // Create a depth buffer
        let depth_image_view = util::create_depth(
            memory.memory_allocator.clone(),
            dimensions.into(),
            depth_format,
            sample_count,
        )?;

        // Create an intermediate multisampled image and view if needed
        let msaa_option = if sample_count == SampleCount::Sample1 {
            None
        } else {
            Some(util::create_msaa(
                memory.memory_allocator.clone(),
                swapchain.image_extent(),
                colour_format,
                sample_count,
            )?)
        };

        // Collect data into a convenient `RenderFormat`
        let render_format = RenderFormat {
            colour_format,
            depth_format,
            sample_count,
        };

        // Create a command buffer builder. Some modules will record commands to
        // this during initialization.
        let mut cbb = util::create_primary_cbb(
            &memory.command_buffer_allocator,
            window.graphics_queue(),
        )
        .map_err(Validated::unwrap)?;
        let device_access = DeviceAccess {
            device: window.device().clone(),
            set_allocator: memory.set_allocator.clone(),
            cbb: &mut cbb,
        };

        // Create the postprocess layer
        let postprocess = PostProcess::new(
            &device_access,
            memory.memory_allocator.clone(),
            swapchain.image_extent(),
            colour_format, // output format from main rendering pass
            swapchain.image_format(), // output format from postprocessing
            None,          // sampler option, None = create automatically
        )?;

        // Create a ModelManager
        let model_manager = ModelManager::new(
            device_access,
            memory.memory_allocator.clone(),
            render_format,
        )?;

        // No more commands need to be recorded, so execute the command buffer.
        // Initialization of unrelated modules could continue after this
        // (if there were any).
        let benchmark = Instant::now();
        let cb = cbb.build().map_err(Validated::unwrap)?;
        let future = cb
            .execute(window.graphics_queue().clone())?
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap)?;

        // Wait for the GPU to finish the commands submitted for the
        // module creation. Not very efficient but it shouldn't take very long.
        // Probably could return the future and let the caller use it as a
        // condition for a later queue submission, but that adds complexity.
        future.wait(None).map_err(Validated::unwrap)?;
        debug!("Initial GPU transfer took {:?}", benchmark.elapsed());

        Ok(Self {
            window,
            render_format,
            memory,
            swapchain,
            swapchain_views,
            postprocess,
            viewport,
            depth_image_view,
            msaa_option,
            window_resized: false,
            recreate_swapchain: false,
            frame_control: FrameControl::new(frames_in_flight),
            model_manager,
            timing: None,
            limiter: None,
            loop_start: None,
            vsync,
        })
    }

    /// # Errors
    /// May return `RhError`
    pub fn resize(&mut self) -> Result<(), RhError> {
        let dimensions = self.window.dimensions()?;
        self.viewport.extent = dimensions.into();
        self.depth_image_view = util::create_depth(
            self.memory.memory_allocator.clone(),
            dimensions.into(),
            self.render_format.depth_format,
            self.render_format.sample_count,
        )?;
        if self.msaa_option.is_some() {
            self.msaa_option = Some(util::create_msaa(
                self.memory.memory_allocator.clone(),
                dimensions.into(),
                self.render_format.colour_format,
                self.render_format.sample_count,
            )?);
        }
        Ok(())
    }

    /// Recreates the swapchain after a window resize etc.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn recreate(&mut self) -> Result<(), RhError> {
        let dimensions = self.window.dimensions()?;
        let (new_swapchain, new_images) =
            match self.swapchain.recreate(SwapchainCreateInfo {
                image_extent: dimensions.into(),
                ..self.swapchain.create_info()
            }) {
                Ok(r) => r,
                // Error can happen during resizing, just retry
                // Maybe not anymore with vulkano 0.34?
                /*Err(SwapchainCreationError::ImageExtentNotSupported {
                    ..
                }) => {
                    warn!("retry for image extent not supported");
                    return Err(RhError::RetryRecreate);
                }*/
                Err(e) => {
                    error!("Could not recreate swapchain: {e:?}");
                    return Err(RhError::RecreateFailed);
                }
            };
        self.swapchain = new_swapchain;
        self.swapchain_views = util::create_image_views(&new_images)?;
        Ok(())
    }

    /// Creates a primary command buffer
    ///
    /// # Errors
    /// May return `RhError`
    pub fn create_primary_cbb(
        &self,
    ) -> Result<
        AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        Validated<VulkanError>,
    > {
        util::create_primary_cbb(
            &self.memory.command_buffer_allocator,
            self.window.graphics_queue(),
        )
    }

    #[allow(clippy::type_complexity)]
    /// Acquires the next image from the swapchain
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Panics if vulkano returns a validation error
    pub fn acquire_next_image(
        &mut self,
    ) -> Result<(u32, SwapchainAcquireFuture), RhError> {
        // The acquire_future will be signaled when the swapchain image is
        // ready. This may panic.
        let (image_index, suboptimal, acquire_future) =
            match vulkano::swapchain::acquire_next_image(
                self.swapchain.clone(),
                None,
            )
            .map_err(Validated::unwrap)
            {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    // This seems to happen on `present` after window resizing
                    // but not here
                    debug!("acquire_next_image says swapchain is out of date");
                    self.recreate_swapchain = true;
                    return Err(RhError::SwapchainOutOfDate);
                }
                Err(e) => {
                    error!("Could not acquire next image: {e:?}");
                    return Err(RhError::AcquireFailed);
                }
            };

        // "Some drivers might return suboptimal during window resize"
        if suboptimal {
            debug!("recreate_swapchain for suboptimal");
            self.recreate_swapchain = true;
        }

        Ok((image_index, acquire_future))
    }

    #[must_use]
    /// Creates a `RenderingAttachmentInfo` struct compatible with rendering
    /// using the provided background colour
    pub fn attachment_info(
        &self,
        background: [f32; 4],
    ) -> RenderingAttachmentInfo {
        util::attachment_info(
            background,
            self.postprocess.image_view(),
            self.msaa_option.clone(),
        )
    }

    /// Performs the postprocessing render pass
    ///
    /// # Errors
    /// May return `RhError`
    ///
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    pub fn render_pass_postprocess<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        image_index: u32,
    ) -> Result<(), RhError> {
        cbb.begin_rendering(RenderingInfo {
            color_attachments: vec![Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::DontCare,
                store_op: AttachmentStoreOp::Store,
                ..RenderingAttachmentInfo::image_view(
                    self.swapchain_views[image_index as usize].clone(),
                )
            })],
            ..Default::default()
        })
        .unwrap() // This is a Box<ValidationError>
        .set_viewport(0, std::iter::once(self.viewport.clone()).collect())
        .unwrap(); // This is a Box<ValidationError>
        self.postprocess.draw(cbb)
    }

    /// Renders the main pass with all models to the given camera. Call after
    /// `render_start`.
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    pub fn render_main_pass<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        background: [f32; 4],
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        let attachment_info = self.attachment_info(background);

        cbb.begin_rendering(RenderingInfo {
            color_attachments: vec![Some(attachment_info)],
            depth_attachment: Some(RenderingAttachmentInfo {
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::DontCare,
                clear_value: Some(1f32.into()),
                ..RenderingAttachmentInfo::image_view(
                    self.depth_image_view.clone(),
                )
            }),
            ..Default::default()
        })
        .unwrap(); // This is a Box<ValidationError>

        cbb.set_viewport(
            0, // (line break for easier reading)
            std::iter::once(self.viewport.clone()).collect(),
        )
        .unwrap(); // This is a Box<ValidationError>

        self.model_manager.draw_all(
            cbb,
            &self.memory.set_allocator,
            camera,
            lights,
        )?;

        self.render_pass_end(cbb)
    }

    /// Ends a rendering pass
    ///
    /// # Errors
    /// Will NOT return `RhError`. This is leftover from earlier code.
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    pub fn render_pass_end<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        cbb.end_rendering().unwrap(); // This is a Box<ValidationError>
        Ok(())
    }

    #[must_use]
    /// Returns the current `RenderFormat`
    pub const fn render_format(&self) -> RenderFormat {
        self.render_format
    }

    /// Starts a rendering frame by checking window and swapchain condition
    /// and cleaning up old resources. Call this at the start of the frame
    /// and check the return value. On `Action::Return` don't do any further
    /// rendering. On `Action::Resize` handle the resized window and proceed
    /// with rendering. On `Action::Continue` proceed with rendering.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn render_start(&mut self) -> Result<Action, RhError> {
        let mut signal_resize = false;

        // Do not draw if the window size is zero
        let dimensions = self.window.dimensions()?;
        if dimensions.width == 0 || dimensions.height == 0 {
            return Ok(Action::Return);
        }

        // Recreate things to the current window size if needed
        if self.recreate_swapchain || self.window_resized {
            match self.recreate() {
                Ok(()) => (),
                Err(RhError::RetryRecreate) => return Ok(Action::Return),
                Err(e) => return Err(e),
            }
            self.recreate_swapchain = false;
            if self.window_resized {
                self.resize()?;
                // Recreate target image and descriptor set
                self.postprocess.resize(
                    self.memory.memory_allocator.clone(),
                    &self.memory.set_allocator,
                    dimensions.into(),
                )?;
                self.window_resized = false;
                signal_resize = true;
            }
        }

        // Do cleanup of resources if possible
        if let Some(f) = &mut self.frame_control.fences
            [self.frame_control.previous_fence_index]
        {
            /* "If possible, checks whether the submission has finished. If so,
            gives up ownership of the resources used by these submissions.
            It is highly recommended to call cleanup_finished from time to
            time. Doing so will prevent memory usage from increasing over
            time, and will also destroy the locks on resources used by the
            GPU." */
            f.cleanup_finished();
            trace!("render_start cleanup_finished");
        }

        // Return
        Ok(if signal_resize {
            Action::Resize(dimensions)
        } else {
            Action::Continue
        })
    }

    /// Wait on the GPU fence, then submit a primary command queue, then present
    /// it to Vulkan.
    ///
    /// The caller should have already waited for the swapchain image to be
    /// available to prevent an issue with Nvidia on Linux with vsync enabled.
    /// However the future still needs to be provided to prevent a Vulkan
    /// validation error and panic.
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if vulkano returns a validation error
    pub fn submit_and_present(
        &mut self,
        cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        before_future: Box<dyn GpuFuture>,
        image_index: u32,
    ) -> Result<(), RhError> {
        let benchmark = Instant::now();
        let command_buffer = cbb.build().map_err(Validated::unwrap)?;

        // Create a joined future by joining the provided `before_future` with
        // the current fence if there is one and the right type of "now" future
        // if there is not.
        let gpu_future = {
            if let Some(x) = self.frame_control.take() {
                before_future.join(x)
            } else {
                before_future.join(self.now_future())
            }
        };

        // Wait for GPU fence then submit and call present
        let after_future = gpu_future.then_execute(
            self.window.graphics_queue().clone(),
            command_buffer,
        )?;
        self.present(after_future.boxed(), image_index)?;
        trace!("submit_and_present took {:?}", benchmark.elapsed());
        Ok(())
    }

    /// Wait for `before_future` then present to Vulkan
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if vulkano returns a validation error
    pub fn present(
        &mut self,
        before_future: Box<dyn GpuFuture>,
        image_index: u32,
    ) -> Result<(), RhError> {
        trace!("present image_index={}", image_index);
        let after_future = before_future
            .then_swapchain_present(
                self.window.graphics_queue().clone(),
                SwapchainPresentInfo::swapchain_image_index(
                    self.swapchain.clone(),
                    image_index,
                ),
            )
            .then_signal_fence_and_flush();

        // See how it went
        let new_fence_result = match after_future.map_err(Validated::unwrap) {
            Ok(future) => Ok(Some(future.boxed())),
            Err(VulkanError::OutOfDate) => {
                // Happens on resizing
                self.recreate_swapchain = true;
                debug!("present says swapchain is out of date");
                Ok(Some(self.now_future()))
            }
            Err(e) => {
                error!("Couldn't flush future: {e:?}");
                Err(RhError::FutureFlush(e))
            }
        };

        self.frame_control.update(new_fence_result?);
        Ok(())
    }

    /// Convenience method to render all models in a main pass, do the
    /// postprocess pass, submit the command buffer, and present the swapchain.
    /// Call `render start` first, followed by any updates needed to model
    /// positions etc. Then call this method. If you need additional passes
    /// for GUI etc. then you will need to perform the steps individually
    /// instead of using this.
    ///
    /// # Errors
    /// Will return `RhError` if an error occurs
    ///
    /// # Panics
    /// Panics if a swapchain image or its future can not be acquired
    pub fn render_all(
        &mut self,
        background: [f32; 4],
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
        gui: Option<&mut impl GuiTrait>,
    ) -> Result<(), RhError> {
        let benchmark = Instant::now();
        if let Some(limiter) = &self.limiter {
            let e = limiter.previous_instant.elapsed();
            if e > Duration::from_micros(1500) {
                debug!("render_all slow start {:?} after limiter reset", e);
            }
        }

        // Acquire an image from the swapchain to draw. Returns a
        // future that indicates when the image will be available and
        // may block until it knows that.
        let wait_time = Instant::now();
        let (image_index, acquire_future) = match self.acquire_next_image() {
            Ok(r) => r,
            Err(RhError::SwapchainOutOfDate) => return Ok(()),
            Err(e) => return Err(e),
        };
        let e = wait_time.elapsed();
        if e > Duration::from_micros(100) {
            debug!(
                "render_all slow to learn image_index={} in {:?}",
                image_index, e
            );
        } else {
            trace!("render_all knew image_index={} in {:?}", image_index, e);
        }

        // Create command buffer. For dynamic rendering, begin_rendering
        // and end_rendering are used instead of begin_render_pass and
        // end_render_pass.
        let mut cbb = self.create_primary_cbb().map_err(Validated::unwrap)?;

        // First pass into postprocess input texture via MSAA if enabled
        trace!(
            "render_all calling render_main_pass at {:?}",
            benchmark.elapsed()
        );
        self.render_main_pass(&mut cbb, background, camera, lights)?;

        // Second pass postprocess to swapchain image
        trace!(
            "render_all calling render_pass_postprocess at {:?}",
            benchmark.elapsed()
        );
        self.render_pass_postprocess(&mut cbb, image_index)?;
        if let Some(g) = gui {
            g.draw(self.swapchain.image_extent(), &mut cbb)?;
        };
        self.render_pass_end(&mut cbb)?;

        // Wait for swapchain image first when vsync is on. This works around
        // vsync problems on Linux + NVIDIA which cause continual 100% CPU
        // core usage and major lag (40+ ms frames) every few seconds. However
        // there are still many slow frames in the 18 to 20 ms range on the
        // test system. Turning off vsync seems to be a better option.
        if self.vsync {
            let wait_time = Instant::now();

            // Still need to use this future later (even though we know it
            // will have been already signalled) to avoid a vulkano validation
            // error / panic
            acquire_future.wait(None)?;

            // The wait time is expected to be long because the vsync delay
            // is done here. But if it is over ~16.5 ms, how are we supposed
            // to have time to actually create a frame?
            let e = wait_time.elapsed();
            if e > Duration::from_micros(16500) {
                debug!(
                    "vsync slow to get swapchain image, waited for {:?}",
                    wait_time.elapsed()
                );
            }
        }

        // Submit command buffer & present swapchain once ready
        trace!(
            "render_all calling submit_and_present at {:?}",
            benchmark.elapsed()
        );
        self.submit_and_present(
            cbb, // (line break for readability)
            Box::new(acquire_future),
            image_index,
        )
    }

    /// Convenience function that creates a batch, loads a single mesh into
    /// it from a file, then adds that to the model manager. Useful if you
    /// only have one model in the scene. Otherwise it is much more efficient
    /// to create a batch and use `load_batch` so that everything is grouped
    /// together.
    ///
    /// This function is equivalent to `load_model`.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_mesh<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        file: &FileToLoad,
    ) -> Result<usize, RhError> {
        // Vertex format was once hard coded in lowest levels but now has made
        // it all the way up to here. Still some more to go.
        let mut batch = Batch::new(Style::Skinned);
        batch.load(file)?;
        let ret = self.model_manager.process_batch(cbb, batch)?;
        Ok(ret[0])
    }

    /// Convenience function that creates a batch, loads a single mesh into
    /// it from a file, then adds that to the model manager. Useful if you
    /// only have one model in the scene. Otherwise it is much more efficient
    /// to create a batch and use `load_batch` so that everything is grouped
    /// together.
    ///
    /// This function is equivalent to `load_mesh`.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_model<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        file: &FileToLoad,
    ) -> Result<usize, RhError> {
        self.load_mesh(cbb, file)
    }

    /// Loads a batch of meshes
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load_batch<T>(
        &mut self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        batch: Batch,
    ) -> Result<Vec<usize>, RhError> {
        self.model_manager.process_batch(cbb, batch)
    }

    #[allow(clippy::cast_possible_truncation)]
    pub fn start_loop(
        &mut self,
        tick_rate: f32,
        limiter_option: Option<Duration>,
    ) {
        self.loop_start = Some(Instant::now());
        self.timing = Some(Timing {
            tick_interval_us: (tick_rate * 1_000_000.0) as i64,
            time_acc_us: 0,
            previous_tick: Instant::now(),
        });
        if let Some(target_duration) = limiter_option {
            self.limiter = Some(Limiter {
                target_duration,
                previous_instant: Instant::now(),
            });
            trace!("start_loop limiter reset");
        };
    }

    /// Handles events for the event loop
    ///
    /// # Panics
    /// Panics if the number of microseconds since the last tick can not fit
    /// into an `i64`. This could happen if you have over 292,000 years between
    /// ticks.
    pub fn handle_event(
        &mut self,
        event: &Event<()>,
        keyboard: &mut impl KeyboardHandler,
        control_flow: &mut ControlFlow,
    ) -> (Option<i64>, bool) {
        let mut do_tick = None;
        let mut do_render = false;
        match event {
            Event::NewEvents(_start) => {}

            Event::WindowEvent {
                event,
                window_id: _winit_window_id,
            } => {
                match event {
                    WindowEvent::Resized(_size) => {
                        self.window_resized = true;
                    }

                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(keycode),
                                state,
                                ..
                            },
                        ..
                    } => keyboard.input(*keycode, *state),

                    WindowEvent::CloseRequested => {
                        // In a multi window application, WindowId should be
                        // checked to make sure this is the right window
                        info!("CloseRequested event");
                        control_flow.set_exit();
                        return (None, false);
                    }

                    WindowEvent::Focused(_focused) => {}

                    WindowEvent::Moved(_position) => {}

                    _ => {} // Default WindowEvent match
                }
            }

            Event::MainEventsCleared => {
                if let Some(timing) = &mut self.timing {
                    let delta = i64::try_from(
                        timing.previous_tick.elapsed().as_micros(),
                    )
                    .expect("time overflowed");
                    timing.time_acc_us += delta;
                    timing.previous_tick = Instant::now();
                    if timing.time_acc_us >= timing.tick_interval_us {
                        do_tick = Some(timing.tick_interval_us);
                    }
                }
                if let Some(limiter) = &mut self.limiter {
                    if limiter.previous_instant.elapsed()
                        > limiter.target_duration
                    {
                        do_render = true;
                        limiter.previous_instant = Instant::now();
                        trace!("MainEventsCleared limiter reset");
                        control_flow.set_wait_until(
                            limiter.previous_instant + limiter.target_duration,
                        );
                    } else {
                        trace!("waiting due to limiter");
                    }
                } else {
                    // Render everytime if limiter not set
                    do_render = true;
                }
            }

            Event::RedrawEventsCleared => {}

            _ => (),
        }
        (do_tick, do_render)
    }

    /// Call to finish the tick. Attempts to "catch up" unless `drop_ticks`
    /// is set
    pub fn finish_tick(&mut self, drop_ticks: bool) -> Option<i64> {
        if let Some(timing) = &mut self.timing {
            if drop_ticks {
                trace!("finish_tick with drop_ticks=true");
                timing.time_acc_us = 0;
            } else {
                timing.time_acc_us -= timing.tick_interval_us;
                if timing.time_acc_us >= timing.tick_interval_us {
                    return Some(timing.tick_interval_us);
                }
            }
        }
        None
    }

    /// Returns the graphics queue
    #[must_use]
    pub const fn graphics_queue(&self) -> &Arc<Queue> {
        self.window.graphics_queue()
    }

    /// Starts a transfer using the graphics queue
    ///
    /// # Errors
    /// May return `RhError`
    pub fn start_transfer(
        &self,
        cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<TransferFuture, RhError> {
        trace!("Starting graphics queue transfer");
        util::start_transfer(cbb, self.graphics_queue().clone())
    }

    /// Returns the `Device` in use
    #[must_use]
    pub const fn device(&self) -> &Arc<Device> {
        self.window.device()
    }

    /// Gets access to the `TextureManager` owned by the `ModelManager` which
    /// is owned by this object
    #[must_use]
    pub const fn texture_manager(&self) -> &Arc<TextureManager> {
        self.model_manager.texture_manager()
    }

    /// Gets a boxed "now" future compatible with the device
    #[must_use]
    pub fn now_future(&self) -> Box<dyn GpuFuture> {
        vulkano::sync::now(self.window.device().clone()).boxed()
    }

    /// Elapsed time since `start_loop`
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.loop_start.map(|loop_start| loop_start.elapsed())
    }

    // Dimensions of the swapchain images
    #[must_use]
    pub fn dimensions(&self) -> [u32; 2] {
        self.swapchain.image_extent()
    }

    // Reference to the `Surface` being used
    #[must_use]
    pub const fn surface(&self) -> &Arc<Surface> {
        self.window.surface()
    }

    // Reference to the `VkWindow` being used
    #[must_use]
    pub const fn window(&self) -> &VkWindow {
        &self.window
    }
}
