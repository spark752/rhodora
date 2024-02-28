use crate::{
    memory::Memory,
    mesh_import::{Batch, FileToLoad, Style},
    model_manager::Manager as ModelManager,
    pbr_lights::PbrLightTrait,
    postprocess::PostProcess,
    rh_error::RhError,
    texture::Manager as TextureManager,
    types::{
        CameraTrait, DeviceAccess, KeyboardHandler, RenderFormat,
        TransferFuture,
    },
    util,
    vk_window::{Properties, VkWindow},
};
use log::{error, info, trace};
use std::time::Instant;
use std::{sync::Arc, time::Duration};
use vulkano::{
    command_buffer::{
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
        PrimaryCommandBufferAbstract, RenderingAttachmentInfo, RenderingInfo,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, Image, SampleCount},
    pipeline::graphics::viewport::Viewport,
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
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

pub struct FrameControl {
    fences: Vec<Option<Box<dyn GpuFuture>>>,
    fence_index: usize,
    previous_fence_index: usize,
    frame_count: usize,
    frames_in_flight: usize,
}

impl FrameControl {
    fn new(frames_in_flight: usize) -> Self {
        // FIXME
        if frames_in_flight != 1 {
            error!("FRAMES IN FLIGHT CURRENTLY BROKEN, SETTING TO 1");
        }
        // The vec! macro doesn't work because the type doesn't implement
        // clone?
        let mut fences = Vec::new();
        for _ in 0..frames_in_flight {
            fences.push(None);
        }
        Self {
            fences,
            fence_index: 0,
            previous_fence_index: 0,
            frame_count: 0,
            frames_in_flight: 1,
        }
    }

    #[allow(clippy::modulo_one)] // FRAMES_IN_FLIGHT might equal 1
    fn update(&mut self, new_fence: Option<Box<dyn GpuFuture>>) {
        self.fences[self.fence_index] = new_fence;
        self.previous_fence_index = self.fence_index;
        self.fence_index = (self.fence_index + 1) % self.frames_in_flight;
        self.frame_count += 1;
    }
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
    pub window: VkWindow,
    render_format: RenderFormat,
    pub memory: Memory,
    pub swapchain: Arc<Swapchain>,
    pub swapchain_views: Vec<Arc<ImageView>>,
    pub postprocess: PostProcess,
    pub viewport: Viewport,
    pub depth_image_view: Arc<ImageView>,
    msaa_option: Option<Arc<ImageView>>,
    window_resized: bool,
    recreate_swapchain: bool,
    frame_control: FrameControl,
    pub model_manager: ModelManager,
    timing: Option<Timing>,
    limiter: Option<Limiter>,
    loop_start: Option<Instant>,
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
        let swapchain_views = Self::create_image_views(&images)?;

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

        // Create postprocess layer
        let mut cbb = util::create_primary_cbb(
            &memory.command_buffer_allocator,
            window.graphics_queue(),
        )
        .map_err(Validated::unwrap)?;
        let postprocess = PostProcess::new(
            &DeviceAccess {
                device: window.device().clone(),
                set_allocator: &memory.set_allocator,
                cbb: &mut cbb,
            },
            memory.memory_allocator.clone(),
            swapchain.image_extent(),
            colour_format, // output format from main rendering pass
            swapchain.image_format(), // output format from postprocessing
            None,          // sampler option, None = create automatically
        )?;
        let cb = cbb.build().map_err(Validated::unwrap)?;
        let future = cb
            .execute(window.graphics_queue().clone())?
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap)?;

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

        let render_format = RenderFormat {
            colour_format,
            depth_format,
            sample_count,
        };

        // Create a ModelManager
        let model_manager = ModelManager::new(
            window.device().clone(),
            memory.memory_allocator.clone(),
            memory.set_allocator.clone(),
            &render_format,
        )?;

        // Wait for the GPU to finish the commands submitted for the
        // postprocess layer creation. Not very efficient but it shouldn't
        // take very long. Probably could return the future and let the caller
        // use it as a condition for a later queue submission, but that adds
        // complexity.
        future.wait(None).map_err(Validated::unwrap)?;

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
                // FIXME Maybe not anymore?
                //Err(SwapchainCreationError::ImageExtentNotSupported {
                //    ..
                //}) => {
                //    info!("retry for image extent not supported");
                //    return Err(RhError::RetryRecreate);
                //}
                Err(e) => {
                    error!("Could not recreate swapchain: {e:?}");
                    return Err(RhError::RecreateFailed);
                }
            };
        self.swapchain = new_swapchain;
        self.swapchain_views = Self::create_image_views(&new_images)?;
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

    fn create_image_views(
        images: &[Arc<Image>],
    ) -> Result<Vec<Arc<ImageView>>, RhError> {
        let mut ret = Vec::new();
        for image in images {
            ret.push(
                ImageView::new_default(image.clone())
                    .map_err(Validated::unwrap)?,
            );
        }
        Ok(ret)
    }

    #[allow(clippy::type_complexity)]
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
                    info!("Recreate swapchain for out of date");
                    self.recreate_swapchain = true;
                    return Err(RhError::SwapchainOutOfDate);
                }
                Err(e) => {
                    error!("Could not acquire next image: {e:?}");
                    return Err(RhError::AcquireFailed);
                }
            };

        // Some drivers might return suboptimal during window resize
        if suboptimal {
            info!("recreate swapchain for suboptimal");
            self.recreate_swapchain = true;
        }

        Ok((image_index, acquire_future)) //.boxed()))
    }

    #[must_use]
    pub fn attachment_info(
        &self,
        background: &[f32; 4],
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
        background: &[f32; 4],
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        //trace!("Getting rendering target attachment");
        let attachment_info = self.attachment_info(background);

        //trace!("Recording 'begin_rendering' to command buffer");
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

        //trace!("Setting the viewport");
        cbb.set_viewport(
            0, // (line break for easier reading)
            std::iter::once(self.viewport.clone()).collect(),
        )
        .unwrap(); // This is a Box<ValidationError>

        //trace!("Calling ModelManager 'draw_all'");
        self.model_manager.draw_all(
            cbb,
            &self.memory.set_allocator,
            camera,
            lights,
        )?;

        //trace!("Ending the render pass");
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
            f.cleanup_finished();
        }

        // Return
        Ok(if signal_resize {
            Action::Resize(dimensions)
        } else {
            Action::Continue
        })
    }

    /// Wait for `before_future`, then submit a primary command queue, then
    /// present to Vulkan
    ///
    /// # Errors
    /// May return `RhError`
    pub fn submit_and_present(
        &mut self,
        cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        before_future: Box<dyn GpuFuture>,
        image_index: u32,
    ) -> Result<(), RhError> {
        let command_buffer = cbb.build().map_err(Validated::unwrap)?;

        // Wait for GPU fence
        // This akward construction creates a JoinedFuture even if there is
        // nothing to join
        let before_future = if self.frame_control.fences
            [self.frame_control.fence_index]
            .is_some()
        {
            before_future.join(
                self.frame_control.fences[self.frame_control.fence_index]
                    .take()
                    .ok_or(RhError::FutureFlush)?, // FIXME incorrect error
            )
        } else {
            before_future.join(self.now_future())
        };

        // Submit and call present
        let after_future = before_future.then_execute(
            self.window.graphics_queue().clone(),
            command_buffer,
        )?;
        self.present(after_future.boxed(), image_index)
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
                info!("Recreate swapchain for flush error");
                Ok(Some(self.now_future()))
            }
            Err(e) => {
                error!("Couldn't flush future: {e:?}");
                Err(RhError::FutureFlush)
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
        background: &[f32; 4],
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        // Acquire an image from the swapchain to draw. Returns a
        // future that indicates when the image will be available and
        // may block until it knows that.
        //trace!("Acquiring swapchain image");
        let (image_index, acquire_future) = match self.acquire_next_image() {
            Ok(r) => r,
            Err(RhError::SwapchainOutOfDate) => return Ok(()),
            Err(e) => return Err(e),
        };

        // Create command buffer. For dynamic rendering, begin_rendering
        // and end_rendering are used instead of begin_render_pass and
        // end_render_pass.
        //trace!("Creating command buffer builder");
        let mut cbb = self.create_primary_cbb().map_err(Validated::unwrap)?;

        // First pass into postprocess input texture via MSAA if enabled
        //trace!("Starting main pass");
        self.render_main_pass(&mut cbb, background, camera, lights)?;

        // Second pass postprocess to swapchain image
        //trace!("Starting postprocess pass");
        self.render_pass_postprocess(&mut cbb, image_index)?;
        self.render_pass_end(&mut cbb)?;

        // Wait for swapchain image first. This avoids stutters and 100%
        // CPU usage on Linux + NVIDIA when vsync is on.
        acquire_future
            .wait(None)
            .expect("Fence error while waiting for swapchain image");

        // Submit command buffer & present swapchain once ready
        //trace!("Submitting command buffer and presenting swapchain");
        self.submit_and_present(
            cbb,
            //previous_future,
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
        };
    }

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
                        control_flow.set_wait_until(
                            limiter.previous_instant + limiter.target_duration,
                        );
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

    pub fn finish_tick(&mut self, drop_ticks: bool) -> Option<i64> {
        if let Some(timing) = &mut self.timing {
            // If the user is doing something slow like loading, they can
            // bypass the catch up mechanism
            if drop_ticks {
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

    #[must_use]
    pub const fn graphics_queue(&self) -> &Arc<Queue> {
        self.window.graphics_queue()
    }

    /// # Errors
    /// May return `RhError`
    pub fn start_transfer(
        &self,
        cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<TransferFuture, RhError> {
        trace!("Staring graphics queue transfer");
        util::start_transfer(cbb, self.graphics_queue().clone())
    }

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

    /// Get a boxed "now" future compatible with the device
    #[must_use]
    pub fn now_future(&self) -> Box<dyn GpuFuture> {
        vulkano::sync::now(self.window.device().clone()).boxed()
    }

    /// Elapsed time since `start_loop`
    #[must_use]
    pub fn elapsed(&self) -> Option<Duration> {
        self.loop_start.map(|loop_start| loop_start.elapsed())
    }
}
