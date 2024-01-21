// This file is based on egui_winit_vulkano which is
// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.
use super::{
    renderer::{RenderResources, Renderer},
    utils::{immutable_texture_from_bytes, immutable_texture_from_file},
};
use crate::rh_error::RhError;
use egui::{ClippedPrimitive, Context, TexturesDelta};
use egui_winit::winit::event_loop::EventLoopWindowTarget;
use std::sync::Arc;
use vulkano::{
    command_buffer::AutoCommandBufferBuilder,
    device::Queue,
    format::{Format, NumericType},
    image::{ImageViewAbstract, SampleCount},
    swapchain::{Surface, SurfaceInfo},
};
use winit::event::Event;
use winit::window::Window;

/// # Panics
/// Panics and should be fixed so that it won't
fn get_surface_image_format(
    surface: &Arc<Surface>,
    preferred_format: Option<Format>,
    gfx_queue: &Arc<Queue>,
) -> vulkano::format::Format {
    preferred_format.unwrap_or_else(|| {
        gfx_queue
            .device()
            .physical_device()
            .surface_formats(surface, SurfaceInfo::default())
            .unwrap()
            .iter()
            .find(|f| f.0.type_color().unwrap() == NumericType::SRGB)
            .unwrap()
            .0
    })
}

pub struct GuiConfig {
    /// Preferred target image format. This should match the surface format.
    /// Sometimes the user may prefer linear color space rather than non linear.
    /// Hence the option. SRGB is selected by default.
    pub preferred_format: Option<Format>,
    /// Multisample count. Defaults to 1. If you use more than 1, you'll have
    /// to ensure your
    /// pipeline and target image matches that.
    pub samples: SampleCount,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            preferred_format: None,
            samples: SampleCount::Sample1,
        }
    }
}

pub struct Gui {
    pub egui_ctx: egui::Context,
    pub egui_winit: egui_winit::State,
    renderer: Renderer,
    surface: Arc<Surface>,
    shapes: Vec<egui::epaint::ClippedShape>,
    textures_delta: egui::TexturesDelta,
}

impl Gui {
    /// # Errors
    /// May return `RhError`
    pub fn new<T>(
        event_loop: &EventLoopWindowTarget<T>,
        surface: Arc<Surface>,
        gfx_queue: Arc<Queue>,
        config: GuiConfig,
    ) -> Result<Self, RhError> {
        // Pick preferred format if provided, otherwise use the default one
        let format = get_surface_image_format(
            &surface,
            config.preferred_format,
            &gfx_queue,
        );
        let max_texture_side = gfx_queue
            .device()
            .physical_device()
            .properties()
            .max_image_array_layers as usize;
        let renderer = Renderer::new(gfx_queue, format)?;
        let mut egui_winit = egui_winit::State::new(event_loop);
        egui_winit.set_max_texture_side(max_texture_side);
        #[allow(clippy::cast_possible_truncation)]
        egui_winit.set_pixels_per_point(
            surface_window(&surface)?.scale_factor() as f32,
        );
        Ok(Self {
            egui_ctx: Context::default(),
            egui_winit,
            renderer,
            surface,
            shapes: vec![],
            textures_delta: TexturesDelta::default(),
        })
    }

    /// Returns a set of resources used to construct the render pipeline. These can be reused
    /// to create additional pipelines and buffers to be rendered in a `PaintCallback`.
    pub fn render_resources(&self) -> RenderResources {
        self.renderer.render_resources()
    }

    /// Updates context state by winit event.
    /// Returns `true` if egui wants exclusive use of this event
    /// (e.g. a mouse click on an egui window, or entering text into a text field).
    /// For instance, if you use egui for a game, you want to first call this
    /// and only when this returns `false` pass on the events to your game.
    ///
    /// Note that egui uses `tab` to move focus between elements, so this will
    /// always return `true` for tabs.
    /*pub fn update(
        &mut self,
        winit_event: &winit::event::WindowEvent<'_>,
    ) -> bool {
        self.egui_winit
            .on_event(&self.egui_ctx, winit_event)
            .consumed
    }*/
    pub fn update(&mut self, event: &Event<()>) -> bool {
        match event {
            Event::WindowEvent {
                event,
                window_id: _winit_window_id,
            } => self.egui_winit.on_event(&self.egui_ctx, event).consumed,
            _ => false,
        }
    }

    /// Begins Egui frame & determines what will be drawn later. This must be
    /// called before draw, and after `update` (winit event).
    ///
    /// # Errors
    /// May return `RhError`
    pub fn immediate_ui(
        &mut self,
        layout_function: impl FnOnce(&mut Self),
    ) -> Result<(), RhError> {
        let raw_input = self
            .egui_winit
            .take_egui_input(surface_window(&self.surface)?);
        self.egui_ctx.begin_frame(raw_input);
        // Render Egui
        layout_function(self);
        Ok(())
    }

    /// If you wish to better control when to begin frame, do so by calling
    /// this function (Finish by drawing)
    ///
    /// # Errors
    /// May return `RhError`
    pub fn begin_frame(&mut self) -> Result<(), RhError> {
        let raw_input = self
            .egui_winit
            .take_egui_input(surface_window(&self.surface)?);
        self.egui_ctx.begin_frame(raw_input);
        Ok(())
    }

    /// # Errors
    /// May return `RhError`
    pub fn draw<T>(
        &mut self,
        dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        let (clipped_meshes, textures_delta) =
            self.extract_draw_data_at_frame_end()?;

        self.renderer.draw(
            &clipped_meshes,
            &textures_delta,
            self.egui_winit.pixels_per_point(),
            dimensions,
            builder,
        )
    }

    fn extract_draw_data_at_frame_end(
        &mut self,
    ) -> Result<(Vec<ClippedPrimitive>, TexturesDelta), RhError> {
        self.end_frame()?;
        let shapes = std::mem::take(&mut self.shapes);
        let textures_delta = std::mem::take(&mut self.textures_delta);
        let clipped_meshes = self.egui_ctx.tessellate(shapes);
        Ok((clipped_meshes, textures_delta))
    }

    fn end_frame(&mut self) -> Result<(), RhError> {
        let egui::FullOutput {
            platform_output,
            repaint_after: _r,
            textures_delta,
            shapes,
        } = self.egui_ctx.end_frame();

        self.egui_winit.handle_platform_output(
            surface_window(&self.surface)?,
            &self.egui_ctx,
            platform_output,
        );
        self.shapes = shapes;
        self.textures_delta = textures_delta;
        Ok(())
    }

    /// Registers a user image from Vulkano image view to be used by egui
    ///
    /// # Errors
    /// May return `RhError`
    pub fn register_user_image_view(
        &mut self,
        image: Arc<dyn ImageViewAbstract + Send + Sync>,
    ) -> Result<egui::TextureId, RhError> {
        self.renderer.register_image(image)
    }

    /// Registers a user image to be used by egui
    /// - `image_file_bytes`: e.g. `include_bytes!("./assets/tree.png")`
    /// - `format`: e.g. `vulkano::format::Format::R8G8B8A8Unorm`
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Panics if the image can not be created
    pub fn register_user_image(
        &mut self,
        image_file_bytes: &[u8],
        format: vulkano::format::Format,
    ) -> Result<egui::TextureId, RhError> {
        let image = immutable_texture_from_file(
            self.renderer.allocators(),
            self.renderer.queue(),
            image_file_bytes,
            format,
        )
        .expect("Failed to create image");
        self.renderer.register_image(image)
    }

    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Panics if the image can not be created
    pub fn register_user_image_from_bytes(
        &mut self,
        image_byte_data: &[u8],
        dimensions: [u32; 2],
        format: vulkano::format::Format,
    ) -> Result<egui::TextureId, RhError> {
        let image = immutable_texture_from_bytes(
            self.renderer.allocators(),
            self.renderer.queue(),
            image_byte_data,
            dimensions,
            format,
        )
        .expect("Failed to create image");
        self.renderer.register_image(image)
    }

    /// Unregisters a user image
    pub fn unregister_user_image(&mut self, texture_id: egui::TextureId) {
        self.renderer.unregister_image(texture_id);
    }

    /// Access egui's context (which can be used to e.g. set fonts, visuals etc)
    pub fn context(&self) -> egui::Context {
        self.egui_ctx.clone()
    }
}

// Helper to retrieve Window from surface object
fn surface_window(surface: &Surface) -> Result<&Window, RhError> {
    surface
        .object()
        .ok_or(RhError::WindowNotFound)?
        .downcast_ref::<Window>()
        .ok_or(RhError::WindowNotFound)
}
