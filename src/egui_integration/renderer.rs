// This file is based on egui_winit_vulkano which is
// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

use std::sync::Arc;

use crate::memory::Memory;
use crate::rh_error::RhError;
use ahash::AHashMap;
use bytemuck::{Pod, Zeroable};
use egui::{
    epaint::{Mesh, Primitive},
    ClippedPrimitive, Rect, TexturesDelta,
};
use log::error;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        BlitImageInfo, CommandBufferUsage, CopyBufferToImageInfo, ImageBlit,
        PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, layout::DescriptorSetLayout,
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Queue,
    format::{Format, NumericType},
    image::{
        view::ImageView, ImageAccess, ImageLayout, ImageUsage,
        ImageViewAbstract, ImmutableImage, SampleCount,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{AttachmentBlend, BlendFactor, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode as CullModeEnum, RasterizationState},
            render_pass::PipelineRenderingCreateInfo,
            vertex_input::Vertex,
            viewport::{Scissor, Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    sampler::{
        Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode,
    },
    sync::GpuFuture,
    DeviceSize,
};

const VERTICES_PER_QUAD: DeviceSize = 4;
const VERTEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * VERTICES_PER_QUAD;
const INDEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * 2;

/// Should match vertex definition of egui (except color is `[f32; 4]`)
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub tex_coords: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

pub struct Renderer {
    gfx_queue: Arc<Queue>,
    need_srgb_conv: bool,

    #[allow(unused)]
    format: vulkano::format::Format,
    sampler: Arc<Sampler>,

    memory: Memory,
    vertex_buffer_pool: SubbufferAllocator,
    index_buffer_pool: SubbufferAllocator,
    pipeline: Arc<GraphicsPipeline>,

    texture_desc_sets: AHashMap<egui::TextureId, Arc<PersistentDescriptorSet>>,
    texture_images: AHashMap<
        egui::TextureId,
        Arc<dyn ImageViewAbstract + Send + Sync + 'static>,
    >,
    next_native_tex_id: u64,
}

impl Renderer {
    pub fn new(
        gfx_queue: Arc<Queue>,
        format: Format,
    ) -> Result<Renderer, RhError> {
        let need_srgb_conv = if let Some(t) = format.type_color() {
            t == NumericType::UNORM
        } else {
            false
        };
        let memory = Memory::new(gfx_queue.device());
        let (vertex_buffer_pool, index_buffer_pool) =
            Self::create_buffers(&memory.memory_allocator);
        let pipeline = Self::create_pipeline(gfx_queue.clone(), format)?;
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )?;
        Ok(Renderer {
            gfx_queue,
            format,
            vertex_buffer_pool,
            index_buffer_pool,
            pipeline,
            texture_desc_sets: AHashMap::default(),
            texture_images: AHashMap::default(),
            next_native_tex_id: 0,
            need_srgb_conv,
            sampler,
            memory,
        })
    }

    fn create_buffers(
        allocator: &Arc<StandardMemoryAllocator>,
    ) -> (SubbufferAllocator, SubbufferAllocator) {
        // Create vertex and index buffers
        let vertex_buffer_pool = SubbufferAllocator::new(
            allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
        );
        vertex_buffer_pool
            .reserve(VERTEX_BUFFER_SIZE)
            .expect("Failed to reserve vertex buffer memory");
        let index_buffer_pool = SubbufferAllocator::new(
            allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
        );
        index_buffer_pool
            .reserve(INDEX_BUFFER_SIZE)
            .expect("Failed to reserve index buffer memory");

        (vertex_buffer_pool, index_buffer_pool)
    }

    fn create_pipeline(
        gfx_queue: Arc<Queue>,
        format: vulkano::format::Format,
    ) -> Result<Arc<GraphicsPipeline>, RhError> {
        let vs = vs::load(gfx_queue.device().clone())
            .expect("failed to create shader module");
        let fs = fs::load(gfx_queue.device().clone())
            .expect("failed to create shader module");

        let mut blend = AttachmentBlend::alpha();
        blend.color_source = BlendFactor::One;
        blend.alpha_source = BlendFactor::OneMinusDstAlpha;
        blend.alpha_destination = BlendFactor::One;
        let blend_state = ColorBlendState::new(1).blend(blend);

        Ok(GraphicsPipeline::start()
            .render_pass(PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(format)],
                ..Default::default()
            })
            .vertex_input_state(EguiVertex::per_vertex())
            .vertex_shader(
                vs.entry_point("main").ok_or(RhError::VertexShaderError)?,
                (),
            )
            .input_assembly_state(InputAssemblyState::new())
            .fragment_shader(
                fs.entry_point("main").ok_or(RhError::VertexShaderError)?,
                (),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_dynamic(1))
            .color_blend_state(blend_state)
            .rasterization_state(
                RasterizationState::new().cull_mode(CullModeEnum::None),
            )
            .multisample_state(MultisampleState {
                rasterization_samples: SampleCount::Sample1,
                ..Default::default()
            })
            .build(gfx_queue.device().clone())?)
    }

    /// Creates a descriptor set for images
    fn sampled_image_desc_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        image: Arc<dyn ImageViewAbstract + 'static>,
    ) -> Result<Arc<PersistentDescriptorSet>, RhError> {
        Ok(PersistentDescriptorSet::new(
            &self.memory.set_allocator,
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                image,
                self.sampler.clone(),
            )],
        )?)
    }

    /// Registers a user texture. User texture needs to be unregistered when it is no longer needed
    pub fn register_image(
        &mut self,
        image: Arc<dyn ImageViewAbstract + Send + Sync>,
    ) -> Result<egui::TextureId, RhError> {
        let layout = self
            .pipeline
            .layout()
            .set_layouts()
            .get(0)
            .ok_or(RhError::PipelineError);
        let desc_set = self.sampled_image_desc_set(layout?, image.clone());
        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;
        self.texture_desc_sets.insert(id, desc_set?);
        self.texture_images.insert(id, image);
        Ok(id)
    }

    /// Unregister user texture.
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        self.texture_desc_sets.remove(&texture_id);
        self.texture_images.remove(&texture_id);
    }

    fn update_texture(
        &mut self,
        texture_id: egui::TextureId,
        delta: &egui::epaint::ImageDelta,
    ) -> Result<(), RhError> {
        // Extract pixel data from egui
        let data: Vec<u8> = match &delta.image {
            egui::ImageData::Color(image) => {
                assert_eq!(
                    image.width() * image.height(),
                    image.pixels.len(),
                    "Mismatch between texture size and texel count"
                );
                image
                    .pixels
                    .iter()
                    .flat_map(|color| color.to_array())
                    .collect()
            }
            egui::ImageData::Font(image) => image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect(),
        };
        // Create buffer to be copied to the image
        let texture_data_buffer = Buffer::from_iter(
            &self.memory.memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            data,
        )?;
        // Create image
        let (img, init) = ImmutableImage::uninitialized(
            &self.memory.memory_allocator,
            vulkano::image::ImageDimensions::Dim2d {
                width: delta.image.width() as u32,
                height: delta.image.height() as u32,
                array_layers: 1,
            },
            Format::R8G8B8A8_SRGB,
            vulkano::image::MipmapsCount::One,
            ImageUsage::TRANSFER_DST
                | ImageUsage::TRANSFER_SRC
                | ImageUsage::SAMPLED,
            Default::default(),
            ImageLayout::ShaderReadOnlyOptimal,
            Some(self.gfx_queue.queue_family_index()),
        )?;
        let font_image = ImageView::new_default(img)?;

        // Create command buffer builder
        let mut cbb = AutoCommandBufferBuilder::primary(
            &self.memory.command_buffer_allocator,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        // Copy buffer to image
        cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            texture_data_buffer,
            init,
        ))?;

        // Blit texture data to existing image if delta pos exists (e.g. font changed)
        if let Some(pos) = delta.pos {
            if let Some(existing_image) = self.texture_images.get(&texture_id) {
                let src_dims = font_image.image().dimensions();
                let top_left = [pos[0] as u32, pos[1] as u32, 0];
                let bottom_right = [
                    pos[0] as u32 + src_dims.width(),
                    pos[1] as u32 + src_dims.height(),
                    1,
                ];

                cbb.blit_image(BlitImageInfo {
                    src_image_layout: ImageLayout::General,
                    dst_image_layout: ImageLayout::General,
                    regions: [ImageBlit {
                        src_subresource: font_image
                            .image()
                            .subresource_layers(),
                        src_offsets: [
                            [0, 0, 0],
                            [src_dims.width(), src_dims.height(), 1],
                        ],
                        dst_subresource: existing_image
                            .image()
                            .subresource_layers(),
                        dst_offsets: [top_left, bottom_right],
                        ..Default::default()
                    }]
                    .into(),
                    filter: Filter::Nearest,
                    ..BlitImageInfo::images(
                        font_image.image().clone(),
                        existing_image.image(),
                    )
                })?;
            }
            // Otherwise save the newly created image
        } else {
            let layout = self
                .pipeline
                .layout()
                .set_layouts()
                .get(0)
                .ok_or(RhError::PipelineError);
            let font_desc_set =
                self.sampled_image_desc_set(layout?, font_image.clone());
            self.texture_desc_sets.insert(texture_id, font_desc_set?);
            self.texture_images.insert(texture_id, font_image);
        }
        // Execute command buffer
        let command_buffer = cbb.build()?;
        let finished = command_buffer.execute(self.gfx_queue.clone())?;
        let _fut = finished.then_signal_fence_and_flush()?;
        Ok(())
    }

    fn get_rect_scissor(
        &self,
        scale_factor: f32,
        dimensions: [u32; 2],
        rect: Rect,
    ) -> Scissor {
        let min = rect.min;
        let min = egui::Pos2 {
            x: min.x * scale_factor,
            y: min.y * scale_factor,
        };
        let min = egui::Pos2 {
            x: min.x.clamp(0.0, dimensions[0] as f32),
            y: min.y.clamp(0.0, dimensions[1] as f32),
        };
        let max = rect.max;
        let max = egui::Pos2 {
            x: max.x * scale_factor,
            y: max.y * scale_factor,
        };
        let max = egui::Pos2 {
            x: max.x.clamp(min.x, dimensions[0] as f32),
            y: max.y.clamp(min.y, dimensions[1] as f32),
        };
        Scissor {
            origin: [min.x.round() as u32, min.y.round() as u32],
            dimensions: [
                (max.x.round() - min.x) as u32,
                (max.y.round() - min.y) as u32,
            ],
        }
    }

    #[allow(clippy::type_complexity)]
    fn create_subbuffers(
        &self,
        mesh: &Mesh,
    ) -> Result<(Subbuffer<[EguiVertex]>, Subbuffer<[u32]>), RhError> {
        // Copy vertices to buffer, mapping to output format
        let mapped: Vec<EguiVertex> = mesh
            .vertices
            .iter()
            .map(|v| EguiVertex {
                position: [v.pos.x, v.pos.y],
                tex_coords: [v.uv.x, v.uv.y],
                color: [
                    v.color.r() as f32 / 255.0,
                    v.color.g() as f32 / 255.0,
                    v.color.b() as f32 / 255.0,
                    v.color.a() as f32 / 255.0,
                ],
            })
            .collect();
        let vertex_buff =
            self.vertex_buffer_pool.allocate_slice(mapped.len() as _)?;
        vertex_buff.write()?.copy_from_slice(&mapped);

        // Copy indices to buffer
        let index_buff = self
            .index_buffer_pool
            .allocate_slice(mesh.indices.len() as _)?;
        index_buff.write()?.copy_from_slice(&mesh.indices);

        Ok((vertex_buff, index_buff))
    }

    pub fn draw<T>(
        &mut self,
        clipped_meshes: &[ClippedPrimitive],
        textures_delta: &TexturesDelta,
        scale_factor: f32,
        dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        for (id, image_delta) in &textures_delta.set {
            self.update_texture(*id, image_delta)?;
        }
        self.draw_egui(scale_factor, clipped_meshes, dimensions, builder)?;

        // FIXME was previously done after build, but not execute.
        // Does it matter?
        for &id in &textures_delta.free {
            self.unregister_image(id);
        }
        Ok(())
    }

    fn draw_egui<T>(
        &mut self,
        scale_factor: f32,
        clipped_meshes: &[ClippedPrimitive],
        dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        let push_constants = vs::PushConstants {
            screen_size: [
                dimensions[0] as f32 / scale_factor,
                dimensions[1] as f32 / scale_factor,
            ],
            need_srgb_conv: self.need_srgb_conv.into(),
        };

        for ClippedPrimitive {
            clip_rect,
            primitive,
        } in clipped_meshes
        {
            match primitive {
                Primitive::Mesh(mesh) => {
                    // Nothing to draw if we don't have vertices & indices
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }
                    if self.texture_desc_sets.get(&mesh.texture_id).is_none() {
                        error!(
                            "This texture no longer exists {:?}",
                            mesh.texture_id
                        );
                        continue;
                    }

                    let scissors = vec![self.get_rect_scissor(
                        scale_factor,
                        dimensions,
                        *clip_rect,
                    )];

                    let (vertices, indices) = self.create_subbuffers(mesh)?;

                    let desc_set = self
                        .texture_desc_sets
                        .get(&mesh.texture_id)
                        .ok_or(RhError::PipelineError)?
                        .clone();
                    builder
                        .bind_pipeline_graphics(self.pipeline.clone())
                        .set_viewport(
                            0,
                            vec![Viewport {
                                origin: [0.0, 0.0],
                                dimensions: [
                                    dimensions[0] as f32,
                                    dimensions[1] as f32,
                                ],
                                depth_range: 0.0..1.0,
                            }],
                        )
                        .set_scissor(0, scissors)
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.pipeline.layout().clone(),
                            0,
                            desc_set.clone(),
                        )
                        .push_constants(
                            self.pipeline.layout().clone(),
                            0,
                            push_constants,
                        )
                        .bind_vertex_buffers(0, vertices.clone())
                        .bind_index_buffer(indices.clone())
                        .draw_indexed(indices.len() as u32, 1, 0, 0, 0)?;
                }
                Primitive::Callback(_callback) => {
                    // FIXME Figure this out
                    error!("RENDER CALLBACK ATTEMPTED");
                }
            }
        }
        Ok(())
    }

    pub fn render_resources(&self) -> RenderResources {
        RenderResources {
            queue: self.queue(),
            //subpass: self.subpass.clone(),
            memory_allocator: self.memory.memory_allocator.clone(),
            descriptor_set_allocator: &self.memory.set_allocator,
            command_buffer_allocator: &self.memory.command_buffer_allocator,
        }
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.gfx_queue.clone()
    }

    pub fn allocators(&self) -> &Memory {
        &self.memory
    }
}

/// A set of resources used to construct the render pipeline. These can be reused
/// to create additional pipelines and buffers to be rendered in a `PaintCallback`.
///
/// # Example
///
/// See the `triangle` demo source for a detailed usage example.
#[derive(Clone)]
pub struct RenderResources<'a> {
    pub memory_allocator: Arc<StandardMemoryAllocator>,
    pub descriptor_set_allocator: &'a StandardDescriptorSetAllocator,
    pub command_buffer_allocator: &'a StandardCommandBufferAllocator,
    pub queue: Arc<Queue>,
    //pub subpass: Subpass,
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 tex_coords;
layout(location = 2) in vec4 color;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_tex_coords;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    int need_srgb_conv;
} push_constants;

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb * 255.0), srgba.a);
}

void main() {
  gl_Position =
      vec4(2.0 * position.x / push_constants.screen_size.x - 1.0,
           2.0 * position.y / push_constants.screen_size.y - 1.0, 0.0, 1.0);
  // We must convert vertex color to linear
  v_color = linear_from_srgba(color);
  v_tex_coords = tex_coords;
}",
    }
}

// Similar to https://github.com/ArjunNair/egui_sdl2_gl/blob/main/src/painter.rs
mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

layout(push_constant) uniform PushConstants {
    vec2 screen_size;
    int need_srgb_conv;
} push_constants;

// 0-255 sRGB  from  0-1 linear
vec3 srgb_from_linear(vec3 rgb) {
  bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
  vec3 lower = rgb * vec3(3294.6);
  vec3 higher = vec3(269.025) * pow(rgb, vec3(1.0 / 2.4)) - vec3(14.025);
  return mix(higher, lower, vec3(cutoff));
}

vec4 srgba_from_linear(vec4 rgba) {
  return vec4(srgb_from_linear(rgba.rgb), 255.0 * rgba.a);
}

// 0-1 linear  from  0-255 sRGB
vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb * 255.0), srgba.a);
}

void main() {
    vec4 texture_color = texture(font_texture, v_tex_coords);

    if (push_constants.need_srgb_conv == 0) {
        f_color = v_color * texture_color;
    } else {
        f_color = srgba_from_linear(v_color * texture_color) / 255.0;
        f_color.a = pow(f_color.a, 1.6);
    }
}"
    }
}
