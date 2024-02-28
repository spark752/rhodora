// This file is based on egui_winit_vulkano which is
// Copyright (c) 2021 Okko Hakola
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or https://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

// Allow conversions of small value `u32` to `f32` to work with different APIs
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

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
    device::{Device, Queue},
    format::{Format, NumericFormat},
    image::sampler::{
        Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode,
    },
    image::{view::ImageView, Image, ImageCreateInfo, ImageLayout, ImageUsage},
    memory::allocator::{
        AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator,
    },
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp,
                ColorBlendAttachmentState, ColorBlendState, ColorComponents,
            },
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint,
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    sync::GpuFuture,
    DeviceSize, Validated,
};

const VERTICES_PER_QUAD: DeviceSize = 4;
const VERTEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * VERTICES_PER_QUAD;
const INDEX_BUFFER_SIZE: DeviceSize = 1024 * 1024 * 2;

/// Should match vertex definition of egui (except colour is `[f32; 4]`)
#[repr(C)]
#[derive(Default, Debug, Clone, Copy, Zeroable, Pod, Vertex)]
pub struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub tex_coords: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub colour: [f32; 4],
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
    texture_images: AHashMap<egui::TextureId, Arc<ImageView>>,
    next_native_tex_id: u64,
}

impl Renderer {
    /// # Errors
    /// May return `RhError`
    pub fn new(gfx_queue: Arc<Queue>, format: Format) -> Result<Self, RhError> {
        let need_srgb_conv = format
            .numeric_format_color()
            .map_or(false, |t| t == NumericFormat::UNORM);
        let memory = Memory::new(gfx_queue.device());
        let (vertex_buffer_pool, index_buffer_pool) =
            create_buffers(&memory.memory_allocator)?;
        let pipeline = create_pipeline(gfx_queue.device().clone(), format)?;
        let sampler = Sampler::new(
            gfx_queue.device().clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;
        Ok(Self {
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

    /// Creates a descriptor set for images from an `ImageView`
    fn sampled_image_desc_set(
        &self,
        layout: &Arc<DescriptorSetLayout>,
        image_view: Arc<ImageView>,
    ) -> Result<Arc<PersistentDescriptorSet>, RhError> {
        Ok(PersistentDescriptorSet::new(
            &self.memory.set_allocator,
            layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                image_view,
                self.sampler.clone(),
            )],
            [],
        )
        .map_err(Validated::unwrap)?)
    }

    /// Registers a user texture. User texture needs to be unregistered when
    /// it is no longer needed.
    ///
    /// This takes an `Image` and creates and stores an `ImageView`.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn register_image(
        &mut self,
        image: Arc<Image>,
    ) -> Result<egui::TextureId, RhError> {
        let image_view =
            ImageView::new_default(image).map_err(Validated::unwrap)?;
        let layout = self
            .pipeline
            .layout()
            .set_layouts()
            .first()
            .ok_or(RhError::PipelineError);
        let desc_set = self.sampled_image_desc_set(layout?, image_view.clone());
        let id = egui::TextureId::User(self.next_native_tex_id);
        self.next_native_tex_id += 1;
        self.texture_desc_sets.insert(id, desc_set?);
        self.texture_images.insert(id, image_view);
        Ok(id)
    }

    /// Unregister user texture.
    pub fn unregister_image(&mut self, texture_id: egui::TextureId) {
        self.texture_desc_sets.remove(&texture_id);
        self.texture_images.remove(&texture_id);
    }

    #[allow(clippy::too_many_lines)]
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
                    .flat_map(egui::Color32::to_array)
                    .collect()
            }
            egui::ImageData::Font(image) => image
                .srgba_pixels(None)
                .flat_map(|color| color.to_array())
                .collect(),
        };

        // Create command buffer builder
        let mut cbb = AutoCommandBufferBuilder::primary(
            &self.memory.command_buffer_allocator,
            self.gfx_queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(Validated::unwrap)?;

        // Create buffer to be copied to the image
        let texture_data_buffer = Buffer::from_iter(
            self.memory.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                // FIXME Guessing at these values
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            data,
        )
        .map_err(Validated::unwrap)?;

        // Create image
        let image = {
            let extent =
                [delta.image.width() as u32, delta.image.height() as u32, 1];
            Image::new(
                self.memory.memory_allocator.clone(),
                ImageCreateInfo {
                    format: Format::R8G8B8A8_SRGB,
                    extent,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .map_err(Validated::unwrap)?
        };

        // Create commands to copy buffer to image
        cbb.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            texture_data_buffer,
            image.clone(),
        ))
        .unwrap(); // This is a Box<ValidationError>

        // Create a image view of the image
        let image_view =
            ImageView::new_default(image).map_err(Validated::unwrap)?;

        // Blit texture data to existing image if delta pos exists
        // (e.g. font changed)
        if let Some(pos) = delta.pos {
            if let Some(existing_image) = self.texture_images.get(&texture_id) {
                let src_dims = image_view.image().extent();
                let top_left = [pos[0] as u32, pos[1] as u32, 0];
                let bottom_right = [
                    pos[0] as u32 + src_dims[0],
                    pos[1] as u32 + src_dims[1],
                    1,
                ];

                cbb.blit_image(BlitImageInfo {
                    src_image_layout: ImageLayout::General,
                    dst_image_layout: ImageLayout::General,
                    regions: [ImageBlit {
                        src_subresource: image_view
                            .image()
                            .subresource_layers(),
                        src_offsets: [[0, 0, 0], [src_dims[0], src_dims[1], 1]],
                        dst_subresource: existing_image
                            .image()
                            .subresource_layers(),
                        dst_offsets: [top_left, bottom_right],
                        ..Default::default()
                    }]
                    .into(),
                    filter: Filter::Nearest,
                    ..BlitImageInfo::images(
                        image_view.image().clone(),
                        existing_image.image().clone(),
                    )
                })
                .unwrap(); // This is a `Box<ValidationError>`
            }
            // Otherwise save the newly created image
        } else {
            let layout = self
                .pipeline
                .layout()
                .set_layouts()
                .first()
                .ok_or(RhError::PipelineError);
            let font_desc_set =
                self.sampled_image_desc_set(layout?, image_view.clone());
            self.texture_desc_sets.insert(texture_id, font_desc_set?);
            self.texture_images.insert(texture_id, image_view);
        }

        // Execute command buffer
        let command_buffer = cbb.build().map_err(Validated::unwrap)?;
        let finished = command_buffer.execute(self.gfx_queue.clone())?;
        let _fut = finished
            .then_signal_fence_and_flush()
            .map_err(Validated::unwrap)?;
        Ok(())
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
                colour: [
                    f32::from(v.color.r()) / 255.0,
                    f32::from(v.color.g()) / 255.0,
                    f32::from(v.color.b()) / 255.0,
                    f32::from(v.color.a()) / 255.0,
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

    /// # Errors
    /// May return `RhError`
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

    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    fn draw_egui<T>(
        &mut self,
        scale_factor: f32,
        clipped_meshes: &[ClippedPrimitive],
        dimensions: [u32; 2],
        builder: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        let extent = [dimensions[0] as f32, dimensions[1] as f32];
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

                    let (vertices, indices) = self.create_subbuffers(mesh)?;

                    let desc_set = self
                        .texture_desc_sets
                        .get(&mesh.texture_id)
                        .ok_or(RhError::PipelineError)?
                        .clone();
                    builder
                        .bind_pipeline_graphics(self.pipeline.clone())
                        .unwrap() // `Box<ValidationError>`
                        .set_viewport(
                            0,
                            std::iter::once(Viewport {
                                offset: [0.0, 0.0],
                                extent,
                                depth_range: 0.0..=1.0,
                            })
                            .collect(),
                        )
                        .unwrap() // `Box<ValidationError>`
                        .set_scissor(
                            0,
                            std::iter::once(get_rect_scissor(
                                scale_factor,
                                dimensions,
                                *clip_rect,
                            ))
                            .collect(),
                        )
                        .unwrap() // `Box<ValidationError>`
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.pipeline.layout().clone(),
                            0,
                            desc_set.clone(),
                        )
                        .unwrap() // `Box<ValidationError>`
                        .push_constants(
                            self.pipeline.layout().clone(),
                            0,
                            push_constants,
                        )
                        .unwrap() // `Box<ValidationError>`
                        .bind_vertex_buffers(0, vertices.clone())
                        .unwrap() // `Box<ValidationError>`
                        .bind_index_buffer(indices.clone())
                        .unwrap() // `Box<ValidationError>`
                        .draw_indexed(
                            u32::try_from(indices.len())
                                .map_err(|_| RhError::IndexCountTooLarge)?,
                            1,
                            0,
                            0,
                            0,
                        )
                        .unwrap(); // `Box<ValidationError>`
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

    pub const fn allocators(&self) -> &Memory {
        &self.memory
    }
}

fn get_rect_scissor(
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

    // min.x and min.y were already clamped to be at least 0
    #[allow(clippy::cast_sign_loss)]
    Scissor {
        offset: [min.x.round() as u32, min.y.round() as u32],
        extent: [
            (max.x.round() - min.x) as u32,
            (max.y.round() - min.y) as u32,
        ],
    }
}

/// Helper function for creating a graphics pipeline
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
#[allow(clippy::needless_pass_by_value)]
fn create_pipeline(
    device: Arc<Device>, // Not a reference
    format: vulkano::format::Format,
) -> Result<Arc<GraphicsPipeline>, RhError> {
    let pipeline = {
        let vs = vs::load(device.clone())
            .map_err(Validated::unwrap)?
            .entry_point("main")
            .ok_or(RhError::VertexShaderError)?;
        let fs = fs::load(device.clone())
            .map_err(Validated::unwrap)?
            .entry_point("main")
            .ok_or(RhError::FragmentShaderError)?;

        let vertex_input_state = EguiVertex::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap(); // `Box<ValidationError>`

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())?,
        )
        .map_err(Validated::unwrap)?;

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(format)],
            ..Default::default()
        };

        GraphicsPipeline::new(
            device,
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState::default()),
                depth_stencil_state: None,
                color_blend_state: Some(ColorBlendState {
                    attachments: vec![ColorBlendAttachmentState {
                        blend: Some(AttachmentBlend {
                            // Not typical alpha blend values. Ported from
                            // previous code to vulkano 0.34.
                            src_color_blend_factor: BlendFactor::One,
                            dst_color_blend_factor:
                                BlendFactor::OneMinusSrcAlpha,
                            color_blend_op: BlendOp::Add,
                            src_alpha_blend_factor:
                                BlendFactor::OneMinusDstAlpha,
                            dst_alpha_blend_factor: BlendFactor::One,
                            alpha_blend_op: BlendOp::Add,
                        }),
                        color_write_mask: ColorComponents::all(),
                        color_write_enable: true,
                    }],
                    ..Default::default()
                }),
                // Previous code used `viewport_dynamic_scissor_dynamic`.
                // This is a guess at how to do that in vulkan 0.34.
                // `viewport_state` might also need to be involved.
                // FIXME
                dynamic_state: [DynamicState::Viewport, DynamicState::Scissor]
                    .into_iter()
                    .collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .map_err(Validated::unwrap)?
    };
    Ok(pipeline)
}

/// Helper function for creating buffers
///
/// # Errors
/// May return `RhError`
fn create_buffers(
    allocator: &Arc<StandardMemoryAllocator>,
) -> Result<(SubbufferAllocator, SubbufferAllocator), RhError> {
    // Create vertex and index buffers
    let vertex_buffer_pool = SubbufferAllocator::new(
        allocator.clone(),
        SubbufferAllocatorCreateInfo {
            buffer_usage: BufferUsage::VERTEX_BUFFER,
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
    );
    index_buffer_pool.reserve(INDEX_BUFFER_SIZE)?;

    Ok((vertex_buffer_pool, index_buffer_pool))
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
        path: "shaders/egui.vert.glsl",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/egui.frag.glsl",
    }
}
