// `vs_quad` and `fs_quad` are not that similar, but clippy disagrees
#![allow(clippy::similar_names)]
use crate::rh_error::RhError;
use crate::types::DeviceAccess;
use crate::util;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    image::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    image::view::ImageView,
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter,
    },
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint,
        PipelineLayout, PipelineShaderStageCreateInfo,
    },
    Validated,
};

// Create vertex format struct
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod, Vertex)]
struct QVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

// Quad drawn efficiently as triangle strip with shader figuring out UV
const QUAD: [QVertex; 4] = [
    QVertex {
        position: [-1.0, -1.0],
    },
    QVertex {
        position: [-1.0, 1.0],
    },
    QVertex {
        position: [1.0, -1.0],
    },
    QVertex {
        position: [1.0, 1.0],
    },
];

pub struct PostProcess {
    vertex_buffer: Subbuffer<[QVertex]>,
    pipeline: Arc<GraphicsPipeline>,
    sampler: Arc<Sampler>,
    image_view: Arc<ImageView>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    input_format: vulkano::format::Format,
}

impl PostProcess {
    /// Records commands to cbb parameter that must be submitted before rendering
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    #[allow(clippy::too_many_lines)]
    pub fn new<T>(
        device_access: &DeviceAccess<T>,
        mem_allocator: Arc<dyn MemoryAllocator>,
        dimensions: [u32; 2],
        input_format: vulkano::format::Format,
        output_format: vulkano::format::Format,
        sampler_option: Option<Arc<Sampler>>,
    ) -> Result<Self, RhError> {
        // FIXME Check if these vertices end up in reasonable memory without
        // queing commands. Even better, generate them in shaders.
        let vertex_buffer = Buffer::from_iter(
            mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            QUAD,
        )
        .map_err(Validated::unwrap)?;

        let pipeline = {
            let vs = vs_quad::load(device_access.device.clone())
                .map_err(Validated::unwrap)?
                .entry_point("main")
                .ok_or(RhError::VertexShaderError)?;
            let fs = fs_quad::load(device_access.device.clone())
                .map_err(Validated::unwrap)?
                .entry_point("main")
                .ok_or(RhError::FragmentShaderError)?;

            let vertex_input_state = QVertex::per_vertex()
                .definition(&vs.info().input_interface)
                .unwrap(); // `Box<ValidationError>`

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                device_access.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(
                        device_access.device.clone(),
                    )?,
            )
            .map_err(Validated::unwrap)?;

            let subpass = PipelineRenderingCreateInfo {
                color_attachment_formats: vec![Some(output_format)],
                ..PipelineRenderingCreateInfo::default()
            };

            GraphicsPipeline::new(
                device_access.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState {
                        topology: PrimitiveTopology::TriangleStrip,
                        ..InputAssemblyState::default()
                    }),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(
                        ColorBlendState::with_attachment_states(
                            u32::try_from(
                                subpass.color_attachment_formats.len(),
                            )
                            .unwrap(),
                            ColorBlendAttachmentState::default(),
                        ),
                    ),
                    dynamic_state: std::iter::once(DynamicState::Viewport)
                        .collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .map_err(Validated::unwrap)?
        };

        let sampler = if let Some(s) = sampler_option {
            s
        } else {
            Sampler::new(
                device_access.device.clone(),
                SamplerCreateInfo {
                    mag_filter: Filter::Linear,
                    min_filter: Filter::Linear,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    ..Default::default()
                },
            )
            .map_err(Validated::unwrap)?
        };
        let (image_view, descriptor_set) = Self::create_target(
            mem_allocator,
            &device_access.set_allocator,
            dimensions,
            input_format,
            &pipeline,
            sampler.clone(),
        )?;

        Ok(Self {
            vertex_buffer,
            pipeline,
            sampler,
            image_view,
            descriptor_set,
            input_format,
        })
    }

    #[must_use]
    pub fn pipeline(&self) -> Arc<GraphicsPipeline> {
        self.pipeline.clone()
    }

    #[must_use]
    pub fn sampler(&self) -> Arc<Sampler> {
        self.sampler.clone()
    }

    #[must_use]
    pub fn image_view(&self) -> Arc<ImageView> {
        self.image_view.clone()
    }

    #[must_use]
    pub fn descriptor_set(&self) -> Arc<PersistentDescriptorSet> {
        self.descriptor_set.clone()
    }

    /// # Errors
    /// May return `RhError`
    pub fn resize(
        &mut self,
        mem_allocator: Arc<dyn MemoryAllocator>,
        set_allocator: &StandardDescriptorSetAllocator,
        dimensions: [u32; 2],
    ) -> Result<(), RhError> {
        (self.image_view, self.descriptor_set) = Self::create_target(
            mem_allocator,
            set_allocator,
            dimensions,
            self.input_format,
            &self.pipeline,
            self.sampler.clone(),
        )?;
        Ok(())
    }

    /// The draw command for the postprocess pass
    ///
    /// # Errors
    /// Currently will NOT return an `RhError`. The return type is leftover
    /// from use with an earlier version of vulkano.
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    pub fn draw<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        // CommandBufferBuilder should have a render pass started
        cbb.bind_pipeline_graphics(self.pipeline.clone())
            .unwrap() // `Box<ValidationError>`
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0, // first_set
                self.descriptor_set.clone(),
            )
            .unwrap() // `Box<ValidationError>`
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap() // `Box<ValidationError>`
            .draw(
                4, // vertex_count
                1, // instance_count
                0, // first_vertex
                0, // first_instance
            )
            .unwrap(); // `Box<ValidationError>`
        Ok(())
    }

    // Private helpers
    fn create_target(
        mem_allocator: Arc<dyn MemoryAllocator>,
        set_allocator: &StandardDescriptorSetAllocator,
        dimensions: [u32; 2],
        format: vulkano::format::Format,
        pipeline: &Arc<GraphicsPipeline>,
        sampler: Arc<Sampler>,
    ) -> Result<(Arc<ImageView>, Arc<PersistentDescriptorSet>), RhError> {
        let target_image_view =
            util::create_target(mem_allocator, dimensions, format)?;
        let descriptor_set = PersistentDescriptorSet::new(
            set_allocator,
            util::get_layout(pipeline, 0)?.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                target_image_view.clone(),
                sampler,
            )],
            [],
        )
        .map_err(Validated::unwrap)?;

        Ok((target_image_view, descriptor_set))
    }
}

mod vs_quad {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/post.vert",
    }
}

mod fs_quad {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/post.frag",
    }
}
