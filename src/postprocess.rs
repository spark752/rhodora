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
    image::view::ImageView,
    image::AttachmentImage,
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryUsage},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            render_pass::PipelineRenderingCreateInfo,
            vertex_input::Vertex,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
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
    image_view: Arc<ImageView<AttachmentImage>>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    input_format: vulkano::format::Format,
}

impl PostProcess {
    /// Records commands to cbb parameter that must be submitted before rendering
    ///
    /// # Errors
    /// May return `RhError`
    pub fn new<T>(
        device_access: &DeviceAccess<T>,
        mem_allocator: &(impl MemoryAllocator + ?Sized),
        dimensions: [u32; 2],
        input_format: vulkano::format::Format,
        output_format: vulkano::format::Format,
        sampler_option: Option<Arc<Sampler>>,
    ) -> Result<Self, RhError> {
        // FIXME This was a DeviceLocalBuffer but the convenience functions
        // for this no longer exist in Vulkano. Temporarily use a different
        // type.
        /*
        let vertex_buffer = DeviceLocalBuffer::from_iter(
            mem_allocator,
            QUAD,
            BufferUsage::VERTEX_BUFFER | BufferUsage::TRANSFER_DST,
            device_access.cbb,
        )?;
         */
        let vertex_buffer = Buffer::from_iter(
            mem_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            QUAD,
        )?;

        let vs_quad = vs_quad::load(device_access.device.clone())?;
        let fs_quad = fs_quad::load(device_access.device.clone())?;
        let pipeline =
            GraphicsPipeline::start()
                .render_pass(PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(output_format)],
                    ..Default::default()
                })
                .vertex_input_state(QVertex::per_vertex())
                .input_assembly_state(
                    InputAssemblyState::new()
                        .topology(PrimitiveTopology::TriangleStrip),
                )
                .vertex_shader(
                    vs_quad
                        .entry_point("main")
                        .ok_or(RhError::VertexShaderError)?,
                    (),
                )
                .viewport_state(
                    ViewportState::viewport_dynamic_scissor_irrelevant(),
                )
                .fragment_shader(
                    fs_quad
                        .entry_point("main")
                        .ok_or(RhError::FragmentShaderError)?,
                    (),
                )
                .build(device_access.device.clone())?;
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
            )?
        };
        let (image_view, descriptor_set) = Self::create_target(
            mem_allocator,
            device_access.set_allocator,
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
    pub fn image_view(&self) -> Arc<ImageView<AttachmentImage>> {
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
        mem_allocator: &(impl MemoryAllocator + ?Sized),
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

    /// # Errors
    /// May return `RhError`
    pub fn draw<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
    ) -> Result<(), RhError> {
        // CommandBufferBuilder should have a render pass started
        cbb.bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0, // first_set
                self.descriptor_set.clone(),
            )
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(
                4, // vertex_count
                1, // instance_count
                0, // first_vertex
                0, // first_instance
            )?;
        Ok(())
    }

    // Private helpers
    fn create_target(
        mem_allocator: &(impl MemoryAllocator + ?Sized),
        set_allocator: &StandardDescriptorSetAllocator,
        dimensions: [u32; 2],
        format: vulkano::format::Format,
        pipeline: &Arc<GraphicsPipeline>,
        sampler: Arc<Sampler>,
    ) -> Result<
        (
            Arc<ImageView<AttachmentImage>>,
            Arc<PersistentDescriptorSet>,
        ),
        RhError,
    > {
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
        )?;

        Ok((target_image_view, descriptor_set))
    }
}

mod vs_quad {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/postprocess.vert",
    }
}

mod fs_quad {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/postprocess.frag",
    }
}
