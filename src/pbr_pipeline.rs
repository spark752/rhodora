use crate::{
    pbr_lights::PbrLightTrait,
    rh_error::RhError,
    types::{CameraTrait, RenderFormat},
    util,
    vertex::Position,
};
use std::sync::Arc;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferUsage,
    },
    command_buffer::AutoCommandBufferBuilder,
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::StandardMemoryAllocator,
    pipeline::PipelineBindPoint,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState, multisample::MultisampleState,
            render_pass::PipelineRenderingCreateInfo, vertex_input::Vertex,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineLayout,
    },
    sampler::{
        Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode, LOD_CLAMP_NONE,
    },
};

const VPL_SET: u32 = 0;
const VPL_BINDING: u32 = 0;

// Two descriptor sets: one for view and projection matrices which change once
// per frame, the other for model matrix which is different for each model
pub type UniformVPL = vs::VPL;
pub type UniformM = vs::M;
pub type PushConstantData = pbr_fs::PushConstantData;

pub struct PbrPipeline {
    pub graphics: Arc<GraphicsPipeline>,
    pub sampler: Arc<Sampler>,
    pub vpl_pool: SubbufferAllocator,
    pub m_pool: SubbufferAllocator,
}

impl PbrPipeline {
    /// Create a PBR pipeline compatible with a `Position` vertex buffer and
    /// an interleaved vertex buffer of type T.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn new<T: Vertex>(
        device: Arc<Device>,
        mem_allocator: Arc<StandardMemoryAllocator>,
        render_format: &RenderFormat,
    ) -> Result<Self, RhError> {
        let vert_shader = vs::load(device.clone())?;
        let frag_shader = pbr_fs::load(device.clone())?;
        //let frag_shader = viz_fs::load(device.clone())?; // TEST
        let pipeline =
            GraphicsPipeline::start()
                .render_pass(PipelineRenderingCreateInfo {
                    color_attachment_formats: vec![Some(
                        render_format.colour_format,
                    )],
                    depth_attachment_format: Some(render_format.depth_format),
                    ..Default::default()
                })
                .multisample_state(MultisampleState {
                    rasterization_samples: render_format.sample_count,
                    ..Default::default()
                })
                .color_blend_state(util::alpha_blend_enable())
                .vertex_input_state([Position::per_vertex(), T::per_vertex()])
                .input_assembly_state(InputAssemblyState::new())
                .vertex_shader(
                    vert_shader
                        .entry_point("main")
                        .ok_or(RhError::VertexShaderError)?,
                    (),
                )
                .viewport_state(
                    ViewportState::viewport_dynamic_scissor_irrelevant(),
                )
                .fragment_shader(
                    frag_shader
                        .entry_point("main")
                        .ok_or(RhError::FragmentShaderError)?,
                    (),
                )
                .depth_stencil_state(DepthStencilState::simple_depth_test())
                .build(device.clone())?;
        let sampler = Sampler::new(
            device,
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Linear,
                lod: 0.0..=LOD_CLAMP_NONE,
                //mip_lod_bias: -2.0, // for testing, negative = use larger map
                ..Default::default()
            },
        )?;
        let vpl_pool = SubbufferAllocator::new(
            mem_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );
        let m_pool = SubbufferAllocator::new(
            mem_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );
        Ok(Self {
            graphics: pipeline,
            sampler,
            vpl_pool,
            m_pool,
        })
    }

    pub fn layout(&self) -> &Arc<PipelineLayout> {
        self.graphics.layout()
    }

    /// # Errors
    /// May return `RhError`
    pub fn start_pass<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        descriptor_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        // Uniform buffer for view and projection matrix is in descriptor set
        // 0 since this will constant for the entire frame.
        // Lights are also here as an experiment.
        let vpl_buffer = {
            let data = UniformVPL {
                proj: camera.proj_matrix().into(),
                ambient: lights.ambient_array(),
                lights: lights.light_array(&camera.view_matrix()),
            };
            let buffer = self.vpl_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let desc_set = PersistentDescriptorSet::new(
            descriptor_set_allocator,
            util::get_layout(&self.graphics, VPL_SET as usize)?.clone(),
            [WriteDescriptorSet::buffer(VPL_BINDING, vpl_buffer)],
        )?;
        cbb.bind_pipeline_graphics(self.graphics.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.layout().clone(),
                VPL_SET,
                desc_set,
            );
        Ok(())
    }
}

// Shaders
mod vs {
    // TODO: It would be nice if we could pass the const to the shader
    use crate::types::MAX_JOINTS;
    #[allow(clippy::assertions_on_constants)]
    const _: () = assert!(MAX_JOINTS == 32, "MAX_JOINTS must be 32");

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/pbr.vert.glsl",
        define: [("MAX_JOINTS", "32")],
    }
}

mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag.glsl",
    }
}

// Visualization shader for development and debugging
mod viz_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag.glsl",
        define: [("VISUALIZE", "1")],
    }
}
