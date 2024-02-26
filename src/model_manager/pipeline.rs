use crate::{
    mesh_import::Style,
    pbr_lights::PbrLightTrait,
    rh_error::RhError,
    types::{CameraTrait, RenderFormat},
    util,
    vertex::{RigidFormat, SkinnedFormat},
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
        GraphicsPipeline, PipelineLayout,
    },
    sampler::{
        Filter, Sampler, SamplerAddressMode, SamplerCreateInfo,
        SamplerMipmapMode, LOD_CLAMP_NONE,
    },
    shader::ShaderModule,
};

const VPL_SET: u32 = 0;
const VPL_BINDING: u32 = 0;

// Two descriptor sets: one for view and projection matrices which change once
// per frame, the other for model matrix which is different for each model
pub type UniformVPL = skinned_vs::VPL;
pub type UniformM = skinned_vs::M;
pub type PushConstantData = pbr_fs::PushConstantData;

pub struct Pipeline {
    pub style: Style,
    pub graphics: Arc<GraphicsPipeline>,
    pub sampler: Arc<Sampler>,
    pub vpl_pool: SubbufferAllocator,
    pub m_pool: SubbufferAllocator,
}

/// Helper function to build a vulkano pipeline compatible with vertex format T
///
/// # Errors
/// May return `RhError`
fn build_vk_pipeline<T: Vertex>(
    device: Arc<Device>,
    render_format: &RenderFormat,
    vert_shader: &Arc<ShaderModule>,
    frag_shader: &Arc<ShaderModule>,
) -> Result<Arc<GraphicsPipeline>, RhError> {
    Ok(GraphicsPipeline::start()
        .render_pass(PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(render_format.colour_format)],
            depth_attachment_format: Some(render_format.depth_format),
            ..Default::default()
        })
        .multisample_state(MultisampleState {
            rasterization_samples: render_format.sample_count,
            ..Default::default()
        })
        .color_blend_state(util::alpha_blend_enable())
        .vertex_input_state(T::per_vertex())
        .input_assembly_state(InputAssemblyState::new())
        .vertex_shader(
            vert_shader
                .entry_point("main")
                .ok_or(RhError::VertexShaderError)?,
            (),
        )
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(
            frag_shader
                .entry_point("main")
                .ok_or(RhError::FragmentShaderError)?,
            (),
        )
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .build(device)?)
}

impl Pipeline {
    /// Creates a graphics pipeline. The `Style` should be selected to be
    /// compatible with the desired functionality and vertex format
    /// (`RigidFormat`, `SkinnedFormat`, etc.) used by the mesh.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn new(
        style: Style,
        device: Arc<Device>,
        mem_allocator: Arc<StandardMemoryAllocator>,
        render_format: &RenderFormat,
    ) -> Result<Self, RhError> {
        // Vulkano pipeline is build based on the style, which determines the
        // shaders and the expected vertex format

        //const USE_VIZ: bool = true; // Use for testing FIXME

        const USE_VIZ: bool = false; // Use for testing FIXME

        let frag_shader = if USE_VIZ {
            viz_fs::load(device.clone())?
        } else {
            pbr_fs::load(device.clone())?
        };
        let graphics = {
            match style {
                Style::Rigid => build_vk_pipeline::<RigidFormat>(
                    device.clone(),
                    render_format,
                    &rigid_vs::load(device.clone())?,
                    &frag_shader,
                ),
                Style::Skinned => build_vk_pipeline::<SkinnedFormat>(
                    device.clone(),
                    render_format,
                    &skinned_vs::load(device.clone())?,
                    &frag_shader,
                ),
            }
        }?;

        // Probably don't need to create a new sampler each time since all
        // of these are the same
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

        // These need to be customizable somehow
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
            style,
            graphics,
            sampler,
            vpl_pool,
            m_pool,
        })
    }

    #[must_use]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        use vulkano::pipeline::Pipeline as VulkanoPipeline; // For trait
        self.graphics.layout()
    }

    /// # Errors
    /// May return `RhError`
    pub fn start_pass<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
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
            desc_set_allocator,
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
mod rigid_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/rigid.vert.glsl",
    }
}

mod skinned_vs {
    // TODO: It would be nice if we could pass the const to the shader
    use crate::types::MAX_JOINTS;
    #[allow(clippy::assertions_on_constants)]
    const _: () = assert!(MAX_JOINTS == 32, "MAX_JOINTS must be 32");

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/skinned.vert.glsl",
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
