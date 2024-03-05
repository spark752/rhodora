use super::layout;
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
    image::sampler::Sampler,
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::PipelineBindPoint,
    pipeline::{
        graphics::{
            depth_stencil::{DepthState, DepthStencilState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        DynamicState, GraphicsPipeline, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::ShaderModule,
    Validated,
};

#[allow(unused_imports)]
use log::{debug, error, info, trace};

use super::layout::{
    LAYOUT_LIGHTS_BINDING, LAYOUT_PASS_SET, LAYOUT_PROJ_BINDING,
};

// Two descriptor sets: one for view and projection matrices which change once
// per frame, the other for model matrix which is different for each model
//pub type UniformVPL = skinned_vs::VPL;
pub type UniformM = skinned_vs::M;

pub struct Pipeline {
    pub style: Style,
    pub graphics: Arc<GraphicsPipeline>,
    pub sampler: Arc<Sampler>,
    pub subbuffer_pool: SubbufferAllocator,
}

/// Helper function to build a vulkano pipeline compatible with vertex format T
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
fn build_vk_pipeline<T: Vertex>(
    device: Arc<Device>,
    render_format: &RenderFormat,
    vert_shader: &Arc<ShaderModule>,
    frag_shader: &Arc<ShaderModule>,
) -> Result<Arc<GraphicsPipeline>, RhError> {
    let pipeline = {
        let vs = vert_shader
            .entry_point("main")
            .ok_or(RhError::VertexShaderError)?;
        let fs = frag_shader
            .entry_point("main")
            .ok_or(RhError::FragmentShaderError)?;

        let vertex_input_state = T::per_vertex()
            .definition(&vs.info().input_interface)
            .unwrap(); // `Box<ValidationError>`

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        // Pipeline layout could potentially be shared. It is put together
        // manually instead of inspecting shaders, so is currently the same
        // for all main pass pipelines.
        let layout = {
            let set_info = layout::pipeline_create_info();
            PipelineLayout::new(
                device.clone(),
                set_info.into_pipeline_layout_create_info(device.clone())?,
            )
            .map_err(Validated::unwrap)?
        };

        let subpass = PipelineRenderingCreateInfo {
            color_attachment_formats: vec![Some(render_format.colour_format)],
            depth_attachment_format: Some(render_format.depth_format),
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
                multisample_state: Some(MultisampleState {
                    rasterization_samples: render_format.sample_count,
                    ..Default::default()
                }),
                depth_stencil_state: Some(DepthStencilState {
                    depth: Some(DepthState::simple()),
                    ..Default::default()
                }),
                color_blend_state: Some(util::alpha_blend_enable()),
                dynamic_state: std::iter::once(DynamicState::Viewport)
                    .collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .map_err(Validated::unwrap)?
    };
    Ok(pipeline)
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
        sampler: Arc<Sampler>,
    ) -> Result<Self, RhError> {
        // Vulkano pipeline is build based on the style, which determines the
        // shaders and the expected vertex format
        let frag_shader =
            pbr_fs::load(device.clone()).map_err(Validated::unwrap)?;
        let graphics = {
            match style {
                Style::Rigid => build_vk_pipeline::<RigidFormat>(
                    device.clone(),
                    render_format,
                    &rigid_vs::load(device).map_err(Validated::unwrap)?,
                    &frag_shader,
                ),
                Style::Skinned => build_vk_pipeline::<SkinnedFormat>(
                    device.clone(),
                    render_format,
                    &skinned_vs::load(device).map_err(Validated::unwrap)?,
                    &frag_shader,
                ),
            }
        }?;

        // Not sure how this works FIXME
        // For vulkano 0.34 these need to have `memory_type_filter` set
        // like below, or similar. Without that they can end up not accesible
        // to the host and result in a `VkHostAccessError(NotHostMapped` when
        // trying to write them from the CPU.
        let subbuffer_pool = SubbufferAllocator::new(
            mem_allocator,
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );

        Ok(Self {
            style,
            graphics,
            sampler,
            subbuffer_pool,
        })
    }

    #[must_use]
    pub fn layout(&self) -> &Arc<PipelineLayout> {
        use vulkano::pipeline::Pipeline as VulkanoPipeline; // For trait
        self.graphics.layout()
    }

    /// Start rendering pass for this pipeline
    ///
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    pub fn start_pass<T>(
        &self,
        cbb: &mut AutoCommandBufferBuilder<T>,
        desc_set_allocator: &StandardDescriptorSetAllocator,
        camera: &impl CameraTrait,
        lights: &impl PbrLightTrait,
    ) -> Result<(), RhError> {
        //trace!("Pipeline start_pass");

        // `LAYOUT_PASS_SET` is descriptor set point used for things that
        // keep the same value for the entire pass, such as projection matrix
        // and lights. Since binding a descriptor set automatically unbinds
        // higher numbered sets it should probably be equal to 0.
        // FIXME Use layouts instead of relying on shader inspection
        let proj_buffer = {
            let data = skinned_vs::VPL {
                proj: camera.proj_matrix().into(),
            };
            let buffer = self.subbuffer_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let lights_buffer = {
            let data = pbr_fs::VPL {
                ambient: lights.ambient_array(),
                lights: lights.light_array(&camera.view_matrix()),
            };
            let buffer = self.subbuffer_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };

        // There are Vulkan alignment requirements for descriptors pointing
        // to the same uniform buffer, but vulkano seems to be taking care
        // of that for us.
        let desc_set = PersistentDescriptorSet::new(
            desc_set_allocator,
            util::get_layout(&self.graphics, LAYOUT_PASS_SET)?.clone(),
            [
                WriteDescriptorSet::buffer(LAYOUT_PROJ_BINDING, proj_buffer),
                WriteDescriptorSet::buffer(
                    LAYOUT_LIGHTS_BINDING,
                    lights_buffer,
                ),
            ],
            [],
        )
        .map_err(Validated::unwrap)?;
        cbb.bind_pipeline_graphics(self.graphics.clone())
            .unwrap() // `Box<ValidationError>`
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.layout().clone(),
                LAYOUT_PASS_SET,
                desc_set,
            )
            .unwrap(); // `Box<ValidationError>`
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

// Shader with Visualization options for development and debugging
#[cfg(feature = "visualize")]
mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag.glsl",
        define: [("VISUALIZE", "1")],
    }
}

// Shader with visualization disabled
#[cfg(not(feature = "visualize"))]
mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag.glsl",
    }
}
