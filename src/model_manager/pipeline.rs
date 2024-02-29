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
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::ShaderModule,
    Validated,
};

#[allow(unused_imports)]
use log::{debug, error, info, trace};

const VPL_SET: u32 = 0;
const VPL_BINDING: u32 = 0;

// Two descriptor sets: one for view and projection matrices which change once
// per frame, the other for model matrix which is different for each model
pub type UniformVPL = skinned_vs::VPL;
pub type UniformM = skinned_vs::M;

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

        // Pipeline layout could potentially be shared. It collects some
        // descriptor set layouts which could also be shared.
        // CAUTION: The descriptor set for the albedo texture is deduced from
        // the shader. It must match the one manually created for the default
        // material. That is done using `layout::create_set`. Ideally that one
        // would be shared or another created the same way, but that doesn't
        // cover the other descriptor sets so this is a possible TODO.
        let layout = {
            let set_info =
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages);
            debug!("PipelineDescriptorSetLayoutCreateInfo={:?}", set_info);
            PipelineLayout::new(
                device.clone(),
                set_info.into_pipeline_layout_create_info(device.clone())?,
            )
            .map_err(Validated::unwrap)?
            /*Debug info is something like this:
            VPL {0: DescriptorSetLayoutBinding { binding_flags: empty(),
                descriptor_type: UniformBuffer, descriptor_count: 1,
                stages: VERTEX | FRAGMENT, immutable_samplers: [],
                _ne: NonExhaustive(()) }}
            M {0: DescriptorSetLayoutBinding { binding_flags: empty(),
                descriptor_type: UniformBuffer, descriptor_count: 1,
                stages: VERTEX, immutable_samplers: [],
                _ne: NonExhaustive(()) }}
            tex {0: DescriptorSetLayoutBinding { binding_flags: empty(),
                descriptor_type: CombinedImageSampler, descriptor_count: 1,
                stages: FRAGMENT, immutable_samplers: [],
                _ne: NonExhaustive(()) }}
            push constants:
                [PushConstantRange { stages: FRAGMENT, offset: 0, size: 24 }]
            */
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

        //const USE_VIZ: bool = true; // Use for testing FIXME

        const USE_VIZ: bool = false; // Use for testing FIXME

        let frag_shader = if USE_VIZ {
            viz_fs::load(device.clone()).map_err(Validated::unwrap)?
        } else {
            pbr_fs::load(device.clone()).map_err(Validated::unwrap)?
        };
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

        // These need to be customizable somehow
        // For vulkano 0.34 these need to have `memory_type_filter` set
        // like below, or similar. Without that they can end up not accesible
        // to the host and result in a `VkHostAccessError(NotHostMapped` when
        // trying to write them from the CPU.
        let vpl_pool = SubbufferAllocator::new(
            mem_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
        );
        let m_pool = SubbufferAllocator::new(
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
            vpl_pool,
            m_pool,
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
        // Uniform buffer for view and projection matrix is in descriptor set
        // 0 since this will constant for the entire frame.
        // Lights are also here as an experiment.
        let vpl_buffer = {
            let data = UniformVPL {
                proj: camera.proj_matrix().into(),
                ambient: lights.ambient_array(),
                lights: lights.light_array(&camera.view_matrix()),
            };
            //trace!("About to write to 'UniformVPL' UBO");
            let buffer = self.vpl_pool.allocate_sized()?;
            *buffer.write()? = data;
            buffer
        };
        let desc_set = PersistentDescriptorSet::new(
            desc_set_allocator,
            util::get_layout(&self.graphics, VPL_SET as usize)?.clone(),
            [WriteDescriptorSet::buffer(VPL_BINDING, vpl_buffer)],
            [],
        )
        .map_err(Validated::unwrap)?;
        cbb.bind_pipeline_graphics(self.graphics.clone())
            .unwrap() // `Box<ValidationError>`
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.layout().clone(),
                VPL_SET,
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
