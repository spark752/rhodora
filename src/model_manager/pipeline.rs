use super::layout;
use crate::{
    mesh_import::Style,
    rh_error::RhError,
    types::RenderFormat,
    util,
    vertex::{RigidFormat, SkinnedFormat},
};
use std::sync::Arc;
use vulkano::{
    device::Device,
    image::sampler::Sampler,
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
    shader::{ShaderModule, ShaderModuleCreateInfo},
    Validated,
};

#[allow(unused_imports)]
use log::{debug, error, info, trace};

pub struct Pipeline {
    pub style: Style,
    pub graphics: Arc<GraphicsPipeline>,
    pub sampler: Arc<Sampler>,
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
        render_format: &RenderFormat,
        sampler: Arc<Sampler>,
    ) -> Result<Self, RhError> {
        // Vulkano pipeline is build based on the style, which determines the
        // shaders and the expected vertex format
        let frag_shader = spirv_to_shader(
            Arc::clone(&device),
            include_bytes!(concat!(env!("OUT_DIR"), "/pbr.spv")),
        )
        .map_err(Validated::unwrap)?;
        let graphics = {
            match style {
                Style::Rigid => build_vk_pipeline::<RigidFormat>(
                    Arc::clone(&device),
                    render_format,
                    &spirv_to_shader(
                        device,
                        include_bytes!(concat!(env!("OUT_DIR"), "/rigid.spv")),
                    )
                    .map_err(Validated::unwrap)?,
                    &frag_shader,
                ),
                Style::Skinned => build_vk_pipeline::<SkinnedFormat>(
                    Arc::clone(&device),
                    render_format,
                    &spirv_to_shader(
                        device,
                        include_bytes!(concat!(
                            env!("OUT_DIR"),
                            "/skinned.spv"
                        )),
                    )
                    .map_err(Validated::unwrap)?,
                    &frag_shader,
                ),
            }
        }?;

        Ok(Self {
            style,
            graphics,
            sampler,
        })
    }
}

/// Convert SPIR-V code in a byte slice into a `ShaderModule`
fn spirv_to_shader(
    device: Arc<Device>,
    bytes: &[u8],
) -> Result<Arc<ShaderModule>, Validated<vulkano::VulkanError>> {
    let words: Vec<u32> = vulkano::shader::spirv::bytes_to_words(bytes)
        .unwrap()
        .into_owned();
    unsafe {
        ShaderModule::new(
            device, //
            ShaderModuleCreateInfo::new(&words),
        )
    }
}

// Shaders
// Dump example plus some other stuff that should be cleaned up
/*
mod rigid_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/rigid.vert",
        vulkan_version: "1.2",
        linalg_type: "nalgebra",
    }
}

mod skinned_vs {
    // TODO: It would be nice if we could pass the const to the shader
    use crate::types::MAX_JOINTS;
    #[allow(clippy::assertions_on_constants)]
    const _: () = assert!(MAX_JOINTS == 32, "MAX_JOINTS must be 32");

    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/skinned.vert",
        define: [("MAX_JOINTS", "32")],
        vulkan_version: "1.2",
        linalg_type: "nalgebra",
        //dump: true,
    }
}

// Shader with Visualization options for development and debugging
#[cfg(feature = "visualize")]
mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag",
        define: [("VISUALIZE", "1")],
    }
}

// Shader with visualization disabled
#[cfg(not(feature = "visualize"))]
mod pbr_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/pbr.frag",
        vulkan_version: "1.2",
        linalg_type: "nalgebra",
        dump: true,
    }
}
*/

/* Runtime compilation example
   This requires `shaderc` and the Vulkan SDK as runtime dependencies so
   is not included here
#[allow(dead_code)]
fn compile_glsl(device: Arc<Device>, glsl: &str) -> Arc<ShaderModule> {
    use shaderc;

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    let binary_result = compiler
        .compile_into_spirv(
            glsl,
            shaderc::ShaderKind::Vertex,
            "some_useful_label",
            "main",
            Some(&options),
        )
        .unwrap();
    let words = binary_result.as_binary();
    unsafe {
        ShaderModule::new(
            device, //
            ShaderModuleCreateInfo::new(words),
        )
    }
    .unwrap()
}
*/
