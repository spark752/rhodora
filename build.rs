//! Build script for compiling shaders used by the main renderpass.
//!
//! The `vulkano-shader` crate provides a `shader!` macro that is used some
//! places in the project, but it has some limitations. It attempts reflection
//! and validation that doesn't always work.
//!
//! For example as of vulkano 0.34, setting an offset on a push constant in
//! a shader seems to always cause a "expected decoration to be 0" build
//! error, even though the GLSL already compiled successfully at that point
//! and the rest of the project hasn't been compiled to detect any actual
//! mismatch.
//!
//! Trying to pass parameters between shader stages in a struct or block
//! causes the `shader!` macro to panic as of vulkano 0.33. (This may have been
//! fixed in 0.34 but hasn't been retested in this project.)
//!
//! The `shaderc` crate is used to do the compilation, but `vulkano-shader`
//! does not expose all of its options. As of vulkano 0.34, it is not possible
//! to target Vulkan 1.3 with the `shader!` macro even though `shaderc`
//! supports that.
//!
//! Creating Rust types from shader code is nice for simple cases but has
//! limitations as well. It tends to confuse Rust analyzer since the type
//! definitions don't appear in the source code. Shader compilation errors
//! lead to a cascade of Rust compilation errors as the types aren't generated
//! at all. Having a mismatch between a shader struct and a Rust struct
//! automatically break the build is nice. But if there are multiple shaders
//! creating multiple structs that are all *supposed to be* the same, and only
//! one is actually used in the Rust code, mismatches may not be detected
//! anyway.
//!
//! For these reasons, the main shaders avoid the attempted reflection and
//! validation of `shader!` by using `shaderc` directly. This build script
//! includes the shader source code files, passes it to the compiler,
//! and writes the output as SPIR-V files. The output directory is in the
//! `OUT_DIR` environment variable provided by `cargo`. The SPIR-V files are
//! then included by the project source code so nothing has to be compiled
//! at runtime.
use shaderc::{CompilationArtifact, Compiler, ShaderKind};
use std::io::Write;

fn main() {
    // Only directory that a build script should write to:
    let out_var = std::env::var("OUT_DIR").unwrap();
    let out_dir = std::path::Path::new(&out_var);

    // Reuse the compiler
    let compiler = Compiler::new().unwrap();

    let source = include_str!("shaders/rigid.vert");
    let artifact = compile_shader(
        &compiler,
        source,
        ShaderKind::Vertex, //
        "rigid.spv",
    );
    let bytes = artifact.as_binary_u8();
    let path = out_dir.join("rigid.spv");
    save_bytes(&path, bytes);

    let source = include_str!("shaders/skinned.vert");
    let artifact = compile_shader(
        &compiler,
        source,
        ShaderKind::Vertex, //
        "skinned.spv",
    );
    let bytes = artifact.as_binary_u8();
    let path = out_dir.join("skinned.spv");
    save_bytes(&path, bytes);

    let source = include_str!("shaders/pbr.frag");
    let artifact = compile_shader(
        &compiler,
        source,
        ShaderKind::Fragment, //
        "pbr.spv",
    );
    let bytes = artifact.as_binary_u8();
    let path = out_dir.join("pbr.spv");
    save_bytes(&path, bytes);
}

fn compile_shader(
    compiler: &Compiler,
    source: &str,
    kind: ShaderKind,
    name: &str,
) -> CompilationArtifact {
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );
    compiler
        .compile_into_spirv(
            source, // GLSL
            kind,
            name, // Used for labels in error messages etc.
            "main",
            Some(&options),
        )
        .unwrap()
}

fn save_bytes(path: &std::path::PathBuf, bytes: &[u8]) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(bytes).unwrap();
}
