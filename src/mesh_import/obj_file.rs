use nalgebra_glm as glm;
use std::path::Path;

use super::types::{
    ImportMaterial, ImportOptions, ImportVertex, MeshLoaded, Submesh,
};
use crate::mesh_import::ImportError;
use crate::rh_error::RhError;
use crate::vertex::{IndexBuffer, InterBuffer};

#[allow(unused_imports)]
use log::{debug, info, warn};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Load a Wavefront OBJ format object from an .obj file. Loads the file into
/// memory and calls `process_obj`. You may call that directly if you've loaded
/// or generated OBJ data some other way.
///
/// # Errors
/// May return `RhError`
pub fn load(
    path: &Path,
    import_options: &ImportOptions,
    vb_index: &mut IndexBuffer,
    vb_inter: &mut InterBuffer,
) -> Result<MeshLoaded, RhError> {
    let load_result = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    let base_path = path.parent().unwrap_or_else(|| Path::new("."));
    process_obj(base_path, import_options, load_result, vb_index, vb_inter)
}

/// Process loaded Wavefront OBJ format data. Called by `load_obj` or can be
/// used with OBJ data loaded or generated some other way.
///
/// # Errors
/// May return `RhError`
pub fn process_obj(
    base_path: &Path,
    import_options: &ImportOptions,
    load_result: tobj::LoadResult,
    vb_index: &mut IndexBuffer,
    vb_inter: &mut InterBuffer,
) -> Result<MeshLoaded, RhError> {
    let (tobj_models, tobj_materials) = load_result?;
    info!("Found {} submeshes", tobj_models.len());

    let scale = import_options.scale;
    let swizzle = import_options.swizzle;

    let mut submeshes = Vec::new();
    let mut first_index = 0_u32;
    let mut vertex_offset = 0_i32;

    // Models aka submeshes
    // Load each mesh sequentially into the vertex buffer
    for m in &tobj_models {
        let benchmark = std::time::Instant::now();
        let mesh = &m.mesh;
        let has_normals = !mesh.normals.is_empty();
        if has_normals && (mesh.positions.len() != mesh.normals.len()) {
            Err(ImportError::CountMismatch)?;
        }
        let pos_count = mesh.positions.len() / 3;
        let idx_count = mesh.indices.len();
        let has_uv = !mesh.texcoords.is_empty();
        info!(
            "Submesh vertices={}, triangles={}, has_normals={}, has_uv={}",
            pos_count,
            idx_count / 3,
            has_normals,
            has_uv,
        );
        let vertex_count = i32::try_from(pos_count)
            .map_err(|_| RhError::VertexCountTooLarge)?;
        let index_count = u32::try_from(mesh.indices.len())
            .map_err(|_| RhError::IndexCountTooLarge)?;

        // Convert to 16 bit indices from the obj file's u32.
        // The possible error & question mark operator makes using a closure
        // messy, so just use a ranged loop.
        let mut import_indices: Vec<u16> = Vec::new();
        for i in 0..idx_count {
            import_indices.push(
                u16::try_from(mesh.indices[i])
                    .map_err(|_| RhError::IndexTooLarge)?,
            );
        }

        // Collect data into the intermediate vertex format.
        // Initial testing of the rayon enabled version shows that it is
        // significantly SLOWER than the serial version.
        #[cfg(feature = "rayon")]
        let it = (0..pos_count).into_par_iter();
        #[cfg(not(feature = "rayon"))]
        let it = 0..pos_count;
        let mut import_vertices: Vec<ImportVertex> = it
            .map(|v| ImportVertex {
                position: if swizzle {
                    glm::Vec3::new(
                        mesh.positions[v * 3] * scale,
                        -mesh.positions[v * 3 + 2] * scale,
                        mesh.positions[v * 3 + 1] * scale,
                    )
                } else {
                    glm::Vec3::new(
                        mesh.positions[v * 3] * scale,
                        mesh.positions[v * 3 + 1] * scale,
                        mesh.positions[v * 3 + 2] * scale,
                    )
                },
                normal: if has_normals {
                    if swizzle {
                        glm::Vec3::new(
                            mesh.normals[v * 3],
                            -mesh.normals[v * 3 + 2],
                            mesh.normals[v * 3 + 1],
                        )
                    } else {
                        glm::Vec3::new(
                            mesh.normals[v * 3],
                            mesh.normals[v * 3 + 1],
                            mesh.normals[v * 3 + 2],
                        )
                    }
                } else {
                    glm::Vec3::new(0.0_f32, 0.0_f32, 0.0_f32)
                },
                tex_coord: if has_uv {
                    [mesh.texcoords[v * 2], 1.0 - mesh.texcoords[v * 2 + 1]]
                } else {
                    [0.0_f32, 0.0_f32]
                },
                ..Default::default()
            })
            .collect();

        // Do possible additional processing of the intermediate data
        if !has_normals {
            warn!("Missing normals are being calculated and might be wrong");
            super::util::calculate_normals(
                &import_indices,
                &mut import_vertices,
            );
        }

        // Collect into the output struct
        vb_inter.append(&import_vertices);
        vb_index.indices.append(&mut import_indices); // Consumes input
        debug!("Processing took {:?}", benchmark.elapsed());

        // Collect information
        submeshes.push(Submesh {
            index_count,
            first_index,
            vertex_offset,
            vertex_count,
            material_id: mesh.material_id.unwrap_or(0),
        });

        // Prepare for next mesh
        vertex_offset += vertex_count;
        first_index += index_count;
    }

    // Materials
    // MTL predates PBR but a proposed extension uses "Pr" for roughness and
    // "Pm" for metalness so that is implemented here.
    let mut materials = Vec::new();
    for m in &tobj_materials.unwrap_or_default() {
        let colour_filename = {
            if m.diffuse_texture.is_empty() {
                String::new()
            } else {
                base_path.join(&m.diffuse_texture).display().to_string()
            }
        };
        info!(
            "Processing material {:?} with texture \"{}\"",
            m.name, colour_filename
        );
        let material = ImportMaterial {
            colour_filename,
            diffuse: m.diffuse,
            roughness: m
                .unknown_param
                .get("Pr")
                .map_or(0.5, |x| x.parse::<f32>().unwrap_or(0.5)),
            metalness: m
                .unknown_param
                .get("Pm")
                .map_or(0.0, |x| x.parse::<f32>().unwrap_or(0.0)),
        };
        materials.push(material);
    }

    Ok(MeshLoaded {
        submeshes,
        materials,
        order_option: import_options.order_option.clone(),
        joint_count: 0, // Not supported by this format
    })
}
