use super::types::{FileToLoad, ImportVertex, Material, MeshLoaded, Submesh};
use crate::rh_error::RhError;
use crate::vertex::{BaseBuffers, InterBuffers, InterVertexTrait};
use log::{info, warn};

/// Load a Wavefront OBJ format object from an .obj file. Loads the file into
/// memory and calls `process_obj`. You may call that directly if you've loaded
/// or generated OBJ data some other way.
///
/// # Errors
/// May return `RhError`
pub fn load<T: InterVertexTrait>(
    file: &FileToLoad,
    vb_base: &mut BaseBuffers,
    vb_inter: &mut InterBuffers<T>,
) -> Result<MeshLoaded, RhError> {
    let load_result = tobj::load_obj(&file.filename, &tobj::GPU_LOAD_OPTIONS);
    process_obj(file, load_result, vb_base, vb_inter)
}

/// Process loaded Wavefront OBJ format data. Called by `load_obj` or can be
/// used with OBJ data loaded or generated some other way.
///
/// # Errors
/// May return `RhError`
pub fn process_obj<T: InterVertexTrait>(
    file: &FileToLoad,
    load_result: tobj::LoadResult,
    vb_base: &mut BaseBuffers,
    vb_inter: &mut InterBuffers<T>,
) -> Result<MeshLoaded, RhError> {
    let (tobj_models, tobj_materials) = load_result?;
    info!("Found {} Models", tobj_models.len());

    let scale = file.scale;
    let swizzle = file.swizzle;

    let mut submeshes = Vec::new();
    let mut first_index = 0u32;
    let mut vertex_offset = 0i32;

    // Models aka submeshes
    // Load each mesh sequentially into the vertex buffer
    for m in &tobj_models {
        let mesh = &m.mesh;
        if mesh.positions.len() != mesh.normals.len() {
            warn!("Mesh has no normals");
            return Err(RhError::UnsupportedFormat);
        }
        let pos_count = mesh.positions.len() / 3;
        let idx_count = mesh.indices.len();
        let has_uv = !mesh.texcoords.is_empty();

        // Convert positions to Z axis up if needed
        for v in 0..pos_count {
            let p = if swizzle {
                [
                    mesh.positions[v * 3] * scale,
                    -mesh.positions[v * 3 + 2] * scale,
                    mesh.positions[v * 3 + 1] * scale,
                ]
            } else {
                [
                    mesh.positions[v * 3] * scale,
                    mesh.positions[v * 3 + 1] * scale,
                    mesh.positions[v * 3 + 2] * scale,
                ]
            };
            vb_base.push_position(&p);
        }

        // Convert normals to Z axis up if needed and interleave. This file
        // format does not support skinning.
        for v in 0..pos_count {
            let bf = ImportVertex {
                normal: if swizzle {
                    [
                        mesh.normals[v * 3],
                        -mesh.normals[v * 3 + 2],
                        mesh.normals[v * 3 + 1],
                    ]
                } else {
                    [
                        mesh.normals[v * 3],
                        mesh.normals[v * 3 + 1],
                        mesh.normals[v * 3 + 2],
                    ]
                },
                tex_coord: if has_uv {
                    [mesh.texcoords[v * 2], 1.0 - mesh.texcoords[v * 2 + 1]]
                } else {
                    [0.0, 0.0]
                },
                ..Default::default()
            };
            vb_inter.push(&bf.into());
        }

        // Convert to 16 bit indices
        for i in 0..idx_count {
            vb_base.push_index(
                u16::try_from(mesh.indices[i])
                    .map_err(|_| RhError::IndexTooLarge)?,
            );
        }

        // Collect information
        let vertex_count = i32::try_from(pos_count)
            .map_err(|_| RhError::VertexCountTooLarge)?;
        let index_count = u32::try_from(mesh.indices.len())
            .map_err(|_| RhError::IndexCountTooLarge)?;
        submeshes.push(Submesh {
            index_count,
            first_index,
            vertex_offset,
            vertex_count,
            material_id: mesh.material_id,
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
        info!("Processing {:?}", m.name);
        let material = Material {
            colour_filename: m.diffuse_texture.clone(),
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
        order_option: file.order_option.clone(),
    })
}
