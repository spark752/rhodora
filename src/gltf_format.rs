use crate::obj_format::{ObjLoaded, ObjMaterial, ObjToLoad};
use crate::rh_error::RhError;
use crate::types::{
    vertex::{Interleaved, Position, VertexBuffers},
    Submesh,
};
use gltf::mesh::util::{
    ReadIndices, ReadNormals, ReadPositions, ReadTexCoords,
};
use gltf::{
    accessor::Dimensions, buffer, image::Source, mesh::Mode, Document, Gltf,
    Semantic,
};
use log::info;
use std::{fs, io, path::Path};

fn load_impl<P>(path: P) -> Result<(Document, Vec<buffer::Data>), RhError>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let file = fs::File::open(path).map_err(RhError::StdIoError)?;
    let reader = io::BufReader::new(file);
    let gltf = Gltf::from_reader(reader)
        .map_err(|e| RhError::GltfError(Box::new(e)))?;
    let buffer_data =
        gltf::import_buffers(&gltf.document, Some(base), gltf.blob)
            .map_err(|e| RhError::GltfError(Box::new(e)))?;
    Ok((gltf.document, buffer_data))
}

/// Load a glTF file. Only a very limited subset of glTF functionality is
/// supported. The current focus of the project is on models which share
/// textures, therefore glTF files that embed images are not supported.
/// Tested with files exported from Blender 3.6.8 using the "glTF Separate"
/// option.
///
/// # Errors
/// May return `RhError`
#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
pub fn load_gltf(
    obj_to_load: &ObjToLoad,
    vb: &mut VertexBuffers,
) -> Result<ObjLoaded, RhError> {
    let scale = obj_to_load.scale;
    let swizzle = obj_to_load.swizzle;
    let mut submeshes = Vec::new();
    let mut first_index = 0u32;
    let mut vertex_offset = 0i32;

    // Load the gltf file and mesh data but not the textures
    let (document, buffers) = load_impl(&obj_to_load.filename)?;
    info!(
        "{}, buffer count={}, buffer length={}, scale={}, swizzle={} ",
        obj_to_load.filename,
        buffers.len(),
        buffers[0].len(),
        scale,
        swizzle
    );

    // Currently only the first mesh is loaded but it may contain primitives
    // aka submeshes similar to the way .obj format works.
    // Load each submesh sequentially into the vertex buffer
    for m in document
        .meshes()
        .next()
        .ok_or(RhError::UnsupportedFormat)?
        .primitives()
    {
        // Mesh must be made of indexed triangles
        if m.mode() != Mode::Triangles {
            return Err(RhError::UnsupportedFormat);
        }
        let indices = (m.indices().ok_or(RhError::UnsupportedFormat))?;
        let idx_count = indices.count();

        // Positions are required
        let positions = (m
            .get(&Semantic::Positions)
            .ok_or(RhError::UnsupportedFormat))?;
        let pos_count = positions.count();

        // Normals are required. There must be the same number of normals as
        // there are positions.
        let normals =
            (m.get(&Semantic::Normals).ok_or(RhError::UnsupportedFormat))?;
        if normals.count() != pos_count {
            return Err(RhError::UnsupportedFormat);
        }

        // Texture coordinate (UVs) are optional, but if they are provided
        // there must be the same number as there are positions as Vec2
        let uv_option = m.get(&Semantic::TexCoords(0));
        {
            if let Some(ref uv) = uv_option {
                if uv.count() != pos_count
                    || uv.dimensions() != Dimensions::Vec2
                {
                    return Err(RhError::UnsupportedFormat);
                }
            }
        }

        // A little debugging info
        info!(
            "Submesh={}, Index count={}, Vertex count={}, Has UV={}",
            m.index(),
            idx_count,
            pos_count,
            uv_option.is_some()
        );

        // Create a reader for the data buffer
        let reader = m.reader(|_| Some(&buffers[0]));

        // Read the indices and store them as 16 bits
        if let Some(idx_data) = reader.read_indices() {
            match idx_data {
                ReadIndices::U8(it) => {
                    for i in it {
                        vb.indices.push(u16::from(i));
                    }
                }
                ReadIndices::U16(it) => {
                    for i in it {
                        vb.indices.push(i);
                    }
                }
                ReadIndices::U32(it) => {
                    for i in it {
                        vb.indices.push(
                            u16::try_from(i)
                                .map_err(|_| RhError::IndexTooLarge)?,
                        );
                    }
                }
            }
        } else {
            return Err(RhError::UnsupportedFormat);
        };
        info!("vb.indices.len() now = {}", vb.indices.len());

        // Read the positions, scale & convert to Z axis up if needed, and store
        if let Some(pos_data) = reader.read_positions() {
            match pos_data {
                ReadPositions::Standard(it) => {
                    for i in it {
                        vb.positions.push(Position {
                            position: if swizzle {
                                [i[0] * scale, i[1] * scale, -i[2] * scale]
                            } else {
                                [i[0] * scale, i[1] * scale, i[2] * scale]
                            },
                        });
                    }
                }
                ReadPositions::Sparse(_) => {
                    return Err(RhError::UnsupportedFormat);
                }
            }
        } else {
            return Err(RhError::UnsupportedFormat);
        }
        info!("vb.positions.len() now = {}", vb.positions.len());

        // Read the texture coordinates if they exist and store them in a
        // temporary buffer so that they can be interleaved with normals
        let mut uv_temp = Vec::new();
        if let Some(uv_data) = reader.read_tex_coords(0) {
            match uv_data {
                ReadTexCoords::F32(it) => {
                    for i in it {
                        uv_temp.push(i);
                    }
                }
                _ => {
                    return Err(RhError::UnsupportedFormat);
                } /* Could probably support these but may have to normalize
                  ReadTexCoords::U16(it) => {
                      for i in it {
                          uv_temp.push([i[0] as f32, i[1] as f32]);
                      }
                  }
                  ReadTexCoords::U8(it) => {
                      for i in it {
                          uv_temp.push([i[0] as f32, i[1] as f32]);
                      }
                  }
                  */
            }
        }

        // Read the normals, convert to Z axis up if needed, and store along
        // with texture coordinates in an interleaved buffer
        if let Some(norm_data) = reader.read_normals() {
            match norm_data {
                ReadNormals::Standard(it) => {
                    for (uv_idx, i) in it.enumerate() {
                        vb.interleaved.push(Interleaved {
                            normal: if swizzle {
                                [i[0], i[1], -i[2]]
                            } else {
                                [i[0], i[1], i[2]]
                            },
                            tex_coord: {
                                if uv_idx < uv_temp.len() {
                                    uv_temp[uv_idx]
                                } else {
                                    [0.0, 0.0]
                                }
                            },
                        });
                    }
                }
                ReadNormals::Sparse(_) => {
                    return Err(RhError::UnsupportedFormat);
                }
            }
        } else {
            return Err(RhError::UnsupportedFormat);
        }
        info!("vb.interleaved.len() now = {}", vb.interleaved.len());

        // Collect information
        let vertex_count = i32::try_from(pos_count)
            .map_err(|_| RhError::VertexCountTooLarge)?;
        let index_count = u32::try_from(idx_count)
            .map_err(|_| RhError::IndexCountTooLarge)?;
        submeshes.push(Submesh {
            index_count,
            first_index,
            vertex_offset,
            vertex_count,
            material_id: m.material().index(),
        });

        // Prepare for next submesh
        vertex_offset += vertex_count;
        first_index += index_count;

        // Some material stuff for debug
        let mat = m.material();
        if let Some(mat_index) = mat.index() {
            if let Some(tex) = mat.pbr_metallic_roughness().base_color_texture()
            {
                let source = tex.texture().source().source();
                if let Source::Uri { uri, mime_type } = source {
                    info!(
                        "    material index={}, name={}, texture index={}, source uri={}, mime type={}",
                        mat_index,
                        mat.name().unwrap_or("N/A"),
                        tex.texture().index(),
                        uri,
                        mime_type.unwrap_or("N/A")
                    );
                } else if let Source::View {
                    view: _view,
                    mime_type,
                } = source
                {
                    info!(
                        "    material index={}, name={}, texture index={}, inline buffer, mime type {}",
                        mat_index,
                        mat.name().unwrap_or("N/A"),
                        tex.texture().index(),
                        mime_type
                    );
                }
            }
        }
    }

    // Materials are currently handled separately because that's how the .obj
    // library works. It could be improved if we focus on glTF.
    let base_path = Path::new(&obj_to_load.filename)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let mut materials = Vec::new();
    for m in document.materials() {
        let pbr = m.pbr_metallic_roughness();
        let diffuse = {
            let base = pbr.base_color_factor();
            [base[0], base[1], base[2]]
        };
        let roughness = pbr.roughness_factor();
        let metalness = pbr.metallic_factor();
        let colour_filename = {
            pbr.base_color_texture().map_or_else(String::new, |tex| {
                let source = tex.texture().source().source();
                if let Source::Uri { uri, mime_type: _ } = source {
                    let ret = base_path.join(uri);
                    ret.display().to_string()
                } else {
                    String::new()
                }
            })
        };
        info!(
            "Material {} name={} texture={} diffuse={:?} roughness={} metalness={}",
            m.index().unwrap_or(0),
            m.name().unwrap_or("N/A"),
            colour_filename,
            diffuse,
            roughness,
            metalness,
        );

        let material = ObjMaterial {
            colour_filename,
            diffuse,
            roughness,
            metalness,
        };
        materials.push(material);
    }

    Ok(ObjLoaded {
        submeshes,
        materials,
        order_option: obj_to_load.order_option.clone(),
    })
}
