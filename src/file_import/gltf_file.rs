use super::types::{FileToLoad, ImportVertex, Material, MeshLoaded, Submesh};
use crate::{
    rh_error::RhError,
    vertex::{BaseBuffers, InterBuffers, InterVertexTrait},
};
use gltf::{
    accessor::Dimensions,
    buffer,
    image::Source,
    mesh::util::{
        ReadIndices, ReadJoints, ReadNormals, ReadPositions, ReadTexCoords,
        ReadWeights,
    },
    mesh::Mode,
    Document, Gltf, Semantic,
};
use log::{info, trace, warn};
use std::{fs, io, path::Path};

struct TempJoints {
    id_array: [u8; 4],
    weights: [f32; 4],
}

impl TempJoints {
    /// Packs joint data for a vertex into the vertex format. Takes ownership
    /// of arguments.
    const fn new(id_array: [u8; 4], weights: [f32; 4]) -> Self {
        Self { id_array, weights }
    }
}

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
/// option. glTF defines +Y up, +Z forward so `swizzle` should usually be
/// set (and is the default.)
///
/// # Errors
/// May return `RhError`
#[allow(clippy::cognitive_complexity)]
#[allow(clippy::too_many_lines)]
pub fn load<T: InterVertexTrait>(
    file: &FileToLoad,
    vb_base: &mut BaseBuffers,
    vb_inter: &mut InterBuffers<T>,
) -> Result<MeshLoaded, RhError> {
    let scale = file.scale;
    let swizzle = file.swizzle;
    let mut submeshes = Vec::new();
    let mut first_index = 0u32;
    let mut vertex_offset = 0i32;

    // Load the gltf file and mesh data but not the textures
    let (document, buffers) = load_impl(&file.filename)?;
    info!(
        "{}, buffer count={}, buffer length={}, scale={}, swizzle={} ",
        file.filename,
        buffers.len(),
        buffers[0].len(),
        scale,
        swizzle
    );

    // Can contain multiple meshes which can contain multiple primitives. Each
    // primitive is treated as a submesh to match the .obj format support.
    for m in document.meshes() {
        info!("Mesh={}, Name={:?}", m.index(), m.name());
        for p in m.primitives() {
            // Mesh must be made of indexed triangles
            if p.mode() != Mode::Triangles {
                return Err(RhError::UnsupportedFormat);
            }
            let indices = (p.indices().ok_or(RhError::UnsupportedFormat))?;
            let idx_count = indices.count();

            // Positions are required
            let positions = (p
                .get(&Semantic::Positions)
                .ok_or(RhError::UnsupportedFormat))?;
            let pos_count = positions.count();

            // Normals are required. There must be the same number of normals as
            // there are positions.
            let normals =
                (p.get(&Semantic::Normals).ok_or(RhError::UnsupportedFormat))?;
            if normals.count() != pos_count {
                return Err(RhError::UnsupportedFormat);
            }

            // Texture coordinate (UVs) are optional, but if they are provided
            // there must be the same number as there are positions as Vec2
            let uv_option = p.get(&Semantic::TexCoords(0));
            {
                if let Some(ref uv) = uv_option {
                    if uv.count() != pos_count
                        || uv.dimensions() != Dimensions::Vec2
                    {
                        return Err(RhError::UnsupportedFormat);
                    }
                }
            }

            // A little info
            info!(
                "Submesh={}, Index count={}, Vertex count={}, Has UV={}",
                p.index(),
                idx_count,
                pos_count,
                uv_option.is_some()
            );

            // Create a reader for the data buffer
            let reader = p.reader(|_| Some(&buffers[0]));

            // Read the indices and store them as 16 bits
            if let Some(idx_data) = reader.read_indices() {
                match idx_data {
                    ReadIndices::U8(it) => {
                        for i in it {
                            vb_base.push_index(u16::from(i));
                        }
                    }
                    ReadIndices::U16(it) => {
                        for i in it {
                            vb_base.push_index(i);
                        }
                    }
                    ReadIndices::U32(it) => {
                        info!("Trying to convert 32 bit indices to 16 bit");
                        for i in it {
                            vb_base.push_index(
                                u16::try_from(i)
                                    .map_err(|_| RhError::IndexTooLarge)?,
                            );
                        }
                    }
                }
            } else {
                warn!("Missing mesh indices");
                return Err(RhError::UnsupportedFormat);
            };

            // Read the positions, scale & convert to Z axis up if needed, and store
            if let Some(pos_data) = reader.read_positions() {
                match pos_data {
                    ReadPositions::Standard(it) => {
                        for i in it {
                            let p = {
                                if swizzle {
                                    [i[0] * scale, -i[2] * scale, i[1] * scale]
                                } else {
                                    [i[0] * scale, i[1] * scale, i[2] * scale]
                                }
                            };
                            vb_base.push_position(&p);
                        }
                    }
                    ReadPositions::Sparse(_) => {
                        warn!("Unsupported sparse format");
                        return Err(RhError::UnsupportedFormat);
                    }
                }
            } else {
                warn!("Missing mesh positions");
                return Err(RhError::UnsupportedFormat);
            }

            // Read the texture coordinates if they exist and store them in a
            // temporary buffer so that they can be interleaved with normals
            let mut uv = Vec::new();
            if let Some(uv_data) = reader.read_tex_coords(0) {
                if let ReadTexCoords::F32(it) = uv_data {
                    for i in it {
                        uv.push(i);
                    }
                } else {
                    // Could support these if we come up with a test case
                    warn!("Unsupported UV format");
                    return Err(RhError::UnsupportedFormat);
                }
            }

            // Read the joints if they exist and store them in a temporary
            // buffer so they can be interleaved with normals
            let mut temp_joints = Vec::new();
            if let Some(d) = reader.read_joints(0) {
                match d {
                    ReadJoints::U8(jit) => {
                        if let Some(d) = reader.read_weights(0) {
                            if let ReadWeights::F32(wit) = d {
                                for (id_array, weights) in jit.zip(wit) {
                                    trace!(
                                        "Joint ids={:?} weights={:?}",
                                        id_array,
                                        weights
                                    );
                                    temp_joints.push(TempJoints::new(
                                        id_array, weights,
                                    ));
                                }
                            } else {
                                // We could try to support these in the
                                // future if we come up with a test case
                                warn!("Unsupported weight format");
                                return Err(RhError::UnsupportedFormat);
                            }
                        } else {
                            warn!("Missing joint weights");
                            return Err(RhError::UnsupportedFormat);
                        }
                    }
                    ReadJoints::U16(_) => {
                        // We could try to support these in the future
                        // if we come up with a test case
                        warn!("Unsupported joint format");
                        return Err(RhError::UnsupportedFormat);
                    }
                }
            }

            // Read the normals, convert to Z axis up if needed, and store along
            // with texture coordinates etc. in an interleaved buffer
            if let Some(norm_data) = reader.read_normals() {
                match norm_data {
                    ReadNormals::Standard(it) => {
                        for (i, norm) in it.enumerate() {
                            vb_inter.push(
                                &ImportVertex {
                                    normal: if swizzle {
                                        [norm[0], -norm[2], norm[1]]
                                    } else {
                                        [norm[0], norm[1], norm[2]]
                                    },
                                    tex_coord: {
                                        if i < uv.len() {
                                            uv[i]
                                        } else {
                                            [0.0, 0.0]
                                        }
                                    },
                                    joint_ids: {
                                        if i < temp_joints.len() {
                                            temp_joints[i].id_array
                                        } else {
                                            [0, 0, 0, 0]
                                        }
                                    },
                                    weights: {
                                        if i < temp_joints.len() {
                                            temp_joints[i].weights
                                        } else {
                                            [0.0, 0.0, 0.0, 0.0]
                                        }
                                    },
                                }
                                .into(),
                            );
                        }
                    }
                    ReadNormals::Sparse(_) => {
                        return Err(RhError::UnsupportedFormat);
                    }
                }
            } else {
                return Err(RhError::UnsupportedFormat);
            }

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
                material_id: p.material().index(),
            });

            // Prepare for next submesh
            vertex_offset += vertex_count;
            first_index += index_count;
        }
    }

    Ok(MeshLoaded {
        submeshes,
        materials: {
            let base_path = Path::new(&file.filename)
                .parent()
                .unwrap_or_else(|| Path::new("."));
            load_materials(base_path, &document)
        },
        order_option: file.order_option.clone(),
    })
}

fn load_materials(base_path: &Path, document: &Document) -> Vec<Material> {
    // Materials are currently handled separately because that's how the .obj
    // library works. It could be improved if we focus on glTF.
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

        let material = Material {
            colour_filename,
            diffuse,
            roughness,
            metalness,
        };
        materials.push(material);
    }
    materials
}
