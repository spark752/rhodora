use super::types::{
    FileToLoad, ImportMaterial, ImportVertex, MeshLoaded, Submesh,
};
use crate::{
    rh_error::RhError,
    vertex::{IndexBuffer, InterBuffer},
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
    Document, Gltf, Primitive, Semantic,
};
use log::{info, trace, warn};
use std::{fs, io, path::Path};

// Validate a glTF for compatibility. Returns index and vertex count.
fn validate(p: &Primitive) -> Result<(usize, usize), RhError> {
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
    let vert_count = positions.count();

    // Normals are required. There must be the same number of normals as
    // there are positions.
    let normals =
        (p.get(&Semantic::Normals).ok_or(RhError::UnsupportedFormat))?;
    if normals.count() != vert_count {
        return Err(RhError::UnsupportedFormat);
    }

    // Texture coordinate (UVs) are optional, but if they are provided they
    // must be in Vec2 and the same number as there are positions
    let uv_option = p.get(&Semantic::TexCoords(0));
    {
        if let Some(ref uv) = uv_option {
            if uv.count() != vert_count || uv.dimensions() != Dimensions::Vec2 {
                return Err(RhError::UnsupportedFormat);
            }
        }
    }

    // Joint data is optional, but if it is provided there must be both indices
    // and weights and the same number as there are positions
    let joint_option = p.get(&Semantic::Joints(0));
    {
        if let Some(ref joints) = joint_option {
            if joints.count() != vert_count {
                return Err(RhError::UnsupportedFormat);
            }
        }
        let weight_option = p.get(&Semantic::Weights(0));
        if let Some(weights) = weight_option {
            if weights.count() != vert_count {
                return Err(RhError::UnsupportedFormat);
            }
        } else {
            return Err(RhError::UnsupportedFormat);
        }
    }

    // A little info
    info!(
        "Submesh={}, Index count={}, Vertex count={}, Has UV={}, Has joints={}",
        p.index(),
        idx_count,
        vert_count,
        uv_option.is_some(),
        joint_option.is_some(),
    );

    Ok((idx_count, vert_count))
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
    let buffers = gltf::import_buffers(&gltf.document, Some(base), gltf.blob)
        .map_err(|e| RhError::GltfError(Box::new(e)))?;

    // Some info
    let buffer_count = buffers.len();
    info!(
        "{:?}, base path={:?}, buffer count={}, first buffer length={} ",
        path,
        base,
        buffer_count,
        buffers[0].len(),
    );
    if buffer_count != 1 {
        warn!(
            "buffer count={} is not 1. This probably won't work",
            buffer_count
        );
    }

    Ok((gltf.document, buffers))
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
//#[allow(clippy::cognitive_complexity)]
#[allow(clippy::too_many_lines)]
pub fn load(
    file: &FileToLoad,
    vb_index: &mut IndexBuffer,
    vb_inter: &mut InterBuffer,
) -> Result<MeshLoaded, RhError> {
    let scale = file.scale;
    let swizzle = file.swizzle;
    let mut submeshes = Vec::new();
    let mut first_index = 0u32;
    let mut vertex_offset = 0i32;

    // Load the gltf file and mesh data but not the textures
    let (document, buffers) = load_impl(&file.filename)?;

    // Can contain multiple meshes which can contain multiple primitives. Each
    // primitive is treated as a submesh to match the .obj format support.
    for m in document.meshes() {
        info!("Mesh={}, Name={:?}", m.index(), m.name());
        for p in m.primitives() {
            // Validate certain aspects of the glTF. These should include that
            // the number of positions is equal to the number of normals etc.
            // since we don't know how to interpret the data if they aren't
            // equal.
            let (idx_count, vert_count) = validate(&p)?;

            // Create a reader for the data buffer
            let reader = p.reader(|_| Some(&buffers[0]));

            // Read the indices and store them as 16 bits directly to the
            // output buffer
            let idx_data =
                reader.read_indices().ok_or(RhError::UnsupportedFormat)?;
            match idx_data {
                ReadIndices::U8(it) => {
                    for i in it {
                        vb_index.push_index(u16::from(i));
                    }
                }
                ReadIndices::U16(it) => {
                    for i in it {
                        vb_index.push_index(i);
                    }
                }
                ReadIndices::U32(it) => {
                    info!("Trying to convert 32 bit indices to 16 bit");
                    for i in it {
                        vb_index.push_index(
                            u16::try_from(i)
                                .map_err(|_| RhError::IndexTooLarge)?,
                        );
                    }
                }
            }

            // Create an import vertex buffer to build up the other data
            let mut verts = Vec::new();

            // Read and store positions into the import vertex buffer, scaling
            // and swizzling if needed
            let pos_data =
                reader.read_positions().ok_or(RhError::UnsupportedFormat)?;
            let ReadPositions::Standard(it) = pos_data else {
                warn!("Unsupported sparse position format");
                return Err(RhError::UnsupportedFormat);
            };
            for i in it {
                let p = {
                    if swizzle {
                        [i[0] * scale, -i[2] * scale, i[1] * scale]
                    } else {
                        [i[0] * scale, i[1] * scale, i[2] * scale]
                    }
                };
                let v = ImportVertex {
                    position: p,
                    ..Default::default()
                };
                verts.push(v);
            }

            // Read and store normals into the import vertex buffer, swizzling
            // if needed
            let norm_data =
                reader.read_normals().ok_or(RhError::UnsupportedFormat)?;
            let ReadNormals::Standard(it) = norm_data else {
                warn!("Unsupported sparse normal format");
                return Err(RhError::UnsupportedFormat);
            };
            for (i, norm) in it.enumerate() {
                if i < verts.len() {
                    verts[i].normal = {
                        if swizzle {
                            [norm[0], -norm[2], norm[1]]
                        } else {
                            [norm[0], norm[1], norm[2]]
                        }
                    };
                }
            }

            // Read and store the texture coordinates if they exist
            if let Some(uv_data) = reader.read_tex_coords(0) {
                let ReadTexCoords::F32(it) = uv_data else {
                    // Could support these if we come up with a test case
                    warn!("Unsupported UV format");
                    return Err(RhError::UnsupportedFormat);
                };
                for (i, uv) in it.enumerate() {
                    if i < verts.len() {
                        verts[i].tex_coord = uv;
                    }
                }
            }

            // Read the store the joints if they exist
            if let Some(joint_data) = reader.read_joints(0) {
                let ReadJoints::U8(joint_it) = joint_data else {
                    warn!("Unsupported joint format");
                    return Err(RhError::UnsupportedFormat);
                };
                let weight_data =
                    reader.read_weights(0).ok_or(RhError::UnsupportedFormat)?;
                let ReadWeights::F32(weight_it) = weight_data else {
                    warn!("Unsupported weight format");
                    return Err(RhError::UnsupportedFormat);
                };
                for (i, (id_array, weights)) in
                    joint_it.zip(weight_it).enumerate()
                {
                    trace!("Joint ids={:?} weights={:?}", id_array, weights);
                    if i < verts.len() {
                        verts[i].joint_ids = id_array;
                        verts[i].weights = weights;
                    }
                }
            }

            // Validate that we have the expected amount of information
            if vert_count != verts.len() {
                warn!(
                    "Vertex count mismatch {} != {}",
                    vert_count,
                    verts.len()
                );
                return Err(RhError::UnsupportedFormat);
            }

            // Push the import vertex data into the output buffer
            for v in verts {
                vb_inter.push(v);
            }

            // Collect information
            let vertex_count = i32::try_from(vert_count)
                .map_err(|_| RhError::VertexCountTooLarge)?;
            let index_count = u32::try_from(idx_count)
                .map_err(|_| RhError::IndexCountTooLarge)?;
            submeshes.push(Submesh {
                index_count,
                first_index,
                vertex_offset,
                vertex_count,
                material_id: p.material().index().unwrap_or(0),
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

fn load_materials(
    base_path: &Path,
    document: &Document,
) -> Vec<ImportMaterial> {
    // Materials are currently handled separately because that's how the .obj
    // library works. It could be improved if we focus on glTF.
    info!("Materials={}", document.materials().count());
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

        let material = ImportMaterial {
            colour_filename,
            diffuse,
            roughness,
            metalness,
        };
        materials.push(material);
    }
    materials
}
