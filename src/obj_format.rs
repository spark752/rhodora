use crate::rh_error::*;
use crate::types::{vertex::*, *};
use log::info;
use serde::{Deserialize, Serialize};

pub struct ObjMaterial {
    pub colour_filename: String, // Colour texture for diffuse
    pub diffuse: [f32; 3],       // Multiplier for diffuse
    pub roughness: f32,
    pub metalness: f32,
}

#[derive(Serialize, Deserialize, PartialEq, Debug)]
pub struct ObjToLoad {
    pub filename: String, // Used for load_obj but not process_obj
    pub scale: f32,
    pub swizzle: bool,
    pub order_option: Option<Vec<usize>>,
}

impl Default for ObjToLoad {
    fn default() -> Self {
        Self {
            filename: "".to_string(),
            scale: 1.0f32,
            swizzle: true,
            order_option: None,
        }
    }
}

#[derive(Default)]
pub struct ObjLoaded {
    pub submeshes: Vec<Submesh>,
    pub materials: Vec<ObjMaterial>,
    pub order_option: Option<Vec<usize>>,
}

/// Load a Wavefront OBJ file
pub fn load_obj(
    obj_to_load: &ObjToLoad,
    vb: &mut VertexBuffers,
) -> Result<ObjLoaded, RhError> {
    let load_result =
        tobj::load_obj(&obj_to_load.filename, &tobj::GPU_LOAD_OPTIONS);
    process_obj(obj_to_load, load_result, vb)
}

/// Process a loaded Wavefront OBJ file. Called by load_obj or can be used
/// with files loaded some other way.
pub fn process_obj(
    obj_to_load: &ObjToLoad,
    load_result: tobj::LoadResult,
    vb: &mut VertexBuffers,
) -> Result<ObjLoaded, RhError> {
    let (tobj_models, tobj_materials) = load_result?;
    info!("Found {} Models", tobj_models.len());

    let scale = obj_to_load.scale;
    let swizzle = obj_to_load.swizzle;

    let mut submeshes = Vec::new();
    let mut first_index = 0u32;
    let mut vertex_offset = 0i32;

    // Models aka submeshes
    // Load each mesh sequentially into the vertex buffer
    for m in tobj_models.iter() {
        let mesh = &m.mesh;
        let pos_count = mesh.positions.len() / 3;
        let idx_count = mesh.indices.len();
        let has_uv = !mesh.texcoords.is_empty();

        // Convert positions to Z axis up if needed
        for v in 0..pos_count {
            vb.positions.push(Position {
                position: if swizzle {
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
                },
            });
        }

        // Convert normals to Z axis up
        // FIXME return error for no normal case
        for v in 0..pos_count {
            vb.interleaved.push(Interleaved {
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
            });
        }

        // Convert to 16 bit indices
        for i in 0..idx_count {
            vb.indices.push(
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
    for m in tobj_materials.unwrap_or_default().iter() {
        info!("Processing {:?}", m.name);
        let material = ObjMaterial {
            colour_filename: m.diffuse_texture.clone(),
            diffuse: m.diffuse,
            roughness: match m.unknown_param.get("Pr") {
                Some(x) => x.parse::<f32>().unwrap_or(0.5),
                None => 0.5,
            },
            metalness: match m.unknown_param.get("Pm") {
                Some(x) => x.parse::<f32>().unwrap_or(0.0),
                None => 0.0,
            },
        };
        materials.push(material);
    }

    Ok(ObjLoaded {
        submeshes,
        materials,
        order_option: obj_to_load.order_option.clone(),
    })
}

pub struct ObjBatch {
    pub vb: VertexBuffers,
    pub obj: Vec<ObjLoaded>,
}

impl Default for ObjBatch {
    fn default() -> Self {
        Self {
            vb: VertexBuffers::new(),
            obj: Vec::new(),
        }
    }
}

impl ObjBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load(&mut self, obj_to_load: &ObjToLoad) -> Result<(), RhError> {
        let obj_loaded = load_obj(obj_to_load, &mut self.vb)?;
        self.obj.push(obj_loaded);
        Ok(())
    }

    pub fn process(
        &mut self,
        obj_to_load: &ObjToLoad,
        load_result: tobj::LoadResult,
    ) -> Result<(), RhError> {
        let obj_loaded = process_obj(obj_to_load, load_result, &mut self.vb)?;
        self.obj.push(obj_loaded);
        Ok(())
    }
}
