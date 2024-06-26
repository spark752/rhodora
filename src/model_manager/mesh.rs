use crate::{
    mesh_import::{MeshLoaded, Submesh},
    rh_error::RhError,
};

pub struct Mesh {
    pub submeshes: Vec<Submesh>,
    pub joint_count: u32,
}

/// Convert a batch of meshes into the format needed by the Model Manager
///
/// # Errors
/// May return `RhError` (but currently doesn't)
#[allow(clippy::unnecessary_wraps)]
pub fn convert_batch_meshes(
    meshes_loaded: &[MeshLoaded],
) -> Result<Vec<Mesh>, RhError> {
    let mut meshes = Vec::new();

    // Adjust submesh structures. The material_id from an .obj file is
    // relative to that .mtl, so is changed to be relative to the whole
    // batch of materials. It is NOT adjusted here for an overall material
    // library to avoid dependency.
    // The first_index and vertex_offset are changed to reflect multiple
    // files being loaded into one DVB. The Mesh does NOT contain an reference
    // to the particular DVB to avoid dependency.
    let mut next_index = 0;
    let mut next_vertex = 0;
    let mut next_material_id = 0;
    for mesh_loaded in meshes_loaded {
        // Collect submeshes for this mesh
        let mut submeshes = Vec::new();
        for sub in &mesh_loaded.submeshes {
            submeshes.push(Submesh {
                index_count: sub.index_count,
                first_index: sub.first_index + next_index,
                vertex_offset: sub.vertex_offset + next_vertex,
                vertex_count: sub.vertex_count,
                material_id: sub.material_id + next_material_id,
            });
        }

        // Update offsets so if there is another mesh in the buffer
        if !submeshes.is_empty() {
            let last_sub = &submeshes[submeshes.len() - 1];
            next_index = last_sub.first_index + last_sub.index_count;
            next_vertex = last_sub.vertex_offset + last_sub.vertex_count;
        }

        // Re-arrange the submeshes if an order was provided
        let ordered_submeshes = {
            if let Some(order) = &mesh_loaded.order_option {
                let mut reorder = Vec::new();
                for i in order {
                    if *i < submeshes.len() {
                        reorder.push(submeshes[*i]); // Copyable type
                    } else {
                        Err(RhError::IndexTooLarge)?;
                    }
                }
                reorder
            } else {
                submeshes
            }
        };
        meshes.push(Mesh {
            submeshes: ordered_submeshes,
            joint_count: mesh_loaded.joint_count,
        });

        // Update material id offset. Normally there will be one per
        // submesh, but use the count from the material file in case it
        // is different.
        next_material_id += mesh_loaded.materials.len();
    }
    Ok(meshes)
}
