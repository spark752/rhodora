use itertools::Itertools;
use nalgebra_glm as glm;

use super::types::ImportVertex;

#[allow(unused_imports)]
use log::debug;

/// Calculates the face normals from vertex data. These are not very useful
/// by themselves but are needed to calculate the vertex normals.
///
/// The face normals are not normalized so there can be some area weighting
/// when averaged into vertex normals. This is not necessarily correct but
/// maybe it is good enough.
#[must_use]
fn calculate_face_normals(
    indices: &[u16],
    vertices: &[ImportVertex],
) -> Vec<glm::Vec3> {
    let mut face_normals = Vec::with_capacity(indices.len() / 3);
    for (i0, i1, i2) in indices.iter().tuples() {
        let v0 = vertices[*i0 as usize].position;
        let v1 = vertices[*i1 as usize].position;
        let v2 = vertices[*i2 as usize].position;
        let va = v0 - v1;
        let vb = v1 - v2;
        face_normals.push(glm::cross(&va, &vb));
    }
    face_normals
}

/// Calculates normals. This is intended for importing meshes that do not
/// contain normals, but it is inefficent and may not be accurate. It is highly
/// recommended that meshes containing normals be used instead.
///
/// # Panics
/// Will panic if a vertex index does not fit in a `u16` however this should
/// have already been checked when the `ImportVertex` was created.
pub fn calculate_normals(indices: &[u16], vertices: &mut [ImportVertex]) {
    let face_normals = calculate_face_normals(indices, vertices);
    for (vertex_index, vertex) in vertices.iter_mut().enumerate() {
        let faces = connected_faces(
            indices,
            // Indices have already been checked to fit in 16 bits so this
            // `try_from` should not fail
            u16::try_from(vertex_index).unwrap(),
        );
        if !faces.is_empty() {
            // Sum the normals from the connected faces and normalize to create
            // sort of an area weighted average.
            let mut vert_norm = glm::Vec3::new(0.0_f32, 0.0_f32, 0.0_f32);
            for index in faces {
                vert_norm += face_normals[index];
            }
            vertex.normal = glm::normalize(&vert_norm);
        }
    }
}

/// Returns the a list of faces that contain a given vertex. Checking every
/// vertex this way is inefficient and slow but it should work.
#[must_use]
fn connected_faces(indices: &[u16], vertex_index: u16) -> Vec<usize> {
    let mut faces = Vec::new();
    for (face_index, (i0, i1, i2)) in indices.iter().tuples().enumerate() {
        if *i0 == vertex_index || *i1 == vertex_index || *i2 == vertex_index {
            faces.push(face_index);
        }
    }
    faces
}

/// Calculates the masks for faces (triangles) from the vertex masks.
#[must_use]
pub fn calculate_masks(indices: &[u16], vertices: &[ImportVertex]) -> Vec<u32> {
    let mut face_masks = Vec::with_capacity(indices.len() / 3);
    for (i0, i1, i2) in indices.iter().tuples() {
        let m0 = vertices[*i0 as usize].mask;
        let m1 = vertices[*i1 as usize].mask;
        let m2 = vertices[*i2 as usize].mask;
        face_masks.push(m0 | m1 | m2);
    }
    face_masks
}

/// Calculates list of indices from an original with zone masking
#[must_use]
#[allow(dead_code)]
pub fn mask_indices(
    indices: &[u16],
    face_masks: &[u32],
    zone_mask: u32,
) -> Vec<u16> {
    // Maxiumum length of output list is the length of the input list.
    // Reserving space for that probably won't hurt.
    let mut indices_out = Vec::with_capacity(indices.len());
    for ((i0, i1, i2), face_mask) in indices.iter().tuples().zip(face_masks) {
        if (face_mask & zone_mask) == 0 {
            // Not masked out so put it in the output list
            indices_out.push(*i0);
            indices_out.push(*i1);
            indices_out.push(*i2);
        }
    }
    indices_out
}
