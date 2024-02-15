use super::{
    gltf_file, obj_file,
    types::{FileToLoad, MeshLoaded},
};
use crate::rh_error::RhError;
use crate::vertex::{IndexBuffer, InterBuffer};
use std::path::Path;

/// Style of batch? mesh? pipeline? something FIXME
#[allow(dead_code)] // Perhaps not all variants are constructed
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Style {
    Rigid,
    Skinned,
}

/// A batch is used to load multiple mesh files into the same buffers. There
/// is no `Default` implementation since it is not clear what the defaults
/// should be. There is a `new` implementation where a `Style` can be provided.
pub struct Batch {
    pub style: Style,
    pub vb_index: IndexBuffer,
    pub vb_inter: InterBuffer,
    pub meshes: Vec<MeshLoaded>,
}

impl Batch {
    #[must_use]
    pub fn new(style: Style) -> Self {
        // FIXME style and vb_inter must be compatible some how
        Self {
            style,
            vb_index: IndexBuffer::new(),
            vb_inter: InterBuffer::new(style),
            meshes: Vec::new(),
        }
    }

    /// Loads a mesh from a file into this batch. Filenames with an ".obj"
    /// extension will be loaded as Wavefront OBJ files. Other files will be
    /// attempted to be loaded as glTF. Note that only a small subset of glTF
    /// files are supported.
    ///
    /// Mesh vertex data is appended to the `vb_index` and `vb_inter` members
    /// of this `Batch` struct. Submesh and material data is appended to
    /// the `meshes` member. This includes the names of texture files but the
    /// textures themselves are not loaded here. Optional data on draw order
    /// for the submeshes can be provided and is added to `meshes` unchanged.
    ///
    /// # Errors
    /// May return `RhError`
    pub fn load(&mut self, file: &FileToLoad) -> Result<(), RhError> {
        // The format specific loader writes directly to `vb` and returns data
        // to put in `meshes`.
        // Assume that files ending with ".obj" are OBJ format and anything
        // else is glTF.
        let mesh_loaded = {
            if let Some(ext) = Path::new(&file.filename).extension() {
                if ext.to_ascii_lowercase() == "obj" {
                    obj_file::load(file, &mut self.vb_index, &mut self.vb_inter)
                } else {
                    gltf_file::load(
                        file,
                        &mut self.vb_index,
                        &mut self.vb_inter,
                    )
                }
            } else {
                gltf_file::load(file, &mut self.vb_index, &mut self.vb_inter)
            }
        }?;
        self.meshes.push(mesh_loaded);
        Ok(())
    }
}
