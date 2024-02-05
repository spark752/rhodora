use super::{
    gltf_file, obj_file,
    types::{FileToLoad, MeshLoaded},
};
use crate::rh_error::RhError;
use crate::vertex::{BaseBuffers, InterBuffers, InterVertexTrait};
use std::path::Path;

/// A batch is used to load multiple files into the same buffers. This type
/// is generic for different vertex formats. They must implement `BuffersTrait`
/// so the format in the file can be converted to the requested format.
#[derive(Default)]
pub struct Batch<T: InterVertexTrait> {
    pub vb_base: BaseBuffers,
    pub vb_inter: InterBuffers<T>,
    pub meshes: Vec<MeshLoaded>,
}

impl<T: InterVertexTrait> Batch<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads a mesh from a file into this batch. Filenames with an ".obj"
    /// extension will be loaded as Wavefront OBJ files. Other files will be
    /// attempted to be loaded as glTF. Note that only a small subset of glTF
    /// files are supported.
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
                    obj_file::load(file, &mut self.vb_base, &mut self.vb_inter)
                } else {
                    gltf_file::load(file, &mut self.vb_base, &mut self.vb_inter)
                }
            } else {
                gltf_file::load(file, &mut self.vb_base, &mut self.vb_inter)
            }
        }?;
        self.meshes.push(mesh_loaded);
        Ok(())
    }
}
