use crate::{
    file_import::DeviceVertexBuffers,
    vertex::{RigidFormat, SkinnedFormat},
};

/// Enum of supported vertex formats. Perhaps not all are constructed.
#[allow(dead_code)]
pub enum DvbWrapper {
    Rigid(DeviceVertexBuffers<RigidFormat>),
    Skinned(DeviceVertexBuffers<SkinnedFormat>),
}

impl From<DeviceVertexBuffers<RigidFormat>> for DvbWrapper {
    fn from(f: DeviceVertexBuffers<RigidFormat>) -> Self {
        Self::Rigid(f)
    }
}

impl From<DeviceVertexBuffers<SkinnedFormat>> for DvbWrapper {
    fn from(f: DeviceVertexBuffers<SkinnedFormat>) -> Self {
        Self::Skinned(f)
    }
}
