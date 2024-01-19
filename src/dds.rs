use crate::rh_error::*;
use crate::types::*;
use log::debug;
use smallvec::SmallVec;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BufferImageCopy, CopyBufferToImageInfo,
    },
    device::DeviceOwned,
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions,
        ImageLayout, ImageSubresourceLayers, ImageUsage, ImmutableImage,
        MipmapsCount,
    },
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryUsage},
};

const MAGIC_HEADER: u32 = 0x20534444;
const FOURCC_DXT1: u32 = 0x31545844;
const FOURCC_DXT5: u32 = 0x35545844;
const FOURCC_DX10: u32 = 0x30315844;
const DXGI_BC7_UNORM: u32 = 98;
const DXGI_BC7_UNORM_SRGB: u32 = 99;

#[allow(dead_code)]
#[derive(Debug)]
struct DdsHeader {
    height: u32,
    width: u32,
    pitch_or_size: u32,
    mipmap_count: u32,
    pixel_flags: u32,
    four_cc: u32,
}

#[allow(dead_code)]
#[derive(Debug)]
struct DdsHeaderDxt10 {
    dxgi_format: u32,
    resource_dimension: u32,
    misc_flag: u32,
    array_size: u32,
    misc_flags2: u32,
}

fn dword(slice: &[u8]) -> Result<u32, RhError> {
    Ok(u32::from_le_bytes(
        slice.try_into().map_err(|_| RhError::DataNotConverted)?,
    ))
}

pub fn load<T>(
    file_path: &str,
    mem_allocator: &(impl MemoryAllocator + ?Sized),
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<TextureView, RhError> {
    let mut f = File::open(file_path)?;

    // Read and process the header
    let mut buffer = [0; 128];
    f.read_exact(&mut buffer)?;

    let magic = dword(&buffer[0..4])?;
    if magic != MAGIC_HEADER {
        return Err(RhError::InvalidFile);
    }

    let header = DdsHeader {
        height: dword(&buffer[12..16])?,
        width: dword(&buffer[16..20])?,
        pitch_or_size: dword(&buffer[20..24])?,
        mipmap_count: dword(&buffer[28..32])?,
        pixel_flags: dword(&buffer[80..84])?,
        four_cc: dword(&buffer[84..88])?,
    };

    let dxt10_header = if header.four_cc == FOURCC_DX10 {
        // Read and process the extended header
        let mut buffer = [0; 20];
        f.read_exact(&mut buffer)?;
        Some(DdsHeaderDxt10 {
            dxgi_format: dword(&buffer[0..4])?,
            resource_dimension: dword(&buffer[4..8])?,
            misc_flag: dword(&buffer[8..12])?,
            array_size: dword(&buffer[12..16])?,
            misc_flags2: dword(&buffer[16..20])?,
        })
    } else {
        None
    };

    debug!(
        "H {}, W {}, SoP {}, mipmaps {}, flags {}",
        header.height,
        header.width,
        header.pitch_or_size,
        header.mipmap_count,
        header.pixel_flags
    );
    debug!("DXT10 extension: {dxt10_header:?}");

    // Vulkano friendly dimensions
    let dimensions = ImageDimensions::Dim2d {
        width: header.width,
        height: header.height,
        array_layers: 1,
    };

    // Mipmaps
    let max_mipmaps = dimensions.max_mip_levels();
    debug!("Max mipmaps based on dimensions: {max_mipmaps}");
    let mip_levels = if header.mipmap_count == 1 {
        MipmapsCount::One
    } else if header.mipmap_count == max_mipmaps {
        MipmapsCount::Log2
    } else {
        MipmapsCount::Specific(header.mipmap_count)
    };
    debug!("Using mipmap levels {mip_levels:?}");

    // pixel_flags bits
    // 0x01 = contains alpha data
    // 0x04 = contains compressed RGB data
    // 0x40 = contains uncompressed RGB data using bitmask flags
    // dxgi_format (if present) can represent many formats including legacy
    // resource_dimension (if present) = 3 for 2D texture or cube map
    // Formats to consider:
    // BC1_RGB_UNORM_BLOCK
    // BC1_RGB_SRGB_BLOCK
    // BC1_RGBA_UNORM_BLOCK
    // BC1_RGBA_SRGB_BLOCK
    // BC3_UNORM_BLOCK
    // BC3_SRGB_BLOCK
    // BC5_UNORM_BLOCK
    // BC5_SNORM_BLOCK
    // BC7_UNORM_BLOCK
    // BC7_SRGB_BLOCK

    // Format comments below based on files created with AMD's "compressonator"
    let (texture_format, block_size) = match header.four_cc {
        // BC1 / DXT1 supports 1 bit alpha which could be indicated in
        // pixel_flags but wasn't with test files so it is ignored.
        // Test files compressed as "BC1" contained "DXT1" identifier
        FOURCC_DXT1 => (Format::BC1_RGB_SRGB_BLOCK, 8),

        // Test files compressed as "BC3" contained "DXT5" identifier
        FOURCC_DXT5 => (Format::BC3_SRGB_BLOCK, 16),

        // The DX10 extension could be used to represent any format, but
        // was not included in BC1 and BC3 test files
        FOURCC_DX10 => {
            if let Some(ext) = dxt10_header {
                match ext.dxgi_format {
                    // Test files used the non sRGB format value but the
                    // data was clearly intended to be sRGB
                    DXGI_BC7_UNORM => (Format::BC7_SRGB_BLOCK, 16),
                    DXGI_BC7_UNORM_SRGB => (Format::BC7_SRGB_BLOCK, 16),
                    _ => {
                        return Err(RhError::UnsupportedFormat);
                    }
                }
            } else {
                return Err(RhError::UnsupportedFormat);
            }
        }

        _ => {
            return Err(RhError::UnsupportedFormat);
        }
    };
    debug!("Using texture format {texture_format:?}");

    // Read the image data
    let mut buffer = Vec::new();
    let data_length = f.read_to_end(&mut buffer)?;
    debug!("Read {data_length} bytes of image data");

    // Create the Vulkan image
    // Ideally it would be as simple as this:
    /*
    let imm_image = ImmutableImage::from_iter(
        memory_allocator,
        buffer,
        dimensions,
        mip_levels,
        texture_format,
        cbb,
    )?;
    */
    // But Vulkano tries to create the mipmaps by scaling and blitting
    // which is no good since the mipmaps already exist in "buffer" and are
    // in a format that doesn't support blitting

    // Start by creating a CpuAccessibleBuffer
    let source = Buffer::from_iter(
        mem_allocator,
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            usage: MemoryUsage::Upload,
            ..Default::default()
        },
        buffer,
    )?;

    // The usage for the immutable image buffer is as the transfer destination
    // from "source" but not a source itself (since no blitting)
    let usage = ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED;

    // The immutable image is created with all the properties and returns a
    // special "initializer" so data can be copied into from "source"
    let (image, initializer) = ImmutableImage::uninitialized(
        mem_allocator,
        dimensions,
        texture_format,
        mip_levels,
        usage,
        ImageCreateFlags::empty(),
        ImageLayout::ShaderReadOnlyOptimal,
        source
            .device()
            .active_queue_family_indices()
            .iter()
            .copied(),
    )?;

    // Next, Vulkano's way is to create a "region" that describes the base image
    // (mipmap level 0) and then record a copy command to copy that from
    // "source" to "image". Then it loops through levels and records blitting
    // commands.
    // Instead, loop through the levels, creating a "region" for each with
    // and then record a copy for that. Hopefully this copies all the image
    // data in a way that Vulkan will understand there are mipmaps and where
    // to find them.
    let mut regions = Vec::new();
    let mut offset = 0;
    let image_access: Arc<dyn ImageAccess> = image.clone();
    for level in 0..image_access.mip_levels() {
        let level_dimensions = dimensions
            .mip_level_dimensions(level)
            .ok_or(RhError::UnsupportedFormat)? // Shouldn't happen...
            .width_height_depth();
        let region = BufferImageCopy {
            buffer_offset: offset,
            image_subresource: ImageSubresourceLayers {
                mip_level: level,
                ..image_access.subresource_layers()
            },
            image_extent: level_dimensions,
            ..Default::default()
        };
        regions.push(region);

        // Calculate the location of the next mipmap in the compressed data
        let pw = std::cmp::max((level_dimensions[0] + 3) / 4, 1);
        let ph = std::cmp::max((level_dimensions[1] + 3) / 4, 1);
        offset += (pw * ph * block_size) as u64;
    }
    debug!("regions vector len {}", regions.len());
    cbb.copy_buffer_to_image(CopyBufferToImageInfo {
        regions: SmallVec::from_vec(regions),
        ..CopyBufferToImageInfo::buffer_image(source, initializer)
    })?;

    Ok(ImageView::new_default(image)?)
}
