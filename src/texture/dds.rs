use crate::rh_error::RhError;
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
    format::Format,
    image::{
        view::ImageView, Image, ImageCreateInfo, ImageSubresourceLayers,
        ImageUsage,
    },
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter,
    },
    Validated,
};

const MAGIC_HEADER: u32 = 0x2053_4444;
const FOURCC_DXT1: u32 = 0x3154_5844;
const FOURCC_DXT5: u32 = 0x3554_5844;
const FOURCC_DX10: u32 = 0x3031_5844;
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

/// Loads a DDS file and records commands to a queue to transfer the data to
/// GPU memory. The queue must be executed to finish the transfer.
///
/// # Errors
/// May return `RhError`
///
/// # Panics
/// Will panic if a `vulkano::ValidationError` is returned by Vulkan
#[allow(clippy::too_many_lines)]
pub fn load<T>(
    file_path: &str,
    mem_allocator: Arc<(dyn MemoryAllocator)>,
    cbb: &mut AutoCommandBufferBuilder<T>,
) -> Result<Arc<ImageView>, RhError> {
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
    let extent = [header.width, header.height, 1];

    // Mipmaps
    debug!(
        "Max mipmaps based on dimensions: {}",
        vulkano::image::max_mip_levels(extent)
    );
    let mip_levels = header.mipmap_count;
    debug!("Using {} mipmap levels", mip_levels);

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
                    DXGI_BC7_UNORM | DXGI_BC7_UNORM_SRGB => {
                        (Format::BC7_SRGB_BLOCK, 16)
                    }
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

    // Originally vulkano tried to create the mipmaps by scaling and blitting
    // which is no good since the mipmaps already exist in the DDS file and are
    // in a format that doesn't support blitting. That has probably been
    // changed now though, but this method seems to work.

    // Read the image data into a CPU accessible buffer as the "source"
    let source = {
        // This Vec<u8> can go out of scope once the data is in the buffer
        let mut image_data = Vec::new();
        let data_length = f.read_to_end(&mut image_data)?;
        debug!("Read {data_length} bytes of image data");

        Buffer::from_iter(
            mem_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            image_data,
        )
        .map_err(Validated::unwrap)?
    };

    // Create a vulkano image for the destination
    let image = Image::new(
        mem_allocator,
        ImageCreateInfo {
            format: texture_format,
            extent,
            array_layers: 1, // Default but listed here for clarity
            mip_levels,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .map_err(Validated::unwrap)?;

    // Next, vulkano's old mipmap creation code's way was to create a "region"
    // that describes the base image (mipmap level 0) and then record a copy
    // command to copy that from `source` to `image`. Then it looped through
    // levels and recorded blitting commands.
    // Instead of doing that, loop through the levels, creating a "region" for
    // each and then record a copy command for that. That seems to get all the
    // image data in a way that Vulkan will understand that there are mipmaps
    // and be able to find them.
    let mut regions = Vec::new();
    let mut offset = 0;
    let image_access = image.clone();
    for level in 0..image_access.mip_levels() {
        let image_extent = vulkano::image::mip_level_extent(extent, level)
            .ok_or(RhError::UnsupportedFormat)?; // Shouldn't happen
        let region = BufferImageCopy {
            buffer_offset: offset,
            image_subresource: ImageSubresourceLayers {
                mip_level: level,
                ..image_access.subresource_layers()
            },
            image_extent,
            ..Default::default()
        };
        regions.push(region);

        // Calculate the location of the next mipmap in the compressed data
        let pw = std::cmp::max((image_extent[0] + 3) / 4, 1);
        let ph = std::cmp::max((image_extent[1] + 3) / 4, 1);
        offset += u64::from(pw * ph * block_size);
    }
    debug!("regions vector len {}", regions.len());

    // Record the commands to transfer the regions from the source buffer
    // to the destination image
    cbb.copy_buffer_to_image(CopyBufferToImageInfo {
        regions: SmallVec::from_vec(regions),
        ..CopyBufferToImageInfo::buffer_image(source, image.clone())
    })
    .unwrap(); // This is a Box<ValidationError>;

    // Create an ImageView to the image and return it
    let image_view =
        ImageView::new_default(image).map_err(Validated::unwrap)?;
    Ok(image_view)
}
