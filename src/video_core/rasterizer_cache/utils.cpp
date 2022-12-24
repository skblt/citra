// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/assert.h"
#include "video_core/rasterizer_cache/morton_swizzle.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/texture/texture_decode.h"

namespace VideoCore {

std::array<u8, 4> MakeFillBuffer(const SurfaceBase& fill_surface, PAddr copy_addr) {
    const PAddr fill_offset = (copy_addr - fill_surface.addr) % fill_surface.fill_size;
    std::array<u8, 4> fill_buffer;

    u32 fill_buff_pos = fill_offset;
    for (std::size_t i = 0; i < fill_buffer.size(); i++) {
        fill_buffer[i] = fill_surface.fill_data[fill_buff_pos++ % fill_surface.fill_size];
    }

    return fill_buffer;
}

ClearValue MakeClearValue(const SurfaceBase& fill_surface, PixelFormat format, SurfaceType type,
                          PAddr copy_addr) {
    const std::array fill_buffer = MakeFillBuffer(fill_surface, copy_addr);

    ClearValue result{};
    switch (type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill: {
        Pica::Texture::TextureInfo tex_info{};
        tex_info.format = static_cast<Pica::TexturingRegs::TextureFormat>(format);
        const auto color = Pica::Texture::LookupTexture(fill_buffer.data(), 0, 0, tex_info);
        result.color = color / 255.f;
        break;
    }
    case SurfaceType::Depth: {
        u32 depth_uint = 0;
        if (format == PixelFormat::D16) {
            std::memcpy(&depth_uint, fill_buffer.data(), 2);
            result.depth = depth_uint / 65535.0f; // 2^16 - 1
        } else if (format == PixelFormat::D24) {
            std::memcpy(&depth_uint, fill_buffer.data(), 3);
            result.depth = depth_uint / 16777215.0f; // 2^24 - 1
        }
        break;
    }
    case SurfaceType::DepthStencil: {
        u32 clear_value_uint;
        std::memcpy(&clear_value_uint, fill_buffer.data(), sizeof(u32));
        result.depth = (clear_value_uint & 0xFFFFFF) / 16777215.0f; // 2^24 - 1
        result.stencil = (clear_value_uint >> 24);
        break;
    }
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    return result;
}

std::array<u32, MAX_PICA_LEVELS> CalculateMipLevelOffsets(const SurfaceParams& params) {
    ASSERT(params.levels <= MAX_PICA_LEVELS && params.stride == params.width);

    const u32 bits_per_pixel = GetFormatBpp(params.pixel_format);
    u32 width = params.width;
    u32 height = params.height;

    std::array<u32, MAX_PICA_LEVELS> offsets{};
    u32 offset = params.addr;
    for (u32 level = 0; level < params.levels; level++) {
        offsets[level] = offset;
        offset += width * height * bits_per_pixel / 8;

        width >>= 1;
        height >>= 1;
    }

    return offsets;
}

u32 CalculateSurfaceSize(const SurfaceParams& params) {
    ASSERT(params.levels <= MAX_PICA_LEVELS && params.stride == params.width);

    const u32 bits_per_pixel = GetFormatBpp(params.pixel_format);
    u32 width = params.width;
    u32 height = params.height;

    u32 size = 0;
    for (u32 level = 0; level < params.levels; level++) {
        size += width * height * bits_per_pixel / 8;
        width >>= 1;
        height >>= 1;
    }

    return size;
}

std::pair<u32, u32> LevelRange(const SurfaceParams& params, SurfaceInterval interval) {
    u32 start_level = params.levels - 1;
    while (params.mipmap_offsets[start_level] > boost::icl::first(interval)) {
        start_level--;
    }

    s32 end_level = params.levels - 1;
    while (params.mipmap_offsets[end_level] > boost::icl::last_next(interval)) {
        end_level--;
    }

    ASSERT(start_level >= 0 && end_level < params.levels && start_level <= end_level);
    return std::make_pair(start_level, end_level);
}

SurfaceInterval LevelInterval(const SurfaceParams& params, u32 level) {
    ASSERT(params.levels > level);
    const PAddr start = params.mipmap_offsets[level];
    const PAddr end = level == (params.levels - 1) ? params.end
                                                   : params.mipmap_offsets[level + 1];
    return SurfaceInterval{start, end};
}

u32 MipLevels(u32 width, u32 height, u32 max_level) {
    u32 levels = 1;
    while (width > 8 && height > 8)  {
        levels++;
        width >>= 1;
        height >>= 1;
    }

    return std::min(levels, max_level + 1);
}

void SwizzleTexture(const SurfaceParams& swizzle_info, PAddr start_addr, PAddr end_addr,
                    std::span<std::byte> source_linear, std::span<std::byte> dest_tiled,
                    bool convert) {
    const u32 func_index = static_cast<u32>(swizzle_info.pixel_format);
    const MortonFunc SwizzleImpl = (convert ? SWIZZLE_TABLE_CONVERTED : SWIZZLE_TABLE)[func_index];
    SwizzleImpl(swizzle_info.width, swizzle_info.height, start_addr - swizzle_info.addr,
                end_addr - swizzle_info.addr, source_linear, dest_tiled);
}

void UnswizzleTexture(const SurfaceParams& unswizzle_info, PAddr start_addr, PAddr end_addr,
                      std::span<std::byte> source_tiled, std::span<std::byte> dest_linear,
                      bool convert) {
    const u32 func_index = static_cast<u32>(unswizzle_info.pixel_format);
    const MortonFunc UnswizzleImpl =
        (convert ? UNSWIZZLE_TABLE_CONVERTED : UNSWIZZLE_TABLE)[func_index];
    UnswizzleImpl(unswizzle_info.width, unswizzle_info.height, start_addr - unswizzle_info.addr,
                  end_addr - unswizzle_info.addr, dest_linear, source_tiled);
}

} // namespace VideoCore
