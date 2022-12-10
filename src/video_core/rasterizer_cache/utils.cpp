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
