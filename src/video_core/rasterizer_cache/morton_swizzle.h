// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "common/alignment.h"
#include "core/memory.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/utils.h"
#include "video_core/video_core.h"

namespace OpenGL {

template <bool morton_to_linear, PixelFormat format>
static void MortonCopyTile(u32 stride, u8* tile_buffer, u8* linear_buffer) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 aligned_bytes_per_pixel = GetBytesPerPixel(format);

    for (u32 y = 0; y < 8; y++) {
        for (u32 x = 0; x < 8; x++) {
            u8* tile_ptr = tile_buffer + VideoCore::MortonInterleave(x, y) * bytes_per_pixel;
            u8* linear_ptr = linear_buffer + ((7 - y) * stride + x) * aligned_bytes_per_pixel;
            if constexpr (morton_to_linear) {
                if constexpr (format == PixelFormat::D24S8) {
                    linear_ptr[0] = tile_ptr[3];
                    std::memcpy(linear_ptr + 1, tile_ptr, 3);
                } else if (format == PixelFormat::RGBA8 && GLES) {
                    // because GLES does not have ABGR format
                    // so we will do byteswapping here
                    linear_ptr[0] = tile_ptr[3];
                    linear_ptr[1] = tile_ptr[2];
                    linear_ptr[2] = tile_ptr[1];
                    linear_ptr[3] = tile_ptr[0];
                } else if (format == PixelFormat::RGB8 && GLES) {
                    linear_ptr[0] = tile_ptr[2];
                    linear_ptr[1] = tile_ptr[1];
                    linear_ptr[2] = tile_ptr[0];
                } else {
                    std::memcpy(linear_ptr, tile_ptr, bytes_per_pixel);
                }
            } else {
                if constexpr (format == PixelFormat::D24S8) {
                    std::memcpy(tile_ptr, linear_ptr + 1, 3);
                    tile_ptr[3] = linear_ptr[0];
                } else if (format == PixelFormat::RGBA8 && GLES) {
                    // because GLES does not have ABGR format
                    // so we will do byteswapping here
                    tile_ptr[0] = linear_ptr[3];
                    tile_ptr[1] = linear_ptr[2];
                    tile_ptr[2] = linear_ptr[1];
                    tile_ptr[3] = linear_ptr[0];
                } else if (format == PixelFormat::RGB8 && GLES) {
                    tile_ptr[0] = linear_ptr[2];
                    tile_ptr[1] = linear_ptr[1];
                    tile_ptr[2] = linear_ptr[0];
                } else {
                    std::memcpy(tile_ptr, linear_ptr, bytes_per_pixel);
                }
            }
        }
    }
}

template <bool morton_to_linear, PixelFormat format>
static void MortonCopy(u32 stride, u32 height, u8* linear_buffer, PAddr base, PAddr start, PAddr end) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 tile_size = bytes_per_pixel * 64;

    constexpr u32 aligned_bytes_per_pixel = GetBytesPerPixel(format);
    static_assert(aligned_bytes_per_pixel >= bytes_per_pixel, "");
    linear_buffer += aligned_bytes_per_pixel - bytes_per_pixel;

    const PAddr aligned_down_start = base + Common::AlignDown(start - base, tile_size);
    const PAddr aligned_start = base + Common::AlignUp(start - base, tile_size);
    PAddr aligned_end = base + Common::AlignDown(end - base, tile_size);

    ASSERT(!morton_to_linear || (aligned_start == start && aligned_end == end));

    const u32 begin_pixel_index = (aligned_down_start - base) / bytes_per_pixel;
    u32 x = (begin_pixel_index % (stride * 8)) / 8;
    u32 y = (begin_pixel_index / (stride * 8)) * 8;

    // In OpenGL the texture origin is in the bottom left corner as opposed to other
    // APIs that have it at the top left. To avoid flipping texture coordinates in
    // the shader we read/write the linear buffer backwards
    linear_buffer += ((height - 8 - y) * stride + x) * aligned_bytes_per_pixel;

    auto linear_next_tile = [&] {
        x = (x + 8) % stride;
        linear_buffer += 8 * aligned_bytes_per_pixel;
        if (!x) {
            y += 8;
            linear_buffer -= stride * 9 * aligned_bytes_per_pixel;
        }
    };

    u8* tile_buffer = VideoCore::g_memory->GetPhysicalPointer(start);

    // If during a texture download the start coordinate is inside a tile, swizzle
    // the tile to a temporary buffer and copy the part we are interested in
    if (start < aligned_start && !morton_to_linear) {
        std::array<u8, tile_size> tmp_buf;
        MortonCopyTile<morton_to_linear, format>(stride, tmp_buf.data(), linear_buffer);
        std::memcpy(tile_buffer, tmp_buf.data() + start - aligned_down_start,
                    std::min(aligned_start, end) - start);

        tile_buffer += aligned_start - start;
        linear_next_tile();
    }

    // Pokemon Super Mystery Dungeon will try to use textures that go beyond
    // the end address of VRAM. Clamp the address to the end of VRAM if that happens
    // TODO: Move this to the rasterizer cache
    if (const u32 clamped_end = VideoCore::g_memory->ClampPhysicalAddress(aligned_start, aligned_end);
            clamped_end != aligned_end) {
        LOG_ERROR(Render_OpenGL, "Out of bound texture read address {:#x}, clamping to {:#x}", aligned_end, clamped_end);
        aligned_end = clamped_end;
    }

    const u8* buffer_end = tile_buffer + aligned_end - aligned_start;
    while (tile_buffer < buffer_end) {
        MortonCopyTile<morton_to_linear, format>(stride, tile_buffer, linear_buffer);
        tile_buffer += tile_size;
        linear_next_tile();
    }

    if (end > std::max(aligned_start, aligned_end) && !morton_to_linear) {
        std::array<u8, tile_size> tmp_buf;
        MortonCopyTile<morton_to_linear, format>(stride, tmp_buf.data(), linear_buffer);
        std::memcpy(tile_buffer, tmp_buf.data(), end - aligned_end);
    }
}

using MortonFunc = void (*)(u32, u32, u8*, PAddr, PAddr, PAddr);

static constexpr std::array<MortonFunc, 18> UNSWIZZLE_TABLE = {
    MortonCopy<true, PixelFormat::RGBA8>,  // 0
    MortonCopy<true, PixelFormat::RGB8>,   // 1
    MortonCopy<true, PixelFormat::RGB5A1>, // 2
    MortonCopy<true, PixelFormat::RGB565>, // 3
    MortonCopy<true, PixelFormat::RGBA4>,  // 4
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,                             // 5 - 13
    MortonCopy<true, PixelFormat::D16>,  // 14
    nullptr,                             // 15
    MortonCopy<true, PixelFormat::D24>,  // 16
    MortonCopy<true, PixelFormat::D24S8> // 17
};

static constexpr std::array<MortonFunc, 18> SWIZZLE_TABLE = {
    MortonCopy<false, PixelFormat::RGBA8>,  // 0
    MortonCopy<false, PixelFormat::RGB8>,   // 1
    MortonCopy<false, PixelFormat::RGB5A1>, // 2
    MortonCopy<false, PixelFormat::RGB565>, // 3
    MortonCopy<false, PixelFormat::RGBA4>,  // 4
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,                              // 5 - 13
    MortonCopy<false, PixelFormat::D16>,  // 14
    nullptr,                              // 15
    MortonCopy<false, PixelFormat::D24>,  // 16
    MortonCopy<false, PixelFormat::D24S8> // 17
};

} // namespace OpenGL
