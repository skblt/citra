// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include <bit>
#include "common/alignment.h"
#include "core/memory.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/utils.h"
#include "video_core/video_core.h"

namespace OpenGL {

inline u32 MakeInt(std::span<std::byte> bytes) {
    u32 integer{};
    std::memcpy(&integer, bytes.data(), sizeof(u32));

    return integer;
}

template <bool morton_to_linear, PixelFormat format>
inline void MortonCopyTile(u32 stride, std::span<std::byte> tile_buffer, std::span<std::byte> linear_buffer) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 linear_bytes_per_pixel = GetBytesPerPixel(format);

    for (u32 y = 0; y < 8; y++) {
        for (u32 x = 0; x < 8; x++) {
            const u32 tile_offset = VideoCore::MortonInterleave(x, y) * bytes_per_pixel;
            const u32 linear_offset = ((7 - y) * stride + x) * linear_bytes_per_pixel;
            auto tile_pixel = tile_buffer.subspan(tile_offset, bytes_per_pixel);
            auto linear_pixel = linear_buffer.subspan(linear_offset, linear_bytes_per_pixel);

            if constexpr (morton_to_linear) {
                if constexpr (format == PixelFormat::D24S8) {
                    const u32 s8d24 = MakeInt(tile_pixel);
                    const u32 d24s8 = std::rotl(s8d24, 8);
                    std::memcpy(linear_pixel.data(), &d24s8, sizeof(u32));
                } else if (format == PixelFormat::RGBA8 && GLES) {
                    const u32 abgr = MakeInt(tile_pixel);
                    const u32 rgba = std::byteswap(abgr);
                    std::memcpy(linear_pixel.data(), &rgba, sizeof(u32));
                } else if (format == PixelFormat::RGB8 && GLES) {
                    std::memcpy(linear_pixel.data(), tile_pixel.data(), 3);
                    std::swap(linear_pixel[0], linear_pixel[2]);
                } else {
                    std::memcpy(linear_pixel.data(), tile_pixel.data(), bytes_per_pixel);
                }
            } else {
                if constexpr (format == PixelFormat::D24S8) {
                    const u32 d24s8 = MakeInt(linear_pixel);
                    const u32 s8d24 = std::rotr(d24s8, 8);
                    std::memcpy(tile_pixel.data(), &s8d24, sizeof(u32));
                } else if (format == PixelFormat::RGBA8 && GLES) {
                    const u32 rgba = MakeInt(linear_pixel);
                    const u32 abgr = std::byteswap(rgba);
                    std::memcpy(tile_pixel.data(), &abgr, sizeof(u32));
                } else if (format == PixelFormat::RGB8 && GLES) {
                    std::memcpy(tile_pixel.data(), linear_pixel.data(), 3);
                    std::swap(tile_pixel[0], tile_pixel[2]);
                } else {
                    std::memcpy(tile_pixel.data(), linear_pixel.data(), bytes_per_pixel);
                }
            }
        }
    }
}

template <bool morton_to_linear, PixelFormat format>
static void MortonCopy(u32 stride, u32 height,
                       std::span<std::byte> linear_buffer, std::span<std::byte> tiled_buffer,
                       PAddr base, PAddr start, PAddr end) {

    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 aligned_bytes_per_pixel = GetBytesPerPixel(format);
    static_assert(aligned_bytes_per_pixel >= bytes_per_pixel, "");

    constexpr u32 tile_size = bytes_per_pixel * 64;
    const u32 linear_tile_size = (7 * stride + 8) * aligned_bytes_per_pixel;

    // This only applies for D24 format, by shifting the span one byte all pixels
    // are written properly without byteswap
    u32 linear_offset = aligned_bytes_per_pixel - bytes_per_pixel;

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
    //linear_buffer += ((height - 8 - y) * stride + x) * aligned_bytes_per_pixel;
    linear_offset += ((height - 8 - y) * stride + x) * aligned_bytes_per_pixel;

    auto linear_next_tile = [&] {
        x = (x + 8) % stride;
        linear_offset += 8 * aligned_bytes_per_pixel;
        if (!x) {
            y  = (y + 8) % height;
            if (!y) {
                return;
            }

            linear_offset -= stride * 9 * aligned_bytes_per_pixel;
        }
    };

    u8* tile_buffer;
    if constexpr (morton_to_linear) {
        tile_buffer = (u8*)tiled_buffer.data();
    } else {
        tile_buffer = VideoCore::g_memory->GetPhysicalPointer(start);
    }

    // If during a texture download the start coordinate is inside a tile, swizzle
    // the tile to a temporary buffer and copy the part we are interested in
    if (start < aligned_start && !morton_to_linear) {
        std::array<std::byte, tile_size> tmp_buf;
        std::span<std::byte> linear_data = linear_buffer.last(linear_buffer.size() - linear_offset);

        MortonCopyTile<morton_to_linear, format>(stride, tmp_buf, linear_data);
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
        std::span<std::byte> linear_data = linear_buffer.last(linear_buffer.size() - linear_offset);
        auto tiled_data = std::span<std::byte>{(std::byte*)tile_buffer, tile_size};

        MortonCopyTile<morton_to_linear, format>(stride, tiled_data, linear_data);
        tile_buffer += tile_size;
        linear_next_tile();
    }

    if (end > std::max(aligned_start, aligned_end) && !morton_to_linear) {
        std::array<std::byte, tile_size> tmp_buf;
        std::span<std::byte> linear_data = linear_buffer.last(linear_buffer.size() - linear_offset);
        MortonCopyTile<morton_to_linear, format>(stride, tmp_buf, linear_data);
        std::memcpy(tile_buffer, tmp_buf.data(), end - aligned_end);
    }
}

using MortonFunc = void (*)(u32, u32, std::span<std::byte>, std::span<std::byte>, PAddr, PAddr, PAddr);

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
