// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <bit>
#include "common/alignment.h"
#include "common/color.h"
#include "core/memory.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/utils.h"
#include "video_core/video_core.h"

namespace OpenGL {

template <PixelFormat format>
constexpr u32 MortonTileIncrement() {
    constexpr u32 bpp = GetFormatBpp(format);

    // 4-bit formats are processed in batches of two pixels since we can't
    // increment by half a byte
    if constexpr (bpp == 4) {
        return 1;
    }

    return bpp / 8;
}

template <PixelFormat format>
constexpr u32 MortonGPUIncrement() {
    if (format == PixelFormat::D24 || GetFormatType(format) == SurfaceType::Texture) {
        if (format == PixelFormat::A8) {
            return 1;
        } else if (format == PixelFormat::RG8 || format == PixelFormat::IA8 ||
                   format == PixelFormat::IA4 || format == PixelFormat::A4) {
            return 2;
        }

        return 4;
    }

    return GetFormatBpp(format) / 8;
}

template <bool morton_to_gpu, PixelFormat format>
constexpr void MortonCopyTile(u32 stride, u8* tile_buffer, u8* gl_buffer) {
    using namespace Pica::Texture;
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 gpu_bytes_per_pixel = GetBytesPerPixel(format);

    // For smaller tiles (8 and 16bit) use SIMD accelerated methods
    // for increased performance. Anything larger than that is faster
    // without vectorization
    if constexpr (morton_to_gpu) {
        if constexpr (bytes_per_pixel == 4 && format != PixelFormat::D24S8) {
            //return DecodeTile32(stride, tile_buffer, gl_buffer);
        } else if constexpr (bytes_per_pixel == 2) {
            //return DecodeTile16(stride, tile_buffer, gl_buffer);
        } else if constexpr (bytes_per_pixel == 1) {
            if constexpr (format == PixelFormat::IA4) {
                //return DecodeTileIA4(stride, tile_buffer, gl_buffer);
            }

            //return DecodeTile8(stride, tile_buffer, gl_buffer);
        }
    } else {
        if constexpr (bytes_per_pixel == 4 && format != PixelFormat::D24S8) {
           // return EncodeTile32(stride, tile_buffer, gl_buffer);
        } else if constexpr (bytes_per_pixel == 2) {
           // return EncodeTile16(stride, gl_buffer, tile_buffer);
        } else if constexpr (bytes_per_pixel == 1) {
            if constexpr (format == PixelFormat::IA4) {
                //return DecodeTileIA4(stride, gl_buffer, gl_buffer);
            }

            //return DecodeTile8(stride, gl_buffer, tile_buffer);
        }
    }

    for (u32 y = 0; y < 8; y++) {
        for (u32 x = 0; x < 8; x++) {
            u8* tile_ptr = tile_buffer + VideoCore::MortonInterleave(x, y) * bytes_per_pixel;
            u8* gl_ptr = gl_buffer + ((7 - y) * stride + x) * gpu_bytes_per_pixel;
            if constexpr (morton_to_gpu) {
                if constexpr (format == PixelFormat::D24S8) {
                    gl_ptr[0] = tile_ptr[3];
                    std::memcpy(gl_ptr + 1, tile_ptr, 3);
                } else if (format == PixelFormat::RGBA8 && GLES) {
                    // because GLES does not have ABGR format
                    // so we will do byteswapping here
                    gl_ptr[0] = tile_ptr[3];
                    gl_ptr[1] = tile_ptr[2];
                    gl_ptr[2] = tile_ptr[1];
                    gl_ptr[3] = tile_ptr[0];
                } else if (format == PixelFormat::RGB8 && GLES) {
                    gl_ptr[1] = tile_ptr[0];
                    gl_ptr[2] = tile_ptr[1];
                    gl_ptr[3] = tile_ptr[2];
                } else if (format == PixelFormat::IA4) {
                    u8 value = tile_ptr[0];
                    gl_ptr[0] = Color::Convert4To8((value & 0xF0) >> 4);
                    gl_ptr[1] = Color::Convert4To8(value & 0xF);
                } else {
                    std::memcpy(gl_ptr, tile_ptr, bytes_per_pixel);
                }
            } else {
                if constexpr (format == PixelFormat::D24S8) {
                    std::memcpy(tile_ptr, gl_ptr + 1, 3);
                    tile_ptr[3] = gl_ptr[0];
                } else if (format == PixelFormat::RGBA8 && GLES) {
                    // because GLES does not have ABGR format
                    // so we will do byteswapping here
                    tile_ptr[0] = gl_ptr[3];
                    tile_ptr[1] = gl_ptr[2];
                    tile_ptr[2] = gl_ptr[1];
                    tile_ptr[3] = gl_ptr[0];
                } else if (format == PixelFormat::RGB8 && GLES) {
                    tile_ptr[0] = gl_ptr[0];
                    tile_ptr[1] = gl_ptr[1];
                    tile_ptr[2] = gl_ptr[2];
                } else if (format == PixelFormat::IA4) {
                    u8 value =  gl_ptr[0] | (gl_ptr[1] << 4);
                    tile_ptr[0] = value;
                } else {
                    std::memcpy(tile_ptr, gl_ptr, bytes_per_pixel);
                }
            }
        }
    }
}

template <bool morton_to_gpu, PixelFormat format>
constexpr void MortonCopy(u32 stride, u32 height, u8* gl_buffer, PAddr base, PAddr start, PAddr end) {
    constexpr u32 bytes_per_pixel = GetFormatBpp(format) / 8;
    constexpr u32 tile_size = bytes_per_pixel * 64;

    constexpr u32 gpu_bytes_per_pixel = GetBytesPerPixel(format);
    static_assert(gpu_bytes_per_pixel >= bytes_per_pixel, "");
    gl_buffer += gpu_bytes_per_pixel - bytes_per_pixel;

    const PAddr aligned_down_start = base + Common::AlignDown(start - base, tile_size);
    const PAddr aligned_start = base + Common::AlignUp(start - base, tile_size);
    const PAddr aligned_end = base + Common::AlignDown(end - base, tile_size);

    ASSERT(!morton_to_gpu || (aligned_start == start && aligned_end == end));

    const u32 begin_pixel_index = (aligned_down_start - base) / bytes_per_pixel;
    u32 x = (begin_pixel_index % (stride * 8)) / 8;
    u32 y = (begin_pixel_index / (stride * 8)) * 8;

    gl_buffer += ((height - 8 - y) * stride + x) * gpu_bytes_per_pixel;

    auto glbuf_next_tile = [&] {
        x = (x + 8) % stride;
        gl_buffer += 8 * gpu_bytes_per_pixel;
        if (x == 0) {
            y += 8;
            gl_buffer -= stride * 9 * gpu_bytes_per_pixel;
        }
    };

    u8* tile_buffer = VideoCore::g_memory->GetPhysicalPointer(start);

    if (start < aligned_start && !morton_to_gpu) {
        std::array<u8, tile_size> tmp_buf;
        MortonCopyTile<morton_to_gpu, format>(stride, &tmp_buf[0], gl_buffer);
        std::memcpy(tile_buffer, &tmp_buf[start - aligned_down_start],
                    std::min(aligned_start, end) - start);

        tile_buffer += aligned_start - start;
        glbuf_next_tile();
    }

    const u8* const buffer_end = tile_buffer + aligned_end - aligned_start;
    PAddr current_paddr = aligned_start;
    while (tile_buffer < buffer_end) {
        // Pokemon Super Mystery Dungeon will try to use textures that go beyond
        // the end address of VRAM. Stop reading if reaches invalid address
        if (!VideoCore::g_memory->IsValidPhysicalAddress(current_paddr) ||
            !VideoCore::g_memory->IsValidPhysicalAddress(current_paddr + tile_size)) {
            LOG_ERROR(Render_OpenGL, "Out of bound texture");
            break;
        }
        MortonCopyTile<morton_to_gpu, format>(stride, tile_buffer, gl_buffer);
        tile_buffer += tile_size;
        current_paddr += tile_size;
        glbuf_next_tile();
    }

    if (end > std::max(aligned_start, aligned_end) && !morton_to_gpu) {
        std::array<u8, tile_size> tmp_buf;
        MortonCopyTile<morton_to_gpu, format>(stride, &tmp_buf[0], gl_buffer);
        std::memcpy(tile_buffer, &tmp_buf[0], end - aligned_end);
    }
}

using MortonFn = void (*)(u32, u32, u8*, PAddr, PAddr, PAddr);

static constexpr std::array<MortonFn, 18> morton_to_gpu_fns = {
    MortonCopy<true, PixelFormat::RGBA8>,  // 0
    MortonCopy<true, PixelFormat::RGB8>,   // 1
    MortonCopy<true, PixelFormat::RGB5A1>, // 2
    MortonCopy<true, PixelFormat::RGB565>, // 3
    MortonCopy<true, PixelFormat::RGBA4>,  // 4
    MortonCopy<true, PixelFormat::IA8>,  // 5
    MortonCopy<true, PixelFormat::RG8>,  // 6
    nullptr,
    MortonCopy<true, PixelFormat::A8>,  // 8
    MortonCopy<true, PixelFormat::IA4>,  // 9
    nullptr,
    MortonCopy<true, PixelFormat::A4>,
    nullptr,
    nullptr,                             // 5 - 13
    MortonCopy<true, PixelFormat::D16>,  // 14
    nullptr,                             // 15
    MortonCopy<true, PixelFormat::D24>,  // 16
    MortonCopy<true, PixelFormat::D24S8> // 17
};

static constexpr std::array<MortonFn, 18> gpu_to_morton_fns = {
    MortonCopy<false, PixelFormat::RGBA8>,  // 0
    MortonCopy<false, PixelFormat::RGB8>,   // 1
    MortonCopy<false, PixelFormat::RGB5A1>, // 2
    MortonCopy<false, PixelFormat::RGB565>, // 3
    MortonCopy<false, PixelFormat::RGBA4>,  // 4
    MortonCopy<false, PixelFormat::IA8>,    // 5
    MortonCopy<false, PixelFormat::RG8>,    // 6
    nullptr,
    MortonCopy<false, PixelFormat::A8>,     // 8
    MortonCopy<false, PixelFormat::IA4>,    // 9
    nullptr,
    MortonCopy<false, PixelFormat::A4>,     // 11
    nullptr,
    nullptr,                              // 5 - 13
    MortonCopy<false, PixelFormat::D16>,  // 14
    nullptr,                              // 15
    MortonCopy<false, PixelFormat::D24>,  // 16
    MortonCopy<false, PixelFormat::D24S8> // 17
};

} // namespace OpenGL
