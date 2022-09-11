// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <glad/glad.h>
#include "common/assert.h"
#include "core/memory.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/rasterizer_cache/morton_swizzle.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/video_core.h"

namespace VideoCore {

void SwizzleTexture(const SurfaceParams& params, u32 start_offset,
                    std::span<std::byte> source_linear, std::span<std::byte> dest_tiled) {
    const u32 func_index = static_cast<u32>(params.pixel_format);
    const MortonFunc SwizzleImpl = SWIZZLE_TABLE[func_index];
    SwizzleImpl(params.stride, params.height, start_offset, source_linear, dest_tiled);
}

void UnswizzleTexture(const SurfaceParams& params, u32 start_offset,
                      std::span<std::byte> source_tiled, std::span<std::byte> dest_linear) {
    const u32 func_index = static_cast<u32>(params.pixel_format);
    const MortonFunc UnswizzleImpl = UNSWIZZLE_TABLE[func_index];
    UnswizzleImpl(params.stride, params.height, start_offset, dest_linear, source_tiled);
}

ClearValue MakeClearValue(SurfaceType type, PixelFormat format, const u8* fill_data) {
    ClearValue result{};
    switch (type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill: {
        Pica::Texture::TextureInfo tex_info{};
        tex_info.format = static_cast<Pica::TexturingRegs::TextureFormat>(format);

        Common::Vec4<u8> color = Pica::Texture::LookupTexture(fill_data, 0, 0, tex_info);
        result.color = color / 255.f;
        break;
    }
    case SurfaceType::Depth: {
        u32 depth_uint = 0;
        if (format == PixelFormat::D16) {
            std::memcpy(&depth_uint, fill_data, 2);
            result.depth = depth_uint / 65535.0f; // 2^16 - 1
        } else if (format == PixelFormat::D24) {
            std::memcpy(&depth_uint, fill_data, 3);
            result.depth = depth_uint / 16777215.0f; // 2^24 - 1
        }
        break;
    }
    case SurfaceType::DepthStencil: {
        u32 clear_value_uint;
        std::memcpy(&clear_value_uint, fill_data, sizeof(u32));

        result.depth = (clear_value_uint & 0xFFFFFF) / 16777215.0f; // 2^24 - 1
        result.stencil = (clear_value_uint >> 24);
        break;
    }
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    return result;
}

} // namespace OpenGL
