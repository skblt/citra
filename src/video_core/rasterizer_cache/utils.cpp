// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <glad/glad.h>
#include "common/assert.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/rasterizer_cache/morton_swizzle.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_opengl/gl_vars.h"

namespace OpenGL {

constexpr FormatTuple tex_tuple = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

static constexpr std::array<FormatTuple, 4> depth_format_tuples = {{
    {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT}, // D16
    {},
    {GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT},   // D24
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8}, // D24S8
}};

static constexpr std::array<FormatTuple, 5> fb_format_tuples = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8},     // RGBA8
    {GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE},              // RGB8
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
}};

// Same as above, with minor changes for OpenGL ES. Replaced
// GL_UNSIGNED_INT_8_8_8_8 with GL_UNSIGNED_BYTE and
// GL_BGR with GL_RGB
static constexpr std::array<FormatTuple, 5> fb_format_tuples_oes = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE},            // RGBA8
    {GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE},              // RGB8
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
}};

const FormatTuple& GetFormatTuple(PixelFormat pixel_format) {
    const SurfaceType type = GetFormatType(pixel_format);
    const std::size_t format_index = static_cast<std::size_t>(pixel_format);

    if (type == SurfaceType::Color) {
        ASSERT(format_index < fb_format_tuples.size());
        return (GLES ? fb_format_tuples_oes : fb_format_tuples)[format_index];
    } else if (type == SurfaceType::Depth || type == SurfaceType::DepthStencil) {
        const std::size_t tuple_idx = format_index - 14;
        ASSERT(tuple_idx < depth_format_tuples.size());
        return depth_format_tuples[tuple_idx];
    }

    return tex_tuple;
}

void SwizzleTexture(const SurfaceParams& params, u32 flush_start, u32 flush_end,
                    std::span<std::byte> source_linear, std::span<std::byte> dest_tiled) {
    const u32 func_index = static_cast<u32>(params.pixel_format);
    const MortonFunc SwizzleImpl = SWIZZLE_TABLE[func_index];

    // TODO: Move memory access out of the morton function
    SwizzleImpl(params.stride, params.height, source_linear, dest_tiled, params.addr, flush_start, flush_end);
}

void UnswizzleTexture(const SurfaceParams& params, u32 load_start, u32 load_end,
                      std::span<std::byte> source_tiled, std::span<std::byte> dest_linear) {
    // TODO: Integrate this to UNSWIZZLE_TABLE
    if (params.type == SurfaceType::Texture) {
        Pica::Texture::TextureInfo tex_info{};
        tex_info.width = params.width;
        tex_info.height = params.height;
        tex_info.format = static_cast<Pica::TexturingRegs::TextureFormat>(params.pixel_format);
        tex_info.SetDefaultStride();
        tex_info.physical_address = params.addr;

        const u32 start_pixel = params.PixelsInBytes(load_start - params.addr);
        const u8* source_data = reinterpret_cast<const u8*>(source_tiled.data());
        for (u32 i = 0; i < params.PixelsInBytes(load_end - load_start); i++) {
            const u32 x = (i + start_pixel) % params.stride;
            const u32 y = (i + start_pixel) / params.stride;

            auto vec4 = Pica::Texture::LookupTexture(source_data, x, params.height - 1 - y, tex_info);
            std::memcpy(dest_linear.data() + i * sizeof(u32), vec4.AsArray(), sizeof(u32));
        }

    } else {
        const u32 func_index = static_cast<u32>(params.pixel_format);
        const MortonFunc UnswizzleImpl = UNSWIZZLE_TABLE[func_index];
        UnswizzleImpl(params.stride, params.height, dest_linear, source_tiled, params.addr, load_start, load_end);
    }
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
