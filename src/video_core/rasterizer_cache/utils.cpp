// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <glad/glad.h>
#include "video_core/texture/texture_decode.h"
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

ClearValue MakeClearValue(Aspect aspect, PixelFormat format, const u8* fill_data) {
    ClearValue result{};
    switch (aspect) {
    case Aspect::Color: {
        Pica::Texture::TextureInfo tex_info{};
        tex_info.format = static_cast<Pica::TexturingRegs::TextureFormat>(format);

        Common::Vec4<u8> color = Pica::Texture::LookupTexture(fill_data, 0, 0, tex_info);
        result.color = color / 255.f;
        break;
    }
    case Aspect::Depth: {
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
    case Aspect::DepthStencil: {
        u32 clear_value_uint;
        std::memcpy(&clear_value_uint, fill_data, sizeof(u32));

        result.depth = (clear_value_uint & 0xFFFFFF) / 16777215.0f; // 2^24 - 1
        result.stencil = (clear_value_uint >> 24);
        break;
    }
    }

    return result;
}

} // namespace OpenGL
