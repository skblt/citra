// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "video_core/rasterizer_cache/rasterizer_cache_utils.h"
#include "video_core/renderer_opengl/gl_vars.h"

namespace OpenGL {

constexpr FormatTuple tex_tuple = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

static constexpr std::array<FormatTuple, 4> depth_format_tuples = {{
    {GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT}, // D16
    {},
    {GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT},   // D24
    {GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8}, // D24S8
}};

static constexpr std::array<FormatTuple, 12> fb_format_tuples = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8},     // RGBA8
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8},     // RGB8
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, {GL_GREEN, GL_GREEN, GL_GREEN, GL_RED}}, // IA8
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, {GL_GREEN, GL_RED, GL_ZERO, GL_ONE}}, // RG8
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_RED, GL_RED, GL_RED, GL_ONE}}, // I8
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_ZERO, GL_ZERO, GL_ZERO, GL_RED}}, // A8
    // The 4-bit texture formats are converted to the conresponding 8-bit ones
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, {GL_RED, GL_RED, GL_RED, GL_GREEN}}, // IA4
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_RED, GL_RED, GL_RED, GL_ONE}}, // I4
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_ZERO, GL_ZERO, GL_ZERO, GL_RED}}, // A4
}};

// Same as above, with minor changes for OpenGL ES. Replaced
// GL_UNSIGNED_INT_8_8_8_8 with GL_UNSIGNED_BYTE and
// GL_BGR with GL_RGB
static constexpr std::array<FormatTuple, 12> fb_format_tuples_oes = {{
    {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE},            // RGBA8
    {GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE},              // RGB8
    {GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    {GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    {GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, {GL_GREEN, GL_GREEN, GL_GREEN, GL_RED}}, // IA8
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, {GL_GREEN, GL_RED, GL_ZERO, GL_ONE}}, // RG8
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_RED, GL_RED, GL_RED, GL_ONE}}, // I8
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_ZERO, GL_ZERO, GL_ZERO, GL_RED}}, // A8
    // The 4-bit texture formats are converted to the conresponding 8-bit ones
    {GL_RG8, GL_RG, GL_UNSIGNED_BYTE, {GL_RED, GL_RED, GL_RED, GL_GREEN}}, // IA4
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_RED, GL_RED, GL_RED, GL_ONE}}, // I4
    {GL_R8, GL_RED, GL_UNSIGNED_BYTE, {GL_ZERO, GL_ZERO, GL_ZERO, GL_RED}}, // A4
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

    // TODO: Remove this
    if (pixel_format == PixelFormat::A8) {
        return fb_format_tuples[8];
    } else if (pixel_format == PixelFormat::RG8) {
        return fb_format_tuples[6];
    } else if (pixel_format == PixelFormat::IA8) {
        return fb_format_tuples[5];
    } else if (pixel_format == PixelFormat::IA4) {
        return fb_format_tuples[9];
    }

    return tex_tuple;
}

} // namespace OpenGL
