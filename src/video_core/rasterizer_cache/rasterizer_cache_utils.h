// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <glad/glad.h>
#include <functional>
#include "common/hash.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace OpenGL {

constexpr std::array<int, 4> DEFAULT_SWIZZLE = {GL_RED, GL_GREEN, GL_BLUE, GL_ALPHA};

struct FormatTuple {
    constexpr FormatTuple() = default;
    constexpr FormatTuple(GLint internal, GLenum format, GLenum type,
                std::array<GLint, 4> mask = DEFAULT_SWIZZLE) : internal_format(internal),
        format(format), type(type), swizzle_mask(mask) {}

    constexpr auto operator<=>(const FormatTuple&) const = default;

    GLint internal_format;
    GLenum format;
    GLenum type;
    std::array<GLint, 4> swizzle_mask;
};

const FormatTuple& GetFormatTuple(PixelFormat pixel_format);

struct HostTextureTag {
    FormatTuple format_tuple{};
    u32 width = 0;
    u32 height = 0;

    constexpr auto operator<=>(const HostTextureTag&) const = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(HostTextureTag));
    }
};

struct TextureCubeConfig {
    PAddr px;
    PAddr nx;
    PAddr py;
    PAddr ny;
    PAddr pz;
    PAddr nz;
    u32 width;
    Pica::TexturingRegs::TextureFormat format;

    constexpr auto operator<=>(const TextureCubeConfig&) const = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(TextureCubeConfig));
    }
};

} // namespace OpenGL

namespace std {
template <>
struct hash<OpenGL::HostTextureTag> {
    std::size_t operator()(const OpenGL::HostTextureTag& tag) const noexcept {
        return tag.Hash();
    }
};

template <>
struct hash<OpenGL::TextureCubeConfig> {
    std::size_t operator()(const OpenGL::TextureCubeConfig& config) const noexcept {
        return config.Hash();
    }
};
} // namespace std
