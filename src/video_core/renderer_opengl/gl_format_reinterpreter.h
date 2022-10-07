// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {

class Surface;

class FormatReinterpreterBase {
public:
    virtual ~FormatReinterpreterBase() = default;

    virtual VideoCore::PixelFormat GetSourceFormat() const = 0;
    virtual void Reinterpret(const Surface& source, VideoCore::Rect2D src_rect,
                             const Surface& dest, VideoCore::Rect2D dst_rect) = 0;
};

using ReinterpreterList = std::vector<std::unique_ptr<FormatReinterpreterBase>>;

class D24S8toRGBA8 final : public FormatReinterpreterBase {
public:
    D24S8toRGBA8(bool use_texture_view);

    [[nodiscard]] VideoCore::PixelFormat GetSourceFormat() const override {
        return VideoCore::PixelFormat::D24S8;
    }

    void Reinterpret(const Surface& source, VideoCore::Rect2D src_rect,
                     const Surface& dest, VideoCore::Rect2D dst_rect) override;

private:
    bool use_texture_view{};
    OGLProgram program{};
    GLint src_offset_loc{-1};
    OGLTexture temp_tex{};
    VideoCore::Rect2D temp_rect{0, 0, 0, 0};
};

class RGBA4toRGB5A1 final : public FormatReinterpreterBase {
public:
    RGBA4toRGB5A1();

    [[nodiscard]] VideoCore::PixelFormat GetSourceFormat() const override {
        return VideoCore::PixelFormat::RGBA4;
    }

    void Reinterpret(const Surface& source, VideoCore::Rect2D src_rect,
                     const Surface& dest, VideoCore::Rect2D dst_rect) override;

private:
    OGLFramebuffer read_fbo;
    OGLFramebuffer draw_fbo;
    OGLProgram program;
    GLint dst_size_loc{-1}, src_size_loc{-1}, src_offset_loc{-1};
    OGLVertexArray vao;
};

} // namespace OpenGL
