// Copyright 2020 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/common_types.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {
class OpenGLState;

class TextureDownloaderES {
public:
    TextureDownloaderES(bool enable_depth_stencil);

    void GetTexImage(GLenum target, GLuint level, GLenum format, const GLenum type,
                     GLint height, GLint width, void* pixels);

private:
    void Test();
    GLuint ConvertDepthToColor(GLuint level, GLenum& format, GLenum& type,
                               GLint height, GLint width);

private:
    static constexpr u16 MAX_SIZE = 1024;

    struct ConversionShader {
        OGLProgram program;
        GLint lod_location{-1};
    };

    OGLVertexArray vao;
    OGLFramebuffer read_fbo_generic;
    OGLFramebuffer depth32_fbo, depth16_fbo;
    OGLRenderbuffer r32ui_renderbuffer, r16_renderbuffer;

    ConversionShader d24_r32ui_conversion_shader;
    ConversionShader d16_r16_conversion_shader;
    ConversionShader d24s8_r32ui_conversion_shader;
    OGLSampler sampler;
};
} // namespace OpenGL
