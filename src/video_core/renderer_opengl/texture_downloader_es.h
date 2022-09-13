// Copyright 2020 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "common/common_types.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {

class OpenGLState;

class TextureDownloaderES {
    static constexpr u16 MAX_SIZE = 1024;
public:
    TextureDownloaderES(bool enable_depth_stencil);

    /**
     * OpenGL ES does not support glGetTexImage. Obtain the pixels by attaching the
     * texture to a framebuffer.
     * Originally from https://github.com/apitrace/apitrace/blob/master/retrace/glstate_images.cpp
     * Depth texture download assumes that the texture's format tuple matches what is found
     * OpenGL::depth_format_tuples
     */
    void GetTexImage(GLenum target, GLuint level, GLenum format, const GLenum type,
                     GLint height, GLint width, void* pixels) const;

private:
    /**
     * OpenGL ES does not support glReadBuffer for depth/stencil formats.
     * This gets around it by converting to a Red surface before downloading
     */
    GLuint ConvertDepthToColor(GLuint level, GLenum& format, GLenum& type,
                               GLint height, GLint width) const;

    /// Self tests for the texture downloader
    void Test();

private:
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
