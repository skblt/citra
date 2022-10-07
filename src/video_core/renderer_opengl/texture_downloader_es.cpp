// Copyright 2020 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <chrono>
#include <vector>
#include <fmt/chrono.h>
#include "common/logging/log.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"
#include "video_core/renderer_opengl/texture_downloader_es.h"

#include "shaders/depth_to_color.frag"
#include "shaders/depth_to_color.vert"
#include "shaders/ds_to_color.frag"

namespace OpenGL {

static constexpr std::array DEPTH_TUPLES_HACK = {
    FormatTuple{GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT},              // D16
    FormatTuple{}, FormatTuple{GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT}, // D24
    FormatTuple{GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8},              // D24S8
};

const FormatTuple& GetFormatTuple(VideoCore::PixelFormat format) {
    return DEPTH_TUPLES_HACK[static_cast<u32>(format)];
}

void TextureDownloaderES::Test() {
    auto cur_state = OpenGLState::GetCurState();
    OpenGLState state;

    {
        GLint range[2];
        GLint precision;
#define PRECISION_TEST(type)                                                                       \
    glGetShaderPrecisionFormat(GL_FRAGMENT_SHADER, type, range, &precision);                       \
    LOG_INFO(Render_OpenGL, #type " range: [{}, {}], precision: {}", range[0], range[1], precision);
        PRECISION_TEST(GL_LOW_INT);
        PRECISION_TEST(GL_MEDIUM_INT);
        PRECISION_TEST(GL_HIGH_INT);
        PRECISION_TEST(GL_LOW_FLOAT);
        PRECISION_TEST(GL_MEDIUM_FLOAT);
        PRECISION_TEST(GL_HIGH_FLOAT);
#undef PRECISION_TEST
    }
    glActiveTexture(GL_TEXTURE0);

    const auto test = [this, &state]<typename T>(FormatTuple tuple, std::vector<T> original_data,
                                                 std::size_t tex_size, auto data_generator) {
        OGLTexture texture;
        texture.Create();
        state.texture_units[0].texture_2d = texture.handle;
        state.Apply();

        original_data.resize(tex_size * tex_size);
        for (std::size_t idx = 0; idx < original_data.size(); ++idx) {
            original_data[idx] = data_generator(idx);
        }
        GLsizei tex_sizei = static_cast<GLsizei>(tex_size);
        glTexStorage2D(GL_TEXTURE_2D, 1, tuple.internal_format, tex_sizei, tex_sizei);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, tex_sizei, tex_sizei, tuple.format, tuple.type,
                        original_data.data());

        std::vector<T> new_data(original_data.size());
        glFinish();
        auto start = std::chrono::high_resolution_clock::now();
        GetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type, tex_sizei, tex_sizei,
                    new_data.data());
        glFinish();
        auto time = std::chrono::high_resolution_clock::now() - start;
        LOG_INFO(Render_OpenGL, "test took {}", std::chrono::duration<double, std::milli>(time));

        T diff = 0;
        for (std::size_t idx = 0; idx < original_data.size(); ++idx)
            if (new_data[idx] - original_data[idx] != diff) {
                diff = new_data[idx] - original_data[idx];
                // every time the error between the real and expected value changes, log it
                // some error is expected in D24 due to floating point precision
                LOG_WARNING(Render_OpenGL, "difference changed at {:#X}: {:#X} -> {:#X}", idx,
                            original_data[idx], new_data[idx]);
            }
    };
    LOG_INFO(Render_OpenGL, "GL_DEPTH24_STENCIL8 download test starting");
    test(GetFormatTuple(VideoCore::PixelFormat::D24S8), std::vector<u32>{}, 4096,
         [](std::size_t idx) { return static_cast<u32>((idx << 8) | (idx & 0xFF)); });
    LOG_INFO(Render_OpenGL, "GL_DEPTH_COMPONENT24 download test starting");
    test(GetFormatTuple(VideoCore::PixelFormat::D24), std::vector<u32>{}, 4096,
         [](std::size_t idx) { return static_cast<u32>(idx << 8); });
    LOG_INFO(Render_OpenGL, "GL_DEPTH_COMPONENT16 download test starting");
    test(GetFormatTuple(VideoCore::PixelFormat::D16), std::vector<u16>{}, 256,
         [](std::size_t idx) { return static_cast<u16>(idx); });

    cur_state.Apply();
}

TextureDownloaderES::TextureDownloaderES(bool enable_depth_stencil) {
    vao.Create();
    read_fbo_generic.Create();

    depth32_fbo.Create();
    r32ui_renderbuffer.Create();
    depth16_fbo.Create();
    r16_renderbuffer.Create();

    const auto init_program = [](ConversionShader& converter, std::string_view frag) {
        converter.program.Create(depth_to_color_vert.data(), frag.data());
        converter.lod_location = glGetUniformLocation(converter.program.handle, "lod");
    };

    // xperia64: The depth stencil shader currently uses a GLES extension that is not supported
    // across all devices Reportedly broken on Tegra devices and the Nexus 6P, so enabling it can be
    // toggled
    if (enable_depth_stencil) {
        init_program(d24s8_r32ui_conversion_shader, ds_to_color_frag);
    }

    init_program(d24_r32ui_conversion_shader, depth_to_color_frag);
    init_program(d16_r16_conversion_shader, R"(
out highp float color;

uniform highp sampler2D depth;
uniform int lod;

void main(){
    color = texelFetch(depth, ivec2(gl_FragCoord.xy), lod).x;
}
)");

    sampler.Create();
    glSamplerParameteri(sampler.handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glSamplerParameteri(sampler.handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    auto cur_state = OpenGLState::GetCurState();
    auto state = cur_state;

    state.draw.shader_program = d24s8_r32ui_conversion_shader.program.handle;
    state.draw.draw_framebuffer = depth32_fbo.handle;
    state.renderbuffer = r32ui_renderbuffer.handle;
    state.Apply();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_R32UI, MAX_SIZE, MAX_SIZE);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                              r32ui_renderbuffer.handle);
    glUniform1i(glGetUniformLocation(d24s8_r32ui_conversion_shader.program.handle, "depth"), 1);

    state.draw.draw_framebuffer = depth16_fbo.handle;
    state.renderbuffer = r16_renderbuffer.handle;
    state.Apply();
    glRenderbufferStorage(GL_RENDERBUFFER, GL_R16, MAX_SIZE, MAX_SIZE);
    glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER,
                              r16_renderbuffer.handle);

    cur_state.Apply();
}

GLuint TextureDownloaderES::ConvertDepthToColor(GLuint level, GLenum& format, GLenum& type,
                                                GLint height, GLint width) const {
    ASSERT(width <= MAX_SIZE && height <= MAX_SIZE);
    const OpenGLState cur_state = OpenGLState::GetCurState();
    OpenGLState state;
    state.texture_units[0] = {cur_state.texture_units[0].texture_2d, sampler.handle};
    state.draw.vertex_array = vao.handle;

    OGLTexture texture_view;
    const ConversionShader* converter = nullptr;
    switch (type) {
    case GL_UNSIGNED_SHORT:
        state.draw.draw_framebuffer = depth16_fbo.handle;
        converter = &d16_r16_conversion_shader;
        format = GL_RED;
        break;
    case GL_UNSIGNED_INT:
        state.draw.draw_framebuffer = depth32_fbo.handle;
        converter = &d24_r32ui_conversion_shader;
        format = GL_RED_INTEGER;
        break;
    case GL_UNSIGNED_INT_24_8:
        state.draw.draw_framebuffer = depth32_fbo.handle;
        converter = &d24s8_r32ui_conversion_shader;
        format = GL_RED_INTEGER;
        type = GL_UNSIGNED_INT;
        break;
    default:
        UNREACHABLE_MSG("Destination type not recognized");
    }

    state.draw.shader_program = converter->program.handle;
    state.viewport = {0, 0, width, height};
    state.Apply();
    if (converter->program.handle == d24s8_r32ui_conversion_shader.program.handle) {
        // TODO BreadFish64: the ARM framebuffer reading extension is probably not the most optimal
        // way to do this, search for another solution
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               state.texture_units[0].texture_2d, level);
    }

    glUniform1i(converter->lod_location, level);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    if (texture_view.handle) {
        glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_DEPTH_COMPONENT);
    }
    return state.draw.draw_framebuffer;
}

void TextureDownloaderES::GetTexImage(GLenum target, GLuint level, GLenum format, GLenum type,
                                      GLint height, GLint width, void* pixels) const {
    OpenGLState state = OpenGLState::GetCurState();
    GLuint texture{};
    const GLuint old_read_buffer = state.draw.read_framebuffer;
    switch (target) {
    case GL_TEXTURE_2D:
        texture = state.texture_units[0].texture_2d;
        break;
    case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        texture = state.texture_cube_unit.texture_cube;
        break;
    default:
        UNIMPLEMENTED_MSG("Unexpected target {:x}", target);
    }

    switch (format) {
    case GL_DEPTH_COMPONENT:
    case GL_DEPTH_STENCIL:
        // unfortunately, the accurate way is too slow for release
        return;
        state.draw.read_framebuffer = ConvertDepthToColor(level, format, type, height, width);
        state.Apply();
        break;
    default:
        state.draw.read_framebuffer = read_fbo_generic.handle;
        state.Apply();
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture,
                               level);
    }
    GLenum status = glCheckFramebufferStatus(GL_READ_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOG_DEBUG(Render_OpenGL, "Framebuffer is incomplete, status: {:X}", status);
    }
    glReadPixels(0, 0, width, height, format, type, pixels);

    state.draw.read_framebuffer = old_read_buffer;
    state.Apply();
}

} // namespace OpenGL
