// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/scope_exit.h"
#include "video_core/renderer_opengl/gl_format_reinterpreter.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"

namespace OpenGL {

D24S8toRGBA8::D24S8toRGBA8(bool use_texture_view) : use_texture_view{use_texture_view} {
    constexpr std::string_view cs_source = R"(
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
layout(binding = 0) uniform sampler2D depth;
layout(binding = 1) uniform usampler2D stencil;
layout(rgba8, binding = 2) uniform writeonly image2D color;

uniform mediump ivec2 src_offset;

void main() {
ivec2 tex_coord = src_offset + ivec2(gl_GlobalInvocationID.xy);

highp uint depth_val =
    uint(texelFetch(depth, tex_coord, 0).x * (exp2(32.0) - 1.0));
lowp uint stencil_val = texelFetch(stencil, tex_coord, 0).x;
highp uvec4 components =
    uvec4(stencil_val, (uvec3(depth_val) >> uvec3(24u, 16u, 8u)) & 0x000000FFu);
imageStore(color, tex_coord, vec4(components) / (exp2(8.0) - 1.0));
}

)";
    program.Create(cs_source);
    src_offset_loc = glGetUniformLocation(program.handle, "src_offset");
}

void D24S8toRGBA8::Reinterpret(const Surface& source, VideoCore::Rect2D src_rect,
                               const Surface& dest, VideoCore::Rect2D dst_rect) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state;
    state.texture_units[0].texture_2d = source.texture.handle;

    // Use glTextureView on desktop to avoid intermediate copy
    if (use_texture_view) {
        temp_tex.Create();
        glActiveTexture(GL_TEXTURE1);
        glTextureView(temp_tex.handle, GL_TEXTURE_2D, source.texture.handle, GL_DEPTH24_STENCIL8, 0,
                      1, 0, 1);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    } else {
        temp_tex.Release();
        temp_tex.Create();
        state.texture_units[1].texture_2d = temp_tex.handle;
        state.Apply();
        glActiveTexture(GL_TEXTURE1);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, src_rect.right, src_rect.top);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        temp_rect = src_rect;
    }

    state.texture_units[1].texture_2d = temp_tex.handle;
    state.draw.shader_program = program.handle;
    state.Apply();

    glBindImageTexture(2, dest.texture.handle, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);

    glActiveTexture(GL_TEXTURE1);
    if (!use_texture_view) {
        glCopyImageSubData(source.texture.handle, GL_TEXTURE_2D, 0, src_rect.left, src_rect.bottom,
                           0, temp_tex.handle, GL_TEXTURE_2D, 0, src_rect.left, src_rect.bottom, 0,
                           src_rect.GetWidth(), src_rect.GetHeight(), 1);
    }
    glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_STENCIL_TEXTURE_MODE, GL_STENCIL_INDEX);

    glUniform2i(src_offset_loc, src_rect.left, src_rect.bottom);
    glDispatchCompute(src_rect.GetWidth() / 32, src_rect.GetHeight() / 32, 1);

    if (use_texture_view) {
        temp_tex.Release();
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

RGBA4toRGB5A1::RGBA4toRGB5A1() {
    constexpr std::string_view vs_source = R"(
out vec2 dst_coord;

uniform mediump ivec2 dst_size;

const vec2 vertices[4] =
vec2[4](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));

void main() {
gl_Position = vec4(vertices[gl_VertexID], 0.0, 1.0);
dst_coord = (vertices[gl_VertexID] / 2.0 + 0.5) * vec2(dst_size);
}
)";

    constexpr std::string_view fs_source = R"(
in mediump vec2 dst_coord;

out lowp vec4 frag_color;

uniform lowp sampler2D source;
uniform mediump ivec2 dst_size;
uniform mediump ivec2 src_size;
uniform mediump ivec2 src_offset;

void main() {
mediump ivec2 tex_coord;
if (src_size == dst_size) {
    tex_coord = ivec2(dst_coord);
} else {
    highp int tex_index = int(dst_coord.y) * dst_size.x + int(dst_coord.x);
    mediump int y = tex_index / src_size.x;
    tex_coord = ivec2(tex_index - y * src_size.x, y);
}
tex_coord -= src_offset;

lowp ivec4 rgba4 = ivec4(texelFetch(source, tex_coord, 0) * (exp2(4.0) - 1.0));
lowp ivec3 rgb5 =
    ((rgba4.rgb << ivec3(1, 2, 3)) | (rgba4.gba >> ivec3(3, 2, 1))) & 0x1F;
frag_color = vec4(vec3(rgb5) / (exp2(5.0) - 1.0), rgba4.a & 0x01);
}
)";
    read_fbo.Create();
    draw_fbo.Create();
    program.Create(vs_source.data(), fs_source.data());
    dst_size_loc = glGetUniformLocation(program.handle, "dst_size");
    src_size_loc = glGetUniformLocation(program.handle, "src_size");
    src_offset_loc = glGetUniformLocation(program.handle, "src_offset");
    vao.Create();
}

void RGBA4toRGB5A1::Reinterpret(const Surface& source, VideoCore::Rect2D src_rect,
                                const Surface& dest, VideoCore::Rect2D dst_rect) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state;
    state.texture_units[0].texture_2d = source.texture.handle;
    state.draw.draw_framebuffer = draw_fbo.handle;
    state.draw.shader_program = program.handle;
    state.draw.vertex_array = vao.handle;
    state.viewport = {static_cast<GLint>(dst_rect.left), static_cast<GLint>(dst_rect.bottom),
                      static_cast<GLsizei>(dst_rect.GetWidth()),
                      static_cast<GLsizei>(dst_rect.GetHeight())};
    state.Apply();

    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           dest.texture.handle, 0);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

    glUniform2i(dst_size_loc, dst_rect.GetWidth(), dst_rect.GetHeight());
    glUniform2i(src_size_loc, src_rect.GetWidth(), src_rect.GetHeight());
    glUniform2i(src_offset_loc, src_rect.left, src_rect.bottom);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
}

} // namespace OpenGL
