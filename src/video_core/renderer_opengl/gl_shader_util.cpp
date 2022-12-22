// Copyright 2014 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <array>
#include <string>
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/settings.h"
#include "video_core/renderer_opengl/gl_shader_util.h"

namespace OpenGL {

constexpr const char* ES_VERSION = R"(#version 320 es
#define CITRA_GLES

#if defined(GL_ANDROID_extension_pack_es31a)
#extension GL_ANDROID_extension_pack_es31a : enable
#endif

#if defined(GL_EXT_clip_cull_distance)
#extension GL_EXT_clip_cull_distance : enable
#endif
)";

static void LogShader(GLuint shader, std::string_view code = {}) {
    GLint shader_status{};
    glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_status);
    if (shader_status == GL_FALSE) {
        LOG_ERROR(Render_OpenGL, "Failed to build shader");
    }
    GLint log_length{};
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length == 0) {
        return;
    }
    std::string log(log_length, 0);
    glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    if (shader_status == GL_FALSE) {
        LOG_ERROR(Render_OpenGL, "{}", log);
        if (!code.empty()) {
            LOG_INFO(Render_OpenGL, "\n{}", code);
        }
    } else {
        LOG_WARNING(Render_OpenGL, "{}", log);
    }
}

GLuint LoadShader(std::string_view source, GLenum type) {
    const bool is_gles = Settings::values.graphics_api == Settings::GraphicsAPI::OpenGLES;
    std::string code = is_gles ? ES_VERSION : "#version 430 core\n";
    code += source;

    const GLuint shader_id = glCreateShader(type);
    const GLsizei length = static_cast<GLsizei>(code.size());
    const GLchar* const code_ptr = code.data();

    glShaderSource(shader_id, 1, &code_ptr, &length);
    glCompileShader(shader_id);
    if (Settings::values.renderer_debug) {
        LogShader(shader_id, code);
    }

    return shader_id;
}

GLuint LoadProgram(bool separable, std::span<const GLuint> shaders) {
    const GLuint program_id = glCreateProgram();

    for (const GLuint shader : shaders) {
        if (shader != 0) {
            glAttachShader(program_id, shader);
        }
    }

    if (separable) {
        glProgramParameteri(program_id, GL_PROGRAM_SEPARABLE, GL_TRUE);
    }

    glProgramParameteri(program_id, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
    glLinkProgram(program_id);

    GLint link_status{};
    glGetProgramiv(program_id, GL_LINK_STATUS, &link_status);

    GLint log_length{};
    glGetProgramiv(program_id, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length == 0) {
        return program_id;
    }
    std::string log(log_length, 0);
    glGetProgramInfoLog(program_id, log_length, nullptr, log.data());
    if (link_status == GL_FALSE) {
        LOG_ERROR(Render_OpenGL, "{}", log);
    } else {
        LOG_WARNING(Render_OpenGL, "{}", log);
    }

    ASSERT_MSG(link_status == GL_TRUE, "Shader not linked");

    for (const GLuint shader : shaders) {
        if (shader != 0) {
            glDetachShader(program_id, shader);
        }
    }

    return program_id;
}

} // namespace OpenGL
