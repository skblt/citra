// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/microprofile.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_format_reinterpreter.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"
#include "video_core/renderer_opengl/pica_to_gl.h"

MICROPROFILE_DEFINE(OpenGL_Upload, "OpenGL", "Texture Upload", MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(OpenGL_Download, "OpenGL", "Texture Download", MP_RGB(128, 192, 64));

namespace OpenGL {

using namespace Pica::Texture;
using VideoCore::AllocationId;
using VideoCore::SurfaceBase;

constexpr u32 UPLOAD_BUFFER_SIZE = 32 * 1024 * 1024;
constexpr u32 DOWNLOAD_BUFFER_SIZE = 32 * 1024 * 1024;

constexpr FormatTuple DEFAULT_TUPLE = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

static constexpr std::array DEPTH_TUPLES = {
    FormatTuple{GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT},              // D16
    FormatTuple{}, FormatTuple{GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT}, // D24
    FormatTuple{GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8},              // D24S8
};

static constexpr std::array COLOR_TUPLES = {
    FormatTuple{GL_RGBA8, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8},     // RGBA8
    FormatTuple{GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE},              // RGB8
    FormatTuple{GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    FormatTuple{GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    FormatTuple{GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
};

static constexpr std::array COLOR_TUPLES_OES = {
    FormatTuple{GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE},            // RGBA8
    FormatTuple{GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE},            // RGB8
    FormatTuple{GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    FormatTuple{GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    FormatTuple{GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
};

[[nodiscard]] GLbitfield MakeBufferMask(VideoCore::SurfaceType type) {
    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        return GL_COLOR_BUFFER_BIT;
    case VideoCore::SurfaceType::Depth:
        return GL_DEPTH_BUFFER_BIT;
    case VideoCore::SurfaceType::DepthStencil:
        return GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
    return GL_COLOR_BUFFER_BIT;
}

[[nodiscard]] u32 FboIndex(VideoCore::SurfaceType type) {
    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        return 0;
    case VideoCore::SurfaceType::Depth:
        return 1;
    case VideoCore::SurfaceType::DepthStencil:
        return 2;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
    return 0;
}

[[nodiscard]] GLuint MakeHandle(GLenum target, GLsizei width, GLsizei height,
                                GLsizei levels, GLint internalformat) {
    GLuint handle{};
    glGenTextures(1, &handle);

    glBindTexture(target, handle);
    glTexStorage2D(target, levels, internalformat, width, height);

    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return handle;
}

TextureRuntime::TextureRuntime(Driver& driver)
    : driver{driver}, filterer{Settings::values.texture_filter_name,
                               VideoCore::GetResolutionScaleFactor()},
      upload_buffer{GL_PIXEL_UNPACK_BUFFER, UPLOAD_BUFFER_SIZE},
      download_buffer{GL_PIXEL_PACK_BUFFER, DOWNLOAD_BUFFER_SIZE, true} {

    for (std::size_t i = 0; i < rescale_draw_fbos.size(); ++i) {
        rescale_draw_fbos[i].Create();
        rescale_read_fbos[i].Create();
    }

    auto Register = [this](VideoCore::PixelFormat dest,
                           std::unique_ptr<FormatReinterpreterBase>&& obj) {
        const u32 dst_index = static_cast<u32>(dest);
        return reinterpreters[dst_index].push_back(std::move(obj));
    };

    Register(VideoCore::PixelFormat::RGBA8, std::make_unique<D24S8toRGBA8>(!driver.IsOpenGLES()));
    Register(VideoCore::PixelFormat::RGB5A1, std::make_unique<RGBA4toRGB5A1>());
}

TextureRuntime::~TextureRuntime() = default;

StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    OGLStreamBuffer& buffer = upload ? upload_buffer : download_buffer;
    const auto [data, offset, invalidate] = buffer.Map(size, 4);

    return StagingData{
        .buffer = buffer.GetHandle(),
        .size = size,
        .mapped = std::span{reinterpret_cast<std::byte*>(data), size},
        .buffer_offset = offset,
    };
}

FormatTuple TextureRuntime::NativeFormat(VideoCore::PixelFormat pixel_format) const {
    const auto type = GetFormatType(pixel_format);
    const std::size_t format_index = static_cast<std::size_t>(pixel_format);

    if (type == VideoCore::SurfaceType::Color) {
        ASSERT(format_index < COLOR_TUPLES.size());
        return (driver.IsOpenGLES() ? COLOR_TUPLES_OES : COLOR_TUPLES)[format_index];
    } else if (type == VideoCore::SurfaceType::Depth ||
               type == VideoCore::SurfaceType::DepthStencil) {
        const std::size_t tuple_idx = format_index - 14;
        ASSERT(tuple_idx < DEPTH_TUPLES.size());
        return DEPTH_TUPLES[tuple_idx];
    }

    return DEFAULT_TUPLE;
}

void TextureRuntime::FormatConvert(const Surface& surface, bool upload, std::span<std::byte> source,
                                   std::span<std::byte> dest) {
    const VideoCore::PixelFormat format = surface.pixel_format;
    if (format == VideoCore::PixelFormat::RGBA8 && driver.IsOpenGLES()) {
        return ConvertABGRToRGBA(source, dest);
    } else if (format == VideoCore::PixelFormat::RGB8 && driver.IsOpenGLES()) {
        return ConvertBGRToRGB(source, dest);
    } else {
        // Sometimes the source size might be larger than the destination.
        // This can happen during texture downloads when FromInterval aligns
        // the flush range to scanline boundaries. In that case only copy
        // what we need
        const std::size_t copy_size = std::min(source.size(), dest.size());
        std::memcpy(dest.data(), source.data(), copy_size);
    }
}

bool TextureRuntime::FramebufferClear(Framebuffer& framebuffer, const VideoCore::TextureClear& clear) {
    OpenGLState prev_state = OpenGLState::GetCurState();

    // Setup scissor rectangle according to the clear rectangle
    OpenGLState state{};
    state.scissor.enabled = true;
    state.scissor.x = clear.texture_rect.left;
    state.scissor.y = clear.texture_rect.bottom;
    state.scissor.width = clear.texture_rect.GetWidth();
    state.scissor.height = clear.texture_rect.GetHeight();
    state.draw.draw_framebuffer = framebuffer.Handle();
    state.Apply();

    switch (clear.type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        state.color_mask.red_enabled = true;
        state.color_mask.green_enabled = true;
        state.color_mask.blue_enabled = true;
        state.color_mask.alpha_enabled = true;
        state.Apply();

        glClearBufferfv(GL_COLOR, 0, clear.value.color.AsArray());
        break;
    case VideoCore::SurfaceType::Depth:
        state.depth.write_mask = GL_TRUE;
        state.Apply();

        glClearBufferfv(GL_DEPTH, 0, &clear.value.depth);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        state.depth.write_mask = GL_TRUE;
        state.stencil.write_mask = -1;
        state.Apply();

        glClearBufferfi(GL_DEPTH_STENCIL, 0, clear.value.depth, clear.value.stencil);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    prev_state.Apply();
    return true;
}

bool TextureRuntime::SurfaceClear(Surface& surface, const VideoCore::TextureClear& clear) {
    return false;
    if (driver.IsOpenGLES()) {
        return false;
    }

    glClearTexSubImage(surface.Handle(), clear.texture_level, clear.texture_rect.left,
                       clear.texture_rect.bottom, 0, clear.texture_rect.GetWidth(),
                       clear.texture_rect.GetHeight(), 1, GL_RGBA, GL_UNSIGNED_BYTE,
                       clear.value.color.AsArray());
    return true;
}

bool TextureRuntime::FramebufferBlit(const Framebuffer& source, const Framebuffer& dest,
                                     const VideoCore::TextureBlit& blit) {
    ASSERT(source.BufferMask() == dest.BufferMask());

    OpenGLState prev_state = OpenGLState::GetCurState();

    OpenGLState state{};
    state.draw.read_framebuffer = source.Handle();
    state.draw.draw_framebuffer = dest.Handle();
    state.scissor.enabled = false;
    state.Apply();

    // TODO (wwylele): use GL_NEAREST for shadow map texture
    // Note: shadow map is treated as RGBA8 format in PICA, as well as in the rasterizer cache, but
    // doing linear intepolation componentwise would cause incorrect value. However, for a
    // well-programmed game this code path should be rarely executed for shadow map with
    // inconsistent scale.
    const GLbitfield buffer_mask = source.BufferMask();
    const GLenum filter = buffer_mask == GL_COLOR_BUFFER_BIT ? GL_LINEAR : GL_NEAREST;
    glBlitFramebuffer(blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right,
                      blit.src_rect.top, blit.dst_rect.left, blit.dst_rect.bottom,
                      blit.dst_rect.right, blit.dst_rect.top, buffer_mask, filter);

    prev_state.Apply();
    return true;
}

const ReinterpreterList& TextureRuntime::GetPossibleReinterpretations(
    VideoCore::PixelFormat dest_format) const {
    return reinterpreters[static_cast<u32>(dest_format)];
}

bool TextureRuntime::NeedsConvertion(VideoCore::PixelFormat format) const {
    return driver.IsOpenGLES() &&
           (format == VideoCore::PixelFormat::RGB8 || format == VideoCore::PixelFormat::RGBA8);
}

Allocation::Allocation(TextureRuntime&) {}

Allocation::Allocation(TextureRuntime& runtime, const VideoCore::SurfaceParams& params) {
    const GLsizei num_levels = params.levels;
    const GLenum target = params.texture_type == VideoCore::TextureType::Texture2D ? GL_TEXTURE_2D
                                                                                   : GL_TEXTURE_CUBE_MAP;

    GLuint old_tex = OpenGLState::GetCurState().texture_units[0].texture_2d;
    glActiveTexture(GL_TEXTURE0);

    tuple = runtime.NativeFormat(params.pixel_format);
    handles[0] = MakeHandle(target, params.GetScaledWidth(), params.GetScaledHeight(),
                            num_levels, tuple.internal_format);

    if (params.res_scale != 1) {
        handles[1] = MakeHandle(target, params.width, params.height,
                                num_levels, tuple.internal_format);
    } else {
        handles[1] = handles[0];
    }

    glBindTexture(GL_TEXTURE_2D, old_tex);
}

Allocation::~Allocation() {
    if (handles[1] && handles[1] != handles[0]) {
        glDeleteTextures(1, &handles[1]);
    }
    if (handles[0]) {
        glDeleteTextures(1, &handles[0]);
    }
}

Surface::Surface(VideoCore::SurfaceParams params) : SurfaceBase{params} {}

Surface::Surface(TextureRuntime& runtime, Allocation&& alloc, SurfaceParams params)
    : SurfaceBase{params}, driver{&runtime.GetDriver()}, runtime{&runtime},
      alloc{std::move(alloc)} {}

Surface::~Surface() = default;

void Surface::Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging) {
    MICROPROFILE_SCOPE(OpenGL_Upload);

    OpenGLState prev_state = OpenGLState::GetCurState();

    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(stride));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, staging.buffer);

    const FormatTuple& tuple = runtime->NativeFormat(pixel_format);
    const void* const offset = reinterpret_cast<const void*>(staging.buffer_offset);
    const GLuint handle = Handle(true);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, handle);

    glTexSubImage2D(GL_TEXTURE_2D, upload.texture_level, upload.texture_rect.left,
                    upload.texture_rect.bottom, upload.texture_rect.GetWidth(),
                    upload.texture_rect.GetHeight(), tuple.format, tuple.type, offset);

    runtime->upload_buffer.Unmap(staging.size);

    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = upload.texture_level,
            .dst_level = upload.texture_level,
            .src_layer = 0,
            .dst_layer = 0,
            .src_rect = upload.texture_rect,
            .dst_rect = upload.texture_rect * res_scale,
        };

        BlitScale(blit, true);
    }

    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    prev_state.Apply();
}

void Surface::Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging) {
    MICROPROFILE_SCOPE(OpenGL_Download);

    const bool is_depth = type == VideoCore::SurfaceType::Depth ||
                          type == VideoCore::SurfaceType::DepthStencil;

    // OpenGL ES does not support depth/stencil downloads using glReadPixels nor
    // glGetTexImage. Ignore the download for now until a proper method is implemented
    if (is_depth && driver->IsOpenGLES()) [[unlikely]] {
        return;
    }

    OpenGLState prev_state = OpenGLState::GetCurState();

    glPixelStorei(GL_PACK_ROW_LENGTH, static_cast<GLint>(stride));
    glBindBuffer(GL_PIXEL_PACK_BUFFER, staging.buffer);

    runtime->download_buffer.Unmap(staging.size);

    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = download.texture_level,
            .dst_level = download.texture_level,
            .src_layer = 0,
            .dst_layer = 0,
            .src_rect = download.texture_rect * res_scale,
            .dst_rect = download.texture_rect,
        };

        BlitScale(blit, false);
    }

    const FormatTuple& tuple = runtime->NativeFormat(pixel_format);
    const GLuint handle = Handle(true);
    void* const offset = reinterpret_cast<void*>(staging.buffer_offset);

    // For full texture downloads (most common case) prefer glGetTexImage
    // to avoid using an FBO
    if (download.texture_rect == GetRect() && !driver->IsOpenGLES()) {
        glBindTexture(GL_TEXTURE_2D, handle);
        glGetTexImage(GL_TEXTURE_2D, download.texture_level, tuple.format, tuple.type, offset);
    } else {
        BindFramebuffer(handle, GL_READ_FRAMEBUFFER, download.texture_level);
        glReadPixels(download.texture_rect.left, download.texture_rect.bottom,
                     download.texture_rect.GetWidth(), download.texture_rect.GetHeight(),
                     tuple.format, tuple.type, offset);
    }

    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    prev_state.Apply();
}

void Surface::BlitScale(const VideoCore::TextureBlit& blit, bool up_scale) {
    OpenGLState prev_state = OpenGLState::GetCurState();

    BindFramebuffer(Handle(up_scale), GL_READ_FRAMEBUFFER, blit.src_level);
    BindFramebuffer(Handle(!up_scale), GL_DRAW_FRAMEBUFFER, blit.dst_level);

    const GLenum buffer_mask = MakeBufferMask(type);
    const GLenum filter = buffer_mask == GL_COLOR_BUFFER_BIT ? GL_LINEAR : GL_NEAREST;
    glBlitFramebuffer(blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right,
                      blit.src_rect.top, blit.dst_rect.left, blit.dst_rect.bottom,
                      blit.dst_rect.right, blit.dst_rect.top, buffer_mask, filter);

    prev_state.Apply();
}

void Surface::BindFramebuffer(GLuint handle, GLenum target, GLsizei level) {
    const u32 fbo_index = FboIndex(type);
    const auto& framebuffers = target == GL_READ_FRAMEBUFFER ? runtime->rescale_read_fbos
                                                             : runtime->rescale_draw_fbos;

    glBindFramebuffer(target, framebuffers[fbo_index].handle);

    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, handle, level);
        glFramebufferTexture2D(target, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
        break;
    case VideoCore::SurfaceType::Depth:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(target, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
                               handle, level);
        glFramebufferTexture2D(target, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        glFramebufferTexture2D(target, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(target, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, handle, level);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }
}

Framebuffer::Framebuffer(TextureRuntime&, Surface* color, Surface* depth,
                         VideoCore::RenderTargets) {
    framebuffer.Create();

    const GLuint prev_fbo = OpenGLState::GetCurState().draw.draw_framebuffer;

    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer.handle);

    if (color) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color->Handle(), 0);
        buffer_mask = GL_COLOR_BUFFER_BIT;
    }

    if (depth) {
        const VideoCore::SurfaceType type = depth->type;
        switch (type) {
        case VideoCore::SurfaceType::Depth:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth->Handle(), 0);
            buffer_mask = GL_DEPTH_BUFFER_BIT;
            break;
        case VideoCore::SurfaceType::DepthStencil:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                                   GL_TEXTURE_2D, depth->Handle(), 0);
            buffer_mask = GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT;
            break;
        default:
            UNREACHABLE_MSG("Depth surface with incompatible surface type!");
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
}

Framebuffer::~Framebuffer() = default;

Sampler::Sampler(TextureRuntime& runtime, VideoCore::SamplerParams params) {
    const GLenum mag_filter = PicaToGL::TextureMagFilterMode(params.min_filter);
    const GLenum min_filter = PicaToGL::TextureMinFilterMode(params.min_filter, params.mip_filter);
    const GLenum wrap_s = PicaToGL::WrapMode(params.wrap_s);
    const GLenum wrap_t = PicaToGL::WrapMode(params.wrap_t);
    const Common::Vec4f gl_color = PicaToGL::ColorRGBA8(params.border_color);
    const float lod_min = params.lod_min;
    const float lod_max = params.lod_max;
    const float lod_bias = params.lod_bias / 256.f;

    sampler.Create();

    const GLuint handle = sampler.handle;
    glSamplerParameteri(handle, GL_TEXTURE_MAG_FILTER, mag_filter);
    glSamplerParameteri(handle, GL_TEXTURE_MIN_FILTER, min_filter);

    glSamplerParameteri(handle, GL_TEXTURE_WRAP_S, wrap_s);
    glSamplerParameteri(handle, GL_TEXTURE_WRAP_T, wrap_t);

    glSamplerParameterfv(handle, GL_TEXTURE_BORDER_COLOR, gl_color.AsArray());

    glSamplerParameterf(handle, GL_TEXTURE_MIN_LOD, lod_min);
    glSamplerParameterf(handle, GL_TEXTURE_MAX_LOD, lod_max);

    const Driver& driver = runtime.GetDriver();
    if (!driver.IsOpenGLES()) {
        glSamplerParameterf(handle, GL_TEXTURE_LOD_BIAS, lod_bias);
    }
}

Sampler::~Sampler() = default;

} // namespace OpenGL
