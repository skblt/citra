// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/scope_exit.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_state.h"

namespace OpenGL {

constexpr FormatTuple DEFAULT_TUPLE = {GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE};

static constexpr std::array DEPTH_TUPLES = {
    FormatTuple{GL_DEPTH_COMPONENT16, GL_DEPTH_COMPONENT, GL_UNSIGNED_SHORT}, // D16
    FormatTuple{},
    FormatTuple{GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT},   // D24
    FormatTuple{GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8}, // D24S8
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
    FormatTuple{GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE},              // RGB8
    FormatTuple{GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1}, // RGB5A1
    FormatTuple{GL_RGB565, GL_RGB, GL_UNSIGNED_SHORT_5_6_5},     // RGB565
    FormatTuple{GL_RGBA4, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4},   // RGBA4
};

GLbitfield MakeBufferMask(VideoCore::SurfaceType type) {
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

TextureRuntime::TextureRuntime(Driver& driver) : driver(driver) {
    read_fbo.Create();
    draw_fbo.Create();
}

const StagingBuffer& TextureRuntime::FindStaging(u32 size, bool upload) {
    const GLenum target = upload ? GL_PIXEL_UNPACK_BUFFER : GL_PIXEL_PACK_BUFFER;
    const GLbitfield access = upload ? GL_MAP_WRITE_BIT : GL_MAP_READ_BIT;
    auto& search = upload ? upload_buffers : download_buffers;

    // Attempt to find a free buffer that fits the requested data
    for (auto it = search.lower_bound({.size = size}); it != search.end(); it++) {
        if (!upload || it->IsFree()) {
            return *it;
        }
    }

    OGLBuffer buffer{};
    buffer.Create();

    glBindBuffer(target, buffer.handle);

    // Allocate a new buffer and map the data to the host
    std::byte* data = nullptr;
    if (driver.IsOpenGLES() && driver.HasExtBufferStorage()) {
        const GLbitfield storage = upload ? GL_MAP_WRITE_BIT : GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT_EXT;
        glBufferStorageEXT(target, size, nullptr, storage | GL_MAP_PERSISTENT_BIT_EXT |
                                                            GL_MAP_COHERENT_BIT_EXT);
        data = reinterpret_cast<std::byte*>(glMapBufferRange(target, 0, size, access | GL_MAP_PERSISTENT_BIT_EXT |
                                                          GL_MAP_COHERENT_BIT_EXT));
    } else if (driver.HasArbBufferStorage()) {
        const GLbitfield storage = upload ? GL_MAP_WRITE_BIT : GL_MAP_READ_BIT | GL_CLIENT_STORAGE_BIT;
        glBufferStorage(target, size, nullptr, storage | GL_MAP_PERSISTENT_BIT |
                                                         GL_MAP_COHERENT_BIT);
        data = reinterpret_cast<std::byte*>(glMapBufferRange(target, 0, size, access | GL_MAP_PERSISTENT_BIT |
                                                          GL_MAP_COHERENT_BIT));
    } else {
        UNIMPLEMENTED();
    }

    glBindBuffer(target, 0);

    StagingBuffer staging = {
        .buffer = std::move(buffer),
        .mapped = std::span{data, size},
        .size = size
    };

    const auto& it = search.emplace(std::move(staging));
    return *it;
}

const FormatTuple& TextureRuntime::GetFormatTuple(VideoCore::PixelFormat pixel_format) {
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

OGLTexture TextureRuntime::Allocate2D(u32 width, u32 height, VideoCore::PixelFormat format) {
    const auto& tuple = GetFormatTuple(format);
    auto recycled_tex = texture2d_recycler.find({format, width, height});
    if (recycled_tex != texture2d_recycler.end()) {
        OGLTexture texture = std::move(recycled_tex->second);
        texture2d_recycler.erase(recycled_tex);
        return texture;
    }

    // Allocate the 2D texture
    OGLTexture texture{};
    texture.Create();
    texture.Allocate(GL_TEXTURE_2D, std::bit_width(std::max(width, height)),
                     tuple.internal_format, width, height);

    return texture;
}

OGLTexture TextureRuntime::AllocateCubeMap(u32 width, VideoCore::PixelFormat format) {
    const auto& tuple = GetFormatTuple(format);

    // Allocate the cube texture
    OGLTexture texture{};
    texture.Create();
    texture.Allocate(GL_TEXTURE_CUBE_MAP, std::bit_width(width),
                     tuple.internal_format, width, width);

    return texture;
}

void TextureRuntime::ReadTexture(OGLTexture& texture, const VideoCore::BufferTextureCopy& copy,
                                 VideoCore::PixelFormat format) {

    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state{};
    state.ResetTexture(texture.handle);
    state.draw.read_framebuffer = read_fbo.handle;
    state.Apply();

    switch (copy.surface_type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture.handle,
                               copy.texture_level);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                               0);
        break;
    case VideoCore::SurfaceType::Depth:
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture.handle,
                               copy.texture_level);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               texture.handle, copy.texture_level);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    const FormatTuple& tuple = GetFormatTuple(format);
    glReadPixels(copy.texture_rect.left, copy.texture_rect.bottom,
                 copy.texture_rect.GetWidth(), copy.texture_rect.GetHeight(),
                 tuple.format, tuple.type, reinterpret_cast<void*>(copy.buffer_offset));
}

bool TextureRuntime::ClearTexture(OGLTexture& texture, const VideoCore::TextureClear& clear,
                                  VideoCore::ClearValue value) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    // Setup scissor rectangle according to the clear rectangle
    OpenGLState state{};
    state.scissor.enabled = true;
    state.scissor.x = clear.texture_rect.left;
    state.scissor.y = clear.texture_rect.bottom;
    state.scissor.width = clear.texture_rect.GetWidth();
    state.scissor.height = clear.texture_rect.GetHeight();
    state.draw.draw_framebuffer = draw_fbo.handle;
    state.Apply();

    switch (clear.surface_type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture.handle,
                               clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                               0);

        state.color_mask.red_enabled = true;
        state.color_mask.green_enabled = true;
        state.color_mask.blue_enabled = true;
        state.color_mask.alpha_enabled = true;
        state.Apply();

        glClearBufferfv(GL_COLOR, 0, value.color.AsArray());
        break;
    case VideoCore::SurfaceType::Depth:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture.handle,
                               clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

        state.depth.write_mask = GL_TRUE;
        state.Apply();

        glClearBufferfv(GL_DEPTH, 0, &value.depth);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               texture.handle, clear.texture_level);

        state.depth.write_mask = GL_TRUE;
        state.stencil.write_mask = -1;
        state.Apply();

        glClearBufferfi(GL_DEPTH_STENCIL, 0, value.depth, value.stencil);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    return true;
}

bool TextureRuntime::CopyTextures(OGLTexture& source, OGLTexture& dest, const VideoCore::TextureCopy& copy) {
    return true;
}

bool TextureRuntime::BlitTextures(OGLTexture& source, OGLTexture& dest, const VideoCore::TextureBlit& blit) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state{};
    state.draw.read_framebuffer = read_fbo.handle;
    state.draw.draw_framebuffer = draw_fbo.handle;
    state.Apply();

    auto BindAttachment = [&blit, &source, &dest](GLenum attachment, u32 src_tex, u32 dst_tex) -> void {
        const GLenum src_target = source.target == GL_TEXTURE_CUBE_MAP ?
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + blit.src_layer : source.target;
        const GLenum dst_target = dest.target == GL_TEXTURE_CUBE_MAP ?
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + blit.dst_layer : dest.target;

        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, attachment, src_target, src_tex, blit.src_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attachment, dst_target, dst_tex, blit.dst_level);
    };

    switch (blit.surface_type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        // Bind only color
        BindAttachment(GL_COLOR_ATTACHMENT0, source.handle, dest.handle);
        BindAttachment(GL_DEPTH_STENCIL_ATTACHMENT, 0, 0);
        break;
    case VideoCore::SurfaceType::Depth:
        // Bind only depth
        BindAttachment(GL_COLOR_ATTACHMENT0, 0, 0);
        BindAttachment(GL_DEPTH_ATTACHMENT, source.handle, dest.handle);
        BindAttachment(GL_STENCIL_ATTACHMENT, 0, 0);
        break;
    case VideoCore::SurfaceType::DepthStencil:
        // Bind to combined depth + stencil
        BindAttachment(GL_COLOR_ATTACHMENT0, 0, 0);
        BindAttachment(GL_DEPTH_STENCIL_ATTACHMENT, source.handle, dest.handle);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    // TODO (wwylele): use GL_NEAREST for shadow map texture
    // Note: shadow map is treated as RGBA8 format in PICA, as well as in the rasterizer cache, but
    // doing linear intepolation componentwise would cause incorrect value. However, for a
    // well-programmed game this code path should be rarely executed for shadow map with
    // inconsistent scale.
    const GLbitfield buffer_mask = MakeBufferMask(blit.surface_type);
    const GLenum filter = buffer_mask == GL_COLOR_BUFFER_BIT ? GL_LINEAR : GL_NEAREST;
    glBlitFramebuffer(blit.src_rect.left, blit.src_rect.bottom, blit.src_rect.right, blit.src_rect.top,
                      blit.dst_rect.left, blit.dst_rect.bottom, blit.dst_rect.right, blit.dst_rect.top,
                      buffer_mask, filter);

    return true;
}

void TextureRuntime::GenerateMipmaps(OGLTexture& texture, u32 max_level) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state{};
    state.texture_units[0].texture_2d = texture.handle;
    state.Apply();

    glActiveTexture(GL_TEXTURE0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, max_level);

    glGenerateMipmap(GL_TEXTURE_2D);
}

MICROPROFILE_DEFINE(RasterizerCache_TextureUL, "RasterizerCache", "Texture Upload", MP_RGB(128, 192, 64));
void CachedSurface::UploadTexture(Common::Rectangle<u32> rect, const StagingBuffer& staging) {
    MICROPROFILE_SCOPE(RasterizerCache_TextureUL);

    const FormatTuple& tuple = runtime.GetFormatTuple(pixel_format);

    // Load data from memory to the surface
    GLint x0 = static_cast<GLint>(rect.left);
    GLint y0 = static_cast<GLint>(rect.bottom);
    std::size_t buffer_offset = (y0 * stride + x0) * GetBytesPerPixel(pixel_format);

    GLuint target_tex = texture.handle;

    // If not 1x scale, create 1x texture that we will blit from to replace texture subrect in surface
    OGLTexture unscaled_tex;
    if (res_scale != 1) {
        x0 = 0;
        y0 = 0;

        unscaled_tex = runtime.Allocate2D(rect.GetWidth(), rect.GetHeight(), pixel_format);
        target_tex = unscaled_tex.handle;
    }

    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, target_tex);

    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(stride));

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, staging.buffer.handle);

    glTexSubImage2D(GL_TEXTURE_2D, 0, x0, y0, static_cast<GLsizei>(rect.GetWidth()),
                    static_cast<GLsizei>(rect.GetHeight()), tuple.format, tuple.type,
                    reinterpret_cast<void*>(buffer_offset));

    // Lock the staging buffer until glTexSubImage completes
    staging.Lock();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    if (res_scale != 1) {
        auto scaled_rect = rect;
        scaled_rect.left *= res_scale;
        scaled_rect.top *= res_scale;
        scaled_rect.right *= res_scale;
        scaled_rect.bottom *= res_scale;

        const Common::Rectangle<u32> from_rect{0, rect.GetHeight(), rect.GetWidth(), 0};
        /*if (!owner.texture_filterer->Filter(unscaled_tex, from_rect, texture, scaled_rect, type)) {
            const TextureBlit texture_blit = {
                .surface_type = type,
                .src_level = 0,
                .dst_level = 0,
                .src_region = Region2D{
                    .start = {0, 0},
                    .end = {width, height}
                },
                .dst_region = Region2D{
                    .start = {rect.left, rect.bottom},
                    .end = {rect.right, rect.top}
                }
            };

            runtime.BlitTextures(unscaled_tex, texture, texture_blit);
        }*/
    }

    InvalidateAllWatcher();
}

MICROPROFILE_DEFINE(RasterizerCache_TextureDL, "RasterizerCache", "Texture Download", MP_RGB(128, 192, 64));
void CachedSurface::DownloadTexture(Common::Rectangle<u32> rect, const StagingBuffer& staging) {
    MICROPROFILE_SCOPE(RasterizerCache_TextureDL);

    const FormatTuple& tuple = runtime.GetFormatTuple(pixel_format);
    const u32 buffer_offset = (rect.bottom * stride + rect.left) * GetBytesPerPixel(pixel_format);

    OpenGLState state = OpenGLState::GetCurState();
    OpenGLState prev_state = state;
    SCOPE_EXIT({ prev_state.Apply(); });

    // Ensure no bad interactions with GL_PACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, static_cast<GLint>(stride));
    glBindBuffer(GL_PIXEL_PACK_BUFFER, staging.buffer.handle);

    // If not 1x scale, blit scaled texture to a new 1x texture and use that to flush
    if (res_scale != 1) {
        auto scaled_rect = rect;
        scaled_rect.left *= res_scale;
        scaled_rect.top *= res_scale;
        scaled_rect.right *= res_scale;
        scaled_rect.bottom *= res_scale;

        OGLTexture unscaled_tex = runtime.Allocate2D(rect.GetWidth(), rect.GetHeight(), pixel_format);

        const VideoCore::TextureBlit texture_blit = {
            .surface_type = type,
            .src_level = 0,
            .dst_level = 0,
            .src_rect = scaled_rect,
            .dst_rect = VideoCore::Rect2D{0, rect.GetHeight(), rect.GetWidth(), 0}
        };

        // Blit scaled texture to the unscaled one
        runtime.BlitTextures(texture, unscaled_tex, texture_blit);

        state.texture_units[0].texture_2d = unscaled_tex.handle;
        state.Apply();

        glActiveTexture(GL_TEXTURE0);

        /*if (GLES) {
            owner.texture_downloader_es->GetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type,
                                                     rect.GetHeight(), rect.GetWidth(),
                                                     reinterpret_cast<void*>(buffer_offset));
        } else {
            glGetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type, reinterpret_cast<void*>(buffer_offset));
        }*/
        glGetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type, reinterpret_cast<void*>(buffer_offset));
    } else {
        const u32 download_size = width * height * GetBytesPerPixel(pixel_format);
        const VideoCore::BufferTextureCopy texture_download = {
            .buffer_offset = buffer_offset,
            .buffer_size = download_size,
            .buffer_row_length = stride,
            .buffer_height = height,
            .surface_type = type,
            .texture_rect = rect,
            .texture_level = 0
        };

        runtime.ReadTexture(texture, texture_download, pixel_format);
    }

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
}

void CachedSurface::Scale() {

}

} // namespace OpenGL
