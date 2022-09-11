// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/scope_exit.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/rasterizer_cache/texture_runtime.h"
#include "video_core/renderer_opengl/gl_driver.h"
#include "video_core/renderer_opengl/gl_state.h"

namespace OpenGL {

GLbitfield MakeBufferMask(SurfaceType type) {
    switch (type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill:
        return GL_COLOR_BUFFER_BIT;
    case SurfaceType::Depth:
        return GL_DEPTH_BUFFER_BIT;
    case SurfaceType::DepthStencil:
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

void TextureRuntime::ReadTexture(OGLTexture& texture, const BufferTextureCopy& copy,
                                 PixelFormat format, std::span<std::byte> pixels) {

    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    OpenGLState state{};
    state.ResetTexture(texture.handle);
    state.draw.read_framebuffer = read_fbo.handle;
    state.Apply();

    switch (copy.surface_type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill:
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture.handle,
                               copy.texture_level);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0,
                               0);
        break;
    case SurfaceType::Depth:
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture.handle,
                               copy.texture_level);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
        break;
    case SurfaceType::DepthStencil:
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D,
                               texture.handle, copy.texture_level);
        break;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    // TODO: Use PBO here
    const FormatTuple& tuple = GetFormatTuple(format);
    glReadPixels(copy.texture_offset.x, copy.texture_offset.y,
                 copy.texture_offset.x + copy.texture_extent.width,
                 copy.texture_offset.y + copy.texture_extent.height,
                 tuple.format, tuple.type, pixels.data() + copy.buffer_offset);
}

bool TextureRuntime::ClearTexture(OGLTexture& texture, const TextureClear& clear, ClearValue value) {
    OpenGLState prev_state = OpenGLState::GetCurState();
    SCOPE_EXIT({ prev_state.Apply(); });

    // Setup scissor rectangle according to the clear rectangle
    OpenGLState state{};
    state.scissor.enabled = true;
    state.scissor.x = clear.rect.offset.x;
    state.scissor.y = clear.rect.offset.y;
    state.scissor.width = clear.rect.extent.width;
    state.scissor.height = clear.rect.extent.height;
    state.draw.draw_framebuffer = draw_fbo.handle;
    state.Apply();

    switch (clear.surface_type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill:
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
    case SurfaceType::Depth:
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture.handle,
                               clear.texture_level);
        glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_2D, 0, 0);

        state.depth.write_mask = GL_TRUE;
        state.Apply();

        glClearBufferfv(GL_DEPTH, 0, &value.depth);
        break;
    case SurfaceType::DepthStencil:
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

bool TextureRuntime::CopyTextures(OGLTexture& source, OGLTexture& dest, const TextureCopy& copy) {
    return true;
}

bool TextureRuntime::BlitTextures(OGLTexture& source, OGLTexture& dest, const TextureBlit& blit) {
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
    case SurfaceType::Color:
    case SurfaceType::Texture:
    case SurfaceType::Fill:
        // Bind only color
        BindAttachment(GL_COLOR_ATTACHMENT0, source.handle, dest.handle);
        BindAttachment(GL_DEPTH_STENCIL_ATTACHMENT, 0, 0);
        break;
    case SurfaceType::Depth:
        // Bind only depth
        BindAttachment(GL_COLOR_ATTACHMENT0, 0, 0);
        BindAttachment(GL_DEPTH_ATTACHMENT, source.handle, dest.handle);
        BindAttachment(GL_STENCIL_ATTACHMENT, 0, 0);
        break;
    case SurfaceType::DepthStencil:
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
    glBlitFramebuffer(blit.src_region.start.x, blit.src_region.start.y,
                      blit.src_region.end.x, blit.src_region.end.y,
                      blit.dst_region.start.x, blit.dst_region.start.y,
                      blit.dst_region.end.x, blit.dst_region.end.y,
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

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    StagingBuffer staging = {
        .buffer = std::move(buffer),
        .mapped = std::span{data, size},
        .size = size
    };

    const auto& it = search.emplace(std::move(staging));
    return *it;
}

} // namespace OpenGL
