// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/microprofile.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "video_core/rasterizer_cache/cached_surface.h"
#include "video_core/rasterizer_cache/rasterizer_cache.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_vars.h"
#include "video_core/renderer_opengl/texture_downloader_es.h"
#include "video_core/renderer_opengl/texture_filters/texture_filterer.h"
#include "video_core/video_core.h"

namespace OpenGL {

CachedSurface::~CachedSurface() {
    if (texture.handle) {
        const auto tag = HostTextureTag{pixel_format, GetScaledWidth(), GetScaledHeight()};
        owner.host_texture_recycler.emplace(tag, std::move(texture));
    }
}

MICROPROFILE_DEFINE(RasterizerCache_SurfaceLoad, "RasterizerCache", "Surface Load", MP_RGB(128, 192, 64));
void CachedSurface::LoadGLBuffer(PAddr load_start, PAddr load_end) {
    DEBUG_ASSERT(load_start >= addr && load_end <= end);

    auto source_ptr = VideoCore::g_memory->GetPhysicalRef(load_start);
    if (!source_ptr) [[unlikely]] {
        return;
    }

    const auto start_offset = load_start - addr;
    const auto upload_data = source_ptr.GetWriteBytes(load_end - load_start);
    const auto upload_size = static_cast<u32>(upload_data.size());

    if (gl_buffer.empty()) {
        gl_buffer.resize(width * height * GetBytesPerPixel(pixel_format));
    }

    MICROPROFILE_SCOPE(RasterizerCache_SurfaceLoad);

    if (!is_tiled) {
        ASSERT(type == SurfaceType::Color);

        const auto dest_buffer = std::span{gl_buffer.begin() + start_offset, upload_size};
        if (pixel_format == PixelFormat::RGBA8 && GLES) {
            Pica::Texture::ConvertABGRToRGBA(upload_data, dest_buffer);
        } else if (pixel_format == PixelFormat::RGB8 && GLES) {
            Pica::Texture::ConvertBGRToRGB(upload_data, dest_buffer);
        } else {
            std::memcpy(dest_buffer.data(), upload_data.data(), upload_size);
        }
    } else {
        UnswizzleTexture(*this, start_offset, upload_data, gl_buffer);
    }
}

MICROPROFILE_DEFINE(RasterizerCache_SurfaceFlush, "RasterizerCache", "Surface Flush", MP_RGB(128, 192, 64));
void CachedSurface::FlushGLBuffer(PAddr flush_start, PAddr flush_end) {
    DEBUG_ASSERT(flush_start >= addr && flush_end <= end);

    auto dest_ptr = VideoCore::g_memory->GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const auto start_offset = flush_start - addr;
    const auto download_dest = dest_ptr.GetWriteBytes(flush_end - flush_start);
    const auto download_size = static_cast<u32>(download_dest.size());

    MICROPROFILE_SCOPE(RasterizerCache_SurfaceFlush);

    if (type == SurfaceType::Fill) {
        const u32 coarse_start_offset = start_offset - (start_offset % fill_size);
        const u32 backup_bytes = start_offset % fill_size;
        std::array<u8, 4> backup_data;
        if (backup_bytes) {
            std::memcpy(backup_data.data(), &dest_ptr[coarse_start_offset], backup_bytes);
        }

        for (u32 offset = coarse_start_offset; offset < download_size; offset += fill_size) {
            std::memcpy(&dest_ptr[offset], &fill_data[0],
                        std::min(fill_size, download_size - offset));
        }

        if (backup_bytes)
            std::memcpy(&dest_ptr[coarse_start_offset], &backup_data[0], backup_bytes);
    } else if (!is_tiled) {
        ASSERT(type == SurfaceType::Color);

        const auto download_data = std::span{gl_buffer.begin() + start_offset, download_size};
        if (pixel_format == PixelFormat::RGBA8 && GLES) {
            Pica::Texture::ConvertABGRToRGBA(gl_buffer, download_data);
        } else if (pixel_format == PixelFormat::RGB8 && GLES) {
            Pica::Texture::ConvertBGRToRGB(gl_buffer, download_data);
        } else {
            std::memcpy(download_dest.data(), download_data.data(), download_size);
        }
    } else {
        SwizzleTexture(*this, start_offset, gl_buffer, download_dest);
    }
}

MICROPROFILE_DEFINE(RasterizerCache_TextureUL, "RasterizerCache", "Texture Upload", MP_RGB(128, 192, 64));
void CachedSurface::UploadGLTexture(Common::Rectangle<u32> rect) {
    MICROPROFILE_SCOPE(RasterizerCache_TextureUL);

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

        unscaled_tex = owner.AllocateSurfaceTexture(pixel_format, rect.GetWidth(), rect.GetHeight());
        target_tex = unscaled_tex.handle;
    }

    OpenGLState cur_state = OpenGLState::GetCurState();

    GLuint old_tex = cur_state.texture_units[0].texture_2d;
    cur_state.texture_units[0].texture_2d = target_tex;
    cur_state.Apply();

    const FormatTuple& tuple = GetFormatTuple(pixel_format);

    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, static_cast<GLint>(stride));

    glActiveTexture(GL_TEXTURE0);

    glTexSubImage2D(GL_TEXTURE_2D, 0, x0, y0, static_cast<GLsizei>(rect.GetWidth()),
                    static_cast<GLsizei>(rect.GetHeight()), tuple.format, tuple.type,
                    &gl_buffer[buffer_offset]);

    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);

    cur_state.texture_units[0].texture_2d = old_tex;
    cur_state.Apply();

    if (res_scale != 1) {
        auto scaled_rect = rect;
        scaled_rect.left *= res_scale;
        scaled_rect.top *= res_scale;
        scaled_rect.right *= res_scale;
        scaled_rect.bottom *= res_scale;

        const Common::Rectangle<u32> from_rect{0, rect.GetHeight(), rect.GetWidth(), 0};
        if (!owner.texture_filterer->Filter(unscaled_tex, from_rect, texture, scaled_rect, type)) {
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
        }
    }

    InvalidateAllWatcher();
}

MICROPROFILE_DEFINE(RasterizerCache_TextureDL, "RasterizerCache", "Texture Download", MP_RGB(128, 192, 64));
void CachedSurface::DownloadGLTexture(const Common::Rectangle<u32>& rect) {
    MICROPROFILE_SCOPE(RasterizerCache_TextureDL);

    OpenGLState state = OpenGLState::GetCurState();
    OpenGLState prev_state = state;
    SCOPE_EXIT({ prev_state.Apply(); });

    const u32 download_size = width * height * GetBytesPerPixel(pixel_format);

    if (gl_buffer.empty()) {
        gl_buffer.resize(download_size);
    }

    // Ensure no bad interactions with GL_PACK_ALIGNMENT
    ASSERT(stride * GetBytesPerPixel(pixel_format) % 4 == 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, static_cast<GLint>(stride));
    const u32 buffer_offset = (rect.bottom * stride + rect.left) * GetBytesPerPixel(pixel_format);

    // If not 1x scale, blit scaled texture to a new 1x texture and use that to flush
    if (res_scale != 1) {
        auto scaled_rect = rect;
        scaled_rect.left *= res_scale;
        scaled_rect.top *= res_scale;
        scaled_rect.right *= res_scale;
        scaled_rect.bottom *= res_scale;

        OGLTexture unscaled_tex = owner.AllocateSurfaceTexture(pixel_format, rect.GetWidth(), rect.GetHeight());

        const TextureBlit texture_blit = {
            .surface_type = type,
            .src_level = 0,
            .dst_level = 0,
            .src_region = Region2D{
                .start = {scaled_rect.left, scaled_rect.bottom},
                .end = {scaled_rect.right, scaled_rect.top}
            },
            .dst_region = Region2D{
                .start = {0, 0},
                .end = {rect.GetWidth(), rect.GetHeight()}
            }
        };

        // Blit scaled texture to the unscaled one
        runtime.BlitTextures(texture, unscaled_tex, texture_blit);

        state.texture_units[0].texture_2d = unscaled_tex.handle;
        state.Apply();

        glActiveTexture(GL_TEXTURE0);

        const FormatTuple& tuple = GetFormatTuple(pixel_format);
        if (GLES) {
            owner.texture_downloader_es->GetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type,
                                                     rect.GetHeight(), rect.GetWidth(),
                                                     &gl_buffer[buffer_offset]);
        } else {
            glGetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type, &gl_buffer[buffer_offset]);
        }
    } else {
        const BufferTextureCopy texture_download = {
            .buffer_offset = buffer_offset,
            .buffer_size = download_size,
            .buffer_row_length = stride,
            .buffer_height = height,
            .surface_type = type,
            .texture_level = 0,
            .texture_offset = {rect.bottom, rect.left},
            .texture_extent = {rect.GetWidth(), rect.GetHeight()}
        };

        runtime.ReadTexture(texture, texture_download, pixel_format, gl_buffer);
    }

    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
}

bool CachedSurface::CanFill(const SurfaceParams& dest_surface,
                            SurfaceInterval fill_interval) const {
    if (type == SurfaceType::Fill && IsRegionValid(fill_interval) &&
        boost::icl::first(fill_interval) >= addr &&
        boost::icl::last_next(fill_interval) <= end && // dest_surface is within our fill range
        dest_surface.FromInterval(fill_interval).GetInterval() ==
            fill_interval) { // make sure interval is a rectangle in dest surface
        if (fill_size * 8 != dest_surface.GetFormatBpp()) {
            // Check if bits repeat for our fill_size
            const u32 dest_bytes_per_pixel = std::max(dest_surface.GetFormatBpp() / 8, 1u);
            std::vector<u8> fill_test(fill_size * dest_bytes_per_pixel);

            for (u32 i = 0; i < dest_bytes_per_pixel; ++i)
                std::memcpy(&fill_test[i * fill_size], &fill_data[0], fill_size);

            for (u32 i = 0; i < fill_size; ++i)
                if (std::memcmp(&fill_test[dest_bytes_per_pixel * i], &fill_test[0],
                                dest_bytes_per_pixel) != 0)
                    return false;

            if (dest_surface.GetFormatBpp() == 4 && (fill_test[0] & 0xF) != (fill_test[0] >> 4))
                return false;
        }
        return true;
    }
    return false;
}

bool CachedSurface::CanCopy(const SurfaceParams& dest_surface,
                            SurfaceInterval copy_interval) const {
    SurfaceParams subrect_params = dest_surface.FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval);
    if (CanSubRect(subrect_params))
        return true;

    if (CanFill(dest_surface, copy_interval))
        return true;

    return false;
}

} // namespace OpenGL
