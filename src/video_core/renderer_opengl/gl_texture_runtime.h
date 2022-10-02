// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include <set>
#include "video_core/rasterizer_cache/rasterizer_cache.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"
#include "video_core/renderer_opengl/texture_filters/texture_filterer.h"
#include "video_core/renderer_opengl/texture_downloader_es.h"

namespace OpenGL {

struct FormatTuple {
    GLint internal_format;
    GLenum format;
    GLenum type;
};

struct StagingBuffer {
    OGLBuffer buffer{};
    mutable OGLSync buffer_lock{};
    std::span<std::byte> mapped{};
    u32 size{};

    bool operator<(const StagingBuffer& other) const {
        return size < other.size;
    }

    /// Returns true if the buffer does not take part in pending transfer operations
    bool IsFree() const {
        if (buffer_lock) {
            GLint status;
            glGetSynciv(buffer_lock.handle, GL_SYNC_STATUS, 1, nullptr, &status);
            return status == GL_SIGNALED;
        }

        return true;
    }

    /// Prevents the runtime from reusing the buffer until the transfer operation is complete
    void Lock() const {
        if (buffer_lock) {
            buffer_lock.Release();
        }

        buffer_lock.Create();
    }
};

class Driver;
class Surface;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;
public:
    TextureRuntime(Driver& driver);
    ~TextureRuntime() = default;

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    const StagingBuffer& FindStaging(u32 size, bool upload);

    /// Returns the OpenGL format tuple associated with the provided pixel format
    const FormatTuple& GetFormatTuple(VideoCore::PixelFormat pixel_format);

    void Finish() const {}

    /// Performs required format convertions on the staging data
    void FormatConvert(const Surface& surface, bool upload,
                       std::span<std::byte> source, std::span<std::byte> dest);

    /// Allocates an OpenGL texture with the specified dimentions and format
    OGLTexture Allocate(u32 width, u32 height, VideoCore::PixelFormat format,
                        VideoCore::TextureType type);

    /// Fills the rectangle of the texture with the clear value provided
    bool ClearTexture(Surface& surface, const VideoCore::TextureClear& clear,
                      VideoCore::ClearValue value);

    /// Copies a rectangle of src_tex to another rectange of dst_rect
    bool CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Blits a rectangle of src_tex to another rectange of dst_rect
    bool BlitTextures(Surface& surface, Surface& dest, const VideoCore::TextureBlit& blit);

    /// Generates mipmaps for all the available levels of the texture
    void GenerateMipmaps(Surface& surface, u32 max_level);

private:
    /// Returns the framebuffer used for texture downloads
    void BindFramebuffer(GLenum target, GLint level, GLenum textarget,
                         VideoCore::SurfaceType type, OGLTexture& texture) const;

    /// Returns the OpenGL driver class
    const Driver& GetDriver() const {
        return driver;
    }

    /// Returns the class that handles texture downloads for OpenGL ES
    const TextureDownloaderES& GetDownloaderES() const {
        return downloader_es;
    }

    /// Returns the class that handles texture filtering
    const TextureFilterer& GetFilterer() const {
        return filterer;
    }

private:
    Driver& driver;
    TextureDownloaderES downloader_es;
    TextureFilterer filterer;

    // Staging buffers stored in increasing size
    std::multiset<StagingBuffer> upload_buffers;
    std::multiset<StagingBuffer> download_buffers;
    OGLFramebuffer read_fbo, draw_fbo;

    // Recycled textures to reduce driver allocation overhead
    std::unordered_multimap<VideoCore::HostTextureTag, OGLTexture> texture_recycler;
};

class Surface : public VideoCore::SurfaceBase<Surface> {
public:
    Surface(VideoCore::SurfaceParams& params, TextureRuntime& runtime);
    ~Surface() override;

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const StagingBuffer& staging);

    /// Downloads pixel data to staging from a rectangle region of the surface texture
    void Download(const VideoCore::BufferTextureCopy& download, const StagingBuffer& staging);

    /// Returns the bpp of the internal surface format
    u32 GetInternalBytesPerPixel() const {
        return VideoCore::GetBytesPerPixel(pixel_format);
    }

private:
    /// Downloads scaled image by downscaling the requested rectangle
    void ScaledDownload(const VideoCore::BufferTextureCopy& download);

    /// Uploads pixel data to scaled texture
    void ScaledUpload(const VideoCore::BufferTextureCopy& upload);

private:
    TextureRuntime& runtime;
    const Driver& driver;

public:
    OGLTexture texture{};
};

struct Traits {
    using RuntimeType = TextureRuntime;
    using SurfaceType = Surface;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace OpenGL
