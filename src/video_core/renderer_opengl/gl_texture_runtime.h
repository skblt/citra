// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <set>
#include <span>
#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_opengl/gl_format_reinterpreter.h"
#include "video_core/renderer_opengl/gl_stream_buffer.h"
#include "video_core/renderer_opengl/texture_filters/texture_filterer.h"

namespace OpenGL {

struct FormatTuple {
    GLint internal_format;
    GLenum format;
    GLenum type;
};

struct StagingData {
    GLuint buffer;
    u32 size = 0;
    std::span<u8> mapped{};
    u64 buffer_offset = 0;
};

class Driver;
class Surface;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;
    friend class Framebuffer;

public:
    TextureRuntime(Driver& driver);
    ~TextureRuntime() = default;

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    StagingData FindStaging(u32 size, bool upload);

    /// Returns the OpenGL format tuple associated with the provided pixel format
    const FormatTuple& GetFormatTuple(VideoCore::PixelFormat pixel_format);

    /// Causes a GPU command flush
    void Finish() const {}

    /// Allocates an OpenGL texture with the specified dimentions and format
    OGLTexture Allocate(u32 width, u32 height, u32 levels, VideoCore::PixelFormat format,
                        VideoCore::TextureType type);

    /// Fills the rectangle of the texture with the clear value provided
    bool ClearTexture(Surface& surface, const VideoCore::TextureClear& clear);

    /// Copies a rectangle of src_tex to another rectange of dst_rect
    bool CopyTextures(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Blits a rectangle of src_tex to another rectange of dst_rect
    bool BlitTextures(Surface& surface, Surface& dest, const VideoCore::TextureBlit& blit);

    /// Generates mipmaps for all the available levels of the texture
    void GenerateMipmaps(Surface& surface, u32 max_level);

    /// Returns all source formats that support reinterpretation to the dest format
    [[nodiscard]] const ReinterpreterList& GetPossibleReinterpretations(
        VideoCore::PixelFormat dest_format) const;

    /// Returns true if the provided pixel format needs convertion
    [[nodiscard]] bool NeedsConvertion(VideoCore::PixelFormat format) const;

private:
    /// Returns the framebuffer used for texture downloads
    void BindFramebuffer(GLenum target, GLint level, GLenum textarget, VideoCore::SurfaceType type,
                         OGLTexture& texture) const;

    /// Returns the OpenGL driver class
    const Driver& GetDriver() const {
        return driver;
    }

    /// Returns the class that handles texture filtering
    const TextureFilterer& GetFilterer() const {
        return filterer;
    }

private:
    Driver& driver;
    TextureFilterer filterer;
    std::array<ReinterpreterList, VideoCore::PIXEL_FORMAT_COUNT> reinterpreters;
    std::unordered_multimap<VideoCore::HostTextureTag, OGLTexture> texture_recycler;
    std::unordered_map<u64, OGLFramebuffer, Common::IdentityHash<u64>> framebuffer_cache;
    StreamBuffer upload_buffer;
    std::vector<u8> download_buffer;
    OGLFramebuffer read_fbo, draw_fbo;
};

class Surface : public VideoCore::SurfaceBase {
public:
    Surface(VideoCore::SurfaceParams& params, TextureRuntime& runtime);
    ~Surface();

    /// Returns the surface image handle
    GLuint Handle() const noexcept {
        return texture.handle;
    }

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads pixel data to staging from a rectangle region of the surface texture
    void Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging);

    /// Returns the bpp of the internal surface format
    u32 GetInternalBytesPerPixel() const {
        return VideoCore::GetBytesPerPixel(pixel_format);
    }

private:
    /// Uploads pixel data to scaled texture
    void ScaledUpload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads scaled image by downscaling the requested rectangle
    void ScaledDownload(const VideoCore::BufferTextureCopy& download, const StagingData& staging);

private:
    TextureRuntime& runtime;
    const Driver& driver;

public:
    OGLTexture texture{};
};

class Framebuffer : public VideoCore::FramebufferBase {
public:
    explicit Framebuffer(TextureRuntime& runtime, Surface* const color,
                         Surface* const depth_stencil, const Pica::Regs& regs,
                         Common::Rectangle<u32> surfaces_rect);
    ~Framebuffer();

    [[nodiscard]] GLuint Handle() const noexcept {
        return handle;
    }

    [[nodiscard]] GLuint Attachment(VideoCore::SurfaceType type) const noexcept {
        return attachments[Index(type)];
    }

    bool HasAttachment(VideoCore::SurfaceType type) const noexcept {
        return static_cast<bool>(attachments[Index(type)]);
    }

private:
    std::array<GLuint, 2> attachments{};
    GLuint handle{};
};

class Sampler {
public:
    explicit Sampler(TextureRuntime& runtime, VideoCore::SamplerParams params);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;

    [[nodiscard]] GLuint Handle() const noexcept {
        return sampler.handle;
    }

private:
    OGLSampler sampler;
};

struct Traits {
    using RuntimeType = TextureRuntime;
    using SurfaceType = Surface;
    using Sampler = Sampler;
    using Framebuffer = Framebuffer;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace OpenGL
