// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <set>
#include <span>
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

    auto operator<=>(const FormatTuple&) const noexcept = default;
};

struct StagingData {
    GLuint buffer;
    u32 size = 0;
    std::span<std::byte> mapped{};
    GLintptr buffer_offset = 0;
};

class Driver;
class Surface;
class Framebuffer;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;

public:
    TextureRuntime(Driver& driver);
    ~TextureRuntime();

    /// Returns the OpenGL driver class
    const Driver& GetDriver() const {
        return driver;
    }

    /// Returns the class that handles texture filtering
    const TextureFilterer& GetFilterer() const {
        return filterer;
    }

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    [[nodiscard]] StagingData FindStaging(u32 size, bool upload);

    /// Returns the OpenGL format tuple associated with the provided pixel format
    FormatTuple NativeFormat(VideoCore::PixelFormat pixel_format) const;

    /// Causes a GPU command flush
    void Finish() const {}

    /// Performs required format convertions on the staging data
    void FormatConvert(const Surface& surface, bool upload, std::span<std::byte> source,
                       std::span<std::byte> dest);

    /// Fills the rectangle of the texture with the clear value provided
    bool SurfaceClear(Surface& surface, const VideoCore::TextureClear& clear);

    /// Fills a partial texture rect using a clear renderpass
    bool FramebufferClear(Framebuffer& framebuffer, const VideoCore::TextureClear& clear);

    /// Blits a rectangle of source framebuffer to another rectange of dest framebuffer
    bool FramebufferBlit(const Framebuffer& source, const Framebuffer& dest,
                         const VideoCore::TextureBlit& blit);

    /// Returns all source formats that support reinterpretation to the dest format
    [[nodiscard]] const ReinterpreterList& GetPossibleReinterpretations(
        VideoCore::PixelFormat dest_format) const;

    /// Returns true if the provided pixel format needs convertion
    [[nodiscard]] bool NeedsConvertion(VideoCore::PixelFormat format) const;

private:
    /// Binds a texture to the appropriate rescale framebuffer
    Framebuffer MakeRescaleFramebuffer(Surface& surface, GLenum target, GLint level,
                                       bool scaled) const;

private:
    Driver& driver;
    TextureFilterer filterer;
    std::array<ReinterpreterList, VideoCore::PIXEL_FORMAT_COUNT> reinterpreters;
    OGLStreamBuffer upload_buffer, download_buffer;
    std::array<OGLFramebuffer, 3> rescale_draw_fbos;
    std::array<OGLFramebuffer, 3> rescale_read_fbos;
};

struct Allocation {
    Allocation() = default;
    Allocation(TextureRuntime&);
    Allocation(TextureRuntime& runtime, const VideoCore::SurfaceParams& params);
    ~Allocation();

    Allocation(const Allocation&) noexcept = delete;
    Allocation& operator=(const Allocation&) noexcept = delete;

    Allocation(Allocation&& o) noexcept {
        std::memcpy(this, &o, sizeof(Allocation));
        o.handles.fill(0);
    }
    Allocation& operator=(Allocation&& o) noexcept {
        std::memcpy(this, &o, sizeof(Allocation));
        o.handles.fill(0);
        return *this;
    }

    std::array<GLuint, 2> handles{};
    FormatTuple tuple;
};

class Surface : public VideoCore::SurfaceBase {
public:
    Surface(VideoCore::SurfaceParams params);
    Surface(TextureRuntime& runtime, Allocation&& alloc, VideoCore::SurfaceParams params);
    ~Surface() override;

    Surface(const Surface&) noexcept = delete;
    Surface& operator=(const Surface&) noexcept = delete;

    Surface(Surface&&) noexcept = default;
    Surface& operator=(Surface&&) noexcept = default;

    [[nodiscard]] Allocation&& Release() noexcept {
        return std::move(alloc);
    }

    [[nodiscard]] GLuint Handle(bool unscaled = false) const noexcept {
        return alloc.handles[unscaled];
    }

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads pixel data to staging from a rectangle region of the surface texture
    void Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging);

    /// Returns the bpp of the internal surface format
    [[nodiscard]] u32 InternalBytesPerPixel() const noexcept {
        return VideoCore::GetBytesPerPixel(pixel_format);
    }

private:
    /// Performs blit between the scaled/unscaled images
    void BlitScale(const VideoCore::TextureBlit& blit, bool up_scale);

    /// Configures and binds the appropriate runtime rescale framebuffer
    void BindFramebuffer(GLuint handle, GLenum target, GLsizei level);

private:
    const Driver* driver{};
    TextureRuntime* runtime{};
    Allocation alloc;
};

class Framebuffer {
public:
    Framebuffer(TextureRuntime&, Surface* color, Surface* depth, VideoCore::RenderTargets);
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    Framebuffer(Framebuffer&& o) = default;
    Framebuffer& operator=(Framebuffer&&) = default;

    [[nodiscard]] GLuint Handle() const noexcept {
        return framebuffer.handle;
    }

    [[nodiscard]] GLbitfield BufferMask() const noexcept {
        return buffer_mask;
    }

    [[nodiscard]] VideoCore::Rect2D RenderArea() const noexcept {
        return render_area;
    }

    void SetRenderArea(VideoCore::Rect2D draw_rect) noexcept {
        render_area = draw_rect;
    }

private:
    OGLFramebuffer framebuffer{};
    GLbitfield buffer_mask{};
    VideoCore::Rect2D render_area{};
};

class Sampler {
public:
    Sampler(TextureRuntime& runtime, VideoCore::SamplerParams params);
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
    static constexpr bool FRAMEBUFFER_BLITS = true;

    using Runtime = OpenGL::TextureRuntime;
    using Surface = OpenGL::Surface;
    using Framebuffer = OpenGL::Framebuffer;
    using Allocation = OpenGL::Allocation;
    using Sampler = OpenGL::Sampler;
    using Format = OpenGL::FormatTuple;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace OpenGL
