// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <set>
#include <span>
#include <vulkan/vulkan_hash.hpp>
#include "video_core/rasterizer_cache/framebuffer_base.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_vulkan/vk_blit_helper.h"
#include "video_core/renderer_vulkan/vk_format_reinterpreter.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Pica {
struct Regs;
}

namespace Vulkan {

struct StagingData {
    vk::Buffer buffer;
    u32 size = 0;
    std::span<u8> mapped{};
    u64 buffer_offset = 0;
};

struct Allocation {
    vk::Image image;
    vk::ImageView image_view;
    vk::ImageView depth_view;
    vk::ImageView stencil_view;
    vk::ImageView storage_view;
    VmaAllocation allocation;
    vk::ImageAspectFlags aspect;
    vk::Format format;
};

class Instance;
class RenderpassCache;
class DescriptorManager;
class Surface;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;
    friend class Sampler;

public:
    explicit TextureRuntime(const Instance& instance, Scheduler& scheduler,
                            RenderpassCache& renderpass_cache, DescriptorManager& desc_manager);
    ~TextureRuntime();

    /// Causes a GPU command flush
    void Finish();

    /// Takes back ownership of the allocation for recycling
    void Recycle(const VideoCore::HostTextureTag tag, Allocation&& alloc);

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    [[nodiscard]] StagingData FindStaging(u32 size, bool upload);

    /// Allocates a vulkan image possibly resusing an existing one
    [[nodiscard]] Allocation Allocate(u32 width, u32 height, u32 levels,
                                      VideoCore::PixelFormat format, VideoCore::TextureType type);

    /// Allocates a vulkan image
    [[nodiscard]] Allocation Allocate(u32 width, u32 height, u32 levels,
                                      VideoCore::PixelFormat pixel_format,
                                      VideoCore::TextureType type, vk::Format format,
                                      vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect);

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

    /// Returns a reference to the renderpass cache
    [[nodiscard]] RenderpassCache& GetRenderpassCache() {
        return renderpass_cache;
    }

private:
    /// Clears a partial texture rect using a clear rectangle
    void ClearTextureWithRenderpass(Surface& surface, const VideoCore::TextureClear& clear);

    /// Returns the current Vulkan instance
    const Instance& GetInstance() const {
        return instance;
    }

    /// Returns the current Vulkan scheduler
    Scheduler& GetScheduler() const {
        return scheduler;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    RenderpassCache& renderpass_cache;
    BlitHelper blit_helper;
    StreamBuffer upload_buffer;
    StreamBuffer download_buffer;
    std::array<ReinterpreterList, VideoCore::PIXEL_FORMAT_COUNT> reinterpreters;
    std::unordered_multimap<VideoCore::HostTextureTag, Allocation> texture_recycler;
};

class Surface : public VideoCore::SurfaceBase {
    friend class TextureRuntime;

public:
    explicit Surface(TextureRuntime& runtime, const VideoCore::SurfaceParams& params);
    explicit Surface(TextureRuntime& runtime, const VideoCore::SurfaceParams& params,
                     vk::Format format, vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect);
    ~Surface() override;

    Surface(const Surface&) noexcept = delete;
    Surface& operator=(const Surface&) noexcept = delete;

    Surface(Surface&&) noexcept = default;
    Surface& operator=(Surface&&) noexcept = default;

    /// Returns the surface aspect
    vk::ImageAspectFlags Aspect() const noexcept {
        return alloc.aspect;
    }

    /// Returns the surface image handle
    vk::Image Image() const noexcept {
        return alloc.image;
    }

    /// Returns an image view used to sample the surface from a shader
    vk::ImageView ImageView() const noexcept {
        return alloc.image_view;
    }

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads pixel data to staging from a rectangle region of the surface texture
    void Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging);

    /// Returns the bpp of the internal surface format
    u32 GetInternalBytesPerPixel() const;

    /// Returns the access flags indicative of the surface
    vk::AccessFlags AccessFlags() const noexcept;

    /// Returns the pipeline stage flags indicative of the surface
    vk::PipelineStageFlags PipelineStageFlags() const noexcept;

    /// Returns the depth only image view of the surface
    vk::ImageView DepthView() noexcept;

    /// Returns the stencil only image view of the surface
    vk::ImageView StencilView() noexcept;

    /// Returns the R32 image view used for atomic load/store
    vk::ImageView StorageView() noexcept;

private:
    /// Uploads pixel data to scaled texture
    void ScaledUpload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads scaled image by downscaling the requested rectangle
    void ScaledDownload(const VideoCore::BufferTextureCopy& download, const StagingData& stagings);

    /// Downloads scaled depth stencil data
    void DepthStencilDownload(const VideoCore::BufferTextureCopy& download,
                              const StagingData& staging);

private:
    TextureRuntime* runtime;
    Scheduler* scheduler;
    vk::Device device;
    Allocation alloc;
    bool is_framebuffer{};
    bool is_storage{};
};

class Framebuffer : public VideoCore::FramebufferBase {
public:
    explicit Framebuffer(Surface* const color, Surface* const depth_stencil,
                         vk::Rect2D render_area);
    explicit Framebuffer(TextureRuntime& runtime, Surface* const color,
                         Surface* const depth_stencil, const Pica::Regs& regs,
                         Common::Rectangle<u32> surfaces_rect);
    ~Framebuffer();

    [[nodiscard]] vk::Image Image(VideoCore::SurfaceType type) const noexcept {
        return images[Index(type)];
    }

    [[nodiscard]] vk::ImageView ImageView(VideoCore::SurfaceType type) const noexcept {
        return image_views[Index(type)];
    }

    bool HasView(VideoCore::SurfaceType type) const noexcept {
        return static_cast<bool>(image_views[Index(type)]);
    }

    u32 Width() const noexcept {
        return width;
    }

    u32 Height() const noexcept {
        return height;
    }

    vk::Rect2D RenderArea() const noexcept {
        return render_area;
    }

private:
    void PrepareImages(Surface* const color, Surface* const depth_stencil);

private:
    std::array<vk::Image, 2> images{};
    std::array<vk::ImageView, 2> image_views{};
    vk::Rect2D render_area{};
    u32 width{};
    u32 height{};
};

/**
 * @brief A sampler is used to configure the sampling parameters of a texture unit
 */
class Sampler {
public:
    Sampler(TextureRuntime& runtime, VideoCore::SamplerParams params);
    ~Sampler();

    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;

    Sampler(Sampler&& o) noexcept {
        std::memcpy(this, &o, sizeof(Sampler));
        o.sampler = VK_NULL_HANDLE;
    }
    Sampler& operator=(Sampler&& o) noexcept {
        std::memcpy(this, &o, sizeof(Sampler));
        o.sampler = VK_NULL_HANDLE;
        return *this;
    }

    [[nodiscard]] vk::Sampler Handle() const noexcept {
        return sampler;
    }

private:
    vk::Device device;
    vk::Sampler sampler;
};

struct Traits {
    using Runtime = TextureRuntime;
    using Surface = Surface;
    using Framebuffer = Framebuffer;
    using Sampler = Sampler;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace Vulkan
