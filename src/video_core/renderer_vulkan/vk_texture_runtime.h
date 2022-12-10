// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <set>
#include <span>
#include <vulkan/vulkan_hash.hpp>
#include "video_core/rasterizer_cache/rasterizer_cache.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/renderer_vulkan/vk_blit_helper.h"
#include "video_core/renderer_vulkan/vk_format_reinterpreter.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"

namespace Vulkan {

struct StagingData {
    vk::Buffer buffer;
    u32 size{0};
    std::span<std::byte> mapped{};
    std::size_t buffer_offset{0};
};

class Instance;
class RenderpassCache;
class DescriptorManager;
class Surface;
class Framebuffer;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
    friend class Surface;

public:
    TextureRuntime(const Instance& instance, Scheduler& scheduler,
                   RenderpassCache& renderpass_cache, DescriptorManager& desc_manager);
    ~TextureRuntime();

    /// Returns the vulkan instance
    [[nodiscard]] const Instance& GetInstance() const noexcept {
        return instance;
    }

    /// Returns the vulkan scheduler
    [[nodiscard]] Scheduler& GetScheduler() const noexcept {
        return scheduler;
    }

    /// Returns the vulkan renderpass cache
    [[nodiscard]] RenderpassCache& GetRenderpassCache() const noexcept {
        return renderpass_cache;
    }

    /// Maps an internal staging buffer of the provided size of pixel uploads/downloads
    [[nodiscard]] StagingData FindStaging(u32 size, bool upload);

    /// Returns the vulkan format associated with the provided pixel format
    vk::Format NativeFormat(VideoCore::PixelFormat pixel_format) const;

    /// Causes a GPU command flush
    void Finish();

    /// Performs required format convertions on the staging data
    void FormatConvert(const Surface& surface, bool upload, std::span<std::byte> source,
                       std::span<std::byte> dest);

    /// Fills a partial texture rect using a clear renderpass
    bool FramebufferClear(Framebuffer& framebuffer, const VideoCore::TextureClear& clear);

    /// Fills the rectangle of the texture with the clear value provided
    bool SurfaceClear(Surface& surface, const VideoCore::TextureClear& clear);

    /// Copies a rectangle of src_tex to another rectange of dst_rect
    bool SurfaceCopy(Surface& source, Surface& dest, const VideoCore::TextureCopy& copy);

    /// Blits a rectangle of src_tex to another rectange of dst_rect
    bool SurfaceBlit(Surface& surface, Surface& dest, const VideoCore::TextureBlit& blit);

    /// Flushes staging buffers
    void FlushBuffers();

    /// Returns all source formats that support reinterpretation to the dest format
    [[nodiscard]] const ReinterpreterList& GetPossibleReinterpretations(
        VideoCore::PixelFormat dest_format) const;

    /// Returns true if the provided pixel format needs convertion
    [[nodiscard]] bool NeedsConvertion(VideoCore::PixelFormat format) const;

public:
    const Instance& instance;
    Scheduler& scheduler;
    RenderpassCache& renderpass_cache;
    DescriptorManager& desc_manager;
    BlitHelper blit_helper;
    StreamBuffer upload_buffer;
    StreamBuffer download_buffer;
    std::array<ReinterpreterList, VideoCore::PIXEL_FORMAT_COUNT> reinterpreters;
};

struct Allocation {
    Allocation() = default;
    Allocation(TextureRuntime& runtime);
    Allocation(TextureRuntime& runtime, const VideoCore::SurfaceParams& params);
    ~Allocation();

    Allocation(const Allocation&) noexcept = delete;
    Allocation& operator=(const Allocation&) noexcept = delete;

    Allocation(Allocation&& o) noexcept {
        std::memcpy(this, &o, sizeof(Allocation));
        o.device = VK_NULL_HANDLE;
    }
    Allocation& operator=(Allocation&& o) noexcept {
        std::memcpy(this, &o, sizeof(Allocation));
        o.device = VK_NULL_HANDLE;
        return *this;
    }

    vk::Device device;
    VmaAllocator allocator;
    std::array<vk::Image, 2> images;
    std::array<VmaAllocation, 2> allocations;
    vk::ImageView image_view;
    vk::ImageView depth_view;
    vk::ImageView stencil_view;
    vk::ImageView storage_view;
    vk::Format format;
    vk::ImageAspectFlags aspect;
};

class Surface : public VideoCore::SurfaceBase {
    friend class TextureRuntime;

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

    [[nodiscard]] vk::Image Handle(bool unscaled = false) const noexcept {
        return alloc.images[unscaled];
    }

    [[nodiscard]] vk::ImageView ImageView() const noexcept {
        return alloc.image_view;
    }

    [[nodiscard]] vk::ImageView DepthView() const noexcept {
        return alloc.depth_view;
    }

    [[nodiscard]] vk::ImageView StencilView() const noexcept {
        return alloc.stencil_view;
    }

    [[nodiscard]] vk::ImageView StorageView() const noexcept {
        ASSERT(alloc.storage_view);
        return alloc.storage_view;
    }

    [[nodiscard]] vk::Format InternalFormat() const noexcept {
        return alloc.format;
    }

    [[nodiscard]] vk::Format NativeFormat() const noexcept {
        return traits.native;
    }

    /// Uploads pixel data in staging to a rectangle region of the surface texture
    void Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging);

    /// Downloads pixel data to staging from a rectangle region of the surface texture
    void Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging);

    /// Returns the bpp of the internal surface format
    u32 InternalBytesPerPixel() const;

private:
    /// Performs blit between the scaled/unscaled images
    void BlitScale(const VideoCore::TextureBlit& blit, bool up_scale);

private:
    Scheduler* scheduler{};
    TextureRuntime* runtime{};
    Allocation alloc;
    FormatTraits traits;
};

class Framebuffer {
public:
    explicit Framebuffer(TextureRuntime& runtime, Surface* color_surface, Surface* depth_surface,
                         VideoCore::RenderTargets key);
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    Framebuffer(Framebuffer&& o) noexcept {
        std::memcpy(this, &o, sizeof(Framebuffer));
        o.framebuffer = VK_NULL_HANDLE;
    }
    Framebuffer& operator=(Framebuffer&& o) noexcept {
        std::memcpy(this, &o, sizeof(Framebuffer));
        o.framebuffer = VK_NULL_HANDLE;
        return *this;
    }

    [[nodiscard]] vk::Framebuffer Handle() const noexcept {
        return framebuffer;
    }

    [[nodiscard]] vk::ImageView ColorView() const noexcept {
        return color_view;
    }

    [[nodiscard]] vk::ImageView DepthView() const noexcept {
        return depth_view;
    }

    [[nodiscard]] VideoCore::Rect2D RenderArea() const noexcept {
        return render_area;
    }

    [[nodiscard]] vk::RenderPass RenderPass(vk::AttachmentLoadOp load_op) const noexcept {
        return renderpass[static_cast<u32>(load_op)];
    }

    void SetRenderArea(VideoCore::Rect2D draw_rect) noexcept {
        render_area = draw_rect;
    }

    /// Begins a renderpass with the stored framebuffer as render target
    void BeginRenderPass();

private:
    RenderpassCache* renderpass_cache{};
    vk::Device device;
    std::array<vk::RenderPass, 2> renderpass;
    vk::ImageView color_view{};
    vk::ImageView depth_view{};
    vk::Framebuffer framebuffer;
    VideoCore::Rect2D render_area;
};

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
    static constexpr bool FRAMEBUFFER_BLITS = false;

    using Runtime = Vulkan::TextureRuntime;
    using Surface = Vulkan::Surface;
    using Framebuffer = Vulkan::Framebuffer;
    using Allocation = Vulkan::Allocation;
    using Sampler = Vulkan::Sampler;
    using Format = vk::Format;
};

using RasterizerCache = VideoCore::RasterizerCache<Traits>;

} // namespace Vulkan
