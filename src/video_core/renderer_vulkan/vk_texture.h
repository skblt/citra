// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/common/texture.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore::Vulkan {

// PICA texture have at most 8 mipmap levels
constexpr u32 TEXTURE_MAX_LEVELS = 8;

class Instance;
class CommandScheduler;

/**
 * A texture located in GPU memory
 */
class Texture : public VideoCore::TextureBase {
public:
    // Default constructor
    Texture(Instance& instance, CommandScheduler& scheduler);

    // Constructor for texture creation
    Texture(Instance& instance, CommandScheduler& scheduler, const TextureInfo& info);

    // Constructor for not owning textures (swapchain)
    Texture(Instance& instance, CommandScheduler& scheduler, vk::Image image, const TextureInfo& info);

    ~Texture() override;

    void Upload(Rect2D rectangle, u32 stride, std::span<const u8> data, u32 level = 0) override;

    void Download(Rect2D rectangle, u32 stride, std::span<u8> data, u32 level = 0) override;

    void BlitTo(TextureHandle dest, Rect2D src_rectangle, Rect2D dest_rect, u32 src_level = 0,
                u32 dest_level = 0, u32 src_layer = 0, u32 dest_layer = 0) override;

    void GenerateMipmaps() override;

    /// Overrides the layout of provided image subresource
    void SetLayout(vk::ImageLayout new_layout, u32 level = 0, u32 level_count = 1);

    /// Transitions part of the image to the provided layout
    void Transition(vk::CommandBuffer command_buffer, vk::ImageLayout new_layout, u32 level = 0,
                    u32 level_count = 1);

    /// Returns the underlying vulkan image handle
    vk::Image GetHandle() const {
        return image;
    }

    /// Returns the Vulka image view
    vk::ImageView GetView() const {
        return image_view;
    }

    /// Returns the internal format backing the texture.
    /// It may not match the input pixel format.
    vk::Format GetInternalFormat() const {
        return internal_format;
    }

    vk::ImageAspectFlags GetAspectFlags() const {
        return aspect;
    }

    /// Returns the current image layout
    vk::ImageLayout GetLayout(u32 level = 0) const {
        return layouts.at(level);
    }

    /// Returns a rectangle that represents the complete area of the texture
    vk::Rect2D GetArea() const {
        return {{0, 0},{info.width, info.height}};
    }

private:
    Instance& instance;
    CommandScheduler& scheduler;

    // Vulkan texture handle
    vk::Image image = VK_NULL_HANDLE;
    vk::ImageView image_view = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    bool is_texture_owned = true;

    // Texture properties
    vk::Format advertised_format = vk::Format::eUndefined;
    vk::Format internal_format = vk::Format::eUndefined;
    vk::ImageAspectFlags aspect = vk::ImageAspectFlagBits::eNone;
    std::array<vk::ImageLayout, TEXTURE_MAX_LEVELS> layouts;
};

/**
 * Staging texture located in CPU memory. Used for intermediate format
 * conversions
 */
class StagingTexture : public VideoCore::TextureBase {
public:
    StagingTexture(Instance& instance, CommandScheduler& scheduler,
                   const TextureInfo& info);
    ~StagingTexture();

    /// Flushes any writes made to texture memory
    void Commit(u32 size);

    /// Returns a span of the mapped texture memory
    void* GetMappedPtr() {
        return mapped_ptr;
    }

    /// Returns the staging image handle
    vk::Image GetHandle() const {
        return image;
    }

private:
    Instance& instance;
    CommandScheduler& scheduler;

    vk::Image image = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    vk::Format format = vk::Format::eUndefined;
    u32 capacity = 0;
    void* mapped_ptr = nullptr;
};

/**
 * Vulkan sampler object
 */
class Sampler : public VideoCore::SamplerBase {
public:
    Sampler(Instance& instance, SamplerInfo info);
    ~Sampler() override;

    /// Returns the underlying vulkan sampler handle
    vk::Sampler GetHandle() const {
        return sampler;
    }

private:
    Instance& instance;
    vk::Sampler sampler;
};

} // namespace Vulkan