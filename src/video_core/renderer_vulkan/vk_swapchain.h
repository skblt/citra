// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class RenderpassCache;

class Swapchain {
public:
    Swapchain(const Instance& instance, RenderpassCache& renderpass_cache);
    ~Swapchain();

    /// Creates (or recreates) the swapchain with a given size.
    void Create(u32 width, u32 height);

    /// Acquires the next image in the swapchain.
    void AcquireNextImage(vk::Semaphore signal_acquired);

    /// Presents the current image and move to the next one
    void Present(vk::Semaphore wait_for_present);

    /// Returns current swapchain state
    vk::Extent2D GetExtent() const {
        return extent;
    }

    /// Returns the swapchain surface
    vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    /// Returns the current framebuffe
    vk::Framebuffer GetFramebuffer() const {
        return swapchain_images[current_image].framebuffer;
    }

    /// Returns the swapchain format
    vk::SurfaceFormatKHR GetSurfaceFormat() const {
        return surface_format;
    }

    /// Returns the Vulkan swapchain handle
    vk::SwapchainKHR GetHandle() const {
        return swapchain;
    }

    /// Returns true when the swapchain should be recreated
    bool NeedsRecreation() const {
        return is_suboptimal || is_outdated;
    }

private:
    void Configure(u32 width, u32 height);

private:
    const Instance& instance;
    RenderpassCache& renderpass_cache;
    vk::SwapchainKHR swapchain{};
    vk::SurfaceKHR surface{};

    // Swapchain properties
    vk::SurfaceFormatKHR surface_format;
    vk::PresentModeKHR present_mode;
    vk::Extent2D extent;
    vk::SurfaceTransformFlagBitsKHR transform;
    u32 image_count;

    struct Image {
        vk::Image image;
        vk::ImageView image_view;
        vk::Framebuffer framebuffer;
    };

    // Swapchain state
    std::vector<Image> swapchain_images;
    u32 current_image = 0;
    bool is_outdated = true;
    bool is_suboptimal = true;
};

} // namespace Vulkan
