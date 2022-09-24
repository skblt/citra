// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <algorithm>
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"

namespace Vulkan {

Swapchain::Swapchain(const Instance& instance, RenderpassCache& renderpass_cache)
    : instance{instance}, renderpass_cache{renderpass_cache}, surface{instance.GetSurface()} {

    // Set the surface format early for RenderpassCache to create the present renderpass
    Configure(0, 0);
    renderpass_cache.CreatePresentRenderpass(surface_format.format);
}

Swapchain::~Swapchain() {
    vk::Device device = instance.GetDevice();
    device.destroySwapchainKHR(swapchain);

    for (auto& image : swapchain_images) {
        device.destroyImageView(image.image_view);
        device.destroyFramebuffer(image.framebuffer);
    }
}

void Swapchain::Create(u32 width, u32 height, bool vsync_enabled) {
    is_outdated = false;
    is_suboptimal = false;

    // Fetch information about the provided surface
    Configure(width, height);

    const std::array queue_family_indices = {
        instance.GetGraphicsQueueFamilyIndex(),
        instance.GetPresentQueueFamilyIndex(),
    };

    const bool exclusive = queue_family_indices[0] == queue_family_indices[1];
    const u32 queue_family_indices_count = exclusive ? 1u : 2u;
    const vk::SharingMode sharing_mode =
            exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent;
    const vk::SwapchainCreateInfoKHR swapchain_info = {
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
        .imageSharingMode = sharing_mode,
        .queueFamilyIndexCount = queue_family_indices_count,
        .pQueueFamilyIndices   = queue_family_indices.data(),
        .preTransform = transform,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = swapchain
    };

    vk::Device device = instance.GetDevice();
    vk::SwapchainKHR new_swapchain = device.createSwapchainKHR(swapchain_info);

    // If an old swapchain exists, destroy it and move the new one to its place.
    if (vk::SwapchainKHR old_swapchain = std::exchange(swapchain, new_swapchain); old_swapchain) {
        device.destroySwapchainKHR(old_swapchain);
    }

    vk::RenderPass present_renderpass = renderpass_cache.GetPresentRenderpass();
    auto images = device.getSwapchainImagesKHR(swapchain);

    // Destroy the previous images
    for (auto& image : swapchain_images) {
        device.destroyImageView(image.image_view);
        device.destroyFramebuffer(image.framebuffer);
    }

    swapchain_images.clear();
    swapchain_images.resize(images.size());

    std::ranges::transform(images, swapchain_images.begin(), [&](vk::Image image) -> Image {
        const vk::ImageViewCreateInfo view_info = {
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = surface_format.format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            }
        };

        vk::ImageView image_view = device.createImageView(view_info);
        const std::array attachments{image_view};

        const vk::FramebufferCreateInfo framebuffer_info = {
            .renderPass = present_renderpass,
            .attachmentCount = 1,
            .pAttachments = attachments.data(),
            .width = extent.width,
            .height = extent.height,
            .layers = 1
        };

        vk::Framebuffer framebuffer = device.createFramebuffer(framebuffer_info);

        return Image{
            .image = image,
            .image_view = image_view,
            .framebuffer = framebuffer
        };
    });
}

// Wait for maximum of 1 second
constexpr u64 ACQUIRE_TIMEOUT = 1000000000;

void Swapchain::AcquireNextImage(vk::Semaphore signal_acquired) {
    vk::Device device = instance.GetDevice();
    vk::Result result = device.acquireNextImageKHR(swapchain, ACQUIRE_TIMEOUT, signal_acquired,
                                                   VK_NULL_HANDLE, &current_image);
    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
        is_suboptimal = true;
        break;
    case vk::Result::eErrorOutOfDateKHR:
        is_outdated = true;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "vkAcquireNextImageKHR returned unknown result");
        break;
    }
}

void Swapchain::Present(vk::Semaphore wait_for_present) {
    const vk::PresentInfoKHR present_info = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &wait_for_present,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &current_image
    };

    vk::Queue present_queue = instance.GetPresentQueue();
    vk::Result result = present_queue.presentKHR(present_info);

    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
        LOG_DEBUG(Render_Vulkan, "Suboptimal swapchain");
        break;
    case vk::Result::eErrorOutOfDateKHR:
        is_outdated = true;
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Swapchain presentation failed");
        break;
    }

    current_frame = (current_frame + 1) % swapchain_images.size();
}

void Swapchain::Configure(u32 width, u32 height) {
    vk::PhysicalDevice physical = instance.GetPhysicalDevice();

    // Choose surface format
    auto formats = physical.getSurfaceFormatsKHR(surface);
    surface_format = formats[0];

    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        surface_format.format = vk::Format::eB8G8R8A8Unorm;
    } else {
        auto it = std::ranges::find_if(formats, [](vk::SurfaceFormatKHR format) -> bool {
            return format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear &&
                format.format == vk::Format::eB8G8R8A8Unorm;
        });

        if (it == formats.end()) {
            LOG_CRITICAL(Render_Vulkan, "Unable to find required swapchain format!");
        } else {
            surface_format = *it;
        }
    }

    // Checks if a particular mode is supported, if it is, returns that mode.
    auto modes = physical.getSurfacePresentModesKHR(surface);

    // FIFO is guaranteed by the Vulkan standard to be available
    present_mode = vk::PresentModeKHR::eFifo;
    auto iter = std::ranges::find_if(modes, [](vk::PresentModeKHR mode) {
        return vk::PresentModeKHR::eMailbox == mode;
    });

    // Prefer Mailbox if present for lowest latency
    if (iter != modes.end()) {
        present_mode = vk::PresentModeKHR::eMailbox;
    }

    // Query surface extent
    auto capabilities = physical.getSurfaceCapabilitiesKHR(surface);
    extent = capabilities.currentExtent;

    if (capabilities.currentExtent.width == std::numeric_limits<u32>::max()) {
        extent.width = std::clamp(width, capabilities.minImageExtent.width,
                                          capabilities.maxImageExtent.width);
        extent.height = std::clamp(height, capabilities.minImageExtent.height,
                                           capabilities.maxImageExtent.height);
    }

    // Select number of images in swap chain, we prefer one buffer in the background to work on
    image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0) {
        image_count = std::min(image_count, capabilities.maxImageCount);
    }

    // Prefer identity transform if possible
    transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if (!(capabilities.supportedTransforms & transform)) {
        transform = capabilities.currentTransform;
    }
}

} // namespace Vulkan
