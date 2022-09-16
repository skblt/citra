// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class TaskScheduler;

constexpr u32 MAX_COLOR_FORMATS = 5;
constexpr u32 MAX_DEPTH_FORMATS = 4;

class RenderpassCache {
public:
    RenderpassCache(const Instance& instance, TaskScheduler& scheduler);
    ~RenderpassCache();

    /// Begins a new renderpass only when no other renderpass is currently active
    void EnterRenderpass(const vk::RenderPassBeginInfo begin_info);

    /// Exits from any currently active renderpass instance
    void ExitRenderpass();

    /// Returns the renderpass associated with the color-depth format pair
    [[nodiscard]] vk::RenderPass GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                               bool is_clear) const;

    /// Returns the swapchain clear renderpass
    [[nodiscard]] vk::RenderPass GetPresentRenderpass() const {
        return present_renderpass;
    }

    /// Creates the renderpass used when rendering to the swapchain
    void CreatePresentRenderpass(vk::Format format);

private:
    /// Creates a renderpass configured appropriately and stores it in cached_renderpasses
    vk::RenderPass CreateRenderPass(vk::Format color, vk::Format depth, vk::AttachmentLoadOp load_op,
                                    vk::ImageLayout initial_layout, vk::ImageLayout final_layout) const;

private:
    const Instance& instance;
    TaskScheduler& scheduler;

    vk::RenderPass active_renderpass = VK_NULL_HANDLE;
    vk::RenderPass present_renderpass{};
    vk::RenderPass cached_renderpasses[MAX_COLOR_FORMATS+1][MAX_DEPTH_FORMATS+1][2];
};

} // namespace Vulkan
