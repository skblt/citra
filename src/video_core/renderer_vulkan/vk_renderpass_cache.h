// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <cstring>
#include <variant>
#include "common/hash.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;
class Surface;

struct FramebufferInfo {
    vk::ImageView color;
    vk::ImageView depth;
    u32 width = 1;
    u32 height = 1;

    auto operator<=>(const FramebufferInfo&) const noexcept = default;
};

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::FramebufferInfo> {
    std::size_t operator()(const Vulkan::FramebufferInfo& info) const noexcept {
        return Common::ComputeStructHash64(info);
    }
};
} // namespace std

namespace Vulkan {

class RenderpassCache {
    static constexpr u32 MAX_COLOR_FORMATS = 5;
    static constexpr u32 MAX_DEPTH_FORMATS = 4;

public:
    RenderpassCache(const Instance& instance, Scheduler& scheduler);
    ~RenderpassCache();

    /// Begins a new renderpass only when no other renderpass is currently active
    void EnterRenderpass(Surface* const color, Surface* const depth_stencil, vk::Rect2D render_area,
                         bool do_clear = false, vk::ClearValue clear = {});

    /// Exits from any currently active renderpass instance
    void ExitRenderpass();

    /// Creates the renderpass used when rendering to the swapchain
    void CreatePresentRenderpass(vk::Format format);

    /// Returns the renderpass associated with the color-depth format pair
    vk::RenderPass GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                 bool is_clear);

    /// Returns the swapchain clear renderpass
    [[nodiscard]] vk::RenderPass GetPresentRenderpass() const {
        return present_renderpass;
    }

private:
    /// Begins a new rendering scope using dynamic rendering
    void BeginRendering(Surface* const color, Surface* const depth_stencil, vk::Rect2D render_area,
                        bool do_clear, vk::ClearValue clear);

    /// Creates a renderpass configured appropriately and stores it in cached_renderpasses
    vk::RenderPass CreateRenderPass(vk::Format color, vk::Format depth,
                                    vk::AttachmentLoadOp load_op, vk::ImageLayout initial_layout,
                                    vk::ImageLayout final_layout) const;

    /// Creates a new Vulkan framebuffer object
    vk::Framebuffer CreateFramebuffer(const FramebufferInfo& info, vk::RenderPass renderpass);

private:
    struct RenderpassState {
        vk::RenderPass renderpass;
        vk::Framebuffer framebuffer;
        vk::Rect2D render_area;
        vk::ClearValue clear;

        [[nodiscard]] bool operator==(const RenderpassState& other) const {
            return std::memcmp(this, &other, sizeof(RenderpassState)) == 0;
        }
    };

    struct RenderingState {
        vk::ImageView color_view;
        vk::ImageView depth_view;
        vk::Rect2D render_area;
        vk::ClearValue clear;
        bool do_clear;

        [[nodiscard]] bool operator==(const RenderpassState& other) const {
            return std::memcmp(this, &other, sizeof(RenderpassState)) == 0;
        }
    };

    const Instance& instance;
    Scheduler& scheduler;
    vk::RenderPass present_renderpass{};
    vk::RenderPass cached_renderpasses[MAX_COLOR_FORMATS + 1][MAX_DEPTH_FORMATS + 1][2];
    std::unordered_map<FramebufferInfo, vk::Framebuffer> framebuffers;
    bool rendering = false;
    bool dynamic_rendering = false;
    u32 cmd_count{};
    u64 state_hash{};
};

} // namespace Vulkan
