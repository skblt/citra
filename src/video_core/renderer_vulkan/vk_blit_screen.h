// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <glm/glm.hpp>
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Core {
class System;
}

namespace Memory {
class MemorySystem;
}

namespace Frontend {
class EmuWindow;
}

namespace VideoCore {
class RasterizerInterface;
}

namespace Layout {
struct FramebufferLayout;
}

namespace Vulkan {

struct ScreenInfo;

class Instance;
class RasterizerVulkan;
class Scheduler;
class Swapchain;
class RenderpassCache;
class DescriptorManager;

struct ScreenInfo {
    vk::ImageView image_view{};
    u32 width{};
    u32 height{};
    Common::Rectangle<f32> texcoords;
};

using Images = std::array<vk::Image, 3>;

struct PresentUniformData {
    glm::mat4 modelview;
    Common::Vec4f i_resolution;
    Common::Vec4f o_resolution;
    int screen_id_l = 0;
    int screen_id_r = 0;
    int layer = 0;
    int reverse_interlaced = 0;

    // Returns an immutable byte view of the uniform data
    auto AsBytes() const {
        return std::as_bytes(std::span{this, 1});
    }
};

constexpr u32 PRESENT_PIPELINES = 3;

class BlitScreen {
public:
    explicit BlitScreen(Frontend::EmuWindow& render_window, const Instance& instance,
                        Scheduler& scheduler, Swapchain& swapchain, RenderpassCache& renderpass_cache,
                        DescriptorManager& desc_manager, std::array<ScreenInfo, 3>& screen_infos);
    ~BlitScreen();

    void Recreate();

    [[nodiscard]] vk::Semaphore Draw(const GPU::Regs::FramebufferConfig& framebuffer,
                                     const vk::Framebuffer& host_framebuffer,
                                     const Layout::FramebufferLayout layout, vk::Extent2D render_area,
                                     bool use_accelerated, u32 screen);

    [[nodiscard]] vk::Semaphore DrawToSwapchain(const GPU::Regs::FramebufferConfig& framebuffer,
                                                bool use_accelerated);

    [[nodiscard]] vk::Framebuffer CreateFramebuffer(const vk::ImageView& image_view,
                                                    vk::Extent2D extent);

    [[nodiscard]] vk::Framebuffer CreateFramebuffer(const vk::ImageView& image_view,
                                                    vk::Extent2D extent, vk::RenderPass& rd);

private:
    void CreateStaticResources();
    void CreateShaders();
    void CreateSemaphores();
    void CreateDescriptorPool();
    void CreateRenderPass();
    vk::RenderPass CreateRenderPassImpl(vk::Format format, bool is_present = true);
    void CreateDescriptorSetLayout();
    void CreateDescriptorSets();
    void CreatePipelineLayout();
    void CreateGraphicsPipeline();
    void CreateSampler();

    void CreateDynamicResources();
    void CreateFramebuffers();

    void RefreshResources(const GPU::Regs::FramebufferConfig& framebuffer);
    void ReleaseRawImages();
    void CreateStagingBuffer(const GPU::Regs::FramebufferConfig& framebuffer);
    void CreateRawImages(const GPU::Regs::FramebufferConfig& framebuffer);

    struct BufferData;

    void UpdateDescriptorSet(std::size_t image_index, bool use_accelerated) const;
    void SetUniformData(BufferData& data, const Layout::FramebufferLayout layout) const;
    void SetVertexData(BufferData& data, const Layout::FramebufferLayout layout) const;

private:
    Frontend::EmuWindow& render_window;
    const Instance& instance;
    Scheduler& scheduler;
    Swapchain& swapchain;
    RenderpassCache& renderpass_cache;
    DescriptorManager& desc_manager;
    Memory::MemorySystem& memory;
    std::array<ScreenInfo, 3>& screen_infos;
    std::size_t image_count;
    PresentUniformData draw_info{};
    StreamBuffer vertex_buffer;

    vk::PipelineLayout pipeline_layout;
    vk::DescriptorSetLayout descriptor_set_layout;
    vk::DescriptorUpdateTemplate update_template;
    std::array<vk::Pipeline, PRESENT_PIPELINES> pipelines;
    std::array<vk::DescriptorSet, PRESENT_PIPELINES> descriptor_sets;
    std::array<vk::ShaderModule, PRESENT_PIPELINES> shaders;
    std::array<vk::Sampler, 2> samplers;
    vk::ShaderModule vertex_shader;
    u32 current_pipeline = 0;
    u32 current_sampler = 0;

    vk::RenderPass renderpass;
    std::vector<vk::Framebuffer> framebuffers;
    std::vector<u64> resource_ticks;
    std::vector<vk::Semaphore> semaphores;
    std::vector<Images> raw_images;
    GPU::Regs::PixelFormat pixel_format;
    u32 raw_width;
    u32 raw_height;
};

} // namespace Vulkan
