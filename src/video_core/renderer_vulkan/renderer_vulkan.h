// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <glm/glm.hpp>
#include "common/common_types.h"
#include "common/math_util.h"
#include "core/hw/gpu.h"
#include "video_core/renderer_base.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_descriptor_manager.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Layout {
struct FramebufferLayout;
}

namespace Vulkan {

class RasterizerVulkan;

class RendererVulkan : public RendererBase {
public:
    RendererVulkan(Frontend::EmuWindow& window);
    ~RendererVulkan() override;

    VideoCore::ResultStatus Init() override;
    VideoCore::RasterizerInterface* Rasterizer() override;
    void ShutDown() override;
    void SwapBuffers() override;
    void TryPresent(int timeout_ms) override {}
    void PrepareVideoDumping() override {}
    void CleanupVideoDumping() override {}
    void Sync() override;
    void FlushBuffers();

private:
    void ReloadSampler();
    void ReloadPipeline();
    void CompileShaders();
    void BuildLayouts();
    void BuildPipelines();
    void ConfigureFramebufferTexture(TextureInfo& texture,
                                     const GPU::Regs::FramebufferConfig& framebuffer);
    void ConfigureRenderPipeline();
    void PrepareRendertarget();
    void BeginRendering();

    void DrawScreens(const Layout::FramebufferLayout& layout, bool flipped);
    void DrawSingleScreenRotated(u32 screen_id, float x, float y, float w, float h);
    void DrawSingleScreen(u32 screen_id, float x, float y, float w, float h);
    void DrawSingleScreenStereoRotated(u32 screen_id_l, u32 screen_id_r, float x, float y, float w,
                                       float h);
    void DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r, float x, float y, float w,
                                float h);

    void UpdateFramerate();

    /// Loads framebuffer from emulated memory into the display information structure
    void LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                            ScreenInfo& screen_info, bool right_eye);

private:
    Instance instance;
    Scheduler scheduler;
    RenderpassCache renderpass_cache;
    DescriptorManager desc_manager;
    TextureRuntime runtime;
    Swapchain swapchain;
    RasterizerVulkan rasterizer;

    // Display information for top and bottom screens respectively
    std::array<ScreenInfo, 3> screen_infos{};
};

} // namespace Vulkan
