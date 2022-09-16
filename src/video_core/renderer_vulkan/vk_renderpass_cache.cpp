// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

namespace Vulkan {

vk::Format ToVkFormatColor(u32 index) {
    switch (index) {
    case 0: return vk::Format::eR8G8B8A8Unorm;
    case 1: return vk::Format::eR8G8B8Unorm;
    case 2: return vk::Format::eR5G5B5A1UnormPack16;
    case 3: return vk::Format::eR5G6B5UnormPack16;
    case 4: return vk::Format::eR4G4B4A4UnormPack16;
    default: return vk::Format::eUndefined;
    }
}

vk::Format ToVkFormatDepth(u32 index) {
    switch (index) {
    case 0: return vk::Format::eD16Unorm;
    case 1: return vk::Format::eX8D24UnormPack32;
    // Notice the similar gap in PixelFormat
    case 3: return vk::Format::eD24UnormS8Uint;
    default: return vk::Format::eUndefined;
    }
}

RenderpassCache::RenderpassCache(const Instance& instance, TaskScheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {
    // Pre-create all needed renderpasses by the renderer
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            const vk::Format color_format =
                    instance.GetFormatAlternative(ToVkFormatColor(color));
            const vk::Format depth_stencil_format =
                    instance.GetFormatAlternative(ToVkFormatDepth(depth));

            if (color_format == vk::Format::eUndefined &&
                    depth_stencil_format == vk::Format::eUndefined) {
                continue;
            }

            cached_renderpasses[color][depth][0] = CreateRenderPass(color_format, depth_stencil_format,
                                                                    vk::AttachmentLoadOp::eLoad,
                                                                    vk::ImageLayout::eColorAttachmentOptimal,
                                                                    vk::ImageLayout::eColorAttachmentOptimal);
            cached_renderpasses[color][depth][1] = CreateRenderPass(color_format, depth_stencil_format,
                                                                    vk::AttachmentLoadOp::eClear,
                                                                    vk::ImageLayout::eColorAttachmentOptimal,
                                                                    vk::ImageLayout::eColorAttachmentOptimal);
        }
    }
}

RenderpassCache::~RenderpassCache() {
    vk::Device device = instance.GetDevice();
    for (u32 color = 0; color <= MAX_COLOR_FORMATS; color++) {
        for (u32 depth = 0; depth <= MAX_DEPTH_FORMATS; depth++) {
            if (vk::RenderPass load_pass = cached_renderpasses[color][depth][0]; load_pass) {
                device.destroyRenderPass(load_pass);
            }

            if (vk::RenderPass clear_pass = cached_renderpasses[color][depth][1]; clear_pass) {
                device.destroyRenderPass(clear_pass);
            }
        }
    }

    device.destroyRenderPass(present_renderpass);
}

void RenderpassCache::EnterRenderpass(const vk::RenderPassBeginInfo begin_info) {
    if (active_renderpass == begin_info.renderPass) {
        return;
    }

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    if (active_renderpass) {
        command_buffer.endRenderPass();
    }

    command_buffer.beginRenderPass(begin_info, vk::SubpassContents::eInline);
    active_renderpass = begin_info.renderPass;
}

void RenderpassCache::ExitRenderpass() {
    if (!active_renderpass) {
        return;
    }

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.endRenderPass();
    active_renderpass = VK_NULL_HANDLE;
}

void RenderpassCache::CreatePresentRenderpass(vk::Format format) {
    if (!present_renderpass) {
        present_renderpass = CreateRenderPass(format, vk::Format::eUndefined,
                                              vk::AttachmentLoadOp::eClear,
                                              vk::ImageLayout::eUndefined,
                                              vk::ImageLayout::ePresentSrcKHR);
    }
}

vk::RenderPass RenderpassCache::GetRenderpass(VideoCore::PixelFormat color, VideoCore::PixelFormat depth,
                                              bool is_clear) const {
    const u32 color_index =
            color == VideoCore::PixelFormat::Invalid ? MAX_COLOR_FORMATS : static_cast<u32>(color);
    const u32 depth_index =
            depth == VideoCore::PixelFormat::Invalid ? MAX_DEPTH_FORMATS : (static_cast<u32>(depth) - 14);

    ASSERT(color_index <= MAX_COLOR_FORMATS && depth_index <= MAX_DEPTH_FORMATS);
    return cached_renderpasses[color_index][depth_index][is_clear];
}

vk::RenderPass RenderpassCache::CreateRenderPass(vk::Format color, vk::Format depth, vk::AttachmentLoadOp load_op,
                                                 vk::ImageLayout initial_layout, vk::ImageLayout final_layout) const {
    // Define attachments
    u32 attachment_count = 0;
    std::array<vk::AttachmentDescription, 2> attachments;

    bool use_color = false;
    vk::AttachmentReference color_attachment_ref{};
    bool use_depth = false;
    vk::AttachmentReference depth_attachment_ref{};

    if (color != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = color,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = initial_layout,
            .finalLayout = final_layout
        };

        color_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eColorAttachmentOptimal
       };

        use_color = true;
    }

    if (depth != vk::Format::eUndefined) {
        attachments[attachment_count] = vk::AttachmentDescription{
            .format = depth,
            .loadOp = load_op,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eLoad,
            .stencilStoreOp = vk::AttachmentStoreOp::eStore,
            .initialLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        depth_attachment_ref = vk::AttachmentReference{
            .attachment = attachment_count++,
            .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal
        };

        use_depth = true;
    }

    // We also require only one subpass
    const vk::SubpassDescription subpass = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = use_color ? 1u : 0u,
        .pColorAttachments = &color_attachment_ref,
        .pResolveAttachments = 0,
        .pDepthStencilAttachment = use_depth ? &depth_attachment_ref : nullptr
    };

    const vk::RenderPassCreateInfo renderpass_info = {
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 0,
        .pDependencies = nullptr
    };

    // Create the renderpass
    vk::Device device = instance.GetDevice();
    return device.createRenderPass(renderpass_info);
}

} // namespace VideoCore::Vulkan
