// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <bit>
#include "video_core/rasterizer_cache/morton_swizzle.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {

vk::ImageAspectFlags ToVkAspect(VideoCore::SurfaceType type) {
    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        return vk::ImageAspectFlagBits::eColor;
    case VideoCore::SurfaceType::Depth:
        return vk::ImageAspectFlagBits::eDepth;
    case VideoCore::SurfaceType::DepthStencil:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    default:
        UNREACHABLE_MSG("Invalid surface type!");
    }

    return vk::ImageAspectFlagBits::eColor;
}

constexpr u32 STAGING_BUFFER_SIZE = 64 * 1024 * 1024;

TextureRuntime::TextureRuntime(const Instance& instance, TaskScheduler& scheduler,
                               RenderpassCache& renderpass_cache)
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache}, blit_helper{
                                                                                        instance,
                                                                                        scheduler} {

    for (auto& buffer : staging_buffers) {
        buffer = std::make_unique<StagingBuffer>(instance, STAGING_BUFFER_SIZE,
                                                 vk::BufferUsageFlagBits::eTransferSrc |
                                                     vk::BufferUsageFlagBits::eTransferDst);
    }

    auto Register = [this](VideoCore::PixelFormat dest,
                           std::unique_ptr<FormatReinterpreterBase>&& obj) {
        const u32 dst_index = static_cast<u32>(dest);
        return reinterpreters[dst_index].push_back(std::move(obj));
    };

    Register(VideoCore::PixelFormat::RGBA8,
             std::make_unique<D24S8toRGBA8>(instance, scheduler, *this));
}

TextureRuntime::~TextureRuntime() {
    VmaAllocator allocator = instance.GetAllocator();
    vk::Device device = instance.GetDevice();
    device.waitIdle();

    for (const auto& [key, alloc] : texture_recycler) {
        vmaDestroyImage(allocator, alloc.image, alloc.allocation);
        device.destroyImageView(alloc.image_view);
        device.destroyImageView(alloc.base_view);
        if (alloc.depth_view) {
            device.destroyImageView(alloc.depth_view);
            device.destroyImageView(alloc.stencil_view);
        }
        if (alloc.storage_view) {
            device.destroyImageView(alloc.storage_view);
        }
    }

    for (const auto& [key, framebuffer] : clear_framebuffers) {
        device.destroyFramebuffer(framebuffer);
    }

    texture_recycler.clear();
}

StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    const u32 current_slot = scheduler.GetCurrentSlotIndex();
    u32& offset = staging_offsets[current_slot];
    // Depth uploads require 4 byte alignment, doesn't hurt to do it for everyone
    offset = Common::AlignUp(offset, 4);

    if (offset + size > STAGING_BUFFER_SIZE) {
        LOG_CRITICAL(Render_Vulkan, "Staging buffer size exceeded!");
        UNREACHABLE();
    }

    const auto& buffer = staging_buffers[current_slot];
    return StagingData{.buffer = buffer->buffer,
                       .size = size,
                       .mapped = buffer->mapped.subspan(offset, size),
                       .buffer_offset = offset};
}

void TextureRuntime::Finish() {
    scheduler.Submit(SubmitMode::Flush);
}

void TextureRuntime::OnSlotSwitch(u32 new_slot) {
    staging_offsets[new_slot] = 0;
}

ImageAlloc TextureRuntime::Allocate(u32 width, u32 height, VideoCore::PixelFormat format,
                                    VideoCore::TextureType type) {
    const FormatTraits traits = instance.GetTraits(format);
    const vk::ImageAspectFlags aspect = ToVkAspect(VideoCore::GetFormatType(format));

    const bool is_suitable = traits.transfer_support && traits.attachment_support &&
                             (traits.blit_support || aspect & vk::ImageAspectFlagBits::eDepth);
    const vk::Format vk_format = is_suitable ? traits.native : traits.fallback;
    const vk::ImageUsageFlags vk_usage = is_suitable ? traits.usage : GetImageUsage(aspect);

    return Allocate(width, height, format, type, vk_format, vk_usage);
}

ImageAlloc TextureRuntime::Allocate(u32 width, u32 height, VideoCore::PixelFormat pixel_format,
                                    VideoCore::TextureType type, vk::Format format,
                                    vk::ImageUsageFlags usage) {
    ImageAlloc alloc{};
    alloc.format = format;
    alloc.levels = std::bit_width(std::max(width, height));
    alloc.layers = type == VideoCore::TextureType::CubeMap ? 6 : 1;
    alloc.aspect = GetImageAspect(format);

    // The internal format does not provide enough guarantee of texture uniqueness
    // especially when many pixel formats fallback to RGBA8
    ASSERT(pixel_format != VideoCore::PixelFormat::Invalid);
    const HostTextureTag key = {.format = format,
                                .pixel_format = pixel_format,
                                .type = type,
                                .width = width,
                                .height = height};

    // Attempt to recycle an unused allocation
    if (auto it = texture_recycler.find(key); it != texture_recycler.end()) {
        ImageAlloc alloc = std::move(it->second);
        texture_recycler.erase(it);
        return alloc;
    }

    const bool create_storage_view = pixel_format == VideoCore::PixelFormat::RGBA8;

    vk::ImageCreateFlags flags;
    if (type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (create_storage_view) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat;
    }

    const vk::ImageCreateInfo image_info = {.flags = flags,
                                            .imageType = vk::ImageType::e2D,
                                            .format = format,
                                            .extent = {width, height, 1},
                                            .mipLevels = alloc.levels,
                                            .arrayLayers = alloc.layers,
                                            .samples = vk::SampleCountFlagBits::e1,
                                            .usage = usage};

    const VmaAllocationCreateInfo alloc_info = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &alloc.allocation, nullptr);
    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }

    const vk::ImageViewType view_type =
        type == VideoCore::TextureType::CubeMap ? vk::ImageViewType::eCube : vk::ImageViewType::e2D;

    alloc.image = vk::Image{unsafe_image};
    const vk::ImageViewCreateInfo view_info = {.image = alloc.image,
                                               .viewType = view_type,
                                               .format = format,
                                               .subresourceRange = {.aspectMask = alloc.aspect,
                                                                    .baseMipLevel = 0,
                                                                    .levelCount = alloc.levels,
                                                                    .baseArrayLayer = 0,
                                                                    .layerCount = alloc.layers}};

    vk::Device device = instance.GetDevice();
    alloc.image_view = device.createImageView(view_info);

    // Also create a base mip view in case this is used as an attachment
    if (alloc.levels > 1) [[likely]] {
        const vk::ImageViewCreateInfo base_view_info = {
            .image = alloc.image,
            .viewType = view_type,
            .format = format,
            .subresourceRange = {.aspectMask = alloc.aspect,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = alloc.layers}};

        alloc.base_view = device.createImageView(base_view_info);
    }

    // Create seperate depth/stencil views in case this gets reinterpreted with a compute shader
    if (alloc.aspect & vk::ImageAspectFlagBits::eStencil) {
        vk::ImageViewCreateInfo view_info = {
            .image = alloc.image,
            .viewType = view_type,
            .format = format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                 .baseMipLevel = 0,
                                 .levelCount = alloc.levels,
                                 .baseArrayLayer = 0,
                                 .layerCount = alloc.layers}};

        alloc.depth_view = device.createImageView(view_info);
        view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eStencil;
        alloc.stencil_view = device.createImageView(view_info);
    }

    if (create_storage_view) {
        const vk::ImageViewCreateInfo storage_view_info = {
            .image = alloc.image,
            .viewType = view_type,
            .format = vk::Format::eR32Uint,
            .subresourceRange = {.aspectMask = alloc.aspect,
                                 .baseMipLevel = 0,
                                 .levelCount = alloc.levels,
                                 .baseArrayLayer = 0,
                                 .layerCount = alloc.layers}};
        alloc.storage_view = device.createImageView(storage_view_info);
    }

    return alloc;
}

void TextureRuntime::Recycle(const HostTextureTag tag, ImageAlloc&& alloc) {
    texture_recycler.emplace(tag, std::move(alloc));
}

void TextureRuntime::FormatConvert(const Surface& surface, bool upload, std::span<std::byte> source,
                                   std::span<std::byte> dest) {
    if (!NeedsConvertion(surface.pixel_format)) {
        std::memcpy(dest.data(), source.data(), source.size());
        return;
    }

    if (upload) {
        switch (surface.pixel_format) {
        case VideoCore::PixelFormat::RGBA8:
            return Pica::Texture::ConvertABGRToRGBA(source, dest);
        case VideoCore::PixelFormat::RGB8:
            return Pica::Texture::ConvertBGRToRGBA(source, dest);
        default:
            break;
        }
    } else {
        switch (surface.pixel_format) {
        case VideoCore::PixelFormat::RGBA8:
            return Pica::Texture::ConvertABGRToRGBA(source, dest);
        case VideoCore::PixelFormat::RGBA4:
            return Pica::Texture::ConvertRGBA8ToRGBA4(source, dest);
        case VideoCore::PixelFormat::RGB8:
            return Pica::Texture::ConvertRGBAToBGR(source, dest);
        default:
            break;
        }
    }

    LOG_WARNING(Render_Vulkan, "Missing linear format convertion: {} {} {}",
                vk::to_string(surface.traits.native), upload ? "->" : "<-",
                vk::to_string(surface.alloc.format));
}

bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear,
                                  VideoCore::ClearValue value) {
    const vk::ImageAspectFlags aspect = ToVkAspect(surface.type);
    renderpass_cache.ExitRenderpass();

    surface.Transition(vk::ImageLayout::eTransferDstOptimal, clear.texture_level, 1);

    vk::ClearValue clear_value{};
    if (aspect & vk::ImageAspectFlagBits::eColor) {
        clear_value.color = vk::ClearColorValue{
            .float32 =
                std::to_array({value.color[0], value.color[1], value.color[2], value.color[3]})};
    } else if (aspect & vk::ImageAspectFlagBits::eDepth ||
               aspect & vk::ImageAspectFlagBits::eStencil) {
        clear_value.depthStencil =
            vk::ClearDepthStencilValue{.depth = value.depth, .stencil = value.stencil};
    }

    // For full clears we can use vkCmdClearColorImage/vkCmdClearDepthStencilImage
    if (clear.texture_rect == surface.GetScaledRect()) {
        const vk::ImageSubresourceRange range = {.aspectMask = aspect,
                                                 .baseMipLevel = clear.texture_level,
                                                 .levelCount = 1,
                                                 .baseArrayLayer = 0,
                                                 .layerCount = 1};

        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        if (aspect & vk::ImageAspectFlagBits::eColor) {
            command_buffer.clearColorImage(surface.alloc.image,
                                           vk::ImageLayout::eTransferDstOptimal, clear_value.color,
                                           range);
        } else if (aspect & vk::ImageAspectFlagBits::eDepth ||
                   aspect & vk::ImageAspectFlagBits::eStencil) {
            command_buffer.clearDepthStencilImage(surface.alloc.image,
                                                  vk::ImageLayout::eTransferDstOptimal,
                                                  clear_value.depthStencil, range);
        }
    } else {
        // For partial clears we begin a clear renderpass with the appropriate render area
        vk::RenderPass clear_renderpass{};
        if (aspect & vk::ImageAspectFlagBits::eColor) {
            clear_renderpass = renderpass_cache.GetRenderpass(
                surface.pixel_format, VideoCore::PixelFormat::Invalid, true);
            surface.Transition(vk::ImageLayout::eColorAttachmentOptimal, 0, 1);
        } else if (aspect & vk::ImageAspectFlagBits::eDepth ||
                   aspect & vk::ImageAspectFlagBits::eStencil) {
            clear_renderpass = renderpass_cache.GetRenderpass(VideoCore::PixelFormat::Invalid,
                                                              surface.pixel_format, true);
            surface.Transition(vk::ImageLayout::eDepthStencilAttachmentOptimal, 0, 1);
        }

        const vk::ImageView framebuffer_view = surface.GetFramebufferView();

        auto [it, new_framebuffer] =
            clear_framebuffers.try_emplace(framebuffer_view, vk::Framebuffer{});
        if (new_framebuffer) {
            const vk::FramebufferCreateInfo framebuffer_info = {.renderPass = clear_renderpass,
                                                                .attachmentCount = 1,
                                                                .pAttachments = &framebuffer_view,
                                                                .width = surface.GetScaledWidth(),
                                                                .height = surface.GetScaledHeight(),
                                                                .layers = 1};

            vk::Device device = instance.GetDevice();
            it->second = device.createFramebuffer(framebuffer_info);
        }

        const vk::RenderPassBeginInfo clear_begin_info = {
            .renderPass = clear_renderpass,
            .framebuffer = it->second,
            .renderArea = vk::Rect2D{.offset = {static_cast<s32>(clear.texture_rect.left),
                                                static_cast<s32>(clear.texture_rect.bottom)},
                                     .extent = {clear.texture_rect.GetWidth(),
                                                clear.texture_rect.GetHeight()}},
            .clearValueCount = 1,
            .pClearValues = &clear_value};

        renderpass_cache.EnterRenderpass(clear_begin_info);
        renderpass_cache.ExitRenderpass();
    }

    return true;
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureCopy& copy) {
    renderpass_cache.ExitRenderpass();

    source.Transition(vk::ImageLayout::eTransferSrcOptimal, copy.src_level, 1);
    dest.Transition(vk::ImageLayout::eTransferDstOptimal, copy.dst_level, 1);

    const vk::ImageCopy image_copy = {
        .srcSubresource = {.aspectMask = ToVkAspect(source.type),
                           .mipLevel = copy.src_level,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
        .srcOffset = {static_cast<s32>(copy.src_offset.x), static_cast<s32>(copy.src_offset.y), 0},
        .dstSubresource = {.aspectMask = ToVkAspect(dest.type),
                           .mipLevel = copy.dst_level,
                           .baseArrayLayer = 0,
                           .layerCount = 1},
        .dstOffset = {static_cast<s32>(copy.dst_offset.x), static_cast<s32>(copy.dst_offset.y), 0},
        .extent = {copy.extent.width, copy.extent.height, 1}};

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.copyImage(source.alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                             dest.alloc.image, vk::ImageLayout::eTransferDstOptimal, image_copy);

    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    renderpass_cache.ExitRenderpass();

    source.Transition(vk::ImageLayout::eTransferSrcOptimal, blit.src_level, 1);
    dest.Transition(vk::ImageLayout::eTransferDstOptimal, blit.dst_level, 1);

    const std::array source_offsets = {vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                                                    static_cast<s32>(blit.src_rect.bottom), 0},
                                       vk::Offset3D{static_cast<s32>(blit.src_rect.right),
                                                    static_cast<s32>(blit.src_rect.top), 1}};

    const std::array dest_offsets = {vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                                                  static_cast<s32>(blit.dst_rect.bottom), 0},
                                     vk::Offset3D{static_cast<s32>(blit.dst_rect.right),
                                                  static_cast<s32>(blit.dst_rect.top), 1}};

    const vk::ImageBlit blit_area = {.srcSubresource = {.aspectMask = ToVkAspect(source.type),
                                                        .mipLevel = blit.src_level,
                                                        .baseArrayLayer = blit.src_layer,
                                                        .layerCount = 1},
                                     .srcOffsets = source_offsets,
                                     .dstSubresource = {.aspectMask = ToVkAspect(dest.type),
                                                        .mipLevel = blit.dst_level,
                                                        .baseArrayLayer = blit.dst_layer,
                                                        .layerCount = 1},
                                     .dstOffsets = dest_offsets};
    // Don't use linear filtering on depth attachments
    const vk::Filter filtering =
        blit_area.srcSubresource.aspectMask & vk::ImageAspectFlagBits::eDepth ||
                blit_area.dstSubresource.aspectMask & vk::ImageAspectFlagBits::eDepth
            ? vk::Filter::eNearest
            : vk::Filter::eLinear;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.blitImage(source.alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                             dest.alloc.image, vk::ImageLayout::eTransferDstOptimal, blit_area,
                             filtering);

    return true;
}

void TextureRuntime::GenerateMipmaps(Surface& surface, u32 max_level) {
    renderpass_cache.ExitRenderpass();

    // TODO: Investigate AMD single pass downsampler
    s32 current_width = surface.GetScaledWidth();
    s32 current_height = surface.GetScaledHeight();

    const u32 levels = std::bit_width(std::max(surface.width, surface.height));
    vk::ImageAspectFlags aspect = ToVkAspect(surface.type);
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    for (u32 i = 1; i < levels; i++) {
        surface.Transition(vk::ImageLayout::eTransferSrcOptimal, i - 1, 1);
        surface.Transition(vk::ImageLayout::eTransferDstOptimal, i, 1);

        const std::array source_offsets = {vk::Offset3D{0, 0, 0},
                                           vk::Offset3D{current_width, current_height, 1}};

        const std::array dest_offsets = {
            vk::Offset3D{0, 0, 0}, vk::Offset3D{current_width > 1 ? current_width / 2 : 1,
                                                current_height > 1 ? current_height / 2 : 1, 1}};

        const vk::ImageBlit blit_area = {.srcSubresource = {.aspectMask = aspect,
                                                            .mipLevel = i - 1,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1},
                                         .srcOffsets = source_offsets,
                                         .dstSubresource = {.aspectMask = aspect,
                                                            .mipLevel = i,
                                                            .baseArrayLayer = 0,
                                                            .layerCount = 1},
                                         .dstOffsets = dest_offsets};

        command_buffer.blitImage(surface.alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                                 surface.alloc.image, vk::ImageLayout::eTransferDstOptimal,
                                 blit_area, vk::Filter::eLinear);
    }
}

const ReinterpreterList& TextureRuntime::GetPossibleReinterpretations(
    VideoCore::PixelFormat dest_format) const {
    return reinterpreters[static_cast<u32>(dest_format)];
}

bool TextureRuntime::NeedsConvertion(VideoCore::PixelFormat format) const {
    const FormatTraits traits = instance.GetTraits(format);
    const VideoCore::SurfaceType type = VideoCore::GetFormatType(format);
    return type == VideoCore::SurfaceType::Color &&
           (format == VideoCore::PixelFormat::RGBA8 || !traits.blit_support ||
            !traits.attachment_support);
}

void TextureRuntime::Transition(vk::CommandBuffer command_buffer, ImageAlloc& alloc,
                                vk::ImageLayout new_layout, u32 level, u32 level_count) {
    LayoutTracker& tracker = alloc.tracker;
    if (tracker.IsRangeEqual(new_layout, level, level_count) || !alloc.image) {
        return;
    }

    renderpass_cache.ExitRenderpass();

    struct LayoutInfo {
        vk::AccessFlags access;
        vk::PipelineStageFlags stage;
    };

    // Get optimal transition settings for every image layout. Settings taken from Dolphin
    auto GetLayoutInfo = [](vk::ImageLayout layout) -> LayoutInfo {
        LayoutInfo info;
        switch (layout) {
        case vk::ImageLayout::eUndefined:
            // Layout undefined therefore contents undefined, and we don't care what happens to it.
            info.access = vk::AccessFlagBits::eNone;
            info.stage = vk::PipelineStageFlagBits::eTopOfPipe;
            break;
        case vk::ImageLayout::ePreinitialized:
            // Image has been pre-initialized by the host, so ensure all writes have completed.
            info.access = vk::AccessFlagBits::eHostWrite;
            info.stage = vk::PipelineStageFlagBits::eHost;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            // Image was being used as a color attachment, so ensure all writes have completed.
            info.access = vk::AccessFlagBits::eColorAttachmentRead |
                          vk::AccessFlagBits::eColorAttachmentWrite;
            info.stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            // Image was being used as a depthstencil attachment, so ensure all writes have
            // completed.
            info.access = vk::AccessFlagBits::eDepthStencilAttachmentRead |
                          vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            info.stage = vk::PipelineStageFlagBits::eEarlyFragmentTests |
                         vk::PipelineStageFlagBits::eLateFragmentTests;
            break;
        case vk::ImageLayout::ePresentSrcKHR:
            info.access = vk::AccessFlagBits::eNone;
            info.stage = vk::PipelineStageFlagBits::eBottomOfPipe;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            // Image was being used as a shader resource, make sure all reads have finished.
            info.access = vk::AccessFlagBits::eShaderRead;
            info.stage = vk::PipelineStageFlagBits::eFragmentShader;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            // Image was being used as a copy source, ensure all reads have finished.
            info.access = vk::AccessFlagBits::eTransferRead;
            info.stage = vk::PipelineStageFlagBits::eTransfer;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            // Image was being used as a copy destination, ensure all writes have finished.
            info.access = vk::AccessFlagBits::eTransferWrite;
            info.stage = vk::PipelineStageFlagBits::eTransfer;
            break;
        case vk::ImageLayout::eGeneral:
            info.access = vk::AccessFlagBits::eInputAttachmentRead;
            info.stage = vk::PipelineStageFlagBits::eColorAttachmentOutput |
                         vk::PipelineStageFlagBits::eFragmentShader |
                         vk::PipelineStageFlagBits::eComputeShader;
            break;
        case vk::ImageLayout::eDepthStencilReadOnlyOptimal:
            // Image is going to be sampled from a compute shader
            info.access = vk::AccessFlagBits::eShaderRead;
            info.stage = vk::PipelineStageFlagBits::eComputeShader;
            break;
        default:
            LOG_CRITICAL(Render_Vulkan, "Unhandled vulkan image layout {}\n", layout);
            UNREACHABLE();
        }

        return info;
    };

    LayoutInfo dest = GetLayoutInfo(new_layout);
    tracker.ForEachLayoutRange(
        level, level_count, new_layout, [&](u32 start, u32 count, vk::ImageLayout old_layout) {
            LayoutInfo source = GetLayoutInfo(old_layout);
            const vk::ImageMemoryBarrier barrier = {
                .srcAccessMask = source.access,
                .dstAccessMask = dest.access,
                .oldLayout = old_layout,
                .newLayout = new_layout,
                .image = alloc.image,
                .subresourceRange = {.aspectMask = alloc.aspect,
                                     .baseMipLevel = start,
                                     .levelCount = count,
                                     .baseArrayLayer = 0,
                                     .layerCount = alloc.layers}};

            command_buffer.pipelineBarrier(source.stage, dest.stage,
                                           vk::DependencyFlagBits::eByRegion, {}, {}, barrier);
        });

    tracker.SetLayout(new_layout, level, level_count);
    for (u32 i = 0; i < level_count; i++) {
        ASSERT(alloc.tracker.GetLayout(level + i) == new_layout);
    }
}

Surface::Surface(TextureRuntime& runtime)
    : runtime{runtime}, instance{runtime.GetInstance()}, scheduler{runtime.GetScheduler()} {}

Surface::Surface(const VideoCore::SurfaceParams& params, TextureRuntime& runtime)
    : VideoCore::SurfaceBase<Surface>{params}, runtime{runtime}, instance{runtime.GetInstance()},
      scheduler{runtime.GetScheduler()}, traits{instance.GetTraits(pixel_format)} {

    if (pixel_format != VideoCore::PixelFormat::Invalid) {
        alloc = runtime.Allocate(GetScaledWidth(), GetScaledHeight(), params.pixel_format,
                                 texture_type);
    }
}

Surface::Surface(const VideoCore::SurfaceParams& params, vk::Format format,
                 vk::ImageUsageFlags usage, TextureRuntime& runtime)
    : VideoCore::SurfaceBase<Surface>{params}, runtime{runtime}, instance{runtime.GetInstance()},
      scheduler{runtime.GetScheduler()} {
    if (format != vk::Format::eUndefined) {
        alloc = runtime.Allocate(GetScaledWidth(), GetScaledHeight(), pixel_format, texture_type,
                                 format, usage);
    }
}

Surface::~Surface() {
    if (pixel_format != VideoCore::PixelFormat::Invalid) {
        const HostTextureTag tag = {.format = alloc.format,
                                    .pixel_format = pixel_format,
                                    .type = texture_type,
                                    .width = GetScaledWidth(),
                                    .height = GetScaledHeight()};

        runtime.Recycle(tag, std::move(alloc));
    }
}

void Surface::Transition(vk::ImageLayout new_layout, u32 level, u32 level_count) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    runtime.Transition(command_buffer, alloc, new_layout, level, level_count);
}

MICROPROFILE_DEFINE(Vulkan_Upload, "VulkanSurface", "Texture Upload", MP_RGB(128, 192, 64));
void Surface::Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging) {
    MICROPROFILE_SCOPE(Vulkan_Upload);

    if (type == VideoCore::SurfaceType::DepthStencil && !traits.blit_support) {
        LOG_ERROR(Render_Vulkan, "Depth blit unsupported by hardware, ignoring");
        return;
    }

    runtime.renderpass_cache.ExitRenderpass();

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledUpload(upload, staging);
    } else {
        u32 region_count = 0;
        std::array<vk::BufferImageCopy, 2> copy_regions;

        const VideoCore::Rect2D rect = upload.texture_rect;
        vk::BufferImageCopy copy_region = {
            .bufferOffset = staging.buffer_offset + upload.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource = {.aspectMask = alloc.aspect,
                                 .mipLevel = upload.texture_level,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1}};

        if (alloc.aspect & vk::ImageAspectFlagBits::eColor) {
            copy_regions[region_count++] = copy_region;
        } else if (alloc.aspect & vk::ImageAspectFlagBits::eDepth) {
            copy_region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
            copy_regions[region_count++] = copy_region;

            if (alloc.aspect & vk::ImageAspectFlagBits::eStencil) {
                copy_region.bufferOffset += UnpackDepthStencil(staging);
                copy_region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
                copy_regions[region_count++] = copy_region;
            }
        }

        Transition(vk::ImageLayout::eTransferDstOptimal, upload.texture_level, 1);

        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        command_buffer.copyBufferToImage(staging.buffer, alloc.image,
                                         vk::ImageLayout::eTransferDstOptimal, region_count,
                                         copy_regions.data());
    }

    InvalidateAllWatcher();

    // Lock this data until the next scheduler switch
    const u32 current_slot = scheduler.GetCurrentSlotIndex();
    runtime.staging_offsets[current_slot] += staging.size;
}

MICROPROFILE_DEFINE(Vulkan_Download, "VulkanSurface", "Texture Download", MP_RGB(128, 192, 64));
void Surface::Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging) {
    MICROPROFILE_SCOPE(Vulkan_Download);

    runtime.renderpass_cache.ExitRenderpass();

    // For depth stencil downloads always use the compute shader fallback
    // to avoid having the interleave the data later. These should(?) be
    // uncommon anyways and the perf hit is very small
    if (type == VideoCore::SurfaceType::DepthStencil) {
        return DepthStencilDownload(download, staging);
    }

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledDownload(download, staging);
    } else {
        const VideoCore::Rect2D rect = download.texture_rect;
        const vk::BufferImageCopy copy_region = {
            .bufferOffset = staging.buffer_offset + download.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource = {.aspectMask = alloc.aspect,
                                 .mipLevel = download.texture_level,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1}};

        Transition(vk::ImageLayout::eTransferSrcOptimal, download.texture_level, 1);

        // Copy pixel data to the staging buffer
        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        command_buffer.copyImageToBuffer(alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                                         staging.buffer, copy_region);
    }

    // Lock this data until the next scheduler switch
    const u32 current_slot = scheduler.GetCurrentSlotIndex();
    runtime.staging_offsets[current_slot] += staging.size;
}

u32 Surface::GetInternalBytesPerPixel() const {
    // Request 5 bytes for D24S8 as well because we can use the
    // extra space when deinterleaving the data during upload
    if (alloc.format == vk::Format::eD24UnormS8Uint) {
        return 5;
    }

    return vk::blockSize(alloc.format);
}

void Surface::ScaledUpload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging) {
    const u32 rect_width = upload.texture_rect.GetWidth();
    const u32 rect_height = upload.texture_rect.GetHeight();
    const auto scaled_rect = upload.texture_rect * res_scale;
    const auto unscaled_rect = VideoCore::Rect2D{0, rect_height, rect_width, 0};

    SurfaceParams unscaled_params = *this;
    unscaled_params.width = rect_width;
    unscaled_params.stride = rect_width;
    unscaled_params.height = rect_height;
    unscaled_params.res_scale = 1;
    Surface unscaled_surface{unscaled_params, runtime};

    const VideoCore::BufferTextureCopy unscaled_upload = {.buffer_offset = upload.buffer_offset,
                                                          .buffer_size = upload.buffer_size,
                                                          .texture_rect = unscaled_rect};

    unscaled_surface.Upload(unscaled_upload, staging);

    const VideoCore::TextureBlit blit = {.src_level = 0,
                                         .dst_level = upload.texture_level,
                                         .src_layer = 0,
                                         .dst_layer = 0,
                                         .src_rect = unscaled_rect,
                                         .dst_rect = scaled_rect};

    runtime.BlitTextures(unscaled_surface, *this, blit);
}

void Surface::ScaledDownload(const VideoCore::BufferTextureCopy& download,
                             const StagingData& staging) {
    const u32 rect_width = download.texture_rect.GetWidth();
    const u32 rect_height = download.texture_rect.GetHeight();
    const VideoCore::Rect2D scaled_rect = download.texture_rect * res_scale;
    const VideoCore::Rect2D unscaled_rect = VideoCore::Rect2D{0, rect_height, rect_width, 0};

    // Allocate an unscaled texture that fits the download rectangle to use as a blit destination
    SurfaceParams unscaled_params = *this;
    unscaled_params.width = rect_width;
    unscaled_params.stride = rect_width;
    unscaled_params.height = rect_height;
    unscaled_params.res_scale = 1;
    Surface unscaled_surface{unscaled_params, runtime};

    const VideoCore::TextureBlit blit = {.src_level = download.texture_level,
                                         .dst_level = 0,
                                         .src_layer = 0,
                                         .dst_layer = 0,
                                         .src_rect = scaled_rect,
                                         .dst_rect = unscaled_rect};

    // Blit the scaled rectangle to the unscaled texture
    runtime.BlitTextures(*this, unscaled_surface, blit);

    const VideoCore::BufferTextureCopy unscaled_download = {.buffer_offset = download.buffer_offset,
                                                            .buffer_size = download.buffer_size,
                                                            .texture_rect = unscaled_rect,
                                                            .texture_level = 0};

    unscaled_surface.Download(unscaled_download, staging);
}

void Surface::DepthStencilDownload(const VideoCore::BufferTextureCopy& download,
                                   const StagingData& staging) {
    const u32 rect_width = download.texture_rect.GetWidth();
    const u32 rect_height = download.texture_rect.GetHeight();
    const VideoCore::Rect2D scaled_rect = download.texture_rect * res_scale;
    const VideoCore::Rect2D unscaled_rect = VideoCore::Rect2D{0, rect_height, rect_width, 0};
    const VideoCore::Rect2D r32_scaled_rect =
        VideoCore::Rect2D{0, scaled_rect.GetHeight(), scaled_rect.GetWidth(), 0};

    // For depth downloads create an R32UI surface and use a compute shader for convert.
    // Then we blit and download that surface.
    // NOTE: We don't need to set pixel format here since R32Uint automatically gives us
    // a storage view. Also the D24S8 creates a unique cache key for it
    SurfaceParams r32_params = *this;
    r32_params.width = scaled_rect.GetWidth();
    r32_params.stride = scaled_rect.GetWidth();
    r32_params.height = scaled_rect.GetHeight();
    r32_params.type = VideoCore::SurfaceType::Color;
    r32_params.res_scale = 1;
    Surface r32_surface{r32_params, vk::Format::eR32Uint,
                        vk::ImageUsageFlagBits::eTransferSrc |
                            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage,
                        runtime};

    const VideoCore::TextureBlit blit = {.src_level = download.texture_level,
                                         .dst_level = 0,
                                         .src_layer = 0,
                                         .dst_layer = 0,
                                         .src_rect = scaled_rect,
                                         .dst_rect = r32_scaled_rect};

    runtime.blit_helper.BlitD24S8ToR32(*this, r32_surface, blit);

    // Blit the upper mip level to the lower one to scale without additional allocations
    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        const VideoCore::TextureBlit r32_blit = {.src_level = 0,
                                                 .dst_level = 1,
                                                 .src_layer = 0,
                                                 .dst_layer = 0,
                                                 .src_rect = r32_scaled_rect,
                                                 .dst_rect = unscaled_rect};

        runtime.BlitTextures(r32_surface, r32_surface, r32_blit);
    }

    const VideoCore::BufferTextureCopy r32_download = {.buffer_offset = download.buffer_offset,
                                                       .buffer_size = download.buffer_size,
                                                       .texture_rect = unscaled_rect,
                                                       .texture_level = is_scaled ? 1u : 0u};

    r32_surface.Download(r32_download, staging);
}

u32 Surface::UnpackDepthStencil(const StagingData& data) {
    u32 depth_offset = 0;
    u32 stencil_offset = 4 * data.size / 5;
    const auto& mapped = data.mapped;

    switch (alloc.format) {
    case vk::Format::eD24UnormS8Uint: {
        for (; stencil_offset < data.size; depth_offset += 4) {
            std::byte* ptr = mapped.data() + depth_offset;
            const u32 d24s8 = VideoCore::MakeInt<u32>(ptr);
            const u32 d24 = d24s8 >> 8;
            mapped[stencil_offset] = static_cast<std::byte>(d24s8 & 0xFF);
            std::memcpy(ptr, &d24, 4);
            stencil_offset++;
        }
        break;
    }
    default:
        LOG_ERROR(Render_Vulkan, "Unimplemtend convertion for depth format {}",
                  vk::to_string(alloc.format));
        UNREACHABLE();
    }

    ASSERT(depth_offset == 4 * data.size / 5);
    return depth_offset;
}

} // namespace Vulkan
