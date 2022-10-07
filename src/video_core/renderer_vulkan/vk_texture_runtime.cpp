// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

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
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache} {

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
    }

    for (const auto& [key, framebuffer] : clear_framebuffers) {
        device.destroyFramebuffer(framebuffer);
    }

    texture_recycler.clear();
}

StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    const u32 current_slot = scheduler.GetCurrentSlotIndex();
    const u32 offset = staging_offsets[current_slot];
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

    const u32 layers = type == VideoCore::TextureType::CubeMap ? 6 : 1;
    const VideoCore::HostTextureTag key = {
        .format = format, .width = width, .height = height, .layers = layers};

    // Attempt to recycle an unused allocation
    if (auto it = texture_recycler.find(key); it != texture_recycler.end()) {
        ImageAlloc alloc = std::move(it->second);
        texture_recycler.erase(it);
        return alloc;
    }

    const FormatTraits traits = instance.GetTraits(format);
    const vk::ImageAspectFlags aspect = ToVkAspect(VideoCore::GetFormatType(format));

    const bool is_suitable = traits.blit_support && traits.attachment_support;
    const vk::Format vk_format = is_suitable ? traits.native : traits.fallback;
    const vk::ImageUsageFlags vk_usage = is_suitable ? traits.usage : GetImageUsage(aspect);

    const u32 levels = std::bit_width(std::max(width, height));
    const vk::ImageCreateInfo image_info = {.flags = type == VideoCore::TextureType::CubeMap
                                                         ? vk::ImageCreateFlagBits::eCubeCompatible
                                                         : vk::ImageCreateFlags{},
                                            .imageType = vk::ImageType::e2D,
                                            .format = vk_format,
                                            .extent = {width, height, 1},
                                            .mipLevels = levels,
                                            .arrayLayers = layers,
                                            .samples = vk::SampleCountFlagBits::e1,
                                            .usage = vk_usage};

    const VmaAllocationCreateInfo alloc_info = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(image_info);
    VmaAllocation allocation;

    VkResult result = vmaCreateImage(instance.GetAllocator(), &unsafe_image_info, &alloc_info,
                                     &unsafe_image, &allocation, nullptr);
    if (result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }

    vk::Image image = vk::Image{unsafe_image};
    const vk::ImageViewCreateInfo view_info = {.image = image,
                                               .viewType = type == VideoCore::TextureType::CubeMap
                                                               ? vk::ImageViewType::eCube
                                                               : vk::ImageViewType::e2D,
                                               .format = vk_format,
                                               .subresourceRange = {.aspectMask = aspect,
                                                                    .baseMipLevel = 0,
                                                                    .levelCount = levels,
                                                                    .baseArrayLayer = 0,
                                                                    .layerCount = layers}};

    // Also create a base mip view in case this is used as an attachment
    const vk::ImageViewCreateInfo base_view_info = {
        .image = image,
        .viewType = type == VideoCore::TextureType::CubeMap ? vk::ImageViewType::eCube
                                                            : vk::ImageViewType::e2D,
        .format = vk_format,
        .subresourceRange = {.aspectMask = aspect,
                             .baseMipLevel = 0,
                             .levelCount = 1,
                             .baseArrayLayer = 0,
                             .layerCount = layers}};

    vk::Device device = instance.GetDevice();
    vk::ImageView image_view = device.createImageView(view_info);
    vk::ImageView base_view = device.createImageView(base_view_info);

    // Create seperate depth/stencil views in case this gets reinterpreted with a compute shader
    vk::ImageView depth_view;
    vk::ImageView stencil_view;
    if (format == VideoCore::PixelFormat::D24S8) {
        vk::ImageViewCreateInfo view_info = {
            .image = image,
            .viewType = type == VideoCore::TextureType::CubeMap ? vk::ImageViewType::eCube
                                                                : vk::ImageViewType::e2D,
            .format = vk_format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                 .baseMipLevel = 0,
                                 .levelCount = levels,
                                 .baseArrayLayer = 0,
                                 .layerCount = layers}};

        depth_view = device.createImageView(view_info);
        view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eStencil;
        stencil_view = device.createImageView(view_info);
    }

    return ImageAlloc{.image = image,
                      .image_view = image_view,
                      .base_view = base_view,
                      .depth_view = depth_view,
                      .stencil_view = stencil_view,
                      .allocation = allocation,
                      .format = vk_format,
                      .aspect = aspect,
                      .levels = levels,
                      .layers = layers};
}

void TextureRuntime::Recycle(const VideoCore::HostTextureTag tag, ImageAlloc&& alloc) {
    texture_recycler.emplace(tag, std::move(alloc));
}

void TextureRuntime::FormatConvert(const Surface& surface, bool upload, std::span<std::byte> source,
                                   std::span<std::byte> dest) {
    if (!surface.NeedsConvert()) {
        std::memcpy(dest.data(), source.data(), source.size());
        return;
    }

    // Since this is the most common case handle it separately
    if (surface.pixel_format == VideoCore::PixelFormat::RGBA8) {
        return Pica::Texture::ConvertABGRToRGBA(source, dest);
    }

    // Handle simple D24S8 interleave case
    if (surface.GetInternalFormat() == vk::Format::eD24UnormS8Uint) {
        return Pica::Texture::InterleaveD24S8(source, dest);
    }

    if (upload) {
        switch (surface.pixel_format) {
        case VideoCore::PixelFormat::RGB8:
            return Pica::Texture::ConvertBGRToRGBA(source, dest);
        case VideoCore::PixelFormat::RGBA4:
            return Pica::Texture::ConvertRGBA4ToRGBA8(source, dest);
        default:
            break;
        }
    } else {
        switch (surface.pixel_format) {
        case VideoCore::PixelFormat::D24S8:
            return Pica::Texture::ConvertD32S8ToD24S8(source, dest);
        case VideoCore::PixelFormat::RGBA4:
            return Pica::Texture::ConvertRGBA8ToRGBA4(source, dest);
        default:
            break;
        }
    }

    LOG_WARNING(Render_Vulkan, "Missing format convertion: {} {} {}",
                vk::to_string(surface.traits.native), upload ? "->" : "<-",
                vk::to_string(surface.alloc.format));
}

bool TextureRuntime::ClearTexture(Surface& surface, const VideoCore::TextureClear& clear,
                                  VideoCore::ClearValue value) {
    const vk::ImageAspectFlags aspect = ToVkAspect(surface.type);
    renderpass_cache.ExitRenderpass();

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    Transition(command_buffer, surface.alloc, vk::ImageLayout::eTransferDstOptimal, 0,
               surface.alloc.levels, 0,
               surface.texture_type == VideoCore::TextureType::CubeMap ? 6 : 1);

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
        ImageAlloc& alloc = surface.alloc;
        if (aspect & vk::ImageAspectFlagBits::eColor) {
            clear_renderpass = renderpass_cache.GetRenderpass(
                surface.pixel_format, VideoCore::PixelFormat::Invalid, true);
            Transition(command_buffer, alloc, vk::ImageLayout::eColorAttachmentOptimal, 0,
                       alloc.levels);
        } else if (aspect & vk::ImageAspectFlagBits::eDepth ||
                   aspect & vk::ImageAspectFlagBits::eStencil) {
            clear_renderpass = renderpass_cache.GetRenderpass(VideoCore::PixelFormat::Invalid,
                                                              surface.pixel_format, true);
            Transition(command_buffer, alloc, vk::ImageLayout::eDepthStencilAttachmentOptimal, 0,
                       alloc.levels);
        }

        auto [it, new_framebuffer] =
            clear_framebuffers.try_emplace(alloc.image_view, vk::Framebuffer{});
        if (new_framebuffer) {
            const vk::ImageView framebuffer_view = surface.GetFramebufferView();
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
    Transition(command_buffer, source.alloc, vk::ImageLayout::eTransferSrcOptimal, 0,
               source.alloc.levels);
    Transition(command_buffer, dest.alloc, vk::ImageLayout::eTransferDstOptimal, 0,
               dest.alloc.levels);

    command_buffer.copyImage(source.alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                             dest.alloc.image, vk::ImageLayout::eTransferDstOptimal, image_copy);

    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    renderpass_cache.ExitRenderpass();

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    Transition(command_buffer, source.alloc, vk::ImageLayout::eTransferSrcOptimal, 0,
               source.alloc.levels, 0,
               source.texture_type == VideoCore::TextureType::CubeMap ? 6 : 1);
    Transition(command_buffer, dest.alloc, vk::ImageLayout::eTransferDstOptimal, 0,
               dest.alloc.levels, 0, dest.texture_type == VideoCore::TextureType::CubeMap ? 6 : 1);

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

    command_buffer.blitImage(source.alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                             dest.alloc.image, vk::ImageLayout::eTransferDstOptimal, blit_area,
                             vk::Filter::eNearest);

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
        Transition(command_buffer, surface.alloc, vk::ImageLayout::eTransferSrcOptimal, i - 1, 1);
        Transition(command_buffer, surface.alloc, vk::ImageLayout::eTransferDstOptimal, i, 1);

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

void TextureRuntime::Transition(vk::CommandBuffer command_buffer, ImageAlloc& alloc,
                                vk::ImageLayout new_layout, u32 level, u32 level_count, u32 layer,
                                u32 layer_count) {
    if (new_layout == alloc.layout || !alloc.image) {
        return;
    }

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

    LayoutInfo source = GetLayoutInfo(alloc.layout);
    LayoutInfo dest = GetLayoutInfo(new_layout);

    const vk::ImageMemoryBarrier barrier = {
        .srcAccessMask = source.access,
        .dstAccessMask = dest.access,
        .oldLayout = alloc.layout,
        .newLayout = new_layout,
        .image = alloc.image,
        .subresourceRange = {.aspectMask = alloc.aspect,
                             .baseMipLevel = /*level*/ 0,
                             .levelCount = /*level_count*/ alloc.levels,
                             .baseArrayLayer = layer,
                             .layerCount = layer_count}};

    command_buffer.pipelineBarrier(source.stage, dest.stage, vk::DependencyFlagBits::eByRegion, {},
                                   {}, barrier);

    alloc.layout = new_layout;
}

Surface::Surface(VideoCore::SurfaceParams& params, TextureRuntime& runtime)
    : VideoCore::SurfaceBase<Surface>{params}, runtime{runtime}, instance{runtime.GetInstance()},
      scheduler{runtime.GetScheduler()}, traits{instance.GetTraits(pixel_format)} {

    if (pixel_format != VideoCore::PixelFormat::Invalid) {
        alloc = runtime.Allocate(GetScaledWidth(), GetScaledHeight(), params.pixel_format,
                                 texture_type);
    }
}

Surface::~Surface() {
    if (pixel_format != VideoCore::PixelFormat::Invalid) {
        const VideoCore::HostTextureTag tag = {
            .format = pixel_format,
            .width = GetScaledWidth(),
            .height = GetScaledHeight(),
            .layers = texture_type == VideoCore::TextureType::CubeMap ? 6u : 1u};

        runtime.Recycle(tag, std::move(alloc));
    }
}

MICROPROFILE_DEFINE(Vulkan_Upload, "VulkanSurface", "Texture Upload", MP_RGB(128, 192, 64));
void Surface::Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging) {
    MICROPROFILE_SCOPE(Vulkan_Upload);

    runtime.renderpass_cache.ExitRenderpass();

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledUpload(upload);
    } else {
        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        const VideoCore::Rect2D rect = upload.texture_rect;
        const vk::BufferImageCopy copy_region = {
            .bufferOffset = staging.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource = {.aspectMask = alloc.aspect,
                                 .mipLevel = upload.texture_level,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1}};

        runtime.Transition(command_buffer, alloc, vk::ImageLayout::eTransferDstOptimal, 0,
                           alloc.levels, 0,
                           texture_type == VideoCore::TextureType::CubeMap ? 6 : 1);
        command_buffer.copyBufferToImage(staging.buffer, alloc.image,
                                         vk::ImageLayout::eTransferDstOptimal, copy_region);
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

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledDownload(download);
    } else {
        u32 region_count = 0;
        std::array<vk::BufferImageCopy, 2> copy_regions;

        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        const VideoCore::Rect2D rect = download.texture_rect;
        vk::BufferImageCopy copy_region = {
            .bufferOffset = staging.buffer_offset + download.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource = {.aspectMask = alloc.aspect,
                                 .mipLevel = download.texture_level,
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
                copy_region.bufferOffset += 4 * staging.size / 5;
                copy_region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
                copy_regions[region_count++] = copy_region;
            }
        }

        runtime.Transition(command_buffer, alloc, vk::ImageLayout::eTransferSrcOptimal, 0,
                           alloc.levels);

        // Copy pixel data to the staging buffer
        command_buffer.copyImageToBuffer(alloc.image, vk::ImageLayout::eTransferSrcOptimal,
                                         staging.buffer, region_count, copy_regions.data());
    }

    // Lock this data until the next scheduler switch
    const u32 current_slot = scheduler.GetCurrentSlotIndex();
    runtime.staging_offsets[current_slot] += staging.size;
}

bool Surface::NeedsConvert() const {
    // RGBA8 needs a byteswap since R8G8B8A8UnormPack32 does not exist
    // D24S8 always needs an interleave pass even if natively supported
    return alloc.format != traits.native || pixel_format == VideoCore::PixelFormat::RGBA8 ||
           pixel_format == VideoCore::PixelFormat::D24S8;
}

u32 Surface::GetInternalBytesPerPixel() const {
    return vk::blockSize(alloc.format);
}

void Surface::ScaledDownload(const VideoCore::BufferTextureCopy& download) {
    /*const u32 rect_width = download.texture_rect.GetWidth();
    const u32 rect_height = download.texture_rect.GetHeight();

    // Allocate an unscaled texture that fits the download rectangle to use as a blit destination
    const ImageAlloc unscaled_tex = runtime.Allocate(rect_width, rect_height, pixel_format,
                                                     VideoCore::TextureType::Texture2D);
    runtime.BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0, GL_TEXTURE_2D, type, unscaled_tex);
    runtime.BindFramebuffer(GL_READ_FRAMEBUFFER, download.texture_level, GL_TEXTURE_2D, type,
    texture);

    // Blit the scaled rectangle to the unscaled texture
    const VideoCore::Rect2D scaled_rect = download.texture_rect * res_scale;
    glBlitFramebuffer(scaled_rect.left, scaled_rect.bottom, scaled_rect.right, scaled_rect.top,
                      0, 0, rect_width, rect_height, MakeBufferMask(type), GL_LINEAR);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, unscaled_tex.handle);

    const auto& tuple = runtime.GetFormatTuple(pixel_format);
    if (driver.IsOpenGLES()) {
        const auto& downloader_es = runtime.GetDownloaderES();
        downloader_es.GetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type,
                                  rect_height, rect_width,
                                  reinterpret_cast<void*>(download.buffer_offset));
    } else {
        glGetTexImage(GL_TEXTURE_2D, 0, tuple.format, tuple.type,
                      reinterpret_cast<void*>(download.buffer_offset));
    }*/
}

void Surface::ScaledUpload(const VideoCore::BufferTextureCopy& upload) {
    /*const u32 rect_width = upload.texture_rect.GetWidth();
    const u32 rect_height = upload.texture_rect.GetHeight();

    OGLTexture unscaled_tex = runtime.Allocate(rect_width, rect_height, pixel_format,
                                               VideoCore::TextureType::Texture2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, unscaled_tex.handle);

    glTexSubImage2D(GL_TEXTURE_2D, upload.texture_level, 0, 0, rect_width, rect_height,
                    tuple.format, tuple.type, reinterpret_cast<void*>(upload.buffer_offset));

    const auto scaled_rect = upload.texture_rect * res_scale;
    const auto unscaled_rect = VideoCore::Rect2D{0, rect_height, rect_width, 0};
    const auto& filterer = runtime.GetFilterer();
    if (!filterer.Filter(unscaled_tex, unscaled_rect, texture, scaled_rect, type)) {
        runtime.BindFramebuffer(GL_READ_FRAMEBUFFER, 0, GL_TEXTURE_2D, type, unscaled_tex);
        runtime.BindFramebuffer(GL_DRAW_FRAMEBUFFER, upload.texture_level, GL_TEXTURE_2D, type,
    texture);

        // If filtering fails, resort to normal blitting
        glBlitFramebuffer(0, 0, rect_width, rect_height,
                          upload.texture_rect.left, upload.texture_rect.bottom,
                          upload.texture_rect.right, upload.texture_rect.top,
                          MakeBufferMask(type), GL_LINEAR);
    }*/
}

} // namespace Vulkan
