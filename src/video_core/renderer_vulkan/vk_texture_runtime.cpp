// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <bit>
#include "common/microprofile.h"
#include "video_core/rasterizer_cache/morton_swizzle.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"
#include "vulkan/vulkan_core.h"
#include "vulkan/vulkan_enums.hpp"
#include "vulkan/vulkan_structs.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_format_traits.hpp>

namespace Vulkan {

constexpr u32 UPLOAD_BUFFER_SIZE = 32 * 1024 * 1024;
constexpr u32 DOWNLOAD_BUFFER_SIZE = 32 * 1024 * 1024;

[[nodiscard]] vk::ImageAspectFlags MakeAspect(VideoCore::SurfaceType type) {
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
        LOG_CRITICAL(Render_Vulkan, "Invalid surface type {}", type);
        UNREACHABLE();
    }

    return vk::ImageAspectFlagBits::eColor;
}

[[nodiscard]] vk::Filter MakeFilter(VideoCore::PixelFormat pixel_format) {
    switch (pixel_format) {
    case VideoCore::PixelFormat::D16:
    case VideoCore::PixelFormat::D24:
    case VideoCore::PixelFormat::D24S8:
        return vk::Filter::eNearest;
    default:
        return vk::Filter::eLinear;
    }
}

[[nodiscard]] vk::ClearValue MakeClearValue(VideoCore::ClearValue clear) {
    static_assert(sizeof(VideoCore::ClearValue) == sizeof(vk::ClearValue));

    vk::ClearValue value{};
    std::memcpy(&value, &clear, sizeof(vk::ClearValue));
    return value;
}

[[nodiscard]] vk::ClearColorValue MakeClearColorValue(VideoCore::ClearValue clear) {
    return vk::ClearColorValue{
        .float32 = std::array{clear.color[0], clear.color[1], clear.color[2], clear.color[3]}};
}

[[nodiscard]] vk::ClearDepthStencilValue MakeClearDepthStencilValue(VideoCore::ClearValue clear) {
    return vk::ClearDepthStencilValue{.depth = clear.depth, .stencil = clear.stencil};
}

TextureRuntime::TextureRuntime(const Instance& instance, Scheduler& scheduler,
                               RenderpassCache& renderpass_cache, DescriptorManager& desc_manager)
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache},
      desc_manager{desc_manager}, blit_helper{instance, scheduler, desc_manager},
      upload_buffer{instance, scheduler, UPLOAD_BUFFER_SIZE}, download_buffer{instance, scheduler,
                                                                              DOWNLOAD_BUFFER_SIZE,
                                                                              true} {

    auto Register = [this](VideoCore::PixelFormat dest,
                           std::unique_ptr<FormatReinterpreterBase>&& obj) {
        const u32 dst_index = static_cast<u32>(dest);
        return reinterpreters[dst_index].push_back(std::move(obj));
    };

    Register(VideoCore::PixelFormat::RGBA8,
             std::make_unique<D24S8toRGBA8>(instance, scheduler, desc_manager, *this));
}

TextureRuntime::~TextureRuntime() {
    VmaAllocator allocator = instance.GetAllocator();
    vk::Device device = instance.GetDevice();
    device.waitIdle();

    for (const auto& [key, alloc] : texture_recycler) {
        vmaDestroyImage(allocator, alloc.image, alloc.allocation);
        device.destroyImageView(alloc.image_view);
        if (alloc.base_view) {
            device.destroyImageView(alloc.base_view);
        }
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
    auto& buffer = upload ? upload_buffer : download_buffer;
    auto [data, offset, invalidate] = buffer.Map(size, 4);

    return StagingData{.buffer = buffer.GetStagingHandle(),
                       .size = size,
                       .mapped = std::span<std::byte>{reinterpret_cast<std::byte*>(data), size},
                       .buffer_offset = offset};
}

void TextureRuntime::FlushBuffers() {
    upload_buffer.Flush();
}

MICROPROFILE_DEFINE(Vulkan_Finish, "Vulkan", "Scheduler Finish", MP_RGB(52, 192, 235));
void TextureRuntime::Finish() {
    MICROPROFILE_SCOPE(Vulkan_Finish);
    renderpass_cache.ExitRenderpass();
    scheduler.Finish();
    download_buffer.Invalidate();
}

ImageAlloc TextureRuntime::Allocate(u32 width, u32 height, VideoCore::PixelFormat format,
                                    VideoCore::TextureType type) {
    const FormatTraits traits = instance.GetTraits(format);
    const vk::ImageAspectFlags aspect = MakeAspect(VideoCore::GetFormatType(format));

    // Depth buffers are not supposed to support blit by the spec so don't require it.
    const bool is_suitable = traits.transfer_support && traits.attachment_support &&
                             (traits.blit_support || aspect & vk::ImageAspectFlagBits::eDepth);
    const vk::Format vk_format = is_suitable ? traits.native : traits.fallback;
    const vk::ImageUsageFlags vk_usage = is_suitable ? traits.usage : GetImageUsage(aspect);

    return Allocate(width, height, format, type, vk_format, vk_usage);
}

MICROPROFILE_DEFINE(Vulkan_ImageAlloc, "Vulkan", "TextureRuntime Finish", MP_RGB(192, 52, 235));
ImageAlloc TextureRuntime::Allocate(u32 width, u32 height, VideoCore::PixelFormat pixel_format,
                                    VideoCore::TextureType type, vk::Format format,
                                    vk::ImageUsageFlags usage) {
    MICROPROFILE_SCOPE(Vulkan_ImageAlloc);

    ImageAlloc alloc{};
    alloc.format = format;
    alloc.aspect = GetImageAspect(format);

    // The internal format does not provide enough guarantee of texture uniqueness
    // especially when many pixel formats fallback to RGBA8
    ASSERT(pixel_format != VideoCore::PixelFormat::Invalid);
    const HostTextureTag key = {.format = format,
                                .pixel_format = pixel_format,
                                .type = type,
                                .width = width,
                                .height = height};

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

    const u32 levels = std::bit_width(std::max(width, height));
    const u32 layers = type == VideoCore::TextureType::CubeMap ? 6 : 1;
    const vk::ImageCreateInfo image_info = {.flags = flags,
                                            .imageType = vk::ImageType::e2D,
                                            .format = format,
                                            .extent = {width, height, 1},
                                            .mipLevels = levels,
                                            .arrayLayers = layers,
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
                                                                    .levelCount = levels,
                                                                    .baseArrayLayer = 0,
                                                                    .layerCount = layers}};

    vk::Device device = instance.GetDevice();
    alloc.image_view = device.createImageView(view_info);

    // Also create a base mip view in case this is used as an attachment
    if (levels > 1) [[likely]] {
        const vk::ImageViewCreateInfo base_view_info = {
            .image = alloc.image,
            .viewType = view_type,
            .format = format,
            .subresourceRange = {.aspectMask = alloc.aspect,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = layers}};

        alloc.base_view = device.createImageView(base_view_info);
    }

    const bool has_stencil = static_cast<bool>(alloc.aspect & vk::ImageAspectFlagBits::eStencil);
    if (has_stencil) {
        vk::ImageViewCreateInfo view_info = {
            .image = alloc.image,
            .viewType = view_type,
            .format = format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eDepth,
                                 .baseMipLevel = 0,
                                 .levelCount = levels,
                                 .baseArrayLayer = 0,
                                 .layerCount = layers}};

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
                                 .levelCount = levels,
                                 .baseArrayLayer = 0,
                                 .layerCount = layers}};
        alloc.storage_view = device.createImageView(storage_view_info);
    }

    scheduler.Record([image = alloc.image, aspect = alloc.aspect](vk::CommandBuffer,
                                                                  vk::CommandBuffer upload_cmdbuf) {
        const vk::ImageMemoryBarrier init_barrier = {.srcAccessMask = vk::AccessFlagBits::eNone,
                                                     .dstAccessMask = vk::AccessFlagBits::eNone,
                                                     .oldLayout = vk::ImageLayout::eUndefined,
                                                     .newLayout = vk::ImageLayout::eGeneral,
                                                     .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                     .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                                     .image = image,
                                                     .subresourceRange{
                                                         .aspectMask = aspect,
                                                         .baseMipLevel = 0,
                                                         .levelCount = VK_REMAINING_MIP_LEVELS,
                                                         .baseArrayLayer = 0,
                                                         .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                                     }};

        upload_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                      vk::PipelineStageFlagBits::eTopOfPipe,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, init_barrier);
    });

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
    renderpass_cache.ExitRenderpass();

    const bool is_color = surface.type != VideoCore::SurfaceType::Depth &&
                          surface.type != VideoCore::SurfaceType::DepthStencil;

    if (clear.texture_rect == surface.GetScaledRect()) {
        scheduler.Record([aspect = MakeAspect(surface.type), image = surface.alloc.image, value,
                          is_color, clear](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
            const vk::ImageSubresourceRange range = {.aspectMask = aspect,
                                                     .baseMipLevel = clear.texture_level,
                                                     .levelCount = 1,
                                                     .baseArrayLayer = 0,
                                                     .layerCount = 1};

            const vk::ImageMemoryBarrier pre_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite |
                                 vk::AccessFlagBits::eColorAttachmentWrite |
                                 vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                 vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{
                    .aspectMask = aspect,
                    .baseMipLevel = clear.texture_level,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                }};

            const vk::ImageMemoryBarrier post_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask =
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                    vk::AccessFlagBits::eColorAttachmentRead |
                    vk::AccessFlagBits::eColorAttachmentWrite |
                    vk::AccessFlagBits::eDepthStencilAttachmentRead |
                    vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                    vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{
                    .aspectMask = aspect,
                    .baseMipLevel = clear.texture_level,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                }};

            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                          vk::PipelineStageFlagBits::eTransfer,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

            if (is_color) {
                render_cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal,
                                              MakeClearColorValue(value), range);
            } else {
                render_cmdbuf.clearDepthStencilImage(image, vk::ImageLayout::eTransferDstOptimal,
                                                     MakeClearDepthStencilValue(value), range);
            }

            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                          vk::PipelineStageFlagBits::eAllCommands,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
        });
        return true;
    }

    ClearTextureWithRenderpass(surface, clear, value);
    return true;
}

void TextureRuntime::ClearTextureWithRenderpass(Surface& surface,
                                                const VideoCore::TextureClear& clear,
                                                VideoCore::ClearValue value) {
    const bool is_color = surface.type != VideoCore::SurfaceType::Depth &&
                          surface.type != VideoCore::SurfaceType::DepthStencil;

    const vk::RenderPass clear_renderpass =
        is_color ? renderpass_cache.GetRenderpass(surface.pixel_format,
                                                  VideoCore::PixelFormat::Invalid, true)
                 : renderpass_cache.GetRenderpass(VideoCore::PixelFormat::Invalid,
                                                  surface.pixel_format, true);

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

    const RenderpassState clear_info = {
        .renderpass = clear_renderpass,
        .framebuffer = it->second,
        .render_area =
            vk::Rect2D{.offset = {static_cast<s32>(clear.texture_rect.left),
                                  static_cast<s32>(clear.texture_rect.bottom)},
                       .extent = {clear.texture_rect.GetWidth(), clear.texture_rect.GetHeight()}},
        .clear = MakeClearValue(value)};

    renderpass_cache.EnterRenderpass(clear_info);
    renderpass_cache.ExitRenderpass();
}

bool TextureRuntime::CopyTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureCopy& copy) {
    renderpass_cache.ExitRenderpass();

    scheduler.Record([src_image = source.alloc.image, dst_image = dest.alloc.image,
                      aspect = MakeAspect(source.type),
                      copy](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const vk::ImageCopy image_copy = {.srcSubresource = {.aspectMask = aspect,
                                                             .mipLevel = copy.src_level,
                                                             .baseArrayLayer = 0,
                                                             .layerCount = 1},
                                          .srcOffset = {static_cast<s32>(copy.src_offset.x),
                                                        static_cast<s32>(copy.src_offset.y), 0},
                                          .dstSubresource = {.aspectMask = aspect,
                                                             .mipLevel = copy.dst_level,
                                                             .baseArrayLayer = 0,
                                                             .layerCount = 1},
                                          .dstOffset = {static_cast<s32>(copy.dst_offset.x),
                                                        static_cast<s32>(copy.dst_offset.y), 0},
                                          .extent = {copy.extent.width, copy.extent.height, 1}};

        const std::array pre_barriers = {
            vk::ImageMemoryBarrier{.srcAccessMask =
                                       vk::AccessFlagBits::eShaderWrite |
                                       vk::AccessFlagBits::eColorAttachmentWrite |
                                       vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                       vk::AccessFlagBits::eTransferWrite,
                                   .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                                   .oldLayout = vk::ImageLayout::eGeneral,
                                   .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = src_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = copy.src_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }},
            vk::ImageMemoryBarrier{.srcAccessMask =
                                       vk::AccessFlagBits::eShaderWrite |
                                       vk::AccessFlagBits::eColorAttachmentWrite |
                                       vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                       vk::AccessFlagBits::eTransferWrite,
                                   .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                                   .oldLayout = vk::ImageLayout::eGeneral,
                                   .newLayout = vk::ImageLayout::eTransferDstOptimal,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = dst_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = copy.dst_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }},
        };
        const std::array post_barriers = {
            vk::ImageMemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eNone,
                                   .dstAccessMask = vk::AccessFlagBits::eNone,
                                   .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                                   .newLayout = vk::ImageLayout::eGeneral,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = src_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = copy.src_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }},
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask =
                    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
                    vk::AccessFlagBits::eColorAttachmentRead |
                    vk::AccessFlagBits::eColorAttachmentWrite |
                    vk::AccessFlagBits::eDepthStencilAttachmentRead |
                    vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                    vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = dst_image,
                .subresourceRange{
                    .aspectMask = aspect,
                    .baseMipLevel = copy.dst_level,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                }},
        };

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, pre_barriers);

        render_cmdbuf.copyImage(src_image, vk::ImageLayout::eTransferSrcOptimal, dst_image,
                                vk::ImageLayout::eTransferDstOptimal, image_copy);

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eAllCommands,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, post_barriers);
    });

    return true;
}

bool TextureRuntime::BlitTextures(Surface& source, Surface& dest,
                                  const VideoCore::TextureBlit& blit) {
    renderpass_cache.ExitRenderpass();

    scheduler.Record([src_image = source.alloc.image, aspect = MakeAspect(source.type),
                      filter = MakeFilter(source.pixel_format), dst_image = dest.alloc.image,
                      blit](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const std::array source_offsets = {vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                                                        static_cast<s32>(blit.src_rect.bottom), 0},
                                           vk::Offset3D{static_cast<s32>(blit.src_rect.right),
                                                        static_cast<s32>(blit.src_rect.top), 1}};

        const std::array dest_offsets = {vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                                                      static_cast<s32>(blit.dst_rect.bottom), 0},
                                         vk::Offset3D{static_cast<s32>(blit.dst_rect.right),
                                                      static_cast<s32>(blit.dst_rect.top), 1}};

        const vk::ImageBlit blit_area = {.srcSubresource = {.aspectMask = aspect,
                                                            .mipLevel = blit.src_level,
                                                            .baseArrayLayer = blit.src_layer,
                                                            .layerCount = 1},
                                         .srcOffsets = source_offsets,
                                         .dstSubresource = {.aspectMask = aspect,
                                                            .mipLevel = blit.dst_level,
                                                            .baseArrayLayer = blit.dst_layer,
                                                            .layerCount = 1},
                                         .dstOffsets = dest_offsets};

        const std::array read_barriers = {
            vk::ImageMemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                                   .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                                   .oldLayout = vk::ImageLayout::eGeneral,
                                   .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = src_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = blit.src_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }},
            vk::ImageMemoryBarrier{.srcAccessMask =
                                       vk::AccessFlagBits::eShaderRead |
                                       vk::AccessFlagBits::eDepthStencilAttachmentRead |
                                       vk::AccessFlagBits::eColorAttachmentRead |
                                       vk::AccessFlagBits::eTransferRead,
                                   .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                                   .oldLayout = vk::ImageLayout::eGeneral,
                                   .newLayout = vk::ImageLayout::eTransferDstOptimal,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = dst_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = blit.dst_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }}};
        const std::array write_barriers = {
            vk::ImageMemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eNone,
                                   .dstAccessMask = vk::AccessFlagBits::eMemoryWrite |
                                                    vk::AccessFlagBits::eMemoryRead,
                                   .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                                   .newLayout = vk::ImageLayout::eGeneral,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = src_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = blit.src_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }},
            vk::ImageMemoryBarrier{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                   .dstAccessMask = vk::AccessFlagBits::eMemoryWrite |
                                                    vk::AccessFlagBits::eMemoryRead,
                                   .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                                   .newLayout = vk::ImageLayout::eGeneral,
                                   .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .image = dst_image,
                                   .subresourceRange{
                                       .aspectMask = aspect,
                                       .baseMipLevel = blit.dst_level,
                                       .levelCount = 1,
                                       .baseArrayLayer = 0,
                                       .layerCount = VK_REMAINING_ARRAY_LAYERS,
                                   }}};

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, read_barriers);

        render_cmdbuf.blitImage(src_image, vk::ImageLayout::eTransferSrcOptimal, dst_image,
                                vk::ImageLayout::eTransferDstOptimal, blit_area, filter);

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eAllCommands,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, write_barriers);
    });

    return true;
}

void TextureRuntime::GenerateMipmaps(Surface& surface, u32 max_level) {
    /*renderpass_cache.ExitRenderpass();

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
    }*/
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
        scheduler.Record([aspect = alloc.aspect, image = alloc.image, format = alloc.format,
                          staging, upload](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
            u32 num_copies = 1;
            std::array<vk::BufferImageCopy, 2> buffer_image_copies;

            const VideoCore::Rect2D rect = upload.texture_rect;
            buffer_image_copies[0] = vk::BufferImageCopy{
                .bufferOffset = staging.buffer_offset + upload.buffer_offset,
                .bufferRowLength = rect.GetWidth(),
                .bufferImageHeight = rect.GetHeight(),
                .imageSubresource = {.aspectMask = aspect,
                                     .mipLevel = upload.texture_level,
                                     .baseArrayLayer = 0,
                                     .layerCount = 1},
                .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
                .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1}};

            if (aspect & vk::ImageAspectFlagBits::eStencil) {
                buffer_image_copies[0].imageSubresource.aspectMask =
                    vk::ImageAspectFlagBits::eDepth;
                vk::BufferImageCopy& stencil_copy = buffer_image_copies[1];
                stencil_copy = buffer_image_copies[0];
                stencil_copy.bufferOffset += UnpackDepthStencil(staging, format);
                stencil_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
                num_copies++;
            }

            static constexpr vk::AccessFlags WRITE_ACCESS_FLAGS =
                vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eColorAttachmentWrite |
                vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            static constexpr vk::AccessFlags READ_ACCESS_FLAGS =
                vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eColorAttachmentRead |
                vk::AccessFlagBits::eDepthStencilAttachmentRead;

            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = WRITE_ACCESS_FLAGS,
                .dstAccessMask = vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect,
                        .baseMipLevel = upload.texture_level,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };
            const vk::ImageMemoryBarrier write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = WRITE_ACCESS_FLAGS | READ_ACCESS_FLAGS,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect,
                        .baseMipLevel = upload.texture_level,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };

            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                          vk::PipelineStageFlagBits::eTransfer,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);

            render_cmdbuf.copyBufferToImage(staging.buffer, image,
                                            vk::ImageLayout::eTransferDstOptimal, num_copies,
                                            buffer_image_copies.data());

            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                          vk::PipelineStageFlagBits::eAllCommands,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, write_barrier);
        });

        runtime.upload_buffer.Commit(staging.size);
    }

    InvalidateAllWatcher();
}

MICROPROFILE_DEFINE(Vulkan_Download, "VulkanSurface", "Texture Download", MP_RGB(128, 192, 64));
void Surface::Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging) {
    MICROPROFILE_SCOPE(Vulkan_Download);

    runtime.renderpass_cache.ExitRenderpass();

    // For depth stencil downloads always use the compute shader fallback
    // to avoid having to interleave the data later. These should(?) be
    // uncommon anyways and the perf hit is very small
    if (type == VideoCore::SurfaceType::DepthStencil) {
        return DepthStencilDownload(download, staging);
    }

    const bool is_scaled = res_scale != 1;
    if (is_scaled) {
        ScaledDownload(download, staging);
    } else {
        scheduler.Record([aspect = alloc.aspect, image = alloc.image, staging,
                          download](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
            const VideoCore::Rect2D rect = download.texture_rect;
            const vk::BufferImageCopy buffer_image_copy = {
                .bufferOffset = staging.buffer_offset + download.buffer_offset,
                .bufferRowLength = rect.GetWidth(),
                .bufferImageHeight = rect.GetHeight(),
                .imageSubresource = {.aspectMask = aspect,
                                     .mipLevel = download.texture_level,
                                     .baseArrayLayer = 0,
                                     .layerCount = 1},
                .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
                .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1}};

            const vk::ImageMemoryBarrier read_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eTransferRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferSrcOptimal,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect,
                        .baseMipLevel = download.texture_level,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };
            const vk::ImageMemoryBarrier image_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange =
                    {
                        .aspectMask = aspect,
                        .baseMipLevel = download.texture_level,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = VK_REMAINING_ARRAY_LAYERS,
                    },
            };
            const vk::MemoryBarrier memory_write_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
            };

            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                          vk::PipelineStageFlagBits::eTransfer,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);

            render_cmdbuf.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal,
                                            staging.buffer, buffer_image_copy);

            render_cmdbuf.pipelineBarrier(
                vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
                vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, image_write_barrier);
        });

        runtime.download_buffer.Commit(staging.size);
    }
}

u32 Surface::GetInternalBytesPerPixel() const {
    // Request 5 bytes for D24S8 as well because we need the
    // extra space when unpacking the data during upload
    if (alloc.format == vk::Format::eD24UnormS8Uint) {
        return 5;
    }

    return vk::blockSize(alloc.format);
}

void Surface::ScaledUpload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging) {
    const u32 rect_width = upload.texture_rect.GetWidth();
    const u32 rect_height = upload.texture_rect.GetHeight();
    const VideoCore::Rect2D scaled_rect = upload.texture_rect * res_scale;
    const VideoCore::Rect2D unscaled_rect = VideoCore::Rect2D{0, rect_height, rect_width, 0};

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

    // For depth downloads create an R32UI surface and use a compute shader for convert.
    // Then we blit and download that surface.
    // NOTE: We keep the pixel format to D24S8 to avoid linear filtering during scale
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

    const VideoCore::Rect2D r32_scaled_rect =
        VideoCore::Rect2D{0, scaled_rect.GetHeight(), scaled_rect.GetWidth(), 0};
    const VideoCore::TextureBlit blit = {.src_level = download.texture_level,
                                         .dst_level = 0,
                                         .src_layer = 0,
                                         .dst_layer = 0,
                                         .src_rect = scaled_rect,
                                         .dst_rect = r32_scaled_rect};

    runtime.blit_helper.BlitD24S8ToR32(*this, r32_surface, blit);

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

u32 Surface::UnpackDepthStencil(const StagingData& data, vk::Format dest) {
    u32 depth_offset = 0;
    u32 stencil_offset = 4 * data.size / 5;
    const auto& mapped = data.mapped;

    switch (dest) {
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
                  vk::to_string(dest));
        UNREACHABLE();
    }

    ASSERT(depth_offset == 4 * data.size / 5);
    return depth_offset;
}

} // namespace Vulkan
