// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/microprofile.h"
#include "video_core/rasterizer_cache/morton_swizzle.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan_format_traits.hpp>

MICROPROFILE_DEFINE(Vulkan_Finish, "Vulkan", "Scheduler Finish", MP_RGB(52, 192, 235));
MICROPROFILE_DEFINE(Vulkan_Upload, "Vulkan", "Texture Upload", MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(Vulkan_Download, "Vulkan", "Texture Download", MP_RGB(128, 192, 64));

namespace Vulkan {

using namespace Pica::Texture;
using VideoCore::SurfaceParams;

constexpr u32 UPLOAD_BUFFER_SIZE = 32 * 1024 * 1024;
constexpr u32 DOWNLOAD_BUFFER_SIZE = 32 * 1024 * 1024;

constexpr static VideoCore::SurfaceParams NULL_PARAMS = {
    .width = 1,
    .height = 1,
    .stride = 1,
    .levels = 1,
    .texture_type = VideoCore::TextureType::Texture2D,
    .pixel_format = VideoCore::PixelFormat::RGBA8,
    .type = VideoCore::SurfaceType::Color,
};

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
    vk::ClearValue value{};
    std::memcpy(&value, &clear, sizeof(vk::ClearValue));
    return value;
}

[[nodiscard]] vk::ClearColorValue MakeClearColorValue(Common::Vec4f color) {
    return vk::ClearColorValue{.float32 = std::array{color[0], color[1], color[2], color[3]}};
}

[[nodiscard]] vk::ClearDepthStencilValue MakeClearDepthStencilValue(float depth, u8 stencil) {
    return vk::ClearDepthStencilValue{.depth = depth, .stencil = stencil};
}

[[nodiscard]] vk::Image MakeImage(VmaAllocator allocator, VmaAllocation& allocation,
                                  const vk::ImageCreateInfo& info) {
    const VmaAllocationCreateInfo alloc_info = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    VkImage unsafe_image{};
    VkImageCreateInfo unsafe_image_info = static_cast<VkImageCreateInfo>(info);

    VkResult result = vmaCreateImage(allocator, &unsafe_image_info, &alloc_info, &unsafe_image,
                                     &allocation, nullptr);

    if (result != VK_SUCCESS) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan, "Failed allocating texture with error {}", result);
        UNREACHABLE();
    }

    return vk::Image{unsafe_image};
}

std::size_t UnpackDepthStencil(const StagingData& data, vk::Format dest) {
    std::size_t depth_offset = 0;
    std::size_t stencil_offset = 4 * data.size / 5;
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
        LOG_ERROR(Render_Vulkan, "Unimplemeted convertion for depth format {}",
                  vk::to_string(dest));
        UNREACHABLE();
    }

    ASSERT(depth_offset == 4 * data.size / 5);
    return depth_offset;
}

TextureRuntime::TextureRuntime(const Instance& instance, Scheduler& scheduler,
                               RenderpassCache& renderpass_cache, DescriptorManager& desc_manager)
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache},
      desc_manager{desc_manager}, blit_helper{instance, scheduler, desc_manager},
      upload_buffer{instance, scheduler, UPLOAD_BUFFER_SIZE}, download_buffer{instance, scheduler,
                                                                              DOWNLOAD_BUFFER_SIZE,
                                                                              16, true} {

    const auto Register = [this](VideoCore::PixelFormat dest,
                                 std::unique_ptr<FormatReinterpreterBase>&& obj) {
        const u32 dst_index = static_cast<u32>(dest);
        return reinterpreters[dst_index].push_back(std::move(obj));
    };

    Register(VideoCore::PixelFormat::RGBA8,
             std::make_unique<D24S8toRGBA8>(instance, scheduler, desc_manager, *this));
}

TextureRuntime::~TextureRuntime() = default;

StagingData TextureRuntime::FindStaging(u32 size, bool upload) {
    StreamBuffer& buffer = upload ? upload_buffer : download_buffer;
    const auto [data, offset, invalidate] = buffer.Map(size);

    return StagingData{
        .buffer = buffer.GetStagingHandle(),
        .size = size,
        .mapped = std::span{reinterpret_cast<std::byte*>(data), size},
        .buffer_offset = offset,
    };
}

vk::Format TextureRuntime::NativeFormat(VideoCore::PixelFormat pixel_format) const {
    const FormatTraits traits = instance.GetTraits(pixel_format);
    const VideoCore::SurfaceType type = VideoCore::GetFormatType(pixel_format);
    const bool is_suitable = traits.transfer_support && traits.attachment_support &&
                             (traits.blit_support || type == VideoCore::SurfaceType::Depth ||
                              type == VideoCore::SurfaceType::DepthStencil);
    return is_suitable ? traits.native : traits.fallback;
}

void TextureRuntime::FlushBuffers() {
    upload_buffer.Flush();
}

void TextureRuntime::Finish() {
    MICROPROFILE_SCOPE(Vulkan_Finish);
    scheduler.Finish();
    download_buffer.Invalidate();
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
            return ConvertABGRToRGBA(source, dest);
        case VideoCore::PixelFormat::RGB8:
            return ConvertBGRToRGBA(source, dest);
        case VideoCore::PixelFormat::RGBA4:
            return ConvertRGBA4ToRGBA8(source, dest);
        default:
            break;
        }
    } else {
        switch (surface.pixel_format) {
        case VideoCore::PixelFormat::RGBA8:
            return ConvertABGRToRGBA(source, dest);
        case VideoCore::PixelFormat::RGBA4:
            return ConvertRGBA8ToRGBA4(source, dest);
        case VideoCore::PixelFormat::RGB8:
            return ConvertRGBAToBGR(source, dest);
        default:
            break;
        }
    }

    LOG_WARNING(Render_Vulkan, "Missing linear format convertion: {} {} {}",
                vk::to_string(surface.NativeFormat()), upload ? "->" : "<-",
                vk::to_string(surface.InternalFormat()));
}

bool TextureRuntime::SurfaceClear(Surface& surface, const VideoCore::TextureClear& clear) {
    if (clear.texture_rect != surface.GetScaledRect()) {
        return false;
    }

    renderpass_cache.ExitRenderpass();

    const bool is_color = surface.type != VideoCore::SurfaceType::Depth &&
                          surface.type != VideoCore::SurfaceType::DepthStencil;

    scheduler.Record([aspect = MakeAspect(surface.type), image = surface.Handle(),
                     is_color, clear, value = clear.value]
                     (vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const vk::ImageSubresourceRange range = {
            .aspectMask = aspect,
            .baseMipLevel = clear.texture_level,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

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
            },
        };

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
            },
        };

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, pre_barrier);

        if (is_color) {
            render_cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal,
                                          MakeClearColorValue(value.color), range);
        } else {
            render_cmdbuf.clearDepthStencilImage(
                image, vk::ImageLayout::eTransferDstOptimal,
                MakeClearDepthStencilValue(value.depth, value.stencil), range);
        }

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eAllCommands,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, post_barrier);
    });
    return true;
}

bool TextureRuntime::FramebufferClear(Framebuffer& framebuffer, const VideoCore::TextureClear& clear) {
    const RenderpassState clear_info = {
        .renderpass = framebuffer.RenderPass(vk::AttachmentLoadOp::eClear),
        .framebuffer = framebuffer.Handle(),
        .render_area =
            vk::Rect2D{
                .offset = {static_cast<s32>(clear.texture_rect.left),
                           static_cast<s32>(clear.texture_rect.bottom)},
                .extent = {clear.texture_rect.GetWidth(), clear.texture_rect.GetHeight()},
            },
        .clear = MakeClearValue(clear.value),
    };

    renderpass_cache.EnterRenderpass(clear_info);
    renderpass_cache.ExitRenderpass();

    return true;
}

bool TextureRuntime::SurfaceCopy(Surface& source, Surface& dest,
                                 const VideoCore::TextureCopy& copy) {
    renderpass_cache.ExitRenderpass();

    scheduler.Record([src_image = source.Handle(), dst_image = dest.Handle(),
                      aspect = MakeAspect(source.type),
                      copy](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const vk::ImageCopy image_copy = {
            .srcSubresource{
                .aspectMask = aspect,
                .mipLevel = copy.src_level,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .srcOffset = {static_cast<s32>(copy.src_offset.x), static_cast<s32>(copy.src_offset.y), 0},
            .dstSubresource{
                .aspectMask = aspect,
                .mipLevel = copy.dst_level,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .dstOffset = {static_cast<s32>(copy.dst_offset.x), static_cast<s32>(copy.dst_offset.y), 0},
            .extent = {copy.extent.width, copy.extent.height, 1},
        };

        const std::array pre_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite |
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
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite |
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
                },
            },
        };
        const std::array post_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
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
                },
            },
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
                },
            },
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

bool TextureRuntime::SurfaceBlit(Surface& source, Surface& dest,
                                 const VideoCore::TextureBlit& blit) {
    renderpass_cache.ExitRenderpass();

    scheduler.Record([src_image = source.Handle(), aspect = MakeAspect(source.type),
                      filter = MakeFilter(source.pixel_format), dst_image = dest.Handle(),
                      blit](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const std::array source_offsets = {
            vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                         static_cast<s32>(blit.src_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.src_rect.right), static_cast<s32>(blit.src_rect.top),
                         1},
        };

        const std::array dest_offsets = {
            vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                         static_cast<s32>(blit.dst_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.dst_rect.right), static_cast<s32>(blit.dst_rect.top),
                         1},
        };

        const vk::ImageBlit blit_area = {
            .srcSubresource = {.aspectMask = aspect,
                               .mipLevel = blit.src_level,
                               .baseArrayLayer = blit.src_layer,
                               .layerCount = 1},
            .srcOffsets = source_offsets,
            .dstSubresource = {.aspectMask = aspect,
                               .mipLevel = blit.dst_level,
                               .baseArrayLayer = blit.dst_layer,
                               .layerCount = 1},
            .dstOffsets = dest_offsets,
        };

        const std::array read_barriers{
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite |
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
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderWrite |
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
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
        };
        const std::array write_barriers{
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferRead,
                .dstAccessMask = vk::AccessFlagBits::eShaderWrite |
                                 vk::AccessFlagBits::eDepthStencilAttachmentWrite |
                                 vk::AccessFlagBits::eTransferWrite,
                .oldLayout = vk::ImageLayout::eTransferSrcOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = src_image,
                .subresourceRange{
                    .aspectMask = aspect,
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
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
                    .baseMipLevel = 0,
                    .levelCount = VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = VK_REMAINING_ARRAY_LAYERS,
                },
            }
        };

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

Allocation::Allocation(TextureRuntime& runtime) : Allocation{runtime, NULL_PARAMS} {}

Allocation::Allocation(TextureRuntime& runtime, const VideoCore::SurfaceParams& params) {
    if (params.pixel_format == VideoCore::PixelFormat::Invalid) {
        ASSERT(false);
    }

    const Instance& instance = runtime.GetInstance();
    const FormatTraits traits = instance.GetTraits(params.pixel_format);
    const bool is_suitable = traits.transfer_support && traits.attachment_support &&
                             (traits.blit_support || aspect & vk::ImageAspectFlagBits::eDepth);

    device = instance.GetDevice();
    allocator = instance.GetAllocator();
    format = is_suitable ? traits.native : traits.fallback;
    aspect = GetImageAspect(format);

    const vk::ImageUsageFlags usage = is_suitable ? traits.usage : GetImageUsage(aspect);
    const bool create_storage_view = params.pixel_format == VideoCore::PixelFormat::RGBA8;
    const u32 layers = params.texture_type == VideoCore::TextureType::CubeMap ? 6 : 1;

    vk::ImageCreateFlags flags;
    if (params.texture_type == VideoCore::TextureType::CubeMap) {
        flags |= vk::ImageCreateFlagBits::eCubeCompatible;
    }
    if (create_storage_view) {
        flags |= vk::ImageCreateFlagBits::eMutableFormat;
    }

    vk::ImageCreateInfo image_info = {
        .flags = flags,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {params.GetScaledWidth(), params.GetScaledHeight(), 1},
        .mipLevels = params.levels,
        .arrayLayers = layers,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = usage,
    };

    images[0] = MakeImage(allocator, allocations[0], image_info);

    if (params.res_scale != 1) {
        image_info.extent.width = params.width;
        image_info.extent.height = params.height;
        images[1] = MakeImage(allocator, allocations[1], image_info);
    } else {
        if (params.height == 800) {
            printf("ff");
        }
        images[1] = images[0];
    }

    const vk::ImageViewType view_type = params.texture_type == VideoCore::TextureType::CubeMap
                                            ? vk::ImageViewType::eCube
                                            : vk::ImageViewType::e2D;

    const vk::ImageViewCreateInfo view_info = {
        .image = images[0],
        .viewType = view_type,
        .format = format,
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = params.levels,
            .baseArrayLayer = 0,
            .layerCount = layers,
        },
    };

    image_view = device.createImageView(view_info);

    if (aspect & vk::ImageAspectFlagBits::eStencil) {
        vk::ImageViewCreateInfo view_info = {
            .image = images[0],
            .viewType = view_type,
            .format = format,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eDepth,
                .baseMipLevel = 0,
                .levelCount = params.levels,
                .baseArrayLayer = 0,
                .layerCount = layers,
            },
        };

        depth_view = device.createImageView(view_info);
        view_info.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eStencil;
        stencil_view = device.createImageView(view_info);
    }

    if (create_storage_view) {
        const vk::ImageViewCreateInfo storage_view_info = {
            .image = images[0],
            .viewType = view_type,
            .format = vk::Format::eR32Uint,
            .subresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = params.levels,
                .baseArrayLayer = 0,
                .layerCount = layers,
            },
        };
        storage_view = device.createImageView(storage_view_info);
    }

    runtime.GetScheduler().Record([image0 = images[0], image1 = images[1], aspect = aspect](
                                      vk::CommandBuffer, vk::CommandBuffer upload_cmdbuf) {
        vk::ImageMemoryBarrier init_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eNone,
            .dstAccessMask = vk::AccessFlagBits::eNone,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image0,
            .subresourceRange{
                .aspectMask = aspect,
                .baseMipLevel = 0,
                .levelCount = VK_REMAINING_MIP_LEVELS,
                .baseArrayLayer = 0,
                .layerCount = VK_REMAINING_ARRAY_LAYERS,
            },
        };

        upload_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                      vk::PipelineStageFlagBits::eTopOfPipe,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, init_barrier);

        if (image0 != image1) {
            init_barrier.image = image1;
            upload_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                          vk::PipelineStageFlagBits::eTopOfPipe,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, init_barrier);
        }
    });
}

Allocation::~Allocation() {
    if (!device) {
        return;
    }
    if (images[1] && images[1] != images[0]) {
        vmaDestroyImage(allocator, images[1], allocations[1]);
    }
    vmaDestroyImage(allocator, images[0], allocations[0]);
    device.destroyImageView(image_view);
    if (stencil_view) {
        device.destroyImageView(stencil_view);
    }
    if (depth_view) {
        device.destroyImageView(depth_view);
    }
    if (storage_view) {
        device.destroyImageView(storage_view);
    }
}

Surface::Surface(VideoCore::SurfaceParams params) : SurfaceBase{params} {}

Surface::Surface(TextureRuntime& runtime, Allocation&& alloc, VideoCore::SurfaceParams params)
    : SurfaceBase{params}, scheduler{&runtime.GetScheduler()}, runtime{&runtime},
      alloc{std::move(alloc)} {
    const Instance& instance = runtime.GetInstance();
    traits = instance.GetTraits(pixel_format);
}

Surface::~Surface() = default;

void Surface::Upload(const VideoCore::BufferTextureCopy& upload, const StagingData& staging) {
    MICROPROFILE_SCOPE(Vulkan_Upload);

    if (type == VideoCore::SurfaceType::DepthStencil && !traits.blit_support) {
        LOG_ERROR(Render_Vulkan, "Depth blit unsupported by hardware, ignoring");
        return;
    }

    runtime->renderpass_cache.ExitRenderpass();

    scheduler->Record([aspect = alloc.aspect, image = Handle(true), format = alloc.format,
                       staging, upload](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        u32 num_copies = 1;
        std::array<vk::BufferImageCopy, 2> buffer_image_copies;

        const VideoCore::Rect2D rect = upload.texture_rect;
        buffer_image_copies[0] = vk::BufferImageCopy{
            .bufferOffset = staging.buffer_offset + upload.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource{
                .aspectMask = aspect,
                .mipLevel = upload.texture_level,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
        };

        if (aspect & vk::ImageAspectFlagBits::eStencil) {
            buffer_image_copies[0].imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
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
            .subresourceRange{
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
            .subresourceRange{
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

        render_cmdbuf.copyBufferToImage(staging.buffer, image, vk::ImageLayout::eTransferDstOptimal,
                                        num_copies, buffer_image_copies.data());

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eAllCommands,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, write_barrier);
    });

    runtime->upload_buffer.Commit(staging.size);

    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = upload.texture_level,
            .dst_level = upload.texture_level,
            .src_layer = 0,
            .dst_layer = 0,
            .src_rect = upload.texture_rect,
            .dst_rect = upload.texture_rect * res_scale,
        };

        BlitScale(blit, true);
    }
}

void Surface::Download(const VideoCore::BufferTextureCopy& download, const StagingData& staging) {
    MICROPROFILE_SCOPE(Vulkan_Download);

    runtime->renderpass_cache.ExitRenderpass();

    if (type == VideoCore::SurfaceType::DepthStencil) {
        LOG_INFO(Render_Vulkan, "Unsupported download!");
        return;
    }

    if (res_scale != 1) {
        const VideoCore::TextureBlit blit = {
            .src_level = download.texture_level,
            .dst_level = download.texture_level,
            .src_layer = 0,
            .dst_layer = 0,
            .src_rect = download.texture_rect * res_scale,
            .dst_rect = download.texture_rect,
        };

        BlitScale(blit, false);
    }

    scheduler->Record([aspect = alloc.aspect, image = Handle(true), staging,
                       download](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const VideoCore::Rect2D rect = download.texture_rect;
        const vk::BufferImageCopy buffer_image_copy = {
            .bufferOffset = staging.buffer_offset + download.buffer_offset,
            .bufferRowLength = rect.GetWidth(),
            .bufferImageHeight = rect.GetHeight(),
            .imageSubresource{
                .aspectMask = aspect,
                .mipLevel = download.texture_level,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = {static_cast<s32>(rect.left), static_cast<s32>(rect.bottom), 0},
            .imageExtent = {rect.GetWidth(), rect.GetHeight(), 1},
        };

        const vk::ImageMemoryBarrier read_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
            .dstAccessMask = vk::AccessFlagBits::eTransferRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eTransferSrcOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange{
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
            .subresourceRange{
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

        render_cmdbuf.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, staging.buffer,
                                        buffer_image_copy);

        render_cmdbuf.pipelineBarrier(
            vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
            vk::DependencyFlagBits::eByRegion, memory_write_barrier, {}, image_write_barrier);
    });
    runtime->download_buffer.Commit(staging.size);
}

u32 Surface::InternalBytesPerPixel() const {
    switch (alloc.format) {
    case vk::Format::eD24UnormS8Uint:
        return 5;
    default:
        return vk::blockSize(alloc.format);
    }
}

void Surface::BlitScale(const VideoCore::TextureBlit& blit, bool up_scale) {
    scheduler->Record([src_image = Handle(up_scale), aspect = alloc.aspect,
                       filter = MakeFilter(pixel_format),
                       dst_image = Handle(!up_scale), blit]
                      (vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        const std::array source_offsets = {
            vk::Offset3D{static_cast<s32>(blit.src_rect.left),
                         static_cast<s32>(blit.src_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.src_rect.right), static_cast<s32>(blit.src_rect.top),
                         1},
        };

        const std::array dest_offsets = {
            vk::Offset3D{static_cast<s32>(blit.dst_rect.left),
                         static_cast<s32>(blit.dst_rect.bottom), 0},
            vk::Offset3D{static_cast<s32>(blit.dst_rect.right), static_cast<s32>(blit.dst_rect.top),
                         1},
        };

        const vk::ImageBlit blit_area = {
            .srcSubresource{
                .aspectMask = aspect,
                .mipLevel = blit.src_level,
                .baseArrayLayer = blit.src_layer,
                .layerCount = 1,
            },
            .srcOffsets = source_offsets,
            .dstSubresource{
                .aspectMask = aspect,
                .mipLevel = blit.dst_level,
                .baseArrayLayer = blit.dst_layer,
                .layerCount = 1,
            },
            .dstOffsets = dest_offsets,
        };

        const std::array read_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
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
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eShaderRead |
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
                },
            },
        };
        const std::array write_barriers = {
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
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
                },
            },
            vk::ImageMemoryBarrier{
                .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                .dstAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eMemoryRead,
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
                },
            },
        };

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
                                      vk::PipelineStageFlagBits::eTransfer,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, read_barriers);

        render_cmdbuf.blitImage(src_image, vk::ImageLayout::eTransferSrcOptimal, dst_image,
                                vk::ImageLayout::eTransferDstOptimal, blit_area, filter);

        render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                      vk::PipelineStageFlagBits::eAllCommands,
                                      vk::DependencyFlagBits::eByRegion, {}, {}, write_barriers);
    });
}

Framebuffer::Framebuffer(TextureRuntime& runtime, Surface* color, Surface* depth,
                         VideoCore::RenderTargets key)
    : renderpass_cache{&runtime.GetRenderpassCache()},
      device{runtime.GetInstance().GetDevice()} {
    u32 attachment_count = 0;
    std::array<vk::ImageView, 2> attachments;

    std::array<VideoCore::PixelFormat, 2> formats;
    formats.fill(VideoCore::PixelFormat::Invalid);

    LOG_INFO(Render_Vulkan, "Creating framebuffer");

    if (color) {
        color_view = color->ImageView();
        attachments[attachment_count++] = color_view;
        formats[0] = color->pixel_format;
    }

    if (depth) {
        depth_view = depth->ImageView();
        attachments[attachment_count++] = depth_view;
        formats[1] = depth->pixel_format;
    }

    RenderpassCache& renderpass_cache = runtime.GetRenderpassCache();
    renderpass[0] = renderpass_cache.GetRenderpass(formats[0], formats[1], false);
    renderpass[1] = renderpass_cache.GetRenderpass(formats[0], formats[1], true);

    const vk::FramebufferCreateInfo framebuffer_info = {
        .renderPass = renderpass[0],
        .attachmentCount = attachment_count,
        .pAttachments = attachments.data(),
        .width = key.size.width,
        .height = key.size.height,
        .layers = 1,
    };

    framebuffer = device.createFramebuffer(framebuffer_info);
}

void Framebuffer::BeginRenderPass() {
    const vk::Rect2D rect{
        .offset{static_cast<s32>(render_area.left), static_cast<s32>(render_area.bottom)},
        .extent = {render_area.GetWidth(), render_area.GetHeight()},
    };

    const RenderpassState renderpass_info = {
        .renderpass = renderpass[0],
        .framebuffer = framebuffer,
        .render_area = rect,
        .clear = {},
    };

    renderpass_cache->EnterRenderpass(renderpass_info);
}

Framebuffer::~Framebuffer() {
    if (framebuffer) {
        device.destroyFramebuffer(framebuffer);
    }
}

Sampler::Sampler(TextureRuntime& runtime, VideoCore::SamplerParams params) :
    device{runtime.GetInstance().GetDevice()} {
    using TextureConfig = VideoCore::SamplerParams::TextureConfig;

    const Instance& instance = runtime.GetInstance();
    const vk::PhysicalDeviceProperties properties = instance.GetPhysicalDevice().getProperties();
    const bool use_border_color =
        instance.IsCustomBorderColorSupported() && (params.wrap_s == TextureConfig::ClampToBorder ||
                                                    params.wrap_t == TextureConfig::ClampToBorder);

    const Common::Vec4f color = PicaToVK::ColorRGBA8(params.border_color);
    const vk::SamplerCustomBorderColorCreateInfoEXT border_color_info = {
        .customBorderColor = MakeClearColorValue(color),
        .format = vk::Format::eUndefined,
    };

    const vk::Filter mag_filter = PicaToVK::TextureFilterMode(params.mag_filter);
    const vk::Filter min_filter = PicaToVK::TextureFilterMode(params.min_filter);
    const vk::SamplerMipmapMode mipmap_mode = PicaToVK::TextureMipFilterMode(params.mip_filter);
    const vk::SamplerAddressMode wrap_u = PicaToVK::WrapMode(params.wrap_s);
    const vk::SamplerAddressMode wrap_v = PicaToVK::WrapMode(params.wrap_t);
    const float lod_bias = /*params.lod_bias / 256.f*/0.f;
    const float lod_min = params.lod_min * 0.f;
    const float lod_max = params.lod_max * 0.f;

    const vk::SamplerCreateInfo sampler_info = {
        .pNext = use_border_color ? &border_color_info : nullptr,
        .magFilter = mag_filter,
        .minFilter = min_filter,
        .mipmapMode = mipmap_mode,
        .addressModeU = wrap_u,
        .addressModeV = wrap_v,
        .mipLodBias = lod_bias,
        .anisotropyEnable = true,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = false,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = lod_min,
        .maxLod = lod_max,
        .borderColor =
            use_border_color ? vk::BorderColor::eFloatCustomEXT : vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = false,
    };

    sampler = device.createSampler(sampler_info);
}

Sampler::~Sampler() {
    if (sampler) {
        device.destroySampler(sampler);
    }
}

} // namespace Vulkan
