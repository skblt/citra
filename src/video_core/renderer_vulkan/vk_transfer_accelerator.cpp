// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_descriptor_manager.h"
#include "video_core/renderer_vulkan/vk_transfer_accelerator.h"

#include "video_core/host_shaders/conversion_shaders/buffer_to_image/a4_to_r8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/a8_to_r8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/etc1_to_rgba8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/etc1a4_to_rgba8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/i4_to_r8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/i8_to_r8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/ia4_to_rg8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/ia8_to_rg8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/rg8_to_rg8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/rgb565_to_rgba8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/rgb5a1_to_rgba8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/rgb8_to_rgba8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/rgba4_to_rgba8_comp_spv.h"
#include "video_core/host_shaders/conversion_shaders/buffer_to_image/rgba8_to_rgba8_comp_spv.h"

namespace Vulkan {

using VideoCore::PixelFormat;
using VideoCore::SurfaceType;

struct PXConversion {
    PixelFormat src_format;
    vk::Format dst_format;
    std::span<const u32> shader_bin;
};

TransferAccelerator::TransferAccelerator(const Instance& instance_, Scheduler& scheduler_,
                                         DescriptorManager& desc_manager_) :
    instance{instance_}, scheduler{scheduler_}, desc_manager{desc_manager_} {

    #define PX_CONVERSION(src, dst, shader)                                                            \
    PXConversion{PixelFormat::src, vk::Format::dst, shader}
    const std::array buffer_to_image_pipeline_specs{
        PX_CONVERSION(RGBA8,    eR8G8B8A8Unorm, RGBA8_TO_RGBA8_COMP_SPV),
        PX_CONVERSION(RGB8,     eR8G8B8A8Unorm, RGB8_TO_RGBA8_COMP_SPV),
        PX_CONVERSION(RGB5A1,   eR8G8B8A8Unorm, RGB5A1_TO_RGBA8_COMP_SPV),
        PX_CONVERSION(RGB565,   eR8G8B8A8Unorm, RGB565_TO_RGBA8_COMP_SPV),
        PX_CONVERSION(RGBA4,    eR8G8B8A8Unorm, RGBA4_TO_RGBA8_COMP_SPV),
        PX_CONVERSION(IA8,      eR8G8Unorm,     IA8_TO_RG8_COMP_SPV),
        PX_CONVERSION(RG8,      eR8G8Unorm,     RG8_TO_RG8_COMP_SPV),
        PX_CONVERSION(I8,       eR8Unorm,       I8_TO_R8_COMP_SPV),
        PX_CONVERSION(A8,       eR8Unorm,       A8_TO_R8_COMP_SPV),
        PX_CONVERSION(IA4,      eR8G8Unorm,     IA4_TO_RG8_COMP_SPV),
        PX_CONVERSION(I4,       eR8Unorm,       I4_TO_R8_COMP_SPV),
        PX_CONVERSION(A4,       eR8Unorm,       A4_TO_R8_COMP_SPV),
        PX_CONVERSION(ETC1,     eR8G8B8A8Unorm, ETC1_TO_RGBA8_COMP_SPV),
        PX_CONVERSION(ETC1A4,   eR8G8B8A8Unorm, ETC1A4_TO_RGBA8_COMP_SPV),
    };

    const std::array image_to_buffer_pipeline_specs{
        PX_CONVERSION(RGBA8, eR8G8B8A8Unorm, RGBA8_TO_RGBA8_COMP_SPV),
    };
    #undef PX_CONVERSION

    const std::array descriptors = {
        vk::DescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
        vk::DescriptorSetLayoutBinding{
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        },
    };

    const vk::DescriptorSetLayoutCreateInfo descriptor_layout_info = {
        .bindingCount = descriptors.size(),
        .pBindings = descriptors.data(),
    };

    vk::Device device = instance.GetDevice();
    buffer_to_image_set_layout = device.createDescriptorSetLayout(descriptor_layout_info);

    const std::array update_template_entries = {
        vk::DescriptorUpdateTemplateEntry{
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .offset = 0,
            .stride = 0,
        },
        vk::DescriptorUpdateTemplateEntry{
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage,
            .offset = sizeof(vk::DescriptorBufferInfo),
            .stride = 0,
        },
    };

    const vk::DescriptorUpdateTemplateCreateInfo template_info = {
        .descriptorUpdateEntryCount = static_cast<u32>(update_template_entries.size()),
        .pDescriptorUpdateEntries = update_template_entries.data(),
        .templateType = vk::DescriptorUpdateTemplateType::eDescriptorSet,
        .descriptorSetLayout = buffer_to_image_set_layout,
    };

    buffer_to_image_update_template = device.createDescriptorUpdateTemplate(template_info);

    const vk::PushConstantRange push_constant_range = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = 8,
    };

    vk::PipelineLayoutCreateInfo pipeline_layout_info = {
        .setLayoutCount = 1,
        .pSetLayouts = &buffer_to_image_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range,
    };

    buffer_to_image_pipeline_layout = device.createPipelineLayout(pipeline_layout_info);

    const auto BuildPipeline = [this](std::span<const u32> code, vk::PipelineLayout layout) {
        vk::Device device = instance.GetDevice();

        const vk::PipelineShaderStageCreateInfo stage_info = {
            .stage = vk::ShaderStageFlagBits::eCompute,
            .module = CompileSPV(code, device),
            .pName = "main",
        };

        const vk::ComputePipelineCreateInfo pipeline_info = {
            .stage = stage_info,
            .layout = layout,
        };

        return device.createComputePipeline({}, pipeline_info).value;
    };

    for (const auto& [src_fmt, dst_fmt, shader_code] : buffer_to_image_pipeline_specs) {
        const vk::Pipeline pipeline = BuildPipeline(shader_code, buffer_to_image_pipeline_layout);
        const std::size_t index = static_cast<std::size_t>(src_fmt);
        buffer_to_image_pipelines[index] = pipeline;
    }
    for (const auto& [src_fmt, dst_fmt, shader_code] : image_to_buffer_pipeline_specs) {
        const vk::Pipeline pipeline = BuildPipeline(shader_code, buffer_to_image_pipeline_layout);
        const std::size_t index = static_cast<std::size_t>(src_fmt);
        image_to_buffer_pipelines[index] = pipeline;
    }
}

TransferAccelerator::~TransferAccelerator() {
    vk::Device device = instance.GetDevice();
    for (std::size_t i = 0; i < buffer_to_image_pipelines.size(); i++) {
        if (vk::Pipeline pipeline = buffer_to_image_pipelines[i]) {
            device.destroyPipeline(pipeline);
        }
        if (vk::Pipeline pipeline = buffer_to_image_pipelines[i]) {
            device.destroyPipeline(pipeline);
        }
    }

    device.destroyPipelineLayout(buffer_to_image_pipeline_layout);
}

void TransferAccelerator::ImageFromBuffer(vk::CommandBuffer cmd_buff, vk::Buffer buffer,
                                       vk::DeviceSize offset, const CachedSurface& surface) {
    switch (surface.pixel_format) {
    case PixelFormat::RGBA8:
    case PixelFormat::RGB8:
    case PixelFormat::RGBA4:
    case PixelFormat::RGB5A1:
    case PixelFormat::RGB565:
    case PixelFormat::IA8:
    case PixelFormat::RG8:
    case PixelFormat::IA4:
    case PixelFormat::I8:
    case PixelFormat::A8:
    case PixelFormat::A4:
    case PixelFormat::I4:
    case PixelFormat::ETC1:
    case PixelFormat::ETC1A4: {
        BufferColorConvert(cmd_buff, Direction::BufferToImage, buffer, offset, surface);
    } break;
    case PixelFormat::D24S8:
    case PixelFormat::D24:
    case PixelFormat::D16: {
        D24S8Convert(cmd_buff, Direction::BufferToImage, buffer, offset, surface);
    } break;
    default:
        UNREACHABLE();
    }
}

void TransferAccelerator::BufferFromImage(vk::CommandBuffer cmd_buff, vk::Buffer buffer,
                                       vk::DeviceSize offset, const CachedSurface& surface) {
    switch (surface.pixel_format) {
    case PixelFormat::RGBA8: {
    //case PixelFormat::RGB8:
    //case PixelFormat::RGBA4:
    //case PixelFormat::RGB5A1:
    //case PixelFormat::RGB565: {
        BufferColorConvert(cmd_buff, Direction::ImageToBuffer, buffer, offset, surface);
    } break;
    //case PX::D24S8:
    //case PX::D24:
    //case PX::D16: {
    //    D24S8Convert(cmd_buff, Direction::ImageToBuffer, buffer, offset, surface);
    //} break;
    default:
        UNREACHABLE();
    }
}

void TransferAccelerator::AssignConversionDescriptor(CachedSurface& surface, vk::Buffer buffer,
                                                  vk::DeviceSize offset) {
    vk::DescriptorSetAllocateInfo descriptor_set_allocate_info;
    descriptor_set_allocate_info.descriptorPool = *descriptor_pool;
    descriptor_set_allocate_info.descriptorSetCount = 1;
    switch (surface.type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
        descriptor_set_allocate_info.pSetLayouts = &*buffer_to_image_set_layout;
        break;
    case SurfaceType::Depth:
    case SurfaceType::DepthStencil:
        descriptor_set_allocate_info.pSetLayouts = &*buffer_to_buffer_set_layout;
        break;
    default:
        UNREACHABLE();
    }
    surface.transfer_descriptor_set =
        std::move(vk_inst.device->allocateDescriptorSetsUnique(descriptor_set_allocate_info)[0]);
    std::array<vk::WriteDescriptorSet, 2> desc_set_writes;
    vk::DescriptorBufferInfo buffer_info;
    buffer_info.buffer = buffer;
    buffer_info.offset = offset;
    buffer_info.range = surface.size;

    desc_set_writes[0].descriptorCount = 1;
    desc_set_writes[0].descriptorType = vk::DescriptorType::eStorageBuffer;
    desc_set_writes[0].dstArrayElement = 0;
    desc_set_writes[0].dstBinding = 0;
    desc_set_writes[0].dstSet = *surface.transfer_descriptor_set;
    desc_set_writes[0].pBufferInfo = &buffer_info;

    vk::DescriptorImageInfo image_info;
    vk::DescriptorBufferInfo temp_buffer_info;
    switch (surface.type) {
    case SurfaceType::Color:
    case SurfaceType::Texture:
        image_info.imageLayout = vk::ImageLayout::eGeneral;
        image_info.imageView = *surface.uint_view;

        desc_set_writes[1].descriptorCount = 1;
        desc_set_writes[1].descriptorType = vk::DescriptorType::eStorageImage;
        desc_set_writes[1].dstArrayElement = 0;
        desc_set_writes[1].dstBinding = 1;
        desc_set_writes[1].dstSet = *surface.transfer_descriptor_set;
        desc_set_writes[1].pImageInfo = &image_info;
        break;
    case SurfaceType::Depth:
    case SurfaceType::DepthStencil:
        temp_buffer_info.buffer = *depth_stencil_temp;
        temp_buffer_info.offset = 0;
        temp_buffer_info.range = 1024 * 1024 * (4 + 1);

        desc_set_writes[1].descriptorCount = 1;
        desc_set_writes[1].descriptorType = vk::DescriptorType::eStorageBuffer;
        desc_set_writes[1].dstArrayElement = 0;
        desc_set_writes[1].dstBinding = 1;
        desc_set_writes[1].dstSet = *surface.transfer_descriptor_set;
        desc_set_writes[1].pBufferInfo = &temp_buffer_info;
        break;
    default:
        UNREACHABLE();
    }

    vk_inst.device->updateDescriptorSets(desc_set_writes, {});
}

void TransferAccelerator::BufferColorConvert(Surface& surface, Direction direction,
                                             const VideoCore::BufferTextureCopy& copy) {
    const auto& conversion_pipelines = direction == Direction::BufferToImage
                                           ? buffer_to_image_pipelines
                                           : image_to_buffer_pipelines;

    const vk::ImageSubresourceRange image_range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = copy.texture_level,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    const std::size_t index = static_cast<std::size_t>(surface.pixel_format);
    const vk::Pipeline pipeline = conversion_pipelines[index];
    const bool is_tiled = surface.is_tiled;
    const u32 tile_dim = (surface.pixel_format == PixelFormat::ETC1 ||
                          surface.pixel_format == PixelFormat::ETC1A4) ? 4 : 8;
    const Common::Vec2u group{surface.width / tile_dim, surface.height / tile_dim};

    scheduler.Record([this, pipeline, group, is_tiled](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
        render_cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                         buffer_to_image_pipeline_layout,
                                         0, surface.transfer_descriptor_set, {});
        render_cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);

        struct {
            alignas(4) bool tiled;
        } push_values{is_tiled};

        render_cmdbuf.pushConstants(buffer_to_image_pipeline_layout,
                                    vk::ShaderStageFlagBits::eCompute, 0,
                                    sizeof(push_values), &push_values);

        render_cmdbuf.dispatch(group.x, group.y, 1);
    });
}

void ConvertaTron5000::D24S8Convert(vk::CommandBuffer cmd_buff, Direction direction,
                                    vk::Buffer buffer, vk::DeviceSize offset,
                                    const CachedSurface& surface) {
    const auto& conversion_pipelines = direction == Direction::BufferToImage
                                           ? buffer_to_image_pipelines
                                           : image_to_buffer_pipelines;
    const auto pipeline = *conversion_pipelines.at(surface.pixel_format);

    vk::ImageSubresourceRange image_range;
    image_range.aspectMask = vk::ImageAspectFlagBits::eDepth;
    if (surface.type == SurfaceParams::SurfaceType::DepthStencil) {
        image_range.aspectMask |= vk::ImageAspectFlagBits::eStencil;
    }
    image_range.baseMipLevel = 0;
    image_range.levelCount = 1;
    image_range.baseArrayLayer = 0;
    image_range.layerCount = 1;
    vk::ImageMemoryBarrier barrier;
    barrier.image = *surface.image;
    barrier.subresourceRange = image_range;
    if (direction == Direction::BufferToImage) {
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        barrier.newLayout = vk::ImageLayout::eTransferDstOptimal;
    } else {
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    }
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    const auto Compute = [&] {
        cmd_buff.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        struct {
            u8 pad[3]{};
            bool tiled{};
        } push_values{surface.is_tiled};
        cmd_buff.pushConstants(*buffer_to_buffer_pipeline_layout, vk::ShaderStageFlagBits::eCompute,
                               0, sizeof(push_values), &push_values);
        cmd_buff.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                    *buffer_to_buffer_pipeline_layout, 0,
                                    *surface.transfer_descriptor_set, {});
        cmd_buff.dispatch(surface.width / 8, surface.height / 8, 1);
    };
    const auto Copy = [&] {
        cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
                                 vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, barrier);
        std::array<vk::BufferImageCopy, 2> copy;
        auto& depth_copy = copy[0];
        depth_copy.imageExtent = vk::Extent3D{surface.width, surface.height, 1};
        depth_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eDepth;
        depth_copy.imageSubresource.baseArrayLayer = 0;
        depth_copy.imageSubresource.layerCount = 1;
        depth_copy.imageSubresource.mipLevel = 0;
        auto& stencil_copy = copy[1];
        stencil_copy.imageSubresource = depth_copy.imageSubresource;
        stencil_copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eStencil;
        stencil_copy.imageExtent = depth_copy.imageExtent;
        stencil_copy.bufferOffset = 1024 * 1024 * 4;
        if (direction == Direction::BufferToImage) {
            cmd_buff.copyBufferToImage(
                *depth_stencil_temp, *surface.image, vk::ImageLayout::eTransferDstOptimal,
                {surface.type == SurfaceParams::SurfaceType::DepthStencil ? 2u : 1u, copy.data()});
            barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
        } else {
            cmd_buff.copyImageToBuffer(
                *surface.image, vk::ImageLayout::eTransferSrcOptimal, *depth_stencil_temp,
                {surface.type == SurfaceParams::SurfaceType::DepthStencil ? 2u : 1u, copy.data()});
            barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
        }
        barrier.newLayout = vk::ImageLayout::eGeneral;

        cmd_buff.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                 vk::PipelineStageFlagBits::eBottomOfPipe, {}, {}, {}, barrier);
    };
    if (direction == Direction::BufferToImage) {
        Compute();
    }
    Copy();
    if (direction == Direction::ImageToBuffer) {
        Compute();
    }
}
} // namespace Vulkan
