// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace VideoCore::Vulkan {

inline vk::BufferUsageFlags ToVkBufferUsage(BufferUsage usage) {
    constexpr std::array vk_buffer_usages = {
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::BufferUsageFlagBits::eIndexBuffer,
        vk::BufferUsageFlagBits::eUniformBuffer,
        vk::BufferUsageFlagBits::eUniformTexelBuffer,
        vk::BufferUsageFlagBits::eTransferSrc
    };

    return vk::BufferUsageFlagBits::eTransferDst |
            vk_buffer_usages.at(static_cast<u32>(usage));
}

inline vk::Format ToVkViewFormat(ViewFormat format) {
    constexpr std::array vk_view_formats = {
        vk::Format::eR32Sfloat,
        vk::Format::eR32G32Sfloat,
        vk::Format::eR32G32B32Sfloat,
        vk::Format::eR32G32B32A32Sfloat
    };

    return vk_view_formats.at(static_cast<u32>(format));
}

Buffer::Buffer(Instance& instance, CommandScheduler& scheduler, const BufferInfo& info) :
        BufferBase(info), instance(instance), scheduler(scheduler) {

    vk::BufferCreateInfo buffer_info = {
        .size = info.capacity,
        .usage = ToVkBufferUsage(info.usage)
    };

    VmaAllocationCreateInfo alloc_create_info = {
        .flags = info.usage == BufferUsage::Staging ?
                (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT) :
                VmaAllocationCreateFlags{},
        .usage = VMA_MEMORY_USAGE_AUTO
    };

    VkBuffer unsafe_buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationInfo alloc_info;
    VmaAllocator allocator = instance.GetAllocator();

    // Allocate texture memory
    vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_create_info,
                    &unsafe_buffer, &allocation, &alloc_info);
    buffer = vk::Buffer{unsafe_buffer};

    u32 view = 0;
    vk::Device device = instance.GetDevice();
    while (info.views[view] != ViewFormat::Undefined) {
        const vk::BufferViewCreateInfo view_info = {
            .buffer = buffer,
            .format = ToVkViewFormat(info.views[view]),
            .range = info.capacity
        };

        views[view++] = device.createBufferView(view_info);
    }

    // Map memory
    if (info.usage == BufferUsage::Staging) {
        mapped_ptr = alloc_info.pMappedData;
    }
}

Buffer::~Buffer() {
    if (buffer) {
        auto deleter = [allocation = allocation,
                        buffer = buffer,
                        views = views](vk::Device device, VmaAllocator allocator) {
            vmaDestroyBuffer(allocator, static_cast<VkBuffer>(buffer), allocation);

            u32 view_index = 0;
            while (views[view_index]) {
                device.destroyBufferView(views[view_index++]);
            }
        };

        // Delete the buffer immediately if it's allocated in host memory
        if (info.usage == BufferUsage::Staging) {
            vk::Device device = instance.GetDevice();
            VmaAllocator allocator = instance.GetAllocator();
            deleter(device, allocator);
        } else {
            scheduler.Schedule(deleter);
        }
    }
}

std::span<u8> Buffer::Map(u32 size, u32 alignment) {
    ASSERT(size <= info.capacity && alignment <= info.capacity);

    if (alignment > 0) {
        buffer_offset = Common::AlignUp<std::size_t>(buffer_offset, alignment);
    }

    // If the buffer is full, invalidate it
    if (buffer_offset + size > info.capacity) {
        Invalidate();
    }

    if (info.usage == BufferUsage::Staging) {
        return std::span<u8>{reinterpret_cast<u8*>(mapped_ptr) + buffer_offset, size};
    } else {
        Buffer& staging = scheduler.GetCommandUploadBuffer();
        return staging.Map(size, alignment);
    }
}

void Buffer::Commit(u32 size) {
    VmaAllocator allocator = instance.GetAllocator();
    if (info.usage == BufferUsage::Staging && size > 0) {
        vmaFlushAllocation(allocator, allocation, buffer_offset, size);
    } else {
        vk::CommandBuffer command_buffer = scheduler.GetUploadCommandBuffer();
        Buffer& staging = scheduler.GetCommandUploadBuffer();

        const vk::BufferCopy copy_region = {
            .srcOffset = staging.GetCurrentOffset(),
            .dstOffset = buffer_offset,
            .size = size
        };

        // Copy staging buffer to device local buffer
        command_buffer.copyBuffer(staging.GetHandle(), buffer, copy_region);

        vk::AccessFlags access_mask;
        vk::PipelineStageFlags stage_mask;
        switch (info.usage) {
        case BufferUsage::Vertex:
            access_mask = vk::AccessFlagBits::eVertexAttributeRead;
            stage_mask = vk::PipelineStageFlagBits::eVertexInput;
            break;
        case BufferUsage::Index:
            access_mask = vk::AccessFlagBits::eIndexRead;
            stage_mask = vk::PipelineStageFlagBits::eVertexInput;
            break;
        case BufferUsage::Uniform:
        case BufferUsage::Texel:
            access_mask = vk::AccessFlagBits::eUniformRead;
            stage_mask = vk::PipelineStageFlagBits::eVertexShader |
                    vk::PipelineStageFlagBits::eFragmentShader;
            break;
        default:
            LOG_CRITICAL(Render_Vulkan, "Unknown BufferUsage flag!");
        }

        const vk::BufferMemoryBarrier buffer_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = access_mask,
            .buffer = buffer,
            .offset = buffer_offset,
            .size = size
        };

        // Add a pipeline barrier for the region modified
        command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, stage_mask,
                                       vk::DependencyFlagBits::eByRegion, {}, buffer_barrier, {});

    }

    buffer_offset += size;
}

}