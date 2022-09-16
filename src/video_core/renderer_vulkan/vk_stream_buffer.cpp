// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <algorithm>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

inline auto ToVkAccessStageFlags(vk::BufferUsageFlagBits usage) {
    std::pair<vk::AccessFlags, vk::PipelineStageFlags> result{};
    switch (usage) {
    case vk::BufferUsageFlagBits::eVertexBuffer:
        result = std::make_pair(vk::AccessFlagBits::eVertexAttributeRead,
                                vk::PipelineStageFlagBits::eVertexInput);
        break;
    case vk::BufferUsageFlagBits::eIndexBuffer:
        result = std::make_pair(vk::AccessFlagBits::eIndexRead,
                                vk::PipelineStageFlagBits::eVertexInput);
    case vk::BufferUsageFlagBits::eUniformBuffer:
        result = std::make_pair(vk::AccessFlagBits::eUniformRead,
                                vk::PipelineStageFlagBits::eVertexShader |
                                vk::PipelineStageFlagBits::eGeometryShader |
                                vk::PipelineStageFlagBits::eFragmentShader);
    case vk::BufferUsageFlagBits::eUniformTexelBuffer:
        result = std::make_pair(vk::AccessFlagBits::eShaderRead,
                                vk::PipelineStageFlagBits::eFragmentShader);
        break;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown usage flag {}", usage);
    }

    return result;
}

StagingBuffer::StagingBuffer(const Instance& instance, u32 size, vk::BufferUsageFlags usage)
    : instance{instance} {
    const vk::BufferCreateInfo buffer_info = {
        .size = size,
        .usage = usage
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST
    };

    VkBuffer unsafe_buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationInfo alloc_info;
    VmaAllocator allocator = instance.GetAllocator();

    vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_create_info,
                    &unsafe_buffer, &allocation, &alloc_info);

    buffer = vk::Buffer{unsafe_buffer};
    mapped = std::span{reinterpret_cast<std::byte*>(alloc_info.pMappedData), size};
}

StagingBuffer::~StagingBuffer() {
    vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(buffer), allocation);
}

StreamBuffer::StreamBuffer(const Instance& instance, TaskScheduler& scheduler, const BufferInfo& info)
    : instance{instance}, scheduler{scheduler}, info{info},
      staging{instance, info.size, vk::BufferUsageFlagBits::eTransferSrc} {

    const vk::BufferCreateInfo buffer_info = {
        .size = info.size,
        .usage = info.usage | vk::BufferUsageFlagBits::eTransferDst
    };

    const VmaAllocationCreateInfo alloc_create_info = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    };

    VkBuffer unsafe_buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationInfo alloc_info;
    VmaAllocator allocator = instance.GetAllocator();

    vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_create_info,
                    &unsafe_buffer, &allocation, &alloc_info);

    buffer = vk::Buffer{unsafe_buffer};

    vk::Device device = instance.GetDevice();
    for (u32 i = 0; i < info.views.size(); i++) {
        if (info.views[i] == vk::Format::eUndefined) {
            view_count = i;
            break;
        }

        const vk::BufferViewCreateInfo view_info = {
            .buffer = buffer,
            .format = info.views[i],
            .range = info.size
        };

        views[i] = device.createBufferView(view_info);
    }

    available_size = info.size;
}

StreamBuffer::~StreamBuffer() {
    if (buffer) {
        vk::Device device = instance.GetDevice();
        vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(buffer), allocation);
        for (u32 i = 0; i < view_count; i++) {
            device.destroyBufferView(views[i]);
        }
    }
}

std::tuple<u8*, u32, bool> StreamBuffer::Map(u32 size, u32 alignment) {
    ASSERT(size <= info.size && alignment <= info.size);

    if (alignment > 0) {
        buffer_offset = Common::AlignUp(buffer_offset, alignment);
    }

    // Have we run out of available space?
    bool invalidate = false;
    if (available_size < size) {
        // Flush any pending writes before continuing
        Flush();

        // If we are at the end of the buffer, start over
        if (buffer_offset + size > info.size) {
            Invalidate();
            invalidate = true;
        }

        // Try to garbage collect old regions
        if (!UnlockFreeRegions(size)) {
            // Nuclear option: stall the GPU to remove all the locks
            LOG_WARNING(Render_Vulkan, "Buffer GPU stall");
            Invalidate();
            regions.clear();
            available_size = info.size;
        }
    }

    u8* mapped = reinterpret_cast<u8*>(staging.mapped.data() + buffer_offset);
    return std::make_tuple(mapped, buffer_offset, invalidate);
}

void StreamBuffer::Commit(u32 size) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();

    auto [access_mask, stage_mask] = ToVkAccessStageFlags(info.usage);
    const vk::BufferMemoryBarrier buffer_barrier = {
        .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
        .dstAccessMask = access_mask,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = buffer,
        .offset = buffer_offset,
        .size = size
    };

    command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, stage_mask,
                                   vk::DependencyFlagBits::eByRegion, {}, buffer_barrier, {});


    buffer_offset += size;
    available_size -= size;
}

void StreamBuffer::Flush() {
    const u32 flush_size = buffer_offset - flush_start;
    if (flush_size > 0) {
        vk::CommandBuffer command_buffer = scheduler.GetUploadCommandBuffer();
        VmaAllocator allocator = instance.GetAllocator();

        const u32 flush_size = buffer_offset - flush_start;
        const vk::BufferCopy copy_region = {
            .srcOffset = flush_start,
            .dstOffset = flush_start,
            .size = flush_size
        };

        vmaFlushAllocation(allocator, allocation, flush_start, flush_size);
        command_buffer.copyBuffer(staging.buffer, buffer, copy_region);

        // Lock the region
        const LockedRegion region = {
            .size = flush_size,
            .fence_counter = scheduler.GetFenceCounter()
        };

        regions.emplace(flush_start, region);
        flush_start = buffer_offset;
    }
}

void StreamBuffer::Invalidate() {
    buffer_offset = 0;
    flush_start = 0;
}

bool StreamBuffer::UnlockFreeRegions(u32 target_size) {
    available_size = 0;

    // Free regions that don't need waiting
    auto it = regions.lower_bound(buffer_offset);
    while (it != regions.end()) {
        const auto& [offset, region] = *it;
        if (region.fence_counter <= scheduler.GetFenceCounter()) {
            available_size += region.size;
            it = regions.erase(it);
        }
        else {
            break;
        }
    }

    // If that wasn't enough, try waiting for some fences
    while (available_size < target_size) {
        const auto& [offset, region] = *it;

        if (region.fence_counter > scheduler.GetFenceCounter()) {
            scheduler.WaitFence(region.fence_counter);
        }

        available_size += region.size;
        it = regions.erase(it);
    }

    return available_size >= target_size;
}

} // namespace Vulkan
