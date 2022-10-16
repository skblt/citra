// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

inline auto ToVkAccessStageFlags(vk::BufferUsageFlagBits usage) {
    std::pair<vk::AccessFlags, vk::PipelineStageFlags> result{};
    switch (usage) {
    case vk::BufferUsageFlagBits::eVertexBuffer:
        result = std::make_pair(vk::AccessFlagBits::eVertexAttributeRead,
                                vk::PipelineStageFlagBits::eVertexInput);
        break;
    case vk::BufferUsageFlagBits::eIndexBuffer:
        result =
            std::make_pair(vk::AccessFlagBits::eIndexRead, vk::PipelineStageFlagBits::eVertexInput);
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
    const vk::BufferCreateInfo buffer_info = {.size = size, .usage = usage};

    const VmaAllocationCreateInfo alloc_create_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};

    VkBuffer unsafe_buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationInfo alloc_info;
    VmaAllocator allocator = instance.GetAllocator();

    vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_create_info, &unsafe_buffer, &allocation,
                    &alloc_info);

    buffer = vk::Buffer{unsafe_buffer};
    mapped = std::span{reinterpret_cast<std::byte*>(alloc_info.pMappedData), size};
}

StagingBuffer::~StagingBuffer() {
    vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(buffer), allocation);
}

StreamBuffer::StreamBuffer(const Instance& instance, TaskScheduler& scheduler, u32 size,
                           vk::BufferUsageFlagBits usage, std::span<const vk::Format> view_formats)
    : instance{instance}, scheduler{scheduler}, staging{instance, size,
                                                        vk::BufferUsageFlagBits::eTransferSrc},
      usage{usage}, total_size{size * SCHEDULER_COMMAND_COUNT} {

    const vk::BufferCreateInfo buffer_info = {
        .size = total_size, .usage = usage | vk::BufferUsageFlagBits::eTransferDst};

    const VmaAllocationCreateInfo alloc_create_info = {.usage =
                                                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};

    VkBuffer unsafe_buffer = VK_NULL_HANDLE;
    VkBufferCreateInfo unsafe_buffer_info = static_cast<VkBufferCreateInfo>(buffer_info);
    VmaAllocationInfo alloc_info;
    VmaAllocator allocator = instance.GetAllocator();

    vmaCreateBuffer(allocator, &unsafe_buffer_info, &alloc_create_info, &unsafe_buffer, &allocation,
                    &alloc_info);

    buffer = vk::Buffer{unsafe_buffer};

    ASSERT(view_formats.size() < MAX_BUFFER_VIEWS);

    vk::Device device = instance.GetDevice();
    for (std::size_t i = 0; i < view_formats.size(); i++) {
        const vk::BufferViewCreateInfo view_info = {
            .buffer = buffer, .format = view_formats[i], .offset = 0, .range = total_size};

        views[i] = device.createBufferView(view_info);
    }

    view_count = view_formats.size();
    bucket_size = size;
}

StreamBuffer::~StreamBuffer() {
    if (buffer) {
        vk::Device device = instance.GetDevice();
        vmaDestroyBuffer(instance.GetAllocator(), static_cast<VkBuffer>(buffer), allocation);
        for (std::size_t i = 0; i < view_count; i++) {
            device.destroyBufferView(views[i]);
        }
    }
}

std::tuple<u8*, u32, bool> StreamBuffer::Map(u32 size, u32 alignment) {
    ASSERT(size <= total_size && alignment <= total_size);

    const u32 current_bucket = scheduler.GetCurrentSlotIndex();
    auto& bucket = buckets[current_bucket];

    if (alignment > 0) {
        bucket.offset = Common::AlignUp(bucket.offset, alignment);
    }

    if (bucket.offset + size > bucket_size) {
        UNREACHABLE();
    }

    bool invalidate = false;
    if (bucket.invalid) {
        invalidate = true;
        bucket.invalid = false;
    }

    const u32 buffer_offset = current_bucket * bucket_size + bucket.offset;
    u8* mapped = reinterpret_cast<u8*>(staging.mapped.data() + buffer_offset);
    return std::make_tuple(mapped, buffer_offset, invalidate);
}

void StreamBuffer::Commit(u32 size) {
    buckets[scheduler.GetCurrentSlotIndex()].offset += size;
}

void StreamBuffer::Flush() {
    const u32 current_bucket = scheduler.GetCurrentSlotIndex();
    const u32 flush_size = buckets[current_bucket].offset;
    ASSERT(flush_size <= bucket_size);

    if (flush_size > 0) {
        vk::CommandBuffer command_buffer = scheduler.GetUploadCommandBuffer();
        VmaAllocator allocator = instance.GetAllocator();

        const u32 flush_start = current_bucket * bucket_size;
        const vk::BufferCopy copy_region = {
            .srcOffset = flush_start, .dstOffset = flush_start, .size = flush_size};

        vmaFlushAllocation(allocator, allocation, flush_start, flush_size);
        command_buffer.copyBuffer(staging.buffer, buffer, copy_region);

        // Add pipeline barrier for the flushed region
        auto [access_mask, stage_mask] = ToVkAccessStageFlags(usage);
        const vk::BufferMemoryBarrier buffer_barrier = {
            .srcAccessMask = vk::AccessFlagBits::eTransferWrite,
            .dstAccessMask = access_mask,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = flush_start,
            .size = flush_size};

        command_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, stage_mask,
                                       vk::DependencyFlagBits::eByRegion, {}, buffer_barrier, {});
    }

    // Reset the offset of the next bucket
    const u32 next_bucket = (current_bucket + 1) % SCHEDULER_COMMAND_COUNT;
    buckets[next_bucket].offset = 0;
    buckets[next_bucket].invalid = true;
}

} // namespace Vulkan
