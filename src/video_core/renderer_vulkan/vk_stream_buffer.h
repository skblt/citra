// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <map>
#include <span>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_common.h"

VK_DEFINE_HANDLE(VmaAllocation)

namespace Vulkan {

class Instance;
class TaskScheduler;

constexpr u32 MAX_BUFFER_VIEWS = 3;

struct LockedRegion {
    u32 size = 0;
    u64 fence_counter = 0;
};

struct StagingBuffer {
    StagingBuffer(const Instance& instance, u32 size, vk::BufferUsageFlags usage);
    ~StagingBuffer();

    const Instance& instance;
    vk::Buffer buffer{};
    VmaAllocation allocation{};
    std::span<std::byte> mapped{};
};

class StreamBuffer {
public:
    StreamBuffer(const Instance& instance, TaskScheduler& scheduler, u32 size,
                 vk::BufferUsageFlagBits usage, std::span<const vk::Format> views);
    ~StreamBuffer();

    std::tuple<u8*, u32, bool> Map(u32 size, u32 alignment = 0);

    /// Commits size bytes from the currently mapped staging memory
    void Commit(u32 size = 0);

    /// Flushes staging memory to the GPU buffer
    void Flush();

    /// Returns the Vulkan buffer handle
    vk::Buffer GetHandle() const {
        return buffer;
    }

    /// Returns an immutable reference to the requested buffer view
    const vk::BufferView& GetView(u32 index = 0) const {
        ASSERT(index < view_count);
        return views[index];
    }

private:
    /// Invalidates the buffer offsets
    void Invalidate();

    /// Removes the lock on regions whose fence counter has been reached by the GPU
    bool UnlockFreeRegions(u32 target_size);

private:
    struct Bucket {
        bool invalid;
        u32 fence_counter;
        u32 offset;
    };

    const Instance& instance;
    TaskScheduler& scheduler;
    StagingBuffer staging;

    vk::Buffer buffer{};
    VmaAllocation allocation{};
    vk::BufferUsageFlagBits usage;
    u32 total_size = 0;
    std::array<vk::BufferView, MAX_BUFFER_VIEWS> views{};
    std::size_t view_count = 0;

    u32 bucket_size = 0;
    std::array<Bucket, SCHEDULER_COMMAND_COUNT> buckets{};
};

} // namespace Vulkan
