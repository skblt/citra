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
class Scheduler;

struct StagingBuffer {
    StagingBuffer(const Instance& instance, std::size_t size, bool readback);
    ~StagingBuffer();

    const Instance& instance;
    vk::Buffer buffer{};
    VmaAllocation allocation{};
    std::span<std::byte> mapped{};
};

class StreamBuffer {
    static constexpr std::size_t MAX_BUFFER_VIEWS = 3;
    static constexpr std::size_t BUCKET_COUNT = 32;

public:
    StreamBuffer(const Instance& instance, Scheduler& scheduler, std::size_t size,
                 std::size_t alignment = 16, bool readback = false);
    StreamBuffer(const Instance& instance, Scheduler& scheduler, std::size_t size,
                 vk::BufferUsageFlagBits usage, std::span<const vk::Format> views,
                 std::size_t alignment = 16, bool readback = false);
    ~StreamBuffer();

    StreamBuffer(const StreamBuffer&) = delete;
    StreamBuffer& operator=(const StreamBuffer&) = delete;

    /// Maps aligned staging memory of size bytes
    std::tuple<u8*, std::size_t, bool> Map(std::size_t size);

    /// Commits size bytes from the currently mapped staging memory
    void Commit(std::size_t size = 0);

    /// Flushes staging memory to the GPU buffer
    void Flush();

    /// Invalidates staging memory for reading
    void Invalidate();

    /// Returns the GPU buffer handle
    [[nodiscard]] vk::Buffer GetHandle() const {
        return gpu_buffer;
    }

    /// Returns the staging buffer handle
    [[nodiscard]] vk::Buffer GetStagingHandle() const {
        return staging.buffer;
    }

    /// Returns an immutable reference to the requested buffer view
    [[nodiscard]] const vk::BufferView& GetView(u32 index = 0) const {
        ASSERT(index < view_count);
        return views[index];
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    StagingBuffer staging;
    vk::Buffer gpu_buffer{};
    VmaAllocation allocation{};
    vk::BufferUsageFlagBits usage;
    std::array<vk::BufferView, MAX_BUFFER_VIEWS> views{};
    std::size_t view_count = 0;
    std::size_t total_size = 0;
    std::size_t bucket_size = 0;
    std::size_t buffer_offset = 0;
    std::size_t flush_offset = 0;
    std::size_t bucket_index = 0;
    std::size_t alignment;
    bool readback = false;
    std::array<u64, BUCKET_COUNT> ticks{};
};

} // namespace Vulkan
