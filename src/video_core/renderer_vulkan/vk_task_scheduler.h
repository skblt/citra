// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <array>
#include <functional>
#include "common/common_types.h"
#include "common/common_funcs.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Buffer;
class Instance;

enum class SubmitMode : u8 {
    SwapchainSynced = 1 << 0, ///< Synchronizes command buffer execution with the swapchain
    Flush = 1 << 1, ///< Causes a GPU command flush, useful for texture downloads
    Shutdown = 1 << 2 ///< Submits all current commands without starting a new command buffer
};

DECLARE_ENUM_FLAG_OPERATORS(SubmitMode);

class TaskScheduler {
public:
    TaskScheduler(const Instance& instance);
    ~TaskScheduler();

    /// Blocks the host until the current command completes execution
    void Synchronize(u32 slot);

    /// Submits the current command to the graphics queue
    void Submit(SubmitMode mode);

    /// Returns the last completed fence counter
    u64 GetFenceCounter() const;

    /// Returns the command buffer used for early upload operations.
    vk::CommandBuffer GetUploadCommandBuffer();

    /// Returns the command buffer used for rendering
    vk::CommandBuffer GetRenderCommandBuffer() const {
        return commands[current_command].render_command_buffer;
    }

    /// Returns the current descriptor pool
    vk::DescriptorPool GetDescriptorPool() const {
        return commands[current_command].descriptor_pool;
    }

    /// Returns the index of the current command slot
    u32 GetCurrentSlotIndex() const {
        return current_command;
    }

    u64 GetHostFenceCounter() const {
        return next_fence_counter - 1;
    }

    vk::Semaphore GetImageAcquiredSemaphore() const {
        return commands[current_command].image_acquired;
    }

    vk::Semaphore GetPresentReadySemaphore() const {
        return commands[current_command].present_ready;
    }

private:
    /// Activates the next command slot and optionally waits for its completion
    void SwitchSlot();

private:
    const Instance& instance;
    u64 next_fence_counter = 1;
    u64 completed_fence_counter = 0;

    struct ExecutionSlot {
        bool use_upload_buffer = false;
        u64 fence_counter = 0;
        vk::Semaphore image_acquired;
        vk::Semaphore present_ready;
        vk::Fence fence;
        vk::DescriptorPool descriptor_pool;
        vk::CommandBuffer render_command_buffer;
        vk::CommandBuffer upload_command_buffer;
    };

    vk::CommandPool command_pool{};
    vk::Semaphore timeline{};
    std::array<ExecutionSlot, SCHEDULER_COMMAND_COUNT> commands{};
    u32 current_command = 0;
};

}  // namespace Vulkan
