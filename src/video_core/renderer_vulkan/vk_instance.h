// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Frontend {
class EmuWindow;
}

namespace Vulkan {

/// The global Vulkan instance
class Instance {
public:
    Instance(Frontend::EmuWindow& window);
    ~Instance();

    /// Returns true when the format supports the provided feature flags
    bool IsFormatSupported(vk::Format format, vk::FormatFeatureFlags usage) const;

    /// Returns the most compatible format that supports the provided feature flags
    vk::Format GetFormatAlternative(vk::Format format) const;

    /// Returns the Vulkan instance
    vk::Instance GetInstance() const {
        return instance;
    }

    /// Returns the Vulkan surface
    vk::SurfaceKHR GetSurface() const {
        return surface;
    }

    /// Returns the current physical device
    vk::PhysicalDevice GetPhysicalDevice() const {
        return physical_device;
    }

    /// Returns the Vulkan device
    vk::Device GetDevice() const {
        return device;
    }

    VmaAllocator GetAllocator() const {
        return allocator;
    }

    /// Retrieve queue information
    u32 GetGraphicsQueueFamilyIndex() const {
        return graphics_queue_family_index;
    }

    u32 GetPresentQueueFamilyIndex() const {
        return present_queue_family_index;
    }

    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }

    vk::Queue GetPresentQueue() const {
        return present_queue;
    }

    /// Returns true when VK_KHR_timeline_semaphore is supported
    bool IsTimelineSemaphoreSupported() const {
        return timeline_semaphores;
    }

    /// Returns true when VK_EXT_extended_dynamic_state is supported
    bool IsExtendedDynamicStateSupported() const {
        return extended_dynamic_state;
    }

    /// Returns true when VK_KHR_push_descriptors is supported
    bool IsPushDescriptorsSupported() const {
        return push_descriptors;
    }

    /// Returns the vendor ID of the physical device
    u32 GetVendorID() const {
        return device_properties.vendorID;
    }

    /// Returns the device ID of the physical device
    u32 GetDeviceID() const {
        return device_properties.deviceID;
    }

    /// Returns the pipeline cache unique identifier
    const auto GetPipelineCacheUUID() const {
        return device_properties.pipelineCacheUUID;
    }

    /// Returns the minimum required alignment for uniforms
    vk::DeviceSize UniformMinAlignment() const {
        return device_properties.limits.minUniformBufferOffsetAlignment;
    }

private:
    /// Creates the logical device opportunistically enabling extensions
    bool CreateDevice();

    /// Creates the VMA allocator handle
    void CreateAllocator();

private:
    vk::Device device;
    vk::PhysicalDevice physical_device;
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDeviceProperties device_properties;
    VmaAllocator allocator;
    vk::Queue present_queue;
    vk::Queue graphics_queue;
    u32 present_queue_family_index = 0;
    u32 graphics_queue_family_index = 0;

    bool timeline_semaphores = false;
    bool extended_dynamic_state = false;
    bool push_descriptors = false;
};

} // namespace Vulkan
