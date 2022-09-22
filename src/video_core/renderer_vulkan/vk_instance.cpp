// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <span>
#include <array>
#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_instance.h"

namespace Vulkan {

Instance::Instance(Frontend::EmuWindow& window) {
    auto window_info = window.GetWindowInfo();

    // Fetch instance independant function pointers
    vk::DynamicLoader dl;
    auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Enable the instance extensions the backend uses
    auto extensions = GetInstanceExtensions(window_info.type, true);

    // We require a Vulkan 1.1 driver
    const u32 available_version = vk::enumerateInstanceVersion();
    if (available_version < VK_API_VERSION_1_1) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan 1.0 is not supported, 1.1 is required!");
    }

    const vk::ApplicationInfo application_info = {
        .pApplicationName = "Citra",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Citra Vulkan",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = available_version
    };

    const std::array layers = {"VK_LAYER_KHRONOS_validation"};
    const vk::InstanceCreateInfo instance_info = {
        .pApplicationInfo = &application_info,
        .enabledLayerCount = static_cast<u32>(layers.size()),
        .ppEnabledLayerNames = layers.data(),
        .enabledExtensionCount = static_cast<u32>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data()
    };

    // Create VkInstance
    instance = vk::createInstance(instance_info);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    surface = CreateSurface(instance, window);

    // TODO: GPU select dialog
    auto physical_devices = instance.enumeratePhysicalDevices();
    physical_device = physical_devices[0];
    device_properties = physical_device.getProperties();

    // Create logical device
    CreateDevice(false);
}

Instance::~Instance() {
    device.waitIdle();
    vmaDestroyAllocator(allocator);
    device.destroy();
    instance.destroySurfaceKHR(surface);
    instance.destroy();
}

bool Instance::IsFormatSupported(vk::Format format, vk::FormatFeatureFlags usage) const {
    static std::unordered_map<vk::Format, vk::FormatProperties> supported;
    if (auto it = supported.find(format); it != supported.end()) {
        return (it->second.optimalTilingFeatures & usage) == usage;
    }

    // Cache format properties so we don't have to query the driver all the time
    const vk::FormatProperties properties = physical_device.getFormatProperties(format);
    supported.insert(std::make_pair(format, properties));

    return (properties.optimalTilingFeatures & usage) == usage;
}

vk::Format Instance::GetFormatAlternative(vk::Format format) const {
    if (format == vk::Format::eUndefined) {
        return format;
    }

    vk::FormatFeatureFlags features = GetFormatFeatures(GetImageAspect(format));
    if (IsFormatSupported(format, features)) {
       return format;
    }

    // Return the most supported alternative format preferably with the
    // same block size according to the Vulkan spec.
    // See 43.3. Required Format Support of the Vulkan spec
    switch (format) {
    case vk::Format::eD24UnormS8Uint:
        return vk::Format::eD32SfloatS8Uint;
    case vk::Format::eX8D24UnormPack32:
        return vk::Format::eD32Sfloat;
    case vk::Format::eR5G5B5A1UnormPack16:
        return vk::Format::eA1R5G5B5UnormPack16;
    case vk::Format::eR8G8B8Unorm:
        return vk::Format::eR8G8B8A8Unorm;
    case vk::Format::eUndefined:
        return vk::Format::eUndefined;
    case vk::Format::eR4G4B4A4UnormPack16:
        // B4G4R4A4 is not guaranteed by the spec to support attachments
        return GetFormatAlternative(vk::Format::eB4G4R4A4UnormPack16);
    default:
        LOG_WARNING(Render_Vulkan, "Format {} doesn't support attachments, falling back to RGBA8",
                                    vk::to_string(format));
        return vk::Format::eR8G8B8A8Unorm;
    }
}

bool Instance::CreateDevice(bool validation_enabled) {
    // Determine required extensions and features
    auto feature_chain = physical_device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                                      vk::PhysicalDeviceDynamicRenderingFeaturesKHR,
                                                      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                                                      vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT>();

    // Not having geometry shaders or wide lines will cause issues with rendering.
    const vk::PhysicalDeviceFeatures available = feature_chain.get().features;
    if (!available.geometryShader && !available.wideLines) {
        LOG_WARNING(Render_Vulkan, "Geometry shaders not availabe! Accelerated rendering not possible!");
    }

    // Enable some common features other emulators like Dolphin use
    const vk::PhysicalDeviceFeatures2 features = {
        .features = {
            .robustBufferAccess = available.robustBufferAccess,
            .geometryShader = available.geometryShader,
            .sampleRateShading = available.sampleRateShading,
            .dualSrcBlend = available.dualSrcBlend,
            .logicOp = available.logicOp,
            .depthClamp = available.depthClamp,
            .largePoints = available.largePoints,
            .samplerAnisotropy = available.samplerAnisotropy,
            .occlusionQueryPrecise = available.occlusionQueryPrecise,
            .fragmentStoresAndAtomics = available.fragmentStoresAndAtomics,
            .shaderStorageImageMultisample = available.shaderStorageImageMultisample,
            .shaderClipDistance = available.shaderClipDistance
        }
    };

    // Enable newer Vulkan features
    auto enabled_features = vk::StructureChain{
        features,
        //feature_chain.get<vk::PhysicalDeviceDynamicRenderingFeaturesKHR>(),
        //feature_chain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>(),
        //feature_chain.get<vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT>()
    };

    auto extension_list = physical_device.enumerateDeviceExtensionProperties();
    if (extension_list.empty()) {
        LOG_CRITICAL(Render_Vulkan, "No extensions supported by device.");
        return false;
    }

    // Helper lambda for adding extensions
    std::array<const char*, 6> enabled_extensions;
    u32 enabled_extension_count = 0;

    auto AddExtension = [&](std::string_view name, bool required) -> bool {
        auto result = std::find_if(extension_list.begin(), extension_list.end(), [&](const auto& prop) {
            return name.compare(prop.extensionName.data());
        });

        if (result != extension_list.end()) {
            LOG_INFO(Render_Vulkan, "Enabling extension: {}", name);
            enabled_extensions[enabled_extension_count++] = name.data();
            return true;
        }

        if (required) {
            LOG_ERROR(Render_Vulkan, "Unable to find required extension {}.", name);
        }

        return false;
    };

    // Add required extensions
    AddExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME, true);

    // Check for optional features
    //dynamic_rendering = AddExtension(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME, false);
    //extended_dynamic_state = AddExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, false);
    //push_descriptors = AddExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, false);

    // Search queue families for graphics and present queues
    auto family_properties = physical_device.getQueueFamilyProperties();
    if (family_properties.empty()) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan physical device reported no queues.");
        return false;
    }

    graphics_queue_family_index = -1;
    present_queue_family_index = -1;
    for (int i = 0; i < family_properties.size(); i++) {
        // Check if queue supports graphics
        if (family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            graphics_queue_family_index = i;

            // If this queue also supports presentation we are finished
            if (physical_device.getSurfaceSupportKHR(i, surface)) {
                present_queue_family_index = i;
                break;
            }
        }

        // Check if queue supports presentation
        if (physical_device.getSurfaceSupportKHR(i, surface)) {
            present_queue_family_index = i;
        }
    }

    if (graphics_queue_family_index == -1 || present_queue_family_index == -1) {
        LOG_CRITICAL(Render_Vulkan, "Unable to find graphics and/or present queues.");
        return false;
    }

    static constexpr float queue_priorities[] = {1.0f};

    const std::array queue_infos = {
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = graphics_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = queue_priorities
        },
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = present_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = queue_priorities
        }
    };

    vk::DeviceCreateInfo device_info = {
        .pNext = &features, // TODO: Change this
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = queue_infos.data(),
        .enabledExtensionCount = enabled_extension_count,
        .ppEnabledExtensionNames = enabled_extensions.data(),
    };

    if (graphics_queue_family_index != present_queue_family_index) {
        device_info.queueCreateInfoCount = 2;
    }

    // Create logical device
    device = physical_device.createDevice(device_info);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    // Grab the graphics and present queues.
    graphics_queue = device.getQueue(graphics_queue_family_index, 0);
    present_queue = device.getQueue(present_queue_family_index, 0);

    // Create the VMA allocator
    CreateAllocator();

    return true;
}

void Instance::CreateAllocator() {
    VmaVulkanFunctions functions = {
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr
    };

    VmaAllocatorCreateInfo allocator_info = {
        .physicalDevice = physical_device,
        .device = device,
        .pVulkanFunctions = &functions,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_1
    };

    if (auto result = vmaCreateAllocator(&allocator_info, &allocator); result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Failed to initialize VMA with error {}", result);
        UNREACHABLE();
    }
}

} // namespace Vulkan
