// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

// Include the vulkan platform specific header
#if defined(ANDROID) || defined (__ANDROID__)
  #define VK_USE_PLATFORM_ANDROID_KHR 1
#elif defined(_WIN32)
  #define VK_USE_PLATFORM_WIN32_KHR 1
#elif defined(__APPLE__)
  #define VK_USE_PLATFORM_MACOS_MVK 1
  #define VK_USE_PLATFORM_METAL_EXT 1
#else
  #ifdef WAYLAND_DISPLAY
    #define VK_USE_PLATFORM_WAYLAND_KHR 1
  #else // wayland
    #define VK_USE_PLATFORM_XLIB_KHR 1
  #endif
#endif

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vector>
#include "common/logging/log.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

inline vk::SurfaceKHR CreateSurface(const vk::Instance& instance, const Frontend::EmuWindow& emu_window) {
    const auto& window_info = emu_window.GetWindowInfo();
    vk::SurfaceKHR surface;

#if VK_USE_PLATFORM_WIN32_KHR
    if (window_info.type == Frontend::WindowSystemType::Windows) {
        const vk::Win32SurfaceCreateInfoKHR win32_ci = {
            .hinstance = nullptr,
            .hwnd = static_cast<HWND>(window_info.render_surface)
        };

        if (instance.createWin32SurfaceKHR(&win32_ci, nullptr, &surface) != vk::Result::eSuccess) {
            LOG_CRITICAL(Render_Vulkan, "Failed to initialize Win32 surface");
        }
    }
#elif VK_USE_PLATFORM_XLIB_KHR
    if (window_info.type == Frontend::WindowSystemType::X11) {
        const vk::XlibSurfaceCreateInfoKHR xlib_ci{{},
            static_cast<Display*>(window_info.display_connection),
            reinterpret_cast<Window>(window_info.render_surface)};
        if (instance.createXlibSurfaceKHR(&xlib_ci, nullptr, &surface) != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Xlib surface");
            UNREACHABLE();
        }
    }

#elif VK_USE_PLATFORM_WAYLAND_KHR
    if (window_info.type == Frontend::WindowSystemType::Wayland) {
        const vk::WaylandSurfaceCreateInfoKHR wayland_ci{{},
            static_cast<wl_display*>(window_info.display_connection),
            static_cast<wl_surface*>(window_info.render_surface)};
        if (instance.createWaylandSurfaceKHR(&wayland_ci, nullptr, &surface) != vk::Result::eSuccess) {
            LOG_ERROR(Render_Vulkan, "Failed to initialize Wayland surface");
            UNREACHABLE();
        }
    }
#endif

    if (!surface) {
        LOG_CRITICAL(Render_Vulkan, "Presentation not supported on this platform");
    }

    return surface;
}

inline auto GetInstanceExtensions(Frontend::WindowSystemType window_type, bool enable_debug_utils) {
    const auto properties = vk::enumerateInstanceExtensionProperties();
    if (properties.empty()) {
        LOG_ERROR(Render_Vulkan, "Failed to query extension properties");
        return std::vector<const char*>{};
    }

    // Add the windowing system specific extension
    std::vector<const char*> extensions;
    extensions.reserve(6);

    switch (window_type) {
    case Frontend::WindowSystemType::Headless:
        break;
#if VK_USE_PLATFORM_WIN32_KHR
    case Frontend::WindowSystemType::Windows:
        extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
        break;
#elif VK_USE_PLATFORM_XLIB_KHR
    case Frontend::WindowSystemType::X11:
        extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
        break;
#elif VK_USE_PLATFORM_WAYLAND_KHR
    case Frontend::WindowSystemType::Wayland:
        extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
        break;
#endif
    default:
        LOG_ERROR(Render_Vulkan, "Presentation not supported on this platform");
        break;
    }

    if (window_type != Frontend::WindowSystemType::Headless) {
        extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    }

    if (enable_debug_utils) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    for (const char* extension : extensions) {
        const auto iter = std::ranges::find_if(properties, [extension](const auto& prop) {
            return std::strcmp(extension, prop.extensionName) == 0;
        });

        if (iter == properties.end()) {
            LOG_ERROR(Render_Vulkan, "Required instance extension {} is not available", extension);
            return std::vector<const char*>{};
        }
    }

    return extensions;
}

} // namespace Vulkan
