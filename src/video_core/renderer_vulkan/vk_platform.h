// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <vector>
#include "common/common_types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Frontend {
class EmuWindow;
enum class WindowSystemType : u8;
} // namespace Frontend

namespace Vulkan {

std::vector<const char*> GetInstanceExtensions(Frontend::WindowSystemType window_type,
                                               bool enable_debug_utils);

vk::InstanceCreateFlags GetInstanceFlags();

vk::SurfaceKHR CreateSurface(vk::Instance instance, const Frontend::EmuWindow& emu_window);

} // namespace Vulkan
