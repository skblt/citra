// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class LayoutTracker {
    static constexpr u32 LAYOUT_BITS = 3;
    static constexpr u32 MAX_LAYOUTS = (1 << LAYOUT_BITS);
    static constexpr u32 LAYOUT_MASK = MAX_LAYOUTS - 1;

    // Build layout pattern masks at compile time for fast range equality checks
    static constexpr auto LAYOUT_PATTERNS = []() {
        std::array<u64, MAX_LAYOUTS> patterns{};
        for (u32 layout = 0; layout < MAX_LAYOUTS; layout++) {
            for (u32 i = 0; i < 16; i++) {
                patterns[layout] <<= LAYOUT_BITS;
                patterns[layout] |= layout;
            }
        }

        return patterns;
    }();

public:
    LayoutTracker() = default;

    /// Returns the image layout of the provided level
    [[nodiscard]] constexpr vk::ImageLayout GetLayout(u32 level) const {
        const u32 shift = level * LAYOUT_BITS;
        return static_cast<vk::ImageLayout>((layouts >> shift) & LAYOUT_MASK);
    }

    /// Returns true if the level and layer range provided has the same layout
    [[nodiscard]] constexpr bool IsRangeEqual(vk::ImageLayout layout, u32 level,
                                              u32 level_count) const {
        const u32 shift = level * LAYOUT_BITS;
        const u64 range_mask = (1ull << level_count * LAYOUT_BITS) - 1;
        const u64 pattern = LAYOUT_PATTERNS[static_cast<u64>(layout)];
        return ((layouts >> shift) & range_mask) == (pattern & range_mask);
    }

    /// Sets the image layout of the provided level
    constexpr void SetLayout(vk::ImageLayout layout, u32 level, u32 level_count = 1) {
        const u32 shift = level * LAYOUT_BITS;
        const u64 range_mask = (1ull << level_count * LAYOUT_BITS) - 1;
        const u64 pattern = LAYOUT_PATTERNS[static_cast<u64>(layout)];
        layouts &= ~(range_mask << shift);
        layouts |= (pattern & range_mask) << shift;
    }

    /// Calls func for each continuous layout range
    template <typename T>
    void ForEachLayoutRange(u32 level, u32 level_count, vk::ImageLayout new_layout, T&& func) {
        u32 start_level = level;
        u32 end_level = level + level_count;
        auto current_layout = GetLayout(level);

        while (level < end_level) {
            level++;
            const auto layout = GetLayout(level);
            if (layout != current_layout || level == end_level) {
                if (current_layout != new_layout) {
                    func(start_level, level - start_level, current_layout);
                }
                current_layout = layout;
                start_level = level;
            }
        }
    }

public:
    u64 layouts{};
};

} // namespace Vulkan
