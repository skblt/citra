// SPDX-FileCopyrightText: Copyright 2020 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <span>

#include "common/hash.h"
#include "video_core/rasterizer_cache/types.h"

namespace VideoCore {

/// Framebuffer properties used to lookup a framebuffer
struct RenderTargets {
    constexpr auto operator<=>(const RenderTargets&) const noexcept = default;

    constexpr bool Contains(SurfaceId surface_id) const noexcept {
        return surface_id && (color_buffer_id == surface_id || depth_buffer_id == surface_id);
    }

    std::size_t Hash() const noexcept {
        return Common::ComputeHash64(this, sizeof(RenderTargets));
    }

    // Framebuffers are tied to the underlying allocations since
    // surfaces can be quite volatile.
    AllocationId color_buffer_id{};
    AllocationId depth_buffer_id{};
    Extent size{};
};

} // namespace VideoCore

namespace std {

template <>
struct hash<VideoCore::RenderTargets> {
    std::size_t operator()(const VideoCore::RenderTargets& rt) const noexcept {
        return rt.Hash();
    }
};

} // namespace std
