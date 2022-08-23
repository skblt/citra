// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <algorithm>
#include <span>
#include <optional>
#include <type_traits>
#include "common/hash.h"
#include "video_core/transform_cache/pixel_format.h"
#include "video_core/transform_cache/types.h"
#include "video_core/transform_cache/slot_vector.h"

namespace VideoCore {

constexpr u32 MAX_PICA_LEVELS = 8;

using SurfaceId = SlotId;
using SurfaceViewId = SlotId;
using SurfaceAllocId = SlotId;
using FramebufferId = SlotId;

enum class SurfaceFlagBits : u32 {
    AcceleratedUpload = 1 << 0,  ///< Upload can be accelerated in the GPU
    RequiresConvertion = 1 << 1, ///< Guest format is not supported natively and it has to be converted
    GPUInvalidated = 1 << 2,     ///< Contents have been modified from the host GPU
    CPUInvalidated = 1 << 3,     ///< Contents have been modified from the guest CPU
    Tracked = 1 << 4,            ///< Writes and reads are being hooked from the CPU JIT
    Registered = 1 << 5,         ///< True when the image is registered
    Picked = 1 << 6,             ///< Temporary flag to mark the image as picked
};

DECLARE_ENUM_FLAG_OPERATORS(SurfaceFlagBits);

struct SurfaceInfo {
    auto operator<=>(const SurfaceInfo& other) const noexcept = default;

    VAddr addr = 0;
    u32 byte_size = 0;
    VAddr addr_end = 0;
    PixelFormat format = PixelFormat::Invalid;
    u32 levels = 1;
    bool is_tiled = false;

    /**
     * The size member dictates what dimentions the allocated texture will have.
     * That sometimes might include padding, especially when the surface is being used
     * as a framebuffer, where games commonly allocate a 256x512 buffer and only render to the
     * lower 240x400 (LCD resolution) portion. This is done due to hardware limitations
     * regarding texture sizes by the PICA and seems to be cheaper than rendering to the
     * entire 256x512 region and downsampling it. The real_size dictates the actual size
     * of the surface and is used in display transfer operations to crop the additional padding.
     **/
    Extent real_size{0, 0};
    Extent size{0, 0};
};

struct NullSurfaceParams {};

/// Properties used to create and locate a SurfaceView
struct SurfaceViewInfo {
    auto operator<=>(const SurfaceViewInfo& other) const noexcept = default;

    [[nodiscard]] bool IsRenderTarget() const noexcept;

    SurfaceViewType type{};
    PixelFormat format{};
    u32 layers = 1;
};

struct Surface {
    explicit Surface(const SurfaceInfo& info);

    [[nodiscard]] std::optional<u32> IsMipLevel(PAddr other_addr);

    [[nodiscard]] SurfaceViewId FindView(const SurfaceViewInfo& view_info) const noexcept;

    void TrackView(const SurfaceViewInfo& view_info, SurfaceViewId image_view_id);

    [[nodiscard]] bool Overlaps(PAddr overlap_addr, u32 overlap_size) const noexcept {
        const PAddr overlap_end = overlap_addr + overlap_size;
        return info.addr < overlap_end && overlap_addr < info.addr_end;
    }

    SurfaceInfo info;
    SurfaceFlagBits flags = SurfaceFlagBits::CPUInvalidated;

    u64 modification_tick = 0;
    u64 frame_tick = 0;

    std::array<u32, MAX_PICA_LEVELS> mip_level_offsets{};
    std::vector<SurfaceViewInfo> surface_view_infos;
    std::vector<SurfaceViewId> surface_view_ids;
};

struct SurfaceView {
    explicit SurfaceView(const SurfaceViewInfo& info,
                         const SurfaceInfo& surface_info, SurfaceId surface_id);

    SurfaceId image_id{};
    PixelFormat format{};
    SurfaceViewType type{};
    u32 layers = 1;
    Extent size{0, 0};

    u64 invalidation_tick = 0;
    u64 modification_tick = 0;
};

/// Framebuffer properties used to lookup a framebuffer
struct RenderTargets {
    constexpr auto operator<=>(const RenderTargets&) const noexcept = default;

    constexpr bool Contains(std::span<const SurfaceViewId> elements) const noexcept {
        const auto contains = [elements](SurfaceViewId item) {
            return std::ranges::find(elements, item) != elements.end();
        };

        return contains(color_buffer_id) || contains(depth_buffer_id);
    }

    SurfaceViewId color_buffer_id;
    SurfaceViewId depth_buffer_id;
    Extent size;
};

} // namespace VideoCore


namespace std {

template <>
struct hash<VideoCore::RenderTargets> {
    size_t operator()(const VideoCore::RenderTargets& rt) const noexcept {
        return Common::ComputeHash64(&rt, sizeof(VideoCore::RenderTargets));
    }
};

} // namespace std
