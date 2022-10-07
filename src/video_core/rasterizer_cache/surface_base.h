// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include "common/alignment.h"
#include "common/assert.h"
#include "video_core/rasterizer_cache/surface_params.h"

namespace VideoCore {

using SurfaceRegions = boost::icl::interval_set<PAddr, std::less, SurfaceInterval>;

/**
 * A watcher that notifies whether a cached surface has been changed. This is useful for caching
 * surface collection objects, including texture cube and mipmap.
 */
template <class S>
class SurfaceWatcher {
public:
    explicit SurfaceWatcher(std::weak_ptr<S>&& surface) : surface(std::move(surface)) {}

    /// Checks whether the surface has been changed.
    bool IsValid() const {
        return !surface.expired() && valid;
    }

    /// Marks that the content of the referencing surface has been updated to the watcher user.
    void Validate() {
        ASSERT(!surface.expired());
        valid = true;
    }

    /// Gets the referencing surface. Returns null if the surface has been destroyed
    std::shared_ptr<S> Get() const {
        return surface.lock();
    }

public:
    std::weak_ptr<S> surface;
    bool valid = false;
};

template <class S>
class SurfaceBase : public SurfaceParams, public std::enable_shared_from_this<S> {
    using Watcher = SurfaceWatcher<S>;

public:
    SurfaceBase(SurfaceParams& params) : SurfaceParams{params} {}
    virtual ~SurfaceBase() = default;

    /// Returns true when this surface can be used to fill the fill_interval of dest_surface
    bool CanFill(const SurfaceParams& dest_surface, SurfaceInterval fill_interval) const;

    /// Returns true when copy_interval of dest_surface can be validated by copying from this
    /// surface
    bool CanCopy(const SurfaceParams& dest_surface, SurfaceInterval copy_interval) const;

    /// Returns the region of the biggest valid rectange within interval
    SurfaceInterval GetCopyableInterval(const SurfaceParams& params) const;

    /// Creates a surface watcher linked to this surface
    std::shared_ptr<Watcher> CreateWatcher();

    /// Invalidates all watchers linked to this surface
    void InvalidateAllWatcher();

    /// Removes any linked watchers from this surface
    void UnlinkAllWatcher();

    /// Returns true when the region denoted by interval is valid
    bool IsRegionValid(SurfaceInterval interval) const {
        return (invalid_regions.find(interval) == invalid_regions.end());
    }

    /// Returns true when the entire surface is invalid
    bool IsSurfaceFullyInvalid() const {
        auto interval = GetInterval();
        return *invalid_regions.equal_range(interval).first == interval;
    }

public:
    bool registered = false;
    SurfaceRegions invalid_regions;
    std::array<std::shared_ptr<Watcher>, 7> level_watchers;
    u32 max_level = 0;
    std::array<u8, 4> fill_data;
    u32 fill_size = 0;

public:
    u32 watcher_count = 0;
    std::array<std::weak_ptr<Watcher>, 8> watchers;
};

template <class S>
bool SurfaceBase<S>::CanFill(const SurfaceParams& dest_surface,
                             SurfaceInterval fill_interval) const {
    if (type == SurfaceType::Fill && IsRegionValid(fill_interval) &&
        boost::icl::first(fill_interval) >= addr &&
        boost::icl::last_next(fill_interval) <= end && // dest_surface is within our fill range
        dest_surface.FromInterval(fill_interval).GetInterval() ==
            fill_interval) { // make sure interval is a rectangle in dest surface

        if (fill_size * 8 != dest_surface.GetFormatBpp()) {
            // Check if bits repeat for our fill_size
            const u32 dest_bytes_per_pixel = std::max(dest_surface.GetFormatBpp() / 8, 1u);
            std::vector<u8> fill_test(fill_size * dest_bytes_per_pixel);

            for (u32 i = 0; i < dest_bytes_per_pixel; ++i)
                std::memcpy(&fill_test[i * fill_size], &fill_data[0], fill_size);

            for (u32 i = 0; i < fill_size; ++i)
                if (std::memcmp(&fill_test[dest_bytes_per_pixel * i], &fill_test[0],
                                dest_bytes_per_pixel) != 0)
                    return false;

            if (dest_surface.GetFormatBpp() == 4 && (fill_test[0] & 0xF) != (fill_test[0] >> 4))
                return false;
        }
        return true;
    }
    return false;
}

template <class S>
bool SurfaceBase<S>::CanCopy(const SurfaceParams& dest_surface,
                             SurfaceInterval copy_interval) const {
    SurfaceParams subrect_params = dest_surface.FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval);
    if (CanSubRect(subrect_params))
        return true;

    if (CanFill(dest_surface, copy_interval))
        return true;

    return false;
}

template <class S>
SurfaceInterval SurfaceBase<S>::GetCopyableInterval(const SurfaceParams& params) const {
    SurfaceInterval result{};
    const u32 tile_align = params.BytesInPixels(params.is_tiled ? 8 * 8 : 1);
    const auto valid_regions =
        SurfaceRegions{params.GetInterval() & GetInterval()} - invalid_regions;

    for (auto& valid_interval : valid_regions) {
        const SurfaceInterval aligned_interval{
            params.addr +
                Common::AlignUp(boost::icl::first(valid_interval) - params.addr, tile_align),
            params.addr +
                Common::AlignDown(boost::icl::last_next(valid_interval) - params.addr, tile_align)};

        if (params.BytesInPixels(tile_align) > boost::icl::length(valid_interval) ||
            boost::icl::length(aligned_interval) == 0) {
            continue;
        }

        // Get the rectangle within aligned_interval
        const u32 stride_bytes = params.BytesInPixels(params.stride) * (params.is_tiled ? 8 : 1);
        SurfaceInterval rect_interval{
            params.addr +
                Common::AlignUp(boost::icl::first(aligned_interval) - params.addr, stride_bytes),
            params.addr + Common::AlignDown(boost::icl::last_next(aligned_interval) - params.addr,
                                            stride_bytes),
        };

        if (boost::icl::first(rect_interval) > boost::icl::last_next(rect_interval)) {
            // 1 row
            rect_interval = aligned_interval;
        } else if (boost::icl::length(rect_interval) == 0) {
            // 2 rows that do not make a rectangle, return the larger one
            const SurfaceInterval row1{boost::icl::first(aligned_interval),
                                       boost::icl::first(rect_interval)};
            const SurfaceInterval row2{boost::icl::first(rect_interval),
                                       boost::icl::last_next(aligned_interval)};
            rect_interval = (boost::icl::length(row1) > boost::icl::length(row2)) ? row1 : row2;
        }

        if (boost::icl::length(rect_interval) > boost::icl::length(result)) {
            result = rect_interval;
        }
    }
    return result;
}

template <class S>
auto SurfaceBase<S>::CreateWatcher() -> std::shared_ptr<Watcher> {
    auto weak_ptr = reinterpret_cast<S*>(this)->weak_from_this();
    auto watcher = std::make_shared<Watcher>(std::move(weak_ptr));
    watchers[watcher_count++] = watcher;
    return watcher;
}

template <class S>
void SurfaceBase<S>::InvalidateAllWatcher() {
    for (const auto& watcher : watchers) {
        if (auto locked = watcher.lock()) {
            locked->valid = false;
        }
    }
}

template <class S>
void SurfaceBase<S>::UnlinkAllWatcher() {
    for (const auto& watcher : watchers) {
        if (auto locked = watcher.lock()) {
            locked->valid = false;
            locked->surface.reset();
        }
    }

    watchers = {};
    watcher_count = 0;
}

} // namespace VideoCore
