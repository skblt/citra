// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <boost/icl/interval_set.hpp>
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/types.h"

namespace VideoCore {

using SurfaceRegions = boost::icl::interval_set<PAddr, std::less, SurfaceInterval>;

class SurfaceBase : public SurfaceParams {
public:
    SurfaceBase();
    SurfaceBase(const SurfaceParams& params);
    virtual ~SurfaceBase();

    [[nodiscard]] bool Overlaps(PAddr overlap_addr, size_t overlap_size) const noexcept {
        const PAddr overlap_end = overlap_addr + static_cast<PAddr>(overlap_size);
        return addr < overlap_end && overlap_addr < end;
    }

    [[nodiscard]] AllocationId AllocId() const noexcept {
        return alloc_id;
    }

    /// Returns true when this surface can be used to fill the fill_interval of dest_surface
    bool CanFill(const SurfaceParams& dest_surface, SurfaceInterval fill_interval) const;

    /// Returns true when copy_interval of dest_surface can be validated by copying from this
    /// surface
    bool CanCopy(const SurfaceParams& dest_surface, SurfaceInterval copy_interval) const;

    /// Returns the region of the biggest valid rectange within interval
    SurfaceInterval GetCopyableInterval(const SurfaceParams& params) const;

    /// Returns true when the region denoted by interval is valid
    bool IsRegionValid(SurfaceInterval interval) const {
        return (invalid_regions.find(interval) == invalid_regions.end());
    }

    /// Returns true when the entire surface is invalid
    bool IsFullyInvalid() const {
        auto interval = GetInterval();
        return *invalid_regions.equal_range(interval).first == interval;
    }

public:
    bool picked = false;
    bool registered = false;
    AllocationId alloc_id;
    SurfaceRegions invalid_regions;
    std::array<u8, 4> fill_data;
    u32 fill_size = 0;
};

} // namespace VideoCore
