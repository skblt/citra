// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <boost/icl/interval.hpp>
#include "common/math_util.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace VideoCore {

enum class SurfaceType {
    Color = 0,
    Texture = 1,
    Depth = 2,
    DepthStencil = 3,
    Fill = 4,
    Invalid = 5
};

class CachedSurface;
using Surface = std::shared_ptr<CachedSurface>;
using SurfaceInterval = boost::icl::right_open_interval<PAddr>;

/**
 * Stores properties about a particular surface such as its address, width, height.
 * Also used by the cache as a key to search for cached surfaces.
 */
struct SurfaceParams {
public:
    /// Returns the surface type the holds the provided pixel format
    static SurfaceType GetFormatType(PixelFormat pixel_format);
    static bool CheckFormatsBlittable(PixelFormat source, PixelFormat dest);

    /// Surface comparison traits
    bool IsExactMatch(const SurfaceParams& other_surface) const;
    bool CanSubRect(const SurfaceParams& sub_surface) const;
    bool CanExpand(const SurfaceParams& expanded_surface) const;
    bool CanTexCopy(const SurfaceParams& texcopy_params) const;

    /// Updates the rest of the members from the already set addr, width, height and pixel_format
    void UpdateParams();

    /// Returns the bounds of the surface being contained in the current one
    Common::Rectangle<u32> GetSubRect(const SurfaceParams& sub_surface) const;
    Common::Rectangle<u32> GetScaledSubRect(const SurfaceParams& sub_surface) const;

    /// Returns the outer rectangle containing "interval"
    SurfaceParams FromInterval(SurfaceInterval interval) const;

    /// Returns the region of the biggest valid rectange within interval
    SurfaceInterval GetCopyableInterval(const Surface& src_surface) const;
    SurfaceInterval GetSubRectInterval(Common::Rectangle<u32> unscaled_rect) const;

    /// Returns the address interval the surface covers
    SurfaceInterval GetInterval() const {
        return SurfaceInterval{addr, end};
    }

    u32 GetScaledWidth() const {
        return width * res_scale;
    }

    u32 GetScaledHeight() const {
        return height * res_scale;
    }

    Common::Rectangle<u32> GetRect() const {
        return {0, height, width, 0};
    }

    Common::Rectangle<u32> GetScaledRect() const {
        return {0, GetScaledHeight(), GetScaledWidth(), 0};
    }

    u32 PixelsInBytes(u32 size) const {
        return size * 8 / GetFormatBpp(pixel_format);
    }

    u32 BytesInPixels(u32 pixels) const {
        return pixels * GetFormatBpp(pixel_format) / 8;
    }

public:
    PAddr addr = 0;
    PAddr end = 0;
    u32 size = 0;

    u32 width = 0;
    u32 height = 0;
    u32 stride = 0;
    u16 res_scale = 1;

    bool is_tiled = false;
    PixelFormat pixel_format = PixelFormat::Invalid;
    SurfaceType type = SurfaceType::Invalid;
};

} // namespace VideoCore
