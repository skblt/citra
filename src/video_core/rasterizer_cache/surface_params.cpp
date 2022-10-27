// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include "video_core/rasterizer_cache/surface_params.h"

namespace VideoCore {

SurfaceParams SurfaceParams::FromInterval(SurfaceInterval interval) const {
    SurfaceParams params = *this;
    const u32 tiled_size = is_tiled ? 8 : 1;
    const u32 stride_tiled_bytes = BytesInPixels(stride * tiled_size);

    PAddr aligned_start =
        addr + Common::AlignDown(boost::icl::first(interval) - addr, stride_tiled_bytes);
    PAddr aligned_end =
        addr + Common::AlignUp(boost::icl::last_next(interval) - addr, stride_tiled_bytes);

    if (aligned_end - aligned_start > stride_tiled_bytes) {
        params.addr = aligned_start;
        params.height = (aligned_end - aligned_start) / BytesInPixels(stride);
    } else {
        // 1 row
        ASSERT(aligned_end - aligned_start == stride_tiled_bytes);
        const u32 tiled_alignment = BytesInPixels(is_tiled ? 8 * 8 : 1);

        aligned_start =
            addr + Common::AlignDown(boost::icl::first(interval) - addr, tiled_alignment);
        aligned_end =
            addr + Common::AlignUp(boost::icl::last_next(interval) - addr, tiled_alignment);

        params.addr = aligned_start;
        params.width = PixelsInBytes(aligned_end - aligned_start) / tiled_size;
        params.stride = params.width;
        params.height = tiled_size;
    }

    params.UpdateParams();
    return params;
}

SurfaceInterval SurfaceParams::GetSubRectInterval(Common::Rectangle<u32> unscaled_rect) const {
    if (unscaled_rect.GetHeight() == 0 || unscaled_rect.GetWidth() == 0) {
        return {};
    }

    if (is_tiled) {
        unscaled_rect.left = Common::AlignDown(unscaled_rect.left, 8) * 8;
        unscaled_rect.bottom = Common::AlignDown(unscaled_rect.bottom, 8) / 8;
        unscaled_rect.right = Common::AlignUp(unscaled_rect.right, 8) * 8;
        unscaled_rect.top = Common::AlignUp(unscaled_rect.top, 8) / 8;
    }

    const u32 stride_tiled = !is_tiled ? stride : stride * 8;

    const u32 pixel_offset =
        stride_tiled * (!is_tiled ? unscaled_rect.bottom : (height / 8) - unscaled_rect.top) +
        unscaled_rect.left;

    const u32 pixels = (unscaled_rect.GetHeight() - 1) * stride_tiled + unscaled_rect.GetWidth();

    return {addr + BytesInPixels(pixel_offset), addr + BytesInPixels(pixel_offset + pixels)};
}

Common::Rectangle<u32> SurfaceParams::GetSubRect(const SurfaceParams& sub_surface) const {
    const u32 begin_pixel_index = PixelsInBytes(sub_surface.addr - addr);

    if (is_tiled) {
        const int x0 = (begin_pixel_index % (stride * 8)) / 8;
        const int y0 = (begin_pixel_index / (stride * 8)) * 8;
        // Top to bottom
        return Common::Rectangle<u32>(x0, height - y0, x0 + sub_surface.width,
                                      height - (y0 + sub_surface.height));
    }

    const int x0 = begin_pixel_index % stride;
    const int y0 = begin_pixel_index / stride;
    // Bottom to top
    return Common::Rectangle<u32>(x0, y0 + sub_surface.height, x0 + sub_surface.width, y0);
}

Common::Rectangle<u32> SurfaceParams::GetScaledSubRect(const SurfaceParams& sub_surface) const {
    auto rect = GetSubRect(sub_surface);
    rect.left = rect.left * res_scale;
    rect.right = rect.right * res_scale;
    rect.top = rect.top * res_scale;
    rect.bottom = rect.bottom * res_scale;
    return rect;
}

bool SurfaceParams::ExactMatch(const SurfaceParams& other_surface) const {
    return std::tie(other_surface.addr, other_surface.width, other_surface.height,
                    other_surface.stride, other_surface.pixel_format, other_surface.is_tiled) ==
               std::tie(addr, width, height, stride, pixel_format, is_tiled) &&
           pixel_format != PixelFormat::Invalid;
}

bool SurfaceParams::CanSubRect(const SurfaceParams& sub_surface) const {
    return sub_surface.addr >= addr && sub_surface.end <= end &&
           sub_surface.pixel_format == pixel_format && pixel_format != PixelFormat::Invalid &&
           sub_surface.is_tiled == is_tiled &&
           (sub_surface.addr - addr) % BytesInPixels(is_tiled ? 64 : 1) == 0 &&
           (sub_surface.stride == stride || sub_surface.height <= (is_tiled ? 8u : 1u)) &&
           GetSubRect(sub_surface).right <= stride;
}

bool SurfaceParams::CanExpand(const SurfaceParams& expanded_surface) const {
    return pixel_format != PixelFormat::Invalid && pixel_format == expanded_surface.pixel_format &&
           addr <= expanded_surface.end && expanded_surface.addr <= end &&
           is_tiled == expanded_surface.is_tiled && stride == expanded_surface.stride &&
           (std::max(expanded_surface.addr, addr) - std::min(expanded_surface.addr, addr)) %
                   BytesInPixels(stride * (is_tiled ? 8 : 1)) ==
               0;
}

bool SurfaceParams::CanTexCopy(const SurfaceParams& texcopy_params) const {
    if (pixel_format == PixelFormat::Invalid || addr > texcopy_params.addr ||
        end < texcopy_params.end) {
        return false;
    }

    if (texcopy_params.width != texcopy_params.stride) {
        const u32 tile_stride = BytesInPixels(stride * (is_tiled ? 8 : 1));
        return (texcopy_params.addr - addr) % BytesInPixels(is_tiled ? 64 : 1) == 0 &&
               texcopy_params.width % BytesInPixels(is_tiled ? 64 : 1) == 0 &&
               (texcopy_params.height == 1 || texcopy_params.stride == tile_stride) &&
               ((texcopy_params.addr - addr) % tile_stride) + texcopy_params.width <= tile_stride;
    }

    return FromInterval(texcopy_params.GetInterval()).GetInterval() == texcopy_params.GetInterval();
}

} // namespace VideoCore
