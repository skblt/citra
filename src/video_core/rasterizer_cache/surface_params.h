// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_cache/utils.h"

namespace VideoCore {

class SurfaceParams {
public:
    /// Returns true if other_surface matches exactly params
    bool ExactMatch(const SurfaceParams& other_surface) const;

    /// Returns true if sub_surface is a subrect of params
    bool CanSubRect(const SurfaceParams& sub_surface) const;

    /// Returns true if params can be expanded to match expanded_surface
    bool CanExpand(const SurfaceParams& expanded_surface) const;

    /// Returns true if params can be used for texcopy
    bool CanTexCopy(const SurfaceParams& texcopy_params) const;

    /// Updates remaining members from the already set addr, width, height and pixel_format
    void UpdateParams();

    /// Returns the unscaled rectangle referenced by sub_surface
    Rect2D GetSubRect(const SurfaceParams& sub_surface) const;

    /// Returns the scaled rectangle referenced by sub_surface
    Rect2D GetScaledSubRect(const SurfaceParams& sub_surface) const;

    /// Returns the outer rectangle containing interval
    SurfaceParams FromInterval(SurfaceInterval interval) const;

    /// Returns the address interval referenced by unscaled_rect
    SurfaceInterval GetSubRectInterval(Rect2D unscaled_rect) const;

    [[nodiscard]] SurfaceInterval GetInterval() const noexcept {
        return SurfaceInterval{addr, end};
    }

    [[nodiscard]] u32 GetFormatBpp() const noexcept {
        return VideoCore::GetFormatBpp(pixel_format);
    }

    [[nodiscard]] u32 GetScaledWidth() const noexcept {
        return width * res_scale;
    }

    [[nodiscard]] u32 GetScaledHeight() const noexcept {
        return height * res_scale;
    }

    [[nodiscard]] Rect2D GetRect() const noexcept {
        return Rect2D{0, height, width, 0};
    }

    [[nodiscard]] Rect2D GetScaledRect() const noexcept {
        return Rect2D{0, GetScaledHeight(), GetScaledWidth(), 0};
    }

    [[nodiscard]] u32 PixelsInBytes(u32 size) const noexcept {
        return size * 8 / GetFormatBpp();
    }

    [[nodiscard]] u32 BytesInPixels(u32 pixels) const noexcept {
        return pixels * GetFormatBpp() / 8;
    }

public:
    PAddr addr = 0;
    PAddr end = 0;
    u32 size = 0;

    u32 width = 0;
    u32 height = 0;
    u32 stride = 0;
    u32 levels = 1;
    u16 res_scale = 1;

    bool is_tiled = false;
    TextureType texture_type = TextureType::Texture2D;
    PixelFormat pixel_format = PixelFormat::Invalid;
    SurfaceType type = SurfaceType::Invalid;
};

} // namespace VideoCore
