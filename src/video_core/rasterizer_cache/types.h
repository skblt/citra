// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "common/math_util.h"
#include "common/vector_math.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace OpenGL {

// A union for both color and depth/stencil clear values
union ClearValue {
    Common::Vec4f color;
    struct {
        float depth;
        u8 stencil;
    };
};

struct Subresource {
    auto operator<=>(const Subresource&) const = default;

    SurfaceType type;
    Common::Rectangle<u32> region;
    u32 level = 0;
    u32 layer = 0;
};

} // namespace OpenGL
