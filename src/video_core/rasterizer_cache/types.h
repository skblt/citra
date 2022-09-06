// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <memory>
#include <set>
#include <tuple>
#include "common/common_types.h"
#include "common/math_util.h"
#include "common/vector_math.h"

namespace OpenGL {

// Describes the type of data a texture holds
enum class Aspect { Color = 0, Depth = 1, DepthStencil = 2 };

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

    Aspect aspect;
    Common::Rectangle<u32> region;
    u32 level = 0;
    u32 layer = 0;
};

} // namespace OpenGL
