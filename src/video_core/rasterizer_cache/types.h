// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "common/common_types.h"
#include "common/math_util.h"
#include "common/vector_math.h"

namespace VideoCore {

using Rect2D = Common::Rectangle<u32>;

struct Offset {
    constexpr auto operator<=>(const Offset&) const noexcept = default;

    u32 x = 0;
    u32 y = 0;
};

struct Extent {
    constexpr auto operator<=>(const Extent&) const noexcept = default;

    u32 width = 1;
    u32 height = 1;
};

union ClearValue {
    Common::Vec4f color;
    struct {
        float depth;
        u8 stencil;
    };
};

struct TextureClear {
    u32 texture_level;
    Rect2D texture_rect;
};

struct TextureCopy {
    u32 src_level;
    u32 dst_level;
    u32 src_layer;
    u32 dst_layer;
    Offset src_offset;
    Offset dst_offset;
    Extent extent;
};

struct TextureBlit {
    u32 src_level;
    u32 dst_level;
    u32 src_layer;
    u32 dst_layer;
    Rect2D src_rect;
    Rect2D dst_rect;
};

struct BufferTextureCopy {
    u32 buffer_offset;
    u32 buffer_size;
    Rect2D texture_rect;
    u32 texture_level;
};

struct BufferCopy {
    u32 src_offset;
    u32 dst_offset;
    u32 size;
};

} // namespace OpenGL
