// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/rasterizer_cache/pixel_format.h"

namespace OpenGL {

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

struct Rect2D {
    constexpr auto operator<=>(const Rect2D&) const noexcept = default;

    Offset offset;
    Extent extent;
};

struct Region2D {
    constexpr auto operator<=>(const Region2D&) const noexcept = default;

    Offset start;
    Offset end;
};

union ClearValue {
    Common::Vec4f color;
    struct {
        float depth;
        u8 stencil;
    };
};

struct ClearRect {
    SurfaceType surface_type;
    u32 texture_level;
    Rect2D rect;
};

struct TextureCopy {
    SurfaceType surface_type;
    u32 src_level;
    u32 dst_level;
    Offset src_offset;
    Offset dst_offset;
    Extent extent;
};

struct TextureBlit {
    SurfaceType surface_type;
    u32 src_level;
    u32 dst_level;
    Region2D src_region;
    Region2D dst_region;
};

struct BufferTextureCopy {
    u32 buffer_offset;
    u32 buffer_size;
    u32 buffer_row_length;
    u32 buffer_height;
    SurfaceType surface_type;
    u32 texture_level;
    Offset texture_offset;
    Extent texture_extent;
};

struct BufferCopy {
    u32 src_offset;
    u32 dst_offset;
    u32 size;
};

} // namespace OpenGL
