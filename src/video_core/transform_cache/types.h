// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <compare>
#include "common/common_types.h"

namespace VideoCore {

enum class SurfaceViewType : u32 {
    e2D,
    eCube,
    eShadow2D,
    eProjection,
    eShadowCube
};

struct Offset {
    constexpr auto operator<=>(const Offset&) const noexcept = default;

    s32 x = 0;
    s32 y = 0;
};

struct Extent {
    constexpr auto operator<=>(const Extent&) const noexcept = default;

    u32 width = 1;
    u32 height = 1;
};

struct SurfaceCopy {
    u32 src_level;
    u32 dst_level;
    Offset src_offset;
    Offset dst_offset;
    Extent extent;
};

struct BufferSurfaceCopy {
    u32 buffer_offset;
    u32 buffer_size;
    u32 buffer_row_length;
    u32 buffer_image_height;
    u32 texture_level;
    Offset texture_offset;
    Extent texture_extent;
};

struct BufferCopy {
    u32 src_offset;
    u32 dst_offset;
    u32 size;
};

} // namespace VideoCore
