// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <span>
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/rasterizer_cache/types.h"

namespace VideoCore {

class SurfaceParams;
class SurfaceBase;

[[nodiscard]] ClearValue MakeClearValue(const SurfaceBase& fill_surface, PixelFormat format,
                                        SurfaceType type, PAddr copy_addr);

void UnswizzleTexture(const SurfaceParams& unswizzle_info, PAddr start_addr, PAddr end_addr,
                      std::span<std::byte> source_tiled, std::span<std::byte> dest_linear,
                      bool convert = false);

void SwizzleTexture(const SurfaceParams& swizzle_info, PAddr start_addr, PAddr end_addr,
                    std::span<std::byte> source_linear, std::span<std::byte> dest_tiled,
                    bool convert = false);

} // namespace VideoCore
