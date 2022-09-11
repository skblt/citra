// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/rasterizer_cache/rasterizer_cache.h"

namespace VideoCore {

MICROPROFILE_DEFINE(RasterizerCache_BlitSurface, "RasterizerCache", "BlitSurface", MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(RasterizerCache_CopySurface, "RasterizerCache", "CopySurface", MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(RasterizerCache_SurfaceLoad, "RasterizerCache", "Surface Load", MP_RGB(128, 192, 64));
MICROPROFILE_DEFINE(RasterizerCache_SurfaceFlush, "RasterizerCache", "Surface Flush", MP_RGB(128, 192, 64));

} // namespace VideoCore
