// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/transform_cache/surface.h"
#include "video_core/transform_cache/utils.h"

namespace VideoCore {

Surface::Surface(const SurfaceInfo& info)
    : info(info), mip_level_offsets(CalculateMipLevelOffsets(info)) {

}

[[nodiscard]] std::optional<u32> Surface::IsMipLevel(PAddr other_addr) {
    const u32 offset = other_addr - info.addr;
    if (other_addr < info.addr || offset > info.byte_size) {
        return std::nullopt;
    }

    // Check if the address is referencing a mip level
    const auto end = mip_level_offsets.begin() + info.levels;
    const auto it = std::find(mip_level_offsets.begin(), end, offset);

    if (it == end) {
        return std::nullopt;
    }

    return *it;
}

} // namespace VideoCore
