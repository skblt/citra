// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <limits>
#include "core/memory.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/video_core.h"

namespace VideoCore {

void RasterizerAccelerated::UpdatePagesCachedCount(PAddr addr, u32 size, int delta) {
    const u32 page_start = addr >> Memory::PAGE_BITS;
    const u32 page_end = ((addr + size - 1) >> Memory::PAGE_BITS);

    u32 uncache_start_addr = 0;
    u32 cache_start_addr = 0;
    u32 uncache_bytes = 0;
    u32 cache_bytes = 0;

    for (u32 page = page_start; page != page_end; page++) {
        auto& count = cached_pages.at(page);

        // Ensure no overflow happens
        if (delta > 0) {
            ASSERT_MSG(count < std::numeric_limits<u16>::max(), "Count will overflow!");
        } else if (delta < 0) {
            ASSERT_MSG(count > 0, "Count will underflow!");
        } else {
            ASSERT_MSG(false, "Delta must be non-zero!");
        }

        // Adds or subtracts 1, as count is a unsigned 8-bit value
        count += delta;

        // Assume delta is either -1 or 1
        if (count == 0) {
            if (uncache_bytes == 0) {
                uncache_start_addr = page << Memory::PAGE_BITS;
            }

            uncache_bytes += Memory::PAGE_SIZE;
        } else if (uncache_bytes > 0) {
            VideoCore::g_memory->RasterizerMarkRegionCached(uncache_start_addr, uncache_bytes,
                                                            false);
            uncache_bytes = 0;
        }

        if (count == 1 && delta > 0) {
            if (cache_bytes == 0) {
                cache_start_addr = page << Memory::PAGE_BITS;
            }

            cache_bytes += Memory::PAGE_SIZE;
        } else if (cache_bytes > 0) {
            VideoCore::g_memory->RasterizerMarkRegionCached(cache_start_addr, cache_bytes,
                                                            true);

            cache_bytes = 0;
        }
    }

    if (uncache_bytes > 0) {
        VideoCore::g_memory->RasterizerMarkRegionCached(uncache_start_addr, uncache_bytes,
                                                        false);
    }

    if (cache_bytes > 0) {
        VideoCore::g_memory->RasterizerMarkRegionCached(cache_start_addr, cache_bytes,
                                                        true);
    }
}

} // namespace VideoCore
