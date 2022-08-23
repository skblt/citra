// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "video_core/rasterizer_interface.h"

namespace VideoCore {

class RasterizerAccelerated : public RasterizerInterface {
public:
    RasterizerAccelerated() = default;
    virtual ~RasterizerAccelerated() override = default;

    /// Increase/decrease the number of surface in pages touching the specified region
    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta) override;

private:
    std::array<u16, 0x18000> cached_pages{};
};
} // namespace VideoCore
