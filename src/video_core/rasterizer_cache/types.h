// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <boost/icl/right_open_interval.hpp>
#include "common/hash.h"
#include "common/math_util.h"
#include "common/vector_math.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/rasterizer_cache/slot_vector.h"

namespace VideoCore {

constexpr std::size_t MAX_PICA_LEVELS = 8;

using SurfaceId = SlotId;
using FramebufferId = SlotId;
using AllocationId = SlotId;
using SamplerId = SlotId;

/// Fake surface ID for null surfaces
constexpr SurfaceId NULL_SURFACE_ID{0};
/// Fake sampler ID for null samplers
constexpr SamplerId NULL_SAMPLER_ID{0};

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
    SurfaceType type;
    ClearValue value;
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

struct TextureCubeConfig {
    PAddr px;
    PAddr nx;
    PAddr py;
    PAddr ny;
    PAddr pz;
    PAddr nz;
    u32 width;
    Pica::TexturingRegs::TextureFormat format;

    auto operator<=>(const TextureCubeConfig&) const noexcept = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(TextureCubeConfig));
    }
};

struct HostTextureTag {
    PixelFormat format = PixelFormat::Invalid;
    TextureType type = TextureType::Texture2D;
    u32 width = 1;
    u32 stride = 1;
    u32 height = 1;
    u32 levels = 1;
    u32 res_scale = 1;

    auto operator<=>(const HostTextureTag&) const noexcept = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(HostTextureTag));
    }
};

using SurfaceInterval = boost::icl::right_open_interval<PAddr>;

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::TextureCubeConfig> {
    std::size_t operator()(const VideoCore::TextureCubeConfig& config) const noexcept {
        return config.Hash();
    }
};
template <>
struct hash<VideoCore::HostTextureTag> {
    std::size_t operator()(const VideoCore::HostTextureTag& tag) const noexcept {
        return tag.Hash();
    }
};
} // namespace std
