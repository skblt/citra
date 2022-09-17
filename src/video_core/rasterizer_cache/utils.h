// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include "common/hash.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/rasterizer_cache/types.h"

namespace VideoCore {

struct HostTextureTag {
    PixelFormat format{};
    u32 width = 0;
    u32 height = 0;
    u32 layers = 1;

    auto operator<=>(const HostTextureTag&) const noexcept = default;

    const u64 Hash() const {
        return Common::ComputeHash64(this, sizeof(HostTextureTag));
    }
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

class SurfaceParams;

[[nodiscard]] ClearValue MakeClearValue(SurfaceType type, PixelFormat format, const u8* fill_data);

void SwizzleTexture(const SurfaceParams& params, u32 start_offset,
                    std::span<std::byte> source_linear, std::span<std::byte> dest_tiled);

/**
 * Converts a morton swizzled texture to linear format.
 *
 * @param params Structure used to query the surface information.
 * @param start_offset Is the offset at which the source_tiled span begins
 * @param source_tiled The source morton swizzled data.
 * @param dest_linear The output buffer where the generated linear data will be written to.
 */
void UnswizzleTexture(const SurfaceParams& params, u32 start_offset,
                      std::span<std::byte> source_tiled, std::span<std::byte> dest_linear);

} // namespace VideoCore

namespace std {
template <>
struct hash<VideoCore::HostTextureTag> {
    std::size_t operator()(const VideoCore::HostTextureTag& tag) const noexcept {
        return tag.Hash();
    }
};

template <>
struct hash<VideoCore::TextureCubeConfig> {
    std::size_t operator()(const VideoCore::TextureCubeConfig& config) const noexcept {
        return config.Hash();
    }
};
} // namespace std
