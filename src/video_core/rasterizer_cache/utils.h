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

/**
 * Converts a morton swizzled texture to linear format.
 *
 * @param unswizzle_info Structure used to query the surface information.
 * @param start_addr The start address of the source_tiled data.
 * @param end_addr The end address of the source_tiled data.
 * @param source_tiled The tiled data to convert.
 * @param dest_linear The output buffer where the generated linear data will be written to.
 */
void UnswizzleTexture(const SurfaceParams& unswizzle_info, PAddr start_addr, PAddr end_addr,
                      std::span<std::byte> source_tiled, std::span<std::byte> dest_linear);

/**
 * Swizzles a linear texture according to the morton code.
 *
 * @param swizzle_info Structure used to query the surface information.
 * @param start_addr The start address of the dest_tiled data.
 * @param end_addr The end address of the dest_tiled data.
 * @param source_tiled The source morton swizzled data.
 * @param dest_linear The output buffer where the generated linear data will be written to.
 */
void SwizzleTexture(const SurfaceParams& swizzle_info, PAddr start_addr, PAddr end_addr,
                    std::span<std::byte> source_linear, std::span<std::byte> dest_tiled);

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
