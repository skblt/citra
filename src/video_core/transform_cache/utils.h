// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <utility>
#include "video_core/transform_cache/surface.h"
#include "video_core/transform_cache/pixel_format.h"

namespace VideoCore {

enum class MatchFlags {
    Invalid = 1 << 0, ///< Candidate is allowed to be invalid
    Exact = 1 << 1,   ///< Candidate must match image exactly
    SubRect = 1 << 2, ///< Candidate is fully encompassed by image
    Copy = 1 << 3,    ///< Candidate can be used as a copy source
    Expand = 1 << 4,  ///< Candidate fully encompasses image
    TexCopy = 1 << 5  ///< Candidate can be used for texture/display transfer
};

DECLARE_ENUM_FLAG_OPERATORS(MatchFlags);

[[nodiscard]] constexpr bool IsBlockAligned(u32 size, const Surface& surface) {
    // Morton tiled imaged are block instead of pixel aligned
    const u32 pixels = surface.info.is_tiled ? 64 : 1;
    return (size % (pixels * GetFormatBpp(surface.info.format)) / 8) == 0;
}

[[nodiscard]] constexpr u32 PixelsInBytes(u32 size, PixelFormat format) {
    return size * 8 / GetFormatBpp(format);
}

[[nodiscard]] constexpr u32 BytesInPixels(u32 pixels, PixelFormat format) {
    return pixels * GetFormatBpp(format) / 8;
}

[[nodiscard]] constexpr auto MakeSurfaceCopyInfosFromTransferConfig(
        const GPU::Regs::DisplayTransferConfig& config) -> std::pair<SurfaceInfo, SurfaceInfo> {
    using ScalingMode = GPU::Regs::DisplayTransferConfig::ScalingMode;

    const SurfaceInfo source_info = {
        .format = PixelFormatFromGPUPixelFormat(config.input_format),
        .size = Extent{config.output_width, config.output_height},
        .is_tiled = !config.input_linear
    };

    const u32 dest_width = config.scaling != ScalingMode::NoScale ? config.output_width.Value() / 2
                                                                  : config.output_width.Value();
    const u32 dest_height = config.scaling == ScalingMode::ScaleXY ? config.output_height.Value() / 2
                                                                   : config.output_height.Value();
    const SurfaceInfo dest_info = {
        .format = PixelFormatFromGPUPixelFormat(config.output_format),
        .size = Extent{dest_width, dest_height},
        .is_tiled = config.input_linear != config.dont_swizzle
    };

    return std::make_pair(source_info, dest_info);
}

[[nodiscard]] constexpr auto CalculateMipLevelOffsets(const SurfaceInfo& info) noexcept
                                                    -> std::array<u32, MAX_PICA_LEVELS> {
    ASSERT(info.levels <= MAX_PICA_LEVELS);

    const u32 bytes_per_pixel = GetBytesPerPixel(info.format);
    u32 width = info.size.width;
    u32 height = info.size.height;

    std::array<u32, MAX_PICA_LEVELS> offsets{};
    u32 offset = 0;
    for (s32 level = 0; level < info.levels; level++) {
        offsets[level] = offset;
        offset += width * height * bytes_per_pixel;

        width >>= 1;
        height >>= 1;
    }

    return offsets;
}

[[nodiscard]] constexpr u32 CalculateSurfaceSize(const SurfaceInfo& info) noexcept {
    const u32 bytes_per_pixel = GetBytesPerPixel(info.format);
    u32 width = info.size.width;
    u32 height = info.size.height;

    u32 size = 0;
    for (s32 level = 0; level < info.levels; level++) {
        size += width * height * bytes_per_pixel;

        width >>= 1;
        height >>= 1;
    }

    return size;
}

// Helper function used to detect a compatible copy surface
[[nodiscard]] constexpr bool CanTexCopy(const SurfaceInfo& info, const Surface& surface) {
    const auto& candidate_info = surface.info;
    if (candidate_info.format == PixelFormat::Invalid) {
        return false;
    }

    const u32 copy_width = info.real_size.width;
    if (info.size.width != info.real_size.width) {
        const u32 stride = candidate_info.size.width;
        const u32 tile_dim = candidate_info.is_tiled ? 8 : 1;
        const u32 tile_stride = BytesInPixels(stride * tile_dim, candidate_info.format);

        const u32 offset = info.addr - candidate_info.addr;
        return IsBlockAligned(offset, surface) &&
               IsBlockAligned(copy_width, surface) &&
               (info.size.height == 1 || stride == tile_stride) &&
               (offset % tile_stride) + copy_width <= tile_stride;
    }

    return true;
};

} // namespace VideoCore
