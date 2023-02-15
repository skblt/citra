// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <boost/container/small_vector.hpp>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "core/memory.h"
#include "video_core/pica_state.h"
#include "video_core/rasterizer_cache/rasterizer_cache_base.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/video_core.h"

namespace VideoCore {

MICROPROFILE_DECLARE(RasterizerCache_BlitSurface);
MICROPROFILE_DECLARE(RasterizerCache_CopySurface);
MICROPROFILE_DECLARE(RasterizerCache_SurfaceLoad);
MICROPROFILE_DECLARE(RasterizerCache_SurfaceFlush);

inline auto RangeFromInterval(auto& map, const auto& interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

template <class T>
RasterizerCache<T>::RasterizerCache(Memory::MemorySystem& memory_, Runtime& runtime_)
    : memory{memory_}, runtime{runtime_}, resolution_scale_factor{
                                              VideoCore::GetResolutionScaleFactor()} {
    using TextureConfig = Pica::TexturingRegs::TextureConfig;

    // Create null handles for all cached resources
    void(slot_samplers.insert(runtime, SamplerParams{
                                           .mag_filter = TextureConfig::TextureFilter::Linear,
                                           .min_filter = TextureConfig::TextureFilter::Linear,
                                           .mip_filter = TextureConfig::TextureFilter::Linear,
                                           .wrap_s = TextureConfig::WrapMode::ClampToBorder,
                                           .wrap_t = TextureConfig::WrapMode::ClampToBorder,
                                       }));
}

template <class T>
RasterizerCache<T>::~RasterizerCache() {
#ifndef ANDROID
    // This is for switching renderers, which is unsupported on Android, and costly on shutdown
    ClearAll(false);
#endif
}

template <class T>
bool RasterizerCache<T>::AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) {
    u32 copy_size = Common::AlignDown(config.texture_copy.size, 16);
    if (copy_size == 0) {
        return false;
    }

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 input_width = config.texture_copy.input_width * 16;
    if (input_width == 0 && input_gap != 0) {
        return false;
    }
    if (input_gap == 0 || input_width >= copy_size) {
        input_width = copy_size;
        input_gap = 0;
    }
    if (copy_size % input_width != 0) {
        return false;
    }

    u32 output_gap = config.texture_copy.output_gap * 16;
    u32 output_width = config.texture_copy.output_width * 16;
    if (output_width == 0 && output_gap != 0) {
        return false;
    }
    if (output_gap == 0 || output_width >= copy_size) {
        output_width = copy_size;
        output_gap = 0;
    }
    if (copy_size % output_width != 0) {
        return false;
    }

    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.stride = input_width + input_gap; // stride in bytes
    src_params.width = input_width;              // width in bytes
    src_params.height = copy_size / input_width;
    src_params.size = ((src_params.height - 1) * src_params.stride) + src_params.width;
    src_params.end = src_params.addr + src_params.size;

    auto [src_surface, src_rect] = GetTexCopySurface(src_params);
    if (src_surface == nullptr) {
        return false;
    }

    if (output_gap != 0 &&
        (output_width != src_surface->BytesInPixels(src_rect.GetWidth() / src_surface->res_scale) *
                             (src_surface->is_tiled ? 8 : 1) ||
         output_gap % src_surface->BytesInPixels(src_surface->is_tiled ? 64 : 1) != 0)) {
        return false;
    }

    SurfaceParams dst_params = *src_surface;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = src_rect.GetWidth() / src_surface->res_scale;
    dst_params.stride = dst_params.width + src_surface->PixelsInBytes(
                                               src_surface->is_tiled ? output_gap / 8 : output_gap);
    dst_params.height = src_rect.GetHeight() / src_surface->res_scale;
    dst_params.res_scale = src_surface->res_scale;
    dst_params.UpdateParams();

    // Since we are going to invalidate the gap if there is one, we will have to load it first
    const bool load_gap = output_gap != 0;
    auto [dst_surface, dst_rect] = GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, load_gap);

    if (!dst_surface || dst_surface->type == SurfaceType::Texture ||
        !CheckFormatsBlittable(src_surface->pixel_format, dst_surface->pixel_format)) {
        return false;
    }

    ASSERT(src_rect.GetWidth() == dst_rect.GetWidth());

    const TextureCopy texture_copy = {
        .src_level = src_surface->LevelOf(src_params.addr),
        .dst_level = dst_surface->LevelOf(dst_params.addr),
        .src_offset = {src_rect.left, src_rect.bottom},
        .dst_offset = {dst_rect.left, dst_rect.bottom},
        .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
    };
    runtime.CopyTextures(*src_surface, *dst_surface, texture_copy);

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface);
    return true;
}

template <class T>
bool RasterizerCache<T>::AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) {
    SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.width = config.output_width;
    src_params.stride = config.input_width;
    src_params.height = config.output_height;
    src_params.is_tiled = !config.input_linear;
    src_params.pixel_format = PixelFormatFromGPUPixelFormat(config.input_format);
    src_params.UpdateParams();

    SurfaceParams dst_params;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = config.scaling != config.NoScale ? config.output_width.Value() / 2
                                                        : config.output_width.Value();
    dst_params.height = config.scaling == config.ScaleXY ? config.output_height.Value() / 2
                                                         : config.output_height.Value();
    dst_params.is_tiled = config.input_linear != config.dont_swizzle;
    dst_params.pixel_format = PixelFormatFromGPUPixelFormat(config.output_format);
    dst_params.UpdateParams();

    auto [src_surface, src_rect] = GetSurfaceSubRect(src_params, ScaleMatch::Ignore, true);
    if (src_surface == nullptr)
        return false;

    dst_params.res_scale = src_surface->res_scale;

    auto [dst_surface, dst_rect] = GetSurfaceSubRect(dst_params, ScaleMatch::Upscale, false);
    if (dst_surface == nullptr) {
        return false;
    }

    if (src_surface->is_tiled != dst_surface->is_tiled) {
        std::swap(src_rect.top, src_rect.bottom);
    }
    if (config.flip_vertically) {
        std::swap(src_rect.top, src_rect.bottom);
    }

    if (!CheckFormatsBlittable(src_surface->pixel_format, dst_surface->pixel_format)) {
        return false;
    }

    const TextureBlit texture_blit = {
        .src_level = src_surface->LevelOf(src_params.addr),
        .dst_level = dst_surface->LevelOf(dst_params.addr),
        .src_rect = src_rect,
        .dst_rect = dst_rect,
    };
    runtime.BlitTextures(*src_surface, *dst_surface, texture_blit);

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface);
    return true;
}

template <class T>
bool RasterizerCache<T>::AccelerateFill(const GPU::Regs::MemoryFillConfig& config) {
    SurfaceParams params;
    params.addr = config.GetStartAddress();
    params.end = config.GetEndAddress();
    params.size = params.end - params.addr;
    params.type = SurfaceType::Fill;
    params.res_scale = std::numeric_limits<u16>::max();

    Surface fill_surface = std::make_shared<typename T::Surface>(params, runtime);

    std::memcpy(&fill_surface->fill_data[0], &config.value_32bit, 4);
    if (config.fill_32bit) {
        fill_surface->fill_size = 4;
    } else if (config.fill_24bit) {
        fill_surface->fill_size = 3;
    } else {
        fill_surface->fill_size = 2;
    }

    RegisterSurface(fill_surface);
    InvalidateRegion(fill_surface->addr, fill_surface->size, fill_surface);
    return true;
}

template <class T>
auto RasterizerCache<T>::GetSampler(SamplerId sampler_id) -> Sampler& {
    return slot_samplers[sampler_id];
}

template <class T>
auto RasterizerCache<T>::GetSampler(const Pica::TexturingRegs::TextureConfig& config) -> Sampler& {
    const SamplerParams params = {
        .mag_filter = config.mag_filter,
        .min_filter = config.min_filter,
        .mip_filter = config.mip_filter,
        .wrap_s = config.wrap_s,
        .wrap_t = config.wrap_t,
        .border_color = config.border_color.raw,
        .lod_min = config.lod.min_level,
        .lod_max = config.lod.max_level,
        .lod_bias = config.lod.bias,
    };

    auto [it, is_new] = samplers.try_emplace(params);
    if (is_new) {
        it->second = slot_samplers.insert(runtime, params);
    }

    return slot_samplers[it->second];
}

template <class T>
template <typename Func>
void RasterizerCache<T>::ForEachSurfaceInRegion(PAddr addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, Surface>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<Surface, 32> surfaces;
    ForEachPage(addr, size, [this, &surfaces, addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const Surface& surface : it->second) {
            if (surface->picked) {
                continue;
            }
            if (!surface->Overlaps(addr, size)) {
                continue;
            }

            surface->picked = true;
            surfaces.push_back(surface);
            if constexpr (BOOL_BREAK) {
                if (func(surface)) {
                    return true;
                }
            } else {
                func(surface);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const Surface surface : surfaces) {
        surface->picked = false;
    }
}

template <class T>
template <MatchFlags find_flags>
auto RasterizerCache<T>::FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                                   std::optional<SurfaceInterval> validate_interval) -> Surface {
    Surface match_surface = nullptr;
    bool match_valid = false;
    u32 match_scale = 0;
    SurfaceInterval match_interval{};

    ForEachSurfaceInRegion(params.addr, params.size, [&](Surface surface) {
        const bool res_scale_matched = match_scale_type == ScaleMatch::Exact
                                           ? (params.res_scale == surface->res_scale)
                                           : (params.res_scale <= surface->res_scale);
        const bool is_valid =
            True(find_flags & MatchFlags::Copy)
                ? true
                : surface->IsRegionValid(validate_interval.value_or(params.GetInterval()));

        const auto IsMatch_Helper = [&](auto check_type, auto match_fn) {
            if (False(find_flags & check_type))
                return;

            bool matched;
            SurfaceInterval surface_interval;
            std::tie(matched, surface_interval) = match_fn();
            if (!matched)
                return;

            if (!res_scale_matched && match_scale_type != ScaleMatch::Ignore &&
                surface->type != SurfaceType::Fill)
                return;

            // Found a match, update only if this is better than the previous one
            const auto UpdateMatch = [&] {
                match_surface = surface;
                match_valid = is_valid;
                match_scale = surface->res_scale;
                match_interval = surface_interval;
            };

            if (surface->res_scale > match_scale) {
                UpdateMatch();
                return;
            } else if (surface->res_scale < match_scale) {
                return;
            }

            if (is_valid && !match_valid) {
                UpdateMatch();
                return;
            } else if (is_valid != match_valid) {
                return;
            }

            if (boost::icl::length(surface_interval) > boost::icl::length(match_interval)) {
                UpdateMatch();
            }
        };

        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Exact>{}, [&] {
            return std::make_pair(surface->ExactMatch(params), surface->GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::SubRect>{}, [&] {
            return std::make_pair(surface->CanSubRect(params), surface->GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Copy>{}, [&] {
            ASSERT(validate_interval);
            auto copy_interval =
                surface->GetCopyableInterval(params.FromInterval(*validate_interval));
            bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                           surface->CanCopy(params, copy_interval);
            return std::make_pair(matched, copy_interval);
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::Expand>{}, [&] {
            return std::make_pair(surface->CanExpand(params), surface->GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchFlags, MatchFlags::TexCopy>{}, [&] {
            return std::make_pair(surface->CanTexCopy(params), surface->GetInterval());
        });
    });

    return match_surface;
}

template <class T>
void RasterizerCache<T>::CopySurface(const Surface& src_surface, const Surface& dst_surface,
                                     SurfaceInterval copy_interval) {
    MICROPROFILE_SCOPE(RasterizerCache_CopySurface);

    const auto subrect_params = dst_surface->FromInterval(copy_interval);
    const Rect2D dst_rect = dst_surface->GetScaledSubRect(subrect_params);
    const PAddr copy_addr = copy_interval.lower();
    ASSERT(subrect_params.GetInterval() == copy_interval && src_surface != dst_surface);

    if (src_surface->type == SurfaceType::Fill) {
        const TextureClear texture_clear = {
            .texture_level = dst_surface->LevelOf(copy_addr),
            .texture_rect = dst_rect,
            .value = src_surface->MakeClearValue(copy_addr, dst_surface->pixel_format),
        };
        runtime.ClearTexture(*dst_surface, texture_clear);
        return;
    }

    const TextureBlit texture_blit = {
        .src_level = src_surface->LevelOf(copy_addr),
        .dst_level = dst_surface->LevelOf(copy_addr),
        .src_rect = src_surface->GetScaledSubRect(subrect_params),
        .dst_rect = dst_rect,
    };
    runtime.BlitTextures(*src_surface, *dst_surface, texture_blit);
}

template <class T>
auto RasterizerCache<T>::GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                                    bool load_if_create) -> Surface {
    if (params.addr == 0 || params.height * params.width == 0) [[unlikely]] {
        return nullptr;
    }

    // Use GetSurfaceSubRect instead
    ASSERT(params.width == params.stride);
    ASSERT(!params.is_tiled || (params.width % 8 == 0 && params.height % 8 == 0));

    // Check for an exact match in existing surfaces
    Surface surface = FindMatch<MatchFlags::Exact>(params, match_res_scale);

    if (!surface) {
        u16 target_res_scale = params.res_scale;
        if (match_res_scale != ScaleMatch::Exact) {
            // This surface may have a subrect of another surface with a higher res_scale, find
            // it to adjust our params
            SurfaceParams find_params = params;
            Surface expandable = FindMatch<MatchFlags::Expand>(find_params, match_res_scale);
            if (expandable && expandable->res_scale > target_res_scale) {
                target_res_scale = expandable->res_scale;
            }

            // Keep res_scale when reinterpreting d24s8 -> rgba8
            if (params.pixel_format == PixelFormat::RGBA8) {
                find_params.pixel_format = PixelFormat::D24S8;
                expandable = FindMatch<MatchFlags::Expand>(find_params, match_res_scale);
                if (expandable && expandable->res_scale > target_res_scale) {
                    target_res_scale = expandable->res_scale;
                }
            }
        }

        SurfaceParams new_params = params;
        new_params.res_scale = target_res_scale;
        surface = CreateSurface(new_params);
        RegisterSurface(surface);
    }

    if (load_if_create) {
        ValidateSurface(surface, params.addr, params.size);
    }

    return surface;
}

template <class T>
auto RasterizerCache<T>::GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                           bool load_if_create) -> SurfaceRect_Tuple {
    if (params.addr == 0 || params.height * params.width == 0) [[unlikely]] {
        return std::make_tuple(nullptr, Common::Rectangle<u32>{});
    }

    // Attempt to find encompassing surface
    Surface surface = FindMatch<MatchFlags::SubRect>(params, match_res_scale);

    // Check if FindMatch failed because of res scaling
    // If that's the case create a new surface with
    // the dimensions of the lower res_scale surface
    // to suggest it should not be used again
    if (!surface && match_res_scale != ScaleMatch::Ignore) {
        surface = FindMatch<MatchFlags::SubRect>(params, ScaleMatch::Ignore);
        if (surface) {
            SurfaceParams new_params = *surface;
            new_params.res_scale = params.res_scale;

            surface = CreateSurface(new_params);
            RegisterSurface(surface);
        }
    }

    SurfaceParams aligned_params = params;
    if (params.is_tiled) {
        aligned_params.height = Common::AlignUp(params.height, 8);
        aligned_params.width = Common::AlignUp(params.width, 8);
        aligned_params.stride = Common::AlignUp(params.stride, 8);
        aligned_params.UpdateParams();
    }

    // Check for a surface we can expand before creating a new one
    if (!surface) {
        surface = FindMatch<MatchFlags::Expand>(aligned_params, match_res_scale);
        if (surface) {
            aligned_params.width = aligned_params.stride;
            aligned_params.UpdateParams();

            SurfaceParams new_params = *surface;
            new_params.addr = std::min(aligned_params.addr, surface->addr);
            new_params.end = std::max(aligned_params.end, surface->end);
            new_params.size = new_params.end - new_params.addr;
            new_params.height =
                new_params.size / aligned_params.BytesInPixels(aligned_params.stride);
            new_params.UpdateParams();
            ASSERT(new_params.size % aligned_params.BytesInPixels(aligned_params.stride) == 0);

            Surface new_surface = CreateSurface(new_params);
            DuplicateSurface(surface, new_surface);
            UnregisterSurface(surface);

            surface = new_surface;
            RegisterSurface(new_surface);
        }
    }

    // No subrect found - create and return a new surface
    if (!surface) {
        SurfaceParams new_params = aligned_params;
        // Can't have gaps in a surface
        new_params.width = aligned_params.stride;
        new_params.UpdateParams();
        // GetSurface will create the new surface and possibly adjust res_scale if necessary
        surface = GetSurface(new_params, match_res_scale, load_if_create);
    } else if (load_if_create) {
        ValidateSurface(surface, aligned_params.addr, aligned_params.size);
    }

    return std::make_tuple(surface, surface->GetScaledSubRect(params));
}

template <class T>
auto RasterizerCache<T>::GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config)
    -> Surface {
    const auto info = Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
    const u32 max_level = MipLevels(info.width, info.height, config.config.lod.max_level) - 1;
    return GetTextureSurface(info, max_level);
}

template <class T>
auto RasterizerCache<T>::GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level)
    -> Surface {
    if (info.physical_address == 0) [[unlikely]] {
        return nullptr;
    }

    SurfaceParams params;
    params.addr = info.physical_address;
    params.width = info.width;
    params.height = info.height;
    params.levels = max_level + 1;
    params.is_tiled = true;
    params.pixel_format = PixelFormatFromTextureFormat(info.format);
    params.UpdateParams();

    const u32 min_width = info.width >> max_level;
    const u32 min_height = info.height >> max_level;
    if (min_width % 8 != 0 || min_height % 8 != 0) {
        LOG_CRITICAL(HW_GPU, "Texture size ({}x{}) is not multiple of 8", min_width, min_height);
        return nullptr;
    }

    if (info.width != (min_width << max_level) || info.height != (min_height << max_level)) {
        LOG_CRITICAL(HW_GPU, "Texture size ({}x{}) does not support required mipmap level ({})",
                     params.width, params.height, max_level);
        return nullptr;
    }

    return GetSurface(params, ScaleMatch::Ignore, true);
}

template <class T>
auto RasterizerCache<T>::GetTextureCube(const TextureCubeConfig& config) -> const Surface& {
    auto [it, new_surface] = texture_cube_cache.try_emplace(config);
    if (new_surface) {
        SurfaceParams cube_params = {
            .addr = config.px,
            .width = config.width,
            .height = config.width,
            .stride = config.width,
            .levels = config.levels,
            .texture_type = TextureType::CubeMap,
            .pixel_format = PixelFormatFromTextureFormat(config.format),
            .type = SurfaceType::Texture,
        };

        it->second = CreateSurface(cube_params);
    }

    Surface& cube = it->second;

    const u32 scaled_size = cube->GetScaledWidth();
    const std::array addresses = {config.px, config.nx, config.py, config.ny, config.pz, config.nz};

    for (std::size_t i = 0; i < addresses.size(); i++) {
        Pica::Texture::TextureInfo info = {
            .physical_address = addresses[i],
            .width = config.width,
            .height = config.width,
            .format = config.format,
        };
        info.SetDefaultStride();

        Surface face_surface = GetTextureSurface(info, config.levels - 1);
        if (!face_surface) {
            continue;
        }

        ASSERT(face_surface->levels == config.levels);
        const u32 face = static_cast<u32>(i);
        for (u32 level = 0; level < face_surface->levels; level++) {
            const TextureCopy texture_copy = {
                .src_level = level,
                .dst_level = level,
                .src_layer = 0,
                .dst_layer = face,
                .src_offset = {0, 0},
                .dst_offset = {0, 0},
                .extent = {scaled_size >> level, scaled_size >> level},
            };
            runtime.CopyTextures(*face_surface, *cube, texture_copy);
        }
    }

    return cube;
}

template <class T>
auto RasterizerCache<T>::GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb)
    -> Framebuffer {
    const auto& regs = Pica::g_state.regs;
    const auto& config = regs.framebuffer.framebuffer;

    // Update resolution_scale_factor and reset cache if changed
    const bool resolution_scale_changed =
        resolution_scale_factor != VideoCore::GetResolutionScaleFactor();
    if (resolution_scale_changed) [[unlikely]] {
        resolution_scale_factor = VideoCore::GetResolutionScaleFactor();
        UnregisterAll();
    }

    const s32 framebuffer_width = config.GetWidth();
    const s32 framebuffer_height = config.GetHeight();
    const auto viewport_rect = regs.rasterizer.GetViewportRect();
    const Common::Rectangle<u32> viewport_clamped = {
        static_cast<u32>(std::clamp(viewport_rect.left, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.top, 0, framebuffer_height)),
        static_cast<u32>(std::clamp(viewport_rect.right, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.bottom, 0, framebuffer_height)),
    };

    // get color and depth surfaces
    SurfaceParams color_params;
    color_params.is_tiled = true;
    color_params.res_scale = resolution_scale_factor;
    color_params.width = config.GetWidth();
    color_params.height = config.GetHeight();
    SurfaceParams depth_params = color_params;

    color_params.addr = config.GetColorBufferPhysicalAddress();
    color_params.pixel_format = PixelFormatFromColorFormat(config.color_format);
    color_params.UpdateParams();

    depth_params.addr = config.GetDepthBufferPhysicalAddress();
    depth_params.pixel_format = PixelFormatFromDepthFormat(config.depth_format);
    depth_params.UpdateParams();

    auto color_vp_interval = color_params.GetSubRectInterval(viewport_clamped);
    auto depth_vp_interval = depth_params.GetSubRectInterval(viewport_clamped);

    // Make sure that framebuffers don't overlap if both color and depth are being used
    if (using_color_fb && using_depth_fb &&
        boost::icl::length(color_vp_interval & depth_vp_interval)) {
        LOG_CRITICAL(HW_GPU, "Color and depth framebuffer memory regions overlap; "
                             "overlapping framebuffers not supported!");
        using_depth_fb = false;
    }

    Common::Rectangle<u32> color_rect{};
    Surface color_surface{};
    if (using_color_fb)
        std::tie(color_surface, color_rect) =
            GetSurfaceSubRect(color_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> depth_rect{};
    Surface depth_surface{};
    if (using_depth_fb)
        std::tie(depth_surface, depth_rect) =
            GetSurfaceSubRect(depth_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> fb_rect{};
    if (color_surface && depth_surface) {
        fb_rect = color_rect;
        // Color and Depth surfaces must have the same dimensions and offsets
        if (color_rect.bottom != depth_rect.bottom || color_rect.top != depth_rect.top ||
            color_rect.left != depth_rect.left || color_rect.right != depth_rect.right) {
            color_surface = GetSurface(color_params, ScaleMatch::Exact, false);
            depth_surface = GetSurface(depth_params, ScaleMatch::Exact, false);
            fb_rect = color_surface->GetScaledRect();
        }
    } else if (color_surface) {
        fb_rect = color_rect;
    } else if (depth_surface) {
        fb_rect = depth_rect;
    }

    if (color_surface) {
        ASSERT_MSG(color_surface->LevelOf(color_params.addr) == 0,
                   "Rendering to mipmap of color surface unsupported");
        ValidateSurface(color_surface, boost::icl::first(color_vp_interval),
                        boost::icl::length(color_vp_interval));
    }
    if (depth_surface) {
        ASSERT_MSG(depth_surface->LevelOf(depth_params.addr) == 0,
                   "Rendering to mipmap of depth surface unsupported");
        ValidateSurface(depth_surface, boost::icl::first(depth_vp_interval),
                        boost::icl::length(depth_vp_interval));
    }

    render_targets = RenderTargets{
        .color_surface = color_surface,
        .depth_surface = depth_surface,
    };

    auto* const color = color_surface ? color_surface.get() : nullptr;
    auto* const depth_stencil = depth_surface ? depth_surface.get() : nullptr;
    return Framebuffer{runtime, color, depth_stencil, regs, fb_rect};
}

template <class T>
void RasterizerCache<T>::InvalidateRenderTargets(const Framebuffer& framebuffer) {
    const auto Invalidate = [&](SurfaceType type, Surface region_owner) {
        const bool has_attachment = framebuffer.HasAttachment(type);
        if (has_attachment) {
            const SurfaceInterval interval = framebuffer.Interval(type);
            InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval),
                             region_owner);
        }
    };

    Invalidate(SurfaceType::Color, render_targets.color_surface);
    Invalidate(SurfaceType::DepthStencil, render_targets.depth_surface);
}

template <class T>
auto RasterizerCache<T>::GetTexCopySurface(const SurfaceParams& params) -> SurfaceRect_Tuple {
    Common::Rectangle<u32> rect{};

    Surface match_surface = FindMatch<MatchFlags::TexCopy>(params, ScaleMatch::Ignore);

    if (match_surface) {
        ValidateSurface(match_surface, params.addr, params.size);

        SurfaceParams match_subrect;
        if (params.width != params.stride) {
            const u32 tiled_size = match_surface->is_tiled ? 8 : 1;
            match_subrect = params;
            match_subrect.width = match_surface->PixelsInBytes(params.width) / tiled_size;
            match_subrect.stride = match_surface->PixelsInBytes(params.stride) / tiled_size;
            match_subrect.height *= tiled_size;
        } else {
            match_subrect = match_surface->FromInterval(params.GetInterval());
            ASSERT(match_subrect.GetInterval() == params.GetInterval());
        }

        rect = match_surface->GetScaledSubRect(match_subrect);
    }

    return std::make_tuple(match_surface, rect);
}

template <class T>
void RasterizerCache<T>::DuplicateSurface(const Surface& src_surface, const Surface& dst_surface) {
    ASSERT(dst_surface->addr <= src_surface->addr && dst_surface->end >= src_surface->end);

    const Rect2D src_rect = src_surface->GetScaledRect();
    const Rect2D dst_rect = dst_surface->GetScaledSubRect(*src_surface);
    ASSERT(src_rect.GetWidth() == dst_rect.GetWidth());

    const TextureCopy texture_copy = {
        .src_level = 0,
        .dst_level = 0,
        .src_offset = {src_rect.left, src_rect.bottom},
        .dst_offset = {dst_rect.left, dst_rect.bottom},
        .extent = {src_rect.GetWidth(), src_rect.GetHeight()},
    };
    runtime.CopyTextures(*src_surface, *dst_surface, texture_copy);

    dst_surface->invalid_regions -= src_surface->GetInterval();
    dst_surface->invalid_regions += src_surface->invalid_regions;

    SurfaceRegions regions;
    for (const auto& pair : RangeFromInterval(dirty_regions, src_surface->GetInterval())) {
        if (pair.second == src_surface) {
            regions += pair.first;
        }
    }

    for (const auto& interval : regions) {
        dirty_regions.set({interval, dst_surface});
    }
}

template <class T>
void RasterizerCache<T>::ValidateSurface(const Surface& surface, PAddr addr, u32 size) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval validate_interval(addr, addr + size);
    const SurfaceRegions validate_regions = surface->invalid_regions & validate_interval;
    if (validate_regions.empty()) {
        return;
    }

    // Fill surfaces must always be valid when used
    if (surface->type == SurfaceType::Fill) {
        ASSERT(surface->IsRegionValid(validate_interval));
        return;
    }

    for (u32 level = surface->LevelOf(addr); level <= surface->LevelOf(addr + size); level++) {
        auto level_regions = validate_regions & surface->LevelInterval(level);
        while (!level_regions.empty()) {
            const SurfaceInterval interval = *level_regions.begin();
            const SurfaceParams params = surface->FromInterval(interval);

            Surface copy_surface =
                FindMatch<MatchFlags::Copy>(params, ScaleMatch::Ignore, interval);
            if (copy_surface) {
                const SurfaceInterval copy_interval = copy_surface->GetCopyableInterval(params);
                CopySurface(copy_surface, surface, copy_interval);
                level_regions.erase(copy_interval);
                continue;
            }

            // Try to find surface in cache with different format
            // that can can be reinterpreted to the requested format.
            if (ValidateByReinterpretation(surface, params, interval)) {
                level_regions.erase(interval);
                continue;
            }
            // Could not find a matching reinterpreter, check if we need to implement a
            // reinterpreter
            if (NoUnimplementedReinterpretations(surface, params, interval) &&
                !IntervalHasInvalidPixelFormat(params, interval)) {
                // No surfaces were found in the cache that had a matching bit-width.
                // If the region was created entirely on the GPU,
                // assume it was a developer mistake and skip flushing.
                if (boost::icl::contains(dirty_regions, interval)) {
                    LOG_DEBUG(HW_GPU, "Region created fully on GPU and reinterpretation is "
                                      "invalid. Skipping validation");
                    level_regions.erase(interval);
                    continue;
                }
            }

            // Load data from 3DS memory
            FlushRegion(params.addr, params.size);
            UploadSurface(surface, interval);
            level_regions.erase(params.GetInterval());
        }
    }

    surface->invalid_regions.erase(validate_interval);
}

template <class T>
void RasterizerCache<T>::UploadSurface(const Surface& surface, SurfaceInterval interval) {
    const SurfaceParams load_info = surface->FromInterval(interval);
    ASSERT(load_info.addr >= surface->addr && load_info.end <= surface->end);

    MICROPROFILE_SCOPE(RasterizerCache_SurfaceLoad);

    const auto staging = runtime.FindStaging(
        load_info.width * load_info.height * surface->GetInternalBytesPerPixel(), true);

    MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
    if (!source_ptr) [[unlikely]] {
        return;
    }

    const auto upload_data = source_ptr.GetWriteBytes(load_info.end - load_info.addr);
    DecodeTexture(load_info, load_info.addr, load_info.end, upload_data, staging.mapped,
                  runtime.NeedsConvertion(surface->pixel_format));

    const BufferTextureCopy upload = {
        .buffer_offset = 0,
        .buffer_size = staging.size,
        .texture_rect = surface->GetSubRect(load_info),
        .texture_level = surface->LevelOf(load_info.addr),
    };
    surface->Upload(upload, staging);
}

template <class T>
void RasterizerCache<T>::DownloadSurface(const Surface& surface, SurfaceInterval interval) {
    const SurfaceParams flush_info = surface->FromInterval(interval);
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface->addr && flush_end <= surface->end);

    const auto staging = runtime.FindStaging(
        flush_info.width * flush_info.height * surface->GetInternalBytesPerPixel(), false);

    const BufferTextureCopy download = {
        .buffer_offset = 0,
        .buffer_size = staging.size,
        .texture_rect = surface->GetSubRect(flush_info),
        .texture_level = surface->LevelOf(flush_start),
    };
    surface->Download(download, staging);

    runtime.Finish();

    MemoryRef dest_ptr = memory.GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const auto download_dest = dest_ptr.GetWriteBytes(flush_end - flush_start);
    EncodeTexture(flush_info, flush_start, flush_end, staging.mapped, download_dest,
                  runtime.NeedsConvertion(surface->pixel_format));
}

template <class T>
void RasterizerCache<T>::DownloadFillSurface(const Surface& surface, SurfaceInterval interval) {
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface->addr && flush_end <= surface->end);

    MemoryRef dest_ptr = memory.GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const u32 start_offset = flush_start - surface->addr;
    const u32 download_size =
        std::clamp(flush_end - flush_start, 0u, static_cast<u32>(dest_ptr.GetSize()));
    const u32 coarse_start_offset = start_offset - (start_offset % surface->fill_size);
    const u32 backup_bytes = start_offset % surface->fill_size;

    std::array<u8, 4> backup_data;
    if (backup_bytes) {
        std::memcpy(backup_data.data(), &dest_ptr[coarse_start_offset], backup_bytes);
    }

    for (u32 offset = coarse_start_offset; offset < download_size; offset += surface->fill_size) {
        std::memcpy(&dest_ptr[offset], &surface->fill_data[0],
                    std::min(surface->fill_size, download_size - offset));
    }

    if (backup_bytes) {
        std::memcpy(&dest_ptr[coarse_start_offset], &backup_data[0], backup_bytes);
    }
}

template <class T>
bool RasterizerCache<T>::NoUnimplementedReinterpretations(const Surface& surface,
                                                          SurfaceParams params,
                                                          SurfaceInterval interval) {
    static constexpr std::array all_formats = {
        PixelFormat::RGBA8, PixelFormat::RGB8,   PixelFormat::RGB5A1, PixelFormat::RGB565,
        PixelFormat::RGBA4, PixelFormat::IA8,    PixelFormat::RG8,    PixelFormat::I8,
        PixelFormat::A8,    PixelFormat::IA4,    PixelFormat::I4,     PixelFormat::A4,
        PixelFormat::ETC1,  PixelFormat::ETC1A4, PixelFormat::D16,    PixelFormat::D24,
        PixelFormat::D24S8,
    };

    bool implemented = true;
    for (PixelFormat format : all_formats) {
        if (GetFormatBpp(format) == surface->GetFormatBpp()) {
            params.pixel_format = format;
            // This could potentially be expensive, although experimentally it hasn't been too bad
            Surface test_surface =
                FindMatch<MatchFlags::Copy>(params, ScaleMatch::Ignore, interval);

            if (test_surface) {
                LOG_WARNING(HW_GPU, "Missing pixel_format reinterpreter: {} -> {}",
                            PixelFormatAsString(format),
                            PixelFormatAsString(surface->pixel_format));
                implemented = false;
            }
        }
    }

    return implemented;
}

template <class T>
bool RasterizerCache<T>::IntervalHasInvalidPixelFormat(SurfaceParams params,
                                                       SurfaceInterval interval) {
    bool invalid_format_found = false;
    ForEachSurfaceInRegion(params.addr, params.end, [&](Surface surface) {
        if (surface->pixel_format == PixelFormat::Invalid) {
            LOG_DEBUG(HW_GPU, "Surface {:#x} found with invalid pixel format", surface->addr);
            invalid_format_found = true;
            return true;
        }
        return false;
    });

    return invalid_format_found;
}

template <class T>
bool RasterizerCache<T>::ValidateByReinterpretation(const Surface& surface, SurfaceParams params,
                                                    SurfaceInterval interval) {
    const PixelFormat dest_format = surface->pixel_format;
    for (const auto& reinterpreter : runtime.GetPossibleReinterpretations(dest_format)) {
        params.pixel_format = reinterpreter->GetSourceFormat();
        Surface reinterpret_surface =
            FindMatch<MatchFlags::Copy>(params, ScaleMatch::Ignore, interval);

        if (reinterpret_surface) {
            auto reinterpret_interval = reinterpret_surface->GetCopyableInterval(params);
            auto reinterpret_params = surface->FromInterval(reinterpret_interval);
            auto src_rect = reinterpret_surface->GetScaledSubRect(reinterpret_params);
            auto dest_rect = surface->GetScaledSubRect(reinterpret_params);

            reinterpreter->Reinterpret(*reinterpret_surface, src_rect, *surface, dest_rect);
            return true;
        }
    }

    return false;
}

template <class T>
void RasterizerCache<T>::ClearAll(bool flush) {
    const auto flush_interval = PageMap::interval_type::right_open(0x0, 0xFFFFFFFF);
    // Force flush all surfaces from the cache
    if (flush) {
        FlushRegion(0x0, 0xFFFFFFFF);
    }
    // Unmark all of the marked pages
    for (auto& pair : RangeFromInterval(cached_pages, flush_interval)) {
        const auto interval = pair.first & flush_interval;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::CITRA_PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::CITRA_PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
    }

    // Remove the whole cache without really looking at it.
    cached_pages -= flush_interval;
    dirty_regions -= SurfaceInterval(0x0, 0xFFFFFFFF);
    page_table.clear();
    remove_surfaces.clear();
}

template <class T>
void RasterizerCache<T>::FlushRegion(PAddr addr, u32 size, Surface flush_surface) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval flush_interval(addr, addr + size);
    SurfaceRegions flushed_intervals;

    for (auto& pair : RangeFromInterval(dirty_regions, flush_interval)) {
        // Small sizes imply that this most likely comes from the cpu, flush the entire region
        // the point is to avoid thousands of small writes every frame if the cpu decides to
        // access that region, anything higher than 8 you're guaranteed it comes from a service
        const auto interval = size <= 8 ? pair.first : pair.first & flush_interval;
        auto& surface = pair.second;

        if (flush_surface && surface != flush_surface) {
            continue;
        }

        // Sanity check, this surface is the last one that marked this region dirty
        ASSERT(surface->IsRegionValid(interval));

        if (surface->type == SurfaceType::Fill) {
            DownloadFillSurface(surface, interval);
        } else {
            DownloadSurface(surface, interval);
        }

        flushed_intervals += interval;
    }

    // Reset dirty regions
    dirty_regions -= flushed_intervals;
}

template <class T>
void RasterizerCache<T>::FlushAll() {
    FlushRegion(0, 0xFFFFFFFF);
}

template <class T>
void RasterizerCache<T>::InvalidateRegion(PAddr addr, u32 size, const Surface& region_owner) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval invalid_interval{addr, addr + size};
    if (region_owner != nullptr) {
        ASSERT(region_owner->type != SurfaceType::Texture);
        ASSERT(addr >= region_owner->addr && addr + size <= region_owner->end);
        // Surfaces can't have a gap
        ASSERT(region_owner->width == region_owner->stride);
        region_owner->invalid_regions.erase(invalid_interval);
    }

    ForEachSurfaceInRegion(addr, size, [&](Surface surface) {
        if (surface == region_owner) {
            return;
        }

        // If the CPU is invalidating this region we want to remove it
        // to (likely) mark the memory pages as uncached
        if (!region_owner && size <= 8) {
            FlushRegion(surface->addr, surface->size, surface);
            remove_surfaces.push_back(surface);
            return;
        }

        const SurfaceInterval interval = surface->GetInterval() & invalid_interval;
        surface->invalid_regions.insert(interval);

        // If the surface has no salvageable data it should be removed from the cache to avoid
        // clogging the data structure
        if (surface->IsFullyInvalid()) {
            remove_surfaces.push_back(surface);
        }
    });

    if (region_owner) {
        dirty_regions.set({invalid_interval, region_owner});
    } else {
        dirty_regions.erase(invalid_interval);
    }

    for (Surface remove_surface : remove_surfaces) {
        UnregisterSurface(remove_surface);
    }
    remove_surfaces.clear();
}

template <class T>
auto RasterizerCache<T>::CreateSurface(SurfaceParams& params) -> Surface {
    Surface surface = std::make_shared<typename T::Surface>(params, runtime);
    surface->invalid_regions.insert(surface->GetInterval());
    return surface;
}

template <class T>
void RasterizerCache<T>::RegisterSurface(const Surface& surface) {
    ASSERT_MSG(!surface->registered, "Trying to register an already registered surface");

    surface->registered = true;
    UpdatePagesCachedCount(surface->addr, surface->size, 1);
    ForEachPage(surface->addr, surface->size,
                [this, surface](u64 page) { page_table[page].push_back(surface); });
}

template <class T>
void RasterizerCache<T>::UnregisterSurface(const Surface& surface) {
    ASSERT_MSG(surface->registered, "Trying to unregister an already unregistered surface");

    surface->registered = false;
    UpdatePagesCachedCount(surface->addr, surface->size, -1);

    ForEachPage(surface->addr, surface->size, [this, surface](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == page_table.end()) {
            ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRA_PAGEBITS);
            return;
        }
        std::vector<Surface>& surfaces = page_it->second;
        const auto vector_it = std::find(surfaces.begin(), surfaces.end(), surface);
        if (vector_it == surfaces.end()) {
            ASSERT_MSG(false, "Unregistering unregistered surface in page=0x{:x}",
                       page << CITRA_PAGEBITS);
            return;
        }
        surfaces.erase(vector_it);
    });
}

template <class T>
void RasterizerCache<T>::UnregisterAll() {
    FlushAll();
    for (auto& [page, surfaces] : page_table) {
        while (!surfaces.empty()) {
            UnregisterSurface(surfaces.back());
        }
    }
    texture_cube_cache.clear();
    remove_surfaces.clear();
}

template <class T>
void RasterizerCache<T>::UpdatePagesCachedCount(PAddr addr, u32 size, int delta) {
    const u32 num_pages =
        ((addr + size - 1) >> Memory::CITRA_PAGE_BITS) - (addr >> Memory::CITRA_PAGE_BITS) + 1;
    const u32 page_start = addr >> Memory::CITRA_PAGE_BITS;
    const u32 page_end = page_start + num_pages;

    // Interval maps will erase segments if count reaches 0, so if delta is negative we have to
    // subtract after iterating
    const auto pages_interval = PageMap::interval_type::right_open(page_start, page_end);
    if (delta > 0) {
        cached_pages.add({pages_interval, delta});
    }

    for (const auto& pair : RangeFromInterval(cached_pages, pages_interval)) {
        const auto interval = pair.first & pages_interval;
        const int count = pair.second;

        const PAddr interval_start_addr = boost::icl::first(interval) << Memory::CITRA_PAGE_BITS;
        const PAddr interval_end_addr = boost::icl::last_next(interval) << Memory::CITRA_PAGE_BITS;
        const u32 interval_size = interval_end_addr - interval_start_addr;

        if (delta > 0 && count == delta) {
            memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, true);
        } else if (delta < 0 && count == -delta) {
            memory.RasterizerMarkRegionCached(interval_start_addr, interval_size, false);
        } else {
            ASSERT(count >= 0);
        }
    }

    if (delta < 0) {
        cached_pages.add({pages_interval, delta});
    }
}

} // namespace VideoCore
