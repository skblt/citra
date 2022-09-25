// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <algorithm>
#include <unordered_map>
#include <optional>
#include <vector>
#include <boost/range/iterator_range.hpp>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "common/microprofile.h"
#include "video_core/pica_state.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/video_core.h"

namespace VideoCore {

inline auto RangeFromInterval(auto& map, SurfaceInterval interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

enum class ScaleMatch {
    Exact,   ///< Only accept same res scale
    Upscale, ///< Only allow higher scale than params
    Ignore   ///< Accept every scaled res
};

enum class MatchFlags {
    Invalid = 1,      ///< Surface is allowed to be only partially valid
    Exact = 1 << 1,   ///< Surface perfectly matches params
    SubRect = 1 << 2, ///< Surface encompasses params
    Copy = 1 << 3,    ///< Surface that can be used as a copy source
    Expand = 1 << 4,  ///< Surface that can expand params
    TexCopy = 1 << 5  ///< Surface that will match a display transfer "texture copy" parameters
};

DECLARE_ENUM_FLAG_OPERATORS(MatchFlags);

class RasterizerAccelerated;

template <class T>
class RasterizerCache : NonCopyable {
public:
    using TextureRuntime = typename T::RuntimeType;
    using Surface = std::shared_ptr<typename T::SurfaceType>;
    using Watcher = SurfaceWatcher<typename T::SurfaceType>;

private:
    /// Declare rasterizer interval types
    using SurfaceSet = std::set<Surface>;
    using SurfaceMap =
        boost::icl::interval_map<PAddr, Surface, boost::icl::partial_absorber, std::less,
                                 boost::icl::inplace_plus, boost::icl::inter_section, SurfaceInterval>;
    using SurfaceCache =
        boost::icl::interval_map<PAddr, SurfaceSet, boost::icl::partial_absorber, std::less,
                                 boost::icl::inplace_plus, boost::icl::inter_section, SurfaceInterval>;

    static_assert(std::is_same<SurfaceRegions::interval_type, typename SurfaceCache::interval_type>() &&
                  std::is_same<typename SurfaceMap::interval_type, typename SurfaceCache::interval_type>(),
                  "Incorrect interval types");

    using SurfaceRect_Tuple = std::tuple<Surface, Common::Rectangle<u32>>;
    using SurfaceSurfaceRect_Tuple = std::tuple<Surface, Surface, Common::Rectangle<u32>>;

public:
    RasterizerCache(VideoCore::RasterizerAccelerated& rasterizer, TextureRuntime& runtime);
    ~RasterizerCache() = default;

    /// Get the best surface match (and its match type) for the given flags
    template <MatchFlags find_flags>
    Surface FindMatch(const SurfaceCache& surface_cache, const SurfaceParams& params,
                      ScaleMatch match_scale_type,
                      std::optional<SurfaceInterval> validate_interval = std::nullopt);

    /// Blit one surface's texture to another
    bool BlitSurfaces(const Surface& src_surface, Common::Rectangle<u32> src_rect,
                      const Surface& dst_surface, Common::Rectangle<u32> dst_rect);

    /// Copy one surface's region to another
    void CopySurface(const Surface& src_surface, const Surface& dst_surface,
                     SurfaceInterval copy_interval);

    /// Load a texture from 3DS memory to OpenGL and cache it (if not already cached)
    Surface GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                       bool load_if_create);

    /// Attempt to find a subrect (resolution scaled) of a surface, otherwise loads a texture from
    /// 3DS memory to OpenGL and caches it (if not already cached)
    SurfaceRect_Tuple GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                        bool load_if_create);

    /// Get a surface based on the texture configuration
    Surface GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config);
    Surface GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level = 0);

    /// Get a texture cube based on the texture configuration
    const Surface& GetTextureCube(const TextureCubeConfig& config);

    /// Get the color and depth surfaces based on the framebuffer configuration
    SurfaceSurfaceRect_Tuple GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb,
                                                    const Common::Rectangle<s32>& viewport_rect);

    /// Get a surface that matches the fill config
    Surface GetFillSurface(const GPU::Regs::MemoryFillConfig& config);

    /// Get a surface that matches a "texture copy" display transfer config
    SurfaceRect_Tuple GetTexCopySurface(const SurfaceParams& params);

    /// Write any cached resources overlapping the region back to memory (if dirty)
    void FlushRegion(PAddr addr, u32 size, Surface flush_surface = nullptr);

    /// Mark region as being invalidated by region_owner (nullptr if 3DS memory)
    void InvalidateRegion(PAddr addr, u32 size, const Surface& region_owner);

    /// Flush all cached resources tracked by this cache manager
    void FlushAll();

private:
    void DuplicateSurface(const Surface& src_surface, const Surface& dest_surface);

    /// Update surface's texture for given region when necessary
    void ValidateSurface(const Surface& surface, PAddr addr, u32 size);

    /// Copies pixel data in interval from the guest VRAM to the host GPU surface
    void UploadSurface(const Surface& surface, SurfaceInterval interval);

    /// Copies pixel data in interval from the host GPU surface to the guest VRAM
    void DownloadSurface(const Surface& surface, SurfaceInterval interval);

    /// Downloads a fill surface to guest VRAM
    void DownloadFillSurface(const Surface& surface, SurfaceInterval interval);

    /// Returns false if there is a surface in the cache at the interval with the same bit-width,
    bool NoUnimplementedReinterpretations(const Surface& surface, SurfaceParams& params,
                                          SurfaceInterval interval);

    /// Return true if a surface with an invalid pixel format exists at the interval
    bool IntervalHasInvalidPixelFormat(SurfaceParams& params, SurfaceInterval interval);

    /// Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(const Surface& surface, const SurfaceParams& params,
                                    SurfaceInterval interval);

    /// Create a new surface
    Surface CreateSurface(SurfaceParams& params);

    /// Register surface into the cache
    void RegisterSurface(const Surface& surface);

    /// Remove surface from the cache
    void UnregisterSurface(const Surface& surface);

private:
    VideoCore::RasterizerAccelerated& rasterizer;
    TextureRuntime& runtime;
    SurfaceCache surface_cache;
    SurfaceMap dirty_regions;
    SurfaceSet remove_surfaces;
    u16 resolution_scale_factor;

    std::unordered_map<TextureCubeConfig, Surface> texture_cube_cache;
    std::recursive_mutex mutex;
};

template <class T>
RasterizerCache<T>::RasterizerCache(VideoCore::RasterizerAccelerated& rasterizer, TextureRuntime& runtime)
    : rasterizer{rasterizer}, runtime{runtime} {
    resolution_scale_factor = VideoCore::GetResolutionScaleFactor();
}

template <class T>
template <MatchFlags find_flags>
auto RasterizerCache<T>::FindMatch(const SurfaceCache& surface_cache, const SurfaceParams& params,
                                   ScaleMatch match_scale_type,
                                   std::optional<SurfaceInterval> validate_interval) -> Surface {
    Surface match_surface = nullptr;
    bool match_valid = false;
    u32 match_scale = 0;
    SurfaceInterval match_interval{};

    for (const auto& pair : RangeFromInterval(surface_cache, params.GetInterval())) {
        for (const auto& surface : pair.second) {
            const bool res_scale_matched = match_scale_type == ScaleMatch::Exact
                                               ? (params.res_scale == surface->res_scale)
                                               : (params.res_scale <= surface->res_scale);
            // validity will be checked in GetCopyableInterval
            bool is_valid =
                True(find_flags & MatchFlags::Copy)
                    ? true
                    : surface->IsRegionValid(validate_interval.value_or(params.GetInterval()));

            if (False(find_flags & MatchFlags::Invalid) && !is_valid)
                continue;

            auto IsMatch_Helper = [&](auto check_type, auto match_fn) {
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
                auto UpdateMatch = [&] {
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
        }
    }
    return match_surface;
}

MICROPROFILE_DECLARE(RasterizerCache_BlitSurface);
template <class T>
bool RasterizerCache<T>::BlitSurfaces(const Surface& src_surface, Common::Rectangle<u32> src_rect,
                                      const Surface& dst_surface, Common::Rectangle<u32> dst_rect) {
    MICROPROFILE_SCOPE(RasterizerCache_BlitSurface);

    if (CheckFormatsBlittable(src_surface->pixel_format, dst_surface->pixel_format)) {
        dst_surface->InvalidateAllWatcher();

        const TextureBlit texture_blit = {
            .src_level = 0,
            .dst_level = 0,
            .src_layer = 0,
            .dst_layer = 0,
            .src_rect = src_rect,
            .dst_rect = dst_rect
        };

        return runtime.BlitTextures(*src_surface, *dst_surface, texture_blit);
    }

    return false;
}

MICROPROFILE_DECLARE(RasterizerCache_CopySurface);
template <class T>
void RasterizerCache<T>::CopySurface(const Surface& src_surface, const Surface& dst_surface,
                                     SurfaceInterval copy_interval) {
    MICROPROFILE_SCOPE(RasterizerCache_CopySurface);

    const SurfaceParams subrect_params = dst_surface->FromInterval(copy_interval);
    ASSERT(subrect_params.GetInterval() == copy_interval && src_surface != dst_surface);

    if (src_surface->type == SurfaceType::Fill) {
        // FillSurface needs a 4 bytes buffer
        const u32 fill_offset =
            (boost::icl::first(copy_interval) - src_surface->addr) % src_surface->fill_size;
        std::array<u8, 4> fill_buffer;

        u32 fill_buff_pos = fill_offset;
        for (std::size_t i = 0; i < fill_buffer.size(); i++) {
            fill_buffer[i] = src_surface->fill_data[fill_buff_pos++ % src_surface->fill_size];
        }

        const ClearValue clear_value =
            MakeClearValue(dst_surface->type, dst_surface->pixel_format, fill_buffer.data());
        const TextureClear clear_rect = {
            .texture_level = 0,
            .texture_rect = dst_surface->GetScaledSubRect(subrect_params)
        };

        runtime.ClearTexture(*dst_surface, clear_rect, clear_value);
        return;
    }

    if (src_surface->CanSubRect(subrect_params)) {
        const TextureBlit texture_blit = {
            .src_level = 0,
            .dst_level = 0,
            .src_layer = 0,
            .dst_layer = 0,
            .src_rect = src_surface->GetScaledSubRect(subrect_params),
            .dst_rect = dst_surface->GetScaledSubRect(subrect_params)
        };

        runtime.BlitTextures(*src_surface, *dst_surface, texture_blit);
        return;
    }

    UNREACHABLE();
}

template <class T>
auto RasterizerCache<T>::GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                                    bool load_if_create) -> Surface {
    if (params.addr == 0 || params.height * params.width == 0) {
        return nullptr;
    }

    // Use GetSurfaceSubRect instead
    ASSERT(params.width == params.stride);
    ASSERT(!params.is_tiled || (params.width % 8 == 0 && params.height % 8 == 0));

    // Check for an exact match in existing surfaces
    Surface surface =
        FindMatch<MatchFlags::Exact | MatchFlags::Invalid>(surface_cache, params, match_res_scale);

    if (surface == nullptr) {
        u16 target_res_scale = params.res_scale;
        if (match_res_scale != ScaleMatch::Exact) {
            // This surface may have a subrect of another surface with a higher res_scale, find
            // it to adjust our params
            SurfaceParams find_params = params;
            Surface expandable = FindMatch<MatchFlags::Expand | MatchFlags::Invalid>(
                surface_cache, find_params, match_res_scale);
            if (expandable != nullptr && expandable->res_scale > target_res_scale) {
                target_res_scale = expandable->res_scale;
            }
            // Keep res_scale when reinterpreting d24s8 -> rgba8
            if (params.pixel_format == PixelFormat::RGBA8) {
                find_params.pixel_format = PixelFormat::D24S8;
                expandable = FindMatch<MatchFlags::Expand | MatchFlags::Invalid>(
                    surface_cache, find_params, match_res_scale);
                if (expandable != nullptr && expandable->res_scale > target_res_scale) {
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
    if (params.addr == 0 || params.height * params.width == 0) {
        return std::make_tuple(nullptr, Common::Rectangle<u32>{});
    }

    // Attempt to find encompassing surface
    Surface surface = FindMatch<MatchFlags::SubRect | MatchFlags::Invalid>(surface_cache, params,
                                                                           match_res_scale);

    // Check if FindMatch failed because of res scaling
    // If that's the case create a new surface with
    // the dimensions of the lower res_scale surface
    // to suggest it should not be used again
    if (surface == nullptr && match_res_scale != ScaleMatch::Ignore) {
        surface = FindMatch<MatchFlags::SubRect | MatchFlags::Invalid>(surface_cache, params,
                                                                       ScaleMatch::Ignore);
        if (surface != nullptr) {
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
    if (surface == nullptr) {
        surface = FindMatch<MatchFlags::Expand | MatchFlags::Invalid>(surface_cache, aligned_params,
                                                                      match_res_scale);
        if (surface != nullptr) {
            aligned_params.width = aligned_params.stride;
            aligned_params.UpdateParams();

            SurfaceParams new_params = *surface;
            new_params.addr = std::min(aligned_params.addr, surface->addr);
            new_params.end = std::max(aligned_params.end, surface->end);
            new_params.size = new_params.end - new_params.addr;
            new_params.height =
                new_params.size / aligned_params.BytesInPixels(aligned_params.stride);
            ASSERT(new_params.size % aligned_params.BytesInPixels(aligned_params.stride) == 0);

            Surface new_surface = CreateSurface(new_params);
            DuplicateSurface(surface, new_surface);

            // Delete the expanded surface, this can't be done safely yet
            // because it may still be in use
            surface->UnlinkAllWatcher(); // unlink watchers as if this surface is already deleted
            remove_surfaces.emplace(surface);

            surface = new_surface;
            RegisterSurface(new_surface);
        }
    }

    // No subrect found - create and return a new surface
    if (surface == nullptr) {
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
auto RasterizerCache<T>::GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config) -> Surface {
    const auto info = Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
    return GetTextureSurface(info, config.config.lod.max_level);
}

template <class T>
auto RasterizerCache<T>::GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level) -> Surface {
    if (info.physical_address == 0) {
        return nullptr;
    }

    SurfaceParams params;
    params.addr = info.physical_address;
    params.width = info.width;
    params.height = info.height;
    params.is_tiled = true;
    params.pixel_format = PixelFormatFromTextureFormat(info.format);
    params.res_scale = /*texture_filterer->IsNull() ?*/ 1 /*: resolution_scale_factor*/;
    params.UpdateParams();

    u32 min_width = info.width >> max_level;
    u32 min_height = info.height >> max_level;
    if (min_width % 8 != 0 || min_height % 8 != 0) {
        LOG_CRITICAL(Render_OpenGL, "Texture size ({}x{}) is not multiple of 8", min_width,
                     min_height);
        return nullptr;
    }
    if (info.width != (min_width << max_level) || info.height != (min_height << max_level)) {
        LOG_CRITICAL(Render_OpenGL,
                     "Texture size ({}x{}) does not support required mipmap level ({})",
                     params.width, params.height, max_level);
        return nullptr;
    }

    auto surface = GetSurface(params, ScaleMatch::Ignore, true);
    if (!surface)
        return nullptr;

    // Update mipmap if necessary
    if (max_level != 0) {
        if (max_level >= 8) {
            // since PICA only supports texture size between 8 and 1024, there are at most eight
            // possible mipmap levels including the base.
            LOG_CRITICAL(Render_OpenGL, "Unsupported mipmap level {}", max_level);
            return nullptr;
        }

        // Allocate more mipmap level if necessary
        if (surface->max_level < max_level) {
            /*if (!texture_filterer->IsNull()) {
                // TODO: proper mipmap support for custom textures
                runtime.GenerateMipmaps(surface->texture, max_level);
            }*/

            surface->max_level = max_level;
        }

        // Blit mipmaps that have been invalidated
        SurfaceParams surface_params = *surface;
        for (u32 level = 1; level <= max_level; ++level) {
            // In PICA all mipmap levels are stored next to each other
            surface_params.addr +=
                surface_params.width * surface_params.height * surface_params.GetFormatBpp() / 8;
            surface_params.width /= 2;
            surface_params.height /= 2;
            surface_params.stride = 0; // reset stride and let UpdateParams re-initialize it
            surface_params.UpdateParams();

            auto& watcher = surface->level_watchers[level - 1];
            if (!watcher || !watcher->Get()) {
                auto level_surface = GetSurface(surface_params, ScaleMatch::Ignore, true);
                if (level_surface) {
                    watcher = level_surface->CreateWatcher();
                } else {
                    watcher = nullptr;
                }
            }

            if (watcher && !watcher->IsValid()) {
                auto level_surface = watcher->Get();
                if (!level_surface->invalid_regions.empty()) {
                    ValidateSurface(level_surface, level_surface->addr, level_surface->size);
                }

                if (/*texture_filterer->IsNull()*/true) {
                    const TextureBlit texture_blit = {
                        .src_level = 0,
                        .dst_level = level,
                        .src_layer = 0,
                        .dst_layer = 0,
                        .src_rect = level_surface->GetScaledRect(),
                        .dst_rect = surface_params.GetScaledRect()
                    };

                    runtime.BlitTextures(*level_surface, *surface, texture_blit);
                }

                watcher->Validate();
            }
        }
    }

    return surface;
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
            .texture_type = TextureType::CubeMap,
            .pixel_format = PixelFormatFromTextureFormat(config.format),
            .type = SurfaceType::Texture
        };

        it->second = CreateSurface(cube_params);
    }

    Surface& cube = it->second;

    // Update surface watchers
    auto& watchers = cube->level_watchers;
    const std::array addresses = {config.px, config.nx, config.py,
                                  config.ny, config.pz, config.nz};

    for (std::size_t i = 0; i < addresses.size(); i++) {
        auto& watcher = watchers[i];
        if (!watcher || !watcher->Get()) {
            Pica::Texture::TextureInfo info = {
                .physical_address = addresses[i],
                .width = config.width,
                .height = config.width,
                .format = config.format,
            };

            info.SetDefaultStride();
            auto surface = GetTextureSurface(info);
            if (surface) {
                watcher = surface->CreateWatcher();
            } else {
                /**
                 * Can occur when texture address is invalid. We mark the watcher with nullptr
                 * in this case and the content of the face wouldn't get updated. These are
                 * usually leftover setup in the texture unit and games are not supposed to draw
                 * using them.
                 */
                watcher = nullptr;
            }
        }
    }

    // Validate the face surfaces
    const u32 scaled_size = cube->GetScaledWidth();
    for (std::size_t i = 0; i < addresses.size(); i++) {
        const auto& watcher = watchers[i];
        if (watcher && !watcher->IsValid()) {
            auto face = watcher->Get();
            if (!face->invalid_regions.empty()) {
                ValidateSurface(face, face->addr, face->size);
            }

            const TextureBlit texture_blit = {
                .src_level = 0,
                .dst_level = 0,
                .src_layer = 0,
                .dst_layer = static_cast<u32>(i),
                .src_rect = face->GetScaledRect(),
                .dst_rect = Rect2D{0, scaled_size, scaled_size, 0}
            };

            runtime.BlitTextures(*face, *cube, texture_blit);
            watcher->Validate();
        }
    }

    return cube;
}

template <class T>
auto RasterizerCache<T>::GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb,
                                                const Common::Rectangle<s32>& viewport_rect)
                                                -> SurfaceSurfaceRect_Tuple {
    const auto& regs = Pica::g_state.regs;
    const auto& config = regs.framebuffer.framebuffer;

    // Update resolution_scale_factor and reset cache if changed
    const bool resolution_scale_changed =
        resolution_scale_factor != VideoCore::GetResolutionScaleFactor();
    const bool texture_filter_changed =
        /*VideoCore::g_texture_filter_update_requested.exchange(false) &&
        texture_filterer->Reset(Settings::values.texture_filter_name,
                                VideoCore::GetResolutionScaleFactor())*/false;

    if (resolution_scale_changed || texture_filter_changed) {
        resolution_scale_factor = VideoCore::GetResolutionScaleFactor();
        FlushAll();
        while (!surface_cache.empty())
            UnregisterSurface(*surface_cache.begin()->second.begin());
        texture_cube_cache.clear();
    }

    Common::Rectangle<u32> viewport_clamped{
        static_cast<u32>(std::clamp(viewport_rect.left, 0, static_cast<s32>(config.GetWidth()))),
        static_cast<u32>(std::clamp(viewport_rect.top, 0, static_cast<s32>(config.GetHeight()))),
        static_cast<u32>(std::clamp(viewport_rect.right, 0, static_cast<s32>(config.GetWidth()))),
        static_cast<u32>(
            std::clamp(viewport_rect.bottom, 0, static_cast<s32>(config.GetHeight())))};

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
        LOG_CRITICAL(Render_OpenGL, "Color and depth framebuffer memory regions overlap; "
                                    "overlapping framebuffers not supported!");
        using_depth_fb = false;
    }

    Common::Rectangle<u32> color_rect{};
    Surface color_surface = nullptr;
    if (using_color_fb)
        std::tie(color_surface, color_rect) =
            GetSurfaceSubRect(color_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> depth_rect{};
    Surface depth_surface = nullptr;
    if (using_depth_fb)
        std::tie(depth_surface, depth_rect) =
            GetSurfaceSubRect(depth_params, ScaleMatch::Exact, false);

    Common::Rectangle<u32> fb_rect{};
    if (color_surface != nullptr && depth_surface != nullptr) {
        fb_rect = color_rect;
        // Color and Depth surfaces must have the same dimensions and offsets
        if (color_rect.bottom != depth_rect.bottom || color_rect.top != depth_rect.top ||
            color_rect.left != depth_rect.left || color_rect.right != depth_rect.right) {
            color_surface = GetSurface(color_params, ScaleMatch::Exact, false);
            depth_surface = GetSurface(depth_params, ScaleMatch::Exact, false);
            fb_rect = color_surface->GetScaledRect();
        }
    } else if (color_surface != nullptr) {
        fb_rect = color_rect;
    } else if (depth_surface != nullptr) {
        fb_rect = depth_rect;
    }

    if (color_surface != nullptr) {
        ValidateSurface(color_surface, boost::icl::first(color_vp_interval),
                        boost::icl::length(color_vp_interval));
        color_surface->InvalidateAllWatcher();
    }
    if (depth_surface != nullptr) {
        ValidateSurface(depth_surface, boost::icl::first(depth_vp_interval),
                        boost::icl::length(depth_vp_interval));
        depth_surface->InvalidateAllWatcher();
    }

    return std::make_tuple(color_surface, depth_surface, fb_rect);
}

template <class T>
auto RasterizerCache<T>::GetFillSurface(const GPU::Regs::MemoryFillConfig& config) -> Surface {
    SurfaceParams params;
    params.addr = config.GetStartAddress();
    params.end = config.GetEndAddress();
    params.size = params.end - params.addr;
    params.type = SurfaceType::Fill;
    params.res_scale = std::numeric_limits<u16>::max();

    Surface new_surface = std::make_shared<typename T::SurfaceType>(params, runtime);

    std::memcpy(&new_surface->fill_data[0], &config.value_32bit, 4);
    if (config.fill_32bit) {
        new_surface->fill_size = 4;
    } else if (config.fill_24bit) {
        new_surface->fill_size = 3;
    } else {
        new_surface->fill_size = 2;
    }

    RegisterSurface(new_surface);
    return new_surface;
}

template <class T>
auto RasterizerCache<T>::GetTexCopySurface(const SurfaceParams& params) -> SurfaceRect_Tuple {
    Common::Rectangle<u32> rect{};

    Surface match_surface = FindMatch<MatchFlags::TexCopy | MatchFlags::Invalid>(
        surface_cache, params, ScaleMatch::Ignore);

    if (match_surface != nullptr) {
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
void RasterizerCache<T>::DuplicateSurface(const Surface& src_surface, const Surface& dest_surface) {
    ASSERT(dest_surface->addr <= src_surface->addr && dest_surface->end >= src_surface->end);

    BlitSurfaces(src_surface, src_surface->GetScaledRect(), dest_surface,
                 dest_surface->GetScaledSubRect(*src_surface));

    dest_surface->invalid_regions -= src_surface->GetInterval();
    dest_surface->invalid_regions += src_surface->invalid_regions;

    SurfaceRegions regions;
    for (const auto& pair : RangeFromInterval(dirty_regions, src_surface->GetInterval())) {
        if (pair.second == src_surface) {
            regions += pair.first;
        }
    }
    for (const auto& interval : regions) {
        dirty_regions.set({interval, dest_surface});
    }
}

template <class T>
void RasterizerCache<T>::ValidateSurface(const Surface& surface, PAddr addr, u32 size) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval validate_interval(addr, addr + size);
    if (surface->type == SurfaceType::Fill) {
        // Sanity check, fill surfaces will always be valid when used
        ASSERT(surface->IsRegionValid(validate_interval));
        return;
    }

    auto validate_regions = surface->invalid_regions & validate_interval;
    auto NotifyValidated = [&](SurfaceInterval interval) {
        surface->invalid_regions.erase(interval);
        validate_regions.erase(interval);
    };

    while (true) {
        const auto it = validate_regions.begin();
        if (it == validate_regions.end()) {
            break;
        }

        const auto interval = *it & validate_interval;
        // Look for a valid surface to copy from
        SurfaceParams params = surface->FromInterval(interval);

        Surface copy_surface =
            FindMatch<MatchFlags::Copy>(surface_cache, params, ScaleMatch::Ignore, interval);
        if (copy_surface != nullptr) {
            SurfaceInterval copy_interval = copy_surface->GetCopyableInterval(params);
            CopySurface(copy_surface, surface, copy_interval);
            NotifyValidated(copy_interval);
            continue;
        }

        // Try to find surface in cache with different format
        // that can can be reinterpreted to the requested format.
        if (ValidateByReinterpretation(surface, params, interval)) {
            NotifyValidated(interval);
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
                LOG_INFO(Render_OpenGL, "Region created fully on GPU and reinterpretation is "
                                         "invalid. Skipping validation");
                validate_regions.erase(interval);
                continue;
            }
        }

        // Load data from 3DS memory
        FlushRegion(params.addr, params.size);
        UploadSurface(surface, interval);
        NotifyValidated(params.GetInterval());
    }
}

MICROPROFILE_DECLARE(RasterizerCache_SurfaceLoad);
template <class T>
void RasterizerCache<T>::UploadSurface(const Surface& surface, SurfaceInterval interval) {
    const SurfaceParams info = surface->FromInterval(interval);
    const u32 load_start = info.addr;
    const u32 load_end = info.end;
    ASSERT(load_start >= surface->addr && load_end <= surface->end);

    const auto& staging = runtime.FindStaging(
                surface->width * surface->height * 4, true);
    MemoryRef source_ptr = VideoCore::g_memory->GetPhysicalRef(info.addr);
    if (!source_ptr) [[unlikely]] {
        return;
    }

    const auto upload_data = source_ptr.GetWriteBytes(load_end - load_start);

    MICROPROFILE_SCOPE(RasterizerCache_SurfaceLoad);

    if (surface->is_tiled) {
        std::vector<std::byte> unswizzled_data(staging.size);
        UnswizzleTexture(*surface, load_start - surface->addr, load_end - surface->addr,
                         upload_data, unswizzled_data);
        runtime.FormatConvert(surface->pixel_format, true, unswizzled_data, staging.mapped);
    } else {
        runtime.FormatConvert(surface->pixel_format, true, upload_data, staging.mapped);
    }

    const BufferTextureCopy upload = {
        .buffer_offset = 0,
        .buffer_size = staging.size,
        .texture_rect = surface->GetSubRect(info),
        .texture_level = 0
    };

    surface->Upload(upload, staging);
}

MICROPROFILE_DECLARE(RasterizerCache_SurfaceFlush);
template <class T>
void RasterizerCache<T>::DownloadSurface(const Surface& surface, SurfaceInterval interval) {
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface->addr && flush_end <= surface->end);

    const auto& staging = runtime.FindStaging(
                surface->width * surface->height * 4, false);
    const SurfaceParams params = surface->FromInterval(interval);
    const BufferTextureCopy download = {
        .buffer_offset = 0,
        .buffer_size = staging.size,
        .texture_rect = surface->GetSubRect(params),
        .texture_level = 0
    };

    surface->Download(download, staging);

    MemoryRef dest_ptr = VideoCore::g_memory->GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const auto download_dest = dest_ptr.GetWriteBytes(flush_end - flush_start);

    MICROPROFILE_SCOPE(RasterizerCache_SurfaceFlush);

    if (surface->is_tiled) {
        std::vector<std::byte> swizzled_data(staging.size);
        SwizzleTexture(*surface, flush_start - surface->addr, flush_end - surface->addr,
                       staging.mapped, swizzled_data);
        runtime.FormatConvert(surface->pixel_format, false, swizzled_data, download_dest);
    } else {
        runtime.FormatConvert(surface->pixel_format, false, staging.mapped, download_dest);
    }
}

template <class T>
void RasterizerCache<T>::DownloadFillSurface(const Surface& surface, SurfaceInterval interval) {
    const u32 flush_start = boost::icl::first(interval);
    const u32 flush_end = boost::icl::last_next(interval);
    ASSERT(flush_start >= surface->addr && flush_end <= surface->end);

    MemoryRef dest_ptr = VideoCore::g_memory->GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const u32 start_offset = flush_start - surface->addr;
    const u32 download_size = std::clamp(flush_end - flush_start, 0u, static_cast<u32>(dest_ptr.GetSize()));
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
bool RasterizerCache<T>::NoUnimplementedReinterpretations(const Surface& surface, SurfaceParams& params,
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
            // This could potentially be expensive,
            // although experimentally it hasn't been too bad
            Surface test_surface =
                FindMatch<MatchFlags::Copy>(surface_cache, params, ScaleMatch::Ignore, interval);
            if (test_surface != nullptr) {
                LOG_WARNING(Render_OpenGL, "Missing pixel_format reinterpreter: {} -> {}",
                            PixelFormatAsString(format),
                            PixelFormatAsString(surface->pixel_format));
                implemented = false;
            }
        }
    }
    return implemented;
}

template <class T>
bool RasterizerCache<T>::IntervalHasInvalidPixelFormat(SurfaceParams& params, SurfaceInterval interval) {
    params.pixel_format = PixelFormat::Invalid;
    for (const auto& set : RangeFromInterval(surface_cache, interval)) {
        for (const auto& surface : set.second) {
            if (surface->pixel_format == PixelFormat::Invalid) {
                LOG_DEBUG(Render_OpenGL, "Surface {:#x} found with invalid pixel format",
                          surface->addr);
                return true;
            }
        }
    }

    return false;
}

template <class T>
bool RasterizerCache<T>::ValidateByReinterpretation(const Surface& surface, const SurfaceParams& params,
                                                    SurfaceInterval interval) {
    /*const PixelFormat dst_format = surface->pixel_format;
    const SurfaceType type = GetFormatType(dst_format);

    for (auto& reinterpreter :
         format_reinterpreter->GetPossibleReinterpretations(surface->pixel_format)) {

        params.pixel_format = reinterpreter->GetSourceFormat();
        Surface reinterpret_surface =
            FindMatch<MatchFlags::Copy>(surface_cache, params, ScaleMatch::Ignore, interval);

        if (reinterpret_surface != nullptr) {
            auto reinterpret_interval = params.GetCopyableInterval(reinterpret_surface);
            auto reinterpret_params = surface->FromInterval(reinterpret_interval);
            auto src_rect = reinterpret_surface->GetScaledSubRect(reinterpret_params);
            auto dest_rect = surface->GetScaledSubRect(reinterpret_params);

            if (!texture_filterer->IsNull() && reinterpret_surface->res_scale == 1 &&
                surface->res_scale == resolution_scale_factor) {
                // The destination surface is either a framebuffer, or a filtered texture.
                // Create an intermediate surface to convert to before blitting to the
                // destination.
                const u32 width = dest_rect.GetHeight() / resolution_scale_factor;
                const u32 height = dest_rect.GetWidth() / resolution_scale_factor;
                const Common::Rectangle<u32> tmp_rect{0, width, height, 0};

                OGLTexture tmp_tex = AllocateSurfaceTexture(dst_format, height, width);
                reinterpreter->Reinterpret(reinterpret_surface->texture, src_rect, tmp_tex,
                                           tmp_rect);

                if (!texture_filterer->Filter(tmp_tex, tmp_rect, surface->texture, dest_rect, type)) {
                    const TextureBlit texture_blit = {
                        .surface_type = type,
                        .src_level = 0,
                        .dst_level = 0,
                        .src_layer = 0,
                        .dst_layer = 0,
                        .src_region = Region2D{
                            .start = {0, 0},
                            .end = {width, height}
                        },
                        .dst_region = Region2D{
                            .start = {dest_rect.left, dest_rect.bottom},
                            .end = {dest_rect.right, dest_rect.top}
                        }
                    };

                    runtime.BlitTextures(tmp_tex, surface->texture, texture_blit);
                }

            } else {
                reinterpreter->Reinterpret(reinterpret_surface->texture, src_rect, surface->texture,
                                           dest_rect);
            }

            return true;
        }
    }*/

    return false;
}

template <class T>
void RasterizerCache<T>::FlushRegion(PAddr addr, u32 size, Surface flush_surface) {
    std::lock_guard lock{mutex};

    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval flush_interval(addr, addr + size);
    SurfaceRegions flushed_intervals;

    for (auto& pair : RangeFromInterval(dirty_regions, flush_interval)) {
        // small sizes imply that this most likely comes from the cpu, flush the entire region
        // the point is to avoid thousands of small writes every frame if the cpu decides to
        // access that region, anything higher than 8 you're guaranteed it comes from a service
        const auto interval = size <= 8 ? pair.first : pair.first & flush_interval;
        auto& surface = pair.second;

        if (flush_surface != nullptr && surface != flush_surface)
            continue;

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
    std::lock_guard lock{mutex};

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

    for (const auto& pair : RangeFromInterval(surface_cache, invalid_interval)) {
        for (const auto& cached_surface : pair.second) {
            if (cached_surface == region_owner)
                continue;

            // If cpu is invalidating this region we want to remove it
            // to (likely) mark the memory pages as uncached
            if (region_owner == nullptr && size <= 8) {
                FlushRegion(cached_surface->addr, cached_surface->size, cached_surface);
                remove_surfaces.emplace(cached_surface);
                continue;
            }

            const auto interval = cached_surface->GetInterval() & invalid_interval;
            cached_surface->invalid_regions.insert(interval);
            cached_surface->InvalidateAllWatcher();

            // If the surface has no salvageable data it should be removed from the cache to avoid
            // clogging the data structure
            if (cached_surface->IsSurfaceFullyInvalid()) {
                remove_surfaces.emplace(cached_surface);
            }
        }
    }

    if (region_owner != nullptr)
        dirty_regions.set({invalid_interval, region_owner});
    else
        dirty_regions.erase(invalid_interval);

    for (const auto& remove_surface : remove_surfaces) {
        if (remove_surface == region_owner) {
            Surface expanded_surface = FindMatch<MatchFlags::SubRect | MatchFlags::Invalid>(
                surface_cache, *region_owner, ScaleMatch::Ignore);
            ASSERT(expanded_surface);

            if ((region_owner->invalid_regions - expanded_surface->invalid_regions).empty()) {
                DuplicateSurface(region_owner, expanded_surface);
            } else {
                continue;
            }
        }
        UnregisterSurface(remove_surface);
    }

    remove_surfaces.clear();
}

template <class T>
auto RasterizerCache<T>::CreateSurface(SurfaceParams& params) -> Surface {
    Surface surface = std::make_shared<typename T::SurfaceType>(params, runtime);
    surface->invalid_regions.insert(surface->GetInterval());

    return surface;
}

template <class T>
void RasterizerCache<T>::RegisterSurface(const Surface& surface) {
    std::lock_guard lock{mutex};

    if (surface->registered) {
        return;
    }
    surface->registered = true;
    surface_cache.add({surface->GetInterval(), SurfaceSet{surface}});
    rasterizer.UpdatePagesCachedCount(surface->addr, surface->size, 1);
}

template <class T>
void RasterizerCache<T>::UnregisterSurface(const Surface& surface) {
    std::lock_guard lock{mutex};

    if (!surface->registered) {
        return;
    }
    surface->registered = false;
    rasterizer.UpdatePagesCachedCount(surface->addr, surface->size, -1);
    surface_cache.subtract({surface->GetInterval(), SurfaceSet{surface}});
}

} // namespace VideoCore
