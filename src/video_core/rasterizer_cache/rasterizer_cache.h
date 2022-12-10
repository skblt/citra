// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <vector>
#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/icl/interval_map.hpp>
#include <boost/range/iterator_range.hpp>
#include "common/alignment.h"
#include "common/logging/log.h"
#include "video_core/pica_state.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/rasterizer_cache/render_targets.h"
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/video_core.h"

namespace VideoCore {

inline auto RangeFromInterval(auto& map, SurfaceInterval interval) {
    return boost::make_iterator_range(map.equal_range(interval));
}

enum class ScaleMatch {
    Exact,   ///< Only accept same res scale
    Upscale, ///< Only allow higher or equal scale than params
    Ignore   ///< Accept every scaled res
};

enum class MatchType {
    Exact = 1 << 0,   ///< Surface perfectly matches params
    SubRect = 1 << 1, ///< Surface encompasses params
    Copy = 1 << 2,    ///< Surface that can be used as a copy source
    Expand = 1 << 3,  ///< Surface that can expand params
    TexCopy = 1 << 4  ///< Surface that will match a display transfer "texture copy" parameters
};

DECLARE_ENUM_FLAG_OPERATORS(MatchType);

class RasterizerAccelerated;

template <class T>
class RasterizerCache {
    /// Address shift for caching surfaces into a hash table
    static constexpr u64 CITRA_PAGEBITS = 18;

    /// Implement blits as copies between framebuffers
    static constexpr bool FRAMEBUFFER_BLITS = T::FRAMEBUFFER_BLITS;

    using Runtime = typename T::Runtime;
    using Surface = typename T::Surface;
    using Framebuffer = typename T::Framebuffer;
    using Allocation = typename T::Allocation;
    using Sampler = typename T::Sampler;
    using Format = typename T::Format;

    /// Declare rasterizer interval types
    using SurfaceMap = boost::icl::interval_map<PAddr, SurfaceId, boost::icl::partial_absorber,
                                                std::less, boost::icl::inplace_plus,
                                                boost::icl::inter_section, SurfaceInterval>;

    using SurfaceRect_Tuple = std::tuple<SurfaceId, Rect2D>;
    using FramebufferRect_Tuple = std::tuple<Framebuffer&, Rect2D>;

public:
    RasterizerCache(VideoCore::RasterizerAccelerated& rasterizer, Memory::MemorySystem& memory,
                    Runtime& runtime);
    ~RasterizerCache() = default;

    /// Perform hardware accelerated texture copy according to the provided configuration
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config);

    /// Perform hardware accelerated display transfer according to the provided configuration
    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config);

    /// Perform hardware accelerated memory fill according to the provided configuration
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config);

    /// Returns a reference to the surface assigned to the provided id
    Surface& GetSurface(SurfaceId surface_id);

    /// Returns a reference to the sampler object matching the provided configuration
    Sampler& GetSampler(const Pica::TexturingRegs::TextureConfig& config);
    Sampler& GetSampler(SamplerId sampler_id);

    /// Blit one surface's texture to another
    bool BlitSurfaces(SurfaceId src_surface_id, Rect2D src_rect, SurfaceId dst_surface_id,
                      Rect2D dst_rect);

    /// Copy one surface's region to another
    void CopySurface(SurfaceId src_id, SurfaceId dst_id, SurfaceInterval copy_interval);

    /// Load a texture from 3DS memory to OpenGL and cache it (if not already cached)
    SurfaceId GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                         bool load_if_create);

    /// Attempt to find a subrect (resolution scaled) of a surface,
    /// otherwise loads a texture from 3DS memory and caches it (if not already cached)
    SurfaceRect_Tuple GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                        bool load_if_create);

    /// Get a surface based on the texture configuration
    Surface& GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config);
    Surface& GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level = 0);

    /// Get a texture cube based on the texture configuration
    Surface& GetTextureCube(const TextureCubeConfig& config);

    /// Get the color and depth surfaces based on the framebuffer configuration
    std::pair<Framebuffer&, Rect2D> GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb);

    /// Get a surface that matches a "texture copy" display transfer config
    SurfaceRect_Tuple GetTexCopySurface(const SurfaceParams& params);

    /// Write any cached resources overlapping the region back to memory (if dirty)
    void FlushRegion(PAddr addr, u32 size, SurfaceId flush_surface_id = SurfaceId{});

    /// Mark region as being invalidated by region_owner (nullptr if 3DS memory)
    void InvalidateRegion(PAddr addr, u32 size, SurfaceId region_owner_id = SurfaceId{});

    /// Flush all cached resources tracked by this cache manager
    void FlushAll();

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_end = (addr + size - 1) >> CITRA_PAGEBITS;
        for (u64 page = addr >> CITRA_PAGEBITS; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Iterates over all the surfaces in a region calling func
    template <typename Func>
    void ForEachSurfaceInRegion(PAddr addr, size_t size, Func&& func);

    /// Get the best surface match (and its match type) for the given flags
    template <MatchType find_flags>
    SurfaceId FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                        std::optional<SurfaceInterval> validate_interval = std::nullopt,
                        bool relaxed_format = false);

    /// Find or create a framebuffer with the given render target parameters
    FramebufferId GetFramebufferId(SurfaceId color_id, SurfaceId depth_id, const RenderTargets& key);

    /// Create a render target from a given image and image view parameters
    [[nodiscard]] Framebuffer& FramebufferFromSurface(SurfaceId surface0_id,
                                                      SurfaceId surface1_id = SurfaceId{});

    void DuplicateSurface(Surface& src_surface, Surface& dest_surface);

    /// Update surface's texture for given region when necessary
    void ValidateSurface(SurfaceId surface_id, PAddr addr, u32 size);

    /// Copies pixel data in interval from the guest VRAM to the host GPU surface
    void UploadSurface(Surface& surface, SurfaceInterval interval);

    /// Copies pixel data in interval from the host GPU surface to the guest VRAM
    void DownloadSurface(Surface& surface, SurfaceInterval interval);

    /// Downloads a fill surface to guest VRAM
    void DownloadFillSurface(Surface& surface, SurfaceInterval interval);

    /// Returns false if there is a surface in the cache at the interval with the same bit-width,
    bool NoUnimplementedReinterpretations(Surface& surface, SurfaceParams& params,
                                          SurfaceInterval interval);

    /// Returns true if a surface with an invalid pixel format exists at the interval
    bool IntervalHasInvalidPixelFormat(SurfaceParams& params, SurfaceInterval interval);

    /// Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(Surface& surface, SurfaceParams& params,
                                    SurfaceInterval interval);

    /// Create a new surface
    SurfaceId CreateSurface(SurfaceParams& params);

    /// Register surface into the cache
    void RegisterSurface(SurfaceId surface_id);

    /// Remove surface from the cache
    void UnregisterSurface(SurfaceId surface_id);

private:
    VideoCore::RasterizerAccelerated& rasterizer;
    Memory::MemorySystem& memory;
    Runtime& runtime;
    SurfaceMap dirty_regions;
    u16 resolution_scale_factor;
    std::vector<std::function<void()>> download_queue;

    RenderTargets render_targets;

    std::unordered_map<RenderTargets, FramebufferId> framebuffers;
    std::unordered_map<u64, std::vector<SurfaceId>, Common::IdentityHash<u64>> page_table;

    struct HostTextureTag {
        Format format{};
        TextureType type = TextureType::Texture2D;
        u32 width = 1;
        u32 height = 1;
        u32 levels = 1;
        u32 res_scale = 1;

        auto operator<=>(const HostTextureTag&) const noexcept = default;
    };

    struct HostTextureTagHash {
        std::size_t operator()(const HostTextureTag& tag) const {
            return Common::ComputeHash64(&tag, sizeof(HostTextureTag));
        }
    };

    std::unordered_map<HostTextureTag, std::vector<AllocationId>, HostTextureTagHash> allocations;
    std::unordered_map<SamplerParams, SamplerId> samplers;

    SlotVector<Surface> slot_surfaces;
    SlotVector<Framebuffer> slot_framebuffers;
    SlotVector<Allocation> slot_allocations;
    SlotVector<Sampler> slot_samplers;

    std::vector<SurfaceId> remove_surfaces;
};

template <class T>
RasterizerCache<T>::RasterizerCache(VideoCore::RasterizerAccelerated& rasterizer,
                                    Memory::MemorySystem& memory, Runtime& runtime)
    : rasterizer{rasterizer}, memory{memory}, runtime{runtime},
      resolution_scale_factor{VideoCore::GetResolutionScaleFactor()} {

    using TextureConfig = Pica::TexturingRegs::TextureConfig;

    // Create null handles for all cached resources
    const AllocationId alloc_id = slot_allocations.insert(runtime);
    Allocation& alloc = slot_allocations[alloc_id];
    void(slot_surfaces.insert(runtime, std::move(alloc),
                              SurfaceParams{
                                  .width = 1,
                                  .height = 1,
                                  .stride = 1,
                                  .pixel_format = PixelFormat::RGBA8,
                              }));
    void(slot_framebuffers.insert(runtime, &slot_surfaces[NULL_SURFACE_ID], nullptr,
                                  RenderTargets{
                                      .size = {1, 1}
                                  }));
    void(slot_samplers.insert(runtime, SamplerParams{
                                           .mag_filter = TextureConfig::TextureFilter::Linear,
                                           .min_filter = TextureConfig::TextureFilter::Linear,
                                           .mip_filter = TextureConfig::TextureFilter::Linear,
                                           .wrap_s = TextureConfig::WrapMode::ClampToBorder,
                                           .wrap_t = TextureConfig::WrapMode::ClampToBorder,
                                       }));
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

    const auto [src_surface_id, src_rect] = GetTexCopySurface(src_params);
    if (!src_surface_id) {
        return false;
    }

    Surface& src_surface = slot_surfaces[src_surface_id];
    if (output_gap != 0 &&
        (output_width != src_surface.BytesInPixels(src_rect.GetWidth() / src_surface.res_scale) *
                             (src_surface.is_tiled ? 8 : 1) ||
         output_gap % src_surface.BytesInPixels(src_surface.is_tiled ? 64 : 1) != 0)) {
        return false;
    }

    SurfaceParams dst_params = src_surface;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = src_rect.GetWidth() / src_surface.res_scale;
    dst_params.stride = dst_params.width + src_surface.PixelsInBytes(
                                               src_surface.is_tiled ? output_gap / 8 : output_gap);
    dst_params.height = src_rect.GetHeight() / src_surface.res_scale;
    dst_params.res_scale = src_surface.res_scale;
    dst_params.UpdateParams();

    // Since we are going to invalidate the gap if there is one, we will have to load it first
    const bool load_gap = output_gap != 0;
    const auto [dst_surface_id, dst_rect] =
        GetSurfaceSubRect(dst_params, VideoCore::ScaleMatch::Upscale, load_gap);

    if (!dst_surface_id) {
        return false;
    }

    Surface& dst_surface = slot_surfaces[dst_surface_id];
    if (dst_surface.type == VideoCore::SurfaceType::Texture ||
        !BlitSurfaces(src_surface_id, src_rect, dst_surface_id, dst_rect)) {
        return false;
    }

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface_id);
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
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.input_format);
    src_params.UpdateParams();

    SurfaceParams dst_params;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = config.scaling != config.NoScale ? config.output_width.Value() / 2
                                                        : config.output_width.Value();
    dst_params.height = config.scaling == config.ScaleXY ? config.output_height.Value() / 2
                                                         : config.output_height.Value();
    dst_params.is_tiled = config.input_linear != config.dont_swizzle;
    dst_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.output_format);
    dst_params.UpdateParams();

    auto [src_surface_id, src_rect] =
        GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);
    if (!src_surface_id) {
        return false;
    }

    Surface& src_surface = slot_surfaces[src_surface_id];
    dst_params.res_scale = src_surface.res_scale;

    const auto [dst_surface_id, dst_rect] =
        GetSurfaceSubRect(dst_params, VideoCore::ScaleMatch::Upscale, false);
    if (!dst_surface_id) {
        return false;
    }

    Surface& dst_surface = slot_surfaces[dst_surface_id];
    if (src_surface.is_tiled != dst_surface.is_tiled)
        std::swap(src_rect.top, src_rect.bottom);

    if (config.flip_vertically)
        std::swap(src_rect.top, src_rect.bottom);

    if (!BlitSurfaces(src_surface_id, src_rect, dst_surface_id, dst_rect))
        return false;

    InvalidateRegion(dst_params.addr, dst_params.size, dst_surface_id);
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

    SurfaceId fill_surface_id = slot_surfaces.insert(params);
    Surface& fill_surface = slot_surfaces[fill_surface_id];

    std::memcpy(fill_surface.fill_data.data(), &config.value_32bit, sizeof(u32));
    if (config.fill_32bit) {
        fill_surface.fill_size = 4;
    } else if (config.fill_24bit) {
        fill_surface.fill_size = 3;
    } else {
        fill_surface.fill_size = 2;
    }

    RegisterSurface(fill_surface_id);
    InvalidateRegion(fill_surface.addr, fill_surface.size, fill_surface_id);

    return true;
}

template <class T>
auto RasterizerCache<T>::GetSurface(SurfaceId surface_id) -> Surface& {
    return slot_surfaces[surface_id];
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

    auto [it, is_new] = samplers.try_emplace(params, SamplerId{});
    if (is_new) {
        it->second = slot_samplers.insert(runtime, params);
    }

    return slot_samplers[it->second];
}

template <class T>
template <typename Func>
void RasterizerCache<T>::ForEachSurfaceInRegion(PAddr addr, size_t size, Func&& func) {
    using FuncReturn = typename std::invoke_result<Func, SurfaceId, Surface&>::type;
    static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
    boost::container::small_vector<SurfaceId, 32> surfaces;
    ForEachPage(addr, size, [this, &surfaces, addr, size, func](u64 page) {
        const auto it = page_table.find(page);
        if (it == page_table.end()) {
            if constexpr (BOOL_BREAK) {
                return false;
            } else {
                return;
            }
        }
        for (const SurfaceId surface_id : it->second) {
            Surface& surface = slot_surfaces[surface_id];
            if (surface.picked) {
                continue;
            }
            if (!surface.Overlaps(addr, size)) {
                continue;
            }

            surface.picked = true;
            surfaces.push_back(surface_id);
            if constexpr (BOOL_BREAK) {
                if (func(surface_id, surface)) {
                    return true;
                }
            } else {
                func(surface_id, surface);
            }
        }
        if constexpr (BOOL_BREAK) {
            return false;
        }
    });
    for (const SurfaceId surface_id : surfaces) {
        slot_surfaces[surface_id].picked = false;
    }
}

template <class T>
template <MatchType find_flags>
auto RasterizerCache<T>::FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                                   std::optional<SurfaceInterval> validate_interval,
                                   bool relaxed_format) -> SurfaceId {
    SurfaceId match_surface{};
    bool match_valid = false;
    u32 match_scale = 0;
    SurfaceInterval match_interval{};

    ForEachSurfaceInRegion(params.addr, params.size, [&](SurfaceId surface_id, Surface& surface) {
        const bool res_scale_matched = match_scale_type == ScaleMatch::Exact
                                           ? (params.res_scale == surface.res_scale)
                                           : (params.res_scale <= surface.res_scale);
        const bool is_valid =
            True(find_flags & MatchType::Copy)
                ? true
                : surface.IsRegionValid(validate_interval.value_or(params.GetInterval()));

        const auto IsMatch_Helper = [&](auto check_type, auto match_fn) {
            if (False(find_flags & check_type))
                return;

            bool matched;
            SurfaceInterval surface_interval;
            std::tie(matched, surface_interval) = match_fn();
            if (!matched)
                return;

            if (!res_scale_matched && match_scale_type != ScaleMatch::Ignore &&
                surface.type != SurfaceType::Fill)
                return;

            // Found a match, update only if this is better than the previous one
            const auto UpdateMatch = [&] {
                match_surface = surface_id;
                match_valid = is_valid;
                match_scale = surface.res_scale;
                match_interval = surface_interval;
            };

            if (surface.res_scale > match_scale) {
                UpdateMatch();
                return;
            } else if (surface.res_scale < match_scale) {
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

        IsMatch_Helper(std::integral_constant<MatchType, MatchType::Exact>{}, [&] {
            return std::make_pair(surface.ExactMatch(params, relaxed_format),
                                  surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchType, MatchType::SubRect>{}, [&] {
            return std::make_pair(surface.CanSubRect(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchType, MatchType::Expand>{}, [&] {
            return std::make_pair(surface.CanExpand(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchType, MatchType::TexCopy>{}, [&] {
            return std::make_pair(surface.CanTexCopy(params), surface.GetInterval());
        });
        IsMatch_Helper(std::integral_constant<MatchType, MatchType::Copy>{}, [&] {
            ASSERT(validate_interval);
            const SurfaceInterval copy_interval =
                surface.GetCopyableInterval(params.FromInterval(*validate_interval));
            const bool matched = boost::icl::length(copy_interval & *validate_interval) != 0 &&
                                 surface.CanCopy(params, copy_interval);
            return std::make_pair(matched, copy_interval);
        });
    });

    return match_surface;
}

template <class T>
FramebufferId RasterizerCache<T>::GetFramebufferId(SurfaceId color_id, SurfaceId depth_id, const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;
    if (!is_new) {
        return framebuffer_id;
    }

    Surface* const color_surface =
        color_id ? &slot_surfaces[color_id] : nullptr;
    Surface* const depth_surface =
        depth_id ? &slot_surfaces[depth_id] : nullptr;

    framebuffer_id = slot_framebuffers.insert(runtime, color_surface, depth_surface, key);
    return framebuffer_id;
}

template <class T>
auto RasterizerCache<T>::FramebufferFromSurface(SurfaceId surface0_id, SurfaceId surface1_id) -> Framebuffer& {
    if (!surface0_id && !surface1_id) [[unlikely]] {
        LOG_ERROR(HW_GPU, "Attempting to create framebuffer without a valid surface");
        return slot_framebuffers[NULL_FRAMEBUFFER_ID];
    }

    SurfaceId color_surface_id{};
    AllocationId color_id{};
    SurfaceId depth_surface_id{};
    AllocationId depth_id{};

    const auto Assign = [&](SurfaceId surface_id) {
        if (!surface_id) {
            return;
        }

        const Surface& surface = slot_surfaces[surface_id];
        const SurfaceType type = GetFormatType(surface.pixel_format);
        const bool is_depth = type == SurfaceType::Depth ||
                              type == SurfaceType::DepthStencil;
        if (is_depth && !depth_id) {
            depth_id = surface.AllocId();
            depth_surface_id = surface_id;
        } else if (!color_id) {
            color_id = surface.AllocId();
            color_surface_id = surface_id;
        } else {
            UNREACHABLE_MSG("Cannot create framebuffer with two surfaces of the same type");
        }
    };

    const Surface* surface =
            surface0_id ? &slot_surfaces[surface0_id] : &slot_surfaces[surface1_id];
    Assign(surface0_id);
    Assign(surface1_id);

    const FramebufferId framebuffer_id = GetFramebufferId(color_surface_id, depth_surface_id,
        RenderTargets{.color_buffer_id = color_id,
                      .depth_buffer_id = depth_id,
                      .size = {surface->GetScaledWidth(), surface->GetScaledHeight()},
                });
    return slot_framebuffers[framebuffer_id];
}

template <class T>
bool RasterizerCache<T>::BlitSurfaces(SurfaceId src_id, Rect2D src_rect, SurfaceId dst_id,
                                      Rect2D dst_rect) {
    Surface& src_surface = slot_surfaces[src_id];
    Surface& dst_surface = slot_surfaces[dst_id];

    const TextureBlit texture_blit = {
        .src_level = 0,
        .dst_level = 0,
        .src_layer = 0,
        .dst_layer = 0,
        .src_rect = src_rect,
        .dst_rect = dst_rect,
    };

    // OpenGL uses framebuffers for blit operations while vulkan does not
    if constexpr (FRAMEBUFFER_BLITS) {
        const Framebuffer& src_framebuffer = FramebufferFromSurface(src_id);
        const Framebuffer& dst_framebuffer = FramebufferFromSurface(dst_id);
        return runtime.FramebufferBlit(src_framebuffer, dst_framebuffer, texture_blit);
    } else {
        return runtime.SurfaceBlit(src_surface, dst_surface, texture_blit);
    }
}

template <class T>
void RasterizerCache<T>::CopySurface(SurfaceId src_id, SurfaceId dst_id,
                                     SurfaceInterval copy_interval) {
    Surface& src_surface = slot_surfaces[src_id];
    Surface& dst_surface = slot_surfaces[dst_id];

    const SurfaceParams subrect_params = dst_surface.FromInterval(copy_interval);
    const Rect2D dst_rect = dst_surface.GetScaledSubRect(subrect_params);
    ASSERT(subrect_params.GetInterval() == copy_interval && src_id != dst_id);

    if (src_surface.type == SurfaceType::Fill) {
        const ClearValue clear_value = MakeClearValue(
            src_surface, dst_surface.pixel_format, dst_surface.type, copy_interval.lower());

        const TextureClear texture_clear = {
            .texture_level = 0,
            .texture_rect = dst_rect,
            .type = dst_surface.type,
            .value = clear_value
        };

        // Attempt to do a framebuffer-less surface clear. This is supported
        // on newer versions of OpenGL and with full clears on Vulkan.
        // For all other cases like OpenGLES and partial vulkan clears, create
        // a framebuffer and use it as the clear target.
        if (!runtime.SurfaceClear(dst_surface, texture_clear)) {
            Framebuffer& dst_framebuffer = FramebufferFromSurface(dst_id);
            runtime.FramebufferClear(dst_framebuffer, texture_clear);
        }
        return;
    }

    if (src_surface.CanSubRect(subrect_params)) {
        const Rect2D src_rect = src_surface.GetScaledSubRect(subrect_params);
        BlitSurfaces(src_id, src_rect, dst_id, dst_rect);
        return;
    }

    UNREACHABLE();
}

template <class T>
auto RasterizerCache<T>::GetSurface(const SurfaceParams& params, ScaleMatch match_res_scale,
                                    bool load_if_create) -> SurfaceId {
    if (params.addr == 0 || params.height * params.width == 0) [[unlikely]] {
        return SurfaceId{};
    }

    // Use GetSurfaceSubRect instead
    ASSERT(params.width == params.stride);
    ASSERT(!params.is_tiled || (params.width % 8 == 0 && params.height % 8 == 0));

    // Check for an exact match in existing surfaces
    SurfaceId surface_id = FindMatch<MatchType::Exact>(params, match_res_scale);

    if (!surface_id) {
        u16 target_res_scale = params.res_scale;
        if (match_res_scale != ScaleMatch::Exact) {
            // This surface may have a subrect of another surface with a higher res_scale, find
            // it to adjust our params
            SurfaceParams find_params = params;
            SurfaceId expandable_id = FindMatch<MatchType::Expand>(find_params, match_res_scale);
            if (expandable_id) {
                Surface& expandable = slot_surfaces[expandable_id];
                if (expandable.res_scale > target_res_scale) {
                    target_res_scale = expandable.res_scale;
                }
            }

            // Keep res_scale when reinterpreting d24s8 -> rgba8
            if (params.pixel_format == PixelFormat::RGBA8) {
                find_params.pixel_format = PixelFormat::D24S8;
                expandable_id = FindMatch<MatchType::Expand>(find_params, match_res_scale);
                if (expandable_id) {
                    Surface& expandable = slot_surfaces[expandable_id];
                    if (expandable.res_scale > target_res_scale) {
                        target_res_scale = expandable.res_scale;
                    }
                }
            }
        }

        SurfaceParams new_params = params;
        new_params.res_scale = target_res_scale;
        surface_id = CreateSurface(new_params);
        RegisterSurface(surface_id);
    }

    if (load_if_create) {
        ValidateSurface(surface_id, params.addr, params.size);
    }

    return surface_id;
}

template <class T>
auto RasterizerCache<T>::GetSurfaceSubRect(const SurfaceParams& params, ScaleMatch match_res_scale,
                                           bool load_if_create) -> SurfaceRect_Tuple {
    if (params.addr == 0 || params.height * params.width == 0) [[unlikely]] {
        return std::make_tuple(SurfaceId{}, Rect2D{});
    }

    // Attempt to find encompassing surface
    SurfaceId surface_id = FindMatch<MatchType::SubRect>(params, match_res_scale);

    // Check if FindMatch failed because of res scaling. If that's the case create a new surface
    // with the dimensions of the lower res_scale surface to suggest it should not be used again
    if (!surface_id && match_res_scale != ScaleMatch::Ignore) {
        surface_id = FindMatch<MatchType::SubRect>(params, ScaleMatch::Ignore);
        if (surface_id) {
            Surface& surface = slot_surfaces[surface_id];

            SurfaceParams new_params = surface;
            new_params.res_scale = params.res_scale;

            surface_id = CreateSurface(new_params);
            RegisterSurface(surface_id);
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
    if (!surface_id) {
        surface_id = FindMatch<MatchType::Expand>(aligned_params, match_res_scale);
        if (surface_id) {
            aligned_params.width = aligned_params.stride;
            aligned_params.UpdateParams();

            Surface& surface = slot_surfaces[surface_id];
            SurfaceParams new_params = surface;
            new_params.addr = std::min(aligned_params.addr, surface.addr);
            new_params.end = std::max(aligned_params.end, surface.end);
            new_params.size = new_params.end - new_params.addr;
            new_params.height =
                new_params.size / aligned_params.BytesInPixels(aligned_params.stride);
            ASSERT(new_params.size % aligned_params.BytesInPixels(aligned_params.stride) == 0);

            SurfaceId new_surface_id = CreateSurface(new_params);
            // DuplicateSurface(surface, new_surface_id);

            // Delete the expanded surface, this can't be done safely yet
            // because it may still be in use
            remove_surfaces.push_back(surface_id);

            surface_id = new_surface_id;
            RegisterSurface(new_surface_id);
        }
    }

    // No subrect found - create and return a new surface
    if (!surface_id) {
        SurfaceParams new_params = aligned_params;
        // Can't have gaps in a surface
        new_params.width = aligned_params.stride;
        new_params.UpdateParams();
        // GetSurface will create the new surface and possibly adjust res_scale if necessary
        surface_id = GetSurface(new_params, match_res_scale, load_if_create);
    } else if (load_if_create) {
        ValidateSurface(surface_id, aligned_params.addr, aligned_params.size);
    }

    return std::make_tuple(surface_id, slot_surfaces[surface_id].GetScaledSubRect(params));
}

template <class T>
auto RasterizerCache<T>::GetTextureSurface(const Pica::TexturingRegs::FullTextureConfig& config)
    -> Surface& {
    const auto info = Pica::Texture::TextureInfo::FromPicaRegister(config.config, config.format);
    return GetTextureSurface(info, config.config.lod.max_level);
}

template <class T>
auto RasterizerCache<T>::GetTextureSurface(const Pica::Texture::TextureInfo& info, u32 max_level)
    -> Surface& {
    if (info.physical_address == 0) [[unlikely]] {
        // Can occur when texture addr is null or its memory is unmapped/invalid
        // HACK: In this case, the correct behaviour for the PICA is to use the last
        // rendered colour. But because this would be impractical to implement, the
        // next best alternative is to use a clear texture, essentially skipping
        // the geometry in question.
        // For example: a bug in Pokemon X/Y causes NULL-texture squares to be drawn
        // on the male character's face, which in the OpenGL default appear black.
        return slot_surfaces[NULL_SURFACE_ID];
    }

    SurfaceParams params;
    params.addr = info.physical_address;
    params.width = info.width;
    params.height = info.height;
    params.levels = 1;
    params.is_tiled = true;
    params.pixel_format = VideoCore::PixelFormatFromTextureFormat(info.format);
    params.UpdateParams();

    if (info.width % 8 != 0 || info.height % 8 != 0) {
        const auto [src_surface_id, rect] = GetSurfaceSubRect(params, ScaleMatch::Ignore, true);
        Surface& src_surface = slot_surfaces[src_surface_id];
        params.res_scale = src_surface.res_scale;

        SurfaceId tmp_surface_id = CreateSurface(params);
        Surface& tmp_surface = slot_surfaces[tmp_surface_id];

        BlitSurfaces(src_surface_id, rect, tmp_surface_id, tmp_surface.GetScaledRect());

        remove_surfaces.push_back(tmp_surface_id);
        return tmp_surface;
    }

    SurfaceId surface_id = GetSurface(params, ScaleMatch::Ignore, true);
    return surface_id ? slot_surfaces[surface_id] : slot_surfaces[NULL_SURFACE_ID];
}

template <class T>
auto RasterizerCache<T>::GetTextureCube(const TextureCubeConfig& config) -> Surface& {
    return slot_surfaces[NULL_SURFACE_ID];
}

template <class T>
auto RasterizerCache<T>::GetFramebufferSurfaces(bool using_color_fb, bool using_depth_fb)
    -> std::pair<Framebuffer&, Rect2D> {
    const auto& regs = Pica::g_state.regs;
    const auto& config = regs.framebuffer.framebuffer;

    // Update resolution_scale_factor and reset cache if changed
    const bool resolution_scale_changed =
        resolution_scale_factor != VideoCore::GetResolutionScaleFactor();
    const bool texture_filter_changed =
        /*VideoCore::g_texture_filter_update_requested.exchange(false) &&
        texture_filterer->Reset(Settings::values.texture_filter_name,
                                VideoCore::GetResolutionScaleFactor())*/
        false;

    if (resolution_scale_changed || texture_filter_changed) [[unlikely]] {
        resolution_scale_factor = VideoCore::GetResolutionScaleFactor();
        FlushAll();
        /*while (!surface_cache.empty()) {
            UnregisterSurface(*surface_cache.begin()->second.begin());
        }*/

        // texture_cube_cache.clear();
    }

    const s32 framebuffer_width = config.GetWidth();
    const s32 framebuffer_height = config.GetHeight();
    const auto viewport_rect = regs.rasterizer.GetViewportRect();
    const Rect2D viewport_clamped = {
        static_cast<u32>(std::clamp(viewport_rect.left, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.top, 0, framebuffer_height)),
        static_cast<u32>(std::clamp(viewport_rect.right, 0, framebuffer_width)),
        static_cast<u32>(std::clamp(viewport_rect.bottom, 0, framebuffer_height)),
    };

    // Get color and depth surfaces
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

    Rect2D color_rect{};
    SurfaceId color_surface_id{};
    if (using_color_fb) {
        std::tie(color_surface_id, color_rect) =
            GetSurfaceSubRect(color_params, ScaleMatch::Exact, false);
    }

    Rect2D depth_rect{};
    SurfaceId depth_surface_id{};
    if (using_depth_fb) {
        std::tie(depth_surface_id, depth_rect) =
            GetSurfaceSubRect(depth_params, ScaleMatch::Exact, false);
    }

    Rect2D fb_rect{};
    if (color_surface_id && depth_surface_id) {
        fb_rect = color_rect;
        // Color and Depth surfaces must have the same dimensions and offsets
        if (color_rect.bottom != depth_rect.bottom || color_rect.top != depth_rect.top ||
            color_rect.left != depth_rect.left || color_rect.right != depth_rect.right) {
            color_surface_id = GetSurface(color_params, ScaleMatch::Exact, false);
            depth_surface_id = GetSurface(depth_params, ScaleMatch::Exact, false);
            fb_rect = slot_surfaces[color_surface_id].GetScaledRect();
        }
    } else if (color_surface_id) {
        fb_rect = color_rect;
    } else if (depth_surface_id) {
        fb_rect = depth_rect;
    }

    if (color_surface_id) {
        ValidateSurface(color_surface_id, boost::icl::first(color_vp_interval),
                        boost::icl::length(color_vp_interval));
    }
    if (depth_surface_id) {
        ValidateSurface(depth_surface_id, boost::icl::first(depth_vp_interval),
                        boost::icl::length(depth_vp_interval));
    }

    const VideoCore::Rect2D draw_rect = {
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(fb_rect.left) +
                                             viewport_rect.left * resolution_scale_factor,
                                         fb_rect.left, fb_rect.right)), // Left
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(fb_rect.bottom) +
                                             viewport_rect.top * resolution_scale_factor,
                                         fb_rect.bottom, fb_rect.top)), // Top
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(fb_rect.left) +
                                             viewport_rect.right * resolution_scale_factor,
                                         fb_rect.left, fb_rect.right)), // Right
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(fb_rect.bottom) +
                                             viewport_rect.bottom * resolution_scale_factor,
                                         fb_rect.bottom, fb_rect.top)),
    };

    // Mark framebuffer surfaces as dirty
    const Common::Rectangle draw_rect_unscaled{draw_rect / resolution_scale_factor};
    if (using_color_fb) {
        const Surface& color_surface = slot_surfaces[color_surface_id];
        const SurfaceInterval interval = color_surface.GetSubRectInterval(draw_rect_unscaled);
        InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval), color_surface_id);
    }

    if (using_depth_fb) {
        const Surface& depth_surface = slot_surfaces[depth_surface_id];
        const SurfaceInterval interval = depth_surface.GetSubRectInterval(draw_rect_unscaled);
        InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval), depth_surface_id);
    }

    Framebuffer& framebuffer = FramebufferFromSurface(color_surface_id, depth_surface_id);
    framebuffer.SetRenderArea(draw_rect);

    return std::make_pair(std::ref(framebuffer), fb_rect);
}

template <class T>
auto RasterizerCache<T>::GetTexCopySurface(const SurfaceParams& params) -> SurfaceRect_Tuple {
    SurfaceId match_surface_id = FindMatch<MatchType::TexCopy>(params, ScaleMatch::Ignore);

    if (match_surface_id) {
        Surface& match_surface = slot_surfaces[match_surface_id];
        ValidateSurface(match_surface_id, params.addr, params.size);

        SurfaceParams match_subrect;
        if (params.width != params.stride) {
            const u32 tiled_size = match_surface.is_tiled ? 8 : 1;
            match_subrect = params;
            match_subrect.width = match_surface.PixelsInBytes(params.width) / tiled_size;
            match_subrect.stride = match_surface.PixelsInBytes(params.stride) / tiled_size;
            match_subrect.height *= tiled_size;
        } else {
            match_subrect = match_surface.FromInterval(params.GetInterval());
            ASSERT(match_subrect.GetInterval() == params.GetInterval());
        }

        const Rect2D rect = match_surface.GetScaledSubRect(match_subrect);
        return std::make_tuple(match_surface_id, rect);
    }

    return {};
}

template <class T>
void RasterizerCache<T>::DuplicateSurface(Surface& src_surface, Surface& dest_surface) {
    /*ASSERT(dest_surface.addr <= src_surface.addr && dest_surface.end >= src_surface.end);

    BlitSurfaces(src_surface, src_surface.GetScaledRect(), dest_surface,
                 dest_surface.GetScaledSubRect(*src_surface));

    dest_surface.invalid_regions -= src_surface.GetInterval();
    dest_surface.invalid_regions += src_surface.invalid_regions;

    SurfaceRegions regions;
    for (const auto& pair : RangeFromInterval(dirty_regions, src_surface.GetInterval())) {
        if (pair.second == src_surface) {
            regions += pair.first;
        }
    }

    for (const auto& interval : regions) {
        dirty_regions.set({interval, dest_surface});
    }*/
}

template <class T>
void RasterizerCache<T>::ValidateSurface(SurfaceId surface_id, PAddr addr, u32 size) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval validate_interval(addr, addr + size);

    Surface& surface = slot_surfaces[surface_id];
    if (surface.type == SurfaceType::Fill) {
        // Sanity check, fill surfaces will always be valid when used
        ASSERT(surface.IsRegionValid(validate_interval));
        return;
    }

    auto validate_regions = surface.invalid_regions & validate_interval;

    const auto NotifyValidated = [&](SurfaceInterval interval) {
        surface.invalid_regions.erase(interval);
        validate_regions.erase(interval);
    };

    while (true) {
        const auto it = validate_regions.begin();
        if (it == validate_regions.end()) {
            break;
        }

        // Look for a valid surface to copy from
        const auto interval = *it & validate_interval;
        SurfaceParams params = surface.FromInterval(interval);

        SurfaceId copy_surface_id =
            FindMatch<MatchType::Copy>(params, ScaleMatch::Ignore, interval);
        if (copy_surface_id) {
            const Surface& copy_surface = slot_surfaces[copy_surface_id];
            SurfaceInterval copy_interval = copy_surface.GetCopyableInterval(params);
            CopySurface(copy_surface_id, surface_id, copy_interval);
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
                LOG_DEBUG(HW_GPU, "Region created fully on GPU and reinterpretation is "
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

template <class T>
void RasterizerCache<T>::UploadSurface(Surface& surface, SurfaceInterval interval) {
    const SurfaceParams load_info = surface.FromInterval(interval);
    ASSERT(load_info.addr >= surface.addr && load_info.end <= surface.end);

    const auto staging = runtime.FindStaging(
        load_info.width * load_info.height * surface.InternalBytesPerPixel(), true);
    MemoryRef source_ptr = memory.GetPhysicalRef(load_info.addr);
    if (!source_ptr) [[unlikely]] {
        return;
    }

    const std::span upload_data = source_ptr.GetWriteBytes(load_info.end - load_info.addr);
    if (surface.is_tiled) {
        UnswizzleTexture(load_info, load_info.addr, load_info.end, upload_data, staging.mapped,
                         runtime.NeedsConvertion(surface.pixel_format));
    } else {
        runtime.FormatConvert(surface, true, upload_data, staging.mapped);
    }

    const BufferTextureCopy upload = {
        .buffer_offset = 0,
        .buffer_size = staging.size,
        .texture_rect = surface.GetSubRect(load_info),
        .texture_level = 0,
    };

    surface.Upload(upload, staging);
}

template <class T>
void RasterizerCache<T>::DownloadSurface(Surface& surface, SurfaceInterval interval) {
    const SurfaceParams flush_info = surface.FromInterval(interval);
    const u32 flush_start = interval.lower();
    const u32 flush_end = interval.upper();
    ASSERT(flush_start >= surface.addr && flush_end <= surface.end);

    const auto staging = runtime.FindStaging(
        flush_info.width * flush_info.height * surface.InternalBytesPerPixel(), false);
    const BufferTextureCopy download = {
        .buffer_offset = 0,
        .buffer_size = staging.size,
        .texture_rect = surface.GetSubRect(flush_info),
        .texture_level = 0,
    };

    surface.Download(download, staging);

    MemoryRef dest_ptr = memory.GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const std::span download_dest = dest_ptr.GetWriteBytes(flush_end - flush_start);
    download_queue.push_back([this, &surface, flush_start, flush_end, flush_info,
                              mapped = staging.mapped, download_dest]() {
        if (surface.is_tiled) {
            SwizzleTexture(flush_info, flush_start, flush_end, mapped, download_dest,
                           runtime.NeedsConvertion(surface.pixel_format));
        } else {
            runtime.FormatConvert(surface, false, mapped, download_dest);
        }
    });
}

template <class T>
void RasterizerCache<T>::DownloadFillSurface(Surface& surface, SurfaceInterval interval) {
    const u32 flush_start = interval.lower();
    const u32 flush_end = interval.upper();
    ASSERT(flush_start >= surface.addr && flush_end <= surface.end);

    MemoryRef dest_ptr = memory.GetPhysicalRef(flush_start);
    if (!dest_ptr) [[unlikely]] {
        return;
    }

    const u32 start_offset = flush_start - surface.addr;
    const u32 download_size =
        std::clamp(flush_end - flush_start, 0u, static_cast<u32>(dest_ptr.GetSize()));
    const u32 coarse_start_offset = start_offset - (start_offset % surface.fill_size);
    const u32 backup_bytes = start_offset % surface.fill_size;

    std::array<u8, 4> backup_data;
    if (backup_bytes) {
        std::memcpy(backup_data.data(), &dest_ptr[coarse_start_offset], backup_bytes);
    }

    for (u32 offset = coarse_start_offset; offset < download_size; offset += surface.fill_size) {
        std::memcpy(&dest_ptr[offset], &surface.fill_data[0],
                    std::min(surface.fill_size, download_size - offset));
    }

    if (backup_bytes) {
        std::memcpy(&dest_ptr[coarse_start_offset], &backup_data[0], backup_bytes);
    }
}

template <class T>
bool RasterizerCache<T>::NoUnimplementedReinterpretations(Surface& surface, SurfaceParams& params,
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
        if (GetFormatBpp(format) == surface.GetFormatBpp()) {
            params.pixel_format = format;
            // This could potentially be expensive, although experimentally it hasn't been too bad
            SurfaceId test_surface_id =
                FindMatch<MatchType::Copy>(params, ScaleMatch::Ignore, interval);

            if (test_surface_id) {
                LOG_WARNING(HW_GPU, "Missing pixel_format reinterpreter: {} -> {}",
                            PixelFormatAsString(format), PixelFormatAsString(surface.pixel_format));
                implemented = false;
            }
        }
    }

    return implemented;
}

template <class T>
bool RasterizerCache<T>::IntervalHasInvalidPixelFormat(SurfaceParams& params,
                                                       SurfaceInterval interval) {
    bool invalid_format_found = false;
    ForEachSurfaceInRegion(params.addr, params.end, [&](SurfaceId surface_id, Surface& surface) {
        if (surface.pixel_format == PixelFormat::Invalid && surface.type != SurfaceType::Fill) {
            LOG_DEBUG(HW_GPU, "Surface {:#x} found with invalid pixel format", surface.addr);
            invalid_format_found = true;
            return true;
        }
        return false;
    });

    return invalid_format_found;
}

template <class T>
bool RasterizerCache<T>::ValidateByReinterpretation(Surface& surface, SurfaceParams& params,
                                                    SurfaceInterval interval) {
    const PixelFormat dest_format = surface.pixel_format;
    for (const auto& reinterpreter : runtime.GetPossibleReinterpretations(dest_format)) {
        params.pixel_format = reinterpreter->GetSourceFormat();
        SurfaceId reinterpret_surface_id =
            FindMatch<MatchType::Copy>(params, ScaleMatch::Ignore, interval);

        if (reinterpret_surface_id) {
            Surface& reinterpret_surface = slot_surfaces[reinterpret_surface_id];
            const SurfaceInterval reinterpret_interval =
                reinterpret_surface.GetCopyableInterval(params);
            const SurfaceParams reinterpret_params = surface.FromInterval(reinterpret_interval);
            const Rect2D src_rect = reinterpret_surface.GetScaledSubRect(reinterpret_params);
            const Rect2D dest_rect = surface.GetScaledSubRect(reinterpret_params);

            reinterpreter->Reinterpret(reinterpret_surface, src_rect, surface, dest_rect);
            return true;
        }
    }

    return false;
}

template <class T>
void RasterizerCache<T>::FlushRegion(PAddr addr, u32 size, SurfaceId flush_surface_id) {
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
        SurfaceId surface_id = pair.second;

        if (flush_surface_id && surface_id != flush_surface_id)
            continue;

        // Sanity check, this surface is the last one that marked this region dirty
        Surface& surface = slot_surfaces[surface_id];
        ASSERT(surface.IsRegionValid(interval));

        if (surface.type == SurfaceType::Fill) {
            DownloadFillSurface(surface, interval);
        } else {
            DownloadSurface(surface, interval);
        }

        flushed_intervals += interval;
    }

    // Batch execute all requested downloads. This gives more time for them to complete
    // before we issue the CPU to GPU flush and reduces scheduler slot switches in Vulkan
    if (!download_queue.empty()) {
        runtime.Finish();
        for (const auto& download_func : download_queue) {
            download_func();
        }

        download_queue.clear();
    }

    // Reset dirty regions
    dirty_regions -= flushed_intervals;
}

template <class T>
void RasterizerCache<T>::FlushAll() {
    FlushRegion(0, 0xFFFFFFFF);
}

template <class T>
void RasterizerCache<T>::InvalidateRegion(PAddr addr, u32 size, SurfaceId region_owner_id) {
    if (size == 0) [[unlikely]] {
        return;
    }

    const SurfaceInterval invalid_interval{addr, addr + size};
    if (region_owner_id) {
        Surface& region_owner = slot_surfaces[region_owner_id];
        region_owner.invalid_regions.erase(invalid_interval);

        // Surfaces can't have a gap
        ASSERT(region_owner.width == region_owner.stride);
        ASSERT(region_owner.type != SurfaceType::Texture);
        ASSERT(addr >= region_owner.addr && addr + size <= region_owner.end);
    }

    ForEachSurfaceInRegion(addr, size, [&](SurfaceId surface_id, Surface& surface) {
        if (surface_id == region_owner_id) {
            return;
        }

        // If cpu is invalidating this region we want to remove it
        // to (likely) mark the memory pages as uncached
        if (!region_owner_id && size <= 8) {
            FlushRegion(surface.addr, surface.size, surface_id);
            remove_surfaces.push_back(surface_id);
            return;
        }

        const SurfaceInterval interval = surface.GetInterval() & invalid_interval;
        surface.invalid_regions.insert(interval);

        // If the surface has no salvageable data it should be removed from the cache to avoid
        // clogging the data structure
        if (surface.IsFullyInvalid()) {
            remove_surfaces.push_back(surface_id);
        }
    });

    if (region_owner_id)
        dirty_regions.set({invalid_interval, region_owner_id});
    else
        dirty_regions.erase(invalid_interval);

    for (const SurfaceId remove_surface_id : remove_surfaces) {
        /*if (&slot_surfaces[remove_surface_id] == region_owner) {
            SurfaceId expanded_surface_id = FindMatch<MatchType::SubRect>(
                        *region_owner, ScaleMatch::Ignore);
            ASSERT(expanded_surface_id);

            SurfaceId
            if ((region_owner->invalid_regions - expanded_surface.invalid_regions).empty()) {
                DuplicateSurface(region_owner, expanded_surface);
            } else {
                continue;
            }
        }*/
        UnregisterSurface(remove_surface_id);
    }

    remove_surfaces.clear();
}

template <class T>
auto RasterizerCache<T>::CreateSurface(SurfaceParams& params) -> SurfaceId {
    const HostTextureTag tag = {
        .format = runtime.NativeFormat(params.pixel_format),
        .type = params.texture_type,
        .width = params.width,
        .height = params.height,
        .levels = params.levels,
        .res_scale = params.res_scale
    };

    AllocationId alloc_id{};
    if (auto it = allocations.find(tag); it != allocations.end() && !it->second.empty()) {
        std::vector<AllocationId>& alloc_ids = it->second;
        alloc_id = alloc_ids.back();
        alloc_ids.pop_back();
    } else {
        alloc_id = slot_allocations.insert(runtime, params);
    }

    Allocation& alloc = slot_allocations[alloc_id];
    const SurfaceId surface_id = slot_surfaces.insert(runtime, std::move(alloc), params);
    Surface& surface = slot_surfaces[surface_id];

    surface.alloc_id = alloc_id;
    surface.invalid_regions.insert(surface.GetInterval());

    return surface_id;
}

template <class T>
void RasterizerCache<T>::RegisterSurface(SurfaceId surface_id) {
    Surface& surface = slot_surfaces[surface_id];
    if (surface.registered) {
        return;
    }

    surface.registered = true;
    rasterizer.UpdatePagesCachedCount(surface.addr, surface.size, 1);
    ForEachPage(surface.addr, surface.size,
                [this, surface_id](u64 page) { page_table[page].push_back(surface_id); });
}

template <class T>
void RasterizerCache<T>::UnregisterSurface(SurfaceId surface_id) {
    Surface& surface = slot_surfaces[surface_id];
    if (!surface.registered) {
        return;
    }

    surface.registered = false;
    rasterizer.UpdatePagesCachedCount(surface.addr, surface.size, -1);

    const AllocationId alloc_id = surface.AllocId();
    if (alloc_id) {
        const HostTextureTag tag = {
            .format = runtime.NativeFormat(surface.pixel_format),
            .type = surface.texture_type,
            .width = surface.width,
            .height = surface.height,
            .levels = surface.levels,
            .res_scale = surface.res_scale
        };

        slot_allocations[alloc_id] = surface.Release();
        allocations[tag].push_back(alloc_id);
    }

    ForEachPage(surface.addr, surface.size, [this, surface_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == page_table.end()) {
            ASSERT_MSG(false, "Unregistering unregistered page=0x{:x}", page << CITRA_PAGEBITS);
            return;
        }
        std::vector<SurfaceId>& surface_ids = page_it->second;
        const auto vector_it = std::ranges::find(surface_ids, surface_id);
        if (vector_it == surface_ids.end()) {
            ASSERT_MSG(false, "Unregistering unregistered surface in page=0x{:x}",
                       page << CITRA_PAGEBITS);
            return;
        }
        surface_ids.erase(vector_it);
    });

    slot_surfaces.erase(surface_id);
}

} // namespace VideoCore
