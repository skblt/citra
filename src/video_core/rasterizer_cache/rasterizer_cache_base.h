// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <optional>
#include <unordered_map>
#include <boost/icl/interval_map.hpp>
#include <boost/range/iterator_range.hpp>
#include "video_core/rasterizer_cache/surface_base.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/texture/texture_decode.h"

namespace Memory {
class MemorySystem;
}

namespace VideoCore {

enum class ScaleMatch {
    Exact,   ///< Only accept same res scale
    Upscale, ///< Only allow higher scale than params
    Ignore   ///< Accept every scaled res
};

enum class MatchFlags {
    Exact = 1 << 0,   ///< Surface perfectly matches params
    SubRect = 1 << 1, ///< Surface encompasses params
    Copy = 1 << 2,    ///< Surface that can be used as a copy source
    Expand = 1 << 3,  ///< Surface that can expand params
    TexCopy = 1 << 4  ///< Surface that will match a display transfer "texture copy" parameters
};

DECLARE_ENUM_FLAG_OPERATORS(MatchFlags);

class RasterizerAccelerated;

template <class T>
class RasterizerCache : NonCopyable {
    /// Address shift for caching surfaces into a hash table
    static constexpr u64 CITRA_PAGEBITS = 18;

    using TextureRuntime = typename T::RuntimeType;
    using Surface = std::shared_ptr<typename T::SurfaceType>;
    using Watcher = SurfaceWatcher<typename T::SurfaceType>;

    /// Declare rasterizer interval types
    using SurfaceMap = boost::icl::interval_map<PAddr, Surface, boost::icl::partial_absorber,
                                                std::less, boost::icl::inplace_plus,
                                                boost::icl::inter_section, SurfaceInterval>;

    using SurfaceRect_Tuple = std::tuple<Surface, Common::Rectangle<u32>>;
    using SurfaceSurfaceRect_Tuple = std::tuple<Surface, Surface, Common::Rectangle<u32>>;
    using PageMap = boost::icl::interval_map<u32, int>;

public:
    RasterizerCache(Memory::MemorySystem& memory, TextureRuntime& runtime);
    ~RasterizerCache();

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

    /// Clear all cached resources tracked by this cache manager
    void ClearAll(bool flush);

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
    template <MatchFlags find_flags>
    Surface FindMatch(const SurfaceParams& params, ScaleMatch match_scale_type,
                      std::optional<SurfaceInterval> validate_interval = std::nullopt);

    /// Duplicates src_surface contents to dest_surface
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
    bool ValidateByReinterpretation(const Surface& surface, SurfaceParams& params,
                                    SurfaceInterval interval);

    /// Create a new surface
    Surface CreateSurface(SurfaceParams& params);

    /// Register surface into the cache
    void RegisterSurface(const Surface& surface);

    /// Remove surface from the cache
    void UnregisterSurface(const Surface& surface);

    /// Unregisters all surfaces from the cache
    void UnregisterAll();

    /// Increase/decrease the number of surface in pages touching the specified region
    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta);

private:
    Memory::MemorySystem& memory;
    TextureRuntime& runtime;
    PageMap cached_pages;
    SurfaceMap dirty_regions;
    std::vector<Surface> remove_surfaces;
    u16 resolution_scale_factor;
    std::unordered_map<TextureCubeConfig, Surface> texture_cube_cache;

    // The internal surface cache is based on buckets of 256KB.
    // This fits better for the purpose of this cache as textures are normaly
    // large in size.
    std::unordered_map<u64, std::vector<Surface>, Common::IdentityHash<u64>> page_table;
};

} // namespace VideoCore
