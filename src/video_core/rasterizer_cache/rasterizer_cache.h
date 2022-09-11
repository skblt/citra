// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <unordered_map>
#include "video_core/rasterizer_cache/cached_surface.h"
#include "video_core/rasterizer_cache/utils.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/texture/texture_decode.h"

namespace VideoCore {
class RasterizerAccelerated;
}

namespace OpenGL {

// Declare rasterizer interval types
using SurfaceSet = std::set<Surface>;
using SurfaceMap =
    boost::icl::interval_map<PAddr, Surface, boost::icl::partial_absorber, std::less,
                             boost::icl::inplace_plus, boost::icl::inter_section, SurfaceInterval>;
using SurfaceCache =
    boost::icl::interval_map<PAddr, SurfaceSet, boost::icl::partial_absorber, std::less,
                             boost::icl::inplace_plus, boost::icl::inter_section, SurfaceInterval>;

static_assert(std::is_same<SurfaceRegions::interval_type, SurfaceCache::interval_type>() &&
                  std::is_same<SurfaceMap::interval_type, SurfaceCache::interval_type>(),
              "Incorrect interval types");

using SurfaceRect_Tuple = std::tuple<Surface, Common::Rectangle<u32>>;
using SurfaceSurfaceRect_Tuple = std::tuple<Surface, Surface, Common::Rectangle<u32>>;

enum class ScaleMatch {
    Exact,   // Only accept same res scale
    Upscale, // Only allow higher scale than params
    Ignore   // Accept every scaled res
};

class Driver;
class TextureDownloaderES;
class TextureFilterer;
class FormatReinterpreterOpenGL;

class RasterizerCache : NonCopyable {
public:
    RasterizerCache(VideoCore::RasterizerAccelerated& rasterizer, Driver& driver);
    ~RasterizerCache();

    /// Blit one surface's texture to another
    bool BlitSurfaces(const Surface& src_surface, const Common::Rectangle<u32>& src_rect,
                      const Surface& dst_surface, const Common::Rectangle<u32>& dst_rect);

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
    const CachedTextureCube& GetTextureCube(const TextureCubeConfig& config);

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

    // Textures from destroyed surfaces are stored here to be recyled to reduce allocation overhead
    // in the driver
    // this must be placed above the surface_cache to ensure all cached surfaces are destroyed
    // before destroying the recycler
    std::unordered_multimap<HostTextureTag, OGLTexture> host_texture_recycler;

private:
    void DuplicateSurface(const Surface& src_surface, const Surface& dest_surface);

    /// Update surface's texture for given region when necessary
    void ValidateSurface(const Surface& surface, PAddr addr, u32 size);

    // Returns false if there is a surface in the cache at the interval with the same bit-width,
    bool NoUnimplementedReinterpretations(const OpenGL::Surface& surface,
                                          OpenGL::SurfaceParams& params,
                                          const OpenGL::SurfaceInterval& interval);

    // Return true if a surface with an invalid pixel format exists at the interval
    bool IntervalHasInvalidPixelFormat(SurfaceParams& params, const SurfaceInterval& interval);

    // Attempt to find a reinterpretable surface in the cache and use it to copy for validation
    bool ValidateByReinterpretation(const Surface& surface, SurfaceParams& params,
                                    const SurfaceInterval& interval);

    /// Create a new surface
    Surface CreateSurface(const SurfaceParams& params);

    /// Register surface into the cache
    void RegisterSurface(const Surface& surface);

    /// Remove surface from the cache
    void UnregisterSurface(const Surface& surface);

    VideoCore::RasterizerAccelerated& rasterizer;
    TextureRuntime runtime;
    SurfaceCache surface_cache;
    SurfaceMap dirty_regions;
    SurfaceSet remove_surfaces;

    u16 resolution_scale_factor;

    std::unordered_map<TextureCubeConfig, CachedTextureCube> texture_cube_cache;

    std::recursive_mutex mutex;

public:
    OGLTexture AllocateSurfaceTexture(PixelFormat format, u32 width, u32 height);

    std::unique_ptr<TextureFilterer> texture_filterer;
    std::unique_ptr<FormatReinterpreterOpenGL> format_reinterpreter;
    std::unique_ptr<TextureDownloaderES> texture_downloader_es;
};

} // namespace OpenGL
