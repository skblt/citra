// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "common/assert.h"
#include "core/custom_tex_cache.h"
#include "video_core/rasterizer_cache/surface_params.h"
#include "video_core/rasterizer_cache/texture_runtime.h"

namespace OpenGL {

using SurfaceRegions = boost::icl::interval_set<PAddr, std::less, SurfaceInterval>;

/**
 * A watcher that notifies whether a cached surface has been changed. This is useful for caching
 * surface collection objects, including texture cube and mipmap.
 */
class SurfaceWatcher {
    friend class CachedSurface;

public:
    explicit SurfaceWatcher(std::weak_ptr<CachedSurface>&& surface) : surface(std::move(surface)) {}

    /// Checks whether the surface has been changed.
    bool IsValid() const {
        return !surface.expired() && valid;
    }

    /// Marks that the content of the referencing surface has been updated to the watcher user.
    void Validate() {
        ASSERT(!surface.expired());
        valid = true;
    }

    /// Gets the referencing surface. Returns null if the surface has been destroyed
    Surface Get() const {
        return surface.lock();
    }

private:
    std::weak_ptr<CachedSurface> surface;
    bool valid = false;
};

class RasterizerCache;
class StagingBuffer;

class CachedSurface : public SurfaceParams, public std::enable_shared_from_this<CachedSurface> {
public:
    CachedSurface(SurfaceParams params, RasterizerCache& owner, TextureRuntime& runtime)
        : SurfaceParams(params), owner(owner), runtime(runtime) {}
    ~CachedSurface();

    /// Upload/Download data in gl_buffer in/to this surface's texture
    void UploadTexture(Common::Rectangle<u32> rect, const StagingBuffer& staging);
    void DownloadTexture(Common::Rectangle<u32> rect, const StagingBuffer& staging);

    bool CanFill(const SurfaceParams& dest_surface, SurfaceInterval fill_interval) const;
    bool CanCopy(const SurfaceParams& dest_surface, SurfaceInterval copy_interval) const;

    bool IsRegionValid(SurfaceInterval interval) const {
        return (invalid_regions.find(interval) == invalid_regions.end());
    }

    bool IsSurfaceFullyInvalid() const {
        auto interval = GetInterval();
        return *invalid_regions.equal_range(interval).first == interval;
    }

    std::shared_ptr<SurfaceWatcher> CreateWatcher() {
        auto watcher = std::make_shared<SurfaceWatcher>(weak_from_this());
        watchers[watcher_count++] = watcher;
        return watcher;
    }

    void InvalidateAllWatcher() {
        for (const auto& watcher : watchers) {
            if (auto locked = watcher.lock()) {
                locked->valid = false;
            }
        }
    }

    void UnlinkAllWatcher() {
        for (const auto& watcher : watchers) {
            if (auto locked = watcher.lock()) {
                locked->valid = false;
                locked->surface.reset();
            }
        }

        watchers = {};
        watcher_count = 0;
    }

public:
    bool registered = false;
    SurfaceRegions invalid_regions;

    // Number of bytes to read from fill_data
    u32 fill_size = 0;
    std::array<u8, 4> fill_data;
    OGLTexture texture;

    std::array<std::shared_ptr<SurfaceWatcher>, 7> level_watchers;
    u32 max_level = 0;

private:
    RasterizerCache& owner;
    TextureRuntime& runtime;
    u32 watcher_count = 0;
    std::array<std::weak_ptr<SurfaceWatcher>, 8> watchers;
};

struct CachedTextureCube {
    OGLTexture texture;
    u16 res_scale = 1;
    std::shared_ptr<SurfaceWatcher> px;
    std::shared_ptr<SurfaceWatcher> nx;
    std::shared_ptr<SurfaceWatcher> py;
    std::shared_ptr<SurfaceWatcher> ny;
    std::shared_ptr<SurfaceWatcher> pz;
    std::shared_ptr<SurfaceWatcher> nz;
};

} // namespace OpenGL
