// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <memory>
#include <boost/icl/interval_set.hpp>
#include "common/assert.h"
#include "video_core/rasterizer_cache/surface_params.h"

namespace VideoCore {

using SurfaceRegions = boost::icl::interval_set<PAddr, std::less, SurfaceInterval>;

/**
 * A watcher that notifies whether a cached surface has been changed. This is useful for caching
 * surface collection objects, including texture cube and mipmap.
 */
class SurfaceWatcher {
public:
    explicit SurfaceWatcher(std::weak_ptr<CachedSurface>&& surface) :
        surface(std::move(surface)) {}

    /// Checks whether the surface has been changed.
    bool IsValid() const {
        return !surface.expired() && valid;
    }

    /// Marks that the content of the referencing surface has been updated to the watcher user.
    void Validate() {
        DEBUG_ASSERT(!surface.expired());
        valid = true;
    }

    /// Gets the referencing surface. Returns null if the surface has been destroyed
    Surface Get() const {
        return surface.lock();
    }

private:
    friend class CachedSurface;
    std::weak_ptr<CachedSurface> surface;
    bool valid = false;
};

class CachedSurface : std::enable_shared_from_this<CachedSurface> {
public:
    CachedSurface(const SurfaceParams& params) : params(params) {}
    ~CachedSurface();

    // Read/Write data in 3DS memory to/from gl_buffer
    void LoadGLBuffer(PAddr load_start, PAddr load_end);
    void FlushGLBuffer(PAddr flush_start, PAddr flush_end);

    // Upload/Download data in gl_buffer in/to this surface's texture
    void UploadTexture(Common::Rectangle<u32> rect);
    void DownloadTexture(Common::Rectangle<u32> rect);

    bool CanFill(const SurfaceParams& dest_surface, SurfaceInterval fill_interval) const;
    bool CanCopy(const SurfaceParams& dest_surface, SurfaceInterval copy_interval) const;

    bool IsRegionValid(SurfaceInterval interval) const {
        return invalid_regions.find(interval) == invalid_regions.end();
    }

    bool IsFullyInvalid() const {
        auto interval = params.GetInterval();
        return *invalid_regions.equal_range(interval).first == interval;
    }

    static constexpr u32 GetBytesPerPixel(PixelFormat format) {
        if (format == PixelFormat::Invalid) {
            return 0;
        }

        if (format == PixelFormat::D24 || SurfaceParams::GetFormatType(format) == SurfaceType::Texture) {
            return 4;
        }

        return GetFormatBpp(format) / 8;
    }

    std::shared_ptr<SurfaceWatcher> CreateWatcher() {
        auto watcher = std::make_shared<SurfaceWatcher>(weak_from_this());
        watchers.push_front(watcher);
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
        watchers.clear();
    }

private:
    const SurfaceParams params{};
    SurfaceRegions invalid_regions;
    bool registered = false;

    u32 fill_size = 0; // Number of bytes to read from fill_data
    std::array<u8, 4> fill_data;

    //OGLTexture texture;

    // max mipmap level that has been attached to the texture
    u32 max_level = 0;
    // level_watchers[i] watches the (i+1)-th level mipmap source surface
    std::array<std::shared_ptr<SurfaceWatcher>, 7> level_watchers;
    std::vector<u8> pixel_buffer;
    std::list<std::weak_ptr<SurfaceWatcher>> watchers;
};

} // namespace VideoCore
