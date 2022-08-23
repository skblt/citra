// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <queue>
#include "common/alignment.h"
#include "common/common_funcs.h"
#include "common/common_types.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/transform_cache/slot_vector.h"
#include "video_core/transform_cache/types.h"
#include "video_core/transform_cache/surface.h"
#include "video_core/texture/texture_decode.h"
#include "video_core/transform_cache/utils.h"
#include "video_core/pica_state.h"

namespace VideoCore {

enum class DirtyFlags : u8 {
    RenderTargets = 0,
    ColorBuffer = 1,
    DepthBuffer = 2
};

DECLARE_ENUM_FLAG_OPERATORS(DirtyFlags);

/// Container to push objects to be destroyed a few ticks in the future
template <typename T, size_t TICKS_TO_DESTROY>
class DelayedDestructionRing {
public:
    void Tick() {
        index = (index + 1) % TICKS_TO_DESTROY;
        elements[index].clear();
    }

    void Push(T&& object) {
        elements[index].push_back(std::move(object));
    }

private:
    size_t index = 0;
    std::array<std::vector<T>, TICKS_TO_DESTROY> elements;
};

template <class P>
class RasterizerCache {
    /// Enables debugging features to the texture cache
    static constexpr bool ENABLE_VALIDATION = P::ENABLE_VALIDATION;
    /// Implement blits as copies between framebuffers
    static constexpr bool FRAMEBUFFER_BLITS = P::FRAMEBUFFER_BLITS;

    /// Image view ID for null descriptors
    static constexpr SurfaceViewId NULL_IMAGE_VIEW_ID{0};

    using Runtime = typename P::Runtime;
    //using Surface = typename P::Surface;
    using SurfaceAlloc = typename P::SurfaceAlloc;
    using SurfaceView = typename P::SurfaceView;
    using Framebuffer = typename P::Framebuffer;

    template <typename T>
    struct IdentityHash {
        [[nodiscard]] size_t operator()(T value) const noexcept {
            return static_cast<size_t>(value);
        }
    };

public:
    explicit RasterizerCache(Runtime& runtime, RasterizerAccelerated& rasterizer);

    /// Notify the cache that a new frame has been queued
    void TickFrame();

    /// Return a constant reference to the given image view id
    [[nodiscard]] const SurfaceView& GetSurfaceView(SurfaceViewId id) const noexcept;

    /// Return a reference to the given image view id
    [[nodiscard]] SurfaceView& GetSurfaceView(SurfaceViewId id) noexcept;

    /// Update bound render targets and upload memory if necessary
    void UpdateRenderTargets(bool is_clear);

    /// Find a framebuffer with the currently bound render targets
    /// UpdateRenderTargets should be called before this
    Framebuffer* GetFramebuffer();

    /// Mark images in a range as modified from the CPU
    void WriteMemory(PAddr cpu_addr, size_t size);

    /// Download contents of host images to guest memory in a region
    void DownloadMemory(PAddr cpu_addr, size_t size);

    /// Remove images in a region
    void UnmapMemory(PAddr cpu_addr, size_t size);

    /// Attempts to perform a PICA texture copy using the config provided
    bool TextureCopy(const GPU::Regs::DisplayTransferConfig& config);

    /// Attempts to perform a PICA display transfer using the config provided
    bool DisplayTransfer(const GPU::Regs::DisplayTransferConfig& config);

    /// Invalidate the contents of the current color buffer
    /// These contents become unspecified, the cache can assume aggressive optimizations.
    void InvalidateColorBuffer();

    /// Invalidate the contents of the depth buffer
    /// These contents become unspecified, the cache can assume aggressive optimizations.
    void InvalidateDepthBuffer();

    /// Try to find a cached surface view in the given CPU address
    [[nodiscard]] SurfaceView* TryFindFramebufferImageView(PAddr cpu_addr);

    /// Return true when a CPU region is modified from the GPU
    [[nodiscard]] bool IsRegionGPUInvalidated(PAddr addr, size_t size);

private:
    /// Iterate over all page indices in a range
    template <typename Func>
    static void ForEachPage(PAddr addr, size_t size, Func&& func) {
        static constexpr bool RETURNS_BOOL = std::is_same_v<std::invoke_result<Func, u64>, bool>;
        const u64 page_start = addr >> Memory::PAGE_BITS;
        const u64 page_end = (addr + size - 1) >> Memory::PAGE_BITS;

        for (u64 page = page_start; page <= page_end; ++page) {
            if constexpr (RETURNS_BOOL) {
                if (func(page)) {
                    break;
                }
            } else {
                func(page);
            }
        }
    }

    /// Iterates over all the images in a region calling func
    template <typename Func>
    void ForEachSurfaceInRegion(PAddr cpu_addr, size_t size, Func&& func) {
        using FuncReturn = typename std::invoke_result_t<Func, SurfaceId, Surface&>;
        static constexpr bool BOOL_BREAK = std::is_same_v<FuncReturn, bool>;
        std::vector<SurfaceId> picked_surfaces;

        ForEachPage(cpu_addr, size, [this, &picked_surfaces, cpu_addr, size, func](u64 page) {
            const auto it = page_table.find(page);
            if (it == page_table.end()) {
                if constexpr (BOOL_BREAK) {
                    return false;
                } else {
                    return;
                }
            }

            for (const SurfaceId image_id : it->second) {
                Surface& surface = slot_images[image_id];

                if (True(surface.flags & SurfaceFlagBits::Picked)) {
                    continue;
                }
                if (!surface.Overlaps(cpu_addr, size)) {
                    continue;
                }

                surface.flags |= SurfaceFlagBits::Picked;
                picked_surfaces.push_back(image_id);

                if constexpr (BOOL_BREAK) {
                    if (func(image_id, surface)) {
                        return true;
                    }

                } else {
                    func(image_id, surface);
                }
            }

            if constexpr (BOOL_BREAK) {
                return false;
            }
        });

        for (const SurfaceId surface_id : picked_surfaces) {
            slot_images[surface_id].flags &= ~SurfaceFlagBits::Picked;
        }
    }

    /// Find a surface in the address range that satisfies the condition function
    template <typename CondFunc>
    [[nodiscard]] SurfaceId FindSurface(const SurfaceInfo& info, CondFunc&& func) {
        SurfaceId image_id;
        ForEachSurfaceInRegion(info.addr, info.byte_size, [&](SurfaceId existing_image_id,
                                                            Surface& existing_surface) {
            if (func(info, existing_surface)) {
                image_id = existing_image_id;
                return true;
            }

            return false;
        });

        return image_id;
    }

    /// Find or create a surface from the given parameters
    template <typename CondFunc>
    [[nodiscard]] SurfaceId FindOrCreateSurface(const SurfaceInfo& info, CondFunc&& func) {
        if (const SurfaceId surface_id = FindSurface(info, func); surface_id) {
            return surface_id;
        }

        return CreateSurface(info);
    }

    /// Returns a pair consisting of a source and destination surfaces matching the config
    [[nodiscard]] std::pair<SurfaceId, SurfaceId> GetTransferSurfacePair(
            const GPU::Regs::DisplayTransferConfig& config);

    /// Find or create a framebuffer with the given render target parameters
    FramebufferId GetFramebufferId(const RenderTargets& key);

    /// Refresh the contents (pixel data) of an image
    void RefreshContents(Surface& image);

    /// Upload data from guest to an image
    void UploadImageContents(Surface& image, auto& map, u32 buffer_offset);

    /// Find or create an image view from a guest descriptor
    [[nodiscard]] SurfaceViewId FindImageView(const Pica::TexturingRegs::FullTextureConfig& config);

    /// Create a new image view from a guest descriptor
    [[nodiscard]] SurfaceViewId CreateImageView(const Pica::TexturingRegs::FullTextureConfig& config);

    /// Creates a surface from the given parameters
    [[nodiscard]] SurfaceId CreateSurface(const SurfaceInfo& info);

    /// Attempts to create a new surface by "stitching" existing surfaces
    [[nodiscard]] SurfaceId StitchSurface(const SurfaceInfo& info);

    /// Find or create an image view for the given color buffer index
    [[nodiscard]] SurfaceViewId FindColorBuffer(bool is_clear);

    /// Find or create an image view for the depth buffer
    [[nodiscard]] SurfaceViewId FindDepthBuffer(bool is_clear);

    /// Find or create a view for a render target with the given image parameters
    [[nodiscard]] SurfaceViewId FindRenderTargetView(const SurfaceInfo& info, bool is_clear);

    /// Find or create an image view in the given image with the passed parameters
    [[nodiscard]] SurfaceViewId FindOrEmplaceImageView(SurfaceId image_id, const SurfaceViewInfo& info);

    /// Register image in the page table
    void RegisterImage(SurfaceId image);

    /// Unregister image from the page table
    void UnregisterImage(SurfaceId image);

    /// Track CPU reads and writes for image
    void TrackImage(Surface& image);

    /// Stop tracking CPU reads and writes for image
    void UntrackSurface(Surface& image);

    /// Delete image from the cache
    void DeleteImage(SurfaceId image);

    /// Remove image views references from the cache
    void RemoveImageViewReferences(std::span<const SurfaceViewId> removed_views);

    /// Remove framebuffers using the given image views from the cache
    void RemoveFramebuffers(std::span<const SurfaceViewId> removed_views);

    /// Mark an image as modified from the GPU
    void MarkModification(Surface& image) noexcept;

    /// Synchronize image aliases, copying data if needed
    void SynchronizeAliases(SurfaceId image_id);

    /// Prepare an image to be used
    void PrepareImage(SurfaceId image_id, bool is_modification, bool invalidate);

    /// Prepare an image view to be used
    void PrepareImageView(SurfaceViewId image_view_id, bool is_modification, bool invalidate);

    /// Execute copies from one image to the other, even if they are incompatible
    void CopyImage(SurfaceId dst_id, SurfaceId src_id, std::span<const SurfaceCopy> copies);

    /// Create a render target from a given image and image view parameters
    [[nodiscard]] std::pair<FramebufferId, SurfaceViewId> RenderTargetFromImage(
        SurfaceId, const SurfaceViewInfo& view_info);

    /// Returns true if the current clear parameters clear the whole image of a given image view
    [[nodiscard]] bool IsFullClear(SurfaceViewId id);

private:
    Runtime& runtime;
    RasterizerAccelerated& rasterizer;

    RenderTargets render_targets;
    DirtyFlags dirty_flags;

    std::unordered_map<Pica::Texture::TextureInfo, SurfaceViewId> image_views;
    std::unordered_map<RenderTargets, FramebufferId> framebuffers;
    std::unordered_map<u64, std::vector<SurfaceId>, IdentityHash<u64>> page_table;

    bool has_deleted_images = false;

    SlotVector<Surface> slot_images;
    SlotVector<SurfaceView> slot_image_views;
    SlotVector<SurfaceAlloc> slot_image_allocs;
    SlotVector<Framebuffer> slot_framebuffers;

    std::vector<SurfaceId> uncommitted_downloads;
    std::queue<std::vector<SurfaceId>> committed_downloads;

    static constexpr size_t TICKS_TO_DESTROY = 6;
    DelayedDestructionRing<Surface, TICKS_TO_DESTROY> sentenced_images;
    DelayedDestructionRing<SurfaceView, TICKS_TO_DESTROY> sentenced_image_view;
    DelayedDestructionRing<Framebuffer, TICKS_TO_DESTROY> sentenced_framebuffers;

    std::unordered_map<PAddr, SurfaceAllocId> image_allocs_table;

    u64 modification_tick = 0;
    u64 frame_tick = 0;
};

template <class P>
RasterizerCache<P>::RasterizerCache(Runtime& runtime, RasterizerAccelerated& rasterizer)
    : runtime(runtime), rasterizer(rasterizer) {
    // Make sure the first index is reserved for the null resources
    // This way the null resource becomes a compile time constant
    slot_image_views.insert(runtime, NullSurfaceParams{});
}

template <class P>
void RasterizerCache<P>::TickFrame() {
    // Tick sentenced resources in this order to ensure they are destroyed in the right order
    sentenced_images.Tick();
    sentenced_framebuffers.Tick();
    sentenced_image_view.Tick();
    frame_tick++;
}

template <class P>
const typename P::SurfaceView& RasterizerCache<P>::GetSurfaceView(SurfaceViewId id) const noexcept {
    return slot_image_views[id];
}

template <class P>
typename P::SurfaceView& RasterizerCache<P>::GetSurfaceView(SurfaceViewId id) noexcept {
    return slot_image_views[id];
}

template <class P>
void RasterizerCache<P>::UpdateRenderTargets(bool is_clear) {
    if (False(dirty_flags & DirtyFlags::RenderTargets)) {
        return;
    }

    dirty_flags &= ~DirtyFlags::RenderTargets;

    const auto BindView = [this, &is_clear](SurfaceViewId& id, DirtyFlags flag) {
        if (True(dirty_flags & flag)) {
            dirty_flags &= ~flag;
            id = FindColorBuffer(is_clear);
        }

        const SurfaceView& image_view = slot_image_views[id];
        PrepareImage(image_view.image_id, true, is_clear && IsFullClear(id));
    };

    // Update color buffer
    if (True(dirty_flags & DirtyFlags::ColorBuffer)) {
        dirty_flags &= ~DirtyFlags::ColorBuffer;
        render_targets.color_buffer_id = FindColorBuffer(is_clear);
    }

    const SurfaceId color_id = render_targets.color_buffer_id;
    const SurfaceView& color_surface_view = slot_image_views[color_id];
    PrepareImage(color_surface_view.image_id, true, is_clear && IsFullClear(color_id));

    // Update depth buffer
    if (True(dirty_flags & DirtyFlags::DepthBuffer)) {
        dirty_flags &= ~DirtyFlags::DepthBuffer;
        render_targets.depth_buffer_id = FindDepthBuffer(is_clear);
    }

    const SurfaceId depth_id = render_targets.depth_buffer_id;
    const SurfaceView& depth_surface_view = slot_image_views[depth_id];
    PrepareImage(depth_surface_view.image_id, true, is_clear && IsFullClear(depth_id));
}

template <class P>
typename P::Framebuffer* RasterizerCache<P>::GetFramebuffer() {
    return &slot_framebuffers[GetFramebufferId(render_targets)];
}

template <class P>
FramebufferId RasterizerCache<P>::GetFramebufferId(const RenderTargets& key) {
    const auto [pair, is_new] = framebuffers.try_emplace(key);
    FramebufferId& framebuffer_id = pair->second;

    if (!is_new) {
        return framebuffer_id;
    }

    const SurfaceView* color_buffer =
        key.color_buffer_id ? &slot_image_views[key.color_buffer_id] : nullptr;
    const SurfaceView* depth_buffer =
        key.depth_buffer_id ? &slot_image_views[key.depth_buffer_id] : nullptr;

    framebuffer_id = slot_framebuffers.insert(runtime, color_buffer, depth_buffer, key);
    return framebuffer_id;
}

template <class P>
void RasterizerCache<P>::WriteMemory(PAddr cpu_addr, size_t size) {
    ForEachSurfaceInRegion(cpu_addr, size, [this](SurfaceId surface_id, Surface& surface) {
        if (True(surface.flags & SurfaceFlagBits::CPUInvalidated)) {
            return;
        }

        surface.flags |= SurfaceFlagBits::CPUInvalidated;
        UntrackSurface(surface);
    });
}

template <class P>
void RasterizerCache<P>::DownloadMemory(PAddr cpu_addr, size_t size) {
    std::vector<SurfaceId> download_surfaces;
    ForEachSurfaceInRegion(cpu_addr, size, [this, &download_surfaces](SurfaceId surface_id, Surface& surface) {
        // Skip surfaces that were not modified from the GPU
        if (False(surface.flags & SurfaceFlagBits::GPUInvalidated)) {
            return;
        }

        // Don't download surfaces that the CPU has modified on the guest.
        // We don't want to override anything the CPU has written there
        if (True(surface.flags & SurfaceFlagBits::CPUInvalidated)) {
            return;
        }

        surface.flags &= ~SurfaceFlagBits::GPUInvalidated;
        download_surfaces.push_back(surface_id);
    });

    if (download_surfaces.empty()) {
        return;
    }

    // Sort images from oldest to newest
    std::ranges::sort(download_surfaces, [this](SurfaceId lhs, SurfaceId rhs) {
        return slot_images[lhs].modification_tick < slot_images[rhs].modification_tick;
    });


    // TODO: Batch download
    for (const SurfaceId surface_id : download_surfaces) {
        const Surface& surface = slot_images[surface_id];
        auto staging_buffer = runtime.MapDownloadBuffer(surface.info.byte_size);

        // TODO: Download only what is needed
        const BufferSurfaceCopy download_copy = {
            .buffer_offset = 0,
            .buffer_size = surface.info.byte_size,
            .buffer_row_length = surface.info.size.width,
            .buffer_image_height = surface.info.size.height,
            .texture_level = 0,
            .texture_offset = Offset{0, 0},
            .texture_extent = Extent{surface.info.size.width, surface.info.size.width}
        };

        runtime.DownloadMemory(surface, staging_buffer, download_copy);
        //SwizzleImage(gpu_memory, image.gpu_addr, image.info, copies, map.Span());
    }
}

template <class P>
void RasterizerCache<P>::UnmapMemory(PAddr cpu_addr, size_t size) {
    std::vector<SurfaceId> deleted_images;
    ForEachSurfaceInRegion(cpu_addr, size, [&](SurfaceId id, Surface&) {
        deleted_images.push_back(id);
    });

    for (const SurfaceId id : deleted_images) {
        Surface& image = slot_images[id];
        if (True(image.flags & SurfaceFlagBits::Tracked)) {
            UntrackSurface(image);
        }

        UnregisterImage(id);
        DeleteImage(id);
    }
}

template <class P>
bool RasterizerCache<P>::TextureCopy(const GPU::Regs::DisplayTransferConfig& config) {
    // Transfers must be 16-byte aligned
    u32 copy_size = Common::AlignDown(config.texture_copy.size, 16);
    if (copy_size == 0) {
        return false;
    }

    // Helper function used to perform some sanity checks on the copy operation
    const auto SanityCheck = [&copy_size](u32& gap, u32& width) {
        if (width == 0 && gap != 0) {
            return false;
        }

        if (gap == 0 || width >= copy_size) {
            width = copy_size;
            gap = 0;
        }

        if (copy_size % width != 0) {
            return false;
        }

        return true;
    };

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 input_width = config.texture_copy.input_width * 16;
    if (!SanityCheck(input_gap, input_width)) {
        return false;
    }

    u32 output_gap = config.texture_copy.output_gap * 16;
    u32 output_width = config.texture_copy.output_width * 16;
    if (!SanityCheck(output_gap, output_width)) {
        return false;
    }

    // Find the source surface
    const u32 src_width = input_width + input_gap;
    const u32 src_height = copy_size / input_width;
    const SurfaceInfo src_info = {
        .addr = config.GetPhysicalInputAddress(),
        .byte_size = (src_height - 1) * src_width + input_width,
        .real_size = Extent{input_width, src_height},
        .size = Extent{src_width, src_height},
    };

    SurfaceId src_id = FindOrCreateSurface(src_info, CanTexCopy);
    const Surface& src_surface = slot_images[src_id];

    // Find the destination surface
    const u32 dst_width = PixelsInBytes(src_surface.info.is_tiled ? output_gap / 8 : output_gap);
    const u32 dst_height = copy_size / output_width;
    const SurfaceInfo dst_info = {
        .addr = config.GetPhysicalOutputAddress(),
        .byte_size = src_info.byte_size,
        .real_size = Extent{output_width, dst_height},
        .size = Extent{dst_width, dst_height}
    };

    SurfaceId dst_id = FindOrCreateSurface(dst_info, CanTexCopy);
    const Surface& dst_surface = slot_images[dst_id];

    // Mark the destination surface as GPU invalidated
    PrepareImage(dst_id, true, false);

    return runtime.BlitSurfaces(src_surface, src_rect, dst_surface, dst_rect);
}

template <class P>
bool RasterizerCache<P>::DisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) {
    using ScalingMode = GPU::Regs::DisplayTransferConfig::ScalingMode;

    const SurfaceInfo src_info = {
        .format = PixelFormatFromGPUPixelFormat(config.input_format),
        .size = Extent{config.input_width, config.input_height},
        .is_tiled = !config.input_linear,
        .real_size = Extent{config.output_width, config.output_height}
    };

    // Downscale if needed
    const u32 dst_width =
            config.scaling != ScalingMode::NoScale ? config.output_width.Value() / 2
                                                   : config.output_width.Value();
    const u32 dst_height =
            config.scaling == ScalingMode::ScaleXY ? config.output_height.Value() / 2
                                                   : config.output_height.Value();

    const SurfaceInfo dst_info = {
        .format = PixelFormatFromGPUPixelFormat(config.output_format),
        .size = Extent{dst_width, dst_height},
        .is_tiled = config.input_linear != config.dont_swizzle,
        .real_size = Extent{dst_width, dst_height}
    };

    // Search the cache for any surfaces that can be used for the transfer
    SurfaceId src_id, dst_id;
    const PAddr src_addr = config.GetPhysicalInputAddress();
    const PAddr dst_addr = config.GetPhysicalOutputAddress();

    do {
        has_deleted_images = false;

        src_id = FindSurface(src_info, src_addr);
        dst_id = FindSurface(dst_info, dst_addr);

        if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
            continue;
        }

        if (!dst_id) {
            dst_id = CreateSurface(dst_info, dst_addr);
        }

        if (!src_id) {
            src_id = CreateSurface(src_info, src_addr);
        }
    } while (has_deleted_images);

    // Mark the surfaces as GPU invalidated
    const Surface& src_surface = slot_images[src_id];
    PrepareImage(src_id, false, false);

    const Surface& dst_surface = slot_images[dst_id];
    PrepareImage(dst_id, true, false);

    const s32 real_width = src_info.real_size.width;
    const s32 real_height = src_info.real_size.height;
    const std::array offsets = {
       Offset{.x = 0, .y = 0},
       Offset{.x = real_width, .y = real_height},
    };

    if (config.flip_vertically) {
        std::swap(offsets[1].y, offsets[0].y);
    }

    return runtime.BlitSurfaces(src_surface, dst_surface, offsets);
}

template <class P>
void RasterizerCache<P>::InvalidateColorBuffer() {
    SurfaceViewId& color_buffer_id = render_targets.color_buffer_id;
    color_buffer_id = FindColorBuffer(false);
    if (!color_buffer_id) {
        LOG_ERROR(HW_GPU, "Invalidating invalid color buffer!");
        return;
    }

    // When invalidating a color buffer, the old contents are no longer relevant
    SurfaceView& color_buffer = slot_image_views[color_buffer_id];
    Surface& image = slot_images[color_buffer.image_id];
    image.flags &= ~SurfaceFlagBits::CPUInvalidated;
    image.flags &= ~SurfaceFlagBits::GPUInvalidated;

    runtime.InvalidateColorBuffer(color_buffer);
}

template <class P>
void RasterizerCache<P>::InvalidateDepthBuffer() {
    SurfaceViewId& depth_buffer_id = render_targets.depth_buffer_id;
    depth_buffer_id = FindDepthBuffer(false);
    if (!depth_buffer_id) {
        LOG_ERROR(HW_GPU, "Invalidating invalid depth buffer");
        return;
    }

    // When invalidating the depth buffer, the old contents are no longer relevant
    Surface& image = slot_images[slot_image_views[depth_buffer_id].image_id];
    image.flags &= ~SurfaceFlagBits::CPUInvalidated;
    image.flags &= ~SurfaceFlagBits::GPUInvalidated;

    SurfaceView& depth_buffer = slot_image_views[depth_buffer_id];
    runtime.InvalidateDepthBuffer(depth_buffer);
}

template <class P>
typename P::SurfaceView* RasterizerCache<P>::TryFindFramebufferImageView(PAddr addr) {
    const auto it = page_table.find(addr >> Memory::PAGE_BITS);
    if (it == page_table.end()) {
        return nullptr;
    }

    for (const SurfaceId& surface_id : it->second) {
        const Surface& surface = slot_images[surface_id];
        if (surface.info.addr != addr) {
            continue;
        }

        if (surface.surface_view_ids.empty()) {
            continue;
        }

        return &slot_image_views[surface.surface_view_ids.at(0)];
    }

    return nullptr;
}

template <class P>
bool RasterizerCache<P>::IsRegionGPUInvalidated(PAddr addr, size_t size) {
    bool is_modified = false;
    ForEachSurfaceInRegion(addr, size, [&is_modified](SurfaceId, Surface& image) {
        if (False(image.flags & SurfaceFlagBits::GPUInvalidated)) {
            return false;
        }

        is_modified = true;
        return true;
    });

    return is_modified;
}

template <class P>
void RasterizerCache<P>::RefreshContents(Surface& image) {
    if (False(image.flags & SurfaceFlagBits::CPUInvalidated)) {
        // Only upload modified images
        return;
    }

    image.flags &= ~SurfaceFlagBits::CPUInvalidated;
    TrackImage(image);

    auto map = runtime.MapUploadBuffer(image.info.byte_size);
    UploadImageContents(image, map, 0);
    runtime.InsertUploadMemoryBarrier();
}

template <class P>
template <typename MapBuffer>
void RasterizerCache<P>::UploadImageContents(Surface& image, MapBuffer& map, size_t buffer_offset) {
    const std::span<u8> mapped_span = map.Span().subspan(buffer_offset);
    const PAddr gpu_addr = image.gpu_addr;

    if (True(image.flags & SurfaceFlagBits::AcceleratedUpload)) {
        gpu_memory.ReadBlockUnsafe(gpu_addr, mapped_span.data(), mapped_span.size_bytes());
        const auto uploads = FullUploadSwizzles(image.info);
        runtime.AccelerateImageUpload(image, map, buffer_offset, uploads);
    } else if (True(image.flags & SurfaceFlagBits::Converted)) {
        std::vector<u8> unswizzled_data(image.unswizzled_size_bytes);
        auto copies = UnswizzleImage(gpu_memory, gpu_addr, image.info, unswizzled_data);
        ConvertImage(unswizzled_data, image.info, mapped_span, copies);
        image.UploadMemory(map, buffer_offset, copies);
    } else if (image.info.type == ImageType::Buffer) {
        const std::array copies{UploadBufferCopy(gpu_memory, gpu_addr, image, mapped_span)};
        image.UploadMemory(map, buffer_offset, copies);
    } else {
        const auto copies = UnswizzleImage(gpu_memory, gpu_addr, image.info, mapped_span);
        image.UploadMemory(map, buffer_offset, copies);
    }
}

template <class P>
SurfaceViewId RasterizerCache<P>::FindImageView(const Pica::TexturingRegs::FullTextureConfig& config) {
    if (!IsValidAddress(gpu_memory, config)) {
        return NULL_IMAGE_VIEW_ID;
    }

    const auto [pair, is_new] = image_views.try_emplace(config);
    SurfaceViewId& image_view_id = pair->second;
    if (is_new) {
        image_view_id = CreateImageView(config);
    }

    return image_view_id;
}

template <class P>
SurfaceViewId RasterizerCache<P>::CreateImageView(const Pica::TexturingRegs::FullTextureConfig& config) {
    const SurfaceInfo info(config);
    const PAddr image_gpu_addr = config.Address() - config.BaseLayer() * info.layer_stride;
    const SurfaceId image_id = FindOrCreateSurface(info, image_gpu_addr);

    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }

    Surface& image = slot_images[image_id];
    const SubresourceBase base = image.TryFindBase(config.Address()).value();
    ASSERT(base.level == 0);
    const SurfaceViewInfo view_info(config, base.layer);
    const SurfaceViewId image_view_id = FindOrEmplaceImageView(image_id, view_info);

    SurfaceView& image_view = slot_image_views[image_view_id];
    image_view.flags |= SurfaceViewFlagBits::Strong;
    image.flags |= SurfaceFlagBits::Strong;
    return image_view_id;
}

template <class P>
SurfaceId RasterizerCache<P>::CreateSurface(const SurfaceInfo& info, PAddr base_addr) {
    const SurfaceId image_id = StitchSurface(info, base_addr);
    const Surface& image = slot_images[image_id];

    const auto [it, is_new] = image_allocs_table.try_emplace(image.addr);
    if (is_new) {
        it->second = slot_image_allocs.insert();
    }

    slot_image_allocs[it->second].images.push_back(image_id);
    return image_id;
}

template <class P>
SurfaceId RasterizerCache<P>::StitchSurface(const SurfaceInfo& info, PAddr base_addr) {
    SurfaceInfo new_info = info;
    const u32 surface_size = CalculateSurfaceSize(new_info);
    std::vector<SurfaceId> overlap_ids;

    ForEachSurfaceInRegion(base_addr, surface_size, [&](SurfaceId overlap_id, Surface& overlap) {
        const auto solution = ResolveOverlap(new_info, base_addr, overlap, true);
        if (solution) {
            base_addr = solution->
            new_info.resources = solution->resources;
            overlap_ids.push_back(overlap_id);
            return;
        }

        static constexpr auto options = RelaxedOptions::Size | RelaxedOptions::Format;
        const Surface new_image{new_info, base_addr};

        if (IsSubresource(new_info, overlap, base_addr, options)) {
            left_aliased_ids.push_back(overlap_id);
        } else if (IsSubresource(overlap.info, new_image_base, overlap.gpu_addr, options)) {
            right_aliased_ids.push_back(overlap_id);
        }
    });

    const SurfaceId new_image_id = slot_images.insert(runtime, new_info, gpu_addr, cpu_addr);
    Surface& new_image = slot_images[new_image_id];

    RefreshContents(new_image);

    for (const SurfaceId overlap_id : overlap_ids) {
        Surface& overlap = slot_images[overlap_id];
        const SubresourceBase base = new_image.TryFindBase(overlap.gpu_addr).value();
        const auto copies = MakeShrinkImageCopies(new_info, overlap.info, base);
        runtime.CopyImage(new_image, overlap, copies);

        if (True(overlap.flags & SurfaceFlagBits::Tracked)) {
            UntrackSurface(overlap);
        }

        UnregisterImage(overlap_id);
        DeleteImage(overlap_id);
    }

    Surface& new_image_base = new_image;
    for (const SurfaceId aliased_id : right_aliased_ids) {
        Surface& aliased = slot_images[aliased_id];
        AddImageAlias(new_image_base, aliased, new_image_id, aliased_id);
    }

    for (const SurfaceId aliased_id : left_aliased_ids) {
        Surface& aliased = slot_images[aliased_id];
        AddImageAlias(aliased, new_image_base, aliased_id, new_image_id);
    }

    RegisterImage(new_image_id);
    return new_image_id;
}

template <class P>
std::pair<SurfaceId, SurfaceId> RasterizerCache<P>::GetTransferSurfacePair(
        const GPU::Regs::DisplayTransferConfig& config) {

    static constexpr auto FIND_OPTIONS = RelaxedOptions::Format | RelaxedOptions::Samples;
    const PAddr dst_addr = config.GetPhysicalInputAddress();
    const PAddr src_addr = config.GetPhysicalOutputAddress();
    const auto [src_info, dst_info] = MakeSurfaceInfosFromTransferConfig(config);

    SurfaceId dst_id;
    SurfaceId src_id;

    do {
        has_deleted_images = false;
        dst_id = FindSurface(dst_info, dst_addr, FIND_OPTIONS);
        src_id = FindSurface(src_info, src_addr, FIND_OPTIONS);
        const Surface* const dst_image = dst_id ? &slot_images[dst_id] : nullptr;
        const Surface* const src_image = src_id ? &slot_images[src_id] : nullptr;

        if (GetFormatType(dst_info.format) != GetFormatType(src_info.format)) {
            continue;
        }

        if (!dst_id) {
            dst_id = InsertImage(dst_info, dst_addr, RelaxedOptions{});
        }

        if (!src_id) {
            src_id = InsertImage(src_info, src_addr, RelaxedOptions{});
        }
    } while (has_deleted_images);

    return CopyImages{
        .dst_id = dst_id,
        .src_id = src_id,
        .dst_format = dst_info.format,
        .src_format = src_info.format,
    };
}

template <class P>
SurfaceViewId RasterizerCache<P>::FindColorBuffer(bool is_clear) {
    const auto& regs = Pica::g_state.regs;

    const bool shadow_rendering = regs.framebuffer.output_merger.fragment_operation_mode ==
                                  Pica::FramebufferRegs::FragmentOperationMode::Shadow;

    auto IsColorWriteEnabled = [&](u32 value) -> bool {
        return regs.framebuffer.framebuffer.allow_color_write != 0 && value != 0;
    };

    const bool color_writes = IsColorWriteEnabled(regs.framebuffer.output_merger.red_enable) ||
            IsColorWriteEnabled(regs.framebuffer.output_merger.green_enable) ||
            IsColorWriteEnabled(regs.framebuffer.output_merger.blue_enable) ||
            IsColorWriteEnabled(regs.framebuffer.output_merger.alpha_enable);

    const bool using_color_fb = shadow_rendering || color_writes;

    if (!using_color_fb) {
        return SurfaceViewId{};
    }

    const PAddr color_addr = regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress();
    if (color_addr == 0) {
        return SurfaceViewId{};
    }

    const SurfaceInfo info{regs};
    return FindRenderTargetView(info, color_addr, is_clear);
}

template <class P>
SurfaceViewId RasterizerCache<P>::FindDepthBuffer(bool is_clear) {
    const auto& regs = Pica::g_state.regs;

    const bool has_stencil =
        regs.framebuffer.framebuffer.depth_format == Pica::FramebufferRegs::DepthFormat::D24S8;

    const bool stencil_test_enabled = regs.framebuffer.output_merger.stencil_test.enable;

    const u32 stencil_write_mask = (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<u32>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;

    const bool depth_test_enabled = regs.framebuffer.output_merger.depth_test_enable == 1 ||
                                    regs.framebuffer.output_merger.depth_write_enable == 1;

    const bool depth_write_mask = (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                                   regs.framebuffer.output_merger.depth_write_enable);

    const bool write_depth_fb =
        (depth_test_enabled && depth_write_mask) ||
        (stencil_test_enabled && stencil_write_mask != 0);

    const bool shadow_rendering = regs.framebuffer.output_merger.fragment_operation_mode ==
                                  Pica::FramebufferRegs::FragmentOperationMode::Shadow;

    const bool using_depth_fb =
        !shadow_rendering && (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
        (has_stencil && stencil_test_enabled));

    if (!using_depth_fb) {
        return SurfaceViewId{};
    }

    const PAddr depth_addr = regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress();
    if (depth_addr == 0) {
        return SurfaceViewId{};
    }

    const SurfaceInfo info{regs};
    return FindRenderTargetView(info, depth_addr, is_clear);
}

template <class P>
SurfaceViewId RasterizerCache<P>::FindRenderTargetView(const SurfaceInfo& info, PAddr target_addr,
                                                  bool is_clear) {
    const auto options = is_clear ? RelaxedOptions::Samples : RelaxedOptions{};
    const SurfaceId image_id = FindOrCreateSurface(info, target_addr, options);
    if (!image_id) {
        return NULL_IMAGE_VIEW_ID;
    }

    Surface& image = slot_images[image_id];
    const ImageViewType view_type = RenderTarGetSurfaceViewType(info);
    SubresourceBase base = SubresourceBase{.level = 0, .layer = 0};

    const SubresourceRange range{
        .base = base,
        .extent = {.levels = 1, .layers = 1},
    };

    return FindOrEmplaceImageView(image_id, SurfaceViewInfo{view_type, info.format, range});
}

template <class P>
SurfaceViewId RasterizerCache<P>::FindOrEmplaceImageView(SurfaceId image_id, const SurfaceViewInfo& info) {
    Surface& image = slot_images[image_id];
    if (const SurfaceViewId image_view_id = image.FindView(info); image_view_id) {
        return image_view_id;
    }

    const SurfaceViewId image_view_id = slot_image_views.insert(runtime, info, image_id, image);
    image.InsertView(info, image_view_id);
    return image_view_id;
}

template <class P>
void RasterizerCache<P>::RegisterImage(SurfaceId image_id) {
    Surface& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & SurfaceFlagBits::Registered),
               "Trying to register an already registered image");

    image.flags |= SurfaceFlagBits::Registered;
    ForEachPage(image.cpu_addr, image.guest_size_bytes,
                [this, image_id](u64 page) { page_table[page].push_back(image_id); });
}

template <class P>
void RasterizerCache<P>::UnregisterImage(SurfaceId image_id) {
    Surface& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & SurfaceFlagBits::Registered),
               "Trying to unregister an already registered image");
    image.flags &= ~SurfaceFlagBits::Registered;

    ForEachPage(image.cpu_addr, image.guest_size_bytes, [this, image_id](u64 page) {
        const auto page_it = page_table.find(page);
        if (page_it == page_table.end()) {
            UNREACHABLE_MSG("Unregistering unregistered page=0x{:x}", page << Memory::PAGE_BITS);
            return;
        }

        std::vector<SurfaceId>& image_ids = page_it->second;
        const auto vector_it = std::ranges::find(image_ids, image_id);
        if (vector_it == image_ids.end()) {
            UNREACHABLE_MSG("Unregistering unregistered image in page=0x{:x}", page << Memory::PAGE_BITS);
            return;
        }

        image_ids.erase(vector_it);
    });
}

template <class P>
void RasterizerCache<P>::TrackImage(Surface& image) {
    ASSERT(False(image.flags & SurfaceFlagBits::Tracked));
    image.flags |= SurfaceFlagBits::Tracked;
    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, 1);
}

template <class P>
void RasterizerCache<P>::UntrackSurface(Surface& image) {
    ASSERT(True(image.flags & SurfaceFlagBits::Tracked));
    image.flags &= ~SurfaceFlagBits::Tracked;
    rasterizer.UpdatePagesCachedCount(image.cpu_addr, image.guest_size_bytes, -1);
}

template <class P>
void RasterizerCache<P>::DeleteImage(SurfaceId image_id) {
    Surface& image = slot_images[image_id];
    const PAddr gpu_addr = image.gpu_addr;
    const auto alloc_it = image_allocs_table.find(gpu_addr);
    if (alloc_it == image_allocs_table.end()) {
        UNREACHABLE_MSG("Trying to delete an image alloc that does not exist in address 0x{:x}",
                        gpu_addr);
        return;
    }
    const ImageAllocId alloc_id = alloc_it->second;
    std::vector<SurfaceId>& alloc_images = slot_image_allocs[alloc_id].images;
    const auto alloc_image_it = std::ranges::find(alloc_images, image_id);
    if (alloc_image_it == alloc_images.end()) {
        UNREACHABLE_MSG("Trying to delete an image that does not exist");
        return;
    }
    ASSERT_MSG(False(image.flags & SurfaceFlagBits::Tracked), "Image was not untracked");
    ASSERT_MSG(False(image.flags & SurfaceFlagBits::Registered), "Image was not unregistered");

    // Mark render targets as dirty
    dirty_flags |= DirtyFlags::RenderTargets;
    dirty_flags |= DirtyFlags::ColorBuffer;
    dirty_flags |= DirtyFlags::DepthBuffer;

    // Check if any view has been bound as a render target and unbind it
    const std::span<const SurfaceViewId> image_view_ids = image.image_view_ids;
    for (const SurfaceViewId image_view_id : image_view_ids) {
        if (render_targets.color_buffer_id == image_view_id) {
            render_targets.color_buffer_id = SurfaceViewId{};
        }

        if (render_targets.depth_buffer_id == image_view_id) {
            render_targets.depth_buffer_id = SurfaceViewId{};
        }
    }

    RemoveImageViewReferences(image_view_ids);
    RemoveFramebuffers(image_view_ids);

    // Iterate over all aliased images and remove any references to the to-be-deleted image
    for (const AliasedSurface& alias : image.aliased_images) {
        Surface& other_image = slot_images[alias.id];
        [[maybe_unused]] const size_t num_removed_aliases =
            std::erase_if(other_image.aliased_images, [image_id](const AliasedSurface& other_alias) {
                return other_alias.id == image_id;
            });

        ASSERT_MSG(num_removed_aliases == 1, "Invalid number of removed aliases: {}",
                   num_removed_aliases);
    }

    for (const SurfaceViewId image_view_id : image_view_ids) {
        sentenced_image_view.Push(std::move(slot_image_views[image_view_id]));
        slot_image_views.erase(image_view_id);
    }

    sentenced_images.Push(std::move(slot_images[image_id]));
    slot_images.erase(image_id);

    alloc_images.erase(alloc_image_it);
    if (alloc_images.empty()) {
        image_allocs_table.erase(alloc_it);
    }

    has_deleted_images = true;
}

template <class P>
void RasterizerCache<P>::RemoveImageViewReferences(std::span<const SurfaceViewId> removed_views) {
    auto it = image_views.begin();
    while (it != image_views.end()) {
        const auto found = std::ranges::find(removed_views, it->second);
        if (found != removed_views.end()) {
            it = image_views.erase(it);
        } else {
            ++it;
        }
    }
}

template <class P>
void RasterizerCache<P>::RemoveFramebuffers(std::span<const SurfaceViewId> removed_views) {
    auto it = framebuffers.begin();
    while (it != framebuffers.end()) {
        if (it->first.Contains(removed_views)) {
            it = framebuffers.erase(it);
        } else {
            ++it;
        }
    }
}

template <class P>
void RasterizerCache<P>::MarkModification(Surface& image) noexcept {
    image.flags |= SurfaceFlagBits::GPUInvalidated;
    image.modification_tick = ++modification_tick;
}

template <class P>
void RasterizerCache<P>::SynchronizeAliases(SurfaceId image_id) {
    std::vector<const AliasedImage*> aliased_images;
    Surface& image = slot_images[image_id];
    u64 most_recent_tick = image.modification_tick;

    for (const AliasedSurface& aliased : image.aliased_images) {
        Surface& aliased_image = slot_images[aliased.id];
        if (image.modification_tick < aliased_image.modification_tick) {
            most_recent_tick = std::max(most_recent_tick, aliased_image.modification_tick);
            aliased_images.push_back(&aliased);
        }
    }

    if (aliased_images.empty()) {
        return;
    }

    image.modification_tick = most_recent_tick;
    std::ranges::sort(aliased_images, [this](const AliasedImage* lhs, const AliasedImage* rhs) {
        const Surface& lhs_image = slot_images[lhs->id];
        const Surface& rhs_image = slot_images[rhs->id];
        return lhs_image.modification_tick < rhs_image.modification_tick;
    });

    for (const AliasedImage* const aliased : aliased_images) {
        CopyImage(image_id, aliased->id, aliased->copies);
    }
}

template <class P>
void RasterizerCache<P>::PrepareImage(SurfaceId image_id, bool is_modification, bool invalidate) {
    Surface& image = slot_images[image_id];
    if (invalidate) {
        image.flags &= ~(SurfaceFlagBits::CPUInvalidated | SurfaceFlagBits::GPUInvalidated);

        if (False(image.flags & SurfaceFlagBits::Tracked)) {
            TrackImage(image);
        }

    } else {
        RefreshContents(image);
    }

    if (is_modification) {
        MarkModification(image);
    }

    image.frame_tick = frame_tick;
}

template <class P>
void RasterizerCache<P>::CopyImage(SurfaceId dst_id, SurfaceId src_id, std::span<const SurfaceCopy> copies) {
    Surface& dst = slot_images[dst_id];
    Surface& src = slot_images[src_id];

    const auto dst_format_type = GetFormatType(dst.info.format);
    const auto src_format_type = GetFormatType(src.info.format);
    if (src_format_type == dst_format_type) {
        return runtime.CopyImage(dst, src, copies);
    }

    UNIMPLEMENTED_IF(dst.info.type != ImageType::e2D);
    UNIMPLEMENTED_IF(src.info.type != ImageType::e2D);

    for (const SurfaceCopy& copy : copies) {
        UNIMPLEMENTED_IF(copy.dst_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_subresource.num_layers != 1);
        UNIMPLEMENTED_IF(copy.src_offset != Offset{});
        UNIMPLEMENTED_IF(copy.dst_offset != Offset{});

        const SubresourceBase dst_base{
            .level = copy.dst_subresource.base_level,
            .layer = copy.dst_subresource.base_layer,
        };

        const SubresourceBase src_base{
            .level = copy.src_subresource.base_level,
            .layer = copy.src_subresource.base_layer,
        };

        const SubresourceExtent dst_extent{.levels = 1, .layers = 1};
        const SubresourceExtent src_extent{.levels = 1, .layers = 1};
        const SubresourceRange dst_range{.base = dst_base, .extent = dst_extent};
        const SubresourceRange src_range{.base = src_base, .extent = src_extent};
        const SurfaceViewInfo dst_view_info{ImageViewType::e2D, dst.info.format, dst_range};
        const SurfaceViewInfo src_view_info{ImageViewType::e2D, src.info.format, src_range};

        const auto [dst_framebuffer_id, dst_view_id] = RenderTargetFromImage(dst_id, dst_view_info);
        Framebuffer* const dst_framebuffer = &slot_framebuffers[dst_framebuffer_id];
        const SurfaceViewId src_view_id = FindOrEmplaceImageView(src_id, src_view_info);
        SurfaceView& dst_view = slot_image_views[dst_view_id];
        SurfaceView& src_view = slot_image_views[src_view_id];

        [[maybe_unused]] const Extent expected_size{
            .width = std::min(dst_view.size.width, src_view.size.width),
            .height = std::min(dst_view.size.height, src_view.size.height),
            .depth = std::min(dst_view.size.depth, src_view.size.depth),
        };

        UNIMPLEMENTED_IF(copy.extent != expected_size);
        runtime.ConvertImage(dst_framebuffer, dst_view, src_view);
    }
}

template <class P>
std::pair<FramebufferId, SurfaceViewId> RasterizerCache<P>::RenderTargetFromImage(
    SurfaceId image_id, const SurfaceViewInfo& view_info) {
    const SurfaceViewId view_id = FindOrEmplaceImageView(image_id, view_info);
    const Surface& image = slot_images[image_id];
    const bool is_color = GetFormatType(image.info.format) == SurfaceType::Color;
    const SurfaceViewId color_view_id = is_color ? view_id : SurfaceViewId{};
    const SurfaceViewId depth_view_id = is_color ? SurfaceViewId{} : view_id;
    const Extent extent = MipSize(image.info.size, view_info.range.base.level);
    const u32 num_samples = image.info.num_samples;
    const auto [samples_x, samples_y] = SamplesLog2(num_samples);
    const FramebufferId framebuffer_id = GetFramebufferId(RenderTargets{
        .color_buffer_ids = {color_view_id},
        .depth_buffer_id = depth_view_id,
        .size = {extent.width >> samples_x, extent.height >> samples_y},
    });

    return {framebuffer_id, view_id};
}

template <class P>
bool RasterizerCache<P>::IsFullClear(SurfaceViewId id) {
    if (!id) {
        return true;
    }

    const SurfaceView& image_view = slot_image_views[id];
    const Surface& image = slot_images[image_view.image_id];
    const Extent size = image_view.size;
    const auto& regs = Pica::g_state.regs;

    if (image.info.resources.levels > 1 || image.info.resources.layers > 1) {
        // Images with multiple resources can't be cleared in a single call
        return false;
    }

    // Make sure the clear covers all texels in the subresource
    auto& scissor = regs.rasterizer.scissor_test;
    return scissor.x1 == 0 && scissor.y1 == 0 && scissor.x2 >= size.width &&
           scissor.y2 >= size.height;
}

} // namespace VideoCommon
