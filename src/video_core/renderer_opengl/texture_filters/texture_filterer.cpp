/// Copyright 2020 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <algorithm>
#include <functional>
#include <unordered_map>
#include "common/logging/log.h"
#include "video_core/renderer_opengl/texture_filters/anime4k/anime4k_ultrafast.h"
#include "video_core/renderer_opengl/texture_filters/bicubic/bicubic.h"
#include "video_core/renderer_opengl/texture_filters/nearest_neighbor/nearest_neighbor.h"
#include "video_core/renderer_opengl/texture_filters/scale_force/scale_force.h"
#include "video_core/renderer_opengl/texture_filters/texture_filter_base.h"
#include "video_core/renderer_opengl/texture_filters/texture_filterer.h"
#include "video_core/renderer_opengl/texture_filters/xbrz/xbrz_freescale.h"

namespace OpenGL {

namespace {

using Settings::TextureFilter;
using TextureFilterContructor = std::function<std::unique_ptr<TextureFilterBase>(u16)>;

template <TextureFilter filter, typename T>
std::pair<TextureFilter, TextureFilterContructor> FilterMapPair() {
    return {filter, std::make_unique<T, u16>};
};

static const std::unordered_map<TextureFilter, TextureFilterContructor> filter_map{
    {TextureFilter::Linear, [](u16) { return nullptr; }},
    FilterMapPair<TextureFilter::Anime4K, Anime4kUltrafast>(),
    FilterMapPair<TextureFilter::Bicubic, Bicubic>(),
    FilterMapPair<TextureFilter::NearestNeighbor, NearestNeighbor>(),
    FilterMapPair<TextureFilter::ScaleForce, ScaleForce>(),
    FilterMapPair<TextureFilter::xBRZ, XbrzFreescale>(),
};

} // namespace

TextureFilterer::TextureFilterer(TextureFilter filter_name, u16 scale_factor) {
    Reset(filter_name, scale_factor);
}

bool TextureFilterer::Reset(TextureFilter new_filter_name, u16 new_scale_factor) {
    if (filter_name == new_filter_name && (IsNull() || filter->scale_factor == new_scale_factor))
        return false;

    auto iter = filter_map.find(new_filter_name);
    if (iter == filter_map.end()) {
        LOG_ERROR(Render_OpenGL, "Invalid texture filter: {}", new_filter_name);
        filter = nullptr;
        return true;
    }

    filter_name = iter->first;
    filter = iter->second(new_scale_factor);
    return true;
}

bool TextureFilterer::IsNull() const {
    return !filter;
}

bool TextureFilterer::Filter(const OGLTexture& src_tex, Common::Rectangle<u32> src_rect,
                             const OGLTexture& dst_tex, Common::Rectangle<u32> dst_rect,
                             SurfaceType type) {

    // Depth/Stencil texture filtering is not supported for now
    if (IsNull() || (type != SurfaceType::Color && type != SurfaceType::Texture)) {
        return false;
    }

    filter->Filter(src_tex, src_rect, dst_tex, dst_rect);
    return true;
}

} // namespace OpenGL
