// Copyright 2020 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <memory>
#include <string_view>
#include <vector>
#include "common/settings.h"
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_opengl/texture_filters/texture_filter_base.h"

namespace Settings {
enum class TextureFilter : u32;
}

namespace OpenGL {

class TextureFilterer {
public:
    static constexpr Settings::TextureFilter NONE = Settings::TextureFilter::Linear;

public:
    explicit TextureFilterer(Settings::TextureFilter filter, u16 scale_factor);

    // Returns true if the filter actually changed
    bool Reset(Settings::TextureFilter new_filter_name, u16 new_scale_factor);

    // Returns true if there is no active filter
    bool IsNull() const;

    // Returns true if the texture was able to be filtered
    bool Filter(const OGLTexture& src_tex, Common::Rectangle<u32> src_rect,
                const OGLTexture& dst_tex, Common::Rectangle<u32> dst_rect, SurfaceType type);

private:
    Settings::TextureFilter filter_name;
    std::unique_ptr<TextureFilterBase> filter;
};

} // namespace OpenGL
