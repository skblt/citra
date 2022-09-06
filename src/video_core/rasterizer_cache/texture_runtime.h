// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include "video_core/rasterizer_cache/types.h"
#include "video_core/renderer_opengl/gl_resource_manager.h"

namespace OpenGL {

struct FormatTuple;

/**
 * Provides texture manipulation functions to the rasterizer cache
 * Separating this into a class makes it easier to abstract graphics API code
 */
class TextureRuntime {
public:
    TextureRuntime();
    ~TextureRuntime() = default;

    // Copies the GPU pixel data to the provided pixels buffer
    void ReadTexture(const OGLTexture& tex, Subresource subresource, const FormatTuple& tuple,
                     u8* pixels);

    // Fills the rectangle of the texture with the clear value provided
    bool ClearTexture(const OGLTexture& texture, Subresource subresource, ClearValue value);

    // Copies a rectangle of src_tex to another rectange of dst_rect
    // NOTE: The width and height of the rectangles must be equal
    bool CopyTextures(const OGLTexture& src_tex, Subresource src_subresource,
                      const OGLTexture& dst_tex, Subresource dst_subresource);

    // Copies a rectangle of src_tex to another rectange of dst_rect performing
    // scaling and format conversions
    bool BlitTextures(const OGLTexture& src_tex, Subresource src_subresource,
                      const OGLTexture& dst_tex, Subresource dst_subresource);

    // Generates mipmaps for all the available levels of the texture
    void GenerateMipmaps(const OGLTexture& tex, u32 max_level);

private:
    OGLFramebuffer read_fbo, draw_fbo;
};

} // namespace OpenGL
