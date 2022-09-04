// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <span>
#include <vector>
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/regs_texturing.h"

namespace Pica::Texture {

struct TextureInfo {
    PAddr address;
    u32 width;
    u32 height;
    TexturingRegs::TextureFormat format;

    static TextureInfo FromPicaRegister(const TexturingRegs::TextureConfig& config,
                                        const TexturingRegs::TextureFormat& format);
};

/**
 * Lookup texel located at the given coordinates and return an RGBA vector of its color.
 * @param source Source pointer to read data from
 * @param x,y Texture coordinates to read from
 * @param info TextureInfo object describing the texture setup
 * @param disable_alpha This is used for debug widgets which use this method to display textures
 * without providing a good way to visualize alpha by themselves. If true, this will return 255 for
 * the alpha component, and either drop the information entirely or store it in an "unused" color
 * channel.
 * @todo Eventually we should get rid of the disable_alpha parameter.
 */
Common::Vec4<u8> LookupTexture(const u8* source, unsigned int x, unsigned int y,
                               const TextureInfo& info, bool disable_alpha = false);

/**
 * Looks up a texel from a single 8x8 texture tile.
 *
 * @param source Pointer to the beginning of the tile.
 * @param x, y In-tile coordinates to read from. Must be < 8.
 * @param info TextureInfo describing the texture format.
 * @param disable_alpha Used for debugging. Sets the result alpha to 255 and either discards the
 *                      real alpha or inserts it in an otherwise unused channel.
 */
Common::Vec4<u8> LookupTexelInTile(const u8* source, unsigned int x, unsigned int y,
                                   const TextureInfo& info, bool disable_alpha);

/**
 * Converts a morton swizzled 8 * 8 block of pixels to linear format
 *
 * @param stride The width in pixels of the source texture.
 * @param tile_buffer The morton pixel data
 * @param gpu_buffer The output buffer where the linear data is written
 */
void DecodeTile8(u32 stride, const u8* tile_buffer, u8* gpu_buffer);
void DecodeTileIA4(u32 stride, const u8* tile_buffer, u8* gpu_buffer);
void DecodeTile16(u32 stride, const u8* tile_buffer, u8* gpu_buffer);
void DecodeTile32(u32 stride, const u8* tile_buffer, u8* gpu_buffer);

/**
 * Performs morton swizzling on a linear 8 * 8 block of pixels
 *
 * @param stride The width in pixels of the source texture.
 * @param tile_buffer The output buffer where the morton data is written
 * @param gpu_buffer The linear pixel data
 */
void EncodeTile8(u32 stride, u8* tile_buffer, const u8* gpu_buffer);
void EncodeTile16(u32 stride, u8* tile_buffer, const u8* gpu_buffer);
void EncodeTile32(u32 stride, u8* tile_buffer, const u8* gpu_buffer);

/**
 * Converts RGB8 encoded pixel data to RGBA8 by inserting a dummy alpha channel
 * @param byte_count The number of bytes to read from input_buffer
 * @param input_buffer The source RGB pixel data
 * @param output_buffer The output buffer where the RGBA pixel data is written
 */
void ConvertRGBToRGBA(u32 byte_count, const u8* input_buffer, u8* output_buffer);

} // namespace Pica::Texture
