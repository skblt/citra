// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <span>
#include <vector>
#include "common/common_types.h"

namespace Common {

/**
 * @brief DecodePNG Given a buffer of png input data decodes said data to RGBA8 format
 * and writes the result to out_data, updating width and height to match the file dimentions
 * @param in_data The input png data
 * @param out_data The decoded RGBA8 pixel data
 * @param width The output width of the png image
 * @param height The output height of the png image
 * @return true on decode success, false otherwise
 */
bool DecodePNG(std::span<const u8> in_data, std::vector<u8>& out_data, u32& width, u32& height);

bool EncodePNG(std::span<u8> in_data, const std::string& out_path, u32 width, u32 height,
               u32 stride, s32 level);

} // namespace Common
