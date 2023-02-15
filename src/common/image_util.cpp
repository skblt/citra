// Copyright 2023 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <spng.h>
#include "common/dds.h"
#include "common/file_util.h"
#include "common/image_util.h"
#include "common/logging/log.h"

namespace Common {

using namespace Common::DirectX;

namespace {

void spng_free(spng_ctx* ctx) {
    if (ctx) {
        spng_ctx_free(ctx);
    }
}

auto make_spng_ctx(int flags) {
    return std::unique_ptr<spng_ctx, decltype(&spng_free)>(spng_ctx_new(flags), spng_free);
}

} // Anonymous namespace

bool DecodePNG(std::span<const u8> in_data, std::vector<u8>& out_data, u32& width, u32& height) {
    auto ctx = make_spng_ctx(0);
    if (!ctx) [[unlikely]] {
        return false;
    }

    if (spng_set_png_buffer(ctx.get(), in_data.data(), in_data.size())) {
        return false;
    }

    spng_ihdr ihdr{};
    if (spng_get_ihdr(ctx.get(), &ihdr)) {
        return false;
    }

    const int format = SPNG_FMT_RGBA8;
    size_t decoded_len = 0;
    if (spng_decoded_image_size(ctx.get(), format, &decoded_len)) {
        return false;
    }

    out_data.resize(decoded_len);
    if (spng_decode_image(ctx.get(), out_data.data(), decoded_len, format, SPNG_DECODE_TRNS)) {
        return false;
    }

    width = ihdr.width;
    height = ihdr.height;
    return true;
}

bool EncodePNG(std::span<u8> in_data, const std::string& out_path, u32 width, u32 height,
               u32 stride, s32 level) {
    auto ctx = make_spng_ctx(SPNG_CTX_ENCODER);
    if (!ctx) [[unlikely]] {
        return false;
    }

    auto outfile = FileUtil::IOFile(out_path, "wb");
    if (spng_set_png_file(ctx.get(), outfile.Handle())) {
        return false;
    }

    if (spng_set_option(ctx.get(), SPNG_IMG_COMPRESSION_LEVEL, level)) {
        return false;
    }

    spng_ihdr ihdr{};
    ihdr.width = width;
    ihdr.height = height;
    ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA;
    ihdr.bit_depth = 8;
    if (spng_set_ihdr(ctx.get(), &ihdr)) {
        return false;
    }

    if (spng_encode_image(ctx.get(), nullptr, 0, SPNG_FMT_PNG,
                          SPNG_ENCODE_PROGRESSIVE | SPNG_ENCODE_FINALIZE)) {
        return false;
    }

    for (u32 row = 0; row < height; row++) {
        const int err = spng_encode_row(ctx.get(), &in_data[row * stride], stride);
        if (err == SPNG_EOI) {
            break;
        }

        if (err) {
            LOG_ERROR(Common, "Failed to save {} by {} image to {} at level {}: error {}", width,
                      height, out_path, level, err);
            return false;
        }
    }

    size_t image_len = 0;
    spng_decoded_image_size(ctx.get(), SPNG_FMT_PNG, &image_len);
    LOG_ERROR(Common, "{} byte {} by {} image saved to {} at level {}", image_len, width, height,
              out_path, level);

    return true;
}

} // namespace Common
