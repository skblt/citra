// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <immintrin.h>
#include "video_core/texture/texture_decode.h"

namespace Pica::Texture {

constexpr u8 SHUFFLE_MASK = 0b11011000;

void DecodeTile8(u32 stride, const u8* tile_buffer, u8* gpu_buffer) {
    const __m256i* input_data = reinterpret_cast<const __m256i*>(tile_buffer);

    const __m256i PERMUTE_MASK = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i load1 = _mm256_load_si256(input_data);
    __m256i load2 = _mm256_load_si256(input_data + 1);

    __m256i shuffled1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(load1,
                                               SHUFFLE_MASK), SHUFFLE_MASK);
    __m256i shuffled2 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(load2,
                                               SHUFFLE_MASK), SHUFFLE_MASK);

    __m256i unswizzled1 = _mm256_permutevar8x32_epi32(shuffled1, PERMUTE_MASK);
    __m256i unswizzled2 = _mm256_permutevar8x32_epi32(shuffled2, PERMUTE_MASK);

    std::array<u64, 8> lines;
    lines[0] = _mm_extract_epi64(_mm256_castsi256_si128(unswizzled1), 0);
    lines[1] = _mm_extract_epi64(_mm256_castsi256_si128(unswizzled1), 1);
    lines[2] = _mm_extract_epi64(_mm256_extractf128_si256(unswizzled1, 1), 0);
    lines[3] = _mm_extract_epi64(_mm256_extractf128_si256(unswizzled1, 1), 1);
    lines[4] = _mm_extract_epi64(_mm256_castsi256_si128(unswizzled2), 0);
    lines[5] = _mm_extract_epi64(_mm256_castsi256_si128(unswizzled2), 1);
    lines[6] = _mm_extract_epi64(_mm256_extractf128_si256(unswizzled2, 1), 0);
    lines[7] = _mm_extract_epi64(_mm256_extractf128_si256(unswizzled2, 1), 1);

    for (u32 y = 0; y < 8; y++) {
        const u32 offset = ((7 - y) * stride);
        u64* addr = reinterpret_cast<u64*>(gpu_buffer + offset);
        *addr = lines[y];
    }
}

void DecodeTileIA4(u32 stride, const u8* tile_buffer, u8* gpu_buffer) {
    const __m256i* input_data = reinterpret_cast<const __m256i*>(tile_buffer);

    const __m256i PERMUTE_MASK = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i load1 = _mm256_load_si256(input_data);
    __m256i load2 = _mm256_load_si256(input_data + 1);

    __m256i shuffled1 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(load1,
        SHUFFLE_MASK), SHUFFLE_MASK);
    __m256i shuffled2 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(load2,
        SHUFFLE_MASK), SHUFFLE_MASK);

    __m256i unswizzled1 = _mm256_permutevar8x32_epi32(shuffled1, PERMUTE_MASK);
    __m256i unswizzled2 = _mm256_permutevar8x32_epi32(shuffled2, PERMUTE_MASK);

    // Now we convert every 8bit IA4 pixel to IA8
    // We start by zero extending every 8bit value to 16bit: 0xIA -> 0x00IA
    __m256i batch1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(unswizzled1));
    __m256i batch2 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(unswizzled1, 1));
    __m256i batch3 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(unswizzled2));
    __m256i batch4 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(unswizzled2, 1));

    // shifted = 0xIA | (0xIA << 8) which results in 0xIAIA
    __m256i shifted1 = _mm256_or_si256(batch1, _mm256_slli_epi16(batch1, 8));
    __m256i shifted2 = _mm256_or_si256(batch2, _mm256_slli_epi16(batch2, 8));
    __m256i shifted3 = _mm256_or_si256(batch3, _mm256_slli_epi16(batch3, 8));
    __m256i shifted4 = _mm256_or_si256(batch4, _mm256_slli_epi16(batch4, 8));

    // The mask 0xF00F is to clear the middle two nibbles;
    // converted = (0xIAIA & 0xF00F) | (0xIA << 4) which results to
    // 0xI00A | (0xIA << 4) = 0xIIAA that is our desired result
    const __m256i AND_MASK = _mm256_set1_epi16(0xF00F);
    __m256i converted1 = _mm256_or_si256(_mm256_slli_epi16(batch1, 4),
                                         _mm256_and_si256(shifted1, AND_MASK));
    __m256i converted2 = _mm256_or_si256(_mm256_slli_epi16(batch2, 4),
                                         _mm256_and_si256(shifted2, AND_MASK));
    __m256i converted3 = _mm256_or_si256(_mm256_slli_epi16(batch3, 4),
                                         _mm256_and_si256(shifted3, AND_MASK));
    __m256i converted4 = _mm256_or_si256(_mm256_slli_epi16(batch4, 4),
                                         _mm256_and_si256(shifted4, AND_MASK));

    // Extract individual lines and store them separately
    std::array<__m128i, 8> lines;
    lines[0] = _mm256_castsi256_si128(converted1);
    lines[1] = _mm256_extractf128_si256(converted1, 0x1);
    lines[2] = _mm256_castsi256_si128(converted2);
    lines[3] = _mm256_extractf128_si256(converted2, 0x1);
    lines[4] = _mm256_castsi256_si128(converted3);
    lines[5] = _mm256_extractf128_si256(converted3, 0x1);
    lines[6] = _mm256_castsi256_si128(converted4);
    lines[7] = _mm256_extractf128_si256(converted4, 0x1);

    for (u32 y = 0; y < 8; y++) {
        const u32 offset = ((7 - y) * stride * 2);
        __m128i* addr = reinterpret_cast<__m128i*>(gpu_buffer + offset);
        *addr = lines[y];
    }
}

void DecodeTile16(u32 stride, const u8* tile_buffer, u8* gpu_buffer) {
    const __m256i* input_data = reinterpret_cast<const __m256i*>(tile_buffer);

    __m256i line1 = _mm256_load_si256(input_data);
    __m256i line2 = _mm256_load_si256(input_data + 1);
    __m256i line3 = _mm256_load_si256(input_data + 2);
    __m256i line4 = _mm256_load_si256(input_data + 3);

    const __m256i PERMUTE_MASK = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    __m256i shuffled1 = _mm256_permutevar8x32_epi32(line1, PERMUTE_MASK);
    __m256i shuffled2 = _mm256_permutevar8x32_epi32(line2, PERMUTE_MASK);
    __m256i shuffled3 = _mm256_permutevar8x32_epi32(line3, PERMUTE_MASK);
    __m256i shuffled4 = _mm256_permutevar8x32_epi32(line4, PERMUTE_MASK);

    __m256i interleaved1 = _mm256_unpacklo_epi64(shuffled1, shuffled2);
    __m256i interleaved2 = _mm256_unpackhi_epi64(shuffled1, shuffled2);
    __m256i interleaved3 = _mm256_unpacklo_epi64(shuffled3, shuffled4);
    __m256i interleaved4 = _mm256_unpackhi_epi64(shuffled3, shuffled4);

    std::array<__m128i, 8> lines;
    lines[0] = _mm256_castsi256_si128(interleaved1);
    lines[1] = _mm256_extractf128_si256(interleaved1, 0x1);
    lines[2] = _mm256_castsi256_si128(interleaved2);
    lines[3] = _mm256_extractf128_si256(interleaved2, 0x1);
    lines[4] = _mm256_castsi256_si128(interleaved3);
    lines[5] = _mm256_extractf128_si256(interleaved3, 0x1);
    lines[6] = _mm256_castsi256_si128(interleaved4);
    lines[7] = _mm256_extractf128_si256(interleaved4, 0x1);

    // Store the each line separately
    for (u32 y = 0; y < 8; y++) {
        const u32 offset = ((7 - y) * stride * 2);
        __m128i* addr = reinterpret_cast<__m128i*>(gpu_buffer + offset);
        *addr = lines[y];
    }
}

void DecodeTile32(u32 stride, const u8* tile_buffer, u8* gpu_buffer) {
    const __m256i* input_data = reinterpret_cast<const __m256i*>(tile_buffer);

    __m256i load1 = _mm256_load_si256(input_data);
    __m256i load2 = _mm256_load_si256(input_data + 1);
    __m256i load3 = _mm256_load_si256(input_data + 2);
    __m256i load4 = _mm256_load_si256(input_data + 3);
    __m256i load5 = _mm256_load_si256(input_data + 4);
    __m256i load6 = _mm256_load_si256(input_data + 5);
    __m256i load7 = _mm256_load_si256(input_data + 6);
    __m256i load8 = _mm256_load_si256(input_data + 7);

    __m256i shuffled1 = _mm256_permute4x64_epi64(load1, SHUFFLE_MASK);
    __m256i shuffled2 = _mm256_permute4x64_epi64(load2, SHUFFLE_MASK);
    __m256i shuffled3 = _mm256_permute4x64_epi64(load3, SHUFFLE_MASK);
    __m256i shuffled4 = _mm256_permute4x64_epi64(load4, SHUFFLE_MASK);
    __m256i shuffled5 = _mm256_permute4x64_epi64(load5, SHUFFLE_MASK);
    __m256i shuffled6 = _mm256_permute4x64_epi64(load6, SHUFFLE_MASK);
    __m256i shuffled7 = _mm256_permute4x64_epi64(load7, SHUFFLE_MASK);
    __m256i shuffled8 = _mm256_permute4x64_epi64(load8, SHUFFLE_MASK);

    std::array<__m256i, 8> lines;
    lines[0] = _mm256_permute2x128_si256(shuffled1, shuffled3, 0x20);
    lines[1] = _mm256_permute2x128_si256(shuffled1, shuffled3, 0x31);
    lines[2] = _mm256_permute2x128_si256(shuffled2, shuffled4, 0x20);
    lines[3] = _mm256_permute2x128_si256(shuffled2, shuffled4, 0x31);
    lines[4] = _mm256_permute2x128_si256(shuffled5, shuffled7, 0x20);
    lines[5] = _mm256_permute2x128_si256(shuffled5, shuffled7, 0x31);
    lines[6] = _mm256_permute2x128_si256(shuffled6, shuffled8, 0x20);
    lines[7] = _mm256_permute2x128_si256(shuffled6, shuffled8, 0x31);

    // Store the each line separately
    for (u32 y = 0; y < 8; y++) {
        const u32 offset = ((7 - y) * stride * 4);
        __m256i* addr = reinterpret_cast<__m256i*>(gpu_buffer + offset);
        *addr = lines[y];
    }
}

void EncodeTile8(u32 stride, u8* tile_buffer, const u8* gpu_buffer) {
    __m256i* output_data = reinterpret_cast<__m256i*>(tile_buffer);
     const __m128i indices = _mm_set_epi32(3 * stride, 2 * stride, stride, 0);
     const __int64* base_addr = reinterpret_cast<const __int64*>(gpu_buffer);

     std::array<__m256i, 2> load;
     for (u32 y = 0; y < 8; y += 4) {
         load[y >> 2] = _mm256_i32gather_epi64(base_addr, indices, 1);
         base_addr += stride / 2;
     }

     const __m256i PERMUTE_MASK = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
}

void EncodeTile16(u32 stride, u8* tile_buffer, const u8* gpu_buffer) {
    __m256i* output_data = reinterpret_cast<__m256i*>(tile_buffer);

    std::array<__m256i, 4> load;
    for (u32 y = 0; y < 8; y += 2) {
        const u32 offset_lo = (y * stride * 2);
        const u32 offset_hi = ((y + 1) * stride * 2);
        load[y >> 1] = _mm256_loadu2_m128i(reinterpret_cast<const __m128i*>(gpu_buffer + offset_hi),
                                           reinterpret_cast<const __m128i*>(gpu_buffer + offset_lo));
    }

    const __m256i PERMUTE_MASK = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    __m256i shuffled1 = _mm256_permutevar8x32_epi32(load[0], PERMUTE_MASK);
    __m256i shuffled2 = _mm256_permutevar8x32_epi32(load[1], PERMUTE_MASK);
    __m256i shuffled3 = _mm256_permutevar8x32_epi32(load[2], PERMUTE_MASK);
    __m256i shuffled4 = _mm256_permutevar8x32_epi32(load[3], PERMUTE_MASK);

    constexpr u8 LINE_MASK_LO = 0b00100000;
    constexpr u8 LINE_MASK_HI = 0b00110001;
    __m256i swizzled1 = _mm256_permute2x128_si256(shuffled1, shuffled2, LINE_MASK_LO);
    __m256i swizzled2 = _mm256_permute2x128_si256(shuffled1, shuffled2, LINE_MASK_HI);
    __m256i swizzled3 = _mm256_permute2x128_si256(shuffled3, shuffled4, LINE_MASK_LO);
    __m256i swizzled4 = _mm256_permute2x128_si256(shuffled2, shuffled4, LINE_MASK_HI);

    _mm256_store_si256(output_data, swizzled1);
    _mm256_store_si256(output_data + 1, swizzled2);
    _mm256_store_si256(output_data + 2, swizzled3);
    _mm256_store_si256(output_data + 3, swizzled4);
}


void EncodeTile32(u32 stride, u8* tile_buffer, const u8* gpu_buffer) {
    __m256i* output_data = reinterpret_cast<__m256i*>(tile_buffer);

    std::array<__m256i, 8> load;
    for (u32 y = 0; y < 8; y++) {
        const u32 offset = (y * stride * 4);
        const __m256i* load_addr = reinterpret_cast<const __m256i*>(gpu_buffer + offset);
        load[y] = _mm256_load_si256(load_addr);
    }

    __m256i shuffled1 = _mm256_permute4x64_epi64(load[0], SHUFFLE_MASK);
    __m256i shuffled2 = _mm256_permute4x64_epi64(load[1], SHUFFLE_MASK);
    __m256i shuffled3 = _mm256_permute4x64_epi64(load[2], SHUFFLE_MASK);
    __m256i shuffled4 = _mm256_permute4x64_epi64(load[3], SHUFFLE_MASK);
    __m256i shuffled5 = _mm256_permute4x64_epi64(load[4], SHUFFLE_MASK);
    __m256i shuffled6 = _mm256_permute4x64_epi64(load[5], SHUFFLE_MASK);
    __m256i shuffled7 = _mm256_permute4x64_epi64(load[6], SHUFFLE_MASK);
    __m256i shuffled8 = _mm256_permute4x64_epi64(load[7], SHUFFLE_MASK);

    __m256i swizzled1 = _mm256_unpacklo_epi64(shuffled8, shuffled7);
    __m256i swizzled2 = _mm256_unpacklo_epi64(shuffled6, shuffled5);
    __m256i swizzled3 = _mm256_unpackhi_epi64(shuffled8, shuffled7);
    __m256i swizzled4 = _mm256_unpackhi_epi64(shuffled6, shuffled5);
    __m256i swizzled5 = _mm256_unpacklo_epi64(shuffled4, shuffled3);
    __m256i swizzled6 = _mm256_unpacklo_epi64(shuffled2, shuffled1);
    __m256i swizzled7 = _mm256_unpackhi_epi64(shuffled4, shuffled3);
    __m256i swizzled8 = _mm256_unpackhi_epi64(shuffled2, shuffled1);

    _mm256_store_si256(output_data, swizzled1);
    _mm256_store_si256(output_data + 1, swizzled2);
    _mm256_store_si256(output_data + 2, swizzled3);
    _mm256_store_si256(output_data + 3, swizzled4);
    _mm256_store_si256(output_data + 4, swizzled5);
    _mm256_store_si256(output_data + 5, swizzled6);
    _mm256_store_si256(output_data + 6, swizzled7);
    _mm256_store_si256(output_data + 7, swizzled8);
}

void ConvertRGBToRGBA(u32 byte_count, const u8* input_buffer, u8* output_buffer) {
    static std::array<u8, 64> aligned_storage;
    aligned_storage.fill(0xFF);

    const __m256i* load_addr = reinterpret_cast<const __m256i*>(aligned_storage.data());
    const __m256i SHUFFLE_MASK = _mm256_set_epi8(11, 10, 9, 15, 8, 7, 6, 14, 5, 4, 3, 13, 2, 1, 0,
                                                 12, 15, 14, 13, 3, 12, 11, 10, 2, 9, 8, 7, 1, 6, 5, 4, 0);
    u32 input_offset = 0;
    u32 output_offset = 0;
    while (input_offset <= byte_count - 48) {
        // Processing 16 pixels every loop iteration seems to give the best performance
        std::memcpy(aligned_storage.data() + 4, input_buffer + input_offset, 24);
        std::memcpy(aligned_storage.data() + 36, input_buffer + input_offset + 24, 24);
        input_offset += 48;

        __m256i load1 = _mm256_load_si256(load_addr);
        __m256i load2 = _mm256_load_si256(load_addr+1);
        __m256i rgba1 = _mm256_shuffle_epi8(load1, SHUFFLE_MASK);
        __m256i rgba2 = _mm256_shuffle_epi8(load2, SHUFFLE_MASK);

        _mm256_store_si256(reinterpret_cast<__m256i*>(output_buffer + output_offset), rgba1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(output_buffer + output_offset + 32), rgba2);
        output_buffer += 64;
    }

    // Process any remaining pixels if the input data wasn't aligned to 48 bytes
    if (byte_count > input_offset) {
        for (; input_offset < byte_count; input_offset += 3) {
            output_buffer[output_offset] = 255;
            output_buffer[output_offset+1] = input_buffer[input_offset];
            output_buffer[output_offset+2] = input_buffer[input_offset+1];
            output_buffer[output_offset+3] = input_buffer[input_offset+2];
            output_offset += 4;
        }
    }
}

} // namespace Pica::Texture
