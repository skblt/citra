// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once
#include <string_view>
#include "core/hw/gpu.h"
#include "video_core/regs_texturing.h"
#include "video_core/regs_framebuffer.h"

namespace VideoCore {

enum class PixelFormat : u8 {
    // First 5 formats are shared between textures and color buffers
    RGBA8 = 0,
    RGB8 = 1,
    RGB5A1 = 2,
    RGB565 = 3,
    RGBA4 = 4,
    // Texture-only formats
    IA8 = 5,
    RG8 = 6,
    I8 = 7,
    A8 = 8,
    IA4 = 9,
    I4 = 10,
    A4 = 11,
    ETC1 = 12,
    ETC1A4 = 13,
    // Depth buffer-only formats
    D16 = 14,
    D24 = 16,
    D24S8 = 17,
    Invalid = 255,
};

inline constexpr u32 GetFormatBpp(PixelFormat format) {
    switch (format) {
    case PixelFormat::RGBA8:
    case PixelFormat::D24S8:
        return 32;
    case PixelFormat::RGB8:
    case PixelFormat::D24:
        return 24;
    case PixelFormat::RGB5A1:
    case PixelFormat::RGB565:
    case PixelFormat::RGBA4:
    case PixelFormat::IA8:
    case PixelFormat::RG8:
    case PixelFormat::D16:
        return 16;
    case PixelFormat::I8:
    case PixelFormat::A8:
    case PixelFormat::IA4:
    case PixelFormat::ETC1A4:
        return 8;
    case PixelFormat::I4:
    case PixelFormat::A4:
    case PixelFormat::ETC1:
        return 4;
    case PixelFormat::Invalid:
        return 1;
    }
}

inline std::string_view GetFormatName(PixelFormat format) {
    switch (format) {
    case PixelFormat::RGBA8:
        return "RGBA8";
    case PixelFormat::RGB8:
        return "RGB8";
    case PixelFormat::RGB5A1:
        return "RGB5A1";
    case PixelFormat::RGB565:
        return "RGB565";
    case PixelFormat::RGBA4:
        return "RGBA4";
    case PixelFormat::IA8:
        return "IA8";
    case PixelFormat::RG8:
        return "RG8";
    case PixelFormat::I8:
        return "I8";
    case PixelFormat::A8:
        return "A8";
    case PixelFormat::IA4:
        return "IA4";
    case PixelFormat::I4:
        return "I4";
    case PixelFormat::A4:
        return "A4";
    case PixelFormat::ETC1:
        return "ETC1";
    case PixelFormat::ETC1A4:
        return "ETC1A4";
    case PixelFormat::D16:
        return "D16";
    case PixelFormat::D24:
        return "D24";
    case PixelFormat::D24S8:
        return "D24S8";
    default:
        return "Not a real pixel format";
    }
}

inline PixelFormat PixelFormatFromTextureFormat(Pica::TexturingRegs::TextureFormat format) {
    const u32 format_index = static_cast<u32>(format);
    return (format_index < 14) ? static_cast<PixelFormat>(format) : PixelFormat::Invalid;
}

inline PixelFormat PixelFormatFromColorFormat(Pica::FramebufferRegs::ColorFormat format) {
    const u32 format_index = static_cast<u32>(format);
    return (format_index < 5) ? static_cast<PixelFormat>(format) : PixelFormat::Invalid;
}

inline PixelFormat PixelFormatFromDepthFormat(Pica::FramebufferRegs::DepthFormat format) {
    const u32 format_index = static_cast<u32>(format);
    return (format_index < 4) ? static_cast<PixelFormat>(format_index + 14)
                              : PixelFormat::Invalid;
}

inline PixelFormat PixelFormatFromGPUPixelFormat(GPU::Regs::PixelFormat format) {
    const u32 format_index = static_cast<u32>(format);
    switch (format) {
    // RGB565 and RGB5A1 are switched in PixelFormat compared to ColorFormat
    case GPU::Regs::PixelFormat::RGB565:
        return PixelFormat::RGB565;
    case GPU::Regs::PixelFormat::RGB5A1:
        return PixelFormat::RGB5A1;
    default:
        return (format_index < 5) ? static_cast<PixelFormat>(format) : PixelFormat::Invalid;
    }
}

} // namespace VideoCore
