//--------------------------------------------------------------------------------------
// DDS.h
//
// This header defines constants and structures that are useful when parsing
// DDS files.  DDS files were originally designed to use several structures
// and constants that are native to DirectDraw and are defined in ddraw.h,
// such as DDSURFACEDESC2 and DDSCAPS2.  This file defines similar
// (compatible) constants and structures so that one can use DDS files
// without needing to include ddraw.h.
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// http://go.microsoft.com/fwlink/?LinkId=248926
// http://go.microsoft.com/fwlink/?LinkId=248929
// http://go.microsoft.com/fwlink/?LinkID=615561
//--------------------------------------------------------------------------------------

#pragma once

#include <cstdint>

namespace Common::DirectX {

#pragma pack(push, 1)

const uint32_t DDS_MAGIC = 0x20534444; // "DDS "

struct DDS_PIXELFORMAT {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwFourCC;
    uint32_t dwRGBBitCount;
    uint32_t dwRBitMask;
    uint32_t dwGBitMask;
    uint32_t dwBBitMask;
    uint32_t dwABitMask;
};

#define DDS_FOURCC 0x00000004     // DDPF_FOURCC
#define DDS_RGB 0x00000040        // DDPF_RGB
#define DDS_RGBA 0x00000041       // DDPF_RGB | DDPF_ALPHAPIXELS
#define DDS_LUMINANCE 0x00020000  // DDPF_LUMINANCE
#define DDS_LUMINANCEA 0x00020001 // DDPF_LUMINANCE | DDPF_ALPHAPIXELS
#define DDS_ALPHA 0x00000002      // DDPF_ALPHA
#define DDS_PAL8 0x00000020       // DDPF_PALETTEINDEXED8
#define DDS_PAL8A 0x00000021      // DDPF_PALETTEINDEXED8 | DDPF_ALPHAPIXELS
#define DDS_BUMPDUDV 0x00080000   // DDPF_BUMPDUDV

#ifndef MAKEFOURCC
#define MAKEFOURCC(ch0, ch1, ch2, ch3)                                                             \
    ((uint32_t)(uint8_t)(ch0) | ((uint32_t)(uint8_t)(ch1) << 8) |                                  \
     ((uint32_t)(uint8_t)(ch2) << 16) | ((uint32_t)(uint8_t)(ch3) << 24))
#endif /* defined(MAKEFOURCC) */

#define DDS_HEADER_FLAGS_TEXTURE                                                                   \
    0x00001007 // DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
#define DDS_HEADER_FLAGS_MIPMAP 0x00020000     // DDSD_MIPMAPCOUNT
#define DDS_HEADER_FLAGS_VOLUME 0x00800000     // DDSD_DEPTH
#define DDS_HEADER_FLAGS_PITCH 0x00000008      // DDSD_PITCH
#define DDS_HEADER_FLAGS_LINEARSIZE 0x00080000 // DDSD_LINEARSIZE

// Subset here matches D3D10_RESOURCE_DIMENSION and D3D11_RESOURCE_DIMENSION
enum DDS_RESOURCE_DIMENSION {
    DDS_DIMENSION_TEXTURE1D = 2,
    DDS_DIMENSION_TEXTURE2D = 3,
    DDS_DIMENSION_TEXTURE3D = 4,
};

struct DDS_HEADER {
    uint32_t dwSize;
    uint32_t dwFlags;
    uint32_t dwHeight;
    uint32_t dwWidth;
    uint32_t dwPitchOrLinearSize;
    uint32_t dwDepth; // only if DDS_HEADER_FLAGS_VOLUME is set in dwFlags
    uint32_t dwMipMapCount;
    uint32_t dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    uint32_t dwCaps;
    uint32_t dwCaps2;
    uint32_t dwCaps3;
    uint32_t dwCaps4;
    uint32_t dwReserved2;
};

struct DDS_HEADER_DXT10 {
    uint32_t dxgiFormat;
    uint32_t resourceDimension;
    uint32_t miscFlag; // see DDS_RESOURCE_MISC_FLAG
    uint32_t arraySize;
    uint32_t miscFlags2; // see DDS_MISC_FLAGS2
};

#pragma pack(pop)

static_assert(sizeof(DDS_HEADER) == 124, "DDS Header size mismatch");
static_assert(sizeof(DDS_HEADER_DXT10) == 20, "DDS DX10 Extended Header size mismatch");

constexpr DDS_PIXELFORMAT DDSPF_A8R8G8B8 = {
    sizeof(DDS_PIXELFORMAT), DDS_RGBA, 0, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000};
constexpr DDS_PIXELFORMAT DDSPF_X8R8G8B8 = {
    sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000};
constexpr DDS_PIXELFORMAT DDSPF_A8B8G8R8 = {
    sizeof(DDS_PIXELFORMAT), DDS_RGBA, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000};
constexpr DDS_PIXELFORMAT DDSPF_X8B8G8R8 = {
    sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000};
constexpr DDS_PIXELFORMAT DDSPF_R8G8B8 = {
    sizeof(DDS_PIXELFORMAT), DDS_RGB, 0, 24, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000};

} // namespace Common::DirectX
