// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <limits>
#include "core/memory.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/pica_state.h"
#include "video_core/video_core.h"

namespace VideoCore {

static Common::Vec4f ColorRGBA8(const u32 color) {
    const auto rgba =
        Common::Vec4u{color >> 0 & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF, color >> 24 & 0xFF};
    return rgba / 255.0f;
}

static Common::Vec3f LightColor(const Pica::LightingRegs::LightColor& color) {
    return Common::Vec3u{color.r, color.g, color.b} / 255.0f;
}

RasterizerAccelerated::RasterizerAccelerated() {
    uniform_block_data.lighting_lut_dirty.fill(true);
}

void RasterizerAccelerated::UpdatePagesCachedCount(PAddr addr, u32 size, int delta) {
    const u32 page_start = addr >> Memory::CITRA_PAGE_BITS;
    const u32 page_end = ((addr + size - 1) >> Memory::CITRA_PAGE_BITS) + 1;

    u32 uncache_start_addr = 0;
    u32 cache_start_addr = 0;
    u32 uncache_bytes = 0;
    u32 cache_bytes = 0;

    for (u32 page = page_start; page != page_end; page++) {
        auto& count = cached_pages.at(page);

        // Ensure no overflow happens
        if (delta > 0) {
            ASSERT_MSG(count < std::numeric_limits<u16>::max(), "Count will overflow!");
        } else if (delta < 0) {
            ASSERT_MSG(count > 0, "Count will underflow!");
        } else {
            ASSERT_MSG(false, "Delta must be non-zero!");
        }

        // Adds or subtracts 1, as count is a unsigned 8-bit value
        count += delta;

        // Assume delta is either -1 or 1
        if (count == 0) {
            if (uncache_bytes == 0) {
                uncache_start_addr = page << Memory::CITRA_PAGE_BITS;
            }

            uncache_bytes += Memory::CITRA_PAGE_SIZE;
        } else if (uncache_bytes > 0) {
            VideoCore::g_memory->RasterizerMarkRegionCached(uncache_start_addr, uncache_bytes,
                                                            false);
            uncache_bytes = 0;
        }

        if (count == 1 && delta > 0) {
            if (cache_bytes == 0) {
                cache_start_addr = page << Memory::CITRA_PAGE_BITS;
            }

            cache_bytes += Memory::CITRA_PAGE_SIZE;
        } else if (cache_bytes > 0) {
            VideoCore::g_memory->RasterizerMarkRegionCached(cache_start_addr, cache_bytes, true);

            cache_bytes = 0;
        }
    }

    if (uncache_bytes > 0) {
        VideoCore::g_memory->RasterizerMarkRegionCached(uncache_start_addr, uncache_bytes, false);
    }

    if (cache_bytes > 0) {
        VideoCore::g_memory->RasterizerMarkRegionCached(cache_start_addr, cache_bytes, true);
    }
}

void RasterizerAccelerated::ClearAll(bool flush) {
    // Force flush all surfaces from the cache
    if (flush) {
        FlushRegion(0x0, 0xFFFFFFFF);
    }

    u32 uncache_start_addr = 0;
    u32 uncache_bytes = 0;

    for (u32 page = 0; page != cached_pages.size(); page++) {
        auto& count = cached_pages.at(page);

        // Assume delta is either -1 or 1
        if (count != 0) {
            if (uncache_bytes == 0) {
                uncache_start_addr = page << Memory::CITRA_PAGE_BITS;
            }

            uncache_bytes += Memory::CITRA_PAGE_SIZE;
        } else if (uncache_bytes > 0) {
            VideoCore::g_memory->RasterizerMarkRegionCached(uncache_start_addr, uncache_bytes,
                                                            false);
            uncache_bytes = 0;
        }
    }

    if (uncache_bytes > 0) {
        VideoCore::g_memory->RasterizerMarkRegionCached(uncache_start_addr, uncache_bytes, false);
    }

    cached_pages = {};
}

void RasterizerAccelerated::SyncDepthScale() {
    float depth_scale =
        Pica::float24::FromRaw(Pica::g_state.regs.rasterizer.viewport_depth_range).ToFloat32();

    if (depth_scale != uniform_block_data.data.depth_scale) {
        uniform_block_data.data.depth_scale = depth_scale;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncDepthOffset() {
    float depth_offset =
        Pica::float24::FromRaw(Pica::g_state.regs.rasterizer.viewport_depth_near_plane).ToFloat32();

    if (depth_offset != uniform_block_data.data.depth_offset) {
        uniform_block_data.data.depth_offset = depth_offset;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncFogColor() {
    const auto& regs = Pica::g_state.regs;
    uniform_block_data.data.fog_color = {
        regs.texturing.fog_color.r.Value() / 255.0f,
        regs.texturing.fog_color.g.Value() / 255.0f,
        regs.texturing.fog_color.b.Value() / 255.0f,
    };
    uniform_block_data.dirty = true;
}

void RasterizerAccelerated::SyncProcTexNoise() {
    const auto& regs = Pica::g_state.regs.texturing;
    uniform_block_data.data.proctex_noise_f = {
        Pica::float16::FromRaw(regs.proctex_noise_frequency.u).ToFloat32(),
        Pica::float16::FromRaw(regs.proctex_noise_frequency.v).ToFloat32(),
    };
    uniform_block_data.data.proctex_noise_a = {
        regs.proctex_noise_u.amplitude / 4095.0f,
        regs.proctex_noise_v.amplitude / 4095.0f,
    };
    uniform_block_data.data.proctex_noise_p = {
        Pica::float16::FromRaw(regs.proctex_noise_u.phase).ToFloat32(),
        Pica::float16::FromRaw(regs.proctex_noise_v.phase).ToFloat32(),
    };

    uniform_block_data.dirty = true;
}

void RasterizerAccelerated::SyncProcTexBias() {
    const auto& regs = Pica::g_state.regs.texturing;
    uniform_block_data.data.proctex_bias =
        Pica::float16::FromRaw(regs.proctex.bias_low | (regs.proctex_lut.bias_high << 8))
            .ToFloat32();

    uniform_block_data.dirty = true;
}

void RasterizerAccelerated::SyncAlphaTest() {
    const auto& regs = Pica::g_state.regs;
    if (regs.framebuffer.output_merger.alpha_test.ref != uniform_block_data.data.alphatest_ref) {
        uniform_block_data.data.alphatest_ref = regs.framebuffer.output_merger.alpha_test.ref;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncCombinerColor() {
    auto combiner_color =
        ColorRGBA8(Pica::g_state.regs.texturing.tev_combiner_buffer_color.raw);
    if (combiner_color != uniform_block_data.data.tev_combiner_buffer_color) {
        uniform_block_data.data.tev_combiner_buffer_color = combiner_color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncTevConstColor(std::size_t stage_index,
                                         const Pica::TexturingRegs::TevStageConfig& tev_stage) {
    const auto const_color = ColorRGBA8(tev_stage.const_color);

    if (const_color == uniform_block_data.data.const_color[stage_index]) {
        return;
    }

    uniform_block_data.data.const_color[stage_index] = const_color;
    uniform_block_data.dirty = true;
}

void RasterizerAccelerated::SyncGlobalAmbient() {
    auto color = LightColor(Pica::g_state.regs.lighting.global_ambient);
    if (color != uniform_block_data.data.lighting_global_ambient) {
        uniform_block_data.data.lighting_global_ambient = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpecular0(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].specular_0);
    if (color != uniform_block_data.data.light_src[light_index].specular_0) {
        uniform_block_data.data.light_src[light_index].specular_0 = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpecular1(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].specular_1);
    if (color != uniform_block_data.data.light_src[light_index].specular_1) {
        uniform_block_data.data.light_src[light_index].specular_1 = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDiffuse(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].diffuse);
    if (color != uniform_block_data.data.light_src[light_index].diffuse) {
        uniform_block_data.data.light_src[light_index].diffuse = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightAmbient(int light_index) {
    auto color = LightColor(Pica::g_state.regs.lighting.light[light_index].ambient);
    if (color != uniform_block_data.data.light_src[light_index].ambient) {
        uniform_block_data.data.light_src[light_index].ambient = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightPosition(int light_index) {
    const Common::Vec3f position = {
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].x).ToFloat32(),
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].y).ToFloat32(),
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].z).ToFloat32()};

    if (position != uniform_block_data.data.light_src[light_index].position) {
        uniform_block_data.data.light_src[light_index].position = position;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightSpotDirection(int light_index) {
    const auto& light = Pica::g_state.regs.lighting.light[light_index];
    const auto spot_direction = Common::Vec3f{light.spot_x / 2047.0f, light.spot_y / 2047.0f, light.spot_z / 2047.0f};

    if (spot_direction != uniform_block_data.data.light_src[light_index].spot_direction) {
        uniform_block_data.data.light_src[light_index].spot_direction = spot_direction;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDistanceAttenuationBias(int light_index) {
    float dist_atten_bias =
        Pica::float20::FromRaw(Pica::g_state.regs.lighting.light[light_index].dist_atten_bias)
            .ToFloat32();

    if (dist_atten_bias != uniform_block_data.data.light_src[light_index].dist_atten_bias) {
        uniform_block_data.data.light_src[light_index].dist_atten_bias = dist_atten_bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncLightDistanceAttenuationScale(int light_index) {
    float dist_atten_scale =
        Pica::float20::FromRaw(Pica::g_state.regs.lighting.light[light_index].dist_atten_scale)
            .ToFloat32();

    if (dist_atten_scale != uniform_block_data.data.light_src[light_index].dist_atten_scale) {
        uniform_block_data.data.light_src[light_index].dist_atten_scale = dist_atten_scale;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncShadowBias() {
    const auto& shadow = Pica::g_state.regs.framebuffer.shadow;
    float constant = Pica::float16::FromRaw(shadow.constant).ToFloat32();
    float linear = Pica::float16::FromRaw(shadow.linear).ToFloat32();

    if (constant != uniform_block_data.data.shadow_bias_constant ||
        linear != uniform_block_data.data.shadow_bias_linear) {
        uniform_block_data.data.shadow_bias_constant = constant;
        uniform_block_data.data.shadow_bias_linear = linear;
        uniform_block_data.dirty = true;
    }
}

void RasterizerAccelerated::SyncShadowTextureBias() {
    int bias = Pica::g_state.regs.texturing.shadow.bias << 1;
    if (bias != uniform_block_data.data.shadow_texture_bias) {
        uniform_block_data.data.shadow_texture_bias = bias;
        uniform_block_data.dirty = true;
    }
}

} // namespace VideoCore
