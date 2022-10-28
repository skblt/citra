// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/regs_texturing.h"
#include "video_core/shader/shader_uniforms.h"

namespace VideoCore {

class RasterizerAccelerated : public RasterizerInterface {
public:
    RasterizerAccelerated();
    virtual ~RasterizerAccelerated() = default;

    void UpdatePagesCachedCount(PAddr addr, u32 size, int delta) override;

    void ClearAll(bool flush) override;

protected:
    /// Syncs the depth scale to match the PICA register
    void SyncDepthScale();

    /// Syncs the depth offset to match the PICA register
    void SyncDepthOffset();

    /// Syncs the fog states to match the PICA register
    void SyncFogColor();

    /// Sync the procedural texture noise configuration to match the PICA register
    void SyncProcTexNoise();

    /// Sync the procedural texture bias configuration to match the PICA register
    void SyncProcTexBias();

    /// Syncs the alpha test states to match the PICA register
    void SyncAlphaTest();

    /// Syncs the TEV combiner color buffer to match the PICA register
    void SyncCombinerColor();

    /// Syncs the TEV constant color to match the PICA register
    void SyncTevConstColor(std::size_t tev_index,
                           const Pica::TexturingRegs::TevStageConfig& tev_stage);

    /// Syncs the lighting global ambient color to match the PICA register
    void SyncGlobalAmbient();

    /// Syncs the specified light's specular 0 color to match the PICA register
    void SyncLightSpecular0(int light_index);

    /// Syncs the specified light's specular 1 color to match the PICA register
    void SyncLightSpecular1(int light_index);

    /// Syncs the specified light's diffuse color to match the PICA register
    void SyncLightDiffuse(int light_index);

    /// Syncs the specified light's ambient color to match the PICA register
    void SyncLightAmbient(int light_index);

    /// Syncs the specified light's position to match the PICA register
    void SyncLightPosition(int light_index);

    /// Syncs the specified spot light direcition to match the PICA register
    void SyncLightSpotDirection(int light_index);

    /// Syncs the specified light's distance attenuation bias to match the PICA register
    void SyncLightDistanceAttenuationBias(int light_index);

    /// Syncs the specified light's distance attenuation scale to match the PICA register
    void SyncLightDistanceAttenuationScale(int light_index);

    /// Syncs the shadow rendering bias to match the PICA register
    void SyncShadowBias();

    /// Syncs the shadow texture bias to match the PICA register
    void SyncShadowTextureBias();

private:
    struct UniformBlockData {
        Pica::Shader::UniformData data{};
        std::array<bool, Pica::LightingRegs::NumLightingSampler> lighting_lut_dirty{};
        bool lighting_lut_dirty_any = true;
        bool fog_lut_dirty = true;
        bool proctex_noise_lut_dirty = true;
        bool proctex_color_map_dirty = true;
        bool proctex_alpha_map_dirty = true;
        bool proctex_lut_dirty = true;
        bool proctex_diff_lut_dirty = true;
        bool dirty = true;
    };

protected:
    std::array<u16, 0x30000> cached_pages{};

    UniformBlockData uniform_block_data{};
    std::array<std::array<Common::Vec2f, 256>, Pica::LightingRegs::NumLightingSampler>
        lighting_lut_data{};
    std::array<Common::Vec2f, 128> fog_lut_data{};
    std::array<Common::Vec2f, 128> proctex_noise_lut_data{};
    std::array<Common::Vec2f, 128> proctex_color_map_data{};
    std::array<Common::Vec2f, 128> proctex_alpha_map_data{};
    std::array<Common::Vec4f, 256> proctex_lut_data{};
    std::array<Common::Vec4f, 256> proctex_diff_lut_data{};
};
} // namespace VideoCore
