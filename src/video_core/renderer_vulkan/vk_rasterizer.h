// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "common/bit_field.h"
#include "common/common_types.h"
#include "common/vector_math.h"
#include "core/hw/gpu.h"
#include "video_core/pica_state.h"
#include "video_core/pica_types.h"
#include "video_core/rasterizer_interface.h"
#include "video_core/regs_framebuffer.h"
#include "video_core/regs_lighting.h"
#include "video_core/regs_rasterizer.h"
#include "video_core/regs_texturing.h"
#include "video_core/shader/shader.h"
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/renderer_vulkan/vk_rasterizer_cache.h"

namespace Frontend {
class EmuWindow;
}

namespace Vulkan {

enum class UniformBindings : u32 { Common, VS, GS };

struct LightSrc {
    alignas(16) glm::vec3 specular_0;
    alignas(16) glm::vec3 specular_1;
    alignas(16) glm::vec3 diffuse;
    alignas(16) glm::vec3 ambient;
    alignas(16) glm::vec3 position;
    alignas(16) glm::vec3 spot_direction; // negated
    float dist_atten_bias;
    float dist_atten_scale;
};

/// Uniform structure for the Uniform Buffer Object, all vectors must be 16-byte aligned
// NOTE: Always keep a vec4 at the end. The GL spec is not clear wether the alignment at
//       the end of a uniform block is included in UNIFORM_BLOCK_DATA_SIZE or not.
//       Not following that rule will cause problems on some AMD drivers.
struct UniformData {
    int framebuffer_scale;
    int alphatest_ref;
    float depth_scale;
    float depth_offset;
    float shadow_bias_constant;
    float shadow_bias_linear;
    int scissor_x1;
    int scissor_y1;
    int scissor_x2;
    int scissor_y2;
    int fog_lut_offset;
    int proctex_noise_lut_offset;
    int proctex_color_map_offset;
    int proctex_alpha_map_offset;
    int proctex_lut_offset;
    int proctex_diff_lut_offset;
    float proctex_bias;
    int shadow_texture_bias;
    alignas(16) glm::ivec4 lighting_lut_offset[Pica::LightingRegs::NumLightingSampler / 4];
    alignas(16) glm::vec3 fog_color;
    alignas(8) glm::vec2 proctex_noise_f;
    alignas(8) glm::vec2 proctex_noise_a;
    alignas(8) glm::vec2 proctex_noise_p;
    alignas(16) glm::vec3 lighting_global_ambient;
    LightSrc light_src[8];
    alignas(16) glm::vec4 const_color[6]; // A vec4 color for each of the six tev stages
    alignas(16) glm::vec4 tev_combiner_buffer_color;
    alignas(16) glm::vec4 clip_coef;
};

static_assert(
    sizeof(UniformData) == 0x4F0,
    "The size of the UniformData structure has changed, update the structure in the shader");
static_assert(sizeof(UniformData) < 16384,
              "UniformData structure must be less than 16kb as per the OpenGL spec");

/// Uniform struct for the Uniform Buffer Object that contains PICA vertex/geometry shader uniforms.
// NOTE: the same rule from UniformData also applies here.
struct PicaUniformsData {
    void SetFromRegs(const Pica::ShaderRegs& regs, const Pica::Shader::ShaderSetup& setup);

    struct BoolAligned {
        alignas(16) int b;
    };

    std::array<BoolAligned, 16> bools;
    alignas(16) std::array<glm::uvec4, 4> i;
    alignas(16) std::array<glm::vec4, 96> f;
};

struct VSUniformData {
    PicaUniformsData uniforms;
};
static_assert(
    sizeof(VSUniformData) == 1856,
    "The size of the VSUniformData structure has changed, update the structure in the shader");
static_assert(sizeof(VSUniformData) < 16384,
              "VSUniformData structure must be less than 16kb as per the OpenGL spec");

struct ScreenInfo;

class RasterizerVulkan : public VideoCore::RasterizerInterface {
public:
    explicit RasterizerVulkan(Frontend::EmuWindow& emu_window);
    ~RasterizerVulkan() override;

    void LoadDiskResources(const std::atomic_bool& stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void AddTriangle(const Pica::Shader::OutputVertex& v0, const Pica::Shader::OutputVertex& v1,
                     const Pica::Shader::OutputVertex& v2) override;
    void DrawTriangles() override;
    void NotifyPicaRegisterChanged(u32 id) override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const GPU::Regs::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, Vulkan::ScreenInfo& screen_info) override;
    bool AccelerateDrawBatch(bool is_indexed) override { return false; }

    /// Syncs entire status to match PICA registers
    void SyncEntireState() override;

private:
    /// Syncs the clip enabled status to match the PICA register
    void SyncClipEnabled();

    /// Syncs the clip coefficients to match the PICA register
    void SyncClipCoef();

    /// Sets the OpenGL shader in accordance with the current PICA register state
    void SetShader();

    /// Syncs the cull mode to match the PICA register
    void SyncCullMode();

    /// Syncs the depth scale to match the PICA register
    void SyncDepthScale();

    /// Syncs the depth offset to match the PICA register
    void SyncDepthOffset();

    /// Syncs the blend enabled status to match the PICA register
    void SyncBlendEnabled();

    /// Syncs the blend functions to match the PICA register
    void SyncBlendFuncs();

    /// Syncs the blend color to match the PICA register
    void SyncBlendColor();

    /// Syncs the fog states to match the PICA register
    void SyncFogColor();

    /// Sync the procedural texture noise configuration to match the PICA register
    void SyncProcTexNoise();

    /// Sync the procedural texture bias configuration to match the PICA register
    void SyncProcTexBias();

    /// Syncs the alpha test states to match the PICA register
    void SyncAlphaTest();

    /// Syncs the logic op states to match the PICA register
    void SyncLogicOp();

    /// Syncs the color write mask to match the PICA register state
    void SyncColorWriteMask();

    /// Syncs the stencil write mask to match the PICA register state
    void SyncStencilWriteMask();

    /// Syncs the depth write mask to match the PICA register state
    void SyncDepthWriteMask();

    /// Syncs the stencil test states to match the PICA register
    void SyncStencilTest();

    /// Syncs the depth test states to match the PICA register
    void SyncDepthTest();

    /// Syncs the TEV combiner color buffer to match the PICA register
    void SyncCombinerColor();

    /// Syncs the TEV constant color to match the PICA register
    void SyncTevConstColor(std::size_t tev_index, const Pica::TexturingRegs::TevStageConfig& tev_stage);

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

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    struct VertexArrayInfo {
        u32 vs_input_index_min;
        u32 vs_input_index_max;
        u32 vs_input_size;
    };

private:
    RasterizerCacheVulkan res_cache;
    std::vector<HardwareVertex> vertex_batch;
    bool shader_dirty = true;

    struct {
        UniformData data;
        std::array<bool, Pica::LightingRegs::NumLightingSampler> lighting_lut_dirty;
        bool lighting_lut_dirty_any;
        bool fog_lut_dirty;
        bool proctex_noise_lut_dirty;
        bool proctex_color_map_dirty;
        bool proctex_alpha_map_dirty;
        bool proctex_lut_dirty;
        bool proctex_diff_lut_dirty;
        bool dirty;
    } uniform_block_data = {};

    // They shall be big enough for about one frame.
    static constexpr std::size_t VERTEX_BUFFER_SIZE = 64 * 1024 * 1024;
    static constexpr std::size_t INDEX_BUFFER_SIZE = 16 * 1024 * 1024;
    static constexpr std::size_t UNIFORM_BUFFER_SIZE = 2 * 1024 * 1024;
    static constexpr std::size_t TEXTURE_BUFFER_SIZE = 1 * 1024 * 1024;

    Buffer vertex_buffer, index_buffer;
    StreamBuffer uniform_buffer, texture_buffer_lut_lf, texture_buffer_lut;

    u32 uniform_buffer_alignment;
    u32 uniform_size_aligned_vs, uniform_size_aligned_fs;

    std::array<std::array<glm::vec2, 256>,
               Pica::LightingRegs::NumLightingSampler> lighting_lut_data{};
    std::array<glm::vec2, 128> fog_lut_data{};
    std::array<glm::vec2, 128> proctex_noise_lut_data{};
    std::array<glm::vec2, 128> proctex_color_map_data{};
    std::array<glm::vec2, 128> proctex_alpha_map_data{};
    std::array<glm::vec4, 256> proctex_lut_data{};
    std::array<glm::vec4, 256> proctex_diff_lut_data{};

    bool allow_shadow{};
};

} // namespace OpenGL
