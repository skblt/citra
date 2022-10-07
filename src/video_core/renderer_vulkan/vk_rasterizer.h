// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "common/vector_math.h"
#include "core/hw/gpu.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/regs_lighting.h"
#include "video_core/regs_texturing.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"
#include "video_core/shader/shader.h"
#include "video_core/shader/shader_uniforms.h"

namespace Frontend {
class EmuWindow;
}

namespace Vulkan {

struct ScreenInfo;

class Instance;
class TaskScheduler;
class RenderpassCache;

struct SamplerInfo {
    using TextureConfig = Pica::TexturingRegs::TextureConfig;
    TextureConfig::TextureFilter mag_filter;
    TextureConfig::TextureFilter min_filter;
    TextureConfig::TextureFilter mip_filter;
    TextureConfig::WrapMode wrap_s;
    TextureConfig::WrapMode wrap_t;
    u32 border_color = 0;
    float lod_min = 0;
    float lod_max = 0;
    float lod_bias = 0;

    // TODO(wwylele): remove this once mipmap for cube is implemented
    bool supress_mipmap_for_cube = false;

    auto operator<=>(const SamplerInfo&) const noexcept = default;
};

struct FramebufferInfo {
    vk::ImageView color;
    vk::ImageView depth;
    vk::RenderPass renderpass;
    u32 width = 1;
    u32 height = 1;

    auto operator<=>(const FramebufferInfo&) const noexcept = default;
};

} // namespace Vulkan

namespace std {
template <>
struct hash<Vulkan::SamplerInfo> {
    std::size_t operator()(const Vulkan::SamplerInfo& info) const noexcept {
        return Common::ComputeHash64(&info, sizeof(Vulkan::SamplerInfo));
    }
};

template <>
struct hash<Vulkan::FramebufferInfo> {
    std::size_t operator()(const Vulkan::FramebufferInfo& info) const noexcept {
        return Common::ComputeHash64(&info, sizeof(Vulkan::FramebufferInfo));
    }
};
} // namespace std

namespace Vulkan {

class RasterizerVulkan : public VideoCore::RasterizerAccelerated {
    friend class RendererVulkan;

public:
    explicit RasterizerVulkan(Frontend::EmuWindow& emu_window, const Instance& instance,
                              TaskScheduler& scheduler, TextureRuntime& runtime,
                              RenderpassCache& renderpass_cache);
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
    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const GPU::Regs::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed) override;

    /// Syncs entire status to match PICA registers
    void SyncEntireState() override;

    /// Sync fixed function pipeline state
    void SyncFixedState();

    /// Flushes all rasterizer owned buffers
    void FlushBuffers();

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

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed);

    struct VertexArrayInfo {
        u32 vs_input_index_min;
        u32 vs_input_index_max;
        u32 vs_input_size;
    };

    /// Retrieve the range and the size of the input vertex
    VertexArrayInfo AnalyzeVertexArray(bool is_indexed);

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray(u32 vs_input_size, u32 vs_input_index_min, u32 vs_input_index_max);

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

    /// Creates a new sampler object
    vk::Sampler CreateSampler(const SamplerInfo& info);

    /// Creates a new Vulkan framebuffer object
    vk::Framebuffer CreateFramebuffer(const FramebufferInfo& info);

private:
    const Instance& instance;
    TaskScheduler& scheduler;
    TextureRuntime& runtime;
    RenderpassCache& renderpass_cache;
    RasterizerCache res_cache;
    PipelineCache pipeline_cache;
    bool shader_dirty = true;

    /// Structure that the hardware rendered vertices are composed of
    struct HardwareVertex {
        HardwareVertex() = default;
        HardwareVertex(const Pica::Shader::OutputVertex& v, bool flip_quaternion);

        constexpr static VertexLayout GetVertexLayout();

        Common::Vec4f position;
        Common::Vec4f color;
        Common::Vec2f tex_coord0;
        Common::Vec2f tex_coord1;
        Common::Vec2f tex_coord2;
        float tex_coord0_w;
        Common::Vec4f normquat;
        Common::Vec3f view;
    };

    std::vector<HardwareVertex> vertex_batch;
    ImageAlloc default_texture;
    vk::Sampler default_sampler;

    struct {
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
    } uniform_block_data = {};

    std::array<bool, 16> hw_enabled_attributes{};

    std::array<SamplerInfo, 3> texture_samplers;
    SamplerInfo texture_cube_sampler;
    std::unordered_map<SamplerInfo, vk::Sampler> samplers;
    std::unordered_map<FramebufferInfo, vk::Framebuffer> framebuffers;

    StreamBuffer vertex_buffer;
    StreamBuffer uniform_buffer;
    StreamBuffer index_buffer;
    StreamBuffer texture_buffer;
    StreamBuffer texture_lf_buffer;
    PipelineInfo pipeline_info;
    std::size_t uniform_buffer_alignment;
    std::size_t uniform_size_aligned_vs;
    std::size_t uniform_size_aligned_fs;

    std::array<std::array<Common::Vec2f, 256>, Pica::LightingRegs::NumLightingSampler>
        lighting_lut_data{};
    std::array<Common::Vec2f, 128> fog_lut_data{};
    std::array<Common::Vec2f, 128> proctex_noise_lut_data{};
    std::array<Common::Vec2f, 128> proctex_color_map_data{};
    std::array<Common::Vec2f, 128> proctex_alpha_map_data{};
    std::array<Common::Vec4f, 256> proctex_lut_data{};
    std::array<Common::Vec4f, 256> proctex_diff_lut_data{};
};

} // namespace Vulkan
