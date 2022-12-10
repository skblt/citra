// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "core/hw/gpu.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_stream_buffer.h"
#include "video_core/renderer_vulkan/vk_texture_runtime.h"

namespace Memory {
class MemorySystem;
}

namespace Frontend {
class EmuWindow;
}

namespace Vulkan {

struct ScreenInfo;

class Instance;
class Scheduler;
class RenderpassCache;
class DescriptorManager;

class RasterizerVulkan : public VideoCore::RasterizerAccelerated {
    friend class RendererVulkan;

public:
    explicit RasterizerVulkan(Core::System& system, Frontend::EmuWindow& emu_window,
                              const Instance& instance, Scheduler& scheduler,
                              DescriptorManager& desc_manager, TextureRuntime& runtime,
                              RenderpassCache& renderpass_cache);
    ~RasterizerVulkan() override;

    void LoadDiskResources(const std::atomic_bool& stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void DrawTriangles() override;
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
    void NotifyFixedFunctionPicaRegisterChanged(u32 id) override;

    /// Syncs the clip enabled status to match the PICA register
    void SyncClipEnabled();

    /// Syncs the clip coefficients to match the PICA register
    void SyncClipCoef();

    /// Syncs the cull mode to match the PICA register
    void SyncCullMode();

    /// Syncs the blend enabled status to match the PICA register
    void SyncBlendEnabled();

    /// Syncs the blend functions to match the PICA register
    void SyncBlendFuncs();

    /// Syncs the blend color to match the PICA register
    void SyncBlendColor();

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

    /// Syncs and uploads the lighting, fog and proctex LUTs
    void SyncAndUploadLUTs();
    void SyncAndUploadLUTsLF();

    /// Upload the uniform blocks to the uniform buffer object
    void UploadUniforms(bool accelerate_draw);

    /// Generic draw function for DrawTriangles and AccelerateDrawBatch
    bool Draw(bool accelerate, bool is_indexed);

    /// Internal implementation for AccelerateDrawBatch
    bool AccelerateDrawBatchInternal(bool is_indexed);

    /// Setup vertex array for AccelerateDrawBatch
    void SetupVertexArray(u32 vs_input_size, u32 vs_input_index_min, u32 vs_input_index_max);

    /// Setup the fixed attribute emulation in vulkan
    void SetupFixedAttribs();

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

    /// Setup texture units for drawing
    void SetupTextureUnits();

    /// Creates the vertex layout struct used for software shader pipelines
    void MakeSoftwareVertexLayout();

private:
    const Instance& instance;
    Scheduler& scheduler;
    TextureRuntime& runtime;
    RenderpassCache& renderpass_cache;
    DescriptorManager& desc_manager;
    RasterizerCache res_cache;
    PipelineCache pipeline_cache;

    VertexLayout software_layout;
    std::array<u64, 16> binding_offsets{};
    std::array<bool, 16> enable_attributes{};

    std::size_t uniform_buffer_alignment;
    std::size_t uniform_size_aligned_vs;
    std::size_t uniform_size_aligned_fs;
    StreamBuffer vertex_buffer;
    StreamBuffer uniform_buffer;
    StreamBuffer index_buffer;
    StreamBuffer texture_buffer;
    StreamBuffer texture_lf_buffer;
    PipelineInfo pipeline_info;
};

} // namespace Vulkan
