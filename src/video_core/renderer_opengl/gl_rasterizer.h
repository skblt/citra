// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "core/hw/gpu.h"
#include "video_core/rasterizer_accelerated.h"
#include "video_core/renderer_opengl/gl_shader_manager.h"
#include "video_core/renderer_opengl/gl_state.h"
#include "video_core/renderer_opengl/gl_stream_buffer.h"
#include "video_core/renderer_opengl/gl_texture_runtime.h"

namespace Frontend {
class EmuWindow;
}

namespace OpenGL {

struct ScreenInfo;

class Driver;
class ShaderProgramManager;

class RasterizerOpenGL : public VideoCore::RasterizerAccelerated {
public:
    explicit RasterizerOpenGL(Memory::MemorySystem& memory, Frontend::EmuWindow& emu_window,
                              Driver& driver);
    ~RasterizerOpenGL() override;

    void LoadDiskResources(const std::atomic_bool& stop_loading,
                           const VideoCore::DiskResourceLoadCallback& callback) override;

    void DrawTriangles() override;
    void FlushAll() override;
    void FlushRegion(PAddr addr, u32 size) override;
    void InvalidateRegion(PAddr addr, u32 size) override;
    void FlushAndInvalidateRegion(PAddr addr, u32 size) override;
    void ClearAll(bool flush) override;
    bool AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) override;
    bool AccelerateFill(const GPU::Regs::MemoryFillConfig& config) override;
    bool AccelerateDisplay(const GPU::Regs::FramebufferConfig& config, PAddr framebuffer_addr,
                           u32 pixel_stride, ScreenInfo& screen_info);
    bool AccelerateDrawBatch(bool is_indexed) override;

    void SyncFixedState() override;

private:
    void NotifyFixedFunctionPicaRegisterChanged(u32 id) override;

    /// Syncs the clip enabled status to match the PICA register
    void SyncClipEnabled();

    /// Sets the OpenGL shader in accordance with the current PICA register state
    void SetShader();

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
    void SetupVertexArray(u8* array_ptr, GLintptr buffer_offset, GLuint vs_input_index_min,
                          GLuint vs_input_index_max);

    /// Setup vertex shader for AccelerateDrawBatch
    bool SetupVertexShader();

    /// Setup geometry shader for AccelerateDrawBatch
    bool SetupGeometryShader();

private:
    Driver& driver;
    OpenGLState state;
    TextureRuntime runtime;
    RasterizerCache res_cache;
    ShaderProgramManager shader_program_manager;

    OGLVertexArray sw_vao; // VAO for software shader draw
    OGLVertexArray hw_vao; // VAO for hardware shader / accelerate draw
    std::array<bool, 16> hw_vao_enabled_attributes{};

    OGLTexture default_texture;
    StreamBuffer vertex_buffer;
    StreamBuffer uniform_buffer;
    StreamBuffer index_buffer;
    StreamBuffer texture_buffer;
    StreamBuffer texture_lf_buffer;
    OGLFramebuffer framebuffer;
    GLint uniform_buffer_alignment;
    std::size_t uniform_size_aligned_vs;
    std::size_t uniform_size_aligned_fs;

    OGLTexture texture_buffer_lut_lf;
    OGLTexture texture_buffer_lut_rg;
    OGLTexture texture_buffer_lut_rgba;
};

} // namespace OpenGL
