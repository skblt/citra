// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "common/alignment.h"
#include "common/logging/log.h"
#include "common/math_util.h"
#include "common/microprofile.h"
#include "video_core/pica_state.h"
#include "video_core/regs_framebuffer.h"
#include "video_core/regs_rasterizer.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/video_core.h"

namespace Vulkan {

MICROPROFILE_DEFINE(OpenGL_VAO, "OpenGL", "Vertex Array Setup", MP_RGB(255, 128, 0));
MICROPROFILE_DEFINE(OpenGL_VS, "OpenGL", "Vertex Shader Setup", MP_RGB(192, 128, 128));
MICROPROFILE_DEFINE(OpenGL_GS, "OpenGL", "Geometry Shader Setup", MP_RGB(128, 192, 128));
MICROPROFILE_DEFINE(OpenGL_Drawing, "OpenGL", "Drawing", MP_RGB(128, 128, 192));
MICROPROFILE_DEFINE(OpenGL_Blits, "OpenGL", "Blits", MP_RGB(100, 100, 255));
MICROPROFILE_DEFINE(OpenGL_CacheManagement, "OpenGL", "Cache Mgmt", MP_RGB(100, 255, 100));

RasterizerVulkan::HardwareVertex::HardwareVertex(const Pica::Shader::OutputVertex& v,
                                                 bool flip_quaternion) {
    position[0] = v.pos.x.ToFloat32();
    position[1] = v.pos.y.ToFloat32();
    position[2] = v.pos.z.ToFloat32();
    position[3] = v.pos.w.ToFloat32();
    color[0] = v.color.x.ToFloat32();
    color[1] = v.color.y.ToFloat32();
    color[2] = v.color.z.ToFloat32();
    color[3] = v.color.w.ToFloat32();
    tex_coord0[0] = v.tc0.x.ToFloat32();
    tex_coord0[1] = v.tc0.y.ToFloat32();
    tex_coord1[0] = v.tc1.x.ToFloat32();
    tex_coord1[1] = v.tc1.y.ToFloat32();
    tex_coord2[0] = v.tc2.x.ToFloat32();
    tex_coord2[1] = v.tc2.y.ToFloat32();
    tex_coord0_w = v.tc0_w.ToFloat32();
    normquat[0] = v.quat.x.ToFloat32();
    normquat[1] = v.quat.y.ToFloat32();
    normquat[2] = v.quat.z.ToFloat32();
    normquat[3] = v.quat.w.ToFloat32();
    view[0] = v.view.x.ToFloat32();
    view[1] = v.view.y.ToFloat32();
    view[2] = v.view.z.ToFloat32();

    if (flip_quaternion) {
        normquat = -normquat;
    }
}

/**
 * This maps to the following layout in GLSL code:
 *  layout(location = 0) in vec4 vert_position;
 *  layout(location = 1) in vec4 vert_color;
 *  layout(location = 2) in vec2 vert_texcoord0;
 *  layout(location = 3) in vec2 vert_texcoord1;
 *  layout(location = 4) in vec2 vert_texcoord2;
 *  layout(location = 5) in float vert_texcoord0_w;
 *  layout(location = 6) in vec4 vert_normquat;
 *  layout(location = 7) in vec3 vert_view;
 */
constexpr VertexLayout RasterizerVulkan::HardwareVertex::GetVertexLayout() {
    VertexLayout layout{};
    layout.attribute_count = 8;
    layout.binding_count = 1;

    // Define binding
    layout.bindings[0].binding.Assign(0);
    layout.bindings[0].fixed.Assign(0);
    layout.bindings[0].stride.Assign(sizeof(HardwareVertex));

    // Define attributes
    constexpr std::array sizes = {4, 4, 2, 2, 2, 1, 4, 3};
    u32 offset = 0;

    for (u32 loc = 0; loc < 8; loc++) {
        VertexAttribute& attribute = layout.attributes[loc];
        attribute.binding.Assign(0);
        attribute.location.Assign(loc);
        attribute.offset.Assign(offset);
        attribute.type.Assign(AttribType::Float);
        attribute.size.Assign(sizes[loc]);
        offset += sizes[loc] * sizeof(float);
    }

    return layout;
}

constexpr u32 VERTEX_BUFFER_SIZE = 128 * 1024 * 1024;
constexpr u32 INDEX_BUFFER_SIZE = 8 * 1024 * 1024;
constexpr u32 UNIFORM_BUFFER_SIZE = 16 * 1024 * 1024;
constexpr u32 TEXTURE_BUFFER_SIZE = 16 * 1024 * 1024;

constexpr std::array TEXTURE_BUFFER_LF_FORMATS = {vk::Format::eR32G32Sfloat};

constexpr std::array TEXTURE_BUFFER_FORMATS = {vk::Format::eR32G32Sfloat,
                                               vk::Format::eR32G32B32A32Sfloat};

RasterizerVulkan::RasterizerVulkan(Frontend::EmuWindow& emu_window, const Instance& instance,
                                   TaskScheduler& scheduler, TextureRuntime& runtime,
                                   RenderpassCache& renderpass_cache)
    : instance{instance}, scheduler{scheduler}, runtime{runtime},
      renderpass_cache{renderpass_cache}, res_cache{*this, runtime},
      pipeline_cache{instance, scheduler, renderpass_cache},
      vertex_buffer{
          instance, scheduler, VERTEX_BUFFER_SIZE, vk::BufferUsageFlagBits::eVertexBuffer, {}},
      uniform_buffer{
          instance, scheduler, UNIFORM_BUFFER_SIZE, vk::BufferUsageFlagBits::eUniformBuffer, {}},
      index_buffer{
          instance, scheduler, INDEX_BUFFER_SIZE, vk::BufferUsageFlagBits::eIndexBuffer, {}},
      texture_buffer{instance, scheduler, TEXTURE_BUFFER_SIZE,
                     vk::BufferUsageFlagBits::eUniformTexelBuffer, TEXTURE_BUFFER_FORMATS},
      texture_lf_buffer{instance, scheduler, TEXTURE_BUFFER_SIZE,
                        vk::BufferUsageFlagBits::eUniformTexelBuffer, TEXTURE_BUFFER_LF_FORMATS} {

    // Create a 1x1 clear texture to use in the NULL case,
    default_texture =
        runtime.Allocate(1, 1, VideoCore::PixelFormat::RGBA8, VideoCore::TextureType::Texture2D);
    runtime.Transition(scheduler.GetUploadCommandBuffer(), default_texture,
                       vk::ImageLayout::eShaderReadOnlyOptimal, 0, 1);

    uniform_block_data.lighting_lut_dirty.fill(true);

    uniform_buffer_alignment = instance.UniformMinAlignment();
    uniform_size_aligned_vs =
        Common::AlignUp<std::size_t>(sizeof(Pica::Shader::VSUniformData), uniform_buffer_alignment);
    uniform_size_aligned_fs =
        Common::AlignUp<std::size_t>(sizeof(Pica::Shader::UniformData), uniform_buffer_alignment);

    // Define vertex layout for software shaders
    pipeline_info.vertex_layout = HardwareVertex::GetVertexLayout();

    const SamplerInfo default_sampler_info = {
        .mag_filter = Pica::TexturingRegs::TextureConfig::TextureFilter::Linear,
        .min_filter = Pica::TexturingRegs::TextureConfig::TextureFilter::Linear,
        .mip_filter = Pica::TexturingRegs::TextureConfig::TextureFilter::Linear,
        .wrap_s = Pica::TexturingRegs::TextureConfig::WrapMode::ClampToBorder,
        .wrap_t = Pica::TexturingRegs::TextureConfig::WrapMode::ClampToBorder};

    default_sampler = CreateSampler(default_sampler_info);

    // Since we don't have access to VK_EXT_descriptor_indexing we need to intiallize
    // all descriptor sets even the ones we don't use. Use default_texture for this
    const u32 vs_uniform_size = sizeof(Pica::Shader::VSUniformData);
    const u32 fs_uniform_size = sizeof(Pica::Shader::UniformData);
    pipeline_cache.BindBuffer(0, uniform_buffer.GetHandle(), 0, vs_uniform_size);
    pipeline_cache.BindBuffer(1, uniform_buffer.GetHandle(), vs_uniform_size, fs_uniform_size);
    pipeline_cache.BindTexelBuffer(2, texture_lf_buffer.GetView());
    pipeline_cache.BindTexelBuffer(3, texture_buffer.GetView(0));
    pipeline_cache.BindTexelBuffer(4, texture_buffer.GetView(1));

    for (u32 i = 0; i < 4; i++) {
        pipeline_cache.BindTexture(i, default_texture.image_view);
        pipeline_cache.BindSampler(i, default_sampler);
    }

    for (u32 i = 0; i < 7; i++) {
        pipeline_cache.BindStorageImage(i, default_texture.image_view);
    }

    // Explicitly call the derived version to avoid warnings about calling virtual
    // methods in the constructor
    RasterizerVulkan::SyncEntireState();
}

RasterizerVulkan::~RasterizerVulkan() {
    renderpass_cache.ExitRenderpass();
    scheduler.Submit(SubmitMode::Flush | SubmitMode::Shutdown);

    vk::Device device = instance.GetDevice();

    for (auto& [key, sampler] : samplers) {
        device.destroySampler(sampler);
    }

    for (auto& [key, framebuffer] : framebuffers) {
        device.destroyFramebuffer(framebuffer);
    }

    const VideoCore::HostTextureTag tag = {
        .format = VideoCore::PixelFormat::RGBA8, .width = 1, .height = 1};

    runtime.Recycle(tag, std::move(default_texture));
    device.destroySampler(default_sampler);
}

void RasterizerVulkan::LoadDiskResources(const std::atomic_bool& stop_loading,
                                         const VideoCore::DiskResourceLoadCallback& callback) {
    // shader_program_manager->LoadDiskCache(stop_loading, callback);
}

void RasterizerVulkan::SyncEntireState() {
    // Sync fixed function Vulkan state
    SyncFixedState();

    // Sync uniforms
    SyncClipCoef();
    SyncDepthScale();
    SyncDepthOffset();
    SyncAlphaTest();
    SyncCombinerColor();
    auto& tev_stages = Pica::g_state.regs.texturing.GetTevStages();
    for (std::size_t index = 0; index < tev_stages.size(); ++index)
        SyncTevConstColor(index, tev_stages[index]);

    SyncGlobalAmbient();
    for (unsigned light_index = 0; light_index < 8; light_index++) {
        SyncLightSpecular0(light_index);
        SyncLightSpecular1(light_index);
        SyncLightDiffuse(light_index);
        SyncLightAmbient(light_index);
        SyncLightPosition(light_index);
        SyncLightDistanceAttenuationBias(light_index);
        SyncLightDistanceAttenuationScale(light_index);
    }

    SyncFogColor();
    SyncProcTexNoise();
    SyncProcTexBias();
    SyncShadowBias();
    SyncShadowTextureBias();
}

void RasterizerVulkan::SyncFixedState() {
    SyncClipEnabled();
    SyncCullMode();
    SyncBlendEnabled();
    SyncBlendFuncs();
    SyncBlendColor();
    SyncLogicOp();
    SyncStencilTest();
    SyncDepthTest();
    SyncColorWriteMask();
    SyncStencilWriteMask();
    SyncDepthWriteMask();
}

/**
 * This is a helper function to resolve an issue when interpolating opposite quaternions. See below
 * for a detailed description of this issue (yuriks):
 *
 * For any rotation, there are two quaternions Q, and -Q, that represent the same rotation. If you
 * interpolate two quaternions that are opposite, instead of going from one rotation to another
 * using the shortest path, you'll go around the longest path. You can test if two quaternions are
 * opposite by checking if Dot(Q1, Q2) < 0. In that case, you can flip either of them, therefore
 * making Dot(Q1, -Q2) positive.
 *
 * This solution corrects this issue per-vertex before passing the quaternions to OpenGL. This is
 * correct for most cases but can still rotate around the long way sometimes. An implementation
 * which did `lerp(lerp(Q1, Q2), Q3)` (with proper weighting), applying the dot product check
 * between each step would work for those cases at the cost of being more complex to implement.
 *
 * Fortunately however, the 3DS hardware happens to also use this exact same logic to work around
 * these issues, making this basic implementation actually more accurate to the hardware.
 */
static bool AreQuaternionsOpposite(Common::Vec4<Pica::float24> qa, Common::Vec4<Pica::float24> qb) {
    Common::Vec4f a{qa.x.ToFloat32(), qa.y.ToFloat32(), qa.z.ToFloat32(), qa.w.ToFloat32()};
    Common::Vec4f b{qb.x.ToFloat32(), qb.y.ToFloat32(), qb.z.ToFloat32(), qb.w.ToFloat32()};

    return (Common::Dot(a, b) < 0.f);
}

void RasterizerVulkan::AddTriangle(const Pica::Shader::OutputVertex& v0,
                                   const Pica::Shader::OutputVertex& v1,
                                   const Pica::Shader::OutputVertex& v2) {
    vertex_batch.emplace_back(v0, false);
    vertex_batch.emplace_back(v1, AreQuaternionsOpposite(v0.quat, v1.quat));
    vertex_batch.emplace_back(v2, AreQuaternionsOpposite(v0.quat, v2.quat));
}

static constexpr std::array vs_attrib_types = {
    AttribType::Byte,  // VertexAttributeFormat::BYTE
    AttribType::Ubyte, // VertexAttributeFormat::UBYTE
    AttribType::Short, // VertexAttributeFormat::SHORT
    AttribType::Float  // VertexAttributeFormat::FLOAT
};

struct VertexArrayInfo {
    u32 vs_input_index_min;
    u32 vs_input_index_max;
    u32 vs_input_size;
};

RasterizerVulkan::VertexArrayInfo RasterizerVulkan::AnalyzeVertexArray(bool is_indexed) {
    const auto& regs = Pica::g_state.regs;
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;

    u32 vertex_min;
    u32 vertex_max;
    if (is_indexed) {
        const auto& index_info = regs.pipeline.index_array;
        const PAddr address = vertex_attributes.GetPhysicalBaseAddress() + index_info.offset;
        const u8* index_address_8 = VideoCore::g_memory->GetPhysicalPointer(address);
        const u16* index_address_16 = reinterpret_cast<const u16*>(index_address_8);
        const bool index_u16 = index_info.format != 0;

        vertex_min = 0xFFFF;
        vertex_max = 0;
        const u32 size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);
        res_cache.FlushRegion(address, size, nullptr);
        for (u32 index = 0; index < regs.pipeline.num_vertices; ++index) {
            const u32 vertex = index_u16 ? index_address_16[index] : index_address_8[index];
            vertex_min = std::min(vertex_min, vertex);
            vertex_max = std::max(vertex_max, vertex);
        }
    } else {
        vertex_min = regs.pipeline.vertex_offset;
        vertex_max = regs.pipeline.vertex_offset + regs.pipeline.num_vertices - 1;
    }

    const u32 vertex_num = vertex_max - vertex_min + 1;
    u32 vs_input_size = 0;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count != 0) {
            vs_input_size += loader.byte_count * vertex_num;
        }
    }

    return {vertex_min, vertex_max, vs_input_size};
}

void RasterizerVulkan::SetupVertexArray(u32 vs_input_size, u32 vs_input_index_min,
                                        u32 vs_input_index_max) {
    auto [array_ptr, array_offset, invalidate] = vertex_buffer.Map(vs_input_size, 4);

    // The Nintendo 3DS has 12 attribute loaders which are used to tell the GPU
    // how to interpret vertex data. The program firsts sets GPUREG_ATTR_BUF_BASE to the base
    // address containing the vertex array data. The data for each attribute loader (i) can be found
    // by adding GPUREG_ATTR_BUFi_OFFSET to the base address. Attribute loaders can be thought
    // as something analogous to Vulkan bindings. The user can store attributes in separate loaders
    // or interleave them in the same loader.
    const auto& regs = Pica::g_state.regs;
    const auto& vertex_attributes = regs.pipeline.vertex_attributes;
    PAddr base_address = vertex_attributes.GetPhysicalBaseAddress(); // GPUREG_ATTR_BUF_BASE

    std::array<bool, 16> enable_attributes{};
    VertexLayout layout{};

    u32 buffer_offset = array_offset;
    for (const auto& loader : vertex_attributes.attribute_loaders) {
        if (loader.component_count == 0 || loader.byte_count == 0) {
            continue;
        }

        // Analyze the attribute loader by checking which attributes it provides
        u32 offset = 0;
        for (u32 comp = 0; comp < loader.component_count && comp < 12; comp++) {
            u32 attribute_index = loader.GetComponent(comp);
            if (attribute_index < 12) {
                if (u32 size = vertex_attributes.GetNumElements(attribute_index); size != 0) {
                    offset = Common::AlignUp(
                        offset, vertex_attributes.GetElementSizeInBytes(attribute_index));

                    const u32 input_reg = regs.vs.GetRegisterForAttribute(attribute_index);
                    const u32 attrib_format =
                        static_cast<u32>(vertex_attributes.GetFormat(attribute_index));
                    const AttribType type = vs_attrib_types[attrib_format];

                    // Define the attribute
                    VertexAttribute& attribute = layout.attributes[layout.attribute_count++];
                    attribute.binding.Assign(layout.binding_count);
                    attribute.location.Assign(input_reg);
                    attribute.offset.Assign(offset);
                    attribute.type.Assign(type);
                    attribute.size.Assign(size);

                    enable_attributes[input_reg] = true;
                    offset += vertex_attributes.GetStride(attribute_index);
                }

            } else {
                // Attribute ids 12, 13, 14 and 15 signify 4, 8, 12 and 16-byte paddings
                // respectively
                offset = Common::AlignUp(offset, 4);
                offset += (attribute_index - 11) * 4;
            }
        }

        const PAddr data_addr =
            base_address + loader.data_offset + (vs_input_index_min * loader.byte_count);
        const u32 vertex_num = vs_input_index_max - vs_input_index_min + 1;
        u32 data_size = loader.byte_count * vertex_num;

        res_cache.FlushRegion(data_addr, data_size, nullptr);
        std::memcpy(array_ptr, VideoCore::g_memory->GetPhysicalPointer(data_addr), data_size);

        // Create the binding associated with this loader
        VertexBinding& binding = layout.bindings[layout.binding_count];
        binding.binding.Assign(layout.binding_count);
        binding.fixed.Assign(0);
        binding.stride.Assign(loader.byte_count);

        // Keep track of the binding offsets so we can bind the vertex buffer later
        binding_offsets[layout.binding_count++] = buffer_offset;
        data_size = Common::AlignUp(data_size, 16);
        array_ptr += data_size;
        buffer_offset += data_size;
    }

    // Reserve the last binding for fixed and default attributes
    // Place the default attrib at offset zero for easy access
    constexpr Common::Vec4f default_attrib = Common::MakeVec(0.f, 0.f, 0.f, 1.f);
    u32 offset = sizeof(Common::Vec4f);
    std::memcpy(array_ptr, default_attrib.AsArray(), sizeof(Common::Vec4f));
    array_ptr += sizeof(Common::Vec4f);

    // Find all fixed attributes and assign them to the last binding
    for (std::size_t i = 0; i < 16; i++) {
        if (vertex_attributes.IsDefaultAttribute(i)) {
            const u32 reg = regs.vs.GetRegisterForAttribute(i);
            if (!enable_attributes[reg]) {
                const auto& attr = Pica::g_state.input_default_attributes.attr[i];
                const std::array data = {attr.x.ToFloat32(), attr.y.ToFloat32(), attr.z.ToFloat32(),
                                         attr.w.ToFloat32()};

                const u32 data_size = sizeof(float) * static_cast<u32>(data.size());
                std::memcpy(array_ptr, data.data(), data_size);

                VertexAttribute& attribute = layout.attributes[layout.attribute_count++];
                attribute.binding.Assign(layout.binding_count);
                attribute.location.Assign(reg);
                attribute.offset.Assign(offset);
                attribute.type.Assign(AttribType::Float);
                attribute.size.Assign(4);

                offset += data_size;
                array_ptr += data_size;
                enable_attributes[reg] = true;
            }
        }
    }

    // Loop one more time to find unused attributes and assign them to the default one
    // This needs to happen because i = 2 might be assigned to location = 3 so the loop
    // above would skip setting it
    for (std::size_t i = 0; i < 16; i++) {
        // If the attribute is just disabled, shove the default attribute to avoid
        // errors if the shader ever decides to use it. The pipeline cache can discard
        // this if needed since it has access to the usage mask from the code generator
        if (!enable_attributes[i]) {
            VertexAttribute& attribute = layout.attributes[layout.attribute_count++];
            attribute.binding.Assign(layout.binding_count);
            attribute.location.Assign(i);
            attribute.offset.Assign(0);
            attribute.type.Assign(AttribType::Float);
            attribute.size.Assign(4);
        }
    }

    // Define the fixed+default binding
    VertexBinding& binding = layout.bindings[layout.binding_count];
    binding.binding.Assign(layout.binding_count);
    binding.fixed.Assign(1);
    binding.stride.Assign(offset);
    binding_offsets[layout.binding_count++] = buffer_offset;
    buffer_offset += offset;

    pipeline_info.vertex_layout = layout;
    vertex_buffer.Commit(buffer_offset - array_offset);

    std::array<vk::Buffer, 16> buffers;
    buffers.fill(vertex_buffer.GetHandle());

    // Bind the vertex buffer with all the bindings
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindVertexBuffers(0, layout.binding_count, buffers.data(),
                                     binding_offsets.data());
}

bool RasterizerVulkan::SetupVertexShader() {
    MICROPROFILE_SCOPE(OpenGL_VS);
    return pipeline_cache.UseProgrammableVertexShader(Pica::g_state.regs, Pica::g_state.vs,
                                                      pipeline_info.vertex_layout);
}

bool RasterizerVulkan::SetupGeometryShader() {
    MICROPROFILE_SCOPE(OpenGL_GS);
    const auto& regs = Pica::g_state.regs;

    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        LOG_ERROR(Render_OpenGL, "Accelerate draw doesn't support geometry shader");
        return false;
    }

    pipeline_cache.UseFixedGeometryShader(regs);
    return true;
}

bool RasterizerVulkan::AccelerateDrawBatch(bool is_indexed) {
    const auto& regs = Pica::g_state.regs;
    if (regs.pipeline.use_gs != Pica::PipelineRegs::UseGS::No) {
        if (regs.pipeline.gs_config.mode != Pica::PipelineRegs::GSMode::Point) {
            return false;
        }
        if (regs.pipeline.triangle_topology != Pica::PipelineRegs::TriangleTopology::Shader) {
            return false;
        }
    }

    return Draw(true, is_indexed);
}

bool RasterizerVulkan::AccelerateDrawBatchInternal(bool is_indexed) {
    const auto& regs = Pica::g_state.regs;

    auto [vs_input_index_min, vs_input_index_max, vs_input_size] = AnalyzeVertexArray(is_indexed);

    if (vs_input_size > VERTEX_BUFFER_SIZE) {
        LOG_WARNING(Render_Vulkan, "Too large vertex input size {}", vs_input_size);
        return false;
    }

    SetupVertexArray(vs_input_size, vs_input_index_min, vs_input_index_max);

    if (!SetupVertexShader()) {
        return false;
    }

    if (!SetupGeometryShader()) {
        return false;
    }

    pipeline_info.rasterization.topology.Assign(regs.pipeline.triangle_topology);
    pipeline_cache.BindPipeline(pipeline_info);

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    if (is_indexed) {
        bool index_u16 = regs.pipeline.index_array.format != 0;
        const u64 index_buffer_size = regs.pipeline.num_vertices * (index_u16 ? 2 : 1);

        if (index_buffer_size > INDEX_BUFFER_SIZE) {
            LOG_WARNING(Render_Vulkan, "Too large index input size {}", index_buffer_size);
            return false;
        }

        const u8* index_data = VideoCore::g_memory->GetPhysicalPointer(
            regs.pipeline.vertex_attributes.GetPhysicalBaseAddress() +
            regs.pipeline.index_array.offset);

        // Upload index buffer data to the GPU
        auto [index_ptr, index_offset, _] = index_buffer.Map(index_buffer_size, 4);
        std::memcpy(index_ptr, index_data, index_buffer_size);
        index_buffer.Commit(index_buffer_size);

        vk::IndexType index_type = index_u16 ? vk::IndexType::eUint16 : vk::IndexType::eUint8EXT;
        command_buffer.bindIndexBuffer(index_buffer.GetHandle(), index_offset, index_type);

        // Submit draw
        command_buffer.drawIndexed(regs.pipeline.num_vertices, 1, 0, -vs_input_index_min, 0);
    } else {
        command_buffer.draw(regs.pipeline.num_vertices, 1, 0, 0);
    }

    return true;
}

void RasterizerVulkan::DrawTriangles() {
    if (vertex_batch.empty()) {
        return;
    }

    Draw(false, false);
}

bool RasterizerVulkan::Draw(bool accelerate, bool is_indexed) {
    MICROPROFILE_SCOPE(OpenGL_Drawing);
    const auto& regs = Pica::g_state.regs;

    const bool shadow_rendering = regs.framebuffer.output_merger.fragment_operation_mode ==
                                  Pica::FramebufferRegs::FragmentOperationMode::Shadow;
    const bool has_stencil =
        regs.framebuffer.framebuffer.depth_format == Pica::FramebufferRegs::DepthFormat::D24S8;
    const bool write_color_fb = shadow_rendering || pipeline_info.blending.color_write_mask.Value();
    const bool write_depth_fb = pipeline_info.IsDepthWriteEnabled();
    const bool using_color_fb =
        regs.framebuffer.framebuffer.GetColorBufferPhysicalAddress() != 0 && write_color_fb;
    const bool using_depth_fb =
        !shadow_rendering && regs.framebuffer.framebuffer.GetDepthBufferPhysicalAddress() != 0 &&
        (write_depth_fb || regs.framebuffer.output_merger.depth_test_enable != 0 ||
         (has_stencil && pipeline_info.depth_stencil.stencil_test_enable));

    const auto viewport_rect_unscaled = Common::Rectangle<s32>{
        // These registers hold half-width and half-height, so must be multiplied by 2
        regs.rasterizer.viewport_corner.x,  // left
        regs.rasterizer.viewport_corner.y + // top
            static_cast<s32>(Pica::float24::FromRaw(regs.rasterizer.viewport_size_y).ToFloat32() *
                             2),
        regs.rasterizer.viewport_corner.x + // right
            static_cast<s32>(Pica::float24::FromRaw(regs.rasterizer.viewport_size_x).ToFloat32() *
                             2),
        regs.rasterizer.viewport_corner.y // bottom
    };

    auto [color_surface, depth_surface, surfaces_rect] =
        res_cache.GetFramebufferSurfaces(using_color_fb, using_depth_fb, viewport_rect_unscaled);

    pipeline_info.color_attachment =
        color_surface ? color_surface->pixel_format : VideoCore::PixelFormat::Invalid;
    pipeline_info.depth_attachment =
        depth_surface ? depth_surface->pixel_format : VideoCore::PixelFormat::Invalid;

    const u16 res_scale = color_surface != nullptr
                              ? color_surface->res_scale
                              : (depth_surface == nullptr ? 1u : depth_surface->res_scale);

    const VideoCore::Rect2D draw_rect = {
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.left) +
                                             viewport_rect_unscaled.left * res_scale,
                                         surfaces_rect.left, surfaces_rect.right)), // Left
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.bottom) +
                                             viewport_rect_unscaled.top * res_scale,
                                         surfaces_rect.bottom, surfaces_rect.top)), // Top
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.left) +
                                             viewport_rect_unscaled.right * res_scale,
                                         surfaces_rect.left, surfaces_rect.right)), // Right
        static_cast<u32>(std::clamp<s32>(static_cast<s32>(surfaces_rect.bottom) +
                                             viewport_rect_unscaled.bottom * res_scale,
                                         surfaces_rect.bottom, surfaces_rect.top))};

    if (uniform_block_data.data.framebuffer_scale != res_scale) {
        uniform_block_data.data.framebuffer_scale = res_scale;
        uniform_block_data.dirty = true;
    }

    // Scissor checks are window-, not viewport-relative, which means that if the cached texture
    // sub-rect changes, the scissor bounds also need to be updated.
    int scissor_x1 =
        static_cast<int>(surfaces_rect.left + regs.rasterizer.scissor_test.x1 * res_scale);
    int scissor_y1 =
        static_cast<int>(surfaces_rect.bottom + regs.rasterizer.scissor_test.y1 * res_scale);

    // x2, y2 have +1 added to cover the entire pixel area, otherwise you might get cracks when
    // scaling or doing multisampling.
    int scissor_x2 =
        static_cast<int>(surfaces_rect.left + (regs.rasterizer.scissor_test.x2 + 1) * res_scale);
    int scissor_y2 =
        static_cast<int>(surfaces_rect.bottom + (regs.rasterizer.scissor_test.y2 + 1) * res_scale);

    if (uniform_block_data.data.scissor_x1 != scissor_x1 ||
        uniform_block_data.data.scissor_x2 != scissor_x2 ||
        uniform_block_data.data.scissor_y1 != scissor_y1 ||
        uniform_block_data.data.scissor_y2 != scissor_y2) {

        uniform_block_data.data.scissor_x1 = scissor_x1;
        uniform_block_data.data.scissor_x2 = scissor_x2;
        uniform_block_data.data.scissor_y1 = scissor_y1;
        uniform_block_data.data.scissor_y2 = scissor_y2;
        uniform_block_data.dirty = true;
    }

    auto CheckBarrier = [this, &color_surface = color_surface](vk::ImageView image_view,
                                                               u32 texture_index) {
        if (color_surface && color_surface->alloc.image_view == image_view) {
            // auto temp_tex = backend->CreateTexture(texture->GetInfo());
            // temp_tex->CopyFrom(texture);
            pipeline_cache.BindTexture(texture_index, image_view);
        } else {
            pipeline_cache.BindTexture(texture_index, image_view);
        }
    };

    const auto BindCubeFace = [&](Pica::TexturingRegs::CubeFace face,
                                  Pica::Texture::TextureInfo& info) {
        info.physical_address = regs.texturing.GetCubePhysicalAddress(face);
        auto surface = res_cache.GetTextureSurface(info);

        const u32 binding = static_cast<u32>(face);
        if (surface != nullptr) {
            pipeline_cache.BindStorageImage(binding, surface->alloc.image_view);
        } else {
            pipeline_cache.BindStorageImage(binding, default_texture.image_view);
        }
    };

    const auto BindSampler = [&](u32 binding, SamplerInfo& info,
                                 const Pica::TexturingRegs::TextureConfig& config) {
        // TODO(GPUCode): Cubemaps don't contain any mipmaps for now, so sampling from them returns
        // nothing Always sample from the base level until mipmaps for texture cubes are implemented
        // NOTE: There are no Vulkan filter modes that directly correspond to OpenGL minification
        // filters GL_LINEAR/GL_NEAREST so emulate them by setting minLod = 0, and maxLod = 0.25,
        // and using minFilter = VK_FILTER_LINEAR or minFilter = VK_FILTER_NEAREST
        const bool skip_mipmap = config.type == Pica::TexturingRegs::TextureConfig::TextureCube;
        info =
            SamplerInfo{.mag_filter = config.mag_filter,
                        .min_filter = config.min_filter,
                        .mip_filter = config.mip_filter,
                        .wrap_s = config.wrap_s,
                        .wrap_t = config.wrap_t,
                        .border_color = config.border_color.raw,
                        .lod_min = skip_mipmap ? 0.f : static_cast<float>(config.lod.min_level),
                        .lod_max = skip_mipmap ? 0.25f : static_cast<float>(config.lod.max_level),
                        .lod_bias = static_cast<float>(config.lod.bias)};

        // Search the cache and bind the appropriate sampler
        if (auto it = samplers.find(info); it != samplers.end()) {
            pipeline_cache.BindSampler(binding, it->second);
        } else {
            vk::Sampler texture_sampler = CreateSampler(info);
            samplers.emplace(info, texture_sampler);
            pipeline_cache.BindSampler(binding, texture_sampler);
        }
    };

    renderpass_cache.ExitRenderpass();

    // Sync and bind the texture surfaces
    const auto pica_textures = regs.texturing.GetTextures();
    for (unsigned texture_index = 0; texture_index < pica_textures.size(); ++texture_index) {
        const auto& texture = pica_textures[texture_index];

        if (texture.enabled) {
            if (texture_index == 0) {
                using TextureType = Pica::TexturingRegs::TextureConfig::TextureType;
                switch (texture.config.type.Value()) {
                case TextureType::Shadow2D: {
                    auto surface = res_cache.GetTextureSurface(texture);
                    if (surface != nullptr) {
                        pipeline_cache.BindStorageImage(0, surface->alloc.image_view);
                    } else {
                        pipeline_cache.BindStorageImage(0, default_texture.image_view);
                    }
                    continue;
                }
                case TextureType::ShadowCube: {
                    using CubeFace = Pica::TexturingRegs::CubeFace;
                    auto info = Pica::Texture::TextureInfo::FromPicaRegister(texture.config,
                                                                             texture.format);
                    BindCubeFace(CubeFace::PositiveX, info);
                    BindCubeFace(CubeFace::NegativeX, info);
                    BindCubeFace(CubeFace::PositiveY, info);
                    BindCubeFace(CubeFace::NegativeY, info);
                    BindCubeFace(CubeFace::PositiveZ, info);
                    BindCubeFace(CubeFace::NegativeZ, info);
                    continue;
                }
                case TextureType::TextureCube: {
                    using CubeFace = Pica::TexturingRegs::CubeFace;
                    const VideoCore::TextureCubeConfig config = {
                        .px = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveX),
                        .nx = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeX),
                        .py = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveY),
                        .ny = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeY),
                        .pz = regs.texturing.GetCubePhysicalAddress(CubeFace::PositiveZ),
                        .nz = regs.texturing.GetCubePhysicalAddress(CubeFace::NegativeZ),
                        .width = texture.config.width,
                        .format = texture.format};

                    auto surface = res_cache.GetTextureCube(config);
                    if (surface != nullptr) {
                        runtime.Transition(scheduler.GetRenderCommandBuffer(), surface->alloc,
                                           vk::ImageLayout::eShaderReadOnlyOptimal, 0,
                                           surface->alloc.levels, 0, 6);
                        pipeline_cache.BindTexture(3, surface->alloc.image_view);
                    } else {
                        pipeline_cache.BindTexture(3, default_texture.image_view);
                    }

                    BindSampler(3, texture_cube_sampler, texture.config);
                    continue; // Texture unit 0 setup finished. Continue to next unit
                }
                default:
                    break;
                }
            }

            // Update sampler key
            BindSampler(texture_index, texture_samplers[texture_index], texture.config);

            auto surface = res_cache.GetTextureSurface(texture);
            if (surface != nullptr) {
                runtime.Transition(scheduler.GetRenderCommandBuffer(), surface->alloc,
                                   vk::ImageLayout::eShaderReadOnlyOptimal, 0,
                                   surface->alloc.levels);
                CheckBarrier(surface->alloc.image_view, texture_index);
            } else {
                // Can occur when texture addr is null or its memory is unmapped/invalid
                // HACK: In this case, the correct behaviour for the PICA is to use the last
                // rendered colour. But because this would be impractical to implement, the
                // next best alternative is to use a clear texture, essentially skipping
                // the geometry in question.
                // For example: a bug in Pokemon X/Y causes NULL-texture squares to be drawn
                // on the male character's face, which in the OpenGL default appear black.
                // state.texture_units[texture_index].texture_2d = default_texture;
                pipeline_cache.BindTexture(texture_index, default_texture.image_view);
            }
        } else {
            pipeline_cache.BindTexture(texture_index, default_texture.image_view);
            pipeline_cache.BindSampler(texture_index, default_sampler);
        }
    }

    // NOTE: From here onwards its a safe zone to set the draw state, doing that any earlier will
    // cause issues as the rasterizer cache might cause a scheduler switch and invalidate our state

    // Sync the viewport
    pipeline_cache.SetViewport(surfaces_rect.left + viewport_rect_unscaled.left * res_scale,
                               surfaces_rect.bottom + viewport_rect_unscaled.bottom * res_scale,
                               viewport_rect_unscaled.GetWidth() * res_scale,
                               viewport_rect_unscaled.GetHeight() * res_scale);

    // Sync and bind the shader
    if (shader_dirty) {
        SetShader();
        shader_dirty = false;
    }

    // Sync the LUTs within the texture buffer
    SyncAndUploadLUTs();
    SyncAndUploadLUTsLF();

    // Sync the uniform data
    UploadUniforms(accelerate);

    // Viewport can have negative offsets or larger dimensions than our framebuffer sub-rect.
    // Enable scissor test to prevent drawing outside of the framebuffer region
    pipeline_cache.SetScissor(draw_rect.left, draw_rect.bottom, draw_rect.GetWidth(),
                              draw_rect.GetHeight());

    auto valid_surface = color_surface ? color_surface : depth_surface;
    const FramebufferInfo framebuffer_info = {
        .color = color_surface ? color_surface->GetFramebufferView() : VK_NULL_HANDLE,
        .depth = depth_surface ? depth_surface->GetFramebufferView() : VK_NULL_HANDLE,
        .renderpass = renderpass_cache.GetRenderpass(pipeline_info.color_attachment,
                                                     pipeline_info.depth_attachment, false),
        .width = valid_surface->GetScaledWidth(),
        .height = valid_surface->GetScaledHeight()};

    auto [it, new_framebuffer] = framebuffers.try_emplace(framebuffer_info, vk::Framebuffer{});
    if (new_framebuffer) {
        it->second = CreateFramebuffer(framebuffer_info);
    }

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    if (color_surface) {
        runtime.Transition(command_buffer, color_surface->alloc,
                           vk::ImageLayout::eColorAttachmentOptimal, 0,
                           color_surface->alloc.levels);
    }

    if (depth_surface) {
        runtime.Transition(command_buffer, depth_surface->alloc,
                           vk::ImageLayout::eDepthStencilAttachmentOptimal, 0,
                           depth_surface->alloc.levels);
    }

    const vk::RenderPassBeginInfo renderpass_begin = {
        .renderPass = renderpass_cache.GetRenderpass(pipeline_info.color_attachment,
                                                     pipeline_info.depth_attachment, false),
        .framebuffer = it->second,
        .renderArea = vk::Rect2D{.offset = {static_cast<s32>(draw_rect.left),
                                            static_cast<s32>(draw_rect.bottom)},
                                 .extent = {draw_rect.GetWidth(), draw_rect.GetHeight()}},

        .clearValueCount = 0,
        .pClearValues = nullptr};

    renderpass_cache.EnterRenderpass(renderpass_begin);

    // Draw the vertex batch
    bool succeeded = true;
    if (accelerate) {
        succeeded = AccelerateDrawBatchInternal(is_indexed);
    } else {
        pipeline_info.rasterization.topology.Assign(Pica::PipelineRegs::TriangleTopology::List);
        pipeline_info.vertex_layout = HardwareVertex::GetVertexLayout();
        pipeline_cache.UseTrivialVertexShader();
        pipeline_cache.UseTrivialGeometryShader();
        pipeline_cache.BindPipeline(pipeline_info);

        // Bind the vertex buffer at the current mapped offset. This effectively means
        // that when base_vertex is zero the GPU will start drawing from the current mapped
        // offset not the start of the buffer.
        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        command_buffer.bindVertexBuffers(0, vertex_buffer.GetHandle(),
                                         vertex_buffer.GetBufferOffset());

        const u32 max_vertices = VERTEX_BUFFER_SIZE / sizeof(HardwareVertex);
        const u32 batch_size = static_cast<u32>(vertex_batch.size());
        for (u32 base_vertex = 0; base_vertex < batch_size; base_vertex += max_vertices) {
            const u32 vertices = std::min(max_vertices, batch_size - base_vertex);
            const u32 vertex_size = vertices * sizeof(HardwareVertex);

            // Copy vertex data
            auto [array_ptr, offset, _] = vertex_buffer.Map(vertex_size, sizeof(HardwareVertex));
            std::memcpy(array_ptr, vertex_batch.data() + base_vertex, vertex_size);
            vertex_buffer.Commit(vertex_size);

            command_buffer.draw(vertices, 1, base_vertex, 0);
        }
    }

    vertex_batch.clear();

    // Mark framebuffer surfaces as dirty
    const VideoCore::Rect2D draw_rect_unscaled = {
        draw_rect.left / res_scale, draw_rect.top / res_scale, draw_rect.right / res_scale,
        draw_rect.bottom / res_scale};

    if (color_surface != nullptr && write_color_fb) {
        auto interval = color_surface->GetSubRectInterval(draw_rect_unscaled);
        res_cache.InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval),
                                   color_surface);
    }

    if (depth_surface != nullptr && write_depth_fb) {
        auto interval = depth_surface->GetSubRectInterval(draw_rect_unscaled);
        res_cache.InvalidateRegion(boost::icl::first(interval), boost::icl::length(interval),
                                   depth_surface);
    }

    return succeeded;
}

void RasterizerVulkan::NotifyPicaRegisterChanged(u32 id) {
    const auto& regs = Pica::g_state.regs;

    switch (id) {
    // Culling
    case PICA_REG_INDEX(rasterizer.cull_mode):
        SyncCullMode();
        break;

    // Clipping plane
    case PICA_REG_INDEX(rasterizer.clip_enable):
        SyncClipEnabled();
        break;

    case PICA_REG_INDEX(rasterizer.clip_coef[0]):
    case PICA_REG_INDEX(rasterizer.clip_coef[1]):
    case PICA_REG_INDEX(rasterizer.clip_coef[2]):
    case PICA_REG_INDEX(rasterizer.clip_coef[3]):
        SyncClipCoef();
        break;

    // Depth modifiers
    case PICA_REG_INDEX(rasterizer.viewport_depth_range):
        SyncDepthScale();
        break;
    case PICA_REG_INDEX(rasterizer.viewport_depth_near_plane):
        SyncDepthOffset();
        break;

    // Depth buffering
    case PICA_REG_INDEX(rasterizer.depthmap_enable):
        shader_dirty = true;
        break;

    // Blending
    case PICA_REG_INDEX(framebuffer.output_merger.alphablend_enable):
        SyncBlendEnabled();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_blending):
        SyncBlendFuncs();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.blend_const):
        SyncBlendColor();
        break;

    // Shadow texture
    case PICA_REG_INDEX(texturing.shadow):
        SyncShadowTextureBias();
        break;

    // Fog state
    case PICA_REG_INDEX(texturing.fog_color):
        SyncFogColor();
        break;
    case PICA_REG_INDEX(texturing.fog_lut_data[0]):
    case PICA_REG_INDEX(texturing.fog_lut_data[1]):
    case PICA_REG_INDEX(texturing.fog_lut_data[2]):
    case PICA_REG_INDEX(texturing.fog_lut_data[3]):
    case PICA_REG_INDEX(texturing.fog_lut_data[4]):
    case PICA_REG_INDEX(texturing.fog_lut_data[5]):
    case PICA_REG_INDEX(texturing.fog_lut_data[6]):
    case PICA_REG_INDEX(texturing.fog_lut_data[7]):
        uniform_block_data.fog_lut_dirty = true;
        break;

    // ProcTex state
    case PICA_REG_INDEX(texturing.proctex):
    case PICA_REG_INDEX(texturing.proctex_lut):
    case PICA_REG_INDEX(texturing.proctex_lut_offset):
        SyncProcTexBias();
        shader_dirty = true;
        break;

    case PICA_REG_INDEX(texturing.proctex_noise_u):
    case PICA_REG_INDEX(texturing.proctex_noise_v):
    case PICA_REG_INDEX(texturing.proctex_noise_frequency):
        SyncProcTexNoise();
        break;

    case PICA_REG_INDEX(texturing.proctex_lut_data[0]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[1]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[2]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[3]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[4]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[5]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[6]):
    case PICA_REG_INDEX(texturing.proctex_lut_data[7]):
        using Pica::TexturingRegs;
        switch (regs.texturing.proctex_lut_config.ref_table.Value()) {
        case TexturingRegs::ProcTexLutTable::Noise:
            uniform_block_data.proctex_noise_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorMap:
            uniform_block_data.proctex_color_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::AlphaMap:
            uniform_block_data.proctex_alpha_map_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::Color:
            uniform_block_data.proctex_lut_dirty = true;
            break;
        case TexturingRegs::ProcTexLutTable::ColorDiff:
            uniform_block_data.proctex_diff_lut_dirty = true;
            break;
        }
        break;

    // Alpha test
    case PICA_REG_INDEX(framebuffer.output_merger.alpha_test):
        SyncAlphaTest();
        shader_dirty = true;
        break;

    // Sync GL stencil test + stencil write mask
    // (Pica stencil test function register also contains a stencil write mask)
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_func):
        SyncStencilTest();
        SyncStencilWriteMask();
        break;
    case PICA_REG_INDEX(framebuffer.output_merger.stencil_test.raw_op):
    case PICA_REG_INDEX(framebuffer.framebuffer.depth_format):
        SyncStencilTest();
        break;

    // Sync GL depth test + depth and color write mask
    // (Pica depth test function register also contains a depth and color write mask)
    case PICA_REG_INDEX(framebuffer.output_merger.depth_test_enable):
        SyncDepthTest();
        SyncDepthWriteMask();
        SyncColorWriteMask();
        break;

    // Sync GL depth and stencil write mask
    // (This is a dedicated combined depth / stencil write-enable register)
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_depth_stencil_write):
        SyncDepthWriteMask();
        SyncStencilWriteMask();
        break;

    // Sync GL color write mask
    // (This is a dedicated color write-enable register)
    case PICA_REG_INDEX(framebuffer.framebuffer.allow_color_write):
        SyncColorWriteMask();
        break;

    case PICA_REG_INDEX(framebuffer.shadow):
        SyncShadowBias();
        break;

    // Scissor test
    case PICA_REG_INDEX(rasterizer.scissor_test.mode):
        shader_dirty = true;
        break;

    // Logic op
    case PICA_REG_INDEX(framebuffer.output_merger.logic_op):
        SyncLogicOp();
        break;

    case PICA_REG_INDEX(texturing.main_config):
        shader_dirty = true;
        break;

    // Texture 0 type
    case PICA_REG_INDEX(texturing.texture0.type):
        shader_dirty = true;
        break;

    // TEV stages
    // (This also syncs fog_mode and fog_flip which are part of tev_combiner_buffer_input)
    case PICA_REG_INDEX(texturing.tev_stage0.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage0.color_op):
    case PICA_REG_INDEX(texturing.tev_stage0.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage1.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage1.color_op):
    case PICA_REG_INDEX(texturing.tev_stage1.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage2.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage2.color_op):
    case PICA_REG_INDEX(texturing.tev_stage2.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage3.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage3.color_op):
    case PICA_REG_INDEX(texturing.tev_stage3.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage4.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage4.color_op):
    case PICA_REG_INDEX(texturing.tev_stage4.color_scale):
    case PICA_REG_INDEX(texturing.tev_stage5.color_source1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_modifier1):
    case PICA_REG_INDEX(texturing.tev_stage5.color_op):
    case PICA_REG_INDEX(texturing.tev_stage5.color_scale):
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_input):
        shader_dirty = true;
        break;
    case PICA_REG_INDEX(texturing.tev_stage0.const_r):
        SyncTevConstColor(0, regs.texturing.tev_stage0);
        break;
    case PICA_REG_INDEX(texturing.tev_stage1.const_r):
        SyncTevConstColor(1, regs.texturing.tev_stage1);
        break;
    case PICA_REG_INDEX(texturing.tev_stage2.const_r):
        SyncTevConstColor(2, regs.texturing.tev_stage2);
        break;
    case PICA_REG_INDEX(texturing.tev_stage3.const_r):
        SyncTevConstColor(3, regs.texturing.tev_stage3);
        break;
    case PICA_REG_INDEX(texturing.tev_stage4.const_r):
        SyncTevConstColor(4, regs.texturing.tev_stage4);
        break;
    case PICA_REG_INDEX(texturing.tev_stage5.const_r):
        SyncTevConstColor(5, regs.texturing.tev_stage5);
        break;

    // TEV combiner buffer color
    case PICA_REG_INDEX(texturing.tev_combiner_buffer_color):
        SyncCombinerColor();
        break;

    // Fragment lighting switches
    case PICA_REG_INDEX(lighting.disable):
    case PICA_REG_INDEX(lighting.max_light_index):
    case PICA_REG_INDEX(lighting.config0):
    case PICA_REG_INDEX(lighting.config1):
    case PICA_REG_INDEX(lighting.abs_lut_input):
    case PICA_REG_INDEX(lighting.lut_input):
    case PICA_REG_INDEX(lighting.lut_scale):
    case PICA_REG_INDEX(lighting.light_enable):
        break;

    // Fragment lighting specular 0 color
    case PICA_REG_INDEX(lighting.light[0].specular_0):
        SyncLightSpecular0(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_0):
        SyncLightSpecular0(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_0):
        SyncLightSpecular0(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_0):
        SyncLightSpecular0(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_0):
        SyncLightSpecular0(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_0):
        SyncLightSpecular0(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_0):
        SyncLightSpecular0(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_0):
        SyncLightSpecular0(7);
        break;

    // Fragment lighting specular 1 color
    case PICA_REG_INDEX(lighting.light[0].specular_1):
        SyncLightSpecular1(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].specular_1):
        SyncLightSpecular1(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].specular_1):
        SyncLightSpecular1(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].specular_1):
        SyncLightSpecular1(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].specular_1):
        SyncLightSpecular1(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].specular_1):
        SyncLightSpecular1(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].specular_1):
        SyncLightSpecular1(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].specular_1):
        SyncLightSpecular1(7);
        break;

    // Fragment lighting diffuse color
    case PICA_REG_INDEX(lighting.light[0].diffuse):
        SyncLightDiffuse(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].diffuse):
        SyncLightDiffuse(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].diffuse):
        SyncLightDiffuse(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].diffuse):
        SyncLightDiffuse(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].diffuse):
        SyncLightDiffuse(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].diffuse):
        SyncLightDiffuse(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].diffuse):
        SyncLightDiffuse(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].diffuse):
        SyncLightDiffuse(7);
        break;

    // Fragment lighting ambient color
    case PICA_REG_INDEX(lighting.light[0].ambient):
        SyncLightAmbient(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].ambient):
        SyncLightAmbient(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].ambient):
        SyncLightAmbient(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].ambient):
        SyncLightAmbient(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].ambient):
        SyncLightAmbient(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].ambient):
        SyncLightAmbient(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].ambient):
        SyncLightAmbient(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].ambient):
        SyncLightAmbient(7);
        break;

    // Fragment lighting position
    case PICA_REG_INDEX(lighting.light[0].x):
    case PICA_REG_INDEX(lighting.light[0].z):
        SyncLightPosition(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].x):
    case PICA_REG_INDEX(lighting.light[1].z):
        SyncLightPosition(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].x):
    case PICA_REG_INDEX(lighting.light[2].z):
        SyncLightPosition(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].x):
    case PICA_REG_INDEX(lighting.light[3].z):
        SyncLightPosition(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].x):
    case PICA_REG_INDEX(lighting.light[4].z):
        SyncLightPosition(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].x):
    case PICA_REG_INDEX(lighting.light[5].z):
        SyncLightPosition(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].x):
    case PICA_REG_INDEX(lighting.light[6].z):
        SyncLightPosition(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].x):
    case PICA_REG_INDEX(lighting.light[7].z):
        SyncLightPosition(7);
        break;

    // Fragment spot lighting direction
    case PICA_REG_INDEX(lighting.light[0].spot_x):
    case PICA_REG_INDEX(lighting.light[0].spot_z):
        SyncLightSpotDirection(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].spot_x):
    case PICA_REG_INDEX(lighting.light[1].spot_z):
        SyncLightSpotDirection(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].spot_x):
    case PICA_REG_INDEX(lighting.light[2].spot_z):
        SyncLightSpotDirection(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].spot_x):
    case PICA_REG_INDEX(lighting.light[3].spot_z):
        SyncLightSpotDirection(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].spot_x):
    case PICA_REG_INDEX(lighting.light[4].spot_z):
        SyncLightSpotDirection(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].spot_x):
    case PICA_REG_INDEX(lighting.light[5].spot_z):
        SyncLightSpotDirection(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].spot_x):
    case PICA_REG_INDEX(lighting.light[6].spot_z):
        SyncLightSpotDirection(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].spot_x):
    case PICA_REG_INDEX(lighting.light[7].spot_z):
        SyncLightSpotDirection(7);
        break;

    // Fragment lighting light source config
    case PICA_REG_INDEX(lighting.light[0].config):
    case PICA_REG_INDEX(lighting.light[1].config):
    case PICA_REG_INDEX(lighting.light[2].config):
    case PICA_REG_INDEX(lighting.light[3].config):
    case PICA_REG_INDEX(lighting.light[4].config):
    case PICA_REG_INDEX(lighting.light[5].config):
    case PICA_REG_INDEX(lighting.light[6].config):
    case PICA_REG_INDEX(lighting.light[7].config):
        shader_dirty = true;
        break;

    // Fragment lighting distance attenuation bias
    case PICA_REG_INDEX(lighting.light[0].dist_atten_bias):
        SyncLightDistanceAttenuationBias(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_bias):
        SyncLightDistanceAttenuationBias(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_bias):
        SyncLightDistanceAttenuationBias(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_bias):
        SyncLightDistanceAttenuationBias(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_bias):
        SyncLightDistanceAttenuationBias(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_bias):
        SyncLightDistanceAttenuationBias(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_bias):
        SyncLightDistanceAttenuationBias(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_bias):
        SyncLightDistanceAttenuationBias(7);
        break;

    // Fragment lighting distance attenuation scale
    case PICA_REG_INDEX(lighting.light[0].dist_atten_scale):
        SyncLightDistanceAttenuationScale(0);
        break;
    case PICA_REG_INDEX(lighting.light[1].dist_atten_scale):
        SyncLightDistanceAttenuationScale(1);
        break;
    case PICA_REG_INDEX(lighting.light[2].dist_atten_scale):
        SyncLightDistanceAttenuationScale(2);
        break;
    case PICA_REG_INDEX(lighting.light[3].dist_atten_scale):
        SyncLightDistanceAttenuationScale(3);
        break;
    case PICA_REG_INDEX(lighting.light[4].dist_atten_scale):
        SyncLightDistanceAttenuationScale(4);
        break;
    case PICA_REG_INDEX(lighting.light[5].dist_atten_scale):
        SyncLightDistanceAttenuationScale(5);
        break;
    case PICA_REG_INDEX(lighting.light[6].dist_atten_scale):
        SyncLightDistanceAttenuationScale(6);
        break;
    case PICA_REG_INDEX(lighting.light[7].dist_atten_scale):
        SyncLightDistanceAttenuationScale(7);
        break;

    // Fragment lighting global ambient color (emission + ambient * ambient)
    case PICA_REG_INDEX(lighting.global_ambient):
        SyncGlobalAmbient();
        break;

    // Fragment lighting lookup tables
    case PICA_REG_INDEX(lighting.lut_data[0]):
    case PICA_REG_INDEX(lighting.lut_data[1]):
    case PICA_REG_INDEX(lighting.lut_data[2]):
    case PICA_REG_INDEX(lighting.lut_data[3]):
    case PICA_REG_INDEX(lighting.lut_data[4]):
    case PICA_REG_INDEX(lighting.lut_data[5]):
    case PICA_REG_INDEX(lighting.lut_data[6]):
    case PICA_REG_INDEX(lighting.lut_data[7]): {
        const auto& lut_config = regs.lighting.lut_config;
        uniform_block_data.lighting_lut_dirty[lut_config.type] = true;
        uniform_block_data.lighting_lut_dirty_any = true;
        break;
    }
    }
}

void RasterizerVulkan::FlushAll() {
    MICROPROFILE_SCOPE(OpenGL_CacheManagement);
    res_cache.FlushAll();
}

void RasterizerVulkan::FlushRegion(PAddr addr, u32 size) {
    MICROPROFILE_SCOPE(OpenGL_CacheManagement);
    res_cache.FlushRegion(addr, size);
}

void RasterizerVulkan::InvalidateRegion(PAddr addr, u32 size) {
    MICROPROFILE_SCOPE(OpenGL_CacheManagement);
    res_cache.InvalidateRegion(addr, size, nullptr);
}

void RasterizerVulkan::FlushAndInvalidateRegion(PAddr addr, u32 size) {
    MICROPROFILE_SCOPE(OpenGL_CacheManagement);
    res_cache.FlushRegion(addr, size);
    res_cache.InvalidateRegion(addr, size, nullptr);
}

bool RasterizerVulkan::AccelerateDisplayTransfer(const GPU::Regs::DisplayTransferConfig& config) {
    MICROPROFILE_SCOPE(OpenGL_Blits);

    VideoCore::SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.width = config.output_width;
    src_params.stride = config.input_width;
    src_params.height = config.output_height;
    src_params.is_tiled = !config.input_linear;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.input_format);
    src_params.UpdateParams();

    VideoCore::SurfaceParams dst_params;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = config.scaling != config.NoScale ? config.output_width.Value() / 2
                                                        : config.output_width.Value();
    dst_params.height = config.scaling == config.ScaleXY ? config.output_height.Value() / 2
                                                         : config.output_height.Value();
    dst_params.is_tiled = config.input_linear != config.dont_swizzle;
    dst_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.output_format);
    dst_params.UpdateParams();

    auto [src_surface, src_rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);
    if (src_surface == nullptr)
        return false;

    dst_params.res_scale = src_surface->res_scale;

    auto [dst_surface, dst_rect] =
        res_cache.GetSurfaceSubRect(dst_params, VideoCore::ScaleMatch::Upscale, false);
    if (dst_surface == nullptr) {
        return false;
    }

    if (src_surface->is_tiled != dst_surface->is_tiled)
        std::swap(src_rect.top, src_rect.bottom);

    if (config.flip_vertically)
        std::swap(src_rect.top, src_rect.bottom);

    if (!res_cache.BlitSurfaces(src_surface, src_rect, dst_surface, dst_rect))
        return false;

    res_cache.InvalidateRegion(dst_params.addr, dst_params.size, dst_surface);
    return true;
}

bool RasterizerVulkan::AccelerateTextureCopy(const GPU::Regs::DisplayTransferConfig& config) {
    u32 copy_size = Common::AlignDown(config.texture_copy.size, 16);
    if (copy_size == 0) {
        return false;
    }

    u32 input_gap = config.texture_copy.input_gap * 16;
    u32 input_width = config.texture_copy.input_width * 16;
    if (input_width == 0 && input_gap != 0) {
        return false;
    }
    if (input_gap == 0 || input_width >= copy_size) {
        input_width = copy_size;
        input_gap = 0;
    }
    if (copy_size % input_width != 0) {
        return false;
    }

    u32 output_gap = config.texture_copy.output_gap * 16;
    u32 output_width = config.texture_copy.output_width * 16;
    if (output_width == 0 && output_gap != 0) {
        return false;
    }
    if (output_gap == 0 || output_width >= copy_size) {
        output_width = copy_size;
        output_gap = 0;
    }
    if (copy_size % output_width != 0) {
        return false;
    }

    VideoCore::SurfaceParams src_params;
    src_params.addr = config.GetPhysicalInputAddress();
    src_params.stride = input_width + input_gap; // stride in bytes
    src_params.width = input_width;              // width in bytes
    src_params.height = copy_size / input_width;
    src_params.size = ((src_params.height - 1) * src_params.stride) + src_params.width;
    src_params.end = src_params.addr + src_params.size;

    auto [src_surface, src_rect] = res_cache.GetTexCopySurface(src_params);
    if (src_surface == nullptr) {
        return false;
    }

    if (output_gap != 0 &&
        (output_width != src_surface->BytesInPixels(src_rect.GetWidth() / src_surface->res_scale) *
                             (src_surface->is_tiled ? 8 : 1) ||
         output_gap % src_surface->BytesInPixels(src_surface->is_tiled ? 64 : 1) != 0)) {
        return false;
    }

    VideoCore::SurfaceParams dst_params = *src_surface;
    dst_params.addr = config.GetPhysicalOutputAddress();
    dst_params.width = src_rect.GetWidth() / src_surface->res_scale;
    dst_params.stride = dst_params.width + src_surface->PixelsInBytes(
                                               src_surface->is_tiled ? output_gap / 8 : output_gap);
    dst_params.height = src_rect.GetHeight() / src_surface->res_scale;
    dst_params.res_scale = src_surface->res_scale;
    dst_params.UpdateParams();

    // Since we are going to invalidate the gap if there is one, we will have to load it first
    const bool load_gap = output_gap != 0;
    auto [dst_surface, dst_rect] =
        res_cache.GetSurfaceSubRect(dst_params, VideoCore::ScaleMatch::Upscale, load_gap);

    if (!dst_surface || dst_surface->type == VideoCore::SurfaceType::Texture ||
        !res_cache.BlitSurfaces(src_surface, src_rect, dst_surface, dst_rect)) {
        return false;
    }

    res_cache.InvalidateRegion(dst_params.addr, dst_params.size, dst_surface);
    return true;
}

bool RasterizerVulkan::AccelerateFill(const GPU::Regs::MemoryFillConfig& config) {
    auto dst_surface = res_cache.GetFillSurface(config);
    if (dst_surface == nullptr)
        return false;

    res_cache.InvalidateRegion(dst_surface->addr, dst_surface->size, dst_surface);
    return true;
}

bool RasterizerVulkan::AccelerateDisplay(const GPU::Regs::FramebufferConfig& config,
                                         PAddr framebuffer_addr, u32 pixel_stride,
                                         ScreenInfo& screen_info) {
    if (framebuffer_addr == 0) {
        return false;
    }
    MICROPROFILE_SCOPE(OpenGL_CacheManagement);

    VideoCore::SurfaceParams src_params;
    src_params.addr = framebuffer_addr;
    src_params.width = std::min(config.width.Value(), pixel_stride);
    src_params.height = config.height;
    src_params.stride = pixel_stride;
    src_params.is_tiled = false;
    src_params.pixel_format = VideoCore::PixelFormatFromGPUPixelFormat(config.color_format);
    src_params.UpdateParams();

    const auto [src_surface, src_rect] =
        res_cache.GetSurfaceSubRect(src_params, VideoCore::ScaleMatch::Ignore, true);

    if (src_surface == nullptr) {
        return false;
    }

    u32 scaled_width = src_surface->GetScaledWidth();
    u32 scaled_height = src_surface->GetScaledHeight();

    screen_info.display_texcoords = Common::Rectangle<float>(
        (float)src_rect.bottom / (float)scaled_height, (float)src_rect.left / (float)scaled_width,
        (float)src_rect.top / (float)scaled_height, (float)src_rect.right / (float)scaled_width);

    screen_info.display_texture = &src_surface->alloc;

    return true;
}

vk::Sampler RasterizerVulkan::CreateSampler(const SamplerInfo& info) {
    auto properties = instance.GetPhysicalDevice().getProperties();
    const vk::SamplerCreateInfo sampler_info = {
        .magFilter = PicaToVK::TextureFilterMode(info.mag_filter),
        .minFilter = PicaToVK::TextureFilterMode(info.min_filter),
        .mipmapMode = PicaToVK::TextureMipFilterMode(info.mip_filter),
        .addressModeU = PicaToVK::WrapMode(info.wrap_s),
        .addressModeV = PicaToVK::WrapMode(info.wrap_t),
        .mipLodBias = info.lod_bias / 256.0f,
        .anisotropyEnable = true,
        .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
        .compareEnable = false,
        .compareOp = vk::CompareOp::eAlways,
        .minLod = info.lod_min,
        .maxLod = info.lod_max,
        .borderColor = vk::BorderColor::eIntOpaqueBlack,
        .unnormalizedCoordinates = false};

    vk::Device device = instance.GetDevice();
    return device.createSampler(sampler_info);
}

vk::Framebuffer RasterizerVulkan::CreateFramebuffer(const FramebufferInfo& info) {
    u32 attachment_count = 0;
    std::array<vk::ImageView, 2> attachments;

    if (info.color) {
        attachments[attachment_count++] = info.color;
    }

    if (info.depth) {
        attachments[attachment_count++] = info.depth;
    }

    const vk::FramebufferCreateInfo framebuffer_info = {.renderPass = info.renderpass,
                                                        .attachmentCount = attachment_count,
                                                        .pAttachments = attachments.data(),
                                                        .width = info.width,
                                                        .height = info.height,
                                                        .layers = 1};

    vk::Device device = instance.GetDevice();
    return device.createFramebuffer(framebuffer_info);
}

void RasterizerVulkan::FlushBuffers() {
    vertex_buffer.Flush();
    uniform_buffer.Flush();
    index_buffer.Flush();
    texture_buffer.Flush();
    texture_lf_buffer.Flush();
}

void RasterizerVulkan::SetShader() {
    pipeline_cache.UseFragmentShader(Pica::g_state.regs);
}

void RasterizerVulkan::SyncClipEnabled() {
    uniform_block_data.data.enable_clip1 = Pica::g_state.regs.rasterizer.clip_enable != 0;
}

void RasterizerVulkan::SyncClipCoef() {
    const auto raw_clip_coef = Pica::g_state.regs.rasterizer.GetClipCoef();
    const Common::Vec4f new_clip_coef = {raw_clip_coef.x.ToFloat32(), raw_clip_coef.y.ToFloat32(),
                                         raw_clip_coef.z.ToFloat32(), raw_clip_coef.w.ToFloat32()};
    if (new_clip_coef != uniform_block_data.data.clip_coef) {
        uniform_block_data.data.clip_coef = new_clip_coef;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncCullMode() {
    const auto& regs = Pica::g_state.regs;
    if (instance.IsExtendedDynamicStateSupported()) {
        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        command_buffer.setCullModeEXT(PicaToVK::CullMode(regs.rasterizer.cull_mode));
        command_buffer.setFrontFaceEXT(PicaToVK::FrontFace(regs.rasterizer.cull_mode));
    }

    pipeline_info.rasterization.cull_mode.Assign(regs.rasterizer.cull_mode);
}

void RasterizerVulkan::SyncDepthScale() {
    float depth_scale =
        Pica::float24::FromRaw(Pica::g_state.regs.rasterizer.viewport_depth_range).ToFloat32();

    if (depth_scale != uniform_block_data.data.depth_scale) {
        uniform_block_data.data.depth_scale = depth_scale;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncDepthOffset() {
    float depth_offset =
        Pica::float24::FromRaw(Pica::g_state.regs.rasterizer.viewport_depth_near_plane).ToFloat32();

    if (depth_offset != uniform_block_data.data.depth_offset) {
        uniform_block_data.data.depth_offset = depth_offset;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncBlendEnabled() {
    pipeline_info.blending.blend_enable.Assign(
        Pica::g_state.regs.framebuffer.output_merger.alphablend_enable);
}

void RasterizerVulkan::SyncBlendFuncs() {
    const auto& regs = Pica::g_state.regs;

    pipeline_info.blending.color_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_rgb);
    pipeline_info.blending.alpha_blend_eq.Assign(
        regs.framebuffer.output_merger.alpha_blending.blend_equation_a);
    pipeline_info.blending.src_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_rgb);
    pipeline_info.blending.dst_color_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_rgb);
    pipeline_info.blending.src_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_source_a);
    pipeline_info.blending.dst_alpha_blend_factor.Assign(
        regs.framebuffer.output_merger.alpha_blending.factor_dest_a);
}

void RasterizerVulkan::SyncBlendColor() {
    auto blend_color =
        PicaToVK::ColorRGBA8(Pica::g_state.regs.framebuffer.output_merger.blend_const.raw);

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setBlendConstants(blend_color.AsArray());
}

void RasterizerVulkan::SyncFogColor() {
    const auto& regs = Pica::g_state.regs;
    uniform_block_data.data.fog_color = {
        regs.texturing.fog_color.r.Value() / 255.0f,
        regs.texturing.fog_color.g.Value() / 255.0f,
        regs.texturing.fog_color.b.Value() / 255.0f,
    };
    uniform_block_data.dirty = true;
}

void RasterizerVulkan::SyncProcTexNoise() {
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

void RasterizerVulkan::SyncProcTexBias() {
    const auto& regs = Pica::g_state.regs.texturing;
    uniform_block_data.data.proctex_bias =
        Pica::float16::FromRaw(regs.proctex.bias_low | (regs.proctex_lut.bias_high << 8))
            .ToFloat32();

    uniform_block_data.dirty = true;
}

void RasterizerVulkan::SyncAlphaTest() {
    const auto& regs = Pica::g_state.regs;
    if (regs.framebuffer.output_merger.alpha_test.ref != uniform_block_data.data.alphatest_ref) {
        uniform_block_data.data.alphatest_ref = regs.framebuffer.output_merger.alpha_test.ref;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLogicOp() {
    const auto& regs = Pica::g_state.regs;
    pipeline_info.blending.logic_op.Assign(regs.framebuffer.output_merger.logic_op);
}

void RasterizerVulkan::SyncColorWriteMask() {
    const auto& regs = Pica::g_state.regs;
    const u32 color_mask = (regs.framebuffer.output_merger.depth_color_mask >> 8) & 0xF;
    pipeline_info.blending.color_write_mask.Assign(color_mask);
}

void RasterizerVulkan::SyncStencilWriteMask() {
    const auto& regs = Pica::g_state.regs;
    pipeline_info.depth_stencil.stencil_write_mask =
        (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0)
            ? static_cast<u32>(regs.framebuffer.output_merger.stencil_test.write_mask)
            : 0;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       pipeline_info.depth_stencil.stencil_write_mask);
}

void RasterizerVulkan::SyncDepthWriteMask() {
    const auto& regs = Pica::g_state.regs;

    const bool write_enable = (regs.framebuffer.framebuffer.allow_depth_stencil_write != 0 &&
                               regs.framebuffer.output_merger.depth_write_enable);

    if (instance.IsExtendedDynamicStateSupported()) {
        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        command_buffer.setDepthWriteEnableEXT(write_enable);
    }

    pipeline_info.depth_stencil.depth_write_enable.Assign(write_enable);
}

void RasterizerVulkan::SyncStencilTest() {
    const auto& regs = Pica::g_state.regs;

    const auto& stencil_test = regs.framebuffer.output_merger.stencil_test;
    const bool test_enable = stencil_test.enable && regs.framebuffer.framebuffer.depth_format ==
                                                        Pica::FramebufferRegs::DepthFormat::D24S8;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_test.input_mask);
    command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_test.reference_value);

    if (instance.IsExtendedDynamicStateSupported()) {
        command_buffer.setStencilTestEnableEXT(test_enable);
        command_buffer.setStencilOpEXT(vk::StencilFaceFlagBits::eFrontAndBack,
                                       PicaToVK::StencilOp(stencil_test.action_stencil_fail),
                                       PicaToVK::StencilOp(stencil_test.action_depth_pass),
                                       PicaToVK::StencilOp(stencil_test.action_depth_fail),
                                       PicaToVK::CompareFunc(stencil_test.func));
    }

    pipeline_info.depth_stencil.stencil_test_enable.Assign(test_enable);
    pipeline_info.depth_stencil.stencil_fail_op.Assign(stencil_test.action_stencil_fail);
    pipeline_info.depth_stencil.stencil_pass_op.Assign(stencil_test.action_depth_pass);
    pipeline_info.depth_stencil.stencil_depth_fail_op.Assign(stencil_test.action_depth_fail);
    pipeline_info.depth_stencil.stencil_compare_op.Assign(stencil_test.func);
    pipeline_info.depth_stencil.stencil_reference = stencil_test.reference_value;
    pipeline_info.depth_stencil.stencil_write_mask = stencil_test.input_mask;
}

void RasterizerVulkan::SyncDepthTest() {
    const auto& regs = Pica::g_state.regs;

    const bool test_enabled = regs.framebuffer.output_merger.depth_test_enable == 1 ||
                              regs.framebuffer.output_merger.depth_write_enable == 1;
    const auto compare_op = regs.framebuffer.output_merger.depth_test_enable == 1
                                ? regs.framebuffer.output_merger.depth_test_func.Value()
                                : Pica::FramebufferRegs::CompareFunc::Always;

    if (instance.IsExtendedDynamicStateSupported()) {
        vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
        command_buffer.setDepthCompareOpEXT(PicaToVK::CompareFunc(compare_op));
        command_buffer.setDepthTestEnableEXT(test_enabled);
    }

    pipeline_info.depth_stencil.depth_test_enable.Assign(test_enabled);
    pipeline_info.depth_stencil.depth_compare_op.Assign(compare_op);
}

void RasterizerVulkan::SyncCombinerColor() {
    auto combiner_color =
        PicaToVK::ColorRGBA8(Pica::g_state.regs.texturing.tev_combiner_buffer_color.raw);
    if (combiner_color != uniform_block_data.data.tev_combiner_buffer_color) {
        uniform_block_data.data.tev_combiner_buffer_color = combiner_color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncTevConstColor(std::size_t stage_index,
                                         const Pica::TexturingRegs::TevStageConfig& tev_stage) {
    const auto const_color = PicaToVK::ColorRGBA8(tev_stage.const_color);

    if (const_color == uniform_block_data.data.const_color[stage_index]) {
        return;
    }

    uniform_block_data.data.const_color[stage_index] = const_color;
    uniform_block_data.dirty = true;
}

void RasterizerVulkan::SyncGlobalAmbient() {
    auto color = PicaToVK::LightColor(Pica::g_state.regs.lighting.global_ambient);
    if (color != uniform_block_data.data.lighting_global_ambient) {
        uniform_block_data.data.lighting_global_ambient = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightSpecular0(int light_index) {
    auto color = PicaToVK::LightColor(Pica::g_state.regs.lighting.light[light_index].specular_0);
    if (color != uniform_block_data.data.light_src[light_index].specular_0) {
        uniform_block_data.data.light_src[light_index].specular_0 = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightSpecular1(int light_index) {
    auto color = PicaToVK::LightColor(Pica::g_state.regs.lighting.light[light_index].specular_1);
    if (color != uniform_block_data.data.light_src[light_index].specular_1) {
        uniform_block_data.data.light_src[light_index].specular_1 = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightDiffuse(int light_index) {
    auto color = PicaToVK::LightColor(Pica::g_state.regs.lighting.light[light_index].diffuse);
    if (color != uniform_block_data.data.light_src[light_index].diffuse) {
        uniform_block_data.data.light_src[light_index].diffuse = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightAmbient(int light_index) {
    auto color = PicaToVK::LightColor(Pica::g_state.regs.lighting.light[light_index].ambient);
    if (color != uniform_block_data.data.light_src[light_index].ambient) {
        uniform_block_data.data.light_src[light_index].ambient = color;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightPosition(int light_index) {
    const Common::Vec3f position = {
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].x).ToFloat32(),
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].y).ToFloat32(),
        Pica::float16::FromRaw(Pica::g_state.regs.lighting.light[light_index].z).ToFloat32()};

    if (position != uniform_block_data.data.light_src[light_index].position) {
        uniform_block_data.data.light_src[light_index].position = position;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightSpotDirection(int light_index) {
    const auto& light = Pica::g_state.regs.lighting.light[light_index];
    const auto spot_direction = Common::Vec3f{light.spot_x, light.spot_y, light.spot_z} / 2047.0f;

    if (spot_direction != uniform_block_data.data.light_src[light_index].spot_direction) {
        uniform_block_data.data.light_src[light_index].spot_direction = spot_direction;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightDistanceAttenuationBias(int light_index) {
    float dist_atten_bias =
        Pica::float20::FromRaw(Pica::g_state.regs.lighting.light[light_index].dist_atten_bias)
            .ToFloat32();

    if (dist_atten_bias != uniform_block_data.data.light_src[light_index].dist_atten_bias) {
        uniform_block_data.data.light_src[light_index].dist_atten_bias = dist_atten_bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncLightDistanceAttenuationScale(int light_index) {
    float dist_atten_scale =
        Pica::float20::FromRaw(Pica::g_state.regs.lighting.light[light_index].dist_atten_scale)
            .ToFloat32();

    if (dist_atten_scale != uniform_block_data.data.light_src[light_index].dist_atten_scale) {
        uniform_block_data.data.light_src[light_index].dist_atten_scale = dist_atten_scale;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncShadowBias() {
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

void RasterizerVulkan::SyncShadowTextureBias() {
    int bias = Pica::g_state.regs.texturing.shadow.bias << 1;
    if (bias != uniform_block_data.data.shadow_texture_bias) {
        uniform_block_data.data.shadow_texture_bias = bias;
        uniform_block_data.dirty = true;
    }
}

void RasterizerVulkan::SyncAndUploadLUTsLF() {
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 256 * Pica::LightingRegs::NumLightingSampler +
        sizeof(Common::Vec2f) * 128; // fog

    if (!uniform_block_data.lighting_lut_dirty_any && !uniform_block_data.fog_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_lf_buffer.Map(max_size, sizeof(Common::Vec4f));

    // Sync the lighting luts
    if (uniform_block_data.lighting_lut_dirty_any || invalidate) {
        for (unsigned index = 0; index < uniform_block_data.lighting_lut_dirty.size(); index++) {
            if (uniform_block_data.lighting_lut_dirty[index] || invalidate) {
                std::array<Common::Vec2f, 256> new_data;
                const auto& source_lut = Pica::g_state.lighting.luts[index];
                std::transform(source_lut.begin(), source_lut.end(), new_data.begin(),
                               [](const auto& entry) {
                                   return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                               });

                if (new_data != lighting_lut_data[index] || invalidate) {
                    lighting_lut_data[index] = new_data;
                    std::memcpy(buffer + bytes_used, new_data.data(),
                                new_data.size() * sizeof(Common::Vec2f));
                    uniform_block_data.data.lighting_lut_offset[index / 4][index % 4] =
                        static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
                    uniform_block_data.dirty = true;
                    bytes_used += new_data.size() * sizeof(Common::Vec2f);
                }
                uniform_block_data.lighting_lut_dirty[index] = false;
            }
        }
        uniform_block_data.lighting_lut_dirty_any = false;
    }

    // Sync the fog lut
    if (uniform_block_data.fog_lut_dirty || invalidate) {
        std::array<Common::Vec2f, 128> new_data;

        std::transform(Pica::g_state.fog.lut.begin(), Pica::g_state.fog.lut.end(), new_data.begin(),
                       [](const auto& entry) {
                           return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
                       });

        if (new_data != fog_lut_data || invalidate) {
            fog_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec2f));
            uniform_block_data.data.fog_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec2f);
        }
        uniform_block_data.fog_lut_dirty = false;
    }

    texture_lf_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::SyncAndUploadLUTs() {
    constexpr std::size_t max_size =
        sizeof(Common::Vec2f) * 128 * 3 + // proctex: noise + color + alpha
        sizeof(Common::Vec4f) * 256 +     // proctex
        sizeof(Common::Vec4f) * 256;      // proctex diff

    if (!uniform_block_data.proctex_noise_lut_dirty &&
        !uniform_block_data.proctex_color_map_dirty &&
        !uniform_block_data.proctex_alpha_map_dirty && !uniform_block_data.proctex_lut_dirty &&
        !uniform_block_data.proctex_diff_lut_dirty) {
        return;
    }

    std::size_t bytes_used = 0;
    auto [buffer, offset, invalidate] = texture_buffer.Map(max_size, sizeof(Common::Vec4f));

    // helper function for SyncProcTexNoiseLUT/ColorMap/AlphaMap
    auto SyncProcTexValueLUT =
        [this, &buffer = buffer, &offset = offset, &invalidate = invalidate,
         &bytes_used](const std::array<Pica::State::ProcTex::ValueEntry, 128>& lut,
                      std::array<Common::Vec2f, 128>& lut_data, int& lut_offset) {
            std::array<Common::Vec2f, 128> new_data;
            std::transform(lut.begin(), lut.end(), new_data.begin(), [](const auto& entry) {
                return Common::Vec2f{entry.ToFloat(), entry.DiffToFloat()};
            });

            if (new_data != lut_data || invalidate) {
                lut_data = new_data;
                std::memcpy(buffer + bytes_used, new_data.data(),
                            new_data.size() * sizeof(Common::Vec2f));
                lut_offset = static_cast<int>((offset + bytes_used) / sizeof(Common::Vec2f));
                uniform_block_data.dirty = true;
                bytes_used += new_data.size() * sizeof(Common::Vec2f);
            }
        };

    // Sync the proctex noise lut
    if (uniform_block_data.proctex_noise_lut_dirty || invalidate) {
        SyncProcTexValueLUT(Pica::g_state.proctex.noise_table, proctex_noise_lut_data,
                            uniform_block_data.data.proctex_noise_lut_offset);
        uniform_block_data.proctex_noise_lut_dirty = false;
    }

    // Sync the proctex color map
    if (uniform_block_data.proctex_color_map_dirty || invalidate) {
        SyncProcTexValueLUT(Pica::g_state.proctex.color_map_table, proctex_color_map_data,
                            uniform_block_data.data.proctex_color_map_offset);
        uniform_block_data.proctex_color_map_dirty = false;
    }

    // Sync the proctex alpha map
    if (uniform_block_data.proctex_alpha_map_dirty || invalidate) {
        SyncProcTexValueLUT(Pica::g_state.proctex.alpha_map_table, proctex_alpha_map_data,
                            uniform_block_data.data.proctex_alpha_map_offset);
        uniform_block_data.proctex_alpha_map_dirty = false;
    }

    // Sync the proctex lut
    if (uniform_block_data.proctex_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::transform(Pica::g_state.proctex.color_table.begin(),
                       Pica::g_state.proctex.color_table.end(), new_data.begin(),
                       [](const auto& entry) {
                           auto rgba = entry.ToVector() / 255.0f;
                           return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
                       });

        if (new_data != proctex_lut_data || invalidate) {
            proctex_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec4f));
            uniform_block_data.data.proctex_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }
        uniform_block_data.proctex_lut_dirty = false;
    }

    // Sync the proctex difference lut
    if (uniform_block_data.proctex_diff_lut_dirty || invalidate) {
        std::array<Common::Vec4f, 256> new_data;

        std::transform(Pica::g_state.proctex.color_diff_table.begin(),
                       Pica::g_state.proctex.color_diff_table.end(), new_data.begin(),
                       [](const auto& entry) {
                           auto rgba = entry.ToVector() / 255.0f;
                           return Common::Vec4f{rgba.r(), rgba.g(), rgba.b(), rgba.a()};
                       });

        if (new_data != proctex_diff_lut_data || invalidate) {
            proctex_diff_lut_data = new_data;
            std::memcpy(buffer + bytes_used, new_data.data(),
                        new_data.size() * sizeof(Common::Vec4f));
            uniform_block_data.data.proctex_diff_lut_offset =
                static_cast<int>((offset + bytes_used) / sizeof(Common::Vec4f));
            uniform_block_data.dirty = true;
            bytes_used += new_data.size() * sizeof(Common::Vec4f);
        }
        uniform_block_data.proctex_diff_lut_dirty = false;
    }

    texture_buffer.Commit(static_cast<u32>(bytes_used));
}

void RasterizerVulkan::UploadUniforms(bool accelerate_draw) {
    const bool sync_vs = accelerate_draw;
    const bool sync_fs = uniform_block_data.dirty;

    if (!sync_vs && !sync_fs) {
        return;
    }

    u32 used_bytes = 0;
    const u32 uniform_size = static_cast<u32>(uniform_size_aligned_vs + uniform_size_aligned_fs);
    auto [uniforms, offset, invalidate] =
        uniform_buffer.Map(uniform_size, static_cast<u32>(uniform_buffer_alignment));

    if (sync_vs) {
        Pica::Shader::VSUniformData vs_uniforms;
        vs_uniforms.uniforms.SetFromRegs(Pica::g_state.regs.vs, Pica::g_state.vs);
        std::memcpy(uniforms + used_bytes, &vs_uniforms, sizeof(vs_uniforms));

        pipeline_cache.BindBuffer(0, uniform_buffer.GetHandle(), offset + used_bytes,
                                  sizeof(vs_uniforms));
        used_bytes += static_cast<u32>(uniform_size_aligned_vs);
    }

    if (sync_fs || invalidate) {
        std::memcpy(uniforms + used_bytes, &uniform_block_data.data,
                    sizeof(Pica::Shader::UniformData));

        pipeline_cache.BindBuffer(1, uniform_buffer.GetHandle(), offset + used_bytes,
                                  sizeof(uniform_block_data.data));
        uniform_block_data.dirty = false;
        used_bytes += static_cast<u32>(uniform_size_aligned_fs);
    }

    uniform_buffer.Commit(used_bytes);
}

} // namespace Vulkan
