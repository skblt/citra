// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/common_paths.h"
#include "common/file_util.h"
#include "common/logging/log.h"
#include "video_core/renderer_vulkan/pica_to_vk.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

namespace Vulkan {

struct Bindings {
    std::array<vk::DescriptorType, MAX_DESCRIPTORS> bindings;
    u32 binding_count;
};

constexpr u32 RASTERIZER_SET_COUNT = 4;
constexpr static std::array RASTERIZER_SETS = {
    Bindings{
        // Utility set
        .bindings = {
            vk::DescriptorType::eUniformBuffer,
            vk::DescriptorType::eUniformBuffer,
            vk::DescriptorType::eUniformTexelBuffer,
            vk::DescriptorType::eUniformTexelBuffer,
            vk::DescriptorType::eUniformTexelBuffer
        },
        .binding_count = 5
    },
    Bindings{
        // Texture set
        .bindings = {
            vk::DescriptorType::eSampledImage,
            vk::DescriptorType::eSampledImage,
            vk::DescriptorType::eSampledImage,
            vk::DescriptorType::eSampledImage
        },
        .binding_count = 4
    },
    Bindings{
        // Sampler set
        .bindings = {
            vk::DescriptorType::eSampler,
            vk::DescriptorType::eSampler,
            vk::DescriptorType::eSampler,
            vk::DescriptorType::eSampler
        },
        .binding_count = 4
    },
    Bindings {
        // Shadow set
        .bindings = {
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eStorageImage,
            vk::DescriptorType::eStorageImage
        },
        .binding_count = 7
    }
};

constexpr vk::ShaderStageFlags ToVkStageFlags(vk::DescriptorType type) {
    vk::ShaderStageFlags flags;
    switch (type) {
    case vk::DescriptorType::eSampler:
    case vk::DescriptorType::eSampledImage:
    case vk::DescriptorType::eUniformTexelBuffer:
    case vk::DescriptorType::eStorageImage:
        flags = vk::ShaderStageFlagBits::eFragment;
        break;
    case vk::DescriptorType::eUniformBuffer:
    case vk::DescriptorType::eUniformBufferDynamic:
        flags = vk::ShaderStageFlagBits::eFragment |
                vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eGeometry |
                vk::ShaderStageFlagBits::eCompute;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown descriptor type!");
    }

    return flags;
}

u32 AttribBytes(VertexAttribute attrib) {
    switch (attrib.type) {
    case AttribType::Float:
        return sizeof(float) * attrib.size;
    case AttribType::Int:
        return sizeof(u32) * attrib.size;
    case AttribType::Short:
        return sizeof(u16) * attrib.size;
    case AttribType::Byte:
    case AttribType::Ubyte:
        return sizeof(u8) * attrib.size;
    }
}

vk::Format ToVkAttributeFormat(VertexAttribute attrib) {
    switch (attrib.type) {
    case AttribType::Float:
        switch (attrib.size) {
        case 1: return vk::Format::eR32Sfloat;
        case 2: return vk::Format::eR32G32Sfloat;
        case 3: return vk::Format::eR32G32B32Sfloat;
        case 4: return vk::Format::eR32G32B32A32Sfloat;
        }
    default:
        LOG_CRITICAL(Render_Vulkan, "Unimplemented vertex attribute format!");
        UNREACHABLE();
    }

    return vk::Format::eR32Sfloat;
}

vk::ShaderStageFlagBits ToVkShaderStage(std::size_t index) {
    switch (index) {
    case 0: return vk::ShaderStageFlagBits::eVertex;
    case 1: return vk::ShaderStageFlagBits::eFragment;
    case 2: return vk::ShaderStageFlagBits::eGeometry;
    default:
        LOG_CRITICAL(Render_Vulkan, "Invalid shader stage index!");
        UNREACHABLE();
    }

    return vk::ShaderStageFlagBits::eVertex;
}

PipelineCache::PipelineCache(const Instance& instance, TaskScheduler& scheduler, RenderpassCache& renderpass_cache)
    : instance{instance}, scheduler{scheduler}, renderpass_cache{renderpass_cache} {
    descriptor_dirty.fill(true);

    LoadDiskCache();
}

PipelineCache::~PipelineCache() {
    vk::Device device = instance.GetDevice();

    SaveDiskCache();

    device.destroyPipelineLayout(layout);
    for (std::size_t i = 0; i < MAX_DESCRIPTOR_SETS; i++) {
        device.destroyDescriptorSetLayout(descriptor_set_layouts[i]);
        device.destroyDescriptorUpdateTemplate(update_templates[i]);
    }

    for (const auto& [hash, pipeline] : graphics_pipelines) {
        device.destroyPipeline(pipeline);
    }

    graphics_pipelines.clear();
}

void PipelineCache::BindPipeline(const PipelineInfo& info) {
    ApplyDynamic(info);

    u64 shader_hash = 0;
    for (u32 i = 0; i < MAX_SHADER_STAGES; i++) {
        shader_hash = Common::HashCombine(shader_hash, shader_hashes[i]);
    }

    const u64 info_hash_size = instance.IsExtendedDynamicStateSupported() ?
            offsetof(PipelineInfo, rasterization) :
            offsetof(PipelineInfo, depth_stencil) + offsetof(DepthStencilState, stencil_reference);

    u64 info_hash = Common::ComputeHash64(&info, info_hash_size);
    u64 pipeline_hash = Common::HashCombine(shader_hash, info_hash);

    auto [it, new_pipeline] = graphics_pipelines.try_emplace(pipeline_hash, vk::Pipeline{});
    if (new_pipeline) {
        it->second = BuildPipeline(info);
    }

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, it->second);

    BindDescriptorSets();
}

bool PipelineCache::UseProgrammableVertexShader(const Pica::Regs& regs, Pica::Shader::ShaderSetup& setup) {
    const PicaVSConfig config{regs.vs, setup};
    auto [handle, result] = programmable_vertex_shaders.Get(config, setup, vk::ShaderStageFlagBits::eVertex,
                                                            instance.GetDevice(), ShaderOptimization::Debug);
    if (!handle) {
        return false;
    }

    current_shaders[ProgramType::VS] = handle;
    shader_hashes[ProgramType::VS] = config.Hash();
    return true;
}

void PipelineCache::UseTrivialVertexShader() {
    current_shaders[ProgramType::VS] = trivial_vertex_shader;
    shader_hashes[ProgramType::VS] = 0;
}

void PipelineCache::UseFixedGeometryShader(const Pica::Regs& regs) {
    const PicaFixedGSConfig gs_config{regs};
    auto [handle, _] = fixed_geometry_shaders.Get(gs_config, vk::ShaderStageFlagBits::eGeometry,
                                                  instance.GetDevice(), ShaderOptimization::Debug);
    current_shaders[ProgramType::GS] = handle;
    shader_hashes[ProgramType::GS] = gs_config.Hash();
}

void PipelineCache::UseTrivialGeometryShader() {
    current_shaders[ProgramType::GS] = VK_NULL_HANDLE;
    shader_hashes[ProgramType::GS] = 0;
}

void PipelineCache::UseFragmentShader(const Pica::Regs& regs) {
    const PicaFSConfig config = PicaFSConfig::BuildFromRegs(regs);
    auto [handle, result] = fragment_shaders.Get(config, vk::ShaderStageFlagBits::eFragment,
                                                 instance.GetDevice(), ShaderOptimization::Debug);
    current_shaders[ProgramType::FS] = handle;
    shader_hashes[ProgramType::FS] = config.Hash();
}

void PipelineCache::BindTexture(u32 set, u32 descriptor, vk::ImageView image_view) {
    const DescriptorData data = {
        .image_info = vk::DescriptorImageInfo{
            .imageView = image_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        }
    };

    SetBinding(set, descriptor, data);
}

void PipelineCache::BindBuffer(u32 set, u32 descriptor, vk::Buffer buffer, u32 offset, u32 size) {
    const DescriptorData data = {
        .buffer_info = vk::DescriptorBufferInfo{
            .buffer = buffer,
            .offset = offset,
            .range = size
        }
    };

    SetBinding(set, descriptor, data);
}

void PipelineCache::BindTexelBuffer(u32 set, u32 descriptor, vk::BufferView buffer_view) {
    const DescriptorData data = {
        .buffer_view = buffer_view
    };

    SetBinding(set, descriptor, data);
}

void PipelineCache::BindSampler(u32 set, u32 descriptor, vk::Sampler sampler) {
    const DescriptorData data = {
        .image_info = vk::DescriptorImageInfo{
            .sampler = sampler
        }
    };

    SetBinding(set, descriptor, data);
}

void PipelineCache::SetViewport(float x, float y, float width, float height) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setViewport(0, vk::Viewport{x, y, width, height, 0.f, 1.f});
}

void PipelineCache::SetScissor(s32 x, s32 y, u32 width, u32 height) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setScissor(0, vk::Rect2D{{x, y}, {width, height}});
}

void PipelineCache::MarkDescriptorSetsDirty() {
    descriptor_dirty.fill(true);
}

void PipelineCache::ApplyDynamic(const PipelineInfo& info) {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, info.depth_stencil.stencil_compare_mask);
    command_buffer.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, info.depth_stencil.stencil_write_mask);
    command_buffer.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, info.depth_stencil.stencil_reference);

    if (instance.IsExtendedDynamicStateSupported()) {
        command_buffer.setCullModeEXT(PicaToVK::CullMode(info.rasterization.cull_mode));
        command_buffer.setDepthCompareOpEXT(PicaToVK::CompareFunc(info.depth_stencil.depth_compare_op));
        command_buffer.setDepthTestEnableEXT(info.depth_stencil.depth_test_enable);
        command_buffer.setDepthWriteEnableEXT(info.depth_stencil.depth_write_enable);
        command_buffer.setFrontFaceEXT(PicaToVK::FrontFace(info.rasterization.cull_mode));
        command_buffer.setPrimitiveTopologyEXT(PicaToVK::PrimitiveTopology(info.rasterization.topology));
        command_buffer.setStencilTestEnableEXT(info.depth_stencil.stencil_test_enable);
        command_buffer.setStencilOpEXT(vk::StencilFaceFlagBits::eFrontAndBack,
                                       PicaToVK::StencilOp(info.depth_stencil.stencil_fail_op),
                                       PicaToVK::StencilOp(info.depth_stencil.stencil_pass_op),
                                       PicaToVK::StencilOp(info.depth_stencil.stencil_depth_fail_op),
                                       PicaToVK::CompareFunc(info.depth_stencil.stencil_compare_op));
    }
}

void PipelineCache::SetBinding(u32 set, u32 binding, DescriptorData data) {
    if (update_data[set][binding] != data) {
        update_data[set][binding] = data;
        descriptor_dirty[set] = true;
    }
}

void PipelineCache::BuildLayout() {
    std::array<vk::DescriptorSetLayoutBinding, MAX_DESCRIPTORS> set_bindings;
    std::array<vk::DescriptorUpdateTemplateEntry, MAX_DESCRIPTORS> update_entries;

    vk::Device device = instance.GetDevice();
    for (u32 i = 0; i < RASTERIZER_SET_COUNT; i++) {
        const auto& set = RASTERIZER_SETS[i];
        for (u32 j = 0; j < set.binding_count; j++) {
            vk::DescriptorType type = set.bindings[j];
            set_bindings[j] = vk::DescriptorSetLayoutBinding{
                .binding = j,
                .descriptorType = type,
                .descriptorCount = 1,
                .stageFlags = ToVkStageFlags(type)
            };

            update_entries[j] = vk::DescriptorUpdateTemplateEntry{
                .dstBinding = j,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = type,
                .offset = j * sizeof(DescriptorData),
                .stride = 0
            };
        }

        const vk::DescriptorSetLayoutCreateInfo layout_info = {
            .bindingCount = set.binding_count,
            .pBindings = set_bindings.data()
        };

        // Create descriptor set layout
        descriptor_set_layouts[i] = device.createDescriptorSetLayout(layout_info);

        const vk::DescriptorUpdateTemplateCreateInfo template_info = {
            .descriptorUpdateEntryCount = set.binding_count,
            .pDescriptorUpdateEntries = update_entries.data(),
            .descriptorSetLayout = descriptor_set_layouts[i]
        };

        // Create descriptor set update template
        update_templates[i] = device.createDescriptorUpdateTemplate(template_info);
    }

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = RASTERIZER_SET_COUNT,
        .pSetLayouts = descriptor_set_layouts.data(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr
    };

    layout = device.createPipelineLayout(layout_info);
}

vk::Pipeline PipelineCache::BuildPipeline(const PipelineInfo& info) {
    vk::Device device = instance.GetDevice();

    u32 shader_count = 0;
    std::array<vk::PipelineShaderStageCreateInfo, MAX_SHADER_STAGES> shader_stages;
    for (std::size_t i = 0; i < current_shaders.size(); i++) {
        vk::ShaderModule shader = current_shaders[i];
        if (!shader) {
            continue;
        }

        shader_stages[i] = vk::PipelineShaderStageCreateInfo{
            .stage = ToVkShaderStage(i),
            .module = shader,
            .pName = "main"
        };
    }

    /**
     * Vulkan doesn't intuitively support fixed attributes. To avoid duplicating the data and increasing
     * data upload, when the fixed flag is true, we specify VK_VERTEX_INPUT_RATE_INSTANCE as the input rate.
     * Since one instance is all we render, the shader will always read the single attribute.
     */
    std::array<vk::VertexInputBindingDescription, MAX_VERTEX_BINDINGS> bindings;
    for (u32 i = 0; i < info.vertex_layout.binding_count; i++) {
        const auto& binding = info.vertex_layout.bindings[i];
        bindings[i] = vk::VertexInputBindingDescription{
            .binding = binding.binding,
            .stride = binding.stride,
            .inputRate = binding.fixed.Value() ? vk::VertexInputRate::eInstance
                                               : vk::VertexInputRate::eVertex
        };
    }

    // Populate vertex attribute structures
    std::array<vk::VertexInputAttributeDescription, MAX_VERTEX_ATTRIBUTES> attributes;
    for (u32 i = 0; i < info.vertex_layout.attribute_count; i++) {
        const auto& attr = info.vertex_layout.attributes[i];
        attributes[i] = vk::VertexInputAttributeDescription{
            .location = attr.location,
            .binding = attr.binding,
            .format = ToVkAttributeFormat(attr),
            .offset = attr.offset
        };
    }

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = info.vertex_layout.binding_count,
        .pVertexBindingDescriptions = bindings.data(),
        .vertexAttributeDescriptionCount = info.vertex_layout.attribute_count,
        .pVertexAttributeDescriptions = attributes.data()
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = PicaToVK::PrimitiveTopology(info.rasterization.topology),
        .primitiveRestartEnable = false
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = PicaToVK::CullMode(info.rasterization.cull_mode),
        .frontFace = PicaToVK::FrontFace(info.rasterization.cull_mode),
        .depthBiasEnable = false,
        .lineWidth = 1.0f
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples  = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false
    };

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = info.blending.blend_enable.Value(),
        .srcColorBlendFactor = PicaToVK::BlendFunc(info.blending.src_color_blend_factor),
        .dstColorBlendFactor = PicaToVK::BlendFunc(info.blending.dst_color_blend_factor),
        .colorBlendOp = PicaToVK::BlendEquation(info.blending.color_blend_eq),
        .srcAlphaBlendFactor = PicaToVK::BlendFunc(info.blending.src_alpha_blend_factor),
        .dstAlphaBlendFactor = PicaToVK::BlendFunc(info.blending.dst_alpha_blend_factor),
        .alphaBlendOp = PicaToVK::BlendEquation(info.blending.alpha_blend_eq),
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = info.blending.logic_op_enable.Value(),
        .logicOp = PicaToVK::LogicOp(info.blending.logic_op),
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f}
    };

    const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &placeholder_viewport,
        .scissorCount = 1,
        .pScissors = &placeholder_scissor,
    };

    const bool extended_dynamic_states = instance.IsExtendedDynamicStateSupported();
    const std::array dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
        vk::DynamicState::eLineWidth,
        vk::DynamicState::eStencilCompareMask,
        vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilReference,
        // VK_EXT_extended_dynamic_state
        vk::DynamicState::eCullModeEXT,
        vk::DynamicState::eDepthCompareOpEXT,
        vk::DynamicState::eDepthTestEnableEXT,
        vk::DynamicState::eDepthWriteEnableEXT,
        vk::DynamicState::eFrontFaceEXT,
        vk::DynamicState::ePrimitiveTopologyEXT,
        vk::DynamicState::eStencilOpEXT,
        vk::DynamicState::eStencilTestEnableEXT,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = extended_dynamic_states ? 14u : 6u,
        .pDynamicStates = dynamic_states.data()
    };

    const vk::StencilOpState stencil_op_state = {
        .failOp = PicaToVK::StencilOp(info.depth_stencil.stencil_fail_op),
        .passOp = PicaToVK::StencilOp(info.depth_stencil.stencil_pass_op),
        .depthFailOp = PicaToVK::StencilOp(info.depth_stencil.stencil_depth_fail_op),
        .compareOp = PicaToVK::CompareFunc(info.depth_stencil.stencil_compare_op)
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = static_cast<u32>(info.depth_stencil.depth_test_enable.Value()),
        .depthWriteEnable = static_cast<u32>(info.depth_stencil.depth_write_enable.Value()),
        .depthCompareOp = PicaToVK::CompareFunc(info.depth_stencil.depth_compare_op),
        .depthBoundsTestEnable = false,
        .stencilTestEnable = static_cast<u32>(info.depth_stencil.stencil_test_enable.Value()),
        .front = stencil_op_state,
        .back = stencil_op_state
    };

    const vk::GraphicsPipelineCreateInfo pipeline_info = {
        .stageCount = shader_count,
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_state,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth_info,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = layout,
        .renderPass = renderpass_cache.GetRenderpass(info.color_attachment,
                                                     info.depth_attachment, false)
    };

    if (const auto result = device.createGraphicsPipeline(pipeline_cache, pipeline_info);
            result.result == vk::Result::eSuccess) {
        return result.value;
    } else {
       LOG_CRITICAL(Render_Vulkan, "Graphics pipeline creation failed!");
       UNREACHABLE();
    }

    return VK_NULL_HANDLE;
}

void PipelineCache::BindDescriptorSets() {
    vk::Device device = instance.GetDevice();
    for (u32 i = 0; i < RASTERIZER_SET_COUNT; i++) {
        if (descriptor_dirty[i] || !descriptor_sets[i]) {
            const vk::DescriptorSetAllocateInfo alloc_info = {
                .descriptorPool = scheduler.GetDescriptorPool(),
                .descriptorSetCount = 1,
                .pSetLayouts = &descriptor_set_layouts[i]
            };

            vk::DescriptorSet set = device.allocateDescriptorSets(alloc_info)[0];
            device.updateDescriptorSetWithTemplate(set, update_templates[i], update_data[i].data());

            descriptor_sets[i] = set;
            descriptor_dirty[i] = false;
        }
    }

    // Bind the descriptor sets
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, RASTERIZER_SET_COUNT,
                                      descriptor_sets.data(), 0, nullptr);
}

void PipelineCache::LoadDiskCache() {
    const std::string cache_path =
            FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + DIR_SEP "vulkan" + DIR_SEP "pipelines.bin";

    FileUtil::IOFile cache_file{cache_path, "r"};
    if (!cache_file.IsOpen()) {
        LOG_INFO(Render_Vulkan, "No pipeline cache found");
    }

    const u32 cache_file_size = cache_file.GetSize();
    auto cache_data = std::vector<u8>(cache_file_size);
    if (!cache_file.ReadBytes(cache_data.data(), cache_file_size)) {
        LOG_WARNING(Render_Vulkan, "Error during pipeline cache read");
        return;
    }

    cache_file.Close();

    const bool is_valid = ValidateData(cache_data.data(), cache_file_size);
    const vk::PipelineCacheCreateInfo cache_info = {
        .initialDataSize = is_valid ? cache_file_size : 0,
        .pInitialData = cache_data.data()
    };

    vk::Device device = instance.GetDevice();
    pipeline_cache = device.createPipelineCache(cache_info);
}

void PipelineCache::SaveDiskCache() {
    const std::string cache_path =
            FileUtil::GetUserPath(FileUtil::UserPath::ShaderDir) + DIR_SEP "vulkan" + DIR_SEP "pipelines.bin";

    FileUtil::IOFile cache_file{cache_path, "w"};
    if (!cache_file.IsOpen()) {
        LOG_INFO(Render_Vulkan, "Unable to open pipeline cache for writing");
        return;
    }

    vk::Device device = instance.GetDevice();
    auto cache_data = device.getPipelineCacheData(pipeline_cache);
    if (!cache_file.WriteBytes(cache_data.data(), cache_data.size())) {
        LOG_WARNING(Render_Vulkan, "Error during pipeline cache write");
        return;
    }

    cache_file.Close();
}

bool PipelineCache::ValidateData(const u8* data, u32 size) {
    if (size < sizeof(vk::PipelineCacheHeaderVersionOne)) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Invalid header");
        return false;
    }

    vk::PipelineCacheHeaderVersionOne header;
    std::memcpy(&header, data, sizeof(header));
    if (header.headerSize < sizeof(header)) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Invalid header length");
        return false;
    }

    if (header.headerVersion != vk::PipelineCacheHeaderVersion::eOne) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Invalid header version");
        return false;
    }

    if (u32 vendor_id = instance.GetVendorID(); header.vendorID != vendor_id) {
        LOG_ERROR(Render_Vulkan,
                  "Pipeline cache failed validation: Incorrect vendor ID (file: {:#X}, device: {:#X})",
                   header.vendorID, vendor_id);
        return false;
    }

    if (u32 device_id = instance.GetDeviceID(); header.deviceID != device_id) {
        LOG_ERROR(Render_Vulkan,
                  "Pipeline cache failed validation: Incorrect device ID (file: {:#X}, device: {:#X})",
                  header.deviceID, device_id);
        return false;
    }

    if (header.pipelineCacheUUID != instance.GetPipelineCacheUUID()) {
        LOG_ERROR(Render_Vulkan, "Pipeline cache failed validation: Incorrect UUID");
        return false;
    }

    return true;
}

} // namespace Vulkan
