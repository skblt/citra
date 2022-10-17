// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include "video_core/renderer_vulkan/vk_descriptor_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"

namespace Vulkan {

struct Bindings {
    std::array<vk::DescriptorType, MAX_DESCRIPTORS> bindings;
    u32 binding_count;
};

constexpr u32 DESCRIPTOR_BATCH_SIZE = 8;
constexpr u32 RASTERIZER_SET_COUNT = 4;
constexpr static std::array RASTERIZER_SETS = {
    Bindings{// Utility set
             .bindings = {vk::DescriptorType::eUniformBuffer, vk::DescriptorType::eUniformBuffer,
                          vk::DescriptorType::eUniformTexelBuffer,
                          vk::DescriptorType::eUniformTexelBuffer,
                          vk::DescriptorType::eUniformTexelBuffer},
             .binding_count = 5},
    Bindings{// Texture set
             .bindings = {vk::DescriptorType::eSampledImage, vk::DescriptorType::eSampledImage,
                          vk::DescriptorType::eSampledImage, vk::DescriptorType::eSampledImage},
             .binding_count = 4},
    Bindings{// Sampler set
             .bindings = {vk::DescriptorType::eSampler, vk::DescriptorType::eSampler,
                          vk::DescriptorType::eSampler, vk::DescriptorType::eSampler},
             .binding_count = 4},
    Bindings{// Shadow set
             .bindings = {vk::DescriptorType::eStorageImage, vk::DescriptorType::eStorageImage,
                          vk::DescriptorType::eStorageImage, vk::DescriptorType::eStorageImage,
                          vk::DescriptorType::eStorageImage, vk::DescriptorType::eStorageImage,
                          vk::DescriptorType::eStorageImage},
             .binding_count = 7}};

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
        flags = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex |
                vk::ShaderStageFlagBits::eGeometry | vk::ShaderStageFlagBits::eCompute;
        break;
    default:
        LOG_ERROR(Render_Vulkan, "Unknown descriptor type!");
    }

    return flags;
}

DescriptorManager::DescriptorManager(const Instance& instance, TaskScheduler& scheduler)
    : instance{instance}, scheduler{scheduler} {
    descriptor_dirty.fill(true);
    BuildLayouts();
}

DescriptorManager::~DescriptorManager() {
    vk::Device device = instance.GetDevice();
    device.destroyPipelineLayout(layout);

    for (std::size_t i = 0; i < MAX_DESCRIPTOR_SETS; i++) {
        device.destroyDescriptorSetLayout(descriptor_set_layouts[i]);
        device.destroyDescriptorUpdateTemplate(update_templates[i]);
    }
}

void DescriptorManager::SetBinding(u32 set, u32 binding, DescriptorData data) {
    if (update_data[set][binding] != data) {
        update_data[set][binding] = data;
        descriptor_dirty[set] = true;
    }
}

void DescriptorManager::BindDescriptorSets() {
    vk::Device device = instance.GetDevice();
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_BATCH_SIZE> layouts;

    for (u32 i = 0; i < RASTERIZER_SET_COUNT; i++) {
        if (descriptor_dirty[i] || !descriptor_sets[i]) {
            auto& batch = descriptor_batch[i];
            if (batch.empty()) {
                layouts.fill(descriptor_set_layouts[i]);
                const vk::DescriptorSetAllocateInfo alloc_info = {
                    .descriptorPool = scheduler.GetDescriptorPool(),
                    .descriptorSetCount = DESCRIPTOR_BATCH_SIZE,
                    .pSetLayouts = layouts.data()};

                try {
                    batch = device.allocateDescriptorSets(alloc_info);
                } catch (vk::OutOfPoolMemoryError& err) {
                    LOG_CRITICAL(Render_Vulkan, "Run out of pool memory for layout {}: {}", i,
                                 err.what());
                    UNREACHABLE();
                }
            }

            vk::DescriptorSet set = batch.back();
            device.updateDescriptorSetWithTemplate(set, update_templates[i], update_data[i][0]);

            descriptor_sets[i] = set;
            descriptor_dirty[i] = false;
            batch.pop_back();
        }
    }

    // Bind the descriptor sets
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0,
                                      RASTERIZER_SET_COUNT, descriptor_sets.data(), 0, nullptr);
}

void DescriptorManager::MarkDirty() {
    descriptor_dirty.fill(true);
    for (auto& batch : descriptor_batch) {
        batch.clear();
    }
}

void DescriptorManager::BuildLayouts() {
    std::array<vk::DescriptorSetLayoutBinding, MAX_DESCRIPTORS> set_bindings;
    std::array<vk::DescriptorUpdateTemplateEntry, MAX_DESCRIPTORS> update_entries;

    vk::Device device = instance.GetDevice();
    for (u32 i = 0; i < RASTERIZER_SET_COUNT; i++) {
        const auto& set = RASTERIZER_SETS[i];
        for (u32 j = 0; j < set.binding_count; j++) {
            vk::DescriptorType type = set.bindings[j];
            set_bindings[j] = vk::DescriptorSetLayoutBinding{.binding = j,
                                                             .descriptorType = type,
                                                             .descriptorCount = 1,
                                                             .stageFlags = ToVkStageFlags(type)};

            update_entries[j] =
                vk::DescriptorUpdateTemplateEntry{.dstBinding = j,
                                                  .dstArrayElement = 0,
                                                  .descriptorCount = 1,
                                                  .descriptorType = type,
                                                  .offset = j * sizeof(DescriptorData),
                                                  .stride = 0};
        }

        const vk::DescriptorSetLayoutCreateInfo layout_info = {.bindingCount = set.binding_count,
                                                               .pBindings = set_bindings.data()};

        // Create descriptor set layout
        descriptor_set_layouts[i] = device.createDescriptorSetLayout(layout_info);

        const vk::DescriptorUpdateTemplateCreateInfo template_info = {
            .descriptorUpdateEntryCount = set.binding_count,
            .pDescriptorUpdateEntries = update_entries.data(),
            .templateType = vk::DescriptorUpdateTemplateType::eDescriptorSet,
            .descriptorSetLayout = descriptor_set_layouts[i]};

        // Create descriptor set update template
        update_templates[i] = device.createDescriptorUpdateTemplate(template_info);
    }

    const vk::PipelineLayoutCreateInfo layout_info = {.setLayoutCount = RASTERIZER_SET_COUNT,
                                                      .pSetLayouts = descriptor_set_layouts.data(),
                                                      .pushConstantRangeCount = 0,
                                                      .pPushConstantRanges = nullptr};

    layout = device.createPipelineLayout(layout_info);
}

} // namespace Vulkan
