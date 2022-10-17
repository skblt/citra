// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

constexpr u32 MAX_DESCRIPTORS = 8;
constexpr u32 MAX_DESCRIPTOR_SETS = 6;

union DescriptorData {
    vk::DescriptorImageInfo image_info;
    vk::DescriptorBufferInfo buffer_info;
    vk::BufferView buffer_view;

    bool operator!=(const DescriptorData& other) const {
        return std::memcmp(this, &other, sizeof(DescriptorData)) != 0;
    }
};

using DescriptorSetData = std::array<DescriptorData, MAX_DESCRIPTORS>;

class Instance;
class TaskScheduler;

class DescriptorManager {
public:
    DescriptorManager(const Instance& instance, TaskScheduler& scheduler);
    ~DescriptorManager();

    /// Binds a resource to the provided binding
    void SetBinding(u32 set, u32 binding, DescriptorData data);

    /// Builds descriptor sets that reference the currently bound resources
    void BindDescriptorSets();

    /// Marks cached descriptor state dirty
    void MarkDirty();

    /// Returns the rasterizer pipeline layout
    vk::PipelineLayout GetPipelineLayout() const {
        return layout;
    }

private:
    /// Builds the rasterizer pipeline layout objects
    void BuildLayouts();

private:
    const Instance& instance;
    TaskScheduler& scheduler;

    // Cached layouts for the rasterizer pipelines
    vk::PipelineLayout layout;
    std::array<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS> descriptor_set_layouts;
    std::array<vk::DescriptorUpdateTemplate, MAX_DESCRIPTOR_SETS> update_templates;

    // Current data for the descriptor sets
    std::array<DescriptorSetData, MAX_DESCRIPTOR_SETS> update_data{};
    std::array<bool, MAX_DESCRIPTOR_SETS> descriptor_dirty{};
    std::array<vk::DescriptorSet, MAX_DESCRIPTOR_SETS> descriptor_sets;
    std::array<std::vector<vk::DescriptorSet>, MAX_DESCRIPTOR_SETS> descriptor_batch;
};

} // namespace Vulkan
