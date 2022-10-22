// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/renderer_vulkan/vk_resource_pool.h"

namespace Vulkan {

constexpr u32 MAX_DESCRIPTORS = 8;
constexpr u32 MAX_DESCRIPTOR_SETS = 4;

union DescriptorData {
    vk::DescriptorImageInfo image_info;
    vk::DescriptorBufferInfo buffer_info;
    vk::BufferView buffer_view;

    [[nodiscard]] bool operator!=(const DescriptorData& other) const noexcept {
        return std::memcmp(this, &other, sizeof(DescriptorData)) != 0;
    }
};

using DescriptorSetData = std::array<DescriptorData, MAX_DESCRIPTORS>;

class Instance;
class Scheduler;

class DescriptorManager {
public:
    DescriptorManager(const Instance& instance, Scheduler& scheduler);
    ~DescriptorManager();

    /// Allocates a descriptor set of the provided layout
    vk::DescriptorSet AllocateSet(vk::DescriptorSetLayout layout);

    /// Binds a resource to the provided binding
    void SetBinding(u32 set, u32 binding, DescriptorData data);

    /// Builds descriptor sets that reference the currently bound resources
    void BindDescriptorSets();

    /// Returns the rasterizer pipeline layout
    [[nodiscard]] vk::PipelineLayout GetPipelineLayout() const noexcept {
        return pipeline_layout;
    }

private:
    /// Builds the rasterizer pipeline layout objects
    void BuildLayouts();

private:
    const Instance& instance;
    Scheduler& scheduler;
    DescriptorPool pool_provider;
    vk::PipelineLayout pipeline_layout;
    vk::DescriptorPool current_pool;
    std::array<vk::DescriptorSetLayout, MAX_DESCRIPTOR_SETS> descriptor_set_layouts;
    std::array<vk::DescriptorUpdateTemplate, MAX_DESCRIPTOR_SETS> update_templates;
    std::array<DescriptorSetData, MAX_DESCRIPTOR_SETS> update_data{};
    std::array<vk::DescriptorSet, MAX_DESCRIPTOR_SETS> descriptor_sets{};
    std::array<bool, MAX_DESCRIPTOR_SETS> descriptor_set_dirty{};
};

} // namespace Vulkan
