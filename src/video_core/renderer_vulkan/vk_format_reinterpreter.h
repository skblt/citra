// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include "video_core/rasterizer_cache/utils.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Surface;
class Instance;
class TaskScheduler;
class TextureRuntime;

class FormatReinterpreterBase {
public:
    FormatReinterpreterBase(const Instance& instance, TaskScheduler& scheduler,
                            TextureRuntime& runtime)
        : instance{instance}, scheduler{scheduler}, runtime{runtime} {}
    virtual ~FormatReinterpreterBase() = default;

    virtual VideoCore::PixelFormat GetSourceFormat() const = 0;
    virtual void Reinterpret(Surface& source, VideoCore::Rect2D src_rect, Surface& dest,
                             VideoCore::Rect2D dst_rect) = 0;

protected:
    const Instance& instance;
    TaskScheduler& scheduler;
    TextureRuntime& runtime;
};

using ReinterpreterList = std::vector<std::unique_ptr<FormatReinterpreterBase>>;

class D24S8toRGBA8 final : public FormatReinterpreterBase {
public:
    D24S8toRGBA8(const Instance& instance, TaskScheduler& scheduler, TextureRuntime& runtime);
    ~D24S8toRGBA8();

    [[nodiscard]] VideoCore::PixelFormat GetSourceFormat() const override {
        return VideoCore::PixelFormat::D24S8;
    }

    void Reinterpret(Surface& source, VideoCore::Rect2D src_rect, Surface& dest,
                     VideoCore::Rect2D dst_rect) override;

private:
    vk::Device device;
    vk::Pipeline compute_pipeline;
    vk::PipelineLayout compute_pipeline_layout;
    vk::DescriptorSetLayout descriptor_layout;
    vk::DescriptorSet descriptor_set;
    vk::DescriptorUpdateTemplate update_template;
    vk::ShaderModule compute_shader;
    VideoCore::Rect2D temp_rect{0, 0, 0, 0};
};

} // namespace Vulkan
