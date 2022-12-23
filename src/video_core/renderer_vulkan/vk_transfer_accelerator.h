// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <unordered_map>
#include "video_core/rasterizer_cache/pixel_format.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {
struct BufferTextureCopy;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Surface;
class DescriptorManager;
struct StagingData;

class TransferAccelerator {
public:
    TransferAccelerator(const Instance& instance, Scheduler& scheduler,
                        DescriptorManager& desc_manager);
    ~TransferAccelerator();

    void ImageFromBuffer();

private:
    enum class Direction : u8 {
        BufferToImage,
        ImageToBuffer,
    };

    void BufferColorConvert(Surface& surface, Direction direction,
                            const VideoCore::BufferTextureCopy& copy);

private:
    const Instance& instance;
    Scheduler& scheduler;
    DescriptorManager& desc_manager;
    std::array<vk::Pipeline, VideoCore::PIXEL_FORMAT_COUNT> buffer_to_image_pipelines;
    std::array<vk::Pipeline, VideoCore::PIXEL_FORMAT_COUNT> image_to_buffer_pipelines;
    vk::DescriptorSetLayout buffer_to_image_set_layout;
    vk::DescriptorUpdateTemplate buffer_to_image_update_template;
    vk::PipelineLayout buffer_to_image_pipeline_layout;
};

} // namespace Vulkan
