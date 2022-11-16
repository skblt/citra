// SPDX-FileCopyrightText: Copyright 2018 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>
#include "common/common_types.h"
#include "core/frontend/emu_window.h"
#include "video_core/renderer_vulkan/vk_blit_screen.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_renderpass_cache.h"
#include "video_core/renderer_vulkan/vk_swapchain.h"

namespace Vulkan {

namespace {

constexpr std::string_view vertex_shader_source = R"(
#version 450 core
#extension GL_ARB_separate_shader_objects : enable
layout (location = 0) in vec2 vert_position;
layout (location = 1) in vec2 vert_tex_coord;
layout (location = 0) out vec2 frag_tex_coord;

// This is a truncated 3x3 matrix for 2D transformations:
// The upper-left 2x2 submatrix performs scaling/rotation/mirroring.
// The third column performs translation.
// The third row could be used for projection, which we don't need in 2D. It hence is assumed to
// implicitly be [0, 0, 1]
layout (push_constant, std140) uniform DrawInfo {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
};

void main() {
    vec4 position = vec4(vert_position, 0.0, 1.0) * modelview_matrix;
    gl_Position = vec4(position.x, -position.y, 0.0, 1.0);
    frag_tex_coord = vert_tex_coord;
}
)";

constexpr std::string_view fragment_shader_source = R"(
#version 450 core
#extension GL_ARB_separate_shader_objects : enable
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

layout (push_constant, std140) uniform DrawInfo {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
    int reverse_interlaced;
};

layout (set = 0, binding = 0) uniform texture2D screen_textures[3];
layout (set = 0, binding = 1) uniform sampler screen_sampler;

void main() {
    color = texture(sampler2D(screen_textures[screen_id_l], screen_sampler), frag_tex_coord);
}
)";

constexpr std::string_view fragment_shader_anaglyph_source = R"(
#version 450 core
#extension GL_ARB_separate_shader_objects : enable
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

// Anaglyph Red-Cyan shader based on Dubois algorithm
// Constants taken from the paper:
// "Conversion of a Stereo Pair to Anaglyph with
// the Least-Squares Projection Method"
// Eric Dubois, March 2009
const mat3 l = mat3( 0.437, 0.449, 0.164,
              -0.062,-0.062,-0.024,
              -0.048,-0.050,-0.017);
const mat3 r = mat3(-0.011,-0.032,-0.007,
               0.377, 0.761, 0.009,
              -0.026,-0.093, 1.234);

layout (push_constant, std140) uniform DrawInfo {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
    int reverse_interlaced;
};

layout (set = 0, binding = 0) uniform texture2D screen_textures[3];
layout (set = 0, binding = 1) uniform sampler screen_sampler;

void main() {
    vec4 color_tex_l = texture(sampler2D(screen_textures[screen_id_l], screen_sampler), frag_tex_coord);
    vec4 color_tex_r = texture(sampler2D(screen_textures[screen_id_r], screen_sampler), frag_tex_coord);
    color = vec4(color_tex_l.rgb*l+color_tex_r.rgb*r, color_tex_l.a);
}
)";

constexpr std::string_view fragment_shader_interlaced_source = R"(
#version 450 core
#extension GL_ARB_separate_shader_objects : enable
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

layout (push_constant, std140) uniform DrawInfo {
    mat4 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
    int reverse_interlaced;
};

layout (set = 0, binding = 0) uniform texture2D screen_textures[3];
layout (set = 0, binding = 1) uniform sampler screen_sampler;

void main() {
    float screen_row = o_resolution.x * frag_tex_coord.x;
    if (int(screen_row) % 2 == reverse_interlaced)
        color = texture(sampler2D(screen_textures[screen_id_l], screen_sampler), frag_tex_coord);
    else
        color = texture(sampler2D(screen_textures[screen_id_r], screen_sampler), frag_tex_coord);
}
)";

struct ScreenRectVertex {
    ScreenRectVertex() = default;
    explicit ScreenRectVertex(f32 x, f32 y, f32 u, f32 v) : position{x, y}, tex_coord{u, v} {}

    Common::Vec2f position;
    Common::Vec2f tex_coord;

    static vk::VertexInputBindingDescription GetDescription() {
        return vk::VertexInputBindingDescription{
            .binding = 0,
            .stride = sizeof(ScreenRectVertex),
            .inputRate = vk::VertexInputRate::eVertex,
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> GetAttributes() {
        return {{
            vk::VertexInputAttributeDescription{
                .location = 0,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = static_cast<u32>(offsetof(ScreenRectVertex, position)),
            },
            vk::VertexInputAttributeDescription{
                .location = 1,
                .binding = 0,
                .format = vk::Format::eR32G32Sfloat,
                .offset = static_cast<u32>(offsetof(ScreenRectVertex, tex_coord)),
            },
        }};
    }
};

constexpr std::array<f32, 4 * 4> MakeOrthographicMatrix(f32 width, f32 height) {
    // clang-format off
    return { 2.f / width, 0.f,          0.f, 0.f,
             0.f,         2.f / height, 0.f, 0.f,
             0.f,         0.f,          1.f, 0.f,
            -1.f,        -1.f,          0.f, 1.f};
    // clang-format on
}

} // Anonymous namespace

struct BlitScreen::BufferData {
    std::array<ScreenRectVertex, 4> vertices;
};

constexpr u32 VERTEX_BUFFER_SIZE = sizeof(ScreenRectVertex) * 8192;

BlitScreen::BlitScreen(Frontend::EmuWindow& render_window_, const Instance& instance_,
                       Scheduler& scheduler_, Swapchain& swapchain_, RenderpassCache& renderpass_cache_,
                       DescriptorManager& desc_manager_, std::array<ScreenInfo, 3>& screen_infos_)
    : render_window{render_window_}, instance{instance_}, scheduler{scheduler_}, swapchain{swapchain_},
      renderpass_cache{renderpass_cache_}, desc_manager{desc_manager_}, memory{*VideoCore::g_memory},
      screen_infos{screen_infos_}, image_count{swapchain.GetImageCount()},
      vertex_buffer{instance, scheduler, VERTEX_BUFFER_SIZE, vk::BufferUsageFlagBits::eVertexBuffer, {}} {
    resource_ticks.resize(image_count);

    CreateStaticResources();
    CreateDynamicResources();
}

BlitScreen::~BlitScreen() = default;

void BlitScreen::Recreate() {
    CreateDynamicResources();
}

vk::Semaphore BlitScreen::Draw(const GPU::Regs::FramebufferConfig& framebuffer,
                               const vk::Framebuffer& host_framebuffer,
                               const Layout::FramebufferLayout layout, vk::Extent2D render_area,
                               bool use_accelerated, u32 screen) {
    RefreshResources(framebuffer);

    // Finish any pending renderpass
    renderpass_cache.ExitRenderpass();

    if (const auto swapchain_images = swapchain.GetImageCount(); swapchain_images != image_count) {
        image_count = swapchain_images;
        Recreate();
    }

    const std::size_t image_index = swapchain.GetImageIndex();

    scheduler.Wait(resource_ticks[image_index]);
    resource_ticks[image_index] = scheduler.CurrentTick();

    vk::ImageView source_image_view =
        use_accelerated ? screen_infos[screen].image_view : raw_image_views[image_index][screen];

    BufferData data;
    SetUniformData(data, layout);
    SetVertexData(data, layout);

    auto [ptr, offset, invalidate] = vertex_buffer.Map(sizeof(data));
    std::memcpy(ptr, &data, sizeof(data));
    vertex_buffer.Commit(sizeof(data));

    if (!use_accelerated) {
        bool right_eye = true;
        if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0)
            right_eye = false;

        const PAddr framebuffer_addr =
            framebuffer.active_fb == 0
                ? (!right_eye ? framebuffer.address_left1 : framebuffer.address_right1)
                : (!right_eye ? framebuffer.address_left2 : framebuffer.address_right2);

        LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
                  framebuffer.stride * framebuffer.height, framebuffer_addr, framebuffer.width.Value(),
                  framebuffer.height.Value(), framebuffer.format);

        s32 bpp = GPU::Regs::BytesPerPixel(framebuffer.color_format);
        std::size_t pixel_stride = framebuffer.stride / bpp;

        Memory::RasterizerFlushRegion(framebuffer_addr, framebuffer.stride * framebuffer.height);

        const u8* framebuffer_data = memory.GetPhysicalPointer(framebuffer_addr);

        const vk::BufferImageCopy copy = {
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource =
               {
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .mipLevel = 0,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            .imageOffset = {.x = 0, .y = 0, .z = 0},
            .imageExtent =
                {
                    .width = framebuffer.width,
                    .height = framebuffer.height,
                    .depth = 1,
                },
        };
        scheduler.Record([this, copy, image_index](vk::CommandBuffer render_cmdbuf, vk::CommandBuffer) {
            const vk::Image image = raw_images[image_index];
            const vk::ImageMemoryBarrier base_barrier = {
                .srcAccessMask = vk::AccessFlagBits::eNone,
                .dstAccessMask = vk::AccessFlagBits::eNone,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                .image = image,
                .subresourceRange{
                    .aspectMask = vk::ImageAspectFlagBits::eColor,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            vk::ImageMemoryBarrier read_barrier = base_barrier;
            read_barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
            read_barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            read_barrier.oldLayout = vk::ImageLayout::eUndefined;

            vk::ImageMemoryBarrier write_barrier = base_barrier;
            write_barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            write_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eTransfer,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, read_barrier);
            render_cmdbuf.copyBufferToImage(buffer, image, vk::ImageLayout::eGeneral, copy);
            render_cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
                                          vk::DependencyFlagBits::eByRegion, {}, {}, write_barrier);
        });
    }

    UpdateDescriptorSet(image_index, source_image_view);

    scheduler.Record(
        [this, host_framebuffer, image_index, size = render_area](vk::CommandBuffer cmdbuf) {
            const f32 bg_red = Settings::values.bg_red / 255.0f;
            const f32 bg_green = Settings::values.bg_green / 255.0f;
            const f32 bg_blue = Settings::values.bg_blue / 255.0f;
            const vk::ClearValue clear_color = {
                .color = {.float32 = std::array{bg_red, bg_green, bg_blue, 1.0f}},
            };
            const VkRenderPassBeginInfo renderpass_bi{
                .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                .pNext = nullptr,
                .renderPass = *renderpass,
                .framebuffer = host_framebuffer,
                .renderArea =
                    {
                        .offset = {0, 0},
                        .extent = size,
                    },
                .clearValueCount = 1,
                .pClearValues = &clear_color,
            };
            const VkViewport viewport{
                .x = 0.0f,
                .y = 0.0f,
                .width = static_cast<float>(size.width),
                .height = static_cast<float>(size.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
            const VkRect2D scissor{
                .offset = {0, 0},
                .extent = size,
            };
            cmdbuf.BeginRenderPass(renderpass_bi, VK_SUBPASS_CONTENTS_INLINE);
            auto graphics_pipeline = [this]() {
                switch (Settings::values.scaling_filter.GetValue()) {
                case Settings::ScalingFilter::NearestNeighbor:
                case Settings::ScalingFilter::Bilinear:
                    return *bilinear_pipeline;
                case Settings::ScalingFilter::Bicubic:
                    return *bicubic_pipeline;
                case Settings::ScalingFilter::Gaussian:
                    return *gaussian_pipeline;
                case Settings::ScalingFilter::ScaleForce:
                    return *scaleforce_pipeline;
                default:
                    return *bilinear_pipeline;
                }
            }();
            cmdbuf.BindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);
            cmdbuf.SetViewport(0, viewport);
            cmdbuf.SetScissor(0, scissor);

            cmdbuf.BindVertexBuffer(0, *buffer, offsetof(BufferData, vertices));
            cmdbuf.BindDescriptorSets(VK_PIPELINE_BIND_POINT_GRAPHICS, *pipeline_layout, 0,
                                      descriptor_sets[image_index], {});
            cmdbuf.Draw(4, 1, 0, 0);
            cmdbuf.EndRenderPass();
        });
    return *semaphores[image_index];
}

vk::Semaphore BlitScreen::DrawToSwapchain(const GPU::Regs::FramebufferConfig& framebuffer,
                                          bool use_accelerated) {
    const std::size_t image_index = swapchain.GetImageIndex();
    const vk::Extent2D render_area = swapchain.GetExtent();
    const Layout::FramebufferLayout layout = render_window.GetFramebufferLayout();
    return Draw(framebuffer, framebuffers[image_index], layout, render_area, use_accelerated);
}

vk::Framebuffer BlitScreen::CreateFramebuffer(const vk::ImageView& image_view, vk::Extent2D extent) {
    return CreateFramebuffer(image_view, extent, renderpass);
}

vk::Framebuffer BlitScreen::CreateFramebuffer(const vk::ImageView& image_view, vk::Extent2D extent,
                                              vk::RenderPass& renderpass) {
    return instance.GetDevice().createFramebuffer(vk::FramebufferCreateInfo{
        .renderPass = renderpass,
        .attachmentCount = 1,
        .pAttachments = &image_view,
        .width = extent.width,
        .height = extent.height,
        .layers = 1,
    });
}

void BlitScreen::CreateStaticResources() {
    CreateShaders();
    CreateSampler();
}

void BlitScreen::CreateDynamicResources() {
    CreateSemaphores();
    CreateDescriptorPool();
    CreateDescriptorSetLayout();
    CreateDescriptorSets();
    CreatePipelineLayout();
    CreateRenderPass();
    CreateFramebuffers();
    CreateGraphicsPipeline();
}

void BlitScreen::RefreshResources(const GPU::Regs::FramebufferConfig& framebuffer) {
    if (framebuffer.width == raw_width && framebuffer.height == raw_height &&
        framebuffer.color_format == pixel_format && !raw_images.empty()) {
        return;
    }

    raw_width = framebuffer.width;
    raw_height = framebuffer.height;
    pixel_format = framebuffer.color_format;

    ReleaseRawImages();

    CreateStagingBuffer(framebuffer);
    CreateRawImages(framebuffer);
}

void BlitScreen::CreateShaders() {
    vk::Device device = instance.GetDevice();
    vertex_shader =
        Compile(vertex_shader_source, vk::ShaderStageFlagBits::eVertex, device, ShaderOptimization::High);
    shaders[0] = Compile(fragment_shader_source, vk::ShaderStageFlagBits::eFragment, device,
                                 ShaderOptimization::High);
    shaders[1] = Compile(fragment_shader_anaglyph_source, vk::ShaderStageFlagBits::eFragment,
                                 device, ShaderOptimization::High);
    shaders[2] = Compile(fragment_shader_interlaced_source, vk::ShaderStageFlagBits::eFragment,
                                 device, ShaderOptimization::High);
}

void BlitScreen::CreateSemaphores() {
    semaphores.resize(image_count);
    for (vk::Semaphore& semaphore : semaphores) {
        semaphore = instance.GetDevice().createSemaphore({});
    };
}

void BlitScreen::CreateDescriptorPool() {
    const std::array pool_sizes = {
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eSampledImage,
            .descriptorCount = static_cast<u32>(image_count),
        },
        vk::DescriptorPoolSize{
            .type = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
        },
    };

    const vk::DescriptorPoolCreateInfo pool_create_info = {
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = static_cast<u32>(image_count),
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };
    descriptor_pool = instance.GetDevice().createDescriptorPool(pool_create_info);
}

void BlitScreen::CreateRenderPass() {
    renderpass = CreateRenderPassImpl(swapchain.GetImageFormat());
}

vk::RenderPass BlitScreen::CreateRenderPassImpl(vk::Format format, bool is_present) {
    const vk::AttachmentDescription color_attachment{
        .format = format,
        .samples = vk::SampleCountFlagBits::e1,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
        .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
        .initialLayout = vk::ImageLayout::eUndefined,
        .finalLayout = is_present ? vk::ImageLayout::ePresentSrcKHR : vk::ImageLayout::eGeneral,
    };

    const vk::AttachmentReference color_attachment_ref = {
        .attachment = 0,
        .layout = vk::ImageLayout::eGeneral,
    };

    const vk::SubpassDescription subpass_description = {
        .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref
    };

    const vk::SubpassDependency dependency = {
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite
    };

    const vk::RenderPassCreateInfo renderpass_create_info = {
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass_description,
        .dependencyCount = 1,
        .pDependencies = &dependency,
    };

    return instance.GetDevice().createRenderPass(renderpass_create_info);
}

void BlitScreen::CreateDescriptorSetLayout() {
    const std::array layout_bindings = {
        vk::DescriptorSetLayoutBinding{.binding = 0,
                                       .descriptorType = vk::DescriptorType::eSampledImage,
                                       .descriptorCount = 3,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment},
        vk::DescriptorSetLayoutBinding{.binding = 1,
                                       .descriptorType = vk::DescriptorType::eSampler,
                                       .descriptorCount = 1,
                                       .stageFlags = vk::ShaderStageFlagBits::eFragment}
    };

    const vk::DescriptorSetLayoutCreateInfo present_layout_info = {
        .bindingCount = static_cast<u32>(layout_bindings.size()),
        .pBindings = layout_bindings.data()
    };

    vk::Device device = instance.GetDevice();
    descriptor_set_layout = device.createDescriptorSetLayout(present_layout_info);
}

void BlitScreen::CreateDescriptorSets() {
    const std::vector layouts(image_count, *descriptor_set_layout);

    const VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *descriptor_pool,
        .descriptorSetCount = static_cast<u32>(image_count),
        .pSetLayouts = layouts.data(),
    };

    const VkDescriptorSetAllocateInfo ai_aa{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *aa_descriptor_pool,
        .descriptorSetCount = static_cast<u32>(image_count),
        .pSetLayouts = layouts_aa.data(),
    };

    descriptor_sets = descriptor_pool.Allocate(ai);
    aa_descriptor_sets = aa_descriptor_pool.Allocate(ai_aa);
}

void BlitScreen::CreatePipelineLayout() {
    const VkPipelineLayoutCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = descriptor_set_layout.address(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    const VkPipelineLayoutCreateInfo ci_aa{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = aa_descriptor_set_layout.address(),
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };
    pipeline_layout = device.GetLogical().CreatePipelineLayout(ci);
    aa_pipeline_layout = device.GetLogical().CreatePipelineLayout(ci_aa);
}

void BlitScreen::CreateGraphicsPipeline() {
    const std::array attributes = {
        vk::VertexInputAttributeDescription{.location = 0,
                                            .binding = 0,
                                            .format = vk::Format::eR32G32Sfloat,
                                            .offset = offsetof(ScreenRectVertex, position)},
        vk::VertexInputAttributeDescription{.location = 1,
                                            .binding = 0,
                                            .format = vk::Format::eR32G32Sfloat,
                                            .offset = offsetof(ScreenRectVertex, tex_coord)}};

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data()};

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleStrip, .primitiveRestartEnable = false};

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f};

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = false};

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &colorblend_attachment,
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f}};

    const vk::Viewport placeholder_viewport = vk::Viewport{0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
    const vk::Rect2D placeholder_scissor = vk::Rect2D{{0, 0}, {1, 1}};
    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .viewportCount = 1,
        .pViewports = &placeholder_viewport,
        .scissorCount = 1,
        .pScissors = &placeholder_scissor,
    };

    const std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data()};

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {.depthTestEnable = false,
                                                                .depthWriteEnable = false,
                                                                .depthCompareOp =
                                                                    vk::CompareOp::eAlways,
                                                                .depthBoundsTestEnable = false,
                                                                .stencilTestEnable = false};

    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        const std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex,
                                              .module = present_vertex_shader,
                                              .pName = "main"},
            vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment,
                                              .module = present_shaders[i],
                                              .pName = "main"},
        };

        const vk::GraphicsPipelineCreateInfo pipeline_info = {
            .stageCount = static_cast<u32>(shader_stages.size()),
            .pStages = shader_stages.data(),
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_info,
            .pRasterizationState = &raster_state,
            .pMultisampleState = &multisampling,
            .pDepthStencilState = &depth_info,
            .pColorBlendState = &color_blending,
            .pDynamicState = &dynamic_info,
            .layout = present_pipeline_layout,
            .renderPass = renderpass_cache.GetPresentRenderpass()};

        vk::Device device = instance.GetDevice();
        if (const auto result = device.createGraphicsPipeline({}, pipeline_info);
            result.result == vk::Result::eSuccess) {
            present_pipelines[i] = result.value;
        } else {
            LOG_CRITICAL(Render_Vulkan, "Unable to build present pipelines");
            UNREACHABLE();
        }
    }
}

void BlitScreen::CreateSampler() {
    auto properties = instance.GetPhysicalDevice().getProperties();
    for (std::size_t i = 0; i < samplers.size(); i++) {
        const vk::Filter filter_mode = i == 0 ? vk::Filter::eLinear : vk::Filter::eNearest;
        const vk::SamplerCreateInfo sampler_info = {
            .magFilter = filter_mode,
            .minFilter = filter_mode,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .anisotropyEnable = true,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = false,
            .compareOp = vk::CompareOp::eAlways,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = false};

        samplers[i] = instance.GetDevice().createSampler(sampler_info);
    }
}

void BlitScreen::CreateFramebuffers() {
    const VkExtent2D size{swapchain.GetSize()};
    framebuffers.resize(image_count);

    for (std::size_t i = 0; i < image_count; ++i) {
        const VkImageView image_view{swapchain.GetImageViewIndex(i)};
        framebuffers[i] = CreateFramebuffer(image_view, size, renderpass);
    }
}

void BlitScreen::ReleaseRawImages() {
    for (const u64 tick : resource_ticks) {
        scheduler.Wait(tick);
    }
    raw_images.clear();
    raw_buffer_commits.clear();

    aa_image_view.reset();
    aa_image.reset();
    aa_commit = MemoryCommit{};

    buffer.reset();
    buffer_commit = MemoryCommit{};
}

void BlitScreen::CreateStagingBuffer(const Tegra::FramebufferConfig& framebuffer) {
    const VkBufferCreateInfo ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = CalculateBufferSize(framebuffer),
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    buffer = device.GetLogical().CreateBuffer(ci);
    buffer_commit = memory_allocator.Commit(buffer, MemoryUsage::Upload);
}

void BlitScreen::CreateRawImages(const Tegra::FramebufferConfig& framebuffer) {
    raw_images.resize(image_count);
    raw_image_views.resize(image_count);
    raw_buffer_commits.resize(image_count);

    const auto create_image = [&](bool used_on_framebuffer = false, u32 up_scale = 1,
                                  u32 down_shift = 0) {
        u32 extra_usages = used_on_framebuffer ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                               : VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        return device.GetLogical().CreateImage(VkImageCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = GetFormat(framebuffer),
            .extent =
                {
                    .width = (up_scale * framebuffer.width) >> down_shift,
                    .height = (up_scale * framebuffer.height) >> down_shift,
                    .depth = 1,
                },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = used_on_framebuffer ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR,
            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | extra_usages,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = nullptr,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        });
    };
    const auto create_commit = [&](vk::Image& image) {
        return memory_allocator.Commit(image, MemoryUsage::DeviceLocal);
    };
    const auto create_image_view = [&](vk::Image& image) {
        return device.GetLogical().CreateImageView(VkImageViewCreateInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = *image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = GetFormat(framebuffer),
            .components =
                {
                    .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                    .a = VK_COMPONENT_SWIZZLE_IDENTITY,
                },
            .subresourceRange =
                {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
        });
    };

    for (size_t i = 0; i < image_count; ++i) {
        raw_images[i] = create_image();
        raw_buffer_commits[i] = create_commit(raw_images[i]);
        raw_image_views[i] = create_image_view(raw_images[i]);
    }

    // AA Resources
    const u32 up_scale = Settings::values.resolution_info.up_scale;
    const u32 down_shift = Settings::values.resolution_info.down_shift;
    aa_image = create_image(true, up_scale, down_shift);
    aa_commit = create_commit(aa_image);
    aa_image_view = create_image_view(aa_image);
    VkExtent2D size{
        .width = (up_scale * framebuffer.width) >> down_shift,
        .height = (up_scale * framebuffer.height) >> down_shift,
    };
    if (aa_renderpass) {
        aa_framebuffer = CreateFramebuffer(*aa_image_view, size, aa_renderpass);
        return;
    }
    aa_renderpass = CreateRenderPassImpl(GetFormat(framebuffer), false);
    aa_framebuffer = CreateFramebuffer(*aa_image_view, size, aa_renderpass);

    const std::array<VkPipelineShaderStageCreateInfo, 2> fxaa_shader_stages{{
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_VERTEX_BIT,
            .module = *fxaa_vertex_shader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = *fxaa_fragment_shader,
            .pName = "main",
            .pSpecializationInfo = nullptr,
        },
    }};

    const auto vertex_binding_description = ScreenRectVertex::GetDescription();
    const auto vertex_attrs_description = ScreenRectVertex::GetAttributes();

    const VkPipelineVertexInputStateCreateInfo vertex_input_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_binding_description,
        .vertexAttributeDescriptionCount = u32{vertex_attrs_description.size()},
        .pVertexAttributeDescriptions = vertex_attrs_description.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo input_assembly_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkPipelineViewportStateCreateInfo viewport_state_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .viewportCount = 1,
        .pViewports = nullptr,
        .scissorCount = 1,
        .pScissors = nullptr,
    };

    const VkPipelineRasterizationStateCreateInfo rasterization_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .depthBiasConstantFactor = 0.0f,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = 0.0f,
        .lineWidth = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisampling_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable = VK_FALSE,
        .minSampleShading = 0.0f,
        .pSampleMask = nullptr,
        .alphaToCoverageEnable = VK_FALSE,
        .alphaToOneEnable = VK_FALSE,
    };

    const VkPipelineColorBlendAttachmentState color_blend_attachment{
        .blendEnable = VK_FALSE,
        .srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
        .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
        .alphaBlendOp = VK_BLEND_OP_ADD,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo color_blend_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .logicOpEnable = VK_FALSE,
        .logicOp = VK_LOGIC_OP_COPY,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
        .blendConstants = {0.0f, 0.0f, 0.0f, 0.0f},
    };

    static constexpr std::array dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    const VkPipelineDynamicStateCreateInfo dynamic_state_ci{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    const VkGraphicsPipelineCreateInfo fxaa_pipeline_ci{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<u32>(fxaa_shader_stages.size()),
        .pStages = fxaa_shader_stages.data(),
        .pVertexInputState = &vertex_input_ci,
        .pInputAssemblyState = &input_assembly_ci,
        .pTessellationState = nullptr,
        .pViewportState = &viewport_state_ci,
        .pRasterizationState = &rasterization_ci,
        .pMultisampleState = &multisampling_ci,
        .pDepthStencilState = nullptr,
        .pColorBlendState = &color_blend_ci,
        .pDynamicState = &dynamic_state_ci,
        .layout = *aa_pipeline_layout,
        .renderPass = *aa_renderpass,
        .subpass = 0,
        .basePipelineHandle = 0,
        .basePipelineIndex = 0,
    };

    // AA
    aa_pipeline = device.GetLogical().CreateGraphicsPipeline(fxaa_pipeline_ci);
}

void BlitScreen::UpdateDescriptorSet(std::size_t image_index, vk::ImageView image_view) const {
    std::array<vk::DescriptorImageInfo, 4> image_infos;
    for (std::size_t i = 0; i < screen_infos.size(); i++) {
        const auto& info = screen_infos[i];
        image_infos[i] = vk::DescriptorImageInfo{
            .imageView = info.display_texture ? info.display_texture->image_view
                                              : info.texture.alloc.image_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
    }

    image_infos[3] = vk::DescriptorImageInfo{.sampler = samplers[current_sampler]};

    const vk::DescriptorImageInfo image_info{
        .sampler = samplers[current_sampler],
        .imageView = image_view,
        .imageLayout = vk::ImageLayout::eGeneral,
    };

    device.GetLogical().UpdateDescriptorSets(std::array{ubo_write, sampler_write}, {});
}

void BlitScreen::SetUniformData(BufferData& data, const Layout::FramebufferLayout layout) const {
    data.uniform.modelview_matrix =
        MakeOrthographicMatrix(static_cast<f32>(layout.width), static_cast<f32>(layout.height));
}

void BlitScreen::SetVertexData(BufferData& data, const Layout::FramebufferLayout layout) const {
    /*const auto screen_vertices = [](f32 x, f32 y, f32 w, f32 h) {
        data.vertices[0] = ScreenRectVertex{x, y,}
    };


    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};*/
}

} // namespace Vulkan
