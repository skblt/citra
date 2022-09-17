// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#define VULKAN_HPP_NO_CONSTRUCTORS
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/core.h"
#include "core/frontend/emu_window.h"
#include "core/frontend/framebuffer_layout.h"
#include "core/hw/gpu.h"
#include "core/hw/hw.h"
#include "core/hw/lcd.h"
#include "core/settings.h"
#include "video_core/renderer_vulkan/renderer_vulkan.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "video_core/renderer_vulkan/vk_shader.h"
#include "video_core/renderer_vulkan/vk_task_scheduler.h"
#include "video_core/video_core.h"

namespace Vulkan {

constexpr std::string_view vertex_shader = R"(
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
layout (push_constant) uniform DrawInfo {
    mat3x2 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
};

void main() {
    // Multiply input position by the rotscale part of the matrix and then manually translate by
    // the last column. This is equivalent to using a full 3x3 matrix and expanding the vector
    // to `vec3(vert_position.xy, 1.0)`
    gl_Position = vec4(mat2(modelview_matrix) * vert_position + modelview_matrix[2], 0.0, 1.0);
    gl_Position.y = -gl_Position.y;
    frag_tex_coord = vert_tex_coord;
}
)";

constexpr std::string_view fragment_shader = R"(
version 450 core
#extension GL_ARB_separate_shader_objects : enable
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

layout (push_constant) uniform DrawInfo {
    mat3x2 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
};

layout (set = 0, binding = 0) uniform sampler2D screen_textures[3];

void main() {
    color = texture(screen_textures[screen_id_l], frag_tex_coord);
}
)";

constexpr std::string_view fragment_shader_anaglyph = R"(
version 450 core
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

layout (push_constant) uniform DrawInfo {
    mat3x2 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
};

layout (set = 0, binding = 0) uniform sampler2D screen_textures[3];

void main() {
    vec4 color_tex_l = texture(screen_textures[screen_id_l], frag_tex_coord);
    vec4 color_tex_r = texture(screen_textures[screen_id_r], frag_tex_coord);
    color = vec4(color_tex_l.rgb*l+color_tex_r.rgb*r, color_tex_l.a);
}
)";

constexpr std::string_view fragment_shader_interlaced = R"(
version 450 core
#extension GL_ARB_separate_shader_objects : enable
layout (location = 0) in vec2 frag_tex_coord;
layout (location = 0) out vec4 color;

layout (push_constant) uniform DrawInfo {
    mat3x2 modelview_matrix;
    vec4 i_resolution;
    vec4 o_resolution;
    int screen_id_l;
    int screen_id_r;
    int layer;
    int reverse_interlaced;
};

layout (set = 0, binding = 0) uniform sampler2D screen_textures[3];

void main() {
    float screen_row = o_resolution.x * frag_tex_coord.x;
    if (int(screen_row) % 2 == reverse_interlaced)
        color = texture(screen_textures[screen_id_l], frag_tex_coord);
    else
        color = texture(screen_textures[screen_id_r], frag_tex_coord);
}
)";


/// Vertex structure that the drawn screen rectangles are composed of.
struct ScreenRectVertex {
    ScreenRectVertex() = default;
    ScreenRectVertex(float x, float y, float u, float v) :
        position{Common::MakeVec(x, y)}, tex_coord{Common::MakeVec(u, v)} {}

    Common::Vec2f position;
    Common::Vec2f tex_coord;
};

constexpr u32 VERTEX_BUFFER_SIZE = sizeof(ScreenRectVertex) * 64;

/**
 * Defines a 1:1 pixel ortographic projection matrix with (0,0) on the top-left
 * corner and (width, height) on the lower-bottom.
 *
 * The projection part of the matrix is trivial, hence these operations are represented
 * by a 3x2 matrix.
 *
 * @param flipped Whether the frame should be flipped upside down.
 */
static std::array<float, 3 * 2> MakeOrthographicMatrix(float width, float height, bool flipped) {

    std::array<float, 3 * 2> matrix; // Laid out in column-major order

    // Last matrix row is implicitly assumed to be [0, 0, 1].
    if (flipped) {
        // clang-format off
        matrix[0] = 2.f / width; matrix[2] = 0.f;           matrix[4] = -1.f;
        matrix[1] = 0.f;         matrix[3] = 2.f / height;  matrix[5] = -1.f;
        // clang-format on
    } else {
        // clang-format off
        matrix[0] = 2.f / width; matrix[2] = 0.f;           matrix[4] = -1.f;
        matrix[1] = 0.f;         matrix[3] = -2.f / height; matrix[5] = 1.f;
        // clang-format on
    }

    return matrix;
}

RendererVulkan::RendererVulkan(Frontend::EmuWindow& window)
    : RendererBase{window}, instance{window}, scheduler{instance}, renderpass_cache{instance, scheduler},
      runtime{instance, scheduler, renderpass_cache}, swapchain{instance, renderpass_cache},
      vertex_buffer{instance, scheduler, VERTEX_BUFFER_SIZE, vk::BufferUsageFlagBits::eVertexBuffer, {}} {

    auto& telemetry_session = Core::System::GetInstance().TelemetrySession();
    constexpr auto user_system = Common::Telemetry::FieldType::UserSystem;
    telemetry_session.AddField(user_system, "GPU_Vendor", "NVIDIA");
    telemetry_session.AddField(user_system, "GPU_Model", "GTX 1650");
    telemetry_session.AddField(user_system, "GPU_Vulkan_Version", "Vulkan 1.1");

    window.mailbox = nullptr;
}

RendererVulkan::~RendererVulkan() {
    vk::Device device = instance.GetDevice();

    device.destroyPipelineLayout(present_pipeline_layout);
    device.destroyDescriptorSetLayout(present_descriptor_layout);
    device.destroyDescriptorUpdateTemplate(present_update_template);
    device.destroyShaderModule(present_vertex_shader);
    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        device.destroyPipeline(present_pipelines[i]);
        device.destroyShaderModule(present_shaders[i]);
    }
    for (std::size_t i = 0; i < present_samplers.size(); i++) {
        device.destroySampler(present_samplers[i]);
    }
}

VideoCore::ResultStatus RendererVulkan::Init() {
    CompileShaders();
    BuildLayouts();
    BuildPipelines();

    // Create the rasterizer
    rasterizer = std::make_unique<RasterizerVulkan>(render_window, instance, scheduler,
                                                    runtime, renderpass_cache);

    return VideoCore::ResultStatus::Success;
}

VideoCore::RasterizerInterface* RendererVulkan::Rasterizer() {
    return rasterizer.get();
}

void RendererVulkan::ShutDown() {}

void RendererVulkan::Sync() {
    rasterizer->SyncEntireState();
}

void RendererVulkan::PrepareRendertarget() {
    for (int i = 0; i < 3; i++) {
        int fb_id = i == 2 ? 1 : 0;
        const auto& framebuffer = GPU::g_regs.framebuffer_config[fb_id];

        // Main LCD (0): 0x1ED02204, Sub LCD (1): 0x1ED02A04
        u32 lcd_color_addr =
            (fb_id == 0) ? LCD_REG_INDEX(color_fill_top) : LCD_REG_INDEX(color_fill_bottom);
        lcd_color_addr = HW::VADDR_LCD + 4 * lcd_color_addr;
        LCD::Regs::ColorFill color_fill = {0};
        LCD::Read(color_fill.raw, lcd_color_addr);

        if (color_fill.is_enabled) {
            LoadColorToActiveGLTexture(color_fill.color_r, color_fill.color_g, color_fill.color_b,
                                       screen_infos[i].texture);
        } else {
            if (screen_infos[i].texture.width != framebuffer.width ||
                screen_infos[i].texture.height != framebuffer.height ||
                screen_infos[i].texture.format != framebuffer.color_format) {
                // Reallocate texture if the framebuffer size has changed.
                // This is expected to not happen very often and hence should not be a
                // performance problem.
                ConfigureFramebufferTexture(screen_infos[i].texture, framebuffer);
            }

            LoadFBToScreenInfo(framebuffer, screen_infos[i], i == 1);

            // Resize the texture in case the framebuffer size has changed
            screen_infos[i].texture.width = framebuffer.width;
            screen_infos[i].texture.height = framebuffer.height;
        }
    }
}

void RendererVulkan::BeginRendering() {
    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, present_pipelines[current_pipeline]);

    for (std::size_t i = 0; i < screen_infos.size(); i++) {
        runtime.Transition(command_buffer, screen_infos[i].display_texture,
                           vk::ImageLayout::eShaderReadOnlyOptimal, 0, 1);
    }

    const std::array present_textures = {
        vk::DescriptorImageInfo{
            .sampler = present_samplers[current_sampler],
            .imageView = screen_infos[0].display_texture.image_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        },
        vk::DescriptorImageInfo{
            .sampler = present_samplers[current_sampler],
            .imageView = screen_infos[1].display_texture.image_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        },
        vk::DescriptorImageInfo{
            .sampler = present_samplers[current_sampler],
            .imageView = screen_infos[2].display_texture.image_view,
            .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal
        },
    };

    const vk::DescriptorSetAllocateInfo alloc_info = {
        .descriptorPool = scheduler.GetDescriptorPool(),
        .descriptorSetCount = 1,
        .pSetLayouts = &present_descriptor_layout
    };

    vk::Device device = instance.GetDevice();
    vk::DescriptorSet set = device.allocateDescriptorSets(alloc_info)[0];
    device.updateDescriptorSetWithTemplate(set, present_update_template, present_textures.data());

    command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, present_pipeline_layout,
                                      0, 1, &set, 0, nullptr);
}

void RendererVulkan::LoadFBToScreenInfo(const GPU::Regs::FramebufferConfig& framebuffer,
                                        ScreenInfo& screen_info, bool right_eye) {

    if (framebuffer.address_right1 == 0 || framebuffer.address_right2 == 0)
        right_eye = false;

    const PAddr framebuffer_addr =
        framebuffer.active_fb == 0
            ? (!right_eye ? framebuffer.address_left1 : framebuffer.address_right1)
            : (!right_eye ? framebuffer.address_left2 : framebuffer.address_right2);

    LOG_TRACE(Render_Vulkan, "0x{:08x} bytes from 0x{:08x}({}x{}), fmt {:x}",
              framebuffer.stride * framebuffer.height, framebuffer_addr, framebuffer.width.Value(),
              framebuffer.height.Value(), framebuffer.format);

    int bpp = GPU::Regs::BytesPerPixel(framebuffer.color_format);
    std::size_t pixel_stride = framebuffer.stride / bpp;

    // OpenGL only supports specifying a stride in units of pixels, not bytes, unfortunately
    ASSERT(pixel_stride * bpp == framebuffer.stride);

    // Ensure no bad interactions with GL_UNPACK_ALIGNMENT, which by default
    // only allows rows to have a memory alignement of 4.
    ASSERT(pixel_stride % 4 == 0);

    if (!rasterizer->AccelerateDisplay(framebuffer, framebuffer_addr, static_cast<u32>(pixel_stride), screen_info)) {
        ASSERT(false);
        // Reset the screen info's display texture to its own permanent texture
        /*screen_info.display_texture = &screen_info.texture;
        screen_info.display_texcoords = Common::Rectangle<float>(0.f, 0.f, 1.f, 1.f);

        Memory::RasterizerFlushRegion(framebuffer_addr, framebuffer.stride * framebuffer.height);

        vk::Rect2D region{{0, 0}, {framebuffer.width, framebuffer.height}};
        std::span<u8> framebuffer_data(VideoCore::g_memory->GetPhysicalPointer(framebuffer_addr),
                                       screen_info.texture.GetSize());

        screen_info.texture.Upload(0, 1, pixel_stride, region, framebuffer_data);*/
    }
}

void RendererVulkan::LoadColorToActiveGLTexture(u8 color_r, u8 color_g, u8 color_b, const TextureInfo& texture) {
    const auto color = std::array<float, 4>{color_r / 255.0f, color_g / 255.0f, color_b / 255.0f, 1};
    const vk::ClearColorValue clear_color = {
        .float32 = color
    };

    const vk::ImageSubresourceRange range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1,
    };

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.clearColorImage(texture.alloc.image, vk::ImageLayout::eShaderReadOnlyOptimal,
                                   clear_color, range);
}

void RendererVulkan::CompileShaders() {
    vk::Device device = instance.GetDevice();
    present_vertex_shader = Compile(vertex_shader, vk::ShaderStageFlagBits::eVertex,
                                    device, ShaderOptimization::Debug);
    present_shaders[0] = Compile(fragment_shader, vk::ShaderStageFlagBits::eFragment,
                                    device, ShaderOptimization::Debug);
    present_shaders[1] = Compile(fragment_shader_anaglyph, vk::ShaderStageFlagBits::eFragment,
                                    device, ShaderOptimization::Debug);
    present_shaders[2] = Compile(fragment_shader_interlaced, vk::ShaderStageFlagBits::eFragment,
                                    device, ShaderOptimization::Debug);

    auto properties = instance.GetPhysicalDevice().getProperties();
    for (std::size_t i = 0; i < present_samplers.size(); i++) {
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
            .unnormalizedCoordinates = false
        };

        present_samplers[i] = device.createSampler(sampler_info);
    }
}

void RendererVulkan::BuildLayouts() {
    const vk::DescriptorSetLayoutBinding present_layout_binding = {
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 3,
        .stageFlags = vk::ShaderStageFlagBits::eFragment
    };

    const vk::DescriptorSetLayoutCreateInfo present_layout_info = {
        .bindingCount = 1,
        .pBindings = &present_layout_binding
    };

    const vk::DescriptorUpdateTemplateEntry update_template_entry = {
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 3,
        .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
        .offset = 0,
        .stride = sizeof(vk::DescriptorImageInfo)
    };

    const vk::DescriptorUpdateTemplateCreateInfo template_info = {
        .descriptorUpdateEntryCount = 1,
        .pDescriptorUpdateEntries = &update_template_entry,
        .descriptorSetLayout = present_descriptor_layout
    };

    vk::Device device = instance.GetDevice();
    present_descriptor_layout = device.createDescriptorSetLayout(present_layout_info);
    present_update_template = device.createDescriptorUpdateTemplate(template_info);

    const vk::PushConstantRange push_range = {
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PresentUniformData),
    };

    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1,
        .pSetLayouts = &present_descriptor_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range
    };

    present_pipeline_layout = device.createPipelineLayout(layout_info);
}

void RendererVulkan::BuildPipelines() {
    const vk::VertexInputBindingDescription binding = {
        .binding = 0,
        .stride = sizeof(ScreenRectVertex),
        .inputRate = vk::VertexInputRate::eVertex
    };

    const std::array attributes = {
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, position)
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ScreenRectVertex, tex_coord)
        }
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<u32>(attributes.size()),
        .pVertexAttributeDescriptions = attributes.data()
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = vk::PrimitiveTopology::eTriangleStrip,
        .primitiveRestartEnable = false
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state = {
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .cullMode = vk::CullModeFlagBits::eNone,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling = {
        .rasterizationSamples  = vk::SampleCountFlagBits::e1,
        .sampleShadingEnable = false
    };

    const vk::PipelineColorBlendAttachmentState colorblend_attachment = {
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable = false,
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

    const std::array dynamic_states = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data()
    };

    const vk::PipelineDepthStencilStateCreateInfo depth_info = {
        .depthTestEnable = false,
        .depthWriteEnable = false,
        .depthCompareOp = vk::CompareOp::eAlways,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false
    };

    for (u32 i = 0; i < PRESENT_PIPELINES; i++) {
        const std::array shader_stages = {
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eVertex,
                .module = present_vertex_shader,
                .pName = "main"
            },
            vk::PipelineShaderStageCreateInfo{
                .stage = vk::ShaderStageFlagBits::eFragment,
                .module = present_shaders[i],
                .pName = "main"
            },
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
            .renderPass = renderpass_cache.GetPresentRenderpass()
        };

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

void RendererVulkan::ReloadSampler() {
    current_sampler = !Settings::values.filter_mode;
}

void RendererVulkan::ReloadPipeline() {
    switch (Settings::values.render_3d) {
    case Settings::StereoRenderOption::Anaglyph:
        current_pipeline = 1;
        break;
    case Settings::StereoRenderOption::Interlaced:
    case Settings::StereoRenderOption::ReverseInterlaced:
        current_pipeline = 2;
        draw_info.reverse_interlaced =
                Settings::values.render_3d == Settings::StereoRenderOption::ReverseInterlaced;
        break;
    default:
        current_pipeline = 0;
        break;
    }
}

void RendererVulkan::ConfigureFramebufferTexture(TextureInfo& texture,
                                                 const GPU::Regs::FramebufferConfig& framebuffer) {
    texture.format = framebuffer.color_format;
    texture.width = framebuffer.width;
    texture.height = framebuffer.height;
    texture.alloc = runtime.Allocate(framebuffer.width, framebuffer.height,
                                     VideoCore::PixelFormatFromGPUPixelFormat(framebuffer.color_format),
                                     VideoCore::TextureType::Texture2D);
}

void RendererVulkan::DrawSingleScreenRotated(u32 screen_id, float x, float y, float w, float h) {
    auto& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.display_texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size);

    const std::array vertices = {
        ScreenRectVertex{x, y, texcoords.bottom, texcoords.left},
        ScreenRectVertex{x + w, y, texcoords.bottom, texcoords.right},
        ScreenRectVertex{x, y + h, texcoords.top, texcoords.left},
        ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.right},
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    // As this is the "DrawSingleScreenRotated" function, the output resolution dimensions have been
    // swapped. If a non-rotated draw-screen function were to be added for book-mode games, those
    // should probably be set to the standard (w, h, 1.0 / w, 1.0 / h) ordering.
    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info.texture.width);
    const float height = static_cast<float>(screen_info.texture.height);

    draw_info.i_resolution = Common::Vec4f{width * scale_factor, height * scale_factor,
                                           1.0f / (width * scale_factor),
                                           1.0f / (height * scale_factor)};
    draw_info.o_resolution = Common::Vec4f{h, w, 1.0f / h, 1.0f / w};
    draw_info.screen_id_l = screen_id;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.pushConstants(present_pipeline_layout,
                                 vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                 0, sizeof(draw_info), &draw_info);

    command_buffer.bindVertexBuffers(0, vertex_buffer.GetHandle(), {0});
    command_buffer.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
}

void RendererVulkan::DrawSingleScreen(u32 screen_id, float x, float y, float w, float h) {
    auto& screen_info = screen_infos[screen_id];
    const auto& texcoords = screen_info.display_texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size);

    const std::array vertices = {
        ScreenRectVertex{x, y, texcoords.bottom, texcoords.right},
        ScreenRectVertex{x + w, y, texcoords.top, texcoords.right},
        ScreenRectVertex{x, y + h, texcoords.bottom, texcoords.left},
        ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.left},
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info.texture.width);
    const float height = static_cast<float>(screen_info.texture.height);

    draw_info.i_resolution = Common::Vec4f{width * scale_factor, height * scale_factor,
                                       1.0f / (width * scale_factor),
                                       1.0f / (height * scale_factor)};
    draw_info.o_resolution = Common::Vec4f{h, w, 1.0f / h, 1.0f / w};
    draw_info.screen_id_l = screen_id;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.pushConstants(present_pipeline_layout,
                                 vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                 0, sizeof(draw_info), &draw_info);

    command_buffer.bindVertexBuffers(0, vertex_buffer.GetHandle(), {0});
    command_buffer.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
}

void RendererVulkan::DrawSingleScreenStereoRotated(u32 screen_id_l, u32 screen_id_r,
                                                   float x, float y, float w, float h) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.display_texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size);

    const std::array vertices = {
        ScreenRectVertex{x, y, texcoords.bottom, texcoords.left},
        ScreenRectVertex{x + w, y, texcoords.bottom, texcoords.right},
        ScreenRectVertex{x, y + h, texcoords.top, texcoords.left},
        ScreenRectVertex{x + w, y + h, texcoords.top, texcoords.right}
    };

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info_l.texture.width);
    const float height = static_cast<float>(screen_info_l.texture.height);

    draw_info.i_resolution = Common::Vec4f{width * scale_factor, height * scale_factor,
                                           1.0f / (width * scale_factor),
                                           1.0f / (height * scale_factor)};

    draw_info.o_resolution = Common::Vec4f{h, w, 1.0f / h, 1.0f / w};
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.pushConstants(present_pipeline_layout,
                                 vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                 0, sizeof(draw_info), &draw_info);

    command_buffer.bindVertexBuffers(0, vertex_buffer.GetHandle(), {0});
    command_buffer.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
}

void RendererVulkan::DrawSingleScreenStereo(u32 screen_id_l, u32 screen_id_r,
                                            float x, float y, float w, float h) {
    const ScreenInfo& screen_info_l = screen_infos[screen_id_l];
    const auto& texcoords = screen_info_l.display_texcoords;

    u32 size = sizeof(ScreenRectVertex) * 4;
    auto [ptr, offset, invalidate] = vertex_buffer.Map(size);

    const std::array<ScreenRectVertex, 4> vertices = {{
        ScreenRectVertex(x, y, texcoords.bottom, texcoords.right),
        ScreenRectVertex(x + w, y, texcoords.top, texcoords.right),
        ScreenRectVertex(x, y + h, texcoords.bottom, texcoords.left),
        ScreenRectVertex(x + w, y + h, texcoords.top, texcoords.left),
    }};

    std::memcpy(ptr, vertices.data(), size);
    vertex_buffer.Commit(size);

    const u16 scale_factor = VideoCore::GetResolutionScaleFactor();
    const float width = static_cast<float>(screen_info_l.texture.width);
    const float height = static_cast<float>(screen_info_l.texture.height);

    draw_info.i_resolution = Common::Vec4f{width * scale_factor, height * scale_factor,
                                           1.0f / (width * scale_factor),
                                           1.0f / (height * scale_factor)};

    draw_info.o_resolution = Common::Vec4f{w, h, 1.0f / w, 1.0f / h};
    draw_info.screen_id_l = screen_id_l;
    draw_info.screen_id_r = screen_id_r;

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.pushConstants(present_pipeline_layout,
                                 vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eVertex,
                                 0, sizeof(draw_info), &draw_info);

    command_buffer.bindVertexBuffers(0, vertex_buffer.GetHandle(), {0});
    command_buffer.draw(4, 1, offset / sizeof(ScreenRectVertex), 0);
}

void RendererVulkan::DrawScreens(const Layout::FramebufferLayout& layout, bool flipped) {
    if (VideoCore::g_renderer_bg_color_update_requested.exchange(false)) {
        // Update background color before drawing
        clear_color.float32[0] = Settings::values.bg_red;
        clear_color.float32[1] = Settings::values.bg_green;
        clear_color.float32[2] = Settings::values.bg_blue;
    }

    if (VideoCore::g_renderer_sampler_update_requested.exchange(false)) {
        // Set the new filtering mode for the sampler
        ReloadSampler();
    }

    if (VideoCore::g_renderer_shader_update_requested.exchange(false)) {
        ReloadPipeline();
    }

    const auto& top_screen = layout.top_screen;
    const auto& bottom_screen = layout.bottom_screen;

    // Set projection matrix
    draw_info.modelview =
        MakeOrthographicMatrix(static_cast<float>(layout.width), static_cast<float>(layout.height), flipped);

    const bool stereo_single_screen =
        Settings::values.render_3d == Settings::StereoRenderOption::Anaglyph ||
        Settings::values.render_3d == Settings::StereoRenderOption::Interlaced ||
        Settings::values.render_3d == Settings::StereoRenderOption::ReverseInterlaced;

    // Bind necessary state before drawing the screens
    BeginRendering();

    draw_info.layer = 0;
    if (layout.top_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(0, top_screen.left,
                                        top_screen.top, top_screen.GetWidth(),
                                        top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(0, (float)top_screen.left / 2,
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(1,
                                        ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                        (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                        (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(0, layout.top_screen.left,
                                        layout.top_screen.top, layout.top_screen.GetWidth(),
                                        layout.top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(1,
                                        layout.cardboard.top_screen_right_eye +
                                            ((float)layout.width / 2),
                                        layout.top_screen.top, layout.top_screen.GetWidth(),
                                        layout.top_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereoRotated(0, 1, (float)top_screen.left, (float)top_screen.top,
                    (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            }
        } else {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreen(0, (float)top_screen.left, (float)top_screen.top,
                                 (float)top_screen.GetWidth(), (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(0, (float)top_screen.left / 2, (float)top_screen.top,
                                 (float)top_screen.GetWidth() / 2, (float)top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(1,
                                 ((float)top_screen.left / 2) + ((float)layout.width / 2),
                                 (float)top_screen.top, (float)top_screen.GetWidth() / 2,
                                 (float)top_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(0, layout.top_screen.left, layout.top_screen.top,
                                 layout.top_screen.GetWidth(), layout.top_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(1,
                                 layout.cardboard.top_screen_right_eye + ((float)layout.width / 2),
                                 layout.top_screen.top, layout.top_screen.GetWidth(),
                                 layout.top_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereo(0, 1, (float)top_screen.left,
                                       (float)top_screen.top, (float)top_screen.GetWidth(),
                                       (float)top_screen.GetHeight());
            }
        }
    }

    draw_info.layer = 0;
    if (layout.bottom_screen_enabled) {
        if (layout.is_rotated) {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreenRotated(2, (float)bottom_screen.left,
                                        (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                        (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreenRotated(
                    2, (float)bottom_screen.left / 2, (float)bottom_screen.top,
                    (float)bottom_screen.GetWidth() / 2, (float)bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(
                    2, ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                    (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                    (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreenRotated(2, layout.bottom_screen.left,
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreenRotated(2,
                                        layout.cardboard.bottom_screen_right_eye +
                                            ((float)layout.width / 2),
                                        layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                        layout.bottom_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereoRotated(2, 2, (float)bottom_screen.left, (float)bottom_screen.top,
                                              (float)bottom_screen.GetWidth(),
                                              (float)bottom_screen.GetHeight());
            }
        } else {
            if (Settings::values.render_3d == Settings::StereoRenderOption::Off) {
                DrawSingleScreen(2, (float)bottom_screen.left,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::SideBySide) {
                DrawSingleScreen(2, (float)bottom_screen.left / 2,
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(2,
                                 ((float)bottom_screen.left / 2) + ((float)layout.width / 2),
                                 (float)bottom_screen.top, (float)bottom_screen.GetWidth() / 2,
                                 (float)bottom_screen.GetHeight());
            } else if (Settings::values.render_3d == Settings::StereoRenderOption::CardboardVR) {
                DrawSingleScreen(2, layout.bottom_screen.left,
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
                draw_info.layer = 1;
                DrawSingleScreen(2,
                                 layout.cardboard.bottom_screen_right_eye +
                                     ((float)layout.width / 2),
                                 layout.bottom_screen.top, layout.bottom_screen.GetWidth(),
                                 layout.bottom_screen.GetHeight());
            } else if (stereo_single_screen) {
                DrawSingleScreenStereo(2, 2, (float)bottom_screen.left,
                                       (float)bottom_screen.top, (float)bottom_screen.GetWidth(),
                                       (float)bottom_screen.GetHeight());
            }
        }
    }
}

void RendererVulkan::SwapBuffers() {
    const auto& layout = render_window.GetFramebufferLayout();
    PrepareRendertarget();

    // Create swapchain if needed
    if (swapchain.NeedsRecreation()) {
        swapchain.Create(layout.width, layout.height, false);
    }

    const vk::Viewport viewport = {
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(layout.width),
        .height = static_cast<float>(layout.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f
    };

    const vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = {layout.width, layout.height}
    };

    const vk::ClearValue clear_value = {
        .color = clear_color
    };

    const vk::RenderPassBeginInfo begin_info = {
        .renderPass = renderpass_cache.GetPresentRenderpass(),
        .framebuffer = swapchain.GetFramebuffer(),
        .clearValueCount = 1,
        .pClearValues = &clear_value,
    };

    vk::CommandBuffer command_buffer = scheduler.GetRenderCommandBuffer();
    command_buffer.setViewport(0, viewport);
    command_buffer.setScissor(0, scissor);
    command_buffer.beginRenderPass(begin_info, vk::SubpassContents::eInline);

    DrawScreens(layout, false);

    // Flush all buffers to make the data visible to the GPU before submitting
    vertex_buffer.Flush();
    rasterizer->FlushBuffers();

    command_buffer.endRenderPass();
    scheduler.Submit(false, true, swapchain.GetAvailableSemaphore(), swapchain.GetPresentSemaphore());

    // Inform texture runtime about the switch
    runtime.OnSlotSwitch(scheduler.GetCurrentSlotIndex());
}

} // namespace Vulkan
