// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <set>
#include <span>
#include "common/assert.h"
#include "common/settings.h"
#include "core/frontend/emu_window.h"
#include "video_core/rasterizer_cache/custom_tex_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#include <vk_mem_alloc.h>

namespace Vulkan {

vk::DynamicLoader Instance::dl;

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data, void* user_data) {

    switch (callback_data->messageIdNumber) {
    case 0x609a13b: // Vertex attribute at location not consumed by shader
        return VK_FALSE;
    default:
        break;
    }

    Log::Level level{};
    switch (severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        level = Log::Level::Error;
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        level = Log::Level::Info;
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        level = Log::Level::Debug;
        break;
    default:
        level = Log::Level::Info;
    }

    LOG_GENERIC(Log::Class::Render_Vulkan, level, "{}: {}",
                callback_data->pMessageIdName ? callback_data->pMessageIdName : "<null>",
                callback_data->pMessage ? callback_data->pMessage : "<null>");

    return VK_FALSE;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL DebugReportCallback(VkDebugReportFlagsEXT flags,
                                                          VkDebugReportObjectTypeEXT objectType,
                                                          uint64_t object, size_t location,
                                                          int32_t messageCode,
                                                          const char* pLayerPrefix,
                                                          const char* pMessage, void* pUserData) {

    const VkDebugReportFlagBitsEXT severity = static_cast<VkDebugReportFlagBitsEXT>(flags);
    Log::Level level{};
    switch (severity) {
    case VK_DEBUG_REPORT_ERROR_BIT_EXT:
        level = Log::Level::Error;
        break;
    case VK_DEBUG_REPORT_INFORMATION_BIT_EXT:
        level = Log::Level::Warning;
        break;
    case VK_DEBUG_REPORT_DEBUG_BIT_EXT:
    case VK_DEBUG_REPORT_WARNING_BIT_EXT:
    case VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT:
        level = Log::Level::Debug;
        break;
    default:
        level = Log::Level::Info;
    }

    const vk::DebugReportObjectTypeEXT type = static_cast<vk::DebugReportObjectTypeEXT>(objectType);
    LOG_GENERIC(Log::Class::Render_Vulkan, level,
                "type = {}, object = {} | MessageCode = {:#x}, LayerPrefix = {} | {}",
                vk::to_string(type), object, messageCode, pLayerPrefix, pMessage);

    return VK_FALSE;
}

vk::Format MakeFormat(VideoCore::PixelFormat format) {
    switch (format) {
    case VideoCore::PixelFormat::RGBA8:
        return vk::Format::eR8G8B8A8Unorm;
    case VideoCore::PixelFormat::RGB8:
        return vk::Format::eB8G8R8Unorm;
    case VideoCore::PixelFormat::RGB5A1:
        return vk::Format::eR5G5B5A1UnormPack16;
    case VideoCore::PixelFormat::RGB565:
        return vk::Format::eR5G6B5UnormPack16;
    case VideoCore::PixelFormat::RGBA4:
        return vk::Format::eR4G4B4A4UnormPack16;
    case VideoCore::PixelFormat::D16:
        return vk::Format::eD16Unorm;
    case VideoCore::PixelFormat::D24:
        return vk::Format::eX8D24UnormPack32;
    case VideoCore::PixelFormat::D24S8:
        return vk::Format::eD24UnormS8Uint;
    case VideoCore::PixelFormat::Invalid:
        LOG_ERROR(Render_Vulkan, "Unknown texture format {}!", format);
        return vk::Format::eUndefined;
    default:
        return vk::Format::eR8G8B8A8Unorm; ///< Use default case for the texture formats
    }
}

vk::Format MakeCustomFormat(VideoCore::CustomPixelFormat format) {
    switch (format) {
    case VideoCore::CustomPixelFormat::RGBA8:
        return vk::Format::eR8G8B8A8Unorm;
    case VideoCore::CustomPixelFormat::BC1:
        return vk::Format::eBc1RgbaUnormBlock;
    case VideoCore::CustomPixelFormat::BC3:
        return vk::Format::eBc3UnormBlock;
    case VideoCore::CustomPixelFormat::BC5:
        return vk::Format::eBc5UnormBlock;
    case VideoCore::CustomPixelFormat::BC7:
        return vk::Format::eBc7UnormBlock;
    case VideoCore::CustomPixelFormat::ASTC4:
        return vk::Format::eAstc4x4UnormBlock;
    case VideoCore::CustomPixelFormat::ASTC6:
        return vk::Format::eAstc6x6UnormBlock;
    case VideoCore::CustomPixelFormat::ASTC8:
        return vk::Format::eAstc8x6UnormBlock;
    }
    LOG_ERROR(Render_Vulkan, "Unknown custom format {}", format);
    return vk::Format::eR8G8B8A8Unorm;
}

vk::Format MakeAttributeFormat(Pica::PipelineRegs::VertexAttributeFormat format, u32 count,
                               bool scaled = true) {
    static constexpr std::array attrib_formats_scaled = {
        vk::Format::eR8Sscaled,        vk::Format::eR8G8Sscaled,
        vk::Format::eR8G8B8Sscaled,    vk::Format::eR8G8B8A8Sscaled,
        vk::Format::eR8Uscaled,        vk::Format::eR8G8Uscaled,
        vk::Format::eR8G8B8Uscaled,    vk::Format::eR8G8B8A8Uscaled,
        vk::Format::eR16Sscaled,       vk::Format::eR16G16Sscaled,
        vk::Format::eR16G16B16Sscaled, vk::Format::eR16G16B16A16Sscaled,
        vk::Format::eR32Sfloat,        vk::Format::eR32G32Sfloat,
        vk::Format::eR32G32B32Sfloat,  vk::Format::eR32G32B32A32Sfloat,
    };
    static constexpr std::array attrib_formats_int = {
        vk::Format::eR8Sint,          vk::Format::eR8G8Sint,
        vk::Format::eR8G8B8Sint,      vk::Format::eR8G8B8A8Sint,
        vk::Format::eR8Uint,          vk::Format::eR8G8Uint,
        vk::Format::eR8G8B8Uint,      vk::Format::eR8G8B8A8Uint,
        vk::Format::eR16Sint,         vk::Format::eR16G16Sint,
        vk::Format::eR16G16B16Sint,   vk::Format::eR16G16B16A16Sint,
        vk::Format::eR32Sfloat,       vk::Format::eR32G32Sfloat,
        vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32B32A32Sfloat,
    };

    const u32 index = static_cast<u32>(format);
    return (scaled ? attrib_formats_scaled : attrib_formats_int)[index * 4 + count - 1];
}

vk::ImageAspectFlags MakeAspect(VideoCore::SurfaceType type) {
    switch (type) {
    case VideoCore::SurfaceType::Color:
    case VideoCore::SurfaceType::Texture:
    case VideoCore::SurfaceType::Fill:
        return vk::ImageAspectFlagBits::eColor;
    case VideoCore::SurfaceType::Depth:
        return vk::ImageAspectFlagBits::eDepth;
    case VideoCore::SurfaceType::DepthStencil:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    default:
        LOG_CRITICAL(Render_Vulkan, "Invalid surface type {}", type);
        UNREACHABLE();
    }

    return vk::ImageAspectFlagBits::eColor;
}

[[nodiscard]] vk::DebugUtilsMessengerCreateInfoEXT MakeDebugUtilsMessengerInfo() {
    return vk::DebugUtilsMessengerCreateInfoEXT{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = DebugUtilsCallback,
    };
}

[[nodiscard]] vk::DebugReportCallbackCreateInfoEXT MakeDebugReportCallbackInfo() {
    return vk::DebugReportCallbackCreateInfoEXT{
        .flags = vk::DebugReportFlagBitsEXT::eDebug | vk::DebugReportFlagBitsEXT::eInformation |
                 vk::DebugReportFlagBitsEXT::eError |
                 vk::DebugReportFlagBitsEXT::ePerformanceWarning |
                 vk::DebugReportFlagBitsEXT::eWarning,
        .pfnCallback = DebugReportCallback,
    };
}

std::vector<std::string> GetSupportedExtensions(vk::PhysicalDevice physical) {
    const std::vector extensions = physical.enumerateDeviceExtensionProperties();
    std::vector<std::string> supported_extensions;
    supported_extensions.reserve(extensions.size());
    for (const auto& extension : extensions) {
        supported_extensions.emplace_back(extension.extensionName.data());
    }
    return supported_extensions;
}

Instance::Instance(bool validation, bool dump_command_buffers)
    : enable_validation{validation}, dump_command_buffers{dump_command_buffers} {
    auto vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Enable the instance extensions the platform requires
    const std::vector extensions =
        GetInstanceExtensions(Frontend::WindowSystemType::Headless, false);

    const vk::ApplicationInfo application_info = {
        .pApplicationName = "Citra",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Citra Vulkan",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_0,
    };

    std::array<const char*, 2> layers;
    u32 layer_count = 0;

    if (enable_validation) {
        layers[layer_count++] = "VK_LAYER_KHRONOS_validation";
    }
    if (dump_command_buffers) {
        layers[layer_count++] = "VK_LAYER_LUNARG_api_dump";
    }

    const vk::StructureChain instance_chain = {
        vk::InstanceCreateInfo{
            .flags = GetInstanceFlags(),
            .pApplicationInfo = &application_info,
            .enabledLayerCount = layer_count,
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<u32>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        },
    };

    instance = vk::createInstance(instance_chain.get());

    // Load required function pointers for querying the physical device
    VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumeratePhysicalDevices = PFN_vkEnumeratePhysicalDevices(
        vkGetInstanceProcAddr(instance, "vkEnumeratePhysicalDevices"));
    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPhysicalDeviceProperties = PFN_vkGetPhysicalDeviceProperties(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties"));
    VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyInstance =
        PFN_vkDestroyInstance(vkGetInstanceProcAddr(instance, "vkDestroyInstance"));

    physical_devices = instance.enumeratePhysicalDevices();
}

Instance::Instance(Frontend::EmuWindow& window, u32 physical_device_index)
    : enable_validation{Settings::values.renderer_debug},
      dump_command_buffers{Settings::values.dump_command_buffers} {
    const Frontend::EmuWindow::WindowSystemInfo window_info = window.GetWindowInfo();

    // Fetch instance independant function pointers
    auto vkGetInstanceProcAddr =
        dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    // Enable the instance extensions the backend uses
    const std::vector extensions = GetInstanceExtensions(window_info.type, enable_validation);

    // We require a Vulkan 1.1 driver
    const u32 available_version = vk::enumerateInstanceVersion();
    if (available_version < VK_API_VERSION_1_1) {
        LOG_CRITICAL(Render_Vulkan, "Vulkan 1.0 is not supported, 1.1 is required!");
        return;
    }

    const vk::ApplicationInfo application_info = {
        .pApplicationName = "Citra",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "Citra Vulkan",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = available_version,
    };

    std::array<const char*, 3> layers;
#ifdef ANDROID
    u32 layer_count = 1;
    layers[0] = "VK_LAYER_KHRONOS_timeline_semaphore";
#else
    u32 layer_count = 0;
#endif

    if (enable_validation) {
        layers[layer_count++] = "VK_LAYER_KHRONOS_validation";
    }
    if (dump_command_buffers) {
        layers[layer_count++] = "VK_LAYER_LUNARG_api_dump";
    }

    vk::StructureChain instance_chain = {
        vk::InstanceCreateInfo{
            .flags = GetInstanceFlags(),
            .pApplicationInfo = &application_info,
            .enabledLayerCount = layer_count,
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<u32>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        },
        MakeDebugUtilsMessengerInfo(),
    };

    const auto IsSupported = [&extensions](std::string_view requested) {
        const auto it =
            std::find_if(extensions.begin(), extensions.end(),
                         [requested](const char* extension) { return requested == extension; });

        return it != extensions.end();
    };

    debug_messenger_supported = IsSupported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    debug_report_supported = IsSupported(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
    if (!debug_messenger_supported || !enable_validation) {
        instance_chain.unlink<vk::DebugUtilsMessengerCreateInfoEXT>();
    }

    try {
        instance = vk::createInstance(instance_chain.get());
    } catch (vk::LayerNotPresentError& err) {
        LOG_CRITICAL(Render_Vulkan, "Validation requested but layer is not available {}",
                     err.what());
        UNREACHABLE();
    }

    surface = CreateSurface(instance, window);

    // If validation is enabled attempt to also enable debug messenger
    if (enable_validation) {
        if (debug_messenger_supported) {
            debug_messenger = instance.createDebugUtilsMessengerEXT(MakeDebugUtilsMessengerInfo());
        } else if (debug_report_supported) {
            callback = instance.createDebugReportCallbackEXT(MakeDebugReportCallbackInfo());
        }
    }

    // Pick physical device
    physical_devices = instance.enumeratePhysicalDevices();
    if (const u16 physical_device_count = static_cast<u16>(physical_devices.size());
        physical_device_index >= physical_devices.size()) [[unlikely]] {
        LOG_CRITICAL(Render_Vulkan,
                     "Invalid physical device index {} provided when only {} devices exist",
                     physical_device_index, physical_device_count);
        UNREACHABLE();
    }

    physical_device = physical_devices[physical_device_index];
    properties = physical_device.getProperties();
    limits = properties.limits;

    LOG_INFO(Render_Vulkan, "Creating logical device for physical device: {}",
             properties.deviceName);

    CollectTelemetryParameters();
    CreateDevice();
    CreateFormatTable();
    CreateCustomFormatTable();
    CreateAttribTable();
}

Instance::~Instance() {
    if (device) {
        vmaDestroyAllocator(allocator);
        device.destroy();

        if (debug_messenger) {
            instance.destroyDebugUtilsMessengerEXT(debug_messenger);
        }
        if (callback) {
            instance.destroyDebugReportCallbackEXT(callback);
        }
    }

    instance.destroy();
}

const FormatTraits& Instance::GetTraits(VideoCore::PixelFormat pixel_format) const {
    if (pixel_format == VideoCore::PixelFormat::Invalid) [[unlikely]] {
        constexpr static FormatTraits null_traits{};
        return null_traits;
    }

    const u32 index = static_cast<u32>(pixel_format);
    return format_table[index];
}

const FormatTraits& Instance::GetTraits(VideoCore::CustomPixelFormat pixel_format) const {
    const u32 index = static_cast<u32>(pixel_format);
    return custom_format_table[index];
}

const FormatTraits& Instance::GetTraits(Pica::PipelineRegs::VertexAttributeFormat format,
                                        u32 count) const {
    if (count == 0) [[unlikely]] {
        ASSERT_MSG(false, "Unable to retrieve traits for invalid attribute component count");
    }

    const u32 index = static_cast<u32>(format);
    return attrib_table[index * 4 + count - 1];
}

FormatTraits Instance::DetermineTraits(VideoCore::PixelFormat pixel_format, vk::Format format) {
    const vk::ImageAspectFlags format_aspect = MakeAspect(VideoCore::GetFormatType(pixel_format));
    const vk::FormatProperties format_properties = physical_device.getFormatProperties(format);

    const vk::FormatFeatureFlagBits attachment_usage =
        (format_aspect & vk::ImageAspectFlagBits::eDepth)
            ? vk::FormatFeatureFlagBits::eDepthStencilAttachment
            : vk::FormatFeatureFlagBits::eColorAttachmentBlend;

    const vk::FormatFeatureFlags storage_usage = vk::FormatFeatureFlagBits::eStorageImage;
    const vk::FormatFeatureFlags transfer_usage = vk::FormatFeatureFlagBits::eSampledImage;
    const vk::FormatFeatureFlags blit_usage =
        vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;

    const bool supports_transfer =
        (format_properties.optimalTilingFeatures & transfer_usage) == transfer_usage;
    const bool supports_blit = (format_properties.optimalTilingFeatures & blit_usage) == blit_usage;
    const bool supports_attachment =
        (format_properties.optimalTilingFeatures & attachment_usage) == attachment_usage &&
        pixel_format != VideoCore::PixelFormat::RGB8;
    const bool supports_storage =
        (format_properties.optimalTilingFeatures & storage_usage) == storage_usage;
    const bool requires_conversion =
        // Requires component flip.
        pixel_format == VideoCore::PixelFormat::RGBA8 ||
        // Requires (de)interleaving.
        pixel_format == VideoCore::PixelFormat::D24S8;

    // Find the most inclusive usage flags for this format
    vk::ImageUsageFlags best_usage{};
    if (supports_blit || supports_transfer) {
        best_usage |= vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                      vk::ImageUsageFlagBits::eTransferSrc;
    }
    if (supports_attachment) {
        best_usage |= (format_aspect & vk::ImageAspectFlagBits::eDepth)
                          ? vk::ImageUsageFlagBits::eDepthStencilAttachment
                          : vk::ImageUsageFlagBits::eColorAttachment;
    }
    if (supports_storage) {
        best_usage |= vk::ImageUsageFlagBits::eStorage;
    }

    return FormatTraits{
        .transfer_support = supports_transfer,
        .blit_support = supports_blit,
        .attachment_support = supports_attachment,
        .storage_support = supports_storage,
        .requires_conversion = requires_conversion,
        .usage = best_usage,
        .aspect = format_aspect,
        .native = format,
    };
}

void Instance::CreateFormatTable() {
    constexpr std::array pixel_formats = {
        VideoCore::PixelFormat::RGBA8,  VideoCore::PixelFormat::RGB8,
        VideoCore::PixelFormat::RGB5A1, VideoCore::PixelFormat::RGB565,
        VideoCore::PixelFormat::RGBA4,  VideoCore::PixelFormat::IA8,
        VideoCore::PixelFormat::RG8,    VideoCore::PixelFormat::I8,
        VideoCore::PixelFormat::A8,     VideoCore::PixelFormat::IA4,
        VideoCore::PixelFormat::I4,     VideoCore::PixelFormat::A4,
        VideoCore::PixelFormat::ETC1,   VideoCore::PixelFormat::ETC1A4,
        VideoCore::PixelFormat::D16,    VideoCore::PixelFormat::D24,
        VideoCore::PixelFormat::D24S8,
    };

    for (const auto& pixel_format : pixel_formats) {
        const vk::Format format = MakeFormat(pixel_format);
        FormatTraits traits = DetermineTraits(pixel_format, format);

        const bool is_suitable =
            traits.transfer_support && traits.attachment_support &&
            (traits.blit_support || traits.aspect & vk::ImageAspectFlagBits::eDepth);
        // Fall back if the native format is not suitable.
        if (!is_suitable) {
            // Always fallback to RGBA8 or D32(S8) for convenience
            auto fallback = vk::Format::eR8G8B8A8Unorm;
            if (traits.aspect & vk::ImageAspectFlagBits::eDepth) {
                fallback = vk::Format::eD32Sfloat;
                if (traits.aspect & vk::ImageAspectFlagBits::eStencil) {
                    fallback = vk::Format::eD32SfloatS8Uint;
                }
            }
            LOG_WARNING(Render_Vulkan, "Format {} unsupported, falling back unconditionally to {}",
                        vk::to_string(format), vk::to_string(fallback));
            traits = DetermineTraits(pixel_format, fallback);
            // Always requires conversion if backing format does not match.
            traits.requires_conversion = true;
        }

        const u32 index = static_cast<u32>(pixel_format);
        format_table[index] = traits;
    }
}

void Instance::CreateCustomFormatTable() {
    // The traits are the same for RGBA8
    custom_format_table[0] = format_table[static_cast<u32>(VideoCore::PixelFormat::RGBA8)];

    constexpr std::array custom_formats = {
        VideoCore::CustomPixelFormat::BC1,   VideoCore::CustomPixelFormat::BC3,
        VideoCore::CustomPixelFormat::BC5,   VideoCore::CustomPixelFormat::BC7,
        VideoCore::CustomPixelFormat::ASTC4, VideoCore::CustomPixelFormat::ASTC6,
        VideoCore::CustomPixelFormat::ASTC8,
    };

    for (const auto& custom_format : custom_formats) {
        const vk::Format format = MakeCustomFormat(custom_format);
        const vk::FormatProperties format_properties = physical_device.getFormatProperties(format);

        // Compressed formats don't support blit_dst in general so just check for transfer
        const vk::FormatFeatureFlags transfer_usage = vk::FormatFeatureFlagBits::eSampledImage;
        const bool supports_transfer =
            (format_properties.optimalTilingFeatures & transfer_usage) == transfer_usage;

        vk::ImageUsageFlags best_usage{};
        if (supports_transfer) {
            best_usage |= vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                          vk::ImageUsageFlagBits::eTransferSrc;
        }

        const u32 index = static_cast<u32>(custom_format);
        custom_format_table[index] = FormatTraits{
            .transfer_support = supports_transfer,
            .usage = best_usage,
            .aspect = vk::ImageAspectFlagBits::eColor,
            .native = format,
        };
    }
}

void Instance::DetermineEmulation(Pica::PipelineRegs::VertexAttributeFormat format,
                                  bool& needs_cast) {
    // Check if (u)scaled formats can be used to emulate the 3 component format
    vk::Format two_comp_format = MakeAttributeFormat(format, 2);
    vk::FormatProperties format_properties = physical_device.getFormatProperties(two_comp_format);
    needs_cast = !(format_properties.bufferFeatures & vk::FormatFeatureFlagBits::eVertexBuffer);
}

void Instance::CreateAttribTable() {
    constexpr std::array attrib_formats = {
        Pica::PipelineRegs::VertexAttributeFormat::BYTE,
        Pica::PipelineRegs::VertexAttributeFormat::UBYTE,
        Pica::PipelineRegs::VertexAttributeFormat::SHORT,
        Pica::PipelineRegs::VertexAttributeFormat::FLOAT,
    };

    for (const auto& format : attrib_formats) {
        for (u32 count = 1; count <= 4; count++) {
            bool needs_cast{false};
            bool needs_emulation{false};
            vk::Format attrib_format = MakeAttributeFormat(format, count);
            vk::FormatProperties format_properties =
                physical_device.getFormatProperties(attrib_format);
            if (!(format_properties.bufferFeatures & vk::FormatFeatureFlagBits::eVertexBuffer)) {
                needs_cast = true;
                attrib_format = MakeAttributeFormat(format, count, false);
                format_properties = physical_device.getFormatProperties(attrib_format);
                if (!(format_properties.bufferFeatures &
                      vk::FormatFeatureFlagBits::eVertexBuffer)) {
                    ASSERT_MSG(
                        count == 3,
                        "Vertex attribute emulation is only supported for 3 component formats");
                    DetermineEmulation(format, needs_cast);
                    needs_emulation = true;
                }
            }

            const u32 index = static_cast<u32>(format) * 4 + count - 1;
            attrib_table[index] = FormatTraits{
                .requires_conversion = needs_cast,
                .requires_emulation = needs_emulation,
                .native = attrib_format,
            };
        }
    }
}

bool Instance::CreateDevice() {
    const vk::StructureChain feature_chain = physical_device.getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDevicePortabilitySubsetFeaturesKHR,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT,
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
        vk::PhysicalDeviceCustomBorderColorFeaturesEXT, vk::PhysicalDeviceIndexTypeUint8FeaturesEXT,
        vk::PhysicalDevicePipelineCreationCacheControlFeaturesEXT,
        vk::PhysicalDeviceDynamicRenderingFeaturesKHR>();
    const vk::StructureChain properties_chain =
        physical_device.getProperties2<vk::PhysicalDeviceProperties2,
                                       vk::PhysicalDevicePortabilitySubsetPropertiesKHR>();

    // Not having geometry shaders will cause issues with accelerated rendering.
    features = feature_chain.get().features;
    if (!features.geometryShader) {
        LOG_WARNING(Render_Vulkan,
                    "Geometry shaders not availabe! Accelerated rendering not possible!");
    }

    available_extensions = GetSupportedExtensions(physical_device);
    if (available_extensions.empty()) {
        LOG_CRITICAL(Render_Vulkan, "No extensions supported by device.");
        return false;
    }

    std::vector<const char*> enabled_extensions;
    const auto AddExtension = [&](std::string_view extension, bool blacklist = false,
                                  std::string_view reason = "") -> bool {
        auto result = std::find_if(available_extensions.begin(), available_extensions.end(),
                                   [&](const std::string& name) { return name == extension; });

        if (result != available_extensions.end() && !blacklist) {
            LOG_INFO(Render_Vulkan, "Enabling extension: {}", extension);
            enabled_extensions.push_back(extension.data());
            return true;
        } else if (blacklist) {
            LOG_WARNING(Render_Vulkan, "Extension {} has been blacklisted because {}", extension,
                        reason);
        }

        LOG_WARNING(Render_Vulkan, "Extension {} unavailable.", extension);
        return false;
    };

    const bool is_arm = driver_id == vk::DriverIdKHR::eArmProprietary;
    const bool is_qualcomm = driver_id == vk::DriverIdKHR::eQualcommProprietary;
    const bool is_radv = driver_id == vk::DriverIdKHR::eMesaRadv;

    AddExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    timeline_semaphores = AddExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    image_format_list = AddExtension(VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME);
    pipeline_creation_feedback = AddExtension(VK_EXT_PIPELINE_CREATION_FEEDBACK_EXTENSION_NAME);
    shader_stencil_export = AddExtension(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    bool has_portability_subset = AddExtension(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME);
    bool has_dynamic_rendering = AddExtension(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    bool has_extended_dynamic_state =
        AddExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME, is_arm || is_qualcomm,
                     "it is broken on Qualcomm and ARM drivers");
    bool has_extended_dynamic_state2 =
        AddExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME, is_qualcomm,
                     "it is broken on Qualcomm drivers");
    bool has_extended_dynamic_state3 = AddExtension(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
                                                    is_radv, "it is broken on RADV drivers");
    bool has_custom_border_color = AddExtension(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    bool has_index_type_uint8 = AddExtension(VK_EXT_INDEX_TYPE_UINT8_EXTENSION_NAME);
    bool has_pipeline_creation_cache_control =
        AddExtension(VK_EXT_PIPELINE_CREATION_CACHE_CONTROL_EXTENSION_NAME);

    // Search queue families for graphics and present queues
    auto family_properties = physical_device.getQueueFamilyProperties();
    if (family_properties.empty()) {
        LOG_CRITICAL(Render_Vulkan, "Physical device reported no queues.");
        return false;
    }

    bool graphics_queue_found = false;
    bool present_queue_found = false;
    for (std::size_t i = 0; i < family_properties.size(); i++) {
        // Check if queue supports graphics
        const u32 index = static_cast<u32>(i);
        if (family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            graphics_queue_family_index = index;
            graphics_queue_found = true;

            // If this queue also supports presentation we are finished
            if (physical_device.getSurfaceSupportKHR(static_cast<u32>(i), surface)) {
                present_queue_family_index = index;
                present_queue_found = true;
                break;
            }
        }

        // Check if queue supports presentation
        if (physical_device.getSurfaceSupportKHR(index, surface)) {
            present_queue_family_index = index;
            present_queue_found = true;
        }
    }

    if (!graphics_queue_found || !present_queue_found) {
        LOG_CRITICAL(Render_Vulkan, "Unable to find graphics and/or present queues.");
        return false;
    }

    static constexpr float queue_priorities[] = {1.0f};

    const std::array queue_infos = {
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = graphics_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = queue_priorities,
        },
        vk::DeviceQueueCreateInfo{
            .queueFamilyIndex = present_queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = queue_priorities,
        },
    };

    const u32 queue_count = graphics_queue_family_index != present_queue_family_index ? 2u : 1u;
    vk::StructureChain device_chain = {
        vk::DeviceCreateInfo{
            .queueCreateInfoCount = queue_count,
            .pQueueCreateInfos = queue_infos.data(),
            .enabledExtensionCount = static_cast<u32>(enabled_extensions.size()),
            .ppEnabledExtensionNames = enabled_extensions.data(),
        },
        vk::PhysicalDeviceFeatures2{
            .features{
                .geometryShader = features.geometryShader,
                .logicOp = features.logicOp,
                .depthClamp = features.depthClamp,
                .largePoints = features.largePoints,
                .samplerAnisotropy = features.samplerAnisotropy,
                .fragmentStoresAndAtomics = features.fragmentStoresAndAtomics,
                .shaderClipDistance = features.shaderClipDistance,
            },
        },
        vk::PhysicalDevicePortabilitySubsetFeaturesKHR{},
#ifndef ANDROID ///< Android uses the extension layer. Avoid enabling the feature because it messes
                ///< with the pNext chain
        vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR{
            .timelineSemaphore = true,
        },
#endif
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT{},
        vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT{},
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT{},
        vk::PhysicalDeviceDynamicRenderingFeaturesKHR{},
        vk::PhysicalDeviceCustomBorderColorFeaturesEXT{},
        vk::PhysicalDeviceIndexTypeUint8FeaturesEXT{},
        vk::PhysicalDevicePipelineCreationCacheControlFeaturesEXT{},
    };

#define PROP_GET(structName, prop, property) property = properties_chain.get<structName>().prop;

#define FEAT_SET(structName, feature, property)                                                    \
    if (feature_chain.get<structName>().feature) {                                                 \
        property = true;                                                                           \
        device_chain.get<structName>().feature = true;                                             \
    } else {                                                                                       \
        property = false;                                                                          \
        device_chain.get<structName>().feature = false;                                            \
    }

    if (has_portability_subset) {
        FEAT_SET(vk::PhysicalDevicePortabilitySubsetFeaturesKHR, triangleFans,
                 triangle_fan_supported)
        FEAT_SET(vk::PhysicalDevicePortabilitySubsetFeaturesKHR, imageViewFormatReinterpretation,
                 image_view_reinterpretation)
        PROP_GET(vk::PhysicalDevicePortabilitySubsetPropertiesKHR,
                 minVertexInputBindingStrideAlignment, min_vertex_stride_alignment)
    } else {
        device_chain.unlink<vk::PhysicalDevicePortabilitySubsetFeaturesKHR>();
    }

    if (has_index_type_uint8) {
        FEAT_SET(vk::PhysicalDeviceIndexTypeUint8FeaturesEXT, indexTypeUint8, index_type_uint8)
    } else {
        device_chain.unlink<vk::PhysicalDeviceIndexTypeUint8FeaturesEXT>();
    }

    if (has_extended_dynamic_state) {
        FEAT_SET(vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT, extendedDynamicState,
                 extended_dynamic_state)
    } else {
        device_chain.unlink<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    }

    if (has_extended_dynamic_state2) {
        FEAT_SET(vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT, extendedDynamicState2LogicOp,
                 extended_dynamic_state2)
    } else {
        device_chain.unlink<vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT>();
    }

    if (has_extended_dynamic_state3) {
        FEAT_SET(vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
                 extendedDynamicState3LogicOpEnable, extended_dynamic_state3_logicop_enable)
        FEAT_SET(vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
                 extendedDynamicState3ColorBlendEnable, extended_dynamic_state3_color_blend_enable)
        FEAT_SET(vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
                 extendedDynamicState3ColorBlendEquation, extended_dynamic_state3_color_blend_eq)
        FEAT_SET(vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
                 extendedDynamicState3ColorWriteMask, extended_dynamic_state3_color_write_mask)
    } else {
        device_chain.unlink<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>();
    }

    if (has_dynamic_rendering) {
        FEAT_SET(vk::PhysicalDeviceDynamicRenderingFeaturesKHR, dynamicRendering,
                 dynamic_rendering);
    } else {
        device_chain.unlink<vk::PhysicalDeviceDynamicRenderingFeaturesKHR>();
    }

    if (has_custom_border_color) {
        FEAT_SET(vk::PhysicalDeviceCustomBorderColorFeaturesEXT, customBorderColors,
                 custom_border_color)
        FEAT_SET(vk::PhysicalDeviceCustomBorderColorFeaturesEXT, customBorderColorWithoutFormat,
                 custom_border_color)
    } else {
        device_chain.unlink<vk::PhysicalDeviceCustomBorderColorFeaturesEXT>();
    }

    if (has_pipeline_creation_cache_control) {
        FEAT_SET(vk::PhysicalDevicePipelineCreationCacheControlFeaturesEXT,
                 pipelineCreationCacheControl, pipeline_creation_cache_control)
    } else {
        device_chain.unlink<vk::PhysicalDevicePipelineCreationCacheControlFeaturesEXT>();
    }

#undef PROP_GET
#undef FEAT_SET

    try {
        device = physical_device.createDevice(device_chain.get());
    } catch (vk::ExtensionNotPresentError& err) {
        LOG_CRITICAL(Render_Vulkan, "Some required extensions are not available {}", err.what());
        UNREACHABLE();
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    graphics_queue = device.getQueue(graphics_queue_family_index, 0);
    present_queue = device.getQueue(present_queue_family_index, 0);

    CreateAllocator();
    return true;
}

void Instance::CreateAllocator() {
    const VmaVulkanFunctions functions = {
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };

    const VmaAllocatorCreateInfo allocator_info = {
        .physicalDevice = physical_device,
        .device = device,
        .pVulkanFunctions = &functions,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_1,
    };

    if (VkResult result = vmaCreateAllocator(&allocator_info, &allocator); result != VK_SUCCESS) {
        LOG_CRITICAL(Render_Vulkan, "Failed to initialize VMA with error {}", result);
        UNREACHABLE();
    }
}

void Instance::CollectTelemetryParameters() {
    const vk::StructureChain property_chain =
        physical_device
            .getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDriverProperties>();
    const vk::PhysicalDeviceDriverProperties driver =
        property_chain.get<vk::PhysicalDeviceDriverProperties>();

    driver_id = driver.driverID;
    vendor_name = driver.driverName.data();
}

} // namespace Vulkan
