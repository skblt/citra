// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <sirit/sirit.h>
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

using Sirit::Id;

struct VectorIds {
    /// Returns the type id of the vector with the provided size
    [[nodiscard]] constexpr Id Get(u32 size) const {
        return ids[size - 2];
    }

    std::array<Id, 3> ids;
};

class FragmentModule : public Sirit::Module {
    static constexpr u32 NUM_TEV_STAGES = 6;
    static constexpr u32 NUM_LIGHTS = 8;
    static constexpr u32 NUM_LIGHTING_SAMPLERS = 24;
public:
    FragmentModule(const PicaFSConfig& config);
    ~FragmentModule();

    /// Emits SPIR-V bytecode corresponding to the provided pica fragment configuration
    void Generate();

    /// Undos the vulkan perspective transformation and applies the pica one
    void WriteDepth();

    /// Writes the code to emulate fragment lighting
    void WriteLighting();

    /// Writes the code to emulate the specified TEV stage
    void WriteTevStage(s32 index);

    /// Writes the if-statement condition used to evaluate alpha testing.
    /// Returns true if the fragment was discarded
    [[nodiscard]] bool WriteAlphaTestCondition(Pica::FramebufferRegs::CompareFunc func);

    /// Samples the current fragment texel from the provided texture unit
    [[nodiscard]] Id SampleTexture(u32 texture_unit);

    /// Rounds the provided variable to the nearest 1/255th
    [[nodiscard]] Id Byteround(Id variable_id, u32 size = 1);

    /// Lookups the lighting LUT at the provided lut_index
    [[nodiscard]] Id LookupLightingLUT(Id lut_index, Id index, Id delta);

    /// Writes the specified TEV stage source component(s)
    [[nodiscard]] Id AppendSource(Pica::TexturingRegs::TevStageConfig::Source source, s32 index);

    /// Writes the color components to use for the specified TEV stage color modifier
    [[nodiscard]] Id AppendColorModifier(Pica::TexturingRegs::TevStageConfig::ColorModifier modifier,
                                         Pica::TexturingRegs::TevStageConfig::Source source, s32 index);

    /// Writes the alpha component to use for the specified TEV stage alpha modifier
    [[nodiscard]] Id AppendAlphaModifier(Pica::TexturingRegs::TevStageConfig::AlphaModifier modifier,
                                         Pica::TexturingRegs::TevStageConfig::Source source, s32 index);

    /// Writes the combiner function for the color components for the specified TEV stage operation
    [[nodiscard]] Id AppendColorCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation);

    /// Writes the combiner function for the alpha component for the specified TEV stage operation
    [[nodiscard]] Id AppendAlphaCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation);

    /// Loads the member specified from the shader_data uniform struct
    template <typename... Ids>
    [[nodiscard]] Id GetShaderDataMember(Id type, Ids... ids) {
        const Id uniform_ptr{TypePointer(spv::StorageClass::Uniform, type)};
        return OpLoad(type, OpAccessChain(uniform_ptr, shader_data_id, ids...));
    }

    /// Pads the provided vector by inserting args at the end
    template <typename... Args>
    [[nodiscard]] Id PadVectorF32(Id vector, Id pad_type_id, Args&&... args) {
        return OpCompositeConstruct(pad_type_id, vector, ConstF32(args...));
    }

    /// Defines a input variable
    [[nodiscard]] Id DefineInput(Id type, u32 location) {
        const Id input_id{DefineVar(type, spv::StorageClass::Input)};
        Decorate(input_id, spv::Decoration::Location, location);
        return input_id;
    }

    /// Defines a input variable
    [[nodiscard]] Id DefineOutput(Id type, u32 location) {
        const Id output_id{DefineVar(type, spv::StorageClass::Output)};
        Decorate(output_id, spv::Decoration::Location, location);
        return output_id;
    }

    /// Defines a uniform constant variable
    [[nodiscard]] Id DefineUniformConst(Id type, u32 set, u32 binding) {
        const Id uniform_id{DefineVar(type, spv::StorageClass::UniformConstant)};
        Decorate(uniform_id, spv::Decoration::DescriptorSet, set);
        Decorate(uniform_id, spv::Decoration::Binding, binding);
        return uniform_id;
    }

    [[nodiscard]] Id DefineVar(Id type, spv::StorageClass storage_class) {
        const Id pointer_type_id{TypePointer(storage_class, type)};
        return AddGlobalVariable(pointer_type_id, storage_class);
    }

    /// Returns the id of a signed integer constant of value
    [[nodiscard]] Id ConstU32(u32 value) {
        return Constant(u32_id, value);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstU32(Args&&... values) {
        constexpr auto size = sizeof...(values);
        static_assert(size >= 2 && size <= 4);
        const std::array constituents{Constant(u32_id, values)...};
        return ConstantComposite(uvec_ids.Get(size), constituents);
    }

    /// Returns the id of a signed integer constant of value
    [[nodiscard]] Id ConstS32(s32 value) {
        return Constant(i32_id, value);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstS32(Args&&... values) {
        constexpr auto size = sizeof...(values);
        static_assert(size >= 2 && size <= 4);
        const std::array constituents{Constant(i32_id, values)...};
        return ConstantComposite(ivec_ids.Get(size), constituents);
    }

    /// Returns the id of a float constant of value
    [[nodiscard]] Id ConstF32(float value) {
        return Constant(f32_id, value);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstF32(Args... values) {
        constexpr auto size = sizeof...(values);
        static_assert(size >= 2 && size <= 4);
        const std::array constituents{Constant(f32_id, values)...};
        return ConstantComposite(vec_ids.Get(size), constituents);
    }

private:
    void DefineArithmeticTypes();
    void DefineEntryPoint();
    void DefineUniformStructs();
    void DefineInterface();

private:
    PicaFSConfig config;
    Id void_id{};
    Id bool_id{};
    Id f32_id{};
    Id i32_id{};
    Id u32_id{};

    VectorIds vec_ids{};
    VectorIds ivec_ids{};
    VectorIds uvec_ids{};

    Id image2d_id{};
    Id image_cube_id{};
    Id image_buffer_id{};
    Id sampler_id{};
    Id shader_data_id{};

    Id primary_color_id{};
    Id texcoord0_id{};
    Id texcoord1_id{};
    Id texcoord2_id{};
    Id texcoord0_w_id{};
    Id normquat_id{};
    Id view_id{};
    Id color_id{};

    Id gl_frag_coord_id{};
    Id gl_frag_depth_id{};

    Id tex0_id{};
    Id tex1_id{};
    Id tex2_id{};
    Id tex_cube_id{};
    Id tex0_sampler_id{};
    Id tex1_sampler_id{};
    Id tex2_sampler_id{};
    Id tex_cube_sampler_id{};
    Id texture_buffer_lut_lf_id{};
    Id texture_buffer_lut_rg_id{};
    Id texture_buffer_lut_rgba_id{};

    Id texture_buffer_lut_lf{};

    Id rounded_primary_color{};
    Id primary_fragment_color{};
    Id secondary_fragment_color{};
    Id combiner_buffer{};
    Id next_combiner_buffer{};
    Id last_tex_env_out{};

    Id color_results_1{};
    Id color_results_2{};
    Id color_results_3{};
    Id alpha_results_1{};
    Id alpha_results_2{};
    Id alpha_results_3{};
};

/**
 * Generates the SPIR-V fragment shader program source code for the current Pica state
 * @param config ShaderCacheKey object generated for the current Pica state, used for the shader
 *               configuration (NOTE: Use state in this struct only, not the Pica registers!)
 * @param separable_shader generates shader that can be used for separate shader object
 * @returns String of the shader source code
 */
std::vector<u32> GenerateFragmentShaderSPV(const PicaFSConfig& config);

} // namespace Vulkan
