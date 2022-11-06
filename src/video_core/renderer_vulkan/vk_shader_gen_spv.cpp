// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <fstream>
#include "video_core/regs.h"
#include "video_core/renderer_vulkan/vk_shader_gen_spv.h"
#include "video_core/shader/shader_uniforms.h"

using Pica::FramebufferRegs;
using Pica::LightingRegs;
using Pica::RasterizerRegs;
using Pica::TexturingRegs;
using TevStageConfig = TexturingRegs::TevStageConfig;

namespace Vulkan {

FragmentModule::FragmentModule(const PicaFSConfig& config) : Sirit::Module{0x00010300}, config{config} {
    DefineArithmeticTypes();
    DefineUniformStructs();
    DefineInterface();
    DefineEntryPoint();
}

FragmentModule::~FragmentModule() = default;

void FragmentModule::Generate() {
    const PicaFSConfigState& state = config.state;
    AddLabel(OpLabel());

    rounded_primary_color = Byteround(OpLoad(vec_ids.Get(4), primary_color_id), 4);
    primary_fragment_color = ConstF32(0.f, 0.f, 0.f, 0.f);
    secondary_fragment_color = ConstF32(0.f, 0.f, 0.f, 0.f);

    // Do not do any sort of processing if it's obvious we're not going to pass the alpha test
    if (state.alpha_test_func == Pica::FramebufferRegs::CompareFunc::Never) {
        OpKill();
        OpFunctionEnd();
        return;
    }

    // After perspective divide, OpenGL transform z_over_w from [-1, 1] to [near, far]. Here we use
    // default near = 0 and far = 1, and undo the transformation to get the original z_over_w, then
    // do our own transformation according to PICA specification.
    WriteDepth();

    // Write shader bytecode to emulate all enabled PICA lights
    if (state.lighting.enable) {
        WriteLighting();
    }

    combiner_buffer = ConstF32(0.f, 0.f, 0.f, 0.f);
    next_combiner_buffer = GetShaderDataMember(vec_ids.Get(4), ConstS32(27));
    last_tex_env_out = ConstF32(0.f, 0.f, 0.f, 0.f);

    // Write shader bytecode to emulate PICA TEV stages
    for (std::size_t index = 0; index < state.tev_stages.size(); ++index) {
        WriteTevStage(static_cast<s32>(index));
    }

    // Write output color
    OpStore(color_id, Byteround(last_tex_env_out, 4));
    OpReturn();
    OpFunctionEnd();
}

void FragmentModule::WriteDepth() {
    const Id input_pointer_id{TypePointer(spv::StorageClass::Input, f32_id)};
    const Id gl_frag_coord_z{OpLoad(f32_id, OpAccessChain(input_pointer_id, gl_frag_coord_id, ConstU32(2u)))};
    const Id z_over_w{OpFma(f32_id, ConstF32(2.f), gl_frag_coord_z, ConstF32(-1.f))};
    const Id uniform_pointer_id{TypePointer(spv::StorageClass::Uniform, f32_id)};
    const Id depth_scale{OpLoad(f32_id, OpAccessChain(uniform_pointer_id, shader_data_id, ConstS32(2)))};
    const Id depth_offset{OpLoad(f32_id, OpAccessChain(uniform_pointer_id, shader_data_id, ConstS32(3)))};
    const Id depth{OpFma(f32_id, z_over_w, depth_scale, depth_offset)};
    if (config.state.depthmap_enable == Pica::RasterizerRegs::DepthBuffering::WBuffering) {
        const Id gl_frag_coord_w{OpLoad(f32_id, OpAccessChain(input_pointer_id, gl_frag_coord_id, ConstU32(3u)))};
        const Id depth_over_w{OpFDiv(f32_id, depth, gl_frag_coord_w)};
        OpStore(gl_frag_depth_id, depth_over_w);
    } else {
        OpStore(gl_frag_depth_id, depth);
    }
}

void FragmentModule::WriteLighting() {
    const auto& lighting = config.state.lighting;

    // Define lighting globals
    Id diffuse_sum{ConstF32(0.f, 0.f, 0.f, 1.f)};
    Id specular_sum{ConstF32(0.f, 0.f, 0.f, 1.f)};
    Id light_vector{ConstF32(0.f, 0.f, 0.f)};
    Id spot_dir{ConstF32(0.f, 0.f, 0.f)};
    Id half_vector{ConstF32(0.f, 0.f, 0.f)};
    Id dot_product{ConstF32(0.f)};
    Id clamp_highlights{ConstF32(1.f)};
    Id geo_factor{ConstF32(1.f)};
    Id surface_normal{};
    Id surface_tangent{};

    // Compute fragment normals and tangents
    const auto Perturbation = [&]() -> Id {
        const Id texel{SampleTexture(lighting.bump_selector)};
        const Id texel_rgb{OpVectorShuffle(vec_ids.Get(3), texel, texel, 0, 1, 2)};
        const Id rgb_mul_two{OpVectorTimesScalar(vec_ids.Get(3), texel_rgb, ConstF32(2.f))};
        return OpFSub(vec_ids.Get(3), rgb_mul_two, ConstF32(1.f, 1.f, 1.f));
    };

    if (lighting.bump_mode == LightingRegs::LightingBumpMode::NormalMap) {
        // Bump mapping is enabled using a normal map
        surface_normal = Perturbation();

        // Recompute Z-component of perturbation if 'renorm' is enabled, this provides a higher
        // precision result
        if (lighting.bump_renorm) {
            const Id normal_x{OpCompositeExtract(f32_id, surface_normal, 0)};
            const Id normal_y{OpCompositeExtract(f32_id, surface_normal, 1)};
            const Id y_mul_y{OpFMul(f32_id, normal_y, normal_y)};
            const Id val{OpFSub(f32_id, ConstF32(1.f), OpFma(f32_id, normal_x, normal_x, y_mul_y))};
            const Id normal_z{OpSqrt(f32_id, OpFMax(f32_id, val, ConstF32(0.f)))};
            surface_normal = OpCompositeConstruct(vec_ids.Get(3), normal_x, normal_y, normal_z);
        }

        // The tangent vector is not perturbed by the normal map and is just a unit vector.
        surface_tangent = ConstF32(1.f, 0.f, 0.f);
    } else if (lighting.bump_mode == LightingRegs::LightingBumpMode::TangentMap) {
        // Bump mapping is enabled using a tangent map
        surface_tangent = Perturbation();

        // Mathematically, recomputing Z-component of the tangent vector won't affect the relevant
        // computation below, which is also confirmed on 3DS. So we don't bother recomputing here
        // even if 'renorm' is enabled.

        // The normal vector is not perturbed by the tangent map and is just a unit vector.
        surface_normal = ConstF32(0.f, 0.f, 1.f);
    } else {
        // No bump mapping - surface local normal and tangent are just unit vectors
        surface_normal = ConstF32(0.f, 0.f, 1.f);
        surface_tangent = ConstF32(1.f, 0.f, 0.f);
    }

    // Rotate the vector v by the quaternion q
    const auto QuaternionRotate = [this](Id q, Id v) -> Id {
        const Id q_xyz{OpVectorShuffle(vec_ids.Get(3), q, q, 0, 1, 2)};
        const Id q_xyz_cross_v{OpCross(vec_ids.Get(3), q_xyz, v)};
        const Id q_w{OpCompositeExtract(f32_id, q, 3)};
        const Id val1{OpFAdd(vec_ids.Get(3), q_xyz_cross_v, OpVectorTimesScalar(vec_ids.Get(3), v, q_w))};
        const Id val2{OpVectorTimesScalar(vec_ids.Get(3), OpCross(vec_ids.Get(3), q_xyz, val1), ConstF32(2.f))};
        return OpFAdd(vec_ids.Get(3), v, val2);
    };

    // Rotate the surface-local normal by the interpolated normal quaternion to convert it to
    // eyespace.
    const Id normalized_normquat{OpNormalize(vec_ids.Get(4), OpLoad(vec_ids.Get(4), normquat_id))};
    const Id normal{QuaternionRotate(normalized_normquat, surface_normal)};
    const Id tangent{QuaternionRotate(normalized_normquat, surface_tangent)};

    Id shadow{ConstF32(1.f, 1.f, 1.f, 1.f)};
    if (lighting.enable_shadow && false) {
        shadow = SampleTexture(lighting.shadow_selector);
        if (lighting.shadow_invert) {
            shadow = OpFSub(vec_ids.Get(4), ConstF32(1.f, 1.f, 1.f, 1.f), shadow);
        }
    }

    const auto LookupLightingLUTUnsigned = [this](Id lut_index, Id pos) -> Id {
        const Id pos_int{OpConvertFToS(i32_id, OpFMul(f32_id, pos, ConstF32(255.f)))};
        const Id index{OpSClamp(i32_id, pos_int, ConstS32(0), ConstS32(255))};
        const Id neg_index{OpFNegate(f32_id, OpConvertSToF(f32_id, index))};
        const Id delta{OpFma(f32_id, pos, ConstF32(255.f), neg_index)};
        return LookupLightingLUT(lut_index, index, delta);
    };

    const auto LookupLightingLUTSigned = [this](Id lut_index, Id pos) -> Id {
        const Id pos_int{OpConvertFToS(i32_id, OpFMul(f32_id, pos, ConstF32(128.f)))};
        const Id index{OpSClamp(i32_id, pos_int, ConstS32(-128), ConstS32(127))};
        const Id neg_index{OpFNegate(f32_id, OpConvertSToF(f32_id, index))};
        const Id delta{OpFma(f32_id, pos, ConstF32(128.f), neg_index)};
        const Id increment{OpSelect(i32_id, OpSLessThan(bool_id, index, ConstS32(0)), ConstS32(255), ConstS32(0))};
        return LookupLightingLUT(lut_index, OpIAdd(i32_id, index, increment), delta);
    };

    // Samples the specified lookup table for specular lighting
    const Id view{OpLoad(vec_ids.Get(3), view_id)};
    const auto GetLutValue = [&](LightingRegs::LightingSampler sampler, u32 light_num,
                                 LightingRegs::LightingLutInput input, bool abs) -> Id {
        Id index{};
        switch (input) {
        case LightingRegs::LightingLutInput::NH:
            index = OpDot(f32_id, normal, OpNormalize(vec_ids.Get(3), half_vector));
            break;
        case LightingRegs::LightingLutInput::VH:
            index = OpDot(f32_id, OpNormalize(vec_ids.Get(3), view), OpNormalize(vec_ids.Get(3), half_vector));
            break;
        case LightingRegs::LightingLutInput::NV:
            index = OpDot(f32_id, normal, OpNormalize(vec_ids.Get(3), view));
            break;
        case LightingRegs::LightingLutInput::LN:
            index = OpDot(f32_id, light_vector, normal);
            break;
        case LightingRegs::LightingLutInput::SP:
            index = OpDot(f32_id, light_vector, spot_dir);
            break;
        case LightingRegs::LightingLutInput::CP:
            // CP input is only available with configuration 7
            if (lighting.config == LightingRegs::LightingConfig::Config7) {
                // Note: even if the normal vector is modified by normal map, which is not the
                // normal of the tangent plane anymore, the half angle vector is still projected
                // using the modified normal vector.
                const Id normalized_half_vector{OpNormalize(vec_ids.Get(3), half_vector)};
                const Id normal_dot_half_vector{OpDot(f32_id, normal, normalized_half_vector)};
                const Id normal_mul_dot{OpVectorTimesScalar(vec_ids.Get(3), normal, normal_dot_half_vector)};
                const Id half_angle_proj{OpFSub(vec_ids.Get(3), normalized_half_vector, normal_mul_dot)};

                // Note: the half angle vector projection is confirmed not normalized before the dot
                // product. The result is in fact not cos(phi) as the name suggested.
                index = OpDot(f32_id, half_angle_proj, tangent);
            } else {
                index = ConstF32(0.f);
            }
            break;
        default:
            LOG_CRITICAL(HW_GPU, "Unknown lighting LUT input {}", (int)input);
            UNIMPLEMENTED();
            index = ConstF32(0.f);
            break;
        }

        const Id sampler_index{ConstU32(static_cast<u32>(sampler))};
        if (abs) {
            // LUT index is in the range of (0.0, 1.0)
            index = lighting.light[light_num].two_sided_diffuse
                        ? OpFAbs(f32_id, index)
                        : OpFMax(f32_id, index, ConstF32(0.f));
            return LookupLightingLUTUnsigned(sampler_index, index);
        } else {
            // LUT index is in the range of (-1.0, 1.0)
            return LookupLightingLUTSigned(sampler_index, index);
        }
    };

    // Write the code to emulate each enabled light
    for (u32 light_index = 0; light_index < lighting.src_num; ++light_index) {
        const auto& light_config = lighting.light[light_index];

        const auto GetLightMember = [&](s32 member) -> Id {
            const Id member_type = member < 6 ? vec_ids.Get(3) : f32_id;
            const Id uniform_pointer_id{TypePointer(spv::StorageClass::Uniform, member_type)};
            const Id light_num{ConstS32(static_cast<s32>(lighting.light[light_index].num.Value()))};
            return OpLoad(member_type, OpAccessChain(uniform_pointer_id, shader_data_id, ConstS32(25),
                                                     light_num, ConstS32(member)));
        };

        // Compute light vector (directional or positional)
        const Id light_position{GetLightMember(4)};
        if (light_config.directional) {
            light_vector = OpNormalize(vec_ids.Get(3), light_position);
        } else {
            light_vector = OpNormalize(vec_ids.Get(3), OpFAdd(vec_ids.Get(3), light_position, view));
        }

        spot_dir = GetLightMember(5);
        half_vector = OpFAdd(vec_ids.Get(3), OpNormalize(vec_ids.Get(3), view), light_vector);

        // Compute dot product of light_vector and normal, adjust if lighting is one-sided or
        // two-sided
        if (light_config.two_sided_diffuse) {
            dot_product = OpFAbs(f32_id, OpDot(f32_id, light_vector, normal));
        } else {
            dot_product = OpFMax(f32_id, OpDot(f32_id, light_vector, normal), ConstF32(0.f));
        }

        // If enabled, clamp specular component if lighting result is zero
        if (lighting.clamp_highlights) {
            clamp_highlights = OpFSign(f32_id, dot_product);
        }

        // If enabled, compute spot light attenuation value
        Id spot_atten{ConstF32(1.f)};
        if (light_config.spot_atten_enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::SpotlightAttenuation)) {
            const Id value{GetLutValue(LightingRegs::SpotlightAttenuationSampler(light_config.num),
                            light_config.num, lighting.lut_sp.type, lighting.lut_sp.abs_input)};
            spot_atten = OpFMul(f32_id, ConstF32(lighting.lut_sp.scale), value);
        }

        // If enabled, compute distance attenuation value
        Id dist_atten{ConstF32(1.f)};
        if (light_config.dist_atten_enable) {
            const Id dist_atten_scale{GetLightMember(7)};
            const Id dist_atten_bias{GetLightMember(6)};
            const Id min_view_min_pos{OpFSub(vec_ids.Get(3), OpFNegate(vec_ids.Get(3), view), light_position)};
            const Id index{OpFma(f32_id, dist_atten_scale, OpLength(f32_id, min_view_min_pos), dist_atten_bias)};
            const Id clamped_index{OpFClamp(f32_id, index, ConstF32(0.f), ConstF32(1.f))};
            const Id sampler{ConstS32(static_cast<s32>(LightingRegs::DistanceAttenuationSampler(light_config.num)))};
            dist_atten = LookupLightingLUTUnsigned(sampler, clamped_index);
        }

        if (light_config.geometric_factor_0 || light_config.geometric_factor_1) {
            geo_factor = OpDot(f32_id, half_vector, half_vector);
            const Id dot_div_geo{OpFMin(f32_id, OpFDiv(f32_id, dot_product, geo_factor), ConstF32(1.f))};
            const Id is_geo_factor_zero{OpFOrdEqual(bool_id, geo_factor, ConstF32(0.f))};
            geo_factor = OpSelect(f32_id, is_geo_factor_zero, ConstF32(0.f), dot_div_geo);
        }

        // Specular 0 component
        Id d0_lut_value{ConstF32(1.f)};
        if (lighting.lut_d0.enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::Distribution0)) {
            // Lookup specular "distribution 0" LUT value
            const Id value{GetLutValue(LightingRegs::LightingSampler::Distribution0, light_config.num,
                            lighting.lut_d0.type, lighting.lut_d0.abs_input)};
            d0_lut_value = OpFMul(f32_id, ConstF32(lighting.lut_d0.scale), value);
        }

        Id specular_0{OpVectorTimesScalar(vec_ids.Get(3), GetLightMember(0), d0_lut_value)};
        if (light_config.geometric_factor_0) {
            specular_0 = OpVectorTimesScalar(vec_ids.Get(3), specular_0, geo_factor);
        }

        // If enabled, lookup ReflectRed value, otherwise, 1.0 is used
        Id refl_value_r{ConstF32(1.f)};
        if (lighting.lut_rr.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::ReflectRed)) {
            const Id value{GetLutValue(LightingRegs::LightingSampler::ReflectRed, light_config.num,
                           lighting.lut_rr.type, lighting.lut_rr.abs_input)};

            refl_value_r = OpFMul(f32_id, ConstF32(lighting.lut_rr.scale), value);
        }

        // If enabled, lookup ReflectGreen value, otherwise, ReflectRed value is used
        Id refl_value_g{refl_value_r};
        if (lighting.lut_rg.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::ReflectGreen)) {
            const Id value{GetLutValue(LightingRegs::LightingSampler::ReflectGreen, light_config.num,
                           lighting.lut_rg.type, lighting.lut_rg.abs_input)};

            refl_value_g = OpFMul(f32_id, ConstF32(lighting.lut_rg.scale), value);
        }

        // If enabled, lookup ReflectBlue value, otherwise, ReflectRed value is used
        Id refl_value_b{refl_value_r};
        if (lighting.lut_rb.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::ReflectBlue)) {
            const Id value{GetLutValue(LightingRegs::LightingSampler::ReflectBlue, light_config.num,
                           lighting.lut_rb.type, lighting.lut_rb.abs_input)};
            refl_value_b = OpFMul(f32_id, ConstF32(lighting.lut_rb.scale), value);
        }

        // Specular 1 component
        Id d1_lut_value{ConstF32(1.f)};
        if (lighting.lut_d1.enable &&
            LightingRegs::IsLightingSamplerSupported(
                lighting.config, LightingRegs::LightingSampler::Distribution1)) {
            // Lookup specular "distribution 1" LUT value
            const Id value{GetLutValue(LightingRegs::LightingSampler::Distribution1, light_config.num,
                            lighting.lut_d1.type, lighting.lut_d1.abs_input)};
            d1_lut_value = OpFMul(f32_id, ConstF32(lighting.lut_d1.scale), value);
        }

        const Id refl_value{OpCompositeConstruct(vec_ids.Get(3), refl_value_r, refl_value_g, refl_value_b)};
        const Id light_specular_1{GetLightMember(1)};
        Id specular_1{OpFMul(vec_ids.Get(3), OpVectorTimesScalar(vec_ids.Get(3), refl_value, d1_lut_value), light_specular_1)};
        if (light_config.geometric_factor_1) {
            specular_1 = OpVectorTimesScalar(vec_ids.Get(3), specular_1, geo_factor);
        }

        // Fresnel
        // Note: only the last entry in the light slots applies the Fresnel factor
        if (light_index == lighting.src_num - 1 && lighting.lut_fr.enable &&
            LightingRegs::IsLightingSamplerSupported(lighting.config,
                                                     LightingRegs::LightingSampler::Fresnel)) {
            // Lookup fresnel LUT value
            Id value{GetLutValue(LightingRegs::LightingSampler::Fresnel, light_config.num,
                     lighting.lut_fr.type, lighting.lut_fr.abs_input)};
            value = OpFMul(f32_id, ConstF32(lighting.lut_fr.scale), value);

            // Enabled for diffuse lighting alpha component
            if (lighting.enable_primary_alpha) {
                diffuse_sum = OpCompositeInsert(vec_ids.Get(4), value, diffuse_sum, 3);
            }

            // Enabled for the specular lighting alpha component
            if (lighting.enable_secondary_alpha) {
                specular_sum = OpCompositeInsert(vec_ids.Get(4), value, specular_sum, 3);
            }
        }

        const bool shadow_primary_enable = lighting.shadow_primary && light_config.shadow_enable;
        const bool shadow_secondary_enable = lighting.shadow_secondary && light_config.shadow_enable;
        const Id shadow_rgb{OpVectorShuffle(vec_ids.Get(3), shadow, shadow, 0, 1, 2)};

        const Id light_diffuse{GetLightMember(2)};
        const Id light_ambient{GetLightMember(3)};
        const Id diffuse_mul_dot{OpVectorTimesScalar(vec_ids.Get(3),light_diffuse, dot_product)};

        // Compute primary fragment color (diffuse lighting) function
        Id diffuse_sum_rgb{OpFAdd(vec_ids.Get(3), diffuse_mul_dot, light_ambient)};
        diffuse_sum_rgb = OpVectorTimesScalar(vec_ids.Get(3), diffuse_sum_rgb, dist_atten);
        diffuse_sum_rgb = OpVectorTimesScalar(vec_ids.Get(3), diffuse_sum_rgb, spot_atten);
        if (shadow_primary_enable) {
            diffuse_sum_rgb = OpFMul(vec_ids.Get(3), diffuse_sum_rgb, shadow_rgb);
        }

        // Compute secondary fragment color (specular lighting) function
        const Id specular_01{OpFAdd(vec_ids.Get(3), specular_0, specular_1)};
        Id specular_sum_rgb{OpVectorTimesScalar(vec_ids.Get(3), specular_01, clamp_highlights)};
        specular_sum_rgb = OpVectorTimesScalar(vec_ids.Get(3), specular_sum_rgb, dist_atten);
        specular_sum_rgb = OpVectorTimesScalar(vec_ids.Get(3), specular_sum_rgb, spot_atten);
        if (shadow_secondary_enable) {
            specular_sum_rgb = OpFMul(vec_ids.Get(3), specular_sum_rgb, shadow_rgb);
        }

        // Accumulate the fragment colors
        const Id diffuse_sum_rgba{PadVectorF32(diffuse_sum_rgb, vec_ids.Get(4), 0.f)};
        const Id specular_sum_rgba{PadVectorF32(specular_sum_rgb, vec_ids.Get(4), 0.f)};
        diffuse_sum = OpFAdd(vec_ids.Get(4), diffuse_sum, diffuse_sum_rgba);
        specular_sum = OpFAdd(vec_ids.Get(4), specular_sum, specular_sum_rgba);
    }

    // Apply shadow attenuation to alpha components if enabled
    if (lighting.shadow_alpha) {
        const Id shadow_a{OpCompositeExtract(vec_ids.Get(4), shadow, 3)};
        const Id shadow_a_vec{OpCompositeConstruct(vec_ids.Get(4), ConstF32(1.f, 1.f, 1.f), shadow_a)};
        if (lighting.enable_primary_alpha) {
            diffuse_sum = OpFMul(vec_ids.Get(4), diffuse_sum, shadow_a_vec);
        }
        if (lighting.enable_secondary_alpha) {
            specular_sum = OpFMul(vec_ids.Get(4), specular_sum, shadow_a_vec);
        }
    }

    // Sum final lighting result
    const Id lighting_global_ambient{GetShaderDataMember(vec_ids.Get(3), ConstS32(24))};
    const Id lighting_global_ambient_rgba{PadVectorF32(lighting_global_ambient, vec_ids.Get(4), 0.f)};
    const Id zero_vec{ConstF32(0.f, 0.f, 0.f, 0.f)};
    const Id one_vec{ConstF32(1.f, 1.f, 1.f, 1.f)};
    diffuse_sum = OpFAdd(vec_ids.Get(4), diffuse_sum, lighting_global_ambient_rgba);
    primary_fragment_color = OpFClamp(vec_ids.Get(4), diffuse_sum, zero_vec, one_vec);
    secondary_fragment_color = OpFClamp(vec_ids.Get(4), specular_sum, zero_vec, one_vec);
}

void FragmentModule::WriteTevStage(s32 index) {
    const TexturingRegs::TevStageConfig stage =
        static_cast<const TexturingRegs::TevStageConfig>(config.state.tev_stages[index]);

    // Detects if a TEV stage is configured to be skipped (to avoid generating unnecessary code)
    const auto IsPassThroughTevStage = [](const TevStageConfig& stage) {
        return (stage.color_op == TevStageConfig::Operation::Replace &&
                stage.alpha_op == TevStageConfig::Operation::Replace &&
                stage.color_source1 == TevStageConfig::Source::Previous &&
                stage.alpha_source1 == TevStageConfig::Source::Previous &&
                stage.color_modifier1 == TevStageConfig::ColorModifier::SourceColor &&
                stage.alpha_modifier1 == TevStageConfig::AlphaModifier::SourceAlpha &&
                stage.GetColorMultiplier() == 1 && stage.GetAlphaMultiplier() == 1);
    };

    if (!IsPassThroughTevStage(stage)) {
        color_results_1 = AppendColorModifier(stage.color_modifier1, stage.color_source1, index);
        color_results_2 = AppendColorModifier(stage.color_modifier2, stage.color_source2, index);
        color_results_3 = AppendColorModifier(stage.color_modifier3, stage.color_source3, index);

        // Round the output of each TEV stage to maintain the PICA's 8 bits of precision
        Id color_output{Byteround(AppendColorCombiner(stage.color_op), 3)};
        Id alpha_output{};

        if (stage.color_op == TevStageConfig::Operation::Dot3_RGBA) {
            // result of Dot3_RGBA operation is also placed to the alpha component
            alpha_output = OpCompositeExtract(f32_id, color_output, 0);
        } else {
            alpha_results_1 = AppendAlphaModifier(stage.alpha_modifier1, stage.alpha_source1, index);
            alpha_results_2 = AppendAlphaModifier(stage.alpha_modifier2, stage.alpha_source2, index);
            alpha_results_3 = AppendAlphaModifier(stage.alpha_modifier3, stage.alpha_source3, index);

            alpha_output = Byteround(AppendAlphaCombiner(stage.alpha_op));
        }

        color_output = OpVectorTimesScalar(vec_ids.Get(3), color_output, ConstF32(static_cast<float>(stage.GetColorMultiplier())));
        color_output = OpFClamp(vec_ids.Get(3), color_output, ConstF32(0.f, 0.f, 0.f), ConstF32(1.f, 1.f, 1.f));
        alpha_output = OpFMul(f32_id, alpha_output, ConstF32(static_cast<float>(stage.GetAlphaMultiplier())));
        alpha_output = OpFClamp(f32_id, alpha_output, ConstF32(0.f), ConstF32(1.f));
        last_tex_env_out = OpCompositeConstruct(vec_ids.Get(4), color_output, alpha_output);
    }

    combiner_buffer = next_combiner_buffer;
    if (config.TevStageUpdatesCombinerBufferColor(index)) {
        next_combiner_buffer = OpVectorShuffle(vec_ids.Get(4), last_tex_env_out, next_combiner_buffer, 0, 1, 2, 7);
    }

    if (config.TevStageUpdatesCombinerBufferAlpha(index)) {
        next_combiner_buffer = OpVectorShuffle(vec_ids.Get(4), next_combiner_buffer, last_tex_env_out, 0, 1, 2, 7);
    }
}

Id FragmentModule::SampleTexture(u32 texture_unit) {
    const PicaFSConfigState& state = config.state;

    // PICA's LOD formula for 2D textures.
    // This LOD formula is the same as the LOD lower limit defined in OpenGL.
    // f(x, y) >= max{m_u, m_v, m_w}
    // (See OpenGL 4.6 spec, 8.14.1 - Scale Factor and Level-of-Detail)
    const auto SampleLod = [this](Id tex_id, Id tex_sampler_id, Id texcoord_id) {
        const Id tex{OpLoad(image2d_id, tex_id)};
        const Id tex_sampler{OpLoad(sampler_id, tex_sampler_id)};
        const Id sampled_image{OpSampledImage(TypeSampledImage(image2d_id), tex, tex_sampler)};
        const Id tex_image{OpImage(image2d_id, sampled_image)};
        const Id tex_size{OpImageQuerySizeLod(ivec_ids.Get(2), tex_image, ConstS32(0))};
        const Id texcoord{OpLoad(vec_ids.Get(2), texcoord_id)};
        const Id coord{OpFMul(vec_ids.Get(2), texcoord, OpConvertSToF(vec_ids.Get(2), tex_size))};
        const Id abs_dfdx_coord{OpFAbs(vec_ids.Get(2), OpDPdx(vec_ids.Get(2), coord))};
        const Id abs_dfdy_coord{OpFAbs(vec_ids.Get(2), OpDPdy(vec_ids.Get(2), coord))};
        const Id d{OpFMax(vec_ids.Get(2), abs_dfdx_coord, abs_dfdy_coord)};
        const Id dx_dy_max{OpFMax(f32_id, OpCompositeExtract(f32_id, d, 0), OpCompositeExtract(f32_id, d, 1))};
        const Id lod{OpLog2(f32_id, dx_dy_max)};
        return OpImageSampleExplicitLod(vec_ids.Get(4), sampled_image, texcoord, spv::ImageOperandsMask::Lod, lod);
    };

    const auto Sample = [this](Id tex_id, Id tex_sampler_id, bool projection) {
        const Id tex{OpLoad(image2d_id, tex_id)};
        const Id tex_sampler{OpLoad(sampler_id, tex_sampler_id)};
        const Id sampled_image{OpSampledImage(TypeSampledImage(image2d_id), tex, tex_sampler)};
        const Id texcoord0{OpLoad(vec_ids.Get(2), texcoord0_id)};
        const Id texcoord0_w{OpLoad(f32_id, texcoord0_w_id)};
        const Id coord{OpCompositeConstruct(vec_ids.Get(3), OpCompositeExtract(f32_id, texcoord0, 0),
                                                            OpCompositeExtract(f32_id, texcoord0, 1),
                                                            texcoord0_w)};
        if (projection) {
            return OpImageSampleProjImplicitLod(vec_ids.Get(4), sampled_image, coord);
        } else {
            return OpImageSampleImplicitLod(vec_ids.Get(4), sampled_image, coord);
        }
    };

    switch (texture_unit) {
    case 0:
        // Only unit 0 respects the texturing type
        switch (state.texture0_type) {
        case Pica::TexturingRegs::TextureConfig::Texture2D:
            return SampleLod(tex0_id, tex0_sampler_id, texcoord0_id);
        case Pica::TexturingRegs::TextureConfig::Projection2D:
            return Sample(tex0_id, tex0_sampler_id, true);
        case Pica::TexturingRegs::TextureConfig::TextureCube:
            return Sample(tex_cube_id, tex_cube_sampler_id, false);
        //case Pica::TexturingRegs::TextureConfig::Shadow2D:
            //return "shadowTexture(texcoord0, texcoord0_w)";
        //case Pica::TexturingRegs::TextureConfig::ShadowCube:
            //return "shadowTextureCube(texcoord0, texcoord0_w)";
        case Pica::TexturingRegs::TextureConfig::Disabled:
            return ConstF32(0.f, 0.f, 0.f, 0.f);
        default:
            LOG_CRITICAL(Render_Vulkan, "Unhandled texture type {:x}", state.texture0_type);
            UNIMPLEMENTED();
            return void_id;
        }
    case 1:
        return SampleLod(tex1_id, tex1_sampler_id, texcoord1_id);
    case 2:
        if (state.texture2_use_coord1)
            return SampleLod(tex2_id, tex2_sampler_id, texcoord1_id);
        else
            return SampleLod(tex2_id, tex2_sampler_id, texcoord2_id);
    case 3:
        if (false && state.proctex.enable) {
            //return "ProcTex()";
        } else {
            LOG_DEBUG(Render_Vulkan, "Using Texture3 without enabling it");
            return ConstF32(0.f, 0.f, 0.f, 0.f);
        }
    default:
        UNREACHABLE();
        return void_id;
    }
}

Id FragmentModule::Byteround(Id variable_id, u32 size) {
    if (size > 1) {
        const Id scaled_vec_id{OpVectorTimesScalar(vec_ids.Get(size), variable_id, ConstF32(255.f))};
        const Id rounded_id{OpRound(vec_ids.Get(size), scaled_vec_id)};
        return OpVectorTimesScalar(vec_ids.Get(size), rounded_id, ConstF32(1.f / 255.f));
    } else {
        const Id rounded_id{OpRound(f32_id, OpFMul(f32_id, variable_id, ConstF32(255.f)))};
        return OpFMul(f32_id, rounded_id, ConstF32(1.f / 255.f));
    }
}

Id FragmentModule::LookupLightingLUT(Id lut_index, Id index, Id delta) {
    // Only load the texture buffer lut once
    if (!Sirit::ValidId(texture_buffer_lut_lf)) {
        const Id sampled_image{TypeSampledImage(image_buffer_id)};
        texture_buffer_lut_lf = OpLoad(sampled_image, texture_buffer_lut_lf_id);
    }

    const Id lut_index_x{OpShiftRightArithmetic(i32_id, lut_index, ConstS32(2))};
    const Id lut_index_y{OpBitwiseAnd(i32_id, lut_index, ConstS32(3))};
    const Id lut_offset{GetShaderDataMember(i32_id, ConstS32(19), lut_index_x, lut_index_y)};
    const Id coord{OpIAdd(i32_id, lut_offset, index)};
    const Id entry{OpImageFetch(vec_ids.Get(4), OpImage(image_buffer_id, texture_buffer_lut_lf), coord)};
    const Id entry_r{OpCompositeExtract(f32_id, entry, 0)};
    const Id entry_g{OpCompositeExtract(f32_id, entry, 1)};
    return OpFma(f32_id, entry_g, delta, entry_r);
}

Id FragmentModule::AppendSource(TevStageConfig::Source source, s32 index) {
    using Source = TevStageConfig::Source;
    switch (source) {
    case Source::PrimaryColor:
        return rounded_primary_color;
    case Source::PrimaryFragmentColor:
        return primary_fragment_color;
    case Source::SecondaryFragmentColor:
        return secondary_fragment_color;
    case Source::Texture0:
        return SampleTexture(0);
    case Source::Texture1:
        return SampleTexture(1);
    case Source::Texture2:
        return SampleTexture(2);
    case Source::Texture3:
        return SampleTexture(3);
    case Source::PreviousBuffer:
        return combiner_buffer;
    case Source::Constant:
        return GetShaderDataMember(vec_ids.Get(4), ConstS32(26), ConstS32(index));
    case Source::Previous:
        return last_tex_env_out;
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown source op {}", source);
        return ConstF32(0.f, 0.f, 0.f, 0.f);
    }
}

Id FragmentModule::AppendColorModifier(TevStageConfig::ColorModifier modifier,
                                       TevStageConfig::Source source, s32 index) {
    using ColorModifier = TevStageConfig::ColorModifier;
    const Id source_color{AppendSource(source, index)};
    const Id one_vec{ConstF32(1.f, 1.f, 1.f)};

    const auto Shuffle = [&](s32 r, s32 g, s32 b) -> Id {
        return OpVectorShuffle(vec_ids.Get(3), source_color, source_color, r, g, b);
    };

    switch (modifier) {
    case ColorModifier::SourceColor:
        return Shuffle(0, 1, 2);
    case ColorModifier::OneMinusSourceColor:
        return OpFSub(vec_ids.Get(3), one_vec, Shuffle(0, 1, 2));
    case ColorModifier::SourceRed:
        return Shuffle(0, 0, 0);
    case ColorModifier::OneMinusSourceRed:
        return OpFSub(vec_ids.Get(3), one_vec, Shuffle(0, 0, 0));
    case ColorModifier::SourceGreen:
        return Shuffle(1, 1, 1);
    case ColorModifier::OneMinusSourceGreen:
        return OpFSub(vec_ids.Get(3), one_vec, Shuffle(1, 1, 1));
    case ColorModifier::SourceBlue:
        return Shuffle(2, 2, 2);
    case ColorModifier::OneMinusSourceBlue:
        return OpFSub(vec_ids.Get(3), one_vec, Shuffle(2, 2, 2));
    case ColorModifier::SourceAlpha:
        return Shuffle(3, 3, 3);
    case ColorModifier::OneMinusSourceAlpha:
        return OpFSub(vec_ids.Get(3), one_vec, Shuffle(3, 3, 3));
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown color modifier op {}", modifier);
        return one_vec;
    }
}

Id FragmentModule::AppendAlphaModifier(TevStageConfig::AlphaModifier modifier,
                                       TevStageConfig::Source source, s32 index) {
    using AlphaModifier = TevStageConfig::AlphaModifier;
    const Id source_color{AppendSource(source, index)};
    const Id one_f32{ConstF32(1.f)};

    const auto Component = [&](s32 c) -> Id {
        return OpCompositeExtract(f32_id, source_color, c);
    };

    switch (modifier) {
    case AlphaModifier::SourceAlpha:
        return Component(3);
    case AlphaModifier::OneMinusSourceAlpha:
        return OpFSub(f32_id, one_f32, Component(3));
    case AlphaModifier::SourceRed:
        return Component(0);
    case AlphaModifier::OneMinusSourceRed:
        return OpFSub(f32_id, one_f32, Component(0));
    case AlphaModifier::SourceGreen:
        return Component(1);
    case AlphaModifier::OneMinusSourceGreen:
        return OpFSub(f32_id, one_f32, Component(1));
    case AlphaModifier::SourceBlue:
        return Component(2);
    case AlphaModifier::OneMinusSourceBlue:
        return OpFSub(f32_id, one_f32, Component(2));
    default:
        LOG_CRITICAL(Render_Vulkan, "Unknown alpha modifier op {}", modifier);
        return one_f32;
    }
}

Id FragmentModule::AppendColorCombiner(Pica::TexturingRegs::TevStageConfig::Operation operation) {
    using Operation = TevStageConfig::Operation;
    const Id half_vec{ConstF32(0.5f, 0.5f, 0.5f)};
    const Id one_vec{ConstF32(1.f, 1.f, 1.f)};
    const Id zero_vec{ConstF32(0.f, 0.f, 0.f)};
    Id color{};

    switch (operation) {
    case Operation::Replace:
        color = color_results_1;
        break;
    case Operation::Modulate:
        color = OpFMul(vec_ids.Get(3), color_results_1, color_results_2);
        break;
    case Operation::Add:
        color = OpFAdd(vec_ids.Get(3), color_results_1, color_results_2);
        break;
    case Operation::AddSigned:
        color = OpFSub(vec_ids.Get(3), OpFAdd(vec_ids.Get(3), color_results_1, color_results_2), half_vec);
        break;
    case Operation::Lerp:
        color = OpFMix(vec_ids.Get(3), color_results_2, color_results_1, color_results_3);
        break;
    case Operation::Subtract:
        color = OpFSub(vec_ids.Get(3), color_results_1, color_results_2);
        break;
    case Operation::MultiplyThenAdd:
        color = OpFma(vec_ids.Get(3), color_results_1, color_results_2, color_results_3);
        break;
    case Operation::AddThenMultiply:
        color = OpFMin(vec_ids.Get(3), OpFAdd(vec_ids.Get(3), color_results_1, color_results_2), one_vec);
        color = OpFMul(vec_ids.Get(3), color, color_results_3);
        break;
    case Operation::Dot3_RGB:
    case Operation::Dot3_RGBA:
        color = OpDot(f32_id, OpFSub(vec_ids.Get(3), color_results_1, half_vec),
                              OpFSub(vec_ids.Get(3), color_results_2, half_vec));
        color = OpFMul(f32_id, color, ConstF32(4.f));
        color = OpCompositeConstruct(vec_ids.Get(3), color, color, color);
        break;
    default:
        color = zero_vec;
        LOG_CRITICAL(Render_Vulkan, "Unknown color combiner operation: {}", operation);
        break;
    }

    // Clamp result to 0.0, 1.0
    return OpFClamp(vec_ids.Get(3), color, zero_vec, one_vec);
}

Id FragmentModule::AppendAlphaCombiner(TevStageConfig::Operation operation) {
    using Operation = TevStageConfig::Operation;
    Id color{};

    switch (operation) {
    case Operation::Replace:
        color = alpha_results_1;
        break;
    case Operation::Modulate:
        color = OpFMul(f32_id, alpha_results_1, alpha_results_2);
        break;
    case Operation::Add:
        color = OpFAdd(f32_id, alpha_results_1, alpha_results_2);
        break;
    case Operation::AddSigned:
        color = OpFSub(f32_id, OpFAdd(f32_id, alpha_results_1, alpha_results_2), ConstF32(0.5f));
        break;
    case Operation::Lerp:
        color = OpFMix(f32_id, alpha_results_2, alpha_results_1, alpha_results_3);
        break;
    case Operation::Subtract:
        color = OpFSub(f32_id, alpha_results_1, alpha_results_2);
        break;
    case Operation::MultiplyThenAdd:
        color = OpFma(f32_id, alpha_results_1, alpha_results_2, alpha_results_3);
        break;
    case Operation::AddThenMultiply:
        color = OpFMin(f32_id, OpFAdd(f32_id, alpha_results_1, alpha_results_2), ConstF32(1.f));
        color = OpFMul(f32_id, color, alpha_results_3);
        break;
    default:
        color = ConstF32(0.f);
        LOG_CRITICAL(Render_Vulkan, "Unknown alpha combiner operation: {}", operation);
        break;
    }

    return OpFClamp(f32_id, color, ConstF32(0.f), ConstF32(1.f));
}

void FragmentModule::DefineArithmeticTypes() {
    void_id = Name(TypeVoid(), "void_id");
    bool_id = Name(TypeBool(), "bool_id");
    f32_id = Name(TypeFloat(32), "f32_id");
    i32_id = Name(TypeSInt(32), "i32_id");
    u32_id = Name(TypeUInt(32), "u32_id");

    for (u32 size = 2; size <= 4; size++) {
        const u32 i = size - 2;
        vec_ids.ids[i] = Name(TypeVector(f32_id, size), fmt::format("vec{}_id", size));
        ivec_ids.ids[i] = Name(TypeVector(i32_id, size), fmt::format("ivec{}_id", size));
        uvec_ids.ids[i] = Name(TypeVector(u32_id, size), fmt::format("uvec{}_id", size));
    }
}

void FragmentModule::DefineEntryPoint() {
    AddCapability(spv::Capability::Shader);
    AddCapability(spv::Capability::SampledBuffer);
    AddCapability(spv::Capability::ImageQuery);
    SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);

    const Id main_type{TypeFunction(TypeVoid())};
    const Id main_func{OpFunction(TypeVoid(), spv::FunctionControlMask::MaskNone, main_type)};
    AddEntryPoint(spv::ExecutionModel::Fragment, main_func, "main", primary_color_id, texcoord0_id,
                  texcoord1_id, texcoord2_id, texcoord0_w_id, normquat_id, view_id, color_id,
                  gl_frag_coord_id, gl_frag_depth_id);
    AddExecutionMode(main_func, spv::ExecutionMode::OriginUpperLeft);
    AddExecutionMode(main_func, spv::ExecutionMode::DepthReplacing);
}

void FragmentModule::DefineUniformStructs() {
    const Id light_src_struct_id{TypeStruct(vec_ids.Get(3), vec_ids.Get(3), vec_ids.Get(3), vec_ids.Get(3),
                                      vec_ids.Get(3), vec_ids.Get(3), f32_id, f32_id)};

    const Id light_src_array_id{TypeArray(light_src_struct_id, ConstU32(NUM_LIGHTS))};
    const Id lighting_lut_array_id{TypeArray(ivec_ids.Get(4), ConstU32(NUM_LIGHTING_SAMPLERS / 4))};
    const Id const_color_array_id{TypeArray(vec_ids.Get(4), ConstU32(NUM_TEV_STAGES))};

    const Id shader_data_struct_id{TypeStruct(i32_id, i32_id, f32_id, f32_id, f32_id, f32_id, i32_id,
                                              i32_id, i32_id, i32_id, i32_id, i32_id, i32_id, i32_id, i32_id,
                                              i32_id, f32_id, i32_id, u32_id, lighting_lut_array_id, vec_ids.Get(3),
                                              vec_ids.Get(2), vec_ids.Get(2), vec_ids.Get(2), vec_ids.Get(3),
                                              light_src_array_id, const_color_array_id, vec_ids.Get(4), vec_ids.Get(4))};

    constexpr std::array light_src_offsets{0u, 16u, 32u, 48u, 64u, 80u, 92u, 96u};
    constexpr std::array shader_data_offsets{0u, 4u, 8u, 12u, 16u, 20u, 24u, 28u, 32u, 36u, 40u, 44u, 48u,
                                             52u, 56u, 60u, 64u, 68u, 72u, 80u, 176u, 192u, 200u, 208u,
                                             224u, 240u, 1136u, 1232u, 1248u};

    Decorate(lighting_lut_array_id, spv::Decoration::ArrayStride, 16u);
    Decorate(light_src_array_id, spv::Decoration::ArrayStride, 112u);
    Decorate(const_color_array_id, spv::Decoration::ArrayStride, 16u);
    for (u32 i = 0; i < static_cast<u32>(light_src_offsets.size()); i++) {
        MemberDecorate(light_src_struct_id, i, spv::Decoration::Offset, light_src_offsets[i]);
    }
    for (u32 i = 0; i < static_cast<u32>(shader_data_offsets.size()); i++) {
        MemberDecorate(shader_data_struct_id, i, spv::Decoration::Offset, shader_data_offsets[i]);
    }
    Decorate(shader_data_struct_id, spv::Decoration::Block);

    shader_data_id = AddGlobalVariable(TypePointer(spv::StorageClass::Uniform, shader_data_struct_id),
                                       spv::StorageClass::Uniform);
    Decorate(shader_data_id, spv::Decoration::DescriptorSet, 0);
    Decorate(shader_data_id, spv::Decoration::Binding, 1);
}

void FragmentModule::DefineInterface() {
    // Define interface block
    primary_color_id = DefineInput(vec_ids.Get(4), 1);
    texcoord0_id = DefineInput(vec_ids.Get(2), 2);
    texcoord1_id = DefineInput(vec_ids.Get(2), 3);
    texcoord2_id = DefineInput(vec_ids.Get(2), 4);
    texcoord0_w_id = DefineInput(f32_id, 5);
    normquat_id = DefineInput(vec_ids.Get(4), 6);
    view_id = DefineInput(vec_ids.Get(3), 7);
    color_id = DefineOutput(vec_ids.Get(4), 0);

    // Define the texture unit samplers/uniforms
    image_buffer_id = TypeImage(f32_id, spv::Dim::Buffer, 0, 0, 0, 1, spv::ImageFormat::Unknown);
    image2d_id = TypeImage(f32_id, spv::Dim::Dim2D, 0, 0, 0, 1, spv::ImageFormat::Unknown);
    image_cube_id = TypeImage(f32_id, spv::Dim::Cube, 0, 0, 0, 1, spv::ImageFormat::Unknown);
    sampler_id = TypeSampler();

    texture_buffer_lut_lf_id = DefineUniformConst(TypeSampledImage(image_buffer_id), 0, 2);
    texture_buffer_lut_rg_id = DefineUniformConst(TypeSampledImage(image_buffer_id), 0, 3);
    texture_buffer_lut_rgba_id = DefineUniformConst(TypeSampledImage(image_buffer_id), 0, 4);
    tex0_id = DefineUniformConst(image2d_id, 1, 0);
    tex1_id = DefineUniformConst(image2d_id, 1, 1);
    tex2_id = DefineUniformConst(image2d_id, 1, 2);
    tex_cube_id = DefineUniformConst(image_cube_id, 1, 3);
    tex0_sampler_id = DefineUniformConst(sampler_id, 2, 0);
    tex1_sampler_id = DefineUniformConst(sampler_id, 2, 1);
    tex2_sampler_id = DefineUniformConst(sampler_id, 2, 2);
    tex_cube_sampler_id = DefineUniformConst(sampler_id, 2, 3);

    // Define built-ins
    gl_frag_coord_id = DefineVar(vec_ids.Get(4), spv::StorageClass::Input);
    gl_frag_depth_id = DefineVar(f32_id, spv::StorageClass::Output);
    Decorate(gl_frag_coord_id, spv::Decoration::BuiltIn, spv::BuiltIn::FragCoord);
    Decorate(gl_frag_depth_id, spv::Decoration::BuiltIn, spv::BuiltIn::FragDepth);
}

std::vector<u32> GenerateFragmentShaderSPV(const PicaFSConfig& config) {
    FragmentModule module{config};
    module.Generate();
    return module.Assemble();
}

} // namespace Vulkan
