// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <exception>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <boost/container/small_vector.hpp>
#include <fmt/format.h>
#include <nihstro/shader_bytecode.h>
#include "common/assert.h"
#include "common/file_util.h"
#include "video_core/renderer_vulkan/vk_shader_decompiler.h"

namespace Vulkan {

int i = 0;

using nihstro::Instruction;
using nihstro::OpCode;
using nihstro::RegisterType;
using nihstro::SourceRegister;
using nihstro::SwizzlePattern;

VertexModule::VertexModule(const Pica::Shader::ShaderSetup& setup,
                           const PicaVSConfig& config) : Sirit::Module{0x00010300},
    config{config}, program_code{setup.program_code}, swizzle_data{setup.swizzle_data},
    main_offset{config.state.main_offset}, sanitize_mul{config.state.sanitize_mul},
    subroutines{ControlFlowAnalyzer(program_code, main_offset).MoveSubroutines()} {
    DefineArithmeticTypes();
    DefineUniformStructs();
    DefineInterface();
}

VertexModule::~VertexModule() = default;

ControlFlowAnalyzer::ControlFlowAnalyzer(const Pica::Shader::ProgramCode& program_code, u32 main_offset)
    : program_code(program_code) {
    // Recursively finds all subroutines.
    const Subroutine& program_main = AddSubroutine(main_offset, PROGRAM_END);
    if (program_main.exit_method != ExitMethod::AlwaysEnd) {
        throw DecompileFail("Program does not always end");
    }
}

const Subroutine& ControlFlowAnalyzer::AddSubroutine(u32 begin, u32 end) {
    auto iter = subroutines.find(Subroutine{begin, end});
    if (iter != subroutines.end())
        return *iter;

    Subroutine subroutine{begin, end};
    subroutine.exit_method = Scan(begin, end, subroutine.labels);
    if (subroutine.exit_method == ExitMethod::Undetermined)
        throw DecompileFail("Recursive function detected");
    return *subroutines.insert(std::move(subroutine)).first;
}

ExitMethod ControlFlowAnalyzer::Scan(u32 begin, u32 end, std::set<u32>& labels) {
    auto [iter, inserted] =
        exit_method_map.emplace(std::make_pair(begin, end), ExitMethod::Undetermined);
    ExitMethod& exit_method = iter->second;
    if (!inserted)
        return exit_method;

    using nihstro::Instruction;
    using nihstro::OpCode;

    for (u32 offset = begin; offset != end && offset != PROGRAM_END; ++offset) {
        const Instruction instr = {program_code[offset]};
        switch (instr.opcode.Value()) {
        case OpCode::Id::END: {
            return exit_method = ExitMethod::AlwaysEnd;
        }
        case OpCode::Id::JMPC:
        case OpCode::Id::JMPU: {
            labels.insert(instr.flow_control.dest_offset);
            ExitMethod no_jmp = Scan(offset + 1, end, labels);
            ExitMethod jmp = Scan(instr.flow_control.dest_offset, end, labels);
            return exit_method = ParallelExit(no_jmp, jmp);
        }
        case OpCode::Id::CALL: {
            auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                       instr.flow_control.dest_offset +
                                           instr.flow_control.num_instructions);
            if (call.exit_method == ExitMethod::AlwaysEnd)
                return exit_method = ExitMethod::AlwaysEnd;
            ExitMethod after_call = Scan(offset + 1, end, labels);
            return exit_method = SeriesExit(call.exit_method, after_call);
        }
        case OpCode::Id::LOOP: {
            auto& loop = AddSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
            if (loop.exit_method == ExitMethod::AlwaysEnd)
                return exit_method = ExitMethod::AlwaysEnd;
            ExitMethod after_loop = Scan(instr.flow_control.dest_offset + 1, end, labels);
            return exit_method = SeriesExit(loop.exit_method, after_loop);
        }
        case OpCode::Id::CALLC:
        case OpCode::Id::CALLU: {
            auto& call = AddSubroutine(instr.flow_control.dest_offset,
                                       instr.flow_control.dest_offset +
                                           instr.flow_control.num_instructions);
            ExitMethod after_call = Scan(offset + 1, end, labels);
            return exit_method = SeriesExit(
                       ParallelExit(call.exit_method, ExitMethod::AlwaysReturn), after_call);
        }
        case OpCode::Id::IFU:
        case OpCode::Id::IFC: {
            auto& if_sub = AddSubroutine(offset + 1, instr.flow_control.dest_offset);
            ExitMethod else_method;
            if (instr.flow_control.num_instructions != 0) {
                auto& else_sub = AddSubroutine(instr.flow_control.dest_offset,
                                               instr.flow_control.dest_offset +
                                                   instr.flow_control.num_instructions);
                else_method = else_sub.exit_method;
            } else {
                else_method = ExitMethod::AlwaysReturn;
            }

            ExitMethod both = ParallelExit(if_sub.exit_method, else_method);
            if (both == ExitMethod::AlwaysEnd)
                return exit_method = ExitMethod::AlwaysEnd;
            ExitMethod after_call =
                Scan(instr.flow_control.dest_offset + instr.flow_control.num_instructions, end,
                     labels);
            return exit_method = SeriesExit(both, after_call);
        }
        default:
            break;
        }
    }
    return exit_method = ExitMethod::AlwaysReturn;
}

ExitMethod ControlFlowAnalyzer::SeriesExit(ExitMethod a, ExitMethod b) {
    // This should be handled before evaluating b.
    DEBUG_ASSERT(a != ExitMethod::AlwaysEnd);

    if (a == ExitMethod::Undetermined) {
        return ExitMethod::Undetermined;
    }

    if (a == ExitMethod::AlwaysReturn) {
        return b;
    }

    if (b == ExitMethod::Undetermined || b == ExitMethod::AlwaysEnd) {
        return ExitMethod::AlwaysEnd;
    }

    return ExitMethod::Conditional;
}

ExitMethod ControlFlowAnalyzer::ParallelExit(ExitMethod a, ExitMethod b) {
    if (a == ExitMethod::Undetermined) {
        return b;
    }
    if (b == ExitMethod::Undetermined) {
        return a;
    }
    if (a == b) {
        return a;
    }
    return ExitMethod::Conditional;
}

/// An adaptor for getting swizzle pattern string from nihstro interfaces.
template <SwizzlePattern::Selector (SwizzlePattern::*getter)(int) const>
Id GetSelectorSrc(VertexModule& m, const Id vector, const SwizzlePattern& pattern) {
    bool identity = true;
    std::array<Sirit::Literal, 4> components;
    for (u32 i = 0; i < 4; ++i) {
        const SwizzlePattern::Selector selector = (pattern.*getter)(i);
        const u32 index = static_cast<u32>(selector);
        identity &= (i == index);
        components[i] = index;
    }

    if (identity) {
        return vector;
    }
    return m.OpVectorShuffle(m.vec_ids.Get(4), vector, vector, components);
}

constexpr auto GetSelectorSrc1 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc1>;
constexpr auto GetSelectorSrc2 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc2>;
constexpr auto GetSelectorSrc3 = GetSelectorSrc<&SwizzlePattern::GetSelectorSrc3>;

const Subroutine& VertexModule::GetSubroutine(u32 begin, u32 end) const {
    auto iter = subroutines.find(Subroutine{begin, end});
    ASSERT(iter != subroutines.end());
    return *iter;
}

Id VertexModule::EvaluateCondition(Instruction::FlowControlType flow_control) {
    using Op = Instruction::FlowControlType::Op;

    const Id cond_code{OpLoad(bvec_ids.Get(2), conditional_code)};
    const Id cond_x{OpCompositeExtract(bool_id, cond_code, 0)};
    const Id cond_y{OpCompositeExtract(bool_id, cond_code, 1)};

    const Id result_x =
        flow_control.refx.Value() ? cond_x : OpLogicalNot(bool_id, cond_x);;
    const Id result_y =
        flow_control.refy.Value() ? cond_y : OpLogicalNot(bool_id, cond_y);

    const auto Condition = [&]() -> Id {
        if (flow_control.refx.Value() && flow_control.refy.Value()) {
            return cond_code;
        } else if (!flow_control.refx.Value() && !flow_control.refy.Value()) {
            return OpLogicalNot(bvec_ids.Get(2), cond_code);
        } else {
            return OpCompositeConstruct(bvec_ids.Get(2), result_x, result_y);
        }
    };

    switch (flow_control.op) {
    case Op::JustX:
        return result_x;
    case Op::JustY:
        return result_y;
    case Op::Or:
        return OpAny(bool_id, Condition());
    case Op::And:
        return OpAll(bool_id, Condition());
    default:
        UNREACHABLE();
        return Id{};
    }
}

Id VertexModule::GetSourceRegister(const SourceRegister& source_reg, u32 address_register_index) {
    const u32 index = static_cast<u32>(source_reg.GetIndex());

    switch (source_reg.GetRegisterType()) {
    case RegisterType::Input: {
        if (!used_regs[index]) {
            const Id type{AttribType(index)};
            const Id vs_in_typed_reg = DefineInput(type, index);
            const Id typed_reg{OpLoad(type, vs_in_typed_reg)};
            input_typed_regs[index] = vs_in_typed_reg;
            input_regs[index] = AttribCast(index, typed_reg);
            used_regs[index] = true;
        }
        return input_regs[index];
    }
    case RegisterType::Temporary: {
        return OpLoad(vec_ids.Get(4), tmp_regs[index]);
    }
    case RegisterType::FloatUniform: {
        Id uniform_index{ConstU32(index)};
        if (address_register_index != 0) {
            const Id private_ptr{TypePointer(spv::StorageClass::Private, i32_id)};
            const Id component{ConstU32(address_register_index - 1)};
            const Id offset{OpLoad(i32_id, OpAccessChain(private_ptr,
                                                         address_registers,
                                                         component))};
            uniform_index = OpIAdd(i32_id, uniform_index, offset);
        }
        return GetVsUniformMember(vec_ids.Get(4), ConstS32(2), uniform_index);
    }
    default:
        UNREACHABLE();
    }
    return Id{};
}

Id VertexModule::GetDestRegister(const DestRegister& dest_reg) {
    const u32 index = static_cast<u32>(dest_reg.GetIndex());

    switch (dest_reg.GetRegisterType()) {
    case RegisterType::Temporary:
        return tmp_regs[index];
    case RegisterType::Output:
        if (config.state.output_map[index] < config.state.num_outputs) {
            return output_regs[index];
        }
        break;
    default:
        UNREACHABLE();
    }
    return Id{};
}

Id VertexModule::GetDestPointer(const DestRegister& dest_reg) {
    switch (dest_reg.GetRegisterType()) {
    case RegisterType::Temporary:
        return TypePointer(spv::StorageClass::Private, f32_id);
    case RegisterType::Output:
        return TypePointer(spv::StorageClass::Output, f32_id);
    default:
        UNREACHABLE();
    }
    return Id{};
}

void VertexModule::CallSubroutine(const Subroutine& subroutine) {
    if (subroutine.exit_method == ExitMethod::AlwaysEnd) {
        OpFunctionCall(bool_id, subroutine.function);
        OpReturnValue(ConstBool(true));
        OpFunctionEnd();
    } else if (subroutine.exit_method == ExitMethod::Conditional) {
        ASSERT_MSG(false, "Conditional exit method not implemented");
        //shader.AddLine("if ({}()) {{ return true; }}", subroutine.GetName());
    } else {
        OpFunctionCall(bool_id, subroutine.function);
    }
}

void VertexModule::SetDest(const nihstro::SwizzlePattern& swizzle, Id dest, Id value,
                           Id reg_pointer, u32 dest_num_components, u32 value_num_components) {
    u32 dest_mask_num_components = 0;
    std::array<u32, 4> dest_mask_swizzle;

    for (u32 i = 0; i < dest_num_components; ++i) {
        if (swizzle.DestComponentEnabled(static_cast<int>(i))) {
            dest_mask_swizzle[dest_mask_num_components++] = i;
        }
    }

    if (!Sirit::ValidId(dest) || dest_mask_num_components == 0) {
        return;
    }
    DEBUG_ASSERT(value_num_components >= dest_num_components || value_num_components == 1);

    Id src{value};
    if (value_num_components == 1) {
        if (dest_mask_num_components == 4) {
            src = OpCompositeConstruct(vec_ids.Get(4), src, src, src, src);
            OpStore(dest, src);
        } else {
            for (u32 i = 0; i < dest_mask_num_components; i++) {
                const u32 comp = dest_mask_swizzle[i];
                const Id pointer{OpAccessChain(reg_pointer, dest, ConstU32(comp))};
                OpStore(pointer, src);
            }
        }
    } else {
        if (dest_mask_num_components == 4) {
            OpStore(dest, src);
        } else {
            for (u32 i = 0; i < dest_mask_num_components; i++) {
                const u32 comp = dest_mask_swizzle[i];
                const Id pointer{OpAccessChain(reg_pointer, dest, ConstU32(comp))};
                const Id result_type{dest.value == address_registers.value ? i32_id : f32_id};
                const Id val{OpCompositeExtract(result_type, src, comp)};
                OpStore(pointer, val);
            }
        }
    }
}

Id VertexModule::SanitizeMul(Id lhs, Id rhs) {
    const Id product{OpFMul(vec_ids.Get(4), lhs, rhs)};
    const Id zero_vec{ConstF32(0.f, 0.f, 0.f, 0.f)};
    const Id product_nan{OpIsNan(bvec_ids.Get(4), product)};

#ifdef ANDROID
    // Use a cheaper sanitize_mul on Android, as mobile GPUs struggle here
    // This seems to be sufficient at least for Ocarina of Time and Attack on Titan accurate
    // multiplication bugs
    return OpSelect(vec_ids.Get(4), product_nan, zero_vec, product);
#else
    const Id rhs_nan{OpIsNan(bvec_ids.Get(4), rhs)};
    const Id lhs_nan{OpIsNan(bvec_ids.Get(4), lhs)};
    return OpSelect(vec_ids.Get(4), product_nan,
                    OpSelect(vec_ids.Get(4), lhs_nan, product,
                    OpSelect(vec_ids.Get(4), rhs_nan, product, zero_vec)),
                    product);
#endif
}

SpirvParams params;

u32 VertexModule::CompileInstr(u32 offset) {
    const Instruction instr = {program_code[offset]};

    std::size_t swizzle_offset =
        instr.opcode.Value().GetInfo().type == OpCode::Type::MultiplyAdd
            ? instr.mad.operand_desc_id
            : instr.common.operand_desc_id;
    const SwizzlePattern swizzle = {swizzle_data[swizzle_offset]};

    //shader.AddLine("// {}: {}", offset, instr.opcode.Value().GetInfo().name);

    switch (instr.opcode.Value().GetInfo().type) {
    case OpCode::Type::Arithmetic: {
        const bool is_inverted =
            (0 != (instr.opcode.Value().GetInfo().subtype & OpCode::Info::SrcInversed));

        Id src1{GetSourceRegister(instr.common.GetSrc1(is_inverted),
                                  !is_inverted * instr.common.address_register_index)};
        if (swizzle.negate_src1) {
            src1 = OpFNegate(vec_ids.Get(4), src1);
        }
        src1 = GetSelectorSrc1(*this, src1, swizzle);

        Id src2{GetSourceRegister(instr.common.GetSrc2(is_inverted),
                                  is_inverted * instr.common.address_register_index)};
        if (swizzle.negate_src2) {
            src2 = OpFNegate(vec_ids.Get(4), src2);
        }
        src2 = GetSelectorSrc2(*this, src2, swizzle);

        const Id dest_reg{GetDestRegister(instr.common.dest.Value())};
        const Id reg_pointer{GetDestPointer(instr.common.dest.Value())};

        switch (instr.opcode.Value().EffectiveOpCode()) {
        case OpCode::Id::ADD: {
            SetDest(swizzle, dest_reg, OpFAdd(vec_ids.Get(4), src1, src2), reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::MUL: {
            Id product{};
            if (sanitize_mul) {
                product = SanitizeMul(src1, src2);
            } else {
                product = OpFMul(vec_ids.Get(4), src1, src2);
            }

            SetDest(swizzle, dest_reg, product, reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::FLR: {
            SetDest(swizzle, dest_reg, OpFloor(vec_ids.Get(4), src1), reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::MAX: {
            SetDest(swizzle, dest_reg, OpFMax(vec_ids.Get(4), src1, src2), reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::MIN: {
            SetDest(swizzle, dest_reg, OpFMin(vec_ids.Get(4), src1, src2), reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::DP3:
        case OpCode::Id::DP4:
        case OpCode::Id::DPH:
        case OpCode::Id::DPHI: {
            OpCode::Id opcode = instr.opcode.Value().EffectiveOpCode();
            Id dot{};
            if (opcode == OpCode::Id::DP3) {
                if (sanitize_mul) {
                    const Id product{SanitizeMul(src1, src2)};
                    const Id product_xyz{OpVectorShuffle(vec_ids.Get(3), product, product, 0, 1, 2)};
                    dot = OpDot(f32_id, product_xyz, ConstF32(1.f, 1.f, 1.f));
                } else {
                    const Id src1_xyz{OpVectorShuffle(vec_ids.Get(3), src1, src1, 0, 1, 2)};
                    const Id src2_xyz{OpVectorShuffle(vec_ids.Get(3), src2, src2, 0, 1, 2)};
                    dot = OpDot(f32_id, src1_xyz, src2_xyz);
                }
            } else {
                if (sanitize_mul) {
                    const Id src1_ =
                        (opcode == OpCode::Id::DPH || opcode == OpCode::Id::DPHI)
                            ? OpCompositeInsert(vec_ids.Get(4), ConstF32(1.f), src1, 3)
                            : src1;

                    dot = OpDot(f32_id, SanitizeMul(src1_, src2), ConstF32(1.f, 1.f, 1.f, 1.f));
                } else {
                    dot = OpDot(f32_id, src1, src2);
                }
            }

            SetDest(swizzle, dest_reg, dot, reg_pointer, 4, 1);
            break;
        }

        case OpCode::Id::RCP: {
            //if (!sanitize_mul) {
                // When accurate multiplication is OFF, NaN are not really handled. This is a
                // workaround to cheaply avoid NaN. Fixes graphical issues in Ocarina of Time.
                //shader.AddLine("if ({}.x != 0.0)", src1);
            //}
            const Id src1_x{OpCompositeExtract(f32_id, src1, 0)};
            const Id rcp{OpFDiv(f32_id, ConstF32(1.f), src1_x)};
            SetDest(swizzle, dest_reg, rcp, reg_pointer, 4, 1);
            break;
        }

        case OpCode::Id::RSQ: {
            //if (!sanitize_mul) {
                // When accurate multiplication is OFF, NaN are not really handled. This is a
                // workaround to cheaply avoid NaN. Fixes graphical issues in Ocarina of Time.
                //shader.AddLine("if ({}.x > 0.0)", src1);
            //}
            const Id src1_x{OpCompositeExtract(f32_id, src1, 0)};
            const Id rsq{OpInverseSqrt(f32_id, src1_x)};
            SetDest(swizzle, dest_reg, rsq, reg_pointer, 4, 1);
            break;
        }

        case OpCode::Id::MOVA: {
            const Id src1i{OpConvertFToS(ivec_ids.Get(4), src1)};
            const Id src1i_xy{OpVectorShuffle(ivec_ids.Get(2), src1i, src1i, 0, 1)};
            SetDest(swizzle, address_registers, src1i_xy,
                    TypePointer(spv::StorageClass::Private, i32_id), 2, 2);
            break;
        }

        case OpCode::Id::MOV: {
            SetDest(swizzle, dest_reg, src1, reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::SGE:
        case OpCode::Id::SGEI: {
            const Id one_vec{ConstF32(1.f, 1.f, 1.f, 1.f)};
            const Id zero_vec{ConstF32(0.f, 0.f, 0.f, 0.f)};
            const Id geq{OpFOrdGreaterThanEqual(bvec_ids.Get(4), src1, src2)};
            const Id geqf{OpSelect(vec_ids.Get(4), geq, one_vec, zero_vec)};
            SetDest(swizzle, dest_reg, geqf, reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::SLT:
        case OpCode::Id::SLTI: {
            const Id one_vec{ConstF32(1.f, 1.f, 1.f, 1.f)};
            const Id zero_vec{ConstF32(0.f, 0.f, 0.f, 0.f)};
            const Id le{OpFOrdLessThan(bvec_ids.Get(4), src1, src2)};
            const Id lef{OpSelect(vec_ids.Get(4), le, one_vec, zero_vec)};
            SetDest(swizzle, dest_reg, lef, reg_pointer, 4, 4);
            break;
        }

        case OpCode::Id::CMP: {
            using CompareOp = Instruction::Common::CompareOpType::Op;
            const auto Compare = [&](CompareOp op, Id type, Id lhs, Id rhs) -> Id {
                switch (op) {
                case CompareOp::Equal:
                    return OpFOrdEqual(type, lhs, rhs);
                case CompareOp::NotEqual:
                    return OpFOrdNotEqual(type, lhs, rhs);
                case CompareOp::LessThan:
                    return OpFOrdLessThan(type, lhs, rhs);
                case CompareOp::LessEqual:
                    return OpFOrdLessThanEqual(type, lhs, rhs);
                case CompareOp::GreaterThan:
                    return OpFOrdGreaterThan(type, lhs, rhs);
                case CompareOp::GreaterEqual:
                    return OpFOrdGreaterThanEqual(type, lhs, rhs);
                default:
                    LOG_ERROR(HW_GPU, "Unknown compare mode {:x}", op);
                }
                return Id{};
            };

            const CompareOp op_x = instr.common.compare_op.x.Value();
            const CompareOp op_y = instr.common.compare_op.y.Value();

            if (op_x != op_y) {
                const Id src1_x{OpCompositeExtract(f32_id, src1, 0)};
                const Id src2_x{OpCompositeExtract(f32_id, src2, 0)};
                const Id cond_code_x{Compare(op_x, bool_id, src1_x, src2_x)};

                const Id src1_y{OpCompositeExtract(f32_id, src1, 1)};
                const Id src2_y{OpCompositeExtract(f32_id, src2, 1)};
                const Id cond_code_y{Compare(op_y, bool_id, src1_y, src2_y)};

                const Id cond_code{OpCompositeConstruct(bvec_ids.Get(2), cond_code_x, cond_code_y)};
                OpStore(conditional_code, cond_code);
            } else {
                const Id src1_xy{OpVectorShuffle(vec_ids.Get(2), src1, src1, 0, 1)};
                const Id src2_xy{OpVectorShuffle(vec_ids.Get(2), src2, src2, 0, 1)};
                const Id cond_code{Compare(op_x, bvec_ids.Get(2), src1_xy, src2_xy)};
                OpStore(conditional_code, cond_code);
            }
            break;
        }

        case OpCode::Id::EX2: {
            const Id src1_x{OpCompositeExtract(f32_id, src1, 0)};
            const Id exp2{OpExp2(f32_id, src1_x)};
            SetDest(swizzle, dest_reg, exp2, reg_pointer, 4, 1);
            break;
        }

        case OpCode::Id::LG2: {
            const Id src1_x{OpCompositeExtract(f32_id, src1, 0)};
            const Id log2{OpLog2(f32_id, src1_x)};
            SetDest(swizzle, dest_reg, log2, reg_pointer, 4, 1);
            break;
        }

        default: {
            LOG_ERROR(HW_GPU, "Unhandled arithmetic instruction: 0x{:02x} ({}): 0x{:08x}",
                      (int)instr.opcode.Value().EffectiveOpCode(),
                      instr.opcode.Value().GetInfo().name, instr.hex);
            throw DecompileFail("Unhandled instruction");
            break;
        }
        }

        break;
    }

    case OpCode::Type::MultiplyAdd: {
        if ((instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MAD) ||
            (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI)) {
            bool is_inverted = (instr.opcode.Value().EffectiveOpCode() == OpCode::Id::MADI);

            Id src1{GetSourceRegister(instr.mad.GetSrc1(is_inverted), 0)};
            if (swizzle.negate_src1) {
                src1 = OpFNegate(vec_ids.Get(4), src1);
            }
            src1 = GetSelectorSrc1(*this, src1, swizzle);

            Id src2{GetSourceRegister(instr.mad.GetSrc2(is_inverted),
                                      !is_inverted * instr.mad.address_register_index)};
            if (swizzle.negate_src2) {
                src2 = OpFNegate(vec_ids.Get(4), src2);
            }
            src2 = GetSelectorSrc2(*this, src2, swizzle);

            Id src3{GetSourceRegister(instr.mad.GetSrc3(is_inverted),
                                      is_inverted * instr.mad.address_register_index)};
            if (swizzle.negate_src3) {
                src3 = OpFNegate(vec_ids.Get(4), src3);
            }
            src3 = GetSelectorSrc3(*this, src3, swizzle);

            Id dest_reg =
                (instr.mad.dest.Value() < 0x10)
                    ? output_regs[instr.mad.dest.Value().GetIndex()]
                : (instr.mad.dest.Value() < 0x20)
                    ? tmp_regs[instr.mad.dest.Value().GetIndex()]
                    : Id{};
            Id reg_pointer =
                (instr.mad.dest.Value() < 0x10)
                    ? TypePointer(spv::StorageClass::Output, f32_id)
                : (instr.mad.dest.Value() < 0x20)
                    ? TypePointer(spv::StorageClass::Private, f32_id)
                    : Id{};

            if (sanitize_mul) {
                const Id src12{SanitizeMul(src1, src2)};
                const Id result{OpFAdd(vec_ids.Get(4), src12, src3)};
                SetDest(swizzle, dest_reg, result, reg_pointer, 4, 4);
            } else {
                const Id result{OpFma(vec_ids.Get(4), src1, src2, src3)};
                SetDest(swizzle, dest_reg, result, reg_pointer, 4, 4);
            }
        } else {
            LOG_ERROR(HW_GPU, "Unhandled multiply-add instruction: 0x{:02x} ({}): 0x{:08x}",
                      (int)instr.opcode.Value().EffectiveOpCode(),
                      instr.opcode.Value().GetInfo().name, instr.hex);
            throw DecompileFail("Unhandled instruction");
        }
        break;
    }

    default: {
        switch (instr.opcode.Value()) {
        case OpCode::Id::END: {
            OpReturnValue(ConstBool(true));
            offset = PROGRAM_END - 1;
            break;
        }

        case OpCode::Id::JMPC:
        case OpCode::Id::JMPU: {
            Id condition{};
            if (instr.opcode.Value() == OpCode::Id::JMPC) {
                condition = EvaluateCondition(instr.flow_control);
            } else {
                const bool invert_test = instr.flow_control.num_instructions & 1;
                condition = GetUniformBool(instr.flow_control.bool_uniform_id);
                if (invert_test) {
                    condition = OpLogicalNot(bool_id, condition);
                }
            }

            const Id merge_block{OpLabel()};
            const Id true_label{OpLabel()};
            OpSelectionMerge(merge_block, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(condition, true_label, merge_block);

            AddLabel(true_label);
            OpStore(params.jmp_to, ConstU32(instr.flow_control.dest_offset.Value()));
            OpBranch(params.switch_merge_block);

            AddLabel(merge_block);
            break;
        }

        case OpCode::Id::CALL:
        case OpCode::Id::CALLC:
        case OpCode::Id::CALLU: {
            Id condition{};
            if (instr.opcode.Value() == OpCode::Id::CALLC) {
                condition = EvaluateCondition(instr.flow_control);
            } else if (instr.opcode.Value() == OpCode::Id::CALLU) {
                condition = GetUniformBool(instr.flow_control.bool_uniform_id);
            }

            auto& call_sub = GetSubroutine(instr.flow_control.dest_offset,
                                           instr.flow_control.dest_offset +
                                           instr.flow_control.num_instructions);

            if (!Sirit::ValidId(condition)) {
                CallSubroutine(call_sub);
            } else {
                const Id true_label{OpLabel()};
                const Id false_label{OpLabel()};

                OpSelectionMerge(false_label, spv::SelectionControlMask::MaskNone);
                OpBranchConditional(condition, true_label, false_label);

                AddLabel(true_label);
                CallSubroutine(call_sub);
                AddLabel(false_label);
            }

            if (instr.opcode.Value() == OpCode::Id::CALL &&
                call_sub.exit_method == ExitMethod::AlwaysEnd) {
                offset = PROGRAM_END - 1;
            }

            break;
        }

        case OpCode::Id::NOP: {
            break;
        }

        case OpCode::Id::IFC:
        case OpCode::Id::IFU: {
            Id condition{};
            if (instr.opcode.Value() == OpCode::Id::IFC) {
                condition = EvaluateCondition(instr.flow_control);
            } else {
                condition = GetUniformBool(instr.flow_control.bool_uniform_id);
            }

            const u32 if_offset = offset + 1;
            const u32 else_offset = instr.flow_control.dest_offset;
            const u32 endif_offset =
                instr.flow_control.dest_offset + instr.flow_control.num_instructions;

            const Id merge_block{OpLabel()};
            const Id true_label{OpLabel()};
            const Id false_label{OpLabel()};

            OpSelectionMerge(merge_block, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(condition, true_label, false_label);

            AddLabel(true_label);

            auto& if_sub = GetSubroutine(if_offset, else_offset);
            CallSubroutine(if_sub);
            offset = else_offset - 1;

            OpBranch(merge_block);
            AddLabel(false_label);
            if (instr.flow_control.num_instructions != 0) {
                auto& else_sub = GetSubroutine(else_offset, endif_offset);
                CallSubroutine(else_sub);
                offset = endif_offset - 1;

                if (if_sub.exit_method == ExitMethod::AlwaysEnd &&
                    else_sub.exit_method == ExitMethod::AlwaysEnd) {
                    offset = PROGRAM_END - 1;
                }
            }

            OpBranch(merge_block);
            AddLabel(merge_block);
            break;
        }

        case OpCode::Id::LOOP: {
            const Id int_uniform{GetVsUniformMember(uvec_ids.Get(4), ConstS32(1),
                                 ConstS32(static_cast<s32>(instr.flow_control.int_uniform_id.Value())))};
            const Id int_x{OpCompositeExtract(u32_id, int_uniform, 0)};
            const Id int_y{OpBitcast(i32_id, OpCompositeExtract(u32_id, int_uniform, 1))};
            const Id int_z{OpBitcast(i32_id, OpCompositeExtract(u32_id, int_uniform, 2))};
            const Id loop_id{params.vars[params.used_vars++]};

            const Id addr_regs_pointer{TypePointer(spv::StorageClass::Private, i32_id)};
            const Id addr_regs_z_id{OpAccessChain(addr_regs_pointer, address_registers, ConstU32(2u))};
            OpStore(addr_regs_z_id, int_y);
            OpStore(loop_id, ConstU32(0u));

            const Id for_loop_label{OpLabel()};
            const Id merge_block{OpLabel()};
            const Id continue_target{OpLabel()};
            const Id label{OpLabel()};

            OpBranch(for_loop_label);
            AddLabel(for_loop_label);
            OpLoopMerge(merge_block, continue_target, spv::LoopControlMask::MaskNone);
            OpBranch(label);
            AddLabel(label);

            const Id loop{OpLoad(u32_id, loop_id)};
            const Id condition{OpULessThanEqual(bool_id, loop, int_x)};
            const Id true_label{OpLabel()};
            OpBranchConditional(condition, true_label, merge_block);

            AddLabel(true_label);

            auto& loop_sub = GetSubroutine(offset + 1, instr.flow_control.dest_offset + 1);
            CallSubroutine(loop_sub);
            OpBranch(continue_target);

            AddLabel(continue_target);
            const Id addr_regs_z{OpLoad(i32_id, addr_regs_z_id)};
            const Id addr_regs_z_inc{OpIAdd(i32_id, addr_regs_z, int_z)};
            OpStore(addr_regs_z_id, addr_regs_z_inc);
            OpBranch(for_loop_label);

            AddLabel(merge_block);

            offset = instr.flow_control.dest_offset;
            if (loop_sub.exit_method == ExitMethod::AlwaysEnd) {
                offset = PROGRAM_END - 1;
            }

            break;
        }

        case OpCode::Id::EMIT:
        case OpCode::Id::SETEMIT:
            LOG_ERROR(HW_GPU, "Geometry shader operation detected in vertex shader");
            break;

        default: {
            LOG_ERROR(HW_GPU, "Unhandled instruction: 0x{:02x} ({}): 0x{:08x}",
                      (int)instr.opcode.Value().EffectiveOpCode(),
                      instr.opcode.Value().GetInfo().name, instr.hex);
            throw DecompileFail("Unhandled instruction");
            break;
        }
        }

        break;
    }
    }
    return offset + 1;
}

u32 VertexModule::CompileRange(u32 begin, u32 end) {
    u32 program_counter;
    for (program_counter = begin; program_counter < (begin > end ? PROGRAM_END : end);) {
        program_counter = CompileInstr(program_counter);
    }
    return program_counter;
}

void VertexModule::Generate() {
    // Add declarations for all subroutines
    for (const Subroutine& subroutine : subroutines) {
        subroutine.function = OpFunction();
    }

    // Add definitions for all subroutines
    const Id func_type{TypeFunction(bool_id)};
    for (const Subroutine& subroutine : subroutines) {
        const Id function{subroutine.function};
        AddFunction(bool_id, function, spv::FunctionControlMask::MaskNone, func_type);
        AddLabel(OpLabel());

        // Define a list of variables that can be used for LOOP
        for (Id& var : params.vars) {
            var = DefineVar<false>(u32_id, spv::StorageClass::Function);
        }

        std::set<u32> labels = subroutine.labels;
        if (labels.empty()) {
            if (CompileRange(subroutine.begin, subroutine.end) != PROGRAM_END) {
                OpReturnValue(ConstantFalse(bool_id));
                OpFunctionEnd();
            }
        } else {
            labels.insert(subroutine.begin);

            const Id jmp_to_id{DefineVar<false>(u32_id, spv::StorageClass::Function)};
            OpStore(jmp_to_id, ConstU32(subroutine.begin));

            const Id while_label{OpLabel()};
            const Id while_merge_block{OpLabel()};
            const Id while_continue_block{OpLabel()};
            const Id switch_label{OpLabel()};
            const Id switch_merge_block{OpLabel()};

            const Id jmp_to{OpLoad(u32_id, jmp_to_id)};
            const Id default_label{OpLabel()};

            // Define the while loop header
            OpBranch(while_label);
            AddLabel(while_label);
            OpLoopMerge(while_merge_block, while_continue_block, spv::LoopControlMask::MaskNone);
            OpBranch(switch_label);

            // Define the switch statement header
            AddLabel(switch_label);
            OpSelectionMerge(switch_merge_block, spv::SelectionControlMask::MaskNone);

            // Generate spirv labels for all switch targets
            boost::container::small_vector<Sirit::Literal, 8> spv_literals;
            boost::container::small_vector<Id, 8> spv_labels;
            for (u32 label : labels) {
                spv_labels.push_back(OpLabel());
                spv_literals.push_back(label);
            }

            OpSwitch(jmp_to, default_label, spv_literals, spv_labels);

            params = SpirvParams{
                .jmp_to = jmp_to_id,
                .while_label = while_label,
                .switch_label = switch_label,
                .switch_merge_block = switch_merge_block,
            };

            for (auto it = labels.begin(); it != labels.end(); it++) {
                u32 label = *it;
                u32 index = std::distance(labels.begin(), it);
                AddLabel(spv_labels[index]);

                auto next_it = labels.lower_bound(label + 1);
                u32 next_label = next_it == labels.end() ? subroutine.end : *next_it;

                u32 compile_end = CompileRange(label, next_label);
                if (compile_end > next_label && compile_end != PROGRAM_END) {
                    ASSERT_MSG(false, "Unimplemented jump label stuff");
                    // This happens only when there is a label inside a IF/LOOP block
                    //OpStore(jmp_to_id, ConstU32(compile_end));
                    //OpBranch(switch_merge_block);
                    //labels.emplace(compile_end);
                }

                Id next_spv_label{};
                if (next_label == subroutine.end) {
                    next_spv_label = default_label;
                } else {
                    u32 next_index = std::distance(labels.begin(), next_it);
                    next_spv_label = spv_labels[next_index];
                }

                if (compile_end != PROGRAM_END) {
                    OpBranch(next_spv_label);
                }
            }

            AddLabel(switch_merge_block);
            OpBranch(while_continue_block);

            AddLabel(default_label);
            OpBranch(while_merge_block);

            AddLabel(while_continue_block);
            OpBranch(while_label);

            AddLabel(while_merge_block);
            OpReturnValue(ConstBool(false));
            OpFunctionEnd();
        }
    }

    // Define the shader execution entry subroutine
    const Id exec_shader{OpFunction(bool_id, spv::FunctionControlMask::MaskNone, func_type)};
    AddLabel();

    // Call main subroutine
    CallSubroutine(GetSubroutine(main_offset, PROGRAM_END));

    // Add the main entry point
    DefineEntryPoint();

    // Initialize registers
    OpStore(conditional_code, ConstBool(false, false));
    OpStore(address_registers, ConstS32(0, 0, 0));
    for (int i = 0; i < 16; ++i) {
        OpStore(tmp_regs[i], ConstF32(0.f, 0.f, 0.f, 1.f));
    }

    // Call exec_shader
    OpFunctionCall(bool_id, exec_shader);

    OpReturn();
    OpFunctionEnd();
}

void VertexModule::DefineArithmeticTypes() {
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
        bvec_ids.ids[i] = Name(TypeVector(bool_id, size), fmt::format("bvec{}_id", size));
    }
}

void VertexModule::DefineEntryPoint() {
    AddCapability(spv::Capability::Shader);
    SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);

    const Id main_type{TypeFunction(TypeVoid())};
    const Id main_func{OpFunction(TypeVoid(), spv::FunctionControlMask::MaskNone, main_type)};
    AddLabel();

    boost::container::small_vector<Id, 32> interfaces;
    /*interfaces.push_back(conditional_code);
    interfaces.push_back(address_registers);
    interfaces.push_back(vs_uniforms);
    for (const Id& tmp_reg : tmp_regs) {
        interfaces.push_back(tmp_reg);
    }*/
    for (size_t i = 0; i < input_typed_regs.size(); i++) {
        if (used_regs[i]) {
            ASSERT(Sirit::ValidId(input_typed_regs[i]));
            interfaces.push_back(input_typed_regs[i]);
        }
    }
    for (u32 i = 0; i < config.state.num_outputs; ++i) {
        interfaces.push_back(output_regs[i]);
    }

    AddEntryPoint(spv::ExecutionModel::Vertex, main_func, "main", interfaces);
}

void VertexModule::DefineUniformStructs() {
    // glslang uses uint for representing bools
    const Id barray{TypeArray(u32_id, ConstU32(16u))};
    const Id iarray{TypeArray(uvec_ids.Get(4), ConstU32(4u))};
    const Id farray{TypeArray(vec_ids.Get(4), ConstU32(96u))};
    const Id vs_config_id{TypeStruct(barray, iarray, farray)};
    constexpr std::array vs_config_offsets{0u, 256u, 320u};

    Decorate(vs_config_id, spv::Decoration::Block);
    Decorate(barray, spv::Decoration::ArrayStride, 16);
    Decorate(iarray, spv::Decoration::ArrayStride, 16);
    Decorate(farray, spv::Decoration::ArrayStride, 16);
    for (u32 i = 0; i < static_cast<u32>(vs_config_offsets.size()); i++) {
        MemberDecorate(vs_config_id, i, spv::Decoration::Offset, vs_config_offsets[i]);
    }

    vs_uniforms = AddGlobalVariable(TypePointer(spv::StorageClass::Uniform, vs_config_id),
                                    spv::StorageClass::Uniform);
    Decorate(vs_uniforms, spv::Decoration::DescriptorSet, 0);
    Decorate(vs_uniforms, spv::Decoration::Binding, 0);
}

void VertexModule::DefineInterface() {
    // Add declarations for registers
    conditional_code = DefineVar(bvec_ids.Get(2), spv::StorageClass::Private);
    address_registers = DefineVar(ivec_ids.Get(3), spv::StorageClass::Private);
    for (std::size_t i = 0; i < tmp_regs.size(); i++) {
        tmp_regs[i] = DefineVar(vec_ids.Get(4), spv::StorageClass::Private);
    }
    for (u32 i = 0; i < config.state.num_outputs; ++i) {
        output_regs[i] = DefineOutput(vec_ids.Get(4), i);
    }
}

std::optional<std::vector<u32>> GenerateVertexShaderSPV(const Pica::Shader::ShaderSetup& setup,
                                                        const PicaVSConfig& config) {
    try {
        VertexModule module(setup, config);
        module.Generate();
        const std::vector<u32> code = module.Assemble();

        FileUtil::IOFile file{fmt::format("vert{}.spv", i++), "wb"};
        file.WriteBytes(code.data(), code.size() * sizeof(u32));
        file.Flush();
        file.Close();

        return code;
    } catch (const DecompileFail& exception) {
        LOG_INFO(HW_GPU, "Shader decompilation failed: {}", exception.what());
        return std::nullopt;
    }
}

} // namespace Vulkan
