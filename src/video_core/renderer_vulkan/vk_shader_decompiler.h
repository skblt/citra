// Copyright 2022 Citra Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <exception>
#include <map>
#include <set>
#include <optional>
#include <sirit/sirit.h>
#include <nihstro/shader_bytecode.h>
#include "video_core/renderer_vulkan/vk_shader_gen.h"

namespace Vulkan {

using Sirit::Id;

constexpr u32 PROGRAM_END = Pica::Shader::MAX_PROGRAM_CODE_LENGTH;

class DecompileFail : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// Describes the behaviour of code path of a given entry point and a return point.
enum class ExitMethod {
    Undetermined, ///< Internal value. Only occur when analyzing JMP loop.
    AlwaysReturn, ///< All code paths reach the return point.
    Conditional,  ///< Code path reaches the return point or an END instruction conditionally.
    AlwaysEnd,    ///< All code paths reach a END instruction.
};

/// A label is an offset into the code assigned to the SPIR-V lavel
struct Label {
    u32 label;
    mutable Id spv_label;

    Label operator+(u32 other) const {
        return Label{.label = label + other, .spv_label = spv_label};
    }

    bool operator<(const Label& other) const {
        return label < other.label;
    }
};

struct SpirvParams {
    Id jmp_to;              ///< Temporary holding the current jump target
    Id while_label;         ///< Label to the beginning of the while loop
    Id switch_label;        ///< Label to the beginning of the switch statement
    Id switch_merge_block;  ///< Label to the merge block of the switch statement
    std::array<Id, 3> vars; ///< Available function variables used for LOOP
    u32 used_vars = 0;
};

/// A subroutine is a range of code refereced by a CALL, IF or LOOP instruction.
struct Subroutine {
    u32 begin;              ///< Entry point of the subroutine.
    u32 end;                ///< Return point of the subroutine.
    ExitMethod exit_method; ///< Exit method of the subroutine.
    std::set<u32> labels;   ///< Addresses refereced by JMP instructions.
    mutable Id function;    ///< Function label of the subroutine

    bool operator<(const Subroutine& rhs) const {
        return std::tie(begin, end) < std::tie(rhs.begin, rhs.end);
    }
};

/// Analyzes shader code and produces a set of subroutines.
class ControlFlowAnalyzer {
public:
    ControlFlowAnalyzer(const Pica::Shader::ProgramCode& program_code, u32 main_offset);

    [[nodiscard]] std::set<Subroutine> MoveSubroutines() {
        return std::move(subroutines);
    }

private:
    /// Adds and analyzes a new subroutine if it is not added yet.
    const Subroutine& AddSubroutine(u32 begin, u32 end);

    /// Merges exit method of two parallel branches.
    ExitMethod ParallelExit(ExitMethod a, ExitMethod b);

    /// Cascades exit method of two blocks of code.
    ExitMethod SeriesExit(ExitMethod a, ExitMethod b);

    /// Scans a range of code for labels and determines the exit method.
    ExitMethod Scan(u32 begin, u32 end, std::set<u32>& labels);

private:
    const Pica::Shader::ProgramCode& program_code;
    std::set<Subroutine> subroutines;
    std::map<std::pair<u32, u32>, ExitMethod> exit_method_map;
};

class VertexModule : public Sirit::Module {
    struct VectorIds {
        /// Returns the type id of the vector with the provided size
        [[nodiscard]] constexpr Id Get(u32 size) const {
            return ids[size - 2];
        }

        std::array<Id, 3> ids;
    };

public:
    VertexModule(const Pica::Shader::ShaderSetup& setup,
                 const PicaVSConfig& config);
    ~VertexModule();

    void Generate();

private:
    /// Gets the Subroutine object corresponding to the specified address.
    const Subroutine& GetSubroutine(u32 begin, u32 end) const;

    /// Generates code to evaluate a shader control flow instruction
    Id EvaluateCondition(nihstro::Instruction::FlowControlType flow_control);

    /// Generates code representing a source register.
    Id GetSourceRegister(const SourceRegister& source_reg, u32 address_register_index);

    /// Generates code representing a destination register.
    Id GetDestRegister(const DestRegister& dest_reg);

    /// Returns the pointer type of the destination register.
    Id GetDestPointer(const DestRegister& dest_reg);

    /// Attemps to sanitize multiplication result to match PICA expected behaviour.
    Id SanitizeMul(Id lhs, Id rhs);

    /**
     * Adds code that calls a subroutine.
     * @param subroutine the subroutine to call.
     */
    void CallSubroutine(const Subroutine& subroutine);

    /**
     * Writes code that does an assignment operation.
     * @param swizzle the swizzle data of the current instruction.
     * @param reg the destination register code.
     * @param value the code representing the value to assign.
     * @param storage_class storage specifier of reg.
     * @param value_num_components number of components of the value to assign.
     */
    void SetDest(const nihstro::SwizzlePattern& swizzle, Id reg, Id value,
                 Id reg_pointer, u32 dest_num_components, u32 value_num_components);

    /**
     * Compiles a single instruction from PICA to GLSL.
     * @param offset the offset of the PICA shader instruction.
     * @return the offset of the next instruction to execute. Usually it is the current offset + 1.
     * If the current instruction is IF or LOOP, the next instruction is after the IF or LOOP block.
     * If the current instruction always terminates the program, returns PROGRAM_END.
     */
    u32 CompileInstr(u32 offset);

    /**
     * Compiles a range of instructions from PICA to GLSL.
     * @param begin the offset of the starting instruction.
     * @param end the offset where the compilation should stop (exclusive).
     * @return the offset of the next instruction to compile. PROGRAM_END if the program terminates.
     */
    u32 CompileRange(u32 begin, u32 end);

private:
    /// Returns an id of the attribute type
    Id AttribType(u32 index) const {
        switch (config.state.attrib_types[index]) {
        case Pica::PipelineRegs::VertexAttributeFormat::FLOAT:
            return vec_ids.Get(4);
        case Pica::PipelineRegs::VertexAttributeFormat::BYTE:
        case Pica::PipelineRegs::VertexAttributeFormat::SHORT:
            return ivec_ids.Get(4);
        case Pica::PipelineRegs::VertexAttributeFormat::UBYTE:
            return uvec_ids.Get(4);
        default:
            UNREACHABLE();
        }
        return Id{};
    }

    /// Returns the attribute casted to float
    Id AttribCast(u32 index, Id typed_reg) {
        switch (config.state.attrib_types[index]) {
        case Pica::PipelineRegs::VertexAttributeFormat::FLOAT:
            break;
        case Pica::PipelineRegs::VertexAttributeFormat::BYTE:
        case Pica::PipelineRegs::VertexAttributeFormat::SHORT:
            return OpConvertSToF(ivec_ids.Get(4), typed_reg);
        case Pica::PipelineRegs::VertexAttributeFormat::UBYTE:
            return OpConvertUToF(uvec_ids.Get(4), typed_reg);
        default:
            UNREACHABLE();
        }
        return typed_reg;
    }

    /// Loads the member specified from the vs_uniforms uniform struct
    template <typename... Ids>
    [[nodiscard]] Id GetVsUniformMember(Id type, Ids... ids) {
        const Id uniform_ptr{TypePointer(spv::StorageClass::Uniform, type)};
        return OpLoad(type, OpAccessChain(uniform_ptr, vs_uniforms, ids...));
    }

    /// Generates code representing a bool uniform
    Id GetUniformBool(u32 index) {
        const Id value{GetVsUniformMember(u32_id, ConstU32(0u), ConstU32(index))};
        return OpINotEqual(bool_id, value, ConstU32(0u));
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
    [[nodiscard]] Id DefineUniformConst(Id type, u32 set, u32 binding, bool readonly = false) {
        const Id uniform_id{DefineVar(type, spv::StorageClass::UniformConstant)};
        Decorate(uniform_id, spv::Decoration::DescriptorSet, set);
        Decorate(uniform_id, spv::Decoration::Binding, binding);
        if (readonly) {
            Decorate(uniform_id, spv::Decoration::NonWritable);
        }
        return uniform_id;
    }

    template <bool global = true>
    [[nodiscard]] Id DefineVar(Id type, spv::StorageClass storage_class) {
        const Id pointer_type_id{TypePointer(storage_class, type)};
        return global ? AddGlobalVariable(pointer_type_id, storage_class)
                      : AddLocalVariable(pointer_type_id, storage_class);
    }

    /// Returns the id of a signed integer constant of value
    [[nodiscard]] Id ConstBool(bool value) {
        return value ? ConstantTrue(bool_id) : ConstantFalse(bool_id);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstBool(Args&&... values) {
        constexpr u32 size = static_cast<u32>(sizeof...(values));
        static_assert(size >= 2);
        const std::array constituents{ConstBool(values)...};
        const Id type = size <= 4 ? bvec_ids.Get(size) : TypeArray(bool_id, ConstU32(size));
        return ConstantComposite(type, constituents);
    }

    /// Returns the id of a signed integer constant of value
    [[nodiscard]] Id ConstU32(u32 value) {
        return Constant(u32_id, value);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstU32(Args&&... values) {
        constexpr u32 size = static_cast<u32>(sizeof...(values));
        static_assert(size >= 2);
        const std::array constituents{Constant(u32_id, values)...};
        const Id type = size <= 4 ? uvec_ids.Get(size) : TypeArray(u32_id, ConstU32(size));
        return ConstantComposite(type, constituents);
    }

    /// Returns the id of a signed integer constant of value
    [[nodiscard]] Id ConstS32(s32 value) {
        return Constant(i32_id, value);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstS32(Args&&... values) {
        constexpr u32 size = static_cast<u32>(sizeof...(values));
        static_assert(size >= 2);
        const std::array constituents{Constant(i32_id, values)...};
        const Id type = size <= 4 ? ivec_ids.Get(size) : TypeArray(i32_id, ConstU32(size));
        return ConstantComposite(type, constituents);
    }

    /// Returns the id of a float constant of value
    [[nodiscard]] Id ConstF32(f32 value) {
        return Constant(f32_id, value);
    }

    template <typename... Args>
    [[nodiscard]] Id ConstF32(Args... values) {
        constexpr u32 size = static_cast<u32>(sizeof...(values));
        static_assert(size >= 2);
        const std::array constituents{Constant(f32_id, values)...};
        const Id type = size <= 4 ? vec_ids.Get(size) : TypeArray(f32_id, ConstU32(size));
        return ConstantComposite(type, constituents);
    }

    void DefineArithmeticTypes();
    void DefineEntryPoint();
    void DefineUniformStructs();
    void DefineInterface();

public:
    Id void_id{};
    Id bool_id{};
    Id f32_id{};
    Id i32_id{};
    Id u32_id{};

    VectorIds vec_ids{};
    VectorIds ivec_ids{};
    VectorIds uvec_ids{};
    VectorIds bvec_ids{};

private:
    const PicaVSConfig& config;
    const Pica::Shader::ProgramCode& program_code;
    const Pica::Shader::SwizzleData& swizzle_data;
    u32 main_offset;
    bool sanitize_mul;
    std::set<Subroutine> subroutines;

    /**
     * PICA input registers are float but vulkan doesn't have the
     * ability to cast integer attributes to float. Thus they are
     * manually cast if needed
     **/
    std::array<Id, 16> input_typed_regs{};
    std::array<Id, 16> input_regs{};
    std::array<bool, 16> used_regs{};
    std::array<Id, 16> output_regs{};
    std::array<Id, 16> tmp_regs{};

    Id vs_uniforms{};
    Id conditional_code{};
    Id address_registers{};
};

/**
 * Generates the SPIRV vertex shader program source code for the given VS program
 * @returns String of the shader source code; boost::none on failure
 */
std::optional<std::vector<u32>> GenerateVertexShaderSPV(const Pica::Shader::ShaderSetup& setup,
                                         const PicaVSConfig& config);


} // namespace Vulkan
