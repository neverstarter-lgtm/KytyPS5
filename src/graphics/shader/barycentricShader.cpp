#include "graphics/shader/barycentricShader.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "spirv/unified1/spirv.hpp11"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <utility>

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::Spirv::Builder;
namespace IR = ShaderRecompiler::IR;

constexpr uint32_t SpirvVersion15 = 0x00010500u;

template <typename T>
constexpr uint32_t Word(T value) {
	return static_cast<uint32_t>(value);
}

// A parameter the pixel shader reads.
struct Parameter {
	uint32_t input_location  = 0; // where the geometry shader reads it (the vertex shader's, or the tessellation's)
	uint32_t output_location = 0; // where the pixel shader reads it (its first location for a per-vertex one)
	bool     flat            = false;
	bool     per_vertex      = false;
};

struct Plan {
	std::vector<Parameter> parameters;
	uint32_t               smooth_location        = UINT32_MAX;
	uint32_t               noperspective_location = UINT32_MAX;
	uint32_t               clip_distances         = 0;
	uint32_t               cull_distances         = 0;
	bool                   valid                  = false;
};

Plan MakePlan(const ShaderVertexInputInfo& vertex_info, const ShaderPixelInputInfo& pixel_info, bool rect_list) {
	Plan plan;
	if (vertex_info.stage.program == nullptr || pixel_info.stage.program == nullptr) {
		return plan;
	}

	std::array<uint32_t, 32> active {};
	std::array<bool, 32>     per_vertex {};
	size_t                   count          = 0;
	bool                     smooth_weights = false;
	bool                     linear_weights = false;
	bool                     emulated       = false;
	for (const auto& input: pixel_info.stage.program->info.inputs) {
		if (input.kind == IR::StageInputKind::Parameter) {
			if (count == active.size()) {
				return plan;
			}
			active[count]     = input.location;
			per_vertex[count] = input.per_vertex;
			emulated          = emulated || input.per_vertex;
			count++;
		} else if (input.kind == IR::StageInputKind::BaryCoordSmooth) {
			smooth_weights = emulated = true;
		} else if (input.kind == IR::StageInputKind::BaryCoordNoPerspective) {
			linear_weights = emulated = true;
		}
	}
	if (!emulated) {
		return plan;
	}
	const auto layout = ShaderPixelEmulatedBarycentrics(pixel_info, {active.data(), count}, {per_vertex.data(), count},
	                                                    smooth_weights, linear_weights);
	if (!layout.ok) {
		return plan;
	}

	for (size_t i = 0; i < count; i++) {
		const auto mapped = ShaderPixelParameterMappedLocation(pixel_info, active[i]);
		// (As the rect list's: a parameter the vertex shader does not export has no value to hand on.)
		if (mapped >= 32u || (vertex_info.stage.program->param_export_mask & (1u << mapped)) == 0) {
			continue;
		}
		const auto location = ShaderPixelParameterLocation(pixel_info, {active.data(), count}, active[i]);
		Parameter  parameter;
		// (After a rect list's tessellation the parameters are at the pixel shader's locations already.)
		parameter.input_location  = rect_list ? location : mapped;
		parameter.output_location = per_vertex[i] ? layout.vertex_location[i] : location;
		parameter.flat            = ShaderPixelParameterIsFlat(pixel_info, active[i]);
		parameter.per_vertex      = per_vertex[i];
		plan.parameters.push_back(parameter);
	}
	plan.smooth_location        = layout.smooth_location;
	plan.noperspective_location = layout.noperspective_location;

	// What the vertex shader writes besides the position and the parameters: a rect list's tessellation writes neither the
	// clip nor the cull distances, so there is nothing to hand on there.
	if (!rect_list) {
		for (const auto& output: vertex_info.stage.program->info.outputs) {
			switch (output.kind) {
				case IR::StageOutputKind::ClipDistance:
					plan.clip_distances = std::max(plan.clip_distances, output.index + 1u);
					break;
				case IR::StageOutputKind::CullDistance:
					plan.cull_distances = std::max(plan.cull_distances, output.index + 1u);
					break;
				case IR::StageOutputKind::Layer:
				case IR::StageOutputKind::ViewportIndex:
					// (A geometry shader cannot read them from the vertex shader.)
					return plan;
				default: break;
			}
		}
	}
	plan.valid = true;
	return plan;
}

class BarycentricEmitter {
public:
	BarycentricEmitter(const Plan& plan_, bool provoking_last_): plan(plan_), provoking_last(provoking_last_) {}

	std::vector<uint32_t> Emit() {
		builder.AddMemoryModel({Word(spv::AddressingModel::Logical), Word(spv::MemoryModel::GLSL450)});

		void_type       = Type(spv::Op::OpTypeVoid);
		uint_type       = Type(spv::Op::OpTypeInt, 32u, 0u);
		int_type        = Type(spv::Op::OpTypeInt, 32u, 1u);
		float_type      = Type(spv::Op::OpTypeFloat, 32u);
		vec3_float_type = Type(spv::Op::OpTypeVector, float_type, 3u);
		vec4_float_type = Type(spv::Op::OpTypeVector, float_type, 4u);
		function_type   = Type(spv::Op::OpTypeFunction, void_type);

		per_vertex_type = builder.DecoratedType(
		    Word(spv::Op::OpTypeStruct), {vec4_float_type},
		    {{Word(spv::Op::OpMemberDecorate),
		      {0u, Word(spv::Decoration::BuiltIn), Word(spv::BuiltIn::Position)}},
		     {Word(spv::Op::OpDecorate), {Word(spv::Decoration::Block)}}});

		ptr_input_vec4_float  = Pointer(spv::StorageClass::Input, vec4_float_type);
		ptr_output_vec4_float = Pointer(spv::StorageClass::Output, vec4_float_type);
		ptr_output_vec3_float = Pointer(spv::StorageClass::Output, vec3_float_type);

		builder.RequireCapability(Word(spv::Capability::Shader));
		builder.RequireCapability(Word(spv::Capability::Geometry));
		if (plan.clip_distances != 0) {
			builder.RequireCapability(Word(spv::Capability::ClipDistance));
		}
		if (plan.cull_distances != 0) {
			builder.RequireCapability(Word(spv::Capability::CullDistance));
		}

		main = Result(spv::Op::OpFunction, void_type, spv::FunctionControlMask::MaskNone, function_type);
		builder.AddExecutionMode({main, Word(spv::ExecutionMode::Triangles)});
		builder.AddExecutionMode({main, Word(spv::ExecutionMode::OutputTriangleStrip)});
		builder.AddExecutionMode({main, Word(spv::ExecutionMode::OutputVertices), 3u});
		DefineInterface();
		builder.AddEntryPoint(Word(spv::ExecutionModel::Geometry), main, "main", interfaces);
		ResultWithoutType(spv::Op::OpLabel);

		const auto zero = Constant(float_type, std::bit_cast<uint32_t>(0.0f));
		const auto one  = Constant(float_type, std::bit_cast<uint32_t>(1.0f));
		const std::array<std::array<uint32_t, 3>, 3> unit {{{one, zero, zero}, {zero, one, zero}, {zero, zero, one}}};

		// The flat parameters come from the provoking vertex of the triangle the pipeline's state names.
		const uint32_t provoking = provoking_last ? 2u : 0u;
		for (uint32_t vertex = 0; vertex < 3u; vertex++) {
			// (The outputs are undefined after each EmitVertex: every vertex writes them all.)
			Store(Access(ptr_output_vec4_float, gl_out, Int(0)),
			      Load(vec4_float_type, Access(ptr_input_vec4_float, gl_in, Int(vertex), Int(0))));
			if (plan.clip_distances != 0) {
				Store(clip_out, Load(clip_array_type, Access(ptr_input_clip, clip_in, Int(vertex))));
			}
			if (plan.cull_distances != 0) {
				Store(cull_out, Load(cull_array_type, Access(ptr_input_cull, cull_in, Int(vertex))));
			}
			for (size_t i = 0; i < plan.parameters.size(); i++) {
				const auto& parameter = plan.parameters[i];
				if (parameter.per_vertex) {
					for (uint32_t other = 0; other < 3u; other++) {
						Store(Access(ptr_output_vec4_float, outputs[i], Int(other)),
						      Load(vec4_float_type, Access(ptr_input_vec4_float, inputs[i], Int(other))));
					}
				} else {
					const auto source = parameter.flat ? provoking : vertex;
					Store(outputs[i], Load(vec4_float_type, Access(ptr_input_vec4_float, inputs[i], Int(source))));
				}
			}
			const auto weights = builder.Constant(Word(spv::Op::OpConstantComposite), vec3_float_type,
			                                      {unit[vertex][0], unit[vertex][1], unit[vertex][2]});
			if (smooth_out != 0) {
				Store(smooth_out, weights);
			}
			if (linear_out != 0) {
				Store(linear_out, weights);
			}
			Emit(spv::Op::OpEmitVertex);
		}
		Emit(spv::Op::OpEndPrimitive);
		Emit(spv::Op::OpReturn);
		Emit(spv::Op::OpFunctionEnd);
		return builder.Build();
	}

private:
	template <typename... Args>
	uint32_t Type(spv::Op opcode, Args... operands) {
		return builder.Type(Word(opcode), {Word(operands)...});
	}

	uint32_t Constant(uint32_t type, uint32_t value) { return builder.Constant(Word(spv::Op::OpConstant), type, {value}); }

	uint32_t Pointer(spv::StorageClass storage, uint32_t type) { return Type(spv::Op::OpTypePointer, storage, type); }

	uint32_t Array(uint32_t type, uint32_t size) { return Type(spv::Op::OpTypeArray, type, Uint(size)); }

	template <typename... Args>
	uint32_t Result(spv::Op opcode, uint32_t type, Args... operands) {
		const auto id = builder.AllocateId();
		builder.AddFunction({Word(opcode), type, id, Word(operands)...});
		return id;
	}

	template <typename... Args>
	uint32_t ResultWithoutType(spv::Op opcode, Args... operands) {
		const auto id = builder.AllocateId();
		builder.AddFunction({Word(opcode), id, Word(operands)...});
		return id;
	}

	template <typename... Args>
	void Emit(spv::Op opcode, Args... operands) {
		builder.AddFunction({Word(opcode), Word(operands)...});
	}

	template <typename... Args>
	uint32_t Access(uint32_t pointer_type, uint32_t base, Args... indices) {
		return Result(spv::Op::OpAccessChain, pointer_type, base, Word(indices)...);
	}

	uint32_t Load(uint32_t type, uint32_t pointer) { return Result(spv::Op::OpLoad, type, pointer); }

	void Store(uint32_t pointer, uint32_t value) { Emit(spv::Op::OpStore, pointer, value); }

	uint32_t Int(uint32_t value) { return Constant(int_type, value); }

	uint32_t Uint(uint32_t value) { return Constant(uint_type, value); }

	uint32_t AddInterface(spv::StorageClass storage, uint32_t type) {
		const auto variable = builder.DefineGlobalVariable(Pointer(storage, type), Word(storage));
		interfaces.push_back(variable);
		return variable;
	}

	void Decorate(uint32_t target, spv::Decoration decoration, uint32_t value) {
		builder.AddAnnotation({Word(spv::Op::OpDecorate), target, Word(decoration), value});
	}

	void DecorateFlag(uint32_t target, spv::Decoration decoration) {
		builder.AddAnnotation({Word(spv::Op::OpDecorate), target, Word(decoration)});
	}

	void DefineInterface() {
		// The vertices of the triangle: the positions, then the clip and cull distances.
		gl_in = AddInterface(spv::StorageClass::Input, Array(per_vertex_type, 3u));
		if (plan.clip_distances != 0) {
			clip_array_type = Array(float_type, plan.clip_distances);
			ptr_input_clip  = Pointer(spv::StorageClass::Input, clip_array_type);
			clip_in         = AddInterface(spv::StorageClass::Input, Array(clip_array_type, 3u));
			Decorate(clip_in, spv::Decoration::BuiltIn, Word(spv::BuiltIn::ClipDistance));
		}
		if (plan.cull_distances != 0) {
			cull_array_type = Array(float_type, plan.cull_distances);
			ptr_input_cull  = Pointer(spv::StorageClass::Input, cull_array_type);
			cull_in         = AddInterface(spv::StorageClass::Input, Array(cull_array_type, 3u));
			Decorate(cull_in, spv::Decoration::BuiltIn, Word(spv::BuiltIn::CullDistance));
		}
		inputs.resize(plan.parameters.size());
		for (size_t i = 0; i < plan.parameters.size(); i++) {
			inputs[i] = AddInterface(spv::StorageClass::Input, Array(vec4_float_type, 3u));
			Decorate(inputs[i], spv::Decoration::Location, plan.parameters[i].input_location);
		}

		gl_out = AddInterface(spv::StorageClass::Output, per_vertex_type);
		if (plan.clip_distances != 0) {
			clip_out = AddInterface(spv::StorageClass::Output, clip_array_type);
			Decorate(clip_out, spv::Decoration::BuiltIn, Word(spv::BuiltIn::ClipDistance));
		}
		if (plan.cull_distances != 0) {
			cull_out = AddInterface(spv::StorageClass::Output, cull_array_type);
			Decorate(cull_out, spv::Decoration::BuiltIn, Word(spv::BuiltIn::CullDistance));
		}
		outputs.resize(plan.parameters.size());
		for (size_t i = 0; i < plan.parameters.size(); i++) {
			const auto& parameter = plan.parameters[i];
			// (A per-vertex parameter's array takes three locations from `output_location`.)
			outputs[i] = AddInterface(spv::StorageClass::Output,
			                          parameter.per_vertex ? Array(vec4_float_type, 3u) : vec4_float_type);
			Decorate(outputs[i], spv::Decoration::Location, parameter.output_location);
			if (parameter.flat || parameter.per_vertex) {
				DecorateFlag(outputs[i], spv::Decoration::Flat);
			}
		}
		if (plan.smooth_location != UINT32_MAX) {
			smooth_out = AddInterface(spv::StorageClass::Output, vec3_float_type);
			Decorate(smooth_out, spv::Decoration::Location, plan.smooth_location);
		}
		if (plan.noperspective_location != UINT32_MAX) {
			linear_out = AddInterface(spv::StorageClass::Output, vec3_float_type);
			Decorate(linear_out, spv::Decoration::Location, plan.noperspective_location);
			DecorateFlag(linear_out, spv::Decoration::NoPerspective);
		}
	}

	Builder               builder {SpirvVersion15};
	const Plan&           plan;
	bool                  provoking_last;
	std::vector<uint32_t> interfaces;
	std::vector<uint32_t> inputs;
	std::vector<uint32_t> outputs;
	uint32_t              main                  = 0;
	uint32_t              void_type             = 0;
	uint32_t              uint_type             = 0;
	uint32_t              int_type              = 0;
	uint32_t              float_type            = 0;
	uint32_t              vec3_float_type       = 0;
	uint32_t              vec4_float_type       = 0;
	uint32_t              function_type         = 0;
	uint32_t              per_vertex_type       = 0;
	uint32_t              ptr_input_vec4_float  = 0;
	uint32_t              ptr_output_vec4_float = 0;
	uint32_t              ptr_output_vec3_float = 0;
	uint32_t              clip_array_type       = 0;
	uint32_t              cull_array_type       = 0;
	uint32_t              ptr_input_clip        = 0;
	uint32_t              ptr_input_cull        = 0;
	uint32_t              gl_in                 = 0;
	uint32_t              gl_out                = 0;
	uint32_t              clip_in               = 0;
	uint32_t              cull_in               = 0;
	uint32_t              clip_out              = 0;
	uint32_t              cull_out              = 0;
	uint32_t              smooth_out            = 0;
	uint32_t              linear_out            = 0;
};

} // namespace

std::vector<uint32_t> BuildEmulatedBarycentricShader(const ShaderVertexInputInfo& vertex_info,
                                                     const ShaderPixelInputInfo&  pixel_info, bool rect_list,
                                                     bool provoking_last) {
	const auto plan = MakePlan(vertex_info, pixel_info, rect_list);
	if (!plan.valid) {
		return {};
	}
	BarycentricEmitter emitter(plan, provoking_last);
	return emitter.Emit();
}

} // namespace Libs::Graphics
