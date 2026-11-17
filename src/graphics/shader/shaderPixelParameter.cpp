#include "graphics/shader/shader.h"

#include "common/assert.h"

#include <algorithm>

namespace Libs::Graphics {

namespace {

constexpr uint32_t PsInputOffsetMask   = 0x0000001fu;
constexpr uint32_t PsInputUseDefault   = 0x00000020u;
constexpr uint32_t PsInputDefaultShift = 8u;
constexpr uint32_t PsInputFlatShade    = 0x00000400u;

} // namespace

uint32_t ShaderPixelParameterMappedLocation(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num ? info.interpolator_settings[input] & PsInputOffsetMask : input;
}

uint32_t ShaderPixelParameterLocation(const ShaderPixelInputInfo& info,
                                      std::span<const uint32_t> active_inputs, uint32_t input) {
	std::array<bool, 32> used_locations {};
	for (const auto active_input: active_inputs) {
		auto location = ShaderPixelParameterMappedLocation(info, active_input);
		if (location < used_locations.size() && used_locations[location]) {
			location = active_input;
			while (location < used_locations.size() && used_locations[location]) {
				location++;
			}
			EXIT_NOT_IMPLEMENTED(location >= used_locations.size());
		}

		if (active_input == input) {
			return location;
		}
		used_locations[location] = true;
	}
	return ShaderPixelParameterMappedLocation(info, input);
}

bool ShaderPixelParameterIsFlat(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < info.input_num && (info.interpolator_settings[input] & PsInputFlatShade) != 0 &&
	       !ShaderPixelParameterIsCustom(info, input);
}

bool ShaderPixelParameterIsCustom(const ShaderPixelInputInfo& info, uint32_t input) {
	return input < 32u && (info.custom_interpolation_mask & (1u << input)) != 0;
}

ShaderEmulatedBarycentricLayout ShaderPixelEmulatedBarycentrics(const ShaderPixelInputInfo&    info,
                                                                std::span<const uint32_t>      active_inputs,
                                                                std::span<const bool>          per_vertex,
                                                                bool smooth_weights, bool noperspective_weights) {
	constexpr uint32_t LocationCount = 32;
	ShaderEmulatedBarycentricLayout layout;
	if (active_inputs.size() > layout.vertex_location.size() || per_vertex.size() != active_inputs.size()) {
		return layout;
	}
	// The ordinary parameters keep their locations; the per-vertex ones (three locations each) and the weights follow.
	uint32_t next = 0;
	for (size_t i = 0; i < active_inputs.size(); i++) {
		if (!per_vertex[i]) {
			next = std::max(next, ShaderPixelParameterLocation(info, active_inputs, active_inputs[i]) + 1u);
		}
	}
	for (size_t i = 0; i < active_inputs.size(); i++) {
		if (per_vertex[i]) {
			layout.vertex_location[i] = next;
			next += 3u;
		}
	}
	if (smooth_weights) layout.smooth_location = next++;
	if (noperspective_weights) layout.noperspective_location = next++;
	layout.ok = next <= LocationCount;
	return layout;
}

bool ShaderPixelParameterDefault(const ShaderPixelInputInfo& info, uint32_t input,
                                 uint32_t component, uint32_t& bits) {
	if (input >= info.input_num || (info.interpolator_settings[input] & PsInputUseDefault) == 0 ||
	    (info.interpolator_settings[input] & PsInputFlatShade) != 0 ||
	    ShaderPixelParameterIsCustom(info, input)) {
		return false;
	}
	const auto value = (info.interpolator_settings[input] >> PsInputDefaultShift) & 0x3u;
	const bool one   = value == 3u || (value == 1u && component == 3u) ||
	                 (value == 2u && component < 3u);
	bits = one ? 0x3f800000u : 0u;
	return true;
}

} // namespace Libs::Graphics
