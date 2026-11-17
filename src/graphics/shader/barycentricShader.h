#ifndef EMULATOR_SRC_GRAPHICS_SHADER_BARYCENTRICSHADER_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_BARYCENTRICSHADER_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

struct ShaderPixelInputInfo;
struct ShaderVertexInputInfo;

// The geometry shader that stands in for VK_KHR_fragment_shader_barycentric on a GPU without it (NVIDIA's GeForce GTX 10
// series and older), for a pipeline whose pixel shader was translated with
// ShaderRecompiler::Frontend::TranslatorFeatureEmulatedBarycentrics and reads per-vertex inputs or barycentric weights.
// It takes a triangle (from the vertex shader, or from a rect list's tessellation evaluation shader: `rect_list`) and
// writes it again with, at each of its three vertices:
//   - gl_Position, the clip and cull distances of the vertex shader, and the ordinary parameters of the pixel shader
//     (flat ones from the provoking vertex: the first, the last with `provoking_last`),
//   - the three vertices' values of each per-vertex parameter, as a flat array of three vec4 at the pixel shader's
//     emulated locations (ShaderPixelEmulatedBarycentrics),
//   - the weights as a varying: (1,0,0), (0,1,0) and (0,0,1) at the first, second and third vertex, so that the pixel
//     shader gets the weights of PerVertexKHR's vertex order (a triangle list's).
// Empty where the pipeline needs none (the pixel shader reads neither) or cannot have one (its interface does not fit in 32
// locations, or the vertex shader writes a layer or a viewport index, which a geometry shader cannot pass on).
[[nodiscard]] std::vector<uint32_t> BuildEmulatedBarycentricShader(const ShaderVertexInputInfo& vertex_info,
                                                                   const ShaderPixelInputInfo&  pixel_info,
                                                                   bool rect_list, bool provoking_last);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_SHADER_BARYCENTRICSHADER_H_
