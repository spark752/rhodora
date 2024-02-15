#version 450

// Vertex format
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

// Outputs to fragment shader
layout(location = 0) out vec3 f_normal;
layout(location = 1) out vec3 f_position;
layout(location = 2) out vec2 f_tex_coord;

// Uniform buffer definition shared with fragment shader
#include "vpl.glsl"

// norm_view is the matrix for converting normals to view space which could be
// a mat3, but a mat3 pads each column (making it the same size as a mat3x4)
// and there is no built in conversion from glm matrices to the rust arrays
// created for either mat3 or mat3x4.
layout(set = 1, binding = 0) uniform M {
    mat4 model_view;
} m;

void main() {
    f_tex_coord = tex_coord;

    vec4 pos_vs = m.model_view * vec4(position, 1.0);
    f_position = vec3(pos_vs.xyz);

    vec4 norm_vs = m.model_view * vec4(normal, 0.0);
    f_normal = vec3(norm_vs.xyz);

    gl_Position = vpl.proj * pos_vs;
}
