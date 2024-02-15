#version 450
// #define MAX_JOINTS when compiling

// Vertex format
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;
layout(location = 3) in uint joint_ids;
layout(location = 4) in vec4 weights;

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
    mat2x4 joints[MAX_JOINTS];
} m;

void main() {
    // Texture coordinates will be interpolated
    f_tex_coord = tex_coord;

    // Blend dual quaternions based on pose and joint weighting
    mat2x4 blend = m.joints[joint_ids >> 24] * weights.x +
        m.joints[joint_ids >> 16 & 0xff] * weights.y +
        m.joints[joint_ids >> 8 & 0xff] * weights.z +
        m.joints[joint_ids & 0xff] * weights.w;
    float len = length(blend[0]);
    if (len < 0.01) len = 1.0;
    mat2x4 dq = blend / len;

    // Dual quaternion for position translation and rotation
    vec4 ap = vec4(position + 2.0 * cross(dq[0].xyz,
        cross(dq[0].xyz, position) + dq[0].w * position) +
        2.0 * (dq[0].w * dq[1].xyz - dq[1].w * dq[0].xyz +
        cross(dq[0].xyz, dq[1].xyz)), 1.0);
    vec4 pos_vs = m.model_view * ap;
    f_position = vec3(pos_vs.xyz);

    // Real part of dual quaternion for normal rotation
    vec4 an = vec4(normal + 2.0 * cross(dq[0].xyz,
        cross(dq[0].xyz, normal) + dq[0].w * normal), 0.0);
    f_normal = vec3(m.model_view * an);

    // Projected position as OpenGL style output
    gl_Position = vpl.proj * pos_vs;
}
