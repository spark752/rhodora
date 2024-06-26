#version 460

// Vertex format
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;
layout(location = 3) in uint joint_ids;
layout(location = 4) in vec4 weights;

// Outputs to fragment shader. Trying to pass these as a struct or block in
// vulkano 0.33 causes the shader macro to panic.
layout(location = 0) out vec3 f_normal;
layout(location = 1) out vec3 f_position;
layout(location = 2) out vec2 f_tex_coord;

layout(set = 0, binding = 0) buffer Matrices {
    mat4 proj;
    mat4 model_view[];
};

layout(set = 1, binding = 0) buffer Joints {
    mat2x4 joints[];
};

layout(push_constant, std430) uniform VertData {
   uint model_index;
   uint joint_data_offset;
};

void main() {
    // Texture coordinates will be interpolated
    f_tex_coord = tex_coord;

    // Find the relevant dual quaternions
    mat2x4 dq0 = joints[(joint_ids >> 24) + joint_data_offset];
    mat2x4 dq1 = joints[(joint_ids >> 16 & 0xff) + joint_data_offset];
    mat2x4 dq2 = joints[(joint_ids >> 8 & 0xff) + joint_data_offset];
    mat2x4 dq3 = joints[(joint_ids & 0xff) + joint_data_offset];

    // Antipodality: quaternion q and -q represent the same rotation, but
    // one will blend correctly and the other will lead to nightmares
    if (dot(dq0[0], dq1[0]) < 0.0) dq1 *= -1.0;
    if (dot(dq0[0], dq2[0]) < 0.0) dq2 *= -1.0;
    if (dot(dq0[0], dq3[0]) < 0.0) dq3 *= -1.0;

    // Linear blend and pseudo-normalize. A correct normal has an extra term
    // for the dual part but that cancels out later on the transform.
    mat2x4 blend =
        dq0 * weights.x +
        dq1 * weights.y +
        dq2 * weights.z +
        dq3 * weights.w;
    blend = blend / max(length(blend[0]), 0.001);

    // Position
    vec4 ap = vec4(position + 2.0 * cross(blend[0].xyz,
        cross(blend[0].xyz, position) + blend[0].w * position) +
        2.0 * (blend[0].w * blend[1].xyz - blend[1].w * blend[0].xyz +
        cross(blend[0].xyz, blend[1].xyz)), 1.0);
    vec4 pos_vs = model_view[model_index] * ap;
    f_position = pos_vs.xyz;

    // Normal
    vec4 an = vec4(normal + 2.0 * cross(blend[0].xyz,
        cross(blend[0].xyz, normal) + blend[0].w * normal), 0.0);
    vec4 norm_vs = model_view[model_index] * an;
    f_normal = norm_vs.xyz;

    // Projected position as OpenGL style output
    gl_Position = proj * pos_vs;
}
