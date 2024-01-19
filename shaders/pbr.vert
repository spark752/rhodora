#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 tex_coord;

layout(location = 0) out vec3 f_normal;
layout(location = 1) out vec3 f_position;
layout(location = 2) out vec2 f_tex_coord;

layout(set = 0, binding = 0) uniform VPL {
    mat4 view;
    mat4 proj;
    vec4 ambient;
    vec4 lights[4];
} vpl;

layout(set = 1, binding = 0) uniform M {
    mat4 model;
} m;

void main() {
    mat4 mv = vpl.view * m.model;
    vec4 pos_vs = mv * vec4(position, 1.0);
    f_position = vec3(pos_vs.xyz);
    f_normal = transpose(inverse(mat3(mv))) * normal;
    f_tex_coord = tex_coord;
    gl_Position = vpl.proj * mv * vec4(position, 1.0);
}
