#version 450

// Basic sort of fragment shader for showcase like application

layout(location = 0) in vec3 f_normal;
layout(location = 1) in vec3 f_position;
layout(location = 2) in vec2 f_tex_coord;

layout(set = 1, binding = 1) uniform sampler2D tex;

layout(location = 0) out vec4 frag;

void main() {
    vec4 colour = texture(tex, f_tex_coord);
    vec3 ldir = vec3(0.0, -0.28, 0.96); // Towards light in viewspace (y axis down in Vulkan), normalized
    vec3 n = normalize(f_normal);
    float intensity = max(dot(n, ldir) * 0.4, 0.08); // Diffuse with a little ambient
    frag = vec4(colour.rgb * intensity, 1.0);
    //frag = vec4((n + 1.0) /2.0, 1.0);
    //frag = vec4(f_tex_coord.x, f_tex_coord.y, 0.0, 1.0);
    //frag = vec4(colour.r, 0.0, 0.0, 1.0);
}
