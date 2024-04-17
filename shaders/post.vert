#version 450

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 f_tex_coord;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    f_tex_coord = (position + vec2(1.0)) / 2.0;
}
