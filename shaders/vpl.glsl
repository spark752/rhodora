// Unifrom buffer definition for vertex and fragment shader
layout(set = 0, binding = 0) uniform VPL {
    mat4 proj;
    vec4 ambient;
    vec4 lights[4];
} vpl;
