// Uniform buffer definition shared between vertex and fragment shader.
// This is non-model specific data that is constant across the render
// pass. Due to the way Vulkan works, less frequently changed data should
// be in lower discriptor set numbers, so this is set 0.
// The name comes from View/Projection/Lights but that no longer
// matches the contents. It was more efficient to combine the View and
// Model matrices for model specific data and so the View matrix was
// no longer being used. It is unlikely that the fragment shader needs
// the Projection matrix, nor the vertex shader the lighting information
// but it cuts down on the number of blocks.
layout(set = 0, binding = 0) uniform VPL {
    mat4 proj;
    vec4 ambient;
    vec4 lights[4];
} vpl;
