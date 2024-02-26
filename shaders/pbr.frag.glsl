#version 450

// Basic sort of fragment shader for PBR lighting
// Based in part on learnopengl.com and on "Moving Frostbite to Physically
// Based Rendering" by Legarde and de Rousiers, SIGGRAPH 2014

// Define this for a visualization shader used for development and debugging
#ifdef VISUALIZE
    #define USE_AMBIENT 0   // Default
    #define NO_AMBIENT 1
    #define AMBIENT_ONLY 2  // Overrides `specular_mode`
    #define USE_SPECULAR 0  // Default
    #define NO_SPECULAR 1
    #define SPECULAR_ONLY 2
    #define USE_LIGHTING 0      // Default
    #define SHOW_NORMALS 1      // Overrides lighting
    #define SHOW_ROUGHNESS 2    // Overrides lighting
    #define SHOW_METALNESS 3    // Overrides lighting
    #define SHOW_MIPMAP 4       // Overrides lighting
    #define SHOW_DEPTH 5        // Overrides lighting

    // TODO: Move these to push constants
    uint ambient_mode = USE_AMBIENT;
    uint specular_mode = USE_SPECULAR;
    uint override_mode = SHOW_DEPTH;
#endif

// Inputs from vertex shaders. Trying to pass these as a struct or block in
// vulkano 0.33 causes the shader macro to panic.
layout(location = 0) in vec3 f_normal;
layout(location = 1) in vec3 f_position;
layout(location = 2) in vec2 f_tex_coord;

// Matrices are used by vertex shader but are part of the same uniform buffer.
// The lights array currently contains point lights with each position in view
// space in the xyz elements and its intensity in the w element.
#include "vpl.glsl"

// Texture for albedo
layout(set = 2, binding = 0) uniform sampler2D tex;

// Material parameters as push constants. Push constants are faster than uniform
// buffers but Vulkan only requires a minimum size of 128 bytes for the block.
layout(push_constant) uniform PushConstantData {
    vec4 diffuse; // Multiplied by diffuse texture contents (alpha not used)
    float roughness;
    float metalness;
} material;

// Output
layout(location = 0) out vec4 frag;

const float PI = 3.1415926535897932;
const float ALMOST_ZERO = 0.00000001;

vec3 fresnel(vec3 H, vec3 V, vec3 F0) {
    float cos_theta = clamp(dot(H, V), 0.0, 1.0);
    return F0 + (1.0 - F0) * pow(1.0 - cos_theta, 5.0);
}

float distribution(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float d = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / max((PI * d * d), ALMOST_ZERO);
}

float geometry(float Ndot, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return Ndot / (Ndot * (1.0 - k) + k);
}

float attenuation(float distance) {
    // This inverse square law is physically correct but many prefer other
    // functions
    return 1.0 / (distance * distance);
}

void main() {
    // Clamp material uniforms to reasonable range
    float roughness = clamp(material.roughness, 0.1, 1.0);
    float metalness = clamp(material.metalness, 0.0, 1.0);

    // Calculate pre-multiply alpha
    vec4 colour_tex = texture(tex, f_tex_coord);
    float alpha = colour_tex.a;
    vec3 albedo = colour_tex.rgb * material.diffuse.rgb * colour_tex.a;

    // Use a standard value for reflectance for non-metals. Else use the albedo
    // value.
    vec3 F0 = mix(vec3(0.04), albedo, metalness);

    // Non light specific calculations
    vec3 N = normalize(f_normal);      // Normalized surface normal
    vec3 V = normalize(-f_position);   // Normalized position vector
    vec3 Lout = vec3(0.0);              // Initialize

    // Loop over lights
    for (int i = 0; i < 4; ++i) {
        vec3 L = normalize(vpl.lights[i].xyz - f_position);
        vec3 H = normalize(L + V);          // Half vector
        float NdotL = max(dot(N, L), 0.0);
        float NdotV = max(dot(N, V), 0.0);
        float distance = length(vpl.lights[i].xyz - f_position);
        vec3 radiance = vec3(vpl.lights[i].w * attenuation(distance));

        // Specular
        float NDF = distribution(N, H, roughness);
        float G = geometry(NdotL, roughness)
            * geometry(NdotV, roughness);
        vec3 F = fresnel(H, V, F0);
        vec3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, ALMOST_ZERO);

        // Diffuse
        vec3 kD = vec3(1.0) - F; // kS = F
        kD *= 1.0 - metalness; // Pure metal should have no diffuse

        // Add to total. There should be some multiply PI here somewhere but
        // since the input light level is just an arbitrary value it isn't
        // really necessary.
        #ifdef VISUALIZE
            if (specular_mode == NO_SPECULAR) {
                Lout += (kD * albedo / PI) * radiance * NdotL;
            } else if (specular_mode == SPECULAR_ONLY) {
                Lout += specular * radiance * NdotL;
            } else {
                Lout += (kD * albedo / PI + specular) * radiance * NdotL;
            }
        #else
            Lout += (kD * albedo / PI + specular) * radiance * NdotL;
        #endif
    }

    // Add some ambient. Very important for metals that have no diffuse.
    // Ambient light level is in last element of the ambient vector
    vec3 ambient = vpl.ambient.rgb * albedo * vpl.ambient.a;
    #ifdef VISUALIZE
        vec3 shade = Lout + ambient;
        if (ambient_mode == NO_AMBIENT) {
            shade = Lout;
        } else if (ambient_mode == AMBIENT_ONLY) {
            shade = ambient;
        }
    #else
        vec3 shade = Lout + ambient;
    #endif

    // Overrides
    #ifdef VISUALIZE
        vec3 rainbow[8] = {
            vec3(1.0, 0.0, 0.0), // R
            vec3(1.0, 0.5, 0.0), // O
            vec3(1.0, 1.0, 0.0), // Y
            vec3(0.0, 1.0, 0.0), // G
            vec3(0.0, 0.0, 1.0), // B
            vec3(0.3, 0.0, 0.5), // I
            vec3(1.0, 0.0, 1.0), // V
            vec3(0.0, 0.0, 0.0)  // K
        };
        switch (override_mode) {
            case SHOW_NORMALS:
                shade = vec3((f_normal + 1.0) * 0.5);
                break;
            case SHOW_ROUGHNESS:
                shade = vec3(roughness);
                break;
            case SHOW_METALNESS:
                shade = vec3(metalness);
                break;
            case SHOW_MIPMAP:
                int mml = clamp(int(textureQueryLod(tex, f_tex_coord).x), 0, 7);
                shade = rainbow[mml];
                break;
            case SHOW_DEPTH:
                const float NEAR = 0.2; // Default near plane
                const float FAR = 100.0; // Default far plane
                float depth = (NEAR * FAR /
                    (FAR + gl_FragCoord.z * (NEAR - FAR))) / FAR; // Linearized
                shade = vec3(depth);
                break;
            default:
                break;
        }
    #endif

    // Result is HDR and in linear space but let postprocess deal with that
    frag = vec4(shade, alpha);
}
