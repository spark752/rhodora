#version 450

// Basic sort of fragment shader for PBR lighting
// Based in part on learnopengl.com and on "Moving Frostbite to Physically
// Based Rendering" by Legarde and de Rousiers, SIGGRAPH 2014

layout(location = 0) in vec3 f_normal;
layout(location = 1) in vec3 f_position;
layout(location = 2) in vec2 f_tex_coord;

// Matrices are used by vertex shader but are part of the same uniform buffer.
// The lights array currently contains point lights with each position in view
// space in the xyz elements and its intensity in the w element.
layout(set = 0, binding = 0) uniform VPL {
    mat4 view;
    mat4 proj;
    vec4 ambient;
    vec4 lights[4];
} vpl;

layout(set = 2, binding = 0) uniform sampler2D tex;

// Material parameters
layout(push_constant) uniform PushConstantData {
    vec4 diffuse; // Multiplied by diffuse texture contents (alpha not used)
    float roughness;
    float metalness;
} material;

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

    // Ambient light level is in last element of the ambient vector
    float amb_level = vpl.ambient.w;

    vec4 colour_tex = texture(tex, f_tex_coord);
    float alpha = colour_tex.a;
    vec3 albedo = vec3(colour_tex.rgb) * vec3(material.diffuse.rgb);

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
        Lout += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Add some ambient. Very important for metals that have no diffuse.
    vec3 ambient = amb_level * albedo;
    vec3 shade = Lout + ambient;

    // Result is HDR and in linear space but let postprocess deal with that
    frag = vec4(shade, alpha);
}
