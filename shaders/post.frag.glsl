#version 450

layout(location = 0) in vec2 f_tex_coord;
layout(location = 0) out vec4 frag;

layout(set = 0, binding = 0) uniform sampler2D tex;

// Some Filmic things
const mat3 ACESInputMat = mat3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777);

const mat3 ACESOutputMat = mat3(
    1.60475, -0.53108, -0.07367,
    -0.10208, 1.10813, -0.00605,
    -0.00327, -0.07276, 1.07602);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

void main() {
    // Sample is in HDR linear. Tonemap for dynamic range but let output surface
    // take care of sRGB conversion.
    vec3 colour = texture(tex, f_tex_coord).xyz;

    // Debug test to see if HDR is being used
    //if (colour.r > 1.0) colour.r = 0.0;
    //if (colour.g > 1.0) colour.g = 0.0;
    //if (colour.b > 1.0) colour.b = 0.0;

    // Tonemapping options:
    // Reinhard
    // Apparently this should really be done in CIE xyY colour space to
    // prevent desaturating
    //colour = colour / (colour + 1.0);

    // Reinhard2
    //colour = (colour * (1.0 + colour / 16.0)) / (colour + 1.0);

    // Filmic (origin?)
    // This one was really bright compared to the others so the exposure
    // adjustment was added but has not been tuned
    //colour = colour * 0.4; // Arbitrary exposure adjustment
    //vec3 X = max(vec3(0.0), colour - 0.004);
    //colour = (X * (6.2 * X + 0.5)) / (X * (6.2 * X + 1.7) + 0.06);

    // Simplified ACES Filmic, Narkowicz 2015
    // Notes say it is "a very simple luminance only fit, which over saturates
    // brights. This was actually something consistent with our art direction."
    // Exposure adjustment arbitrarily adjusts closer to Reinhard brightness
    colour = colour * 0.8; // Arbitrary exposure adjustment
    colour = clamp((colour * (2.51 * colour + 0.03)) /
        (colour * (2.43 * colour + 0.59) + 0.14), 0.0, 1.0);

    // ACES Flimic, Steven Hill
    // Converted from HLSL. Original notes on the matrix mention sRGB and other
    // things but these are the colour gamut and not the display transform
    // (which is non linear and so couldn't be in a matrix).
    // Exposure adjustment gives this one a boost to be closer to ACES in
    // brightness. It does appear less saturated. Whether it is worth the extra
    // processing is TBD.
    //colour = colour * 1.3; // Arbitrary exposure adjustment
    //colour = colour * ACESInputMat;
    //colour = RRTAndODTFit(colour);
    //colour = colour * ACESOutputMat;
    //colour = clamp(colour, 0.0, 1.0);

    frag = vec4(colour, 1.0);
}
