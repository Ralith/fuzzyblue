#version 450

/* From 2000 ASTM Standard Extraterrestrial Spectrum Reference */
const vec3 SOLAR_IRRADIANCE = vec3(1558, 1916, 2051) * 6e-3;

layout (location = 0) in vec2 screen;

layout (location = 0) out vec4 f_color;

layout (set = 0, binding = 0) uniform Params {
    float H_atm;
    float R_planet;
    float H_r;
    float H_m;
    vec3 beta_r;
    float beta_m;
};
layout (set = 0, binding = 1) uniform sampler3D scattering;

layout (push_constant) uniform Uniforms {
    mat4 inverse_viewproj;
    vec3 zenith;
    float height;
    vec3 sun_direction;
};

#include "mapping.h"

float phase_r(float cos_theta) {
    return 0.8 * (1.4 + 0.5 * cos_theta);
}

const float g = 0.76;

float phase_m(float cos_theta) {
    return (3 * (1 - g * g) / (2 * (2 + g * g)))
        * (1 + cos_theta * cos_theta) / pow(1 + g * g + - 2 * g * cos_theta, 1.5);
}

void main() {
    vec3 view = normalize((inverse_viewproj * vec4(2*screen - 1, 0, 1)).xyz);
    float cos_view = dot(view, zenith);
    float cos_sun = dot(sun_direction, zenith);
    vec3 coords = vec3(height_to_coord(height), cos_view_to_coord(height, cos_view), cos_sun_to_coord(cos_sun));
    vec4 value = texture(scattering, coords);
    vec3 rayleigh = value.rgb;
    vec3 mie;
    if (value.r < 0.0001) {
        mie = vec3(0);
    } else {
        mie = rayleigh * (value.a / value.r) * beta_r.r / beta_m * vec3(beta_m) / beta_r;
    }
    float cos_theta = dot(view, sun_direction);
    vec3 color = SOLAR_IRRADIANCE * (phase_r(cos_theta) * rayleigh + phase_m(cos_theta) * mie);
    f_color = vec4(color, 0);
}
