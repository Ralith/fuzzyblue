// View of the sky from inside the atmosphere
#version 450

layout (location = 0) in vec2 screen;

layout (location = 0) out vec4 f_color;

#include "params.h"

layout (set = 0, binding = 1) uniform sampler3D scattering;

#include "inscattering.h"

#include "draw_params.h"

void main() {
    vec3 view = normalize((inverse_viewproj * vec4(2*screen - 1, 0, 1)).xyz);
    vec3 color = solar_irradiance * inscattering(scattering, view, zenith, height, sun_direction, mie_anisotropy);
    f_color = vec4(color, 0);
}
