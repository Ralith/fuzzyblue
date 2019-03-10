// View of the sky from inside the atmosphere
#version 450

layout (location = 0) in vec2 screen;

layout (location = 0) out vec4 f_color;

#include "params.h"

layout (set = 0, binding = 1) uniform sampler3D scattering;
layout (set = 0, binding = 2) uniform sampler2D transmittance;
layout (set = 1, input_attachment_index = 0, binding = 0) uniform subpassInput depth;

#include "inscattering.h"

#include "draw_params.h"

void main() {
    vec3 view = normalize((inverse_viewproj * vec4(2*screen - 1, 0, 1)).xyz);
    vec4 iabtab = inscattering(scattering, view, zenith, height, sun_direction);
    vec4 world_pre = (inverse_viewproj * vec4(2*screen - 1, subpassLoad(depth).x, 1));
    vec3 world = world_pre.xyz / world_pre.w;
    vec3 world_zenith = normalize(world);
    float world_height = length(world) - R_planet;
    vec4 isbtsb = inscattering(scattering, view, world_zenith, world_height, sun_direction);
    vec2 low;
    vec2 mid;
    if (height > world_height) { // FIXME: Numerically unstable comparison, produces flickering!
        low = vec2(height_to_coord(world_height), cos_view_to_coord(world_height, dot(world_zenith, -view)));
        mid = vec2(height_to_coord(height), cos_view_to_coord(height, dot(zenith, -view)));
    } else {
        low = vec2(height_to_coord(height), cos_view_to_coord(height, dot(zenith, view)));
        mid = vec2(height_to_coord(world_height), cos_view_to_coord(world_height, dot(world_zenith, view)));
    }
    vec3 t_low_sky = texture(transmittance, low).rgb;
    // Special case needed until there's an atmosphere backface pass
    vec3 t_mid_sky = abs(world_pre.w) > 0 ? texture(transmittance, mid).rgb : vec3(1);
    vec3 tas = t_low_sky / t_mid_sky;
    vec4 iastas = max(iabtab - tas.rgbr * isbtsb, vec4(0));
    vec3 color = solar_irradiance * inscattering_ratios(vec4(iastas.rgb, float(world_height > H_atm) * iastas.a), view, sun_direction, mie_anisotropy);
    f_color = vec4(color, 0);
}
