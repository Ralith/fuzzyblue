// View from space
#version 450

layout (location = 0) in vec2 screen;

layout (location = 0) out vec4 f_color;

#include "params.h"

layout (set = 0, binding = 1) uniform sampler3D scattering;
layout (set = 1, input_attachment_index = 0, binding = 0) uniform subpassInput depth;

#include "inscattering.h"

#include "draw_params.h"

bool ray_sphere(vec3 pos, vec3 dir, float radius, out float t) {
    float a = dot(dir, dir);
    float b = 2 * dot(pos, dir);
    float c = dot(pos, pos) - radius*radius;
    float delta = b*b - 4*a*c;
    if (delta < 0) { return false; }
    float t1 = (-b + sqrt(delta)) / (2 * a);
    t = (-b - sqrt(delta)) / (2 * a);
    //t = min(t1, t2);
    return t > 0;
}

void main() {
    vec3 view = normalize((inverse_viewproj * vec4(2*screen - 1, 0, 1)).xyz);
    float t;
    vec3 camera_pos = zenith * (R_planet + height);
    // TODO: approximate atmosphere limits with rasterization
    if (ray_sphere(camera_pos, view, R_planet + H_atm, t)) {
        vec3 hit_pos = camera_pos + t * view;
        vec3 hit_zenith = normalize(hit_pos);
        vec3 s = solar_irradiance * inscattering(scattering, view, hit_zenith, H_atm, sun_direction, mie_anisotropy);
        f_color = vec4(s, 1);
    } else {
        f_color = vec4(0);
    }
    // TODO: Transmittance:
    // 1. rasterize icosphere depth-only with frontfaces culled
    // 2. read depth to reconstruct zenith + height
    // 3. transmittance from LUT based on (height, dot(zenith, view))
}
