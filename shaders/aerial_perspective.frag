#version 450

#include "aerial_perspective.h"
#include "params.h"

layout (location = 0) in vec2 screen;
layout (location = 0, index = 0) out vec4 f_inscattering;
layout (location = 0, index = 1) out vec4 f_transmittance;

layout (set = 1, binding = 0, input_attachment_index = 0) uniform subpassInput depth;
layout (set = 1, binding = 1) uniform sampler3D scattering;
layout (set = 1, binding = 2) uniform sampler3D transmittance;

#include "inscattering.h"
#include "draw_params.h"

void main() {
    vec3 camera = zenith * (R_planet + height);
    vec4 world_pre = (inverse_viewproj * vec4(2*screen - 1, subpassLoad(depth).x, 1));
    vec3 world = world_pre.xyz / world_pre.w;
    vec3 wrt_camera = world - camera;
    vec3 camera_forwards = (inverse_viewproj * vec4(0, 0, 0, 1)).xyz;
    // Project ray from camera to fragment onto unit forwards vector to obtain viewspace Z
    // a onto b = (a.b)/|b|; |b| = 1
    float z = dot(wrt_camera, camera_forwards);

    vec3 coords = vec3(screen, z_to_coord(max_ap_depth, z));
    vec4 value = texture(scattering, coords);
    vec3 rayleigh = value.rgb;
    vec3 mie;
    if (value.r < 0.0001) {
        mie = vec3(0);
    } else {
        mie = rayleigh * (value.a / value.r) * beta_r.r / beta_m * vec3(beta_m) / beta_r;
    }
    vec3 view = normalize(wrt_camera);
    float cos_theta = dot(view, sun_direction);
    vec3 s = phase_r(cos_theta) * rayleigh + phase_m(cos_theta, mie_anisotropy) * mie;
    f_inscattering = vec4(solar_irradiance * s, 0);
    f_transmittance = texture(transmittance, coords);
}
