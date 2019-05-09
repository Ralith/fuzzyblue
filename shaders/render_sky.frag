#version 450

#include "params.h"
#include "render_sky.h"

layout (location=0) in vec2 screen_coords;

layout (location=0, index=0) out vec4 color_out;
layout (location=0, index=1) out vec4 transmittance_out;

layout (set=0, binding=0) uniform Params {
    AtmosphereParameters atmosphere;
};
layout (set=0, binding=1) uniform sampler2D transmittance_texture;
layout (set=0, binding=2) uniform sampler3D scattering_texture;
layout (push_constant) uniform DrawParams {
    mat4 inverse_viewproj;
    vec3 camera_position;
    vec3 sun_direction;
};

void main() {
    vec3 view = normalize((inverse_viewproj * vec4(2*screen_coords - 1, 0, 1)).xyz);
    vec3 transmittance;
    // TODO: ...ToPoint using depth buffer
    vec3 color = GetSkyRadiance(
        atmosphere, transmittance_texture, scattering_texture,
        camera_position, view, sun_direction,
        transmittance);
    color_out = vec4(color, 0);
    transmittance_out = vec4(transmittance, 1);
}
