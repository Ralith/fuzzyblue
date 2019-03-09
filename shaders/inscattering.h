#include "mapping.h"

float phase_r(float cos_theta) {
    return 0.8 * (1.4 + 0.5 * cos_theta);
}

float phase_m(float cos_theta, float g) {
    return (3 * (1 - g * g) / (2 * (2 + g * g)))
        * (1 + cos_theta * cos_theta) / pow(1 + g * g - 2 * g * cos_theta, 1.5);
}

vec3 inscattering(sampler3D lut, vec3 view, vec3 zenith, float height, vec3 sun_direction, float g) {
    float cos_view = dot(view, zenith);
    float cos_sun = dot(sun_direction, zenith);
    vec3 coords = vec3(height_to_coord(height), cos_view_to_coord(height, cos_view), cos_sun_to_coord(cos_sun));
    vec4 value = texture(lut, coords);
    vec3 rayleigh = value.rgb;
    vec3 mie;
    if (value.r < 0.0001) {
        mie = vec3(0);
    } else {
        mie = rayleigh * (value.a / value.r) * beta_r.r / beta_m * vec3(beta_m) / beta_r;
    }
    float cos_theta = dot(view, sun_direction);
    return phase_r(cos_theta) * rayleigh + phase_m(cos_theta, g) * mie;
}
