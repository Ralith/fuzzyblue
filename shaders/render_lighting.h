// Efficiently compute approximate illumination of a surface within the atmosphere
#ifndef FUZZYBLUE_RENDER_LIGHTING_H_
#define FUZZYBLUE_RENDER_LIGHTING_H_

#include "params.h"
#include "transmittance.h"
#include "irradiance.h"

// Returns direct illumination, last argument outputs indirect illumination
vec3 GetSunAndSkyIrradiance(
    AtmosphereParameters atmosphere,
    sampler2D transmittance_texture,
    sampler2D irradiance_texture,
    vec3 point, vec3 normal, vec3 sun_direction,
    out vec3 sky_irradiance) {
    float r = length(point);
    float mu_s = dot(point, sun_direction) / r;

    // Indirect irradiance (approximated if the surface is not horizontal).
    sky_irradiance = GetIrradiance(atmosphere, irradiance_texture, r, mu_s) *
        (1.0 + dot(normal, point) / r) * 0.5;

    // Direct irradiance.
    return atmosphere.solar_irradiance *
        GetTransmittanceToSun(
            atmosphere, transmittance_texture, r, mu_s) *
        max(dot(normal, sun_direction), 0.0);
}

#endif
