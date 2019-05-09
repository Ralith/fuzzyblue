#ifndef TRANSMITTANCE_H_
#define TRANSMITTANCE_H_

#include "util.h"
#include "params.h"

vec2 GetTransmittanceTextureUvFromRMu(AtmosphereParameters atmosphere, float r, float mu) {
    // assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
    // assert(mu >= -1.0 && mu <= 1.0);
    // Distance to top atmosphere boundary for a horizontal ray at ground level.
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
                   atmosphere.bottom_radius * atmosphere.bottom_radius);
    // Distance to the horizon.
    float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
    // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
    float d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
    float d_min = atmosphere.top_radius - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    return vec2(GetTextureCoordFromUnitRange(x_mu, atmosphere.transmittance_texture_mu_size),
                GetTextureCoordFromUnitRange(x_r, atmosphere.transmittance_texture_r_size));
}

vec3 GetTransmittanceToTopAtmosphereBoundary(
    AtmosphereParameters atmosphere,
    sampler2D transmittance_texture,
    float r, float mu) {
    // assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
    vec2 uv = GetTransmittanceTextureUvFromRMu(atmosphere, r, mu);
    return texture(transmittance_texture, uv).rgb;
}

vec3 GetTransmittance(
    AtmosphereParameters atmosphere,
    sampler2D transmittance_texture,
    float r, float mu, float d, bool ray_r_mu_intersects_ground) {
    // assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
    // assert(mu >= -1.0 && mu <= 1.0);
    // assert(d >= 0.0 * m);

    float r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);

    vec3 quotient;
    if (ray_r_mu_intersects_ground) {
        quotient =
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, r_d, -mu_d) /
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, r, -mu);
    } else {
        quotient =
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, r, mu) /
            GetTransmittanceToTopAtmosphereBoundary(
                atmosphere, transmittance_texture, r_d, mu_d);
    }
    return min(quotient, vec3(1.0));
}

vec3 GetTransmittanceToSun(
    AtmosphereParameters atmosphere,
    sampler2D transmittance_texture,
    float r, float mu_s) {
    float sin_theta_h = atmosphere.bottom_radius / r;
    float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    return GetTransmittanceToTopAtmosphereBoundary(
        atmosphere, transmittance_texture, r, mu_s) *
        smoothstep(-sin_theta_h * atmosphere.sun_angular_radius,
                   sin_theta_h * atmosphere.sun_angular_radius,
                   mu_s - cos_theta_h);
}

#endif
