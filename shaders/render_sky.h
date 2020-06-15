// Efficiently compute incoming light from atmospheric scattering
#ifndef FUZZYBLUE_RENDER_SKY_H_
#define FUZZYBLUE_RENDER_SKY_H_

#include "params.h"
#include "transmittance.h"
#include "scattering.h"

vec3 GetExtrapolatedSingleMieScattering(
    AtmosphereParameters atmosphere, vec4 scattering) {
    // Algebraically this can never be negative, but rounding errors can produce that effect for
    // sufficiently short view rays.
    if (scattering.r <= 0.0) {
        return vec3(0.0);
    }
    return scattering.rgb * scattering.a / scattering.r *
        (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
        (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}

vec3 GetCombinedScattering(
    AtmosphereParameters atmosphere,
    sampler3D scattering_texture,
    float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground,
    out vec3 single_mie_scattering) {
    vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    float tex_coord_x = uvwz.x * float(atmosphere.scattering_texture_nu_size - 1);
    float tex_x = floor(tex_coord_x);
    float lerp = tex_coord_x - tex_x;
    vec3 uvw0 = vec3((tex_x + uvwz.y) / float(atmosphere.scattering_texture_nu_size),
                     uvwz.z, uvwz.w);
    vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(atmosphere.scattering_texture_nu_size),
                     uvwz.z, uvwz.w);
    vec4 combined_scattering = mix(
        texture(scattering_texture, uvw0),
        texture(scattering_texture, uvw1),
        lerp);
    single_mie_scattering =
        GetExtrapolatedSingleMieScattering(atmosphere, combined_scattering);
    return combined_scattering.rgb;
}

vec3 GetSkyRadiance(
    AtmosphereParameters atmosphere,
    sampler2D transmittance_texture,
    sampler3D scattering_texture,
    vec3 camera, vec3 view_ray,
    vec3 sun_direction, out vec3 transmittance) {
    // Compute the distance to the top atmosphere boundary along the view ray,
    // assuming the viewer is in space (or NaN if the view ray does not intersect
    // the atmosphere).
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top_atmosphere_boundary = -rmu -
        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
    // If the viewer is in space and the view ray intersects the atmosphere, move
    // the viewer to the top atmosphere boundary (along the view ray):
    if (distance_to_top_atmosphere_boundary > 0.0) {
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = atmosphere.top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    } else if (r > atmosphere.top_radius) {
        // If the view ray does not intersect the atmosphere, simply return 0.
        transmittance = vec3(1.0);
        // watt_per_square_meter_per_sr_per_nm
        return vec3(0.0);
    }
    // Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

    transmittance = ray_r_mu_intersects_ground ? vec3(0.0) :
        GetTransmittanceToTopAtmosphereBoundary(
            atmosphere, transmittance_texture, r, mu);
    vec3 single_mie_scattering;
    vec3 scattering;
    // if (shadow_length == 0.0 * m) {
    scattering = GetCombinedScattering(
        atmosphere, scattering_texture,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground,
        single_mie_scattering);
    // } else {
    //     // Case of light shafts (shadow_length is the total length noted l in our
    //     // paper): we omit the scattering between the camera and the point at
    //     // distance l, by implementing Eq. (18) of the paper (shadow_transmittance
    //     // is the T(x,x_s) term, scattering is the S|x_s=x+lv term).
    //     Length d = shadow_length;
    //     Length r_p =
    //         ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
    //     Number mu_p = (r * mu + d) / r_p;
    //     Number mu_s_p = (r * mu_s + d * nu) / r_p;

    //     scattering = GetCombinedScattering(
    //         atmosphere, scattering_texture, single_mie_scattering_texture,
    //         r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
    //         single_mie_scattering);
    //     DimensionlessSpectrum shadow_transmittance =
    //         GetTransmittance(atmosphere, transmittance_texture,
    //                          r, mu, shadow_length, ray_r_mu_intersects_ground);
    //     scattering = scattering * shadow_transmittance;
    //     single_mie_scattering = single_mie_scattering * shadow_transmittance;
    // }
    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
        MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

vec3 GetSkyRadianceToPoint(
    AtmosphereParameters atmosphere,
    sampler2D transmittance_texture,
    sampler3D scattering_texture,
    vec3 camera, vec3 view_ray, vec3 point,
    vec3 sun_direction, out vec3 transmittance) {
    // Compute the distance to the top atmosphere boundary along the view ray,
    // assuming the viewer is in space (or NaN if the view ray does not intersect
    // the atmosphere).
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top_atmosphere_boundary = -rmu -
        sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);

    // If the viewer is in space and the view ray intersects the atmosphere, move
    // the viewer to the top atmosphere boundary (along the view ray):
    if (distance_to_top_atmosphere_boundary > 0.0) {
        camera = camera + view_ray * distance_to_top_atmosphere_boundary;
        r = atmosphere.top_radius;
        rmu += distance_to_top_atmosphere_boundary;
    } else if (r > atmosphere.top_radius) {
        // If the view ray does not intersect the atmosphere, simply return 0.
        transmittance = vec3(1.0);
        // watt_per_square_meter_per_sr_per_nm
        return vec3(0.0);
    }

    // Compute the r, mu, mu_s and nu parameters for the first texture lookup.
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    float d = length(point - camera);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

    transmittance = GetTransmittance(atmosphere, transmittance_texture,
                                     r, mu, d, ray_r_mu_intersects_ground);

    vec3 single_mie_scattering;
    vec3 scattering = GetCombinedScattering(
        atmosphere, scattering_texture,
        r, mu, mu_s, nu, ray_r_mu_intersects_ground,
        single_mie_scattering);

    if (!isinf(d)) {
        // Compute the r, mu, mu_s and nu parameters for the second texture lookup.
        // If shadow_length is not 0 (case of light shafts), we want to ignore the
        // scattering along the last shadow_length meters of the view ray, which we
        // do by subtracting shadow_length from d (this way scattering_p is equal to
        // the S|x_s=x_0-lv term in Eq. (17) of our paper).
        // d = max(d - shadow_length, 0.0 * m);
        float r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
        float mu_p = (r * mu + d) / r_p;
        float mu_s_p = (r * mu_s + d * nu) / r_p;

        vec3 single_mie_scattering_p;
        vec3 scattering_p = GetCombinedScattering(
                                                  atmosphere, scattering_texture,
                                                  r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
                                                  single_mie_scattering_p);

        // Combine the lookup results to get the scattering between camera and point.
        vec3 shadow_transmittance = transmittance;
        // if (shadow_length > 0.0 * m) {
        //   // This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
        //   shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
        //       r, mu, d, ray_r_mu_intersects_ground);
        // }
        scattering = scattering - shadow_transmittance * scattering_p;
        single_mie_scattering =
            single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
        single_mie_scattering = GetExtrapolatedSingleMieScattering(
                                                                   atmosphere, vec4(scattering, single_mie_scattering.r));

        // Hack to avoid rendering artifacts when the sun is below the horizon.
        single_mie_scattering = single_mie_scattering *
            smoothstep(0.0, 0.01, mu_s);
    }

    return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
        MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}

#endif
