#ifndef FUZZYBLUE_SCATTERING_H_
#define FUZZYBLUE_SCATTERING_H_

#include "params.h"

vec4 GetScatteringTextureUvwzFromRMuMuSNu(AtmosphereParameters atmosphere,
                                          float r, float mu, float mu_s, float nu,
                                          bool ray_r_mu_intersects_ground) {
    // assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
    // assert(mu >= -1.0 && mu <= 1.0);
    // assert(mu_s >= -1.0 && mu_s <= 1.0);
    // assert(nu >= -1.0 && nu <= 1.0);

    // Distance to top atmosphere boundary for a horizontal ray at ground level.
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
                   atmosphere.bottom_radius * atmosphere.bottom_radius);
    // Distance to the horizon.
    float rho =
        SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    float u_r = GetTextureCoordFromUnitRange(rho / H, atmosphere.scattering_texture_r_size);

    // Discriminant of the quadratic equation for the intersections of the ray
    // (r,mu) with the ground (see RayIntersectsGround).
    float r_mu = r * mu;
    float discriminant =
        r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
    float u_mu;
    if (ray_r_mu_intersects_ground) {
        // Distance to the ground for the ray (r,mu), and its minimum and maximum
        // values over all mu - obtained for (r,-1) and (r,mu_horizon).
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - atmosphere.bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
                                                        (d - d_min) / (d_max - d_min), atmosphere.scattering_texture_mu_size / 2);
    } else {
        // Distance to the top atmosphere boundary for the ray (r,mu), and its
        // minimum and maximum values over all mu - obtained for (r,1) and
        // (r,mu_horizon).
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = atmosphere.top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d - d_min) / (d_max - d_min), atmosphere.scattering_texture_mu_size / 2);
    }

    float d = DistanceToTopAtmosphereBoundary(
        atmosphere, atmosphere.bottom_radius, mu_s);
    float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float A =
        -2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
    float u_mu_s = GetTextureCoordFromUnitRange(
        max(1.0 - a / A, 0.0) / (1.0 + a), atmosphere.scattering_texture_mu_s_size);

    float u_nu = (nu + 1.0) / 2.0;
    return vec4(u_nu, u_mu_s, u_mu, u_r);
}

void GetRMuMuSNuFromScatteringTextureUvwz(AtmosphereParameters atmosphere,
                                          vec4 uvwz, out float r, out float mu, out float mu_s,
                                          out float nu, out bool ray_r_mu_intersects_ground) {
    // assert(uvwz.x >= 0.0 && uvwz.x <= 1.0);
    // assert(uvwz.y >= 0.0 && uvwz.y <= 1.0);
    // assert(uvwz.z >= 0.0 && uvwz.z <= 1.0);
    // assert(uvwz.w >= 0.0 && uvwz.w <= 1.0);

    // Distance to top atmosphere boundary for a horizontal ray at ground level.
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
                   atmosphere.bottom_radius * atmosphere.bottom_radius);
    // Distance to the horizon.
    float rho =
        H * GetUnitRangeFromTextureCoord(uvwz.w, atmosphere.scattering_texture_r_size);
    r = sqrt(rho * rho + atmosphere.bottom_radius * atmosphere.bottom_radius);

    if (uvwz.z < 0.5) {
        // Distance to the ground for the ray (r,mu), and its minimum and maximum
        // values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
        // we can recover mu:
        float d_min = r - atmosphere.bottom_radius;
        float d_max = rho;
        float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            1.0 - 2.0 * uvwz.z, atmosphere.scattering_texture_mu_size / 2);
        mu = d == 0.0 ? -1.0 :
            ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = true;
    } else {
        // Distance to the top atmosphere boundary for the ray (r,mu), and its
        // minimum and maximum values over all mu - obtained for (r,1) and
        // (r,mu_horizon) - from which we can recover mu:
        float d_min = atmosphere.top_radius - r;
        float d_max = rho + H;
        float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            2.0 * uvwz.z - 1.0, atmosphere.scattering_texture_mu_size / 2);
        mu = d == 0.0 ? 1.0 :
            ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = false;
    }

    float x_mu_s =
        GetUnitRangeFromTextureCoord(uvwz.y, atmosphere.scattering_texture_mu_s_size);
    float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    float d_max = H;
    float A =
        -2.0 * atmosphere.mu_s_min * atmosphere.bottom_radius / (d_max - d_min);
    float a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
    float d = d_min + min(a, A) * (d_max - d_min);
    mu_s = d == 0.0 ? 1.0 :
        ClampCosine((H * H - d * d) / (2.0 * atmosphere.bottom_radius * d));

    nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

void GetRMuMuSNuFromScatteringTextureFragCoord(
    AtmosphereParameters atmosphere, vec3 frag_coord,
    out float r, out float mu, out float mu_s, out float nu,
    out bool ray_r_mu_intersects_ground) {
    const vec4 SCATTERING_TEXTURE_SIZE = vec4(
        atmosphere.scattering_texture_nu_size - 1,
        atmosphere.scattering_texture_mu_s_size,
        atmosphere.scattering_texture_mu_size,
        atmosphere.scattering_texture_r_size);
    float frag_coord_nu =
        floor(frag_coord.x / float(atmosphere.scattering_texture_mu_s_size));
    float frag_coord_mu_s =
        mod(frag_coord.x, float(atmosphere.scattering_texture_mu_s_size));
    vec4 uvwz =
        vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /
        SCATTERING_TEXTURE_SIZE;
    GetRMuMuSNuFromScatteringTextureUvwz(
        atmosphere, uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    // Clamp nu to its valid range of values, given mu and mu_s.
    nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
               mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

vec3 GetScattering(
    AtmosphereParameters atmosphere,
    sampler3D scattering_texture,
    float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground) {
    vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
        atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    float tex_coord_x = uvwz.x * float(atmosphere.scattering_texture_nu_size - 1);
    float tex_x = floor(tex_coord_x);
    float lerp = tex_coord_x - tex_x;
    vec3 uvw0 = vec3((tex_x + uvwz.y) / float(atmosphere.scattering_texture_nu_size),
                     uvwz.z, uvwz.w);
    vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(atmosphere.scattering_texture_nu_size),
                     uvwz.z, uvwz.w);
    return vec3(texture(scattering_texture, uvw0) * (1.0 - lerp) +
                texture(scattering_texture, uvw1) * lerp);
}

vec3 GetScattering(
    AtmosphereParameters atmosphere,
    sampler3D single_rayleigh_scattering_texture,
    sampler3D single_mie_scattering_texture,
    sampler3D multiple_scattering_texture,
    float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground,
    int scattering_order) {
    if (scattering_order == 1) {
        vec3 rayleigh = GetScattering(
            atmosphere, single_rayleigh_scattering_texture, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
        vec3 mie = GetScattering(
            atmosphere, single_mie_scattering_texture, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
        return rayleigh * RayleighPhaseFunction(nu) +
            mie * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
    } else {
        return GetScattering(
            atmosphere, multiple_scattering_texture, r, mu, mu_s, nu,
            ray_r_mu_intersects_ground);
    }
}

#endif
