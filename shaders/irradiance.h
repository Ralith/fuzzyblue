#ifndef FUZZYBLUE_IRRADIANCE_H_
#define FUZZYBLUE_IRRADIANCE_H_

#include "params.h"
#include "util.h"

void GetRMuSFromIrradianceUnitRange(
    AtmosphereParameters atmosphere,
    float x_mu_s, float x_r,
    out float r, out float mu_s) {
    // assert(uv.x >= 0.0 && uv.x <= 1.0);
    // assert(uv.y >= 0.0 && uv.y <= 1.0);
    // Number x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    // Number x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    r = atmosphere.bottom_radius +
        x_r * (atmosphere.top_radius - atmosphere.bottom_radius);
    mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
}

vec2 GetIrradianceTextureUvFromRMuS(
    AtmosphereParameters atmosphere,
    float r, float mu_s) {
    // assert(r >= atmosphere.bottom_radius && r <= atmosphere.top_radius);
    // assert(mu_s >= -1.0 && mu_s <= 1.0);
    float x_r = (r - atmosphere.bottom_radius) /
        (atmosphere.top_radius - atmosphere.bottom_radius);
    float x_mu_s = mu_s * 0.5 + 0.5;
    return vec2(GetTextureCoordFromUnitRange(x_mu_s, atmosphere.irradiance_texture_mu_s_size),
                GetTextureCoordFromUnitRange(x_r, atmosphere.irradiance_texture_r_size));
}

vec3 GetIrradiance(
    AtmosphereParameters atmosphere,
    sampler2D irradiance_texture,
    float r, float mu_s) {
    vec2 uv = GetIrradianceTextureUvFromRMuS(atmosphere, r, mu_s);
    return texture(irradiance_texture, uv).rgb;
}

#endif
