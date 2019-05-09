#ifndef FUZZYBLUE_UTIL_H_
#define FUZZYBLUE_UTIL_H_

const float PI = 3.14159265358979323846;

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float ClampDistance(float d) {
    return max(d, 0.0);
}

float SafeSqrt(float area) {
    return sqrt(max(area, 0.0));
}

float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

float RayleighPhaseFunction(float nu) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + nu * nu);
}

float MiePhaseFunction(float g, float nu) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

float GetFragCoordFromTexel(uint x, uint texture_size) {
    return texture_size * GetTextureCoordFromUnitRange(x / float(texture_size - 1), int(texture_size));
}

vec3 GetFragCoordFromTexel(uvec3 v, uvec3 size) {
    return vec3(
        GetFragCoordFromTexel(v.x, size.x),
        GetFragCoordFromTexel(v.y, size.y),
        GetFragCoordFromTexel(v.z, size.z));
}

#endif
