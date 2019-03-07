const int STEPS = 30;

#include "params.h"

float density_r(float h) { return exp(-h/H_r); }
float density_m(float h) { return exp(-h/H_m); }

float point_height(vec2 p) { return length(p) - R_planet; }

#include "mapping.h"

const float infinity = 1. / 0.;

float ray_circle(vec2 start, vec2 dir, float radius, bool nearest) {
    float c = dot(start, start) - (radius*radius);
    float b = dot(dir, start);
    float d = b*b - c;
    if (d < 0.0) return infinity;
    float t0 = -b - sqrt(d);
    float t1 = -b + sqrt(d);
    float ta = min(t0, t1);
    float tb = max(t0, t1);
    if (tb < 0.0) { return infinity; }
    else if (nearest) { return ta > 0.0 ? ta : tb; }
    else { return tb; }
}

vec2 intersection(vec2 start, vec2 dir) {
    float t = ray_circle(start, dir, R_planet, true);
    if (isinf(t)) { t = ray_circle(start, dir, R_planet + H_atm, false); }
    if (isinf(t)) { t = 0; }
    return start + t * dir;
}

vec2 cos_view_dir(float cos_view) {
    return vec2(cos_view, sqrt(1 - cos_view * cos_view));
}

const float pi = 3.1415926538;
