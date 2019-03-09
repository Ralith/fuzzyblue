float height_to_coord(float h) { return pow(h/H_atm, 0.5); }
float coord_to_height(float u_h) { return max(u_h * u_h * H_atm, 0.1); }

float cos_view_to_coord(float height, float c_v) {
    const float c_h = -sqrt(height * (2 * R_planet + height)) / (R_planet + height);
    if (c_v > c_h) {
        return 0.5 * pow((c_v - c_h) / (1 - c_h), 0.2) + 0.5;
    } else {
        return 0.5 - (0.5 * pow((c_h - c_v) / (c_h + 1), 0.2));
    }
}
float coord_to_cos_view(float height, float u_v) {
    const float c_h = -sqrt(height * (2 * R_planet + height)) / (R_planet + height);
    float result;
    if (u_v > 0.5) {
        result = c_h + pow(2*u_v - 1, 5) * (1 - c_h);
    } else {
        result = c_h - pow(2*(0.5 - u_v), 5) * (1 + c_h);
    }
    return clamp(result, -1, 1);
}

float cos_sun_to_coord(float c_s) {
    return 0.5 * (atan(max(c_s, -0.1975) * tan(1.26 * 1.1)) / 1.1 + (1 - 0.26));
}
float coord_to_cos_sun(float u_s) {
    return clamp(tan((2 * u_s - 1 + 0.26) * 0.75) / tan(1.26 * 0.75), -1, 1);
}
