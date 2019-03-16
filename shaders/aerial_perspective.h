const int AP_DEPTH_EXPONENT = 2;

float coord_to_z(float max_z, float coord) {
    return max_z * pow(coord, AP_DEPTH_EXPONENT);
}

float z_to_coord(float max_z, float z) {
    return pow(z / max_z, 1.0/AP_DEPTH_EXPONENT);
}
