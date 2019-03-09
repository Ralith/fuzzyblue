layout (push_constant) uniform Uniforms {
    mat4 inverse_viewproj;
    vec3 zenith;
    float height;
    vec3 sun_direction;
    float mie_anisotropy;
    vec3 solar_irradiance;
};
