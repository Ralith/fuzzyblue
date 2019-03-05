#version 450

layout(constant_id = 0) const float DEPTH = 1.0;

layout(location = 0) out vec2 texcoords;

void main()  {
    texcoords = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(texcoords * 2.0f + -1.0f, 0.0f, DEPTH);
}
