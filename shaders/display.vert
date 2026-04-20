#version 460

// Fullscreen triangle — no vertex buffer needed
const vec2 POSITIONS[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);
const vec2 TEXCOORDS[3] = vec2[](
    vec2(0.0, 0.0),
    vec2(2.0, 0.0),
    vec2(0.0, 2.0)
);

layout(location = 0) out vec2 frag_uv;

void main() {
    frag_uv     = TEXCOORDS[gl_VertexIndex];
    gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
}
