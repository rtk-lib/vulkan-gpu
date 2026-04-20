#version 460

layout(location = 0) in  vec2 frag_uv;
layout(location = 0) out vec4 out_color;

layout(binding = 0) uniform sampler2D render_tex;

void main() {
    out_color = texture(render_tex, frag_uv);
}
