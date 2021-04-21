// shader.vert
#version 450

layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_color;
layout(location=0) out vec3 v_color;

// uniform Uniforms {
//     vec2 viewport;
// };

void main() {
    v_color = a_color;
    v_color.z = 0.9;
    gl_Position = vec4(a_position, 1.0);
}
