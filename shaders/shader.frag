// shader.frag
#version 450

layout(location=0) in vec3 v_color;
layout(location=0) out vec4 f_color;
layout(set=0, binding=0) uniform Uniforms {
    vec2 viewport;
};


void main() {
    vec2 frame = gl_FragCoord.xy/viewport;
    if (frame.x <= 0.5) {
        f_color = vec4(
              0.0,
              1.0,
              0.0,
              0.7
          );
          return;
    }
    f_color = vec4(v_color, 0.9);
}
 
