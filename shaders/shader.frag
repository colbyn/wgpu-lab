// shader.frag
#version 450
#define PI 3.1415926538
#define TAU 6.283185307

layout(location=0) in vec3 v_color;
layout(location=0) out vec4 f_color;
layout(binding=0) uniform Uniforms {
    vec2 viewport;
    float time;
};


float linear_scale(
    vec2 domain,
    vec2 codomain,
    float input_value
) {
    const float min_input = domain.x;
    const float max_input = domain.y;
    const float min_output = codomain.x;
    const float max_output = codomain.y;
    return (
          (max_output - min_output)
        * (input_value - min_input)
        / (max_input - min_input)
        + min_output);
}

#define SCREEN_TO_PCT_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, 1.0), a)
#define PCT_TO_WIDTH_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, viewport.x), a)
#define PCT_TO_HEIGHT_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, viewport.y), a)

vec4 cubic_bezier(vec4 A, vec4 B, vec4 C, vec4 D, float t) {
    vec4 E = mix(A, B, t);
    vec4 F = mix(B, C, t);
    vec4 G = mix(C, D, t);

    vec4 H = mix(E, F, t);
    vec4 I = mix(F, G, t);

    vec4 P = mix(H, I, t);

    return P;
}

void main() {
    const float max_width = viewport.x;
    const float max_height = viewport.y;
    const float topLeft = max_height/max_width;
    const float x_pos = gl_FragCoord.x;
    const float y_pos = gl_FragCoord.y;
    const vec2 st = gl_FragCoord.xy/viewport;

    if (st.x >= 0.5) {
        f_color = vec4(abs(sin(time)), 0.0, 0.0, 1.0);
        return;
    }

    // DEFAULT
    f_color = vec4(0.0, 0.0, 0.0, 0.0);
}

