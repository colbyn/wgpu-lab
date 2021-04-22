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

// const vec2 u_resolution = viewport;
const float u_time = TAU * 0.25;

float quadraticBezier(float x, vec2 a){
    float epsilon = 0.00001;
    a.x = clamp(a.x,0.0,1.0);
    a.y = clamp(a.y,0.0,1.0);
    if (a.x == 0.5){
        a += epsilon;
    }

    // solve t from x (an inverse operation)
    float om2a = 1.0 - 2.0 * a.x;
    float t = (sqrt(a.x*a.x + om2a*x) - a.x)/om2a;
    float y = (1.0-2.0*a.y)*(t*t) + (2.0*a.y)*t;
    return y;
}



// Helper functions:
float slopeFromT (float t, float A, float B, float C){
    float dtdx = 1.0/(3.0*A*t*t + 2.0*B*t + C); 
    return dtdx;
}
float xFromT (float t, float A, float B, float C, float D){
    float x = A*(t*t*t) + B*(t*t) + C*t + D;
    return x;
}
float yFromT (float t, float E, float F, float G, float H){
    float y = E*(t*t*t) + F*(t*t) + G*t + H;
    return y;
}
float B0 (float t){
    return (1.0-t)*(1.0-t)*(1.0-t);
}
float B1 (float t){
    return  3.0*t*(1.0-t)*(1.0-t);
}
float B2 (float t){
    return 3.0*t*t* (1.0-t);
}
float B3 (float t){
    return t*t*t;
}
float findx (float t, float x0, float x1, float x2, float x3){
    return x0*B0(t) + x1*B1(t) + x2*B2(t) + x3*B3(t);
}
float findy (float t, float y0, float y1, float y2, float y3){
    return y0*B0(t) + y1*B1(t) + y2*B2(t) + y3*B3(t);
}

float cubicBezier(float x, vec2 a, vec2 b){
    float y0a = 0.0; // initial y
    float x0a = 0.0; // initial x 
    float y1a = a.y;    // 1st influence y   
    float x1a = a.x;    // 1st influence x 
    float y2a = b.y;    // 2nd influence y
    float x2a = b.x;    // 2nd influence x
    float y3a = 1.0; // final y 
    float x3a = 1.0; // final x 

    float A =   x3a - 3.0*x2a + 3.0*x1a - x0a;
    float B = 3.0*x2a - 6.0*x1a + 3.0*x0a;
    float C = 3.0*x1a - 3.0*x0a;   
    float D =   x0a;

    float E =   y3a - 3.0*y2a + 3.0*y1a - y0a;    
    float F = 3.0*y2a - 6.0*y1a + 3.0*y0a;             
    float G = 3.0*y1a - 3.0*y0a;             
    float H =   y0a;

    // Solve for t given x (using Newton-Raphelson), then solve for y given t.
    // Assume for the first guess that t = x.
    float currentt = x;
    for (int i=0; i < 5; i++){
        float currentx = xFromT (currentt, A,B,C,D); 
        float currentslope = slopeFromT (currentt, A,B,C);
        currentt -= (currentx - x)*(currentslope);
        currentt = clamp(currentt,0.0,1.0); 
    } 

    float y = yFromT (currentt,  E,F,G,H);
    return y;
}

float cubicBezierNearlyThroughTwoPoints(float x, vec2 a, vec2 b){
    float y = 0.0;
    float epsilon = 0.00001;
    float min_param_a = 0.0 + epsilon;
    float max_param_a = 1.0 - epsilon;
    float min_param_b = 0.0 + epsilon;
    float max_param_b = 1.0 - epsilon;
    a.x = max(min_param_a, min(max_param_a, a.x));
    a.y = max(min_param_b, min(max_param_b, a.y));

    float x0 = 0.0;  
    float y0 = 0.0;
    float x4 = a.x;  
    float y4 = a.y;
    float x5 = b.x;  
    float y5 = b.y;
    float x3 = 1.0;  
    float y3 = 1.0;
    float x1,y1,x2,y2; // to be solved.

    // arbitrary but reasonable 
    // t-values for interior control points
    float t1 = 0.3;
    float t2 = 0.7;

    float B0t1 = B0(t1);
    float B1t1 = B1(t1);
    float B2t1 = B2(t1);
    float B3t1 = B3(t1);
    float B0t2 = B0(t2);
    float B1t2 = B1(t2);
    float B2t2 = B2(t2);
    float B3t2 = B3(t2);

    float ccx = x4 - x0*B0t1 - x3*B3t1;
    float ccy = y4 - y0*B0t1 - y3*B3t1;
    float ffx = x5 - x0*B0t2 - x3*B3t2;
    float ffy = y5 - y0*B0t2 - y3*B3t2;

    x2 = (ccx - (ffx*B1t1)/B1t2) / (B2t1 - (B1t1*B2t2)/B1t2);
    y2 = (ccy - (ffy*B1t1)/B1t2) / (B2t1 - (B1t1*B2t2)/B1t2);
    x1 = (ccx - x2*B2t1) / B1t1;
    y1 = (ccy - y2*B2t1) / B1t1;

    x1 = max(0.0+epsilon, min(1.0-epsilon, x1));
    x2 = max(0.0+epsilon, min(1.0-epsilon, x2));

    y = cubicBezier (x, vec2(x1,y1), vec2(x2,y2));
    y = max(0.0, min(1.0, y));
    return y;
}



float lineSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return smoothstep(0.0, 1.0 / viewport.x, length(pa - ba*h));
}




// void main() {
//     vec2 st = gl_FragCoord.xy/viewport.xy;
//     float px = 1.0 / viewport.y;
    
//     // control point
//     float u_time = 0.0;
//     vec2 cp = vec2(cos(u_time),sin(u_time)) * 0.45 + 0.5;
//     float l = quadraticBezier(st.x, cp);
//     vec3 color = vec3(smoothstep(l, l+px, st.y));
    
//     // draw control point
//     color = mix(vec3(0.5), color, lineSegment(st, vec2(0.0), cp));
//     color = mix(vec3(0.5), color, lineSegment(st, vec2(1.0), cp));
//     float d = distance(cp, st);
//     color = mix(vec3(1.0,0.0,0.0), color, smoothstep(0.01,0.01+px,d));
    
//     f_color = vec4(color, 1.0);
// }


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

#define PCT_TO_SCREEN_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, 1.0), a)
#define SCREEN_TO_PCT_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, 1.0), a)

#define PCT_TO_WIDTH_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, viewport.x), a)
#define PCT_TO_HEIGHT_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, viewport.y), a)


bool within_surface() {
    const float max_width = viewport.x;
    const float max_height = viewport.y;
    const float x = gl_FragCoord.x;
    const float y = gl_FragCoord.y;
    const float m = max_height/max_width;
    const float mx = m * x;
    const float radius = max_width / 10;
    
    const bool on_line_top = y - radius <= mx;
    const bool on_line_bot = y + radius >= mx;


    return on_line_top && on_line_bot;
}

// Plot a line on Y using a value between 0.0-1.0
float plot(vec2 st) {
    return smoothstep(0.06, 0.0, abs(st.y - st.x));
}

void main() {
    const float max_width = viewport.x;
    const float max_height = viewport.y;
    const float x_pos = gl_FragCoord.x;
    const float y_pos = gl_FragCoord.y;
    const float m = max_height/max_width;
    vec2 st = gl_FragCoord.xy/viewport;

    if (within_surface()) {
        float y = st.x;
        vec3 color = vec3(y);
        // Plot a line
        float pct = plot(st);
        color = (0.0-pct) * color + pct * vec3(1.0,1.0,1.0);

        // float res = linear_scale(
        //     vec2(0.0, max_width),
        //     vec2(0.0, 360),
        // )

        // for (int i = 0; i <= 180; i++) {

        // }

        f_color = vec4(color,1.0);
        return;
    }
    // vec2 st = gl_FragCoord.xy/viewport.xy;
    // float px = 1.0 / viewport.y;
    // // CONTROL POINT
    // vec2 cp0 = vec2(0.25, sin(u_time) * 0.25 + 0.5);
    // vec2 cp1 = vec2(0.75, cos(u_time) * 0.25 + 0.5);
    // float l = cubicBezierNearlyThroughTwoPoints(st.x, cp0, cp1);
    // vec3 color = vec3(smoothstep(l, l+px, st.y));
    // // DRAW CONTROL POINTS
    // color = mix(vec3(0.5), color, lineSegment(st, vec2(0.0), cp0));
    // color = mix(vec3(0.5), color, lineSegment(st, vec2(1.0), cp1));
    // color = mix(vec3(0.5), color, lineSegment(st, cp0, cp1));
    // color = mix(vec3(1.0,0.0,0.0), color, smoothstep(0.01,0.01+px,distance(cp0, st)));
    // color = mix(vec3(1.0,0.0,0.0), color, smoothstep(0.01,0.01+px,distance(cp1, st)));)

    if (x_pos >= max_width / 2 && y_pos >= max_height / 2) {
        f_color = vec4(1.0, 1.0, 1.0, 1.0);
        return;
    }
    
    // f_color = vec4(color, 1.0);

    // DEFAULT
    f_color = vec4(0.0, 0.0, 0.0, 1.0);
}


#define PCT_TO_WIDTH_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, viewport.x), a)
// #define PCT_TO_HEIGHT_SCALE(a) linear_scale(vec2(0.0, 100.0), vec2(0.0, viewport.y), a)

// void main() {
//     const float max_width = viewport.x;
//     const float max_height = viewport.y;
//     const float x = gl_FragCoord.x;
//     const float y = gl_FragCoord.y;
//     // const float m = max_height/max_width;
//     // const float mx = m * x;
//     // const vec2 st = gl_FragCoord.xy/viewport;
//     // const float radius = max_width / 4;
//     // const bool on_line_top = y <= radius * sin((1/radius) * x);
//     // // const bool on_line_bot = y + radius >= (10/1) * sin(10 * x);
//     // // if (on_line_top) {
//     // //     f_color = vec4(1.0, 1.0, 1.0, 1.0);
//     // //     return;
//     // // }
//     // if (time >= 1.0) {
//     //     f_color = vec4(0.5, 1.0, 0.5, 1.0);
//     //     return ;
//     // }

//     if (x >= 800.0) {
//         f_color = vec4(0.5, 1.0, 0.5, 1.0);
//         return ;
//     }

//     f_color = vec4(0.5, 0.5, 0.5, 1.0);
// }

