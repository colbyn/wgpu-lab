[[location(0)]]
var<in> in_position: vec2<f32>;
[[location(1)]]
var<in> in_color_vs: vec4<f32>;

[[location(0)]]
var<out> out_color_vs: vec4<f32>;
[[builtin(position)]]
var<out> out_position: vec4<f32>;

[[stage(vertex)]]
fn vs_main() {
    out_position = vec4<f32>(in_position, 0.0, 1.0);
    if (out_position[0] < 0) {
        out_color_vs = vec4<f32>(
            1.0,
            in_color_vs[1],
            in_color_vs[2],
            0.9
        );
    } else {
        out_color_vs = in_color_vs;
    }
}

[[location(0)]]
var<in> in_color_fs: vec4<f32>;
[[builtin(position)]]
var<in> in_position_fs: vec4<f32>;
[[location(0)]]
var<out> out_color_fs: vec4<f32>;

[[stage(fragment)]]
fn fs_main() {
    if (in_position_fs[0] > 0.0) {
        out_color_fs = in_color_fs;
    } else {
        out_color_fs = in_color_fs;
    }
}
