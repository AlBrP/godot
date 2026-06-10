shader_type spatial;
render_mode cull_front, depth_draw_opaque, unshaded, shadows_disabled;

uniform float outline_thickness : hint_range(0.0, 0.05) = 0.015;
uniform vec3 outline_color : source_color = vec3(0.05, 0.04, 0.08);

void vertex() {
	VERTEX += NORMAL * outline_thickness;
}

void fragment() {
	ALBEDO = outline_color;
}
