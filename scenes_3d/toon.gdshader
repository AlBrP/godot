shader_type spatial;
render_mode blend_mix, depth_draw_opaque, cull_back, specular_disabled;

uniform float toon_levels : hint_range(1.0, 8.0) = 3.0;
uniform float shadow_floor : hint_range(0.0, 0.8) = 0.35;
uniform float rim_strength : hint_range(0.0, 2.0) = 0.6;
uniform float rim_power : hint_range(0.5, 8.0) = 3.0;
uniform vec3 rim_color : source_color = vec3(1.0, 0.95, 0.85);
uniform vec3 base_color : source_color = vec3(0.55, 0.72, 0.45);

void light() {
	float nl = clamp(dot(NORMAL, LIGHT), 0.0, 1.0);
	float steps = max(toon_levels - 1.0, 1.0);
	float lit = floor(nl * steps + 0.5) / steps;
	lit = mix(shadow_floor, 1.0, lit);
	DIFFUSE_LIGHT += LIGHT_COLOR * ATTENUATION * lit;
}

void fragment() {
	ALBEDO = base_color;
	float rim = 1.0 - clamp(dot(NORMAL, VIEW), 0.0, 1.0);
	rim = pow(rim, rim_power);
	EMISSION = rim_color * rim * rim_strength;
}
