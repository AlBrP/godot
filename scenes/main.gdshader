shader_type canvas_item;
uniform vec2 ball_position_uv[2000];
uniform vec4 hint_color = vec4(1.0);
uniform sampler2D SCREEN_TEXTURE:hint_screen_texture,filter_linear_mipmap;
uniform bool render_enabled = true;

void fragment()
{
	float weight = 0.0;

	if (render_enabled) {
		for (int i = 0; i < ball_position_uv.length(); i++) {
			float dst = length(UV - ball_position_uv[i]);
			weight += 0.000015/(dst*dst);
		}
	}

	if (!render_enabled) {
		COLOR = vec4(0, 0, 0, 0);
	}
	else if (weight > 3.7) {
		vec4 color = texture(SCREEN_TEXTURE, SCREEN_UV);
		vec4 water = vec4(0.15, 0.58, 0.78, 1.0);
		color = mix(color, water, 0.4);
		color = mix(vec4(0.5), color, 1.5);
		COLOR = color;
	}
	else if (weight > 3.4) {
		COLOR = vec4(0.8, 1, 1, 1.0);
	}
	else {
		COLOR = vec4(0, 0, 0, 0);
	}
}
