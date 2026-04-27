shader_type canvas_item;

uniform sampler2D raw_position_texture;
uniform sampler2D hash_lookup_texture;
uniform int particle_count = 5000;
uniform vec2 texture_size = vec2(72.0, 70.0);
uniform bool render_enabled = true;
uniform vec2 viewport_size = vec2(1152.0, 648.0);
uniform vec2 bbox_min = vec2(0.0, 0.0);
uniform vec2 bbox_max = vec2(1152.0, 648.0);

uniform float grid_cell_size = 17.0;
uniform vec2 ground_offset = vec2(0.0, 0.0);

uniform float field_strength = 1.8;
uniform float sigma = 9.0;
uniform float body_threshold = 8.0;
uniform float edge_threshold = 4.5;
uniform float stretch_scale = 0.008;
uniform float density_scale = 0.3;
uniform float edge_sharpness = 0.82;
uniform float field_scale = 0.008;

// Toon water palette
uniform vec3 water_deep      = vec3(0.04, 0.18, 0.42);
uniform vec3 water_mid       = vec3(0.08, 0.42, 0.68);
uniform vec3 water_light     = vec3(0.18, 0.62, 0.85);
uniform vec3 water_bright    = vec3(0.60, 0.88, 1.00);
uniform vec3 water_foam      = vec3(0.95, 0.98, 1.00);
uniform vec3 water_spec      = vec3(1.00, 1.00, 1.00);
uniform float time = 0.0;

uniform float spec_strength = 0.5;
uniform float toon_levels = 0.0;  // 0=off (continuous gradient), >0=quantized levels

uniform int sim_mode = 0;
uniform int debug_mode = 0;

// Hash constants (must match C# side)
const int HASH_K1 = 15823;
const int HASH_K2 = 9737333;

// Spatial hash function (must match C# HashCell exactly)
uint hash_cell(ivec2 cell) {
	uint cx = uint(cell.x);
	uint cy = uint(cell.y);
	uint a = cx * uint(HASH_K1);
	uint b = cy * uint(HASH_K2);
	return (a + b) % uint(particle_count);
}

float particle_contrib(vec2 diff, float inv_2sq, float cutoff_sq) {
	float dist_sq = dot(diff, diff);
	if (dist_sq > cutoff_sq) return 0.0;
	return exp(-dist_sq * inv_2sq) * field_strength * 2.2;
}

void fragment() {
	if (!render_enabled) discard;

	vec2 worldPos = UV * viewport_size;

	// Bbox early-discard
	float cutoff = sigma * 3.0;
	if (worldPos.x < bbox_min.x - cutoff || worldPos.x > bbox_max.x + cutoff ||
		worldPos.y < bbox_min.y - cutoff || worldPos.y > bbox_max.y + cutoff) discard;

	float total = 0.0;
	vec2 avg_vel = vec2(0.0);
	float vel_weight_sum = 0.0;
	float avg_vx = 0.0;       // fire mode: lateral velocity
	float sum_pos_x = 0.0;    // fire mode: centroid X
	float sum_pos_y = 0.0;    // fire mode: centroid Y
	float inv_2sq = 1.0 / (2.0 * sigma * sigma);
	float cutoff_sq = 9.0 * sigma * sigma;

	// Compute hash grid cell for current pixel (using local coords)
	vec2 localPos = worldPos - ground_offset;
	ivec2 center_cell = ivec2(floor(localPos / grid_cell_size));

	// Dynamic search radius: ensure 3-sigma coverage
	int search_r = int(ceil(3.0 * sigma / grid_cell_size));
	search_r = min(search_r, 6); // safety cap

	int tex_w = int(texture_size.x);

	// Iterate surrounding (2*search_r+1)^2 cells
	for (int dy = -6; dy <= 6; dy++) {
		for (int dx = -6; dx <= 6; dx++) {
			// Skip if outside search radius (compiler can unroll constant loop)
			if (abs(dx) > search_r || abs(dy) > search_r) continue;

			ivec2 neighbor_cell = center_cell + ivec2(dx, dy);
			uint h = hash_cell(neighbor_cell);

			// Query hash lookup texture
			int hx = int(h) % tex_w;
			int hy = int(h) / tex_w;
			vec2 range = texelFetch(hash_lookup_texture, ivec2(hx, hy), 0).rg;

			int start = int(range.r);
			int end = int(range.g);

			// Invalid bucket
			if (start < 0) continue;

			// Iterate all particles in this bucket
			for (int j = start; j <= end; j++) {
				int px = j % tex_w;
				int py = j / tex_w;
				vec4 pdata = texelFetch(raw_position_texture, ivec2(px, py), 0);
				float w;
				vec2 pdiff = worldPos - pdata.rg;
				if (sim_mode == 2) {
					float temp_n = pdata.b;  // 0-1, mapped to 0-1500K
					if (temp_n < 0.12) { w = 0.0; }
					else {
						float r = length(pdiff);
						float dy = worldPos.y - pdata.rg.y;  // positive = above particle
						float sigma_x = 10.0;
						float sigma_up = 12.0;
						float sigma_down = 24.0;
						float sy = dy > 0.0 ? dy / sigma_up : -dy / sigma_down;
						float sx = r / sigma_x;
						float tongue_field = exp(-0.5 * (sx * sx + sy * sy));
						float heat = pow(temp_n, 1.5);
						w = tongue_field * field_strength * heat * 5.0;
					}
					// Decode lateral velocity from .a channel
					float vx = (pdata.a - 0.5) * 2.0 * 2000.0;
					avg_vx += vx * w;
					sum_pos_x += pdata.r * w;
					sum_pos_y += pdata.g * w;
				} else {
					w = particle_contrib(pdiff, inv_2sq, cutoff_sq);
				}
				total += w;
				if (sim_mode != 2) {
					avg_vel += (pdata.ba - 0.5) * w;
				}
				vel_weight_sum += w;
			}
		}
	}

	if (vel_weight_sum > 0.001) avg_vel /= vel_weight_sum;

	// Fire mode: hierarchical cartoon flame (main body + side tongues)
	if (sim_mode == 2) {
		float ft = time * 1.6;

		// Flame centroid and stats
		float flame_cx = vel_weight_sum > 0.001 ? (sum_pos_x / vel_weight_sum) : worldPos.x;
		float flame_cy = vel_weight_sum > 0.001 ? (sum_pos_y / vel_weight_sum) : worldPos.y;
		float mean_vx = vel_weight_sum > 0.001 ? (avg_vx / vel_weight_sum) : 0.0;

		// Flame scale from particle density (grows with mass)
		float mass_norm = clamp(total / 350.0, 0.3, 2.0);
		float flame_h = 240.0 * pow(mass_norm, 0.28);
		float flame_hw = flame_h * 0.15;

		// Local Y: 0=base, positive going upward (Godot Y increases downward)
		float base_y = flame_cy + 48.0;
		float local_y = base_y - worldPos.y;
		float ny = local_y / max(flame_h, 1.0); // normalized height

		// Wind sways entire flame (tips more than base)
		float wind = mean_vx * 0.0002;
		float flame_axis_x = flame_cx + wind * (1.0 + ny * 0.6);

		// ---- Main flame body (narrow central column) ----
		float main_w = flame_hw * 0.35 * mix(1.0, 0.04, smoothstep(0.0, 0.9, ny));
		float main_dist = abs((worldPos.x - flame_axis_x) / max(main_w, 0.3));
		float main_core = 1.0 - main_dist * main_dist;
		float main_edge = smoothstep(1.0, 0.0, main_dist);
		float main_shape = max(0.0, main_core) * main_edge;
		float main_ve = smoothstep(-0.03, 0.05, ny) * (1.0 - smoothstep(0.55, 0.95, ny));
		float main_body = main_shape * main_ve;

		// ---- Side tongues (4 detached small flames) ----
		float side_field = 0.0;

		for (int ti = 0; ti < 4; ti++) {
			float sign_side[4] = float[4](-1.0, -1.0, 1.0, 1.0);
			float offset_x[4] = float[4](0.60, 0.40, 0.40, 0.60);
			float s_hfrac[4]  = float[4](0.22, 0.42, 0.40, 0.20);
			float s_sway[4]   = float[4](10.0, 5.0, 6.0, 11.0);
			float s_sfreq[4]  = float[4](0.040, 0.032, 0.030, 0.042);
			float s_phase[4]  = float[4](1.2, 0.5, 0.7, 1.4);
			float s_tpow[4]   = float[4](2.2, 2.8, 2.9, 2.1);
			float s_bw[4]     = float[4](0.06, 0.08, 0.07, 0.05);

			float th = flame_h * s_hfrac[ti];
			float ty = local_y / max(th, 1.0);
			if (ty < -0.04 || ty > 1.06) continue;

			float time_i = ft * (1.0 + float(ti) * 0.15);
			float sw = sin(local_y * s_sfreq[ti] + time_i + s_phase[ti]) * s_sway[ti];
			sw += sin(local_y * s_sfreq[ti] * 2.3 + time_i * 1.5 + s_phase[ti] * 1.5) * s_sway[ti] * 0.25;
			float cx = flame_axis_x + sign_side[ti] * offset_x[ti] * flame_hw + sw;

			float bw = s_bw[ti] * flame_hw * (0.75 + mass_norm * 0.25);
			float width = bw * pow(max(0.004, 1.0 - ty), s_tpow[ti]);

			float dist = abs(worldPos.x - cx);
			float dn = dist / max(width, 0.001);
			if (dn >= 1.0) continue;

			float cs = (1.0 - dn * dn) * smoothstep(1.0, 0.0, dn);
			float ve = smoothstep(-0.04, 0.05, ty) * (1.0 - smoothstep(0.72, 1.06, ty));
			side_field = max(side_field, cs * ve);
		}

		// ---- Composite: main body dominates, side tongues add flank detail ----
		float shape_field = max(main_body * 1.1, side_field * 0.75);

		// Particle density: near-binary gate (eliminate contour lines)
		float density_gate = smoothstep(0.01, 0.12, total * 0.01);
		float flame_field = shape_field * density_gate;

		// Edge AA
		float edge_width = fwidth(flame_field) * 1.4;
		float in_flame = smoothstep(0.06 - edge_width, 0.06 + edge_width, flame_field);
		if (in_flame < 0.005) discard;

		if (debug_mode == 1) {
			float t = clamp(flame_field / 4.0, 0.0, 1.0);
			vec3 heat;
			if (t < 0.25) heat = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 0.5), t / 0.25);
			else if (t < 0.5) heat = mix(vec3(0.0, 0.5, 0.5), vec3(0.0, 1.0, 0.0), (t - 0.25) / 0.25);
			else if (t < 0.75) heat = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.5) / 0.25);
			else heat = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.75) / 0.25);
			COLOR = vec4(heat, 0.9);
		} else {
			vec3 core_yellow  = vec3(1.0, 0.97, 0.38);
			vec3 inner_gold   = vec3(1.0, 0.82, 0.12);
			vec3 mid_orange   = vec3(1.0, 0.60, 0.06);
			vec3 outer_orange = vec3(0.96, 0.42, 0.06);
			vec3 tip_red      = vec3(0.88, 0.28, 0.08);

			// Radial temp gradient: center=hot(yellow), edge=cool(red), base=hot, tip=cool
			float dist_from_axis = abs(worldPos.x - flame_axis_x) / max(flame_hw * 0.55, 1.0);
			float coreness = (1.0 - ny) * (1.0 - clamp(dist_from_axis, 0.0, 1.0));
			float color_t = clamp(coreness, 0.0, 1.0);

			vec3 flame_color;
			if (toon_levels > 0.5) {
				int band = int(floor(color_t * 5.0));
				if (band >= 4)       flame_color = core_yellow;
				else if (band == 3)  flame_color = inner_gold;
				else if (band == 2)  flame_color = mid_orange;
				else if (band == 1)  flame_color = outer_orange;
				else                 flame_color = tip_red;
			} else {
				if (color_t > 0.80)       flame_color = mix(inner_gold,   core_yellow,   (color_t - 0.80) / 0.20);
				else if (color_t > 0.58)  flame_color = mix(mid_orange,   inner_gold,    (color_t - 0.58) / 0.22);
				else if (color_t > 0.34)  flame_color = mix(outer_orange, mid_orange,    (color_t - 0.34) / 0.24);
				else if (color_t > 0.14)  flame_color = mix(tip_red,      outer_orange,  (color_t - 0.14) / 0.20);
				else                      flame_color = tip_red;
			}

			// Cartoon outline
			float field_grad = fwidth(flame_field);
			float is_outline = smoothstep(0.003, 0.014, field_grad) *
			                   smoothstep(0.3, 1.0, flame_field) *
			                   (1.0 - smoothstep(3.0, 6.0, flame_field));
			vec3 stroke_color = vec3(0.5, 0.10, 0.02);
			flame_color = mix(flame_color, stroke_color, is_outline * 0.20);

			float flame_alpha = 0.4 + in_flame * 0.55;
			COLOR = vec4(flame_color, flame_alpha);
		}
	} else if (sim_mode == 1) {
		float avg_temp = vel_weight_sum > 0.001 ? avg_vel.x : 0.0;
		float temp_opacity = smoothstep(0.15, 0.55, avg_temp);
		float smoke_alpha = smoothstep(0.2, 2.0, total) * 0.65 * temp_opacity;
		// Steam white: hot vapor (dense white) to cool condensate (thin gray-white)
		vec3 dense_steam   = vec3(0.95, 0.97, 1.0);
		vec3 steam         = vec3(0.88, 0.90, 0.93);
		vec3 thin_steam    = vec3(0.7, 0.72, 0.75);
		vec3 fading_steam  = vec3(0.92, 0.92, 0.93);
		vec3 smoke_color;
		if (avg_temp > 0.6) {
			smoke_color = mix(steam, dense_steam, (avg_temp - 0.6) / 0.4);
		} else if (avg_temp > 0.25) {
			smoke_color = mix(thin_steam, steam, (avg_temp - 0.25) / 0.35);
		} else {
			smoke_color = mix(fading_steam, thin_steam, avg_temp / 0.25);
		}
		COLOR = vec4(smoke_color, smoke_alpha);
	} else if (debug_mode == 1) {
		float t = clamp(total / (body_threshold * 1.5), 0.0, 1.0);
		vec3 heat;
		if (t < 0.5) heat = mix(vec3(0.0, 0.0, 0.2), vec3(0.0, 1.0, 0.0), t * 2.0);
		else heat = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
		COLOR = vec4(heat, 1.0);
	} else if (debug_mode == 2) {
		float t = clamp(total / (body_threshold * 1.5), 0.0, 1.0);
		vec3 heat = t < 0.5 ? mix(vec3(0.0, 0.0, 0.2), vec3(0.0, 1.0, 0.0), t * 2.0)
			: mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
		COLOR = vec4(heat, 1.0);
	} else {
		float edge_width = fwidth(total) * 1.5;
		float aa = smoothstep(2.5 - edge_width, 2.5 + edge_width, total);
		if (aa < 0.001) discard;

		float n = total * field_scale;

		// Toon quantization: map continuous n to discrete levels
		if (toon_levels > 0.5) {
			n = floor(n * toon_levels) / toon_levels;
		}

		float rim = smoothstep(2.5, 4.0, total);
		vec3 rim_color = mix(water_foam, water_bright, rim);

		vec3 color;
		if (n > 1.35) {
			color = water_deep;
		} else if (n > 0.75) {
			color = mix(water_mid, water_deep, smoothstep(0.75, 1.35, n));
		} else if (n > 0.3) {
			color = mix(water_light, water_mid, smoothstep(0.3, 0.75, n));
		} else {
			color = mix(water_bright, water_light, smoothstep(0.05, 0.3, n));
		}

		float rim_blend = 1.0 - smoothstep(3.0, 6.0, total);
		color = mix(color, rim_color, rim_blend * 0.7);

		vec2 p1 = UV * 25.0 + time * 2.3;
		vec2 p2 = UV * 41.0 - time * 1.7;
		float noise1 = fract(sin(dot(p1, vec2(12.9898, 78.233))) * 43758.5453);
		float noise2 = fract(sin(dot(p2, vec2(31.4159, 67.182))) * 23456.789);
		float noise = (noise1 + noise2) * 0.5;

		color = mix(color, color * 0.92 + 0.04, noise * 0.20);

		float foam = 0.0;
		float speed_mag = length(avg_vel);
		foam = smoothstep(0.06, 0.20, speed_mag) * 0.8;
		foam *= smoothstep(0.03, 0.15, n);
		foam *= (0.5 + noise * 0.5);
		foam = clamp(foam, 0.0, 1.0);

		color = mix(color, water_foam, foam);

		// Velocity-driven specular: faster flow = brighter
		float vel_mag = length(avg_vel);
		float spec = smoothstep(0.005, 0.06, vel_mag);
		spec *= (0.4 + 0.6 * pow(max(dot(normalize(avg_vel + vec2(0.0001)), vec2(0.35, -0.6)), 0.0), 2.0));
		color = mix(color, water_spec, spec * spec_strength);

		COLOR = vec4(clamp(color, 0.0, 1.0), aa);
	}
}
