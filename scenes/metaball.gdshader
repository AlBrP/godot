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

// 卡通水配色
uniform vec3 water_deep      = vec3(0.04, 0.18, 0.42);
uniform vec3 water_mid       = vec3(0.08, 0.42, 0.68);
uniform vec3 water_light     = vec3(0.18, 0.62, 0.85);
uniform vec3 water_bright    = vec3(0.60, 0.88, 1.00);
uniform vec3 water_foam      = vec3(0.95, 0.98, 1.00);
uniform vec3 water_spec      = vec3(1.00, 1.00, 1.00);
uniform float time = 0.0;

uniform int debug_mode = 0;

// 哈希常数（必须与C#端一致）
const int HASH_K1 = 15823;
const int HASH_K2 = 9737333;

// 空间哈希函数（与C# HashCell完全一致）
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

	// 包围盒 early-discard
	float cutoff = sigma * 3.0;
	if (worldPos.x < bbox_min.x - cutoff || worldPos.x > bbox_max.x + cutoff ||
		worldPos.y < bbox_min.y - cutoff || worldPos.y > bbox_max.y + cutoff) discard;

	float total = 0.0;
	vec2 avg_vel = vec2(0.0);
	float vel_weight_sum = 0.0;
	float inv_2sq = 1.0 / (2.0 * sigma * sigma);
	float cutoff_sq = 9.0 * sigma * sigma;

	// 计算当前像素在hash grid中的cell（用local坐标）
	vec2 localPos = worldPos - ground_offset;
	ivec2 center_cell = ivec2(floor(localPos / grid_cell_size));

	// 动态搜索半径：保证覆盖3σ
	int search_r = int(ceil(3.0 * sigma / grid_cell_size));
	search_r = min(search_r, 6); // 安全上限

	int tex_w = int(texture_size.x);

	// 遍历周围 (2*search_r+1)^2 个cell
	for (int dy = -6; dy <= 6; dy++) {
		for (int dx = -6; dx <= 6; dx++) {
			// 超出搜索半径则跳过（编译器可展开常量循环）
			if (abs(dx) > search_r || abs(dy) > search_r) continue;

			ivec2 neighbor_cell = center_cell + ivec2(dx, dy);
			uint h = hash_cell(neighbor_cell);

			// 查询hash lookup纹理
			int hx = int(h) % tex_w;
			int hy = int(h) / tex_w;
			vec2 range = texelFetch(hash_lookup_texture, ivec2(hx, hy), 0).rg;

			int start = int(range.r);
			int end = int(range.g);

			// 无效bucket
			if (start < 0) continue;

			// 遍历该bucket中的所有粒子
			for (int j = start; j <= end; j++) {
				int px = j % tex_w;
				int py = j / tex_w;
				vec4 pdata = texelFetch(raw_position_texture, ivec2(px, py), 0);
				float w = particle_contrib(worldPos - pdata.rg, inv_2sq, cutoff_sq);
				total += w;
				avg_vel += (pdata.ba - 0.5) * w;
				vel_weight_sum += w;
			}
		}
	}

	if (vel_weight_sum > 0.001) avg_vel /= vel_weight_sum;

	// Heatmap debug
	if (debug_mode == 1) {
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

		COLOR = vec4(clamp(color, 0.0, 1.0), aa);
	}
}
