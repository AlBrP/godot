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
uniform float toon_levels = 0.0;
uniform float color_band = 2.2;     // >1 pushes bands inward (less yellow), <1 spreads out (more yellow) // 0=off (continuous gradient), >0=quantized levels

uniform int debug_mode = 0;
uniform sampler2D physics_texture;

// Per-body waterline anchors. waterlines_world[b] = (xL, yL, xR, yR)
// in viewport-world coordinates. valid_mask bit b means body b has a
// usable waterline this frame. bodies_world[b] = (cx, cy, r, 0) is the
// body center + radius, used to test whether a fragment is inside.
const int MAX_BODIES_RIM = 4;
const int FAKE_PER_BODY = 120;
const int FAKE_TOTAL = 480; // MAX_BODIES_RIM * FAKE_PER_BODY
uniform vec4 waterlines_world[4];
uniform vec4 bodies_world[4];
uniform int waterline_valid_mask = 0;
uniform vec2 fake_particles[480];

// Particle type constants
const int PTYPE_WATER = 0;
const int PTYPE_FIRE = 1;
const int PTYPE_SMOKE = 2;
const int PTYPE_STEAM = 3;

int particle_type(vec4 pdata) {
    return int(pdata.b * 16.0 + 0.5);
}

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
	float total_water = 0.0;
	float total_fire = 0.0;
	// drew flag drives the priority cascade at the bottom: each branch
	// sets drew=true after writing COLOR, downstream branches gate on
	// !drew so they don't overwrite. Godot canvas_item fragment shader
	// doesn't allow `return`, so this is the canonical way.
	bool drew = false;
	// Smoke is SDF-based, but combined via exponential soft-min instead
	// of polynomial smooth-min: -k*log(sum_i exp(-d_i/k)). This is
	// commutative AND associative, so the result is independent of the
	// bucket iteration order. Polynomial smooth-min is non-associative,
	// and since sorted_indices reshuffles every frame, the same pixel
	// got different sdf values across frames -> visible flicker. It also
	// produced D-shape / sliced lobes on isolated puffs because adjacent
	// fragments saw different orderings and computed different sdfs.
	// k=3 matches the visual hardness of the old polynomial blend_k=9
	// (k_exp ≈ blend_k_poly * 0.36, derived from equal 2-particle merge
	// depth: polynomial gives a - 0.25*k_poly, exp gives a - k_exp*ln2).
	const float smoke_k = 3.0;
	float smoke_softmin_sum = 0.0;
	// Max single-particle contribution. Used to derive "concentration" =
	// max/sum at output: ~1 when one puff dominates this fragment, low
	// (~1/N) when N far-away puffs are summing up into ghost-bridging
	// territory. Outline is gated by this so ghost regions don't paint
	// phantom ink-lines, while real silhouettes (single dominant puff)
	// still get the full cartoon stroke.
	float smoke_max_contrib = 0.0;
	// Average remaining-life of smoke particles contributing here. Used at
	// output to fade alpha in the last ~20% of lifetime. Geometry doesn't
	// shrink — only alpha drops, so the puff disappears in place instead
	// of visibly collapsing inward.
	float smoke_life_sum = 0.0;
	float smoke_life_weight = 0.0;
	vec2 sdf_grad = vec2(0.0);
	float grad_weight_sum = 0.0;
	vec2 avg_vel = vec2(0.0);
	float vel_weight_sum = 0.0;
	float sum_pos_x = 0.0;
	float sum_pos_y = 0.0;
	float fire_weight_sum = 0.0;
	float avg_vx = 0.0, avg_vy = 0.0;
	float vel_weight_sum_fire = 0.0;
	float avg_curl = 0.0, curl_weight_sum = 0.0;
	float avg_pressure = 0.0, press_weight = 0.0;
	float fire_temp_sum = 0.0, fire_temp_weight = 0.0;
	float inv_2sq = 1.0 / (2.0 * sigma * sigma);
	float cutoff_sq = 9.0 * sigma * sigma;

	// Compute hash grid cell for current pixel (using local coords)
	vec2 localPos = worldPos - ground_offset;
	ivec2 center_cell = ivec2(floor(localPos / grid_cell_size));

	// Dynamic search radius: 3-sigma coverage for water/fire kernel.
	// Smoke uses SDF -- search 3 cells covers puff_radius(22) + a few
	// blend_k(9) widths beyond which smooth-min contribution is ~0.
	// search_r=6 was 6.76x the work and tanked fps to 30-50.
	int search_r_kernel = int(ceil(3.0 * sigma / grid_cell_size));
	int search_r_smoke  = 3;
	int search_r = max(search_r_kernel, search_r_smoke);
	search_r = min(search_r, 6); // safety cap

	int tex_w = int(texture_size.x);

	// Iterate surrounding (2*search_r+1)^2 cells
	for (int dy = -3; dy <= 3; dy++) {
		for (int dx = -3; dx <= 3; dx++) {
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
				int ptype = particle_type(pdata);
				float extra = pdata.a; // temp_n for fire/smoke, density_n for water
				if (ptype == PTYPE_FIRE) {
					float temp_n = extra;
					if (temp_n < 0.12) { w = 0.0; }
					else {
						float r = length(pdiff);
						float dy = worldPos.y - pdata.g;
						float sigma_x = 10.0;
						float sigma_up = 12.0;
						float sigma_down = 24.0;
						float sy = dy > 0.0 ? dy / sigma_up : -dy / sigma_down;
						float sx = r / sigma_x;
						float tongue_field = exp(-0.5 * (sx * sx + sy * sy));
						float heat = pow(temp_n, 1.5);
						w = tongue_field * field_strength * heat * 5.0;

						// Accumulate physics data from physics texture
						ivec2 ptex_coord = ivec2(px, py);
						vec4 pphys = texelFetch(physics_texture, ptex_coord, 0);
						vec2 pvel = pphys.rg;
						float pcurl = pphys.b;
						float ppress = pphys.a;

						float vel_weight = temp_n * w;
						avg_vx += pvel.x * vel_weight;
						avg_vy += pvel.y * vel_weight;
						vel_weight_sum_fire += vel_weight;

						avg_curl += pcurl * w;
						curl_weight_sum += w;

						avg_pressure += ppress * w;
						press_weight += w;

						fire_temp_sum += temp_n * w;
						fire_temp_weight += w;
					}
					total_fire += w;
					sum_pos_x += pdata.r * w;
					sum_pos_y += pdata.g * w;
					fire_weight_sum += w;
				} else if (ptype == PTYPE_SMOKE || ptype == PTYPE_STEAM) {
					// Cartoon smoke: each particle is a hard disc (SDF) at
					// distance d = |pdiff| - puff_radius. Combined across
					// particles via exponential soft-min (accumulated in
					// smoke_softmin_sum, finalised after the loop). See
					// header comment near smoke_k for rationale.
					float temp_n = extra;
					if (temp_n > 0.02) {
						float fade = smoothstep(0.02, 0.35, temp_n);
						float puff_radius = 22.0 * fade;
						float dist = length(pdiff);
						// Wispy edge: dying puffs get organic SDF noise that
						// breaks their silhouette into 2-3 large lobes/dents
						// (mimicking how real smoke disperses into fragments).
						// Wavelengths ~ puff_radius so we get smooth bumps,
						// not high-frequency spikes. Particle position seeds
						// per-puff phase so neighbours deform independently.
						// wispy uses pow(.., 2.0) so mid-life puffs stay nearly
						// circular and only old ones visibly fragment.
						float plife = texelFetch(physics_texture, ivec2(px, py), 0).b;
						float wispy = pow(clamp(1.0 - plife, 0.0, 1.0), 2.0);
						float n1 = sin(pdiff.x * 0.12 + pdata.r * 0.05) * cos(pdiff.y * 0.10 + pdata.g * 0.03);
						float n2 = sin(pdiff.x * 0.20 - pdata.g * 0.07) * cos(pdiff.y * 0.18 + pdata.r * 0.04);
						float wisp_noise = n1 * 0.6 + n2 * 0.4;
						float wisp_disp = wisp_noise * wispy * 10.0;
						float d = dist - puff_radius + wisp_disp;
						// Clamp scaled value so exp() stays in a sane range.
						// d_min = -puff_radius = -22 -> scaled = 22/3 ≈ 7.3
						// d_max in influence ~ 5*k ≈ 15 -> scaled ≈ -5
						float scaled = clamp(-d / smoke_k, -8.0, 8.0);
						float contrib = exp(scaled);
						smoke_softmin_sum += contrib;
						smoke_max_contrib = max(smoke_max_contrib, contrib);
						smoke_life_sum += plife * contrib;
						smoke_life_weight += contrib;
						if (dist > 0.001) {
							float gw = max(0.0, 1.0 - abs(d) / (puff_radius + smoke_k));
							sdf_grad += (pdiff / dist) * gw;
							grad_weight_sum += gw;
						}
					}
					w = 0.0;
				} else {
					// Water
					w = particle_contrib(pdiff, inv_2sq, cutoff_sq);
					total_water += w;
				}
				total += w;
				vel_weight_sum += w;
			}
		}
	}

	// Finalise exponential soft-min: sdf = -k * log(sum_i exp(-d_i/k)).
	// Empty sum -> no smoke contribution, fall through to 1e10 sentinel
	// (same value the old polynomial path initialised to).
	float sdf_smoke = smoke_softmin_sum > 1e-6 ? -smoke_k * log(smoke_softmin_sum) : 1e10;

	if (fire_weight_sum > 0.001) {
		sum_pos_x /= fire_weight_sum;
		sum_pos_y /= fire_weight_sum;
	}
	if (vel_weight_sum_fire > 0.001) {
		avg_vx /= vel_weight_sum_fire;
		avg_vy /= vel_weight_sum_fire;
	}
	if (curl_weight_sum > 0.001) {
		avg_curl /= curl_weight_sum;
	}
	if (press_weight > 0.001) {
		avg_pressure /= press_weight;
	}

	// Fire: hierarchical cartoon flame (main body + side tongues)
	if (total_fire > 0.01) {
		float ft = time * 1.6;

		// Flame centroid and stats
		float flame_cx = fire_weight_sum > 0.001 ? sum_pos_x : worldPos.x;
		float flame_cy = fire_weight_sum > 0.001 ? sum_pos_y : worldPos.y;

		// Flame scale: nearly constant (not per-pixel!) to avoid shape following particle cone
		float mass_norm = 0.82 + 0.18 * smoothstep(8.0, 60.0, total_fire);
		// Physics-driven flame height: proportional to temperature and upward velocity
		float avg_upward = vel_weight_sum_fire > 0.001 ? abs(min(avg_vy, 0.0)) : 1.0;
		float avg_temp_clamped = fire_temp_weight > 0.001 ? clamp(fire_temp_sum / fire_temp_weight, 0.0, 1.0) : 0.5;
		float thermal_power = avg_temp_clamped * (0.5 + avg_upward * 0.008);
		float flame_h = 160.0 + 200.0 * thermal_power * pow(mass_norm, 0.25);
		float flame_hw = flame_h * 0.15;

		// Local Y: 0=base, positive going upward (Godot Y increases downward)
		float base_y = flame_cy + 48.0;
		float local_y = base_y - worldPos.y;
		float ny = local_y / max(flame_h, 1.0); // normalized height

		// Flame axis: center on centroid
		float flame_axis_x = flame_cx;

		// Edge noise: subtle jitter on tongue edges only (not main body)
		float edge_noise = fract(sin(dot(worldPos.xy * 0.08 + ft * 2.0, vec2(12.9898, 78.233))) * 43758.5453);
		edge_noise = mix(0.95, 1.05, edge_noise);

		// Main body width: solid pillar, no holes
		float pressure_factor = 0.7 + 0.3 * clamp(avg_pressure / 5000.0, 0.0, 1.0);
		float main_w = flame_hw * 0.55 * pressure_factor * mix(1.0, 0.08, smoothstep(0.0, 1.05, ny));

		// ---- Main flame body (wide central pillar) ----
		float main_dist = abs((worldPos.x - flame_axis_x) / max(main_w, 0.5));
		float main_core = 1.0 - main_dist * main_dist;
		float main_edge = smoothstep(1.0, 0.0, main_dist);
		float main_shape = max(0.0, main_core) * main_edge;
		float main_ve = smoothstep(-0.04, 0.07, ny) * (1.0 - smoothstep(0.72, 1.10, ny));
		float main_body = main_shape * main_ve;

		// ---- Side tongues (3 medium + 2 small embers + 3 tip tongues) ----
		float side_field = 0.0;

		// Medium tongues: taller, wider, more visible
		for (int ti = 0; ti < 3; ti++) {
			float sign_side[3] = float[3](-1.0, 1.0, -1.0);
			float offset_x[3] = float[3](0.62, 0.56, 0.74);
			float s_hfrac[3]  = float[3](0.72, 0.78, 0.60);
			float s_sway[3]   = float[3](18.0, 15.0, 13.0);
			float s_sfreq[3]  = float[3](0.038, 0.034, 0.042);
			float s_phase[3]  = float[3](1.1, 0.6, 1.7);
			float s_tpow[3]   = float[3](1.7, 1.9, 2.1);
			float s_bw[3]     = float[3](0.14, 0.13, 0.11);

			float th = flame_h * s_hfrac[ti];
			float ty = local_y / max(th, 1.0);
			if (ty < -0.04 || ty > 1.08) continue;

			float time_i = ft * (1.0 + float(ti) * 0.12);
			// Physics-driven sway: horizontal velocity + curl jitter + residual procedural
			float phys_sway = avg_vx * 0.15;
			float curl_jitter = (avg_curl - 0.5) * s_sway[ti] * 0.25;
			float residual = sin(local_y * s_sfreq[ti] + s_phase[ti]) * s_sway[ti] * 0.08;
			float sw = phys_sway + curl_jitter + residual;
			float cx = flame_axis_x + sign_side[ti] * offset_x[ti] * flame_hw * edge_noise + sw;

			float bw = s_bw[ti] * flame_hw * (0.80 + mass_norm * 0.20) * edge_noise;
			float width = bw * pow(max(0.005, 1.0 - ty), s_tpow[ti]);

			float dist = abs(worldPos.x - cx);
			float dn = dist / max(width, 0.002);
			if (dn >= 1.0) continue;

			float cs = (1.0 - dn * dn) * smoothstep(1.0, 0.0, dn);
			float ve = smoothstep(-0.04, 0.07, ty) * (1.0 - smoothstep(0.82, 1.08, ty));
			side_field = max(side_field, cs * ve);
		}

		// Small ember tongues (detached sparks)
		for (int ei = 0; ei < 2; ei++) {
			float e_sign[2]    = float[2](1.0, -1.0);
			float e_ox[2]      = float[2](0.34, 0.42);
			float e_hf[2]      = float[2](0.24, 0.18);
			float e_yoff[2]    = float[2](0.54, 0.70);
			float e_sw[2]      = float[2](7.0, 5.0);
			float e_sf[2]      = float[2](0.058, 0.070);
			float e_ph[2]      = float[2](2.2, 0.8);

			float th = flame_h * e_hf[ei];
			float y_start = flame_h * e_yoff[ei];
			float ty = (local_y - y_start) / max(th, 1.0);
			if (ty < -0.05 || ty > 1.06) continue;

			float time_i = ft * (1.3 + float(ei) * 0.2);
			float phys_sway = avg_vx * 0.15;
			float curl_jitter = (avg_curl - 0.5) * e_sw[ei] * 0.30;
			float residual = sin(local_y * e_sf[ei] + e_ph[ei]) * e_sw[ei] * 0.08;
			float sw = phys_sway + curl_jitter + residual;
			float cx = flame_axis_x + e_sign[ei] * e_ox[ei] * flame_hw * edge_noise + sw;

			float bw = 0.042 * flame_hw * (0.85 + mass_norm * 0.15) * edge_noise;
			float width = bw * pow(max(0.006, 1.0 - ty), 2.4);

			float dist = abs(worldPos.x - cx);
			float dn = dist / max(width, 0.001);
			if (dn >= 1.0) continue;

			float cs = (1.0 - dn * dn * 0.7) * smoothstep(1.0, 0.0, dn);
			float ve = smoothstep(-0.02, 0.09, ty) * (1.0 - smoothstep(0.86, 1.06, ty));
			side_field = max(side_field, cs * ve * 0.7);
		}

		// Tip tongues: tall, thin, reach above all others -> arched crown
		for (int tpi = 0; tpi < 3; tpi++) {
			float tp_sign[3] = float[3](1.0, -1.0, 1.0);
			float tp_ox[3]   = float[3](0.08, 0.12, 0.30);
			float tp_hf[3]   = float[3](0.38, 0.32, 0.26);
			float tp_yoff[3] = float[3](0.58, 0.64, 0.70);
			float tp_sw[3]   = float[3](9.0, 8.0, 6.0);
			float tp_sf[3]   = float[3](0.044, 0.050, 0.056);
			float tp_ph[3]   = float[3](0.3, 1.8, 3.1);
			float tp_bw[3]   = float[3](0.055, 0.048, 0.040);
			float tp_tp[3]   = float[3](2.0, 2.2, 2.5);

			float th = flame_h * tp_hf[tpi];
			float y_start = flame_h * tp_yoff[tpi];
			float ty = (local_y - y_start) / max(th, 1.0);
			if (ty < -0.06 || ty > 1.10) continue;

			float time_i = ft * (1.6 + float(tpi) * 0.18);
			float phys_sway = avg_vx * 0.15;
			float curl_jitter = (avg_curl - 0.5) * tp_sw[tpi] * 0.35;
			float residual = sin(local_y * tp_sf[tpi] + tp_ph[tpi]) * tp_sw[tpi] * 0.08;
			float sw = phys_sway + curl_jitter + residual;
			float cx = flame_axis_x + tp_sign[tpi] * tp_ox[tpi] * flame_hw * edge_noise + sw;

			float bw = tp_bw[tpi] * flame_hw * edge_noise;
			float width = bw * pow(max(0.004, 1.0 - ty), tp_tp[tpi]);

			float dist = abs(worldPos.x - cx);
			float dn = dist / max(width, 0.0005);
			if (dn >= 1.0) continue;

			float cs = (1.0 - dn * dn * 0.8) * smoothstep(1.0, 0.0, dn);
			float ve = smoothstep(-0.03, 0.08, ty) * (1.0 - smoothstep(0.90, 1.10, ty));
			side_field = max(side_field, cs * ve * 0.85);
		}

		// ---- Solid ellipse base (fills all interior holes) ----
		float sbx = abs(worldPos.x - flame_axis_x) / max(flame_hw * 0.55, 1.0);
		float sby = ny;
		float ellipse_w = mix(1.0, 0.20, smoothstep(0.0, 1.0, sby));
		float solid_base = (1.0 - sbx * sbx / max(ellipse_w * ellipse_w, 0.01))
		                  * smoothstep(-0.05, 0.08, sby)
		                  * (1.0 - smoothstep(0.92, 1.12, sby));
		solid_base = max(solid_base, 0.0) * 0.6;

		// ---- Composite: tongues on top of solid base ----
		float shape_field = max(main_body * 1.05, side_field * 0.90);
		shape_field = max(shape_field, solid_base);

		// Concave arc base
		float base_arc = 0.0;
		if (ny > -0.08 && ny < 0.18) {
			float bny = (ny + 0.08) / 0.26;
			float bx = abs(worldPos.x - flame_axis_x) / max(flame_hw * 0.50, 1.0);
			float b_curve = bx * bx * (1.8 - 0.9 * bny);
			base_arc = (1.0 - b_curve) * smoothstep(0.0, 1.0, bny) * smoothstep(1.0, 0.7, bny);
		}
		shape_field = max(shape_field, base_arc * 0.7);

		// Taper: convex width envelope + center spike (scaled to visible ny ~0-0.3)
		{
			float ny_norm = clamp(ny / 0.28, 0.0, 1.0);
			float rate = 1.0 - ny_norm;
			float tdist = abs(worldPos.x - flame_axis_x);
			float tmax = 65.0 * rate;
			float tsoft = 3.0 + rate * 7.0;
			shape_field *= (1.0 - smoothstep(tmax - tsoft, tmax, tdist));

			// Center spike: physics-driven axis tongue for pointed tip
			float sny = ny * 3.5;
			float th = flame_h * 0.25;
			float ystart = flame_h * 0.08;
			float sty = (local_y - ystart) / max(th, 1.0);
			if (sty >= -0.04 && sty <= 1.08) {
				float psway = avg_vx * 0.12 + (avg_curl - 0.5) * 2.0 + sin(local_y * 0.06 + 1.5) * 1.5 * 0.05;
				float scx = flame_axis_x + psway;
				float sbw = 0.12 * flame_hw * edge_noise;
				float swidth = sbw * pow(max(0.0, 1.0 - sty * 0.9), 2.5);
				float sdist = abs(worldPos.x - scx);
				float sdn = sdist / max(swidth, 0.001);
				if (sdn < 1.0) {
					float sc = pow(max(0.0, 1.0 - sdn), 3.0 + 5.0 * sny);
					float sve = smoothstep(-0.03, 0.10, sty) * pow(max(0.0, 1.0 - smoothstep(0.80, 1.05, sty)), 1.5);
					shape_field = max(shape_field, sc * sve * 2.5);
				}
			}
		}

		// Particle density gate
		float density_gate = smoothstep(0.01, 0.12, total_fire * 0.01);
		float flame_field = shape_field * density_gate;

		// Edge AA
		float edge_width = fwidth(flame_field) * 1.4;
		float in_flame = smoothstep(0.06 - edge_width, 0.06 + edge_width, flame_field);
		// When fragment isn't inside the procedural flame body (cavity edge,
		// e.g. where SPH pushed fire and water apart), DON'T discard — let
		// it fall through to the smoke/steam SDF branch below so STEAM
		// particles that actually live in the cavity get rendered. Use
		// early return on the success path, drop into smoke/water on the
		// fail path.
		if (in_flame >= 0.005) {

		if (debug_mode == 1) {
			float t = clamp(flame_field / 4.0, 0.0, 1.0);
			vec3 heat;
			if (t < 0.25) heat = mix(vec3(0.0, 0.0, 0.5), vec3(0.0, 0.5, 0.5), t / 0.25);
			else if (t < 0.5) heat = mix(vec3(0.0, 0.5, 0.5), vec3(0.0, 1.0, 0.0), (t - 0.25) / 0.25);
			else if (t < 0.75) heat = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.5, 0.0), (t - 0.5) / 0.25);
			else heat = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 1.0), (t - 0.75) / 0.25);
			COLOR = vec4(heat, 0.9);
		} else {
			// Colors matching cartoon ref: yellow core -> orange body -> red tips
			vec3 hot_white   = vec3(1.0, 1.0, 0.65);
			vec3 core_yellow = vec3(1.0, 0.92, 0.16);
			vec3 inner_gold  = vec3(1.0, 0.68, 0.04);
			vec3 body_orange = vec3(1.0, 0.48, 0.02);
			vec3 edge_orange = vec3(0.92, 0.30, 0.04);
			vec3 tip_red     = vec3(0.72, 0.12, 0.02);

			// Physics-driven color: temperature + geometric blend
			float avg_temp = fire_temp_weight > 0.001 ? fire_temp_sum / fire_temp_weight : 0.0;
			float temp_coreness = pow(clamp(avg_temp, 0.0, 1.0), 1.2);

			float dist_from_axis = abs(worldPos.x - flame_axis_x) / max(flame_hw * 0.52, 1.0);
			float height_cool = smoothstep(0.0, 1.0, ny);
			float radial_cool = smoothstep(0.0, 1.0, dist_from_axis * 1.7);
			float geo_coreness = (1.0 - height_cool * 0.88) * (1.0 - radial_cool * 0.82);

			// Temperature dominates (70%), geometry as fallback (30%)
			float coreness = temp_coreness * 0.7 + geo_coreness * 0.3;
			float color_t = clamp(coreness, 0.0, 1.0);
			if (color_band > 1.01 || color_band < 0.99)
				color_t = pow(color_t, color_band);

			vec3 flame_color;
			if (toon_levels > 0.5) {
				int band = int(floor(color_t * 6.0));
				if (band >= 5)       flame_color = hot_white;
				else if (band == 4)  flame_color = core_yellow;
				else if (band == 3)  flame_color = inner_gold;
				else if (band == 2)  flame_color = body_orange;
				else if (band == 1)  flame_color = edge_orange;
				else                 flame_color = tip_red;
			} else {
				if (color_t > 0.90)       flame_color = mix(core_yellow, hot_white,    (color_t - 0.90) / 0.10);
				else if (color_t > 0.70)  flame_color = mix(inner_gold,  core_yellow,   (color_t - 0.70) / 0.20);
				else if (color_t > 0.46)  flame_color = mix(body_orange, inner_gold,    (color_t - 0.46) / 0.24);
				else if (color_t > 0.24)  flame_color = mix(edge_orange, body_orange,   (color_t - 0.24) / 0.22);
				else if (color_t > 0.08)  flame_color = mix(tip_red,     edge_orange,    (color_t - 0.08) / 0.16);
				else                      flame_color = tip_red;
			}

			// Inner glow: subtle bright spot at bottom-center
			float glow_r = dist_from_axis + ny * 0.8;
			float inner_glow = exp(-glow_r * glow_r * 5.0) * 0.14;
			flame_color = mix(flame_color, core_yellow, inner_glow * smoothstep(0.25, 0.85, flame_field));

			// Cartoon outline: only at real edges, NOT inside
			float field_grad = fwidth(flame_field);
			float is_outline = smoothstep(0.003, 0.015, field_grad) *
			                   smoothstep(0.15, 0.5, flame_field) *
			                   (1.0 - smoothstep(2.5, 4.5, flame_field));
			// Suppress internal strokes: kill outline where solid_base is strong
			float interior_mask = 1.0 - smoothstep(0.25, 0.6, solid_base);
			is_outline *= interior_mask;
			float edge_sel = 1.0 - smoothstep(0.4, 1.8, shape_field);
			is_outline *= (0.55 + edge_sel * 0.45);
			vec3 stroke_color = vec3(0.38, 0.07, 0.01);
			flame_color = mix(flame_color, stroke_color, is_outline * 0.42);

			float flame_alpha = 0.45 + in_flame * 0.52;
			COLOR = vec4(flame_color, flame_alpha);
		}
		drew = true;
		}
	}
	if (!drew && sdf_smoke < 0.5) {
		// Cartoon smoke: SDF-based render. AA is driven by the SDF value
		// itself (linear ramp over ~1.5 px around sdf=0), NOT by fwidth.
		// fwidth would spike at hash-cell boundaries where the visible
		// particle set jumps, smearing ghost ramps across the screen.
		if (sdf_smoke > 0.0) discard; // hard cull outside, no leak from spikes
		float aa = clamp(-sdf_smoke / 1.5, 0.0, 1.0);

		// Depth into the lobe (0 at surface, grows toward core).
		float depth = -sdf_smoke;

		// Cell-shaded core->rim ramp. 3 discrete bands when toon_levels>0.
		float ramp = clamp(depth / 14.0, 0.0, 1.0);
		if (toon_levels > 0.5) {
			ramp = floor(ramp * toon_levels) / toon_levels;
		}

		// Brighter palette overall; cartoon smoke reads as a white mass with
		// shadow only on the dark side, not a continuous gradient.
		vec3 smoke_bright = vec3(0.98, 0.99, 1.00);   // deep core
		vec3 smoke_light  = vec3(0.88, 0.90, 0.93);   // mid
		vec3 smoke_shade  = vec3(0.66, 0.68, 0.74);   // near-rim shadow

		vec3 color;
		if (ramp > 0.55)      color = smoke_bright;
		else if (ramp > 0.25) color = mix(smoke_light, smoke_bright, smoothstep(0.25, 0.55, ramp));
		else                  color = mix(smoke_shade, smoke_light,  smoothstep(0.0,  0.25, ramp));

		// Cel-shaded per-bubble lighting: only the side facing AWAY from the
		// light gets one discrete shadow step. The lit side and equator stay
		// at full base color. This is what makes cartoon art read as flat
		// shapes with a single shadow plane, not 3D plastic balls.
		vec2 normal = grad_weight_sum > 0.001 ? normalize(sdf_grad) : vec2(0.0, -1.0);
		vec2 light_dir = normalize(vec2(-0.45, -0.85)); // upper-left key
		float NdotL = dot(normal, light_dir);
		float shadow_strength = smoothstep(-0.15, -0.55, NdotL);
		if (toon_levels > 0.5) {
			shadow_strength = step(0.5, shadow_strength);
		}
		color *= mix(1.0, 0.78, shadow_strength);

		// Outline gated by concentration = max_contrib / sum: ~1 when one
		// puff truly owns this fragment (real silhouette), drops fast when
		// many distant puffs each contribute a small amount (ghost zone
		// where exp soft-min synthesises an sdf inside [-3.5, 0] band).
		// This kills phantom ink-blobs in sparse top-of-plume regions
		// without breaking real silhouettes or puff-valley outlines.
		float concentration = smoke_softmin_sum > 0.001 ? smoke_max_contrib / smoke_softmin_sum : 0.0;
		float outline_gate = smoothstep(0.35, 0.65, concentration);
		float outline = smoothstep(-3.5, -1.0, sdf_smoke) * (1.0 - smoothstep(-1.0, 0.0, sdf_smoke));
		outline *= outline_gate;
		// Fading puffs need the outline to weaken with them, otherwise the
		// near-black stroke_color stays opaque-ish under a low-alpha disc
		// and reads as a dark blot instead of a white wisp.
		float avg_life = smoke_life_weight > 0.001 ? smoke_life_sum / smoke_life_weight : 1.0;
		// life_alpha curves: pow(life, 0.5) keeps young puffs near opaque
		// while letting old ones get noticeably translucent (full lifetime
		// fade, not just last 20%). Paired with the wispy SDF displacement
		// this reads as "soft puff slowly dispersing into wisps", not "puff
		// suddenly turning grey and vanishing".
		float life_alpha = pow(clamp(avg_life, 0.0, 1.0), 0.5);
		outline *= life_alpha;
		vec3 outline_col = vec3(0.18, 0.20, 0.26);
		color = mix(color, outline_col, outline * 0.95);

		// Aggressive trend toward smoke_bright as the puff dies, so the
		// last visible frames are a white wisp rather than a grey blot
		// (shadow + outline tint under low alpha would otherwise read as
		// dark grey when composited over the dim grey background).
		color = mix(smoke_bright, color, pow(life_alpha, 2.5));

		float final_alpha = max(aa, outline) * life_alpha;
		COLOR = vec4(clamp(color, 0.0, 1.0), final_alpha);
		drew = true;
	}
	if (!drew && debug_mode == 1) {
		// placeholder -- real debug heat draw happens after fake loop below
	}
	if (!drew && debug_mode == 2) {
		// placeholder -- real debug heat draw happens after fake loop below
	}

	// Fake-particle contribution: treat fakes identically to real water so
	// the metaball field is uniform across the body's cavity and the
	// surrounding pool. No special atten / waterline gate / cavity fill --
	// fakes plug into the same Gaussian kernel (sigma=inv_2sq, cutoff_sq)
	// the real-water loop uses, and total_water sees one homogeneous field.
	// The pool's natural density gradient (foam rim near the surface,
	// mid-blue deeper) emerges from where the fakes sit relative to the
	// fragment, just like for real particles.
	//
	// FAKE_KERNEL_WEIGHT is a single uniform scale (no positional logic).
	// 1.0 would treat fakes as one-for-one with real-water particles, but
	// fake hex spacing is denser (10 px vs SPH ~17 px) AND the cavity is
	// a single layer of fakes while the pool stacks many layers. The net
	// per-fragment count inside the cavity ends up roughly matching the
	// pool's, so a weight modestly below 1 keeps the body's water from
	// reading darker than the surrounding pool while still filling the
	// cavity solidly.
	//
	// Per-fake self-weight (own-body fade): fade by d_fake against the
	// fake's own body so the outermost ring (d_fake ~ +15, cavity edge)
	// doesn't double up with real-water Gaussian tails just outside the
	// cavity -- without that fade the two kernels stack in d=0..10 and
	// total_water spikes into a "fish-tail" lobe under the body. This is
	// driven by the fake's own position, not the fragment's.
	// FAKE_KERNEL_WEIGHT = 0.45. The own-body fade is direction-aware
	// (fade_upper interpolates 12..26 by facing_gap), so the per-fake
	// weight is naturally lower on the pool-facing side (real-water tail
	// overlap suppressed) and higher on the gap-facing side (no real
	// water to overlap, gap needs the fill). One global scalar suffices;
	// the asymmetry comes from the fade, not from per-fake bookkeeping.
	const float FAKE_KERNEL_WEIGHT = 0.45;
	// Per-fake self-weight: own-body fade only.
	//
	// Earlier iterations also did a neighbour-body fade loop here
	// (smoothstep against d_other for every other body) to kill an
	// imagined "two bodies' Gaussian tails stack in the gap" artifact.
	// The debug heatmap proved that wrong: the actual dark lens under
	// each body is uniform across the whole submerged hemisphere, not
	// concentrated in the inter-body gap, so it's caused by fake over-
	// density per body (fixed by FAKE_KERNEL_WEIGHT) rather than inter-
	// body tail overlap. The neighbour loop cost FAKE_TOTAL * MAX_BODIES
	// = 1280 distance() calls per fragment for nothing -- removing it
	// brought FPS from ~50 back to the usual 170+.
	// Per-body body-cull: a fake of body b sits at most br+15 px from
	// body b's centre (cavity outer ring), and its Gaussian reaches
	// another 3*sigma=27 px. So if worldPos is further than br+42 from
	// body b's centre, NONE of b's 80 fakes can contribute and we skip
	// the inner loop entirely. For most fragments (open pool, no nearby
	// body) all 4 bodies cull -> zero fake work.
	const float FAKE_CUTOFF_RADIUS = 27.0;
	const float CAVITY_OUTER = 15.0;
	for (int b = 0; b < MAX_BODIES_RIM; b++) {
		float br = bodies_world[b].z;
		if (br < 0.1) continue;
		float d_body_frag = distance(worldPos, bodies_world[b].xy) - br;
		if (d_body_frag > CAVITY_OUTER + FAKE_CUTOFF_RADIUS) continue;

		int base = b * FAKE_PER_BODY;
		for (int i = 0; i < FAKE_PER_BODY; i++) {
			vec2 fp = fake_particles[base + i];
			if (fp.x < -1e5) continue;

			float d_self = distance(fp, bodies_world[b].xy) - br;
			float fake_self_weight = 1.0 - smoothstep(0.0, 12.0, d_self);

			// surface_boost: cavity is single-layer fakes vs pool's multi-
			// layer real water at the waterline. Driven by fake's own y
			// vs body centre. Combined with F_water lifting the fake
			// layer 5 px above the local waterline (Ground.cs), the
			// metaball isosurface around bodies sits a few px above the
			// open-pool surface -- the "water hugs the body" look.
			float depth_below = max(fp.y - bodies_world[b].y, 0.0);
			float surface_boost = 1.0 + 0.9 * exp(-depth_below * depth_below / 128.0);

			vec2 pdiff = worldPos - fp;
			total_water += particle_contrib(pdiff, inv_2sq, cutoff_sq) * FAKE_KERNEL_WEIGHT * fake_self_weight * surface_boost;
		}
	}


	// Debug heat: must run AFTER fake loop so total_water includes fake
	// contributions (otherwise the cavity-fill / under-body lens we're
	// trying to diagnose is invisible in the heatmap).
	// Use n = total_water * field_scale so the heat axis maps 1:1 onto
	// the real color ramp's thresholds:
	//   n < 0.3   -> bright/light band
	//   n 0.3-0.75 -> light -> mid
	//   n 0.75-1.35 -> mid -> deep
	//   n > 1.35  -> deep (saturates)
	// Heat color stages mirror that:
	//   black -> blue -> cyan -> green -> yellow -> red -> magenta
	// across n in [0, 1.6] so the deep-lens band shows up as red/magenta
	// while the open pool sits around green/yellow.
	if (!drew && (debug_mode == 1 || debug_mode == 2)) {
		float n_dbg = total_water * field_scale;
		float t = clamp(n_dbg / 1.6, 0.0, 1.0);
		vec3 heat;
		if (t < 0.166)      heat = mix(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), t / 0.166);
		else if (t < 0.333) heat = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (t - 0.166) / 0.166);
		else if (t < 0.5)   heat = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.333) / 0.166);
		else if (t < 0.666) heat = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5)   / 0.166);
		else if (t < 0.833) heat = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.666) / 0.166);
		else                heat = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.0, 1.0), (t - 0.833) / 0.166);
		COLOR = vec4(heat, 1.0);
		drew = true;
	}

	if (!drew && total_water > 0.01) {
		float edge_width = fwidth(total_water) * 1.5;
		float aa = smoothstep(2.5 - edge_width, 2.5 + edge_width, total_water);
		if (aa < 0.001) discard;

		float n = total_water * field_scale;

		// Toon quantization: map continuous n to discrete levels
		if (toon_levels > 0.5) {
			n = floor(n * toon_levels) / toon_levels;
		}

		float rim = smoothstep(2.5, 4.0, total_water);
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


		float rim_blend = 1.0 - smoothstep(3.0, 6.0, total_water);
		color = mix(color, rim_color, rim_blend * 0.7);

		vec2 p1 = UV * 25.0 + time * 2.3;
		vec2 p2 = UV * 41.0 - time * 1.7;
		float noise1 = fract(sin(dot(p1, vec2(12.9898, 78.233))) * 43758.5453);
		float noise2 = fract(sin(dot(p2, vec2(31.4159, 67.182))) * 23456.789);
		float noise = (noise1 + noise2) * 0.5;

		color = mix(color, color * 0.92 + 0.04, noise * 0.20);

		float foam = 0.0;
		foam *= smoothstep(0.03, 0.15, n);
		foam *= (0.5 + noise * 0.5);
		foam = clamp(foam, 0.0, 1.0);

		color = mix(color, water_foam, foam);

		// Velocity-driven specular (disabled: velocity data no longer in output texture)
		// TODO: re-implement with field gradient or separate velocity texture

		COLOR = vec4(clamp(color, 0.0, 1.0), aa);
		drew = true;
	}

	if (!drew) discard;
}
