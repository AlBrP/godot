shader_type canvas_item;
render_mode blend_mix;

// SPH GPU pipeline output (stable view, pidx-indexed). Channels:
//   rg = world position
//   b  = particle type / 16 (decoded by *16)
//   a  = remaining life fraction for smoke/steam: 1.0 = just spawned,
//        0.0 = about to be recycled. Used to drive age/color AND fade-out.
uniform sampler2D position_tex : filter_nearest;
uniform int tex_width = 72;

// Fade-out window. The last `fade_out_window` fraction of life ramps alpha
// from 1 -> 0 so the particle disappears smoothly instead of popping when
// integrate.glsl teleports it off-screen.
uniform float fade_out_window = 0.25;

// Per-particle visual is driven by temperature_norm (a channel of position_tex).
// For smoke, init_temp = 800K, ambient = 300K, so temp_norm runs ~1.0 (fresh)
// down to ~0.375 (fully aged). We map this to age_t in [0, 1].
// Tuned to match the reference photo: overall white puffs, tightly overlapping,
// small age-driven gradient. Avoid extreme alpha drop (ghost-disc artifacts) and
// avoid huge size growth (sparse aged-particle gaps).
// Cartoon-style: each puff has a solid interior with a clean rounded lobe edge.
// alpha is high so the interior reads as opaque flat color (cartoon shading),
// but a moderate edge gradient still lets adjacent discs merge into lobed
// silhouettes rather than hard intersecting circles.
uniform float sprite_size_min = 60.0;
uniform float sprite_size_max = 100.0;
// Low single-sprite alpha: many sprites must stack to read as opaque white.
// This lets age-based color survive overdraw and matches the reference's
// "translucent puffs that consolidate into a solid mass" look.
uniform float alpha_fresh = 0.55;
uniform float alpha_aged = 0.38;
// Gaussian falloff sharpness. Lower = puffier (very soft, blends seamlessly
// with neighbors). Higher = tighter core (visible disc edge returns).
// disc = exp(-dist² * falloff_k). At dist=1 (sprite edge mid): k=3 -> 0.05,
// k=2 -> 0.13. The smaller, the more each puff fades to nothing at its edge.
uniform float falloff_k = 3.0;
// Three-stop gradient. Driven by v_age_t (temperature) so each puff starts
// gray at the spawn point and whitens as it cools — independent of where on
// screen the spawner is. Color is safe to key off per-sprite age now because
// stable_position_tex keeps INSTANCE_ID -> particle binding fixed: MultiMesh
// draws in stable order each frame, so alpha-over composition is also stable
// (the historical "world-Y" workaround was needed only while sorted_indices
// reshuffled draw order every frame).
uniform vec4 hot_color  : source_color = vec4(0.62, 0.65, 0.70, 1.0);
uniform vec4 mid_color  : source_color = vec4(0.86, 0.88, 0.91, 1.0);
uniform vec4 cool_color : source_color = vec4(1.00, 1.00, 1.00, 1.0);
// age_t range over which color transitions: <lo = fully gray (fresh),
// >hi = fully white (aged). Tweak hi up to keep gray longer.
uniform float color_t_lo = 0.05;
uniform float color_t_hi = 0.55;

// Per-sprite static deformation: each INSTANCE_ID picks a fixed
// aspect-ratio and rotation derived from a hash. Stays constant for the
// sprite's whole life -> no flicker. This breaks the "perfect circle" look.
uniform float aspect_min = 0.65;
uniform float aspect_max = 1.45;

const int PTYPE_SMOKE = 2;
const int PTYPE_STEAM = 3;

varying flat float v_visible;
varying flat float v_age_t;     // 0 = fresh (gray), 1 = aged (white); drives size/alpha/color
varying flat float v_life_frac; // remaining life, drives end-of-life alpha fade

void vertex() {
	int idx = INSTANCE_ID;
	int x = idx % tex_width;
	int y = idx / tex_width;
	vec4 pdata = texelFetch(position_tex, ivec2(x, y), 0);
	vec2 worldPos = pdata.rg;
	int ptype = int(round(pdata.b * 16.0));
	float life_frac = pdata.a;

	bool is_smoke = (ptype == PTYPE_SMOKE || ptype == PTYPE_STEAM);
	bool active = worldPos.y > -500.0;
	v_visible = (is_smoke && active) ? 1.0 : 0.0;

	v_life_frac = life_frac;
	// age_t: 0 (fresh) -> 1 (aged). Directly inverted from life_frac since
	// the sph_output pass already scaled it to a clean [0, 1] range.
	v_age_t = clamp(1.0 - life_frac, 0.0, 1.0);

	if (v_visible < 0.5) {
		VERTEX = vec2(-100000.0, -100000.0);
	} else {
		float sz = mix(sprite_size_min, sprite_size_max, v_age_t);

		// Per-sprite stable deformation derived from INSTANCE_ID.
		// fract(sin()) hash gives stable per-instance noise so the shape
		// is unique per particle but does NOT change frame to frame.
		float seed = float(idx) * 0.017 + 0.123;
		float rA = fract(sin(seed * 12.989) * 43758.545);
		float rR = fract(sin(seed * 78.233) * 24634.612);
		float aspect = mix(aspect_min, aspect_max, rA);   // x scale
		float angle  = rR * 6.28318;                       // base rotation
		float ca = cos(angle);
		float sa = sin(angle);

		// Per-instance static aspect + rotation only. Velocity-driven
		// stretching was removed because physics_tex.rg jitters between SPH
		// substeps, which made every sprite visibly "tremble" each frame.
		vec2 v = vec2(VERTEX.x * aspect, VERTEX.y / aspect);
		v = vec2(ca * v.x - sa * v.y, sa * v.x + ca * v.y);

		VERTEX = v * sz + worldPos;
	}
}

void fragment() {
	if (v_visible < 0.5) discard;

	// Pure Gaussian disc. No flat-top core -> overlapping sprites blend into
	// one continuous puff instead of showing each other's elliptical outline.
	vec2 centered = UV - vec2(0.5);
	float dist_sq = dot(centered, centered) * 4.0;  // dist=1 at sprite edge
	float disc = exp(-dist_sq * falloff_k);

	// Color driven by per-sprite age (temperature). Safe now that
	// stable_position_tex keeps INSTANCE_ID -> particle binding fixed across
	// frames, so MultiMesh draw order is stable and overlapping sprites no
	// longer flip the resulting blend each frame.
	float t = smoothstep(color_t_lo, color_t_hi, v_age_t);
	vec3 col;
	if (t < 0.5) col = mix(hot_color.rgb, mid_color.rgb, t * 2.0);
	else         col = mix(mid_color.rgb, cool_color.rgb, (t - 0.5) * 2.0);
	float a = mix(alpha_fresh, alpha_aged, v_age_t);
	// Smoothly fade out during the last `fade_out_window` of life so the
	// particle dissolves rather than popping when integrate teleports it.
	float fade_out = smoothstep(0.0, fade_out_window, v_life_frac);
	COLOR = vec4(col, disc * a * fade_out);
}
