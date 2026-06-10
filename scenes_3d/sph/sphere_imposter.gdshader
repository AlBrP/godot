// Sphere imposter: render a camera-facing quad per particle, compute a sphere
// surface in fragment, write DEPTH so spheres occlude each other correctly.
// Position fetched per-instance from a sampler2D updated by sph3d_output.glsl.
//
// Toon shading: cel-quantised NdotL + rim light, matching scenes_3d/toon.gdshader
// but spatial (not light-pass) since we need to read DEPTH and a custom UV.

shader_type spatial;
render_mode unshaded, depth_draw_always, cull_disabled;

uniform sampler2D position_tex : filter_nearest;
uniform int tex_w = 100;
uniform int tex_h = 100;
uniform float particle_radius = 0.07;

uniform vec3 light_dir = vec3(-0.5, -0.8, -0.3);
uniform vec3 base_color : source_color = vec3(0.30, 0.55, 0.85);
uniform vec3 deep_color : source_color = vec3(0.08, 0.22, 0.45);
uniform float toon_levels = 3.0;
uniform float shadow_floor = 0.25;
uniform float rim_strength = 0.4;
uniform float rim_power = 3.0;
uniform vec3  rim_color : source_color = vec3(1.0, 0.97, 0.90);

varying vec3 v_world_center;
varying vec2 v_local_uv;
varying float v_alpha;

void vertex() {
	// MultiMesh expands one quad per instance; VERTEX is the quad-local position.
	int idx = int(INSTANCE_ID);
	ivec2 tex_coord = ivec2(idx % tex_w, idx / tex_w);
	vec4 sample = texelFetch(position_tex, tex_coord, 0);
	vec3 center = sample.xyz;
	v_alpha = sample.w;

	// quad-local space (in this MultiMesh, instance transform = identity, mesh
	// is a unit quad with corners (±1, ±1, 0)). Build a camera-aligned billboard.
	vec3 right = vec3(INV_VIEW_MATRIX[0].x, INV_VIEW_MATRIX[0].y, INV_VIEW_MATRIX[0].z);
	vec3 up    = vec3(INV_VIEW_MATRIX[1].x, INV_VIEW_MATRIX[1].y, INV_VIEW_MATRIX[1].z);
	vec3 offset = (right * VERTEX.x + up * VERTEX.y) * particle_radius;

	vec3 world_pos = center + offset;
	v_world_center = center;
	v_local_uv = VERTEX.xy;  // [-1, 1]

	// To clip
	POSITION = PROJECTION_MATRIX * VIEW_MATRIX * vec4(world_pos, 1.0);
}

void fragment() {
	float d2 = dot(v_local_uv, v_local_uv);
	if (d2 > 1.0) discard;

	// Compute the sphere normal in view-aligned camera basis, then transform
	// back to world for shading. nz = sqrt(1-r^2) since |n|=1.
	float nz = sqrt(1.0 - d2);
	vec3 right = vec3(INV_VIEW_MATRIX[0].x, INV_VIEW_MATRIX[0].y, INV_VIEW_MATRIX[0].z);
	vec3 up    = vec3(INV_VIEW_MATRIX[1].x, INV_VIEW_MATRIX[1].y, INV_VIEW_MATRIX[1].z);
	vec3 fwd   = vec3(INV_VIEW_MATRIX[2].x, INV_VIEW_MATRIX[2].y, INV_VIEW_MATRIX[2].z);
	vec3 normal = normalize(right * v_local_uv.x + up * v_local_uv.y + fwd * nz);

	// Correct world position of the actual sphere surface point so DEPTH is right.
	vec3 surface_world = v_world_center + normal * particle_radius;
	vec4 clip = PROJECTION_MATRIX * VIEW_MATRIX * vec4(surface_world, 1.0);
	DEPTH = clip.z / clip.w;

	// Toon ramp lighting
	vec3 L = normalize(-light_dir);
	float ndl = max(dot(normal, L), 0.0);
	float cel = floor(ndl * toon_levels) / toon_levels;
	cel = max(cel, shadow_floor);

	// Fresnel-ish rim from view-aligned distance from sphere center
	float rim = pow(d2, rim_power) * rim_strength;

	vec3 col = mix(deep_color, base_color, cel);
	col += rim_color * rim;

	ALBEDO = col;
}
