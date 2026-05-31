using Godot;
using System;

public partial class Ground : StaticBody2D
{
	// Physics params
	[Export] public float smoothing_radius { get; set; } = 17f;
	[Export] public float target_density { get; set; } = 0.01f;
	[Export] public float pressure_multiplier { get; set; } = 1000000f;
	[Export] public float near_pressure_multiplier { get; set; } = 10000f;
	[Export] public float viscosity_strength { get; set; } = 350f;
	[Export] public float velocity_damping { get; set; } = 0.99f;
	[Export] public float gravity { get; set; } = 2500f;
	[Export] public float collision_damping { get; set; } = 0.6f;
	[Export] public int iterations_per_frame { get; set; } = 2;

	// Rendering params
	[Export] public float field_strength { get; set; } = 1.8f;
	[Export] public float sigma { get; set; } = 9f;
	[Export] public float body_threshold { get; set; } = 6.5f;
	[Export] public float edge_threshold { get; set; } = 3.8f;
	[Export] public float stretch_scale { get; set; } = 0.012f;
	[Export] public float density_scale { get; set; } = 0.3f;
	[Export] public float field_scale { get; set; } = 0.018f;
	[Export] public float edge_sharpness { get; set; } = 0.78f;
	[Export] public float spec_strength { get; set; } = 0.5f;
	[Export] public float toon_levels { get; set; } = 0f;
	[Export] public float color_band { get; set; } = 2.2f;

	// Slider panel
	private bool show_sliders_ = false;
	private bool debug_sprites_ = true;
	private int debug_mode_ = 0;
	public int DebugModeVal => debug_mode_;
	private Panel slider_panel_;
	private HSlider toon_slider_, spec_slider_, band_slider_;
	private Label toon_label_, spec_label_, band_label_;

	// Particles
	public const int ball_nums_ = 5000;
	public Vector2[] pos_ = new Vector2[ball_nums_];
	public Vector2[] vel_ = new Vector2[ball_nums_];

	// Bounds
	private Vector2 bounds_min_, bounds_max_;
	private const float BOUND_MARGIN = 5f;

	// Particle sprites
	private Sprite2D[] particle_sprites_;
	[Export] public Texture2D particle_texture { get; set; }

	// Mouse
	public bool mouse_pressed_ = false;
	public bool mouse_right_pressed_ = false;
	private Vector2 mouse_position_ = Vector2.Zero;
	private int spawn_index_ = 0;
	private bool spray_mode_ = true;
	private int last_synced_spawn_ = 0;

	// FPS
	private float fps_time_accum_ = 0f;
	private int fps_frame_count_ = 0;
	private float current_fps_ = 0f;

	// Render
	private Color_Rect colorRect_;
	private SmokeSpriteRenderer smokeSprites_;

	// Pause
	private bool paused_ = false;
	public bool Paused => paused_;

	// GPU
	private SphGpu sph_gpu_;

	// Multi-body
	private const int MAX_BODIES = SphGpu.MAX_BODIES;
	[Export] public NodePath BodyPath0 { get; set; }
	[Export] public NodePath BodyPath1 { get; set; }
	[Export] public NodePath BodyPath2 { get; set; }
	[Export] public NodePath BodyPath3 { get; set; }
	[Export] public float BodyRadius { get; set; } = 50f;
	[Export] public float BodyBV { get; set; } = 100f;        // boundary_volume
	[Export] public float BodyBPS { get; set; } = 0.03f;       // boundary_pressure_scale

	// Gas params
	[Export] public float gas_stiffness { get; set; } = 3.0f;
	[Export] public float buoyancy_alpha { get; set; } = 0.2f;
	[Export] public float ambient_temperature { get; set; } = 300f;
	[Export] public float vorticity_epsilon { get; set; } = 50.0f;
	[Export] public float temp_diffusion_rate { get; set; } = 2.0f;
	[Export] public float particle_lifetime { get; set; } = 2.5f;
	[Export] public float cooling_rate { get; set; } = 180.0f;
	[Export] public float gas_viscosity_ratio { get; set; } = 0.1f;
	[Export] public float body_drag_gas { get; set; } = 0.3f;
	private bool smoke_mode_ = false;
	private bool fire_mode_ = false;
	public bool SmokeMode => smoke_mode_;
	public bool FireMode => fire_mode_;
	[Export] public float fire_temperature { get; set; } = 1500f;
	private float[] smoke_temp_ = new float[ball_nums_];
	private int[] particle_types_ = new int[ball_nums_];

	private int body_count_ = 0;
	private int current_body_ = 0;
	private RigidBody2D[] body_nodes_ = new RigidBody2D[MAX_BODIES];
	private Vector2[] body_pos_ = new Vector2[MAX_BODIES];
	private Vector2[] body_vel_ = new Vector2[MAX_BODIES];
	private float[] body_angle_ = new float[MAX_BODIES];
	private bool[] body_enabled_ = new bool[MAX_BODIES];
	private float[][] sdf_data_ = new float[MAX_BODIES][];
	private Vector2[] sdf_half_extents_ = new Vector2[MAX_BODIES];
	private float[] sdf_shape_radius_ = new float[MAX_BODIES];
	private bool[] sdf_is_box_ = new bool[MAX_BODIES];
	private Polygon2D[] body_visual_ = new Polygon2D[MAX_BODIES];
	private Vector2[] last_body_force_ = new Vector2[MAX_BODIES];
	private const int SDF_SIZE = 64;

	// Waterline-through-body: each body gets a left (xL,yL) and right
	// (xR,yR) anchor. yL/yR are the median y of the 10 water particles
	// nearest to the body's outer left/right edge in x distance.
	// Mapped over [xL..xR] this defines the waterline visually crossing
	// the body, with the median giving immunity to splash outliers.
	private struct Waterline { public float xL, yL, xR, yR; public bool valid; }
	private Waterline[] waterlines_ = new Waterline[MAX_BODIES];
	// Temporal median over the last N frames of raw yL/yR. EMA tracked the
	// Akinci boundary eddy's phase (~0.3s period) and the rim oscillated
	// in sync. A median is immune to periodic noise: even with half the
	// samples high and half low, the middle one lands on the cycle center.
	// 30 @ 60Hz = 0.5s window, fully covering the eddy period. Lag is ~0.25s
	// for body rise/fall, acceptable.
	private const int WATERLINE_HISTORY = 12;
	private float[,] waterline_hist_yL_ = new float[MAX_BODIES, WATERLINE_HISTORY];
	private float[,] waterline_hist_yR_ = new float[MAX_BODIES, WATERLINE_HISTORY];
	private int[] waterline_hist_count_ = new int[MAX_BODIES];
	private int[] waterline_hist_idx_ = new int[MAX_BODIES];

	// Returns waterlines in world coordinates (Ground-local + Position).
	// Output layout for the shader: vec4 per body = (xL, yL, xR, yR);
	// invalid bodies report all zeros and waterline_valid_mask has the
	// corresponding bit cleared so the shader can skip them cheaply.
	public Godot.Collections.Array<Vector4> GetWaterlinesWorld()
	{
		var arr = new Godot.Collections.Array<Vector4>();
		for (int b = 0; b < MAX_BODIES; b++)
		{
			if (waterlines_[b].valid)
			{
				var w = waterlines_[b];
				arr.Add(new Vector4(w.xL + Position.X, w.yL + Position.Y,
				                     w.xR + Position.X, w.yR + Position.Y));
			}
			else
				arr.Add(Vector4.Zero);
		}
		return arr;
	}

	public int GetWaterlineValidMask()
	{
		int m = 0;
		for (int b = 0; b < MAX_BODIES; b++)
			if (waterlines_[b].valid) m |= (1 << b);
		return m;
	}

	// Returns body centers + radii in world coords (vec4 = cx, cy, r, 0).
	// Shader uses these to test whether a fragment is inside a body, so
	// the in-body rim/underwater logic only fires on body fragments.
	public Godot.Collections.Array<Vector4> GetBodiesWorld()
	{
		var arr = new Godot.Collections.Array<Vector4>();
		for (int b = 0; b < MAX_BODIES; b++)
		{
			if (body_enabled_[b])
				arr.Add(new Vector4(body_pos_[b].X + Position.X,
				                     body_pos_[b].Y + Position.Y,
				                     sdf_shape_radius_[b], 0));
			else
				arr.Add(Vector4.Zero);
		}
		return arr;
	}

	// Generates fake water particles for each body to fill the SPH
	// cavity. The hex lattice spacing is tuned per body to the local
	// real-water density: count real water particles in a square window
	// near the body, estimate the average particle spacing, and use that
	// for the fake lattice. This keeps the in-body metaball lobe at the
	// same density as the surrounding pool whether the pool is dense
	// (compressed by the body's weight) or sparse (post-splash). Each
	// body gets up to FAKE_MAX_PER_BODY slots; tighter spacing fills
	// more of them.
	// Persistent fake-particle pool per body. Each particle is a small
	// fluid-like proxy that moves continuously frame to frame instead of
	// being deleted and recreated -- this is what eliminates the visible
	// flicker around the body's rim.
	// Position is stored in BODY-LOCAL space (un-rotated body frame), so
	// translating/rotating the body automatically moves the cluster with
	// it without re-seeding. fake_pos_local_[b, i] is in pixels relative
	// to body center; fake_vel_local_[b, i] is local px/s.
	private const int FAKE_PER_BODY = 80;
	private const int FAKE_TOTAL = MAX_BODIES * FAKE_PER_BODY;
	private Vector2[,] fake_pos_local_ = new Vector2[MAX_BODIES, FAKE_PER_BODY];
	private Vector2[,] fake_vel_local_ = new Vector2[MAX_BODIES, FAKE_PER_BODY];
	private bool[,] fake_alive_ = new bool[MAX_BODIES, FAKE_PER_BODY];
	private bool fake_seeded_ = false;
	// F_real spatial grid: accelerates fake<->real particle repulsion from
	// O(FAKE_PER_BODY * N) to O(FAKE_PER_BODY). Grid is rebuilt once per
	// frame (lazily on first call) and reused across subsequent callers.
	private float f_real_grid_min_x_, f_real_grid_min_y_;
	private int f_real_grid_w_, f_real_grid_h_;
	private const float F_REAL_CELL_SIZE = 11f;
	private int[] f_real_cell_offsets_;
	private int[] f_real_cell_counts_;
	private int[] f_real_particles_;
	private ulong f_real_grid_frame_ = ulong.MaxValue;
	// Per-frame step + output. The pool is initialized lazily (first time
	// a body becomes submerged) and then ONLY moves frame to frame. No
	// despawn / respawn under normal flow -- a fake only resets if it
	// drifts WAY out of bounds.
	//
	// Forces per fake particle each frame:
	//   F_home  = pull toward body center (keeps cluster bound when body moves)
	//   F_sdf   = push along outward SDF normal if it's outside the body+cavity
	//             (effectively the body's surface tension on the fake)
	//   F_real  = repulsion from any nearby real water particle (soft Gaussian)
	//   F_water = push downward if it tries to surface above the waterline
	// + heavy linear damping to keep them roughly stationary unless pushed.
	private void BuildFRealSpatialGrid()
	{
		float minX = bounds_min_.X + Position.X;
		float minY = bounds_min_.Y + Position.Y;
		float maxX = bounds_max_.X + Position.X;
		float maxY = bounds_max_.Y + Position.Y;
		int gw = (int)((maxX - minX) / F_REAL_CELL_SIZE) + 2;
		int gh = (int)((maxY - minY) / F_REAL_CELL_SIZE) + 2;
		int numCells = gw * gh;

		if (f_real_cell_offsets_ == null || f_real_cell_offsets_.Length != numCells)
		{
			f_real_cell_offsets_ = new int[numCells];
			f_real_cell_counts_ = new int[numCells];
		}
		else
		{
			Array.Clear(f_real_cell_offsets_, 0, numCells);
			Array.Clear(f_real_cell_counts_, 0, numCells);
		}
		if (f_real_particles_ == null || f_real_particles_.Length < ball_nums_)
			f_real_particles_ = new int[ball_nums_];

		f_real_grid_w_ = gw;
		f_real_grid_h_ = gh;
		f_real_grid_min_x_ = minX;
		f_real_grid_min_y_ = minY;

		for (int pi = 0; pi < ball_nums_; pi++)
		{
			if (particle_types_[pi] != 0) continue;
			float wx = pos_[pi].X + Position.X;
			float wy = pos_[pi].Y + Position.Y;
			int cx = (int)((wx - minX) / F_REAL_CELL_SIZE);
			int cy = (int)((wy - minY) / F_REAL_CELL_SIZE);
			if (cx < 0) cx = 0; if (cx >= gw) cx = gw - 1;
			if (cy < 0) cy = 0; if (cy >= gh) cy = gh - 1;
			f_real_cell_counts_[cy * gw + cx]++;
		}

		int total = 0;
		for (int i = 0; i < numCells; i++)
		{
			f_real_cell_offsets_[i] = total;
			int cnt = f_real_cell_counts_[i];
			f_real_cell_counts_[i] = 0;
			total += cnt;
		}

		for (int pi = 0; pi < ball_nums_; pi++)
		{
			if (particle_types_[pi] != 0) continue;
			float wx = pos_[pi].X + Position.X;
			float wy = pos_[pi].Y + Position.Y;
			int cx = (int)((wx - minX) / F_REAL_CELL_SIZE);
			int cy = (int)((wy - minY) / F_REAL_CELL_SIZE);
			if (cx < 0) cx = 0; if (cx >= gw) cx = gw - 1;
			if (cy < 0) cy = 0; if (cy >= gh) cy = gh - 1;
			int cellIdx = cy * gw + cx;
			int slot = f_real_cell_offsets_[cellIdx] + f_real_cell_counts_[cellIdx];
			f_real_particles_[slot] = pi;
			f_real_cell_counts_[cellIdx]++;
		}
	}

	public Godot.Collections.Array<Vector2> GetFakeParticlesWorld()
	{
		var arr = new Godot.Collections.Array<Vector2>();
		const float SDF_PADDING = 1.6f;
		const float SPACING = 10f;
		float dt = paused_ ? 0f : Mathf.Min((float)GetProcessDeltaTime(), 1f / 30f);
		if (!fake_seeded_) { SeedFakePoolInitial(SDF_PADDING, SPACING); fake_seeded_ = true; }

		ulong currentFrame = Engine.GetProcessFrames();
		if (f_real_grid_frame_ != currentFrame)
		{
			BuildFRealSpatialGrid();
			f_real_grid_frame_ = currentFrame;
		}

		for (int b = 0; b < MAX_BODIES; b++)
		{
			// Submerged gate (same as before): valid waterline + waterline
			// reasonably close to body + body bottom below waterline.
			bool submerged = false;
			float water_y_flat = 0f;
			float cx = 0f, cy = 0f, ang = 0f;
			float wlxL = 0f, wlyL = 0f, wlxR = 0f, wlyR = 0f, dx_wl = 1f;
			if (waterlines_[b].valid && body_enabled_[b])
			{
				var w = waterlines_[b];
				cx = body_pos_[b].X + Position.X;
				cy = body_pos_[b].Y + Position.Y;
				ang = body_angle_[b];
				float bound = Mathf.Max(sdf_half_extents_[b].X, sdf_half_extents_[b].Y) / SDF_PADDING;
				wlxL = w.xL + Position.X; wlyL = w.yL + Position.Y;
				wlxR = w.xR + Position.X; wlyR = w.yR + Position.Y;
				dx_wl = Mathf.Max(wlxR - wlxL, 0.001f);
				water_y_flat = (wlyL + wlyR) * 0.5f;
				bool near_enough = Mathf.Abs(water_y_flat - cy) < bound * 2.5f;
				bool body_in_water = (cy + bound) > water_y_flat;
				submerged = near_enough && body_in_water;
			}

			if (submerged)
			{
				const float CAVITY_MAX = 15f;
				float ca = Mathf.Cos(-ang), sa = Mathf.Sin(-ang);
				float ca2 = Mathf.Cos(ang), sa2 = Mathf.Sin(ang);
				// Stable per-body grid index for body-local sampling: body's
				// own SDF data (CPU) and the real-water rejection.
				for (int i = 0; i < FAKE_PER_BODY; i++)
				{
					if (!fake_alive_[b, i])
					{
						RespawnFake(b, i, SDF_PADDING);
					}
					Vector2 lp = fake_pos_local_[b, i];
					Vector2 lv = fake_vel_local_[b, i];

					Vector2 force_local = Vector2.Zero;

					// F_sdf: keep inside body+cavity.
					float d = SampleSdfCpu(b, lp);
					const float SDF_BOUNDARY = 12f;
					if (d > -SDF_BOUNDARY)
					{
						Vector2 grad = SampleSdfGradient(b, lp, 1.5f);
						float push_mag = Mathf.Max(0f, d + SDF_BOUNDARY) * 12f;
						if (d > 0f) push_mag += d * 14f;
						force_local -= grad * push_mag;
					}

					// F_self: pair-wise repulsion between fakes within the same body.
					const float R_SELF = 16f;
					const float R_SELF_SQ = R_SELF * R_SELF;
					Vector2 self_repel = Vector2.Zero;
					for (int j = 0; j < FAKE_PER_BODY; j++)
					{
						if (j == i) continue;
						if (!fake_alive_[b, j]) continue;
						Vector2 op = fake_pos_local_[b, j];
						float ox = lp.X - op.X;
						if (ox > R_SELF || ox < -R_SELF) continue;
						float oy = lp.Y - op.Y;
						if (oy > R_SELF || oy < -R_SELF) continue;
						float r2 = ox * ox + oy * oy;
						if (r2 > R_SELF_SQ || r2 < 0.01f) continue;
						float r = Mathf.Sqrt(r2);
						float strength = (R_SELF - r) * 25f;
						self_repel.X += ox / r * strength;
						self_repel.Y += oy / r * strength;
					}
					force_local += self_repel;

					// F_water: linear + quadratic push so surface fakes settle
					// 2-4px below waterline. Old above*20 left ~12-15px of
					// residual lift because F_self (R=16, k=25) on a layer
					// of neighbours produces ~300-400 px/s^2 upward and a
					// linear restoring force needs that much "above" to
					// balance it. Quadratic term hardens response once a
					// fake drifts more than a couple px above target; no
					// hard clamp so no teleport / tangential slide.
					// F_water: push fakes 2 px below the local waterline (soft
					// pull, equilibrium ~3-5 px below the surface). Now that
					// ComputeWaterlines no longer pollutes yR/yL with water
					// from inside a neighbour body, water_y_flat is honest
					// even in the inter-body gap and target_wy lands at the
					// real surface.
					{
						float wy_pre = cy + sa2 * lp.X + ca2 * lp.Y;
						float target_wy = water_y_flat + 2f;
						float above = target_wy - wy_pre;
						if (above > 0f)
						{
							float force_wy = above * 60f + above * above * 5f;
							force_local.X += force_wy * sa2;
							force_local.Y += force_wy * ca2;
						}
					}

					// World position for F_real lookup.
					float wx = cx + ca2 * lp.X - sa2 * lp.Y;
					float wy = cy + sa2 * lp.X + ca2 * lp.Y;

					// F_real: repulsion from nearby real water particles (spatial-grid O(1)).
					const float R_REPEL = 11f;
					const float R_REPEL_SQ = R_REPEL * R_REPEL;
					Vector2 repel_world = Vector2.Zero;
					int gc_x = (int)((wx - f_real_grid_min_x_) / F_REAL_CELL_SIZE);
					int gc_y = (int)((wy - f_real_grid_min_y_) / F_REAL_CELL_SIZE);
					for (int gdx = -1; gdx <= 1; gdx++)
					{
						int cell_x = gc_x + gdx;
						if (cell_x < 0 || cell_x >= f_real_grid_w_) continue;
						for (int gdy = -1; gdy <= 1; gdy++)
						{
							int cell_y = gc_y + gdy;
							if (cell_y < 0 || cell_y >= f_real_grid_h_) continue;
							int cellIdx = cell_y * f_real_grid_w_ + cell_x;
							int start = f_real_cell_offsets_[cellIdx];
							int cnt = f_real_cell_counts_[cellIdx];
							for (int k = 0; k < cnt; k++)
							{
								int pi = f_real_particles_[start + k];
								float rx = (pos_[pi].X + Position.X) - wx;
								if (rx > R_REPEL || rx < -R_REPEL) continue;
								float ry = (pos_[pi].Y + Position.Y) - wy;
								if (ry > R_REPEL || ry < -R_REPEL) continue;
								float r2 = rx * rx + ry * ry;
								if (r2 > R_REPEL_SQ || r2 < 0.01f) continue;
								float r = Mathf.Sqrt(r2);
								float strength = (R_REPEL - r) * 12f;
								repel_world.X -= rx / r * strength;
								repel_world.Y -= ry / r * strength;
							}
						}
					}
					Vector2 repel_local = new Vector2(
						ca * repel_world.X - sa * repel_world.Y,
						sa * repel_world.X + ca * repel_world.Y);
					force_local += repel_local;

					// Integrate.
					lv += force_local * dt;
					lv *= Mathf.Max(0f, 1f - 3f * dt);
					float sp = lv.Length();
					if (sp > 140f) lv = lv * (140f / sp);
					lp += lv * dt;

					// Hard SDF clamp: safety net when integration carries
					// the fake past the body surface. Always recompute the
					// gradient at the post-integration position.
					float d_post = SampleSdfCpu(b, lp);
					if (d_post > CAVITY_MAX)
					{
						Vector2 n = SampleSdfGradient(b, lp, 1.5f);
						lp -= n * (d_post - CAVITY_MAX);
						float vn = lv.X * n.X + lv.Y * n.Y;
						if (vn > 0f) lv -= n * vn;
					}

					fake_pos_local_[b, i] = lp;
					fake_vel_local_[b, i] = lv;

					float bound2 = Mathf.Max(sdf_half_extents_[b].X, sdf_half_extents_[b].Y) / SDF_PADDING + 30f + CAVITY_MAX;
					if (lp.X * lp.X + lp.Y * lp.Y > bound2 * bound2) fake_alive_[b, i] = false;

					float owx = cx + ca2 * lp.X - sa2 * lp.Y;
					float owy = cy + sa2 * lp.X + ca2 * lp.Y;
					arr.Add(new Vector2(owx, owy));
				}
			}
			else
			{
				// Body not submerged: output sentinel (so shader skips fake
				// for this body) but KEEP the existing pos/vel intact. If we
				// mark them dead, the next submersion event re-spawns all
				// 60 fakes at body center with zero velocity, and then
				// F_water slams them collectively into the body's lower
				// half (visible as "fakes cluster at the bottom" right
				// after re-entry). Keeping positions means fakes are still
				// nicely spread inside the body when it re-enters water.
				for (int i = 0; i < FAKE_PER_BODY; i++)
				{
					arr.Add(new Vector2(-1e6f, -1e6f));
				}
			}
		}
		return arr;
	}

	// Initialize fake pool: drop FAKE_PER_BODY particles spread inside each
	// body's true interior on a coarse jittered grid, zero velocity.
	// Initial seed: hex lattice for perfectly even distribution so the
	// very first submersion doesn't cluster particles at body center.
	private void SeedFakePoolInitial(float sdfPadding, float spacing)
	{
		for (int b = 0; b < MAX_BODIES; b++)
		{
			float ex = sdf_half_extents_[b].X / sdfPadding;
			float ey = sdf_half_extents_[b].Y / sdfPadding;
			float hex_h = spacing * 0.8660254f;
			int cols = (int)(ex * 2f / spacing) + 2;
			int rows = (int)(ey * 2f / hex_h) + 2;
			int placed = 0;
			for (int row = 0; row < rows && placed < FAKE_PER_BODY; row++)
			{
				float offset_x = (row % 2 == 0) ? 0f : spacing * 0.5f;
				for (int col = 0; col < cols && placed < FAKE_PER_BODY; col++)
				{
					float lx = -ex + col * spacing + offset_x;
					float ly = -ey + row * hex_h;
					if (SampleSdfCpu(b, new Vector2(lx, ly)) < -12f)
					{
						fake_pos_local_[b, placed] = new Vector2(lx, ly);
						fake_vel_local_[b, placed] = Vector2.Zero;
						fake_alive_[b, placed] = true;
						placed++;
					}
				}
			}
			var rng = new System.Random(12345 + b * 1000);
			for (int i = placed; i < FAKE_PER_BODY; i++) RespawnFake(b, i, sdfPadding, rng);
		}
	}

	private System.Random respawn_rng_ = new System.Random(67890);
	private void RespawnFake(int b, int i, float sdfPadding, System.Random rng = null)
	{
		var r = rng ?? respawn_rng_;
		// Mirror a surviving fake's position so the new particle inherits
		// the cluster's spatial distribution. When a body re-enters water,
		// survivors are already pushed below the waterline by F_water --
		// mirroring places the new particle in the same region instead of
		// the geometric center (visible as "growing from bottom").
		for (int mirror = 0; mirror < 10; mirror++)
		{
			int src = r.Next(FAKE_PER_BODY);
			if (!fake_alive_[b, src]) continue;
			float jitter = 3f;
			float lx = fake_pos_local_[b, src].X + ((float)r.NextDouble() * 2f - 1f) * jitter;
			float ly = fake_pos_local_[b, src].Y + ((float)r.NextDouble() * 2f - 1f) * jitter;
			if (SampleSdfCpu(b, new Vector2(lx, ly)) < -12f)
			{
				fake_pos_local_[b, i] = new Vector2(lx, ly);
				fake_vel_local_[b, i] = Vector2.Zero;
				fake_alive_[b, i] = true;
				return;
			}
		}
		// Fallback: rejection-sample inside the body.
		float ex = sdf_half_extents_[b].X / sdfPadding;
		float ey = sdf_half_extents_[b].Y / sdfPadding;
		for (int t = 0; t < 16; t++)
		{
			float lx = ((float)r.NextDouble() * 2f - 1f) * ex;
			float ly = ((float)r.NextDouble() * 2f - 1f) * ey;
			if (SampleSdfCpu(b, new Vector2(lx, ly)) < -12f)
			{
				fake_pos_local_[b, i] = new Vector2(lx, ly);
				fake_vel_local_[b, i] = Vector2.Zero;
				fake_alive_[b, i] = true;
				return;
			}
		}
		// Last resort: dead center.
		fake_pos_local_[b, i] = Vector2.Zero;
		fake_vel_local_[b, i] = Vector2.Zero;
		fake_alive_[b, i] = true;
	}

	// Tight candidate set: only the very top of the water column near the
	// body edge. Pick N_PICK=10, drop none, average the top K_TOP=2 as
	// the surface layer. Temporal median over WATERLINE_HISTORY frames
	// rejects the periodic Akinci boundary eddy noise -- see the history
	// buffer below.
	private const int WATERLINE_N_PICK = 10;
	private const int WATERLINE_K_SKIP = 0;
	private const int WATERLINE_K_TOP  = 4;
	// Independent debug toggles:
	//   show_fake_debug_      — green dots at fake-particle positions (J toggles body, fake dots stay)
	//   show_waterline_debug_ — yellow waterline anchors + magenta/cyan top candidate circles
	private bool show_fake_debug_ = true;
	private bool show_waterline_debug_ = false;
	[Export] public Vector2 WaterlineDebugOffset { get; set; } = Vector2.Zero;
	private Vector2[,] waterline_picks_left_ = new Vector2[MAX_BODIES, WATERLINE_N_PICK];
	private Vector2[,] waterline_picks_right_ = new Vector2[MAX_BODIES, WATERLINE_N_PICK];

	public override void _Ready()
	{
		ProcessMode = ProcessModeEnum.Always;
		var vs = GetViewportRect().Size;
		bounds_min_ = new Vector2(BOUND_MARGIN, BOUND_MARGIN);
		bounds_max_ = new Vector2(vs.X - BOUND_MARGIN, vs.Y - 100f);

		particle_sprites_ = new Sprite2D[ball_nums_];
		for (int i = 0; i < ball_nums_; i++)
		{
			particle_sprites_[i] = new Sprite2D();
			particle_sprites_[i].Texture = particle_texture;
			particle_sprites_[i].Scale = new Vector2(0.03f, 0.03f);
			particle_sprites_[i].Visible = true;
			AddChild(particle_sprites_[i]);
		}
		ResetParticles();
		for (int i = 0; i < 4; i++)
		{
			var cs = GetChild(i) as CollisionShape2D;
			if (cs != null && cs.GetChildCount() > 0)
			{ var s = cs.GetChild(0) as Sprite2D; if (s != null) s.Visible = false; }
		}

		colorRect_ = GetNode<Color_Rect>("../CanvasLayer/SubVPContainer/SubVP/ColorRect");
		sph_gpu_ = new SphGpu();
		sph_gpu_.Init();
		colorRect_.SetGpuTextures(sph_gpu_.PositionTex, sph_gpu_.HashLookupTex);
		colorRect_.SetGpuPhysicsTexture(sph_gpu_.PhysicsTex);
		smokeSprites_ = GetNode<SmokeSpriteRenderer>("../CanvasLayer/SubVPContainer/SubVP/SmokeSprites");
		smokeSprites_.SetGpuTextures(sph_gpu_.PositionTex, sph_gpu_.PhysicsTex, sph_gpu_.StablePositionTex);
		SetParticleSpritesVisible(true);
		colorRect_.SetGpuMode(true);
		sph_gpu_.ResetParticles(pos_, vel_, Position);
		{
			float warmDt = 1f / 60f;
			DispatchGpu(warmDt, warmDt / iterations_per_frame);
		}
		toon_levels = 4f;

		// Setup bodies
		var bodyPaths = new[] { BodyPath0, BodyPath1, BodyPath2, BodyPath3 };
		for (int b = 0; b < MAX_BODIES; b++)
		{
			if (bodyPaths[b] == null || bodyPaths[b].IsEmpty) continue;
			var node = GetNode<RigidBody2D>(bodyPaths[b]);
			if (node == null) continue;
			body_nodes_[b] = node;
			body_enabled_[b] = true;
			body_count_++;

			var physMat = new PhysicsMaterial();
			physMat.Friction = 0.1f; physMat.Rough = false;
			node.PhysicsMaterialOverride = physMat;
			node.LinearDamp = 0.15f; node.AngularDamp = 3.0f;

			GenerateSdfCircle(b, BodyRadius);
			sph_gpu_.UploadSdfTexture(b, sdf_data_[b]);

			body_visual_[b] = new Polygon2D();
			body_visual_[b].Color = new Color(0.3f + b * 0.15f, 0.85f - b * 0.1f, 0.4f + b * 0.15f);
			body_visual_[b].Visible = false;
			UpdateBodyVisual(b);
			node.CallDeferred(Node.MethodName.AddChild, body_visual_[b]);
		}
		if (body_count_ > 0)
			CreateBodyBounds();

		CreateSliderPanel();
	}

	private void CreateSliderPanel()
	{
		slider_panel_ = new Panel();
		slider_panel_.Position = new Vector2(10, 10);
		slider_panel_.Size = new Vector2(220, 150);
		slider_panel_.Visible = false;
		slider_panel_.Modulate = new Color(1, 1, 1, 0.85f);

		toon_label_ = new Label();
		toon_label_.Position = new Vector2(10, 8);
		toon_label_.Text = "Toon Levels: 0";
		toon_label_.AddThemeFontSizeOverride("font_size", 14);
		slider_panel_.AddChild(toon_label_);

		toon_slider_ = new HSlider();
		toon_slider_.Position = new Vector2(10, 32);
		toon_slider_.Size = new Vector2(200, 20);
		toon_slider_.MinValue = 0; toon_slider_.MaxValue = 10; toon_slider_.Step = 1;
		toon_slider_.Value = toon_levels;
		toon_slider_.ValueChanged += (double val) => { toon_levels = (float)val; };
		slider_panel_.AddChild(toon_slider_);

		spec_label_ = new Label();
		spec_label_.Position = new Vector2(10, 55);
		spec_label_.Text = "Spec: 0.50";
		spec_label_.AddThemeFontSizeOverride("font_size", 14);
		slider_panel_.AddChild(spec_label_);

		spec_slider_ = new HSlider();
		spec_slider_.Position = new Vector2(10, 78);
		spec_slider_.Size = new Vector2(200, 20);
		spec_slider_.MinValue = 0; spec_slider_.MaxValue = 1; spec_slider_.Step = 0.05f;
		spec_slider_.Value = spec_strength;
		spec_slider_.ValueChanged += (double val) => { spec_strength = (float)val; };
		slider_panel_.AddChild(spec_slider_);

		band_label_ = new Label();
		band_label_.Position = new Vector2(10, 80);
		band_label_.Text = "ColorBand: 1.50";
		band_label_.AddThemeFontSizeOverride("font_size", 14);
		slider_panel_.AddChild(band_label_);

		band_slider_ = new HSlider();
		band_slider_.Position = new Vector2(10, 104);
		band_slider_.Size = new Vector2(200, 20);
		band_slider_.MinValue = 0.5; band_slider_.MaxValue = 4.0; band_slider_.Step = 0.1;
		band_slider_.Value = color_band;
		band_slider_.ValueChanged += (double val) => { color_band = (float)val; };
		slider_panel_.AddChild(band_slider_);

		AddChild(slider_panel_);
	}

	private void CreateBodyBounds()
	{
		var vs = GetViewportRect().Size; float t = 50f;
		float left = BOUND_MARGIN, right = vs.X - BOUND_MARGIN;
		float top = BOUND_MARGIN, bottom = vs.Y - 100f;
		float cx = (left + right) / 2f, cy = (top + bottom) / 2f;
		float w = right - left + t * 2f, h = bottom - top + t * 2f;
		CreateBoundWall(new Vector2(cx, bottom + t / 2f), new Vector2(w, t));
		CreateBoundWall(new Vector2(cx, top - t / 2f), new Vector2(w, t));
		CreateBoundWall(new Vector2(left - t / 2f, cy), new Vector2(t, h));
		CreateBoundWall(new Vector2(right + t / 2f, cy), new Vector2(t, h));
	}
	private void CreateBoundWall(Vector2 pos, Vector2 size)
	{
		var w = new StaticBody2D(); w.CollisionLayer = 2; w.CollisionMask = 2; w.Position = pos;
		var s = new RectangleShape2D(); s.Size = size;
		var cs = new CollisionShape2D(); cs.Shape = s; w.AddChild(cs);
		AddChild(w);
	}

	// Bilinear sample of the CPU-side SDF for body b at the given LOCAL
	// (body-relative, un-rotated) position. Returns +infinity-ish outside
	// the texture window so callers can treat "out of range" as far away.
	// d < 0 means inside the shape, d > 0 means outside (in pixels).
	private float SampleSdfCpu(int b, Vector2 localPos)
	{
		float ex = sdf_half_extents_[b].X, ey = sdf_half_extents_[b].Y;
		if (ex <= 0f || ey <= 0f) return 1e6f;
		float u = (localPos.X / ex * 0.5f + 0.5f) * SDF_SIZE - 0.5f;
		float v = (localPos.Y / ey * 0.5f + 0.5f) * SDF_SIZE - 0.5f;
		if (u < 0f || v < 0f || u > SDF_SIZE - 1f || v > SDF_SIZE - 1f) return 1e6f;
		int u0 = (int)Mathf.Floor(u);
		int v0 = (int)Mathf.Floor(v);
		int u1 = Mathf.Min(u0 + 1, SDF_SIZE - 1);
		int v1 = Mathf.Min(v0 + 1, SDF_SIZE - 1);
		float fu = u - u0, fv = v - v0;
		var sdf = sdf_data_[b];
		float d00 = sdf[v0 * SDF_SIZE + u0];
		float d10 = sdf[v0 * SDF_SIZE + u1];
		float d01 = sdf[v1 * SDF_SIZE + u0];
		float d11 = sdf[v1 * SDF_SIZE + u1];
		float d0 = d00 + (d10 - d00) * fu;
		float d1 = d01 + (d11 - d01) * fu;
		return d0 + (d1 - d0) * fv;
	}

	// SDF gradient with sample clamping to avoid texture-boundary artifacts.
	// Falls back to body-center direction when the gradient is degenerate.
	private Vector2 SampleSdfGradient(int b, Vector2 localPos, float eps)
	{
		float ex = sdf_half_extents_[b].X, ey = sdf_half_extents_[b].Y;
		float clampX = Mathf.Max(ex * 0.97f, 1f);
		float clampY = Mathf.Max(ey * 0.97f, 1f);
		float xp = Mathf.Clamp(localPos.X + eps, -clampX, clampX);
		float xm = Mathf.Clamp(localPos.X - eps, -clampX, clampX);
		float yp = Mathf.Clamp(localPos.Y + eps, -clampY, clampY);
		float ym = Mathf.Clamp(localPos.Y - eps, -clampY, clampY);
		float gx = SampleSdfCpu(b, new Vector2(xp, localPos.Y))
		         - SampleSdfCpu(b, new Vector2(xm, localPos.Y));
		float gy = SampleSdfCpu(b, new Vector2(localPos.X, yp))
		         - SampleSdfCpu(b, new Vector2(localPos.X, ym));
		float gl = Mathf.Sqrt(gx * gx + gy * gy);
		if (gl > 0.001f) return new Vector2(gx / gl, gy / gl);
		// Degenerate gradient (e.g. outside SDF domain) — fall back to
		// body-center direction so the particle is at least pulled inward.
		float dl = localPos.Length();
		if (dl > 0.001f) return -localPos / dl;
		return new Vector2(0f, -1f);
	}

	private void GenerateSdfCircle(int bodyIdx, float radius, float padding = 1.6f)
	{
		sdf_shape_radius_[bodyIdx] = radius;
		float ext = radius * padding;
		sdf_half_extents_[bodyIdx] = new Vector2(ext, ext);
		sdf_data_[bodyIdx] = new float[SDF_SIZE * SDF_SIZE];
		for (int y = 0; y < SDF_SIZE; y++)
			for (int x = 0; x < SDF_SIZE; x++)
			{
				float wx = ((x + 0.5f) / SDF_SIZE * 2f - 1f) * ext;
				float wy = ((y + 0.5f) / SDF_SIZE * 2f - 1f) * ext;
				sdf_data_[bodyIdx][y * SDF_SIZE + x] = Mathf.Sqrt(wx * wx + wy * wy) - radius;
			}
	}

	private void GenerateSdfBox(int bodyIdx, float halfW, float halfH, float padding = 1.6f)
	{
		sdf_shape_radius_[bodyIdx] = Mathf.Max(halfW, halfH);
		float extX = halfW * padding, extY = halfH * padding;
		sdf_half_extents_[bodyIdx] = new Vector2(extX, extY);
		sdf_data_[bodyIdx] = new float[SDF_SIZE * SDF_SIZE];
		for (int y = 0; y < SDF_SIZE; y++)
			for (int x = 0; x < SDF_SIZE; x++)
			{
				float wx = ((x + 0.5f) / SDF_SIZE * 2f - 1f) * extX;
				float wy = ((y + 0.5f) / SDF_SIZE * 2f - 1f) * extY;
				float dx = Mathf.Abs(wx) - halfW, dy = Mathf.Abs(wy) - halfH;
				float outside = Mathf.Sqrt(Mathf.Max(dx, 0f) * Mathf.Max(dx, 0f) + Mathf.Max(dy, 0f) * Mathf.Max(dy, 0f));
				float inside = Mathf.Min(Mathf.Max(dx, dy), 0f);
				sdf_data_[bodyIdx][y * SDF_SIZE + x] = outside + inside;
			}
	}

	private void SwitchSdfShape()
	{
		if (body_count_ == 0) return;
		int b = current_body_;
		sdf_is_box_[b] = !sdf_is_box_[b];
		if (sdf_is_box_[b])
			GenerateSdfBox(b, BodyRadius, BodyRadius * 0.7f);
		else
			GenerateSdfCircle(b, BodyRadius);
		sph_gpu_.UploadSdfTexture(b, sdf_data_[b]);
		UpdateBodyVisual(b);
	}

	private void UpdateBodyVisual(int b)
	{
		var vis = body_visual_[b]; if (vis == null) return;
		if (sdf_is_box_[b])
		{
			float hw = BodyRadius, hh = BodyRadius * 0.7f;
			vis.Polygon = new Vector2[] {
				new Vector2(-hw, -hh), new Vector2(hw, -hh),
				new Vector2(hw, hh), new Vector2(-hw, hh) };
		}
		else
		{
			int seg = 32; var pts = new Vector2[seg];
			for (int j = 0; j < seg; j++)
			{ float a = j * Mathf.Pi * 2f / seg; pts[j] = new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * BodyRadius; }
			vis.Polygon = pts;
		}
	}

	public void ResetParticles()
	{
		var vs = GetViewportRect().Size;
		int rows = (int)Mathf.Sqrt(ball_nums_);
		int cols = (ball_nums_ - 1) / rows + 1;
		float spacing = 10f;
		float ox = vs.X / 2, oy = vs.Y * 0.3f;
		for (int i = 0; i < ball_nums_; i++)
		{
			if (spray_mode_ || smoke_mode_ || fire_mode_)
			{
				pos_[i] = new Vector2(-1000f, -1000f); vel_[i] = Vector2.Zero;
				smoke_temp_[i] = 0f;
				particle_types_[i] = 0;
			}
			else
			{
				float x = (i % rows - rows / 2f + 0.5f) * spacing + ox;
				float y = (i / rows - cols / 2f + 0.5f) * spacing + oy;
				pos_[i] = new Vector2(x, y); vel_[i] = Vector2.Zero;
				particle_types_[i] = 0;
			}
		}
		spawn_index_ = 0; last_synced_spawn_ = 0;
	}

	public override void _Process(double delta)
	{
		var vs = GetViewportRect().Size;
		bounds_max_ = new Vector2(vs.X - BOUND_MARGIN, vs.Y - 100f);

		fps_time_accum_ += (float)delta; fps_frame_count_++;
		if (fps_time_accum_ >= 0.5f)
		{ current_fps_ = fps_frame_count_ / fps_time_accum_; fps_time_accum_ = 0f; fps_frame_count_ = 0; }

		if (debug_sprites_)
		{
			// Read back GPU particle positions for accurate sprite display
			byte[] gpuData = sph_gpu_.ReadBackParticleBuffer();
			for (int i = 0; i < ball_nums_; i++)
			{
				float px = BitConverter.ToSingle(gpuData, i * 8);
				float py = BitConverter.ToSingle(gpuData, i * 8 + 4);
				pos_[i] = new Vector2(px, py);
				particle_sprites_[i].Position = pos_[i];
				// Color by type: water=blue, fire=red, smoke=gray, steam=white.
				// type field is stored as float in the particle buffer (so the
				// shader can index ptype tables uniformly), so read it as a
				// float and cast. ToInt32 on those 4 bytes would interpret the
				// IEEE-754 bit pattern as an integer (fire=1.0 -> 0x3F800000 ->
				// int 1065353216 != 1) - that's why everything looked blue.
				int ptype = (int)BitConverter.ToSingle(gpuData, ball_nums_ * 40 + i * 4);
				particle_types_[i] = ptype;
				if (ptype == 1) particle_sprites_[i].Modulate = new Color(1f, 0.3f, 0.05f, 0.8f);
				else if (ptype == 2) particle_sprites_[i].Modulate = new Color(0.6f, 0.6f, 0.6f, 0.8f);
				else if (ptype == 3) particle_sprites_[i].Modulate = new Color(0.9f, 0.9f, 1f, 0.8f);
				else particle_sprites_[i].Modulate = new Color(0.2f, 0.5f, 1f, 0.8f);
			}
			SetParticleSpritesVisible(true);
			colorRect_.Visible = false;
		}
		else
		{
			// Lightweight readback just for waterline computation (pos + type).
			// Same call as debug_sprites_ path but result is consumed only
			// for the waterline; no sprite update.
			byte[] gpuData2 = sph_gpu_.ReadBackParticleBuffer();
			for (int i = 0; i < ball_nums_; i++)
			{
				pos_[i].X = BitConverter.ToSingle(gpuData2, i * 8);
				pos_[i].Y = BitConverter.ToSingle(gpuData2, i * 8 + 4);
				particle_types_[i] = (int)BitConverter.ToSingle(gpuData2, ball_nums_ * 40 + i * 4);
			}
			SetParticleSpritesVisible(false);
			colorRect_.Visible = true;
		}

		ComputeWaterlines();

		QueueRedraw();

		if (slider_panel_ != null && slider_panel_.Visible)
		{ toon_label_.Text = $"Toon Levels: {toon_levels:0}"; spec_label_.Text = $"Spec: {spec_strength:0.00}"; band_label_.Text = $"ColorBand: {color_band:0.2}"; }

		if (!paused_)
		{
			if (!spray_mode_ && !smoke_mode_ && !fire_mode_ && mouse_pressed_)
			{
				float rg = 120f, rs = 30f, gs = 1500f;
				for (int i = 0; i < ball_nums_; i++)
				{
					var diff = mouse_position_ - pos_[i]; float len = diff.Length();
					if (len < rg && len > 0.001f)
					{
						var dir = diff / len;
						if (len < rs) { vel_[i] = dir * gs * (len / rs); vel_[i] *= 0.3f; }
						else { float t = 1f - (len - rs) / (rg - rs); vel_[i] += dir * gs * t * 0.3f; vel_[i] *= 0.95f; }
					}
				}
			}
		}
	}

	public override void _PhysicsProcess(double delta)
	{
		if (paused_) return;

		// Spawn particles at fixed physics rate (60Hz, independent of render FPS)
		if (fire_mode_ && mouse_pressed_)
		{
			int num_jets = 3;
			float jet_spacing = 14f;
			int per_jet = 8;
			float[] jet_temp = { 1450f, 1550f, 1450f };
			float[] jet_vscale = { 0.9f, 1.0f, 0.9f };
			for (int j = 0; j < num_jets; j++)
			{
				float offset_x = (j - 1) * jet_spacing;
				for (int s = 0; s < per_jet; s++)
				{
					int idx = spawn_index_ % ball_nums_;
					spawn_index_++;
					float angle = -Mathf.Pi * 0.5f + (float)GD.RandRange(-Mathf.Pi * 0.18f, Mathf.Pi * 0.18f);
					float speed = (float)GD.RandRange(6f, 28f) * jet_vscale[j];
					var vel = new Vector2(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed);
					vel_[idx] = vel;
					var spawnOffset = new Vector2(offset_x + (float)(GD.Randf() - 0.5) * 4f, 40f + (float)(GD.Randf() - 0.5) * 4f);
					pos_[idx] = mouse_position_ + spawnOffset;
					smoke_temp_[idx] = jet_temp[j];
					particle_types_[idx] = 1; // PTYPE_FIRE
				}
			}
		}
		else if (smoke_mode_ && mouse_pressed_)
		{
			int burst = 12;
			float initTemp = 800f;
			for (int s = 0; s < burst; s++)
			{
				int idx = spawn_index_ % ball_nums_;
				spawn_index_++;
				float angle = -Mathf.Pi * 0.5f + (float)GD.RandRange(-Mathf.Pi * 0.12f, Mathf.Pi * 0.12f);
				float speed = (float)GD.RandRange(100f, 400f);
				var vel = new Vector2(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed);
				vel_[idx] = vel;
				pos_[idx] = mouse_position_ + new Vector2(
					(float)(GD.Randf() - 0.5) * 50f,
					(float)(GD.Randf() - 0.5) * 30f
				);
				smoke_temp_[idx] = initTemp;
				particle_types_[idx] = 2; // PTYPE_SMOKE
			}
		}
		else if (spray_mode_)
		{
			// Left click: water spray (Lv1 default behaviour)
			if (mouse_pressed_)
			{
				int spawn_count = 8;
				for (int s = 0; s < spawn_count; s++)
				{
					int idx = spawn_index_ % ball_nums_;
					spawn_index_++;
					float angle = -Mathf.Pi * 0.5f + (float)GD.RandRange(-Mathf.Pi * 0.04f, Mathf.Pi * 0.04f);
					float speed = (float)GD.RandRange(2000f, 2100f);
					var vel = new Vector2(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed);
					vel_[idx] = vel;
					float along = (float)GD.RandRange(50f, 110f);
					pos_[idx] = mouse_position_ + vel.Normalized() * along;
					smoke_temp_[idx] = ambient_temperature; // 300K — sets water baseline so heat diffusion has the right starting point
					particle_types_[idx] = 0; // PTYPE_WATER
				}
			}
			// Right click: fire jet (Lv2 — emergent phase transition trigger).
			// Small bursts so a water pool can actually heat up rather than
			// getting overwhelmed by fire. Same physics as FIRE mode (ptype
			// table drives stiffness/buoyancy/cooling/lifetime) — only the
			// spawn cadence is lighter.
			if (mouse_right_pressed_)
			{
				int per_burst = 12;
				for (int s = 0; s < per_burst; s++)
				{
					int idx = spawn_index_ % ball_nums_;
					spawn_index_++;
					float angle = -Mathf.Pi * 0.5f + (float)GD.RandRange(-Mathf.Pi * 0.18f, Mathf.Pi * 0.18f);
					float speed = (float)GD.RandRange(8f, 30f);
					var vel = new Vector2(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed);
					vel_[idx] = vel;
					pos_[idx] = mouse_position_ + new Vector2(
						(float)(GD.Randf() - 0.5) * 14f,
						40f + (float)(GD.Randf() - 0.5) * 4f
					);
					smoke_temp_[idx] = fire_temperature; // 1500K — matches ptype_init_temp[FIRE]
					particle_types_[idx] = 1;            // PTYPE_FIRE
				}
			}
		}

		float frameDt = (float)delta;
		float subDt = frameDt / iterations_per_frame;

		for (int b = 0; b < body_count_; b++)
		{
			if (!body_enabled_[b] || body_nodes_[b] == null) continue;
			body_pos_[b] = body_nodes_[b].GlobalPosition - Position;
			body_vel_[b] = body_nodes_[b].LinearVelocity;
			body_angle_[b] = body_nodes_[b].Rotation;
		}

		if ((spray_mode_ || smoke_mode_ || fire_mode_) && spawn_index_ != last_synced_spawn_)
		{
			int cur = spawn_index_ % ball_nums_;
			int last = last_synced_spawn_ % ball_nums_;
			if (cur > last)
			{
				int count = cur - last;
				sph_gpu_.UpdateParticlesBatchWithType(last, pos_, vel_, smoke_temp_, particle_types_, count);
			}
			else if (cur < last)
			{
				// Wrap-around: sync tail then head
				int count1 = ball_nums_ - last;
				int count2 = cur;
				sph_gpu_.UpdateParticlesBatchWithType(last, pos_, vel_, smoke_temp_, particle_types_, count1);
				sph_gpu_.UpdateParticlesBatchWithType(0, pos_, vel_, smoke_temp_, particle_types_, count2);
			}
			last_synced_spawn_ = spawn_index_;
		}
		DispatchGpu(frameDt, subDt);

		// Apply force to each body
		for (int b = 0; b < body_count_; b++)
		{
			if (!body_enabled_[b] || body_nodes_[b] == null) continue;
			var node = body_nodes_[b];

			var (collisionForce, collisionTorque) = sph_gpu_.ReadBackForce(b);
			last_body_force_[b] = collisionForce;

			Vector2 bvel = node.LinearVelocity;
			float maxForce = 120000f;
			float fMag = collisionForce.Length();
			if (fMag > maxForce) collisionForce = collisionForce.Normalized() * maxForce;
			if (fMag > 0.1f) bvel += collisionForce / node.Mass * frameDt;

			float maxVel = 4000f;
			if (bvel.LengthSquared() > maxVel * maxVel) bvel = bvel.Normalized() * maxVel;

			float sr = sdf_shape_radius_[b];
			Vector2 gpos = node.GlobalPosition;
			float gL = bounds_min_.X + Position.X + sr, gR = bounds_max_.X + Position.X - sr;
			float gT = bounds_min_.Y + Position.Y + sr, gB = bounds_max_.Y + Position.Y - sr;
			bool clamped = false;
			if (gpos.X < gL) { gpos.X = gL; if (bvel.X < 0) bvel.X *= -collision_damping; clamped = true; }
			if (gpos.X > gR) { gpos.X = gR; if (bvel.X > 0) bvel.X *= -collision_damping; clamped = true; }
			if (gpos.Y < gT) { gpos.Y = gT; if (bvel.Y < 0) bvel.Y *= -collision_damping; clamped = true; }
			if (gpos.Y > gB) { gpos.Y = gB; if (bvel.Y > 0) bvel.Y *= -collision_damping; clamped = true; }
			if (clamped) node.GlobalPosition = gpos;
			node.LinearVelocity = bvel;

			float inertia = node.Inertia;
			if (inertia < 0.01f) inertia = 0.5f * node.Mass * sr * sr;
			collisionTorque = Mathf.Clamp(collisionTorque, -50000f, 50000f);
			float avel = node.AngularVelocity + (collisionTorque / inertia) * frameDt;
			avel *= Mathf.Max(0f, 1.0f - 3.0f * frameDt);
			avel = Mathf.Clamp(avel, -12f, 12f);
			node.AngularVelocity = avel;
		}
	}

	// Compute per-body waterline anchors (left + right). For each side
	// we look at water particles in a narrow x-band just outside the
	// body, pick the K with smallest y (highest in Y-down) and take
	// their median y. "x-nearest" was wrong: the water pool is a 2D
	// volume, so the 10 particles nearest the body's edge in x include
	// the full water column from surface to floor and the median lands
	// mid-pool. "Highest in a narrow band" anchors on the actual surface;
	// the median over K=10 still rejects a few splash outliers.
	private void ComputeWaterlines()
	{
		// Sample band sits 20-60 px out from the body edge. Earlier 10-35 px
		// landed inside the body's own splash crown after a body got
		// knocked into the air and fell back, so the anchor latched onto
		// the wave-trough surface (10-20 px below the rest pool) and the
		// fake target lagged for ~1-2 s until the splash settled. 20-60 px
		// reaches past the splash crown to the unperturbed pool, so the
		// anchor tracks the true rest surface within one frame.
		const float BAND_INNER = 20f;
		const float BAND_OUTER = 60f;
		// Pick N_PICK highest particles (smallest y in Y-down). Sort by y
		// and drop the top K_SKIP as splash outliers, take median of rest.
		float[] yL_top = new float[WATERLINE_N_PICK];
		float[] xL_top = new float[WATERLINE_N_PICK];
		float[] yR_top = new float[WATERLINE_N_PICK];
		float[] xR_top = new float[WATERLINE_N_PICK];

		for (int b = 0; b < MAX_BODIES; b++)
		{
			waterlines_[b].valid = false;
			if (!body_enabled_[b]) continue;

			float bx = body_pos_[b].X;
			// Per-body horizontal half-width: box bodies are wider than
			// circles, so use sdf_half_extents (unpadded) to anchor the
			// left/right windows. Avoids the box's flat top getting
			// candidates from inside its footprint.
			const float SDF_PAD_W = 1.6f;
			float hw_body = sdf_half_extents_[b].X / SDF_PAD_W;
			float xL = bx - hw_body;
			float xR = bx + hw_body;

			// Neighbour-body window-blocking: when another body overlaps this
			// body's left or right candidate window, "water" in that window is
			// actually leaked-in real water clinging to the lower hemisphere
			// of the neighbour (Akinci SDF push is soft -- a thin sub-surface
			// crust of real water lives a few px inside each body). Those
			// crust particles sit 15-30 px BELOW the true surface, so
			// sampling them tilts water_y_flat downward and drags fake
			// target_wy with it -- exactly what made the gap-side waterline
			// dip toward the inter-body cavity. Block the affected side and
			// mirror the y from the opposite (clean) side at compute time.
			bool xL_blocked = false, xR_blocked = false;
			for (int b2 = 0; b2 < body_count_; b2++)
			{
				if (b2 == b) continue;
				if (!body_enabled_[b2]) continue;
				float bx2 = body_pos_[b2].X;
				float hw2 = sdf_half_extents_[b2].X / SDF_PAD_W;
				float b2_xL = bx2 - hw2;
				float b2_xR = bx2 + hw2;
				if (b2_xR > xL - BAND_OUTER && b2_xL < xL - BAND_INNER) xL_blocked = true;
				if (b2_xR > xR + BAND_INNER && b2_xL < xR + BAND_OUTER) xR_blocked = true;
			}

			int nL = 0, nR = 0;
			float worstL_y = float.NegativeInfinity; int worstL_i = 0;
			float worstR_y = float.NegativeInfinity; int worstR_i = 0;

			for (int i = 0; i < ball_nums_; i++)
			{
				if (particle_types_[i] != 0) continue;
				float px = pos_[i].X;
				float py = pos_[i].Y;
				if (py < -500f) continue;
				if (!xL_blocked && px < xL - BAND_INNER && px >= xL - BAND_OUTER)
				{
					if (nL < WATERLINE_N_PICK)
					{
						yL_top[nL] = py; xL_top[nL] = px;
						if (py > worstL_y) { worstL_y = py; worstL_i = nL; }
						nL++;
						if (nL == WATERLINE_N_PICK)
						{
							worstL_y = yL_top[0]; worstL_i = 0;
							for (int j = 1; j < WATERLINE_N_PICK; j++) if (yL_top[j] > worstL_y) { worstL_y = yL_top[j]; worstL_i = j; }
						}
					}
					else if (py < worstL_y)
					{
						yL_top[worstL_i] = py; xL_top[worstL_i] = px;
						worstL_y = yL_top[0]; worstL_i = 0;
						for (int j = 1; j < WATERLINE_N_PICK; j++) if (yL_top[j] > worstL_y) { worstL_y = yL_top[j]; worstL_i = j; }
					}
				}
				else if (!xR_blocked && px > xR + BAND_INNER && px <= xR + BAND_OUTER)
				{
					if (nR < WATERLINE_N_PICK)
					{
						yR_top[nR] = py; xR_top[nR] = px;
						if (py > worstR_y) { worstR_y = py; worstR_i = nR; }
						nR++;
						if (nR == WATERLINE_N_PICK)
						{
							worstR_y = yR_top[0]; worstR_i = 0;
							for (int j = 1; j < WATERLINE_N_PICK; j++) if (yR_top[j] > worstR_y) { worstR_y = yR_top[j]; worstR_i = j; }
						}
					}
					else if (py < worstR_y)
					{
						yR_top[worstR_i] = py; xR_top[worstR_i] = px;
						worstR_y = yR_top[0]; worstR_i = 0;
						for (int j = 1; j < WATERLINE_N_PICK; j++) if (yR_top[j] > worstR_y) { worstR_y = yR_top[j]; worstR_i = j; }
					}
				}
			}

			if (xL_blocked && xR_blocked) continue;
			if (!xL_blocked && nL < WATERLINE_N_PICK) continue;
			if (!xR_blocked && nR < WATERLINE_N_PICK) continue;

			// Sort y ascending (smallest = highest in Y-down). Drop the
			// first K_SKIP (splash outliers). Average the next K_TOP -- the
			// surface layer mean -- instead of taking the median which falls
			// into layer-2 when the surface is only 1-2 particles thick.
			float medL = 0f, medR = 0f;
			if (!xL_blocked)
			{
				float[] sortL = new float[WATERLINE_N_PICK];
				Array.Copy(yL_top, sortL, WATERLINE_N_PICK); Array.Sort(sortL);
				float sumL = 0f;
				for (int j = WATERLINE_K_SKIP; j < WATERLINE_K_SKIP + WATERLINE_K_TOP; j++) sumL += sortL[j];
				medL = sumL / WATERLINE_K_TOP;
			}
			if (!xR_blocked)
			{
				float[] sortR = new float[WATERLINE_N_PICK];
				Array.Copy(yR_top, sortR, WATERLINE_N_PICK); Array.Sort(sortR);
				float sumR = 0f;
				for (int j = WATERLINE_K_SKIP; j < WATERLINE_K_SKIP + WATERLINE_K_TOP; j++) sumR += sortR[j];
				medR = sumR / WATERLINE_K_TOP;
			}
			// Mirror the blocked side from the clean side.
			if (xL_blocked) medL = medR;
			if (xR_blocked) medR = medL;

			waterlines_[b].xL = xL;
			waterlines_[b].xR = xR;
			// Push raw median into ring buffer, then take temporal median
			// over the window. Filters the periodic boundary-eddy noise.
			int hi = waterline_hist_idx_[b];
			waterline_hist_yL_[b, hi] = medL;
			waterline_hist_yR_[b, hi] = medR;
			waterline_hist_idx_[b] = (hi + 1) % WATERLINE_HISTORY;
			if (waterline_hist_count_[b] < WATERLINE_HISTORY) waterline_hist_count_[b]++;
			int hc = waterline_hist_count_[b];
			float[] hyL = new float[hc];
			float[] hyR = new float[hc];
			for (int j = 0; j < hc; j++) { hyL[j] = waterline_hist_yL_[b, j]; hyR[j] = waterline_hist_yR_[b, j]; }
			Array.Sort(hyL); Array.Sort(hyR);
			waterlines_[b].yL = hc % 2 == 1 ? hyL[hc / 2] : (hyL[hc / 2 - 1] + hyL[hc / 2]) * 0.5f;
			waterlines_[b].yR = hc % 2 == 1 ? hyR[hc / 2] : (hyR[hc / 2 - 1] + hyR[hc / 2]) * 0.5f;
			waterlines_[b].valid = true;

			for (int j = 0; j < WATERLINE_N_PICK; j++)
			{
				waterline_picks_left_[b, j]  = new Vector2(xL_top[j], yL_top[j]);
				waterline_picks_right_[b, j] = new Vector2(xR_top[j], yR_top[j]);
			}
		}
	}

	private void DispatchGpu(float frameDt, float subDt)
	{
		var bodies = new SphGpu.BodyInfo[MAX_BODIES];
		for (int b = 0; b < body_count_; b++)
		{
			bodies[b] = new SphGpu.BodyInfo
			{
				pos = body_pos_[b],
				vel = body_vel_[b],
				sdfHalfExtents = sdf_half_extents_[b],
				angle = body_angle_[b],
				shapeRadius = sdf_shape_radius_[b],
				boundaryVolume = BodyBV,
				bpScale = BodyBPS,
				enabled = body_enabled_[b] ? 1 : 0
			};
		}
		sph_gpu_.UploadBodyData(bodies);
		sph_gpu_.UpdateParams(frameDt, subDt, gravity, velocity_damping,
			smoothing_radius, pressure_multiplier, near_pressure_multiplier,
			viscosity_strength, collision_damping, 1f / 60f, 2000f,
			bounds_min_, bounds_max_, mouse_pressed_, mouse_position_,
			120f, 30f, 1500f, Position, spray_mode_, 1.8f, target_density, body_count_,
			gas_stiffness, buoyancy_alpha, fire_mode_ ? 2 : (smoke_mode_ ? 1 : 0),
			vorticity_epsilon, temp_diffusion_rate,
			(fire_mode_ ? particle_lifetime * 0.3f : particle_lifetime), ambient_temperature,
			cooling_rate, gas_viscosity_ratio,
			body_drag_gas,
			// ptype_stiffness: [water=0, fire=0.75, smoke=3.0, steam=3.0]
			0f, gas_stiffness * 0.25f, gas_stiffness, gas_stiffness,
			// ptype_buoyancy: [water=0, fire=0.14, smoke=0.2, steam=0.2]
			0f, buoyancy_alpha * 0.7f, buoyancy_alpha, buoyancy_alpha,
			// ptype_viscosity: [water=1.0, fire=0.1, smoke=0.1, steam=0.1]
			1.0f, gas_viscosity_ratio, gas_viscosity_ratio, gas_viscosity_ratio,
			// ptype_vorticity: [water=0, fire=50, smoke=50, steam=50]
			0f, vorticity_epsilon, vorticity_epsilon, vorticity_epsilon,
			// ptype_diffusion: [water=100, fire=2, smoke=2, steam=2]
			// Water diffusion bumped to 100: the poly6 kernel coupling is
			// sparse and water has a much larger effective heat capacity
			// than gas, so a moderate value still leaves a visible gap
			// around the fire jet. 100 lets the cavity edge water boil
			// fast enough to fill the gap with rising steam.
			100f, temp_diffusion_rate, temp_diffusion_rate, temp_diffusion_rate,
			// ptype_cooling: [water=30, fire=2700, smoke=180, steam=0]
			// Steam cooling = 0 keeps boiled-off vapour at its init_temp so it
			// doesn't immediately condense back into water (user prefers the
			// vapour to ride up and disperse, not rain back down).
			30f, cooling_rate * 15f, cooling_rate, 0f,
			// ptype_init_temp: [water=0, fire=1500, smoke=800, steam=500]
			0f, fire_temperature, 800f, 500f,
			// ptype_lifetime: [water=99999, fire=lifetime*0.3, smoke=lifetime*0.5, steam=6.0]
			// Steam lifetime extended (was 2.0s) so the rising vapour cloud
			// has time to thicken — short lifetime leaves sparse SDF puffs
			// with visible gaps between them in the metaball render.
			99999f, particle_lifetime * 0.3f, particle_lifetime * 0.5f, 6.0f,
			// ptype_near_pressure_scale defaults (water=1, fire=0.5, smoke=0.3, steam=0.3)
			1f, 0.5f, 0.3f, 0.3f,
			// ptype_boil_point: water boils at 320K (~47°C demo threshold).
			// Real water needs 373K but the SPH temperature scale is loose
			// and the visible cavity around a fire jet looks broken if water
			// can't boil fast enough to fill it. 320 lets cavity-edge water
			// flip to STEAM within ~1s of contact.
			320f, -1f, -1f, -1f,
			// ptype_boil_product: water -> STEAM (=3); others disabled
			3f, -1f, -1f, -1f,
			// ptype_condense_point: disabled across the board (steam previously
			// re-condensed below 320K, but user wants boiled vapour to stay
			// vapour — keep the field for future liquids that should reverse).
			-1f, -1f, -1f, -1f,
			// ptype_condense_product: all disabled (paired with condense_point)
			-1f, -1f, -1f, -1f,
			// Akinci 2012 boundary friction: damps tangential component of
			// water velocity relative to a body's surface so gravity doesn't
			// drive sliding around the submerged surface. 0.5 = 50% damp at
			// the surface, fades to 0 at smoothing_radius. Tangential only,
			// so buoyancy (normal pressure response) is preserved.
			0.5f);
		sph_gpu_.DispatchFrame(frameDt, iterations_per_frame);
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventMouse mouse)
		{ mouse_position_ = mouse.Position - Position; mouse_pressed_ = Input.IsMouseButtonPressed(MouseButton.Left); mouse_right_pressed_ = Input.IsMouseButtonPressed(MouseButton.Right); }
		if (@event is InputEventKey key && key.Pressed && !key.Echo)
		{
			if (key.Keycode == Key.Space) { paused_ = !paused_; GetTree().Paused = paused_; }
			if (key.Keycode == Key.H) spec_strength = spec_strength > 0.01f ? 0f : 0.5f;
			if (key.Keycode == Key.T) toon_levels = toon_levels < 0.5f ? 4f : 0f;
			if (key.Keycode == Key.V) { show_sliders_ = !show_sliders_; if (slider_panel_ != null) slider_panel_.Visible = show_sliders_; }
			if (key.Keycode == Key.F) { fire_mode_ = !fire_mode_; if (fire_mode_) smoke_mode_ = false; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
			if (key.Keycode == Key.Key1) { debug_mode_ = (debug_mode_ + 1) % 3; }
			if (key.Keycode == Key.M) { smoke_mode_ = !smoke_mode_; if (smoke_mode_) fire_mode_ = false; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
			if (key.Keycode == Key.G) { debug_sprites_ = !debug_sprites_; SetParticleSpritesVisible(debug_sprites_); }
			if (key.Keycode == Key.N && body_count_ > 0) { current_body_ = (current_body_ + 1) % body_count_; SwitchSdfShape(); }
			if (key.Keycode == Key.B) { spray_mode_ = !spray_mode_; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
			if (key.Keycode == Key.J)
			{
				// Hide body polygons so the fake-particle distribution is
				// directly visible against the water render. Toggles all 4.
				for (int bi = 0; bi < MAX_BODIES; bi++)
					if (body_visual_[bi] != null) body_visual_[bi].Visible = !body_visual_[bi].Visible;
			}
			if (key.Keycode == Key.K) show_waterline_debug_ = !show_waterline_debug_;
			if (key.Keycode == Key.L) show_fake_debug_ = !show_fake_debug_;
		}
	}

	public void SetParticleSpritesVisible(bool visible)
	{ for (int i = 0; i < ball_nums_; i++) particle_sprites_[i].Visible = visible; }

	public override void _Draw()
	{
		var font = ThemeDB.FallbackFont;
		var vs = GetViewportRect().Size;
		const float right_margin = 12f;
		float rx = vs.X - right_margin, y = 22f, lh = 20f;

		var fps_text = $"FPS: {current_fps_:0}";
		var fps_color = current_fps_ >= 60 ? Colors.Green : current_fps_ >= 30 ? Colors.Yellow : Colors.Red;
		DrawString(font, new Vector2(rx - font.GetStringSize(fps_text).X, y), fps_text, fontSize: 16, modulate: fps_color);
		y += lh;
		if (paused_) { DrawString(font, new Vector2(rx - font.GetStringSize("PAUSED").X, y), "PAUSED", fontSize: 16, modulate: Colors.Yellow); y += lh; }
		var spec_text = $"Spec: {spec_strength:0.00}";
		DrawString(font, new Vector2(rx - font.GetStringSize(spec_text).X, y), spec_text, fontSize: 14, modulate: spec_strength > 0.01f ? Colors.White : Colors.Gray);
		y += lh;
		var toon_text = $"Toon: {toon_levels:0}";
		DrawString(font, new Vector2(rx - font.GetStringSize(toon_text).X, y), toon_text, fontSize: 14, modulate: toon_levels > 0.5f ? Colors.Orange : Colors.Gray);
		y += lh;
		var mode_text = fire_mode_ ? "FIRE" : (smoke_mode_ ? "SMOKE" : (spray_mode_ ? "SPRAY (L:water R:fire)" : "GRAB"));
		var mode_color = fire_mode_ ? Colors.Red : (smoke_mode_ ? Colors.Orange : (spray_mode_ ? Colors.Cyan : Colors.Green));
		DrawString(font, new Vector2(rx - font.GetStringSize(mode_text).X, y), mode_text, fontSize: 14, modulate: mode_color);
		y += lh;
		if (smoke_mode_)
		{
			var gas_text = $"Gas K:{gas_stiffness:0.0} Buoy:{buoyancy_alpha:0.0} Vort:{vorticity_epsilon:0.0}";
			DrawString(font, new Vector2(rx - font.GetStringSize(gas_text).X, y), gas_text, fontSize: 12, modulate: Colors.Gray);
		}
		y += lh;
		if (body_count_ > 0)
			DrawString(font, new Vector2(rx - font.GetStringSize($"Body:{current_body_}").X, y), $"Body:{current_body_}", fontSize: 14, modulate: Colors.Yellow);

		for (int b = 0; b < body_count_; b++)
		{
			if (!body_enabled_[b]) continue;
			DrawSetTransform(body_pos_[b], body_angle_[b], Vector2.One);
			if (sdf_is_box_[b])
			{
				float hw = BodyRadius, hh = BodyRadius * 0.7f;
				var rect = new Rect2(-hw, -hh, hw * 2, hh * 2);
				DrawRect(rect, new Color(0.4f, 0.4f, 0.4f, 0.5f));
				DrawRect(rect, new Color(0.8f, 0.8f, 0.8f, 0.8f), false, 2f);
			}
			else
			{
				DrawCircle(Vector2.Zero, sdf_shape_radius_[b], new Color(0.4f, 0.4f, 0.4f, 0.5f));
				DrawCircle(Vector2.Zero, sdf_shape_radius_[b], new Color(0.8f, 0.8f, 0.8f, 0.8f), false, 2f);
			}
			DrawSetTransform(Vector2.Zero, 0f, Vector2.One);

			var ft = last_body_force_[b];
			var ft_text = $"F{ft.Length():0}";
			DrawString(font, new Vector2(body_pos_[b].X - 30, body_pos_[b].Y - sdf_shape_radius_[b] - 22), ft_text, fontSize: 13,
				modulate: ft.LengthSquared() > 1f ? Colors.Yellow : Colors.Gray);
		}

		// Debug overlays (toggled via L/K).
		Vector2 dbgOff = WaterlineDebugOffset;
		if (show_fake_debug_)
		{
			var fakes = GetFakeParticlesWorld();
			foreach (var fp in fakes)
			{
				if (fp.X < -1e5f) continue;
				DrawCircle(fp - Position + dbgOff, 3f, new Color(0f, 1f, 0f, 0.9f));
			}
		}
		if (show_waterline_debug_)
		{
			Vector2 off = dbgOff;
			for (int b = 0; b < MAX_BODIES; b++)
			{
				if (!waterlines_[b].valid) continue;
				var w = waterlines_[b];
				DrawLine(new Vector2(w.xL, w.yL) + off, new Vector2(w.xR, w.yR) + off, Colors.Yellow, 2f);
				DrawCircle(new Vector2(w.xL, w.yL) + off, 4f, Colors.Yellow);
				DrawCircle(new Vector2(w.xR, w.yR) + off, 4f, Colors.Yellow);

				int[] orderL = new int[WATERLINE_N_PICK];
				int[] orderR = new int[WATERLINE_N_PICK];
				for (int j = 0; j < WATERLINE_N_PICK; j++) { orderL[j] = j; orderR[j] = j; }
				Array.Sort(orderL, (a, c) => waterline_picks_left_[b, a].Y.CompareTo(waterline_picks_left_[b, c].Y));
				Array.Sort(orderR, (a, c) => waterline_picks_right_[b, a].Y.CompareTo(waterline_picks_right_[b, c].Y));

				for (int rank = 0; rank < WATERLINE_N_PICK; rank++)
				{
					bool dropped = rank < WATERLINE_K_SKIP;
					float a = dropped ? 0.25f : 0.9f;
					DrawCircle(waterline_picks_left_[b,  orderL[rank]] + off, 5f, new Color(1, 0, 1, a), false, 1.5f);
					DrawCircle(waterline_picks_right_[b, orderR[rank]] + off, 5f, new Color(0, 1, 1, a), false, 1.5f);
				}
			}
		}
	}
}
