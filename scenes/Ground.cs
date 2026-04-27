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
	private bool debug_sprites_ = false;
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

	public override void _Ready()
	{
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
		SetParticleSpritesVisible(false);
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
			}
			else
			{
				float x = (i % rows - rows / 2f + 0.5f) * spacing + ox;
				float y = (i / rows - cols / 2f + 0.5f) * spacing + oy;
				pos_[i] = new Vector2(x, y); vel_[i] = Vector2.Zero;
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
				particle_sprites_[i].Position = Position + pos_[i];
				// Color by temperature
				float temp = BitConverter.ToSingle(gpuData, ball_nums_ * 32 + i * 4);
				float tn = Mathf.Clamp(temp / 1000f, 0f, 1f);
				particle_sprites_[i].Modulate = new Color(1f, 0.5f * (1f - tn) + 0.1f, tn * 0.3f, 0.8f);
			}
			SetParticleSpritesVisible(true);
			colorRect_.Visible = false;
		}
		else
		{
			SetParticleSpritesVisible(false);
			colorRect_.Visible = true;
		}

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
			int per_jet = 4;
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
			}
		}
		else if (spray_mode_ && mouse_pressed_)
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
				if (smoke_mode_ || fire_mode_) sph_gpu_.UpdateParticlesBatchWithTemp(last, pos_, vel_, smoke_temp_, count);
				else sph_gpu_.UpdateParticlesBatch(last, pos_, vel_, count);
			}
			else if (cur < last)
			{
				// Wrap-around: sync tail then head
				int count1 = ball_nums_ - last;
				int count2 = cur;
				if (smoke_mode_ || fire_mode_)
				{
					sph_gpu_.UpdateParticlesBatchWithTemp(last, pos_, vel_, smoke_temp_, count1);
					sph_gpu_.UpdateParticlesBatchWithTemp(0, pos_, vel_, smoke_temp_, count2);
				}
				else
				{
					sph_gpu_.UpdateParticlesBatch(last, pos_, vel_, count1);
					sph_gpu_.UpdateParticlesBatch(0, pos_, vel_, count2);
				}
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
			body_drag_gas);
		sph_gpu_.DispatchFrame(frameDt, iterations_per_frame);
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventMouse mouse)
		{ mouse_position_ = mouse.Position - Position; mouse_pressed_ = Input.IsMouseButtonPressed(MouseButton.Left); }
		if (@event is InputEventKey key && key.Pressed && !key.Echo)
		{
			if (key.Keycode == Key.Space) paused_ = !paused_;
			if (key.Keycode == Key.H) spec_strength = spec_strength > 0.01f ? 0f : 0.5f;
			if (key.Keycode == Key.T) toon_levels = toon_levels < 0.5f ? 4f : 0f;
			if (key.Keycode == Key.V) { show_sliders_ = !show_sliders_; if (slider_panel_ != null) slider_panel_.Visible = show_sliders_; }
			if (key.Keycode == Key.F) { fire_mode_ = !fire_mode_; if (fire_mode_) smoke_mode_ = false; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
			if (key.Keycode == Key.Key1) { if (fire_mode_) debug_mode_ = (debug_mode_ + 1) % 3; }
			if (key.Keycode == Key.M) { smoke_mode_ = !smoke_mode_; if (smoke_mode_) fire_mode_ = false; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
			if (key.Keycode == Key.G) { debug_sprites_ = !debug_sprites_; SetParticleSpritesVisible(debug_sprites_); }
			if (key.Keycode == Key.N && body_count_ > 0) { current_body_ = (current_body_ + 1) % body_count_; SwitchSdfShape(); }
			if (key.Keycode == Key.B) { spray_mode_ = !spray_mode_; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
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
		var mode_text = fire_mode_ ? "FIRE" : (smoke_mode_ ? "SMOKE" : (spray_mode_ ? "SPRAY" : "GRAB"));
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
	}
}
