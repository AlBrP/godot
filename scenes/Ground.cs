using Godot;
using System;

public partial class Ground : StaticBody2D
{
	// 物理参数 — 仅 Inspector 调整
	[Export] public float smoothing_radius { get; set; } = 17f;
	[Export] public float target_density { get; set; } = 0.01f;
	[Export] public float pressure_multiplier { get; set; } = 1000000f;
	[Export] public float near_pressure_multiplier { get; set; } = 10000f;
	[Export] public float viscosity_strength { get; set; } = 350f;
	[Export] public float velocity_damping { get; set; } = 0.99f;
	[Export] public float gravity { get; set; } = 2500f;
	[Export] public float collision_damping { get; set; } = 0.6f;
	[Export] public int iterations_per_frame { get; set; } = 2;

	// 渲染参数（运行时可调）
	[Export] public float field_strength { get; set; } = 1.8f;
	[Export] public float sigma { get; set; } = 9f;
	[Export] public float body_threshold { get; set; } = 6.5f;
	[Export] public float edge_threshold { get; set; } = 3.8f;
	[Export] public float stretch_scale { get; set; } = 0.012f;
	[Export] public float density_scale { get; set; } = 0.3f;
	[Export] public float field_scale { get; set; } = 0.018f;

	// 卡通水参数
	[Export] public float edge_sharpness { get; set; } = 0.78f;
	[Export] public float spec_strength { get; set; } = 0.5f;
	[Export] public float toon_levels { get; set; } = 0f;

	// 运行时滑块面板
	private bool show_sliders_ = false;
	private Panel slider_panel_;
	private HSlider toon_slider_;
	private Label toon_label_;
	private HSlider spec_slider_;
	private Label spec_label_;

	// 粒子数量
	public const int ball_nums_ = 5000;

	// 粒子数据（仅用于spawn同步到GPU）
	public Vector2[] pos_ = new Vector2[ball_nums_];
	public Vector2[] vel_ = new Vector2[ball_nums_];

	// 边界
	private Vector2 bounds_min_;
	private Vector2 bounds_max_;
	private const float BOUND_MARGIN = 5f;

	// 粒子精灵
	private Sprite2D[] particle_sprites_;
	[Export] public Texture2D particle_texture { get; set; }

	// 鼠标交互
	public bool mouse_pressed_ = false;
	private Vector2 mouse_position_ = Vector2.Zero;
	private int spawn_index_ = 0;
	private bool spray_mode_ = true;
	private int last_synced_spawn_ = 0;

	// FPS 统计
	private float fps_time_accum_ = 0f;
	private int fps_frame_count_ = 0;
	private float current_fps_ = 0f;

	// metaball 渲染引用
	private Color_Rect colorRect_;

	// 暂停
	private bool paused_ = false;
	public bool Paused => paused_;

	// GPU Compute
	private SphGpu sph_gpu_;

	// 刚体
	[Export] public NodePath BodyPath { get; set; }
	[Export] public float BodyRadius { get; set; } = 50f;
	private RigidBody2D body_node_;
	private Vector2 body_pos_ = Vector2.Zero;
	private Vector2 body_vel_ = Vector2.Zero;
	private bool body_enabled_ = false;
	private float body_angle_ = 0f;
	private Vector2 last_body_force_;

	// SDF
	private const int SDF_SIZE = 64;
	private float[] sdf_data_;
	private Vector2 sdf_half_extents_ = Vector2.Zero;
	private float sdf_shape_radius_ = 0f;
	private bool sdf_is_box_ = false;
	private Polygon2D body_visual_;


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
			{
				var sprite = cs.GetChild(0) as Sprite2D;
				if (sprite != null) sprite.Visible = false;
			}
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

		if (BodyPath != null && !BodyPath.IsEmpty)
		{
			body_node_ = GetNode<RigidBody2D>(BodyPath);
			body_enabled_ = body_node_ != null;
			if (body_enabled_)
			{
				GenerateSdfCircle(BodyRadius);
				sph_gpu_.UploadSdfTexture(sdf_data_);
			}
		}

		if (body_enabled_)
		{
			var physMat = new PhysicsMaterial();
			physMat.Friction = 0.1f;
			physMat.Rough = false;
			body_node_.PhysicsMaterialOverride = physMat;
			body_node_.LinearDamp = 0.15f;
			body_node_.AngularDamp = 3.0f;
		}

		if (body_enabled_)
		{
			CreateBodyBounds();
			body_visual_ = new Polygon2D();
			body_visual_.Color = new Color(0.3f, 0.85f, 0.4f);
			UpdateBodyVisual();
			body_node_.CallDeferred(Node.MethodName.AddChild, body_visual_);
		}

		CreateSliderPanel();
	}

	private void CreateSliderPanel()
	{
		slider_panel_ = new Panel();
		slider_panel_.Position = new Vector2(10, 10);
		slider_panel_.Size = new Vector2(220, 100);
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
		toon_slider_.MinValue = 0;
		toon_slider_.MaxValue = 10;
		toon_slider_.Step = 1;
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
		spec_slider_.MinValue = 0;
		spec_slider_.MaxValue = 1;
		spec_slider_.Step = 0.05;
		spec_slider_.Value = spec_strength;
		spec_slider_.ValueChanged += (double val) => { spec_strength = (float)val; };
		slider_panel_.AddChild(spec_slider_);

		AddChild(slider_panel_);
	}

	private void CreateBodyBounds()
	{
		var vs = GetViewportRect().Size;
		float t = 50f;
		float left = BOUND_MARGIN;
		float right = vs.X - BOUND_MARGIN;
		float top = BOUND_MARGIN;
		float bottom = vs.Y - 100f;
		float cx = (left + right) / 2f;
		float cy = (top + bottom) / 2f;
		float w = right - left + t * 2f;
		float h = bottom - top + t * 2f;

		CreateBoundWall(new Vector2(cx, bottom + t / 2f), new Vector2(w, t));
		CreateBoundWall(new Vector2(cx, top - t / 2f), new Vector2(w, t));
		CreateBoundWall(new Vector2(left - t / 2f, cy), new Vector2(t, h));
		CreateBoundWall(new Vector2(right + t / 2f, cy), new Vector2(t, h));
	}

	private void CreateBoundWall(Vector2 position, Vector2 size)
	{
		var wall = new StaticBody2D();
		wall.CollisionLayer = 2;
		wall.CollisionMask = 2;
		wall.Position = position;
		var shape = new RectangleShape2D();
		shape.Size = size;
		var cs = new CollisionShape2D();
		cs.Shape = shape;
		wall.AddChild(cs);
		AddChild(wall);
	}

	private void GenerateSdfCircle(float radius, float padding = 1.6f)
	{
		sdf_shape_radius_ = radius;
		float ext = radius * padding;
		sdf_half_extents_ = new Vector2(ext, ext);
		sdf_data_ = new float[SDF_SIZE * SDF_SIZE];
		for (int y = 0; y < SDF_SIZE; y++)
			for (int x = 0; x < SDF_SIZE; x++)
			{
				float wx = ((x + 0.5f) / SDF_SIZE * 2f - 1f) * ext;
				float wy = ((y + 0.5f) / SDF_SIZE * 2f - 1f) * ext;
				sdf_data_[y * SDF_SIZE + x] = Mathf.Sqrt(wx * wx + wy * wy) - radius;
			}
	}

	private void GenerateSdfBox(float halfW, float halfH, float padding = 1.6f)
	{
		sdf_shape_radius_ = Mathf.Max(halfW, halfH);
		float extX = halfW * padding, extY = halfH * padding;
		sdf_half_extents_ = new Vector2(extX, extY);
		sdf_data_ = new float[SDF_SIZE * SDF_SIZE];
		for (int y = 0; y < SDF_SIZE; y++)
			for (int x = 0; x < SDF_SIZE; x++)
			{
				float wx = ((x + 0.5f) / SDF_SIZE * 2f - 1f) * extX;
				float wy = ((y + 0.5f) / SDF_SIZE * 2f - 1f) * extY;
				float dx = Mathf.Abs(wx) - halfW, dy = Mathf.Abs(wy) - halfH;
				float outside = Mathf.Sqrt(Mathf.Max(dx, 0f) * Mathf.Max(dx, 0f) + Mathf.Max(dy, 0f) * Mathf.Max(dy, 0f));
				float inside = Mathf.Min(Mathf.Max(dx, dy), 0f);
				sdf_data_[y * SDF_SIZE + x] = outside + inside;
			}
	}

	private void SwitchSdfShape()
	{
		sdf_is_box_ = !sdf_is_box_;
		if (sdf_is_box_)
			GenerateSdfBox(BodyRadius, BodyRadius * 0.7f);
		else
			GenerateSdfCircle(BodyRadius);
		sph_gpu_.UploadSdfTexture(sdf_data_);
		UpdateBodyVisual();
	}

	private void UpdateBodyVisual()
	{
		if (body_visual_ == null) return;
		if (sdf_is_box_)
		{
			float hw = BodyRadius, hh = BodyRadius * 0.7f;
			body_visual_.Polygon = new Vector2[] {
				new Vector2(-hw, -hh), new Vector2(hw, -hh),
				new Vector2(hw, hh), new Vector2(-hw, hh)
			};
		}
		else
		{
			int seg = 32; var pts = new Vector2[seg];
			for (int j = 0; j < seg; j++)
			{ float a = j * Mathf.Pi * 2f / seg; pts[j] = new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * BodyRadius; }
			body_visual_.Polygon = pts;
		}
	}

	public void ResetParticles()
	{
		var vs = GetViewportRect().Size;
		int rows = (int)Mathf.Sqrt(ball_nums_);
		int cols = (ball_nums_ - 1) / rows + 1;
		float spacing = 10f;
		float offset_x = vs.X / 2;
		float offset_y = vs.Y * 0.3f;

		for (int i = 0; i < ball_nums_; i++)
		{
			if (spray_mode_)
			{
				pos_[i] = new Vector2(-1000f, -1000f);
				vel_[i] = Vector2.Zero;
			}
			else
			{
				float x = (i % rows - rows / 2f + 0.5f) * spacing + offset_x;
				float y = (i / rows - cols / 2f + 0.5f) * spacing + offset_y;
				pos_[i] = new Vector2(x, y);
				vel_[i] = Vector2.Zero;
			}
		}
		spawn_index_ = 0;
		last_synced_spawn_ = 0;
	}

	public override void _Process(double delta)
	{
		var vs = GetViewportRect().Size;
		bounds_max_ = new Vector2(vs.X - BOUND_MARGIN, vs.Y - 100f);

		fps_time_accum_ += (float)delta;
		fps_frame_count_++;
		if (fps_time_accum_ >= 0.5f)
		{
			current_fps_ = fps_frame_count_ / fps_time_accum_;
			fps_time_accum_ = 0f;
			fps_frame_count_ = 0;
		}

		if (colorRect_ == null || !colorRect_.MetaballEnabled)
		{
			for (int i = 0; i < ball_nums_; i++)
				particle_sprites_[i].Position = pos_[i];
		}

		QueueRedraw();

		if (slider_panel_ != null && slider_panel_.Visible)
		{
			toon_label_.Text = $"Toon Levels: {toon_levels:0}";
			spec_label_.Text = $"Spec: {spec_strength:0.00}";
		}

		if (!paused_)
		{
			if (spray_mode_ && mouse_pressed_)
			{
				int spawn_count = 8;
				for (int s = 0; s < spawn_count && spawn_index_ < ball_nums_; s++, spawn_index_++)
				{
					float angle = (float)GD.RandRange(-Mathf.Pi * 0.252, -Mathf.Pi * 0.238);
					float speed = (float)GD.RandRange(2000f, 2100f);
					Vector2 vel = new Vector2(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed);
					vel_[spawn_index_] = vel;
					float along_dist = (float)GD.RandRange(50f, 110f);
					Vector2 dir_norm = vel.Normalized();
					pos_[spawn_index_] = mouse_position_ + dir_norm * along_dist;
				}
			}
			else if (!spray_mode_ && mouse_pressed_)
			{
				float r_grab = 120f, r_stick = 30f, grab_speed = 1500f;
				for (int i = 0; i < ball_nums_; i++)
				{
					Vector2 diff = mouse_position_ - pos_[i];
					float len = diff.Length();
					if (len < r_grab && len > 0.001f)
					{
						Vector2 dir = diff / len;
						if (len < r_stick) { vel_[i] = dir * grab_speed * (len / r_stick); vel_[i] *= 0.3f; }
						else { float t = 1f - (len - r_stick) / (r_grab - r_stick); vel_[i] += dir * grab_speed * t * 0.3f; vel_[i] *= 0.95f; }
					}
				}
			}
		}
	}

	public override void _PhysicsProcess(double delta)
	{
		if (paused_) return;

		float frameDt = (float)delta;
		float subDt = frameDt / iterations_per_frame;

		if (body_enabled_ && body_node_ != null)
		{
			body_pos_ = body_node_.GlobalPosition - Position;
			body_vel_ = body_node_.LinearVelocity;
			body_angle_ = body_node_.Rotation;
		}

		if (spray_mode_ && spawn_index_ > last_synced_spawn_)
		{
			int count = spawn_index_ - last_synced_spawn_;
			if (count > 0)
				sph_gpu_.UpdateParticlesBatch(last_synced_spawn_, pos_, vel_, count);
			last_synced_spawn_ = spawn_index_;
		}
		DispatchGpu(frameDt, subDt);

		// Apply SPH collision force to rigid body
		if (body_enabled_ && body_node_ != null)
		{
			var (collisionForce, collisionTorque) = sph_gpu_.ReadBackForce();
			last_body_force_ = collisionForce;

			Vector2 bvel = body_node_.LinearVelocity;
			float maxForce = 120000f;
			float fMag = collisionForce.Length();
			if (fMag > maxForce)
				collisionForce = collisionForce.Normalized() * maxForce;
			if (fMag > 0.1f)
				bvel += collisionForce / body_node_.Mass * frameDt;

			float maxVel = 4000f;
			if (bvel.LengthSquared() > maxVel * maxVel)
				bvel = bvel.Normalized() * maxVel;

			Vector2 gpos = body_node_.GlobalPosition;
			float gLeft = bounds_min_.X + Position.X + sdf_shape_radius_;
			float gRight = bounds_max_.X + Position.X - sdf_shape_radius_;
			float gTop = bounds_min_.Y + Position.Y + sdf_shape_radius_;
			float gBottom = bounds_max_.Y + Position.Y - sdf_shape_radius_;
			bool clamped = false;
			if (gpos.X < gLeft) { gpos.X = gLeft; if (bvel.X < 0) bvel.X *= -collision_damping; clamped = true; }
			if (gpos.X > gRight) { gpos.X = gRight; if (bvel.X > 0) bvel.X *= -collision_damping; clamped = true; }
			if (gpos.Y < gTop) { gpos.Y = gTop; if (bvel.Y < 0) bvel.Y *= -collision_damping; clamped = true; }
			if (gpos.Y > gBottom) { gpos.Y = gBottom; if (bvel.Y > 0) bvel.Y *= -collision_damping; clamped = true; }
			if (clamped) body_node_.GlobalPosition = gpos;
			body_node_.LinearVelocity = bvel;

			float inertia = body_node_.Inertia;
			if (inertia < 0.01f)
				inertia = 0.5f * body_node_.Mass * sdf_shape_radius_ * sdf_shape_radius_;
			collisionTorque = Mathf.Clamp(collisionTorque, -50000f, 50000f);
			float avel = body_node_.AngularVelocity + (collisionTorque / inertia) * frameDt;
			avel *= Mathf.Max(0f, 1.0f - 3.0f * frameDt);
			avel = Mathf.Clamp(avel, -12f, 12f);
			body_node_.AngularVelocity = avel;
		}
	}

	private void DispatchGpu(float frameDt, float subDt)
	{
		sph_gpu_.UpdateParams(
			frameDt, subDt, gravity, velocity_damping,
			smoothing_radius, pressure_multiplier, near_pressure_multiplier,
			viscosity_strength, collision_damping,
			1f / 60f, 2000f,
			bounds_min_, bounds_max_,
			mouse_pressed_, mouse_position_,
			120f, 30f, 1500f,
			Position,
			spray_mode_,
			body_pos_, sdf_half_extents_, body_enabled_,
			body_vel_, body_angle_, sdf_shape_radius_,
			100f, 0.03f, 1.8f, target_density
		);
		sph_gpu_.DispatchFrame(frameDt, iterations_per_frame);
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventMouse mouse)
		{
			mouse_position_ = mouse.Position - Position;
			mouse_pressed_ = Input.IsMouseButtonPressed(MouseButton.Left);
		}
		if (@event is InputEventKey key && key.Pressed && !key.Echo)
		{
			if (key.Keycode == Key.Space) paused_ = !paused_;
			if (key.Keycode == Key.H) spec_strength = spec_strength > 0.01f ? 0f : 0.5f;
			if (key.Keycode == Key.T) toon_levels = toon_levels < 0.5f ? 4f : 0f;
			if (key.Keycode == Key.V) { show_sliders_ = !show_sliders_; if (slider_panel_ != null) slider_panel_.Visible = show_sliders_; }
			if (key.Keycode == Key.N && body_enabled_) SwitchSdfShape();
			if (key.Keycode == Key.B) { spray_mode_ = !spray_mode_; ResetParticles(); sph_gpu_.ResetParticles(pos_, vel_, Position); }
		}
	}

	public void SetParticleSpritesVisible(bool visible)
	{
		for (int i = 0; i < ball_nums_; i++)
			particle_sprites_[i].Visible = visible;
	}

	public override void _Draw()
	{
		var font = ThemeDB.FallbackFont;
		var vs = GetViewportRect().Size;
		const float right_margin = 12f;
		float right_x = vs.X - right_margin;
		float y = 22f;
		float line_h = 20f;

		var fps_text = $"FPS: {current_fps_:0}";
		var fps_color = current_fps_ >= 60 ? Colors.Green : current_fps_ >= 30 ? Colors.Yellow : Colors.Red;
		DrawString(font, new Vector2(right_x - font.GetStringSize(fps_text).X, y), fps_text, fontSize: 16, modulate: fps_color);
		y += line_h;

		if (paused_)
		{
			DrawString(font, new Vector2(right_x - font.GetStringSize("PAUSED").X, y), "PAUSED", fontSize: 16, modulate: Colors.Yellow);
			y += line_h;
		}

		var spec_text = $"Spec: {spec_strength:0.00}";
		DrawString(font, new Vector2(right_x - font.GetStringSize(spec_text).X, y), spec_text, fontSize: 14,
			modulate: spec_strength > 0.01f ? Colors.White : Colors.Gray);
		y += line_h;

		var toon_text = $"Toon: {toon_levels:0}";
		DrawString(font, new Vector2(right_x - font.GetStringSize(toon_text).X, y), toon_text, fontSize: 14,
			modulate: toon_levels > 0.5f ? Colors.Orange : Colors.Gray);
		y += line_h;

		var mode_text = spray_mode_ ? "SPRAY" : "GRAB";
		DrawString(font, new Vector2(right_x - font.GetStringSize(mode_text).X, y), mode_text, fontSize: 14,
			modulate: spray_mode_ ? Colors.Cyan : Colors.Green);

		if (body_enabled_)
		{
			DrawSetTransform(body_pos_, body_angle_, Vector2.One);
			if (sdf_is_box_)
			{
				float hw = BodyRadius, hh = BodyRadius * 0.7f;
				var rect = new Rect2(-hw, -hh, hw * 2, hh * 2);
				DrawRect(rect, new Color(0.4f, 0.4f, 0.4f, 0.5f));
				DrawRect(rect, new Color(0.8f, 0.8f, 0.8f, 0.8f), false, 2f);
			}
			else
			{
				DrawCircle(Vector2.Zero, sdf_shape_radius_, new Color(0.4f, 0.4f, 0.4f, 0.5f));
				DrawCircle(Vector2.Zero, sdf_shape_radius_, new Color(0.8f, 0.8f, 0.8f, 0.8f), false, 2f);
			}
			DrawSetTransform(Vector2.Zero, 0f, Vector2.One);

			var force_text = $"Force:{last_body_force_.Length():0}";
			DrawString(font, new Vector2(body_pos_.X - 60, body_pos_.Y - sdf_shape_radius_ - 22), force_text, fontSize: 13,
				modulate: last_body_force_.LengthSquared() > 1f ? Colors.Yellow : Colors.Gray);
		}
	}
}
