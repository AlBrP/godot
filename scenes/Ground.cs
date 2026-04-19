using Godot;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class Ground : StaticBody2D
{
	// 物理参数（参考Fluid-Sim）— 仅 Inspector 调整，运行时不再显示
	[Export] public float smoothing_radius { get; set; } = 17f;
	[Export] public float target_density { get; set; } = 0.01f;
	[Export] public float pressure_multiplier { get; set; } = 1000000f;
	[Export] public float near_pressure_multiplier { get; set; } = 10000f;
	[Export] public float viscosity_strength { get; set; } = 350f;
	[Export] public float velocity_damping { get; set; } = 0.99f;
	[Export] public float gravity { get; set; } = 2500f;
	[Export] public float collision_damping { get; set; } = 0.3f;
	[Export] public int iterations_per_frame { get; set; } = 2;

	// 渲染参数（运行时可调）
	[Export] public float field_strength { get; set; } = 1.8f;
	[Export] public float sigma { get; set; } = 9f;           // 世界坐标像素（原0.012归一化）
	[Export] public float body_threshold { get; set; } = 6.5f;
	[Export] public float edge_threshold { get; set; } = 3.8f;
	[Export] public float stretch_scale { get; set; } = 0.012f;
	[Export] public float density_scale { get; set; } = 0.3f;
	[Export] public float field_scale { get; set; } = 0.008f;

	// 卡通水参数（已简化）
	[Export] public float edge_sharpness { get; set; } = 0.78f;  // 越高边缘越窄锐利

	// UI滑块（渲染参数）
	private HSlider fieldStrengthSlider;
	private HSlider sigmaSlider;
	private HSlider bodyThresholdSlider;
	private HSlider edgeThresholdSlider;
	private HSlider stretchScaleSlider;
	private HSlider densityScaleSlider;
	private HSlider fieldScaleSlider;

	private Label fieldStrengthLabel;
	private Label sigmaLabel;
	private Label bodyThresholdLabel;
	private Label edgeThresholdLabel;
	private Label stretchScaleLabel;
	private Label densityScaleLabel;
	private Label fieldScaleLabel;

	// 粒子数量
	public const int ball_nums_ = 5000;

	// 数据数组
	public Vector2[] pos_ = new Vector2[ball_nums_];
	private Vector2[] predicted_pos_ = new Vector2[ball_nums_];  // 预测位置
	public Vector2[] vel_ = new Vector2[ball_nums_];
	private float[] density_ = new float[ball_nums_];
	private float[] near_density_ = new float[ball_nums_];  // 近密度

	// 核函数缩放因子
	private float poly6_scale;
	private float spiky_pow2_scale;
	private float spiky_pow3_scale;
	private float spiky_pow2_deriv_scale;
	private float spiky_pow3_deriv_scale;

	// 边界
	private Vector2 bounds_min_;
	private Vector2 bounds_max_;
	private const float BOUND_MARGIN = 5f;

	// 邻居缓存（方案4：数组替代List）
	private int[][] neighbor_cache_;
	private int[] neighbor_count_;
	private const int MAX_NEIGHBORS = 100;

	// 粒子精灵（渲染OFF模式）
	private Sprite2D[] particle_sprites_;
	[Export] public Texture2D particle_texture { get; set; }

	// 鼠标交互
	public bool mouse_pressed_ = false;
	private Vector2 mouse_position_ = Vector2.Zero;

	// FPS 统计
	private float fps_time_accum_ = 0f;
	private int fps_frame_count_ = 0;
	private float current_fps_ = 0f;
	public float physics_ms_ = 0f;
	public float build_tex_ms_ = 0f;

	// metaball 渲染引用
	private Color_Rect colorRect_;

	// 当前子步 dt
	private float current_sub_dt_ = 0f;

	// 暂停
	private bool paused_ = false;
	public bool Paused => paused_;


	public override void _Ready()
	{
		// 初始化边界（底部留出空间）
		var viewport_size = GetViewportRect().Size;
		bounds_min_ = new Vector2(BOUND_MARGIN, BOUND_MARGIN);
		bounds_max_ = new Vector2(viewport_size.X - BOUND_MARGIN, viewport_size.Y - 100f);

		// 初始化邻居缓存（方案4）
		neighbor_cache_ = new int[ball_nums_][];
		neighbor_count_ = new int[ball_nums_];
		for (int i = 0; i < ball_nums_; i++)
		{
			neighbor_cache_[i] = new int[MAX_NEIGHBORS];
		}

		// 初始化粒子精灵
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

		// 隐藏旧边界精灵，用 _Draw() 画动态边界线
		for (int i = 0; i < 4; i++)
		{
			var cs = GetChild(i) as CollisionShape2D;
			if (cs != null && cs.GetChildCount() > 0)
			{
				var sprite = cs.GetChild(0) as Sprite2D;
				if (sprite != null) sprite.Visible = false;
			}
		}

		HashInit();
		UpdateKernelScales();

		// 获取 metaball ColorRect 引用
		colorRect_ = GetNode<Color_Rect>("../CanvasLayer/SubVPContainer/SubVP/ColorRect");

		// 创建UI（合并面板）
		CreateControlPanel();
	}

	private void CreateControlPanel()
	{
		var canvas = new CanvasLayer();
		canvas.Layer = 10;
		AddChild(canvas);

		// 半透明面板容器
		var panel = new PanelContainer();
		panel.Position = new Vector2(8, 8);

		var style = new StyleBoxFlat();
		style.BgColor = new Godot.Color(0.08f, 0.08f, 0.12f, 0.85f);
		style.CornerRadiusTopLeft = 6;
		style.CornerRadiusTopRight = 6;
		style.CornerRadiusBottomLeft = 6;
		style.CornerRadiusBottomRight = 6;
		style.ContentMarginLeft = 10;
		style.ContentMarginRight = 10;
		style.ContentMarginTop = 8;
		style.ContentMarginBottom = 8;
		panel.AddThemeStyleboxOverride("panel", style);
		canvas.AddChild(panel);

		var vbox = new VBoxContainer();
		vbox.AddThemeConstantOverride("separation", 4);
		panel.AddChild(vbox);

		// 标题行：Reset按钮 + 渲染模式提示
		var titleRow = new HBoxContainer();
		vbox.AddChild(titleRow);

		var resetBtn = new Button();
		resetBtn.Text = "Reset [Space]";
		resetBtn.CustomMinimumSize = new Vector2(110, 28);
		resetBtn.Pressed += ResetParticles;
		titleRow.AddChild(resetBtn);

		var hintLabel = new Label();
		hintLabel.Text = "  R: toggle render   P: pause";
		hintLabel.Modulate = new Godot.Color(0.7f, 0.7f, 0.75f, 1f);
		hintLabel.VerticalAlignment = VerticalAlignment.Center;
		titleRow.AddChild(hintLabel);

		// 分隔线
		var hSep = new HSeparator();
		vbox.AddChild(hSep);

		// 滑块区
		var sliderVBox = new VBoxContainer();
		sliderVBox.AddThemeConstantOverride("separation", 2);
		vbox.AddChild(sliderVBox);

		AddSlider(sliderVBox, "Field", ref fieldStrengthLabel, ref fieldStrengthSlider,
			field_strength, 0.1f, 10.0f, 0.1f, (v) => { field_strength = v; fieldStrengthLabel.Text = $"Field: {v:F1}"; });

		AddSlider(sliderVBox, "Sigma", ref sigmaLabel, ref sigmaSlider,
			sigma, 4f, 30f, 1f, (v) => { sigma = v; sigmaLabel.Text = $"Sigma: {v:F0}"; });

		AddSlider(sliderVBox, "BodyThresh", ref bodyThresholdLabel, ref bodyThresholdSlider,
			body_threshold, 1.0f, 30.0f, 0.5f, (v) => { body_threshold = v; bodyThresholdLabel.Text = $"BodyThresh: {v:F1}"; });

		AddSlider(sliderVBox, "EdgeThresh", ref edgeThresholdLabel, ref edgeThresholdSlider,
			edge_threshold, 1.0f, 10.0f, 0.1f, (v) => { edge_threshold = v; edgeThresholdLabel.Text = $"EdgeThresh: {v:F1}"; });

		AddSlider(sliderVBox, "Stretch", ref stretchScaleLabel, ref stretchScaleSlider,
			stretch_scale, 0.0f, 0.01f, 0.001f, (v) => { stretch_scale = v; stretchScaleLabel.Text = $"Stretch: {v:F3}"; });

		AddSlider(sliderVBox, "Density", ref densityScaleLabel, ref densityScaleSlider,
			density_scale, 0.0f, 2.0f, 0.1f, (v) => { density_scale = v; densityScaleLabel.Text = $"Density: {v:F1}"; });

		AddSlider(sliderVBox, "FieldScale", ref fieldScaleLabel, ref fieldScaleSlider,
			field_scale, 0.001f, 0.05f, 0.001f, (v) => { field_scale = v; fieldScaleLabel.Text = $"FieldScale: {v:F3}"; });
	}

	private void AddSlider(VBoxContainer parent, string name, ref Label label, ref HSlider slider, float value, float min, float max, float step, Action<float> onChanged)
	{
		var row = new HBoxContainer();

		label = new Label();
		label.Text = $"{name}: {value}";
		label.CustomMinimumSize = new Vector2(100, 20);
		row.AddChild(label);

		slider = new HSlider();
		slider.MinValue = min;
		slider.MaxValue = max;
		slider.Step = step;
		slider.Value = value;
		slider.SizeFlagsHorizontal = Control.SizeFlags.ExpandFill;
		slider.CustomMinimumSize = new Vector2(120, 18);
		slider.ValueChanged += (double v) => onChanged((float)v);
		row.AddChild(slider);

		parent.AddChild(row);
	}

	public void ResetParticles()
	{
		var viewport_size = GetViewportRect().Size;
		int rows = (int)Mathf.Sqrt(ball_nums_);
		int cols = (ball_nums_ - 1) / rows + 1;
		float spacing = 10f;
		float offset_x = viewport_size.X / 2;
		float offset_y = viewport_size.Y * 0.3f;

		for (int i = 0; i < ball_nums_; i++)
		{
			float x = (i % rows - rows / 2f + 0.5f) * spacing + offset_x;
			float y = (i / rows - cols / 2f + 0.5f) * spacing + offset_y;
			pos_[i] = new Vector2(x, y);
			vel_[i] = Vector2.Zero;
			density_[i] = 0f;
			near_density_[i] = 0f;
			predicted_pos_[i] = pos_[i];
		}
		GD.Print("Particles reset");
	}

	// 更新核函数缩放因子（参考Fluid-Sim）
	private void UpdateKernelScales()
	{
		float r = smoothing_radius;
		poly6_scale = 4f / (Mathf.Pi * Mathf.Pow(r, 8));
		spiky_pow2_scale = 6f / (Mathf.Pi * Mathf.Pow(r, 4));
		spiky_pow3_scale = 10f / (Mathf.Pi * Mathf.Pow(r, 5));
		spiky_pow2_deriv_scale = 12f / (Mathf.Pi * Mathf.Pow(r, 4));
		spiky_pow3_deriv_scale = 30f / (Mathf.Pi * Mathf.Pow(r, 5));
	}

	// Poly6 核函数（用于粘度）
	private float Poly6Kernel(float dst)
	{
		if (dst >= smoothing_radius) return 0f;
		float v = smoothing_radius * smoothing_radius - dst * dst;
		return v * v * v * poly6_scale;
	}

	// Spiky Pow2 核函数（用于密度）
	private float SpikyPow2(float dst)
	{
		if (dst >= smoothing_radius) return 0f;
		float v = smoothing_radius - dst;
		return v * v * spiky_pow2_scale;
	}

	// Spiky Pow3 核函数（用于近密度）
	private float SpikyPow3(float dst)
	{
		if (dst >= smoothing_radius) return 0f;
		float v = smoothing_radius - dst;
		return v * v * v * spiky_pow3_scale;
	}

	// Spiky Pow2 导数（用于压力力）
	private float SpikyPow2Derivative(float dst)
	{
		if (dst >= smoothing_radius) return 0f;
		float v = smoothing_radius - dst;
		return -v * spiky_pow2_deriv_scale;
	}

	// Spiky Pow3 导数（用于近压力力）
	private float SpikyPow3Derivative(float dst)
	{
		if (dst >= smoothing_radius) return 0f;
		float v = smoothing_radius - dst;
		return -v * v * spiky_pow3_deriv_scale;
	}

	// 压力计算
	private float PressureFromDensity(float density)
	{
		return (density - target_density) * pressure_multiplier;
	}

	// 近压力计算（始终正）
	private float NearPressureFromDensity(float nearDensity)
	{
		return near_pressure_multiplier * nearDensity;
	}

	// 密度计算（参考Fluid-Sim）
	private void CalculateDensity(int index)
	{
		Vector2 pos = predicted_pos_[index];
		float density = 0;
		float nearDensity = 0;

		var neighbors = neighbor_cache_[index];
		int count = neighbor_count_[index];

		for (int i = 0; i < count; i++)
		{
			int j = neighbors[i];
			Vector2 offset = predicted_pos_[j] - pos;
			float dst = offset.Length();
			if (dst < smoothing_radius)
			{
				density += SpikyPow2(dst);
				nearDensity += SpikyPow3(dst);
			}
		}

		density_[index] = density;
		near_density_[index] = nearDensity;
	}

	// 压力力计算（参考Fluid-Sim）
	private void CalculatePressureForce(int index)
	{
		float density = density_[index];
		float nearDensity = near_density_[index];
		float pressure = PressureFromDensity(density);
		float nearPressure = NearPressureFromDensity(nearDensity);
		Vector2 pressureForce = Vector2.Zero;

		Vector2 pos = predicted_pos_[index];
		var neighbors = neighbor_cache_[index];
		int count = neighbor_count_[index];

		for (int i = 0; i < count; i++)
		{
			int j = neighbors[i];
			if (j == index) continue;

			Vector2 offset = predicted_pos_[j] - pos;
			float dst = offset.Length();
			if (dst >= smoothing_radius || dst < 0.0001f) continue;

			Vector2 dir = offset / dst;

			float neighbourDensity = density_[j];
			float neighbourNearDensity = near_density_[j];
			float neighbourPressure = PressureFromDensity(neighbourDensity);
			float neighbourNearPressure = NearPressureFromDensity(neighbourNearDensity);

			float sharedPressure = (pressure + neighbourPressure) * 0.5f;
			float sharedNearPressure = (nearPressure + neighbourNearPressure) * 0.5f;

			pressureForce += dir * SpikyPow2Derivative(dst) * sharedPressure / neighbourDensity;
			pressureForce += dir * SpikyPow3Derivative(dst) * sharedNearPressure / neighbourNearDensity;
		}

		Vector2 acceleration = pressureForce / density;
		vel_[index] += acceleration * current_sub_dt_;
	}

	// 粘度力计算（参考Fluid-Sim）
	private void CalculateViscosity(int index)
	{
		Vector2 pos = predicted_pos_[index];
		Vector2 velocity = vel_[index];
		Vector2 viscosityForce = Vector2.Zero;

		var neighbors = neighbor_cache_[index];
		int count = neighbor_count_[index];

		for (int i = 0; i < count; i++)
		{
			int j = neighbors[i];
			if (j == index) continue;

			Vector2 offset = predicted_pos_[j] - pos;
			float dst = offset.Length();
			if (dst >= smoothing_radius) continue;

			Vector2 neighbourVelocity = vel_[j];
			viscosityForce += (neighbourVelocity - velocity) * Poly6Kernel(dst);
		}

		vel_[index] += viscosityForce * viscosity_strength * current_sub_dt_;
	}

	// 外力（重力+预测位置）
	private void ApplyExternalForces(float dt)
	{
		for (int i = 0; i < ball_nums_; i++)
		{
			// 重力
			vel_[i].Y += gravity * dt;

			// 速度阻尼（消耗能量，让浪涌消退）
			vel_[i] *= velocity_damping;

			// 预测位置（用于密度和力计算）
			float predictionFactor = 1f / 60f;
			predicted_pos_[i] = pos_[i] + vel_[i] * predictionFactor;
		}
	}

	// 积分步（参考Fluid-Sim）
	private void Integrate(float dt)
	{
		for (int i = 0; i < ball_nums_; i++)
		{
			// 更新位置
			pos_[i] += vel_[i] * dt;

			// 边界碰撞
			if (pos_[i].X < bounds_min_.X)
			{
				pos_[i].X = bounds_min_.X;
				vel_[i].X *= -collision_damping;
			}
			else if (pos_[i].X > bounds_max_.X)
			{
				pos_[i].X = bounds_max_.X;
				vel_[i].X *= -collision_damping;
			}

			if (pos_[i].Y < bounds_min_.Y)
			{
				pos_[i].Y = bounds_min_.Y;
				vel_[i].Y *= -collision_damping;
			}
			else if (pos_[i].Y > bounds_max_.Y)
			{
				pos_[i].Y = bounds_max_.Y;
				vel_[i].Y *= -collision_damping;
			}
		}
	}

	public override void _Process(double delta)
	{
		// 响应窗口大小变化
		var viewport_size = GetViewportRect().Size;
		bounds_max_ = new Vector2(viewport_size.X - BOUND_MARGIN, viewport_size.Y - 100f);

		// FPS 统计
		fps_time_accum_ += (float)delta;
		fps_frame_count_++;
		if (fps_time_accum_ >= 0.5f)
		{
			current_fps_ = fps_frame_count_ / fps_time_accum_;
			GD.Print($"FPS:{current_fps_:0} Physics:{physics_ms_:F1}ms BuildTex:{build_tex_ms_:F1}ms");
			fps_time_accum_ = 0f;
			fps_frame_count_ = 0;
		}

		// 同步粒子精灵位置（metaball模式下跳过）
		if (colorRect_ == null || !colorRect_.MetaballEnabled)
		{
			for (int i = 0; i < ball_nums_; i++)
			{
				particle_sprites_[i].Position = pos_[i];
			}
		}

		QueueRedraw();

		// 鼠标交互（强力抓取模式）— 暂停时也暂停
		if (mouse_pressed_ && !paused_)
		{
			float r_grab = 120f;        // 抓取范围
			float r_stick = 30f;         // 粘住核心区
			float grab_speed = 1500f;     // 拉向鼠标的速度上限

			for (int i = 0; i < ball_nums_; i++)
			{
				Vector2 diff = mouse_position_ - pos_[i];
				float len = diff.Length();
				if (len < r_grab && len > 0.001f)
				{
					Vector2 dir = diff / len;

					if (len < r_stick)
					{
						// 核心区：直接设置速度跟鼠标，强阻尼稳住
						vel_[i] = dir * grab_speed * (len / r_stick);
						vel_[i] *= 0.3f;  // 强阻尼
					}
					else
					{
						// 外围区：加速拉过来
						float t = 1f - (len - r_stick) / (r_grab - r_stick);
						vel_[i] += dir * grab_speed * t * 0.3f;
						vel_[i] *= 0.95f;  // 中等阻尼
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
		current_sub_dt_ = subDt;

		long t0 = (long)Time.GetTicksUsec();

		for (int iter = 0; iter < iterations_per_frame; iter++)
		{
			// 1. 外力（重力+预测位置）
			ApplyExternalForces(subDt);

			// 2. 空间哈希
			Hashing();

			// 3. 邻居搜索
			Parallel.For(0, ball_nums_, i =>
			{
				neighbor_count_[i] = 0;
				SearchNeighborsToCache(i, ref neighbor_cache_[i], ref neighbor_count_[i]);
			});

			// 4. 密度计算
			Parallel.For(0, ball_nums_, i => CalculateDensity(i));

			// 5. 压力力计算
			Parallel.For(0, ball_nums_, i => CalculatePressureForce(i));

			// 6. 粘度力计算
			if (viscosity_strength > 0)
			{
				Parallel.For(0, ball_nums_, i => CalculateViscosity(i));
			}

			// 7. 更新位置
			Integrate(subDt);
		}

		long t1 = (long)Time.GetTicksUsec();
		physics_ms_ = (t1 - t0) / 1000f;

		// 通知 ColorRect 重建 tile 数据
		if (colorRect_ != null)
			colorRect_.RequestBuild();
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
			if (key.Keycode == Key.Space)
			{
				ResetParticles();
			}
			else if (key.Keycode == Key.P)
			{
				paused_ = !paused_;
				GD.Print(paused_ ? "Paused" : "Resumed");
			}
		}
	}

	// 切换粒子精灵可见性
	public void SetParticleSpritesVisible(bool visible)
	{
		for (int i = 0; i < ball_nums_; i++)
		{
			particle_sprites_[i].Visible = visible;
		}
	}

	public override void _Draw()
	{
		var font = ThemeDB.FallbackFont;

		// 边界线（动态适配窗口）
		var vs = GetViewportRect().Size;
		var line_color = new Godot.Color(0.2f, 0.5f, 1.0f, 0.8f);
		float line_w = 2f;
		DrawLine(new Vector2(0, 0), new Vector2(vs.X, 0), line_color, line_w);
		DrawLine(new Vector2(vs.X, 0), new Vector2(vs.X, vs.Y), line_color, line_w);
		DrawLine(new Vector2(0, vs.Y), new Vector2(vs.X, vs.Y), line_color, line_w);
		DrawLine(new Vector2(0, 0), new Vector2(0, vs.Y), line_color, line_w);

		// FPS显示（右上角）
		var fps_text = $"FPS: {current_fps_:0}";
		var fps_color = current_fps_ >= 60 ? Colors.Green : current_fps_ >= 30 ? Colors.Yellow : Colors.Red;
		DrawString(font, new Vector2(vs.X - 120, 24), fps_text, fontSize: 18, modulate: fps_color);

		// 暂停状态（右上角FPS下方）
		if (paused_)
		{
			DrawString(font, new Vector2(vs.X - 80, 46), "PAUSED", fontSize: 16, modulate: Colors.Yellow);
		}

		// 鼠标交互圈
		DrawArc(mouse_position_, 70f, 0, Mathf.Tau, 32, new Godot.Color(1, 0, 0, 0.5f), 2f, false);
	}
}
