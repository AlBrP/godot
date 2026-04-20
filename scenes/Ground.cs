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
	[Export] public float field_scale { get; set; } = 0.018f;

	// 卡通水参数（已简化）
	[Export] public float edge_sharpness { get; set; } = 0.78f;  // 越高边缘越窄锐利
	[Export] public float spec_strength { get; set; } = 0.5f;  // 速度高光强度，0=关闭
	[Export] public float toon_levels { get; set; } = 0f;     // Toon量化色阶数，0=连续渐变

	// 运行时滑块面板
	private bool show_sliders_ = false;
	private Panel slider_panel_;
	private HSlider toon_slider_;
	private Label toon_label_;
	private HSlider spec_slider_;
	private Label spec_label_;

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
	private int spawn_index_ = 0;       // 下一个待喷出粒子的索引
	private bool spray_mode_ = true;    // 喷水模式：按住鼠标喷出粒子
	private int last_synced_spawn_ = 0;     // 上次同步到GPU的spawn位置（累计上传用）

	// FPS 统计
	private float fps_time_accum_ = 0f;
	private int fps_frame_count_ = 0;
	private float current_fps_ = 0f;

	// metaball 渲染引用
	private Color_Rect colorRect_;

	// 当前子步 dt
	private float current_sub_dt_ = 0f;

	// 暂停
	private bool paused_ = false;
	public bool Paused => paused_;

	// GPU Compute模式
	private bool gpu_mode_ = false;
	public bool GpuMode => gpu_mode_;
	private SphGpu sph_gpu_;


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

		// 初始化GPU compute
		sph_gpu_ = new SphGpu();
		sph_gpu_.Init();
		colorRect_.SetGpuTextures(sph_gpu_.PositionTex, sph_gpu_.HashLookupTex);

		// 默认开启metaball渲染，隐藏粒子精灵
		SetParticleSpritesVisible(false);

		// 默认开启GPU模式
		gpu_mode_ = true;
		colorRect_.SetGpuMode(true);
		sph_gpu_.ResetParticles(pos_, vel_, Position);
		{
			float warmDt = 1f / 60f;
			float warmSubDt = warmDt / iterations_per_frame;
			DispatchGpu(warmDt, warmSubDt);
		}

		// 默认开启Toon
		toon_levels = 4f;

		// 创建运行时滑块面板
		CreateSliderPanel();
	}

	private void CreateSliderPanel()
	{
		slider_panel_ = new Panel();
		slider_panel_.Position = new Vector2(10, 10);
		slider_panel_.Size = new Vector2(220, 100);
		slider_panel_.Visible = false;
		slider_panel_.Modulate = new Color(1, 1, 1, 0.85f);

		// Toon Levels
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

		// Spec Strength
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
			if (spray_mode_)
			{
				// 喷水模式：全部粒子放到屏幕外，等待鼠标喷出
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
			density_[i] = 0f;
			near_density_[i] = 0f;
			predicted_pos_[i] = pos_[i];
		}
		spawn_index_ = 0;
		last_synced_spawn_ = 0;
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
		// 非活跃粒子跳过
		if (predicted_pos_[index].Y < -500f)
		{
			density_[index] = 0f;
			near_density_[index] = 0f;
			return;
		}

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
		// 非活跃粒子跳过
		if (predicted_pos_[index].Y < -500f) return;

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
		// 非活跃粒子跳过
		if (predicted_pos_[index].Y < -500f) return;

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
			// 跳过非活跃粒子
			if (pos_[i].Y < -500f) continue;

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
			// 跳过非活跃粒子（不clamp到边界）
			if (pos_[i].Y < -500f) continue;

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

		// 更新滑块标签
		if (slider_panel_ != null && slider_panel_.Visible)
		{
			toon_label_.Text = $"Toon Levels: {toon_levels:0}";
			spec_label_.Text = $"Spec: {spec_strength:0.00}";
		}

		// 鼠标交互
		if (!paused_)
		{
			if (spray_mode_ && mouse_pressed_)
			{
				// 喷水模式：稀疏+高速+大间距 → 形成连贯水柱，不被SPH压力炸散
				int spawn_count = 8;

				for (int s = 0; s < spawn_count && spawn_index_ < ball_nums_; s++, spawn_index_++)
				{
					// 极窄 ±1° 扇形（水柱集中）
					float angle = (float)GD.RandRange(-Mathf.Pi * 0.252, -Mathf.Pi * 0.238);
					// 更高速
					float speed = (float)GD.RandRange(2000f, 2100f);
					Vector2 vel = new Vector2(Mathf.Cos(angle) * speed, Mathf.Sin(angle) * speed);
					vel_[spawn_index_] = vel;

					// 沿速度方向错开，间距更大让粒子不重叠
					float along_dist = (float)GD.RandRange(50f, 110f);
					Vector2 dir_norm = vel.Normalized();
					pos_[spawn_index_] = mouse_position_ + dir_norm * along_dist;
					predicted_pos_[spawn_index_] = pos_[spawn_index_];
				}
			}
			else if (!spray_mode_ && mouse_pressed_)
			{
				// 抓取模式
				float r_grab = 120f;
				float r_stick = 30f;
				float grab_speed = 1500f;

				for (int i = 0; i < ball_nums_; i++)
				{
					Vector2 diff = mouse_position_ - pos_[i];
					float len = diff.Length();
					if (len < r_grab && len > 0.001f)
					{
						Vector2 dir = diff / len;

						if (len < r_stick)
						{
							vel_[i] = dir * grab_speed * (len / r_stick);
							vel_[i] *= 0.3f;
						}
						else
						{
							float t = 1f - (len - r_stick) / (r_grab - r_stick);
							vel_[i] += dir * grab_speed * t * 0.3f;
							vel_[i] *= 0.95f;
						}
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

		if (gpu_mode_)
		{
			// 喷水模式下累计同步所有未上传的spawn粒子到GPU
			// （_Process可能每帧跑多次，必须累计，不能只传最后一批）
			if (spray_mode_ && spawn_index_ > last_synced_spawn_)
			{
				int count = spawn_index_ - last_synced_spawn_;
				if (count > 0)
					sph_gpu_.UpdateParticlesBatch(last_synced_spawn_, pos_, vel_, count);
				last_synced_spawn_ = spawn_index_;
			}
			DispatchGpu(frameDt, subDt);
		}
		else
		{
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

			// 通知 ColorRect 重建数据
			if (colorRect_ != null)
				colorRect_.RequestBuild();
		}
	}

	private void DispatchGpu(float frameDt, float subDt)
	{
		sph_gpu_.UpdateParams(
			frameDt, subDt, gravity, velocity_damping,
			smoothing_radius, pressure_multiplier, near_pressure_multiplier,
			viscosity_strength, collision_damping,
			1f / 60f, 2000f,    // predictionFactor, maxVel
			bounds_min_, bounds_max_,
			mouse_pressed_, mouse_position_,
			120f, 30f, 1500f,   // mouseRadiusGrab, mouseRadiusStick, grabSpeed
			Position,
			spray_mode_         // sprayMode
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
			if (key.Keycode == Key.Space)
			{
				paused_ = !paused_;
			}
			if (key.Keycode == Key.G)
			{
				ToggleGpuMode();
			}
			if (key.Keycode == Key.H)
			{
				spec_strength = spec_strength > 0.01f ? 0f : 0.5f;
			}
			if (key.Keycode == Key.T)
			{
				toon_levels = toon_levels < 0.5f ? 4f : 0f;
			}
			if (key.Keycode == Key.V)
			{
				show_sliders_ = !show_sliders_;
				if (slider_panel_ != null) slider_panel_.Visible = show_sliders_;
			}
			if (key.Keycode == Key.B)
			{
				spray_mode_ = !spray_mode_;
				ResetParticles();
				if (gpu_mode_)
				{
					sph_gpu_.ResetParticles(pos_, vel_, Position);
				}
			}
		}
	}

	private void ToggleGpuMode()
	{
		if (gpu_mode_)
		{
			// GPU → CPU: 读回粒子位置和速度
			byte[] data = sph_gpu_.ReadBackParticleBuffer();
			for (int i = 0; i < ball_nums_; i++)
			{
				int off = i * 8;
				pos_[i] = new Vector2(
					BitConverter.ToSingle(data, off),
					BitConverter.ToSingle(data, off + 4)
				);
				int voff = ball_nums_ * 8 + i * 8;
				vel_[i] = new Vector2(
					BitConverter.ToSingle(data, voff),
					BitConverter.ToSingle(data, voff + 4)
				);
			}
			gpu_mode_ = false;
			colorRect_.SetGpuMode(false);
		}
		else
		{
			// CPU → GPU: 上传粒子位置和速度
			sph_gpu_.ResetParticles(pos_, vel_, Position);
			gpu_mode_ = true;
			colorRect_.SetGpuMode(true);

			// Warm up: 立即dispatch一帧，填充输出纹理，避免首帧空白
			float warmDt = 1f / 60f;
			float warmSubDt = warmDt / iterations_per_frame;
			DispatchGpu(warmDt, warmSubDt);
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
		var vs = GetViewportRect().Size;

		// 状态显示（右上角，右对齐）
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

		if (gpu_mode_)
		{
			DrawString(font, new Vector2(right_x - font.GetStringSize("GPU MODE").X, y), "GPU MODE", fontSize: 14, modulate: Colors.Cyan);
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
	}
}
