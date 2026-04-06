using Godot;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class Ground : StaticBody2D
{
	// 物理参数（参考Fluid-Sim）
	[Export] public float smoothing_radius { get; set; } = 15f;
	[Export] public float target_density { get; set; } = 0.02f;
	[Export] public float pressure_multiplier { get; set; } = 50000f;
	[Export] public float near_pressure_multiplier { get; set; } = 100000f;
	[Export] public float viscosity_strength { get; set; } = 30f;
	[Export] public float gravity { get; set; } = 300f;
	[Export] public float collision_damping { get; set; } = 0.9f;

	// UI滑块
	private HSlider densitySlider;
	private HSlider viscositySlider;
	private Label densityLabel;
	private Label viscosityLabel;

	// 粒子数量
	public const int ball_nums_ = 5000;

	// 数据数组
	public Vector2[] pos_ = new Vector2[ball_nums_];
	private Vector2[] predicted_pos_ = new Vector2[ball_nums_];  // 预测位置
	private Vector2[] vel_ = new Vector2[ball_nums_];
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
	private bool mouse_pressed_ = false;
	private Vector2 mouse_position_ = Vector2.Zero;

	// FPS 统计
	private float fps_time_accum_ = 0f;
	private int fps_frame_count_ = 0;
	private float current_fps_ = 0f;


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

		// 初始化粒子位置（网格排列，密集放置模拟流体）
		int rows = (int)Mathf.Sqrt(ball_nums_);
		int cols = (ball_nums_ - 1) / rows + 1;
		float spacing = 10f;  // 增大间距，减少初始密度
		float offset_x = viewport_size.X / 2;
		float offset_y = viewport_size.Y * 0.3f;

		for (int i = 0; i < ball_nums_; i++)
		{
			float x = (i % rows - rows / 2f + 0.5f) * spacing + offset_x;
			float y = (i / rows - cols / 2f + 0.5f) * spacing + offset_y;
			pos_[i] = new Vector2(x, y);
			vel_[i] = Vector2.Zero;
			density_[i] = 0f;
			predicted_pos_[i] = pos_[i];
		}

		HashInit();
		UpdateKernelScales();

		// 创建UI
		CreateParameterUI();
	}

	private void CreateParameterUI()
	{
		// 创建CanvasLayer确保UI在最上层
		var canvas = new CanvasLayer();
		canvas.Layer = 10;
		AddChild(canvas);

		// 创建容器
		var container = new VBoxContainer();
		container.Position = new Vector2(10, 80);
		canvas.AddChild(container);

		// target_density 滑块
		var densityRow = new HBoxContainer();
		densityLabel = new Label();
		densityLabel.Text = $"Density: {target_density:F1}";
		densityLabel.CustomMinimumSize = new Vector2(120, 0);
		densityRow.AddChild(densityLabel);

		densitySlider = new HSlider();
		densitySlider.MinValue = 0.05f;
		densitySlider.MaxValue = 0.5f;
		densitySlider.Step = 0.01f;
		densitySlider.Value = target_density;
		densitySlider.CustomMinimumSize = new Vector2(200, 20);
		densitySlider.ValueChanged += OnDensityChanged;
		densityRow.AddChild(densitySlider);
		container.AddChild(densityRow);

		// viscosity 滑块
		var viscosityRow = new HBoxContainer();
		viscosityLabel = new Label();
		viscosityLabel.Text = $"Viscosity: {viscosity_strength:F0}";
		viscosityLabel.CustomMinimumSize = new Vector2(120, 0);
		viscosityRow.AddChild(viscosityLabel);

		viscositySlider = new HSlider();
		viscositySlider.MinValue = 0f;
		viscositySlider.MaxValue = 2000f;
		viscositySlider.Step = 5f;
		viscositySlider.Value = viscosity_strength;
		viscositySlider.CustomMinimumSize = new Vector2(200, 20);
		viscositySlider.ValueChanged += OnViscosityChanged;
		viscosityRow.AddChild(viscositySlider);
		container.AddChild(viscosityRow);
	}

	private void OnDensityChanged(double value)
	{
		target_density = (float)value;
		densityLabel.Text = $"Density: {target_density:F1}";
	}

	private void OnViscosityChanged(double value)
	{
		viscosity_strength = (float)value;
		viscosityLabel.Text = $"Viscosity: {viscosity_strength:F0}";
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
		vel_[index] += acceleration * (float)GetPhysicsProcessDeltaTime();

		// 调试：找密度最高的粒子
		if (density > 0.4f && index == FindMaxDensityIndex())
		{
			GD.Print($"[High Density] idx={index}, density={density:F3}, pressure={pressure:F1}");
			GD.Print($"  pos=({pos.X:F1}, {pos.Y:F1})");
			GD.Print($"  pressureForce=({pressureForce.X:F1}, {pressureForce.Y:F1})");
			GD.Print($"  acceleration=({acceleration.X:F1}, {acceleration.Y:F1})");
		}
	}

	private int FindMaxDensityIndex()
	{
		int maxIdx = 0;
		float maxD = 0;
		for (int i = 0; i < ball_nums_; i++)
		{
			if (density_[i] > maxD)
			{
				maxD = density_[i];
				maxIdx = i;
			}
		}
		return maxIdx;
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

		vel_[index] += viscosityForce * viscosity_strength * (float)GetPhysicsProcessDeltaTime();
	}

	// 外力（重力+预测位置）
	private void ApplyExternalForces(float dt)
	{
		for (int i = 0; i < ball_nums_; i++)
		{
			// 重力
			vel_[i].Y += gravity * dt;

			// 预测位置（用于密度和力计算）
			float predictionFactor = 1f / 120f;
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
		// FPS 统计
		fps_time_accum_ += (float)delta;
		fps_frame_count_++;
		if (fps_time_accum_ >= 0.5f)
		{
			current_fps_ = fps_frame_count_ / fps_time_accum_;

			// 按高度分层统计密度
			float yMin = float.MaxValue, yMax = float.MinValue;
			for (int i = 0; i < ball_nums_; i++)
			{
				if (pos_[i].Y < yMin) yMin = pos_[i].Y;
				if (pos_[i].Y > yMax) yMax = pos_[i].Y;
			}
			float yRange = yMax - yMin;
			if (yRange < 1f) yRange = 1f;
			int numLayers = 5;
			float layerHeight = yRange / numLayers;
			float[] layerDensity = new float[numLayers];
			int[] layerCount = new int[numLayers];

			for (int i = 0; i < ball_nums_; i++)
			{
				int layer = Math.Clamp((int)((pos_[i].Y - yMin) / layerHeight), 0, numLayers - 1);
				layerDensity[layer] += density_[i];
				layerCount[layer]++;
			}

			GD.Print($"FPS: {current_fps_:0}, Y: {yMin:F0}~{yMax:F0}");
			for (int l = 0; l < numLayers; l++)
			{
				float avgD = layerCount[l] > 0 ? layerDensity[l] / layerCount[l] : 0;
				GD.Print($"  Layer{l}: avgDensity={avgD:F4}, count={layerCount[l]}");
			}

			fps_time_accum_ = 0f;
			fps_frame_count_ = 0;
		}

		// 同步粒子精灵位置
		for (int i = 0; i < ball_nums_; i++)
		{
			particle_sprites_[i].Position = pos_[i];
		}

		QueueRedraw();

		// 鼠标交互
		if (mouse_pressed_)
		{
			float r_interaction = 70f;
			float kp = 500f;

			for (int i = 0; i < ball_nums_; i++)
			{
				Vector2 diff = mouse_position_ - pos_[i];
				float len = diff.Length();
				if (len < r_interaction && len > 0.001f)
				{
					Vector2 dir = diff / len;
					float weight = 1 - len / r_interaction;
					// 直接设置速度，让粒子跟随鼠标
					vel_[i] = vel_[i] * 0.8f + dir * weight * kp;
					vel_[i].Y -= gravity * (float)delta * weight * 0.5f;
				}
			}
		}
	}

	public override void _PhysicsProcess(double delta)
	{
		float dt = (float)delta;

		// 1. 外力（重力+预测位置）
		ApplyExternalForces(dt);

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
		Integrate(dt);
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventMouse mouse)
		{
			mouse_position_ = mouse.Position - Position;
			mouse_pressed_ = Input.IsMouseButtonPressed(MouseButton.Left);
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

		// FPS显示
		var fps_text = $"FPS: {current_fps_:0}";
		var fps_color = current_fps_ >= 60 ? Colors.Green : current_fps_ >= 30 ? Colors.Yellow : Colors.Red;
		DrawString(font, new Vector2(10, 30), fps_text, fontSize: 20, modulate: fps_color);

		// 渲染状态显示
		var color_rect = GetParent().GetChild(0) as Color_Rect;
		var render_text = color_rect.MetaballEnabled ? "Metaball" : "Particles";
		var render_hint = " [R]";
		DrawString(font, new Vector2(10, 55), $"Render: {render_text}{render_hint}", fontSize: 16, modulate: Colors.Cyan);

		// 鼠标交互圈
		DrawArc(mouse_position_, 70f, 0, Mathf.Tau, 32, new Godot.Color(1, 0, 0, 0.5f), 2f, false);
	}
}
