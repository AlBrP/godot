using Godot;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

public partial class Ground : StaticBody2D
{
	// 物理参数
	[Export] public float smoothing_length { get; set; } = 17.5f;
	[Export] public float pressure_stiffness { get; set; } = 0.001f;
	[Export] public float target_density { get; set; } = 0.007f;
	[Export] public float viscosity_gain { get; set; } = 250f;
	[Export] public int gamma { get; set; } = 7;
	[Export] public float gravity { get; set; } = 980f;
	[Export] public float mass { get; set; } = 1f;

	// 粒子数量
	public const int ball_nums_ = 2000;

	// 纯数据数组（方案2）
	public Vector2[] pos_ = new Vector2[ball_nums_];
	private Vector2[] vel_ = new Vector2[ball_nums_];
	private Vector2[] force_acc_ = new Vector2[ball_nums_];
	private float[] density_ = new float[ball_nums_];
	private Vector2[] pressure_force_ = new Vector2[ball_nums_];

	// 边界
	private Vector2 bounds_min_;
	private Vector2 bounds_max_;
	private const float BOUND_MARGIN = 5f;
	private const float DAMPING = 0.998f;
	private const float MAX_VELOCITY = 300f;

	// 邻居缓存（方案4：数组替代List）
	private int[][] neighbor_cache_;
	private int[] neighbor_count_;
	private const int MAX_NEIGHBORS = 256;

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

	// 物理计时
	private ulong time_integrate = 0;
	private ulong time_hashing = 0;
	private ulong time_neighbor = 0;
	private ulong time_density = 0;
	private ulong time_pressure = 0;
	private int frame_count = 0;

	// 子步数
	private int substeps_ = 4;

	public override void _Ready()
	{
		// 初始化边界
		var viewport_size = GetViewportRect().Size;
		bounds_min_ = new Vector2(BOUND_MARGIN, BOUND_MARGIN);
		bounds_max_ = viewport_size - new Vector2(BOUND_MARGIN, BOUND_MARGIN);

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
			particle_sprites_[i].Scale = new Vector2(0.04f, 0.04f);
			particle_sprites_[i].Visible = false;
			AddChild(particle_sprites_[i]);
		}

		// 初始化粒子位置（网格排列）
		int rows = (int)Mathf.Sqrt(ball_nums_);
		int cols = (ball_nums_ - 1) / rows + 1;
		float spacing = 15f;
		float offset_x = viewport_size.X / 2;
		float offset_y = viewport_size.Y / 2;

		for (int i = 0; i < ball_nums_; i++)
		{
			float x = (i % rows - rows / 2f + 0.5f) * spacing + offset_x;
			float y = (i / rows - cols / 2f + 0.5f) * spacing + offset_y;
			pos_[i] = new Vector2(x, y);
			vel_[i] = Vector2.Zero;
			density_[i] = 0f;
			force_acc_[i] = Vector2.Zero;
		}

		HashInit();
	}

	// 核函数
	private float SmoothingKernel(float h, float dst)
	{
		float q = dst / h;
		if (q >= 2) return 0;
		float weight;
		if (q < 1)
			weight = 0.341f * ((2 - q) * (2 - q) * (2 - q) - 4 * (1 - q) * (1 - q) * (1 - q));
		else
			weight = 0.341f * (2 - q) * (2 - q) * (2 - q);
		return weight / (h * h);
	}

	private float SmoothingKernelDerivative(float h, float dst)
	{
		float q = dst / h;
		if (q >= 2) return 0;
		float weight;
		if (q < 1)
			weight = 0.341f * (-3 * (2 - q) * (2 - q) + 12 * (1 - q) * (1 - q)) / h;
		else
			weight = -1.023f * (2 - q) * (2 - q) / h;
		return weight / (h * h);
	}

	private float CalculatePressure(float density)
	{
		return pressure_stiffness * (Mathf.Pow(density / target_density, gamma) - 1);
	}

	// 密度计算（纯数组版）
	private void CalculateDensity(int index)
	{
		density_[index] = 0;
		var neighbors = neighbor_cache_[index];
		int count = neighbor_count_[index];
		for (int i = 0; i < count; i++)
		{
			int j = neighbors[i];
			float dst = (pos_[j] - pos_[index]).Length();
			density_[index] += mass * SmoothingKernel(smoothing_length, dst);
		}
	}

	// 压力+粘度力计算（纯数组版）
	private void CalculateForces(int index)
	{
		float p_i = CalculatePressure(density_[index]);
		Vector2 pressure_acc = Vector2.Zero;
		Vector2 viscosity_acc = Vector2.Zero;

		var neighbors = neighbor_cache_[index];
		int count = neighbor_count_[index];
		for (int i = 0; i < count; i++)
		{
			int j = neighbors[i];
			float p_j = CalculatePressure(density_[j]);

			Vector2 dir = pos_[j] - pos_[index];
			float dst = dir.Length();
			if (dst < 0.0001f) continue;

			float kernel_deriv = SmoothingKernelDerivative(smoothing_length, dst);
			Vector2 dir_norm = dir / dst;

			// 压力力
			pressure_acc += mass * (p_i / (density_[index] * density_[index]) + p_j / (density_[j] * density_[j])) * dir_norm * kernel_deriv;

			// 粘度力
			Vector2 v_rel = vel_[j] - vel_[index];
			float weight = SmoothingKernel(smoothing_length, dst);
			viscosity_acc += v_rel * weight * viscosity_gain;
		}

		force_acc_[index] = pressure_acc * mass + viscosity_acc;
	}

	// 积分步（半隐式欧拉）
	private void Integrate(float dt)
	{
		for (int i = 0; i < ball_nums_; i++)
		{
			// 重力
			vel_[i].Y += gravity * dt;

			// 压力+粘度力
			vel_[i] += force_acc_[i] * dt;

			// 速度限制
			float speed = vel_[i].Length();
			if (speed > MAX_VELOCITY)
			{
				vel_[i] = vel_[i] / speed * MAX_VELOCITY;
			}

			// 更新位置
			pos_[i] += vel_[i] * dt;

			// 边界碰撞
			if (pos_[i].X < bounds_min_.X)
			{
				pos_[i].X = bounds_min_.X;
				vel_[i].X = 0;
			}
			else if (pos_[i].X > bounds_max_.X)
			{
				pos_[i].X = bounds_max_.X;
				vel_[i].X = 0;
			}

			if (pos_[i].Y < bounds_min_.Y)
			{
				pos_[i].Y = bounds_min_.Y;
				vel_[i].Y = 0;
			}
			else if (pos_[i].Y > bounds_max_.Y)
			{
				pos_[i].Y = bounds_max_.Y;
				vel_[i].Y = 0;
			}

			// 阻尼
			vel_[i] *= DAMPING;
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
			GD.Print($"FPS: {current_fps_:0}");
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
			float kp = 6000f;

			for (int i = 0; i < ball_nums_; i++)
			{
				Vector2 diff = mouse_position_ - pos_[i];
				float len = diff.Length();
				if (len < r_interaction && len > 0.001f)
				{
					Vector2 dir = diff / len;
					float weight = 1 - len / r_interaction;
					vel_[i] += dir * weight * kp * (float)delta;
					vel_[i].Y -= gravity * (float)delta * weight;
				}
			}
		}
	}

	public override void _PhysicsProcess(double delta)
	{
		ulong t0, t1, t2, t3, t4, t5;
		t0 = Time.GetTicksUsec();

		// 子步积分
		float sub_dt = (float)delta / substeps_;

		for (int step = 0; step < substeps_; step++)
		{
			// 1. 空间哈希
			Hashing();
			t1 = Time.GetTicksUsec();

			// 2. 邻居搜索
			Parallel.For(0, ball_nums_, i =>
			{
				neighbor_count_[i] = 0;
				SearchNeighborsToCache(i, ref neighbor_cache_[i], ref neighbor_count_[i]);
			});
			t2 = Time.GetTicksUsec();

			// 3. 密度计算
			Parallel.For(0, ball_nums_, i => CalculateDensity(i));
			t3 = Time.GetTicksUsec();

			// 4. 力计算
			Parallel.For(0, ball_nums_, i => CalculateForces(i));
			t4 = Time.GetTicksUsec();

			// 5. 积分
			Integrate(sub_dt);
			t5 = Time.GetTicksUsec();

			// 只统计第一次子步的耗时
			if (step == 0)
			{
				time_hashing += t1 - t0;
				time_neighbor += t2 - t1;
				time_density += t3 - t2;
				time_pressure += t4 - t3;
				time_integrate += t5 - t4;
			}
		}

		frame_count++;

		if (frame_count >= 60)
		{
			GD.Print($"=== Physics Timing (avg over {frame_count} frames, {substeps_} substeps) ===");
			GD.Print($"  Hashing:         {time_hashing / (ulong)frame_count} us");
			GD.Print($"  Neighbor Search: {time_neighbor / (ulong)frame_count} us");
			GD.Print($"  Density:         {time_density / (ulong)frame_count} us");
			GD.Print($"  Force Calc:      {time_pressure / (ulong)frame_count} us");
			GD.Print($"  Integrate:       {time_integrate / (ulong)frame_count} us");
			GD.Print($"  Total Physics:   {(time_hashing + time_neighbor + time_density + time_pressure + time_integrate) / (ulong)frame_count} us");

			time_hashing = time_neighbor = time_density = time_pressure = time_integrate = 0;
			frame_count = 0;
		}
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
