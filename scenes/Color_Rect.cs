using Godot;
using System;

public partial class Color_Rect : ColorRect
{
	Ground ground_;

	// 位置纹理：72x70 = 5040 像素（按hash排序顺序写入）
	private Image raw_position_image_;
	private ImageTexture raw_position_texture_;
	private const int TEXTURE_WIDTH = 72;
	private const int TEXTURE_HEIGHT = 70;

	// Hash lookup 纹理：RG32F，每个像素 R=start, G=end
	private Image hash_lookup_image_;
	private ImageTexture hash_lookup_texture_;

	// 速度归一化参数
	private const float MAX_VEL = 2000f;

	// 渲染开关
	private bool metaball_enabled_ = false;
	public bool MetaballEnabled => metaball_enabled_;

	// Debug 模式：0=off, 1=heatmap+grid, 2=brute force
	private int debug_mode_ = 0;

	// 粒子包围盒
	private Vector2 bbox_min_ = Vector2.Zero;
	private Vector2 bbox_max_ = Vector2.Zero;

	// 批量数据数组
	private byte[] _rawBytes;
	private byte[] _hashBytes; // RG32F: 8 bytes per pixel

	public override void _Ready()
	{
		ground_ = GetNode<Ground>("/root/main/Ground");

		MouseFilter = MouseFilterEnum.Ignore;

		// SubViewportContainer 也不能拦截鼠标
		var container = GetParent().GetParent() as SubViewportContainer;
		if (container != null)
			container.MouseFilter = MouseFilterEnum.Ignore;

		raw_position_image_ = Image.CreateEmpty(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.Format.Rgbah);
		raw_position_texture_ = ImageTexture.CreateFromImage(raw_position_image_);

		hash_lookup_image_ = Image.CreateEmpty(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.Format.Rgf);
		hash_lookup_texture_ = ImageTexture.CreateFromImage(hash_lookup_image_);

		if (Material is ShaderMaterial shaderMaterial)
		{
			shaderMaterial.SetShaderParameter("raw_position_texture", raw_position_texture_);
			shaderMaterial.SetShaderParameter("hash_lookup_texture", hash_lookup_texture_);
			shaderMaterial.SetShaderParameter("particle_count", Ground.ball_nums_);
			shaderMaterial.SetShaderParameter("texture_size", new Vector2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
			shaderMaterial.SetShaderParameter("max_vel", (double)MAX_VEL);
		}

		_rawBytes = new byte[TEXTURE_WIDTH * TEXTURE_HEIGHT * 8]; // RGBA16F: 8 bytes per pixel
		_hashBytes = new byte[TEXTURE_WIDTH * TEXTURE_HEIGHT * 8]; // RG32F: 8 bytes per pixel

		GD.Print($"ColorRect Ready: size={Size.X}x{Size.Y}");
	}

	private bool _needsBuild = false;

	public void RequestBuild()
	{
		_needsBuild = true;
	}

	public override void _Process(double delta)
	{
		if (Material is not ShaderMaterial shaderMaterial) return;

		if (metaball_enabled_ && _needsBuild)
		{
			long t0 = (long)Time.GetTicksUsec();
			BuildRawPositionTexture();
			_needsBuild = false;
			ground_.build_tex_ms_ = ((long)Time.GetTicksUsec() - t0) / 1000f;
		}

		shaderMaterial.SetShaderParameter("render_enabled", metaball_enabled_);
		shaderMaterial.SetShaderParameter("debug_mode", debug_mode_);

		shaderMaterial.SetShaderParameter("field_strength", (double)ground_.field_strength);
		shaderMaterial.SetShaderParameter("body_threshold", (double)ground_.body_threshold);
		shaderMaterial.SetShaderParameter("edge_threshold", (double)ground_.edge_threshold);
		shaderMaterial.SetShaderParameter("sigma", (double)ground_.sigma);
		shaderMaterial.SetShaderParameter("stretch_scale", (double)ground_.stretch_scale);
		shaderMaterial.SetShaderParameter("density_scale", (double)ground_.density_scale);
		shaderMaterial.SetShaderParameter("edge_sharpness", (double)ground_.edge_sharpness);
		shaderMaterial.SetShaderParameter("field_scale", (double)ground_.field_scale);
		shaderMaterial.SetShaderParameter("time", Time.GetTicksMsec() / 1000.0f);

		// viewport_size: shader用它把屏幕UV转成世界坐标
		var vpSize = ground_.GetViewportRect().Size;
		shaderMaterial.SetShaderParameter("viewport_size", vpSize);
		shaderMaterial.SetShaderParameter("bbox_min", bbox_min_);
		shaderMaterial.SetShaderParameter("bbox_max", bbox_max_);

		// 空间哈希参数
		shaderMaterial.SetShaderParameter("grid_cell_size", (double)ground_.grid_cell_size);
		shaderMaterial.SetShaderParameter("ground_offset", ground_.Position);
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventKey key && key.Pressed && !key.Echo)
		{
			if (key.Keycode == Key.R)
			{
				metaball_enabled_ = !metaball_enabled_;
				ground_.SetParticleSpritesVisible(!metaball_enabled_);
				if (metaball_enabled_)
					BuildRawPositionTexture();
				GD.Print($"Render: {(metaball_enabled_ ? "Metaball" : "Particles")}");
			}
			else if (key.Keycode == Key.D && metaball_enabled_)
			{
				debug_mode_ = (debug_mode_ + 1) % 3;
				string[] names = { "OFF", "Heatmap+Grid", "BruteForce" };
				GD.Print($"Debug: {names[debug_mode_]}");
			}
		}
	}

	// ===== 将 float 编码为 RGBA16F (half-float per channel) =====
	private static ushort HalfFloat(float f)
	{
		uint bits = (uint)BitConverter.SingleToInt32Bits(f);
		uint sign = (bits >> 31) & 0x0001;
		uint exp = (bits >> 23) & 0x00FF;
		uint frac = bits & 0x007FFFFF;

		if (exp == 0) // zero or subnormal
		{
			return (ushort)(sign << 15);
		}
		if (exp == 0xFF) // Inf or NaN
		{
			return (ushort)((sign << 15) | 0x7C00 | ((frac != 0) ? 1u : 0u));
		}
		int newExp = (int)exp - 127;
		newExp += 15;
		return (ushort)((sign << 15) | ((uint)newExp << 10) | (frac >> 13));
	}

	private static void WriteRgba16F(byte[] buf, int pixelIdx, float r, float g, float b, float a)
	{
		int i = pixelIdx * 8;
		ushort hr = HalfFloat(r);
		ushort hg = HalfFloat(g);
		ushort hb = HalfFloat(b);
		ushort ha = HalfFloat(a);
		buf[i]     = (byte)(hr & 0xFF); buf[i + 1] = (byte)(hr >> 8);
		buf[i + 2] = (byte)(hg & 0xFF); buf[i + 3] = (byte)(hg >> 8);
		buf[i + 4] = (byte)(hb & 0xFF); buf[i + 5] = (byte)(hb >> 8);
		buf[i + 6] = (byte)(ha & 0xFF); buf[i + 7] = (byte)(ha >> 8);
	}

	// 写 RG32F（两个 float32）
	private static void WriteRg32F(byte[] buf, int pixelIdx, float r, float g)
	{
		int i = pixelIdx * 8;
		byte[] rb = BitConverter.GetBytes(r);
		byte[] gb = BitConverter.GetBytes(g);
		buf[i] = rb[0]; buf[i + 1] = rb[1]; buf[i + 2] = rb[2]; buf[i + 3] = rb[3];
		buf[i + 4] = gb[0]; buf[i + 5] = gb[1]; buf[i + 6] = gb[2]; buf[i + 7] = gb[3];
	}

	// 按hash排序顺序写入位置纹理 + 构建hash lookup纹理
	private void BuildRawPositionTexture()
	{
		int totalPixels = TEXTURE_WIDTH * TEXTURE_HEIGHT;
		int n = Ground.ball_nums_;

		// 计算粒子包围盒
		float minX = float.MaxValue, minY = float.MaxValue;
		float maxX = float.MinValue, maxY = float.MinValue;

		// 按 sorted_indices 顺序写入位置纹理
		var sorted = ground_.sorted_indices;
		for (int i = 0; i < n; i++)
		{
			int pidx = sorted[i];
			Vector2 worldPos = ground_.pos_[pidx] + ground_.Position;
			Vector2 vel = ground_.vel_[pidx];
			float vn_x = Mathf.Clamp(vel.X / MAX_VEL * 0.5f + 0.5f, 0f, 1f);
			float vn_y = Mathf.Clamp(vel.Y / MAX_VEL * 0.5f + 0.5f, 0f, 1f);
			WriteRgba16F(_rawBytes, i, worldPos.X, worldPos.Y, vn_x, vn_y);

			if (worldPos.X < minX) minX = worldPos.X;
			if (worldPos.Y < minY) minY = worldPos.Y;
			if (worldPos.X > maxX) maxX = worldPos.X;
			if (worldPos.Y > maxY) maxY = worldPos.Y;
		}

		bbox_min_ = new Vector2(minX, minY);
		bbox_max_ = new Vector2(maxX, maxY);

		// 填充剩余像素
		for (int i = n; i < totalPixels; i++)
		{
			WriteRgba16F(_rawBytes, i, 0f, 0f, 0f, 0f);
		}

		raw_position_image_.SetData(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.Format.Rgbah, _rawBytes);
		raw_position_texture_.Update(raw_position_image_);

		// 构建 hash lookup 纹理
		// 每个 hash bucket: R=start index, G=end index
		// 无效 bucket: R=-1
		var hashSt = ground_.hash_values_st;
		var hashEd = ground_.hash_values_ed;
		int bigNum = ground_.big_num;

		for (int h = 0; h < n; h++)
		{
			if (hashSt[h] >= bigNum)
			{
				WriteRg32F(_hashBytes, h, -1f, -1f);
			}
			else
			{
				WriteRg32F(_hashBytes, h, (float)hashSt[h], (float)hashEd[h]);
			}
		}
		// 填充剩余像素（hash值0..4999, 剩余40个像素）
		for (int h = n; h < totalPixels; h++)
		{
			WriteRg32F(_hashBytes, h, -1f, -1f);
		}

		hash_lookup_image_.SetData(TEXTURE_WIDTH, TEXTURE_HEIGHT, false, Image.Format.Rgf, _hashBytes);
		hash_lookup_texture_.Update(hash_lookup_image_);
	}
}
