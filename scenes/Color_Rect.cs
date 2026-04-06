using Godot;
using System;

public partial class Color_Rect : ColorRect
{
	Ground ground_;
	Vector2[] ball_position_uv = new Vector2[5000];
	Vector2 ground_position_;
	float inv_size_x_;
	float inv_size_y_;

	// 渲染开关（true = Metaball, false = 粒子精灵）
	private bool metaball_enabled_ = false;

	public bool MetaballEnabled => metaball_enabled_;

	public override void _Ready()
	{
		ground_ = GetParent().GetChild(1) as Ground;
		ground_position_ = (GetTree().CurrentScene.GetChild(1) as StaticBody2D).Position;
		inv_size_x_ = 1f / Size.X;
		inv_size_y_ = 1f / Size.Y;
	}

	public override void _Input(InputEvent @event)
	{
		// 按R键切换渲染模式
		if (@event is InputEventKey key && key.Pressed && !key.Echo && key.Keycode == Key.R)
		{
			metaball_enabled_ = !metaball_enabled_;

			// 切换粒子精灵可见性
			ground_.SetParticleSpritesVisible(!metaball_enabled_);

			GD.Print($"Render: {(metaball_enabled_ ? "Metaball" : "Particles")}");
		}
	}

	public override void _Process(double delta)
	{
		// 更新shader参数
		((ShaderMaterial)Material).SetShaderParameter("render_enabled", metaball_enabled_);

		// 只在Metaball模式下更新位置数据
		if (metaball_enabled_)
		{
			for (int i = 0; i < Ground.ball_nums_; i++)
			{
				ball_position_uv[i] = ground_.pos_[i] + ground_position_;
				ball_position_uv[i].X *= inv_size_x_;
				ball_position_uv[i].Y *= inv_size_y_;
			}
			((ShaderMaterial)Material).SetShaderParameter("ball_position_uv", ball_position_uv);
		}
	}
}
