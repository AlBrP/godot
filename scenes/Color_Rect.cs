using Godot;
using System;
using System.ComponentModel;

public partial class Color_Rect : ColorRect
{
	// Called when the node enters the scene tree for the first time.
	Ground ground = new Ground();
	Vector2[] ball_position_uv = new Vector2[Ground.ball_nums_];
	Vector2 ground_position;
	float viewport_x_scale;
	float viewport_y_scale;
	public override void _Ready()
	{
		ground = GetParent().GetChild(1) as Ground;
		ground_position = (GetTree().CurrentScene.GetChild(1) as StaticBody2D).Position;
		viewport_x_scale = 1/Size.X;
		viewport_y_scale = 1/Size.Y;
	}
	
	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		for (int i = 0; i < Ground.ball_nums_; i++)
		{
			ball_position_uv[i] = ground.ball_position_[i] + ground_position;
			ball_position_uv[i][0] *= viewport_x_scale;
			ball_position_uv[i][1] *= viewport_y_scale;
			// ground.ball_position_[i]
			//1. 把所有ball的position转到左上角的坐标系
			//2. ball_position转uv
			//3. set_shader_params 把ball_position set下去
			//4. shader接收，在fragment里面用UV和ball position做为距离公式的变量，改变COLOR
		}
		((ShaderMaterial)Material).SetShaderParameter("ball_position_uv", ball_position_uv);
	}
}
