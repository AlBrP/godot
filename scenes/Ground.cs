using Godot;
using Godot.NativeInterop;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Reflection.Metadata.Ecma335;
using System.Threading;
using System.Threading.Tasks;
public partial class Ground : StaticBody2D
{
	/*Built-in types.*/
	[Export] public PackedScene ball_scene { get; set; }
	[Export(PropertyHint.Range, "0,100,")] public float smoothing_length { get; set; } = 25;
	[Export] public float pressure_stiffness { get; set; } = 0.01f;
	[Export(PropertyHint.Range, "0.001,0.05,")] public double target_density { get; set; } = 0.005;
	[Export] public float vicosity_gain { get; set; } = 0.5f;

	/*global predefined variables*/
	private bool simulation_start_ = false;
	private List<Ball> ball_array_ = new List<Ball>();
	public const int ball_nums_ = 1200;
	private const int looking_idx_ = 0;
	private ulong random_seed_ = 10;
	public Vector2[] ball_position_ = new Vector2[ball_nums_];
	bool mouse_pressed_ = false;
	Vector2 mouse_position_ = Vector2.Zero;

	//pressure calculation parameters
	private int gamma_ = 7;
	// #real 2d water density is 1000 kg/(m^2)
	// #since 1m = 100 unit
	// #1000 kg/(m^2) = 0.1 kg/(unit^2)
	// #so target density = 0.1
	public void _ButtonPressed()
	{
		simulation_start_ = true;
		GD.Print("simulation started!");
		
	}
	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
		// var wind = GetWindow();
		// wind.Size = new Vector2I(1920, 1080);
		RandomNumberGenerator rng = new RandomNumberGenerator();
		rng.Seed = random_seed_;

		float ball_rand_x_max = GetViewportRect().Size.X;
		float ball_rand_y_max = GetViewportRect().Size.Y;

		// for(int i = 0;i < ball_nums_;i++){
		// 	ball_array_.Add(ball_scene.Instantiate() as Ball);
		// 	AddChild(ball_array_[i]);
		//  	ball_array_[i].Position = new Vector2(
		// 		rng.RandfRange(-0.4f, 0.4f) * ball_rand_x_max,
		// 	    -ball_rand_y_max*0.5f + rng.RandfRange(-0.4f, 0.4f) * ball_rand_y_max);
		// }


		// 计算行列数（假设粒子总数是平方数）
		int rows = (int)Mathf.Sqrt(ball_nums_);
		int cols =( ball_nums_ -1) / rows +1;
		float spacing = 10;
		for (int i = 0; i < ball_nums_; i++)
		{
			ball_array_.Add(ball_scene.Instantiate() as Ball);
			AddChild(ball_array_[i]);
			float x = (i % rows - rows / 2f + 0.5f) * spacing + ball_rand_x_max/2;
			float y = (i / rows - cols / 2f + 0.5f) * spacing + ball_rand_y_max/2;
			ball_array_[i].Position = new Vector2(x, y);
		}
		Button button  = new Button();
		button.Text = "Press To Start Simulation!";
		button.Position = new Vector2(ball_rand_x_max/2,0);
		button.Pressed +=_ButtonPressed;
		AddChild(button);
		HashInit();
	}
	//fluid simulation used functions
	public float SmoothingKernel(float h, float dst){
		float q = dst / h;
		
		float weight;
		if (q >= 0 && q < 1)
			weight = 0.341f * ((2 - q) * (2 - q) * (2 - q) - 4 * (1 - q) * (1 - q) * (1 - q));
		else if (q >= 1 && q <= 2)
			weight = 0.341f * (2 - q) * (2 - q) * (2 - q);
		else
			weight = 0;
		return weight / h / h;
	}
	public float SmoothingKernelDerivative(float h, float dst) {
		float q = dst / h;
		float weight;
		if (q >= 0 && q < 1)
			weight = 0.341f * (-3 * (2 - q) * (2 - q) + 12 * (1 - q) * (1 - q)) / h;
		else if (q >= 1 && q <= 2)
			weight = -1.023f * (2 - q) * (2 - q) / h;
		else
			weight = 0;
		return weight / h / h;
	}
	// public float SmoothingKernel(float h, float dst){
	// 	if (dst >= h) return 0;
	// 	float v = (float)(Math.PI * Math.Pow(h, 4) / 6);
	// 	return (h - dst) *(h - dst) / v;
	// }

	// public float SmoothingKernelDerivative(float h, float dst){
	// 	if (dst >= h) return 0;
	// 	float scale = (float)(12/(Math.Pow(h, 4) * Math.PI));
	// 	return (dst - h) *scale;
	// }
	public void CalculateDensity(int index)
	{
		ball_array_[index].density = 0;
		var neighbor_idx_list = NeighborhoodSearch(index);
		for (int i = 0; i < neighbor_idx_list.Count; i++)
		{
			var dst = (ball_position_[neighbor_idx_list[i]] - ball_position_[index]).Length();
			ball_array_[index].density += ball_array_[neighbor_idx_list[i]].Mass * SmoothingKernel(smoothing_length, dst);
		}

	}
	public float CalculatePressure(float density)
	{
		return pressure_stiffness * (float)(Math.Pow(density / target_density, gamma_) - 1);
	}

	public Vector2 CalculateViscosityForce(int index, int i, float h)
	{

		Vector2 v_relative = ball_array_[i].LinearVelocity - ball_array_[index].LinearVelocity;
		float dst = (ball_position_[index] - ball_position_[i]).Length();
		float weight = SmoothingKernel(h, dst);
		return  v_relative * weight * vicosity_gain;
	} 

	public void CalculatePressureForce(int index)
	{
		float index_pressure = CalculatePressure(ball_array_[index].density);
		Vector2 acc = Vector2.Zero,
				viscosity_force = Vector2.Zero;
		var neighbor_idx_list = NeighborhoodSearch(index);
		for (int i = 0; i < neighbor_idx_list.Count; i++)
		{

			float i_pressure = CalculatePressure(ball_array_[neighbor_idx_list[i]].density);
			Vector2 dir_vec = ball_position_[neighbor_idx_list[i]] - ball_position_[index];
			float dst = dir_vec.Length();
			acc += ball_array_[neighbor_idx_list[i]].Mass * (
				index_pressure / ball_array_[index].density / ball_array_[index].density +
				i_pressure / ball_array_[neighbor_idx_list[i]].density / ball_array_[neighbor_idx_list[i]].density) *
				dir_vec.Normalized() * SmoothingKernelDerivative(smoothing_length, dst);
			viscosity_force += CalculateViscosityForce(index, neighbor_idx_list[i], smoothing_length);

		}
		ball_array_[index].pressure = acc * ball_array_[index].Mass + viscosity_force;
	}

	public override void _Input(InputEvent @event)
	{
		// Mouse in viewport coordinates.
		if (@event is InputEventMouse mouse)
		{
			// GD.Print(GetParent().GetNode("Ground").Name);
			mouse_position_ = mouse.Position - (GetParent().GetNode("Ground") as StaticBody2D).Position;
			if (Input.IsMouseButtonPressed(MouseButton.Left))
				mouse_pressed_ = true;
			else
				mouse_pressed_ = false;
		}
	}
	public override void _Draw()
	{
		// for (int i = 0; i < ball_nums_; i++)
		// {
		// 	float vel = ball_array_[i].LinearVelocity.Length();
		// 	Godot.Color color = new Godot.Color();
		// 	color.R = 0 + vel / 40;
		// 	color.G = 0 + vel / 40;
		// 	color.B = 255;
		// 	color.A = 0.85f;
		// 	(ball_array_[i].GetChild(0).GetChild(0) as CanvasItem).SelfModulate = color;
		// }
		// DrawLine(ball_position_[looking_idx_], mouse_position_, Colors.Red, 2);
		float circle_radius =35.0f * 2.0f;
		var circle_color = new Godot.Color(255,0,0,0.5f);

		DrawArc(mouse_position_, circle_radius, 0, (float)(2 * Math.PI), 32, circle_color, 6.0f, false);
    }
	
	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
		// if (!simulation_start_)
		// 	ball_array_.ForEach(item => item.Sleeping = true);
		// else
		// 	ball_array_.ForEach(item => item.Sleeping = false);
		QueueRedraw();
		ball_array_.ForEach(item => item.externel_force = Vector2.Zero);
		if (mouse_pressed_)
		{
			for (int i = 0; i < ball_nums_; i++)
			{
				float len = (mouse_position_ - ball_position_[i]).Length();
				float r_interaction = 35.0f * 2.0f;
				float kp = 6000;
				if (len < r_interaction)
				{
					Vector2 dir = (mouse_position_ - ball_position_[i]).Normalized();
					float weight = 1 - len / r_interaction;
					ball_array_[i].externel_force = dir * weight * kp - GetGravity();
				}

			}
		}
	}

    public override void _PhysicsProcess(double delta)
    {
		// GD.Print("-----------_PhysicsProcess started------------");
		// ulong start_time = Time.GetTicksMsec();
		for (int i = 0; i < ball_nums_; i++)
		{
			ball_position_[i] = ball_array_[i].Position + (float)delta * ball_array_[i].LinearVelocity;
		}
		Hashing();
		Parallel.For(0, ball_nums_, i =>
		{
			CalculateDensity(i);
		});
		Parallel.For(0,ball_nums_, i =>
		{
			CalculatePressureForce(i);
		});
		// GD.Print(" pressure = ", ball_array_[looking_idx_].pressure);
		// GD.Print("density = ",ball_array_[looking_idx_].density);
		// GD.Print("takes ",Time.GetTicksMsec() - start_time ," ms");	
		// GD.Print("-----------_PhysicsProcess ended------------");
	}
}
