using Godot;
using System;

public partial class Ball : RigidBody2D
{
	public float density = 1;
	public Vector2 pressure = new Vector2(0, 0);
	public Vector2 externel_force = new Vector2(0, 0);

	// Called when the node enters the scene tree for the first time.
	public override void _Ready()
	{
	}

	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public override void _Process(double delta)
	{
	}
	public override void _IntegrateForces(PhysicsDirectBodyState2D state)
	{
		state.ApplyCentralForce(pressure + externel_force);
    } 
}
