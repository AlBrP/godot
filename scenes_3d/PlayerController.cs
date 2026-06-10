using Godot;
using System;

public partial class PlayerController : CharacterBody3D
{
	[Export] public float MoveSpeed = 6.0f;
	[Export] public float JumpVelocity = 12.0f;   // can clear ~3.3m with gravity=22
	[Export] public float Gravity = 22.0f;
	[Export] public float MouseSensitivity = 0.0035f;
	[Export] public float CameraDistance = 6.0f;
	[Export] public float MinPitch = -1.2f;
	[Export] public float MaxPitch = 0.6f;

	private Node3D camera_yaw_;
	private Node3D camera_pitch_;
	private SpringArm3D spring_arm_;
	private Camera3D camera_;
	private float pitch_ = -0.4f;
	private bool mouse_captured_ = true;

	public override void _Ready()
	{
		BuildBody();
		BuildCamera();
		Input.MouseMode = Input.MouseModeEnum.Captured;
		mouse_captured_ = true;
	}

	private void BuildBody()
	{
		var mesh = new MeshInstance3D { Name = "Mesh" };
		var capsule = new CapsuleMesh { Radius = 0.4f, Height = 1.8f };
		mesh.Mesh = capsule;
		mesh.MaterialOverride = SceneRoot.PlayerMaterial;
		mesh.Position = new Vector3(0, 0.9f, 0);
		AddChild(mesh);

		var col = new CollisionShape3D { Name = "Col" };
		var capsuleShape = new CapsuleShape3D { Radius = 0.4f, Height = 1.8f };
		col.Shape = capsuleShape;
		col.Position = new Vector3(0, 0.9f, 0);
		AddChild(col);
	}

	private void BuildCamera()
	{
		camera_yaw_ = new Node3D { Name = "CameraYaw" };
		camera_yaw_.Position = new Vector3(0, 1.4f, 0);
		AddChild(camera_yaw_);

		camera_pitch_ = new Node3D { Name = "CameraPitch" };
		camera_yaw_.AddChild(camera_pitch_);

		spring_arm_ = new SpringArm3D { Name = "SpringArm" };
		spring_arm_.SpringLength = CameraDistance;
		spring_arm_.Margin = 0.2f;
		var arm_shape = new SphereShape3D { Radius = 0.2f };
		spring_arm_.Shape = arm_shape;
		camera_pitch_.AddChild(spring_arm_);

		camera_ = new Camera3D { Name = "Camera" };
		camera_.Position = Vector3.Zero;
		spring_arm_.AddChild(camera_);

		ApplyCameraPitch();
	}

	public override void _Input(InputEvent @event)
	{
		if (@event is InputEventKey key && key.Pressed && !key.Echo && key.Keycode == Key.Escape)
		{
			mouse_captured_ = !mouse_captured_;
			Input.MouseMode = mouse_captured_ ? Input.MouseModeEnum.Captured : Input.MouseModeEnum.Visible;
		}
		if (@event is InputEventMouseMotion mm && mouse_captured_)
		{
			camera_yaw_.RotateY(-mm.Relative.X * MouseSensitivity);
			pitch_ = Mathf.Clamp(pitch_ - mm.Relative.Y * MouseSensitivity, MinPitch, MaxPitch);
			ApplyCameraPitch();
		}
	}

	private void ApplyCameraPitch()
	{
		if (camera_pitch_ == null) return;
		var r = camera_pitch_.Rotation;
		r.X = pitch_;
		camera_pitch_.Rotation = r;
	}

	public override void _PhysicsProcess(double delta)
	{
		var velocity = Velocity;

		if (!IsOnFloor())
			velocity.Y -= Gravity * (float)delta;
		else if (Input.IsKeyPressed(Key.Space))
			velocity.Y = JumpVelocity;

		float forwardAxis = 0f;
		float rightAxis = 0f;
		if (Input.IsKeyPressed(Key.W)) forwardAxis += 1f;
		if (Input.IsKeyPressed(Key.S)) forwardAxis -= 1f;
		if (Input.IsKeyPressed(Key.A)) rightAxis -= 1f;
		if (Input.IsKeyPressed(Key.D)) rightAxis += 1f;

		Vector2 input_dir = new Vector2(rightAxis, forwardAxis);
		if (input_dir.LengthSquared() > 0f)
		{
			input_dir = input_dir.Normalized();
			Vector3 forward = -camera_yaw_.GlobalTransform.Basis.Z;
			Vector3 right = camera_yaw_.GlobalTransform.Basis.X;
			forward.Y = 0f; forward = forward.Normalized();
			right.Y = 0f; right = right.Normalized();
			Vector3 motion = (right * input_dir.X + forward * input_dir.Y) * MoveSpeed;
			velocity.X = motion.X;
			velocity.Z = motion.Z;
		}
		else
		{
			velocity.X = Mathf.Lerp(velocity.X, 0f, 15.0f * (float)delta);
			velocity.Z = Mathf.Lerp(velocity.Z, 0f, 15.0f * (float)delta);
		}

		Velocity = velocity;
		MoveAndSlide();
	}
}
