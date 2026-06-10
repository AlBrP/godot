using Godot;
using System;

// Procedural 6×4×6 m water container ("a bowl"): 5 StaticBody3D faces (bottom
// + 4 walls) + 1 transparent toon mesh per face for visual outline. The SPH
// solver does the actual fluid containment via 6 plane SDF in
// sph3d_integrate.glsl (bounds_min/bounds_max). The CharacterBody3D player
// uses these StaticBody3D faces to stand on / not fall through.
public partial class WaterContainer : Node3D
{
	[Export] public Vector3 Size = new Vector3(6f, 4f, 6f);
	[Export] public float WallThickness = 0.2f;
	[Export] public Color GlassTint = new Color(0.4f, 0.65f, 0.85f, 0.18f);

	public Vector3 BoundsMin { get; private set; }
	public Vector3 BoundsMax { get; private set; }

	public override void _Ready()
	{
		// Container origin = container floor center; bounds_min/max are world space
		Vector3 origin = GlobalPosition;
		BoundsMin = origin + new Vector3(-Size.X / 2f, 0f, -Size.Z / 2f);
		BoundsMax = origin + new Vector3(Size.X / 2f, Size.Y, Size.Z / 2f);

		BuildFloor();
		BuildWall(new Vector3(0, Size.Y / 2f, -Size.Z / 2f), new Vector3(Size.X, Size.Y, WallThickness));
		BuildWall(new Vector3(0, Size.Y / 2f,  Size.Z / 2f), new Vector3(Size.X, Size.Y, WallThickness));
		BuildWall(new Vector3(-Size.X / 2f, Size.Y / 2f, 0), new Vector3(WallThickness, Size.Y, Size.Z));
		BuildWall(new Vector3( Size.X / 2f, Size.Y / 2f, 0), new Vector3(WallThickness, Size.Y, Size.Z));
	}

	private void BuildFloor()
	{
		var body = new StaticBody3D { Name = "Floor" };
		body.Position = new Vector3(0, -WallThickness / 2f, 0);

		var mesh = new MeshInstance3D();
		var box = new BoxMesh { Size = new Vector3(Size.X, WallThickness, Size.Z) };
		mesh.Mesh = box;
		// Use the existing toon material for visible coherence with platforms.
		mesh.MaterialOverride = SceneRoot.ToonMaterial;
		body.AddChild(mesh);

		var col = new CollisionShape3D();
		col.Shape = new BoxShape3D { Size = new Vector3(Size.X, WallThickness, Size.Z) };
		body.AddChild(col);

		AddChild(body);
	}

	private void BuildWall(Vector3 localPos, Vector3 size)
	{
		// Walls are visual-only: SPH already keeps particles in bounds via the
		// integrate-pass plane clamp, and a physical wall would block the Player
		// from wading into the water. Floor still gets collision so the Player
		// has a surface to stand on inside the pool.
		var node = new Node3D { Name = $"Wall_{localPos.X:0}_{localPos.Z:0}" };
		node.Position = localPos;

		var mesh = new MeshInstance3D();
		var box = new BoxMesh { Size = size };
		mesh.Mesh = box;

		var glass = new StandardMaterial3D();
		glass.AlbedoColor = GlassTint;
		glass.Transparency = BaseMaterial3D.TransparencyEnum.Alpha;
		glass.CullMode = BaseMaterial3D.CullModeEnum.Disabled;
		glass.Roughness = 0.1f;
		mesh.MaterialOverride = glass;
		node.AddChild(mesh);

		AddChild(node);
	}
}
