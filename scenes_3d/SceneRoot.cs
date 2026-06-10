using Godot;
using System;

public partial class SceneRoot : Node3D
{
	public static ShaderMaterial ToonMaterial;
	public static ShaderMaterial OutlineMaterial;
	public static ShaderMaterial PlayerMaterial;
	public static ShaderMaterial PlayerOutlineMaterial;

	public override void _Ready()
	{
		BuildEnvironment();
		BuildToonMaterials();
		BuildLight();
		BuildGround();
		BuildPlatforms();
		BuildPlayer();           // must come BEFORE BuildWaterContainer so SPH can resolve PlayerPath
		BuildWaterContainer();
		BuildUI();
	}

	private void BuildEnvironment()
	{
		var envNode = new WorldEnvironment { Name = "WorldEnv" };
		var e = new Godot.Environment();

		var sky = new Sky();
		var skyMat = new ProceduralSkyMaterial();
		skyMat.SkyTopColor = new Color(0.35f, 0.55f, 0.85f);
		skyMat.SkyHorizonColor = new Color(0.75f, 0.82f, 0.92f);
		skyMat.GroundBottomColor = new Color(0.30f, 0.32f, 0.34f);
		skyMat.GroundHorizonColor = new Color(0.50f, 0.50f, 0.48f);
		sky.SkyMaterial = skyMat;

		e.BackgroundMode = Godot.Environment.BGMode.Sky;
		e.Sky = sky;
		// Ambient: sky 存在时默认就是 sky-driven。仅调强度
		e.AmbientLightEnergy = 0.25f;
		e.FogEnabled = true;
		e.FogLightColor = new Color(0.75f, 0.82f, 0.92f);
		e.FogDensity = 0.0015f;
		e.TonemapMode = Godot.Environment.ToneMapper.Linear;

		envNode.Environment = e;
		AddChild(envNode);
	}

	private void BuildToonMaterials()
	{
		var toonShader = GD.Load<Shader>("res://scenes_3d/toon.gdshader");
		var outlineShader = GD.Load<Shader>("res://scenes_3d/toon_outline.gdshader");

		ToonMaterial = new ShaderMaterial { Shader = toonShader };
		ToonMaterial.SetShaderParameter("base_color", new Color(0.45f, 0.62f, 0.32f));
		ToonMaterial.SetShaderParameter("toon_levels", 3.0f);
		ToonMaterial.SetShaderParameter("shadow_floor", 0.18f);
		ToonMaterial.SetShaderParameter("rim_strength", 0.35f);
		ToonMaterial.SetShaderParameter("rim_power", 3.0f);
		ToonMaterial.SetShaderParameter("rim_color", new Color(1.0f, 0.95f, 0.85f));

		OutlineMaterial = new ShaderMaterial { Shader = outlineShader };
		OutlineMaterial.SetShaderParameter("outline_thickness", 0.015f);
		OutlineMaterial.SetShaderParameter("outline_color", new Color(0.05f, 0.04f, 0.08f));
		ToonMaterial.NextPass = OutlineMaterial;

		PlayerMaterial = new ShaderMaterial { Shader = toonShader };
		PlayerMaterial.SetShaderParameter("base_color", new Color(0.92f, 0.55f, 0.22f));
		PlayerMaterial.SetShaderParameter("toon_levels", 3.0f);
		PlayerMaterial.SetShaderParameter("shadow_floor", 0.20f);
		PlayerMaterial.SetShaderParameter("rim_strength", 0.45f);
		PlayerMaterial.SetShaderParameter("rim_power", 3.0f);
		PlayerMaterial.SetShaderParameter("rim_color", new Color(1.0f, 0.95f, 0.85f));

		PlayerOutlineMaterial = new ShaderMaterial { Shader = outlineShader };
		PlayerOutlineMaterial.SetShaderParameter("outline_thickness", 0.020f);
		PlayerOutlineMaterial.SetShaderParameter("outline_color", new Color(0.05f, 0.04f, 0.08f));
		PlayerMaterial.NextPass = PlayerOutlineMaterial;
	}

	private void BuildLight()
	{
		var sun = new DirectionalLight3D { Name = "Sun" };
		sun.RotationDegrees = new Vector3(-55, -45, 0);
		sun.LightEnergy = 0.9f;
		sun.LightColor = new Color(1.0f, 0.97f, 0.90f);
		sun.ShadowEnabled = true;
		AddChild(sun);
	}

	private void BuildGround()
	{
		var ground = new StaticBody3D { Name = "Ground" };

		var meshInstance = new MeshInstance3D();
		var planeMesh = new PlaneMesh
		{
			Size = new Vector2(80, 80),
			SubdivideWidth = 4,
			SubdivideDepth = 4
		};
		meshInstance.Mesh = planeMesh;
		meshInstance.MaterialOverride = ToonMaterial;
		ground.AddChild(meshInstance);

		var col = new CollisionShape3D();
		var box = new BoxShape3D { Size = new Vector3(80, 0.2f, 80) };
		col.Shape = box;
		col.Position = new Vector3(0, -0.1f, 0);
		ground.AddChild(col);

		AddChild(ground);
	}

	private void BuildPlatforms()
	{
		Vector3[] positions = new Vector3[]
		{
			new Vector3(  6f, 2.0f, -4f),
			new Vector3( -7f, 3.5f,  6f),
			new Vector3(  2f, 5.0f,  9f),
			new Vector3( 10f, 4.0f,  6f),
			new Vector3(-10f, 6.5f, -8f),
		};
		Vector3[] sizes = new Vector3[]
		{
			new Vector3(4f, 0.6f, 4f),
			new Vector3(5f, 0.6f, 3f),
			new Vector3(3f, 0.6f, 3f),
			new Vector3(3f, 0.6f, 5f),
			new Vector3(4f, 0.6f, 4f),
		};
		for (int i = 0; i < positions.Length; i++)
		{
			var p = new StaticBody3D { Name = $"Platform{i}" };
			p.Position = positions[i];

			var m = new MeshInstance3D();
			var box = new BoxMesh { Size = sizes[i] };
			m.Mesh = box;
			m.MaterialOverride = ToonMaterial;
			p.AddChild(m);

			var col = new CollisionShape3D();
			var shape = new BoxShape3D { Size = sizes[i] };
			col.Shape = shape;
			p.AddChild(col);

			AddChild(p);
		}
	}

	private void BuildPlayer()
	{
		var player = new PlayerController { Name = "Player" };
		player.Position = new Vector3(0, 1.5f, 0);
		AddChild(player);
	}

	private void BuildWaterContainer()
	{
		var container = new WaterContainer { Name = "WaterContainer" };
		// Wide and shallow: 8x8 footprint, 1.5m deep -> water column
		// stays under ~0.7m which avoids the deep-column resonance that
		// kicks in past ~1m with current EOS-only SPH params.
		container.Position = new Vector3(0f, 0.3f, 5f);
		container.Size = new Vector3(8f, 1.5f, 8f);
		AddChild(container);

		// SPH renderer needs container to exist (reads BoundsMin/Max in _Ready)
		// before its own _Ready fires; Godot _Ready order is bottom-up, but
		// SetUp by deferring renderer add until container is in tree.
		var renderer = new SphRenderer3D { Name = "SphRenderer3D" };
		renderer.ContainerPath = container.GetPath();
		// Player is added before this in BuildPlayer (called earlier in _Ready),
		// so we can resolve its path now.
		var player = GetNodeOrNull<Node3D>("Player");
		if (player != null) renderer.PlayerPath = player.GetPath();
		AddChild(renderer);
	}

	private void BuildUI()
	{
		var canvas = new CanvasLayer { Name = "UI" };
		AddChild(canvas);
		var panel = new ToonPanel { Name = "ToonPanel" };
		canvas.AddChild(panel);
	}
}
