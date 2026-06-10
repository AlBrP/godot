using Godot;
using System;

// Drives the per-frame SPH dispatch + binds the output Texture2Drd to the
// sphere imposter MultiMeshInstance3D shader. Owns the SphGpu3D instance.
public partial class SphRenderer3D : Node3D
{
	[Export] public NodePath ContainerPath;

	// SPH parameters. KEY INSIGHT: at h=0.18 / lattice spacing 0.13, only the 6
	// axial neighbours fall inside the kernel (face-diagonals at 0.184 > h are
	// cut off). Lattice center density ~= 409 (self) + 6 * 31.6 = 599. If
	// target_density < 599 the bulk has POSITIVE pressure -> particles repel
	// outward -> "fluid" fills the whole container (what frames 1-9 showed).
	// target_density ABOVE lattice density makes cohesion dominate and the
	// particles settle into a real water layer at the bottom.
	[Export] public float Gravity = 9.8f;
	[Export] public float SmoothingRadius = 0.25f;
	[Export] public float TargetDensity = 370f;
	[Export] public float PressureMultiplier = 400f;
	[Export] public float NearPressureMultiplier = 3.0f;
	[Export] public float ViscosityStrength = 0f;
	[Export] public float CollisionDamping = 0.3f;
	[Export] public float PredictionFactor = 1f / 60f;
	// Hard CFL: single-step displacement must stay < smoothing_radius/2 so the
	// spatial hash neighbour lookup remains valid. With SubSteps=3, sub_dt =
	// 1/180s, so max_vel <= h/(2*sub_dt) = 0.18 * 90 = 16.2. Pick 8 for safety.
	[Export] public float MaxVel = 8f;
	[Export] public float VelocityDamping = 0.975f;
	[Export] public float ParticleRadius = 0.04f;
	[Export] public int SubSteps = 3;

	[Export] public float PlayerPushRadius = 1.2f;
	[Export] public NodePath PlayerPath;

	private SphGpu3D gpu_;
	private MultiMeshInstance3D mmi_;
	private MultiMesh mm_;
	private WaterContainer container_;
	private Vector3 bounds_min_;
	private Vector3 bounds_max_;
	private int dbg_frame_ = 0;
	private Node3D player_;
	private bool antigravity_ = false;

	public override void _Ready()
	{
		container_ = GetNode<WaterContainer>(ContainerPath);
		bounds_min_ = container_.BoundsMin;
		bounds_max_ = container_.BoundsMax;

		gpu_ = new SphGpu3D();
		gpu_.Init();

		SeedParticles();
		BuildMultiMesh();

		if (PlayerPath != null && !PlayerPath.IsEmpty)
			player_ = GetNodeOrNull<Node3D>(PlayerPath);

		GD.Print($"[SPH3D] bounds_min={bounds_min_} bounds_max={bounds_max_}");
		GD.Print($"[SPH3D] N={SphGpu3D.N} TEX={SphGpu3D.TEX_W}x{SphGpu3D.TEX_H} smoothing_radius={SmoothingRadius} max_vel={MaxVel}");
		GD.Print($"[SPH3D] Keys: R=reset particles, G=toggle antigravity, walk into water to push");
	}

	private void SeedParticles()
	{
		// Cubic lattice covering most of the container's lower half. With a
		// shallow 8x1.5x8 pool, 20k particles at lattice spacing 0.13 fill
		// ~0.5m of depth (Vy = 20000 * 0.13^3 / (7.7*7.7) = 0.74m of full
		// volume / footprint), so we cap fill at 90% height which leaves
		// headroom for splashes.
		int n = SphGpu3D.N;
		float h = SmoothingRadius * 0.65f;
		float margin = h * 1.2f;
		Vector3 vmin = bounds_min_ + new Vector3(margin, margin, margin);
		Vector3 vmax = bounds_max_ - new Vector3(margin, 0f, margin);
		vmax.Y = bounds_min_.Y + container_.Size.Y * 0.9f;

		int per_x = Mathf.Max(1, (int)((vmax.X - vmin.X) / h));
		int per_z = Mathf.Max(1, (int)((vmax.Z - vmin.Z) / h));
		int per_layer = per_x * per_z;
		int layers = Mathf.Max(1, (n + per_layer - 1) / per_layer);

		Vector3[] positions = new Vector3[n];
		var rng = new RandomNumberGenerator();
		rng.Seed = 1234;
		for (int i = 0; i < n; i++)
		{
			int layer = i / per_layer;
			int rem = i % per_layer;
			int ix = rem % per_x;
			int iz = rem / per_x;
			Vector3 p = new Vector3(
				vmin.X + (ix + 0.5f) * h,
				vmin.Y + (layer + 0.5f) * h,
				vmin.Z + (iz + 0.5f) * h);
			// tiny jitter to break the lattice symmetry so SPH cohesion has gradient
			p += new Vector3(rng.Randf() - 0.5f, rng.Randf() - 0.5f, rng.Randf() - 0.5f) * (h * 0.05f);
			// clamp inside container so seeded particles never start outside walls
			p.X = Mathf.Clamp(p.X, bounds_min_.X + 0.01f, bounds_max_.X - 0.01f);
			p.Y = Mathf.Clamp(p.Y, bounds_min_.Y + 0.01f, bounds_max_.Y - 0.01f);
			p.Z = Mathf.Clamp(p.Z, bounds_min_.Z + 0.01f, bounds_max_.Z - 0.01f);
			positions[i] = p;
		}
		gpu_.ResetParticles(positions);
		GD.Print($"[SPH3D] seed p[0]={positions[0]} p[5000]={positions[5000]} p[9999]={positions[9999]}");
	}

	private void BuildMultiMesh()
	{
		mm_ = new MultiMesh();
		mm_.TransformFormat = MultiMesh.TransformFormatEnum.Transform3D;
		mm_.UseCustomData = false;
		// Unit quad as imposter base.
		var quad = new QuadMesh();
		quad.Size = new Vector2(2f, 2f);  // VERTEX in [-1, 1]
		mm_.Mesh = quad;
		mm_.InstanceCount = SphGpu3D.N;

		// Instance transforms stay at identity -- the shader places each particle
		// via texelFetch(position_tex, INSTANCE_ID) + camera billboard math.
		Transform3D id = Transform3D.Identity;
		for (int i = 0; i < SphGpu3D.N; i++)
		{
			mm_.SetInstanceTransform(i, id);
		}

		mmi_ = new MultiMeshInstance3D { Name = "ParticleMM" };
		mmi_.Multimesh = mm_;
		// Big enough custom AABB so frustum culling never drops the whole mesh
		// (instance transforms are identity at origin -- without a custom AABB
		// Godot would conclude the mesh is a 2x2 quad at origin and cull it).
		mmi_.CustomAabb = new Aabb(bounds_min_, bounds_max_ - bounds_min_);

		var mat = new ShaderMaterial();
		mat.Shader = GD.Load<Shader>("res://scenes_3d/sph/sphere_imposter.gdshader");
		mat.SetShaderParameter("position_tex", gpu_.PositionTex);
		mat.SetShaderParameter("tex_w", SphGpu3D.TEX_W);
		mat.SetShaderParameter("tex_h", SphGpu3D.TEX_H);
		mat.SetShaderParameter("particle_radius", ParticleRadius);
		mat.SetShaderParameter("light_dir", new Vector3(-0.5f, -0.8f, -0.3f));
		mat.SetShaderParameter("base_color", new Vector3(0.30f, 0.55f, 0.85f));
		mat.SetShaderParameter("deep_color", new Vector3(0.08f, 0.22f, 0.45f));
		mat.SetShaderParameter("toon_levels", 3.0f);
		mat.SetShaderParameter("shadow_floor", 0.25f);
		mat.SetShaderParameter("rim_strength", 0.4f);
		mat.SetShaderParameter("rim_power", 3.0f);
		mat.SetShaderParameter("rim_color", new Vector3(1.0f, 0.97f, 0.90f));
		mmi_.MaterialOverride = mat;

		AddChild(mmi_);
	}

	public override void _PhysicsProcess(double delta)
	{
		if (gpu_ == null) return;
		float dt = (float)delta;
		float subDt = dt / Mathf.Max(1, SubSteps);
		float g = antigravity_ ? -Gravity * 0.5f : Gravity;
		Vector3 ppos = Vector3.Zero; float pr = 0f;
		if (player_ != null)
		{
			// Push sphere center at player's feet (his GlobalPosition, before
			// the capsule offset). Water layer is only ~0.25m thick: a center
			// at waist (y+0.9) sits half a metre above the water and never
			// touches a particle. With radius 1.2m the sphere envelops the
			// whole capsule + a generous ring of water around the feet.
			ppos = player_.GlobalPosition;
			pr = PlayerPushRadius;
		}
		gpu_.UpdateParams(
			bounds_min_, bounds_max_,
			dt, subDt, g, VelocityDamping,
			SmoothingRadius, PressureMultiplier, NearPressureMultiplier,
			ViscosityStrength, CollisionDamping, PredictionFactor, MaxVel,
			TargetDensity,
			ppos, pr);
		gpu_.DispatchFrame(SubSteps);

		dbg_frame_++;
		if (dbg_frame_ % 30 == 0)
		{
			int[] sample_ids = { 0, 100, 1000, 2500, 5000, 7500, 9000, 9999 };
			float den_min = 1e9f, den_max = 0f, den_sum = 0f;
			float vel_min = 1e9f, vel_max = 0f, vel_sum = 0f;
			float py_min = 1e9f, py_max = 0f;
			foreach (int id in sample_ids)
			{
				var pp = gpu_.ReadBackParticle(id);
				float vmag = pp.vel.Length();
				den_min = Mathf.Min(den_min, pp.density);
				den_max = Mathf.Max(den_max, pp.density);
				den_sum += pp.density;
				vel_min = Mathf.Min(vel_min, vmag);
				vel_max = Mathf.Max(vel_max, vmag);
				vel_sum += vmag;
				py_min = Mathf.Min(py_min, pp.pos.Y);
				py_max = Mathf.Max(py_max, pp.pos.Y);
			}
			GD.Print($"[SPH3D] f={dbg_frame_} den[min/avg/max]={den_min:0.0}/{den_sum/8:0.0}/{den_max:0.0} vel[min/avg/max]={vel_min:0.00}/{vel_sum/8:0.00}/{vel_max:0.00} y[{py_min:0.00},{py_max:0.00}] AG={antigravity_} player={ppos} r={pr}");
		}
	}

	public override void _UnhandledInput(InputEvent @event)
	{
		if (@event is InputEventKey key && key.Pressed && !key.Echo)
		{
			if (key.Keycode == Key.R)
			{
				SeedParticles();
				dbg_frame_ = 0;
				GD.Print("[SPH3D] Reset particles");
			}
			else if (key.Keycode == Key.G)
			{
				antigravity_ = !antigravity_;
				GD.Print($"[SPH3D] Antigravity = {antigravity_}");
			}
		}
	}

	// Called by ToonPanel slider hooks (if present).
	public void SetParam(string name, float v)
	{
		switch (name)
		{
			case "gravity": Gravity = v; break;
			case "smoothing_radius": SmoothingRadius = v; break;
			case "target_density": TargetDensity = v; break;
			case "pressure_multiplier": PressureMultiplier = v; break;
			case "near_pressure_multiplier": NearPressureMultiplier = v; break;
			case "viscosity_strength": ViscosityStrength = v; break;
			case "collision_damping": CollisionDamping = v; break;
			case "max_vel": MaxVel = v; break;
			case "velocity_damping": VelocityDamping = v; break;
			case "particle_radius":
				ParticleRadius = v;
				if (mmi_ != null && mmi_.MaterialOverride is ShaderMaterial sm)
					sm.SetShaderParameter("particle_radius", v);
				break;
		}
	}
}
