using Godot;

public partial class SmokeSpriteRenderer : MultiMeshInstance2D
{
	private const int PARTICLE_COUNT = 5000;

	private ShaderMaterial shader_material_;

	public override void _Ready()
	{
		// Build a unit QuadMesh (size 1x1, centered at origin -> VERTEX in [-0.5, 0.5]).
		var quad = new QuadMesh();
		quad.Size = new Vector2(1.0f, 1.0f);

		var mm = new MultiMesh();
		mm.TransformFormat = MultiMesh.TransformFormatEnum.Transform2D;
		mm.UseColors = false;
		mm.UseCustomData = false;
		mm.Mesh = quad;
		mm.InstanceCount = PARTICLE_COUNT;
		// All instances at identity transform; shader uses INSTANCE_ID to fetch
		// the actual particle position from position_tex.
		var identity = Transform2D.Identity;
		for (int i = 0; i < PARTICLE_COUNT; i++)
			mm.SetInstanceTransform2D(i, identity);
		Multimesh = mm;

		shader_material_ = new ShaderMaterial();
		shader_material_.Shader = ResourceLoader.Load<Shader>("res://scenes/smoke_sprite.gdshader");
		Material = shader_material_;
	}

	public void SetGpuTextures(Texture2Drd posTex, Texture2Drd physTex, Texture2Drd stableTex)
	{
		if (shader_material_ == null) return;
		// Must use the stable (pidx-indexed) view, not the sorted-slot one,
		// otherwise each sprite's INSTANCE_ID maps to a different particle
		// every frame as the spatial hash reshuffles -> visible per-sprite jitter.
		shader_material_.SetShaderParameter("position_tex", stableTex);
	}
}
