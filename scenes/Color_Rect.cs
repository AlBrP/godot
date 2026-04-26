using Godot;
using System;

public partial class Color_Rect : ColorRect
{
	Ground ground_;

	private const int TEXTURE_WIDTH = 72;
	private const int TEXTURE_HEIGHT = 70;
	private const float MAX_VEL = 2000f;

	private bool metaball_enabled_ = true;
	public bool MetaballEnabled => metaball_enabled_;

	private Texture2Drd gpu_position_tex_;
	private Texture2Drd gpu_hashlookup_tex_;

	private int debug_mode_ = 0;

	public override void _Ready()
	{
		ground_ = GetNode<Ground>("/root/main/Ground");
		MouseFilter = MouseFilterEnum.Ignore;

		var container = GetParent().GetParent() as SubViewportContainer;
		if (container != null)
			container.MouseFilter = MouseFilterEnum.Ignore;

		if (Material is ShaderMaterial shaderMaterial)
		{
			shaderMaterial.SetShaderParameter("particle_count", Ground.ball_nums_);
			shaderMaterial.SetShaderParameter("texture_size", new Vector2(TEXTURE_WIDTH, TEXTURE_HEIGHT));
			shaderMaterial.SetShaderParameter("max_vel", (double)MAX_VEL);
		}
	}

	public void SetGpuTextures(Texture2Drd posTex, Texture2Drd hlTex)
	{
		gpu_position_tex_ = posTex;
		gpu_hashlookup_tex_ = hlTex;
		if (Material is ShaderMaterial shaderMaterial)
		{
			shaderMaterial.SetShaderParameter("raw_position_texture", gpu_position_tex_);
			shaderMaterial.SetShaderParameter("hash_lookup_texture", gpu_hashlookup_tex_);
		}
	}

	public void SetGpuMode(bool enabled) { } // no-op: always GPU

	public override void _Process(double delta)
	{
		if (Material is not ShaderMaterial shaderMaterial) return;

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
		shaderMaterial.SetShaderParameter("spec_strength", (double)ground_.spec_strength);
		shaderMaterial.SetShaderParameter("toon_levels", (double)ground_.toon_levels);
		shaderMaterial.SetShaderParameter("time", Time.GetTicksMsec() / 1000.0f);

		var vpSize = ground_.GetViewportRect().Size;
		shaderMaterial.SetShaderParameter("viewport_size", vpSize);
		shaderMaterial.SetShaderParameter("bbox_min", ground_.Position);
		shaderMaterial.SetShaderParameter("bbox_max", new Vector2(vpSize.X, vpSize.Y - 100f));

		shaderMaterial.SetShaderParameter("grid_cell_size", (double)ground_.smoothing_radius);
		shaderMaterial.SetShaderParameter("ground_offset", ground_.Position);
		shaderMaterial.SetShaderParameter("sim_mode", ground_.SmokeMode ? 1 : 0);
	}

	public override void _Input(InputEvent @event) { }
}
