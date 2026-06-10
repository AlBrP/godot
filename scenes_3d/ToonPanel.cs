using Godot;
using System;

public partial class ToonPanel : Control
{
	private bool panel_visible_ = false;
	private VBoxContainer container_;

	public override void _Ready()
	{
		// Anchor top-left
		AnchorLeft = 0; AnchorTop = 0; AnchorRight = 0; AnchorBottom = 0;
		OffsetLeft = 12; OffsetTop = 12;
		CustomMinimumSize = new Vector2(380, 280);

		var bg = new Panel { Name = "Bg" };
		bg.AnchorRight = 1; bg.AnchorBottom = 1;
		bg.MouseFilter = MouseFilterEnum.Pass;
		AddChild(bg);

		container_ = new VBoxContainer { Name = "Box" };
		container_.AnchorRight = 1; container_.AnchorBottom = 1;
		container_.OffsetLeft = 10; container_.OffsetTop = 10;
		container_.OffsetRight = -10; container_.OffsetBottom = -10;
		AddChild(container_);

		AddLabel("Toon 调参 (按 V 切换)");
		AddSeparator();

		AddSlider("toon_levels",       1f,   8f,    1f,    "toon_levels",       true);
		AddSlider("shadow_floor",      0f,   0.8f,  0.02f, "shadow_floor",      true);
		AddSlider("rim_strength",      0f,   2f,    0.05f, "rim_strength",      true);
		AddSlider("rim_power",         0.5f, 8f,    0.1f,  "rim_power",         true);
		AddOutlineSlider("outline_thickness", 0f, 0.05f, 0.001f);

		AddSeparator();
		AddLabel("SPH 流体参数");
		AddSphSlider("smoothing_radius",        0.05f,  0.5f,   0.005f);
		AddSphSlider("target_density",          50f,    3000f,  10f);
		AddSphSlider("pressure_multiplier",     1f,     500f,   1f);
		AddSphSlider("near_pressure_multiplier", 0f,     20f,    0.1f);
		AddSphSlider("viscosity_strength",      0f,     2f,     0.02f);
		AddSphSlider("collision_damping",       0f,     1f,     0.02f);
		AddSphSlider("max_vel",                 1f,     50f,    0.5f);
		AddSphSlider("velocity_damping",        0.9f,   1f,     0.001f);
		AddSphSlider("particle_radius",         0.02f,  0.2f,   0.005f);
		AddSphSlider("gravity",                 0f,     30f,    0.5f);

		Visible = panel_visible_;
	}

	public override void _UnhandledInput(InputEvent @event)
	{
		if (@event is InputEventKey key && key.Pressed && !key.Echo && key.Keycode == Key.V)
		{
			panel_visible_ = !panel_visible_;
			Visible = panel_visible_;
		}
	}

	private void AddLabel(string text)
	{
		var l = new Label { Text = text };
		container_.AddChild(l);
	}

	private void AddSeparator()
	{
		container_.AddChild(new HSeparator());
	}

	private void AddSlider(string display, float min, float max, float step, string param, bool applyToPlayer)
	{
		var row = new HBoxContainer();
		var nameLabel = new Label { Text = display, CustomMinimumSize = new Vector2(150, 0) };
		row.AddChild(nameLabel);

		var slider = new HSlider();
		slider.MinValue = min; slider.MaxValue = max; slider.Step = step;
		slider.SizeFlagsHorizontal = SizeFlags.ExpandFill;
		slider.CustomMinimumSize = new Vector2(140, 0);
		var initial = (float)(double)SceneRoot.ToonMaterial.GetShaderParameter(param);
		slider.Value = initial;

		var valueLabel = new Label { Text = initial.ToString("0.000"), CustomMinimumSize = new Vector2(60, 0) };

		slider.ValueChanged += (newVal) =>
		{
			float v = (float)newVal;
			SceneRoot.ToonMaterial.SetShaderParameter(param, v);
			if (applyToPlayer) SceneRoot.PlayerMaterial.SetShaderParameter(param, v);
			valueLabel.Text = v.ToString("0.000");
		};

		row.AddChild(slider);
		row.AddChild(valueLabel);
		container_.AddChild(row);
	}

	private void AddOutlineSlider(string display, float min, float max, float step)
	{
		var row = new HBoxContainer();
		var nameLabel = new Label { Text = display, CustomMinimumSize = new Vector2(150, 0) };
		row.AddChild(nameLabel);

		var slider = new HSlider();
		slider.MinValue = min; slider.MaxValue = max; slider.Step = step;
		slider.SizeFlagsHorizontal = SizeFlags.ExpandFill;
		slider.CustomMinimumSize = new Vector2(140, 0);
		var initial = (float)(double)SceneRoot.OutlineMaterial.GetShaderParameter("outline_thickness");
		slider.Value = initial;

		var valueLabel = new Label { Text = initial.ToString("0.000"), CustomMinimumSize = new Vector2(60, 0) };

		slider.ValueChanged += (newVal) =>
		{
			float v = (float)newVal;
			SceneRoot.OutlineMaterial.SetShaderParameter("outline_thickness", v);
			SceneRoot.PlayerOutlineMaterial.SetShaderParameter("outline_thickness", v * 1.3f);
			valueLabel.Text = v.ToString("0.000");
		};

		row.AddChild(slider);
		row.AddChild(valueLabel);
		container_.AddChild(row);
	}

	private SphRenderer3D FindSph()
	{
		// SceneRoot adds SphRenderer3D as a sibling under root. Walk up from
		// this UI control's parent chain (CanvasLayer -> SceneRoot) and look.
		var root = GetTree().Root.GetNodeOrNull("Main3D");
		return root?.GetNodeOrNull<SphRenderer3D>("SphRenderer3D");
	}

	private void AddSphSlider(string param, float min, float max, float step)
	{
		var row = new HBoxContainer();
		var nameLabel = new Label { Text = param, CustomMinimumSize = new Vector2(150, 0) };
		row.AddChild(nameLabel);

		var slider = new HSlider();
		slider.MinValue = min; slider.MaxValue = max; slider.Step = step;
		slider.SizeFlagsHorizontal = SizeFlags.ExpandFill;
		slider.CustomMinimumSize = new Vector2(140, 0);

		// Defer reading initial until SphRenderer3D is in tree (it's added in
		// SceneRoot._Ready right before BuildUI, so should be present here).
		var sph = FindSph();
		float initial = 0f;
		if (sph != null) initial = ReadSphParam(sph, param);
		slider.Value = initial;

		var valueLabel = new Label { Text = initial.ToString("0.000"), CustomMinimumSize = new Vector2(60, 0) };

		slider.ValueChanged += (newVal) =>
		{
			float v = (float)newVal;
			var s = FindSph();
			if (s != null) s.SetParam(param, v);
			valueLabel.Text = v.ToString("0.000");
		};

		row.AddChild(slider);
		row.AddChild(valueLabel);
		container_.AddChild(row);
	}

	private float ReadSphParam(SphRenderer3D s, string param)
	{
		switch (param)
		{
			case "gravity":                  return s.Gravity;
			case "smoothing_radius":          return s.SmoothingRadius;
			case "target_density":           return s.TargetDensity;
			case "pressure_multiplier":      return s.PressureMultiplier;
			case "near_pressure_multiplier": return s.NearPressureMultiplier;
			case "viscosity_strength":       return s.ViscosityStrength;
			case "collision_damping":        return s.CollisionDamping;
			case "max_vel":                  return s.MaxVel;
			case "velocity_damping":         return s.VelocityDamping;
			case "particle_radius":          return s.ParticleRadius;
			default: return 0f;
		}
	}
}
