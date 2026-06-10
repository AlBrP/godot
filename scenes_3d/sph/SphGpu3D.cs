using Godot;
using System;

// 3D SPH GPU pipeline. 10-pass architecture mirrors the 2D SphGpu but stripped
// down to "pure water in a box" for P2.1 (no ptype/temp/age/curl/body/sdf).
//
// Particle SoA layout (PARTICLE_BUF, std430):
//   pos.xyz[N]      bytes 0           .. N*12
//   vel.xyz[N]      bytes N*12        .. N*24
//   pred.xyz[N]     bytes N*24        .. N*36
//   density[N]      bytes N*36        .. N*40
//   near_density[N] bytes N*40        .. N*44
//
// Hash buffer (std430):
//   hash_values[N]       bytes 0 .. N*4
//   grid_cell_coord[N]   bytes N*4 .. N*4 + N*16   (ivec3 stride = 16 in std430)
public class SphGpu3D
{
	private RenderingDevice RD;

	public const int N = 20000;
	public const int TEX_W = 200;
	public const int TEX_H = 100;
	private const int WORKGROUP = 256;
	private const int NUM_GROUPS = (N + WORKGROUP - 1) / WORKGROUP;
	private const int OUTPUT_GROUPS = (TEX_W * TEX_H + WORKGROUP - 1) / WORKGROUP;

	private Rid particle_buf;
	private Rid hash_buf;
	private Rid sort_buf;
	private Rid hashtable_buf;
	private Rid histogram_buf;
	private Rid prefix_buf;
	private Rid block_sums_buf;
	private Rid params_ubuf;

	private Rid position_tex_rid;
	public Texture2Drd PositionTex { get; private set; }

	private Rid shader_forces_hash, pipeline_forces_hash;
	private Rid shader_histogram, pipeline_histogram;
	private Rid shader_prefix_local, pipeline_prefix_local;
	private Rid shader_prefix_top, pipeline_prefix_top;
	private Rid shader_prefix_apply, pipeline_prefix_apply;
	private Rid shader_scatter, pipeline_scatter;
	private Rid shader_density, pipeline_density;
	private Rid shader_pressure_visc, pipeline_pressure_visc;
	private Rid shader_integrate, pipeline_integrate;
	private Rid shader_output, pipeline_output;

	private Rid uniform_set_forces_hash;
	private Rid uniform_set_histogram;
	private Rid uniform_set_prefix_local;
	private Rid uniform_set_prefix_top;
	private Rid uniform_set_prefix_apply;
	private Rid uniform_set_scatter;
	private Rid uniform_set_density;
	private Rid uniform_set_pressure_visc;
	private Rid uniform_set_integrate;
	private Rid uniform_set_output_0;
	private Rid uniform_set_output_1;

	private const int PARTICLE_BUF_SIZE = N * (12 + 12 + 12 + 4 + 4); // 440 KB
	private const int HASH_BUF_SIZE = N * 4 + N * 16;                  // 200 KB (uint + ivec3 std430 stride 16)
	private const int SORT_BUF_SIZE = N * (4 + 4);
	private const int HASHTABLE_BUF_SIZE = N * (4 + 4);
	private const int HISTOGRAM_BUF_SIZE = N * 4;
	private const int PREFIX_BUF_SIZE = N * 4;
	private const int BLOCK_SUMS_BUF_SIZE = NUM_GROUPS * 4;
	// 96 bytes core + 16 bytes player_data (xyz=pos, w=push_radius)
	private const int PARAMS_BUF_SIZE = 112;

	public void Init()
	{
		RD = RenderingServer.GetRenderingDevice();
		CreateBuffers();
		CreateTextures();
		LoadShaders();
		CreateUniformSets();
	}

	private void CreateBuffers()
	{
		particle_buf = RD.StorageBufferCreate(PARTICLE_BUF_SIZE, new byte[PARTICLE_BUF_SIZE]);
		hash_buf = RD.StorageBufferCreate(HASH_BUF_SIZE, new byte[HASH_BUF_SIZE]);
		sort_buf = RD.StorageBufferCreate(SORT_BUF_SIZE, new byte[SORT_BUF_SIZE]);
		hashtable_buf = RD.StorageBufferCreate(HASHTABLE_BUF_SIZE, new byte[HASHTABLE_BUF_SIZE]);
		histogram_buf = RD.StorageBufferCreate(HISTOGRAM_BUF_SIZE, new byte[HISTOGRAM_BUF_SIZE]);
		prefix_buf = RD.StorageBufferCreate(PREFIX_BUF_SIZE, new byte[PREFIX_BUF_SIZE]);
		block_sums_buf = RD.StorageBufferCreate(BLOCK_SUMS_BUF_SIZE, new byte[BLOCK_SUMS_BUF_SIZE]);
		params_ubuf = RD.UniformBufferCreate(PARAMS_BUF_SIZE, new byte[PARAMS_BUF_SIZE]);
	}

	private void CreateTextures()
	{
		var defaultView = new RDTextureView();

		// R32G32B32A32: 3D world positions need full float precision; R16F clips
		// values past ~65k and loses sub-millimeter resolution at world-scale.
		var posFmt = new RDTextureFormat();
		posFmt.Width = TEX_W; posFmt.Height = TEX_H;
		posFmt.Format = RenderingDevice.DataFormat.R32G32B32A32Sfloat;
		posFmt.UsageBits = RenderingDevice.TextureUsageBits.StorageBit | RenderingDevice.TextureUsageBits.SamplingBit;
		position_tex_rid = RD.TextureCreate(posFmt, defaultView);
		PositionTex = new Texture2Drd();
		PositionTex.TextureRdRid = position_tex_rid;
	}

	private Rid LoadComputeShader(string path)
	{
		var shaderFile = GD.Load<RDShaderFile>(path);
		var spirv = shaderFile.GetSpirV();
		return RD.ShaderCreateFromSpirV(spirv);
	}

	private void LoadShaders()
	{
		string basePath = "res://scenes_3d/sph/compute/";
		shader_forces_hash = LoadComputeShader(basePath + "sph3d_forces_hash.glsl");
		pipeline_forces_hash = RD.ComputePipelineCreate(shader_forces_hash);
		shader_histogram = LoadComputeShader(basePath + "sph3d_histogram.glsl");
		pipeline_histogram = RD.ComputePipelineCreate(shader_histogram);
		shader_prefix_local = LoadComputeShader(basePath + "sph3d_prefix_local.glsl");
		pipeline_prefix_local = RD.ComputePipelineCreate(shader_prefix_local);
		shader_prefix_top = LoadComputeShader(basePath + "sph3d_prefix_top.glsl");
		pipeline_prefix_top = RD.ComputePipelineCreate(shader_prefix_top);
		shader_prefix_apply = LoadComputeShader(basePath + "sph3d_prefix_apply.glsl");
		pipeline_prefix_apply = RD.ComputePipelineCreate(shader_prefix_apply);
		shader_scatter = LoadComputeShader(basePath + "sph3d_scatter.glsl");
		pipeline_scatter = RD.ComputePipelineCreate(shader_scatter);
		shader_density = LoadComputeShader(basePath + "sph3d_density.glsl");
		pipeline_density = RD.ComputePipelineCreate(shader_density);
		shader_pressure_visc = LoadComputeShader(basePath + "sph3d_pressure_viscosity.glsl");
		pipeline_pressure_visc = RD.ComputePipelineCreate(shader_pressure_visc);
		shader_integrate = LoadComputeShader(basePath + "sph3d_integrate.glsl");
		pipeline_integrate = RD.ComputePipelineCreate(shader_integrate);
		shader_output = LoadComputeShader(basePath + "sph3d_output.glsl");
		pipeline_output = RD.ComputePipelineCreate(shader_output);
	}

	private RDUniform Storage(int binding, Rid buf)
	{
		var u = new RDUniform();
		u.UniformType = RenderingDevice.UniformType.StorageBuffer;
		u.Binding = binding; u.AddId(buf);
		return u;
	}
	private RDUniform UniformBuf(int binding, Rid buf)
	{
		var u = new RDUniform();
		u.UniformType = RenderingDevice.UniformType.UniformBuffer;
		u.Binding = binding; u.AddId(buf);
		return u;
	}
	private RDUniform Image(int binding, Rid tex)
	{
		var u = new RDUniform();
		u.UniformType = RenderingDevice.UniformType.Image;
		u.Binding = binding; u.AddId(tex);
		return u;
	}

	private void CreateUniformSets()
	{
		// bindings: 0=particle, 1=hash, 2=sort, 3=hashtable,
		//           4=histogram, 5=prefix, 6=block_sums, 7=params
		uniform_set_forces_hash = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(0, particle_buf), Storage(1, hash_buf), UniformBuf(7, params_ubuf),
		}, shader_forces_hash, 0);

		uniform_set_histogram = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(1, hash_buf), Storage(4, histogram_buf), UniformBuf(7, params_ubuf),
		}, shader_histogram, 0);

		uniform_set_prefix_local = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(4, histogram_buf), Storage(5, prefix_buf),
			Storage(6, block_sums_buf), UniformBuf(7, params_ubuf),
		}, shader_prefix_local, 0);

		uniform_set_prefix_top = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(6, block_sums_buf), UniformBuf(7, params_ubuf),
		}, shader_prefix_top, 0);

		uniform_set_prefix_apply = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(3, hashtable_buf), Storage(4, histogram_buf),
			Storage(5, prefix_buf), Storage(6, block_sums_buf),
			UniformBuf(7, params_ubuf),
		}, shader_prefix_apply, 0);

		uniform_set_scatter = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(1, hash_buf), Storage(2, sort_buf),
			Storage(5, prefix_buf), UniformBuf(7, params_ubuf),
		}, shader_scatter, 0);

		uniform_set_density = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(0, particle_buf), Storage(1, hash_buf),
			Storage(2, sort_buf), Storage(3, hashtable_buf),
			UniformBuf(7, params_ubuf),
		}, shader_density, 0);

		uniform_set_pressure_visc = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(0, particle_buf), Storage(1, hash_buf),
			Storage(2, sort_buf), Storage(3, hashtable_buf),
			UniformBuf(7, params_ubuf),
		}, shader_pressure_visc, 0);

		uniform_set_integrate = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(0, particle_buf), UniformBuf(7, params_ubuf),
		}, shader_integrate, 0);

		uniform_set_output_0 = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Storage(0, particle_buf), Storage(2, sort_buf), UniformBuf(7, params_ubuf),
		}, shader_output, 0);
		uniform_set_output_1 = RD.UniformSetCreate(new Godot.Collections.Array<RDUniform> {
			Image(0, position_tex_rid),
		}, shader_output, 1);
	}

	private static void WriteFloat(byte[] buf, ref int o, float v)
	{
		byte[] b = BitConverter.GetBytes(v);
		buf[o] = b[0]; buf[o + 1] = b[1]; buf[o + 2] = b[2]; buf[o + 3] = b[3];
		o += 4;
	}
	private static void WriteInt(byte[] buf, ref int o, int v)
	{
		byte[] b = BitConverter.GetBytes(v);
		buf[o] = b[0]; buf[o + 1] = b[1]; buf[o + 2] = b[2]; buf[o + 3] = b[3];
		o += 4;
	}
	private static void WriteVec3Pad(byte[] buf, ref int o, Vector3 v)
	{
		WriteFloat(buf, ref o, v.X); WriteFloat(buf, ref o, v.Y); WriteFloat(buf, ref o, v.Z);
		o += 4;  // std140 vec3 padding (next member aligned to vec4 boundary)
	}

	public void UpdateParams(
		Vector3 boundsMin, Vector3 boundsMax,
		float dt, float subDt, float gravity, float velocityDamping,
		float smoothingRadius, float pressureMultiplier, float nearPressureMultiplier,
		float viscosityStrength, float collisionDamping, float predictionFactor, float maxVel,
		float targetDensity,
		Vector3 playerPos, float pushRadius)
	{
		byte[] data = new byte[PARAMS_BUF_SIZE];
		int o = 0;
		WriteVec3Pad(data, ref o, boundsMin);                        // 0
		WriteVec3Pad(data, ref o, boundsMax);                        // 16
		WriteFloat(data, ref o, dt);                                  // 32
		WriteFloat(data, ref o, subDt);                              // 36
		WriteFloat(data, ref o, gravity);                            // 40
		WriteFloat(data, ref o, velocityDamping);                    // 44
		WriteFloat(data, ref o, smoothingRadius);                    // 48
		WriteFloat(data, ref o, smoothingRadius * smoothingRadius);  // 52
		WriteFloat(data, ref o, 1f / smoothingRadius);                // 56
		WriteFloat(data, ref o, pressureMultiplier);                 // 60
		WriteFloat(data, ref o, nearPressureMultiplier);             // 64
		WriteFloat(data, ref o, viscosityStrength);                  // 68
		WriteFloat(data, ref o, collisionDamping);                   // 72
		WriteFloat(data, ref o, predictionFactor);                   // 76
		WriteFloat(data, ref o, maxVel);                              // 80
		WriteFloat(data, ref o, targetDensity);                      // 84
		WriteInt(data, ref o, N);                                     // 88
		WriteFloat(data, ref o, 0f);                                  // 92 _pad
		// player_data: xyz = world pos, w = push radius (<=0 disables)
		WriteFloat(data, ref o, playerPos.X);                         // 96
		WriteFloat(data, ref o, playerPos.Y);                         // 100
		WriteFloat(data, ref o, playerPos.Z);                         // 104
		WriteFloat(data, ref o, pushRadius);                          // 108
		RD.BufferUpdate(params_ubuf, 0, PARAMS_BUF_SIZE, data);
	}

	public void DispatchFrame(int iterations)
	{
		for (int iter = 0; iter < iterations; iter++)
		{
			RD.BufferClear(histogram_buf, 0, HISTOGRAM_BUF_SIZE);
			RD.BufferClear(prefix_buf, 0, PREFIX_BUF_SIZE);
			RD.BufferClear(block_sums_buf, 0, BLOCK_SUMS_BUF_SIZE);

			var cl = RD.ComputeListBegin();

			RD.ComputeListBindComputePipeline(cl, pipeline_forces_hash);
			RD.ComputeListBindUniformSet(cl, uniform_set_forces_hash, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_histogram);
			RD.ComputeListBindUniformSet(cl, uniform_set_histogram, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_prefix_local);
			RD.ComputeListBindUniformSet(cl, uniform_set_prefix_local, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_prefix_top);
			RD.ComputeListBindUniformSet(cl, uniform_set_prefix_top, 0);
			RD.ComputeListDispatch(cl, 1, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_prefix_apply);
			RD.ComputeListBindUniformSet(cl, uniform_set_prefix_apply, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_scatter);
			RD.ComputeListBindUniformSet(cl, uniform_set_scatter, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_density);
			RD.ComputeListBindUniformSet(cl, uniform_set_density, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_pressure_visc);
			RD.ComputeListBindUniformSet(cl, uniform_set_pressure_visc, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			RD.ComputeListBindComputePipeline(cl, pipeline_integrate);
			RD.ComputeListBindUniformSet(cl, uniform_set_integrate, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);

			RD.ComputeListEnd();
		}

		{
			var cl = RD.ComputeListBegin();
			RD.ComputeListBindComputePipeline(cl, pipeline_output);
			RD.ComputeListBindUniformSet(cl, uniform_set_output_0, 0);
			RD.ComputeListBindUniformSet(cl, uniform_set_output_1, 1);
			RD.ComputeListDispatch(cl, OUTPUT_GROUPS, 1, 1);
			RD.ComputeListEnd();
		}
	}

	// Reset particles: positions array of length N (Vector3 each).
	// Velocity/pred/density/near_density are cleared to 0.
	public void ResetParticles(Vector3[] positions)
	{
		byte[] posData = new byte[N * 12];
		for (int i = 0; i < N; i++)
		{
			int off = i * 12;
			byte[] xb = BitConverter.GetBytes(positions[i].X);
			byte[] yb = BitConverter.GetBytes(positions[i].Y);
			byte[] zb = BitConverter.GetBytes(positions[i].Z);
			for (int k = 0; k < 4; k++)
			{
				posData[off + k] = xb[k];
				posData[off + 4 + k] = yb[k];
				posData[off + 8 + k] = zb[k];
			}
		}
		RD.BufferUpdate(particle_buf, 0, (uint)posData.Length, posData);
		RD.BufferClear(particle_buf, (uint)(N * 12), (uint)(N * 12));  // vel
		RD.BufferClear(particle_buf, (uint)(N * 24), (uint)(N * 12));  // pred
		RD.BufferClear(particle_buf, (uint)(N * 36), (uint)(N * 4));   // density
		RD.BufferClear(particle_buf, (uint)(N * 40), (uint)(N * 4));   // near_density
	}

	// Debug: read back one particle's state (pos, vel, density, near_density).
	public (Vector3 pos, Vector3 vel, float density, float near_density) ReadBackParticle(int i)
	{
		byte[] posBytes = RD.BufferGetData(particle_buf, (uint)(i * 12), 12);
		byte[] velBytes = RD.BufferGetData(particle_buf, (uint)(N * 12 + i * 12), 12);
		byte[] denBytes = RD.BufferGetData(particle_buf, (uint)(N * 36 + i * 4), 4);
		byte[] ndBytes = RD.BufferGetData(particle_buf, (uint)(N * 40 + i * 4), 4);
		Vector3 pos = new Vector3(
			BitConverter.ToSingle(posBytes, 0),
			BitConverter.ToSingle(posBytes, 4),
			BitConverter.ToSingle(posBytes, 8));
		Vector3 vel = new Vector3(
			BitConverter.ToSingle(velBytes, 0),
			BitConverter.ToSingle(velBytes, 4),
			BitConverter.ToSingle(velBytes, 8));
		float density = BitConverter.ToSingle(denBytes, 0);
		float near_density = BitConverter.ToSingle(ndBytes, 0);
		return (pos, vel, density, near_density);
	}

	// Debug: read back the entire Params UBO and return the 24 floats so the
	// caller can verify std140 layout end-to-end (catches the bounds_min/max
	// vec3-padding bug + any future drift).
	public float[] ReadBackParams()
	{
		byte[] data = RD.BufferGetData(params_ubuf, 0, PARAMS_BUF_SIZE);
		float[] floats = new float[PARAMS_BUF_SIZE / 4];
		for (int i = 0; i < floats.Length; i++)
			floats[i] = BitConverter.ToSingle(data, i * 4);
		return floats;
	}
}
