using Godot;
using System;

public class SphGpu
{
	private RenderingDevice RD;

	private const int N = 5000;
	private const int TEX_W = 72;
	private const int TEX_H = 70;
	private const int WORKGROUP = 256;
	private const int NUM_GROUPS = (N + WORKGROUP - 1) / WORKGROUP; // 20
	private const int OUTPUT_GROUPS = (TEX_W * TEX_H + WORKGROUP - 1) / WORKGROUP; // 20

	// Buffer RIDs
	private Rid particle_buf;
	private Rid hash_buf;
	private Rid sort_buf;
	private Rid hashtable_buf;
	private Rid histogram_buf;
	private Rid prefix_buf;
	private Rid block_sums_buf;
	private Rid params_ubuf;

	// Texture RIDs + Texture2Drd (bridge RD texture to scene tree)
	private Rid position_tex_rid;
	private Rid hashlookup_tex_rid;
	public Texture2Drd PositionTex { get; private set; }
	public Texture2Drd HashLookupTex { get; private set; }

	// Force accumulator buffer (2 × float, for GPU→CPU readback)
	private Rid force_accum_buf;

	// Water info buffer (water_count + water_y_min_scaled, for Archimedes buoyancy)
	private Rid water_info_buf;

	// Shader + Pipeline RIDs
	private Rid shader_forces_hash;
	private Rid pipeline_forces_hash;
	private Rid shader_histogram;
	private Rid pipeline_histogram;
	private Rid shader_prefix_local;
	private Rid pipeline_prefix_local;
	private Rid shader_prefix_top;
	private Rid pipeline_prefix_top;
	private Rid shader_prefix_apply;
	private Rid pipeline_prefix_apply;
	private Rid shader_scatter;
	private Rid pipeline_scatter;
	private Rid shader_density;
	private Rid pipeline_density;
	private Rid shader_pressure_visc;
	private Rid pipeline_pressure_visc;
	private Rid shader_integrate;
	private Rid pipeline_integrate;
	private Rid shader_output;
	private Rid pipeline_output;

	// Uniform Set RIDs
	private Rid uniform_set_forces_hash;   // set=0: particle(0), hash(1), params(7)
	private Rid uniform_set_histogram;      // set=0: hash(1), histogram(4)
	private Rid uniform_set_prefix_local;   // set=0: histogram(4), prefix(5), block_sums(6)
	private Rid uniform_set_prefix_top;     // set=0: block_sums(6)
	private Rid uniform_set_prefix_apply;  // set=0: hashtable(3), histogram(4), prefix(5), block_sums(6)
	private Rid uniform_set_scatter;       // set=0: hash(1), sort(2), prefix(5)
	private Rid uniform_set_density;       // set=0: particle(0), hash(1), sort(2), hashtable(3), params(7)
	private Rid uniform_set_pressure_visc; // set=0: particle(0), hash(1), sort(2), hashtable(3), params(7)
	private Rid uniform_set_integrate;     // set=0: particle(0), params(7)
	private Rid uniform_set_output_0;      // set=0: particle(0), sort(2), hashtable(3), params(7)
	private Rid uniform_set_output_1;      // set=1: position_tex(0), hashlookup_tex(1)

	// Buffer sizes
	private const int PARTICLE_BUF_SIZE = N * (8 + 8 + 8 + 4 + 4); // 160000
	private const int HASH_BUF_SIZE = N * (4 + 8);                 // 60000
	private const int SORT_BUF_SIZE = N * (4 + 4);                // 40000
	private const int HASHTABLE_BUF_SIZE = N * (4 + 4);           // 40000
	private const int HISTOGRAM_BUF_SIZE = N * 4;                 // 20000
	private const int PREFIX_BUF_SIZE = N * 4;                    // 20000
	private const int BLOCK_SUMS_BUF_SIZE = NUM_GROUPS * 4;       // 80
	private const int PARAMS_BUF_SIZE = 160;                      // std140: 160 bytes (10 * 16)

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

		// Position texture: RGBA16F 72x70
		var posFmt = new RDTextureFormat();
		posFmt.Width = TEX_W;
		posFmt.Height = TEX_H;
		posFmt.Format = RenderingDevice.DataFormat.R16G16B16A16Sfloat;
		posFmt.UsageBits = RenderingDevice.TextureUsageBits.StorageBit | RenderingDevice.TextureUsageBits.SamplingBit;
		position_tex_rid = RD.TextureCreate(posFmt, defaultView);

		// Hash lookup texture: RG32F 72x70
		var hlFmt = new RDTextureFormat();
		hlFmt.Width = TEX_W;
		hlFmt.Height = TEX_H;
		hlFmt.Format = RenderingDevice.DataFormat.R32G32Sfloat;
		hlFmt.UsageBits = RenderingDevice.TextureUsageBits.StorageBit | RenderingDevice.TextureUsageBits.SamplingBit;
		hashlookup_tex_rid = RD.TextureCreate(hlFmt, defaultView);

		// Bridge to Texture2D for canvas_item shader
		PositionTex = new Texture2Drd();
		PositionTex.TextureRdRid = position_tex_rid;
		HashLookupTex = new Texture2Drd();
		HashLookupTex.TextureRdRid = hashlookup_tex_rid;

		// Force accumulator buffer: 8 bytes (2 × uint for atomicAdd, reinterpreted as float on CPU)
		force_accum_buf = RD.StorageBufferCreate(8, new byte[8]);
		water_info_buf = RD.StorageBufferCreate(8, new byte[8]);
	}

	private Rid LoadComputeShader(string path)
	{
		var shaderFile = GD.Load<RDShaderFile>(path);
		var spirv = shaderFile.GetSpirV();
		return RD.ShaderCreateFromSpirV(spirv);
	}

	private void LoadShaders()
	{
		string basePath = "res://scenes/compute/";

		shader_forces_hash = LoadComputeShader(basePath + "sph_forces_hash.glsl");
		pipeline_forces_hash = RD.ComputePipelineCreate(shader_forces_hash);

		shader_histogram = LoadComputeShader(basePath + "sph_histogram.glsl");
		pipeline_histogram = RD.ComputePipelineCreate(shader_histogram);

		shader_prefix_local = LoadComputeShader(basePath + "sph_prefix_local.glsl");
		pipeline_prefix_local = RD.ComputePipelineCreate(shader_prefix_local);

		shader_prefix_top = LoadComputeShader(basePath + "sph_prefix_top.glsl");
		pipeline_prefix_top = RD.ComputePipelineCreate(shader_prefix_top);

		shader_prefix_apply = LoadComputeShader(basePath + "sph_prefix_apply.glsl");
		pipeline_prefix_apply = RD.ComputePipelineCreate(shader_prefix_apply);

		shader_scatter = LoadComputeShader(basePath + "sph_scatter.glsl");
		pipeline_scatter = RD.ComputePipelineCreate(shader_scatter);

		shader_density = LoadComputeShader(basePath + "sph_density.glsl");
		pipeline_density = RD.ComputePipelineCreate(shader_density);

		shader_pressure_visc = LoadComputeShader(basePath + "sph_pressure_viscosity.glsl");
		pipeline_pressure_visc = RD.ComputePipelineCreate(shader_pressure_visc);

		shader_integrate = LoadComputeShader(basePath + "sph_integrate.glsl");
		pipeline_integrate = RD.ComputePipelineCreate(shader_integrate);

		shader_output = LoadComputeShader(basePath + "sph_output.glsl");
		pipeline_output = RD.ComputePipelineCreate(shader_output);
	}

	private RDUniform MakeStorageUniform(int binding, Rid buffer)
	{
		var u = new RDUniform();
		u.UniformType = RenderingDevice.UniformType.StorageBuffer;
		u.Binding = binding;
		u.AddId(buffer);
		return u;
	}

	private RDUniform MakeUniformUniform(int binding, Rid buffer)
	{
		var u = new RDUniform();
		u.UniformType = RenderingDevice.UniformType.UniformBuffer;
		u.Binding = binding;
		u.AddId(buffer);
		return u;
	}

	private RDUniform MakeImageUniform(int binding, Rid texture)
	{
		var u = new RDUniform();
		u.UniformType = RenderingDevice.UniformType.Image;
		u.Binding = binding;
		u.AddId(texture);
		return u;
	}

	private Rid MakeUniformSet(Rid shader, uint set, Godot.Collections.Array<RDUniform> uniforms)
	{
		return RD.UniformSetCreate(uniforms, shader, set);
	}

	private void CreateUniformSets()
	{
		// Pass 1: Forces+Hash — particle(0), hash(1), params(7)
		uniform_set_forces_hash = MakeUniformSet(shader_forces_hash, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(0, particle_buf),
			MakeStorageUniform(1, hash_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 2: Histogram — hash(1), histogram(4), params(7)
		uniform_set_histogram = MakeUniformSet(shader_histogram, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(1, hash_buf),
			MakeStorageUniform(4, histogram_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 3: Prefix Local — histogram(4), prefix(5), block_sums(6), params(7)
		uniform_set_prefix_local = MakeUniformSet(shader_prefix_local, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(4, histogram_buf),
			MakeStorageUniform(5, prefix_buf),
			MakeStorageUniform(6, block_sums_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 4: Prefix Top — block_sums(6), params(7)
		uniform_set_prefix_top = MakeUniformSet(shader_prefix_top, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(6, block_sums_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 5: Prefix Apply — hashtable(3), histogram(4), prefix(5), block_sums(6), params(7)
		uniform_set_prefix_apply = MakeUniformSet(shader_prefix_apply, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(3, hashtable_buf),
			MakeStorageUniform(4, histogram_buf),
			MakeStorageUniform(5, prefix_buf),
			MakeStorageUniform(6, block_sums_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 6: Scatter — hash(1), sort(2), prefix(5), params(7)
		uniform_set_scatter = MakeUniformSet(shader_scatter, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(1, hash_buf),
			MakeStorageUniform(2, sort_buf),
			MakeStorageUniform(5, prefix_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 7: Density — particle(0), hash(1), sort(2), hashtable(3), params(7)
		uniform_set_density = MakeUniformSet(shader_density, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(0, particle_buf),
			MakeStorageUniform(1, hash_buf),
			MakeStorageUniform(2, sort_buf),
			MakeStorageUniform(3, hashtable_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 8: Pressure+Viscosity — particle(0), hash(1), sort(2), hashtable(3), params(7)
		uniform_set_pressure_visc = MakeUniformSet(shader_pressure_visc, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(0, particle_buf),
			MakeStorageUniform(1, hash_buf),
			MakeStorageUniform(2, sort_buf),
			MakeStorageUniform(3, hashtable_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 9: Integrate — particle(0), force_accum(1), water_info(2), params(7)
		uniform_set_integrate = MakeUniformSet(shader_integrate, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(0, particle_buf),
			MakeStorageUniform(1, force_accum_buf),
			MakeStorageUniform(2, water_info_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 10: Output set=0 — particle(0), sort(2), hashtable(3), params(7)
		uniform_set_output_0 = MakeUniformSet(shader_output, 0, new Godot.Collections.Array<RDUniform> {
			MakeStorageUniform(0, particle_buf),
			MakeStorageUniform(2, sort_buf),
			MakeStorageUniform(3, hashtable_buf),
			MakeUniformUniform(7, params_ubuf),
		});

		// Pass 10: Output set=1 — position_tex(0), hashlookup_tex(1)
		uniform_set_output_1 = MakeUniformSet(shader_output, 1, new Godot.Collections.Array<RDUniform> {
			MakeImageUniform(0, position_tex_rid),
			MakeImageUniform(1, hashlookup_tex_rid),
		});
	}

	public void UpdateParams(float dt, float subDt, float gravity, float velocityDamping,
		float smoothingRadius, float pressureMultiplier, float nearPressureMultiplier,
		float viscosityStrength, float collisionDamping, float predictionFactor, float maxVel,
		Vector2 boundsMin, Vector2 boundsMax,
		bool mousePressed, Vector2 mousePos,
		float mouseRadiusGrab, float mouseRadiusStick, float grabSpeed,
		Vector2 groundOffset, bool sprayMode,
		Vector2 bodyPos, float bodyRadius, bool bodyEnabled,
		Vector2 bodyVel, float forceScale)
	{
		byte[] data = new byte[PARAMS_BUF_SIZE];
		int offset = 0;
		WriteVec2(data, ref offset, boundsMin);       // 0
		WriteVec2(data, ref offset, boundsMax);         // 8
		WriteVec2(data, ref offset, mousePos);         // 16
		WriteFloat(data, ref offset, dt);               // 24
		WriteFloat(data, ref offset, subDt);            // 28
		WriteFloat(data, ref offset, gravity);          // 32
		WriteFloat(data, ref offset, velocityDamping);  // 36
		WriteFloat(data, ref offset, smoothingRadius);  // 40
		WriteFloat(data, ref offset, smoothingRadius * smoothingRadius); // 44
		WriteFloat(data, ref offset, 1f / smoothingRadius); // 48
		WriteFloat(data, ref offset, pressureMultiplier);   // 52
		WriteFloat(data, ref offset, nearPressureMultiplier); // 56
		WriteFloat(data, ref offset, viscosityStrength);     // 60
		WriteFloat(data, ref offset, collisionDamping);      // 64
		WriteFloat(data, ref offset, predictionFactor);      // 68
		WriteFloat(data, ref offset, maxVel);                // 72
		WriteFloat(data, ref offset, mouseRadiusGrab);      // 76
		WriteFloat(data, ref offset, mouseRadiusStick);     // 80
		WriteFloat(data, ref offset, grabSpeed);            // 84
		WriteInt(data, ref offset, N);                      // 88
		WriteInt(data, ref offset, mousePressed ? 1 : 0);   // 92
		WriteVec2(data, ref offset, groundOffset);           // 96
		WriteInt(data, ref offset, sprayMode ? 1 : 0);      // 104
		WriteFloat(data, ref offset, 0f);                    // 108 padding for vec2 alignment
		WriteVec2(data, ref offset, bodyPos);                // 112
		WriteFloat(data, ref offset, bodyRadius);            // 120
		WriteInt(data, ref offset, bodyEnabled ? 1 : 0);    // 124
		WriteVec2(data, ref offset, bodyVel);               // 128
		WriteFloat(data, ref offset, forceScale);            // 136
		WriteFloat(data, ref offset, 0f);                    // 140 padding
		WriteFloat(data, ref offset, 0f);                    // 144 padding
		WriteFloat(data, ref offset, 0f);                    // 148 padding
		WriteFloat(data, ref offset, 0f);                    // 152 padding
		// Total: 160 bytes

		RD.BufferUpdate(params_ubuf, 0, PARAMS_BUF_SIZE, data);
	}

	private static void WriteFloat(byte[] buf, ref int offset, float v)
	{
		byte[] bytes = BitConverter.GetBytes(v);
		buf[offset++] = bytes[0]; buf[offset++] = bytes[1];
		buf[offset++] = bytes[2]; buf[offset++] = bytes[3];
	}

	private static void WriteInt(byte[] buf, ref int offset, int v)
	{
		byte[] bytes = BitConverter.GetBytes(v);
		buf[offset++] = bytes[0]; buf[offset++] = bytes[1];
		buf[offset++] = bytes[2]; buf[offset++] = bytes[3];
	}

	private static void WriteVec2(byte[] buf, ref int offset, Vector2 v)
	{
		WriteFloat(buf, ref offset, v.X);
		WriteFloat(buf, ref offset, v.Y);
	}

	public void DispatchFrame(float frameDt, int iterations)
	{
		for (int iter = 0; iter < iterations; iter++)
		{
			// Clear per-substep buffers (must be outside ComputeList)
			RD.BufferClear(histogram_buf, 0, HISTOGRAM_BUF_SIZE);
			RD.BufferClear(prefix_buf, 0, PREFIX_BUF_SIZE);
			RD.BufferClear(block_sums_buf, 0, BLOCK_SUMS_BUF_SIZE);

			// Clear force accumulator + water info (first substep only — accumulate across substeps)
			if (iter == 0)
			{
				RD.BufferClear(force_accum_buf, 0, 8);
				// Init water_info: both counts = 0
				byte[] wiInit = new byte[8];
				BitConverter.GetBytes((uint)0).CopyTo(wiInit, 0);
				BitConverter.GetBytes((uint)0).CopyTo(wiInit, 4);
				RD.BufferUpdate(water_info_buf, 0, 8, wiInit);
			}

			var cl = RD.ComputeListBegin();

			// Pass 1: Forces + Hash
			RD.ComputeListBindComputePipeline(cl, pipeline_forces_hash);
			RD.ComputeListBindUniformSet(cl, uniform_set_forces_hash, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 2: Histogram
			RD.ComputeListBindComputePipeline(cl, pipeline_histogram);
			RD.ComputeListBindUniformSet(cl, uniform_set_histogram, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 3: Prefix Local
			RD.ComputeListBindComputePipeline(cl, pipeline_prefix_local);
			RD.ComputeListBindUniformSet(cl, uniform_set_prefix_local, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 4: Prefix Top
			RD.ComputeListBindComputePipeline(cl, pipeline_prefix_top);
			RD.ComputeListBindUniformSet(cl, uniform_set_prefix_top, 0);
			RD.ComputeListDispatch(cl, 1, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 5: Prefix Apply + Hash Table
			RD.ComputeListBindComputePipeline(cl, pipeline_prefix_apply);
			RD.ComputeListBindUniformSet(cl, uniform_set_prefix_apply, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 6: Scatter
			RD.ComputeListBindComputePipeline(cl, pipeline_scatter);
			RD.ComputeListBindUniformSet(cl, uniform_set_scatter, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 7: Density
			RD.ComputeListBindComputePipeline(cl, pipeline_density);
			RD.ComputeListBindUniformSet(cl, uniform_set_density, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 8: Pressure + Viscosity
			RD.ComputeListBindComputePipeline(cl, pipeline_pressure_visc);
			RD.ComputeListBindUniformSet(cl, uniform_set_pressure_visc, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);
			RD.ComputeListAddBarrier(cl);

			// Pass 9: Integrate
			RD.ComputeListBindComputePipeline(cl, pipeline_integrate);
			RD.ComputeListBindUniformSet(cl, uniform_set_integrate, 0);
			RD.ComputeListDispatch(cl, NUM_GROUPS, 1, 1);

			RD.ComputeListEnd();
		}

		// Pass 10: Output (once per frame, separate compute list)
		{
			var cl = RD.ComputeListBegin();
			RD.ComputeListBindComputePipeline(cl, pipeline_output);
			RD.ComputeListBindUniformSet(cl, uniform_set_output_0, 0);
			RD.ComputeListBindUniformSet(cl, uniform_set_output_1, 1);
			RD.ComputeListDispatch(cl, OUTPUT_GROUPS, 1, 1);
			RD.ComputeListEnd();
		}
	}

	public void ResetParticles(Vector2[] positions, Vector2[] velocities, Vector2 groundOffset)
	{
		// Upload pos[N] + vel[N]: offset 0 = pos, offset N*8 = vel
		byte[] posVelData = new byte[N * 16]; // pos(N*8) + vel(N*8)
		for (int i = 0; i < N; i++)
		{
			int off = i * 8;
			byte[] xb = BitConverter.GetBytes(positions[i].X);
			byte[] yb = BitConverter.GetBytes(positions[i].Y);
			posVelData[off] = xb[0]; posVelData[off + 1] = xb[1]; posVelData[off + 2] = xb[2]; posVelData[off + 3] = xb[3];
			posVelData[off + 4] = yb[0]; posVelData[off + 5] = yb[1]; posVelData[off + 6] = yb[2]; posVelData[off + 7] = yb[3];

			int voff = N * 8 + i * 8;
			byte[] vxb = BitConverter.GetBytes(velocities[i].X);
			byte[] vyb = BitConverter.GetBytes(velocities[i].Y);
			posVelData[voff] = vxb[0]; posVelData[voff + 1] = vxb[1]; posVelData[voff + 2] = vxb[2]; posVelData[voff + 3] = vxb[3];
			posVelData[voff + 4] = vyb[0]; posVelData[voff + 5] = vyb[1]; posVelData[voff + 6] = vyb[2]; posVelData[voff + 7] = vyb[3];
		}
		RD.BufferUpdate(particle_buf, 0, N * 16, posVelData);

		// Zero out predicted_pos + density + near_density
		RD.BufferClear(particle_buf, N * 16, N * 8);  // predicted_pos
		RD.BufferClear(particle_buf, N * 24, N * 8);    // density + near_density
	}

	public byte[] ReadBackParticleBuffer()
	{
		return RD.BufferGetData(particle_buf, 0, PARTICLE_BUF_SIZE);
	}

	public Vector2 ReadBackForce()
	{
		byte[] data = RD.BufferGetData(force_accum_buf, 0, 8);
		int ix = BitConverter.ToInt32(data, 0);
		int iy = BitConverter.ToInt32(data, 4);
		return new Vector2((float)ix, (float)iy);
	}

	// Read back water info: (count_below_center, count_above_center)
	public (uint countBelow, uint countAbove) ReadBackWaterInfo()
	{
		byte[] data = RD.BufferGetData(water_info_buf, 0, 8);
		uint countBelow = BitConverter.ToUInt32(data, 0);
		uint countAbove = BitConverter.ToUInt32(data, 4);
		return (countBelow, countAbove);
	}

	// 上传单个粒子的pos和vel到GPU buffer
	public void UpdateParticle(int index, Vector2 pos, Vector2 vel)
	{
		uint off = (uint)(index * 8);
		uint voff = (uint)(N * 8 + index * 8);
		byte[] posData = new byte[8];
		byte[] xb = BitConverter.GetBytes(pos.X);
		byte[] yb = BitConverter.GetBytes(pos.Y);
		posData[0] = xb[0]; posData[1] = xb[1]; posData[2] = xb[2]; posData[3] = xb[3];
		posData[4] = yb[0]; posData[5] = yb[1]; posData[6] = yb[2]; posData[7] = yb[3];
		RD.BufferUpdate(particle_buf, off, 8, posData);

		byte[] velData = new byte[8];
		byte[] vxb = BitConverter.GetBytes(vel.X);
		byte[] vyb = BitConverter.GetBytes(vel.Y);
		velData[0] = vxb[0]; velData[1] = vxb[1]; velData[2] = vxb[2]; velData[3] = vxb[3];
		velData[4] = vyb[0]; velData[5] = vyb[1]; velData[6] = vyb[2]; velData[7] = vyb[3];
		RD.BufferUpdate(particle_buf, voff, 8, velData);
	}

	// 批量上传粒子pos+vel到GPU buffer（减少BufferUpdate调用次数）
	public void UpdateParticlesBatch(int startIndex, Vector2[] positions, Vector2[] velocities, int count)
	{
		byte[] posData = new byte[count * 8];
		byte[] velData = new byte[count * 8];
		for (int i = 0; i < count; i++)
		{
			int idx = startIndex + i;
			int off = i * 8;
			byte[] xb = BitConverter.GetBytes(positions[idx].X);
			byte[] yb = BitConverter.GetBytes(positions[idx].Y);
			posData[off + 0] = xb[0]; posData[off + 1] = xb[1]; posData[off + 2] = xb[2]; posData[off + 3] = xb[3];
			posData[off + 4] = yb[0]; posData[off + 5] = yb[1]; posData[off + 6] = yb[2]; posData[off + 7] = yb[3];

			byte[] vxb = BitConverter.GetBytes(velocities[idx].X);
			byte[] vyb = BitConverter.GetBytes(velocities[idx].Y);
			velData[off + 0] = vxb[0]; velData[off + 1] = vxb[1]; velData[off + 2] = vxb[2]; velData[off + 3] = vxb[3];
			velData[off + 4] = vyb[0]; velData[off + 5] = vyb[1]; velData[off + 6] = vyb[2]; velData[off + 7] = vyb[3];
		}
		RD.BufferUpdate(particle_buf, (uint)(startIndex * 8), (uint)posData.Length, posData);
		RD.BufferUpdate(particle_buf, (uint)(N * 8 + startIndex * 8), (uint)velData.Length, velData);
	}

	public void Free()
	{
		RD.FreeRid(position_tex_rid);
		RD.FreeRid(hashlookup_tex_rid);
		RD.FreeRid(force_accum_buf);
		RD.FreeRid(water_info_buf);
		RD.FreeRid(particle_buf);
		RD.FreeRid(hash_buf);
		RD.FreeRid(sort_buf);
		RD.FreeRid(hashtable_buf);
		RD.FreeRid(histogram_buf);
		RD.FreeRid(prefix_buf);
		RD.FreeRid(block_sums_buf);
		RD.FreeRid(params_ubuf);
		RD.FreeRid(shader_forces_hash);
		RD.FreeRid(shader_histogram);
		RD.FreeRid(shader_prefix_local);
		RD.FreeRid(shader_prefix_top);
		RD.FreeRid(shader_prefix_apply);
		RD.FreeRid(shader_scatter);
		RD.FreeRid(shader_density);
		RD.FreeRid(shader_pressure_visc);
		RD.FreeRid(shader_integrate);
		RD.FreeRid(shader_output);
	}
}
