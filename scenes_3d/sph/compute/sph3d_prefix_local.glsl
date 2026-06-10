#[compute]
#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 4, std430) readonly buffer HistogramBuffer {
    uint histogram[];
};

layout(set = 0, binding = 5, std430) buffer PrefixBuffer {
    uint prefix[];
};

layout(set = 0, binding = 6, std430) buffer BlockSumsBuffer {
    uint block_sums[];
};

shared uint shared_data[256];

layout(set = 0, binding = 7, std140) uniform Params {
    vec4 bounds_min;
    vec4 bounds_max;
    float dt;
    float sub_dt;
    float gravity;
    float velocity_damping;
    float smoothing_radius;
    float smoothing_radius_sq;
    float grid_cell_inverse;
    float pressure_multiplier;
    float near_pressure_multiplier;
    float viscosity_strength;
    float collision_damping;
    float prediction_factor;
    float max_vel;
    float target_density;
    int particle_count;
    float _pad0;
    vec4 player_data;
};

void main() {
    uint g = gl_WorkGroupID.x;
    uint lane = gl_LocalInvocationID.x;
    uint chunk_start = g * 256u;

    uint val = 0u;
    if (chunk_start + lane < uint(particle_count)) {
        val = histogram[chunk_start + lane];
    }
    shared_data[lane] = val;
    barrier();

    for (uint d = 0; d < 8; d++) {
        uint stride = 1u << d;
        uint idx = (lane + 1) * stride * 2u - 1u;
        if (idx < 256u) {
            shared_data[idx] += shared_data[idx - stride];
        }
        barrier();
    }

    if (lane == 0u) {
        shared_data[255] = 0u;
    }
    barrier();

    for (uint d = 8; d > 0; d--) {
        uint stride = 1u << (d - 1u);
        uint idx = (lane + 1) * stride * 2u - 1u;
        if (idx < 256u) {
            uint tmp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += tmp;
        }
        barrier();
    }

    if (chunk_start + lane < uint(particle_count)) {
        prefix[chunk_start + lane] = shared_data[lane];
    }

    if (lane == 0u) {
        uint last_idx = min(255u, uint(particle_count) - chunk_start - 1u);
        uint block_total = shared_data[last_idx];
        if (chunk_start + last_idx < uint(particle_count)) {
            block_total += histogram[chunk_start + last_idx];
        }
        block_sums[g] = block_total;
    }
}
