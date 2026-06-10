#[compute]
#version 450

layout(local_size_x = 256) in;

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
    uint lane = gl_LocalInvocationID.x;
    uint num_blocks = (uint(particle_count) + 255u) / 256u;

    uint val = 0u;
    if (lane < num_blocks) {
        val = block_sums[lane];
    }
    shared_data[lane] = val;
    barrier();

    for (uint d = 0; d < 8; d++) {
        uint stride = 1u << d;
        uint idx = (lane + 1u) * stride * 2u - 1u;
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
        uint idx = (lane + 1u) * stride * 2u - 1u;
        if (idx < 256u) {
            uint tmp = shared_data[idx - stride];
            shared_data[idx - stride] = shared_data[idx];
            shared_data[idx] += tmp;
        }
        barrier();
    }

    if (lane < num_blocks) {
        block_sums[lane] = shared_data[lane];
    }
}
