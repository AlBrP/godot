#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;

layout(set = 0, binding = 3, std430) buffer HashTableBuffer {
    int ht_start[N];
    int ht_end[];
};

layout(set = 0, binding = 4, std430) readonly buffer HistogramBuffer {
    uint histogram[];
};

layout(set = 0, binding = 5, std430) buffer PrefixBuffer {
    uint prefix[];
};

layout(set = 0, binding = 6, std430) readonly buffer BlockSumsBuffer {
    uint block_sums[];
};

layout(set = 0, binding = 7, std140) uniform Params {
    vec2 bounds_min;
    vec2 bounds_max;
    vec2 mouse_pos;
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
    float mouse_radius_grab;
    float mouse_radius_stick;
    float grab_speed;
    int particle_count;
    int mouse_pressed;
    vec2 ground_offset;
    int spray_mode;
    float _pad_sp0;
    float _pad_sp1;
    float _pad_sp2;
    float _pad_sp3;
    float _pad_sp4;
    int body_count;
    float _pad_sp5;
    float _pad_sp6;
    float _pad_sp7;
    float _pad_sp8;
    float fluid_particle_mass;
    float _pad_sp9;
    float target_density;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(particle_count)) return;

    uint block_idx = i / 256u;
    uint complete_prefix = prefix[i] + block_sums[block_idx];
    prefix[i] = complete_prefix;

    uint count = histogram[i];
    if (count > 0u) {
        ht_start[i] = int(complete_prefix);
        ht_end[i] = int(complete_prefix + count - 1u);
    } else {
        ht_start[i] = -1;
        ht_end[i] = -1;
    }
}
