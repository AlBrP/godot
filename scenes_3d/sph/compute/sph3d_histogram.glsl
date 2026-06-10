#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 20000;

layout(set = 0, binding = 1, std430) readonly buffer HashBuffer {
    uint hash_values[N];
    ivec3 grid_cell_coord[];   // std430 ivec3 stride = 16
};

layout(set = 0, binding = 4, std430) buffer HistogramBuffer {
    uint histogram[];
};

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
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(particle_count)) return;

    atomicAdd(histogram[hash_values[i]], 1u);
}
