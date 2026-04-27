#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;

layout(set = 0, binding = 1, std430) readonly buffer HashBuffer {
    uint hash_values[N];
};

layout(set = 0, binding = 4, std430) buffer HistogramBuffer {
    uint histogram[];
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
    int sim_mode;
    float gas_stiffness;
    float buoyancy_alpha;
    float vorticity_epsilon;
    float temp_diffusion_rate;
    int body_count;
    float particle_lifetime;
    float ambient_temperature;
    float cooling_rate;
    float gas_viscosity_ratio;
    float fluid_particle_mass;
    float body_drag_gas;
    float target_density;
};

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(particle_count)) return;

    atomicAdd(histogram[hash_values[i]], 1u);
}
