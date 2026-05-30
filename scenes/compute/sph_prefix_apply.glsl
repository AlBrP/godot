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
    vec4 ptype_stiffness;
    vec4 ptype_buoyancy;
    vec4 ptype_viscosity;
    vec4 ptype_vorticity;
    vec4 ptype_diffusion;
    vec4 ptype_cooling;
    vec4 ptype_init_temp;
    vec4 ptype_lifetime;
    vec4 ptype_near_pressure_scale;
    vec4 ptype_boil_point;
    vec4 ptype_boil_product;
    vec4 ptype_condense_point;
    vec4 ptype_condense_product;
    float boundary_friction;
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
