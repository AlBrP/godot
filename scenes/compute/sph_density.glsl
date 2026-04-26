#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;
const int MAX_BODIES = 4;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 1, std430) readonly buffer HashBuffer {
    uint hash_values[N];
    ivec2 grid_cell_coord[];
};

layout(set = 0, binding = 2, std430) readonly buffer SortBuffer {
    uint sorted_indices[N];
    uint sorted_hashes[];
};

layout(set = 0, binding = 3, std430) readonly buffer HashTableBuffer {
    int ht_start[N];
    int ht_end[];
};

layout(set = 0, binding = 4, std430) readonly buffer SdfBuffer {
    float sdf_data[];          // MAX_BODIES * 64 * 64 SDF values
};

struct Body {
    vec2 pos;
    vec2 vel;
    vec2 sdf_half_extents;
    float angle;
    float shape_radius;
    float boundary_volume;
    float bp_scale;
    int enabled;
    float _pad;
};

layout(set = 0, binding = 8, std430) readonly buffer BodyDataBuffer {
    Body bodies[MAX_BODIES];
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
    float _pad9;
    float target_density;
};

float spiky_pow2_scale;
float spiky_pow3_scale;

void init_kernel_scales() {
    float r = smoothing_radius;
    spiky_pow2_scale = 6.0 / (3.14159265 * r * r * r * r);
    spiky_pow3_scale = 10.0 / (3.14159265 * r * r * r * r * r);
}

float spiky_pow2(float dst) {
    if (dst >= smoothing_radius) return 0.0;
    float v = smoothing_radius - dst;
    return v * v * spiky_pow2_scale;
}

float spiky_pow3(float dst) {
    if (dst >= smoothing_radius) return 0.0;
    float v = smoothing_radius - dst;
    return v * v * v * spiky_pow3_scale;
}

vec2 rotate2d(vec2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

float sampleSdf(vec2 localPos, int bodyIdx) {
    float ux = (localPos.x / bodies[bodyIdx].sdf_half_extents.x + 1.0) * 0.5;
    float uy = (localPos.y / bodies[bodyIdx].sdf_half_extents.y + 1.0) * 0.5;
    int ix = int(clamp(ux * 64.0, 0.0, 63.0));
    int iy = int(clamp(uy * 64.0, 0.0, 63.0));
    return sdf_data[bodyIdx * 4096 + iy * 64 + ix];
}

int vel_offset(int i) { return particle_count * 2 + i * 2; }
int pred_offset(int i) { return particle_count * 4 + i * 2; }
int density_offset(int i) { return particle_count * 6 + i; }
int near_density_offset(int i) { return particle_count * 7 + i; }

const int HASH_K1 = 15823;
const int HASH_K2 = 9737333;

uint hash_cell(ivec2 cell) {
    uint cx = uint(cell.x);
    uint cy = uint(cell.y);
    uint a = cx * uint(HASH_K1);
    uint b = cy * uint(HASH_K2);
    return (a + b) % uint(particle_count);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(particle_count)) return;

    init_kernel_scales();

    int pi = pred_offset(int(idx));
    vec2 my_pos = vec2(particle_data[pi], particle_data[pi + 1]);

    if (my_pos.y < -500.0) {
        particle_data[density_offset(int(idx))] = 0.0;
        particle_data[near_density_offset(int(idx))] = 0.0;
        return;
    }

    float density = 0.0;
    float near_density = 0.0;

    ivec2 my_cell = grid_cell_coord[idx];
    float sr_sq = smoothing_radius_sq;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ivec2 neighbor_cell = my_cell + ivec2(dx, dy);
            uint h = hash_cell(neighbor_cell);
            int start = ht_start[h];
            int end = ht_end[h];
            if (start < 0) continue;
            for (int j = start; j <= end; j++) {
                uint neighbor_idx = sorted_indices[j];
                int pn = pred_offset(int(neighbor_idx));
                vec2 other_pos = vec2(particle_data[pn], particle_data[pn + 1]);
                vec2 diff = other_pos - my_pos;
                float dist_sq = dot(diff, diff);
                if (dist_sq < sr_sq) {
                    float dist = sqrt(dist_sq);
                    density += spiky_pow2(dist);
                    near_density += spiky_pow3(dist);
                }
            }
        }
    }

    // Boundary density contribution from all active bodies
    for (int b = 0; b < body_count; b++) {
        if (bodies[b].enabled == 0) continue;
        vec2 localPos = rotate2d(my_pos - bodies[b].pos, -bodies[b].angle);
        if (abs(localPos.x) >= bodies[b].sdf_half_extents.x || abs(localPos.y) >= bodies[b].sdf_half_extents.y) continue;

        float d = sampleSdf(localPos, b);
        if (d > -smoothing_radius && d < smoothing_radius) {
            float dist = abs(d);
            float bv = target_density * bodies[b].boundary_volume;
            density += bv * spiky_pow2(dist);
            near_density += bv * spiky_pow3(dist);
        }
    }

    particle_data[density_offset(int(idx))] = density;
    particle_data[near_density_offset(int(idx))] = near_density;
}
