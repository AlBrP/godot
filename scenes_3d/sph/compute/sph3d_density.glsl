#[compute]
#version 450

// Pass 7: density + near-density accumulation via 27-cell neighbour lookup.
// 3D Spiky kernel family (Müller 2003-style), normalised over the sphere.
//   W_spiky2 = (15 / (2π h^5)) (h - r)^2          -- "spiky_pow2", density
//   W_spiky3 = (15 /     (π h^6)) (h - r)^3       -- "spiky_pow3", near density
// (Coefficient scales picked so volume integrals over the sphere of radius h
//  equal 1. See Müller / Becker-Teschner SPH references.)

layout(local_size_x = 256) in;

const int N = 20000;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 1, std430) readonly buffer HashBuffer {
    uint hash_values[N];
    ivec3 grid_cell_coord[];
};

layout(set = 0, binding = 2, std430) readonly buffer SortBuffer {
    uint sorted_indices[N];
    uint sorted_hashes[];
};

layout(set = 0, binding = 3, std430) readonly buffer HashTableBuffer {
    int ht_start[N];
    int ht_end[];
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

float spiky_pow2_scale;
float spiky_pow3_scale;

void init_kernel_scales() {
    float r = smoothing_radius;
    float r5 = r * r * r * r * r;
    float r6 = r5 * r;
    spiky_pow2_scale = 15.0 / (2.0 * 3.14159265 * r5);
    spiky_pow3_scale = 15.0 / (3.14159265 * r6);
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

int pred_offset(int i) { return particle_count * 6 + i * 3; }
int density_offset(int i) { return particle_count * 9 + i; }
int near_density_offset(int i) { return particle_count * 10 + i; }

const int HASH_K1 = 15823;
const int HASH_K2 = 9737333;
const int HASH_K3 = 440817757;

uint hash_cell(ivec3 cell) {
    uint cx = uint(cell.x);
    uint cy = uint(cell.y);
    uint cz = uint(cell.z);
    return (cx * uint(HASH_K1) + cy * uint(HASH_K2) + cz * uint(HASH_K3)) % uint(particle_count);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(particle_count)) return;

    init_kernel_scales();

    int pi = pred_offset(int(idx));
    vec3 my_pos = vec3(particle_data[pi], particle_data[pi + 1], particle_data[pi + 2]);

    float density = 0.0;
    float near_density = 0.0;

    ivec3 my_cell = grid_cell_coord[idx];
    float sr_sq = smoothing_radius_sq;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                ivec3 ncell = my_cell + ivec3(dx, dy, dz);
                uint h = hash_cell(ncell);
                int start = ht_start[h];
                int end = ht_end[h];
                if (start < 0) continue;
                for (int j = start; j <= end; j++) {
                    uint neighbor_idx = sorted_indices[j];
                    int pn = pred_offset(int(neighbor_idx));
                    vec3 other_pos = vec3(particle_data[pn], particle_data[pn + 1], particle_data[pn + 2]);
                    vec3 diff = other_pos - my_pos;
                    float dist_sq = dot(diff, diff);
                    if (dist_sq < sr_sq) {
                        float dist = sqrt(dist_sq);
                        density += spiky_pow2(dist);
                        near_density += spiky_pow3(dist);
                    }
                }
            }
        }
    }

    particle_data[density_offset(int(idx))] = density;
    particle_data[near_density_offset(int(idx))] = near_density;
}
