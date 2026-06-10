#[compute]
#version 450

// Pass 8: pressure + viscosity forces from 27-cell neighbour lookup.
//   pressure force:   -m_j (p_i + p_j) / (2 rho_j) * grad W_spiky
//   near pressure:    same with near-pressure kernel (cohesion + anti-cluster)
//   viscosity force:  (v_j - v_i) * Poly6 weight, scaled by viscosity_strength
//
// 3D kernel gradients (radial scalar derivative, negative since dW/dr < 0):
//   grad W_spiky2 = -(15 /     (π h^5)) (h - r)        * r_hat
//   grad W_spiky3 = -(45 /     (π h^6)) (h - r)^2      * r_hat
//   W_poly6      =  (315 / (64 π h^9)) (h^2 - r^2)^3

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

float spiky_pow2_deriv_scale;
float spiky_pow3_deriv_scale;
float poly6_scale;

void init_kernel_scales() {
    float r = smoothing_radius;
    float r5 = r * r * r * r * r;
    float r6 = r5 * r;
    float r9 = r6 * r * r * r;
    spiky_pow2_deriv_scale = 15.0 / (3.14159265 * r5);
    spiky_pow3_deriv_scale = 45.0 / (3.14159265 * r6);
    poly6_scale = 315.0 / (64.0 * 3.14159265 * r9);
}

float spiky_pow2_derivative(float dst) {
    if (dst >= smoothing_radius) return 0.0;
    float v = smoothing_radius - dst;
    return -v * spiky_pow2_deriv_scale;
}

float spiky_pow3_derivative(float dst) {
    if (dst >= smoothing_radius) return 0.0;
    float v = smoothing_radius - dst;
    return -v * v * spiky_pow3_deriv_scale;
}

float poly6_kernel(float dst) {
    if (dst >= smoothing_radius) return 0.0;
    float v = smoothing_radius_sq - dst * dst;
    return v * v * v * poly6_scale;
}

	float viscosity_kernel(float dst) {
	    if (dst >= smoothing_radius) return 0.0;
	    return smoothing_radius - dst;
	}

int vel_offset(int i) { return particle_count * 3 + i * 3; }
int pred_offset(int i) { return particle_count * 6 + i * 3; }
int density_offset(int i) { return particle_count * 9 + i; }
int near_density_offset(int i) { return particle_count * 10 + i; }

float pressure_from_density(float density) {
    float ratio = density / target_density;
    float ratio2 = ratio * ratio;
    float ratio4 = ratio2 * ratio2;
    float ratio5 = ratio4 * ratio;
    return pressure_multiplier * (ratio5 - 1.0);
}

float near_pressure_from_density(float near_density) {
    return near_pressure_multiplier * near_density;
}

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

    int ii = int(idx);
    int pi = pred_offset(ii);
    vec3 my_pos = vec3(particle_data[pi], particle_data[pi + 1], particle_data[pi + 2]);

    float my_density = particle_data[density_offset(ii)];
    float my_near_density = particle_data[near_density_offset(ii)];

    int vi = vel_offset(ii);
    vec3 my_vel = vec3(particle_data[vi], particle_data[vi + 1], particle_data[vi + 2]);

    float my_pressure = pressure_from_density(my_density);
    float my_near_pressure = near_pressure_from_density(my_near_density);

    vec3 pressure_force = vec3(0.0);
    vec3 viscosity_force = vec3(0.0);

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
                    if (int(neighbor_idx) == ii) continue;
                    int pn = pred_offset(int(neighbor_idx));
                    vec3 other_pos = vec3(particle_data[pn], particle_data[pn + 1], particle_data[pn + 2]);
                    vec3 diff = other_pos - my_pos;
                    float dist_sq = dot(diff, diff);
                    if (dist_sq >= sr_sq || dist_sq < 0.000001) continue;
                    float dist = sqrt(dist_sq);
                    vec3 dir = diff / dist;

                    float nb_density = particle_data[density_offset(int(neighbor_idx))];
                    float nb_near_density = particle_data[near_density_offset(int(neighbor_idx))];
                    float nb_pressure = pressure_from_density(nb_density);
                    float nb_near_pressure = near_pressure_from_density(nb_near_density);
                    float shared_pressure = (my_pressure + nb_pressure) * 0.5;
                    float shared_near_pressure = (my_near_pressure + nb_near_pressure) * 0.5;

                    pressure_force += dir * spiky_pow2_derivative(dist) * shared_pressure / max(nb_density, 0.0001);
                    pressure_force += dir * spiky_pow3_derivative(dist) * shared_near_pressure / max(nb_density, 0.0001);

                    int vn = vel_offset(int(neighbor_idx));
                    vec3 other_vel = vec3(particle_data[vn], particle_data[vn + 1], particle_data[vn + 2]);
                    viscosity_force += (other_vel - my_vel) * viscosity_kernel(dist);
                }
            }
        }
    }

    vec3 acceleration = pressure_force / max(my_density, 0.0001);

    particle_data[vi]     += acceleration.x * sub_dt + viscosity_force.x * viscosity_strength * sub_dt;
    particle_data[vi + 1] += acceleration.y * sub_dt + viscosity_force.y * viscosity_strength * sub_dt;
    particle_data[vi + 2] += acceleration.z * sub_dt + viscosity_force.z * viscosity_strength * sub_dt;
}
