#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;

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
    float sdf_data[];
};

layout(set = 0, binding = 5, std430) buffer ForceAccum {
    int force_accum[3];  // [0]=fx, [1]=fy, [2]=torque
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
    float boundary_volume;
    vec2 body_pos;
    vec2 sdf_half_extents;
    int body_enabled;
    float boundary_pressure_scale;
    vec2 body_vel;
    float body_angle;
    float fluid_particle_mass;
    float sdf_shape_radius;
    float target_density;
};

float spiky_pow2_deriv_scale;
float spiky_pow3_deriv_scale;
float poly6_scale;

void init_kernel_scales() {
    float r = smoothing_radius;
    spiky_pow2_deriv_scale = 12.0 / (3.14159265 * r * r * r * r);
    spiky_pow3_deriv_scale = 30.0 / (3.14159265 * r * r * r * r * r);
    poly6_scale = 4.0 / (3.14159265 * r * r * r * r * r * r * r * r);
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
    float v = smoothing_radius * smoothing_radius - dst * dst;
    return v * v * v * poly6_scale;
}

vec2 rotate2d(vec2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

float sampleSdf(vec2 localPos) {
    float ux = (localPos.x / sdf_half_extents.x + 1.0) * 0.5;
    float uy = (localPos.y / sdf_half_extents.y + 1.0) * 0.5;
    const int SDF_SIZE = 64;
    int ix = int(clamp(ux * float(SDF_SIZE), 0.0, float(SDF_SIZE - 1)));
    int iy = int(clamp(uy * float(SDF_SIZE), 0.0, float(SDF_SIZE - 1)));
    return sdf_data[iy * SDF_SIZE + ix];
}

vec2 sampleSdfNormal(vec2 localPos) {
    float epsX = 2.0 * sdf_half_extents.x / 64.0;
    float epsY = 2.0 * sdf_half_extents.y / 64.0;
    float dx = (sampleSdf(localPos + vec2(epsX, 0.0)) - sampleSdf(localPos - vec2(epsX, 0.0))) / (2.0 * epsX);
    float dy = (sampleSdf(localPos + vec2(0.0, epsY)) - sampleSdf(localPos - vec2(0.0, epsY))) / (2.0 * epsY);
    vec2 grad = vec2(dx, dy);
    float len = length(grad);
    return len > 0.0001 ? grad / len : vec2(0.0, -1.0);
}

float pressure_from_density(float density) {
    return (density - target_density) * pressure_multiplier;
}

float near_pressure_from_density(float near_density) {
    return near_pressure_multiplier * near_density;
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

    int idxi = int(idx);
    int pi = pred_offset(idxi);
    vec2 my_pos = vec2(particle_data[pi], particle_data[pi + 1]);

    // Inactive particle: skip force computation
    if (my_pos.y < -500.0) return;

    float my_density = particle_data[density_offset(idxi)];
    float my_near_density = particle_data[near_density_offset(idxi)];
    float my_pressure = pressure_from_density(my_density);
    float my_near_pressure = near_pressure_from_density(my_near_density);

    int vi = vel_offset(idxi);
    vec2 my_vel = vec2(particle_data[vi], particle_data[vi + 1]);

    vec2 pressure_force = vec2(0.0);
    vec2 viscosity_force = vec2(0.0);

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
                if (int(neighbor_idx) == idxi) continue;

                int pn = pred_offset(int(neighbor_idx));
                vec2 other_pos = vec2(particle_data[pn], particle_data[pn + 1]);
                vec2 diff = other_pos - my_pos;
                float dist_sq = dot(diff, diff);
                if (dist_sq >= sr_sq || dist_sq < 0.000001) continue;

                float dist = sqrt(dist_sq);
                vec2 dir = diff / dist;

                float nb_density = particle_data[density_offset(int(neighbor_idx))];
                float nb_near_density = particle_data[near_density_offset(int(neighbor_idx))];
                float nb_pressure = pressure_from_density(nb_density);
                float nb_near_pressure = near_pressure_from_density(nb_near_density);

                float shared_pressure = (my_pressure + nb_pressure) * 0.5;
                float shared_near_pressure = (my_near_pressure + nb_near_pressure) * 0.5;

                pressure_force += dir * spiky_pow2_derivative(dist) * shared_pressure / nb_density;
                pressure_force += dir * spiky_pow3_derivative(dist) * shared_near_pressure / nb_near_density;

                int vn = vel_offset(int(neighbor_idx));
                vec2 other_vel = vec2(particle_data[vn], particle_data[vn + 1]);
                viscosity_force += (other_vel - my_vel) * poly6_kernel(dist);
            }
        }
    }

    vec2 acceleration = pressure_force / my_density;
    particle_data[vi] += acceleration.x * sub_dt + viscosity_force.x * viscosity_strength * sub_dt;
    particle_data[vi + 1] += acceleration.y * sub_dt + viscosity_force.y * viscosity_strength * sub_dt;

    // Boundary pressure force via SDF (Akinci 2012: Versatile Rigid-Fluid Coupling)
    // The body boundary participates as "virtual boundary particles".
    // Pressure force on fluid from boundary, with equal-and-opposite reaction on body.
    if (body_enabled != 0) {
        vec2 localPos = rotate2d(my_pos - body_pos, -body_angle);
        if (abs(localPos.x) < sdf_half_extents.x && abs(localPos.y) < sdf_half_extents.y) {
            float d = sampleSdf(localPos);
            // Apply within kernel radius of surface (both outside AND inside body)
            if (d > -smoothing_radius && d < smoothing_radius) {
                vec2 localNormal = sampleSdfNormal(localPos);
                vec2 sdf_normal = rotate2d(localNormal, body_angle);  // world-space outward normal

                // Direct pressure-based boundary force (simpler and stronger than kernel gradient)
                // Force = pressure * boundary_pressure_scale in surface normal direction.
                // Only positive pressure (compression) pushes; negative pressure (tension) ignored.
                float press = max(my_pressure, 0.0);
                float force_magnitude = press * boundary_pressure_scale;
                vec2 force_on_fluid = sdf_normal * force_magnitude;

                // Apply to fluid velocity (acceleration = force / density)
                particle_data[vi] += force_on_fluid.x / my_density * sub_dt;
                particle_data[vi + 1] += force_on_fluid.y / my_density * sub_dt;

                // Newton's 3rd law: equal-and-opposite reaction on rigid body
                // Contact point is on body surface: xb = fluid_pos - sdf_normal * d
                vec2 contact_point = my_pos - sdf_normal * d;
                vec2 reaction_force = -force_on_fluid * fluid_particle_mass;
                atomicAdd(force_accum[0], int(reaction_force.x));
                atomicAdd(force_accum[1], int(reaction_force.y));
                // Torque = (contact_point - body_COM) x reaction_force
                vec2 arm = contact_point - body_pos;
                atomicAdd(force_accum[2], int(arm.x * reaction_force.y - arm.y * reaction_force.x));
            }
        }
    }
}
