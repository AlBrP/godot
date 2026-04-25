#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
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

int vel_offset(int i) { return particle_count * 2 + i * 2; }

vec2 rotate2d(vec2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

// These duplicate what's in other shaders because integrate has no SDF buffer binding.
// We do a cheap sphere-based push-out as safety net, avoiding complex SDF sampling here.
// The primary force comes from SPH pressure in density+pressure_visc passes.

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(particle_count)) return;

    int i = int(idx);

    vec2 p = vec2(particle_data[i * 2], particle_data[i * 2 + 1]);
    if (p.y < -500.0) return;

    int vi = vel_offset(i);
    vec2 v = vec2(particle_data[vi], particle_data[vi + 1]);

    // Simple Euler integration
    p += v * sub_dt;

    // Light SDF safety push-out for penetrated particles
    // (Primary force is SPH boundary pressure in earlier passes; this catches rare penetrations)
    if (body_enabled != 0) {
        vec2 diff = p - body_pos;
        float dist = length(diff);
        if (dist < sdf_shape_radius && dist > 0.0001) {
            // Approximate push-out using sphere (SDF unavailable in this pass)
            float penetration = sdf_shape_radius - dist;
            vec2 normal = diff / dist;
            p += normal * penetration;
            // Reflect inward velocity component
            vec2 rel_v = v - body_vel;
            float vn = dot(rel_v, normal);
            if (vn < 0.0) {
                v = body_vel + reflect(rel_v, normal) * collision_damping;
            }
        }
    }

    // Boundary walls (4 sides)
    if (p.x < bounds_min.x) { p.x = bounds_min.x; v.x *= -collision_damping; }
    else if (p.x > bounds_max.x) { p.x = bounds_max.x; v.x *= -collision_damping; }

    if (p.y < bounds_min.y) { p.y = bounds_min.y; v.y *= -collision_damping; }
    else if (p.y > bounds_max.y) { p.y = bounds_max.y; v.y *= -collision_damping; }

    particle_data[i * 2] = p.x;
    particle_data[i * 2 + 1] = p.y;
    particle_data[vi] = v.x;
    particle_data[vi + 1] = v.y;
}
