#[compute]
#version 450

layout(local_size_x = 256) in;

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
};

int vel_offset(int i) { return particle_count * 2 + i * 2; }

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(particle_count)) return;

    int i = int(idx);

    vec2 p = vec2(particle_data[i * 2], particle_data[i * 2 + 1]);

    // Inactive particle: don't clamp to bounds
    if (p.y < -500.0) return;

    int vi = vel_offset(i);
    vec2 v = vec2(particle_data[vi], particle_data[vi + 1]);

    p += v * sub_dt;

    if (p.x < bounds_min.x) {
        p.x = bounds_min.x;
        v.x *= -collision_damping;
    } else if (p.x > bounds_max.x) {
        p.x = bounds_max.x;
        v.x *= -collision_damping;
    }

    if (p.y < bounds_min.y) {
        p.y = bounds_min.y;
        v.y *= -collision_damping;
    } else if (p.y > bounds_max.y) {
        p.y = bounds_max.y;
        v.y *= -collision_damping;
    }

    particle_data[i * 2] = p.x;
    particle_data[i * 2 + 1] = p.y;
    particle_data[vi] = v.x;
    particle_data[vi + 1] = v.y;
}
