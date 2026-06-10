#[compute]
#version 450

// Pass 9: integrate position from velocity, clamp velocity, reflect off
// container walls. Six axis-aligned planes from bounds_min / bounds_max.

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
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

int vel_offset(int i) { return particle_count * 3 + i * 3; }

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(particle_count)) return;

    int i = int(idx);
    int pi_pos = i * 3;
    int vi = vel_offset(i);

    vec3 p = vec3(particle_data[pi_pos], particle_data[pi_pos + 1], particle_data[pi_pos + 2]);
    vec3 v = vec3(particle_data[vi], particle_data[vi + 1], particle_data[vi + 2]);

    // Boundary repulsion: push particles away from walls before they touch.
    // Prevents the "candied hawthorn" wall-climb from neighbor pressure.
    float wall_margin = smoothing_radius * 0.15;
    float wall_push = 8.0 * sub_dt;
    if (p.x < bounds_min.x + wall_margin) v.x += wall_push * (1.0 - (p.x - bounds_min.x) / wall_margin);
    else if (p.x > bounds_max.x - wall_margin) v.x -= wall_push * (1.0 - (bounds_max.x - p.x) / wall_margin);
    if (p.y < bounds_min.y + wall_margin) v.y += wall_push * (1.0 - (p.y - bounds_min.y) / wall_margin);
    else if (p.y > bounds_max.y - wall_margin) v.y -= wall_push * (1.0 - (bounds_max.y - p.y) / wall_margin);
    if (p.z < bounds_min.z + wall_margin) v.z += wall_push * (1.0 - (p.z - bounds_min.z) / wall_margin);
    else if (p.z > bounds_max.z - wall_margin) v.z -= wall_push * (1.0 - (bounds_max.z - p.z) / wall_margin);

    p += v * sub_dt;
    // Six-wall clamp + reflect. collision_damping ~0.3 = 70% energy lost per bounce.
    // Strict clamp first (push back inside), then flip velocity *only if it was
    // moving outward*. Without the velocity-sign check, particles riding the
    // wall lose to numerical drift -> clamp -> flip -> drift back -> clamp ->
    // flip again, perpetual jitter on the boundary. Reflect-only-when-outward
    // lets a settled particle sit at the wall with v ~ 0.
    if (p.x < bounds_min.x) { p.x = bounds_min.x; if (v.x < 0.0) v.x = -v.x * collision_damping; }
    else if (p.x > bounds_max.x) { p.x = bounds_max.x; if (v.x > 0.0) v.x = -v.x * collision_damping; }
    if (p.y < bounds_min.y) { p.y = bounds_min.y; if (v.y < 0.0) v.y = -v.y * collision_damping; }
    else if (p.y > bounds_max.y) { p.y = bounds_max.y; if (v.y > 0.0) v.y = -v.y * collision_damping; }
    if (p.z < bounds_min.z) { p.z = bounds_min.z; if (v.z < 0.0) v.z = -v.z * collision_damping; }
    else if (p.z > bounds_max.z) { p.z = bounds_max.z; if (v.z > 0.0) v.z = -v.z * collision_damping; }

    // Velocity clamp -- crucial guard from 2D experience. Pressure spikes at
    // air-water interface can otherwise fling surface particles to enormous
    // speeds in a single sub-step.
    float speed = length(v);
    if (speed > max_vel) v = v * (max_vel / speed);

    particle_data[pi_pos]     = p.x;
    particle_data[pi_pos + 1] = p.y;
    particle_data[pi_pos + 2] = p.z;
    particle_data[vi]     = v.x;
    particle_data[vi + 1] = v.y;
    particle_data[vi + 2] = v.z;
}
