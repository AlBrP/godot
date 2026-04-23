#[compute]
#version 450

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 1, std430) buffer ForceAccum {
    int force_accum[2];  // [0]=x, [1]=y — collision penalty reaction force
};

layout(set = 0, binding = 2, std430) buffer WaterInfo {
    uint water_count_below;    // particles in shell zone with Y >= body_pos.y (below center)
    uint water_count_above;    // particles in shell zone with Y < body_pos.y (above center)
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
    float _pad0;
    vec2 body_pos;
    float body_radius;
    int body_enabled;
    vec2 body_vel;
    float force_scale;
    float _pad1;
    float _pad2;
    float _pad3;
    float _pad4;
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

    // Circle body collision + water surface tracking
    if (body_enabled != 0) {
        vec2 diff = p - body_pos;
        float dist = length(diff);
        if (dist < body_radius && dist > 0.001) {
            vec2 normal = diff / dist;
            float penetration = body_radius - dist;
            p += normal * penetration;
            vec2 rel_v = v - body_vel;
            float vn = dot(rel_v, normal);
            if (vn < 0.0) {
                v = body_vel + reflect(rel_v, normal) * collision_damping;
                // Only accumulate force for ACTIVE collisions (particle moving toward ball)
                float f = penetration * force_scale * min(1.0, abs(vn) / 1000.0);
                atomicAdd(force_accum[0], int(-normal.x * f));
                atomicAdd(force_accum[1], int(-normal.y * f));
            }
        }

        // Track water near ball — split by above/below ball center
        // count_below: particles below ball center = water ball displaces → drives buoyancy
        // count_above: particles above ball center → ignored for buoyancy (prevents splash false-positive)
        float dist_to_ball = length(p - body_pos);
        if (dist_to_ball >= body_radius * 0.8f && dist_to_ball < body_radius * 2.5f) {
            if (p.y >= body_pos.y) {
                atomicAdd(water_count_below, 1u);
            } else {
                atomicAdd(water_count_above, 1u);
            }
        }
    }

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
