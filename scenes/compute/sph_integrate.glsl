#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;
const int MAX_BODIES = 4;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
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
};

int vel_offset(int i) { return particle_count * 2 + i * 2; }
int age_offset(int i) { return particle_count * 9 + i; }
int type_offset(int i) { return particle_count * 10 + i; }

#define PTYPE_WATER  0
#define PTYPE_FIRE   1
#define PTYPE_SMOKE  2
#define PTYPE_STEAM  3

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(particle_count)) return;

    int i = int(idx);

    vec2 p = vec2(particle_data[i * 2], particle_data[i * 2 + 1]);
    if (p.y < -500.0) return;

    int vi = vel_offset(i);
    vec2 v = vec2(particle_data[vi], particle_data[vi + 1]);

    p += v * sub_dt;

    int my_type = int(particle_data[type_offset(i)]);

    // Light sphere safety push-out for all active bodies
    for (int b = 0; b < body_count; b++) {
        if (bodies[b].enabled == 0) continue;
        vec2 diff = p - bodies[b].pos;
        float sr = bodies[b].shape_radius;
        float dist = length(diff);
        if (dist < sr && dist > 0.0001) {
            float penetration = sr - dist;
            vec2 normal = diff / dist;
            float push_strength = 1.0;
            p += normal * penetration * push_strength;
            vec2 rel_v = v - bodies[b].vel;
            float vn = dot(rel_v, normal);
            if (vn < 0.0)
                v = bodies[b].vel + reflect(rel_v, normal) * collision_damping;
        }
    }

    // Boundary walls
    if (p.x < bounds_min.x) { p.x = bounds_min.x; v.x *= -collision_damping; }
    else if (p.x > bounds_max.x) { p.x = bounds_max.x; v.x *= -collision_damping; }
    if (p.y < bounds_min.y) { p.y = bounds_min.y; v.y *= -collision_damping; }
    else if (p.y > bounds_max.y) { p.y = bounds_max.y; v.y *= -collision_damping; }

    // Gas particle lifecycle
    if (my_type != PTYPE_WATER) {
        int ai = age_offset(i);
        particle_data[ai] += sub_dt;
        float lifetime = ptype_lifetime[my_type];
        if (particle_data[ai] > lifetime) {
            p.y = -600.0;
            v = vec2(0.0);
            particle_data[ai] = 0.0;
            int ti = particle_count * 8 + i;
            particle_data[ti] = 0.0;
        }
    }

    particle_data[i * 2] = p.x;
    particle_data[i * 2 + 1] = p.y;
    particle_data[vi] = v.x;
    particle_data[vi + 1] = v.y;
}
