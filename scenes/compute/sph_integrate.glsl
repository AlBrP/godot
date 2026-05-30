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
    vec4 ptype_boil_point;
    vec4 ptype_boil_product;
    vec4 ptype_condense_point;
    vec4 ptype_condense_product;
    float boundary_friction;
};

int vel_offset(int i) { return particle_count * 2 + i * 2; }
int temperature_offset(int i) { return particle_count * 8 + i; }
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

    // Phase transition: temperature crossing a threshold rewrites the
    // particle's type, letting the ptype tables (stiffness, buoyancy,
    // cooling, lifetime, ...) flip to the new phase next frame. age and
    // temperature are kept so the new phase inherits kinematics smoothly;
    // resetting age=0 would let forces_hash re-init temperature to the
    // new ptype_init_temp, undoing the threshold crossing.
    float my_temp = particle_data[temperature_offset(i)];
    int new_type = my_type;
    float boil_pt = ptype_boil_point[my_type];
    if (boil_pt > 0.0 && my_temp >= boil_pt) {
        int b = int(ptype_boil_product[my_type] + 0.5);
        if (b >= 0 && b < 4) new_type = b;
    }
    float cond_pt = ptype_condense_point[my_type];
    if (cond_pt > 0.0 && my_temp <= cond_pt) {
        int c = int(ptype_condense_product[my_type] + 0.5);
        if (c >= 0 && c < 4) new_type = c;
    }
    if (new_type != my_type) {
        particle_data[type_offset(i)] = float(new_type);
        // Bump age past the forces_hash "freshly spawned" window
        // (my_age < sub_dt*3) so the new type's init_temp doesn't
        // overwrite my_temp next frame and undo the threshold crossing.
        // WATER particles in particular sit at age=0 forever (WATER skips
        // the age++ in this pass), so without this any water->steam
        // transition would be re-initialised to 500K immediately.
        int ai = age_offset(i);
        particle_data[ai] = max(particle_data[ai], sub_dt * 5.0);
        my_type = new_type;
    }

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

    // Clamp velocity to max_vel to prevent runaway from pressure spikes.
    // High-speed spray particles (~2000 px/s) smashing into the water pool
    // create an instant density imbalance at the air-water interface; the
    // pressure response (pressure_multiplier=1e6) is large enough that
    // surface particles can be reversed to enormous upward speeds in one
    // sub-step and fling far above the surface (no neighbours up there
    // -> no viscosity damping, only gravity). max_vel was declared in the
    // Params UBO but never enforced. Doing it here keeps the configured
    // 2000 px/s cap meaningful.
    float speed = length(v);
    if (speed > max_vel) v = v * (max_vel / speed);

    particle_data[i * 2] = p.x;
    particle_data[i * 2 + 1] = p.y;
    particle_data[vi] = v.x;
    particle_data[vi + 1] = v.y;
}
