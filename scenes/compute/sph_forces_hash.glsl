#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 1, std430) buffer HashBuffer {
    uint hash_values[N];
    ivec2 grid_cell_coord[];
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
};

const int MAX_BODIES = 4;

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

int vel_offset(int i) { return particle_count * 2 + i * 2; }
int pred_offset(int i) { return particle_count * 4 + i * 2; }
int temperature_offset(int i) { return particle_count * 8 + i; }
int age_offset(int i) { return particle_count * 9 + i; }
int type_offset(int i) { return particle_count * 10 + i; }
int curl_offset(int i) { return particle_count * 11 + i; }

#define PTYPE_WATER  0
#define PTYPE_FIRE   1
#define PTYPE_SMOKE  2
#define PTYPE_STEAM  3
#define PTYPE_COUNT  4

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
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(particle_count)) return;

    vec2 p = vec2(particle_data[i * 2], particle_data[i * 2 + 1]);
    int vi = vel_offset(int(i));
    vec2 v = vec2(particle_data[vi], particle_data[vi + 1]);

    // Inactive particle: skip simulation, stay off-screen
    if (p.y < -500.0) {
        // predicted_pos = pos (off-screen)
        int pi = pred_offset(int(i));
        particle_data[pi] = p.x;
        particle_data[pi + 1] = p.y;
        // scatter inactive hashes to avoid bucket overflow
        grid_cell_coord[i] = ivec2(int(0x80000000), int(0x80000000));
        hash_values[i] = i % uint(particle_count);
        return;
    }

    int my_type = int(particle_data[type_offset(int(i))]);

    // Per-type physics via ptype tables
    if (my_type != PTYPE_WATER) {
        // Gas/fire/smoke/steam physics
        int ti = temperature_offset(int(i));
        int ai = age_offset(int(i));
        float my_temp = particle_data[ti];
        float my_age = particle_data[ai];
        float init_temp = ptype_init_temp[my_type];
        float cool_rate = ptype_cooling[my_type];
        // Init temperature for freshly spawned particles
        if (my_age < sub_dt * 3.0) {
            my_temp = init_temp;
            particle_data[ti] = my_temp;
        }
        // Uniform Newton cooling (no core protection — lets extinguishing emerge)
        my_temp -= cool_rate * sub_dt;
        my_temp = max(my_temp, ambient_temperature);
        particle_data[ti] = my_temp;
        // Ideal gas buoyancy: rho ∝ 1/T (PV=nRT), F = g*(1 - T_ambient/T)
        float bf = ptype_buoyancy[my_type];
        float T_ratio = my_temp / max(ambient_temperature, 0.001);
        v.y -= gravity * bf * (T_ratio - 1.0) * sub_dt;
        // Subgrid eddy model (LES)
        float temp_factor = my_temp / max(init_temp, 1.0);
        float eddy_x, eddy_y;
        if (my_type == PTYPE_FIRE) {
            // Real curl-driven eddy: force perpendicular to velocity, strength from stored curl
            float my_curl = particle_data[curl_offset(int(i))];
            vec2 vel_dir = normalize(v + vec2(0.001));
            vec2 lateral = vec2(-vel_dir.y, vel_dir.x);
            float strength = clamp(my_curl * 400.0, -2000.0, 2000.0) * temp_factor;
            eddy_x = lateral.x * strength;
            eddy_y = lateral.y * strength * 0.15;
        } else {
            // Smoke/steam: original eddy model
            float phase = float(i) * 2.399 + p.x * 0.037 + p.y * 0.029;
            float phase2 = float(i) * 1.713 + p.x * 0.053 - p.y * 0.041;
            eddy_x = (sin(phase * 0.9 + cos(phase2 * 0.7) * 1.6)
                    + cos(phase2 * 1.1) * sin(phase * 0.6) * 0.7) * 900.0 * temp_factor;
            eddy_y = (cos(phase * 1.3) * sin(phase2 * 0.8) * 0.5
                    + sin(phase * 0.5 + phase2) * 0.3
                    + cos(phase2 * 0.4 + phase) * 0.2) * 500.0 * temp_factor;
        }
        v.x += eddy_x * sub_dt;
        v.y += eddy_y * sub_dt;
        // Body interaction: boundary layer drag + wake
        for (int b = 0; b < body_count; b++) {
            if (bodies[b].enabled == 0) continue;
            vec2 to_body = p - bodies[b].pos;
            float dist = length(to_body);
            float radius = bodies[b].shape_radius;
            float influence = radius * 3.5;
            if (dist < influence && dist > 0.001) {
                vec2 dir = to_body / dist;
                float drag = (1.0 - smoothstep(radius, influence, dist)) * body_drag_gas;
                v = mix(v, bodies[b].vel, drag * sub_dt);
                vec2 body_vel_dir = normalize(bodies[b].vel + vec2(0.001));
                float behind = -dot(body_vel_dir, dir);
                if (behind > 0.0) {
                    float taper = 1.0 - abs(behind);
                    float wake_pressure = behind * taper * body_drag_gas * 2.0;
                    float wake_dist = (dist - radius) * 0.5;
                    v += body_vel_dir * wake_pressure / max(wake_dist, 0.1) * sub_dt;
                }
            }
        }
        // Minimal weight for gas
        v.y += gravity * 0.03 * sub_dt;
    } else {
        // Water mode: full gravity
        v.y += gravity * sub_dt;
    }

    // Apply velocity damping
    v *= velocity_damping;

    // Mouse interaction (skip grab in spray mode)
    if (mouse_pressed != 0 && spray_mode == 0) {
        vec2 diff = mouse_pos - p;
        float len = length(diff);
        if (len < mouse_radius_grab && len > 0.001) {
            vec2 dir = diff / len;
            if (len < mouse_radius_stick) {
                v = dir * grab_speed * (len / mouse_radius_stick);
                v *= 0.3;
            } else {
                float t = 1.0 - (len - mouse_radius_stick) / (mouse_radius_grab - mouse_radius_stick);
                v += dir * grab_speed * t * 0.3;
                v *= 0.95;
            }
        }
    }

    // Write back velocity
    particle_data[vi] = v.x;
    particle_data[vi + 1] = v.y;

    // Compute predicted position
    vec2 pred = p + v * prediction_factor;
    int pi = pred_offset(int(i));
    particle_data[pi] = pred.x;
    particle_data[pi + 1] = pred.y;

    // Compute grid cell and hash
    ivec2 cell = ivec2(floor(pred * grid_cell_inverse));
    grid_cell_coord[i] = cell;
    hash_values[i] = hash_cell(cell);
}
