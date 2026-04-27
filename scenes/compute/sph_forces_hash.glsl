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

    // Gas/smoke/fire physics: shared formulation, parameterized by init_temp
    if (sim_mode == 1 || sim_mode == 2) {
        int ti = temperature_offset(int(i));
        int ai = age_offset(int(i));
        float my_temp = particle_data[ti];
        float my_age = particle_data[ai];
        float init_temp = sim_mode == 2 ? 1500.0 : 800.0;
        float cool_rate = sim_mode == 2 ? cooling_rate * 15.0 : cooling_rate;
        // Init temperature for freshly spawned particles
        if (my_age < sub_dt * 3.0) {
            my_temp = init_temp;
            particle_data[ti] = my_temp;
        }
        // Newton cooling: hot core protected, edges cool fast (flame shape driver)
        float core_protection = smoothstep(ambient_temperature, init_temp, my_temp);
        float cool_factor = 1.0 - core_protection * 0.8;
        my_temp -= cool_rate * cool_factor * sub_dt;
        my_temp = max(my_temp, ambient_temperature);
        particle_data[ti] = my_temp;
        // Boussinesq buoyancy: fire reduced 30%
        float bf = sim_mode == 2 ? buoyancy_alpha * 0.7 : buoyancy_alpha;
        v.y += gravity * bf * (1.0 - my_temp / ambient_temperature) * sub_dt;
        // Subgrid eddy model (LES)
        float temp_factor = my_temp / init_temp;
        float eddy_x, eddy_y;
        if (sim_mode == 2) {
            // Fire: reduced lateral spread for columnar flame shape
            float sway = p.y * 0.031;
            float pid = float(i) * 0.7;
            // Height-dependent lateral boost: sides peel off at top -> arched tip
            float rise_h = clamp((mouse_pos.y - p.y) / 100.0, 0.0, 1.0);
            float top_spread = 1.0 + rise_h * 2.5;  // 1x at base, 3.5x at top
            // Lateral eddy: moderate spread for flame tongue formation
            eddy_x = sin(sway + pid) * 1100.0 * temp_factor * top_spread;
            eddy_x += cos(sway * 2.1 + pid * 1.3) * 360.0 * temp_factor * top_spread;
            // Vertical eddy: slight upward turbulence (billow shape driver)
            eddy_y = cos(sway * 1.4 + pid * 0.6) * 120.0 * temp_factor;
            eddy_y += sin(sway * 3.0 + pid) * 60.0 * temp_factor;
        } else {
            // Smoke: original eddy model
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
        // Mild column guidance: gentle centering only at mid-height where spread is worst
        if (sim_mode == 2) {
            float dx = p.x - mouse_pos.x;
            float rise = mouse_pos.y - p.y;
            // Only active in middle band (40-180px up), not near base or far tip
            float guide_zone = smoothstep(20.0, 50.0, rise) * smoothstep(120.0, 90.0, rise);
            v.x -= dx * 2.8 * guide_zone * temp_factor * sub_dt;
        }
        // Body interaction: boundary layer drag + wake (physical, no trick)
        for (int b = 0; b < body_count; b++) {
            if (bodies[b].enabled == 0) continue;
            vec2 to_body = p - bodies[b].pos;
            float dist = length(to_body);
            float radius = bodies[b].shape_radius;
            float influence = radius * 3.5;
            if (dist < influence && dist > 0.001) {
                vec2 dir = to_body / dist;
                // Partial no-slip: boundary layer drag
                float drag = (1.0 - smoothstep(radius, influence, dist)) * body_drag_gas;
                v = mix(v, bodies[b].vel, drag * sub_dt);
                // Wake: low pressure behind moving body (Bernoulli)
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
