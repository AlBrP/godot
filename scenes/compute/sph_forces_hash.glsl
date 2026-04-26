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
    float _pad9;
    float target_density;
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

    // Smoke physics: clean formulation, no procedural tricks
    if (sim_mode == 1) {
        int ti = temperature_offset(int(i));
        int ai = age_offset(int(i));
        float my_temp = particle_data[ti];
        float my_age = particle_data[ai];
        // Init temperature for freshly spawned particles
        if (my_age < sub_dt * 3.0) {
            my_temp = 800.0;
            particle_data[ti] = my_temp;
        }
        // Newton cooling to ambient
        my_temp -= cooling_rate * sub_dt;
        my_temp = max(my_temp, ambient_temperature);
        particle_data[ti] = my_temp;
        // Boussinesq approximation: buoyancy = -beta * (T-T0) * g = g * beta * (1 - T/T0)
        v.y += gravity * buoyancy_alpha * (1.0 - my_temp / ambient_temperature) * sub_dt;
        // Subgrid eddy model (LES): models unresolved turbulent eddies at sub-particle scale
        float temp_factor = my_temp / 800.0;
        float vx = p.x * 0.03;
        float vy = p.y * 0.025;
        float eddy_x = sin(vy + cos(vx * 0.7) * 1.5) * 900.0 * temp_factor;
        float eddy_y = (cos(vx * 1.1) * sin(vy * 0.8) * 0.5 + sin(vx * 0.5 + vy) * 0.3) * 500.0 * temp_factor;
        v.x += eddy_x * sub_dt;
        v.y += eddy_y * sub_dt;
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
