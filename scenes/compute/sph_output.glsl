#[compute]
#version 450

layout(local_size_x = 256) in;

const int N = 5000;

layout(set = 0, binding = 0, std430) readonly buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 2, std430) readonly buffer SortBuffer {
    uint sorted_indices[N];
    uint sorted_hashes[];
};

layout(set = 0, binding = 3, std430) readonly buffer HashTableBuffer {
    int ht_start[N];
    int ht_end[];
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

layout(set = 1, binding = 0, rgba16f) uniform image2D position_tex;
layout(set = 1, binding = 1, rg32f) uniform image2D hashlookup_tex;
layout(set = 1, binding = 2, rgba16f) uniform image2D physics_tex;
// stable_position_tex: same channel layout as position_tex, but indexed by
// raw particle id (pidx) instead of sorted-slot. Used by INSTANCE_ID-based
// sprite renderers (e.g. smoke_sprite) so each particle stays bound to one
// pixel even when the spatial-hash sort reshuffles sorted_indices each frame.
layout(set = 1, binding = 3, rgba16f) uniform image2D stable_position_tex;

const int TEX_W = 72;
const int TEX_H = 70;
const int TOTAL_PIXELS = TEX_W * TEX_H;

int vel_offset(int i) { return particle_count * 2 + i * 2; }
int density_offset(int i) { return particle_count * 6 + i; }
int temperature_offset(int i) { return particle_count * 8 + i; }
int age_offset(int i) { return particle_count * 9 + i; }
int type_offset(int i) { return particle_count * 10 + i; }
int curl_offset(int i) { return particle_count * 11 + i; }

#define PTYPE_WATER  0
#define PTYPE_FIRE   1
#define PTYPE_SMOKE  2
#define PTYPE_STEAM  3

void main() {
    uint j = gl_GlobalInvocationID.x;
    if (j >= uint(TOTAL_PIXELS)) return;

    ivec2 tex_coord = ivec2(int(j) % TEX_W, int(j) / TEX_W);

    if (int(j) < particle_count) {
        uint pidx = sorted_indices[j];
        vec2 p = vec2(particle_data[pidx * 2], particle_data[pidx * 2 + 1]);
        vec2 world_pos = p + ground_offset;
        int my_type = int(particle_data[type_offset(int(pidx))]);
        float type_encoded = float(my_type) / 16.0;

        float extra = 0.0;
        if (my_type != PTYPE_WATER) {
            float temp = particle_data[temperature_offset(int(pidx))];
            float init_temp = ptype_init_temp[my_type];
            extra = clamp(temp / max(init_temp, 1.0), 0.0, 1.0);
        } else {
            float density_val = particle_data[density_offset(int(pidx))];
            extra = clamp(density_val * 0.05, 0.0, 1.0);
        }
        imageStore(position_tex, tex_coord, vec4(world_pos, type_encoded, extra));

        // Stable view: indexed by pidx, used by INSTANCE_ID-based sprite
        // renderers. For smoke/steam, replace .a (extra) with remaining-life
        // fraction so the sprite shader can smoothly fade alpha to 0 right
        // before integrate.glsl teleports the particle off-screen. For other
        // types, keep extra as-is (temp/density) since fire metaball still
        // reads .a for color via position_tex anyway.
        float stable_extra = extra;
        if (my_type == PTYPE_SMOKE || my_type == PTYPE_STEAM) {
            float age = particle_data[age_offset(int(pidx))];
            float lifetime = ptype_lifetime[my_type];
            stable_extra = clamp(1.0 - age / max(lifetime, 0.001), 0.0, 1.0);
        }
        ivec2 stable_coord = ivec2(int(pidx) % TEX_W, int(pidx) / TEX_W);
        imageStore(stable_position_tex, stable_coord, vec4(world_pos, type_encoded, stable_extra));

        // -- Physics data: velocity, curl proxy, pressure --
        vec2 vel_out = vec2(0.0);
        float curl_out = 0.0;
        float pressure_out = 0.0;

        if (my_type == PTYPE_FIRE) {
            int vi = vel_offset(int(pidx));
            vel_out = vec2(particle_data[vi], particle_data[vi + 1]);

            float dens = particle_data[density_offset(int(pidx))];
            float temp = particle_data[temperature_offset(int(pidx))];

            // Gas pressure: density * stiffness * (T / T_ambient)
            float gs = ptype_stiffness[my_type];
            pressure_out = dens * gs * (temp / max(ambient_temperature, 1.0));

            // Real curl from pressure_viscosity pass (stored in particle buffer)
            curl_out = particle_data[curl_offset(int(pidx))];
        } else if (my_type == PTYPE_SMOKE || my_type == PTYPE_STEAM) {
            int vi = vel_offset(int(pidx));
            vel_out = vec2(particle_data[vi], particle_data[vi + 1]);

            float dens = particle_data[density_offset(int(pidx))];
            float temp = particle_data[temperature_offset(int(pidx))];
            float gs = ptype_stiffness[my_type];
            pressure_out = dens * gs * (temp / max(ambient_temperature, 1.0));

            // .b channel: remaining-life fraction. Read by metaball smoke
            // shader to fade alpha to 0 in the last ~20% of lifetime so the
            // puff vanishes smoothly instead of popping out the frame
            // integrate.glsl teleports it to -600. Dead particles get 0
            // (caught by temp_n>0.02 gate too, but kept defensive).
            if (p.y < -500.0) {
                curl_out = 0.0;
            } else {
                float age = particle_data[age_offset(int(pidx))];
                float lifetime = ptype_lifetime[my_type];
                curl_out = clamp(1.0 - age / max(lifetime, 0.001), 0.0, 1.0);
            }
        } else {
            int vi_w = vel_offset(int(pidx));
            vel_out = vec2(particle_data[vi_w], particle_data[vi_w + 1]);
            float dens_w = particle_data[density_offset(int(pidx))];
            pressure_out = max(dens_w - target_density, 0.0) * pressure_multiplier;
            curl_out = 0.0;
        }

        imageStore(physics_tex, tex_coord, vec4(vel_out, curl_out, pressure_out));

        int start = ht_start[int(j)];
        int end = ht_end[int(j)];
        if (start >= 0) {
            imageStore(hashlookup_tex, tex_coord, vec4(float(start), float(end), 0.0, 0.0));
        } else {
            imageStore(hashlookup_tex, tex_coord, vec4(-1.0, -1.0, 0.0, 0.0));
        }
    } else {
        imageStore(position_tex, tex_coord, vec4(0.0, 0.0, 0.0, 0.0));
        imageStore(hashlookup_tex, tex_coord, vec4(-1.0, -1.0, 0.0, 0.0));
        imageStore(physics_tex, tex_coord, vec4(0.0, 0.0, 0.0, 0.0));
        // stable_position_tex shares the same TEX_W x TEX_H grid as position_tex.
        // Pixels past particle_count are unused by the pidx mapping too, so
        // mirror the clear here for consistency / predictable sampling.
        imageStore(stable_position_tex, tex_coord, vec4(0.0, 0.0, 0.0, 0.0));
    }
}
