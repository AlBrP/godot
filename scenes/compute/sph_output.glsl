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
};

layout(set = 1, binding = 0, rgba16f) uniform image2D position_tex;
layout(set = 1, binding = 1, rg32f) uniform image2D hashlookup_tex;

const int TEX_W = 72;
const int TEX_H = 70;
const int TOTAL_PIXELS = TEX_W * TEX_H;

int vel_offset(int i) { return particle_count * 2 + i * 2; }

void main() {
    uint j = gl_GlobalInvocationID.x;
    if (j >= uint(TOTAL_PIXELS)) return;

    ivec2 tex_coord = ivec2(int(j) % TEX_W, int(j) / TEX_W);

    if (int(j) < particle_count) {
        uint pidx = sorted_indices[j];
        vec2 p = vec2(particle_data[pidx * 2], particle_data[pidx * 2 + 1]);
        vec2 world_pos = p + ground_offset;
        int vi = vel_offset(int(pidx));
        vec2 v = vec2(particle_data[vi], particle_data[vi + 1]);
        float vn_x = clamp(v.x / max_vel * 0.5 + 0.5, 0.0, 1.0);
        float vn_y = clamp(v.y / max_vel * 0.5 + 0.5, 0.0, 1.0);
        imageStore(position_tex, tex_coord, vec4(world_pos, vn_x, vn_y));

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
    }
}
