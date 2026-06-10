#[compute]
#version 450

// Pass 10: write world positions into a 2D texture so the imposter renderer
// (sph_sphere_imposter.gdshader, MultiMeshInstance3D) can sample them via
// INSTANCE_ID without a CPU readback.
//
// We index by sorted-slot rather than raw particle id -- visually identical
// since every particle gets exactly one slot, and saves the stable-id buffer.

layout(local_size_x = 256) in;

const int N = 20000;

layout(set = 0, binding = 0, std430) readonly buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 2, std430) readonly buffer SortBuffer {
    uint sorted_indices[N];
    uint sorted_hashes[];
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

layout(set = 1, binding = 0, rgba32f) uniform image2D position_tex;

const int TEX_W = 200;
const int TEX_H = 100;
const int TOTAL_PIXELS = TEX_W * TEX_H;

int vel_offset(int i) { return particle_count * 3 + i * 3; }
int density_offset(int i) { return particle_count * 9 + i; }

void main() {
    uint j = gl_GlobalInvocationID.x;
    if (j >= uint(TOTAL_PIXELS)) return;

    ivec2 tex_coord = ivec2(int(j) % TEX_W, int(j) / TEX_W);

    if (int(j) < particle_count) {
        uint pidx = sorted_indices[j];
        int pi_pos = int(pidx) * 3;
        vec3 p = vec3(particle_data[pi_pos], particle_data[pi_pos + 1], particle_data[pi_pos + 2]);

        // .a slot: optional payload. Encode density / target_density - 1.0 so the
        // renderer can tint by compression / sparsity if it wants to.
        float dens = particle_data[density_offset(int(pidx))];
        float alpha = dens / max(target_density, 0.0001);
        imageStore(position_tex, tex_coord, vec4(p, alpha));
    } else {
        // Outside particle range -- park at far -Y so imposter culls it.
        imageStore(position_tex, tex_coord, vec4(0.0, -10000.0, 0.0, 0.0));
    }
}
