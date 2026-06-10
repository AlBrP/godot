#[compute]
#version 450

// Pass 1: forces + spatial hash write.
// Applies gravity to velocity, predicts position, then writes cell-coord +
// hash index for the spatial sort that follows.
//
// P2.1 simplification: pure water only -- no ptype, no temperature, no mouse
// grab, no body interaction. Just gravity + viscosity damping + predicted
// position + hash write.

layout(local_size_x = 256) in;

const int N = 20000;

layout(set = 0, binding = 0, std430) buffer ParticleBuffer {
    float particle_data[];
};

layout(set = 0, binding = 1, std430) buffer HashBuffer {
    uint hash_values[N];
    ivec3 grid_cell_coord[];
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
    vec4 player_data;  // xyz = world pos, w = push_radius (<=0 disabled)
};

int vel_offset(int i)  { return particle_count * 3 + i * 3; }
int pred_offset(int i) { return particle_count * 6 + i * 3; }

const int HASH_K1 = 15823;
const int HASH_K2 = 9737333;
const int HASH_K3 = 440817757;

uint hash_cell(ivec3 cell) {
    uint cx = uint(cell.x);
    uint cy = uint(cell.y);
    uint cz = uint(cell.z);
    return (cx * uint(HASH_K1) + cy * uint(HASH_K2) + cz * uint(HASH_K3)) % uint(particle_count);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(particle_count)) return;

    int ii = int(i);
    int pi_pos = ii * 3;
    int pi_vel = vel_offset(ii);
    int pi_pred = pred_offset(ii);

    vec3 p = vec3(particle_data[pi_pos], particle_data[pi_pos + 1], particle_data[pi_pos + 2]);
    vec3 v = vec3(particle_data[pi_vel], particle_data[pi_vel + 1], particle_data[pi_vel + 2]);

    // Gravity (Godot Y-up: pull down on -Y; if gravity is negative -> floats up)
    v.y -= gravity * sub_dt;

    // Player push: outward radial impulse if particle inside push sphere.
    // Strength fades from full at center to 0 at radius. Used for "wade through
    // water" feel + reset interaction without needing rigid-body coupling.
    if (player_data.w > 0.0) {
        vec3 to_p = p - player_data.xyz;
        float d2 = dot(to_p, to_p);
        float r = player_data.w;
        if (d2 < r * r && d2 > 0.0001) {
            float d = sqrt(d2);
            float t = 1.0 - d / r;             // 1 at center, 0 at edge
            vec3 dir = to_p / d;
            float push_speed = 8.0 * t;        // m/s outward
            v += dir * push_speed * sub_dt * 60.0; // amortise dt -> per-frame impulse
        }
    }

    // Velocity damping
    v *= velocity_damping;

    // Write back velocity
    particle_data[pi_vel]     = v.x;
    particle_data[pi_vel + 1] = v.y;
    particle_data[pi_vel + 2] = v.z;

    // Predicted position (PCISPH/PBF-style: density/pressure use this so
    // high-speed particles still see "current" neighbours)
    vec3 pred = p + v * prediction_factor;
    particle_data[pi_pred]     = pred.x;
    particle_data[pi_pred + 1] = pred.y;
    particle_data[pi_pred + 2] = pred.z;

    // Spatial hash from predicted position
    ivec3 cell = ivec3(floor(pred * grid_cell_inverse));
    grid_cell_coord[i] = cell;
    hash_values[i] = hash_cell(cell);
}
