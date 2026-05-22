#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#include "builtin.h"

// Map wp.breakpoint() to a device brkpt at the call site so cuda-gdb attributes the stop to the generated .cu line
#if defined(__CUDACC__) && !defined(_MSC_VER)
#define __debugbreak() __brkpt()
#endif

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(_idx, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, _idx, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, _idx, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, _idx, dim)

#define builtin_block_dim() wp::block_dim()



extern "C" __global__ void solve_particle_ground_plane_contacts_a4104fa3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::int32> var_particle_flags,
    wp::float32 var_ground_height,
    wp::float32 var_particle_mu,
    wp::float32 var_dt,
    wp::float32 var_relaxation,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_deltas)
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
            // reset shared memory allocator
        wp::tile_shared_storage_t::init();

        //---------
        // primal vars
        wp::int32 var_0;
        wp::int32* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        const wp::float32 var_12 = 0.0;
        const wp::float32 var_13 = 1.0;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32>* var_15;
        const wp::int32 var_16 = 2;
        wp::float32 var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 0.0;
        bool var_24;
        wp::float32 var_25;
        wp::vec_t<3, wp::float32> var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32>* var_30;
        wp::float32 var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::float32 var_36;
        const wp::float32 var_37 = 1e-08;
        bool var_38;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        //---------
        // forward
        // def solve_particle_ground_plane_contacts(                                              <L 12>
        // tid = wp.tid()                                                                         <L 24>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                                  <L 26>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 27>
            continue;
        }
        // if particle_inv_mass[tid] <= 0.0:                                                      <L 28>
        var_7 = wp::address(var_particle_inv_mass, var_0);
        var_10 = wp::load(var_7);
        var_9 = (var_10 <= var_8);
        if (var_9) {
            // return                                                                             <L 29>
            continue;
        }
        // n = wp.vec3(0.0, 0.0, 1.0)                                                             <L 31>
        var_14 = wp::vec_t<3, wp::float32>(var_11, var_12, var_13);
        // signed_distance = particle_q[tid][2] - particle_radius[tid] - ground_height            <L 32>
        var_15 = wp::address(var_particle_q, var_0);
        var_18 = wp::load(var_15);
        var_17 = wp::extract(var_18, var_16);
        var_19 = wp::address(var_particle_radius, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::sub(var_17, var_21);
        var_22 = wp::sub(var_20, var_ground_height);
        // if signed_distance >= 0.0:                                                             <L 33>
        var_24 = (var_22 >= var_23);
        if (var_24) {
            // return                                                                             <L 34>
            continue;
        }
        // correction = n * (-signed_distance)                                                    <L 36>
        var_25 = wp::neg(var_22);
        var_26 = wp::mul(var_14, var_25);
        // if particle_mu > 0.0:                                                                  <L 38>
        var_28 = (var_particle_mu > var_27);
        if (var_28) {
            // tangential_v = particle_qd[tid] - n * wp.dot(n, particle_qd[tid])                  <L 39>
            var_29 = wp::address(var_particle_qd, var_0);
            var_30 = wp::address(var_particle_qd, var_0);
            var_32 = wp::load(var_30);
            var_31 = wp::dot(var_14, var_32);
            var_33 = wp::mul(var_14, var_31);
            var_35 = wp::load(var_29);
            var_34 = wp::sub(var_35, var_33);
            // tangential_speed = wp.length(tangential_v)                                         <L 40>
            var_36 = wp::length(var_34);
            // if tangential_speed > 1.0e-8:                                                      <L 41>
            var_38 = (var_36 > var_37);
            if (var_38) {
                // max_friction_step = particle_mu * (-signed_distance)                           <L 42>
                var_39 = wp::neg(var_22);
                var_40 = wp::mul(var_particle_mu, var_39);
                // velocity_friction_step = tangential_speed * dt                                 <L 43>
                var_41 = wp::mul(var_36, var_dt);
                // friction_step = wp.min(max_friction_step, velocity_friction_step)              <L 44>
                var_42 = wp::min(var_40, var_41);
                // correction -= tangential_v * (friction_step / tangential_speed)                <L 45>
                var_43 = wp::div(var_42, var_36);
                var_44 = wp::mul(var_34, var_43);
                var_45 = wp::sub(var_26, var_44);
            }
            var_46 = wp::where(var_38, var_45, var_26);
        }
        var_47 = wp::where(var_28, var_46, var_26);
        // wp.atomic_add(particle_deltas, tid, correction * relaxation)                           <L 47>
        var_48 = wp::mul(var_47, var_relaxation);
        var_49 = wp::atomic_add(var_particle_deltas, var_0, var_48);
    }
}

