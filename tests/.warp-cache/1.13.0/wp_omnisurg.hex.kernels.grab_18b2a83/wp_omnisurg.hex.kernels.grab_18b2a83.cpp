#define WP_NO_BFLOAT16

#define WP_TILE_BLOCK_DIM 1
#define WP_NO_CRT
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)

#define builtin_block_dim() wp::block_dim()

struct wp_args_solve_grab_distance_constraints_kernel_e27b06a6 {
    wp::array_t<wp::vec_t<3, wp::float32>> particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> particle_qd;
    wp::array_t<wp::float32> particle_inv_mass;
    wp::array_t<wp::int32> particle_flags;
    wp::array_t<wp::int32> grab_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> grab_offsets;
    wp::vec_t<3, wp::float32> pull_target;
    wp::float32 stiffness;
};


void solve_grab_distance_constraints_kernel_e27b06a6_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_solve_grab_distance_constraints_kernel_e27b06a6 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q = _wp_args->particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd = _wp_args->particle_qd;
    wp::array_t<wp::float32> var_particle_inv_mass = _wp_args->particle_inv_mass;
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::int32> var_grab_indices = _wp_args->grab_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> var_grab_offsets = _wp_args->grab_offsets;
    wp::vec_t<3, wp::float32> var_pull_target = _wp_args->pull_target;
    wp::float32 var_stiffness = _wp_args->stiffness;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    wp::int32* var_6;
    const wp::int32 var_7 = 1;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    wp::float32* var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::float32 var_17 = 0.0;
    bool var_18;
    const wp::float32 var_19 = 0.0;
    wp::float32 var_20;
    const wp::float32 var_21 = 1.0;
    bool var_22;
    const wp::float32 var_23 = 1.0;
    wp::float32 var_24;
    const wp::float32 var_25 = 0.0;
    bool var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 1.0;
    const wp::float32 var_37 = 0.0;
    const wp::float32 var_38 = 0.0;
    wp::vec_t<3, wp::float32> var_39;
    const wp::float32 var_40 = 1e-08;
    bool var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::float32 var_44 = 1e-08;
    bool var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    const wp::float32 var_54 = 0.0;
    const wp::float32 var_55 = 0.0;
    const wp::float32 var_56 = 0.0;
    wp::vec_t<3, wp::float32> var_57;
    //---------
    // forward
    // def solve_grab_distance_constraints_kernel(                                            <L 15>
    // tid = wp.tid()                                                                         <L 25>
    var_0 = builtin_tid1d();
    // particle_idx = grab_indices[tid]                                                       <L 26>
    var_1 = wp::address(var_grab_indices, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if particle_idx < 0:                                                                   <L 27>
    var_5 = (var_2 < var_4);
    if (var_5) {
        // return                                                                             <L 28>
        return;
    }
    // if (particle_flags[particle_idx] & _ACTIVE_BIT) == 0:                                  <L 29>
    var_6 = wp::address(var_particle_flags, var_2);
    var_9 = wp::load(var_6);
    var_8 = wp::bit_and(var_9, var_7);
    var_11 = (var_8 == var_10);
    if (var_11) {
        // return                                                                             <L 30>
        return;
    }
    // if particle_inv_mass[particle_idx] <= 0.0:                                             <L 31>
    var_12 = wp::address(var_particle_inv_mass, var_2);
    var_15 = wp::load(var_12);
    var_14 = (var_15 <= var_13);
    if (var_14) {
        // return                                                                             <L 32>
        return;
    }
    // alpha = stiffness                                                                      <L 34>
    var_16 = wp::copy(var_stiffness);
    // if alpha < 0.0:                                                                        <L 35>
    var_18 = (var_16 < var_17);
    if (var_18) {
        // alpha = 0.0                                                                        <L 36>
    }
    var_20 = wp::where(var_18, var_19, var_16);
    // if alpha > 1.0:                                                                        <L 37>
    var_22 = (var_20 > var_21);
    if (var_22) {
        // alpha = 1.0                                                                        <L 38>
    }
    var_24 = wp::where(var_22, var_23, var_20);
    // if alpha <= 0.0:                                                                       <L 39>
    var_26 = (var_24 <= var_25);
    if (var_26) {
        // return                                                                             <L 40>
        return;
    }
    // q = particle_q[particle_idx]                                                           <L 42>
    var_27 = wp::address(var_particle_q, var_2);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // rest = grab_offsets[tid]                                                               <L 43>
    var_30 = wp::address(var_grab_offsets, var_0);
    var_32 = wp::load(var_30);
    var_31 = wp::copy(var_32);
    // rest_length = wp.length(rest)                                                          <L 44>
    var_33 = wp::length(var_31);
    // delta = q - pull_target                                                                <L 45>
    var_34 = wp::sub(var_28, var_pull_target);
    // dist = wp.length(delta)                                                                <L 46>
    var_35 = wp::length(var_34);
    // direction = wp.vec3(1.0, 0.0, 0.0)                                                     <L 48>
    var_39 = wp::vec_t<3, wp::float32>(var_36, var_37, var_38);
    // if dist > 1.0e-8:                                                                      <L 49>
    var_41 = (var_35 > var_40);
    if (var_41) {
        // direction = delta / dist                                                           <L 50>
        var_42 = wp::div(var_34, var_35);
    }
    var_43 = wp::where(var_41, var_42, var_39);
    if (!var_41) {
        // elif rest_length > 1.0e-8:                                                         <L 51>
        var_45 = (var_33 > var_44);
        if (var_45) {
            // direction = rest / rest_length                                                 <L 52>
            var_46 = wp::div(var_31, var_33);
        }
        var_47 = wp::where(var_45, var_46, var_43);
    }
    var_48 = wp::where(var_41, var_43, var_47);
    // goal = pull_target + direction * rest_length                                           <L 54>
    var_49 = wp::mul(var_48, var_33);
    var_50 = wp::add(var_pull_target, var_49);
    // corrected = q + (goal - q) * alpha                                                     <L 55>
    var_51 = wp::sub(var_50, var_28);
    var_52 = wp::mul(var_51, var_24);
    var_53 = wp::add(var_28, var_52);
    // particle_q[particle_idx] = corrected                                                   <L 56>
    wp::array_store(var_particle_q, var_2, var_53);
    // particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)                                     <L 57>
    var_57 = wp::vec_t<3, wp::float32>(var_54, var_55, var_56);
    wp::array_store(var_particle_qd, var_2, var_57);
}



extern "C" {

// Python CPU entry points
WP_API void solve_grab_distance_constraints_kernel_e27b06a6_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_solve_grab_distance_constraints_kernel_e27b06a6 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        solve_grab_distance_constraints_kernel_e27b06a6_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C

