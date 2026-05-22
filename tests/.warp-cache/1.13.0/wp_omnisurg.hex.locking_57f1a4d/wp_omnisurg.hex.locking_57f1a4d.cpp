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

struct wp_args__enforce_locked_nodes_kernel_3d4c26d0 {
    wp::array_t<wp::int32> locked_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> locked_positions;
    wp::array_t<wp::vec_t<3, wp::float32>> particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> particle_qd;
};


void _enforce_locked_nodes_kernel_3d4c26d0_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args__enforce_locked_nodes_kernel_3d4c26d0 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_locked_indices = _wp_args->locked_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> var_locked_positions = _wp_args->locked_positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q = _wp_args->particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd = _wp_args->particle_qd;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    const wp::float32 var_10 = 0.0;
    wp::vec_t<3, wp::float32> var_11;
    //---------
    // forward
    // def _enforce_locked_nodes_kernel(                                                      <L 28>
    // tid = wp.tid()                                                                         <L 34>
    var_0 = builtin_tid1d();
    // particle_idx = locked_indices[tid]                                                     <L 35>
    var_1 = wp::address(var_locked_indices, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if particle_idx < 0:                                                                   <L 36>
    var_5 = (var_2 < var_4);
    if (var_5) {
        // return                                                                             <L 37>
        return;
    }
    // particle_q[particle_idx] = locked_positions[tid]                                       <L 39>
    var_6 = wp::address(var_locked_positions, var_0);
    var_7 = wp::load(var_6);
    wp::array_store(var_particle_q, var_2, var_7);
    // particle_qd[particle_idx] = wp.vec3(0.0, 0.0, 0.0)                                     <L 40>
    var_11 = wp::vec_t<3, wp::float32>(var_8, var_9, var_10);
    wp::array_store(var_particle_qd, var_2, var_11);
}



extern "C" {

// Python CPU entry points
WP_API void _enforce_locked_nodes_kernel_3d4c26d0_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args__enforce_locked_nodes_kernel_3d4c26d0 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        _enforce_locked_nodes_kernel_3d4c26d0_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C

