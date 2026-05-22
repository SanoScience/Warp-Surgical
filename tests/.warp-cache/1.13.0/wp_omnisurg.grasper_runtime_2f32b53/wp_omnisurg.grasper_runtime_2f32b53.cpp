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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/grasper_runtime.py:45
static wp::vec_t<3, wp::float32> _rotate_point_about_jaw_axis_0(
    wp::vec_t<3, wp::float32> var_local_point,
    wp::float32 var_jaw_sign,
    wp::float32 var_jaw_angle)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::quat_t<wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    //---------
    // forward
    // def _rotate_point_about_jaw_axis(local_point: wp.vec3f, jaw_sign: float, jaw_angle: float) -> wp.vec3f:       <L 46>
    // if jaw_sign == 0.0:                                                                    <L 47>
    var_1 = (var_jaw_sign == var_0);
    if (var_1) {
        // return local_point                                                                 <L 48>
        return var_local_point;
    }
    // jaw_rotation = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), jaw_sign * jaw_angle)       <L 50>
    var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
    var_6 = wp::mul(var_jaw_sign, var_jaw_angle);
    var_7 = wp::quat_from_axis_angle(var_5, var_6);
    // return wp.quat_rotate(jaw_rotation, local_point)                                       <L 51>
    var_8 = wp::quat_rotate(var_7, var_local_point);
    return var_8;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/grasper_runtime.py:45
static void adj__rotate_point_about_jaw_axis_0(
    wp::vec_t<3, wp::float32> var_local_point,
    wp::float32 var_jaw_sign,
    wp::float32 var_jaw_angle,
    wp::vec_t<3, wp::float32> & adj_local_point,
    wp::float32 & adj_jaw_sign,
    wp::float32 & adj_jaw_angle,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 1.0;
    const wp::float32 var_3 = 0.0;
    const wp::float32 var_4 = 0.0;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    wp::quat_t<wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::float32 adj_6 = {};
    wp::quat_t<wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    //---------
    // forward
    // def _rotate_point_about_jaw_axis(local_point: wp.vec3f, jaw_sign: float, jaw_angle: float) -> wp.vec3f:       <L 46>
    // if jaw_sign == 0.0:                                                                    <L 47>
    var_1 = (var_jaw_sign == var_0);
    if (var_1) {
        // return local_point                                                                 <L 48>
        goto label0;
    }
    // jaw_rotation = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), jaw_sign * jaw_angle)       <L 50>
    var_5 = wp::vec_t<3, wp::float32>(var_2, var_3, var_4);
    var_6 = wp::mul(var_jaw_sign, var_jaw_angle);
    var_7 = wp::quat_from_axis_angle(var_5, var_6);
    // return wp.quat_rotate(jaw_rotation, local_point)                                       <L 51>
    var_8 = wp::quat_rotate(var_7, var_local_point);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_8 += adj_ret;
    wp::adj_quat_rotate(var_7, var_local_point, adj_7, adj_local_point, adj_8);
    // adj: return wp.quat_rotate(jaw_rotation, local_point)                                  <L 51>
    wp::adj_quat_from_axis_angle(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_mul(var_jaw_sign, var_jaw_angle, adj_jaw_sign, adj_jaw_angle, adj_6);
    wp::adj_vec_t(var_2, var_3, var_4, adj_2, adj_3, adj_4, adj_5);
    // adj: jaw_rotation = wp.quat_from_axis_angle(wp.vec3f(1.0, 0.0, 0.0), jaw_sign * jaw_angle)  <L 50>
    if (var_1) {
        label0:;
        adj_local_point += adj_ret;
        // adj: return local_point                                                            <L 48>
    }
    // adj: if jaw_sign == 0.0:                                                               <L 47>
    // adj: def _rotate_point_about_jaw_axis(local_point: wp.vec3f, jaw_sign: float, jaw_angle: float) -> wp.vec3f:  <L 46>
    return;
}

struct wp_args_interpolate_grasper_position_592d4f9e {
    wp::array_t<wp::vec_t<3, wp::float32>> root_position_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> root_position_target;
    wp::array_t<wp::vec_t<3, wp::float32>> root_position_current;
    wp::float32 factor;
};


void interpolate_grasper_position_592d4f9e_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_interpolate_grasper_position_592d4f9e *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position_prev = _wp_args->root_position_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position_target = _wp_args->root_position_target;
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position_current = _wp_args->root_position_current;
    wp::float32 var_factor = _wp_args->factor;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::vec_t<3, wp::float32>* var_4;
    const wp::int32 var_5 = 0;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    //---------
    // forward
    // def interpolate_grasper_position(                                                      <L 33>
    // if wp.tid() != 0:                                                                      <L 39>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 40>
        return;
    }
    // root_position_current[0] = wp.lerp(root_position_prev[0], root_position_target[0], factor)       <L 42>
    var_4 = wp::address(var_root_position_prev, var_3);
    var_6 = wp::address(var_root_position_target, var_5);
    var_8 = wp::load(var_4);
    var_9 = wp::load(var_6);
    var_7 = wp::lerp(var_8, var_9, var_factor);
    wp::array_store(var_root_position_current, var_10, var_7);
}



void interpolate_grasper_position_592d4f9e_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_interpolate_grasper_position_592d4f9e *_wp_args,
    wp_args_interpolate_grasper_position_592d4f9e *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position_prev = _wp_args->root_position_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position_target = _wp_args->root_position_target;
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position_current = _wp_args->root_position_current;
    wp::float32 var_factor = _wp_args->factor;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_root_position_prev = _wp_adj_args->root_position_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_root_position_target = _wp_adj_args->root_position_target;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_root_position_current = _wp_adj_args->root_position_current;
    wp::float32 adj_factor = _wp_adj_args->factor;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::vec_t<3, wp::float32>* var_4;
    const wp::int32 var_5 = 0;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 0;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::int32 adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::int32 adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::int32 adj_10 = {};
    //---------
    // forward
    // def interpolate_grasper_position(                                                      <L 33>
    // if wp.tid() != 0:                                                                      <L 39>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 40>
        goto label0;
    }
    // root_position_current[0] = wp.lerp(root_position_prev[0], root_position_target[0], factor)       <L 42>
    var_4 = wp::address(var_root_position_prev, var_3);
    var_6 = wp::address(var_root_position_target, var_5);
    var_8 = wp::load(var_4);
    var_9 = wp::load(var_6);
    var_7 = wp::lerp(var_8, var_9, var_factor);
    // wp::array_store(var_root_position_current, var_10, var_7);
    //---------
    // reverse
    wp::adj_array_store(var_root_position_current, var_10, var_7, adj_root_position_current, adj_10, adj_7);
    wp::adj_lerp(var_8, var_9, var_factor, adj_4, adj_6, adj_factor, adj_7);
    wp::adj_address(var_root_position_target, var_5, adj_root_position_target, adj_5, adj_6);
    wp::adj_address(var_root_position_prev, var_3, adj_root_position_prev, adj_3, adj_4);
    // adj: root_position_current[0] = wp.lerp(root_position_prev[0], root_position_target[0], factor)  <L 42>
    if (var_2) {
        label0:;
        // adj: return                                                                        <L 40>
    }
    // adj: if wp.tid() != 0:                                                                 <L 39>
    // adj: def interpolate_grasper_position(                                                 <L 33>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void interpolate_grasper_position_592d4f9e_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_interpolate_grasper_position_592d4f9e *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        interpolate_grasper_position_592d4f9e_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void interpolate_grasper_position_592d4f9e_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_interpolate_grasper_position_592d4f9e *_wp_args,
    wp_args_interpolate_grasper_position_592d4f9e *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        interpolate_grasper_position_592d4f9e_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_transform_grasper_points_64ad239f {
    wp::array_t<wp::vec_t<3, wp::float32>> root_position;
    wp::array_t<wp::quat_t<wp::float32>> root_rotation;
    wp::array_t<wp::float32> jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> world_points;
    wp::float32 jaw_sign;
};


void transform_grasper_points_64ad239f_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_transform_grasper_points_64ad239f *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position = _wp_args->root_position;
    wp::array_t<wp::quat_t<wp::float32>> var_root_rotation = _wp_args->root_rotation;
    wp::array_t<wp::float32> var_jaw_angle = _wp_args->jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_bind_matrix = _wp_args->bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_points = _wp_args->local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> var_world_points = _wp_args->world_points;
    wp::float32 var_jaw_sign = _wp_args->jaw_sign;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::mat_t<4, 4, wp::float32>* var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::mat_t<4, 4, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32>* var_12;
    const wp::int32 var_13 = 0;
    wp::quat_t<wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    //---------
    // forward
    // def transform_grasper_points(                                                          <L 55>
    // tid = wp.tid()                                                                         <L 64>
    var_0 = builtin_tid1d();
    // local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])       <L 65>
    var_1 = wp::address(var_local_points, var_0);
    var_3 = wp::address(var_jaw_angle, var_2);
    var_5 = wp::load(var_1);
    var_6 = wp::load(var_3);
    var_4 = _rotate_point_about_jaw_axis_0(var_5, var_jaw_sign, var_6);
    // bind_space_point = wp.transform_point(bind_matrix[0], local_point)                     <L 66>
    var_8 = wp::address(var_bind_matrix, var_7);
    var_10 = wp::load(var_8);
    var_9 = wp::transform_point(var_10, var_4);
    // world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)       <L 67>
    var_12 = wp::address(var_root_position, var_11);
    var_14 = wp::address(var_root_rotation, var_13);
    var_16 = wp::load(var_14);
    var_15 = wp::quat_rotate(var_16, var_9);
    var_18 = wp::load(var_12);
    var_17 = wp::add(var_18, var_15);
    wp::array_store(var_world_points, var_0, var_17);
}



void transform_grasper_points_64ad239f_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_transform_grasper_points_64ad239f *_wp_args,
    wp_args_transform_grasper_points_64ad239f *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position = _wp_args->root_position;
    wp::array_t<wp::quat_t<wp::float32>> var_root_rotation = _wp_args->root_rotation;
    wp::array_t<wp::float32> var_jaw_angle = _wp_args->jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_bind_matrix = _wp_args->bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_points = _wp_args->local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> var_world_points = _wp_args->world_points;
    wp::float32 var_jaw_sign = _wp_args->jaw_sign;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_root_position = _wp_adj_args->root_position;
    wp::array_t<wp::quat_t<wp::float32>> adj_root_rotation = _wp_adj_args->root_rotation;
    wp::array_t<wp::float32> adj_jaw_angle = _wp_adj_args->jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> adj_bind_matrix = _wp_adj_args->bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_points = _wp_adj_args->local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_world_points = _wp_adj_args->world_points;
    wp::float32 adj_jaw_sign = _wp_adj_args->jaw_sign;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::mat_t<4, 4, wp::float32>* var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::mat_t<4, 4, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32>* var_12;
    const wp::int32 var_13 = 0;
    wp::quat_t<wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::mat_t<4, 4, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::mat_t<4, 4, wp::float32> adj_10 = {};
    wp::int32 adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::int32 adj_13 = {};
    wp::quat_t<wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::quat_t<wp::float32> adj_16 = {};
    wp::vec_t<3, wp::float32> adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    //---------
    // forward
    // def transform_grasper_points(                                                          <L 55>
    // tid = wp.tid()                                                                         <L 64>
    var_0 = builtin_tid1d();
    // local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])       <L 65>
    var_1 = wp::address(var_local_points, var_0);
    var_3 = wp::address(var_jaw_angle, var_2);
    var_5 = wp::load(var_1);
    var_6 = wp::load(var_3);
    var_4 = _rotate_point_about_jaw_axis_0(var_5, var_jaw_sign, var_6);
    // bind_space_point = wp.transform_point(bind_matrix[0], local_point)                     <L 66>
    var_8 = wp::address(var_bind_matrix, var_7);
    var_10 = wp::load(var_8);
    var_9 = wp::transform_point(var_10, var_4);
    // world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)       <L 67>
    var_12 = wp::address(var_root_position, var_11);
    var_14 = wp::address(var_root_rotation, var_13);
    var_16 = wp::load(var_14);
    var_15 = wp::quat_rotate(var_16, var_9);
    var_18 = wp::load(var_12);
    var_17 = wp::add(var_18, var_15);
    // wp::array_store(var_world_points, var_0, var_17);
    //---------
    // reverse
    wp::adj_array_store(var_world_points, var_0, var_17, adj_world_points, adj_0, adj_17);
    wp::adj_add(var_18, var_15, adj_12, adj_15, adj_17);
    wp::adj_quat_rotate(var_16, var_9, adj_14, adj_9, adj_15);
    wp::adj_address(var_root_rotation, var_13, adj_root_rotation, adj_13, adj_14);
    wp::adj_address(var_root_position, var_11, adj_root_position, adj_11, adj_12);
    // adj: world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)  <L 67>
    wp::adj_transform_point(var_10, var_4, adj_8, adj_4, adj_9);
    wp::adj_address(var_bind_matrix, var_7, adj_bind_matrix, adj_7, adj_8);
    // adj: bind_space_point = wp.transform_point(bind_matrix[0], local_point)                <L 66>
    adj__rotate_point_about_jaw_axis_0(var_5, var_jaw_sign, var_6, adj_1, adj_jaw_sign, adj_3, adj_4);
    wp::adj_address(var_jaw_angle, var_2, adj_jaw_angle, adj_2, adj_3);
    wp::adj_address(var_local_points, var_0, adj_local_points, adj_0, adj_1);
    // adj: local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])  <L 65>
    // adj: tid = wp.tid()                                                                    <L 64>
    // adj: def transform_grasper_points(                                                     <L 55>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void transform_grasper_points_64ad239f_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_transform_grasper_points_64ad239f *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        transform_grasper_points_64ad239f_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void transform_grasper_points_64ad239f_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_transform_grasper_points_64ad239f *_wp_args,
    wp_args_transform_grasper_points_64ad239f *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        transform_grasper_points_64ad239f_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_transform_grasper_spheres_a412fcca {
    wp::array_t<wp::vec_t<3, wp::float32>> root_position;
    wp::array_t<wp::quat_t<wp::float32>> root_rotation;
    wp::array_t<wp::float32> grip_command;
    wp::array_t<wp::float32> jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> world_points;
    wp::array_t<wp::float32> base_radii;
    wp::array_t<wp::float32> active_radii;
    wp::float32 jaw_sign;
};


void transform_grasper_spheres_a412fcca_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_transform_grasper_spheres_a412fcca *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position = _wp_args->root_position;
    wp::array_t<wp::quat_t<wp::float32>> var_root_rotation = _wp_args->root_rotation;
    wp::array_t<wp::float32> var_grip_command = _wp_args->grip_command;
    wp::array_t<wp::float32> var_jaw_angle = _wp_args->jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_bind_matrix = _wp_args->bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_points = _wp_args->local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> var_world_points = _wp_args->world_points;
    wp::array_t<wp::float32> var_base_radii = _wp_args->base_radii;
    wp::array_t<wp::float32> var_active_radii = _wp_args->active_radii;
    wp::float32 var_jaw_sign = _wp_args->jaw_sign;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::mat_t<4, 4, wp::float32>* var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::mat_t<4, 4, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32>* var_12;
    const wp::int32 var_13 = 0;
    wp::quat_t<wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 0;
    wp::float32* var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.0;
    wp::float32* var_25;
    wp::float32 var_26;
    //---------
    // forward
    // def transform_grasper_spheres(                                                         <L 71>
    // tid = wp.tid()                                                                         <L 83>
    var_0 = builtin_tid1d();
    // local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])       <L 84>
    var_1 = wp::address(var_local_points, var_0);
    var_3 = wp::address(var_jaw_angle, var_2);
    var_5 = wp::load(var_1);
    var_6 = wp::load(var_3);
    var_4 = _rotate_point_about_jaw_axis_0(var_5, var_jaw_sign, var_6);
    // bind_space_point = wp.transform_point(bind_matrix[0], local_point)                     <L 85>
    var_8 = wp::address(var_bind_matrix, var_7);
    var_10 = wp::load(var_8);
    var_9 = wp::transform_point(var_10, var_4);
    // world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)       <L 86>
    var_12 = wp::address(var_root_position, var_11);
    var_14 = wp::address(var_root_rotation, var_13);
    var_16 = wp::load(var_14);
    var_15 = wp::quat_rotate(var_16, var_9);
    var_18 = wp::load(var_12);
    var_17 = wp::add(var_18, var_15);
    wp::array_store(var_world_points, var_0, var_17);
    // if grip_command[0] < 0.0:                                                              <L 88>
    var_20 = wp::address(var_grip_command, var_19);
    var_23 = wp::load(var_20);
    var_22 = (var_23 < var_21);
    if (var_22) {
        // active_radii[tid] = 0.0                                                            <L 89>
        wp::array_store(var_active_radii, var_0, var_24);
    }
    if (!var_22) {
        // active_radii[tid] = base_radii[tid]                                                <L 91>
        var_25 = wp::address(var_base_radii, var_0);
        var_26 = wp::load(var_25);
        wp::array_store(var_active_radii, var_0, var_26);
    }
}



void transform_grasper_spheres_a412fcca_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_transform_grasper_spheres_a412fcca *_wp_args,
    wp_args_transform_grasper_spheres_a412fcca *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_root_position = _wp_args->root_position;
    wp::array_t<wp::quat_t<wp::float32>> var_root_rotation = _wp_args->root_rotation;
    wp::array_t<wp::float32> var_grip_command = _wp_args->grip_command;
    wp::array_t<wp::float32> var_jaw_angle = _wp_args->jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> var_bind_matrix = _wp_args->bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> var_local_points = _wp_args->local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> var_world_points = _wp_args->world_points;
    wp::array_t<wp::float32> var_base_radii = _wp_args->base_radii;
    wp::array_t<wp::float32> var_active_radii = _wp_args->active_radii;
    wp::float32 var_jaw_sign = _wp_args->jaw_sign;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_root_position = _wp_adj_args->root_position;
    wp::array_t<wp::quat_t<wp::float32>> adj_root_rotation = _wp_adj_args->root_rotation;
    wp::array_t<wp::float32> adj_grip_command = _wp_adj_args->grip_command;
    wp::array_t<wp::float32> adj_jaw_angle = _wp_adj_args->jaw_angle;
    wp::array_t<wp::mat_t<4, 4, wp::float32>> adj_bind_matrix = _wp_adj_args->bind_matrix;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_local_points = _wp_adj_args->local_points;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_world_points = _wp_adj_args->world_points;
    wp::array_t<wp::float32> adj_base_radii = _wp_adj_args->base_radii;
    wp::array_t<wp::float32> adj_active_radii = _wp_adj_args->active_radii;
    wp::float32 adj_jaw_sign = _wp_adj_args->jaw_sign;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    const wp::int32 var_2 = 0;
    wp::float32* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::float32 var_6;
    const wp::int32 var_7 = 0;
    wp::mat_t<4, 4, wp::float32>* var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::mat_t<4, 4, wp::float32> var_10;
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32>* var_12;
    const wp::int32 var_13 = 0;
    wp::quat_t<wp::float32>* var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::quat_t<wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    const wp::int32 var_19 = 0;
    wp::float32* var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::float32 var_23;
    const wp::float32 var_24 = 0.0;
    wp::float32* var_25;
    wp::float32 var_26;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::float32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::mat_t<4, 4, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::mat_t<4, 4, wp::float32> adj_10 = {};
    wp::int32 adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::int32 adj_13 = {};
    wp::quat_t<wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::quat_t<wp::float32> adj_16 = {};
    wp::vec_t<3, wp::float32> adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::int32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    bool adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    //---------
    // forward
    // def transform_grasper_spheres(                                                         <L 71>
    // tid = wp.tid()                                                                         <L 83>
    var_0 = builtin_tid1d();
    // local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])       <L 84>
    var_1 = wp::address(var_local_points, var_0);
    var_3 = wp::address(var_jaw_angle, var_2);
    var_5 = wp::load(var_1);
    var_6 = wp::load(var_3);
    var_4 = _rotate_point_about_jaw_axis_0(var_5, var_jaw_sign, var_6);
    // bind_space_point = wp.transform_point(bind_matrix[0], local_point)                     <L 85>
    var_8 = wp::address(var_bind_matrix, var_7);
    var_10 = wp::load(var_8);
    var_9 = wp::transform_point(var_10, var_4);
    // world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)       <L 86>
    var_12 = wp::address(var_root_position, var_11);
    var_14 = wp::address(var_root_rotation, var_13);
    var_16 = wp::load(var_14);
    var_15 = wp::quat_rotate(var_16, var_9);
    var_18 = wp::load(var_12);
    var_17 = wp::add(var_18, var_15);
    // wp::array_store(var_world_points, var_0, var_17);
    // if grip_command[0] < 0.0:                                                              <L 88>
    var_20 = wp::address(var_grip_command, var_19);
    var_23 = wp::load(var_20);
    var_22 = (var_23 < var_21);
    if (var_22) {
        // active_radii[tid] = 0.0                                                            <L 89>
        // wp::array_store(var_active_radii, var_0, var_24);
    }
    if (!var_22) {
        // active_radii[tid] = base_radii[tid]                                                <L 91>
        var_25 = wp::address(var_base_radii, var_0);
        var_26 = wp::load(var_25);
        // wp::array_store(var_active_radii, var_0, var_26);
    }
    //---------
    // reverse
    if (!var_22) {
        wp::adj_array_store(var_active_radii, var_0, var_26, adj_active_radii, adj_0, adj_25);
        wp::adj_address(var_base_radii, var_0, adj_base_radii, adj_0, adj_25);
        // adj: active_radii[tid] = base_radii[tid]                                           <L 91>
    }
    if (var_22) {
        wp::adj_array_store(var_active_radii, var_0, var_24, adj_active_radii, adj_0, adj_24);
        // adj: active_radii[tid] = 0.0                                                       <L 89>
    }
    wp::adj_address(var_grip_command, var_19, adj_grip_command, adj_19, adj_20);
    // adj: if grip_command[0] < 0.0:                                                         <L 88>
    wp::adj_array_store(var_world_points, var_0, var_17, adj_world_points, adj_0, adj_17);
    wp::adj_add(var_18, var_15, adj_12, adj_15, adj_17);
    wp::adj_quat_rotate(var_16, var_9, adj_14, adj_9, adj_15);
    wp::adj_address(var_root_rotation, var_13, adj_root_rotation, adj_13, adj_14);
    wp::adj_address(var_root_position, var_11, adj_root_position, adj_11, adj_12);
    // adj: world_points[tid] = root_position[0] + wp.quat_rotate(root_rotation[0], bind_space_point)  <L 86>
    wp::adj_transform_point(var_10, var_4, adj_8, adj_4, adj_9);
    wp::adj_address(var_bind_matrix, var_7, adj_bind_matrix, adj_7, adj_8);
    // adj: bind_space_point = wp.transform_point(bind_matrix[0], local_point)                <L 85>
    adj__rotate_point_about_jaw_axis_0(var_5, var_jaw_sign, var_6, adj_1, adj_jaw_sign, adj_3, adj_4);
    wp::adj_address(var_jaw_angle, var_2, adj_jaw_angle, adj_2, adj_3);
    wp::adj_address(var_local_points, var_0, adj_local_points, adj_0, adj_1);
    // adj: local_point = _rotate_point_about_jaw_axis(local_points[tid], jaw_sign, jaw_angle[0])  <L 84>
    // adj: tid = wp.tid()                                                                    <L 83>
    // adj: def transform_grasper_spheres(                                                    <L 71>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void transform_grasper_spheres_a412fcca_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_transform_grasper_spheres_a412fcca *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        transform_grasper_spheres_a412fcca_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void transform_grasper_spheres_a412fcca_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_transform_grasper_spheres_a412fcca *_wp_args,
    wp_args_transform_grasper_spheres_a412fcca *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        transform_grasper_spheres_a412fcca_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_update_jaw_angle_state_dee1e09a {
    wp::array_t<wp::float32> grip_command;
    wp::array_t<wp::float32> jaw_angle;
    wp::float32 dt;
    wp::float32 jaw_open_angle;
    wp::float32 jaw_closed_angle;
    wp::float32 jaw_response;
};


void update_jaw_angle_state_dee1e09a_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_update_jaw_angle_state_dee1e09a *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_grip_command = _wp_args->grip_command;
    wp::array_t<wp::float32> var_jaw_angle = _wp_args->jaw_angle;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_jaw_open_angle = _wp_args->jaw_open_angle;
    wp::float32 var_jaw_closed_angle = _wp_args->jaw_closed_angle;
    wp::float32 var_jaw_response = _wp_args->jaw_response;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::float32* var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 1.0;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1.0;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32* var_22;
    const wp::int32 var_23 = 0;
    wp::float32* var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    //---------
    // forward
    // def update_jaw_angle_state(                                                            <L 10>
    // if wp.tid() != 0:                                                                      <L 18>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 19>
        return;
    }
    // grip = grip_command[0]                                                                 <L 21>
    var_4 = wp::address(var_grip_command, var_3);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // if grip < 0.0:                                                                         <L 22>
    var_8 = (var_5 < var_7);
    if (var_8) {
        // target_angle = jaw_open_angle                                                      <L 23>
        var_9 = wp::copy(var_jaw_open_angle);
    }
    if (!var_8) {
        // closure = wp.min(wp.max(grip, 0.0), 1.0)                                           <L 25>
        var_11 = wp::max(var_5, var_10);
        var_13 = wp::min(var_11, var_12);
        // target_angle = jaw_open_angle + closure * (jaw_closed_angle - jaw_open_angle)       <L 26>
        var_14 = wp::sub(var_jaw_closed_angle, var_jaw_open_angle);
        var_15 = wp::mul(var_13, var_14);
        var_16 = wp::add(var_jaw_open_angle, var_15);
    }
    var_17 = wp::where(var_8, var_9, var_16);
    // blend = wp.min(1.0, dt * jaw_response)                                                 <L 28>
    var_19 = wp::mul(var_dt, var_jaw_response);
    var_20 = wp::min(var_18, var_19);
    // jaw_angle[0] = jaw_angle[0] + (target_angle - jaw_angle[0]) * blend                    <L 29>
    var_22 = wp::address(var_jaw_angle, var_21);
    var_24 = wp::address(var_jaw_angle, var_23);
    var_26 = wp::load(var_24);
    var_25 = wp::sub(var_17, var_26);
    var_27 = wp::mul(var_25, var_20);
    var_29 = wp::load(var_22);
    var_28 = wp::add(var_29, var_27);
    wp::array_store(var_jaw_angle, var_30, var_28);
}



void update_jaw_angle_state_dee1e09a_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_update_jaw_angle_state_dee1e09a *_wp_args,
    wp_args_update_jaw_angle_state_dee1e09a *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_grip_command = _wp_args->grip_command;
    wp::array_t<wp::float32> var_jaw_angle = _wp_args->jaw_angle;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_jaw_open_angle = _wp_args->jaw_open_angle;
    wp::float32 var_jaw_closed_angle = _wp_args->jaw_closed_angle;
    wp::float32 var_jaw_response = _wp_args->jaw_response;
    wp::array_t<wp::float32> adj_grip_command = _wp_adj_args->grip_command;
    wp::array_t<wp::float32> adj_jaw_angle = _wp_adj_args->jaw_angle;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::float32 adj_jaw_open_angle = _wp_adj_args->jaw_open_angle;
    wp::float32 adj_jaw_closed_angle = _wp_adj_args->jaw_closed_angle;
    wp::float32 adj_jaw_response = _wp_adj_args->jaw_response;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::float32* var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    const wp::float32 var_7 = 0.0;
    bool var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 1.0;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    const wp::float32 var_18 = 1.0;
    wp::float32 var_19;
    wp::float32 var_20;
    const wp::int32 var_21 = 0;
    wp::float32* var_22;
    const wp::int32 var_23 = 0;
    wp::float32* var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 0;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::int32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    bool adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::int32 adj_30 = {};
    //---------
    // forward
    // def update_jaw_angle_state(                                                            <L 10>
    // if wp.tid() != 0:                                                                      <L 18>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 19>
        goto label0;
    }
    // grip = grip_command[0]                                                                 <L 21>
    var_4 = wp::address(var_grip_command, var_3);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // if grip < 0.0:                                                                         <L 22>
    var_8 = (var_5 < var_7);
    if (var_8) {
        // target_angle = jaw_open_angle                                                      <L 23>
        var_9 = wp::copy(var_jaw_open_angle);
    }
    if (!var_8) {
        // closure = wp.min(wp.max(grip, 0.0), 1.0)                                           <L 25>
        var_11 = wp::max(var_5, var_10);
        var_13 = wp::min(var_11, var_12);
        // target_angle = jaw_open_angle + closure * (jaw_closed_angle - jaw_open_angle)       <L 26>
        var_14 = wp::sub(var_jaw_closed_angle, var_jaw_open_angle);
        var_15 = wp::mul(var_13, var_14);
        var_16 = wp::add(var_jaw_open_angle, var_15);
    }
    var_17 = wp::where(var_8, var_9, var_16);
    // blend = wp.min(1.0, dt * jaw_response)                                                 <L 28>
    var_19 = wp::mul(var_dt, var_jaw_response);
    var_20 = wp::min(var_18, var_19);
    // jaw_angle[0] = jaw_angle[0] + (target_angle - jaw_angle[0]) * blend                    <L 29>
    var_22 = wp::address(var_jaw_angle, var_21);
    var_24 = wp::address(var_jaw_angle, var_23);
    var_26 = wp::load(var_24);
    var_25 = wp::sub(var_17, var_26);
    var_27 = wp::mul(var_25, var_20);
    var_29 = wp::load(var_22);
    var_28 = wp::add(var_29, var_27);
    // wp::array_store(var_jaw_angle, var_30, var_28);
    //---------
    // reverse
    wp::adj_array_store(var_jaw_angle, var_30, var_28, adj_jaw_angle, adj_30, adj_28);
    wp::adj_add(var_29, var_27, adj_22, adj_27, adj_28);
    wp::adj_mul(var_25, var_20, adj_25, adj_20, adj_27);
    wp::adj_sub(var_17, var_26, adj_17, adj_24, adj_25);
    wp::adj_address(var_jaw_angle, var_23, adj_jaw_angle, adj_23, adj_24);
    wp::adj_address(var_jaw_angle, var_21, adj_jaw_angle, adj_21, adj_22);
    // adj: jaw_angle[0] = jaw_angle[0] + (target_angle - jaw_angle[0]) * blend               <L 29>
    wp::adj_min(var_18, var_19, adj_18, adj_19, adj_20);
    wp::adj_mul(var_dt, var_jaw_response, adj_dt, adj_jaw_response, adj_19);
    // adj: blend = wp.min(1.0, dt * jaw_response)                                            <L 28>
    wp::adj_where(var_8, var_9, var_16, adj_8, adj_9, adj_16, adj_17);
    if (!var_8) {
        wp::adj_add(var_jaw_open_angle, var_15, adj_jaw_open_angle, adj_15, adj_16);
        wp::adj_mul(var_13, var_14, adj_13, adj_14, adj_15);
        wp::adj_sub(var_jaw_closed_angle, var_jaw_open_angle, adj_jaw_closed_angle, adj_jaw_open_angle, adj_14);
        // adj: target_angle = jaw_open_angle + closure * (jaw_closed_angle - jaw_open_angle)  <L 26>
        wp::adj_min(var_11, var_12, adj_11, adj_12, adj_13);
        wp::adj_max(var_5, var_10, adj_5, adj_10, adj_11);
        // adj: closure = wp.min(wp.max(grip, 0.0), 1.0)                                      <L 25>
    }
    if (var_8) {
        wp::adj_copy(var_jaw_open_angle, adj_jaw_open_angle, adj_9);
        // adj: target_angle = jaw_open_angle                                                 <L 23>
    }
    // adj: if grip < 0.0:                                                                    <L 22>
    wp::adj_copy(var_6, adj_4, adj_5);
    wp::adj_address(var_grip_command, var_3, adj_grip_command, adj_3, adj_4);
    // adj: grip = grip_command[0]                                                            <L 21>
    if (var_2) {
        label0:;
        // adj: return                                                                        <L 19>
    }
    // adj: if wp.tid() != 0:                                                                 <L 18>
    // adj: def update_jaw_angle_state(                                                       <L 10>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void update_jaw_angle_state_dee1e09a_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_update_jaw_angle_state_dee1e09a *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        update_jaw_angle_state_dee1e09a_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void update_jaw_angle_state_dee1e09a_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_update_jaw_angle_state_dee1e09a *_wp_args,
    wp_args_update_jaw_angle_state_dee1e09a *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        update_jaw_angle_state_dee1e09a_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

