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

struct wp_args_update_haptic_proxy_bafefd59 {
    wp::array_t<wp::vec_t<3, wp::float32>> center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> center_target;
    wp::array_t<wp::vec_t<3, wp::float32>> center_current;
    wp::array_t<wp::vec_t<3, wp::float32>> center_scaled_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> center_scaled;
    wp::array_t<wp::transform_t<wp::float32>> body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> body_qd;
    wp::int32 body_id;
    wp::float32 factor;
    wp::float32 position_scale;
    wp::float32 dt;
};


void update_haptic_proxy_bafefd59_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_update_haptic_proxy_bafefd59 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_prev = _wp_args->center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_target = _wp_args->center_target;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_current = _wp_args->center_current;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_scaled_prev = _wp_args->center_scaled_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_scaled = _wp_args->center_scaled;
    wp::array_t<wp::transform_t<wp::float32>> var_body_q = _wp_args->body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd = _wp_args->body_qd;
    wp::int32 var_body_id = _wp_args->body_id;
    wp::float32 var_factor = _wp_args->factor;
    wp::float32 var_position_scale = _wp_args->position_scale;
    wp::float32 var_dt = _wp_args->dt;
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
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32>* var_12;
    const wp::int32 var_13 = 0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    wp::transform_t<wp::float32>* var_17;
    wp::transform_t<wp::float32> var_18;
    wp::transform_t<wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::quat_t<wp::float32> var_21;
    wp::transform_t<wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<6, wp::float32> var_29;
    //---------
    // forward
    // def update_haptic_proxy(                                                               <L 53>
    // if wp.tid() != 0:                                                                      <L 66>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 67>
        return;
    }
    // current = wp.lerp(center_prev[0], center_target[0], factor)                            <L 69>
    var_4 = wp::address(var_center_prev, var_3);
    var_6 = wp::address(var_center_target, var_5);
    var_8 = wp::load(var_4);
    var_9 = wp::load(var_6);
    var_7 = wp::lerp(var_8, var_9, var_factor);
    // center_current[0] = current                                                            <L 70>
    wp::array_store(var_center_current, var_10, var_7);
    // center_scaled_prev[0] = center_scaled[0]                                               <L 72>
    var_12 = wp::address(var_center_scaled, var_11);
    var_14 = wp::load(var_12);
    wp::array_store(var_center_scaled_prev, var_13, var_14);
    // scaled = current * position_scale                                                      <L 73>
    var_15 = wp::mul(var_7, var_position_scale);
    // center_scaled[0] = scaled                                                              <L 74>
    wp::array_store(var_center_scaled, var_16, var_15);
    // xform = body_q[body_id]                                                                <L 76>
    var_17 = wp::address(var_body_q, var_body_id);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // prev = wp.transform_get_translation(xform)                                             <L 77>
    var_20 = wp::transform_get_translation(var_18);
    // body_q[body_id] = wp.transform(scaled, wp.transform_get_rotation(xform))               <L 78>
    var_21 = wp::transform_get_rotation(var_18);
    var_22 = wp::transform_t<wp::float32>(var_15, var_21);
    wp::array_store(var_body_q, var_body_id, var_22);
    // vel = (scaled - prev) / dt                                                             <L 80>
    var_23 = wp::sub(var_15, var_20);
    var_24 = wp::div(var_23, var_dt);
    // body_qd[body_id] = wp.spatial_vector(vel, wp.vec3f(0.0, 0.0, 0.0))                     <L 81>
    var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
    var_29 = wp::vec_t<6, wp::float32>(var_24, var_28);
    wp::array_store(var_body_qd, var_body_id, var_29);
}



void update_haptic_proxy_bafefd59_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_update_haptic_proxy_bafefd59 *_wp_args,
    wp_args_update_haptic_proxy_bafefd59 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_prev = _wp_args->center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_target = _wp_args->center_target;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_current = _wp_args->center_current;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_scaled_prev = _wp_args->center_scaled_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_center_scaled = _wp_args->center_scaled;
    wp::array_t<wp::transform_t<wp::float32>> var_body_q = _wp_args->body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd = _wp_args->body_qd;
    wp::int32 var_body_id = _wp_args->body_id;
    wp::float32 var_factor = _wp_args->factor;
    wp::float32 var_position_scale = _wp_args->position_scale;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_center_prev = _wp_adj_args->center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_center_target = _wp_adj_args->center_target;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_center_current = _wp_adj_args->center_current;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_center_scaled_prev = _wp_adj_args->center_scaled_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_center_scaled = _wp_adj_args->center_scaled;
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q = _wp_adj_args->body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_qd = _wp_adj_args->body_qd;
    wp::int32 adj_body_id = _wp_adj_args->body_id;
    wp::float32 adj_factor = _wp_adj_args->factor;
    wp::float32 adj_position_scale = _wp_adj_args->position_scale;
    wp::float32 adj_dt = _wp_adj_args->dt;
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
    const wp::int32 var_11 = 0;
    wp::vec_t<3, wp::float32>* var_12;
    const wp::int32 var_13 = 0;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    wp::transform_t<wp::float32>* var_17;
    wp::transform_t<wp::float32> var_18;
    wp::transform_t<wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::quat_t<wp::float32> var_21;
    wp::transform_t<wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 0.0;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<6, wp::float32> var_29;
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
    wp::int32 adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::int32 adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::int32 adj_16 = {};
    wp::transform_t<wp::float32> adj_17 = {};
    wp::transform_t<wp::float32> adj_18 = {};
    wp::transform_t<wp::float32> adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::quat_t<wp::float32> adj_21 = {};
    wp::transform_t<wp::float32> adj_22 = {};
    wp::vec_t<3, wp::float32> adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::vec_t<6, wp::float32> adj_29 = {};
    //---------
    // forward
    // def update_haptic_proxy(                                                               <L 53>
    // if wp.tid() != 0:                                                                      <L 66>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 67>
        goto label0;
    }
    // current = wp.lerp(center_prev[0], center_target[0], factor)                            <L 69>
    var_4 = wp::address(var_center_prev, var_3);
    var_6 = wp::address(var_center_target, var_5);
    var_8 = wp::load(var_4);
    var_9 = wp::load(var_6);
    var_7 = wp::lerp(var_8, var_9, var_factor);
    // center_current[0] = current                                                            <L 70>
    // wp::array_store(var_center_current, var_10, var_7);
    // center_scaled_prev[0] = center_scaled[0]                                               <L 72>
    var_12 = wp::address(var_center_scaled, var_11);
    var_14 = wp::load(var_12);
    // wp::array_store(var_center_scaled_prev, var_13, var_14);
    // scaled = current * position_scale                                                      <L 73>
    var_15 = wp::mul(var_7, var_position_scale);
    // center_scaled[0] = scaled                                                              <L 74>
    // wp::array_store(var_center_scaled, var_16, var_15);
    // xform = body_q[body_id]                                                                <L 76>
    var_17 = wp::address(var_body_q, var_body_id);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // prev = wp.transform_get_translation(xform)                                             <L 77>
    var_20 = wp::transform_get_translation(var_18);
    // body_q[body_id] = wp.transform(scaled, wp.transform_get_rotation(xform))               <L 78>
    var_21 = wp::transform_get_rotation(var_18);
    var_22 = wp::transform_t<wp::float32>(var_15, var_21);
    // wp::array_store(var_body_q, var_body_id, var_22);
    // vel = (scaled - prev) / dt                                                             <L 80>
    var_23 = wp::sub(var_15, var_20);
    var_24 = wp::div(var_23, var_dt);
    // body_qd[body_id] = wp.spatial_vector(vel, wp.vec3f(0.0, 0.0, 0.0))                     <L 81>
    var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
    var_29 = wp::vec_t<6, wp::float32>(var_24, var_28);
    // wp::array_store(var_body_qd, var_body_id, var_29);
    //---------
    // reverse
    wp::adj_array_store(var_body_qd, var_body_id, var_29, adj_body_qd, adj_body_id, adj_29);
    wp::adj_vec_t(var_24, var_28, adj_24, adj_28, adj_29);
    wp::adj_vec_t(var_25, var_26, var_27, adj_25, adj_26, adj_27, adj_28);
    // adj: body_qd[body_id] = wp.spatial_vector(vel, wp.vec3f(0.0, 0.0, 0.0))                <L 81>
    wp::adj_div(var_23, var_dt, adj_23, adj_dt, adj_24);
    wp::adj_sub(var_15, var_20, adj_15, adj_20, adj_23);
    // adj: vel = (scaled - prev) / dt                                                        <L 80>
    wp::adj_array_store(var_body_q, var_body_id, var_22, adj_body_q, adj_body_id, adj_22);
    wp::adj_transform_t(var_15, var_21, adj_15, adj_21, adj_22);
    wp::adj_transform_get_rotation(var_18, adj_18, adj_21);
    // adj: body_q[body_id] = wp.transform(scaled, wp.transform_get_rotation(xform))          <L 78>
    wp::adj_transform_get_translation(var_18, adj_18, adj_20);
    // adj: prev = wp.transform_get_translation(xform)                                        <L 77>
    wp::adj_copy(var_19, adj_17, adj_18);
    wp::adj_address(var_body_q, var_body_id, adj_body_q, adj_body_id, adj_17);
    // adj: xform = body_q[body_id]                                                           <L 76>
    wp::adj_array_store(var_center_scaled, var_16, var_15, adj_center_scaled, adj_16, adj_15);
    // adj: center_scaled[0] = scaled                                                         <L 74>
    wp::adj_mul(var_7, var_position_scale, adj_7, adj_position_scale, adj_15);
    // adj: scaled = current * position_scale                                                 <L 73>
    wp::adj_array_store(var_center_scaled_prev, var_13, var_14, adj_center_scaled_prev, adj_13, adj_12);
    wp::adj_address(var_center_scaled, var_11, adj_center_scaled, adj_11, adj_12);
    // adj: center_scaled_prev[0] = center_scaled[0]                                          <L 72>
    wp::adj_array_store(var_center_current, var_10, var_7, adj_center_current, adj_10, adj_7);
    // adj: center_current[0] = current                                                       <L 70>
    wp::adj_lerp(var_8, var_9, var_factor, adj_4, adj_6, adj_factor, adj_7);
    wp::adj_address(var_center_target, var_5, adj_center_target, adj_5, adj_6);
    wp::adj_address(var_center_prev, var_3, adj_center_prev, adj_3, adj_4);
    // adj: current = wp.lerp(center_prev[0], center_target[0], factor)                       <L 69>
    if (var_2) {
        label0:;
        // adj: return                                                                        <L 67>
    }
    // adj: if wp.tid() != 0:                                                                 <L 66>
    // adj: def update_haptic_proxy(                                                          <L 53>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void update_haptic_proxy_bafefd59_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_update_haptic_proxy_bafefd59 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        update_haptic_proxy_bafefd59_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void update_haptic_proxy_bafefd59_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_update_haptic_proxy_bafefd59 *_wp_args,
    wp_args_update_haptic_proxy_bafefd59 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        update_haptic_proxy_bafefd59_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_scale_position_3a4152af {
    wp::array_t<wp::vec_t<3, wp::float32>> src;
    wp::array_t<wp::vec_t<3, wp::float32>> dst;
    wp::float32 scale;
};


void scale_position_3a4152af_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_scale_position_3a4152af *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_src = _wp_args->src;
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst = _wp_args->dst;
    wp::float32 var_scale = _wp_args->scale;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    //---------
    // forward
    // def scale_position(                                                                    <L 85>
    // if wp.tid() != 0:                                                                      <L 90>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 91>
        return;
    }
    // dst[0] = src[0] * scale                                                                <L 93>
    var_4 = wp::address(var_src, var_3);
    var_6 = wp::load(var_4);
    var_5 = wp::mul(var_6, var_scale);
    wp::array_store(var_dst, var_7, var_5);
}



void scale_position_3a4152af_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_scale_position_3a4152af *_wp_args,
    wp_args_scale_position_3a4152af *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_src = _wp_args->src;
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst = _wp_args->dst;
    wp::float32 var_scale = _wp_args->scale;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src = _wp_adj_args->src;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst = _wp_adj_args->dst;
    wp::float32 adj_scale = _wp_adj_args->scale;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::vec_t<3, wp::float32>* var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    const wp::int32 var_7 = 0;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::int32 adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::int32 adj_7 = {};
    //---------
    // forward
    // def scale_position(                                                                    <L 85>
    // if wp.tid() != 0:                                                                      <L 90>
    var_0 = builtin_tid1d();
    var_2 = (var_0 != var_1);
    if (var_2) {
        // return                                                                             <L 91>
        goto label0;
    }
    // dst[0] = src[0] * scale                                                                <L 93>
    var_4 = wp::address(var_src, var_3);
    var_6 = wp::load(var_4);
    var_5 = wp::mul(var_6, var_scale);
    // wp::array_store(var_dst, var_7, var_5);
    //---------
    // reverse
    wp::adj_array_store(var_dst, var_7, var_5, adj_dst, adj_7, adj_5);
    wp::adj_mul(var_6, var_scale, adj_4, adj_scale, adj_5);
    wp::adj_address(var_src, var_3, adj_src, adj_3, adj_4);
    // adj: dst[0] = src[0] * scale                                                           <L 93>
    if (var_2) {
        label0:;
        // adj: return                                                                        <L 91>
    }
    // adj: if wp.tid() != 0:                                                                 <L 90>
    // adj: def scale_position(                                                               <L 85>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void scale_position_3a4152af_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_scale_position_3a4152af *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        scale_position_3a4152af_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void scale_position_3a4152af_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_scale_position_3a4152af *_wp_args,
    wp_args_scale_position_3a4152af *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        scale_position_3a4152af_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

