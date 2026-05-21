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


// /home/pkorzeniowsk/Projects/newton/newton-1.0/newton/_src/solvers/solver.py:51
static void integrate_rigid_body_0(
    wp::transform_t<wp::float32> var_q,
    wp::vec_t<6, wp::float32> var_qd,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<3, wp::float32> var_com,
    wp::mat_t<3, 3, wp::float32> var_inertia,
    wp::float32 var_inv_mass,
    wp::mat_t<3, 3, wp::float32> var_inv_inertia,
    wp::vec_t<3, wp::float32> var_gravity,
    wp::float32 var_angular_damping,
    wp::float32 var_dt,
    wp::transform_t<wp::float32> & ret_0,
    wp::vec_t<6, wp::float32> & ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    wp::quat_t<wp::float32> var_26;
    wp::quat_t<wp::float32> var_27;
    const wp::float32 var_28 = 0.5;
    wp::quat_t<wp::float32> var_29;
    wp::quat_t<wp::float32> var_30;
    wp::quat_t<wp::float32> var_31;
    wp::quat_t<wp::float32> var_32;
    const wp::float32 var_33 = 1.0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::transform_t<wp::float32> var_39;
    wp::vec_t<6, wp::float32> var_40;
    //---------
    // forward
    // def integrate_rigid_body(                                                              <L 52>
    // x0 = wp.transform_get_translation(q)                                                   <L 65>
    var_0 = wp::transform_get_translation(var_q);
    // r0 = wp.transform_get_rotation(q)                                                      <L 66>
    var_1 = wp::transform_get_rotation(var_q);
    // w0 = wp.spatial_bottom(qd)                                                             <L 69>
    var_2 = wp::spatial_bottom(var_qd);
    // v0 = wp.spatial_top(qd)                                                                <L 70>
    var_3 = wp::spatial_top(var_qd);
    // t0 = wp.spatial_bottom(f)                                                              <L 73>
    var_4 = wp::spatial_bottom(var_f);
    // f0 = wp.spatial_top(f)                                                                 <L 74>
    var_5 = wp::spatial_top(var_f);
    // x_com = x0 + wp.quat_rotate(r0, com)                                                   <L 76>
    var_6 = wp::quat_rotate(var_1, var_com);
    var_7 = wp::add(var_0, var_6);
    // v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt                        <L 79>
    var_8 = wp::mul(var_5, var_inv_mass);
    var_9 = wp::nonzero(var_inv_mass);
    var_10 = wp::mul(var_gravity, var_9);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_11, var_dt);
    var_13 = wp::add(var_3, var_12);
    // x1 = x_com + v1 * dt                                                                   <L 80>
    var_14 = wp::mul(var_13, var_dt);
    var_15 = wp::add(var_7, var_14);
    // wb = wp.quat_rotate_inv(r0, w0)                                                        <L 83>
    var_16 = wp::quat_rotate_inv(var_1, var_2);
    // tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces        <L 84>
    var_17 = wp::quat_rotate_inv(var_1, var_4);
    var_18 = wp::mul(var_inertia, var_16);
    var_19 = wp::cross(var_16, var_18);
    var_20 = wp::sub(var_17, var_19);
    // w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)                                    <L 86>
    var_21 = wp::mul(var_inv_inertia, var_20);
    var_22 = wp::mul(var_21, var_dt);
    var_23 = wp::add(var_16, var_22);
    var_24 = wp::quat_rotate(var_1, var_23);
    // r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)                               <L 87>
    var_26 = wp::quat_t<wp::float32>(var_24, var_25);
    var_27 = wp::mul(var_26, var_1);
    var_29 = wp::mul(var_27, var_28);
    var_30 = wp::mul(var_29, var_dt);
    var_31 = wp::add(var_1, var_30);
    var_32 = wp::normalize(var_31);
    // w1 *= 1.0 - angular_damping * dt                                                       <L 90>
    var_34 = wp::mul(var_angular_damping, var_dt);
    var_35 = wp::sub(var_33, var_34);
    var_36 = wp::mul(var_24, var_35);
    // q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)                                 <L 92>
    var_37 = wp::quat_rotate(var_32, var_com);
    var_38 = wp::sub(var_15, var_37);
    var_39 = wp::transform_t<wp::float32>(var_38, var_32);
    // qd_new = wp.spatial_vector(v1, w1)                                                     <L 93>
    var_40 = wp::vec_t<6, wp::float32>(var_13, var_36);
    // return q_new, qd_new                                                                   <L 95>
    ret_0 = var_39;
    ret_1 = var_40;
    return;
}


// /home/pkorzeniowsk/Projects/newton/newton-1.0/newton/_src/solvers/solver.py:51
static void adj_integrate_rigid_body_0(
    wp::transform_t<wp::float32> var_q,
    wp::vec_t<6, wp::float32> var_qd,
    wp::vec_t<6, wp::float32> var_f,
    wp::vec_t<3, wp::float32> var_com,
    wp::mat_t<3, 3, wp::float32> var_inertia,
    wp::float32 var_inv_mass,
    wp::mat_t<3, 3, wp::float32> var_inv_inertia,
    wp::vec_t<3, wp::float32> var_gravity,
    wp::float32 var_angular_damping,
    wp::float32 var_dt,
    wp::transform_t<wp::float32> & ret_0,
    wp::vec_t<6, wp::float32> & ret_1,
    wp::transform_t<wp::float32> & adj_q,
    wp::vec_t<6, wp::float32> & adj_qd,
    wp::vec_t<6, wp::float32> & adj_f,
    wp::vec_t<3, wp::float32> & adj_com,
    wp::mat_t<3, 3, wp::float32> & adj_inertia,
    wp::float32 & adj_inv_mass,
    wp::mat_t<3, 3, wp::float32> & adj_inv_inertia,
    wp::vec_t<3, wp::float32> & adj_gravity,
    wp::float32 & adj_angular_damping,
    wp::float32 & adj_dt,
    wp::transform_t<wp::float32> & adj_ret_0,
    wp::vec_t<6, wp::float32> & adj_ret_1)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::quat_t<wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    const wp::float32 var_25 = 0.0;
    wp::quat_t<wp::float32> var_26;
    wp::quat_t<wp::float32> var_27;
    const wp::float32 var_28 = 0.5;
    wp::quat_t<wp::float32> var_29;
    wp::quat_t<wp::float32> var_30;
    wp::quat_t<wp::float32> var_31;
    wp::quat_t<wp::float32> var_32;
    const wp::float32 var_33 = 1.0;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::transform_t<wp::float32> var_39;
    wp::vec_t<6, wp::float32> var_40;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::quat_t<wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::float32 adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::vec_t<3, wp::float32> adj_16 = {};
    wp::vec_t<3, wp::float32> adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::vec_t<3, wp::float32> adj_22 = {};
    wp::vec_t<3, wp::float32> adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::float32 adj_25 = {};
    wp::quat_t<wp::float32> adj_26 = {};
    wp::quat_t<wp::float32> adj_27 = {};
    wp::float32 adj_28 = {};
    wp::quat_t<wp::float32> adj_29 = {};
    wp::quat_t<wp::float32> adj_30 = {};
    wp::quat_t<wp::float32> adj_31 = {};
    wp::quat_t<wp::float32> adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::transform_t<wp::float32> adj_39 = {};
    wp::vec_t<6, wp::float32> adj_40 = {};
    //---------
    // forward
    // def integrate_rigid_body(                                                              <L 52>
    // x0 = wp.transform_get_translation(q)                                                   <L 65>
    var_0 = wp::transform_get_translation(var_q);
    // r0 = wp.transform_get_rotation(q)                                                      <L 66>
    var_1 = wp::transform_get_rotation(var_q);
    // w0 = wp.spatial_bottom(qd)                                                             <L 69>
    var_2 = wp::spatial_bottom(var_qd);
    // v0 = wp.spatial_top(qd)                                                                <L 70>
    var_3 = wp::spatial_top(var_qd);
    // t0 = wp.spatial_bottom(f)                                                              <L 73>
    var_4 = wp::spatial_bottom(var_f);
    // f0 = wp.spatial_top(f)                                                                 <L 74>
    var_5 = wp::spatial_top(var_f);
    // x_com = x0 + wp.quat_rotate(r0, com)                                                   <L 76>
    var_6 = wp::quat_rotate(var_1, var_com);
    var_7 = wp::add(var_0, var_6);
    // v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt                        <L 79>
    var_8 = wp::mul(var_5, var_inv_mass);
    var_9 = wp::nonzero(var_inv_mass);
    var_10 = wp::mul(var_gravity, var_9);
    var_11 = wp::add(var_8, var_10);
    var_12 = wp::mul(var_11, var_dt);
    var_13 = wp::add(var_3, var_12);
    // x1 = x_com + v1 * dt                                                                   <L 80>
    var_14 = wp::mul(var_13, var_dt);
    var_15 = wp::add(var_7, var_14);
    // wb = wp.quat_rotate_inv(r0, w0)                                                        <L 83>
    var_16 = wp::quat_rotate_inv(var_1, var_2);
    // tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces        <L 84>
    var_17 = wp::quat_rotate_inv(var_1, var_4);
    var_18 = wp::mul(var_inertia, var_16);
    var_19 = wp::cross(var_16, var_18);
    var_20 = wp::sub(var_17, var_19);
    // w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)                                    <L 86>
    var_21 = wp::mul(var_inv_inertia, var_20);
    var_22 = wp::mul(var_21, var_dt);
    var_23 = wp::add(var_16, var_22);
    var_24 = wp::quat_rotate(var_1, var_23);
    // r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)                               <L 87>
    var_26 = wp::quat_t<wp::float32>(var_24, var_25);
    var_27 = wp::mul(var_26, var_1);
    var_29 = wp::mul(var_27, var_28);
    var_30 = wp::mul(var_29, var_dt);
    var_31 = wp::add(var_1, var_30);
    var_32 = wp::normalize(var_31);
    // w1 *= 1.0 - angular_damping * dt                                                       <L 90>
    var_34 = wp::mul(var_angular_damping, var_dt);
    var_35 = wp::sub(var_33, var_34);
    var_36 = wp::mul(var_24, var_35);
    // q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)                                 <L 92>
    var_37 = wp::quat_rotate(var_32, var_com);
    var_38 = wp::sub(var_15, var_37);
    var_39 = wp::transform_t<wp::float32>(var_38, var_32);
    // qd_new = wp.spatial_vector(v1, w1)                                                     <L 93>
    var_40 = wp::vec_t<6, wp::float32>(var_13, var_36);
    // return q_new, qd_new                                                                   <L 95>
    ret_0 = var_39;
    ret_1 = var_40;
    goto label0;
    //---------
    // reverse
    label0:;
    adj_40 += adj_ret_1;
    adj_39 += adj_ret_0;
    // adj: return q_new, qd_new                                                              <L 95>
    wp::adj_vec_t(var_13, var_36, adj_13, adj_36, adj_40);
    // adj: qd_new = wp.spatial_vector(v1, w1)                                                <L 93>
    wp::adj_transform_t(var_38, var_32, adj_38, adj_32, adj_39);
    wp::adj_sub(var_15, var_37, adj_15, adj_37, adj_38);
    wp::adj_quat_rotate(var_32, var_com, adj_32, adj_com, adj_37);
    // adj: q_new = wp.transform(x1 - wp.quat_rotate(r1, com), r1)                            <L 92>
    wp::adj_mul(var_24, var_35, adj_24, adj_35, adj_36);
    wp::adj_sub(var_33, var_34, adj_33, adj_34, adj_35);
    wp::adj_mul(var_angular_damping, var_dt, adj_angular_damping, adj_dt, adj_34);
    // adj: w1 *= 1.0 - angular_damping * dt                                                  <L 90>
    wp::adj_normalize(var_31, adj_31, adj_32);
    wp::adj_add(var_1, var_30, adj_1, adj_30, adj_31);
    wp::adj_mul(var_29, var_dt, adj_29, adj_dt, adj_30);
    wp::adj_mul(var_27, var_28, adj_27, adj_28, adj_29);
    wp::adj_mul(var_26, var_1, adj_26, adj_1, adj_27);
    wp::adj_quat_t(var_24, var_25, adj_24, adj_25, adj_26);
    // adj: r1 = wp.normalize(r0 + wp.quat(w1, 0.0) * r0 * 0.5 * dt)                          <L 87>
    wp::adj_quat_rotate(var_1, var_23, adj_1, adj_23, adj_24);
    wp::adj_add(var_16, var_22, adj_16, adj_22, adj_23);
    wp::adj_mul(var_21, var_dt, adj_21, adj_dt, adj_22);
    wp::adj_mul(var_inv_inertia, var_20, adj_inv_inertia, adj_20, adj_21);
    // adj: w1 = wp.quat_rotate(r0, wb + inv_inertia * tb * dt)                               <L 86>
    wp::adj_sub(var_17, var_19, adj_17, adj_19, adj_20);
    wp::adj_cross(var_16, var_18, adj_16, adj_18, adj_19);
    wp::adj_mul(var_inertia, var_16, adj_inertia, adj_16, adj_18);
    wp::adj_quat_rotate_inv(var_1, var_4, adj_1, adj_4, adj_17);
    // adj: tb = wp.quat_rotate_inv(r0, t0) - wp.cross(wb, inertia * wb)  # coriolis forces   <L 84>
    wp::adj_quat_rotate_inv(var_1, var_2, adj_1, adj_2, adj_16);
    // adj: wb = wp.quat_rotate_inv(r0, w0)                                                   <L 83>
    wp::adj_add(var_7, var_14, adj_7, adj_14, adj_15);
    wp::adj_mul(var_13, var_dt, adj_13, adj_dt, adj_14);
    // adj: x1 = x_com + v1 * dt                                                              <L 80>
    wp::adj_add(var_3, var_12, adj_3, adj_12, adj_13);
    wp::adj_mul(var_11, var_dt, adj_11, adj_dt, adj_12);
    wp::adj_add(var_8, var_10, adj_8, adj_10, adj_11);
    wp::adj_mul(var_gravity, var_9, adj_gravity, adj_9, adj_10);
    wp::adj_mul(var_5, var_inv_mass, adj_5, adj_inv_mass, adj_8);
    // adj: v1 = v0 + (f0 * inv_mass + gravity * wp.nonzero(inv_mass)) * dt                   <L 79>
    wp::adj_add(var_0, var_6, adj_0, adj_6, adj_7);
    wp::adj_quat_rotate(var_1, var_com, adj_1, adj_com, adj_6);
    // adj: x_com = x0 + wp.quat_rotate(r0, com)                                              <L 76>
    wp::adj_spatial_top(var_f, adj_f, adj_5);
    // adj: f0 = wp.spatial_top(f)                                                            <L 74>
    wp::adj_spatial_bottom(var_f, adj_f, adj_4);
    // adj: t0 = wp.spatial_bottom(f)                                                         <L 73>
    wp::adj_spatial_top(var_qd, adj_qd, adj_3);
    // adj: v0 = wp.spatial_top(qd)                                                           <L 70>
    wp::adj_spatial_bottom(var_qd, adj_qd, adj_2);
    // adj: w0 = wp.spatial_bottom(qd)                                                        <L 69>
    wp::adj_transform_get_rotation(var_q, adj_q, adj_1);
    // adj: r0 = wp.transform_get_rotation(q)                                                 <L 66>
    wp::adj_transform_get_translation(var_q, adj_q, adj_0);
    // adj: x0 = wp.transform_get_translation(q)                                              <L 65>
    // adj: def integrate_rigid_body(                                                         <L 52>
    return;
}

struct wp_args_integrate_bodies_6b670ad3 {
    wp::array_t<wp::transform_t<wp::float32>> body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> body_qd;
    wp::array_t<wp::vec_t<6, wp::float32>> body_f;
    wp::array_t<wp::vec_t<3, wp::float32>> body_com;
    wp::array_t<wp::float32> m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> I;
    wp::array_t<wp::float32> inv_m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> inv_I;
    wp::array_t<wp::int32> body_flags;
    wp::array_t<wp::int32> body_world;
    wp::array_t<wp::vec_t<3, wp::float32>> gravity;
    wp::float32 angular_damping;
    wp::float32 dt;
    wp::array_t<wp::transform_t<wp::float32>> body_q_new;
    wp::array_t<wp::vec_t<6, wp::float32>> body_qd_new;
};


void integrate_bodies_6b670ad3_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_integrate_bodies_6b670ad3 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::transform_t<wp::float32>> var_body_q = _wp_args->body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd = _wp_args->body_qd;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_f = _wp_args->body_f;
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com = _wp_args->body_com;
    wp::array_t<wp::float32> var_m = _wp_args->m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_I = _wp_args->I;
    wp::array_t<wp::float32> var_inv_m = _wp_args->inv_m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_inv_I = _wp_args->inv_I;
    wp::array_t<wp::int32> var_body_flags = _wp_args->body_flags;
    wp::array_t<wp::int32> var_body_world = _wp_args->body_world;
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity = _wp_args->gravity;
    wp::float32 var_angular_damping = _wp_args->angular_damping;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::transform_t<wp::float32>> var_body_q_new = _wp_args->body_q_new;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd_new = _wp_args->body_qd_new;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 2;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    wp::transform_t<wp::float32>* var_7;
    wp::transform_t<wp::float32> var_8;
    wp::vec_t<6, wp::float32>* var_9;
    wp::vec_t<6, wp::float32> var_10;
    wp::transform_t<wp::float32>* var_11;
    wp::transform_t<wp::float32> var_12;
    wp::transform_t<wp::float32> var_13;
    wp::vec_t<6, wp::float32>* var_14;
    wp::vec_t<6, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    wp::vec_t<6, wp::float32>* var_17;
    wp::vec_t<6, wp::float32> var_18;
    wp::vec_t<6, wp::float32> var_19;
    wp::float32* var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::mat_t<3, 3, wp::float32>* var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    wp::mat_t<3, 3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32> var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::int32* var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    const wp::int32 var_35 = 0;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::transform_t<wp::float32> var_40;
    wp::vec_t<6, wp::float32> var_41;
    //---------
    // forward
    // def integrate_bodies(                                                                  <L 100>
    // tid = wp.tid()                                                                         <L 118>
    var_0 = builtin_tid1d();
    // if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:                                       <L 120>
    var_1 = wp::address(var_body_flags, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::bit_and(var_4, var_2);
    var_6 = (var_3 != var_5);
    if (var_6) {
        // body_q_new[tid] = body_q[tid]                                                      <L 125>
        var_7 = wp::address(var_body_q, var_0);
        var_8 = wp::load(var_7);
        wp::array_store(var_body_q_new, var_0, var_8);
        // body_qd_new[tid] = body_qd[tid]                                                    <L 126>
        var_9 = wp::address(var_body_qd, var_0);
        var_10 = wp::load(var_9);
        wp::array_store(var_body_qd_new, var_0, var_10);
        // return                                                                             <L 127>
        return;
    }
    // q = body_q[tid]                                                                        <L 130>
    var_11 = wp::address(var_body_q, var_0);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // qd = body_qd[tid]                                                                      <L 131>
    var_14 = wp::address(var_body_qd, var_0);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // f = body_f[tid]                                                                        <L 132>
    var_17 = wp::address(var_body_f, var_0);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // inv_mass = inv_m[tid]  # 1 / mass                                                      <L 135>
    var_20 = wp::address(var_inv_m, var_0);
    var_22 = wp::load(var_20);
    var_21 = wp::copy(var_22);
    // inertia = I[tid]                                                                       <L 137>
    var_23 = wp::address(var_I, var_0);
    var_25 = wp::load(var_23);
    var_24 = wp::copy(var_25);
    // inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix                              <L 138>
    var_26 = wp::address(var_inv_I, var_0);
    var_28 = wp::load(var_26);
    var_27 = wp::copy(var_28);
    // com = body_com[tid]                                                                    <L 140>
    var_29 = wp::address(var_body_com, var_0);
    var_31 = wp::load(var_29);
    var_30 = wp::copy(var_31);
    // world_idx = body_world[tid]                                                            <L 141>
    var_32 = wp::address(var_body_world, var_0);
    var_34 = wp::load(var_32);
    var_33 = wp::copy(var_34);
    // world_g = gravity[wp.max(world_idx, 0)]                                                <L 142>
    var_36 = wp::max(var_33, var_35);
    var_37 = wp::address(var_gravity, var_36);
    var_39 = wp::load(var_37);
    var_38 = wp::copy(var_39);
    // q_new, qd_new = integrate_rigid_body(                                                  <L 144>
    // q,                                                                                     <L 145>
    // qd,                                                                                    <L 146>
    // f,                                                                                     <L 147>
    // com,                                                                                   <L 148>
    // inertia,                                                                               <L 149>
    // inv_mass,                                                                              <L 150>
    // inv_inertia,                                                                           <L 151>
    // world_g,                                                                               <L 152>
    // angular_damping,                                                                       <L 153>
    // dt,                                                                                    <L 154>
    integrate_rigid_body_0(var_12, var_15, var_18, var_30, var_24, var_21, var_27, var_38, var_angular_damping, var_dt, var_40, var_41);
    // body_q_new[tid] = q_new                                                                <L 157>
    wp::array_store(var_body_q_new, var_0, var_40);
    // body_qd_new[tid] = qd_new                                                              <L 158>
    wp::array_store(var_body_qd_new, var_0, var_41);
}



void integrate_bodies_6b670ad3_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_integrate_bodies_6b670ad3 *_wp_args,
    wp_args_integrate_bodies_6b670ad3 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::transform_t<wp::float32>> var_body_q = _wp_args->body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd = _wp_args->body_qd;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_f = _wp_args->body_f;
    wp::array_t<wp::vec_t<3, wp::float32>> var_body_com = _wp_args->body_com;
    wp::array_t<wp::float32> var_m = _wp_args->m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_I = _wp_args->I;
    wp::array_t<wp::float32> var_inv_m = _wp_args->inv_m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_inv_I = _wp_args->inv_I;
    wp::array_t<wp::int32> var_body_flags = _wp_args->body_flags;
    wp::array_t<wp::int32> var_body_world = _wp_args->body_world;
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity = _wp_args->gravity;
    wp::float32 var_angular_damping = _wp_args->angular_damping;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::transform_t<wp::float32>> var_body_q_new = _wp_args->body_q_new;
    wp::array_t<wp::vec_t<6, wp::float32>> var_body_qd_new = _wp_args->body_qd_new;
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q = _wp_adj_args->body_q;
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_qd = _wp_adj_args->body_qd;
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_f = _wp_adj_args->body_f;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_body_com = _wp_adj_args->body_com;
    wp::array_t<wp::float32> adj_m = _wp_adj_args->m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_I = _wp_adj_args->I;
    wp::array_t<wp::float32> adj_inv_m = _wp_adj_args->inv_m;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_inv_I = _wp_adj_args->inv_I;
    wp::array_t<wp::int32> adj_body_flags = _wp_adj_args->body_flags;
    wp::array_t<wp::int32> adj_body_world = _wp_adj_args->body_world;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity = _wp_adj_args->gravity;
    wp::float32 adj_angular_damping = _wp_adj_args->angular_damping;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::array_t<wp::transform_t<wp::float32>> adj_body_q_new = _wp_adj_args->body_q_new;
    wp::array_t<wp::vec_t<6, wp::float32>> adj_body_qd_new = _wp_adj_args->body_qd_new;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 2;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    wp::transform_t<wp::float32>* var_7;
    wp::transform_t<wp::float32> var_8;
    wp::vec_t<6, wp::float32>* var_9;
    wp::vec_t<6, wp::float32> var_10;
    wp::transform_t<wp::float32>* var_11;
    wp::transform_t<wp::float32> var_12;
    wp::transform_t<wp::float32> var_13;
    wp::vec_t<6, wp::float32>* var_14;
    wp::vec_t<6, wp::float32> var_15;
    wp::vec_t<6, wp::float32> var_16;
    wp::vec_t<6, wp::float32>* var_17;
    wp::vec_t<6, wp::float32> var_18;
    wp::vec_t<6, wp::float32> var_19;
    wp::float32* var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::mat_t<3, 3, wp::float32>* var_23;
    wp::mat_t<3, 3, wp::float32> var_24;
    wp::mat_t<3, 3, wp::float32> var_25;
    wp::mat_t<3, 3, wp::float32>* var_26;
    wp::mat_t<3, 3, wp::float32> var_27;
    wp::mat_t<3, 3, wp::float32> var_28;
    wp::vec_t<3, wp::float32>* var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::int32* var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    const wp::int32 var_35 = 0;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32>* var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::transform_t<wp::float32> var_40;
    wp::vec_t<6, wp::float32> var_41;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::transform_t<wp::float32> adj_7 = {};
    wp::transform_t<wp::float32> adj_8 = {};
    wp::vec_t<6, wp::float32> adj_9 = {};
    wp::vec_t<6, wp::float32> adj_10 = {};
    wp::transform_t<wp::float32> adj_11 = {};
    wp::transform_t<wp::float32> adj_12 = {};
    wp::transform_t<wp::float32> adj_13 = {};
    wp::vec_t<6, wp::float32> adj_14 = {};
    wp::vec_t<6, wp::float32> adj_15 = {};
    wp::vec_t<6, wp::float32> adj_16 = {};
    wp::vec_t<6, wp::float32> adj_17 = {};
    wp::vec_t<6, wp::float32> adj_18 = {};
    wp::vec_t<6, wp::float32> adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::mat_t<3, 3, wp::float32> adj_23 = {};
    wp::mat_t<3, 3, wp::float32> adj_24 = {};
    wp::mat_t<3, 3, wp::float32> adj_25 = {};
    wp::mat_t<3, 3, wp::float32> adj_26 = {};
    wp::mat_t<3, 3, wp::float32> adj_27 = {};
    wp::mat_t<3, 3, wp::float32> adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::vec_t<3, wp::float32> adj_31 = {};
    wp::int32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::vec_t<3, wp::float32> adj_39 = {};
    wp::transform_t<wp::float32> adj_40 = {};
    wp::vec_t<6, wp::float32> adj_41 = {};
    //---------
    // forward
    // def integrate_bodies(                                                                  <L 100>
    // tid = wp.tid()                                                                         <L 118>
    var_0 = builtin_tid1d();
    // if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:                                       <L 120>
    var_1 = wp::address(var_body_flags, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::bit_and(var_4, var_2);
    var_6 = (var_3 != var_5);
    if (var_6) {
        // body_q_new[tid] = body_q[tid]                                                      <L 125>
        var_7 = wp::address(var_body_q, var_0);
        var_8 = wp::load(var_7);
        // wp::array_store(var_body_q_new, var_0, var_8);
        // body_qd_new[tid] = body_qd[tid]                                                    <L 126>
        var_9 = wp::address(var_body_qd, var_0);
        var_10 = wp::load(var_9);
        // wp::array_store(var_body_qd_new, var_0, var_10);
        // return                                                                             <L 127>
        goto label0;
    }
    // q = body_q[tid]                                                                        <L 130>
    var_11 = wp::address(var_body_q, var_0);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // qd = body_qd[tid]                                                                      <L 131>
    var_14 = wp::address(var_body_qd, var_0);
    var_16 = wp::load(var_14);
    var_15 = wp::copy(var_16);
    // f = body_f[tid]                                                                        <L 132>
    var_17 = wp::address(var_body_f, var_0);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // inv_mass = inv_m[tid]  # 1 / mass                                                      <L 135>
    var_20 = wp::address(var_inv_m, var_0);
    var_22 = wp::load(var_20);
    var_21 = wp::copy(var_22);
    // inertia = I[tid]                                                                       <L 137>
    var_23 = wp::address(var_I, var_0);
    var_25 = wp::load(var_23);
    var_24 = wp::copy(var_25);
    // inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix                              <L 138>
    var_26 = wp::address(var_inv_I, var_0);
    var_28 = wp::load(var_26);
    var_27 = wp::copy(var_28);
    // com = body_com[tid]                                                                    <L 140>
    var_29 = wp::address(var_body_com, var_0);
    var_31 = wp::load(var_29);
    var_30 = wp::copy(var_31);
    // world_idx = body_world[tid]                                                            <L 141>
    var_32 = wp::address(var_body_world, var_0);
    var_34 = wp::load(var_32);
    var_33 = wp::copy(var_34);
    // world_g = gravity[wp.max(world_idx, 0)]                                                <L 142>
    var_36 = wp::max(var_33, var_35);
    var_37 = wp::address(var_gravity, var_36);
    var_39 = wp::load(var_37);
    var_38 = wp::copy(var_39);
    // q_new, qd_new = integrate_rigid_body(                                                  <L 144>
    // q,                                                                                     <L 145>
    // qd,                                                                                    <L 146>
    // f,                                                                                     <L 147>
    // com,                                                                                   <L 148>
    // inertia,                                                                               <L 149>
    // inv_mass,                                                                              <L 150>
    // inv_inertia,                                                                           <L 151>
    // world_g,                                                                               <L 152>
    // angular_damping,                                                                       <L 153>
    // dt,                                                                                    <L 154>
    integrate_rigid_body_0(var_12, var_15, var_18, var_30, var_24, var_21, var_27, var_38, var_angular_damping, var_dt, var_40, var_41);
    // body_q_new[tid] = q_new                                                                <L 157>
    // wp::array_store(var_body_q_new, var_0, var_40);
    // body_qd_new[tid] = qd_new                                                              <L 158>
    // wp::array_store(var_body_qd_new, var_0, var_41);
    //---------
    // reverse
    wp::adj_array_store(var_body_qd_new, var_0, var_41, adj_body_qd_new, adj_0, adj_41);
    // adj: body_qd_new[tid] = qd_new                                                         <L 158>
    wp::adj_array_store(var_body_q_new, var_0, var_40, adj_body_q_new, adj_0, adj_40);
    // adj: body_q_new[tid] = q_new                                                           <L 157>
    adj_integrate_rigid_body_0(var_12, var_15, var_18, var_30, var_24, var_21, var_27, var_38, var_angular_damping, var_dt, var_40, var_41, adj_12, adj_15, adj_18, adj_30, adj_24, adj_21, adj_27, adj_38, adj_angular_damping, adj_dt, adj_40, adj_41);
    // adj: dt,                                                                               <L 154>
    // adj: angular_damping,                                                                  <L 153>
    // adj: world_g,                                                                          <L 152>
    // adj: inv_inertia,                                                                      <L 151>
    // adj: inv_mass,                                                                         <L 150>
    // adj: inertia,                                                                          <L 149>
    // adj: com,                                                                              <L 148>
    // adj: f,                                                                                <L 147>
    // adj: qd,                                                                               <L 146>
    // adj: q,                                                                                <L 145>
    // adj: q_new, qd_new = integrate_rigid_body(                                             <L 144>
    wp::adj_copy(var_39, adj_37, adj_38);
    wp::adj_address(var_gravity, var_36, adj_gravity, adj_36, adj_37);
    wp::adj_max(var_33, var_35, adj_33, adj_35, adj_36);
    // adj: world_g = gravity[wp.max(world_idx, 0)]                                           <L 142>
    wp::adj_copy(var_34, adj_32, adj_33);
    wp::adj_address(var_body_world, var_0, adj_body_world, adj_0, adj_32);
    // adj: world_idx = body_world[tid]                                                       <L 141>
    wp::adj_copy(var_31, adj_29, adj_30);
    wp::adj_address(var_body_com, var_0, adj_body_com, adj_0, adj_29);
    // adj: com = body_com[tid]                                                               <L 140>
    wp::adj_copy(var_28, adj_26, adj_27);
    wp::adj_address(var_inv_I, var_0, adj_inv_I, adj_0, adj_26);
    // adj: inv_inertia = inv_I[tid]  # inverse of 3x3 inertia matrix                         <L 138>
    wp::adj_copy(var_25, adj_23, adj_24);
    wp::adj_address(var_I, var_0, adj_I, adj_0, adj_23);
    // adj: inertia = I[tid]                                                                  <L 137>
    wp::adj_copy(var_22, adj_20, adj_21);
    wp::adj_address(var_inv_m, var_0, adj_inv_m, adj_0, adj_20);
    // adj: inv_mass = inv_m[tid]  # 1 / mass                                                 <L 135>
    wp::adj_copy(var_19, adj_17, adj_18);
    wp::adj_address(var_body_f, var_0, adj_body_f, adj_0, adj_17);
    // adj: f = body_f[tid]                                                                   <L 132>
    wp::adj_copy(var_16, adj_14, adj_15);
    wp::adj_address(var_body_qd, var_0, adj_body_qd, adj_0, adj_14);
    // adj: qd = body_qd[tid]                                                                 <L 131>
    wp::adj_copy(var_13, adj_11, adj_12);
    wp::adj_address(var_body_q, var_0, adj_body_q, adj_0, adj_11);
    // adj: q = body_q[tid]                                                                   <L 130>
    if (var_6) {
        label0:;
        // adj: return                                                                        <L 127>
        wp::adj_array_store(var_body_qd_new, var_0, var_10, adj_body_qd_new, adj_0, adj_9);
        wp::adj_address(var_body_qd, var_0, adj_body_qd, adj_0, adj_9);
        // adj: body_qd_new[tid] = body_qd[tid]                                               <L 126>
        wp::adj_array_store(var_body_q_new, var_0, var_8, adj_body_q_new, adj_0, adj_7);
        wp::adj_address(var_body_q, var_0, adj_body_q, adj_0, adj_7);
        // adj: body_q_new[tid] = body_q[tid]                                                 <L 125>
    }
    wp::adj_address(var_body_flags, var_0, adj_body_flags, adj_0, adj_1);
    // adj: if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:                                  <L 120>
    // adj: tid = wp.tid()                                                                    <L 118>
    // adj: def integrate_bodies(                                                             <L 100>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void integrate_bodies_6b670ad3_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_integrate_bodies_6b670ad3 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        integrate_bodies_6b670ad3_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void integrate_bodies_6b670ad3_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_integrate_bodies_6b670ad3 *_wp_args,
    wp_args_integrate_bodies_6b670ad3 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        integrate_bodies_6b670ad3_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args__update_effective_inv_mass_inertia_79ac0336 {
    wp::array_t<wp::int32> body_flags;
    wp::array_t<wp::float32> model_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> model_inv_inertia;
    wp::array_t<wp::float32> eff_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> eff_inv_inertia;
};


void _update_effective_inv_mass_inertia_79ac0336_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args__update_effective_inv_mass_inertia_79ac0336 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_body_flags = _wp_args->body_flags;
    wp::array_t<wp::float32> var_model_inv_mass = _wp_args->model_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_model_inv_inertia = _wp_args->model_inv_inertia;
    wp::array_t<wp::float32> var_eff_inv_mass = _wp_args->eff_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_eff_inv_inertia = _wp_args->eff_inv_inertia;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 2;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    const wp::float32 var_10 = 0.0;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.0;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::mat_t<3, 3, wp::float32>* var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    //---------
    // forward
    // def _update_effective_inv_mass_inertia(                                                <L 162>
    // tid = wp.tid()                                                                         <L 169>
    var_0 = builtin_tid1d();
    // if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:                                       <L 170>
    var_1 = wp::address(var_body_flags, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::bit_and(var_4, var_2);
    var_6 = (var_3 != var_5);
    if (var_6) {
        // eff_inv_mass[tid] = 0.0                                                            <L 171>
        wp::array_store(var_eff_inv_mass, var_0, var_7);
        // eff_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 172>
        var_17 = wp::mat_t<3, 3, wp::float32>(var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16);
        wp::array_store(var_eff_inv_inertia, var_0, var_17);
    }
    if (!var_6) {
        // eff_inv_mass[tid] = model_inv_mass[tid]                                            <L 174>
        var_18 = wp::address(var_model_inv_mass, var_0);
        var_19 = wp::load(var_18);
        wp::array_store(var_eff_inv_mass, var_0, var_19);
        // eff_inv_inertia[tid] = model_inv_inertia[tid]                                      <L 175>
        var_20 = wp::address(var_model_inv_inertia, var_0);
        var_21 = wp::load(var_20);
        wp::array_store(var_eff_inv_inertia, var_0, var_21);
    }
}



void _update_effective_inv_mass_inertia_79ac0336_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args__update_effective_inv_mass_inertia_79ac0336 *_wp_args,
    wp_args__update_effective_inv_mass_inertia_79ac0336 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_body_flags = _wp_args->body_flags;
    wp::array_t<wp::float32> var_model_inv_mass = _wp_args->model_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_model_inv_inertia = _wp_args->model_inv_inertia;
    wp::array_t<wp::float32> var_eff_inv_mass = _wp_args->eff_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_eff_inv_inertia = _wp_args->eff_inv_inertia;
    wp::array_t<wp::int32> adj_body_flags = _wp_adj_args->body_flags;
    wp::array_t<wp::float32> adj_model_inv_mass = _wp_adj_args->model_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_model_inv_inertia = _wp_adj_args->model_inv_inertia;
    wp::array_t<wp::float32> adj_eff_inv_mass = _wp_adj_args->eff_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_eff_inv_inertia = _wp_adj_args->eff_inv_inertia;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 2;
    wp::int32 var_3;
    wp::int32 var_4;
    const wp::int32 var_5 = 0;
    bool var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    const wp::float32 var_9 = 0.0;
    const wp::float32 var_10 = 0.0;
    const wp::float32 var_11 = 0.0;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.0;
    const wp::float32 var_15 = 0.0;
    const wp::float32 var_16 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::mat_t<3, 3, wp::float32>* var_20;
    wp::mat_t<3, 3, wp::float32> var_21;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::mat_t<3, 3, wp::float32> adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::mat_t<3, 3, wp::float32> adj_20 = {};
    wp::mat_t<3, 3, wp::float32> adj_21 = {};
    //---------
    // forward
    // def _update_effective_inv_mass_inertia(                                                <L 162>
    // tid = wp.tid()                                                                         <L 169>
    var_0 = builtin_tid1d();
    // if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:                                       <L 170>
    var_1 = wp::address(var_body_flags, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::bit_and(var_4, var_2);
    var_6 = (var_3 != var_5);
    if (var_6) {
        // eff_inv_mass[tid] = 0.0                                                            <L 171>
        // wp::array_store(var_eff_inv_mass, var_0, var_7);
        // eff_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 172>
        var_17 = wp::mat_t<3, 3, wp::float32>(var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16);
        // wp::array_store(var_eff_inv_inertia, var_0, var_17);
    }
    if (!var_6) {
        // eff_inv_mass[tid] = model_inv_mass[tid]                                            <L 174>
        var_18 = wp::address(var_model_inv_mass, var_0);
        var_19 = wp::load(var_18);
        // wp::array_store(var_eff_inv_mass, var_0, var_19);
        // eff_inv_inertia[tid] = model_inv_inertia[tid]                                      <L 175>
        var_20 = wp::address(var_model_inv_inertia, var_0);
        var_21 = wp::load(var_20);
        // wp::array_store(var_eff_inv_inertia, var_0, var_21);
    }
    //---------
    // reverse
    if (!var_6) {
        wp::adj_array_store(var_eff_inv_inertia, var_0, var_21, adj_eff_inv_inertia, adj_0, adj_20);
        wp::adj_address(var_model_inv_inertia, var_0, adj_model_inv_inertia, adj_0, adj_20);
        // adj: eff_inv_inertia[tid] = model_inv_inertia[tid]                                 <L 175>
        wp::adj_array_store(var_eff_inv_mass, var_0, var_19, adj_eff_inv_mass, adj_0, adj_18);
        wp::adj_address(var_model_inv_mass, var_0, adj_model_inv_mass, adj_0, adj_18);
        // adj: eff_inv_mass[tid] = model_inv_mass[tid]                                       <L 174>
    }
    if (var_6) {
        wp::adj_array_store(var_eff_inv_inertia, var_0, var_17, adj_eff_inv_inertia, adj_0, adj_17);
        wp::adj_mat_t(var_8, var_9, var_10, var_11, var_12, var_13, var_14, var_15, var_16, adj_8, adj_9, adj_10, adj_11, adj_12, adj_13, adj_14, adj_15, adj_16, adj_17);
        // adj: eff_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  <L 172>
        wp::adj_array_store(var_eff_inv_mass, var_0, var_7, adj_eff_inv_mass, adj_0, adj_7);
        // adj: eff_inv_mass[tid] = 0.0                                                       <L 171>
    }
    wp::adj_address(var_body_flags, var_0, adj_body_flags, adj_0, adj_1);
    // adj: if (body_flags[tid] & BodyFlags.KINEMATIC) != 0:                                  <L 170>
    // adj: tid = wp.tid()                                                                    <L 169>
    // adj: def _update_effective_inv_mass_inertia(                                           <L 162>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void _update_effective_inv_mass_inertia_79ac0336_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args__update_effective_inv_mass_inertia_79ac0336 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        _update_effective_inv_mass_inertia_79ac0336_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void _update_effective_inv_mass_inertia_79ac0336_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args__update_effective_inv_mass_inertia_79ac0336 *_wp_args,
    wp_args__update_effective_inv_mass_inertia_79ac0336 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        _update_effective_inv_mass_inertia_79ac0336_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_integrate_particles_b575d3f3 {
    wp::array_t<wp::vec_t<3, wp::float32>> x;
    wp::array_t<wp::vec_t<3, wp::float32>> v;
    wp::array_t<wp::vec_t<3, wp::float32>> f;
    wp::array_t<wp::float32> w;
    wp::array_t<wp::int32> particle_flags;
    wp::array_t<wp::int32> particle_world;
    wp::array_t<wp::vec_t<3, wp::float32>> gravity;
    wp::float32 dt;
    wp::float32 v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> x_new;
    wp::array_t<wp::vec_t<3, wp::float32>> v_new;
};


void integrate_particles_b575d3f3_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_integrate_particles_b575d3f3 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_x = _wp_args->x;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v = _wp_args->v;
    wp::array_t<wp::vec_t<3, wp::float32>> var_f = _wp_args->f;
    wp::array_t<wp::float32> var_w = _wp_args->w;
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::int32> var_particle_world = _wp_args->particle_world;
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity = _wp_args->gravity;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_v_max = _wp_args->v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> var_x_new = _wp_args->x_new;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v_new = _wp_args->v_new;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::int32* var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::int32* var_19;
    wp::int32 var_20;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::float32 var_34 = 0.97;
    wp::vec_t<3, wp::float32> var_35;
    wp::float32 var_36;
    bool var_37;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    //---------
    // forward
    // def integrate_particles(                                                               <L 11>
    // tid = wp.tid()                                                                         <L 24>
    var_0 = builtin_tid1d();
    // x0 = x[tid]                                                                            <L 25>
    var_1 = wp::address(var_x, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                                  <L 27>
    var_4 = wp::address(var_particle_flags, var_0);
    var_7 = wp::load(var_4);
    var_6 = wp::bit_and(var_7, var_5);
    var_9 = (var_6 == var_8);
    if (var_9) {
        // x_new[tid] = x0                                                                    <L 28>
        wp::array_store(var_x_new, var_0, var_2);
        // return                                                                             <L 29>
        return;
    }
    // v0 = v[tid]                                                                            <L 31>
    var_10 = wp::address(var_v, var_0);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // f0 = f[tid]                                                                            <L 32>
    var_13 = wp::address(var_f, var_0);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // inv_mass = w[tid]                                                                      <L 34>
    var_16 = wp::address(var_w, var_0);
    var_18 = wp::load(var_16);
    var_17 = wp::copy(var_18);
    // world_idx = particle_world[tid]                                                        <L 35>
    var_19 = wp::address(var_particle_world, var_0);
    var_21 = wp::load(var_19);
    var_20 = wp::copy(var_21);
    // world_g = gravity[wp.max(world_idx, 0)]                                                <L 36>
    var_23 = wp::max(var_20, var_22);
    var_24 = wp::address(var_gravity, var_23);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt                          <L 39>
    var_27 = wp::mul(var_14, var_17);
    var_28 = wp::neg(var_17);
    var_29 = wp::step(var_28);
    var_30 = wp::mul(var_25, var_29);
    var_31 = wp::add(var_27, var_30);
    var_32 = wp::mul(var_31, var_dt);
    var_33 = wp::add(var_11, var_32);
    // v1 = v1 * 0.97                                                                         <L 40>
    var_35 = wp::mul(var_33, var_34);
    // v1_mag = wp.length(v1)                                                                 <L 42>
    var_36 = wp::length(var_35);
    // if v1_mag > v_max:                                                                     <L 43>
    var_37 = (var_36 > var_v_max);
    if (var_37) {
        // v1 *= v_max / v1_mag                                                               <L 44>
        var_38 = wp::div(var_v_max, var_36);
        var_39 = wp::mul(var_35, var_38);
    }
    var_40 = wp::where(var_37, var_39, var_35);
    // x1 = x0 + v1 * dt                                                                      <L 45>
    var_41 = wp::mul(var_40, var_dt);
    var_42 = wp::add(var_2, var_41);
    // x_new[tid] = x1                                                                        <L 47>
    wp::array_store(var_x_new, var_0, var_42);
    // v_new[tid] = v1                                                                        <L 48>
    wp::array_store(var_v_new, var_0, var_40);
}



void integrate_particles_b575d3f3_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_integrate_particles_b575d3f3 *_wp_args,
    wp_args_integrate_particles_b575d3f3 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_x = _wp_args->x;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v = _wp_args->v;
    wp::array_t<wp::vec_t<3, wp::float32>> var_f = _wp_args->f;
    wp::array_t<wp::float32> var_w = _wp_args->w;
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::int32> var_particle_world = _wp_args->particle_world;
    wp::array_t<wp::vec_t<3, wp::float32>> var_gravity = _wp_args->gravity;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_v_max = _wp_args->v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> var_x_new = _wp_args->x_new;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v_new = _wp_args->v_new;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_x = _wp_adj_args->x;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_v = _wp_adj_args->v;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_f = _wp_adj_args->f;
    wp::array_t<wp::float32> adj_w = _wp_adj_args->w;
    wp::array_t<wp::int32> adj_particle_flags = _wp_adj_args->particle_flags;
    wp::array_t<wp::int32> adj_particle_world = _wp_adj_args->particle_world;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_gravity = _wp_adj_args->gravity;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::float32 adj_v_max = _wp_adj_args->v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_x_new = _wp_adj_args->x_new;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_v_new = _wp_adj_args->v_new;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::int32* var_4;
    const wp::int32 var_5 = 1;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::int32* var_19;
    wp::int32 var_20;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::float32 var_34 = 0.97;
    wp::vec_t<3, wp::float32> var_35;
    wp::float32 var_36;
    bool var_37;
    wp::float32 var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::int32 adj_8 = {};
    bool adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::vec_t<3, wp::float32> adj_31 = {};
    wp::vec_t<3, wp::float32> adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::float32 adj_34 = {};
    wp::vec_t<3, wp::float32> adj_35 = {};
    wp::float32 adj_36 = {};
    bool adj_37 = {};
    wp::float32 adj_38 = {};
    wp::vec_t<3, wp::float32> adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    //---------
    // forward
    // def integrate_particles(                                                               <L 11>
    // tid = wp.tid()                                                                         <L 24>
    var_0 = builtin_tid1d();
    // x0 = x[tid]                                                                            <L 25>
    var_1 = wp::address(var_x, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                                  <L 27>
    var_4 = wp::address(var_particle_flags, var_0);
    var_7 = wp::load(var_4);
    var_6 = wp::bit_and(var_7, var_5);
    var_9 = (var_6 == var_8);
    if (var_9) {
        // x_new[tid] = x0                                                                    <L 28>
        // wp::array_store(var_x_new, var_0, var_2);
        // return                                                                             <L 29>
        goto label0;
    }
    // v0 = v[tid]                                                                            <L 31>
    var_10 = wp::address(var_v, var_0);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // f0 = f[tid]                                                                            <L 32>
    var_13 = wp::address(var_f, var_0);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // inv_mass = w[tid]                                                                      <L 34>
    var_16 = wp::address(var_w, var_0);
    var_18 = wp::load(var_16);
    var_17 = wp::copy(var_18);
    // world_idx = particle_world[tid]                                                        <L 35>
    var_19 = wp::address(var_particle_world, var_0);
    var_21 = wp::load(var_19);
    var_20 = wp::copy(var_21);
    // world_g = gravity[wp.max(world_idx, 0)]                                                <L 36>
    var_23 = wp::max(var_20, var_22);
    var_24 = wp::address(var_gravity, var_23);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt                          <L 39>
    var_27 = wp::mul(var_14, var_17);
    var_28 = wp::neg(var_17);
    var_29 = wp::step(var_28);
    var_30 = wp::mul(var_25, var_29);
    var_31 = wp::add(var_27, var_30);
    var_32 = wp::mul(var_31, var_dt);
    var_33 = wp::add(var_11, var_32);
    // v1 = v1 * 0.97                                                                         <L 40>
    var_35 = wp::mul(var_33, var_34);
    // v1_mag = wp.length(v1)                                                                 <L 42>
    var_36 = wp::length(var_35);
    // if v1_mag > v_max:                                                                     <L 43>
    var_37 = (var_36 > var_v_max);
    if (var_37) {
        // v1 *= v_max / v1_mag                                                               <L 44>
        var_38 = wp::div(var_v_max, var_36);
        var_39 = wp::mul(var_35, var_38);
    }
    var_40 = wp::where(var_37, var_39, var_35);
    // x1 = x0 + v1 * dt                                                                      <L 45>
    var_41 = wp::mul(var_40, var_dt);
    var_42 = wp::add(var_2, var_41);
    // x_new[tid] = x1                                                                        <L 47>
    // wp::array_store(var_x_new, var_0, var_42);
    // v_new[tid] = v1                                                                        <L 48>
    // wp::array_store(var_v_new, var_0, var_40);
    //---------
    // reverse
    wp::adj_array_store(var_v_new, var_0, var_40, adj_v_new, adj_0, adj_40);
    // adj: v_new[tid] = v1                                                                   <L 48>
    wp::adj_array_store(var_x_new, var_0, var_42, adj_x_new, adj_0, adj_42);
    // adj: x_new[tid] = x1                                                                   <L 47>
    wp::adj_add(var_2, var_41, adj_2, adj_41, adj_42);
    wp::adj_mul(var_40, var_dt, adj_40, adj_dt, adj_41);
    // adj: x1 = x0 + v1 * dt                                                                 <L 45>
    wp::adj_where(var_37, var_39, var_35, adj_37, adj_39, adj_35, adj_40);
    if (var_37) {
        wp::adj_mul(var_35, var_38, adj_35, adj_38, adj_39);
        wp::adj_div(var_v_max, var_36, var_38, adj_v_max, adj_36, adj_38);
        // adj: v1 *= v_max / v1_mag                                                          <L 44>
    }
    // adj: if v1_mag > v_max:                                                                <L 43>
    wp::adj_length(var_35, var_36, adj_35, adj_36);
    // adj: v1_mag = wp.length(v1)                                                            <L 42>
    wp::adj_mul(var_33, var_34, adj_33, adj_34, adj_35);
    // adj: v1 = v1 * 0.97                                                                    <L 40>
    wp::adj_add(var_11, var_32, adj_11, adj_32, adj_33);
    wp::adj_mul(var_31, var_dt, adj_31, adj_dt, adj_32);
    wp::adj_add(var_27, var_30, adj_27, adj_30, adj_31);
    wp::adj_mul(var_25, var_29, adj_25, adj_29, adj_30);
    wp::adj_neg(var_17, adj_17, adj_28);
    wp::adj_mul(var_14, var_17, adj_14, adj_17, adj_27);
    // adj: v1 = v0 + (f0 * inv_mass + world_g * wp.step(-inv_mass)) * dt                     <L 39>
    wp::adj_copy(var_26, adj_24, adj_25);
    wp::adj_address(var_gravity, var_23, adj_gravity, adj_23, adj_24);
    wp::adj_max(var_20, var_22, adj_20, adj_22, adj_23);
    // adj: world_g = gravity[wp.max(world_idx, 0)]                                           <L 36>
    wp::adj_copy(var_21, adj_19, adj_20);
    wp::adj_address(var_particle_world, var_0, adj_particle_world, adj_0, adj_19);
    // adj: world_idx = particle_world[tid]                                                   <L 35>
    wp::adj_copy(var_18, adj_16, adj_17);
    wp::adj_address(var_w, var_0, adj_w, adj_0, adj_16);
    // adj: inv_mass = w[tid]                                                                 <L 34>
    wp::adj_copy(var_15, adj_13, adj_14);
    wp::adj_address(var_f, var_0, adj_f, adj_0, adj_13);
    // adj: f0 = f[tid]                                                                       <L 32>
    wp::adj_copy(var_12, adj_10, adj_11);
    wp::adj_address(var_v, var_0, adj_v, adj_0, adj_10);
    // adj: v0 = v[tid]                                                                       <L 31>
    if (var_9) {
        label0:;
        // adj: return                                                                        <L 29>
        wp::adj_array_store(var_x_new, var_0, var_2, adj_x_new, adj_0, adj_2);
        // adj: x_new[tid] = x0                                                               <L 28>
    }
    wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_4);
    // adj: if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                             <L 27>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_x, var_0, adj_x, adj_0, adj_1);
    // adj: x0 = x[tid]                                                                       <L 25>
    // adj: tid = wp.tid()                                                                    <L 24>
    // adj: def integrate_particles(                                                          <L 11>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void integrate_particles_b575d3f3_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_integrate_particles_b575d3f3 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        integrate_particles_b575d3f3_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void integrate_particles_b575d3f3_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_integrate_particles_b575d3f3 *_wp_args,
    wp_args_integrate_particles_b575d3f3 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        integrate_particles_b575d3f3_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

