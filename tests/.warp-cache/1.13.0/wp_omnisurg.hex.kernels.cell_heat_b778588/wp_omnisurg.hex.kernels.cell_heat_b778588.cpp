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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:33
static wp::vec_t<3, wp::float32> _cell_deformed_center_0(
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::int32 var_cell_idx)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 0;
    wp::int32* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::int32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 1;
    wp::int32* var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::int32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 2;
    wp::int32* var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::int32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 3;
    wp::int32* var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::int32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 4;
    wp::int32* var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::int32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 5;
    wp::int32* var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::int32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 6;
    wp::int32* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::int32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    const wp::int32 var_46 = 7;
    wp::int32* var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::int32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    const wp::float32 var_52 = 0.125;
    wp::vec_t<3, wp::float32> var_53;
    //---------
    // forward
    // def _cell_deformed_center(                                                             <L 34>
    // centre = wp.vec3(0.0, 0.0, 0.0)                                                        <L 39>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // for local_node in range(8):                                                            <L 40>
    // centre += particle_q[cell_nodes[cell_idx, local_node]]                                 <L 41>
    var_5 = wp::address(var_cell_nodes, var_cell_idx, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::address(var_particle_q, var_7);
    var_9 = wp::load(var_6);
    var_8 = wp::add(var_3, var_9);
    var_11 = wp::address(var_cell_nodes, var_cell_idx, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::address(var_particle_q, var_13);
    var_15 = wp::load(var_12);
    var_14 = wp::add(var_8, var_15);
    var_17 = wp::address(var_cell_nodes, var_cell_idx, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::address(var_particle_q, var_19);
    var_21 = wp::load(var_18);
    var_20 = wp::add(var_14, var_21);
    var_23 = wp::address(var_cell_nodes, var_cell_idx, var_22);
    var_25 = wp::load(var_23);
    var_24 = wp::address(var_particle_q, var_25);
    var_27 = wp::load(var_24);
    var_26 = wp::add(var_20, var_27);
    var_29 = wp::address(var_cell_nodes, var_cell_idx, var_28);
    var_31 = wp::load(var_29);
    var_30 = wp::address(var_particle_q, var_31);
    var_33 = wp::load(var_30);
    var_32 = wp::add(var_26, var_33);
    var_35 = wp::address(var_cell_nodes, var_cell_idx, var_34);
    var_37 = wp::load(var_35);
    var_36 = wp::address(var_particle_q, var_37);
    var_39 = wp::load(var_36);
    var_38 = wp::add(var_32, var_39);
    var_41 = wp::address(var_cell_nodes, var_cell_idx, var_40);
    var_43 = wp::load(var_41);
    var_42 = wp::address(var_particle_q, var_43);
    var_45 = wp::load(var_42);
    var_44 = wp::add(var_38, var_45);
    var_47 = wp::address(var_cell_nodes, var_cell_idx, var_46);
    var_49 = wp::load(var_47);
    var_48 = wp::address(var_particle_q, var_49);
    var_51 = wp::load(var_48);
    var_50 = wp::add(var_44, var_51);
    // return centre * 0.125                                                                  <L 42>
    var_53 = wp::mul(var_50, var_52);
    return var_53;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:45
static wp::int32 _cell_intersects_sphere_0(
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::int32 var_cell_idx,
    wp::vec_t<3, wp::float32> var_sphere_centre,
    wp::float32 var_radius)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    bool var_4;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 0;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    bool var_14;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = 1;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::vec_t<3, wp::float32>* var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::float32 var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 2;
    wp::int32* var_27;
    wp::int32 var_28;
    wp::int32 var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    bool var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = 3;
    wp::int32* var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::vec_t<3, wp::float32>* var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    bool var_44;
    const wp::int32 var_45 = 1;
    const wp::int32 var_46 = 4;
    wp::int32* var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::vec_t<3, wp::float32>* var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    bool var_54;
    const wp::int32 var_55 = 1;
    const wp::int32 var_56 = 5;
    wp::int32* var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::vec_t<3, wp::float32>* var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::float32 var_63;
    bool var_64;
    const wp::int32 var_65 = 1;
    const wp::int32 var_66 = 6;
    wp::int32* var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    wp::vec_t<3, wp::float32>* var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::float32 var_73;
    bool var_74;
    const wp::int32 var_75 = 1;
    const wp::int32 var_76 = 7;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::vec_t<3, wp::float32>* var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::float32 var_83;
    bool var_84;
    const wp::int32 var_85 = 1;
    const wp::int32 var_86 = 0;
    //---------
    // forward
    // def _cell_intersects_sphere(                                                           <L 46>
    // r2 = radius * radius                                                                   <L 53>
    var_0 = wp::mul(var_radius, var_radius);
    // centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)                       <L 54>
    var_1 = _cell_deformed_center_0(var_cell_nodes, var_particle_q, var_cell_idx);
    // delta = centre - sphere_centre                                                         <L 55>
    var_2 = wp::sub(var_1, var_sphere_centre);
    // if wp.dot(delta, delta) <= r2:                                                         <L 56>
    var_3 = wp::dot(var_2, var_2);
    var_4 = (var_3 <= var_0);
    if (var_4) {
        // return 1                                                                           <L 57>
        return var_5;
    }
    // for local_node in range(8):                                                            <L 58>
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_7 = wp::address(var_cell_nodes, var_cell_idx, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_10 = wp::address(var_particle_q, var_8);
    var_12 = wp::load(var_10);
    var_11 = wp::sub(var_12, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_13 = wp::dot(var_11, var_11);
    var_14 = (var_13 <= var_0);
    if (var_14) {
        // return 1                                                                           <L 62>
        return var_15;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_17 = wp::address(var_cell_nodes, var_cell_idx, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_20 = wp::address(var_particle_q, var_18);
    var_22 = wp::load(var_20);
    var_21 = wp::sub(var_22, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_23 = wp::dot(var_21, var_21);
    var_24 = (var_23 <= var_0);
    if (var_24) {
        // return 1                                                                           <L 62>
        return var_25;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_27 = wp::address(var_cell_nodes, var_cell_idx, var_26);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_30 = wp::address(var_particle_q, var_28);
    var_32 = wp::load(var_30);
    var_31 = wp::sub(var_32, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_33 = wp::dot(var_31, var_31);
    var_34 = (var_33 <= var_0);
    if (var_34) {
        // return 1                                                                           <L 62>
        return var_35;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_37 = wp::address(var_cell_nodes, var_cell_idx, var_36);
    var_39 = wp::load(var_37);
    var_38 = wp::copy(var_39);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_40 = wp::address(var_particle_q, var_38);
    var_42 = wp::load(var_40);
    var_41 = wp::sub(var_42, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_43 = wp::dot(var_41, var_41);
    var_44 = (var_43 <= var_0);
    if (var_44) {
        // return 1                                                                           <L 62>
        return var_45;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_47 = wp::address(var_cell_nodes, var_cell_idx, var_46);
    var_49 = wp::load(var_47);
    var_48 = wp::copy(var_49);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_50 = wp::address(var_particle_q, var_48);
    var_52 = wp::load(var_50);
    var_51 = wp::sub(var_52, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_53 = wp::dot(var_51, var_51);
    var_54 = (var_53 <= var_0);
    if (var_54) {
        // return 1                                                                           <L 62>
        return var_55;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_57 = wp::address(var_cell_nodes, var_cell_idx, var_56);
    var_59 = wp::load(var_57);
    var_58 = wp::copy(var_59);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_60 = wp::address(var_particle_q, var_58);
    var_62 = wp::load(var_60);
    var_61 = wp::sub(var_62, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_63 = wp::dot(var_61, var_61);
    var_64 = (var_63 <= var_0);
    if (var_64) {
        // return 1                                                                           <L 62>
        return var_65;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_67 = wp::address(var_cell_nodes, var_cell_idx, var_66);
    var_69 = wp::load(var_67);
    var_68 = wp::copy(var_69);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_70 = wp::address(var_particle_q, var_68);
    var_72 = wp::load(var_70);
    var_71 = wp::sub(var_72, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_73 = wp::dot(var_71, var_71);
    var_74 = (var_73 <= var_0);
    if (var_74) {
        // return 1                                                                           <L 62>
        return var_75;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_77 = wp::address(var_cell_nodes, var_cell_idx, var_76);
    var_79 = wp::load(var_77);
    var_78 = wp::copy(var_79);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_80 = wp::address(var_particle_q, var_78);
    var_82 = wp::load(var_80);
    var_81 = wp::sub(var_82, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_83 = wp::dot(var_81, var_81);
    var_84 = (var_83 <= var_0);
    if (var_84) {
        // return 1                                                                           <L 62>
        return var_85;
    }
    // return 0                                                                               <L 63>
    return var_86;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:15
static wp::float32 _point_segment_distance_sq_1(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 1e-20;
    bool var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    wp::float32 var_12;
    const wp::float32 var_13 = 1.0;
    bool var_14;
    const wp::float32 var_15 = 1.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::float32 var_21;
    //---------
    // forward
    // def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:           <L 16>
    // ab = b - a                                                                             <L 17>
    var_0 = wp::sub(var_b, var_a);
    // denom = wp.dot(ab, ab)                                                                 <L 18>
    var_1 = wp::dot(var_0, var_0);
    // if denom <= 1.0e-20:                                                                   <L 19>
    var_3 = (var_1 <= var_2);
    if (var_3) {
        // d = p - a                                                                          <L 20>
        var_4 = wp::sub(var_p, var_a);
        // return wp.dot(d, d)                                                                <L 21>
        var_5 = wp::dot(var_4, var_4);
        return var_5;
    }
    // t = wp.dot(p - a, ab) / denom                                                          <L 23>
    var_6 = wp::sub(var_p, var_a);
    var_7 = wp::dot(var_6, var_0);
    var_8 = wp::div(var_7, var_1);
    // if t < 0.0:                                                                            <L 24>
    var_10 = (var_8 < var_9);
    if (var_10) {
        // t = 0.0                                                                            <L 25>
    }
    var_12 = wp::where(var_10, var_11, var_8);
    if (!var_10) {
        // elif t > 1.0:                                                                      <L 26>
        var_14 = (var_12 > var_13);
        if (var_14) {
            // t = 1.0                                                                        <L 27>
        }
        var_16 = wp::where(var_14, var_15, var_12);
    }
    var_17 = wp::where(var_10, var_12, var_16);
    // q = a + ab * t                                                                         <L 28>
    var_18 = wp::mul(var_0, var_17);
    var_19 = wp::add(var_a, var_18);
    // d = p - q                                                                              <L 29>
    var_20 = wp::sub(var_p, var_19);
    // return wp.dot(d, d)                                                                    <L 30>
    var_21 = wp::dot(var_20, var_20);
    return var_21;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:66
static wp::int32 _cell_intersects_capsule_0(
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::int32 var_cell_idx,
    wp::vec_t<3, wp::float32> var_capsule_p0,
    wp::vec_t<3, wp::float32> var_capsule_p1,
    wp::float32 var_radius)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    bool var_3;
    const wp::int32 var_4 = 1;
    const wp::int32 var_5 = 0;
    wp::int32* var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    const wp::int32 var_14 = 1;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    bool var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 2;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    bool var_30;
    const wp::int32 var_31 = 1;
    const wp::int32 var_32 = 3;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::float32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    bool var_39;
    const wp::int32 var_40 = 1;
    const wp::int32 var_41 = 4;
    wp::int32* var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::vec_t<3, wp::float32>* var_45;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    bool var_48;
    const wp::int32 var_49 = 1;
    const wp::int32 var_50 = 5;
    wp::int32* var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::vec_t<3, wp::float32>* var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    bool var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 6;
    wp::int32* var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::float32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    bool var_66;
    const wp::int32 var_67 = 1;
    const wp::int32 var_68 = 7;
    wp::int32* var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::vec_t<3, wp::float32>* var_72;
    wp::float32 var_73;
    wp::vec_t<3, wp::float32> var_74;
    bool var_75;
    const wp::int32 var_76 = 1;
    const wp::int32 var_77 = 0;
    //---------
    // forward
    // def _cell_intersects_capsule(                                                          <L 67>
    // r2 = radius * radius                                                                   <L 75>
    var_0 = wp::mul(var_radius, var_radius);
    // centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)                       <L 76>
    var_1 = _cell_deformed_center_0(var_cell_nodes, var_particle_q, var_cell_idx);
    // if _point_segment_distance_sq(centre, capsule_p0, capsule_p1) <= r2:                   <L 77>
    var_2 = _point_segment_distance_sq_1(var_1, var_capsule_p0, var_capsule_p1);
    var_3 = (var_2 <= var_0);
    if (var_3) {
        // return 1                                                                           <L 78>
        return var_4;
    }
    // for local_node in range(8):                                                            <L 79>
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_6 = wp::address(var_cell_nodes, var_cell_idx, var_5);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_9 = wp::address(var_particle_q, var_7);
    var_11 = wp::load(var_9);
    var_10 = _point_segment_distance_sq_1(var_11, var_capsule_p0, var_capsule_p1);
    var_12 = (var_10 <= var_0);
    if (var_12) {
        // return 1                                                                           <L 82>
        return var_13;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_15 = wp::address(var_cell_nodes, var_cell_idx, var_14);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_18 = wp::address(var_particle_q, var_16);
    var_20 = wp::load(var_18);
    var_19 = _point_segment_distance_sq_1(var_20, var_capsule_p0, var_capsule_p1);
    var_21 = (var_19 <= var_0);
    if (var_21) {
        // return 1                                                                           <L 82>
        return var_22;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_24 = wp::address(var_cell_nodes, var_cell_idx, var_23);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_27 = wp::address(var_particle_q, var_25);
    var_29 = wp::load(var_27);
    var_28 = _point_segment_distance_sq_1(var_29, var_capsule_p0, var_capsule_p1);
    var_30 = (var_28 <= var_0);
    if (var_30) {
        // return 1                                                                           <L 82>
        return var_31;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_33 = wp::address(var_cell_nodes, var_cell_idx, var_32);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_36 = wp::address(var_particle_q, var_34);
    var_38 = wp::load(var_36);
    var_37 = _point_segment_distance_sq_1(var_38, var_capsule_p0, var_capsule_p1);
    var_39 = (var_37 <= var_0);
    if (var_39) {
        // return 1                                                                           <L 82>
        return var_40;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_42 = wp::address(var_cell_nodes, var_cell_idx, var_41);
    var_44 = wp::load(var_42);
    var_43 = wp::copy(var_44);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_45 = wp::address(var_particle_q, var_43);
    var_47 = wp::load(var_45);
    var_46 = _point_segment_distance_sq_1(var_47, var_capsule_p0, var_capsule_p1);
    var_48 = (var_46 <= var_0);
    if (var_48) {
        // return 1                                                                           <L 82>
        return var_49;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_51 = wp::address(var_cell_nodes, var_cell_idx, var_50);
    var_53 = wp::load(var_51);
    var_52 = wp::copy(var_53);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_54 = wp::address(var_particle_q, var_52);
    var_56 = wp::load(var_54);
    var_55 = _point_segment_distance_sq_1(var_56, var_capsule_p0, var_capsule_p1);
    var_57 = (var_55 <= var_0);
    if (var_57) {
        // return 1                                                                           <L 82>
        return var_58;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_60 = wp::address(var_cell_nodes, var_cell_idx, var_59);
    var_62 = wp::load(var_60);
    var_61 = wp::copy(var_62);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_63 = wp::address(var_particle_q, var_61);
    var_65 = wp::load(var_63);
    var_64 = _point_segment_distance_sq_1(var_65, var_capsule_p0, var_capsule_p1);
    var_66 = (var_64 <= var_0);
    if (var_66) {
        // return 1                                                                           <L 82>
        return var_67;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_69 = wp::address(var_cell_nodes, var_cell_idx, var_68);
    var_71 = wp::load(var_69);
    var_70 = wp::copy(var_71);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_72 = wp::address(var_particle_q, var_70);
    var_74 = wp::load(var_72);
    var_73 = _point_segment_distance_sq_1(var_74, var_capsule_p0, var_capsule_p1);
    var_75 = (var_73 <= var_0);
    if (var_75) {
        // return 1                                                                           <L 82>
        return var_76;
    }
    // return 0                                                                               <L 83>
    return var_77;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:33
static void adj__cell_deformed_center_0(
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::int32 var_cell_idx,
    wp::array_t<wp::int32> & adj_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_particle_q,
    wp::int32 & adj_cell_idx,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    const wp::int32 var_4 = 0;
    wp::int32* var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::int32 var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    const wp::int32 var_10 = 1;
    wp::int32* var_11;
    wp::vec_t<3, wp::float32>* var_12;
    wp::int32 var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 2;
    wp::int32* var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::int32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    const wp::int32 var_22 = 3;
    wp::int32* var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::int32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::int32 var_28 = 4;
    wp::int32* var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::int32 var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 5;
    wp::int32* var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::int32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    const wp::int32 var_40 = 6;
    wp::int32* var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::int32 var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    const wp::int32 var_46 = 7;
    wp::int32* var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::int32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    const wp::float32 var_52 = 0.125;
    wp::vec_t<3, wp::float32> var_53;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::int32 adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::int32 adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::int32 adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::int32 adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::int32 adj_31 = {};
    wp::vec_t<3, wp::float32> adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::int32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::int32 adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::vec_t<3, wp::float32> adj_39 = {};
    wp::int32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    wp::int32 adj_43 = {};
    wp::vec_t<3, wp::float32> adj_44 = {};
    wp::vec_t<3, wp::float32> adj_45 = {};
    wp::int32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::vec_t<3, wp::float32> adj_48 = {};
    wp::int32 adj_49 = {};
    wp::vec_t<3, wp::float32> adj_50 = {};
    wp::vec_t<3, wp::float32> adj_51 = {};
    wp::float32 adj_52 = {};
    wp::vec_t<3, wp::float32> adj_53 = {};
    //---------
    // forward
    // def _cell_deformed_center(                                                             <L 34>
    // centre = wp.vec3(0.0, 0.0, 0.0)                                                        <L 39>
    var_3 = wp::vec_t<3, wp::float32>(var_0, var_1, var_2);
    // for local_node in range(8):                                                            <L 40>
    // centre += particle_q[cell_nodes[cell_idx, local_node]]                                 <L 41>
    var_5 = wp::address(var_cell_nodes, var_cell_idx, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::address(var_particle_q, var_7);
    var_9 = wp::load(var_6);
    var_8 = wp::add(var_3, var_9);
    var_11 = wp::address(var_cell_nodes, var_cell_idx, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::address(var_particle_q, var_13);
    var_15 = wp::load(var_12);
    var_14 = wp::add(var_8, var_15);
    var_17 = wp::address(var_cell_nodes, var_cell_idx, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::address(var_particle_q, var_19);
    var_21 = wp::load(var_18);
    var_20 = wp::add(var_14, var_21);
    var_23 = wp::address(var_cell_nodes, var_cell_idx, var_22);
    var_25 = wp::load(var_23);
    var_24 = wp::address(var_particle_q, var_25);
    var_27 = wp::load(var_24);
    var_26 = wp::add(var_20, var_27);
    var_29 = wp::address(var_cell_nodes, var_cell_idx, var_28);
    var_31 = wp::load(var_29);
    var_30 = wp::address(var_particle_q, var_31);
    var_33 = wp::load(var_30);
    var_32 = wp::add(var_26, var_33);
    var_35 = wp::address(var_cell_nodes, var_cell_idx, var_34);
    var_37 = wp::load(var_35);
    var_36 = wp::address(var_particle_q, var_37);
    var_39 = wp::load(var_36);
    var_38 = wp::add(var_32, var_39);
    var_41 = wp::address(var_cell_nodes, var_cell_idx, var_40);
    var_43 = wp::load(var_41);
    var_42 = wp::address(var_particle_q, var_43);
    var_45 = wp::load(var_42);
    var_44 = wp::add(var_38, var_45);
    var_47 = wp::address(var_cell_nodes, var_cell_idx, var_46);
    var_49 = wp::load(var_47);
    var_48 = wp::address(var_particle_q, var_49);
    var_51 = wp::load(var_48);
    var_50 = wp::add(var_44, var_51);
    // return centre * 0.125                                                                  <L 42>
    var_53 = wp::mul(var_50, var_52);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_53 += adj_ret;
    wp::adj_mul(var_50, var_52, adj_50, adj_52, adj_53);
    // adj: return centre * 0.125                                                             <L 42>
    wp::adj_add(var_44, var_51, adj_44, adj_48, adj_50);
    wp::adj_address(var_particle_q, var_49, adj_particle_q, adj_47, adj_48);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_46, adj_cell_nodes, adj_cell_idx, adj_46, adj_47);
    wp::adj_add(var_38, var_45, adj_38, adj_42, adj_44);
    wp::adj_address(var_particle_q, var_43, adj_particle_q, adj_41, adj_42);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_40, adj_cell_nodes, adj_cell_idx, adj_40, adj_41);
    wp::adj_add(var_32, var_39, adj_32, adj_36, adj_38);
    wp::adj_address(var_particle_q, var_37, adj_particle_q, adj_35, adj_36);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_34, adj_cell_nodes, adj_cell_idx, adj_34, adj_35);
    wp::adj_add(var_26, var_33, adj_26, adj_30, adj_32);
    wp::adj_address(var_particle_q, var_31, adj_particle_q, adj_29, adj_30);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_28, adj_cell_nodes, adj_cell_idx, adj_28, adj_29);
    wp::adj_add(var_20, var_27, adj_20, adj_24, adj_26);
    wp::adj_address(var_particle_q, var_25, adj_particle_q, adj_23, adj_24);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_22, adj_cell_nodes, adj_cell_idx, adj_22, adj_23);
    wp::adj_add(var_14, var_21, adj_14, adj_18, adj_20);
    wp::adj_address(var_particle_q, var_19, adj_particle_q, adj_17, adj_18);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_16, adj_cell_nodes, adj_cell_idx, adj_16, adj_17);
    wp::adj_add(var_8, var_15, adj_8, adj_12, adj_14);
    wp::adj_address(var_particle_q, var_13, adj_particle_q, adj_11, adj_12);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_10, adj_cell_nodes, adj_cell_idx, adj_10, adj_11);
    wp::adj_add(var_3, var_9, adj_3, adj_6, adj_8);
    wp::adj_address(var_particle_q, var_7, adj_particle_q, adj_5, adj_6);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_4, adj_cell_nodes, adj_cell_idx, adj_4, adj_5);
    // adj: centre += particle_q[cell_nodes[cell_idx, local_node]]                            <L 41>
    // adj: for local_node in range(8):                                                       <L 40>
    wp::adj_vec_t(var_0, var_1, var_2, adj_0, adj_1, adj_2, adj_3);
    // adj: centre = wp.vec3(0.0, 0.0, 0.0)                                                   <L 39>
    // adj: def _cell_deformed_center(                                                        <L 34>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:45
static void adj__cell_intersects_sphere_0(
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::int32 var_cell_idx,
    wp::vec_t<3, wp::float32> var_sphere_centre,
    wp::float32 var_radius,
    wp::array_t<wp::int32> & adj_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_particle_q,
    wp::int32 & adj_cell_idx,
    wp::vec_t<3, wp::float32> & adj_sphere_centre,
    wp::float32 & adj_radius,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    bool var_4;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 0;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::vec_t<3, wp::float32>* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    bool var_14;
    const wp::int32 var_15 = 1;
    const wp::int32 var_16 = 1;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::vec_t<3, wp::float32>* var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::float32 var_23;
    bool var_24;
    const wp::int32 var_25 = 1;
    const wp::int32 var_26 = 2;
    wp::int32* var_27;
    wp::int32 var_28;
    wp::int32 var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::float32 var_33;
    bool var_34;
    const wp::int32 var_35 = 1;
    const wp::int32 var_36 = 3;
    wp::int32* var_37;
    wp::int32 var_38;
    wp::int32 var_39;
    wp::vec_t<3, wp::float32>* var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::float32 var_43;
    bool var_44;
    const wp::int32 var_45 = 1;
    const wp::int32 var_46 = 4;
    wp::int32* var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::vec_t<3, wp::float32>* var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    bool var_54;
    const wp::int32 var_55 = 1;
    const wp::int32 var_56 = 5;
    wp::int32* var_57;
    wp::int32 var_58;
    wp::int32 var_59;
    wp::vec_t<3, wp::float32>* var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::float32 var_63;
    bool var_64;
    const wp::int32 var_65 = 1;
    const wp::int32 var_66 = 6;
    wp::int32* var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    wp::vec_t<3, wp::float32>* var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::float32 var_73;
    bool var_74;
    const wp::int32 var_75 = 1;
    const wp::int32 var_76 = 7;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::vec_t<3, wp::float32>* var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::float32 var_83;
    bool var_84;
    const wp::int32 var_85 = 1;
    const wp::int32 var_86 = 0;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::float32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::float32 adj_13 = {};
    bool adj_14 = {};
    wp::int32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::vec_t<3, wp::float32> adj_22 = {};
    wp::float32 adj_23 = {};
    bool adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::vec_t<3, wp::float32> adj_31 = {};
    wp::vec_t<3, wp::float32> adj_32 = {};
    wp::float32 adj_33 = {};
    bool adj_34 = {};
    wp::int32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::int32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    wp::float32 adj_43 = {};
    bool adj_44 = {};
    wp::int32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::int32 adj_48 = {};
    wp::int32 adj_49 = {};
    wp::vec_t<3, wp::float32> adj_50 = {};
    wp::vec_t<3, wp::float32> adj_51 = {};
    wp::vec_t<3, wp::float32> adj_52 = {};
    wp::float32 adj_53 = {};
    bool adj_54 = {};
    wp::int32 adj_55 = {};
    wp::int32 adj_56 = {};
    wp::int32 adj_57 = {};
    wp::int32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::vec_t<3, wp::float32> adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::vec_t<3, wp::float32> adj_62 = {};
    wp::float32 adj_63 = {};
    bool adj_64 = {};
    wp::int32 adj_65 = {};
    wp::int32 adj_66 = {};
    wp::int32 adj_67 = {};
    wp::int32 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::vec_t<3, wp::float32> adj_70 = {};
    wp::vec_t<3, wp::float32> adj_71 = {};
    wp::vec_t<3, wp::float32> adj_72 = {};
    wp::float32 adj_73 = {};
    bool adj_74 = {};
    wp::int32 adj_75 = {};
    wp::int32 adj_76 = {};
    wp::int32 adj_77 = {};
    wp::int32 adj_78 = {};
    wp::int32 adj_79 = {};
    wp::vec_t<3, wp::float32> adj_80 = {};
    wp::vec_t<3, wp::float32> adj_81 = {};
    wp::vec_t<3, wp::float32> adj_82 = {};
    wp::float32 adj_83 = {};
    bool adj_84 = {};
    wp::int32 adj_85 = {};
    wp::int32 adj_86 = {};
    //---------
    // forward
    // def _cell_intersects_sphere(                                                           <L 46>
    // r2 = radius * radius                                                                   <L 53>
    var_0 = wp::mul(var_radius, var_radius);
    // centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)                       <L 54>
    var_1 = _cell_deformed_center_0(var_cell_nodes, var_particle_q, var_cell_idx);
    // delta = centre - sphere_centre                                                         <L 55>
    var_2 = wp::sub(var_1, var_sphere_centre);
    // if wp.dot(delta, delta) <= r2:                                                         <L 56>
    var_3 = wp::dot(var_2, var_2);
    var_4 = (var_3 <= var_0);
    if (var_4) {
        // return 1                                                                           <L 57>
        goto label0;
    }
    // for local_node in range(8):                                                            <L 58>
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_7 = wp::address(var_cell_nodes, var_cell_idx, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_10 = wp::address(var_particle_q, var_8);
    var_12 = wp::load(var_10);
    var_11 = wp::sub(var_12, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_13 = wp::dot(var_11, var_11);
    var_14 = (var_13 <= var_0);
    if (var_14) {
        // return 1                                                                           <L 62>
        goto label1;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_17 = wp::address(var_cell_nodes, var_cell_idx, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_20 = wp::address(var_particle_q, var_18);
    var_22 = wp::load(var_20);
    var_21 = wp::sub(var_22, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_23 = wp::dot(var_21, var_21);
    var_24 = (var_23 <= var_0);
    if (var_24) {
        // return 1                                                                           <L 62>
        goto label2;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_27 = wp::address(var_cell_nodes, var_cell_idx, var_26);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_30 = wp::address(var_particle_q, var_28);
    var_32 = wp::load(var_30);
    var_31 = wp::sub(var_32, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_33 = wp::dot(var_31, var_31);
    var_34 = (var_33 <= var_0);
    if (var_34) {
        // return 1                                                                           <L 62>
        goto label3;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_37 = wp::address(var_cell_nodes, var_cell_idx, var_36);
    var_39 = wp::load(var_37);
    var_38 = wp::copy(var_39);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_40 = wp::address(var_particle_q, var_38);
    var_42 = wp::load(var_40);
    var_41 = wp::sub(var_42, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_43 = wp::dot(var_41, var_41);
    var_44 = (var_43 <= var_0);
    if (var_44) {
        // return 1                                                                           <L 62>
        goto label4;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_47 = wp::address(var_cell_nodes, var_cell_idx, var_46);
    var_49 = wp::load(var_47);
    var_48 = wp::copy(var_49);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_50 = wp::address(var_particle_q, var_48);
    var_52 = wp::load(var_50);
    var_51 = wp::sub(var_52, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_53 = wp::dot(var_51, var_51);
    var_54 = (var_53 <= var_0);
    if (var_54) {
        // return 1                                                                           <L 62>
        goto label5;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_57 = wp::address(var_cell_nodes, var_cell_idx, var_56);
    var_59 = wp::load(var_57);
    var_58 = wp::copy(var_59);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_60 = wp::address(var_particle_q, var_58);
    var_62 = wp::load(var_60);
    var_61 = wp::sub(var_62, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_63 = wp::dot(var_61, var_61);
    var_64 = (var_63 <= var_0);
    if (var_64) {
        // return 1                                                                           <L 62>
        goto label6;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_67 = wp::address(var_cell_nodes, var_cell_idx, var_66);
    var_69 = wp::load(var_67);
    var_68 = wp::copy(var_69);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_70 = wp::address(var_particle_q, var_68);
    var_72 = wp::load(var_70);
    var_71 = wp::sub(var_72, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_73 = wp::dot(var_71, var_71);
    var_74 = (var_73 <= var_0);
    if (var_74) {
        // return 1                                                                           <L 62>
        goto label7;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 59>
    var_77 = wp::address(var_cell_nodes, var_cell_idx, var_76);
    var_79 = wp::load(var_77);
    var_78 = wp::copy(var_79);
    // particle_delta = particle_q[node_idx] - sphere_centre                                  <L 60>
    var_80 = wp::address(var_particle_q, var_78);
    var_82 = wp::load(var_80);
    var_81 = wp::sub(var_82, var_sphere_centre);
    // if wp.dot(particle_delta, particle_delta) <= r2:                                       <L 61>
    var_83 = wp::dot(var_81, var_81);
    var_84 = (var_83 <= var_0);
    if (var_84) {
        // return 1                                                                           <L 62>
        goto label8;
    }
    // return 0                                                                               <L 63>
    goto label9;
    //---------
    // reverse
    label9:;
    adj_86 += adj_ret;
    // adj: return 0                                                                          <L 63>
    if (var_84) {
        label8:;
        adj_85 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_81, var_81, adj_81, adj_81, adj_83);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_82, var_sphere_centre, adj_80, adj_sphere_centre, adj_81);
    wp::adj_address(var_particle_q, var_78, adj_particle_q, adj_78, adj_80);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_79, adj_77, adj_78);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_76, adj_cell_nodes, adj_cell_idx, adj_76, adj_77);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_74) {
        label7:;
        adj_75 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_71, var_71, adj_71, adj_71, adj_73);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_72, var_sphere_centre, adj_70, adj_sphere_centre, adj_71);
    wp::adj_address(var_particle_q, var_68, adj_particle_q, adj_68, adj_70);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_69, adj_67, adj_68);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_66, adj_cell_nodes, adj_cell_idx, adj_66, adj_67);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_64) {
        label6:;
        adj_65 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_61, var_61, adj_61, adj_61, adj_63);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_62, var_sphere_centre, adj_60, adj_sphere_centre, adj_61);
    wp::adj_address(var_particle_q, var_58, adj_particle_q, adj_58, adj_60);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_59, adj_57, adj_58);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_56, adj_cell_nodes, adj_cell_idx, adj_56, adj_57);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_54) {
        label5:;
        adj_55 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_51, var_51, adj_51, adj_51, adj_53);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_52, var_sphere_centre, adj_50, adj_sphere_centre, adj_51);
    wp::adj_address(var_particle_q, var_48, adj_particle_q, adj_48, adj_50);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_49, adj_47, adj_48);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_46, adj_cell_nodes, adj_cell_idx, adj_46, adj_47);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_44) {
        label4:;
        adj_45 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_41, var_41, adj_41, adj_41, adj_43);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_42, var_sphere_centre, adj_40, adj_sphere_centre, adj_41);
    wp::adj_address(var_particle_q, var_38, adj_particle_q, adj_38, adj_40);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_39, adj_37, adj_38);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_36, adj_cell_nodes, adj_cell_idx, adj_36, adj_37);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_34) {
        label3:;
        adj_35 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_31, var_31, adj_31, adj_31, adj_33);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_32, var_sphere_centre, adj_30, adj_sphere_centre, adj_31);
    wp::adj_address(var_particle_q, var_28, adj_particle_q, adj_28, adj_30);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_29, adj_27, adj_28);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_26, adj_cell_nodes, adj_cell_idx, adj_26, adj_27);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_24) {
        label2:;
        adj_25 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_21, var_21, adj_21, adj_21, adj_23);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_22, var_sphere_centre, adj_20, adj_sphere_centre, adj_21);
    wp::adj_address(var_particle_q, var_18, adj_particle_q, adj_18, adj_20);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_19, adj_17, adj_18);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_16, adj_cell_nodes, adj_cell_idx, adj_16, adj_17);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    if (var_14) {
        label1:;
        adj_15 += adj_ret;
        // adj: return 1                                                                      <L 62>
    }
    wp::adj_dot(var_11, var_11, adj_11, adj_11, adj_13);
    // adj: if wp.dot(particle_delta, particle_delta) <= r2:                                  <L 61>
    wp::adj_sub(var_12, var_sphere_centre, adj_10, adj_sphere_centre, adj_11);
    wp::adj_address(var_particle_q, var_8, adj_particle_q, adj_8, adj_10);
    // adj: particle_delta = particle_q[node_idx] - sphere_centre                             <L 60>
    wp::adj_copy(var_9, adj_7, adj_8);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_6, adj_cell_nodes, adj_cell_idx, adj_6, adj_7);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 59>
    // adj: for local_node in range(8):                                                       <L 58>
    if (var_4) {
        label0:;
        adj_5 += adj_ret;
        // adj: return 1                                                                      <L 57>
    }
    wp::adj_dot(var_2, var_2, adj_2, adj_2, adj_3);
    // adj: if wp.dot(delta, delta) <= r2:                                                    <L 56>
    wp::adj_sub(var_1, var_sphere_centre, adj_1, adj_sphere_centre, adj_2);
    // adj: delta = centre - sphere_centre                                                    <L 55>
    adj__cell_deformed_center_0(var_cell_nodes, var_particle_q, var_cell_idx, adj_cell_nodes, adj_particle_q, adj_cell_idx, adj_1);
    // adj: centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)                  <L 54>
    wp::adj_mul(var_radius, var_radius, adj_radius, adj_radius, adj_0);
    // adj: r2 = radius * radius                                                              <L 53>
    // adj: def _cell_intersects_sphere(                                                      <L 46>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:15
static void adj__point_segment_distance_sq_1(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    const wp::float32 var_2 = 1e-20;
    bool var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::float32 var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    const wp::float32 var_11 = 0.0;
    wp::float32 var_12;
    const wp::float32 var_13 = 1.0;
    bool var_14;
    const wp::float32 var_15 = 1.0;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::float32 var_21;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    bool adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::float32 adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    bool adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    bool adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::float32 adj_21 = {};
    //---------
    // forward
    // def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:           <L 16>
    // ab = b - a                                                                             <L 17>
    var_0 = wp::sub(var_b, var_a);
    // denom = wp.dot(ab, ab)                                                                 <L 18>
    var_1 = wp::dot(var_0, var_0);
    // if denom <= 1.0e-20:                                                                   <L 19>
    var_3 = (var_1 <= var_2);
    if (var_3) {
        // d = p - a                                                                          <L 20>
        var_4 = wp::sub(var_p, var_a);
        // return wp.dot(d, d)                                                                <L 21>
        var_5 = wp::dot(var_4, var_4);
        goto label0;
    }
    // t = wp.dot(p - a, ab) / denom                                                          <L 23>
    var_6 = wp::sub(var_p, var_a);
    var_7 = wp::dot(var_6, var_0);
    var_8 = wp::div(var_7, var_1);
    // if t < 0.0:                                                                            <L 24>
    var_10 = (var_8 < var_9);
    if (var_10) {
        // t = 0.0                                                                            <L 25>
    }
    var_12 = wp::where(var_10, var_11, var_8);
    if (!var_10) {
        // elif t > 1.0:                                                                      <L 26>
        var_14 = (var_12 > var_13);
        if (var_14) {
            // t = 1.0                                                                        <L 27>
        }
        var_16 = wp::where(var_14, var_15, var_12);
    }
    var_17 = wp::where(var_10, var_12, var_16);
    // q = a + ab * t                                                                         <L 28>
    var_18 = wp::mul(var_0, var_17);
    var_19 = wp::add(var_a, var_18);
    // d = p - q                                                                              <L 29>
    var_20 = wp::sub(var_p, var_19);
    // return wp.dot(d, d)                                                                    <L 30>
    var_21 = wp::dot(var_20, var_20);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_21 += adj_ret;
    wp::adj_dot(var_20, var_20, adj_20, adj_20, adj_21);
    // adj: return wp.dot(d, d)                                                               <L 30>
    wp::adj_sub(var_p, var_19, adj_p, adj_19, adj_20);
    // adj: d = p - q                                                                         <L 29>
    wp::adj_add(var_a, var_18, adj_a, adj_18, adj_19);
    wp::adj_mul(var_0, var_17, adj_0, adj_17, adj_18);
    // adj: q = a + ab * t                                                                    <L 28>
    wp::adj_where(var_10, var_12, var_16, adj_10, adj_12, adj_16, adj_17);
    if (!var_10) {
        wp::adj_where(var_14, var_15, var_12, adj_14, adj_15, adj_12, adj_16);
        if (var_14) {
            // adj: t = 1.0                                                                   <L 27>
        }
        // adj: elif t > 1.0:                                                                 <L 26>
    }
    wp::adj_where(var_10, var_11, var_8, adj_10, adj_11, adj_8, adj_12);
    if (var_10) {
        // adj: t = 0.0                                                                       <L 25>
    }
    // adj: if t < 0.0:                                                                       <L 24>
    wp::adj_div(var_7, var_1, var_8, adj_7, adj_1, adj_8);
    wp::adj_dot(var_6, var_0, adj_6, adj_0, adj_7);
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_6);
    // adj: t = wp.dot(p - a, ab) / denom                                                     <L 23>
    if (var_3) {
        label0:;
        adj_5 += adj_ret;
        wp::adj_dot(var_4, var_4, adj_4, adj_4, adj_5);
        // adj: return wp.dot(d, d)                                                           <L 21>
        wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_4);
        // adj: d = p - a                                                                     <L 20>
    }
    // adj: if denom <= 1.0e-20:                                                              <L 19>
    wp::adj_dot(var_0, var_0, adj_0, adj_0, adj_1);
    // adj: denom = wp.dot(ab, ab)                                                            <L 18>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: ab = b - a                                                                        <L 17>
    // adj: def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:      <L 16>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_heat.py:66
static void adj__cell_intersects_capsule_0(
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::int32 var_cell_idx,
    wp::vec_t<3, wp::float32> var_capsule_p0,
    wp::vec_t<3, wp::float32> var_capsule_p1,
    wp::float32 var_radius,
    wp::array_t<wp::int32> & adj_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_particle_q,
    wp::int32 & adj_cell_idx,
    wp::vec_t<3, wp::float32> & adj_capsule_p0,
    wp::vec_t<3, wp::float32> & adj_capsule_p1,
    wp::float32 & adj_radius,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    bool var_3;
    const wp::int32 var_4 = 1;
    const wp::int32 var_5 = 0;
    wp::int32* var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    bool var_12;
    const wp::int32 var_13 = 1;
    const wp::int32 var_14 = 1;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::float32 var_19;
    wp::vec_t<3, wp::float32> var_20;
    bool var_21;
    const wp::int32 var_22 = 1;
    const wp::int32 var_23 = 2;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::float32 var_28;
    wp::vec_t<3, wp::float32> var_29;
    bool var_30;
    const wp::int32 var_31 = 1;
    const wp::int32 var_32 = 3;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::float32 var_37;
    wp::vec_t<3, wp::float32> var_38;
    bool var_39;
    const wp::int32 var_40 = 1;
    const wp::int32 var_41 = 4;
    wp::int32* var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::vec_t<3, wp::float32>* var_45;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    bool var_48;
    const wp::int32 var_49 = 1;
    const wp::int32 var_50 = 5;
    wp::int32* var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::vec_t<3, wp::float32>* var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    bool var_57;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 6;
    wp::int32* var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::float32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    bool var_66;
    const wp::int32 var_67 = 1;
    const wp::int32 var_68 = 7;
    wp::int32* var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::vec_t<3, wp::float32>* var_72;
    wp::float32 var_73;
    wp::vec_t<3, wp::float32> var_74;
    bool var_75;
    const wp::int32 var_76 = 1;
    const wp::int32 var_77 = 0;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::float32 adj_2 = {};
    bool adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::float32 adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    bool adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::float32 adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    bool adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::float32 adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    bool adj_30 = {};
    wp::int32 adj_31 = {};
    wp::int32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::float32 adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    bool adj_39 = {};
    wp::int32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::int32 adj_43 = {};
    wp::int32 adj_44 = {};
    wp::vec_t<3, wp::float32> adj_45 = {};
    wp::float32 adj_46 = {};
    wp::vec_t<3, wp::float32> adj_47 = {};
    bool adj_48 = {};
    wp::int32 adj_49 = {};
    wp::int32 adj_50 = {};
    wp::int32 adj_51 = {};
    wp::int32 adj_52 = {};
    wp::int32 adj_53 = {};
    wp::vec_t<3, wp::float32> adj_54 = {};
    wp::float32 adj_55 = {};
    wp::vec_t<3, wp::float32> adj_56 = {};
    bool adj_57 = {};
    wp::int32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::int32 adj_60 = {};
    wp::int32 adj_61 = {};
    wp::int32 adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::float32 adj_64 = {};
    wp::vec_t<3, wp::float32> adj_65 = {};
    bool adj_66 = {};
    wp::int32 adj_67 = {};
    wp::int32 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::int32 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::vec_t<3, wp::float32> adj_72 = {};
    wp::float32 adj_73 = {};
    wp::vec_t<3, wp::float32> adj_74 = {};
    bool adj_75 = {};
    wp::int32 adj_76 = {};
    wp::int32 adj_77 = {};
    //---------
    // forward
    // def _cell_intersects_capsule(                                                          <L 67>
    // r2 = radius * radius                                                                   <L 75>
    var_0 = wp::mul(var_radius, var_radius);
    // centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)                       <L 76>
    var_1 = _cell_deformed_center_0(var_cell_nodes, var_particle_q, var_cell_idx);
    // if _point_segment_distance_sq(centre, capsule_p0, capsule_p1) <= r2:                   <L 77>
    var_2 = _point_segment_distance_sq_1(var_1, var_capsule_p0, var_capsule_p1);
    var_3 = (var_2 <= var_0);
    if (var_3) {
        // return 1                                                                           <L 78>
        goto label0;
    }
    // for local_node in range(8):                                                            <L 79>
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_6 = wp::address(var_cell_nodes, var_cell_idx, var_5);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_9 = wp::address(var_particle_q, var_7);
    var_11 = wp::load(var_9);
    var_10 = _point_segment_distance_sq_1(var_11, var_capsule_p0, var_capsule_p1);
    var_12 = (var_10 <= var_0);
    if (var_12) {
        // return 1                                                                           <L 82>
        goto label1;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_15 = wp::address(var_cell_nodes, var_cell_idx, var_14);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_18 = wp::address(var_particle_q, var_16);
    var_20 = wp::load(var_18);
    var_19 = _point_segment_distance_sq_1(var_20, var_capsule_p0, var_capsule_p1);
    var_21 = (var_19 <= var_0);
    if (var_21) {
        // return 1                                                                           <L 82>
        goto label2;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_24 = wp::address(var_cell_nodes, var_cell_idx, var_23);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_27 = wp::address(var_particle_q, var_25);
    var_29 = wp::load(var_27);
    var_28 = _point_segment_distance_sq_1(var_29, var_capsule_p0, var_capsule_p1);
    var_30 = (var_28 <= var_0);
    if (var_30) {
        // return 1                                                                           <L 82>
        goto label3;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_33 = wp::address(var_cell_nodes, var_cell_idx, var_32);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_36 = wp::address(var_particle_q, var_34);
    var_38 = wp::load(var_36);
    var_37 = _point_segment_distance_sq_1(var_38, var_capsule_p0, var_capsule_p1);
    var_39 = (var_37 <= var_0);
    if (var_39) {
        // return 1                                                                           <L 82>
        goto label4;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_42 = wp::address(var_cell_nodes, var_cell_idx, var_41);
    var_44 = wp::load(var_42);
    var_43 = wp::copy(var_44);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_45 = wp::address(var_particle_q, var_43);
    var_47 = wp::load(var_45);
    var_46 = _point_segment_distance_sq_1(var_47, var_capsule_p0, var_capsule_p1);
    var_48 = (var_46 <= var_0);
    if (var_48) {
        // return 1                                                                           <L 82>
        goto label5;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_51 = wp::address(var_cell_nodes, var_cell_idx, var_50);
    var_53 = wp::load(var_51);
    var_52 = wp::copy(var_53);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_54 = wp::address(var_particle_q, var_52);
    var_56 = wp::load(var_54);
    var_55 = _point_segment_distance_sq_1(var_56, var_capsule_p0, var_capsule_p1);
    var_57 = (var_55 <= var_0);
    if (var_57) {
        // return 1                                                                           <L 82>
        goto label6;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_60 = wp::address(var_cell_nodes, var_cell_idx, var_59);
    var_62 = wp::load(var_60);
    var_61 = wp::copy(var_62);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_63 = wp::address(var_particle_q, var_61);
    var_65 = wp::load(var_63);
    var_64 = _point_segment_distance_sq_1(var_65, var_capsule_p0, var_capsule_p1);
    var_66 = (var_64 <= var_0);
    if (var_66) {
        // return 1                                                                           <L 82>
        goto label7;
    }
    // node_idx = cell_nodes[cell_idx, local_node]                                            <L 80>
    var_69 = wp::address(var_cell_nodes, var_cell_idx, var_68);
    var_71 = wp::load(var_69);
    var_70 = wp::copy(var_71);
    // if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:       <L 81>
    var_72 = wp::address(var_particle_q, var_70);
    var_74 = wp::load(var_72);
    var_73 = _point_segment_distance_sq_1(var_74, var_capsule_p0, var_capsule_p1);
    var_75 = (var_73 <= var_0);
    if (var_75) {
        // return 1                                                                           <L 82>
        goto label8;
    }
    // return 0                                                                               <L 83>
    goto label9;
    //---------
    // reverse
    label9:;
    adj_77 += adj_ret;
    // adj: return 0                                                                          <L 83>
    if (var_75) {
        label8:;
        adj_76 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_74, var_capsule_p0, var_capsule_p1, adj_72, adj_capsule_p0, adj_capsule_p1, adj_73);
    wp::adj_address(var_particle_q, var_70, adj_particle_q, adj_70, adj_72);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_71, adj_69, adj_70);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_68, adj_cell_nodes, adj_cell_idx, adj_68, adj_69);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_66) {
        label7:;
        adj_67 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_65, var_capsule_p0, var_capsule_p1, adj_63, adj_capsule_p0, adj_capsule_p1, adj_64);
    wp::adj_address(var_particle_q, var_61, adj_particle_q, adj_61, adj_63);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_62, adj_60, adj_61);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_59, adj_cell_nodes, adj_cell_idx, adj_59, adj_60);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_57) {
        label6:;
        adj_58 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_56, var_capsule_p0, var_capsule_p1, adj_54, adj_capsule_p0, adj_capsule_p1, adj_55);
    wp::adj_address(var_particle_q, var_52, adj_particle_q, adj_52, adj_54);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_53, adj_51, adj_52);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_50, adj_cell_nodes, adj_cell_idx, adj_50, adj_51);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_48) {
        label5:;
        adj_49 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_47, var_capsule_p0, var_capsule_p1, adj_45, adj_capsule_p0, adj_capsule_p1, adj_46);
    wp::adj_address(var_particle_q, var_43, adj_particle_q, adj_43, adj_45);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_44, adj_42, adj_43);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_41, adj_cell_nodes, adj_cell_idx, adj_41, adj_42);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_39) {
        label4:;
        adj_40 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_38, var_capsule_p0, var_capsule_p1, adj_36, adj_capsule_p0, adj_capsule_p1, adj_37);
    wp::adj_address(var_particle_q, var_34, adj_particle_q, adj_34, adj_36);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_35, adj_33, adj_34);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_32, adj_cell_nodes, adj_cell_idx, adj_32, adj_33);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_30) {
        label3:;
        adj_31 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_29, var_capsule_p0, var_capsule_p1, adj_27, adj_capsule_p0, adj_capsule_p1, adj_28);
    wp::adj_address(var_particle_q, var_25, adj_particle_q, adj_25, adj_27);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_26, adj_24, adj_25);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_23, adj_cell_nodes, adj_cell_idx, adj_23, adj_24);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_21) {
        label2:;
        adj_22 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_20, var_capsule_p0, var_capsule_p1, adj_18, adj_capsule_p0, adj_capsule_p1, adj_19);
    wp::adj_address(var_particle_q, var_16, adj_particle_q, adj_16, adj_18);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_17, adj_15, adj_16);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_14, adj_cell_nodes, adj_cell_idx, adj_14, adj_15);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    if (var_12) {
        label1:;
        adj_13 += adj_ret;
        // adj: return 1                                                                      <L 82>
    }
    adj__point_segment_distance_sq_1(var_11, var_capsule_p0, var_capsule_p1, adj_9, adj_capsule_p0, adj_capsule_p1, adj_10);
    wp::adj_address(var_particle_q, var_7, adj_particle_q, adj_7, adj_9);
    // adj: if _point_segment_distance_sq(particle_q[node_idx], capsule_p0, capsule_p1) <= r2:  <L 81>
    wp::adj_copy(var_8, adj_6, adj_7);
    wp::adj_address(var_cell_nodes, var_cell_idx, var_5, adj_cell_nodes, adj_cell_idx, adj_5, adj_6);
    // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 80>
    // adj: for local_node in range(8):                                                       <L 79>
    if (var_3) {
        label0:;
        adj_4 += adj_ret;
        // adj: return 1                                                                      <L 78>
    }
    adj__point_segment_distance_sq_1(var_1, var_capsule_p0, var_capsule_p1, adj_1, adj_capsule_p0, adj_capsule_p1, adj_2);
    // adj: if _point_segment_distance_sq(centre, capsule_p0, capsule_p1) <= r2:              <L 77>
    adj__cell_deformed_center_0(var_cell_nodes, var_particle_q, var_cell_idx, adj_cell_nodes, adj_particle_q, adj_cell_idx, adj_1);
    // adj: centre = _cell_deformed_center(cell_nodes, particle_q, cell_idx)                  <L 76>
    wp::adj_mul(var_radius, var_radius, adj_radius, adj_radius, adj_0);
    // adj: r2 = radius * radius                                                              <L 75>
    // adj: def _cell_intersects_capsule(                                                     <L 67>
    return;
}

struct wp_args_apply_diathermy_spheres_kernel_f18770dd {
    wp::int32 num_cells;
    wp::array_t<wp::int32> cell_nodes;
    wp::array_t<wp::int32> cell_material;
    wp::array_t<wp::int32> cell_active;
    wp::array_t<wp::int32> material_cuttable;
    wp::int32 has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_q;
    wp::array_t<wp::int32> sphere_enabled;
    wp::array_t<wp::float32> sphere_power;
    wp::int32 has_sphere_power;
    wp::int32 sphere_count;
    wp::float32 sphere_radius;
    wp::float32 heat_delta;
    wp::array_t<wp::float32> cell_heat;
};


void apply_diathermy_spheres_kernel_f18770dd_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_diathermy_spheres_kernel_f18770dd *_wp_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_nodes = _wp_args->cell_nodes;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::int32> var_material_cuttable = _wp_args->material_cuttable;
    wp::int32 var_has_material_filter = _wp_args->has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q = _wp_args->particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q = _wp_args->sphere_q;
    wp::array_t<wp::int32> var_sphere_enabled = _wp_args->sphere_enabled;
    wp::array_t<wp::float32> var_sphere_power = _wp_args->sphere_power;
    wp::int32 var_has_sphere_power = _wp_args->has_sphere_power;
    wp::int32 var_sphere_count = _wp_args->sphere_count;
    wp::float32 var_sphere_radius = _wp_args->sphere_radius;
    wp::float32 var_heat_delta = _wp_args->heat_delta;
    wp::array_t<wp::float32> var_cell_heat = _wp_args->cell_heat;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    bool var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::int32* var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    wp::float32 var_21;
    wp::range_t var_22;
    wp::int32 var_23;
    wp::int32* var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    wp::int32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::int32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    bool var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    wp::float32* var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    wp::float32* var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    //---------
    // forward
    // def apply_diathermy_spheres_kernel(                                                    <L 87>
    // c = wp.tid()                                                                           <L 105>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 106>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 107>
        return;
    }
    // if cell_active[c] == 0:                                                                <L 108>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 109>
        return;
    }
    // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 110>
    var_8 = (var_has_material_filter != var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_cell_material, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_material_cuttable, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_6 = var_6 && var_13;
    }
    if (var_6) {
        // return                                                                             <L 111>
        return;
    }
    // if sphere_radius <= 0.0 or heat_delta <= 0.0:                                          <L 112>
    var_17 = (var_sphere_radius <= var_16);
    var_15 = var_17;
    if (!var_15) {
        var_19 = (var_heat_delta <= var_18);
        var_15 = var_15 || var_19;
    }
    if (var_15) {
        // return                                                                             <L 113>
        return;
    }
    // total_delta = float(0.0)                                                               <L 115>
    var_21 = wp::float(var_20);
    // for sphere_idx in range(sphere_count):                                                 <L 116>
    var_22 = wp::range(var_sphere_count);
    start_for_4:;
        if (iter_cmp(var_22) == 0) goto end_for_4;
        var_23 = wp::iter_next(var_22);
        // if sphere_enabled[sphere_idx] == 0:                                                <L 117>
        var_24 = wp::address(var_sphere_enabled, var_23);
        var_27 = wp::load(var_24);
        var_26 = (var_27 == var_25);
        if (var_26) {
            // continue                                                                       <L 118>
            goto start_for_4;
        }
        // if _cell_intersects_sphere(cell_nodes, particle_q, c, sphere_q[sphere_idx], sphere_radius) != 0:       <L 119>
        var_28 = wp::address(var_sphere_q, var_23);
        var_30 = wp::load(var_28);
        var_29 = _cell_intersects_sphere_0(var_cell_nodes, var_particle_q, var_0, var_30, var_sphere_radius);
        var_32 = (var_29 != var_31);
        if (var_32) {
            // delta = heat_delta                                                             <L 120>
            var_33 = wp::copy(var_heat_delta);
            // if has_sphere_power != 0:                                                      <L 121>
            var_35 = (var_has_sphere_power != var_34);
            if (var_35) {
                // delta = heat_delta * sphere_power[sphere_idx]                              <L 122>
                var_36 = wp::address(var_sphere_power, var_23);
                var_38 = wp::load(var_36);
                var_37 = wp::mul(var_heat_delta, var_38);
            }
            var_39 = wp::where(var_35, var_37, var_33);
            // if delta > 0.0:                                                                <L 123>
            var_41 = (var_39 > var_40);
            if (var_41) {
                // total_delta += delta                                                       <L 124>
                var_42 = wp::add(var_21, var_39);
            }
            var_43 = wp::where(var_41, var_42, var_21);
        }
        var_44 = wp::where(var_32, var_43, var_21);
        wp::assign(var_21, var_44);
        goto start_for_4;
    end_for_4:;
    // if total_delta > 0.0:                                                                  <L 125>
    var_46 = (var_21 > var_45);
    if (var_46) {
        // cell_heat[c] = cell_heat[c] + total_delta                                          <L 126>
        var_47 = wp::address(var_cell_heat, var_0);
        var_49 = wp::load(var_47);
        var_48 = wp::add(var_49, var_21);
        wp::array_store(var_cell_heat, var_0, var_48);
    }
}



void apply_diathermy_spheres_kernel_f18770dd_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_diathermy_spheres_kernel_f18770dd *_wp_args,
    wp_args_apply_diathermy_spheres_kernel_f18770dd *_wp_adj_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_nodes = _wp_args->cell_nodes;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::int32> var_material_cuttable = _wp_args->material_cuttable;
    wp::int32 var_has_material_filter = _wp_args->has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q = _wp_args->particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q = _wp_args->sphere_q;
    wp::array_t<wp::int32> var_sphere_enabled = _wp_args->sphere_enabled;
    wp::array_t<wp::float32> var_sphere_power = _wp_args->sphere_power;
    wp::int32 var_has_sphere_power = _wp_args->has_sphere_power;
    wp::int32 var_sphere_count = _wp_args->sphere_count;
    wp::float32 var_sphere_radius = _wp_args->sphere_radius;
    wp::float32 var_heat_delta = _wp_args->heat_delta;
    wp::array_t<wp::float32> var_cell_heat = _wp_args->cell_heat;
    wp::int32 adj_num_cells = _wp_adj_args->num_cells;
    wp::array_t<wp::int32> adj_cell_nodes = _wp_adj_args->cell_nodes;
    wp::array_t<wp::int32> adj_cell_material = _wp_adj_args->cell_material;
    wp::array_t<wp::int32> adj_cell_active = _wp_adj_args->cell_active;
    wp::array_t<wp::int32> adj_material_cuttable = _wp_adj_args->material_cuttable;
    wp::int32 adj_has_material_filter = _wp_adj_args->has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q = _wp_adj_args->particle_q;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_q = _wp_adj_args->sphere_q;
    wp::array_t<wp::int32> adj_sphere_enabled = _wp_adj_args->sphere_enabled;
    wp::array_t<wp::float32> adj_sphere_power = _wp_adj_args->sphere_power;
    wp::int32 adj_has_sphere_power = _wp_adj_args->has_sphere_power;
    wp::int32 adj_sphere_count = _wp_adj_args->sphere_count;
    wp::float32 adj_sphere_radius = _wp_adj_args->sphere_radius;
    wp::float32 adj_heat_delta = _wp_adj_args->heat_delta;
    wp::array_t<wp::float32> adj_cell_heat = _wp_adj_args->cell_heat;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    bool var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::int32* var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    wp::int32 var_14;
    bool var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    wp::float32 var_21;
    wp::range_t var_22;
    wp::int32 var_23;
    wp::int32* var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    wp::int32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::int32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    bool var_32;
    wp::float32 var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    wp::float32* var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 0.0;
    bool var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    wp::float32* var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::int32 adj_7 = {};
    bool adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::int32 adj_12 = {};
    bool adj_13 = {};
    wp::int32 adj_14 = {};
    bool adj_15 = {};
    wp::float32 adj_16 = {};
    bool adj_17 = {};
    wp::float32 adj_18 = {};
    bool adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::range_t adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    bool adj_26 = {};
    wp::int32 adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::int32 adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::int32 adj_31 = {};
    bool adj_32 = {};
    wp::float32 adj_33 = {};
    wp::int32 adj_34 = {};
    bool adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    bool adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    bool adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    //---------
    // forward
    // def apply_diathermy_spheres_kernel(                                                    <L 87>
    // c = wp.tid()                                                                           <L 105>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 106>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 107>
        goto label0;
    }
    // if cell_active[c] == 0:                                                                <L 108>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 109>
        goto label1;
    }
    // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 110>
    var_8 = (var_has_material_filter != var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_cell_material, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_material_cuttable, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_6 = var_6 && var_13;
    }
    if (var_6) {
        // return                                                                             <L 111>
        goto label2;
    }
    // if sphere_radius <= 0.0 or heat_delta <= 0.0:                                          <L 112>
    var_17 = (var_sphere_radius <= var_16);
    var_15 = var_17;
    if (!var_15) {
        var_19 = (var_heat_delta <= var_18);
        var_15 = var_15 || var_19;
    }
    if (var_15) {
        // return                                                                             <L 113>
        goto label3;
    }
    // total_delta = float(0.0)                                                               <L 115>
    var_21 = wp::float(var_20);
    // for sphere_idx in range(sphere_count):                                                 <L 116>
    var_22 = wp::range(var_sphere_count);
    // if total_delta > 0.0:                                                                  <L 125>
    var_46 = (var_21 > var_45);
    if (var_46) {
        // cell_heat[c] = cell_heat[c] + total_delta                                          <L 126>
        var_47 = wp::address(var_cell_heat, var_0);
        var_49 = wp::load(var_47);
        var_48 = wp::add(var_49, var_21);
        // wp::array_store(var_cell_heat, var_0, var_48);
    }
    //---------
    // reverse
    if (var_46) {
        wp::adj_array_store(var_cell_heat, var_0, var_48, adj_cell_heat, adj_0, adj_48);
        wp::adj_add(var_49, var_21, adj_47, adj_21, adj_48);
        wp::adj_address(var_cell_heat, var_0, adj_cell_heat, adj_0, adj_47);
        // adj: cell_heat[c] = cell_heat[c] + total_delta                                     <L 126>
    }
    // adj: if total_delta > 0.0:                                                             <L 125>
    var_22 = wp::iter_reverse(var_22);
    start_for_4:;
        if (iter_cmp(var_22) == 0) goto end_for_4;
        var_23 = wp::iter_next(var_22);
    	adj_24 = {};
    	adj_25 = {};
    	adj_26 = {};
    	adj_27 = {};
    	adj_28 = {};
    	adj_29 = {};
    	adj_30 = {};
    	adj_31 = {};
    	adj_32 = {};
    	adj_33 = {};
    	adj_34 = {};
    	adj_35 = {};
    	adj_36 = {};
    	adj_37 = {};
    	adj_38 = {};
    	adj_39 = {};
    	adj_40 = {};
    	adj_41 = {};
    	adj_42 = {};
    	adj_43 = {};
    	adj_44 = {};
        // if sphere_enabled[sphere_idx] == 0:                                                <L 117>
        var_24 = wp::address(var_sphere_enabled, var_23);
        var_27 = wp::load(var_24);
        var_26 = (var_27 == var_25);
        if (var_26) {
            // continue                                                                       <L 118>
            goto start_for_4;
        }
        // if _cell_intersects_sphere(cell_nodes, particle_q, c, sphere_q[sphere_idx], sphere_radius) != 0:       <L 119>
        var_28 = wp::address(var_sphere_q, var_23);
        var_30 = wp::load(var_28);
        var_29 = _cell_intersects_sphere_0(var_cell_nodes, var_particle_q, var_0, var_30, var_sphere_radius);
        var_32 = (var_29 != var_31);
        if (var_32) {
            // delta = heat_delta                                                             <L 120>
            var_33 = wp::copy(var_heat_delta);
            // if has_sphere_power != 0:                                                      <L 121>
            var_35 = (var_has_sphere_power != var_34);
            if (var_35) {
                // delta = heat_delta * sphere_power[sphere_idx]                              <L 122>
                var_36 = wp::address(var_sphere_power, var_23);
                var_38 = wp::load(var_36);
                var_37 = wp::mul(var_heat_delta, var_38);
            }
            var_39 = wp::where(var_35, var_37, var_33);
            // if delta > 0.0:                                                                <L 123>
            var_41 = (var_39 > var_40);
            if (var_41) {
                // total_delta += delta                                                       <L 124>
                var_42 = wp::add(var_21, var_39);
            }
            var_43 = wp::where(var_41, var_42, var_21);
        }
        var_44 = wp::where(var_32, var_43, var_21);
        wp::assign(var_21, var_44);
        wp::adj_assign(var_21, var_44, adj_21, adj_44);
        wp::adj_where(var_32, var_43, var_21, adj_32, adj_43, adj_21, adj_44);
        if (var_32) {
            wp::adj_where(var_41, var_42, var_21, adj_41, adj_42, adj_21, adj_43);
            if (var_41) {
                wp::adj_add(var_21, var_39, adj_21, adj_39, adj_42);
                // adj: total_delta += delta                                                  <L 124>
            }
            // adj: if delta > 0.0:                                                           <L 123>
            wp::adj_where(var_35, var_37, var_33, adj_35, adj_37, adj_33, adj_39);
            if (var_35) {
                wp::adj_mul(var_heat_delta, var_38, adj_heat_delta, adj_36, adj_37);
                wp::adj_address(var_sphere_power, var_23, adj_sphere_power, adj_23, adj_36);
                // adj: delta = heat_delta * sphere_power[sphere_idx]                         <L 122>
            }
            // adj: if has_sphere_power != 0:                                                 <L 121>
            wp::adj_copy(var_heat_delta, adj_heat_delta, adj_33);
            // adj: delta = heat_delta                                                        <L 120>
        }
        adj__cell_intersects_sphere_0(var_cell_nodes, var_particle_q, var_0, var_30, var_sphere_radius, adj_cell_nodes, adj_particle_q, adj_0, adj_28, adj_sphere_radius, adj_29);
        wp::adj_address(var_sphere_q, var_23, adj_sphere_q, adj_23, adj_28);
        // adj: if _cell_intersects_sphere(cell_nodes, particle_q, c, sphere_q[sphere_idx], sphere_radius) != 0:  <L 119>
        if (var_26) {
            // adj: continue                                                                  <L 118>
        }
        wp::adj_address(var_sphere_enabled, var_23, adj_sphere_enabled, adj_23, adj_24);
        // adj: if sphere_enabled[sphere_idx] == 0:                                           <L 117>
    	goto start_for_4;
    end_for_4:;
    wp::adj_range(var_sphere_count, adj_sphere_count, adj_22);
    // adj: for sphere_idx in range(sphere_count):                                            <L 116>
    wp::adj_float(var_20, adj_20, adj_21);
    // adj: total_delta = float(0.0)                                                          <L 115>
    if (var_15) {
        label3:;
        // adj: return                                                                        <L 113>
    }
    if (!var_15) {
    }
    // adj: if sphere_radius <= 0.0 or heat_delta <= 0.0:                                     <L 112>
    if (var_6) {
        label2:;
        // adj: return                                                                        <L 111>
    }
    if (var_6) {
        wp::adj_address(var_material_cuttable, var_11, adj_material_cuttable, adj_9, adj_10);
        wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_9);
    }
    // adj: if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:         <L 110>
    if (var_4) {
        label1:;
        // adj: return                                                                        <L 109>
    }
    wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
    // adj: if cell_active[c] == 0:                                                           <L 108>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 107>
    }
    // adj: if c >= num_cells:                                                                <L 106>
    // adj: c = wp.tid()                                                                      <L 105>
    // adj: def apply_diathermy_spheres_kernel(                                               <L 87>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void apply_diathermy_spheres_kernel_f18770dd_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_apply_diathermy_spheres_kernel_f18770dd *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_diathermy_spheres_kernel_f18770dd_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void apply_diathermy_spheres_kernel_f18770dd_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_apply_diathermy_spheres_kernel_f18770dd *_wp_args,
    wp_args_apply_diathermy_spheres_kernel_f18770dd *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_diathermy_spheres_kernel_f18770dd_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_select_capsule_cells_kernel_eec2ca80 {
    wp::int32 num_cells;
    wp::array_t<wp::int32> cell_nodes;
    wp::array_t<wp::int32> cell_material;
    wp::array_t<wp::int32> cell_active;
    wp::array_t<wp::int32> material_cuttable;
    wp::int32 has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> particle_q;
    wp::vec_t<3, wp::float32> capsule_p0;
    wp::vec_t<3, wp::float32> capsule_p1;
    wp::float32 radius;
    wp::array_t<wp::int32> selected_cells;
    wp::array_t<wp::int32> selected_count;
};


void select_capsule_cells_kernel_eec2ca80_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_select_capsule_cells_kernel_eec2ca80 *_wp_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_nodes = _wp_args->cell_nodes;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::int32> var_material_cuttable = _wp_args->material_cuttable;
    wp::int32 var_has_material_filter = _wp_args->has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q = _wp_args->particle_q;
    wp::vec_t<3, wp::float32> var_capsule_p0 = _wp_args->capsule_p0;
    wp::vec_t<3, wp::float32> var_capsule_p1 = _wp_args->capsule_p1;
    wp::float32 var_radius = _wp_args->radius;
    wp::array_t<wp::int32> var_selected_cells = _wp_args->selected_cells;
    wp::array_t<wp::int32> var_selected_count = _wp_args->selected_count;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    bool var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::int32* var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    wp::int32 var_14;
    const wp::float32 var_15 = 0.0;
    bool var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 0;
    bool var_19;
    const wp::int32 var_20 = 0;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    //---------
    // forward
    // def select_capsule_cells_kernel(                                                       <L 254>
    // c = wp.tid()                                                                           <L 269>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 270>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 271>
        return;
    }
    // if cell_active[c] == 0:                                                                <L 272>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 273>
        return;
    }
    // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 274>
    var_8 = (var_has_material_filter != var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_cell_material, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_material_cuttable, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_6 = var_6 && var_13;
    }
    if (var_6) {
        // return                                                                             <L 275>
        return;
    }
    // if radius < 0.0:                                                                       <L 276>
    var_16 = (var_radius < var_15);
    if (var_16) {
        // return                                                                             <L 277>
        return;
    }
    // if _cell_intersects_capsule(cell_nodes, particle_q, c, capsule_p0, capsule_p1, radius) == 0:       <L 279>
    var_17 = _cell_intersects_capsule_0(var_cell_nodes, var_particle_q, var_0, var_capsule_p0, var_capsule_p1, var_radius);
    var_19 = (var_17 == var_18);
    if (var_19) {
        // return                                                                             <L 280>
        return;
    }
    // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 281>
    var_22 = wp::atomic_add(var_selected_count, var_20, var_21);
    // selected_cells[out_idx] = c                                                            <L 282>
    wp::array_store(var_selected_cells, var_22, var_0);
}



void select_capsule_cells_kernel_eec2ca80_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_select_capsule_cells_kernel_eec2ca80 *_wp_args,
    wp_args_select_capsule_cells_kernel_eec2ca80 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_nodes = _wp_args->cell_nodes;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::int32> var_material_cuttable = _wp_args->material_cuttable;
    wp::int32 var_has_material_filter = _wp_args->has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q = _wp_args->particle_q;
    wp::vec_t<3, wp::float32> var_capsule_p0 = _wp_args->capsule_p0;
    wp::vec_t<3, wp::float32> var_capsule_p1 = _wp_args->capsule_p1;
    wp::float32 var_radius = _wp_args->radius;
    wp::array_t<wp::int32> var_selected_cells = _wp_args->selected_cells;
    wp::array_t<wp::int32> var_selected_count = _wp_args->selected_count;
    wp::int32 adj_num_cells = _wp_adj_args->num_cells;
    wp::array_t<wp::int32> adj_cell_nodes = _wp_adj_args->cell_nodes;
    wp::array_t<wp::int32> adj_cell_material = _wp_adj_args->cell_material;
    wp::array_t<wp::int32> adj_cell_active = _wp_adj_args->cell_active;
    wp::array_t<wp::int32> adj_material_cuttable = _wp_adj_args->material_cuttable;
    wp::int32 adj_has_material_filter = _wp_adj_args->has_material_filter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q = _wp_adj_args->particle_q;
    wp::vec_t<3, wp::float32> adj_capsule_p0 = _wp_adj_args->capsule_p0;
    wp::vec_t<3, wp::float32> adj_capsule_p1 = _wp_adj_args->capsule_p1;
    wp::float32 adj_radius = _wp_adj_args->radius;
    wp::array_t<wp::int32> adj_selected_cells = _wp_adj_args->selected_cells;
    wp::array_t<wp::int32> adj_selected_count = _wp_adj_args->selected_count;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    bool var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    wp::int32* var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    const wp::int32 var_12 = 0;
    bool var_13;
    wp::int32 var_14;
    const wp::float32 var_15 = 0.0;
    bool var_16;
    wp::int32 var_17;
    const wp::int32 var_18 = 0;
    bool var_19;
    const wp::int32 var_20 = 0;
    const wp::int32 var_21 = 1;
    wp::int32 var_22;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::int32 adj_7 = {};
    bool adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::int32 adj_12 = {};
    bool adj_13 = {};
    wp::int32 adj_14 = {};
    wp::float32 adj_15 = {};
    bool adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    bool adj_19 = {};
    wp::int32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    //---------
    // forward
    // def select_capsule_cells_kernel(                                                       <L 254>
    // c = wp.tid()                                                                           <L 269>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 270>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 271>
        goto label0;
    }
    // if cell_active[c] == 0:                                                                <L 272>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 273>
        goto label1;
    }
    // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 274>
    var_8 = (var_has_material_filter != var_7);
    var_6 = var_8;
    if (var_6) {
        var_9 = wp::address(var_cell_material, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_material_cuttable, var_11);
        var_14 = wp::load(var_10);
        var_13 = (var_14 == var_12);
        var_6 = var_6 && var_13;
    }
    if (var_6) {
        // return                                                                             <L 275>
        goto label2;
    }
    // if radius < 0.0:                                                                       <L 276>
    var_16 = (var_radius < var_15);
    if (var_16) {
        // return                                                                             <L 277>
        goto label3;
    }
    // if _cell_intersects_capsule(cell_nodes, particle_q, c, capsule_p0, capsule_p1, radius) == 0:       <L 279>
    var_17 = _cell_intersects_capsule_0(var_cell_nodes, var_particle_q, var_0, var_capsule_p0, var_capsule_p1, var_radius);
    var_19 = (var_17 == var_18);
    if (var_19) {
        // return                                                                             <L 280>
        goto label4;
    }
    // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 281>
    // var_22 = wp::atomic_add(var_selected_count, var_20, var_21);
    // selected_cells[out_idx] = c                                                            <L 282>
    // wp::array_store(var_selected_cells, var_22, var_0);
    //---------
    // reverse
    wp::adj_array_store(var_selected_cells, var_22, var_0, adj_selected_cells, adj_22, adj_0);
    // adj: selected_cells[out_idx] = c                                                       <L 282>
    wp::adj_atomic_add(var_selected_count, var_20, var_21, adj_selected_count, adj_20, adj_21, adj_22);
    // adj: out_idx = wp.atomic_add(selected_count, 0, 1)                                     <L 281>
    if (var_19) {
        label4:;
        // adj: return                                                                        <L 280>
    }
    adj__cell_intersects_capsule_0(var_cell_nodes, var_particle_q, var_0, var_capsule_p0, var_capsule_p1, var_radius, adj_cell_nodes, adj_particle_q, adj_0, adj_capsule_p0, adj_capsule_p1, adj_radius, adj_17);
    // adj: if _cell_intersects_capsule(cell_nodes, particle_q, c, capsule_p0, capsule_p1, radius) == 0:  <L 279>
    if (var_16) {
        label3:;
        // adj: return                                                                        <L 277>
    }
    // adj: if radius < 0.0:                                                                  <L 276>
    if (var_6) {
        label2:;
        // adj: return                                                                        <L 275>
    }
    if (var_6) {
        wp::adj_address(var_material_cuttable, var_11, adj_material_cuttable, adj_9, adj_10);
        wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_9);
    }
    // adj: if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:         <L 274>
    if (var_4) {
        label1:;
        // adj: return                                                                        <L 273>
    }
    wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
    // adj: if cell_active[c] == 0:                                                           <L 272>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 271>
    }
    // adj: if c >= num_cells:                                                                <L 270>
    // adj: c = wp.tid()                                                                      <L 269>
    // adj: def select_capsule_cells_kernel(                                                  <L 254>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void select_capsule_cells_kernel_eec2ca80_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_select_capsule_cells_kernel_eec2ca80 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        select_capsule_cells_kernel_eec2ca80_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void select_capsule_cells_kernel_eec2ca80_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_select_capsule_cells_kernel_eec2ca80 *_wp_args,
    wp_args_select_capsule_cells_kernel_eec2ca80 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        select_capsule_cells_kernel_eec2ca80_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_diffuse_cell_heat_kernel_8ac93069 {
    wp::int32 num_cells;
    wp::array_t<wp::int32> cell_grid_xyz;
    wp::array_t<wp::int32> cell_material;
    wp::array_t<wp::int32> cell_active;
    wp::array_t<wp::int32> grid_to_cell;
    wp::array_t<wp::float32> material_conductivity;
    wp::array_t<wp::float32> heat_in;
    wp::float32 dt;
    wp::float32 diffusion;
    wp::float32 cooling;
    wp::int32 grid_nx;
    wp::int32 grid_ny;
    wp::int32 grid_nz;
    wp::array_t<wp::float32> heat_out;
};


void diffuse_cell_heat_kernel_8ac93069_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_diffuse_cell_heat_kernel_8ac93069 *_wp_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_grid_xyz = _wp_args->cell_grid_xyz;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::int32> var_grid_to_cell = _wp_args->grid_to_cell;
    wp::array_t<wp::float32> var_material_conductivity = _wp_args->material_conductivity;
    wp::array_t<wp::float32> var_heat_in = _wp_args->heat_in;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_diffusion = _wp_args->diffusion;
    wp::float32 var_cooling = _wp_args->cooling;
    wp::int32 var_grid_nx = _wp_args->grid_nx;
    wp::int32 var_grid_ny = _wp_args->grid_ny;
    wp::int32 var_grid_nz = _wp_args->grid_nz;
    wp::array_t<wp::float32> var_heat_out = _wp_args->heat_out;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32* var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::float32* var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32* var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::int32* var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    const wp::float32 var_28 = 0.0;
    wp::float32 var_29;
    const wp::int32 var_30 = 6;
    wp::range_t var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 0;
    bool var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    bool var_42;
    const wp::int32 var_43 = 1;
    wp::int32 var_44;
    wp::int32 var_45;
    const wp::int32 var_46 = 2;
    bool var_47;
    const wp::int32 var_48 = 1;
    wp::int32 var_49;
    wp::int32 var_50;
    const wp::int32 var_51 = 3;
    bool var_52;
    const wp::int32 var_53 = 1;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 4;
    bool var_57;
    const wp::int32 var_58 = 1;
    wp::int32 var_59;
    wp::int32 var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::int32 var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    bool var_72;
    const wp::int32 var_73 = 0;
    bool var_74;
    bool var_75;
    const wp::int32 var_76 = 0;
    bool var_77;
    bool var_78;
    const wp::int32 var_79 = 0;
    bool var_80;
    bool var_81;
    wp::int32* var_82;
    wp::int32 var_83;
    wp::int32 var_84;
    const wp::int32 var_85 = 0;
    bool var_86;
    wp::int32* var_87;
    const wp::int32 var_88 = 0;
    bool var_89;
    wp::int32 var_90;
    wp::int32* var_91;
    wp::float32* var_92;
    wp::int32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32* var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::float32 var_99 = 0.5;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::float32 var_114 = 0.0;
    bool var_115;
    const wp::float32 var_116 = 0.0;
    wp::float32 var_117;
    //---------
    // forward
    // def diffuse_cell_heat_kernel(                                                          <L 130>
    // c = wp.tid()                                                                           <L 147>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 148>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 149>
        return;
    }
    // if cell_active[c] == 0:                                                                <L 150>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // heat_out[c] = 0.0                                                                  <L 151>
        wp::array_store(var_heat_out, var_0, var_6);
        // return                                                                             <L 152>
        return;
    }
    // h = heat_in[c]                                                                         <L 154>
    var_7 = wp::address(var_heat_in, var_0);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // material_c = cell_material[c]                                                          <L 155>
    var_10 = wp::address(var_cell_material, var_0);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // cond_c = material_conductivity[material_c]                                             <L 156>
    var_13 = wp::address(var_material_conductivity, var_11);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // gx = cell_grid_xyz[c, 0]                                                               <L 157>
    var_17 = wp::address(var_cell_grid_xyz, var_0, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // gy = cell_grid_xyz[c, 1]                                                               <L 158>
    var_21 = wp::address(var_cell_grid_xyz, var_0, var_20);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // gz = cell_grid_xyz[c, 2]                                                               <L 159>
    var_25 = wp::address(var_cell_grid_xyz, var_0, var_24);
    var_27 = wp::load(var_25);
    var_26 = wp::copy(var_27);
    // accum = float(0.0)                                                                     <L 160>
    var_29 = wp::float(var_28);
    // for direction in range(6):                                                             <L 162>
    var_31 = wp::range(var_30);
    start_for_2:;
        if (iter_cmp(var_31) == 0) goto end_for_2;
        var_32 = wp::iter_next(var_31);
        // nx = gx                                                                            <L 163>
        var_33 = wp::copy(var_18);
        // ny = gy                                                                            <L 164>
        var_34 = wp::copy(var_22);
        // nz = gz                                                                            <L 165>
        var_35 = wp::copy(var_26);
        // if direction == 0:                                                                 <L 166>
        var_37 = (var_32 == var_36);
        if (var_37) {
            // nx = gx - 1                                                                    <L 167>
            var_39 = wp::sub(var_18, var_38);
        }
        var_40 = wp::where(var_37, var_39, var_33);
        if (!var_37) {
            // elif direction == 1:                                                           <L 168>
            var_42 = (var_32 == var_41);
            if (var_42) {
                // nx = gx + 1                                                                <L 169>
                var_44 = wp::add(var_18, var_43);
            }
            var_45 = wp::where(var_42, var_44, var_40);
            if (!var_42) {
                // elif direction == 2:                                                       <L 170>
                var_47 = (var_32 == var_46);
                if (var_47) {
                    // ny = gy - 1                                                            <L 171>
                    var_49 = wp::sub(var_22, var_48);
                }
                var_50 = wp::where(var_47, var_49, var_34);
                if (!var_47) {
                    // elif direction == 3:                                                   <L 172>
                    var_52 = (var_32 == var_51);
                    if (var_52) {
                        // ny = gy + 1                                                        <L 173>
                        var_54 = wp::add(var_22, var_53);
                    }
                    var_55 = wp::where(var_52, var_54, var_50);
                    if (!var_52) {
                        // elif direction == 4:                                               <L 174>
                        var_57 = (var_32 == var_56);
                        if (var_57) {
                            // nz = gz - 1                                                    <L 175>
                            var_59 = wp::sub(var_26, var_58);
                        }
                        var_60 = wp::where(var_57, var_59, var_35);
                        if (!var_57) {
                            // nz = gz + 1                                                    <L 177>
                            var_62 = wp::add(var_26, var_61);
                        }
                        var_63 = wp::where(var_57, var_60, var_62);
                    }
                    var_64 = wp::where(var_52, var_35, var_63);
                }
                var_65 = wp::where(var_47, var_50, var_55);
                var_66 = wp::where(var_47, var_35, var_64);
            }
            var_67 = wp::where(var_42, var_34, var_65);
            var_68 = wp::where(var_42, var_35, var_66);
        }
        var_69 = wp::where(var_37, var_40, var_45);
        var_70 = wp::where(var_37, var_34, var_67);
        var_71 = wp::where(var_37, var_35, var_68);
        // if nx < 0 or nx >= grid_nx or ny < 0 or ny >= grid_ny or nz < 0 or nz >= grid_nz:       <L 179>
        var_74 = (var_69 < var_73);
        var_72 = var_74;
        if (!var_72) {
            var_75 = (var_69 >= var_grid_nx);
            var_72 = var_72 || var_75;
        }
        if (!var_72) {
            var_77 = (var_70 < var_76);
            var_72 = var_72 || var_77;
        }
        if (!var_72) {
            var_78 = (var_70 >= var_grid_ny);
            var_72 = var_72 || var_78;
        }
        if (!var_72) {
            var_80 = (var_71 < var_79);
            var_72 = var_72 || var_80;
        }
        if (!var_72) {
            var_81 = (var_71 >= var_grid_nz);
            var_72 = var_72 || var_81;
        }
        if (var_72) {
            // continue                                                                       <L 180>
            goto start_for_2;
        }
        // nb = grid_to_cell[nx, ny, nz]                                                      <L 181>
        var_82 = wp::address(var_grid_to_cell, var_69, var_70, var_71);
        var_84 = wp::load(var_82);
        var_83 = wp::copy(var_84);
        // if nb < 0:                                                                         <L 182>
        var_86 = (var_83 < var_85);
        if (var_86) {
            // continue                                                                       <L 183>
            goto start_for_2;
        }
        // if cell_active[nb] == 0:                                                           <L 184>
        var_87 = wp::address(var_cell_active, var_83);
        var_90 = wp::load(var_87);
        var_89 = (var_90 == var_88);
        if (var_89) {
            // continue                                                                       <L 185>
            goto start_for_2;
        }
        // cond_nb = material_conductivity[cell_material[nb]]                                 <L 186>
        var_91 = wp::address(var_cell_material, var_83);
        var_93 = wp::load(var_91);
        var_92 = wp::address(var_material_conductivity, var_93);
        var_95 = wp::load(var_92);
        var_94 = wp::copy(var_95);
        // accum += (heat_in[nb] - h) * 0.5 * (cond_c + cond_nb)                              <L 187>
        var_96 = wp::address(var_heat_in, var_83);
        var_98 = wp::load(var_96);
        var_97 = wp::sub(var_98, var_8);
        var_100 = wp::mul(var_97, var_99);
        var_101 = wp::add(var_14, var_94);
        var_102 = wp::mul(var_100, var_101);
        var_103 = wp::add(var_29, var_102);
        wp::assign(var_29, var_103);
        goto start_for_2;
    end_for_2:;
    // out_h = h + float(dt) * float(diffusion) * accum                                       <L 189>
    var_104 = wp::float(var_dt);
    var_105 = wp::float(var_diffusion);
    var_106 = wp::mul(var_104, var_105);
    var_107 = wp::mul(var_106, var_29);
    var_108 = wp::add(var_8, var_107);
    // out_h = out_h - float(dt) * float(cooling) * h                                         <L 190>
    var_109 = wp::float(var_dt);
    var_110 = wp::float(var_cooling);
    var_111 = wp::mul(var_109, var_110);
    var_112 = wp::mul(var_111, var_8);
    var_113 = wp::sub(var_108, var_112);
    // if out_h < 0.0:                                                                        <L 191>
    var_115 = (var_113 < var_114);
    if (var_115) {
        // out_h = 0.0                                                                        <L 192>
    }
    var_117 = wp::where(var_115, var_116, var_113);
    // heat_out[c] = out_h                                                                    <L 193>
    wp::array_store(var_heat_out, var_0, var_117);
}



void diffuse_cell_heat_kernel_8ac93069_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_diffuse_cell_heat_kernel_8ac93069 *_wp_args,
    wp_args_diffuse_cell_heat_kernel_8ac93069 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_grid_xyz = _wp_args->cell_grid_xyz;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::int32> var_grid_to_cell = _wp_args->grid_to_cell;
    wp::array_t<wp::float32> var_material_conductivity = _wp_args->material_conductivity;
    wp::array_t<wp::float32> var_heat_in = _wp_args->heat_in;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_diffusion = _wp_args->diffusion;
    wp::float32 var_cooling = _wp_args->cooling;
    wp::int32 var_grid_nx = _wp_args->grid_nx;
    wp::int32 var_grid_ny = _wp_args->grid_ny;
    wp::int32 var_grid_nz = _wp_args->grid_nz;
    wp::array_t<wp::float32> var_heat_out = _wp_args->heat_out;
    wp::int32 adj_num_cells = _wp_adj_args->num_cells;
    wp::array_t<wp::int32> adj_cell_grid_xyz = _wp_adj_args->cell_grid_xyz;
    wp::array_t<wp::int32> adj_cell_material = _wp_adj_args->cell_material;
    wp::array_t<wp::int32> adj_cell_active = _wp_adj_args->cell_active;
    wp::array_t<wp::int32> adj_grid_to_cell = _wp_adj_args->grid_to_cell;
    wp::array_t<wp::float32> adj_material_conductivity = _wp_adj_args->material_conductivity;
    wp::array_t<wp::float32> adj_heat_in = _wp_adj_args->heat_in;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::float32 adj_diffusion = _wp_adj_args->diffusion;
    wp::float32 adj_cooling = _wp_adj_args->cooling;
    wp::int32 adj_grid_nx = _wp_adj_args->grid_nx;
    wp::int32 adj_grid_ny = _wp_adj_args->grid_ny;
    wp::int32 adj_grid_nz = _wp_adj_args->grid_nz;
    wp::array_t<wp::float32> adj_heat_out = _wp_adj_args->heat_out;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32* var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::int32* var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::float32* var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::int32 var_16 = 0;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 1;
    wp::int32* var_21;
    wp::int32 var_22;
    wp::int32 var_23;
    const wp::int32 var_24 = 2;
    wp::int32* var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    const wp::float32 var_28 = 0.0;
    wp::float32 var_29;
    const wp::int32 var_30 = 6;
    wp::range_t var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 0;
    bool var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 1;
    bool var_42;
    const wp::int32 var_43 = 1;
    wp::int32 var_44;
    wp::int32 var_45;
    const wp::int32 var_46 = 2;
    bool var_47;
    const wp::int32 var_48 = 1;
    wp::int32 var_49;
    wp::int32 var_50;
    const wp::int32 var_51 = 3;
    bool var_52;
    const wp::int32 var_53 = 1;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 4;
    bool var_57;
    const wp::int32 var_58 = 1;
    wp::int32 var_59;
    wp::int32 var_60;
    const wp::int32 var_61 = 1;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    wp::int32 var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    wp::int32 var_68;
    wp::int32 var_69;
    wp::int32 var_70;
    wp::int32 var_71;
    bool var_72;
    const wp::int32 var_73 = 0;
    bool var_74;
    bool var_75;
    const wp::int32 var_76 = 0;
    bool var_77;
    bool var_78;
    const wp::int32 var_79 = 0;
    bool var_80;
    bool var_81;
    wp::int32* var_82;
    wp::int32 var_83;
    wp::int32 var_84;
    const wp::int32 var_85 = 0;
    bool var_86;
    wp::int32* var_87;
    const wp::int32 var_88 = 0;
    bool var_89;
    wp::int32 var_90;
    wp::int32* var_91;
    wp::float32* var_92;
    wp::int32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32* var_96;
    wp::float32 var_97;
    wp::float32 var_98;
    const wp::float32 var_99 = 0.5;
    wp::float32 var_100;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    wp::float32 var_113;
    const wp::float32 var_114 = 0.0;
    bool var_115;
    const wp::float32 var_116 = 0.0;
    wp::float32 var_117;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    wp::int32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::int32 adj_30 = {};
    wp::range_t adj_31 = {};
    wp::int32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::int32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::int32 adj_36 = {};
    bool adj_37 = {};
    wp::int32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::int32 adj_40 = {};
    wp::int32 adj_41 = {};
    bool adj_42 = {};
    wp::int32 adj_43 = {};
    wp::int32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::int32 adj_46 = {};
    bool adj_47 = {};
    wp::int32 adj_48 = {};
    wp::int32 adj_49 = {};
    wp::int32 adj_50 = {};
    wp::int32 adj_51 = {};
    bool adj_52 = {};
    wp::int32 adj_53 = {};
    wp::int32 adj_54 = {};
    wp::int32 adj_55 = {};
    wp::int32 adj_56 = {};
    bool adj_57 = {};
    wp::int32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::int32 adj_60 = {};
    wp::int32 adj_61 = {};
    wp::int32 adj_62 = {};
    wp::int32 adj_63 = {};
    wp::int32 adj_64 = {};
    wp::int32 adj_65 = {};
    wp::int32 adj_66 = {};
    wp::int32 adj_67 = {};
    wp::int32 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::int32 adj_70 = {};
    wp::int32 adj_71 = {};
    bool adj_72 = {};
    wp::int32 adj_73 = {};
    bool adj_74 = {};
    bool adj_75 = {};
    wp::int32 adj_76 = {};
    bool adj_77 = {};
    bool adj_78 = {};
    wp::int32 adj_79 = {};
    bool adj_80 = {};
    bool adj_81 = {};
    wp::int32 adj_82 = {};
    wp::int32 adj_83 = {};
    wp::int32 adj_84 = {};
    wp::int32 adj_85 = {};
    bool adj_86 = {};
    wp::int32 adj_87 = {};
    wp::int32 adj_88 = {};
    bool adj_89 = {};
    wp::int32 adj_90 = {};
    wp::int32 adj_91 = {};
    wp::float32 adj_92 = {};
    wp::int32 adj_93 = {};
    wp::float32 adj_94 = {};
    wp::float32 adj_95 = {};
    wp::float32 adj_96 = {};
    wp::float32 adj_97 = {};
    wp::float32 adj_98 = {};
    wp::float32 adj_99 = {};
    wp::float32 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float32 adj_102 = {};
    wp::float32 adj_103 = {};
    wp::float32 adj_104 = {};
    wp::float32 adj_105 = {};
    wp::float32 adj_106 = {};
    wp::float32 adj_107 = {};
    wp::float32 adj_108 = {};
    wp::float32 adj_109 = {};
    wp::float32 adj_110 = {};
    wp::float32 adj_111 = {};
    wp::float32 adj_112 = {};
    wp::float32 adj_113 = {};
    wp::float32 adj_114 = {};
    bool adj_115 = {};
    wp::float32 adj_116 = {};
    wp::float32 adj_117 = {};
    //---------
    // forward
    // def diffuse_cell_heat_kernel(                                                          <L 130>
    // c = wp.tid()                                                                           <L 147>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 148>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 149>
        goto label0;
    }
    // if cell_active[c] == 0:                                                                <L 150>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // heat_out[c] = 0.0                                                                  <L 151>
        // wp::array_store(var_heat_out, var_0, var_6);
        // return                                                                             <L 152>
        goto label1;
    }
    // h = heat_in[c]                                                                         <L 154>
    var_7 = wp::address(var_heat_in, var_0);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // material_c = cell_material[c]                                                          <L 155>
    var_10 = wp::address(var_cell_material, var_0);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // cond_c = material_conductivity[material_c]                                             <L 156>
    var_13 = wp::address(var_material_conductivity, var_11);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // gx = cell_grid_xyz[c, 0]                                                               <L 157>
    var_17 = wp::address(var_cell_grid_xyz, var_0, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // gy = cell_grid_xyz[c, 1]                                                               <L 158>
    var_21 = wp::address(var_cell_grid_xyz, var_0, var_20);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // gz = cell_grid_xyz[c, 2]                                                               <L 159>
    var_25 = wp::address(var_cell_grid_xyz, var_0, var_24);
    var_27 = wp::load(var_25);
    var_26 = wp::copy(var_27);
    // accum = float(0.0)                                                                     <L 160>
    var_29 = wp::float(var_28);
    // for direction in range(6):                                                             <L 162>
    var_31 = wp::range(var_30);
    // out_h = h + float(dt) * float(diffusion) * accum                                       <L 189>
    var_104 = wp::float(var_dt);
    var_105 = wp::float(var_diffusion);
    var_106 = wp::mul(var_104, var_105);
    var_107 = wp::mul(var_106, var_29);
    var_108 = wp::add(var_8, var_107);
    // out_h = out_h - float(dt) * float(cooling) * h                                         <L 190>
    var_109 = wp::float(var_dt);
    var_110 = wp::float(var_cooling);
    var_111 = wp::mul(var_109, var_110);
    var_112 = wp::mul(var_111, var_8);
    var_113 = wp::sub(var_108, var_112);
    // if out_h < 0.0:                                                                        <L 191>
    var_115 = (var_113 < var_114);
    if (var_115) {
        // out_h = 0.0                                                                        <L 192>
    }
    var_117 = wp::where(var_115, var_116, var_113);
    // heat_out[c] = out_h                                                                    <L 193>
    // wp::array_store(var_heat_out, var_0, var_117);
    //---------
    // reverse
    wp::adj_array_store(var_heat_out, var_0, var_117, adj_heat_out, adj_0, adj_117);
    // adj: heat_out[c] = out_h                                                               <L 193>
    wp::adj_where(var_115, var_116, var_113, adj_115, adj_116, adj_113, adj_117);
    if (var_115) {
        // adj: out_h = 0.0                                                                   <L 192>
    }
    // adj: if out_h < 0.0:                                                                   <L 191>
    wp::adj_sub(var_108, var_112, adj_108, adj_112, adj_113);
    wp::adj_mul(var_111, var_8, adj_111, adj_8, adj_112);
    wp::adj_mul(var_109, var_110, adj_109, adj_110, adj_111);
    wp::adj_float(var_cooling, adj_cooling, adj_110);
    wp::adj_float(var_dt, adj_dt, adj_109);
    // adj: out_h = out_h - float(dt) * float(cooling) * h                                    <L 190>
    wp::adj_add(var_8, var_107, adj_8, adj_107, adj_108);
    wp::adj_mul(var_106, var_29, adj_106, adj_29, adj_107);
    wp::adj_mul(var_104, var_105, adj_104, adj_105, adj_106);
    wp::adj_float(var_diffusion, adj_diffusion, adj_105);
    wp::adj_float(var_dt, adj_dt, adj_104);
    // adj: out_h = h + float(dt) * float(diffusion) * accum                                  <L 189>
    var_31 = wp::iter_reverse(var_31);
    start_for_2:;
        if (iter_cmp(var_31) == 0) goto end_for_2;
        var_32 = wp::iter_next(var_31);
    	adj_33 = {};
    	adj_34 = {};
    	adj_35 = {};
    	adj_36 = {};
    	adj_37 = {};
    	adj_38 = {};
    	adj_39 = {};
    	adj_40 = {};
    	adj_41 = {};
    	adj_42 = {};
    	adj_43 = {};
    	adj_44 = {};
    	adj_45 = {};
    	adj_46 = {};
    	adj_47 = {};
    	adj_48 = {};
    	adj_49 = {};
    	adj_50 = {};
    	adj_51 = {};
    	adj_52 = {};
    	adj_53 = {};
    	adj_54 = {};
    	adj_55 = {};
    	adj_56 = {};
    	adj_57 = {};
    	adj_58 = {};
    	adj_59 = {};
    	adj_60 = {};
    	adj_61 = {};
    	adj_62 = {};
    	adj_63 = {};
    	adj_64 = {};
    	adj_65 = {};
    	adj_66 = {};
    	adj_67 = {};
    	adj_68 = {};
    	adj_69 = {};
    	adj_70 = {};
    	adj_71 = {};
    	adj_72 = {};
    	adj_73 = {};
    	adj_74 = {};
    	adj_75 = {};
    	adj_76 = {};
    	adj_77 = {};
    	adj_78 = {};
    	adj_79 = {};
    	adj_80 = {};
    	adj_81 = {};
    	adj_82 = {};
    	adj_83 = {};
    	adj_84 = {};
    	adj_85 = {};
    	adj_86 = {};
    	adj_87 = {};
    	adj_88 = {};
    	adj_89 = {};
    	adj_90 = {};
    	adj_91 = {};
    	adj_92 = {};
    	adj_93 = {};
    	adj_94 = {};
    	adj_95 = {};
    	adj_96 = {};
    	adj_97 = {};
    	adj_98 = {};
    	adj_99 = {};
    	adj_100 = {};
    	adj_101 = {};
    	adj_102 = {};
    	adj_103 = {};
        // nx = gx                                                                            <L 163>
        var_33 = wp::copy(var_18);
        // ny = gy                                                                            <L 164>
        var_34 = wp::copy(var_22);
        // nz = gz                                                                            <L 165>
        var_35 = wp::copy(var_26);
        // if direction == 0:                                                                 <L 166>
        var_37 = (var_32 == var_36);
        if (var_37) {
            // nx = gx - 1                                                                    <L 167>
            var_39 = wp::sub(var_18, var_38);
        }
        var_40 = wp::where(var_37, var_39, var_33);
        if (!var_37) {
            // elif direction == 1:                                                           <L 168>
            var_42 = (var_32 == var_41);
            if (var_42) {
                // nx = gx + 1                                                                <L 169>
                var_44 = wp::add(var_18, var_43);
            }
            var_45 = wp::where(var_42, var_44, var_40);
            if (!var_42) {
                // elif direction == 2:                                                       <L 170>
                var_47 = (var_32 == var_46);
                if (var_47) {
                    // ny = gy - 1                                                            <L 171>
                    var_49 = wp::sub(var_22, var_48);
                }
                var_50 = wp::where(var_47, var_49, var_34);
                if (!var_47) {
                    // elif direction == 3:                                                   <L 172>
                    var_52 = (var_32 == var_51);
                    if (var_52) {
                        // ny = gy + 1                                                        <L 173>
                        var_54 = wp::add(var_22, var_53);
                    }
                    var_55 = wp::where(var_52, var_54, var_50);
                    if (!var_52) {
                        // elif direction == 4:                                               <L 174>
                        var_57 = (var_32 == var_56);
                        if (var_57) {
                            // nz = gz - 1                                                    <L 175>
                            var_59 = wp::sub(var_26, var_58);
                        }
                        var_60 = wp::where(var_57, var_59, var_35);
                        if (!var_57) {
                            // nz = gz + 1                                                    <L 177>
                            var_62 = wp::add(var_26, var_61);
                        }
                        var_63 = wp::where(var_57, var_60, var_62);
                    }
                    var_64 = wp::where(var_52, var_35, var_63);
                }
                var_65 = wp::where(var_47, var_50, var_55);
                var_66 = wp::where(var_47, var_35, var_64);
            }
            var_67 = wp::where(var_42, var_34, var_65);
            var_68 = wp::where(var_42, var_35, var_66);
        }
        var_69 = wp::where(var_37, var_40, var_45);
        var_70 = wp::where(var_37, var_34, var_67);
        var_71 = wp::where(var_37, var_35, var_68);
        // if nx < 0 or nx >= grid_nx or ny < 0 or ny >= grid_ny or nz < 0 or nz >= grid_nz:       <L 179>
        var_74 = (var_69 < var_73);
        var_72 = var_74;
        if (!var_72) {
            var_75 = (var_69 >= var_grid_nx);
            var_72 = var_72 || var_75;
        }
        if (!var_72) {
            var_77 = (var_70 < var_76);
            var_72 = var_72 || var_77;
        }
        if (!var_72) {
            var_78 = (var_70 >= var_grid_ny);
            var_72 = var_72 || var_78;
        }
        if (!var_72) {
            var_80 = (var_71 < var_79);
            var_72 = var_72 || var_80;
        }
        if (!var_72) {
            var_81 = (var_71 >= var_grid_nz);
            var_72 = var_72 || var_81;
        }
        if (var_72) {
            // continue                                                                       <L 180>
            goto start_for_2;
        }
        // nb = grid_to_cell[nx, ny, nz]                                                      <L 181>
        var_82 = wp::address(var_grid_to_cell, var_69, var_70, var_71);
        var_84 = wp::load(var_82);
        var_83 = wp::copy(var_84);
        // if nb < 0:                                                                         <L 182>
        var_86 = (var_83 < var_85);
        if (var_86) {
            // continue                                                                       <L 183>
            goto start_for_2;
        }
        // if cell_active[nb] == 0:                                                           <L 184>
        var_87 = wp::address(var_cell_active, var_83);
        var_90 = wp::load(var_87);
        var_89 = (var_90 == var_88);
        if (var_89) {
            // continue                                                                       <L 185>
            goto start_for_2;
        }
        // cond_nb = material_conductivity[cell_material[nb]]                                 <L 186>
        var_91 = wp::address(var_cell_material, var_83);
        var_93 = wp::load(var_91);
        var_92 = wp::address(var_material_conductivity, var_93);
        var_95 = wp::load(var_92);
        var_94 = wp::copy(var_95);
        // accum += (heat_in[nb] - h) * 0.5 * (cond_c + cond_nb)                              <L 187>
        var_96 = wp::address(var_heat_in, var_83);
        var_98 = wp::load(var_96);
        var_97 = wp::sub(var_98, var_8);
        var_100 = wp::mul(var_97, var_99);
        var_101 = wp::add(var_14, var_94);
        var_102 = wp::mul(var_100, var_101);
        var_103 = wp::add(var_29, var_102);
        wp::assign(var_29, var_103);
        wp::adj_assign(var_29, var_103, adj_29, adj_103);
        wp::adj_add(var_29, var_102, adj_29, adj_102, adj_103);
        wp::adj_mul(var_100, var_101, adj_100, adj_101, adj_102);
        wp::adj_add(var_14, var_94, adj_14, adj_94, adj_101);
        wp::adj_mul(var_97, var_99, adj_97, adj_99, adj_100);
        wp::adj_sub(var_98, var_8, adj_96, adj_8, adj_97);
        wp::adj_address(var_heat_in, var_83, adj_heat_in, adj_83, adj_96);
        // adj: accum += (heat_in[nb] - h) * 0.5 * (cond_c + cond_nb)                         <L 187>
        wp::adj_copy(var_95, adj_92, adj_94);
        wp::adj_address(var_material_conductivity, var_93, adj_material_conductivity, adj_91, adj_92);
        wp::adj_address(var_cell_material, var_83, adj_cell_material, adj_83, adj_91);
        // adj: cond_nb = material_conductivity[cell_material[nb]]                            <L 186>
        if (var_89) {
            // adj: continue                                                                  <L 185>
        }
        wp::adj_address(var_cell_active, var_83, adj_cell_active, adj_83, adj_87);
        // adj: if cell_active[nb] == 0:                                                      <L 184>
        if (var_86) {
            // adj: continue                                                                  <L 183>
        }
        // adj: if nb < 0:                                                                    <L 182>
        wp::adj_copy(var_84, adj_82, adj_83);
        wp::adj_address(var_grid_to_cell, var_69, var_70, var_71, adj_grid_to_cell, adj_69, adj_70, adj_71, adj_82);
        // adj: nb = grid_to_cell[nx, ny, nz]                                                 <L 181>
        if (var_72) {
            // adj: continue                                                                  <L 180>
        }
        if (!var_72) {
        }
        if (!var_72) {
        }
        if (!var_72) {
        }
        if (!var_72) {
        }
        if (!var_72) {
        }
        // adj: if nx < 0 or nx >= grid_nx or ny < 0 or ny >= grid_ny or nz < 0 or nz >= grid_nz:  <L 179>
        wp::adj_where(var_37, var_35, var_68, adj_37, adj_35, adj_68, adj_71);
        wp::adj_where(var_37, var_34, var_67, adj_37, adj_34, adj_67, adj_70);
        wp::adj_where(var_37, var_40, var_45, adj_37, adj_40, adj_45, adj_69);
        if (!var_37) {
            wp::adj_where(var_42, var_35, var_66, adj_42, adj_35, adj_66, adj_68);
            wp::adj_where(var_42, var_34, var_65, adj_42, adj_34, adj_65, adj_67);
            if (!var_42) {
                wp::adj_where(var_47, var_35, var_64, adj_47, adj_35, adj_64, adj_66);
                wp::adj_where(var_47, var_50, var_55, adj_47, adj_50, adj_55, adj_65);
                if (!var_47) {
                    wp::adj_where(var_52, var_35, var_63, adj_52, adj_35, adj_63, adj_64);
                    if (!var_52) {
                        wp::adj_where(var_57, var_60, var_62, adj_57, adj_60, adj_62, adj_63);
                        if (!var_57) {
                            wp::adj_add(var_26, var_61, adj_26, adj_61, adj_62);
                            // adj: nz = gz + 1                                               <L 177>
                        }
                        wp::adj_where(var_57, var_59, var_35, adj_57, adj_59, adj_35, adj_60);
                        if (var_57) {
                            wp::adj_sub(var_26, var_58, adj_26, adj_58, adj_59);
                            // adj: nz = gz - 1                                               <L 175>
                        }
                        // adj: elif direction == 4:                                          <L 174>
                    }
                    wp::adj_where(var_52, var_54, var_50, adj_52, adj_54, adj_50, adj_55);
                    if (var_52) {
                        wp::adj_add(var_22, var_53, adj_22, adj_53, adj_54);
                        // adj: ny = gy + 1                                                   <L 173>
                    }
                    // adj: elif direction == 3:                                              <L 172>
                }
                wp::adj_where(var_47, var_49, var_34, adj_47, adj_49, adj_34, adj_50);
                if (var_47) {
                    wp::adj_sub(var_22, var_48, adj_22, adj_48, adj_49);
                    // adj: ny = gy - 1                                                       <L 171>
                }
                // adj: elif direction == 2:                                                  <L 170>
            }
            wp::adj_where(var_42, var_44, var_40, adj_42, adj_44, adj_40, adj_45);
            if (var_42) {
                wp::adj_add(var_18, var_43, adj_18, adj_43, adj_44);
                // adj: nx = gx + 1                                                           <L 169>
            }
            // adj: elif direction == 1:                                                      <L 168>
        }
        wp::adj_where(var_37, var_39, var_33, adj_37, adj_39, adj_33, adj_40);
        if (var_37) {
            wp::adj_sub(var_18, var_38, adj_18, adj_38, adj_39);
            // adj: nx = gx - 1                                                               <L 167>
        }
        // adj: if direction == 0:                                                            <L 166>
        wp::adj_copy(var_26, adj_26, adj_35);
        // adj: nz = gz                                                                       <L 165>
        wp::adj_copy(var_22, adj_22, adj_34);
        // adj: ny = gy                                                                       <L 164>
        wp::adj_copy(var_18, adj_18, adj_33);
        // adj: nx = gx                                                                       <L 163>
    	goto start_for_2;
    end_for_2:;
    wp::adj_range(var_30, adj_30, adj_31);
    // adj: for direction in range(6):                                                        <L 162>
    wp::adj_float(var_28, adj_28, adj_29);
    // adj: accum = float(0.0)                                                                <L 160>
    wp::adj_copy(var_27, adj_25, adj_26);
    wp::adj_address(var_cell_grid_xyz, var_0, var_24, adj_cell_grid_xyz, adj_0, adj_24, adj_25);
    // adj: gz = cell_grid_xyz[c, 2]                                                          <L 159>
    wp::adj_copy(var_23, adj_21, adj_22);
    wp::adj_address(var_cell_grid_xyz, var_0, var_20, adj_cell_grid_xyz, adj_0, adj_20, adj_21);
    // adj: gy = cell_grid_xyz[c, 1]                                                          <L 158>
    wp::adj_copy(var_19, adj_17, adj_18);
    wp::adj_address(var_cell_grid_xyz, var_0, var_16, adj_cell_grid_xyz, adj_0, adj_16, adj_17);
    // adj: gx = cell_grid_xyz[c, 0]                                                          <L 157>
    wp::adj_copy(var_15, adj_13, adj_14);
    wp::adj_address(var_material_conductivity, var_11, adj_material_conductivity, adj_11, adj_13);
    // adj: cond_c = material_conductivity[material_c]                                        <L 156>
    wp::adj_copy(var_12, adj_10, adj_11);
    wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_10);
    // adj: material_c = cell_material[c]                                                     <L 155>
    wp::adj_copy(var_9, adj_7, adj_8);
    wp::adj_address(var_heat_in, var_0, adj_heat_in, adj_0, adj_7);
    // adj: h = heat_in[c]                                                                    <L 154>
    if (var_4) {
        label1:;
        // adj: return                                                                        <L 152>
        wp::adj_array_store(var_heat_out, var_0, var_6, adj_heat_out, adj_0, adj_6);
        // adj: heat_out[c] = 0.0                                                             <L 151>
    }
    wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
    // adj: if cell_active[c] == 0:                                                           <L 150>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 149>
    }
    // adj: if c >= num_cells:                                                                <L 148>
    // adj: c = wp.tid()                                                                      <L 147>
    // adj: def diffuse_cell_heat_kernel(                                                     <L 130>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void diffuse_cell_heat_kernel_8ac93069_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_diffuse_cell_heat_kernel_8ac93069 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        diffuse_cell_heat_kernel_8ac93069_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void diffuse_cell_heat_kernel_8ac93069_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_diffuse_cell_heat_kernel_8ac93069 *_wp_args,
    wp_args_diffuse_cell_heat_kernel_8ac93069 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        diffuse_cell_heat_kernel_8ac93069_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_gather_heat_overlay_kernel_0c6bc357 {
    wp::int32 num_cells;
    wp::array_t<wp::vec_t<3, wp::float32>> cell_center_q;
    wp::array_t<wp::int32> cell_active;
    wp::array_t<wp::float32> cell_heat;
    wp::float32 max_heat;
    wp::float32 min_visible_heat;
    wp::array_t<wp::vec_t<3, wp::float32>> out_points;
    wp::array_t<wp::vec_t<3, wp::float32>> out_colors;
    wp::array_t<wp::int32> out_count;
};


void gather_heat_overlay_kernel_0c6bc357_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_gather_heat_overlay_kernel_0c6bc357 *_wp_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::vec_t<3, wp::float32>> var_cell_center_q = _wp_args->cell_center_q;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::float32> var_cell_heat = _wp_args->cell_heat;
    wp::float32 var_max_heat = _wp_args->max_heat;
    wp::float32 var_min_visible_heat = _wp_args->min_visible_heat;
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points = _wp_args->out_points;
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors = _wp_args->out_colors;
    wp::array_t<wp::int32> var_out_count = _wp_args->out_count;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 1e-06;
    bool var_12;
    const wp::float32 var_13 = 1e-06;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    wp::float32 var_19;
    const wp::float32 var_20 = 1.0;
    bool var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 0;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::float32 var_30 = 0.1;
    const wp::float32 var_31 = 0.9;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.12;
    const wp::float32 var_35 = 0.45;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 1.0;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    //---------
    // forward
    // def gather_heat_overlay_kernel(                                                        <L 286>
    // c = wp.tid()                                                                           <L 298>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 299>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 300>
        return;
    }
    // if cell_active[c] == 0:                                                                <L 301>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 302>
        return;
    }
    // h = cell_heat[c]                                                                       <L 303>
    var_6 = wp::address(var_cell_heat, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if h <= min_visible_heat:                                                              <L 304>
    var_9 = (var_7 <= var_min_visible_heat);
    if (var_9) {
        // return                                                                             <L 305>
        return;
    }
    // denom = max_heat                                                                       <L 307>
    var_10 = wp::copy(var_max_heat);
    // if denom < 1.0e-6:                                                                     <L 308>
    var_12 = (var_10 < var_11);
    if (var_12) {
        // denom = 1.0e-6                                                                     <L 309>
    }
    var_14 = wp::where(var_12, var_13, var_10);
    // t = h / denom                                                                          <L 310>
    var_15 = wp::div(var_7, var_14);
    // if t < 0.0:                                                                            <L 311>
    var_17 = (var_15 < var_16);
    if (var_17) {
        // t = 0.0                                                                            <L 312>
    }
    var_19 = wp::where(var_17, var_18, var_15);
    if (!var_17) {
        // elif t > 1.0:                                                                      <L 313>
        var_21 = (var_19 > var_20);
        if (var_21) {
            // t = 1.0                                                                        <L 314>
        }
        var_23 = wp::where(var_21, var_22, var_19);
    }
    var_24 = wp::where(var_17, var_19, var_23);
    // out_idx = wp.atomic_add(out_count, 0, 1)                                               <L 316>
    var_27 = wp::atomic_add(var_out_count, var_25, var_26);
    // out_points[out_idx] = cell_center_q[c]                                                 <L 317>
    var_28 = wp::address(var_cell_center_q, var_0);
    var_29 = wp::load(var_28);
    wp::array_store(var_out_points, var_27, var_29);
    // out_colors[out_idx] = wp.vec3(0.10 + 0.90 * t, 0.12 + 0.45 * (1.0 - t), 1.0 - t)       <L 318>
    var_32 = wp::mul(var_31, var_24);
    var_33 = wp::add(var_30, var_32);
    var_37 = wp::sub(var_36, var_24);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_34, var_38);
    var_41 = wp::sub(var_40, var_24);
    var_42 = wp::vec_t<3, wp::float32>(var_33, var_39, var_41);
    wp::array_store(var_out_colors, var_27, var_42);
}



void gather_heat_overlay_kernel_0c6bc357_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_gather_heat_overlay_kernel_0c6bc357 *_wp_args,
    wp_args_gather_heat_overlay_kernel_0c6bc357 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::vec_t<3, wp::float32>> var_cell_center_q = _wp_args->cell_center_q;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::float32> var_cell_heat = _wp_args->cell_heat;
    wp::float32 var_max_heat = _wp_args->max_heat;
    wp::float32 var_min_visible_heat = _wp_args->min_visible_heat;
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points = _wp_args->out_points;
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors = _wp_args->out_colors;
    wp::array_t<wp::int32> var_out_count = _wp_args->out_count;
    wp::int32 adj_num_cells = _wp_adj_args->num_cells;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_cell_center_q = _wp_adj_args->cell_center_q;
    wp::array_t<wp::int32> adj_cell_active = _wp_adj_args->cell_active;
    wp::array_t<wp::float32> adj_cell_heat = _wp_adj_args->cell_heat;
    wp::float32 adj_max_heat = _wp_adj_args->max_heat;
    wp::float32 adj_min_visible_heat = _wp_adj_args->min_visible_heat;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_points = _wp_adj_args->out_points;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_colors = _wp_adj_args->out_colors;
    wp::array_t<wp::int32> adj_out_count = _wp_adj_args->out_count;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    bool var_9;
    wp::float32 var_10;
    const wp::float32 var_11 = 1e-06;
    bool var_12;
    const wp::float32 var_13 = 1e-06;
    wp::float32 var_14;
    wp::float32 var_15;
    const wp::float32 var_16 = 0.0;
    bool var_17;
    const wp::float32 var_18 = 0.0;
    wp::float32 var_19;
    const wp::float32 var_20 = 1.0;
    bool var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 0;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::float32 var_30 = 0.1;
    const wp::float32 var_31 = 0.9;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.12;
    const wp::float32 var_35 = 0.45;
    const wp::float32 var_36 = 1.0;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::float32 var_40 = 1.0;
    wp::float32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    bool adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    bool adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    bool adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    bool adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    //---------
    // forward
    // def gather_heat_overlay_kernel(                                                        <L 286>
    // c = wp.tid()                                                                           <L 298>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 299>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 300>
        goto label0;
    }
    // if cell_active[c] == 0:                                                                <L 301>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 302>
        goto label1;
    }
    // h = cell_heat[c]                                                                       <L 303>
    var_6 = wp::address(var_cell_heat, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if h <= min_visible_heat:                                                              <L 304>
    var_9 = (var_7 <= var_min_visible_heat);
    if (var_9) {
        // return                                                                             <L 305>
        goto label2;
    }
    // denom = max_heat                                                                       <L 307>
    var_10 = wp::copy(var_max_heat);
    // if denom < 1.0e-6:                                                                     <L 308>
    var_12 = (var_10 < var_11);
    if (var_12) {
        // denom = 1.0e-6                                                                     <L 309>
    }
    var_14 = wp::where(var_12, var_13, var_10);
    // t = h / denom                                                                          <L 310>
    var_15 = wp::div(var_7, var_14);
    // if t < 0.0:                                                                            <L 311>
    var_17 = (var_15 < var_16);
    if (var_17) {
        // t = 0.0                                                                            <L 312>
    }
    var_19 = wp::where(var_17, var_18, var_15);
    if (!var_17) {
        // elif t > 1.0:                                                                      <L 313>
        var_21 = (var_19 > var_20);
        if (var_21) {
            // t = 1.0                                                                        <L 314>
        }
        var_23 = wp::where(var_21, var_22, var_19);
    }
    var_24 = wp::where(var_17, var_19, var_23);
    // out_idx = wp.atomic_add(out_count, 0, 1)                                               <L 316>
    // var_27 = wp::atomic_add(var_out_count, var_25, var_26);
    // out_points[out_idx] = cell_center_q[c]                                                 <L 317>
    var_28 = wp::address(var_cell_center_q, var_0);
    var_29 = wp::load(var_28);
    // wp::array_store(var_out_points, var_27, var_29);
    // out_colors[out_idx] = wp.vec3(0.10 + 0.90 * t, 0.12 + 0.45 * (1.0 - t), 1.0 - t)       <L 318>
    var_32 = wp::mul(var_31, var_24);
    var_33 = wp::add(var_30, var_32);
    var_37 = wp::sub(var_36, var_24);
    var_38 = wp::mul(var_35, var_37);
    var_39 = wp::add(var_34, var_38);
    var_41 = wp::sub(var_40, var_24);
    var_42 = wp::vec_t<3, wp::float32>(var_33, var_39, var_41);
    // wp::array_store(var_out_colors, var_27, var_42);
    //---------
    // reverse
    wp::adj_array_store(var_out_colors, var_27, var_42, adj_out_colors, adj_27, adj_42);
    wp::adj_vec_t(var_33, var_39, var_41, adj_33, adj_39, adj_41, adj_42);
    wp::adj_sub(var_40, var_24, adj_40, adj_24, adj_41);
    wp::adj_add(var_34, var_38, adj_34, adj_38, adj_39);
    wp::adj_mul(var_35, var_37, adj_35, adj_37, adj_38);
    wp::adj_sub(var_36, var_24, adj_36, adj_24, adj_37);
    wp::adj_add(var_30, var_32, adj_30, adj_32, adj_33);
    wp::adj_mul(var_31, var_24, adj_31, adj_24, adj_32);
    // adj: out_colors[out_idx] = wp.vec3(0.10 + 0.90 * t, 0.12 + 0.45 * (1.0 - t), 1.0 - t)  <L 318>
    wp::adj_array_store(var_out_points, var_27, var_29, adj_out_points, adj_27, adj_28);
    wp::adj_address(var_cell_center_q, var_0, adj_cell_center_q, adj_0, adj_28);
    // adj: out_points[out_idx] = cell_center_q[c]                                            <L 317>
    wp::adj_atomic_add(var_out_count, var_25, var_26, adj_out_count, adj_25, adj_26, adj_27);
    // adj: out_idx = wp.atomic_add(out_count, 0, 1)                                          <L 316>
    wp::adj_where(var_17, var_19, var_23, adj_17, adj_19, adj_23, adj_24);
    if (!var_17) {
        wp::adj_where(var_21, var_22, var_19, adj_21, adj_22, adj_19, adj_23);
        if (var_21) {
            // adj: t = 1.0                                                                   <L 314>
        }
        // adj: elif t > 1.0:                                                                 <L 313>
    }
    wp::adj_where(var_17, var_18, var_15, adj_17, adj_18, adj_15, adj_19);
    if (var_17) {
        // adj: t = 0.0                                                                       <L 312>
    }
    // adj: if t < 0.0:                                                                       <L 311>
    wp::adj_div(var_7, var_14, var_15, adj_7, adj_14, adj_15);
    // adj: t = h / denom                                                                     <L 310>
    wp::adj_where(var_12, var_13, var_10, adj_12, adj_13, adj_10, adj_14);
    if (var_12) {
        // adj: denom = 1.0e-6                                                                <L 309>
    }
    // adj: if denom < 1.0e-6:                                                                <L 308>
    wp::adj_copy(var_max_heat, adj_max_heat, adj_10);
    // adj: denom = max_heat                                                                  <L 307>
    if (var_9) {
        label2:;
        // adj: return                                                                        <L 305>
    }
    // adj: if h <= min_visible_heat:                                                         <L 304>
    wp::adj_copy(var_8, adj_6, adj_7);
    wp::adj_address(var_cell_heat, var_0, adj_cell_heat, adj_0, adj_6);
    // adj: h = cell_heat[c]                                                                  <L 303>
    if (var_4) {
        label1:;
        // adj: return                                                                        <L 302>
    }
    wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
    // adj: if cell_active[c] == 0:                                                           <L 301>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 300>
    }
    // adj: if c >= num_cells:                                                                <L 299>
    // adj: c = wp.tid()                                                                      <L 298>
    // adj: def gather_heat_overlay_kernel(                                                   <L 286>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void gather_heat_overlay_kernel_0c6bc357_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_gather_heat_overlay_kernel_0c6bc357 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        gather_heat_overlay_kernel_0c6bc357_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void gather_heat_overlay_kernel_0c6bc357_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_gather_heat_overlay_kernel_0c6bc357 *_wp_args,
    wp_args_gather_heat_overlay_kernel_0c6bc357 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        gather_heat_overlay_kernel_0c6bc357_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_reset_deleted_cell_heat_kernel_6e71ee1c {
    wp::int32 num_cells;
    wp::array_t<wp::int32> cell_active;
    wp::array_t<wp::float32> cell_heat_a;
    wp::array_t<wp::float32> cell_heat_b;
    wp::array_t<wp::float32> cell_burn;
};


void reset_deleted_cell_heat_kernel_6e71ee1c_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_reset_deleted_cell_heat_kernel_6e71ee1c *_wp_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::float32> var_cell_heat_a = _wp_args->cell_heat_a;
    wp::array_t<wp::float32> var_cell_heat_b = _wp_args->cell_heat_b;
    wp::array_t<wp::float32> var_cell_burn = _wp_args->cell_burn;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    const wp::float32 var_6 = 0.0;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    //---------
    // forward
    // def reset_deleted_cell_heat_kernel(                                                    <L 236>
    // c = wp.tid()                                                                           <L 243>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 244>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 245>
        return;
    }
    // if cell_active[c] != 0:                                                                <L 246>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 != var_3);
    if (var_4) {
        // return                                                                             <L 247>
        return;
    }
    // cell_heat_a[c] = 0.0                                                                   <L 248>
    wp::array_store(var_cell_heat_a, var_0, var_6);
    // cell_heat_b[c] = 0.0                                                                   <L 249>
    wp::array_store(var_cell_heat_b, var_0, var_7);
    // cell_burn[c] = 0.0                                                                     <L 250>
    wp::array_store(var_cell_burn, var_0, var_8);
}



void reset_deleted_cell_heat_kernel_6e71ee1c_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_reset_deleted_cell_heat_kernel_6e71ee1c *_wp_args,
    wp_args_reset_deleted_cell_heat_kernel_6e71ee1c *_wp_adj_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::float32> var_cell_heat_a = _wp_args->cell_heat_a;
    wp::array_t<wp::float32> var_cell_heat_b = _wp_args->cell_heat_b;
    wp::array_t<wp::float32> var_cell_burn = _wp_args->cell_burn;
    wp::int32 adj_num_cells = _wp_adj_args->num_cells;
    wp::array_t<wp::int32> adj_cell_active = _wp_adj_args->cell_active;
    wp::array_t<wp::float32> adj_cell_heat_a = _wp_adj_args->cell_heat_a;
    wp::array_t<wp::float32> adj_cell_heat_b = _wp_adj_args->cell_heat_b;
    wp::array_t<wp::float32> adj_cell_burn = _wp_adj_args->cell_burn;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    const wp::float32 var_6 = 0.0;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    //---------
    // forward
    // def reset_deleted_cell_heat_kernel(                                                    <L 236>
    // c = wp.tid()                                                                           <L 243>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 244>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 245>
        goto label0;
    }
    // if cell_active[c] != 0:                                                                <L 246>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 != var_3);
    if (var_4) {
        // return                                                                             <L 247>
        goto label1;
    }
    // cell_heat_a[c] = 0.0                                                                   <L 248>
    // wp::array_store(var_cell_heat_a, var_0, var_6);
    // cell_heat_b[c] = 0.0                                                                   <L 249>
    // wp::array_store(var_cell_heat_b, var_0, var_7);
    // cell_burn[c] = 0.0                                                                     <L 250>
    // wp::array_store(var_cell_burn, var_0, var_8);
    //---------
    // reverse
    wp::adj_array_store(var_cell_burn, var_0, var_8, adj_cell_burn, adj_0, adj_8);
    // adj: cell_burn[c] = 0.0                                                                <L 250>
    wp::adj_array_store(var_cell_heat_b, var_0, var_7, adj_cell_heat_b, adj_0, adj_7);
    // adj: cell_heat_b[c] = 0.0                                                              <L 249>
    wp::adj_array_store(var_cell_heat_a, var_0, var_6, adj_cell_heat_a, adj_0, adj_6);
    // adj: cell_heat_a[c] = 0.0                                                              <L 248>
    if (var_4) {
        label1:;
        // adj: return                                                                        <L 247>
    }
    wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
    // adj: if cell_active[c] != 0:                                                           <L 246>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 245>
    }
    // adj: if c >= num_cells:                                                                <L 244>
    // adj: c = wp.tid()                                                                      <L 243>
    // adj: def reset_deleted_cell_heat_kernel(                                               <L 236>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void reset_deleted_cell_heat_kernel_6e71ee1c_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_reset_deleted_cell_heat_kernel_6e71ee1c *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        reset_deleted_cell_heat_kernel_6e71ee1c_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void reset_deleted_cell_heat_kernel_6e71ee1c_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_reset_deleted_cell_heat_kernel_6e71ee1c *_wp_args,
    wp_args_reset_deleted_cell_heat_kernel_6e71ee1c *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        reset_deleted_cell_heat_kernel_6e71ee1c_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_select_overheated_cells_kernel_6c7f3511 {
    wp::int32 num_cells;
    wp::array_t<wp::int32> cell_material;
    wp::array_t<wp::int32> cell_active;
    wp::array_t<wp::float32> material_resistance;
    wp::array_t<wp::int32> material_cuttable;
    wp::int32 has_material_filter;
    wp::array_t<wp::float32> cell_heat;
    wp::float32 fulguration;
    wp::array_t<wp::float32> cell_burn;
    wp::array_t<wp::int32> selected_cells;
    wp::array_t<wp::int32> selected_count;
};


void select_overheated_cells_kernel_6c7f3511_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_select_overheated_cells_kernel_6c7f3511 *_wp_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::float32> var_material_resistance = _wp_args->material_resistance;
    wp::array_t<wp::int32> var_material_cuttable = _wp_args->material_cuttable;
    wp::int32 var_has_material_filter = _wp_args->has_material_filter;
    wp::array_t<wp::float32> var_cell_heat = _wp_args->cell_heat;
    wp::float32 var_fulguration = _wp_args->fulguration;
    wp::array_t<wp::float32> var_cell_burn = _wp_args->cell_burn;
    wp::array_t<wp::int32> var_selected_cells = _wp_args->selected_cells;
    wp::array_t<wp::int32> var_selected_count = _wp_args->selected_count;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 15.0;
    bool var_10;
    wp::float32* var_11;
    const wp::float32 var_12 = 0.04;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.05;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 1.0;
    bool var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    bool var_33;
    bool var_34;
    const wp::int32 var_35 = 0;
    bool var_36;
    wp::int32* var_37;
    const wp::int32 var_38 = 0;
    bool var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 0;
    const wp::int32 var_42 = 1;
    wp::int32 var_43;
    //---------
    // forward
    // def select_overheated_cells_kernel(                                                    <L 197>
    // c = wp.tid()                                                                           <L 211>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 212>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 213>
        return;
    }
    // if cell_active[c] == 0:                                                                <L 214>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 215>
        return;
    }
    // h = cell_heat[c]                                                                       <L 217>
    var_6 = wp::address(var_cell_heat, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if h > _BURN_HEAT_THRESHOLD:                                                           <L 218>
    var_10 = (var_7 > var_9);
    if (var_10) {
        // b = cell_burn[c] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration       <L 219>
        var_11 = wp::address(var_cell_burn, var_0);
        var_13 = wp::sub(var_7, var_9);
        var_14 = wp::mul(var_12, var_13);
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::mul(var_16, var_fulguration);
        var_19 = wp::load(var_11);
        var_18 = wp::add(var_19, var_17);
        // if b > 1.0:                                                                        <L 220>
        var_21 = (var_18 > var_20);
        if (var_21) {
            // b = 1.0                                                                        <L 221>
        }
        var_23 = wp::where(var_21, var_22, var_18);
        // cell_burn[c] = b                                                                   <L 222>
        wp::array_store(var_cell_burn, var_0, var_23);
    }
    // material = cell_material[c]                                                            <L 224>
    var_24 = wp::address(var_cell_material, var_0);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // resistance = material_resistance[material]                                             <L 225>
    var_27 = wp::address(var_material_resistance, var_25);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // if resistance <= 0.0 or h <= resistance:                                               <L 226>
    var_32 = (var_28 <= var_31);
    var_30 = var_32;
    if (!var_30) {
        var_33 = (var_7 <= var_28);
        var_30 = var_30 || var_33;
    }
    if (var_30) {
        // return                                                                             <L 227>
        return;
    }
    // if has_material_filter != 0 and material_cuttable[material] == 0:                      <L 228>
    var_36 = (var_has_material_filter != var_35);
    var_34 = var_36;
    if (var_34) {
        var_37 = wp::address(var_material_cuttable, var_25);
        var_40 = wp::load(var_37);
        var_39 = (var_40 == var_38);
        var_34 = var_34 && var_39;
    }
    if (var_34) {
        // return                                                                             <L 229>
        return;
    }
    // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 231>
    var_43 = wp::atomic_add(var_selected_count, var_41, var_42);
    // selected_cells[out_idx] = c                                                            <L 232>
    wp::array_store(var_selected_cells, var_43, var_0);
}



void select_overheated_cells_kernel_6c7f3511_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_select_overheated_cells_kernel_6c7f3511 *_wp_args,
    wp_args_select_overheated_cells_kernel_6c7f3511 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::int32 var_num_cells = _wp_args->num_cells;
    wp::array_t<wp::int32> var_cell_material = _wp_args->cell_material;
    wp::array_t<wp::int32> var_cell_active = _wp_args->cell_active;
    wp::array_t<wp::float32> var_material_resistance = _wp_args->material_resistance;
    wp::array_t<wp::int32> var_material_cuttable = _wp_args->material_cuttable;
    wp::int32 var_has_material_filter = _wp_args->has_material_filter;
    wp::array_t<wp::float32> var_cell_heat = _wp_args->cell_heat;
    wp::float32 var_fulguration = _wp_args->fulguration;
    wp::array_t<wp::float32> var_cell_burn = _wp_args->cell_burn;
    wp::array_t<wp::int32> var_selected_cells = _wp_args->selected_cells;
    wp::array_t<wp::int32> var_selected_count = _wp_args->selected_count;
    wp::int32 adj_num_cells = _wp_adj_args->num_cells;
    wp::array_t<wp::int32> adj_cell_material = _wp_adj_args->cell_material;
    wp::array_t<wp::int32> adj_cell_active = _wp_adj_args->cell_active;
    wp::array_t<wp::float32> adj_material_resistance = _wp_adj_args->material_resistance;
    wp::array_t<wp::int32> adj_material_cuttable = _wp_adj_args->material_cuttable;
    wp::int32 adj_has_material_filter = _wp_adj_args->has_material_filter;
    wp::array_t<wp::float32> adj_cell_heat = _wp_adj_args->cell_heat;
    wp::float32 adj_fulguration = _wp_adj_args->fulguration;
    wp::array_t<wp::float32> adj_cell_burn = _wp_adj_args->cell_burn;
    wp::array_t<wp::int32> adj_selected_cells = _wp_adj_args->selected_cells;
    wp::array_t<wp::int32> adj_selected_count = _wp_adj_args->selected_count;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::int32* var_2;
    const wp::int32 var_3 = 0;
    bool var_4;
    wp::int32 var_5;
    wp::float32* var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 15.0;
    bool var_10;
    wp::float32* var_11;
    const wp::float32 var_12 = 0.04;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 0.05;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    const wp::float32 var_20 = 1.0;
    bool var_21;
    const wp::float32 var_22 = 1.0;
    wp::float32 var_23;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::float32* var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    bool var_33;
    bool var_34;
    const wp::int32 var_35 = 0;
    bool var_36;
    wp::int32* var_37;
    const wp::int32 var_38 = 0;
    bool var_39;
    wp::int32 var_40;
    const wp::int32 var_41 = 0;
    const wp::int32 var_42 = 1;
    wp::int32 var_43;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    bool adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    bool adj_10 = {};
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
    bool adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    bool adj_30 = {};
    wp::float32 adj_31 = {};
    bool adj_32 = {};
    bool adj_33 = {};
    bool adj_34 = {};
    wp::int32 adj_35 = {};
    bool adj_36 = {};
    wp::int32 adj_37 = {};
    wp::int32 adj_38 = {};
    bool adj_39 = {};
    wp::int32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::int32 adj_43 = {};
    //---------
    // forward
    // def select_overheated_cells_kernel(                                                    <L 197>
    // c = wp.tid()                                                                           <L 211>
    var_0 = builtin_tid1d();
    // if c >= num_cells:                                                                     <L 212>
    var_1 = (var_0 >= var_num_cells);
    if (var_1) {
        // return                                                                             <L 213>
        goto label0;
    }
    // if cell_active[c] == 0:                                                                <L 214>
    var_2 = wp::address(var_cell_active, var_0);
    var_5 = wp::load(var_2);
    var_4 = (var_5 == var_3);
    if (var_4) {
        // return                                                                             <L 215>
        goto label1;
    }
    // h = cell_heat[c]                                                                       <L 217>
    var_6 = wp::address(var_cell_heat, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // if h > _BURN_HEAT_THRESHOLD:                                                           <L 218>
    var_10 = (var_7 > var_9);
    if (var_10) {
        // b = cell_burn[c] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration       <L 219>
        var_11 = wp::address(var_cell_burn, var_0);
        var_13 = wp::sub(var_7, var_9);
        var_14 = wp::mul(var_12, var_13);
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::mul(var_16, var_fulguration);
        var_19 = wp::load(var_11);
        var_18 = wp::add(var_19, var_17);
        // if b > 1.0:                                                                        <L 220>
        var_21 = (var_18 > var_20);
        if (var_21) {
            // b = 1.0                                                                        <L 221>
        }
        var_23 = wp::where(var_21, var_22, var_18);
        // cell_burn[c] = b                                                                   <L 222>
        // wp::array_store(var_cell_burn, var_0, var_23);
    }
    // material = cell_material[c]                                                            <L 224>
    var_24 = wp::address(var_cell_material, var_0);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // resistance = material_resistance[material]                                             <L 225>
    var_27 = wp::address(var_material_resistance, var_25);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // if resistance <= 0.0 or h <= resistance:                                               <L 226>
    var_32 = (var_28 <= var_31);
    var_30 = var_32;
    if (!var_30) {
        var_33 = (var_7 <= var_28);
        var_30 = var_30 || var_33;
    }
    if (var_30) {
        // return                                                                             <L 227>
        goto label2;
    }
    // if has_material_filter != 0 and material_cuttable[material] == 0:                      <L 228>
    var_36 = (var_has_material_filter != var_35);
    var_34 = var_36;
    if (var_34) {
        var_37 = wp::address(var_material_cuttable, var_25);
        var_40 = wp::load(var_37);
        var_39 = (var_40 == var_38);
        var_34 = var_34 && var_39;
    }
    if (var_34) {
        // return                                                                             <L 229>
        goto label3;
    }
    // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 231>
    // var_43 = wp::atomic_add(var_selected_count, var_41, var_42);
    // selected_cells[out_idx] = c                                                            <L 232>
    // wp::array_store(var_selected_cells, var_43, var_0);
    //---------
    // reverse
    wp::adj_array_store(var_selected_cells, var_43, var_0, adj_selected_cells, adj_43, adj_0);
    // adj: selected_cells[out_idx] = c                                                       <L 232>
    wp::adj_atomic_add(var_selected_count, var_41, var_42, adj_selected_count, adj_41, adj_42, adj_43);
    // adj: out_idx = wp.atomic_add(selected_count, 0, 1)                                     <L 231>
    if (var_34) {
        label3:;
        // adj: return                                                                        <L 229>
    }
    if (var_34) {
        wp::adj_address(var_material_cuttable, var_25, adj_material_cuttable, adj_25, adj_37);
    }
    // adj: if has_material_filter != 0 and material_cuttable[material] == 0:                 <L 228>
    if (var_30) {
        label2:;
        // adj: return                                                                        <L 227>
    }
    if (!var_30) {
    }
    // adj: if resistance <= 0.0 or h <= resistance:                                          <L 226>
    wp::adj_copy(var_29, adj_27, adj_28);
    wp::adj_address(var_material_resistance, var_25, adj_material_resistance, adj_25, adj_27);
    // adj: resistance = material_resistance[material]                                        <L 225>
    wp::adj_copy(var_26, adj_24, adj_25);
    wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_24);
    // adj: material = cell_material[c]                                                       <L 224>
    if (var_10) {
        wp::adj_array_store(var_cell_burn, var_0, var_23, adj_cell_burn, adj_0, adj_23);
        // adj: cell_burn[c] = b                                                              <L 222>
        wp::adj_where(var_21, var_22, var_18, adj_21, adj_22, adj_18, adj_23);
        if (var_21) {
            // adj: b = 1.0                                                                   <L 221>
        }
        // adj: if b > 1.0:                                                                   <L 220>
        wp::adj_add(var_19, var_17, adj_11, adj_17, adj_18);
        wp::adj_mul(var_16, var_fulguration, adj_16, adj_fulguration, adj_17);
        wp::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
        wp::adj_mul(var_12, var_13, adj_12, adj_13, adj_14);
        wp::adj_sub(var_7, var_9, adj_7, adj_9, adj_13);
        wp::adj_address(var_cell_burn, var_0, adj_cell_burn, adj_0, adj_11);
        // adj: b = cell_burn[c] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration  <L 219>
    }
    // adj: if h > _BURN_HEAT_THRESHOLD:                                                      <L 218>
    wp::adj_copy(var_8, adj_6, adj_7);
    wp::adj_address(var_cell_heat, var_0, adj_cell_heat, adj_0, adj_6);
    // adj: h = cell_heat[c]                                                                  <L 217>
    if (var_4) {
        label1:;
        // adj: return                                                                        <L 215>
    }
    wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
    // adj: if cell_active[c] == 0:                                                           <L 214>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 213>
    }
    // adj: if c >= num_cells:                                                                <L 212>
    // adj: c = wp.tid()                                                                      <L 211>
    // adj: def select_overheated_cells_kernel(                                               <L 197>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void select_overheated_cells_kernel_6c7f3511_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_select_overheated_cells_kernel_6c7f3511 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        select_overheated_cells_kernel_6c7f3511_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void select_overheated_cells_kernel_6c7f3511_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_select_overheated_cells_kernel_6c7f3511 *_wp_args,
    wp_args_select_overheated_cells_kernel_6c7f3511 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        select_overheated_cells_kernel_6c7f3511_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

