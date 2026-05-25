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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:13
static CUDA_CALLABLE wp::float32 _active_particle_inv_mass_0(
    wp::int32 var_particle_index,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::int32* var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    wp::int32 var_6;
    const wp::int32 var_7 = 0;
    bool var_8;
    const wp::float32 var_9 = 0.0;
    wp::float32* var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    const wp::float32 var_13 = 0.0;
    bool var_14;
    const wp::float32 var_15 = 0.0;
    //---------
    // forward
    // def _active_particle_inv_mass(                                                         <L 14>
    // if particle_index < 0:                                                                 <L 19>
    var_1 = (var_particle_index < var_0);
    if (var_1) {
        // return 0.0                                                                         <L 20>
        return var_2;
    }
    // if (particle_flags[particle_index] & _ACTIVE_BIT) == 0:                                <L 21>
    var_3 = wp::address(var_particle_flags, var_particle_index);
    var_6 = wp::load(var_3);
    var_5 = wp::bit_and(var_6, var_4);
    var_8 = (var_5 == var_7);
    if (var_8) {
        // return 0.0                                                                         <L 22>
        return var_9;
    }
    // w = particle_inv_mass[particle_index]                                                  <L 23>
    var_10 = wp::address(var_particle_inv_mass, var_particle_index);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // if w <= 0.0:                                                                           <L 24>
    var_14 = (var_11 <= var_13);
    if (var_14) {
        // return 0.0                                                                         <L 25>
        return var_15;
    }
    // return w                                                                               <L 26>
    return var_11;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:29
static CUDA_CALLABLE wp::float32 _cell_inv_mass_sum_0(
    wp::int32 var_cell_index,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::int32* var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    wp::int32 var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.0;
    wp::float32 var_9;
    const wp::int32 var_10 = 0;
    wp::int32* var_11;
    wp::float32 var_12;
    wp::int32 var_13;
    wp::float32 var_14;
    const wp::int32 var_15 = 1;
    wp::int32* var_16;
    wp::float32 var_17;
    wp::int32 var_18;
    wp::float32 var_19;
    const wp::int32 var_20 = 2;
    wp::int32* var_21;
    wp::float32 var_22;
    wp::int32 var_23;
    wp::float32 var_24;
    const wp::int32 var_25 = 3;
    wp::int32* var_26;
    wp::float32 var_27;
    wp::int32 var_28;
    wp::float32 var_29;
    const wp::int32 var_30 = 4;
    wp::int32* var_31;
    wp::float32 var_32;
    wp::int32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 5;
    wp::int32* var_36;
    wp::float32 var_37;
    wp::int32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 6;
    wp::int32* var_41;
    wp::float32 var_42;
    wp::int32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 7;
    wp::int32* var_46;
    wp::float32 var_47;
    wp::int32 var_48;
    wp::float32 var_49;
    //---------
    // forward
    // def _cell_inv_mass_sum(                                                                <L 30>
    // if cell_index < 0:                                                                     <L 37>
    var_1 = (var_cell_index < var_0);
    if (var_1) {
        // return 0.0                                                                         <L 38>
        return var_2;
    }
    // if cell_active[cell_index] == 0:                                                       <L 39>
    var_3 = wp::address(var_cell_active, var_cell_index);
    var_6 = wp::load(var_3);
    var_5 = (var_6 == var_4);
    if (var_5) {
        // return 0.0                                                                         <L 40>
        return var_7;
    }
    // w = float(0.0)                                                                         <L 42>
    var_9 = wp::float(var_8);
    // for i in range(8):                                                                     <L 43>
    // w = w + _active_particle_inv_mass(cell_nodes[cell_index, i], particle_inv_mass, particle_flags)       <L 44>
    var_11 = wp::address(var_cell_nodes, var_cell_index, var_10);
    var_13 = wp::load(var_11);
    var_12 = _active_particle_inv_mass_0(var_13, var_particle_inv_mass, var_particle_flags);
    var_14 = wp::add(var_9, var_12);
    var_16 = wp::address(var_cell_nodes, var_cell_index, var_15);
    var_18 = wp::load(var_16);
    var_17 = _active_particle_inv_mass_0(var_18, var_particle_inv_mass, var_particle_flags);
    var_19 = wp::add(var_14, var_17);
    var_21 = wp::address(var_cell_nodes, var_cell_index, var_20);
    var_23 = wp::load(var_21);
    var_22 = _active_particle_inv_mass_0(var_23, var_particle_inv_mass, var_particle_flags);
    var_24 = wp::add(var_19, var_22);
    var_26 = wp::address(var_cell_nodes, var_cell_index, var_25);
    var_28 = wp::load(var_26);
    var_27 = _active_particle_inv_mass_0(var_28, var_particle_inv_mass, var_particle_flags);
    var_29 = wp::add(var_24, var_27);
    var_31 = wp::address(var_cell_nodes, var_cell_index, var_30);
    var_33 = wp::load(var_31);
    var_32 = _active_particle_inv_mass_0(var_33, var_particle_inv_mass, var_particle_flags);
    var_34 = wp::add(var_29, var_32);
    var_36 = wp::address(var_cell_nodes, var_cell_index, var_35);
    var_38 = wp::load(var_36);
    var_37 = _active_particle_inv_mass_0(var_38, var_particle_inv_mass, var_particle_flags);
    var_39 = wp::add(var_34, var_37);
    var_41 = wp::address(var_cell_nodes, var_cell_index, var_40);
    var_43 = wp::load(var_41);
    var_42 = _active_particle_inv_mass_0(var_43, var_particle_inv_mass, var_particle_flags);
    var_44 = wp::add(var_39, var_42);
    var_46 = wp::address(var_cell_nodes, var_cell_index, var_45);
    var_48 = wp::load(var_46);
    var_47 = _active_particle_inv_mass_0(var_48, var_particle_inv_mass, var_particle_flags);
    var_49 = wp::add(var_44, var_47);
    // return w                                                                               <L 45>
    return var_49;
}


// /home/pkorzeniowsk/Projects/newton/newton-1.0/newton/_src/geometry/kernels.py:63
static CUDA_CALLABLE void triangle_closest_point_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::int32 & ret_2)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    bool var_5;
    const wp::float32 var_6 = 0.0;
    bool var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::int32 var_10 = 0;
    wp::int32 var_11;
    const wp::float32 var_12 = 1.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.0;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    bool var_21;
    bool var_22;
    const wp::int32 var_23 = 1;
    wp::int32 var_24;
    const wp::float32 var_25 = 0.0;
    const wp::float32 var_26 = 1.0;
    const wp::float32 var_27 = 0.0;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32 var_29;
    wp::vec_t<3, wp::float32> var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    bool var_34;
    const wp::float32 var_35 = 0.0;
    bool var_36;
    bool var_37;
    const wp::int32 var_38 = 2;
    wp::int32 var_39;
    const wp::float32 var_40 = 0.0;
    const wp::float32 var_41 = 0.0;
    const wp::float32 var_42 = 1.0;
    wp::vec_t<3, wp::float32> var_43;
    wp::int32 var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    bool var_49;
    const wp::float32 var_50 = 0.0;
    bool var_51;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    const wp::float32 var_54 = 0.0;
    bool var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    const wp::int32 var_58 = 3;
    wp::int32 var_59;
    const wp::float32 var_60 = 1.0;
    wp::float32 var_61;
    const wp::float32 var_62 = 0.0;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::int32 var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    bool var_71;
    const wp::float32 var_72 = 0.0;
    bool var_73;
    const wp::float32 var_74 = 0.0;
    bool var_75;
    const wp::float32 var_76 = 0.0;
    bool var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    const wp::int32 var_80 = 4;
    wp::int32 var_81;
    const wp::float32 var_82 = 1.0;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.0;
    wp::vec_t<3, wp::float32> var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::int32 var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    bool var_94;
    const wp::float32 var_95 = 0.0;
    bool var_96;
    wp::float32 var_97;
    const wp::float32 var_98 = 0.0;
    bool var_99;
    wp::float32 var_100;
    const wp::float32 var_101 = 0.0;
    bool var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 5;
    wp::int32 var_109;
    const wp::float32 var_110 = 0.0;
    const wp::float32 var_111 = 1.0;
    wp::float32 var_112;
    wp::vec_t<3, wp::float32> var_113;
    wp::vec_t<3, wp::float32> var_114;
    wp::vec_t<3, wp::float32> var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::int32 var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::float32 var_119;
    const wp::float32 var_120 = 1.0;
    wp::float32 var_121;
    wp::float32 var_122;
    wp::float32 var_123;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 6;
    wp::int32 var_127;
    const wp::float32 var_128 = 1.0;
    wp::float32 var_129;
    wp::float32 var_130;
    wp::vec_t<3, wp::float32> var_131;
    wp::vec_t<3, wp::float32> var_132;
    wp::vec_t<3, wp::float32> var_133;
    wp::vec_t<3, wp::float32> var_134;
    wp::vec_t<3, wp::float32> var_135;
    //---------
    // forward
    // def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):            <L 64>
    // ab = b - a                                                                             <L 75>
    var_0 = wp::sub(var_b, var_a);
    // ac = c - a                                                                             <L 76>
    var_1 = wp::sub(var_c, var_a);
    // ap = p - a                                                                             <L 77>
    var_2 = wp::sub(var_p, var_a);
    // d1 = wp.dot(ab, ap)                                                                    <L 79>
    var_3 = wp::dot(var_0, var_2);
    // d2 = wp.dot(ac, ap)                                                                    <L 80>
    var_4 = wp::dot(var_1, var_2);
    // if d1 <= 0.0 and d2 <= 0.0:                                                            <L 81>
    var_7 = (var_3 <= var_6);
    var_5 = var_7;
    if (var_5) {
        var_9 = (var_4 <= var_8);
        var_5 = var_5 && var_9;
    }
    if (var_5) {
        // feature_type = TRI_CONTACT_FEATURE_VERTEX_A                                        <L 82>
        var_11 = wp::copy(var_10);
        // bary = wp.vec3(1.0, 0.0, 0.0)                                                      <L 83>
        var_15 = wp::vec_t<3, wp::float32>(var_12, var_13, var_14);
        // return a, bary, feature_type                                                       <L 84>
        ret_0 = var_a;
        ret_1 = var_15;
        ret_2 = var_11;
        return;
    }
    // bp = p - b                                                                             <L 86>
    var_16 = wp::sub(var_p, var_b);
    // d3 = wp.dot(ab, bp)                                                                    <L 87>
    var_17 = wp::dot(var_0, var_16);
    // d4 = wp.dot(ac, bp)                                                                    <L 88>
    var_18 = wp::dot(var_1, var_16);
    // if d3 >= 0.0 and d4 <= d3:                                                             <L 89>
    var_21 = (var_17 >= var_20);
    var_19 = var_21;
    if (var_19) {
        var_22 = (var_18 <= var_17);
        var_19 = var_19 && var_22;
    }
    if (var_19) {
        // feature_type = TRI_CONTACT_FEATURE_VERTEX_B                                        <L 90>
        var_24 = wp::copy(var_23);
        // bary = wp.vec3(0.0, 1.0, 0.0)                                                      <L 91>
        var_28 = wp::vec_t<3, wp::float32>(var_25, var_26, var_27);
        // return b, bary, feature_type                                                       <L 92>
        ret_0 = var_b;
        ret_1 = var_28;
        ret_2 = var_24;
        return;
    }
    var_29 = wp::where(var_19, var_24, var_11);
    var_30 = wp::where(var_19, var_28, var_15);
    // cp = p - c                                                                             <L 94>
    var_31 = wp::sub(var_p, var_c);
    // d5 = wp.dot(ab, cp)                                                                    <L 95>
    var_32 = wp::dot(var_0, var_31);
    // d6 = wp.dot(ac, cp)                                                                    <L 96>
    var_33 = wp::dot(var_1, var_31);
    // if d6 >= 0.0 and d5 <= d6:                                                             <L 97>
    var_36 = (var_33 >= var_35);
    var_34 = var_36;
    if (var_34) {
        var_37 = (var_32 <= var_33);
        var_34 = var_34 && var_37;
    }
    if (var_34) {
        // feature_type = TRI_CONTACT_FEATURE_VERTEX_C                                        <L 98>
        var_39 = wp::copy(var_38);
        // bary = wp.vec3(0.0, 0.0, 1.0)                                                      <L 99>
        var_43 = wp::vec_t<3, wp::float32>(var_40, var_41, var_42);
        // return c, bary, feature_type                                                       <L 100>
        ret_0 = var_c;
        ret_1 = var_43;
        ret_2 = var_39;
        return;
    }
    var_44 = wp::where(var_34, var_39, var_29);
    var_45 = wp::where(var_34, var_43, var_30);
    // vc = d1 * d4 - d3 * d2                                                                 <L 102>
    var_46 = wp::mul(var_3, var_18);
    var_47 = wp::mul(var_17, var_4);
    var_48 = wp::sub(var_46, var_47);
    // if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                              <L 103>
    var_51 = (var_48 <= var_50);
    var_49 = var_51;
    if (var_49) {
        var_53 = (var_3 >= var_52);
        var_49 = var_49 && var_53;
    }
    if (var_49) {
        var_55 = (var_17 <= var_54);
        var_49 = var_49 && var_55;
    }
    if (var_49) {
        // v = d1 / (d1 - d3)                                                                 <L 104>
        var_56 = wp::sub(var_3, var_17);
        var_57 = wp::div(var_3, var_56);
        // feature_type = TRI_CONTACT_FEATURE_EDGE_AB                                         <L 105>
        var_59 = wp::copy(var_58);
        // bary = wp.vec3(1.0 - v, v, 0.0)                                                    <L 106>
        var_61 = wp::sub(var_60, var_57);
        var_63 = wp::vec_t<3, wp::float32>(var_61, var_57, var_62);
        // return a + v * ab, bary, feature_type                                              <L 107>
        var_64 = wp::mul(var_57, var_0);
        var_65 = wp::add(var_a, var_64);
        ret_0 = var_65;
        ret_1 = var_63;
        ret_2 = var_59;
        return;
    }
    var_66 = wp::where(var_49, var_59, var_44);
    var_67 = wp::where(var_49, var_63, var_45);
    // vb = d5 * d2 - d1 * d6                                                                 <L 109>
    var_68 = wp::mul(var_32, var_4);
    var_69 = wp::mul(var_3, var_33);
    var_70 = wp::sub(var_68, var_69);
    // if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                              <L 110>
    var_73 = (var_70 <= var_72);
    var_71 = var_73;
    if (var_71) {
        var_75 = (var_4 >= var_74);
        var_71 = var_71 && var_75;
    }
    if (var_71) {
        var_77 = (var_33 <= var_76);
        var_71 = var_71 && var_77;
    }
    if (var_71) {
        // v = d2 / (d2 - d6)                                                                 <L 111>
        var_78 = wp::sub(var_4, var_33);
        var_79 = wp::div(var_4, var_78);
        // feature_type = TRI_CONTACT_FEATURE_EDGE_AC                                         <L 112>
        var_81 = wp::copy(var_80);
        // bary = wp.vec3(1.0 - v, 0.0, v)                                                    <L 113>
        var_83 = wp::sub(var_82, var_79);
        var_85 = wp::vec_t<3, wp::float32>(var_83, var_84, var_79);
        // return a + v * ac, bary, feature_type                                              <L 114>
        var_86 = wp::mul(var_79, var_1);
        var_87 = wp::add(var_a, var_86);
        ret_0 = var_87;
        ret_1 = var_85;
        ret_2 = var_81;
        return;
    }
    var_88 = wp::where(var_71, var_81, var_66);
    var_89 = wp::where(var_71, var_85, var_67);
    var_90 = wp::where(var_71, var_79, var_57);
    // va = d3 * d6 - d5 * d4                                                                 <L 116>
    var_91 = wp::mul(var_17, var_33);
    var_92 = wp::mul(var_32, var_18);
    var_93 = wp::sub(var_91, var_92);
    // if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                                <L 117>
    var_96 = (var_93 <= var_95);
    var_94 = var_96;
    if (var_94) {
        var_97 = wp::sub(var_18, var_17);
        var_99 = (var_97 >= var_98);
        var_94 = var_94 && var_99;
    }
    if (var_94) {
        var_100 = wp::sub(var_32, var_33);
        var_102 = (var_100 >= var_101);
        var_94 = var_94 && var_102;
    }
    if (var_94) {
        // v = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                            <L 118>
        var_103 = wp::sub(var_18, var_17);
        var_104 = wp::sub(var_18, var_17);
        var_105 = wp::sub(var_32, var_33);
        var_106 = wp::add(var_104, var_105);
        var_107 = wp::div(var_103, var_106);
        // feature_type = TRI_CONTACT_FEATURE_EDGE_BC                                         <L 119>
        var_109 = wp::copy(var_108);
        // bary = wp.vec3(0.0, 1.0 - v, v)                                                    <L 120>
        var_112 = wp::sub(var_111, var_107);
        var_113 = wp::vec_t<3, wp::float32>(var_110, var_112, var_107);
        // return b + v * (c - b), bary, feature_type                                         <L 121>
        var_114 = wp::sub(var_c, var_b);
        var_115 = wp::mul(var_107, var_114);
        var_116 = wp::add(var_b, var_115);
        ret_0 = var_116;
        ret_1 = var_113;
        ret_2 = var_109;
        return;
    }
    var_117 = wp::where(var_94, var_109, var_88);
    var_118 = wp::where(var_94, var_113, var_89);
    var_119 = wp::where(var_94, var_107, var_90);
    // denom = 1.0 / (va + vb + vc)                                                           <L 123>
    var_121 = wp::add(var_93, var_70);
    var_122 = wp::add(var_121, var_48);
    var_123 = wp::div(var_120, var_122);
    // v = vb * denom                                                                         <L 124>
    var_124 = wp::mul(var_70, var_123);
    // w = vc * denom                                                                         <L 125>
    var_125 = wp::mul(var_48, var_123);
    // feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR                                       <L 126>
    var_127 = wp::copy(var_126);
    // bary = wp.vec3(1.0 - v - w, v, w)                                                      <L 127>
    var_129 = wp::sub(var_128, var_124);
    var_130 = wp::sub(var_129, var_125);
    var_131 = wp::vec_t<3, wp::float32>(var_130, var_124, var_125);
    // return a + v * ab + w * ac, bary, feature_type                                         <L 128>
    var_132 = wp::mul(var_124, var_0);
    var_133 = wp::add(var_a, var_132);
    var_134 = wp::mul(var_125, var_1);
    var_135 = wp::add(var_133, var_134);
    ret_0 = var_135;
    ret_1 = var_131;
    ret_2 = var_127;
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:48
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _sphere_triangle_fallback_dir_0(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_contact_point)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-12;
    bool var_5;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    const wp::float32 var_9 = 3.0;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::float32 var_12;
    const wp::float32 var_13 = 1e-12;
    bool var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    bool var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    bool var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 1e-12;
    bool var_32;
    wp::vec_t<3, wp::float32> var_33;
    const wp::float32 var_34 = 1.0;
    const wp::float32 var_35 = 0.0;
    const wp::float32 var_36 = 0.0;
    wp::vec_t<3, wp::float32> var_37;
    //---------
    // forward
    // def _sphere_triangle_fallback_dir(                                                     <L 49>
    // normal = wp.cross(v1 - v0, v2 - v0)                                                    <L 55>
    var_0 = wp::sub(var_v1, var_v0);
    var_1 = wp::sub(var_v2, var_v0);
    var_2 = wp::cross(var_0, var_1);
    // if wp.dot(normal, normal) > 1.0e-12:                                                   <L 56>
    var_3 = wp::dot(var_2, var_2);
    var_5 = (var_3 > var_4);
    if (var_5) {
        // return wp.normalize(normal)                                                        <L 57>
        var_6 = wp::normalize(var_2);
        return var_6;
    }
    // centroid = (v0 + v1 + v2) / 3.0                                                        <L 59>
    var_7 = wp::add(var_v0, var_v1);
    var_8 = wp::add(var_7, var_v2);
    var_10 = wp::div(var_8, var_9);
    // centroid_dir = centroid - contact_point                                                <L 60>
    var_11 = wp::sub(var_10, var_contact_point);
    // if wp.dot(centroid_dir, centroid_dir) > 1.0e-12:                                       <L 61>
    var_12 = wp::dot(var_11, var_11);
    var_14 = (var_12 > var_13);
    if (var_14) {
        // return wp.normalize(centroid_dir)                                                  <L 62>
        var_15 = wp::normalize(var_11);
        return var_15;
    }
    // d0 = v0 - contact_point                                                                <L 64>
    var_16 = wp::sub(var_v0, var_contact_point);
    // d1 = v1 - contact_point                                                                <L 65>
    var_17 = wp::sub(var_v1, var_contact_point);
    // d2 = v2 - contact_point                                                                <L 66>
    var_18 = wp::sub(var_v2, var_contact_point);
    // best = d0                                                                              <L 68>
    var_19 = wp::copy(var_16);
    // if wp.dot(d1, d1) > wp.dot(best, best):                                                <L 69>
    var_20 = wp::dot(var_17, var_17);
    var_21 = wp::dot(var_19, var_19);
    var_22 = (var_20 > var_21);
    if (var_22) {
        // best = d1                                                                          <L 70>
        var_23 = wp::copy(var_17);
    }
    var_24 = wp::where(var_22, var_23, var_19);
    // if wp.dot(d2, d2) > wp.dot(best, best):                                                <L 71>
    var_25 = wp::dot(var_18, var_18);
    var_26 = wp::dot(var_24, var_24);
    var_27 = (var_25 > var_26);
    if (var_27) {
        // best = d2                                                                          <L 72>
        var_28 = wp::copy(var_18);
    }
    var_29 = wp::where(var_27, var_28, var_24);
    // if wp.dot(best, best) > 1.0e-12:                                                       <L 73>
    var_30 = wp::dot(var_29, var_29);
    var_32 = (var_30 > var_31);
    if (var_32) {
        // return wp.normalize(best)                                                          <L 74>
        var_33 = wp::normalize(var_29);
        return var_33;
    }
    // return wp.vec3(1.0, 0.0, 0.0)                                                          <L 76>
    var_37 = wp::vec_t<3, wp::float32>(var_34, var_35, var_36);
    return var_37;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:79
static CUDA_CALLABLE void _scatter_cell_correction_to_nodes_0(
    wp::int32 var_cell_index,
    wp::vec_t<3, wp::float32> var_correction,
    wp::float32 var_cell_weight,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_delta_accumulator,
    wp::array_t<wp::int32> var_particle_delta_counter)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    bool var_3;
    const wp::int32 var_4 = 0;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::float32 var_8;
    const wp::float32 var_9 = 0.0;
    bool var_10;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    const wp::int32 var_14 = 1;
    wp::int32 var_15;
    const wp::int32 var_16 = 1;
    wp::int32* var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    wp::float32 var_20;
    const wp::float32 var_21 = 0.0;
    bool var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    const wp::int32 var_26 = 1;
    wp::int32 var_27;
    const wp::int32 var_28 = 2;
    wp::int32* var_29;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::float32 var_32;
    const wp::float32 var_33 = 0.0;
    bool var_34;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    const wp::int32 var_40 = 3;
    wp::int32* var_41;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 0.0;
    bool var_46;
    wp::float32 var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    const wp::int32 var_50 = 1;
    wp::int32 var_51;
    const wp::int32 var_52 = 4;
    wp::int32* var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    wp::float32 var_56;
    const wp::float32 var_57 = 0.0;
    bool var_58;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    const wp::int32 var_62 = 1;
    wp::int32 var_63;
    const wp::int32 var_64 = 5;
    wp::int32* var_65;
    wp::int32 var_66;
    wp::int32 var_67;
    wp::float32 var_68;
    const wp::float32 var_69 = 0.0;
    bool var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::int32 var_74 = 1;
    wp::int32 var_75;
    const wp::int32 var_76 = 6;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    wp::float32 var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    const wp::int32 var_88 = 7;
    wp::int32* var_89;
    wp::int32 var_90;
    wp::int32 var_91;
    wp::float32 var_92;
    const wp::float32 var_93 = 0.0;
    bool var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    const wp::int32 var_98 = 1;
    wp::int32 var_99;
    //---------
    // forward
    // def _scatter_cell_correction_to_nodes(                                                 <L 80>
    // if cell_index < 0:                                                                     <L 90>
    var_1 = (var_cell_index < var_0);
    if (var_1) {
        // return                                                                             <L 91>
        return;
    }
    // if cell_weight <= 0.0:                                                                 <L 92>
    var_3 = (var_cell_weight <= var_2);
    if (var_3) {
        // return                                                                             <L 93>
        return;
    }
    // for i in range(8):                                                                     <L 95>
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_5 = wp::address(var_cell_nodes, var_cell_index, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_8 = _active_particle_inv_mass_0(var_6, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_10 = (var_8 > var_9);
    if (var_10) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_11 = wp::div(var_8, var_cell_weight);
        var_12 = wp::mul(var_correction, var_11);
        var_13 = wp::atomic_add(var_particle_delta_accumulator, var_6, var_12);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_15 = wp::atomic_add(var_particle_delta_counter, var_6, var_14);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_17 = wp::address(var_cell_nodes, var_cell_index, var_16);
    var_19 = wp::load(var_17);
    var_18 = wp::copy(var_19);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_20 = _active_particle_inv_mass_0(var_18, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_22 = (var_20 > var_21);
    if (var_22) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_23 = wp::div(var_20, var_cell_weight);
        var_24 = wp::mul(var_correction, var_23);
        var_25 = wp::atomic_add(var_particle_delta_accumulator, var_18, var_24);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_27 = wp::atomic_add(var_particle_delta_counter, var_18, var_26);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_29 = wp::address(var_cell_nodes, var_cell_index, var_28);
    var_31 = wp::load(var_29);
    var_30 = wp::copy(var_31);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_32 = _active_particle_inv_mass_0(var_30, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_34 = (var_32 > var_33);
    if (var_34) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_35 = wp::div(var_32, var_cell_weight);
        var_36 = wp::mul(var_correction, var_35);
        var_37 = wp::atomic_add(var_particle_delta_accumulator, var_30, var_36);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_39 = wp::atomic_add(var_particle_delta_counter, var_30, var_38);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_41 = wp::address(var_cell_nodes, var_cell_index, var_40);
    var_43 = wp::load(var_41);
    var_42 = wp::copy(var_43);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_44 = _active_particle_inv_mass_0(var_42, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_46 = (var_44 > var_45);
    if (var_46) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_47 = wp::div(var_44, var_cell_weight);
        var_48 = wp::mul(var_correction, var_47);
        var_49 = wp::atomic_add(var_particle_delta_accumulator, var_42, var_48);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_51 = wp::atomic_add(var_particle_delta_counter, var_42, var_50);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_53 = wp::address(var_cell_nodes, var_cell_index, var_52);
    var_55 = wp::load(var_53);
    var_54 = wp::copy(var_55);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_56 = _active_particle_inv_mass_0(var_54, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_58 = (var_56 > var_57);
    if (var_58) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_59 = wp::div(var_56, var_cell_weight);
        var_60 = wp::mul(var_correction, var_59);
        var_61 = wp::atomic_add(var_particle_delta_accumulator, var_54, var_60);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_63 = wp::atomic_add(var_particle_delta_counter, var_54, var_62);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_65 = wp::address(var_cell_nodes, var_cell_index, var_64);
    var_67 = wp::load(var_65);
    var_66 = wp::copy(var_67);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_68 = _active_particle_inv_mass_0(var_66, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_70 = (var_68 > var_69);
    if (var_70) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_71 = wp::div(var_68, var_cell_weight);
        var_72 = wp::mul(var_correction, var_71);
        var_73 = wp::atomic_add(var_particle_delta_accumulator, var_66, var_72);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_75 = wp::atomic_add(var_particle_delta_counter, var_66, var_74);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_77 = wp::address(var_cell_nodes, var_cell_index, var_76);
    var_79 = wp::load(var_77);
    var_78 = wp::copy(var_79);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_80 = _active_particle_inv_mass_0(var_78, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_82 = (var_80 > var_81);
    if (var_82) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_83 = wp::div(var_80, var_cell_weight);
        var_84 = wp::mul(var_correction, var_83);
        var_85 = wp::atomic_add(var_particle_delta_accumulator, var_78, var_84);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_87 = wp::atomic_add(var_particle_delta_counter, var_78, var_86);
    }
    // p = cell_nodes[cell_index, i]                                                          <L 96>
    var_89 = wp::address(var_cell_nodes, var_cell_index, var_88);
    var_91 = wp::load(var_89);
    var_90 = wp::copy(var_91);
    // w = _active_particle_inv_mass(p, particle_inv_mass, particle_flags)                    <L 97>
    var_92 = _active_particle_inv_mass_0(var_90, var_particle_inv_mass, var_particle_flags);
    // if w > 0.0:                                                                            <L 98>
    var_94 = (var_92 > var_93);
    if (var_94) {
        // wp.atomic_add(particle_delta_accumulator, p, correction * (w / cell_weight))       <L 99>
        var_95 = wp::div(var_92, var_cell_weight);
        var_96 = wp::mul(var_correction, var_95);
        var_97 = wp::atomic_add(var_particle_delta_accumulator, var_90, var_96);
        // wp.atomic_add(particle_delta_counter, p, 1)                                        <L 100>
        var_99 = wp::atomic_add(var_particle_delta_counter, var_90, var_98);
    }
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:13
static CUDA_CALLABLE void adj__active_particle_inv_mass_0(
    wp::int32 var_particle_index,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::int32 & adj_particle_index,
    wp::array_t<wp::float32> & adj_particle_inv_mass,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:29
static CUDA_CALLABLE void adj__cell_inv_mass_sum_0(
    wp::int32 var_cell_index,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::int32 & adj_cell_index,
    wp::array_t<wp::int32> & adj_cell_nodes,
    wp::array_t<wp::int32> & adj_cell_active,
    wp::array_t<wp::float32> & adj_particle_inv_mass,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/newton/newton-1.0/newton/_src/geometry/kernels.py:63
static CUDA_CALLABLE void adj_triangle_closest_point_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> & ret_0,
    wp::vec_t<3, wp::float32> & ret_1,
    wp::int32 & ret_2,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_c,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_ret_0,
    wp::vec_t<3, wp::float32> & adj_ret_1,
    wp::int32 & adj_ret_2)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:48
static CUDA_CALLABLE void adj__sphere_triangle_fallback_dir_0(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_contact_point,
    wp::vec_t<3, wp::float32> & adj_v0,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_contact_point,
    wp::vec_t<3, wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/instrument.py:79
static CUDA_CALLABLE void adj__scatter_cell_correction_to_nodes_0(
    wp::int32 var_cell_index,
    wp::vec_t<3, wp::float32> var_correction,
    wp::float32 var_cell_weight,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_delta_accumulator,
    wp::array_t<wp::int32> var_particle_delta_counter,
    wp::int32 & adj_cell_index,
    wp::vec_t<3, wp::float32> & adj_correction,
    wp::float32 & adj_cell_weight,
    wp::array_t<wp::int32> & adj_cell_nodes,
    wp::array_t<wp::float32> & adj_particle_inv_mass,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> & adj_particle_delta_accumulator,
    wp::array_t<wp::int32> & adj_particle_delta_counter)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void apply_kinematic_sphere_mc_triangle_node_deltas_kernel_2c1ddbb7_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_delta_accumulator,
    wp::array_t<wp::int32> var_particle_delta_counter)
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
        wp::float32 var_1;
        const wp::float32 var_2 = 0.0;
        bool var_3;
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        //---------
        // forward
        // def apply_kinematic_sphere_mc_triangle_node_deltas_kernel(                             <L 279>
        // p = wp.tid()                                                                           <L 286>
        var_0 = builtin_tid1d();
        // if _active_particle_inv_mass(p, particle_inv_mass, particle_flags) <= 0.0:             <L 287>
        var_1 = _active_particle_inv_mass_0(var_0, var_particle_inv_mass, var_particle_flags);
        var_3 = (var_1 <= var_2);
        if (var_3) {
            // return                                                                             <L 288>
            continue;
        }
        // count = particle_delta_counter[p]                                                      <L 289>
        var_4 = wp::address(var_particle_delta_counter, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if count <= 0:                                                                         <L 290>
        var_8 = (var_5 <= var_7);
        if (var_8) {
            // return                                                                             <L 291>
            continue;
        }
        // particle_q[p] = particle_q[p] + particle_delta_accumulator[p] / float(count)           <L 292>
        var_9 = wp::address(var_particle_q, var_0);
        var_10 = wp::address(var_particle_delta_accumulator, var_0);
        var_11 = wp::float(var_5);
        var_13 = wp::load(var_10);
        var_12 = wp::div(var_13, var_11);
        var_15 = wp::load(var_9);
        var_14 = wp::add(var_15, var_12);
        wp::array_store(var_particle_q, var_0, var_14);
    }
}



extern "C" __global__ void project_kinematic_sphere_particle_positions_kernel_6717a775_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q_prev,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q,
    wp::int32 var_sphere_count,
    wp::float32 var_sphere_radius,
    wp::float32 var_interpolation_alpha,
    wp::float32 var_relaxation,
    wp::float32 var_max_correction)
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
        wp::float32 var_11;
        const wp::float32 var_12 = 0.0;
        bool var_13;
        const wp::float32 var_14 = 0.0;
        wp::float32 var_15;
        const wp::float32 var_16 = 1.0;
        bool var_17;
        const wp::float32 var_18 = 1.0;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::float32 var_21 = 0.0;
        bool var_22;
        const wp::float32 var_23 = 0.0;
        wp::float32 var_24;
        const wp::float32 var_25 = 1.0;
        bool var_26;
        const wp::float32 var_27 = 1.0;
        wp::float32 var_28;
        wp::float32 var_29;
        const wp::float32 var_30 = 0.0;
        bool var_31;
        const wp::float32 var_32 = 0.0;
        wp::float32 var_33;
        wp::vec_t<3, wp::float32>* var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::float32 var_37 = 0.0;
        const wp::float32 var_38 = 0.0;
        const wp::float32 var_39 = 0.0;
        wp::vec_t<3, wp::float32> var_40;
        wp::float32* var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::range_t var_44;
        wp::int32 var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32>* var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.0;
        bool var_59;
        const wp::float32 var_60 = 1.0;
        const wp::float32 var_61 = 0.0;
        const wp::float32 var_62 = 0.0;
        wp::vec_t<3, wp::float32> var_63;
        const wp::float32 var_64 = 1e-08;
        bool var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::float32 var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        bool var_71;
        const wp::float32 var_72 = 0.0;
        bool var_73;
        bool var_74;
        wp::float32 var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        //---------
        // forward
        // def project_kinematic_sphere_particle_positions_kernel(                                <L 104>
        // tid = wp.tid()                                                                         <L 117>
        var_0 = builtin_tid1d();
        // if (particle_flags[tid] & _ACTIVE_BIT) == 0:                                           <L 119>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 120>
            continue;
        }
        // if particle_inv_mass[tid] <= 0.0:                                                      <L 121>
        var_7 = wp::address(var_particle_inv_mass, var_0);
        var_10 = wp::load(var_7);
        var_9 = (var_10 <= var_8);
        if (var_9) {
            // return                                                                             <L 122>
            continue;
        }
        // alpha = interpolation_alpha                                                            <L 124>
        var_11 = wp::copy(var_interpolation_alpha);
        // if alpha < 0.0:                                                                        <L 125>
        var_13 = (var_11 < var_12);
        if (var_13) {
            // alpha = 0.0                                                                        <L 126>
        }
        var_15 = wp::where(var_13, var_14, var_11);
        // if alpha > 1.0:                                                                        <L 127>
        var_17 = (var_15 > var_16);
        if (var_17) {
            // alpha = 1.0                                                                        <L 128>
        }
        var_19 = wp::where(var_17, var_18, var_15);
        // contact_relaxation = relaxation                                                        <L 130>
        var_20 = wp::copy(var_relaxation);
        // if contact_relaxation < 0.0:                                                           <L 131>
        var_22 = (var_20 < var_21);
        if (var_22) {
            // contact_relaxation = 0.0                                                           <L 132>
        }
        var_24 = wp::where(var_22, var_23, var_20);
        // if contact_relaxation > 1.0:                                                           <L 133>
        var_26 = (var_24 > var_25);
        if (var_26) {
            // contact_relaxation = 1.0                                                           <L 134>
        }
        var_28 = wp::where(var_26, var_27, var_24);
        // correction_cap = max_correction                                                        <L 136>
        var_29 = wp::copy(var_max_correction);
        // if correction_cap < 0.0:                                                               <L 137>
        var_31 = (var_29 < var_30);
        if (var_31) {
            // correction_cap = 0.0                                                               <L 138>
        }
        var_33 = wp::where(var_31, var_32, var_29);
        // q = particle_q[tid]                                                                    <L 140>
        var_34 = wp::address(var_particle_q, var_0);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // correction_total = wp.vec3(0.0, 0.0, 0.0)                                              <L 141>
        var_40 = wp::vec_t<3, wp::float32>(var_37, var_38, var_39);
        // combined_radius = particle_radius[tid] + sphere_radius                                 <L 142>
        var_41 = wp::address(var_particle_radius, var_0);
        var_43 = wp::load(var_41);
        var_42 = wp::add(var_43, var_sphere_radius);
        // for sphere_idx in range(sphere_count):                                                 <L 144>
        var_44 = wp::range(var_sphere_count);
        start_for_2:;
            if (iter_cmp(var_44) == 0) goto end_for_2;
            var_45 = wp::iter_next(var_44);
            // center = sphere_q_prev[sphere_idx] + (sphere_q[sphere_idx] - sphere_q_prev[sphere_idx]) * alpha       <L 145>
            var_46 = wp::address(var_sphere_q_prev, var_45);
            var_47 = wp::address(var_sphere_q, var_45);
            var_48 = wp::address(var_sphere_q_prev, var_45);
            var_50 = wp::load(var_47);
            var_51 = wp::load(var_48);
            var_49 = wp::sub(var_50, var_51);
            var_52 = wp::mul(var_49, var_19);
            var_54 = wp::load(var_46);
            var_53 = wp::add(var_54, var_52);
            // delta = q - center                                                                 <L 146>
            var_55 = wp::sub(var_35, var_53);
            // dist = wp.length(delta)                                                            <L 147>
            var_56 = wp::length(var_55);
            // penetration = combined_radius - dist                                               <L 148>
            var_57 = wp::sub(var_42, var_56);
            // if penetration > 0.0:                                                              <L 149>
            var_59 = (var_57 > var_58);
            if (var_59) {
                // normal = wp.vec3(1.0, 0.0, 0.0)                                                <L 150>
                var_63 = wp::vec_t<3, wp::float32>(var_60, var_61, var_62);
                // if dist > 1.0e-8:                                                              <L 151>
                var_65 = (var_56 > var_64);
                if (var_65) {
                    // normal = delta / dist                                                      <L 152>
                    var_66 = wp::div(var_55, var_56);
                }
                var_67 = wp::where(var_65, var_66, var_63);
                // correction = normal * (penetration * contact_relaxation)                       <L 153>
                var_68 = wp::mul(var_57, var_28);
                var_69 = wp::mul(var_67, var_68);
                // correction_len = wp.length(correction)                                         <L 154>
                var_70 = wp::length(var_69);
                // if correction_cap > 0.0 and correction_len > correction_cap:                   <L 155>
                var_73 = (var_33 > var_72);
                var_71 = var_73;
                if (var_71) {
                    var_74 = (var_70 > var_33);
                    var_71 = var_71 && var_74;
                }
                if (var_71) {
                    // correction = correction * (correction_cap / correction_len)                <L 156>
                    var_75 = wp::div(var_33, var_70);
                    var_76 = wp::mul(var_69, var_75);
                }
                var_77 = wp::where(var_71, var_76, var_69);
                // q = q + correction                                                             <L 157>
                var_78 = wp::add(var_35, var_77);
                // correction_total = correction_total + correction                               <L 158>
                var_79 = wp::add(var_40, var_77);
            }
            var_80 = wp::where(var_59, var_78, var_35);
            var_81 = wp::where(var_59, var_79, var_40);
            wp::assign(var_35, var_80);
            wp::assign(var_40, var_81);
            goto start_for_2;
        end_for_2:;
        // if wp.dot(correction_total, correction_total) > 0.0:                                   <L 160>
        var_82 = wp::dot(var_40, var_40);
        var_84 = (var_82 > var_83);
        if (var_84) {
            // particle_q[tid] = q                                                                <L 161>
            wp::array_store(var_particle_q, var_0, var_35);
        }
    }
}



extern "C" __global__ void project_kinematic_sphere_mc_triangle_node_positions_kernel_5af77c50_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::int32> var_tri_indices,
    wp::int32 var_triangle_count,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q_prev,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q,
    wp::int32 var_sphere_count,
    wp::float32 var_sphere_radius,
    wp::float32 var_interpolation_alpha,
    wp::float32 var_relaxation,
    wp::float32 var_max_correction,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_delta_accumulator,
    wp::array_t<wp::int32> var_particle_delta_counter)
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
        bool var_1;
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 1;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 2;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        bool var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        const wp::int32 var_21 = 6;
        wp::int32 var_22;
        const wp::int32 var_23 = 6;
        wp::int32 var_24;
        const wp::int32 var_25 = 6;
        wp::int32 var_26;
        wp::float32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.0;
        bool var_33;
        wp::float32 var_34;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::float32 var_37 = 0.0;
        wp::float32 var_38;
        const wp::float32 var_39 = 1.0;
        bool var_40;
        const wp::float32 var_41 = 1.0;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.0;
        bool var_45;
        const wp::float32 var_46 = 0.0;
        wp::float32 var_47;
        const wp::float32 var_48 = 1.0;
        bool var_49;
        const wp::float32 var_50 = 1.0;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::float32 var_53 = 0.0;
        bool var_54;
        const wp::float32 var_55 = 0.0;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32>* var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32>* var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::range_t var_66;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32>* var_68;
        wp::vec_t<3, wp::float32>* var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::int32 var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        const wp::float32 var_85 = 1.0;
        const wp::float32 var_86 = 0.0;
        const wp::float32 var_87 = 0.0;
        wp::vec_t<3, wp::float32> var_88;
        const wp::float32 var_89 = 1e-08;
        bool var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::float32 var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::float32 var_97;
        bool var_98;
        const wp::float32 var_99 = 0.0;
        bool var_100;
        bool var_101;
        wp::float32 var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::float32 var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::float32 var_107;
        wp::vec_t<3, wp::float32> var_108;
        wp::float32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        //---------
        // forward
        // def project_kinematic_sphere_mc_triangle_node_positions_kernel(                        <L 165>
        // tri = wp.tid()                                                                         <L 183>
        var_0 = builtin_tid1d();
        // if tri >= triangle_count:                                                              <L 184>
        var_1 = (var_0 >= var_triangle_count);
        if (var_1) {
            // return                                                                             <L 185>
            continue;
        }
        // v0_id = tri_indices[tri, 0]                                                            <L 187>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1_id = tri_indices[tri, 1]                                                            <L 188>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2_id = tri_indices[tri, 2]                                                            <L 189>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // if v0_id < 0 or v1_id < 0 or v2_id < 0:                                                <L 190>
        var_16 = (var_4 < var_15);
        var_14 = var_16;
        if (!var_14) {
            var_18 = (var_8 < var_17);
            var_14 = var_14 || var_18;
        }
        if (!var_14) {
            var_20 = (var_12 < var_19);
            var_14 = var_14 || var_20;
        }
        if (var_14) {
            // return                                                                             <L 191>
            continue;
        }
        // c0 = v0_id / 6                                                                         <L 194>
        var_22 = wp::div(var_4, var_21);
        // c1 = v1_id / 6                                                                         <L 195>
        var_24 = wp::div(var_8, var_23);
        // c2 = v2_id / 6                                                                         <L 196>
        var_26 = wp::div(var_12, var_25);
        // w0 = _cell_inv_mass_sum(c0, cell_nodes, cell_active, particle_inv_mass, particle_flags)       <L 198>
        var_27 = _cell_inv_mass_sum_0(var_22, var_cell_nodes, var_cell_active, var_particle_inv_mass, var_particle_flags);
        // w1 = _cell_inv_mass_sum(c1, cell_nodes, cell_active, particle_inv_mass, particle_flags)       <L 199>
        var_28 = _cell_inv_mass_sum_0(var_24, var_cell_nodes, var_cell_active, var_particle_inv_mass, var_particle_flags);
        // w2 = _cell_inv_mass_sum(c2, cell_nodes, cell_active, particle_inv_mass, particle_flags)       <L 200>
        var_29 = _cell_inv_mass_sum_0(var_26, var_cell_nodes, var_cell_active, var_particle_inv_mass, var_particle_flags);
        // weight = w0 + w1 + w2                                                                  <L 201>
        var_30 = wp::add(var_27, var_28);
        var_31 = wp::add(var_30, var_29);
        // if weight <= 0.0:                                                                      <L 202>
        var_33 = (var_31 <= var_32);
        if (var_33) {
            // return                                                                             <L 203>
            continue;
        }
        // alpha = interpolation_alpha                                                            <L 205>
        var_34 = wp::copy(var_interpolation_alpha);
        // if alpha < 0.0:                                                                        <L 206>
        var_36 = (var_34 < var_35);
        if (var_36) {
            // alpha = 0.0                                                                        <L 207>
        }
        var_38 = wp::where(var_36, var_37, var_34);
        // if alpha > 1.0:                                                                        <L 208>
        var_40 = (var_38 > var_39);
        if (var_40) {
            // alpha = 1.0                                                                        <L 209>
        }
        var_42 = wp::where(var_40, var_41, var_38);
        // contact_relaxation = relaxation                                                        <L 211>
        var_43 = wp::copy(var_relaxation);
        // if contact_relaxation < 0.0:                                                           <L 212>
        var_45 = (var_43 < var_44);
        if (var_45) {
            // contact_relaxation = 0.0                                                           <L 213>
        }
        var_47 = wp::where(var_45, var_46, var_43);
        // if contact_relaxation > 1.0:                                                           <L 214>
        var_49 = (var_47 > var_48);
        if (var_49) {
            // contact_relaxation = 1.0                                                           <L 215>
        }
        var_51 = wp::where(var_49, var_50, var_47);
        // correction_cap = max_correction                                                        <L 217>
        var_52 = wp::copy(var_max_correction);
        // if correction_cap < 0.0:                                                               <L 218>
        var_54 = (var_52 < var_53);
        if (var_54) {
            // correction_cap = 0.0                                                               <L 219>
        }
        var_56 = wp::where(var_54, var_55, var_52);
        // p0 = vertex_pos[v0_id]                                                                 <L 221>
        var_57 = wp::address(var_vertex_pos, var_4);
        var_59 = wp::load(var_57);
        var_58 = wp::copy(var_59);
        // p1 = vertex_pos[v1_id]                                                                 <L 222>
        var_60 = wp::address(var_vertex_pos, var_8);
        var_62 = wp::load(var_60);
        var_61 = wp::copy(var_62);
        // p2 = vertex_pos[v2_id]                                                                 <L 223>
        var_63 = wp::address(var_vertex_pos, var_12);
        var_65 = wp::load(var_63);
        var_64 = wp::copy(var_65);
        // for sphere_idx in range(sphere_count):                                                 <L 225>
        var_66 = wp::range(var_sphere_count);
        start_for_3:;
            if (iter_cmp(var_66) == 0) goto end_for_3;
            var_67 = wp::iter_next(var_66);
            // center = sphere_q_prev[sphere_idx] + (sphere_q[sphere_idx] - sphere_q_prev[sphere_idx]) * alpha       <L 226>
            var_68 = wp::address(var_sphere_q_prev, var_67);
            var_69 = wp::address(var_sphere_q, var_67);
            var_70 = wp::address(var_sphere_q_prev, var_67);
            var_72 = wp::load(var_69);
            var_73 = wp::load(var_70);
            var_71 = wp::sub(var_72, var_73);
            var_74 = wp::mul(var_71, var_42);
            var_76 = wp::load(var_68);
            var_75 = wp::add(var_76, var_74);
            // closest_p, _bary, _feature_type = triangle_closest_point(p0, p1, p2, center)       <L 227>
            triangle_closest_point_0(var_58, var_61, var_64, var_75, var_77, var_78, var_79);
            // to_triangle = closest_p - center                                                   <L 229>
            var_80 = wp::sub(var_77, var_75);
            // dist = wp.length(to_triangle)                                                      <L 230>
            var_81 = wp::length(var_80);
            // penetration = sphere_radius - dist                                                 <L 231>
            var_82 = wp::sub(var_sphere_radius, var_81);
            // if penetration <= 0.0:                                                             <L 232>
            var_84 = (var_82 <= var_83);
            if (var_84) {
                // continue                                                                       <L 233>
                goto start_for_3;
            }
            // correction_dir = wp.vec3(1.0, 0.0, 0.0)                                            <L 235>
            var_88 = wp::vec_t<3, wp::float32>(var_85, var_86, var_87);
            // if dist > 1.0e-8:                                                                  <L 236>
            var_90 = (var_81 > var_89);
            if (var_90) {
                // correction_dir = to_triangle / dist                                            <L 237>
                var_91 = wp::div(var_80, var_81);
            }
            var_92 = wp::where(var_90, var_91, var_88);
            if (!var_90) {
                // correction_dir = _sphere_triangle_fallback_dir(p0, p1, p2, center)             <L 239>
                var_93 = _sphere_triangle_fallback_dir_0(var_58, var_61, var_64, var_75);
            }
            var_94 = wp::where(var_90, var_92, var_93);
            // correction = correction_dir * (penetration * contact_relaxation)                   <L 241>
            var_95 = wp::mul(var_82, var_51);
            var_96 = wp::mul(var_94, var_95);
            // correction_len = wp.length(correction)                                             <L 242>
            var_97 = wp::length(var_96);
            // if correction_cap > 0.0 and correction_len > correction_cap:                       <L 243>
            var_100 = (var_56 > var_99);
            var_98 = var_100;
            if (var_98) {
                var_101 = (var_97 > var_56);
                var_98 = var_98 && var_101;
            }
            if (var_98) {
                // correction = correction * (correction_cap / correction_len)                    <L 244>
                var_102 = wp::div(var_56, var_97);
                var_103 = wp::mul(var_96, var_102);
            }
            var_104 = wp::where(var_98, var_103, var_96);
            // _scatter_cell_correction_to_nodes(                                                 <L 246>
            // c0,                                                                                <L 247>
            // correction * (w0 / weight),                                                        <L 248>
            var_105 = wp::div(var_27, var_31);
            var_106 = wp::mul(var_104, var_105);
            // w0,                                                                                <L 249>
            // cell_nodes,                                                                        <L 250>
            // particle_inv_mass,                                                                 <L 251>
            // particle_flags,                                                                    <L 252>
            // particle_delta_accumulator,                                                        <L 253>
            // particle_delta_counter,                                                            <L 254>
            _scatter_cell_correction_to_nodes_0(var_22, var_106, var_27, var_cell_nodes, var_particle_inv_mass, var_particle_flags, var_particle_delta_accumulator, var_particle_delta_counter);
            // _scatter_cell_correction_to_nodes(                                                 <L 256>
            // c1,                                                                                <L 257>
            // correction * (w1 / weight),                                                        <L 258>
            var_107 = wp::div(var_28, var_31);
            var_108 = wp::mul(var_104, var_107);
            // w1,                                                                                <L 259>
            // cell_nodes,                                                                        <L 260>
            // particle_inv_mass,                                                                 <L 261>
            // particle_flags,                                                                    <L 262>
            // particle_delta_accumulator,                                                        <L 263>
            // particle_delta_counter,                                                            <L 264>
            _scatter_cell_correction_to_nodes_0(var_24, var_108, var_28, var_cell_nodes, var_particle_inv_mass, var_particle_flags, var_particle_delta_accumulator, var_particle_delta_counter);
            // _scatter_cell_correction_to_nodes(                                                 <L 266>
            // c2,                                                                                <L 267>
            // correction * (w2 / weight),                                                        <L 268>
            var_109 = wp::div(var_29, var_31);
            var_110 = wp::mul(var_104, var_109);
            // w2,                                                                                <L 269>
            // cell_nodes,                                                                        <L 270>
            // particle_inv_mass,                                                                 <L 271>
            // particle_flags,                                                                    <L 272>
            // particle_delta_accumulator,                                                        <L 273>
            // particle_delta_counter,                                                            <L 274>
            _scatter_cell_correction_to_nodes_0(var_26, var_110, var_29, var_cell_nodes, var_particle_inv_mass, var_particle_flags, var_particle_delta_accumulator, var_particle_delta_counter);
            goto start_for_3;
        end_for_3:;
    }
}

