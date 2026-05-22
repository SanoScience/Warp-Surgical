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


// /home/pkorzeniowsk/Projects/newton/newton-1.0/newton/_src/geometry/kernels.py:63
static void triangle_closest_point_0(
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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/input/haptic_collision.py:11
static wp::vec_t<3, wp::float32> _double_sided_reaction_dir_0(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_contact_point,
    wp::vec_t<3, wp::float32> var_reference_dir)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-12;
    bool var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    const wp::float32 var_6 = 3.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 1e-12;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    bool var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 1e-12;
    bool var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = 0.0;
    const wp::float32 var_33 = 0.0;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // forward
    // def _double_sided_reaction_dir(                                                        <L 12>
    // if wp.length_sq(reference_dir) > 1.0e-12:                                              <L 19>
    var_0 = wp::length_sq(var_reference_dir);
    var_2 = (var_0 > var_1);
    if (var_2) {
        // return wp.normalize(reference_dir)                                                 <L 20>
        var_3 = wp::normalize(var_reference_dir);
        return var_3;
    }
    // centroid = (v0 + v1 + v2) / 3.0                                                        <L 22>
    var_4 = wp::add(var_v0, var_v1);
    var_5 = wp::add(var_4, var_v2);
    var_7 = wp::div(var_5, var_6);
    // centroid_dir = centroid - contact_point                                                <L 23>
    var_8 = wp::sub(var_7, var_contact_point);
    // if wp.length_sq(centroid_dir) > 1.0e-12:                                               <L 24>
    var_9 = wp::length_sq(var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // return wp.normalize(centroid_dir)                                                  <L 25>
        var_12 = wp::normalize(var_8);
        return var_12;
    }
    // d0 = v0 - contact_point                                                                <L 27>
    var_13 = wp::sub(var_v0, var_contact_point);
    // d1 = v1 - contact_point                                                                <L 28>
    var_14 = wp::sub(var_v1, var_contact_point);
    // d2 = v2 - contact_point                                                                <L 29>
    var_15 = wp::sub(var_v2, var_contact_point);
    // best = d0                                                                              <L 31>
    var_16 = wp::copy(var_13);
    // if wp.length_sq(d1) > wp.length_sq(best):                                              <L 32>
    var_17 = wp::length_sq(var_14);
    var_18 = wp::length_sq(var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // best = d1                                                                          <L 33>
        var_20 = wp::copy(var_14);
    }
    var_21 = wp::where(var_19, var_20, var_16);
    // if wp.length_sq(d2) > wp.length_sq(best):                                              <L 34>
    var_22 = wp::length_sq(var_15);
    var_23 = wp::length_sq(var_21);
    var_24 = (var_22 > var_23);
    if (var_24) {
        // best = d2                                                                          <L 35>
        var_25 = wp::copy(var_15);
    }
    var_26 = wp::where(var_24, var_25, var_21);
    // if wp.length_sq(best) > 1.0e-12:                                                       <L 37>
    var_27 = wp::length_sq(var_26);
    var_29 = (var_27 > var_28);
    if (var_29) {
        // return wp.normalize(best)                                                          <L 38>
        var_30 = wp::normalize(var_26);
        return var_30;
    }
    // return wp.vec3f(1.0, 0.0, 0.0)                                                         <L 40>
    var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
    return var_34;
}


// /home/pkorzeniowsk/Projects/newton/newton-1.0/newton/_src/geometry/kernels.py:63
static void adj_triangle_closest_point_0(
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
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    bool adj_5 = {};
    wp::float32 adj_6 = {};
    bool adj_7 = {};
    wp::float32 adj_8 = {};
    bool adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::vec_t<3, wp::float32> adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    bool adj_19 = {};
    wp::float32 adj_20 = {};
    bool adj_21 = {};
    bool adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::int32 adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::vec_t<3, wp::float32> adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    bool adj_34 = {};
    wp::float32 adj_35 = {};
    bool adj_36 = {};
    bool adj_37 = {};
    wp::int32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::vec_t<3, wp::float32> adj_43 = {};
    wp::int32 adj_44 = {};
    wp::vec_t<3, wp::float32> adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    bool adj_49 = {};
    wp::float32 adj_50 = {};
    bool adj_51 = {};
    wp::float32 adj_52 = {};
    bool adj_53 = {};
    wp::float32 adj_54 = {};
    bool adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::int32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::vec_t<3, wp::float32> adj_64 = {};
    wp::vec_t<3, wp::float32> adj_65 = {};
    wp::int32 adj_66 = {};
    wp::vec_t<3, wp::float32> adj_67 = {};
    wp::float32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::float32 adj_70 = {};
    bool adj_71 = {};
    wp::float32 adj_72 = {};
    bool adj_73 = {};
    wp::float32 adj_74 = {};
    bool adj_75 = {};
    wp::float32 adj_76 = {};
    bool adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::int32 adj_80 = {};
    wp::int32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::vec_t<3, wp::float32> adj_85 = {};
    wp::vec_t<3, wp::float32> adj_86 = {};
    wp::vec_t<3, wp::float32> adj_87 = {};
    wp::int32 adj_88 = {};
    wp::vec_t<3, wp::float32> adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float32 adj_91 = {};
    wp::float32 adj_92 = {};
    wp::float32 adj_93 = {};
    bool adj_94 = {};
    wp::float32 adj_95 = {};
    bool adj_96 = {};
    wp::float32 adj_97 = {};
    wp::float32 adj_98 = {};
    bool adj_99 = {};
    wp::float32 adj_100 = {};
    wp::float32 adj_101 = {};
    bool adj_102 = {};
    wp::float32 adj_103 = {};
    wp::float32 adj_104 = {};
    wp::float32 adj_105 = {};
    wp::float32 adj_106 = {};
    wp::float32 adj_107 = {};
    wp::int32 adj_108 = {};
    wp::int32 adj_109 = {};
    wp::float32 adj_110 = {};
    wp::float32 adj_111 = {};
    wp::float32 adj_112 = {};
    wp::vec_t<3, wp::float32> adj_113 = {};
    wp::vec_t<3, wp::float32> adj_114 = {};
    wp::vec_t<3, wp::float32> adj_115 = {};
    wp::vec_t<3, wp::float32> adj_116 = {};
    wp::int32 adj_117 = {};
    wp::vec_t<3, wp::float32> adj_118 = {};
    wp::float32 adj_119 = {};
    wp::float32 adj_120 = {};
    wp::float32 adj_121 = {};
    wp::float32 adj_122 = {};
    wp::float32 adj_123 = {};
    wp::float32 adj_124 = {};
    wp::float32 adj_125 = {};
    wp::int32 adj_126 = {};
    wp::int32 adj_127 = {};
    wp::float32 adj_128 = {};
    wp::float32 adj_129 = {};
    wp::float32 adj_130 = {};
    wp::vec_t<3, wp::float32> adj_131 = {};
    wp::vec_t<3, wp::float32> adj_132 = {};
    wp::vec_t<3, wp::float32> adj_133 = {};
    wp::vec_t<3, wp::float32> adj_134 = {};
    wp::vec_t<3, wp::float32> adj_135 = {};
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
        goto label0;
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
        goto label1;
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
        goto label2;
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
        goto label3;
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
        goto label4;
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
        goto label5;
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
    goto label6;
    //---------
    // reverse
    label6:;
    adj_127 += adj_ret_2;
    adj_131 += adj_ret_1;
    adj_135 += adj_ret_0;
    wp::adj_add(var_133, var_134, adj_133, adj_134, adj_135);
    wp::adj_mul(var_125, var_1, adj_125, adj_1, adj_134);
    wp::adj_add(var_a, var_132, adj_a, adj_132, adj_133);
    wp::adj_mul(var_124, var_0, adj_124, adj_0, adj_132);
    // adj: return a + v * ab + w * ac, bary, feature_type                                    <L 128>
    wp::adj_vec_t(var_130, var_124, var_125, adj_130, adj_124, adj_125, adj_131);
    wp::adj_sub(var_129, var_125, adj_129, adj_125, adj_130);
    wp::adj_sub(var_128, var_124, adj_128, adj_124, adj_129);
    // adj: bary = wp.vec3(1.0 - v - w, v, w)                                                 <L 127>
    wp::adj_copy(var_126, adj_126, adj_127);
    // adj: feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR                                  <L 126>
    wp::adj_mul(var_48, var_123, adj_48, adj_123, adj_125);
    // adj: w = vc * denom                                                                    <L 125>
    wp::adj_mul(var_70, var_123, adj_70, adj_123, adj_124);
    // adj: v = vb * denom                                                                    <L 124>
    wp::adj_div(var_120, var_122, var_123, adj_120, adj_122, adj_123);
    wp::adj_add(var_121, var_48, adj_121, adj_48, adj_122);
    wp::adj_add(var_93, var_70, adj_93, adj_70, adj_121);
    // adj: denom = 1.0 / (va + vb + vc)                                                      <L 123>
    wp::adj_where(var_94, var_107, var_90, adj_94, adj_107, adj_90, adj_119);
    wp::adj_where(var_94, var_113, var_89, adj_94, adj_113, adj_89, adj_118);
    wp::adj_where(var_94, var_109, var_88, adj_94, adj_109, adj_88, adj_117);
    if (var_94) {
        label5:;
        adj_109 += adj_ret_2;
        adj_113 += adj_ret_1;
        adj_116 += adj_ret_0;
        wp::adj_add(var_b, var_115, adj_b, adj_115, adj_116);
        wp::adj_mul(var_107, var_114, adj_107, adj_114, adj_115);
        wp::adj_sub(var_c, var_b, adj_c, adj_b, adj_114);
        // adj: return b + v * (c - b), bary, feature_type                                    <L 121>
        wp::adj_vec_t(var_110, var_112, var_107, adj_110, adj_112, adj_107, adj_113);
        wp::adj_sub(var_111, var_107, adj_111, adj_107, adj_112);
        // adj: bary = wp.vec3(0.0, 1.0 - v, v)                                               <L 120>
        wp::adj_copy(var_108, adj_108, adj_109);
        // adj: feature_type = TRI_CONTACT_FEATURE_EDGE_BC                                    <L 119>
        wp::adj_div(var_103, var_106, var_107, adj_103, adj_106, adj_107);
        wp::adj_add(var_104, var_105, adj_104, adj_105, adj_106);
        wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_105);
        wp::adj_sub(var_18, var_17, adj_18, adj_17, adj_104);
        wp::adj_sub(var_18, var_17, adj_18, adj_17, adj_103);
        // adj: v = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                       <L 118>
    }
    if (var_94) {
        wp::adj_sub(var_32, var_33, adj_32, adj_33, adj_100);
    }
    if (var_94) {
        wp::adj_sub(var_18, var_17, adj_18, adj_17, adj_97);
    }
    // adj: if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                           <L 117>
    wp::adj_sub(var_91, var_92, adj_91, adj_92, adj_93);
    wp::adj_mul(var_32, var_18, adj_32, adj_18, adj_92);
    wp::adj_mul(var_17, var_33, adj_17, adj_33, adj_91);
    // adj: va = d3 * d6 - d5 * d4                                                            <L 116>
    wp::adj_where(var_71, var_79, var_57, adj_71, adj_79, adj_57, adj_90);
    wp::adj_where(var_71, var_85, var_67, adj_71, adj_85, adj_67, adj_89);
    wp::adj_where(var_71, var_81, var_66, adj_71, adj_81, adj_66, adj_88);
    if (var_71) {
        label4:;
        adj_81 += adj_ret_2;
        adj_85 += adj_ret_1;
        adj_87 += adj_ret_0;
        wp::adj_add(var_a, var_86, adj_a, adj_86, adj_87);
        wp::adj_mul(var_79, var_1, adj_79, adj_1, adj_86);
        // adj: return a + v * ac, bary, feature_type                                         <L 114>
        wp::adj_vec_t(var_83, var_84, var_79, adj_83, adj_84, adj_79, adj_85);
        wp::adj_sub(var_82, var_79, adj_82, adj_79, adj_83);
        // adj: bary = wp.vec3(1.0 - v, 0.0, v)                                               <L 113>
        wp::adj_copy(var_80, adj_80, adj_81);
        // adj: feature_type = TRI_CONTACT_FEATURE_EDGE_AC                                    <L 112>
        wp::adj_div(var_4, var_78, var_79, adj_4, adj_78, adj_79);
        wp::adj_sub(var_4, var_33, adj_4, adj_33, adj_78);
        // adj: v = d2 / (d2 - d6)                                                            <L 111>
    }
    if (var_71) {
    }
    if (var_71) {
    }
    // adj: if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                         <L 110>
    wp::adj_sub(var_68, var_69, adj_68, adj_69, adj_70);
    wp::adj_mul(var_3, var_33, adj_3, adj_33, adj_69);
    wp::adj_mul(var_32, var_4, adj_32, adj_4, adj_68);
    // adj: vb = d5 * d2 - d1 * d6                                                            <L 109>
    wp::adj_where(var_49, var_63, var_45, adj_49, adj_63, adj_45, adj_67);
    wp::adj_where(var_49, var_59, var_44, adj_49, adj_59, adj_44, adj_66);
    if (var_49) {
        label3:;
        adj_59 += adj_ret_2;
        adj_63 += adj_ret_1;
        adj_65 += adj_ret_0;
        wp::adj_add(var_a, var_64, adj_a, adj_64, adj_65);
        wp::adj_mul(var_57, var_0, adj_57, adj_0, adj_64);
        // adj: return a + v * ab, bary, feature_type                                         <L 107>
        wp::adj_vec_t(var_61, var_57, var_62, adj_61, adj_57, adj_62, adj_63);
        wp::adj_sub(var_60, var_57, adj_60, adj_57, adj_61);
        // adj: bary = wp.vec3(1.0 - v, v, 0.0)                                               <L 106>
        wp::adj_copy(var_58, adj_58, adj_59);
        // adj: feature_type = TRI_CONTACT_FEATURE_EDGE_AB                                    <L 105>
        wp::adj_div(var_3, var_56, var_57, adj_3, adj_56, adj_57);
        wp::adj_sub(var_3, var_17, adj_3, adj_17, adj_56);
        // adj: v = d1 / (d1 - d3)                                                            <L 104>
    }
    if (var_49) {
    }
    if (var_49) {
    }
    // adj: if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                         <L 103>
    wp::adj_sub(var_46, var_47, adj_46, adj_47, adj_48);
    wp::adj_mul(var_17, var_4, adj_17, adj_4, adj_47);
    wp::adj_mul(var_3, var_18, adj_3, adj_18, adj_46);
    // adj: vc = d1 * d4 - d3 * d2                                                            <L 102>
    wp::adj_where(var_34, var_43, var_30, adj_34, adj_43, adj_30, adj_45);
    wp::adj_where(var_34, var_39, var_29, adj_34, adj_39, adj_29, adj_44);
    if (var_34) {
        label2:;
        adj_39 += adj_ret_2;
        adj_43 += adj_ret_1;
        adj_c += adj_ret_0;
        // adj: return c, bary, feature_type                                                  <L 100>
        wp::adj_vec_t(var_40, var_41, var_42, adj_40, adj_41, adj_42, adj_43);
        // adj: bary = wp.vec3(0.0, 0.0, 1.0)                                                 <L 99>
        wp::adj_copy(var_38, adj_38, adj_39);
        // adj: feature_type = TRI_CONTACT_FEATURE_VERTEX_C                                   <L 98>
    }
    if (var_34) {
    }
    // adj: if d6 >= 0.0 and d5 <= d6:                                                        <L 97>
    wp::adj_dot(var_1, var_31, adj_1, adj_31, adj_33);
    // adj: d6 = wp.dot(ac, cp)                                                               <L 96>
    wp::adj_dot(var_0, var_31, adj_0, adj_31, adj_32);
    // adj: d5 = wp.dot(ab, cp)                                                               <L 95>
    wp::adj_sub(var_p, var_c, adj_p, adj_c, adj_31);
    // adj: cp = p - c                                                                        <L 94>
    wp::adj_where(var_19, var_28, var_15, adj_19, adj_28, adj_15, adj_30);
    wp::adj_where(var_19, var_24, var_11, adj_19, adj_24, adj_11, adj_29);
    if (var_19) {
        label1:;
        adj_24 += adj_ret_2;
        adj_28 += adj_ret_1;
        adj_b += adj_ret_0;
        // adj: return b, bary, feature_type                                                  <L 92>
        wp::adj_vec_t(var_25, var_26, var_27, adj_25, adj_26, adj_27, adj_28);
        // adj: bary = wp.vec3(0.0, 1.0, 0.0)                                                 <L 91>
        wp::adj_copy(var_23, adj_23, adj_24);
        // adj: feature_type = TRI_CONTACT_FEATURE_VERTEX_B                                   <L 90>
    }
    if (var_19) {
    }
    // adj: if d3 >= 0.0 and d4 <= d3:                                                        <L 89>
    wp::adj_dot(var_1, var_16, adj_1, adj_16, adj_18);
    // adj: d4 = wp.dot(ac, bp)                                                               <L 88>
    wp::adj_dot(var_0, var_16, adj_0, adj_16, adj_17);
    // adj: d3 = wp.dot(ab, bp)                                                               <L 87>
    wp::adj_sub(var_p, var_b, adj_p, adj_b, adj_16);
    // adj: bp = p - b                                                                        <L 86>
    if (var_5) {
        label0:;
        adj_11 += adj_ret_2;
        adj_15 += adj_ret_1;
        adj_a += adj_ret_0;
        // adj: return a, bary, feature_type                                                  <L 84>
        wp::adj_vec_t(var_12, var_13, var_14, adj_12, adj_13, adj_14, adj_15);
        // adj: bary = wp.vec3(1.0, 0.0, 0.0)                                                 <L 83>
        wp::adj_copy(var_10, adj_10, adj_11);
        // adj: feature_type = TRI_CONTACT_FEATURE_VERTEX_A                                   <L 82>
    }
    if (var_5) {
    }
    // adj: if d1 <= 0.0 and d2 <= 0.0:                                                       <L 81>
    wp::adj_dot(var_1, var_2, adj_1, adj_2, adj_4);
    // adj: d2 = wp.dot(ac, ap)                                                               <L 80>
    wp::adj_dot(var_0, var_2, adj_0, adj_2, adj_3);
    // adj: d1 = wp.dot(ab, ap)                                                               <L 79>
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_2);
    // adj: ap = p - a                                                                        <L 77>
    wp::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    // adj: ac = c - a                                                                        <L 76>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: ab = b - a                                                                        <L 75>
    // adj: def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):       <L 64>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/input/haptic_collision.py:11
static void adj__double_sided_reaction_dir_0(
    wp::vec_t<3, wp::float32> var_v0,
    wp::vec_t<3, wp::float32> var_v1,
    wp::vec_t<3, wp::float32> var_v2,
    wp::vec_t<3, wp::float32> var_contact_point,
    wp::vec_t<3, wp::float32> var_reference_dir,
    wp::vec_t<3, wp::float32> & adj_v0,
    wp::vec_t<3, wp::float32> & adj_v1,
    wp::vec_t<3, wp::float32> & adj_v2,
    wp::vec_t<3, wp::float32> & adj_contact_point,
    wp::vec_t<3, wp::float32> & adj_reference_dir,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-12;
    bool var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    const wp::float32 var_6 = 3.0;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 1e-12;
    bool var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32> var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    bool var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 1e-12;
    bool var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = 0.0;
    const wp::float32 var_33 = 0.0;
    wp::vec_t<3, wp::float32> var_34;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    bool adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::float32 adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    bool adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::vec_t<3, wp::float32> adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    bool adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    bool adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    bool adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::vec_t<3, wp::float32> adj_34 = {};
    //---------
    // forward
    // def _double_sided_reaction_dir(                                                        <L 12>
    // if wp.length_sq(reference_dir) > 1.0e-12:                                              <L 19>
    var_0 = wp::length_sq(var_reference_dir);
    var_2 = (var_0 > var_1);
    if (var_2) {
        // return wp.normalize(reference_dir)                                                 <L 20>
        var_3 = wp::normalize(var_reference_dir);
        goto label0;
    }
    // centroid = (v0 + v1 + v2) / 3.0                                                        <L 22>
    var_4 = wp::add(var_v0, var_v1);
    var_5 = wp::add(var_4, var_v2);
    var_7 = wp::div(var_5, var_6);
    // centroid_dir = centroid - contact_point                                                <L 23>
    var_8 = wp::sub(var_7, var_contact_point);
    // if wp.length_sq(centroid_dir) > 1.0e-12:                                               <L 24>
    var_9 = wp::length_sq(var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // return wp.normalize(centroid_dir)                                                  <L 25>
        var_12 = wp::normalize(var_8);
        goto label1;
    }
    // d0 = v0 - contact_point                                                                <L 27>
    var_13 = wp::sub(var_v0, var_contact_point);
    // d1 = v1 - contact_point                                                                <L 28>
    var_14 = wp::sub(var_v1, var_contact_point);
    // d2 = v2 - contact_point                                                                <L 29>
    var_15 = wp::sub(var_v2, var_contact_point);
    // best = d0                                                                              <L 31>
    var_16 = wp::copy(var_13);
    // if wp.length_sq(d1) > wp.length_sq(best):                                              <L 32>
    var_17 = wp::length_sq(var_14);
    var_18 = wp::length_sq(var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // best = d1                                                                          <L 33>
        var_20 = wp::copy(var_14);
    }
    var_21 = wp::where(var_19, var_20, var_16);
    // if wp.length_sq(d2) > wp.length_sq(best):                                              <L 34>
    var_22 = wp::length_sq(var_15);
    var_23 = wp::length_sq(var_21);
    var_24 = (var_22 > var_23);
    if (var_24) {
        // best = d2                                                                          <L 35>
        var_25 = wp::copy(var_15);
    }
    var_26 = wp::where(var_24, var_25, var_21);
    // if wp.length_sq(best) > 1.0e-12:                                                       <L 37>
    var_27 = wp::length_sq(var_26);
    var_29 = (var_27 > var_28);
    if (var_29) {
        // return wp.normalize(best)                                                          <L 38>
        var_30 = wp::normalize(var_26);
        goto label2;
    }
    // return wp.vec3f(1.0, 0.0, 0.0)                                                         <L 40>
    var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
    goto label3;
    //---------
    // reverse
    label3:;
    adj_34 += adj_ret;
    wp::adj_vec_t(var_31, var_32, var_33, adj_31, adj_32, adj_33, adj_34);
    // adj: return wp.vec3f(1.0, 0.0, 0.0)                                                    <L 40>
    if (var_29) {
        label2:;
        adj_30 += adj_ret;
        wp::adj_normalize(var_26, var_30, adj_26, adj_30);
        // adj: return wp.normalize(best)                                                     <L 38>
    }
    wp::adj_length_sq(var_26, adj_26, adj_27);
    // adj: if wp.length_sq(best) > 1.0e-12:                                                  <L 37>
    wp::adj_where(var_24, var_25, var_21, adj_24, adj_25, adj_21, adj_26);
    if (var_24) {
        wp::adj_copy(var_15, adj_15, adj_25);
        // adj: best = d2                                                                     <L 35>
    }
    wp::adj_length_sq(var_21, adj_21, adj_23);
    wp::adj_length_sq(var_15, adj_15, adj_22);
    // adj: if wp.length_sq(d2) > wp.length_sq(best):                                         <L 34>
    wp::adj_where(var_19, var_20, var_16, adj_19, adj_20, adj_16, adj_21);
    if (var_19) {
        wp::adj_copy(var_14, adj_14, adj_20);
        // adj: best = d1                                                                     <L 33>
    }
    wp::adj_length_sq(var_16, adj_16, adj_18);
    wp::adj_length_sq(var_14, adj_14, adj_17);
    // adj: if wp.length_sq(d1) > wp.length_sq(best):                                         <L 32>
    wp::adj_copy(var_13, adj_13, adj_16);
    // adj: best = d0                                                                         <L 31>
    wp::adj_sub(var_v2, var_contact_point, adj_v2, adj_contact_point, adj_15);
    // adj: d2 = v2 - contact_point                                                           <L 29>
    wp::adj_sub(var_v1, var_contact_point, adj_v1, adj_contact_point, adj_14);
    // adj: d1 = v1 - contact_point                                                           <L 28>
    wp::adj_sub(var_v0, var_contact_point, adj_v0, adj_contact_point, adj_13);
    // adj: d0 = v0 - contact_point                                                           <L 27>
    if (var_11) {
        label1:;
        adj_12 += adj_ret;
        wp::adj_normalize(var_8, var_12, adj_8, adj_12);
        // adj: return wp.normalize(centroid_dir)                                             <L 25>
    }
    wp::adj_length_sq(var_8, adj_8, adj_9);
    // adj: if wp.length_sq(centroid_dir) > 1.0e-12:                                          <L 24>
    wp::adj_sub(var_7, var_contact_point, adj_7, adj_contact_point, adj_8);
    // adj: centroid_dir = centroid - contact_point                                           <L 23>
    wp::adj_div(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_add(var_4, var_v2, adj_4, adj_v2, adj_5);
    wp::adj_add(var_v0, var_v1, adj_v0, adj_v1, adj_4);
    // adj: centroid = (v0 + v1 + v2) / 3.0                                                   <L 22>
    if (var_2) {
        label0:;
        adj_3 += adj_ret;
        wp::adj_normalize(var_reference_dir, var_3, adj_reference_dir, adj_3);
        // adj: return wp.normalize(reference_dir)                                            <L 20>
    }
    wp::adj_length_sq(var_reference_dir, adj_reference_dir, adj_0);
    // adj: if wp.length_sq(reference_dir) > 1.0e-12:                                         <L 19>
    // adj: def _double_sided_reaction_dir(                                                   <L 12>
    return;
}

struct wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<wp::vec_t<3, wp::float32>> velocities;
    wp::array_t<wp::float32> inv_masses;
    wp::array_t<wp::int32> tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_center;
    wp::float32 sphere_radius;
    wp::float32 sphere_center_scale;
    wp::float32 restitution;
    wp::float32 dt;
    wp::float32 cull_radius;
    wp::array_t<wp::vec_t<3, wp::float32>> delta_accumulator;
    wp::array_t<wp::int32> delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> reaction_accumulator;
    wp::array_t<wp::int32> reaction_counter;
};


void collide_triangles_vs_haptic_sphere_with_reaction_84f216bc_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::float32> var_inv_masses = _wp_args->inv_masses;
    wp::array_t<wp::int32> var_tri_indices = _wp_args->tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_center = _wp_args->sphere_center;
    wp::float32 var_sphere_radius = _wp_args->sphere_radius;
    wp::float32 var_sphere_center_scale = _wp_args->sphere_center_scale;
    wp::float32 var_restitution = _wp_args->restitution;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_cull_radius = _wp_args->cull_radius;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta_accumulator = _wp_args->delta_accumulator;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_reaction_accumulator = _wp_args->reaction_accumulator;
    wp::array_t<wp::int32> var_reaction_counter = _wp_args->reaction_counter;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::shape_t* var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    wp::shape_t var_4;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 2;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::int32 var_27 = 0;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::float32 var_35 = 3.0;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    bool var_39;
    wp::float32* var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32* var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32* var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 0.0;
    bool var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::int32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    bool var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1e-08;
    bool var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::vec_t<3, wp::float32>* var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32>* var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    const wp::float32 var_71 = 3.0;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::float32 var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    const wp::int32 var_88 = 1;
    wp::int32 var_89;
    const wp::int32 var_90 = 1;
    wp::int32 var_91;
    const wp::int32 var_92 = 0;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    const wp::int32 var_95 = 0;
    const wp::int32 var_96 = 1;
    wp::int32 var_97;
    //---------
    // forward
    // def collide_triangles_vs_haptic_sphere_with_reaction(                                  <L 44>
    // tid = wp.tid()                                                                         <L 60>
    var_0 = builtin_tid1d();
    // if tid >= tri_indices.shape[0]:                                                        <L 61>
    var_1 = &(var_tri_indices.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = (var_0 >= var_3);
    if (var_5) {
        // return                                                                             <L 62>
        return;
    }
    // t1 = tri_indices[tid, 0]                                                               <L 64>
    var_7 = wp::address(var_tri_indices, var_0, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // t2 = tri_indices[tid, 1]                                                               <L 65>
    var_11 = wp::address(var_tri_indices, var_0, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // t3 = tri_indices[tid, 2]                                                               <L 66>
    var_15 = wp::address(var_tri_indices, var_0, var_14);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // p1 = positions[t1]                                                                     <L 68>
    var_18 = wp::address(var_positions, var_8);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // p2 = positions[t2]                                                                     <L 69>
    var_21 = wp::address(var_positions, var_12);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // p3 = positions[t3]                                                                     <L 70>
    var_24 = wp::address(var_positions, var_16);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // sphere_pos = sphere_center[0] * sphere_center_scale                                    <L 72>
    var_28 = wp::address(var_sphere_center, var_27);
    var_30 = wp::load(var_28);
    var_29 = wp::mul(var_30, var_sphere_center_scale);
    // if cull_radius > 0.0:                                                                  <L 74>
    var_32 = (var_cull_radius > var_31);
    if (var_32) {
        // centroid = (p1 + p2 + p3) / 3.0                                                    <L 75>
        var_33 = wp::add(var_19, var_22);
        var_34 = wp::add(var_33, var_25);
        var_36 = wp::div(var_34, var_35);
        // if wp.length(centroid - sphere_pos) > cull_radius:                                 <L 76>
        var_37 = wp::sub(var_36, var_29);
        var_38 = wp::length(var_37);
        var_39 = (var_38 > var_cull_radius);
        if (var_39) {
            // return                                                                         <L 77>
            return;
        }
    }
    // w1 = inv_masses[t1]                                                                    <L 79>
    var_40 = wp::address(var_inv_masses, var_8);
    var_42 = wp::load(var_40);
    var_41 = wp::copy(var_42);
    // w2 = inv_masses[t2]                                                                    <L 80>
    var_43 = wp::address(var_inv_masses, var_12);
    var_45 = wp::load(var_43);
    var_44 = wp::copy(var_45);
    // w3 = inv_masses[t3]                                                                    <L 81>
    var_46 = wp::address(var_inv_masses, var_16);
    var_48 = wp::load(var_46);
    var_47 = wp::copy(var_48);
    // weight = w1 + w2 + w3                                                                  <L 82>
    var_49 = wp::add(var_41, var_44);
    var_50 = wp::add(var_49, var_47);
    // if weight <= 0.0:                                                                      <L 83>
    var_52 = (var_50 <= var_51);
    if (var_52) {
        // return                                                                             <L 84>
        return;
    }
    // closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)         <L 86>
    triangle_closest_point_0(var_19, var_22, var_25, var_29, var_53, var_54, var_55);
    // to_sphere = closest_p - sphere_pos                                                     <L 87>
    var_56 = wp::sub(var_53, var_29);
    // dist = wp.length(to_sphere)                                                            <L 88>
    var_57 = wp::length(var_56);
    // if dist >= sphere_radius:                                                              <L 89>
    var_58 = (var_57 >= var_sphere_radius);
    if (var_58) {
        // return                                                                             <L 90>
        return;
    }
    // penetration = sphere_radius - dist                                                     <L 92>
    var_59 = wp::sub(var_sphere_radius, var_57);
    // if dist > 1.0e-8:                                                                      <L 93>
    var_61 = (var_57 > var_60);
    if (var_61) {
        // correction_dir = to_sphere / dist                                                  <L 94>
        var_62 = wp::div(var_56, var_57);
    }
    if (!var_61) {
        // correction_dir = _double_sided_reaction_dir(                                       <L 96>
        // p1,                                                                                <L 97>
        // p2,                                                                                <L 98>
        // p3,                                                                                <L 99>
        // sphere_pos,                                                                        <L 100>
        // -((velocities[t1] + velocities[t2] + velocities[t3]) / 3.0),                       <L 101>
        var_63 = wp::address(var_velocities, var_8);
        var_64 = wp::address(var_velocities, var_12);
        var_66 = wp::load(var_63);
        var_67 = wp::load(var_64);
        var_65 = wp::add(var_66, var_67);
        var_68 = wp::address(var_velocities, var_16);
        var_70 = wp::load(var_68);
        var_69 = wp::add(var_65, var_70);
        var_72 = wp::div(var_69, var_71);
        var_73 = wp::neg(var_72);
        var_74 = _double_sided_reaction_dir_0(var_19, var_22, var_25, var_29, var_73);
    }
    var_75 = wp::where(var_61, var_62, var_74);
    // total_correction = correction_dir * penetration                                        <L 104>
    var_76 = wp::mul(var_75, var_59);
    // d1 = total_correction * (w1 / weight)                                                  <L 105>
    var_77 = wp::div(var_41, var_50);
    var_78 = wp::mul(var_76, var_77);
    // d2 = total_correction * (w2 / weight)                                                  <L 106>
    var_79 = wp::div(var_44, var_50);
    var_80 = wp::mul(var_76, var_79);
    // d3 = total_correction * (w3 / weight)                                                  <L 107>
    var_81 = wp::div(var_47, var_50);
    var_82 = wp::mul(var_76, var_81);
    // wp.atomic_add(delta_accumulator, t1, d1)                                               <L 109>
    var_83 = wp::atomic_add(var_delta_accumulator, var_8, var_78);
    // wp.atomic_add(delta_accumulator, t2, d2)                                               <L 110>
    var_84 = wp::atomic_add(var_delta_accumulator, var_12, var_80);
    // wp.atomic_add(delta_accumulator, t3, d3)                                               <L 111>
    var_85 = wp::atomic_add(var_delta_accumulator, var_16, var_82);
    // wp.atomic_add(delta_counter, t1, 1)                                                    <L 112>
    var_87 = wp::atomic_add(var_delta_counter, var_8, var_86);
    // wp.atomic_add(delta_counter, t2, 1)                                                    <L 113>
    var_89 = wp::atomic_add(var_delta_counter, var_12, var_88);
    // wp.atomic_add(delta_counter, t3, 1)                                                    <L 114>
    var_91 = wp::atomic_add(var_delta_counter, var_16, var_90);
    // wp.atomic_add(reaction_accumulator, 0, -total_correction)                              <L 117>
    var_93 = wp::neg(var_76);
    var_94 = wp::atomic_add(var_reaction_accumulator, var_92, var_93);
    // wp.atomic_add(reaction_counter, 0, 1)                                                  <L 118>
    var_97 = wp::atomic_add(var_reaction_counter, var_95, var_96);
}



void collide_triangles_vs_haptic_sphere_with_reaction_84f216bc_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc *_wp_args,
    wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::float32> var_inv_masses = _wp_args->inv_masses;
    wp::array_t<wp::int32> var_tri_indices = _wp_args->tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_center = _wp_args->sphere_center;
    wp::float32 var_sphere_radius = _wp_args->sphere_radius;
    wp::float32 var_sphere_center_scale = _wp_args->sphere_center_scale;
    wp::float32 var_restitution = _wp_args->restitution;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_cull_radius = _wp_args->cull_radius;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta_accumulator = _wp_args->delta_accumulator;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_reaction_accumulator = _wp_args->reaction_accumulator;
    wp::array_t<wp::int32> var_reaction_counter = _wp_args->reaction_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions = _wp_adj_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_velocities = _wp_adj_args->velocities;
    wp::array_t<wp::float32> adj_inv_masses = _wp_adj_args->inv_masses;
    wp::array_t<wp::int32> adj_tri_indices = _wp_adj_args->tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_center = _wp_adj_args->sphere_center;
    wp::float32 adj_sphere_radius = _wp_adj_args->sphere_radius;
    wp::float32 adj_sphere_center_scale = _wp_adj_args->sphere_center_scale;
    wp::float32 adj_restitution = _wp_adj_args->restitution;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::float32 adj_cull_radius = _wp_adj_args->cull_radius;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_delta_accumulator = _wp_adj_args->delta_accumulator;
    wp::array_t<wp::int32> adj_delta_counter = _wp_adj_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_reaction_accumulator = _wp_adj_args->reaction_accumulator;
    wp::array_t<wp::int32> adj_reaction_counter = _wp_adj_args->reaction_counter;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::shape_t* var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    wp::shape_t var_4;
    bool var_5;
    const wp::int32 var_6 = 0;
    wp::int32* var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32* var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    const wp::int32 var_14 = 2;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    const wp::int32 var_27 = 0;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::float32 var_31 = 0.0;
    bool var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::float32 var_35 = 3.0;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::float32 var_38;
    bool var_39;
    wp::float32* var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32* var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32* var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 0.0;
    bool var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::int32 var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    bool var_58;
    wp::float32 var_59;
    const wp::float32 var_60 = 1e-08;
    bool var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32>* var_63;
    wp::vec_t<3, wp::float32>* var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32>* var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::vec_t<3, wp::float32> var_70;
    const wp::float32 var_71 = 3.0;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32> var_78;
    wp::float32 var_79;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::vec_t<3, wp::float32> var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32> var_84;
    wp::vec_t<3, wp::float32> var_85;
    const wp::int32 var_86 = 1;
    wp::int32 var_87;
    const wp::int32 var_88 = 1;
    wp::int32 var_89;
    const wp::int32 var_90 = 1;
    wp::int32 var_91;
    const wp::int32 var_92 = 0;
    wp::vec_t<3, wp::float32> var_93;
    wp::vec_t<3, wp::float32> var_94;
    const wp::int32 var_95 = 0;
    const wp::int32 var_96 = 1;
    wp::int32 var_97;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::shape_t adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::shape_t adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::vec_t<3, wp::float32> adj_22 = {};
    wp::vec_t<3, wp::float32> adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::int32 adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::float32 adj_31 = {};
    bool adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::vec_t<3, wp::float32> adj_34 = {};
    wp::float32 adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::float32 adj_38 = {};
    bool adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    bool adj_52 = {};
    wp::vec_t<3, wp::float32> adj_53 = {};
    wp::vec_t<3, wp::float32> adj_54 = {};
    wp::int32 adj_55 = {};
    wp::vec_t<3, wp::float32> adj_56 = {};
    wp::float32 adj_57 = {};
    bool adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    bool adj_61 = {};
    wp::vec_t<3, wp::float32> adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::vec_t<3, wp::float32> adj_64 = {};
    wp::vec_t<3, wp::float32> adj_65 = {};
    wp::vec_t<3, wp::float32> adj_66 = {};
    wp::vec_t<3, wp::float32> adj_67 = {};
    wp::vec_t<3, wp::float32> adj_68 = {};
    wp::vec_t<3, wp::float32> adj_69 = {};
    wp::vec_t<3, wp::float32> adj_70 = {};
    wp::float32 adj_71 = {};
    wp::vec_t<3, wp::float32> adj_72 = {};
    wp::vec_t<3, wp::float32> adj_73 = {};
    wp::vec_t<3, wp::float32> adj_74 = {};
    wp::vec_t<3, wp::float32> adj_75 = {};
    wp::vec_t<3, wp::float32> adj_76 = {};
    wp::float32 adj_77 = {};
    wp::vec_t<3, wp::float32> adj_78 = {};
    wp::float32 adj_79 = {};
    wp::vec_t<3, wp::float32> adj_80 = {};
    wp::float32 adj_81 = {};
    wp::vec_t<3, wp::float32> adj_82 = {};
    wp::vec_t<3, wp::float32> adj_83 = {};
    wp::vec_t<3, wp::float32> adj_84 = {};
    wp::vec_t<3, wp::float32> adj_85 = {};
    wp::int32 adj_86 = {};
    wp::int32 adj_87 = {};
    wp::int32 adj_88 = {};
    wp::int32 adj_89 = {};
    wp::int32 adj_90 = {};
    wp::int32 adj_91 = {};
    wp::int32 adj_92 = {};
    wp::vec_t<3, wp::float32> adj_93 = {};
    wp::vec_t<3, wp::float32> adj_94 = {};
    wp::int32 adj_95 = {};
    wp::int32 adj_96 = {};
    wp::int32 adj_97 = {};
    //---------
    // forward
    // def collide_triangles_vs_haptic_sphere_with_reaction(                                  <L 44>
    // tid = wp.tid()                                                                         <L 60>
    var_0 = builtin_tid1d();
    // if tid >= tri_indices.shape[0]:                                                        <L 61>
    var_1 = &(var_tri_indices.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = (var_0 >= var_3);
    if (var_5) {
        // return                                                                             <L 62>
        goto label0;
    }
    // t1 = tri_indices[tid, 0]                                                               <L 64>
    var_7 = wp::address(var_tri_indices, var_0, var_6);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // t2 = tri_indices[tid, 1]                                                               <L 65>
    var_11 = wp::address(var_tri_indices, var_0, var_10);
    var_13 = wp::load(var_11);
    var_12 = wp::copy(var_13);
    // t3 = tri_indices[tid, 2]                                                               <L 66>
    var_15 = wp::address(var_tri_indices, var_0, var_14);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // p1 = positions[t1]                                                                     <L 68>
    var_18 = wp::address(var_positions, var_8);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // p2 = positions[t2]                                                                     <L 69>
    var_21 = wp::address(var_positions, var_12);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // p3 = positions[t3]                                                                     <L 70>
    var_24 = wp::address(var_positions, var_16);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // sphere_pos = sphere_center[0] * sphere_center_scale                                    <L 72>
    var_28 = wp::address(var_sphere_center, var_27);
    var_30 = wp::load(var_28);
    var_29 = wp::mul(var_30, var_sphere_center_scale);
    // if cull_radius > 0.0:                                                                  <L 74>
    var_32 = (var_cull_radius > var_31);
    if (var_32) {
        // centroid = (p1 + p2 + p3) / 3.0                                                    <L 75>
        var_33 = wp::add(var_19, var_22);
        var_34 = wp::add(var_33, var_25);
        var_36 = wp::div(var_34, var_35);
        // if wp.length(centroid - sphere_pos) > cull_radius:                                 <L 76>
        var_37 = wp::sub(var_36, var_29);
        var_38 = wp::length(var_37);
        var_39 = (var_38 > var_cull_radius);
        if (var_39) {
            // return                                                                         <L 77>
            goto label1;
        }
    }
    // w1 = inv_masses[t1]                                                                    <L 79>
    var_40 = wp::address(var_inv_masses, var_8);
    var_42 = wp::load(var_40);
    var_41 = wp::copy(var_42);
    // w2 = inv_masses[t2]                                                                    <L 80>
    var_43 = wp::address(var_inv_masses, var_12);
    var_45 = wp::load(var_43);
    var_44 = wp::copy(var_45);
    // w3 = inv_masses[t3]                                                                    <L 81>
    var_46 = wp::address(var_inv_masses, var_16);
    var_48 = wp::load(var_46);
    var_47 = wp::copy(var_48);
    // weight = w1 + w2 + w3                                                                  <L 82>
    var_49 = wp::add(var_41, var_44);
    var_50 = wp::add(var_49, var_47);
    // if weight <= 0.0:                                                                      <L 83>
    var_52 = (var_50 <= var_51);
    if (var_52) {
        // return                                                                             <L 84>
        goto label2;
    }
    // closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)         <L 86>
    triangle_closest_point_0(var_19, var_22, var_25, var_29, var_53, var_54, var_55);
    // to_sphere = closest_p - sphere_pos                                                     <L 87>
    var_56 = wp::sub(var_53, var_29);
    // dist = wp.length(to_sphere)                                                            <L 88>
    var_57 = wp::length(var_56);
    // if dist >= sphere_radius:                                                              <L 89>
    var_58 = (var_57 >= var_sphere_radius);
    if (var_58) {
        // return                                                                             <L 90>
        goto label3;
    }
    // penetration = sphere_radius - dist                                                     <L 92>
    var_59 = wp::sub(var_sphere_radius, var_57);
    // if dist > 1.0e-8:                                                                      <L 93>
    var_61 = (var_57 > var_60);
    if (var_61) {
        // correction_dir = to_sphere / dist                                                  <L 94>
        var_62 = wp::div(var_56, var_57);
    }
    if (!var_61) {
        // correction_dir = _double_sided_reaction_dir(                                       <L 96>
        // p1,                                                                                <L 97>
        // p2,                                                                                <L 98>
        // p3,                                                                                <L 99>
        // sphere_pos,                                                                        <L 100>
        // -((velocities[t1] + velocities[t2] + velocities[t3]) / 3.0),                       <L 101>
        var_63 = wp::address(var_velocities, var_8);
        var_64 = wp::address(var_velocities, var_12);
        var_66 = wp::load(var_63);
        var_67 = wp::load(var_64);
        var_65 = wp::add(var_66, var_67);
        var_68 = wp::address(var_velocities, var_16);
        var_70 = wp::load(var_68);
        var_69 = wp::add(var_65, var_70);
        var_72 = wp::div(var_69, var_71);
        var_73 = wp::neg(var_72);
        var_74 = _double_sided_reaction_dir_0(var_19, var_22, var_25, var_29, var_73);
    }
    var_75 = wp::where(var_61, var_62, var_74);
    // total_correction = correction_dir * penetration                                        <L 104>
    var_76 = wp::mul(var_75, var_59);
    // d1 = total_correction * (w1 / weight)                                                  <L 105>
    var_77 = wp::div(var_41, var_50);
    var_78 = wp::mul(var_76, var_77);
    // d2 = total_correction * (w2 / weight)                                                  <L 106>
    var_79 = wp::div(var_44, var_50);
    var_80 = wp::mul(var_76, var_79);
    // d3 = total_correction * (w3 / weight)                                                  <L 107>
    var_81 = wp::div(var_47, var_50);
    var_82 = wp::mul(var_76, var_81);
    // wp.atomic_add(delta_accumulator, t1, d1)                                               <L 109>
    // var_83 = wp::atomic_add(var_delta_accumulator, var_8, var_78);
    // wp.atomic_add(delta_accumulator, t2, d2)                                               <L 110>
    // var_84 = wp::atomic_add(var_delta_accumulator, var_12, var_80);
    // wp.atomic_add(delta_accumulator, t3, d3)                                               <L 111>
    // var_85 = wp::atomic_add(var_delta_accumulator, var_16, var_82);
    // wp.atomic_add(delta_counter, t1, 1)                                                    <L 112>
    // var_87 = wp::atomic_add(var_delta_counter, var_8, var_86);
    // wp.atomic_add(delta_counter, t2, 1)                                                    <L 113>
    // var_89 = wp::atomic_add(var_delta_counter, var_12, var_88);
    // wp.atomic_add(delta_counter, t3, 1)                                                    <L 114>
    // var_91 = wp::atomic_add(var_delta_counter, var_16, var_90);
    // wp.atomic_add(reaction_accumulator, 0, -total_correction)                              <L 117>
    var_93 = wp::neg(var_76);
    // var_94 = wp::atomic_add(var_reaction_accumulator, var_92, var_93);
    // wp.atomic_add(reaction_counter, 0, 1)                                                  <L 118>
    // var_97 = wp::atomic_add(var_reaction_counter, var_95, var_96);
    //---------
    // reverse
    wp::adj_atomic_add(var_reaction_counter, var_95, var_96, adj_reaction_counter, adj_95, adj_96, adj_97);
    // adj: wp.atomic_add(reaction_counter, 0, 1)                                             <L 118>
    wp::adj_atomic_add(var_reaction_accumulator, var_92, var_93, adj_reaction_accumulator, adj_92, adj_93, adj_94);
    wp::adj_neg(var_76, adj_76, adj_93);
    // adj: wp.atomic_add(reaction_accumulator, 0, -total_correction)                         <L 117>
    wp::adj_atomic_add(var_delta_counter, var_16, var_90, adj_delta_counter, adj_16, adj_90, adj_91);
    // adj: wp.atomic_add(delta_counter, t3, 1)                                               <L 114>
    wp::adj_atomic_add(var_delta_counter, var_12, var_88, adj_delta_counter, adj_12, adj_88, adj_89);
    // adj: wp.atomic_add(delta_counter, t2, 1)                                               <L 113>
    wp::adj_atomic_add(var_delta_counter, var_8, var_86, adj_delta_counter, adj_8, adj_86, adj_87);
    // adj: wp.atomic_add(delta_counter, t1, 1)                                               <L 112>
    wp::adj_atomic_add(var_delta_accumulator, var_16, var_82, adj_delta_accumulator, adj_16, adj_82, adj_85);
    // adj: wp.atomic_add(delta_accumulator, t3, d3)                                          <L 111>
    wp::adj_atomic_add(var_delta_accumulator, var_12, var_80, adj_delta_accumulator, adj_12, adj_80, adj_84);
    // adj: wp.atomic_add(delta_accumulator, t2, d2)                                          <L 110>
    wp::adj_atomic_add(var_delta_accumulator, var_8, var_78, adj_delta_accumulator, adj_8, adj_78, adj_83);
    // adj: wp.atomic_add(delta_accumulator, t1, d1)                                          <L 109>
    wp::adj_mul(var_76, var_81, adj_76, adj_81, adj_82);
    wp::adj_div(var_47, var_50, var_81, adj_47, adj_50, adj_81);
    // adj: d3 = total_correction * (w3 / weight)                                             <L 107>
    wp::adj_mul(var_76, var_79, adj_76, adj_79, adj_80);
    wp::adj_div(var_44, var_50, var_79, adj_44, adj_50, adj_79);
    // adj: d2 = total_correction * (w2 / weight)                                             <L 106>
    wp::adj_mul(var_76, var_77, adj_76, adj_77, adj_78);
    wp::adj_div(var_41, var_50, var_77, adj_41, adj_50, adj_77);
    // adj: d1 = total_correction * (w1 / weight)                                             <L 105>
    wp::adj_mul(var_75, var_59, adj_75, adj_59, adj_76);
    // adj: total_correction = correction_dir * penetration                                   <L 104>
    wp::adj_where(var_61, var_62, var_74, adj_61, adj_62, adj_74, adj_75);
    if (!var_61) {
        adj__double_sided_reaction_dir_0(var_19, var_22, var_25, var_29, var_73, adj_19, adj_22, adj_25, adj_29, adj_73, adj_74);
        wp::adj_neg(var_72, adj_72, adj_73);
        wp::adj_div(var_69, var_71, adj_69, adj_71, adj_72);
        wp::adj_add(var_65, var_70, adj_65, adj_68, adj_69);
        wp::adj_address(var_velocities, var_16, adj_velocities, adj_16, adj_68);
        wp::adj_add(var_66, var_67, adj_63, adj_64, adj_65);
        wp::adj_address(var_velocities, var_12, adj_velocities, adj_12, adj_64);
        wp::adj_address(var_velocities, var_8, adj_velocities, adj_8, adj_63);
        // adj: -((velocities[t1] + velocities[t2] + velocities[t3]) / 3.0),                  <L 101>
        // adj: sphere_pos,                                                                   <L 100>
        // adj: p3,                                                                           <L 99>
        // adj: p2,                                                                           <L 98>
        // adj: p1,                                                                           <L 97>
        // adj: correction_dir = _double_sided_reaction_dir(                                  <L 96>
    }
    if (var_61) {
        wp::adj_div(var_56, var_57, adj_56, adj_57, adj_62);
        // adj: correction_dir = to_sphere / dist                                             <L 94>
    }
    // adj: if dist > 1.0e-8:                                                                 <L 93>
    wp::adj_sub(var_sphere_radius, var_57, adj_sphere_radius, adj_57, adj_59);
    // adj: penetration = sphere_radius - dist                                                <L 92>
    if (var_58) {
        label3:;
        // adj: return                                                                        <L 90>
    }
    // adj: if dist >= sphere_radius:                                                         <L 89>
    wp::adj_length(var_56, var_57, adj_56, adj_57);
    // adj: dist = wp.length(to_sphere)                                                       <L 88>
    wp::adj_sub(var_53, var_29, adj_53, adj_29, adj_56);
    // adj: to_sphere = closest_p - sphere_pos                                                <L 87>
    adj_triangle_closest_point_0(var_19, var_22, var_25, var_29, var_53, var_54, var_55, adj_19, adj_22, adj_25, adj_29, adj_53, adj_54, adj_55);
    // adj: closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)    <L 86>
    if (var_52) {
        label2:;
        // adj: return                                                                        <L 84>
    }
    // adj: if weight <= 0.0:                                                                 <L 83>
    wp::adj_add(var_49, var_47, adj_49, adj_47, adj_50);
    wp::adj_add(var_41, var_44, adj_41, adj_44, adj_49);
    // adj: weight = w1 + w2 + w3                                                             <L 82>
    wp::adj_copy(var_48, adj_46, adj_47);
    wp::adj_address(var_inv_masses, var_16, adj_inv_masses, adj_16, adj_46);
    // adj: w3 = inv_masses[t3]                                                               <L 81>
    wp::adj_copy(var_45, adj_43, adj_44);
    wp::adj_address(var_inv_masses, var_12, adj_inv_masses, adj_12, adj_43);
    // adj: w2 = inv_masses[t2]                                                               <L 80>
    wp::adj_copy(var_42, adj_40, adj_41);
    wp::adj_address(var_inv_masses, var_8, adj_inv_masses, adj_8, adj_40);
    // adj: w1 = inv_masses[t1]                                                               <L 79>
    if (var_32) {
        if (var_39) {
            label1:;
            // adj: return                                                                    <L 77>
        }
        wp::adj_length(var_37, var_38, adj_37, adj_38);
        wp::adj_sub(var_36, var_29, adj_36, adj_29, adj_37);
        // adj: if wp.length(centroid - sphere_pos) > cull_radius:                            <L 76>
        wp::adj_div(var_34, var_35, adj_34, adj_35, adj_36);
        wp::adj_add(var_33, var_25, adj_33, adj_25, adj_34);
        wp::adj_add(var_19, var_22, adj_19, adj_22, adj_33);
        // adj: centroid = (p1 + p2 + p3) / 3.0                                               <L 75>
    }
    // adj: if cull_radius > 0.0:                                                             <L 74>
    wp::adj_mul(var_30, var_sphere_center_scale, adj_28, adj_sphere_center_scale, adj_29);
    wp::adj_address(var_sphere_center, var_27, adj_sphere_center, adj_27, adj_28);
    // adj: sphere_pos = sphere_center[0] * sphere_center_scale                               <L 72>
    wp::adj_copy(var_26, adj_24, adj_25);
    wp::adj_address(var_positions, var_16, adj_positions, adj_16, adj_24);
    // adj: p3 = positions[t3]                                                                <L 70>
    wp::adj_copy(var_23, adj_21, adj_22);
    wp::adj_address(var_positions, var_12, adj_positions, adj_12, adj_21);
    // adj: p2 = positions[t2]                                                                <L 69>
    wp::adj_copy(var_20, adj_18, adj_19);
    wp::adj_address(var_positions, var_8, adj_positions, adj_8, adj_18);
    // adj: p1 = positions[t1]                                                                <L 68>
    wp::adj_copy(var_17, adj_15, adj_16);
    wp::adj_address(var_tri_indices, var_0, var_14, adj_tri_indices, adj_0, adj_14, adj_15);
    // adj: t3 = tri_indices[tid, 2]                                                          <L 66>
    wp::adj_copy(var_13, adj_11, adj_12);
    wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
    // adj: t2 = tri_indices[tid, 1]                                                          <L 65>
    wp::adj_copy(var_9, adj_7, adj_8);
    wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
    // adj: t1 = tri_indices[tid, 0]                                                          <L 64>
    if (var_5) {
        label0:;
        // adj: return                                                                        <L 62>
    }
    wp::adj_extract(var_4, var_2, adj_1, adj_2, adj_3);
    adj_tri_indices.shape = adj_1;
    // adj: if tid >= tri_indices.shape[0]:                                                   <L 61>
    // adj: tid = wp.tid()                                                                    <L 60>
    // adj: def collide_triangles_vs_haptic_sphere_with_reaction(                             <L 44>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void collide_triangles_vs_haptic_sphere_with_reaction_84f216bc_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        collide_triangles_vs_haptic_sphere_with_reaction_84f216bc_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void collide_triangles_vs_haptic_sphere_with_reaction_84f216bc_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc *_wp_args,
    wp_args_collide_triangles_vs_haptic_sphere_with_reaction_84f216bc *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        collide_triangles_vs_haptic_sphere_with_reaction_84f216bc_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_apply_surface_vertex_truncation_f8f269cd {
    wp::array_t<wp::int32> surface_vertex_ids;
    wp::array_t<wp::float32> truncation_t;
    wp::array_t<wp::vec_t<3, wp::float32>> displacements;
};


void apply_surface_vertex_truncation_f8f269cd_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_surface_vertex_truncation_f8f269cd *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_surface_vertex_ids = _wp_args->surface_vertex_ids;
    wp::array_t<wp::float32> var_truncation_t = _wp_args->truncation_t;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacements = _wp_args->displacements;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::shape_t* var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    wp::shape_t var_4;
    bool var_5;
    wp::int32* var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::float32* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    //---------
    // forward
    // def apply_surface_vertex_truncation(                                                   <L 282>
    // tid = wp.tid()                                                                         <L 287>
    var_0 = builtin_tid1d();
    // if tid >= surface_vertex_ids.shape[0]:                                                 <L 288>
    var_1 = &(var_surface_vertex_ids.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = (var_0 >= var_3);
    if (var_5) {
        // return                                                                             <L 289>
        return;
    }
    // vertex_id = surface_vertex_ids[tid]                                                    <L 291>
    var_6 = wp::address(var_surface_vertex_ids, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]          <L 292>
    var_9 = wp::address(var_displacements, var_7);
    var_10 = wp::address(var_truncation_t, var_7);
    var_12 = wp::load(var_9);
    var_13 = wp::load(var_10);
    var_11 = wp::mul(var_12, var_13);
    wp::array_store(var_displacements, var_7, var_11);
}



void apply_surface_vertex_truncation_f8f269cd_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_surface_vertex_truncation_f8f269cd *_wp_args,
    wp_args_apply_surface_vertex_truncation_f8f269cd *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_surface_vertex_ids = _wp_args->surface_vertex_ids;
    wp::array_t<wp::float32> var_truncation_t = _wp_args->truncation_t;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacements = _wp_args->displacements;
    wp::array_t<wp::int32> adj_surface_vertex_ids = _wp_adj_args->surface_vertex_ids;
    wp::array_t<wp::float32> adj_truncation_t = _wp_adj_args->truncation_t;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_displacements = _wp_adj_args->displacements;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::shape_t* var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    wp::shape_t var_4;
    bool var_5;
    wp::int32* var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::float32* var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::shape_t adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::shape_t adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::float32 adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::float32 adj_13 = {};
    //---------
    // forward
    // def apply_surface_vertex_truncation(                                                   <L 282>
    // tid = wp.tid()                                                                         <L 287>
    var_0 = builtin_tid1d();
    // if tid >= surface_vertex_ids.shape[0]:                                                 <L 288>
    var_1 = &(var_surface_vertex_ids.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = (var_0 >= var_3);
    if (var_5) {
        // return                                                                             <L 289>
        goto label0;
    }
    // vertex_id = surface_vertex_ids[tid]                                                    <L 291>
    var_6 = wp::address(var_surface_vertex_ids, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]          <L 292>
    var_9 = wp::address(var_displacements, var_7);
    var_10 = wp::address(var_truncation_t, var_7);
    var_12 = wp::load(var_9);
    var_13 = wp::load(var_10);
    var_11 = wp::mul(var_12, var_13);
    // wp::array_store(var_displacements, var_7, var_11);
    //---------
    // reverse
    wp::adj_array_store(var_displacements, var_7, var_11, adj_displacements, adj_7, adj_11);
    wp::adj_mul(var_12, var_13, adj_9, adj_10, adj_11);
    wp::adj_address(var_truncation_t, var_7, adj_truncation_t, adj_7, adj_10);
    wp::adj_address(var_displacements, var_7, adj_displacements, adj_7, adj_9);
    // adj: displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]     <L 292>
    wp::adj_copy(var_8, adj_6, adj_7);
    wp::adj_address(var_surface_vertex_ids, var_0, adj_surface_vertex_ids, adj_0, adj_6);
    // adj: vertex_id = surface_vertex_ids[tid]                                               <L 291>
    if (var_5) {
        label0:;
        // adj: return                                                                        <L 289>
    }
    wp::adj_extract(var_4, var_2, adj_1, adj_2, adj_3);
    adj_surface_vertex_ids.shape = adj_1;
    // adj: if tid >= surface_vertex_ids.shape[0]:                                            <L 288>
    // adj: tid = wp.tid()                                                                    <L 287>
    // adj: def apply_surface_vertex_truncation(                                              <L 282>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void apply_surface_vertex_truncation_f8f269cd_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_apply_surface_vertex_truncation_f8f269cd *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_surface_vertex_truncation_f8f269cd_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void apply_surface_vertex_truncation_f8f269cd_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_apply_surface_vertex_truncation_f8f269cd *_wp_args,
    wp_args_apply_surface_vertex_truncation_f8f269cd *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_surface_vertex_truncation_f8f269cd_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_compute_proxy_vertex_truncation_factors_d7c454ca {
    wp::array_t<wp::int32> particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> base_positions;
    wp::array_t<wp::int32> surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_center;
    wp::float32 sphere_radius;
    wp::int32 motion_samples;
    wp::float32 radius_margin;
    wp::float32 safety;
    wp::array_t<wp::float32> truncation_t_out;
};


void compute_proxy_vertex_truncation_factors_d7c454ca_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_compute_proxy_vertex_truncation_factors_d7c454ca *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> var_base_positions = _wp_args->base_positions;
    wp::array_t<wp::int32> var_surface_vertex_ids = _wp_args->surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacement_in = _wp_args->displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_center_prev = _wp_args->sphere_center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_center = _wp_args->sphere_center;
    wp::float32 var_sphere_radius = _wp_args->sphere_radius;
    wp::int32 var_motion_samples = _wp_args->motion_samples;
    wp::float32 var_radius_margin = _wp_args->radius_margin;
    wp::float32 var_safety = _wp_args->safety;
    wp::array_t<wp::float32> var_truncation_t_out = _wp_args->truncation_t_out;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::int32 var_6;
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32* var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32* var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::float32 var_24;
    const wp::float32 var_25 = 1e-12;
    bool var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.0;
    bool var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::vec_t<3, wp::float32>* var_36;
    const wp::int32 var_37 = 0;
    wp::vec_t<3, wp::float32>* var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 1e-12;
    bool var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 1e-12;
    bool var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    const wp::float32 var_58 = 1e-12;
    bool var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::float32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::float32 var_66;
    const wp::float32 var_67 = 0.0;
    bool var_68;
    const wp::float32 var_69 = 1.0;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 1e-08;
    bool var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::float32 var_78 = 0.001;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.0;
    const wp::float32 var_82 = 1.0;
    wp::float32 var_83;
    wp::float32 var_84;
    bool var_85;
    const wp::float32 var_86 = 0.0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    //---------
    // forward
    // def compute_proxy_vertex_truncation_factors(                                           <L 213>
    // tid = wp.tid()                                                                         <L 226>
    var_0 = builtin_tid1d();
    // if tid >= surface_vertex_ids.shape[0] * motion_samples or motion_samples <= 0:         <L 227>
    var_2 = &(var_surface_vertex_ids.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    var_6 = wp::mul(var_4, var_motion_samples);
    var_7 = (var_0 >= var_6);
    var_1 = var_7;
    if (!var_1) {
        var_9 = (var_motion_samples <= var_8);
        var_1 = var_1 || var_9;
    }
    if (var_1) {
        // return                                                                             <L 228>
        return;
    }
    // vertex_idx = tid // motion_samples                                                     <L 230>
    var_10 = wp::floordiv(var_0, var_motion_samples);
    // sample_idx = tid % motion_samples                                                      <L 231>
    var_11 = wp::mod(var_0, var_motion_samples);
    // vertex_id = surface_vertex_ids[vertex_idx]                                             <L 232>
    var_12 = wp::address(var_surface_vertex_ids, var_10);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:                            <L 233>
    var_15 = wp::address(var_particle_flags, var_13);
    var_18 = wp::load(var_15);
    var_17 = wp::bit_and(var_18, var_16);
    var_20 = (var_17 == var_19);
    if (var_20) {
        // return                                                                             <L 234>
        return;
    }
    // displacement = displacement_in[vertex_id]                                              <L 236>
    var_21 = wp::address(var_displacement_in, var_13);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // if wp.length_sq(displacement) <= 1.0e-12:                                              <L 237>
    var_24 = wp::length_sq(var_22);
    var_26 = (var_24 <= var_25);
    if (var_26) {
        // return                                                                             <L 238>
        return;
    }
    // radius = sphere_radius + radius_margin                                                 <L 240>
    var_27 = wp::add(var_sphere_radius, var_radius_margin);
    // if radius <= 0.0:                                                                      <L 241>
    var_29 = (var_27 <= var_28);
    if (var_29) {
        // return                                                                             <L 242>
        return;
    }
    // sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)                <L 244>
    var_31 = wp::add(var_11, var_30);
    var_32 = wp::float32(var_31);
    var_33 = wp::float32(var_motion_samples);
    var_34 = wp::div(var_32, var_33);
    // center = wp.lerp(sphere_center_prev[0], sphere_center[0], sample_factor)               <L 245>
    var_36 = wp::address(var_sphere_center_prev, var_35);
    var_38 = wp::address(var_sphere_center, var_37);
    var_40 = wp::load(var_36);
    var_41 = wp::load(var_38);
    var_39 = wp::lerp(var_40, var_41, var_34);
    // x0 = base_positions[vertex_id]                                                         <L 247>
    var_42 = wp::address(var_base_positions, var_13);
    var_44 = wp::load(var_42);
    var_43 = wp::copy(var_44);
    // x1 = x0 + displacement                                                                 <L 248>
    var_45 = wp::add(var_43, var_22);
    // normal = x1 - center                                                                   <L 250>
    var_46 = wp::sub(var_45, var_39);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 251>
    var_47 = wp::length_sq(var_46);
    var_49 = (var_47 <= var_48);
    if (var_49) {
        // normal = x0 - center                                                               <L 252>
        var_50 = wp::sub(var_43, var_39);
    }
    var_51 = wp::where(var_49, var_50, var_46);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 253>
    var_52 = wp::length_sq(var_51);
    var_54 = (var_52 <= var_53);
    if (var_54) {
        // normal = -displacement                                                             <L 254>
        var_55 = wp::neg(var_22);
    }
    var_56 = wp::where(var_54, var_55, var_51);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 255>
    var_57 = wp::length_sq(var_56);
    var_59 = (var_57 <= var_58);
    if (var_59) {
        // return                                                                             <L 256>
        return;
    }
    // n = wp.normalize(normal)                                                               <L 258>
    var_60 = wp::normalize(var_56);
    // plane_point = center + n * radius                                                      <L 259>
    var_61 = wp::mul(var_60, var_27);
    var_62 = wp::add(var_39, var_61);
    // s0 = wp.dot(n, x0 - plane_point)                                                       <L 261>
    var_63 = wp::sub(var_43, var_62);
    var_64 = wp::dot(var_60, var_63);
    // s1 = wp.dot(n, x1 - plane_point)                                                       <L 262>
    var_65 = wp::sub(var_45, var_62);
    var_66 = wp::dot(var_60, var_65);
    // if s1 >= 0.0:                                                                          <L 263>
    var_68 = (var_66 >= var_67);
    if (var_68) {
        // return                                                                             <L 264>
        return;
    }
    // t = wp.float32(1.0)                                                                    <L 266>
    var_70 = wp::float32(var_69);
    // if s0 > 0.0:                                                                           <L 267>
    var_72 = (var_64 > var_71);
    if (var_72) {
        // denom = s0 - s1                                                                    <L 268>
        var_73 = wp::sub(var_64, var_66);
        // if denom <= 1.0e-8:                                                                <L 269>
        var_75 = (var_73 <= var_74);
        if (var_75) {
            // return                                                                         <L 270>
            return;
        }
        // crossing_t = s0 / denom                                                            <L 271>
        var_76 = wp::div(var_64, var_73);
        // t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)           <L 272>
        var_77 = wp::mul(var_76, var_safety);
        var_79 = wp::sub(var_76, var_78);
        var_80 = wp::min(var_77, var_79);
        var_83 = wp::clamp(var_80, var_81, var_82);
    }
    var_84 = wp::where(var_72, var_83, var_70);
    if (!var_72) {
        // elif s1 < s0:                                                                      <L 273>
        var_85 = (var_66 < var_64);
        if (var_85) {
            // t = 0.0                                                                        <L 274>
        }
        var_87 = wp::where(var_85, var_86, var_84);
        if (!var_85) {
            // return                                                                         <L 276>
            return;
        }
    }
    var_88 = wp::where(var_72, var_84, var_87);
    // wp.atomic_min(truncation_t_out, vertex_id, t)                                          <L 278>
    var_89 = wp::atomic_min(var_truncation_t_out, var_13, var_88);
}



void compute_proxy_vertex_truncation_factors_d7c454ca_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_compute_proxy_vertex_truncation_factors_d7c454ca *_wp_args,
    wp_args_compute_proxy_vertex_truncation_factors_d7c454ca *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> var_base_positions = _wp_args->base_positions;
    wp::array_t<wp::int32> var_surface_vertex_ids = _wp_args->surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacement_in = _wp_args->displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_center_prev = _wp_args->sphere_center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_center = _wp_args->sphere_center;
    wp::float32 var_sphere_radius = _wp_args->sphere_radius;
    wp::int32 var_motion_samples = _wp_args->motion_samples;
    wp::float32 var_radius_margin = _wp_args->radius_margin;
    wp::float32 var_safety = _wp_args->safety;
    wp::array_t<wp::float32> var_truncation_t_out = _wp_args->truncation_t_out;
    wp::array_t<wp::int32> adj_particle_flags = _wp_adj_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_base_positions = _wp_adj_args->base_positions;
    wp::array_t<wp::int32> adj_surface_vertex_ids = _wp_adj_args->surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_displacement_in = _wp_adj_args->displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_center_prev = _wp_adj_args->sphere_center_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_center = _wp_adj_args->sphere_center;
    wp::float32 adj_sphere_radius = _wp_adj_args->sphere_radius;
    wp::int32 adj_motion_samples = _wp_adj_args->motion_samples;
    wp::float32 adj_radius_margin = _wp_adj_args->radius_margin;
    wp::float32 adj_safety = _wp_adj_args->safety;
    wp::array_t<wp::float32> adj_truncation_t_out = _wp_adj_args->truncation_t_out;
    //---------
    // primal vars
    wp::int32 var_0;
    bool var_1;
    wp::shape_t* var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::shape_t var_5;
    wp::int32 var_6;
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32* var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32* var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::int32 var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::vec_t<3, wp::float32> var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::float32 var_24;
    const wp::float32 var_25 = 1e-12;
    bool var_26;
    wp::float32 var_27;
    const wp::float32 var_28 = 0.0;
    bool var_29;
    const wp::int32 var_30 = 1;
    wp::int32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::vec_t<3, wp::float32>* var_36;
    const wp::int32 var_37 = 0;
    wp::vec_t<3, wp::float32>* var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32>* var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::float32 var_47;
    const wp::float32 var_48 = 1e-12;
    bool var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 1e-12;
    bool var_54;
    wp::vec_t<3, wp::float32> var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::float32 var_57;
    const wp::float32 var_58 = 1e-12;
    bool var_59;
    wp::vec_t<3, wp::float32> var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::float32 var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::float32 var_66;
    const wp::float32 var_67 = 0.0;
    bool var_68;
    const wp::float32 var_69 = 1.0;
    wp::float32 var_70;
    const wp::float32 var_71 = 0.0;
    bool var_72;
    wp::float32 var_73;
    const wp::float32 var_74 = 1e-08;
    bool var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    const wp::float32 var_78 = 0.001;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 0.0;
    const wp::float32 var_82 = 1.0;
    wp::float32 var_83;
    wp::float32 var_84;
    bool var_85;
    const wp::float32 var_86 = 0.0;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::shape_t adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::shape_t adj_5 = {};
    wp::int32 adj_6 = {};
    bool adj_7 = {};
    wp::int32 adj_8 = {};
    bool adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    wp::int32 adj_19 = {};
    bool adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::vec_t<3, wp::float32> adj_22 = {};
    wp::vec_t<3, wp::float32> adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    bool adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    bool adj_29 = {};
    wp::int32 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::int32 adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::vec_t<3, wp::float32> adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    wp::vec_t<3, wp::float32> adj_43 = {};
    wp::vec_t<3, wp::float32> adj_44 = {};
    wp::vec_t<3, wp::float32> adj_45 = {};
    wp::vec_t<3, wp::float32> adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    bool adj_49 = {};
    wp::vec_t<3, wp::float32> adj_50 = {};
    wp::vec_t<3, wp::float32> adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    bool adj_54 = {};
    wp::vec_t<3, wp::float32> adj_55 = {};
    wp::vec_t<3, wp::float32> adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    bool adj_59 = {};
    wp::vec_t<3, wp::float32> adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::vec_t<3, wp::float32> adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::float32 adj_64 = {};
    wp::vec_t<3, wp::float32> adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    bool adj_68 = {};
    wp::float32 adj_69 = {};
    wp::float32 adj_70 = {};
    wp::float32 adj_71 = {};
    bool adj_72 = {};
    wp::float32 adj_73 = {};
    wp::float32 adj_74 = {};
    bool adj_75 = {};
    wp::float32 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::float32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    bool adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float32 adj_87 = {};
    wp::float32 adj_88 = {};
    wp::float32 adj_89 = {};
    //---------
    // forward
    // def compute_proxy_vertex_truncation_factors(                                           <L 213>
    // tid = wp.tid()                                                                         <L 226>
    var_0 = builtin_tid1d();
    // if tid >= surface_vertex_ids.shape[0] * motion_samples or motion_samples <= 0:         <L 227>
    var_2 = &(var_surface_vertex_ids.shape);
    var_5 = wp::load(var_2);
    var_4 = wp::extract(var_5, var_3);
    var_6 = wp::mul(var_4, var_motion_samples);
    var_7 = (var_0 >= var_6);
    var_1 = var_7;
    if (!var_1) {
        var_9 = (var_motion_samples <= var_8);
        var_1 = var_1 || var_9;
    }
    if (var_1) {
        // return                                                                             <L 228>
        goto label0;
    }
    // vertex_idx = tid // motion_samples                                                     <L 230>
    var_10 = wp::floordiv(var_0, var_motion_samples);
    // sample_idx = tid % motion_samples                                                      <L 231>
    var_11 = wp::mod(var_0, var_motion_samples);
    // vertex_id = surface_vertex_ids[vertex_idx]                                             <L 232>
    var_12 = wp::address(var_surface_vertex_ids, var_10);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:                            <L 233>
    var_15 = wp::address(var_particle_flags, var_13);
    var_18 = wp::load(var_15);
    var_17 = wp::bit_and(var_18, var_16);
    var_20 = (var_17 == var_19);
    if (var_20) {
        // return                                                                             <L 234>
        goto label1;
    }
    // displacement = displacement_in[vertex_id]                                              <L 236>
    var_21 = wp::address(var_displacement_in, var_13);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // if wp.length_sq(displacement) <= 1.0e-12:                                              <L 237>
    var_24 = wp::length_sq(var_22);
    var_26 = (var_24 <= var_25);
    if (var_26) {
        // return                                                                             <L 238>
        goto label2;
    }
    // radius = sphere_radius + radius_margin                                                 <L 240>
    var_27 = wp::add(var_sphere_radius, var_radius_margin);
    // if radius <= 0.0:                                                                      <L 241>
    var_29 = (var_27 <= var_28);
    if (var_29) {
        // return                                                                             <L 242>
        goto label3;
    }
    // sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)                <L 244>
    var_31 = wp::add(var_11, var_30);
    var_32 = wp::float32(var_31);
    var_33 = wp::float32(var_motion_samples);
    var_34 = wp::div(var_32, var_33);
    // center = wp.lerp(sphere_center_prev[0], sphere_center[0], sample_factor)               <L 245>
    var_36 = wp::address(var_sphere_center_prev, var_35);
    var_38 = wp::address(var_sphere_center, var_37);
    var_40 = wp::load(var_36);
    var_41 = wp::load(var_38);
    var_39 = wp::lerp(var_40, var_41, var_34);
    // x0 = base_positions[vertex_id]                                                         <L 247>
    var_42 = wp::address(var_base_positions, var_13);
    var_44 = wp::load(var_42);
    var_43 = wp::copy(var_44);
    // x1 = x0 + displacement                                                                 <L 248>
    var_45 = wp::add(var_43, var_22);
    // normal = x1 - center                                                                   <L 250>
    var_46 = wp::sub(var_45, var_39);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 251>
    var_47 = wp::length_sq(var_46);
    var_49 = (var_47 <= var_48);
    if (var_49) {
        // normal = x0 - center                                                               <L 252>
        var_50 = wp::sub(var_43, var_39);
    }
    var_51 = wp::where(var_49, var_50, var_46);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 253>
    var_52 = wp::length_sq(var_51);
    var_54 = (var_52 <= var_53);
    if (var_54) {
        // normal = -displacement                                                             <L 254>
        var_55 = wp::neg(var_22);
    }
    var_56 = wp::where(var_54, var_55, var_51);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 255>
    var_57 = wp::length_sq(var_56);
    var_59 = (var_57 <= var_58);
    if (var_59) {
        // return                                                                             <L 256>
        goto label4;
    }
    // n = wp.normalize(normal)                                                               <L 258>
    var_60 = wp::normalize(var_56);
    // plane_point = center + n * radius                                                      <L 259>
    var_61 = wp::mul(var_60, var_27);
    var_62 = wp::add(var_39, var_61);
    // s0 = wp.dot(n, x0 - plane_point)                                                       <L 261>
    var_63 = wp::sub(var_43, var_62);
    var_64 = wp::dot(var_60, var_63);
    // s1 = wp.dot(n, x1 - plane_point)                                                       <L 262>
    var_65 = wp::sub(var_45, var_62);
    var_66 = wp::dot(var_60, var_65);
    // if s1 >= 0.0:                                                                          <L 263>
    var_68 = (var_66 >= var_67);
    if (var_68) {
        // return                                                                             <L 264>
        goto label5;
    }
    // t = wp.float32(1.0)                                                                    <L 266>
    var_70 = wp::float32(var_69);
    // if s0 > 0.0:                                                                           <L 267>
    var_72 = (var_64 > var_71);
    if (var_72) {
        // denom = s0 - s1                                                                    <L 268>
        var_73 = wp::sub(var_64, var_66);
        // if denom <= 1.0e-8:                                                                <L 269>
        var_75 = (var_73 <= var_74);
        if (var_75) {
            // return                                                                         <L 270>
            goto label6;
        }
        // crossing_t = s0 / denom                                                            <L 271>
        var_76 = wp::div(var_64, var_73);
        // t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)           <L 272>
        var_77 = wp::mul(var_76, var_safety);
        var_79 = wp::sub(var_76, var_78);
        var_80 = wp::min(var_77, var_79);
        var_83 = wp::clamp(var_80, var_81, var_82);
    }
    var_84 = wp::where(var_72, var_83, var_70);
    if (!var_72) {
        // elif s1 < s0:                                                                      <L 273>
        var_85 = (var_66 < var_64);
        if (var_85) {
            // t = 0.0                                                                        <L 274>
        }
        var_87 = wp::where(var_85, var_86, var_84);
        if (!var_85) {
            // return                                                                         <L 276>
            goto label7;
        }
    }
    var_88 = wp::where(var_72, var_84, var_87);
    // wp.atomic_min(truncation_t_out, vertex_id, t)                                          <L 278>
    // var_89 = wp::atomic_min(var_truncation_t_out, var_13, var_88);
    //---------
    // reverse
    wp::adj_atomic_min(var_truncation_t_out, var_13, var_88, adj_truncation_t_out, adj_13, adj_88, adj_89);
    // adj: wp.atomic_min(truncation_t_out, vertex_id, t)                                     <L 278>
    wp::adj_where(var_72, var_84, var_87, adj_72, adj_84, adj_87, adj_88);
    if (!var_72) {
        if (!var_85) {
            label7:;
            // adj: return                                                                    <L 276>
        }
        wp::adj_where(var_85, var_86, var_84, adj_85, adj_86, adj_84, adj_87);
        if (var_85) {
            // adj: t = 0.0                                                                   <L 274>
        }
        // adj: elif s1 < s0:                                                                 <L 273>
    }
    wp::adj_where(var_72, var_83, var_70, adj_72, adj_83, adj_70, adj_84);
    if (var_72) {
        wp::adj_clamp(var_80, var_81, var_82, adj_80, adj_81, adj_82, adj_83);
        wp::adj_min(var_77, var_79, adj_77, adj_79, adj_80);
        wp::adj_sub(var_76, var_78, adj_76, adj_78, adj_79);
        wp::adj_mul(var_76, var_safety, adj_76, adj_safety, adj_77);
        // adj: t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)      <L 272>
        wp::adj_div(var_64, var_73, var_76, adj_64, adj_73, adj_76);
        // adj: crossing_t = s0 / denom                                                       <L 271>
        if (var_75) {
            label6:;
            // adj: return                                                                    <L 270>
        }
        // adj: if denom <= 1.0e-8:                                                           <L 269>
        wp::adj_sub(var_64, var_66, adj_64, adj_66, adj_73);
        // adj: denom = s0 - s1                                                               <L 268>
    }
    // adj: if s0 > 0.0:                                                                      <L 267>
    wp::adj_float32(var_69, adj_69, adj_70);
    // adj: t = wp.float32(1.0)                                                               <L 266>
    if (var_68) {
        label5:;
        // adj: return                                                                        <L 264>
    }
    // adj: if s1 >= 0.0:                                                                     <L 263>
    wp::adj_dot(var_60, var_65, adj_60, adj_65, adj_66);
    wp::adj_sub(var_45, var_62, adj_45, adj_62, adj_65);
    // adj: s1 = wp.dot(n, x1 - plane_point)                                                  <L 262>
    wp::adj_dot(var_60, var_63, adj_60, adj_63, adj_64);
    wp::adj_sub(var_43, var_62, adj_43, adj_62, adj_63);
    // adj: s0 = wp.dot(n, x0 - plane_point)                                                  <L 261>
    wp::adj_add(var_39, var_61, adj_39, adj_61, adj_62);
    wp::adj_mul(var_60, var_27, adj_60, adj_27, adj_61);
    // adj: plane_point = center + n * radius                                                 <L 259>
    wp::adj_normalize(var_56, var_60, adj_56, adj_60);
    // adj: n = wp.normalize(normal)                                                          <L 258>
    if (var_59) {
        label4:;
        // adj: return                                                                        <L 256>
    }
    wp::adj_length_sq(var_56, adj_56, adj_57);
    // adj: if wp.length_sq(normal) <= 1.0e-12:                                               <L 255>
    wp::adj_where(var_54, var_55, var_51, adj_54, adj_55, adj_51, adj_56);
    if (var_54) {
        wp::adj_neg(var_22, adj_22, adj_55);
        // adj: normal = -displacement                                                        <L 254>
    }
    wp::adj_length_sq(var_51, adj_51, adj_52);
    // adj: if wp.length_sq(normal) <= 1.0e-12:                                               <L 253>
    wp::adj_where(var_49, var_50, var_46, adj_49, adj_50, adj_46, adj_51);
    if (var_49) {
        wp::adj_sub(var_43, var_39, adj_43, adj_39, adj_50);
        // adj: normal = x0 - center                                                          <L 252>
    }
    wp::adj_length_sq(var_46, adj_46, adj_47);
    // adj: if wp.length_sq(normal) <= 1.0e-12:                                               <L 251>
    wp::adj_sub(var_45, var_39, adj_45, adj_39, adj_46);
    // adj: normal = x1 - center                                                              <L 250>
    wp::adj_add(var_43, var_22, adj_43, adj_22, adj_45);
    // adj: x1 = x0 + displacement                                                            <L 248>
    wp::adj_copy(var_44, adj_42, adj_43);
    wp::adj_address(var_base_positions, var_13, adj_base_positions, adj_13, adj_42);
    // adj: x0 = base_positions[vertex_id]                                                    <L 247>
    wp::adj_lerp(var_40, var_41, var_34, adj_36, adj_38, adj_34, adj_39);
    wp::adj_address(var_sphere_center, var_37, adj_sphere_center, adj_37, adj_38);
    wp::adj_address(var_sphere_center_prev, var_35, adj_sphere_center_prev, adj_35, adj_36);
    // adj: center = wp.lerp(sphere_center_prev[0], sphere_center[0], sample_factor)          <L 245>
    wp::adj_div(var_32, var_33, var_34, adj_32, adj_33, adj_34);
    wp::adj_float32(var_motion_samples, adj_motion_samples, adj_33);
    wp::adj_float32(var_31, adj_31, adj_32);
    wp::adj_add(var_11, var_30, adj_11, adj_30, adj_31);
    // adj: sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)           <L 244>
    if (var_29) {
        label3:;
        // adj: return                                                                        <L 242>
    }
    // adj: if radius <= 0.0:                                                                 <L 241>
    wp::adj_add(var_sphere_radius, var_radius_margin, adj_sphere_radius, adj_radius_margin, adj_27);
    // adj: radius = sphere_radius + radius_margin                                            <L 240>
    if (var_26) {
        label2:;
        // adj: return                                                                        <L 238>
    }
    wp::adj_length_sq(var_22, adj_22, adj_24);
    // adj: if wp.length_sq(displacement) <= 1.0e-12:                                         <L 237>
    wp::adj_copy(var_23, adj_21, adj_22);
    wp::adj_address(var_displacement_in, var_13, adj_displacement_in, adj_13, adj_21);
    // adj: displacement = displacement_in[vertex_id]                                         <L 236>
    if (var_20) {
        label1:;
        // adj: return                                                                        <L 234>
    }
    wp::adj_address(var_particle_flags, var_13, adj_particle_flags, adj_13, adj_15);
    // adj: if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:                       <L 233>
    wp::adj_copy(var_14, adj_12, adj_13);
    wp::adj_address(var_surface_vertex_ids, var_10, adj_surface_vertex_ids, adj_10, adj_12);
    // adj: vertex_id = surface_vertex_ids[vertex_idx]                                        <L 232>
    wp::adj_mod(var_0, var_motion_samples, adj_0, adj_motion_samples, adj_11);
    // adj: sample_idx = tid % motion_samples                                                 <L 231>
    // adj: vertex_idx = tid // motion_samples                                                <L 230>
    if (var_1) {
        label0:;
        // adj: return                                                                        <L 228>
    }
    if (!var_1) {
    }
    wp::adj_mul(var_4, var_motion_samples, adj_4, adj_motion_samples, adj_6);
    wp::adj_extract(var_5, var_3, adj_2, adj_3, adj_4);
    adj_surface_vertex_ids.shape = adj_2;
    // adj: if tid >= surface_vertex_ids.shape[0] * motion_samples or motion_samples <= 0:    <L 227>
    // adj: tid = wp.tid()                                                                    <L 226>
    // adj: def compute_proxy_vertex_truncation_factors(                                      <L 213>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void compute_proxy_vertex_truncation_factors_d7c454ca_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_compute_proxy_vertex_truncation_factors_d7c454ca *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        compute_proxy_vertex_truncation_factors_d7c454ca_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void compute_proxy_vertex_truncation_factors_d7c454ca_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_compute_proxy_vertex_truncation_factors_d7c454ca *_wp_args,
    wp_args_compute_proxy_vertex_truncation_factors_d7c454ca *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        compute_proxy_vertex_truncation_factors_d7c454ca_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

