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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/runtime.py:108
static wp::vec_t<3, wp::float32> _double_sided_fallback_dir_0(
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
    // def _double_sided_fallback_dir(                                                        <L 109>
    // if wp.length_sq(reference_dir) > 1.0e-12:                                              <L 116>
    var_0 = wp::length_sq(var_reference_dir);
    var_2 = (var_0 > var_1);
    if (var_2) {
        // return wp.normalize(reference_dir)                                                 <L 117>
        var_3 = wp::normalize(var_reference_dir);
        return var_3;
    }
    // centroid = (v0 + v1 + v2) / 3.0                                                        <L 119>
    var_4 = wp::add(var_v0, var_v1);
    var_5 = wp::add(var_4, var_v2);
    var_7 = wp::div(var_5, var_6);
    // centroid_dir = centroid - contact_point                                                <L 120>
    var_8 = wp::sub(var_7, var_contact_point);
    // if wp.length_sq(centroid_dir) > 1.0e-12:                                               <L 121>
    var_9 = wp::length_sq(var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // return wp.normalize(centroid_dir)                                                  <L 122>
        var_12 = wp::normalize(var_8);
        return var_12;
    }
    // d0 = v0 - contact_point                                                                <L 124>
    var_13 = wp::sub(var_v0, var_contact_point);
    // d1 = v1 - contact_point                                                                <L 125>
    var_14 = wp::sub(var_v1, var_contact_point);
    // d2 = v2 - contact_point                                                                <L 126>
    var_15 = wp::sub(var_v2, var_contact_point);
    // best = d0                                                                              <L 128>
    var_16 = wp::copy(var_13);
    // if wp.length_sq(d1) > wp.length_sq(best):                                              <L 129>
    var_17 = wp::length_sq(var_14);
    var_18 = wp::length_sq(var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // best = d1                                                                          <L 130>
        var_20 = wp::copy(var_14);
    }
    var_21 = wp::where(var_19, var_20, var_16);
    // if wp.length_sq(d2) > wp.length_sq(best):                                              <L 131>
    var_22 = wp::length_sq(var_15);
    var_23 = wp::length_sq(var_21);
    var_24 = (var_22 > var_23);
    if (var_24) {
        // best = d2                                                                          <L 132>
        var_25 = wp::copy(var_15);
    }
    var_26 = wp::where(var_24, var_25, var_21);
    // if wp.length_sq(best) > 1.0e-12:                                                       <L 134>
    var_27 = wp::length_sq(var_26);
    var_29 = (var_27 > var_28);
    if (var_29) {
        // return wp.normalize(best)                                                          <L 135>
        var_30 = wp::normalize(var_26);
        return var_30;
    }
    // return wp.vec3f(1.0, 0.0, 0.0)                                                         <L 137>
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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/runtime.py:108
static void adj__double_sided_fallback_dir_0(
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
    // def _double_sided_fallback_dir(                                                        <L 109>
    // if wp.length_sq(reference_dir) > 1.0e-12:                                              <L 116>
    var_0 = wp::length_sq(var_reference_dir);
    var_2 = (var_0 > var_1);
    if (var_2) {
        // return wp.normalize(reference_dir)                                                 <L 117>
        var_3 = wp::normalize(var_reference_dir);
        goto label0;
    }
    // centroid = (v0 + v1 + v2) / 3.0                                                        <L 119>
    var_4 = wp::add(var_v0, var_v1);
    var_5 = wp::add(var_4, var_v2);
    var_7 = wp::div(var_5, var_6);
    // centroid_dir = centroid - contact_point                                                <L 120>
    var_8 = wp::sub(var_7, var_contact_point);
    // if wp.length_sq(centroid_dir) > 1.0e-12:                                               <L 121>
    var_9 = wp::length_sq(var_8);
    var_11 = (var_9 > var_10);
    if (var_11) {
        // return wp.normalize(centroid_dir)                                                  <L 122>
        var_12 = wp::normalize(var_8);
        goto label1;
    }
    // d0 = v0 - contact_point                                                                <L 124>
    var_13 = wp::sub(var_v0, var_contact_point);
    // d1 = v1 - contact_point                                                                <L 125>
    var_14 = wp::sub(var_v1, var_contact_point);
    // d2 = v2 - contact_point                                                                <L 126>
    var_15 = wp::sub(var_v2, var_contact_point);
    // best = d0                                                                              <L 128>
    var_16 = wp::copy(var_13);
    // if wp.length_sq(d1) > wp.length_sq(best):                                              <L 129>
    var_17 = wp::length_sq(var_14);
    var_18 = wp::length_sq(var_16);
    var_19 = (var_17 > var_18);
    if (var_19) {
        // best = d1                                                                          <L 130>
        var_20 = wp::copy(var_14);
    }
    var_21 = wp::where(var_19, var_20, var_16);
    // if wp.length_sq(d2) > wp.length_sq(best):                                              <L 131>
    var_22 = wp::length_sq(var_15);
    var_23 = wp::length_sq(var_21);
    var_24 = (var_22 > var_23);
    if (var_24) {
        // best = d2                                                                          <L 132>
        var_25 = wp::copy(var_15);
    }
    var_26 = wp::where(var_24, var_25, var_21);
    // if wp.length_sq(best) > 1.0e-12:                                                       <L 134>
    var_27 = wp::length_sq(var_26);
    var_29 = (var_27 > var_28);
    if (var_29) {
        // return wp.normalize(best)                                                          <L 135>
        var_30 = wp::normalize(var_26);
        goto label2;
    }
    // return wp.vec3f(1.0, 0.0, 0.0)                                                         <L 137>
    var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
    goto label3;
    //---------
    // reverse
    label3:;
    adj_34 += adj_ret;
    wp::adj_vec_t(var_31, var_32, var_33, adj_31, adj_32, adj_33, adj_34);
    // adj: return wp.vec3f(1.0, 0.0, 0.0)                                                    <L 137>
    if (var_29) {
        label2:;
        adj_30 += adj_ret;
        wp::adj_normalize(var_26, var_30, adj_26, adj_30);
        // adj: return wp.normalize(best)                                                     <L 135>
    }
    wp::adj_length_sq(var_26, adj_26, adj_27);
    // adj: if wp.length_sq(best) > 1.0e-12:                                                  <L 134>
    wp::adj_where(var_24, var_25, var_21, adj_24, adj_25, adj_21, adj_26);
    if (var_24) {
        wp::adj_copy(var_15, adj_15, adj_25);
        // adj: best = d2                                                                     <L 132>
    }
    wp::adj_length_sq(var_21, adj_21, adj_23);
    wp::adj_length_sq(var_15, adj_15, adj_22);
    // adj: if wp.length_sq(d2) > wp.length_sq(best):                                         <L 131>
    wp::adj_where(var_19, var_20, var_16, adj_19, adj_20, adj_16, adj_21);
    if (var_19) {
        wp::adj_copy(var_14, adj_14, adj_20);
        // adj: best = d1                                                                     <L 130>
    }
    wp::adj_length_sq(var_16, adj_16, adj_18);
    wp::adj_length_sq(var_14, adj_14, adj_17);
    // adj: if wp.length_sq(d1) > wp.length_sq(best):                                         <L 129>
    wp::adj_copy(var_13, adj_13, adj_16);
    // adj: best = d0                                                                         <L 128>
    wp::adj_sub(var_v2, var_contact_point, adj_v2, adj_contact_point, adj_15);
    // adj: d2 = v2 - contact_point                                                           <L 126>
    wp::adj_sub(var_v1, var_contact_point, adj_v1, adj_contact_point, adj_14);
    // adj: d1 = v1 - contact_point                                                           <L 125>
    wp::adj_sub(var_v0, var_contact_point, adj_v0, adj_contact_point, adj_13);
    // adj: d0 = v0 - contact_point                                                           <L 124>
    if (var_11) {
        label1:;
        adj_12 += adj_ret;
        wp::adj_normalize(var_8, var_12, adj_8, adj_12);
        // adj: return wp.normalize(centroid_dir)                                             <L 122>
    }
    wp::adj_length_sq(var_8, adj_8, adj_9);
    // adj: if wp.length_sq(centroid_dir) > 1.0e-12:                                          <L 121>
    wp::adj_sub(var_7, var_contact_point, adj_7, adj_contact_point, adj_8);
    // adj: centroid_dir = centroid - contact_point                                           <L 120>
    wp::adj_div(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_add(var_4, var_v2, adj_4, adj_v2, adj_5);
    wp::adj_add(var_v0, var_v1, adj_v0, adj_v1, adj_4);
    // adj: centroid = (v0 + v1 + v2) / 3.0                                                   <L 119>
    if (var_2) {
        label0:;
        adj_3 += adj_ret;
        wp::adj_normalize(var_reference_dir, var_3, adj_reference_dir, adj_3);
        // adj: return wp.normalize(reference_dir)                                            <L 117>
    }
    wp::adj_length_sq(var_reference_dir, adj_reference_dir, adj_0);
    // adj: if wp.length_sq(reference_dir) > 1.0e-12:                                         <L 116>
    // adj: def _double_sided_fallback_dir(                                                   <L 109>
    return;
}

struct wp_args_compute_vertex_sphere_truncation_factors_4ef208bd {
    wp::array_t<wp::int32> particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> base_positions;
    wp::array_t<wp::int32> surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_centers;
    wp::array_t<wp::float32> sphere_radii;
    wp::int32 num_spheres;
    wp::int32 motion_samples;
    wp::float32 radius_margin;
    wp::float32 safety;
    wp::array_t<wp::float32> truncation_t_out;
};


void compute_vertex_sphere_truncation_factors_4ef208bd_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_compute_vertex_sphere_truncation_factors_4ef208bd *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> var_base_positions = _wp_args->base_positions;
    wp::array_t<wp::int32> var_surface_vertex_ids = _wp_args->surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacement_in = _wp_args->displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers_prev = _wp_args->sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers = _wp_args->sphere_centers;
    wp::array_t<wp::float32> var_sphere_radii = _wp_args->sphere_radii;
    wp::int32 var_num_spheres = _wp_args->num_spheres;
    wp::int32 var_motion_samples = _wp_args->motion_samples;
    wp::float32 var_radius_margin = _wp_args->radius_margin;
    wp::float32 var_safety = _wp_args->safety;
    wp::array_t<wp::float32> var_truncation_t_out = _wp_args->truncation_t_out;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::shape_t* var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    wp::shape_t var_4;
    bool var_5;
    const wp::int32 var_6 = 0;
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    bool var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::int32* var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::int32* var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 1e-12;
    bool var_32;
    wp::float32* var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32>* var_43;
    wp::vec_t<3, wp::float32>* var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 1e-12;
    bool var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 1e-12;
    bool var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::float32 var_63;
    const wp::float32 var_64 = 1e-12;
    bool var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::float32 var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    const wp::float32 var_75 = 1.0;
    wp::float32 var_76;
    const wp::float32 var_77 = 0.0;
    bool var_78;
    wp::float32 var_79;
    const wp::float32 var_80 = 1e-08;
    bool var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.001;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::float32 var_87 = 0.0;
    const wp::float32 var_88 = 1.0;
    wp::float32 var_89;
    wp::float32 var_90;
    bool var_91;
    const wp::float32 var_92 = 0.0;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    //---------
    // forward
    // def compute_vertex_sphere_truncation_factors(                                          <L 245>
    // tid = wp.tid()                                                                         <L 259>
    var_0 = builtin_tid1d();
    // surface_vertex_count = surface_vertex_ids.shape[0]                                     <L 260>
    var_1 = &(var_surface_vertex_ids.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    // if surface_vertex_count == 0 or num_spheres <= 0 or motion_samples <= 0:               <L 261>
    var_7 = (var_3 == var_6);
    var_5 = var_7;
    if (!var_5) {
        var_9 = (var_num_spheres <= var_8);
        var_5 = var_5 || var_9;
    }
    if (!var_5) {
        var_11 = (var_motion_samples <= var_10);
        var_5 = var_5 || var_11;
    }
    if (var_5) {
        // return                                                                             <L 262>
        return;
    }
    // work_items_per_vertex = num_spheres * motion_samples                                   <L 264>
    var_12 = wp::mul(var_num_spheres, var_motion_samples);
    // vertex_idx = tid // work_items_per_vertex                                              <L 265>
    var_13 = wp::floordiv(var_0, var_12);
    // if vertex_idx >= surface_vertex_count:                                                 <L 266>
    var_14 = (var_13 >= var_3);
    if (var_14) {
        // return                                                                             <L 267>
        return;
    }
    // local_idx = tid % work_items_per_vertex                                                <L 269>
    var_15 = wp::mod(var_0, var_12);
    // sphere_idx = local_idx // motion_samples                                               <L 270>
    var_16 = wp::floordiv(var_15, var_motion_samples);
    // sample_idx = local_idx % motion_samples                                                <L 271>
    var_17 = wp::mod(var_15, var_motion_samples);
    // vertex_id = surface_vertex_ids[vertex_idx]                                             <L 272>
    var_18 = wp::address(var_surface_vertex_ids, var_13);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:                            <L 274>
    var_21 = wp::address(var_particle_flags, var_19);
    var_24 = wp::load(var_21);
    var_23 = wp::bit_and(var_24, var_22);
    var_26 = (var_23 == var_25);
    if (var_26) {
        // return                                                                             <L 275>
        return;
    }
    // displacement = displacement_in[vertex_id]                                              <L 277>
    var_27 = wp::address(var_displacement_in, var_19);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // if wp.length_sq(displacement) <= 1.0e-12:                                              <L 278>
    var_30 = wp::length_sq(var_28);
    var_32 = (var_30 <= var_31);
    if (var_32) {
        // return                                                                             <L 279>
        return;
    }
    // sphere_radius = sphere_radii[sphere_idx] + radius_margin                               <L 281>
    var_33 = wp::address(var_sphere_radii, var_16);
    var_35 = wp::load(var_33);
    var_34 = wp::add(var_35, var_radius_margin);
    // if sphere_radius <= 0.0:                                                               <L 282>
    var_37 = (var_34 <= var_36);
    if (var_37) {
        // return                                                                             <L 283>
        return;
    }
    // sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)                <L 285>
    var_39 = wp::add(var_17, var_38);
    var_40 = wp::float32(var_39);
    var_41 = wp::float32(var_motion_samples);
    var_42 = wp::div(var_40, var_41);
    // sphere_center = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)       <L 286>
    var_43 = wp::address(var_sphere_centers_prev, var_16);
    var_44 = wp::address(var_sphere_centers, var_16);
    var_46 = wp::load(var_43);
    var_47 = wp::load(var_44);
    var_45 = wp::lerp(var_46, var_47, var_42);
    // x0 = base_positions[vertex_id]                                                         <L 288>
    var_48 = wp::address(var_base_positions, var_19);
    var_50 = wp::load(var_48);
    var_49 = wp::copy(var_50);
    // x1 = x0 + displacement                                                                 <L 289>
    var_51 = wp::add(var_49, var_28);
    // normal = x1 - sphere_center                                                            <L 291>
    var_52 = wp::sub(var_51, var_45);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 292>
    var_53 = wp::length_sq(var_52);
    var_55 = (var_53 <= var_54);
    if (var_55) {
        // normal = x0 - sphere_center                                                        <L 293>
        var_56 = wp::sub(var_49, var_45);
    }
    var_57 = wp::where(var_55, var_56, var_52);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 294>
    var_58 = wp::length_sq(var_57);
    var_60 = (var_58 <= var_59);
    if (var_60) {
        // normal = -displacement                                                             <L 295>
        var_61 = wp::neg(var_28);
    }
    var_62 = wp::where(var_60, var_61, var_57);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 296>
    var_63 = wp::length_sq(var_62);
    var_65 = (var_63 <= var_64);
    if (var_65) {
        // return                                                                             <L 297>
        return;
    }
    // n = wp.normalize(normal)                                                               <L 299>
    var_66 = wp::normalize(var_62);
    // plane_point = sphere_center + n * sphere_radius                                        <L 300>
    var_67 = wp::mul(var_66, var_34);
    var_68 = wp::add(var_45, var_67);
    // s0 = wp.dot(n, x0 - plane_point)                                                       <L 302>
    var_69 = wp::sub(var_49, var_68);
    var_70 = wp::dot(var_66, var_69);
    // s1 = wp.dot(n, x1 - plane_point)                                                       <L 303>
    var_71 = wp::sub(var_51, var_68);
    var_72 = wp::dot(var_66, var_71);
    // if s1 >= 0.0:                                                                          <L 304>
    var_74 = (var_72 >= var_73);
    if (var_74) {
        // return                                                                             <L 305>
        return;
    }
    // t = wp.float32(1.0)                                                                    <L 307>
    var_76 = wp::float32(var_75);
    // if s0 > 0.0:                                                                           <L 308>
    var_78 = (var_70 > var_77);
    if (var_78) {
        // denom = s0 - s1                                                                    <L 309>
        var_79 = wp::sub(var_70, var_72);
        // if denom <= 1.0e-8:                                                                <L 310>
        var_81 = (var_79 <= var_80);
        if (var_81) {
            // return                                                                         <L 311>
            return;
        }
        // crossing_t = s0 / denom                                                            <L 312>
        var_82 = wp::div(var_70, var_79);
        // t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)           <L 313>
        var_83 = wp::mul(var_82, var_safety);
        var_85 = wp::sub(var_82, var_84);
        var_86 = wp::min(var_83, var_85);
        var_89 = wp::clamp(var_86, var_87, var_88);
    }
    var_90 = wp::where(var_78, var_89, var_76);
    if (!var_78) {
        // elif s1 < s0:                                                                      <L 314>
        var_91 = (var_72 < var_70);
        if (var_91) {
            // t = 0.0                                                                        <L 315>
        }
        var_93 = wp::where(var_91, var_92, var_90);
        if (!var_91) {
            // return                                                                         <L 317>
            return;
        }
    }
    var_94 = wp::where(var_78, var_90, var_93);
    // wp.atomic_min(truncation_t_out, vertex_id, t)                                          <L 319>
    var_95 = wp::atomic_min(var_truncation_t_out, var_19, var_94);
}



void compute_vertex_sphere_truncation_factors_4ef208bd_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_compute_vertex_sphere_truncation_factors_4ef208bd *_wp_args,
    wp_args_compute_vertex_sphere_truncation_factors_4ef208bd *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> var_base_positions = _wp_args->base_positions;
    wp::array_t<wp::int32> var_surface_vertex_ids = _wp_args->surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacement_in = _wp_args->displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers_prev = _wp_args->sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers = _wp_args->sphere_centers;
    wp::array_t<wp::float32> var_sphere_radii = _wp_args->sphere_radii;
    wp::int32 var_num_spheres = _wp_args->num_spheres;
    wp::int32 var_motion_samples = _wp_args->motion_samples;
    wp::float32 var_radius_margin = _wp_args->radius_margin;
    wp::float32 var_safety = _wp_args->safety;
    wp::array_t<wp::float32> var_truncation_t_out = _wp_args->truncation_t_out;
    wp::array_t<wp::int32> adj_particle_flags = _wp_adj_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_base_positions = _wp_adj_args->base_positions;
    wp::array_t<wp::int32> adj_surface_vertex_ids = _wp_adj_args->surface_vertex_ids;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_displacement_in = _wp_adj_args->displacement_in;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_centers_prev = _wp_adj_args->sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_centers = _wp_adj_args->sphere_centers;
    wp::array_t<wp::float32> adj_sphere_radii = _wp_adj_args->sphere_radii;
    wp::int32 adj_num_spheres = _wp_adj_args->num_spheres;
    wp::int32 adj_motion_samples = _wp_adj_args->motion_samples;
    wp::float32 adj_radius_margin = _wp_adj_args->radius_margin;
    wp::float32 adj_safety = _wp_adj_args->safety;
    wp::array_t<wp::float32> adj_truncation_t_out = _wp_adj_args->truncation_t_out;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::shape_t* var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    wp::shape_t var_4;
    bool var_5;
    const wp::int32 var_6 = 0;
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    bool var_14;
    wp::int32 var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::int32* var_18;
    wp::int32 var_19;
    wp::int32 var_20;
    wp::int32* var_21;
    const wp::int32 var_22 = 1;
    wp::int32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = 0;
    bool var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::float32 var_30;
    const wp::float32 var_31 = 1e-12;
    bool var_32;
    wp::float32* var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::float32 var_36 = 0.0;
    bool var_37;
    const wp::int32 var_38 = 1;
    wp::int32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::vec_t<3, wp::float32>* var_43;
    wp::vec_t<3, wp::float32>* var_44;
    wp::vec_t<3, wp::float32> var_45;
    wp::vec_t<3, wp::float32> var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32>* var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::vec_t<3, wp::float32> var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::float32 var_53;
    const wp::float32 var_54 = 1e-12;
    bool var_55;
    wp::vec_t<3, wp::float32> var_56;
    wp::vec_t<3, wp::float32> var_57;
    wp::float32 var_58;
    const wp::float32 var_59 = 1e-12;
    bool var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::float32 var_63;
    const wp::float32 var_64 = 1e-12;
    bool var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    wp::vec_t<3, wp::float32> var_69;
    wp::float32 var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::float32 var_72;
    const wp::float32 var_73 = 0.0;
    bool var_74;
    const wp::float32 var_75 = 1.0;
    wp::float32 var_76;
    const wp::float32 var_77 = 0.0;
    bool var_78;
    wp::float32 var_79;
    const wp::float32 var_80 = 1e-08;
    bool var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.001;
    wp::float32 var_85;
    wp::float32 var_86;
    const wp::float32 var_87 = 0.0;
    const wp::float32 var_88 = 1.0;
    wp::float32 var_89;
    wp::float32 var_90;
    bool var_91;
    const wp::float32 var_92 = 0.0;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::shape_t adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::shape_t adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    bool adj_7 = {};
    wp::int32 adj_8 = {};
    bool adj_9 = {};
    wp::int32 adj_10 = {};
    bool adj_11 = {};
    wp::int32 adj_12 = {};
    wp::int32 adj_13 = {};
    bool adj_14 = {};
    wp::int32 adj_15 = {};
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
    bool adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    bool adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::float32 adj_36 = {};
    bool adj_37 = {};
    wp::int32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::vec_t<3, wp::float32> adj_43 = {};
    wp::vec_t<3, wp::float32> adj_44 = {};
    wp::vec_t<3, wp::float32> adj_45 = {};
    wp::vec_t<3, wp::float32> adj_46 = {};
    wp::vec_t<3, wp::float32> adj_47 = {};
    wp::vec_t<3, wp::float32> adj_48 = {};
    wp::vec_t<3, wp::float32> adj_49 = {};
    wp::vec_t<3, wp::float32> adj_50 = {};
    wp::vec_t<3, wp::float32> adj_51 = {};
    wp::vec_t<3, wp::float32> adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    bool adj_55 = {};
    wp::vec_t<3, wp::float32> adj_56 = {};
    wp::vec_t<3, wp::float32> adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    bool adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::vec_t<3, wp::float32> adj_62 = {};
    wp::float32 adj_63 = {};
    wp::float32 adj_64 = {};
    bool adj_65 = {};
    wp::vec_t<3, wp::float32> adj_66 = {};
    wp::vec_t<3, wp::float32> adj_67 = {};
    wp::vec_t<3, wp::float32> adj_68 = {};
    wp::vec_t<3, wp::float32> adj_69 = {};
    wp::float32 adj_70 = {};
    wp::vec_t<3, wp::float32> adj_71 = {};
    wp::float32 adj_72 = {};
    wp::float32 adj_73 = {};
    bool adj_74 = {};
    wp::float32 adj_75 = {};
    wp::float32 adj_76 = {};
    wp::float32 adj_77 = {};
    bool adj_78 = {};
    wp::float32 adj_79 = {};
    wp::float32 adj_80 = {};
    bool adj_81 = {};
    wp::float32 adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float32 adj_87 = {};
    wp::float32 adj_88 = {};
    wp::float32 adj_89 = {};
    wp::float32 adj_90 = {};
    bool adj_91 = {};
    wp::float32 adj_92 = {};
    wp::float32 adj_93 = {};
    wp::float32 adj_94 = {};
    wp::float32 adj_95 = {};
    //---------
    // forward
    // def compute_vertex_sphere_truncation_factors(                                          <L 245>
    // tid = wp.tid()                                                                         <L 259>
    var_0 = builtin_tid1d();
    // surface_vertex_count = surface_vertex_ids.shape[0]                                     <L 260>
    var_1 = &(var_surface_vertex_ids.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    // if surface_vertex_count == 0 or num_spheres <= 0 or motion_samples <= 0:               <L 261>
    var_7 = (var_3 == var_6);
    var_5 = var_7;
    if (!var_5) {
        var_9 = (var_num_spheres <= var_8);
        var_5 = var_5 || var_9;
    }
    if (!var_5) {
        var_11 = (var_motion_samples <= var_10);
        var_5 = var_5 || var_11;
    }
    if (var_5) {
        // return                                                                             <L 262>
        goto label0;
    }
    // work_items_per_vertex = num_spheres * motion_samples                                   <L 264>
    var_12 = wp::mul(var_num_spheres, var_motion_samples);
    // vertex_idx = tid // work_items_per_vertex                                              <L 265>
    var_13 = wp::floordiv(var_0, var_12);
    // if vertex_idx >= surface_vertex_count:                                                 <L 266>
    var_14 = (var_13 >= var_3);
    if (var_14) {
        // return                                                                             <L 267>
        goto label1;
    }
    // local_idx = tid % work_items_per_vertex                                                <L 269>
    var_15 = wp::mod(var_0, var_12);
    // sphere_idx = local_idx // motion_samples                                               <L 270>
    var_16 = wp::floordiv(var_15, var_motion_samples);
    // sample_idx = local_idx % motion_samples                                                <L 271>
    var_17 = wp::mod(var_15, var_motion_samples);
    // vertex_id = surface_vertex_ids[vertex_idx]                                             <L 272>
    var_18 = wp::address(var_surface_vertex_ids, var_13);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:                            <L 274>
    var_21 = wp::address(var_particle_flags, var_19);
    var_24 = wp::load(var_21);
    var_23 = wp::bit_and(var_24, var_22);
    var_26 = (var_23 == var_25);
    if (var_26) {
        // return                                                                             <L 275>
        goto label2;
    }
    // displacement = displacement_in[vertex_id]                                              <L 277>
    var_27 = wp::address(var_displacement_in, var_19);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // if wp.length_sq(displacement) <= 1.0e-12:                                              <L 278>
    var_30 = wp::length_sq(var_28);
    var_32 = (var_30 <= var_31);
    if (var_32) {
        // return                                                                             <L 279>
        goto label3;
    }
    // sphere_radius = sphere_radii[sphere_idx] + radius_margin                               <L 281>
    var_33 = wp::address(var_sphere_radii, var_16);
    var_35 = wp::load(var_33);
    var_34 = wp::add(var_35, var_radius_margin);
    // if sphere_radius <= 0.0:                                                               <L 282>
    var_37 = (var_34 <= var_36);
    if (var_37) {
        // return                                                                             <L 283>
        goto label4;
    }
    // sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)                <L 285>
    var_39 = wp::add(var_17, var_38);
    var_40 = wp::float32(var_39);
    var_41 = wp::float32(var_motion_samples);
    var_42 = wp::div(var_40, var_41);
    // sphere_center = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)       <L 286>
    var_43 = wp::address(var_sphere_centers_prev, var_16);
    var_44 = wp::address(var_sphere_centers, var_16);
    var_46 = wp::load(var_43);
    var_47 = wp::load(var_44);
    var_45 = wp::lerp(var_46, var_47, var_42);
    // x0 = base_positions[vertex_id]                                                         <L 288>
    var_48 = wp::address(var_base_positions, var_19);
    var_50 = wp::load(var_48);
    var_49 = wp::copy(var_50);
    // x1 = x0 + displacement                                                                 <L 289>
    var_51 = wp::add(var_49, var_28);
    // normal = x1 - sphere_center                                                            <L 291>
    var_52 = wp::sub(var_51, var_45);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 292>
    var_53 = wp::length_sq(var_52);
    var_55 = (var_53 <= var_54);
    if (var_55) {
        // normal = x0 - sphere_center                                                        <L 293>
        var_56 = wp::sub(var_49, var_45);
    }
    var_57 = wp::where(var_55, var_56, var_52);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 294>
    var_58 = wp::length_sq(var_57);
    var_60 = (var_58 <= var_59);
    if (var_60) {
        // normal = -displacement                                                             <L 295>
        var_61 = wp::neg(var_28);
    }
    var_62 = wp::where(var_60, var_61, var_57);
    // if wp.length_sq(normal) <= 1.0e-12:                                                    <L 296>
    var_63 = wp::length_sq(var_62);
    var_65 = (var_63 <= var_64);
    if (var_65) {
        // return                                                                             <L 297>
        goto label5;
    }
    // n = wp.normalize(normal)                                                               <L 299>
    var_66 = wp::normalize(var_62);
    // plane_point = sphere_center + n * sphere_radius                                        <L 300>
    var_67 = wp::mul(var_66, var_34);
    var_68 = wp::add(var_45, var_67);
    // s0 = wp.dot(n, x0 - plane_point)                                                       <L 302>
    var_69 = wp::sub(var_49, var_68);
    var_70 = wp::dot(var_66, var_69);
    // s1 = wp.dot(n, x1 - plane_point)                                                       <L 303>
    var_71 = wp::sub(var_51, var_68);
    var_72 = wp::dot(var_66, var_71);
    // if s1 >= 0.0:                                                                          <L 304>
    var_74 = (var_72 >= var_73);
    if (var_74) {
        // return                                                                             <L 305>
        goto label6;
    }
    // t = wp.float32(1.0)                                                                    <L 307>
    var_76 = wp::float32(var_75);
    // if s0 > 0.0:                                                                           <L 308>
    var_78 = (var_70 > var_77);
    if (var_78) {
        // denom = s0 - s1                                                                    <L 309>
        var_79 = wp::sub(var_70, var_72);
        // if denom <= 1.0e-8:                                                                <L 310>
        var_81 = (var_79 <= var_80);
        if (var_81) {
            // return                                                                         <L 311>
            goto label7;
        }
        // crossing_t = s0 / denom                                                            <L 312>
        var_82 = wp::div(var_70, var_79);
        // t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)           <L 313>
        var_83 = wp::mul(var_82, var_safety);
        var_85 = wp::sub(var_82, var_84);
        var_86 = wp::min(var_83, var_85);
        var_89 = wp::clamp(var_86, var_87, var_88);
    }
    var_90 = wp::where(var_78, var_89, var_76);
    if (!var_78) {
        // elif s1 < s0:                                                                      <L 314>
        var_91 = (var_72 < var_70);
        if (var_91) {
            // t = 0.0                                                                        <L 315>
        }
        var_93 = wp::where(var_91, var_92, var_90);
        if (!var_91) {
            // return                                                                         <L 317>
            goto label8;
        }
    }
    var_94 = wp::where(var_78, var_90, var_93);
    // wp.atomic_min(truncation_t_out, vertex_id, t)                                          <L 319>
    // var_95 = wp::atomic_min(var_truncation_t_out, var_19, var_94);
    //---------
    // reverse
    wp::adj_atomic_min(var_truncation_t_out, var_19, var_94, adj_truncation_t_out, adj_19, adj_94, adj_95);
    // adj: wp.atomic_min(truncation_t_out, vertex_id, t)                                     <L 319>
    wp::adj_where(var_78, var_90, var_93, adj_78, adj_90, adj_93, adj_94);
    if (!var_78) {
        if (!var_91) {
            label8:;
            // adj: return                                                                    <L 317>
        }
        wp::adj_where(var_91, var_92, var_90, adj_91, adj_92, adj_90, adj_93);
        if (var_91) {
            // adj: t = 0.0                                                                   <L 315>
        }
        // adj: elif s1 < s0:                                                                 <L 314>
    }
    wp::adj_where(var_78, var_89, var_76, adj_78, adj_89, adj_76, adj_90);
    if (var_78) {
        wp::adj_clamp(var_86, var_87, var_88, adj_86, adj_87, adj_88, adj_89);
        wp::adj_min(var_83, var_85, adj_83, adj_85, adj_86);
        wp::adj_sub(var_82, var_84, adj_82, adj_84, adj_85);
        wp::adj_mul(var_82, var_safety, adj_82, adj_safety, adj_83);
        // adj: t = wp.clamp(wp.min(crossing_t * safety, crossing_t - 1.0e-3), 0.0, 1.0)      <L 313>
        wp::adj_div(var_70, var_79, var_82, adj_70, adj_79, adj_82);
        // adj: crossing_t = s0 / denom                                                       <L 312>
        if (var_81) {
            label7:;
            // adj: return                                                                    <L 311>
        }
        // adj: if denom <= 1.0e-8:                                                           <L 310>
        wp::adj_sub(var_70, var_72, adj_70, adj_72, adj_79);
        // adj: denom = s0 - s1                                                               <L 309>
    }
    // adj: if s0 > 0.0:                                                                      <L 308>
    wp::adj_float32(var_75, adj_75, adj_76);
    // adj: t = wp.float32(1.0)                                                               <L 307>
    if (var_74) {
        label6:;
        // adj: return                                                                        <L 305>
    }
    // adj: if s1 >= 0.0:                                                                     <L 304>
    wp::adj_dot(var_66, var_71, adj_66, adj_71, adj_72);
    wp::adj_sub(var_51, var_68, adj_51, adj_68, adj_71);
    // adj: s1 = wp.dot(n, x1 - plane_point)                                                  <L 303>
    wp::adj_dot(var_66, var_69, adj_66, adj_69, adj_70);
    wp::adj_sub(var_49, var_68, adj_49, adj_68, adj_69);
    // adj: s0 = wp.dot(n, x0 - plane_point)                                                  <L 302>
    wp::adj_add(var_45, var_67, adj_45, adj_67, adj_68);
    wp::adj_mul(var_66, var_34, adj_66, adj_34, adj_67);
    // adj: plane_point = sphere_center + n * sphere_radius                                   <L 300>
    wp::adj_normalize(var_62, var_66, adj_62, adj_66);
    // adj: n = wp.normalize(normal)                                                          <L 299>
    if (var_65) {
        label5:;
        // adj: return                                                                        <L 297>
    }
    wp::adj_length_sq(var_62, adj_62, adj_63);
    // adj: if wp.length_sq(normal) <= 1.0e-12:                                               <L 296>
    wp::adj_where(var_60, var_61, var_57, adj_60, adj_61, adj_57, adj_62);
    if (var_60) {
        wp::adj_neg(var_28, adj_28, adj_61);
        // adj: normal = -displacement                                                        <L 295>
    }
    wp::adj_length_sq(var_57, adj_57, adj_58);
    // adj: if wp.length_sq(normal) <= 1.0e-12:                                               <L 294>
    wp::adj_where(var_55, var_56, var_52, adj_55, adj_56, adj_52, adj_57);
    if (var_55) {
        wp::adj_sub(var_49, var_45, adj_49, adj_45, adj_56);
        // adj: normal = x0 - sphere_center                                                   <L 293>
    }
    wp::adj_length_sq(var_52, adj_52, adj_53);
    // adj: if wp.length_sq(normal) <= 1.0e-12:                                               <L 292>
    wp::adj_sub(var_51, var_45, adj_51, adj_45, adj_52);
    // adj: normal = x1 - sphere_center                                                       <L 291>
    wp::adj_add(var_49, var_28, adj_49, adj_28, adj_51);
    // adj: x1 = x0 + displacement                                                            <L 289>
    wp::adj_copy(var_50, adj_48, adj_49);
    wp::adj_address(var_base_positions, var_19, adj_base_positions, adj_19, adj_48);
    // adj: x0 = base_positions[vertex_id]                                                    <L 288>
    wp::adj_lerp(var_46, var_47, var_42, adj_43, adj_44, adj_42, adj_45);
    wp::adj_address(var_sphere_centers, var_16, adj_sphere_centers, adj_16, adj_44);
    wp::adj_address(var_sphere_centers_prev, var_16, adj_sphere_centers_prev, adj_16, adj_43);
    // adj: sphere_center = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)  <L 286>
    wp::adj_div(var_40, var_41, var_42, adj_40, adj_41, adj_42);
    wp::adj_float32(var_motion_samples, adj_motion_samples, adj_41);
    wp::adj_float32(var_39, adj_39, adj_40);
    wp::adj_add(var_17, var_38, adj_17, adj_38, adj_39);
    // adj: sample_factor = wp.float32(sample_idx + 1) / wp.float32(motion_samples)           <L 285>
    if (var_37) {
        label4:;
        // adj: return                                                                        <L 283>
    }
    // adj: if sphere_radius <= 0.0:                                                          <L 282>
    wp::adj_add(var_35, var_radius_margin, adj_33, adj_radius_margin, adj_34);
    wp::adj_address(var_sphere_radii, var_16, adj_sphere_radii, adj_16, adj_33);
    // adj: sphere_radius = sphere_radii[sphere_idx] + radius_margin                          <L 281>
    if (var_32) {
        label3:;
        // adj: return                                                                        <L 279>
    }
    wp::adj_length_sq(var_28, adj_28, adj_30);
    // adj: if wp.length_sq(displacement) <= 1.0e-12:                                         <L 278>
    wp::adj_copy(var_29, adj_27, adj_28);
    wp::adj_address(var_displacement_in, var_19, adj_displacement_in, adj_19, adj_27);
    // adj: displacement = displacement_in[vertex_id]                                         <L 277>
    if (var_26) {
        label2:;
        // adj: return                                                                        <L 275>
    }
    wp::adj_address(var_particle_flags, var_19, adj_particle_flags, adj_19, adj_21);
    // adj: if (particle_flags[vertex_id] & ParticleFlags.ACTIVE) == 0:                       <L 274>
    wp::adj_copy(var_20, adj_18, adj_19);
    wp::adj_address(var_surface_vertex_ids, var_13, adj_surface_vertex_ids, adj_13, adj_18);
    // adj: vertex_id = surface_vertex_ids[vertex_idx]                                        <L 272>
    wp::adj_mod(var_15, var_motion_samples, adj_15, adj_motion_samples, adj_17);
    // adj: sample_idx = local_idx % motion_samples                                           <L 271>
    // adj: sphere_idx = local_idx // motion_samples                                          <L 270>
    wp::adj_mod(var_0, var_12, adj_0, adj_12, adj_15);
    // adj: local_idx = tid % work_items_per_vertex                                           <L 269>
    if (var_14) {
        label1:;
        // adj: return                                                                        <L 267>
    }
    // adj: if vertex_idx >= surface_vertex_count:                                            <L 266>
    // adj: vertex_idx = tid // work_items_per_vertex                                         <L 265>
    wp::adj_mul(var_num_spheres, var_motion_samples, adj_num_spheres, adj_motion_samples, adj_12);
    // adj: work_items_per_vertex = num_spheres * motion_samples                              <L 264>
    if (var_5) {
        label0:;
        // adj: return                                                                        <L 262>
    }
    if (!var_5) {
    }
    if (!var_5) {
    }
    // adj: if surface_vertex_count == 0 or num_spheres <= 0 or motion_samples <= 0:          <L 261>
    wp::adj_extract(var_4, var_2, adj_1, adj_2, adj_3);
    adj_surface_vertex_ids.shape = adj_1;
    // adj: surface_vertex_count = surface_vertex_ids.shape[0]                                <L 260>
    // adj: tid = wp.tid()                                                                    <L 259>
    // adj: def compute_vertex_sphere_truncation_factors(                                     <L 245>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void compute_vertex_sphere_truncation_factors_4ef208bd_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_compute_vertex_sphere_truncation_factors_4ef208bd *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        compute_vertex_sphere_truncation_factors_4ef208bd_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void compute_vertex_sphere_truncation_factors_4ef208bd_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_compute_vertex_sphere_truncation_factors_4ef208bd *_wp_args,
    wp_args_compute_vertex_sphere_truncation_factors_4ef208bd *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        compute_vertex_sphere_truncation_factors_4ef208bd_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_fill_tissue_vertex_blend_colors_da9ba9ee {
    wp::array_t<wp::vec_t<4, wp::float32>> vertex_colors;
    wp::float32 damage;
    wp::float32 coag;
    wp::float32 blood;
};


void fill_tissue_vertex_blend_colors_da9ba9ee_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_fill_tissue_vertex_blend_colors_da9ba9ee *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<4, wp::float32>> var_vertex_colors = _wp_args->vertex_colors;
    wp::float32 var_damage = _wp_args->damage;
    wp::float32 var_coag = _wp_args->coag;
    wp::float32 var_blood = _wp_args->blood;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::float32 var_1 = 0.0;
    wp::vec_t<4, wp::float32> var_2;
    //---------
    // forward
    // def fill_tissue_vertex_blend_colors(                                                   <L 234>
    // tid = wp.tid()                                                                         <L 240>
    var_0 = builtin_tid1d();
    // vertex_colors[tid] = wp.vec4f(damage, coag, blood, 0.0)                                <L 241>
    var_2 = wp::vec_t<4, wp::float32>(var_damage, var_coag, var_blood, var_1);
    wp::array_store(var_vertex_colors, var_0, var_2);
}



void fill_tissue_vertex_blend_colors_da9ba9ee_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_fill_tissue_vertex_blend_colors_da9ba9ee *_wp_args,
    wp_args_fill_tissue_vertex_blend_colors_da9ba9ee *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<4, wp::float32>> var_vertex_colors = _wp_args->vertex_colors;
    wp::float32 var_damage = _wp_args->damage;
    wp::float32 var_coag = _wp_args->coag;
    wp::float32 var_blood = _wp_args->blood;
    wp::array_t<wp::vec_t<4, wp::float32>> adj_vertex_colors = _wp_adj_args->vertex_colors;
    wp::float32 adj_damage = _wp_adj_args->damage;
    wp::float32 adj_coag = _wp_adj_args->coag;
    wp::float32 adj_blood = _wp_adj_args->blood;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::float32 var_1 = 0.0;
    wp::vec_t<4, wp::float32> var_2;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::vec_t<4, wp::float32> adj_2 = {};
    //---------
    // forward
    // def fill_tissue_vertex_blend_colors(                                                   <L 234>
    // tid = wp.tid()                                                                         <L 240>
    var_0 = builtin_tid1d();
    // vertex_colors[tid] = wp.vec4f(damage, coag, blood, 0.0)                                <L 241>
    var_2 = wp::vec_t<4, wp::float32>(var_damage, var_coag, var_blood, var_1);
    // wp::array_store(var_vertex_colors, var_0, var_2);
    //---------
    // reverse
    wp::adj_array_store(var_vertex_colors, var_0, var_2, adj_vertex_colors, adj_0, adj_2);
    wp::adj_vec_t(var_damage, var_coag, var_blood, var_1, adj_damage, adj_coag, adj_blood, adj_1, adj_2);
    // adj: vertex_colors[tid] = wp.vec4f(damage, coag, blood, 0.0)                           <L 241>
    // adj: tid = wp.tid()                                                                    <L 240>
    // adj: def fill_tissue_vertex_blend_colors(                                              <L 234>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void fill_tissue_vertex_blend_colors_da9ba9ee_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_fill_tissue_vertex_blend_colors_da9ba9ee *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        fill_tissue_vertex_blend_colors_da9ba9ee_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void fill_tissue_vertex_blend_colors_da9ba9ee_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_fill_tissue_vertex_blend_colors_da9ba9ee *_wp_args,
    wp_args_fill_tissue_vertex_blend_colors_da9ba9ee *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        fill_tissue_vertex_blend_colors_da9ba9ee_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_collide_triangles_vs_spheres_636d21ad {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<wp::vec_t<3, wp::float32>> velocities;
    wp::array_t<wp::float32> inv_masses;
    wp::array_t<wp::int32> tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> sphere_centers;
    wp::array_t<wp::float32> sphere_radii;
    wp::int32 num_spheres;
    wp::int32 sweep_samples;
    wp::float32 radius_margin;
    wp::float32 restitution;
    wp::float32 dt;
    wp::float32 cull_radius;
    wp::array_t<wp::vec_t<3, wp::float32>> delta_accumulator;
    wp::array_t<wp::int32> delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> reaction_accumulator;
    wp::array_t<wp::int32> reaction_counter;
};


void collide_triangles_vs_spheres_636d21ad_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_collide_triangles_vs_spheres_636d21ad *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::float32> var_inv_masses = _wp_args->inv_masses;
    wp::array_t<wp::int32> var_tri_indices = _wp_args->tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers_prev = _wp_args->sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers = _wp_args->sphere_centers;
    wp::array_t<wp::float32> var_sphere_radii = _wp_args->sphere_radii;
    wp::int32 var_num_spheres = _wp_args->num_spheres;
    wp::int32 var_sweep_samples = _wp_args->sweep_samples;
    wp::float32 var_radius_margin = _wp_args->radius_margin;
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
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    const wp::int32 var_21 = 0;
    wp::int32* var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = 1;
    wp::int32* var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    const wp::int32 var_29 = 2;
    wp::int32* var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::float32* var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32* var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32* var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32>* var_60;
    wp::vec_t<3, wp::float32>* var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::float32 var_65 = 0.0;
    bool var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = 3.0;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::float32 var_72;
    bool var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::int32 var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::float32 var_78;
    bool var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 1e-08;
    bool var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32>* var_84;
    wp::vec_t<3, wp::float32>* var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::float32 var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    const wp::int32 var_101 = 1;
    wp::int32 var_102;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    const wp::int32 var_105 = 1;
    wp::int32 var_106;
    const wp::int32 var_107 = 0;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    const wp::int32 var_110 = 0;
    const wp::int32 var_111 = 1;
    wp::int32 var_112;
    //---------
    // forward
    // def collide_triangles_vs_spheres(                                                      <L 141>
    // tid = wp.tid()                                                                         <L 160>
    var_0 = builtin_tid1d();
    // tri_count = tri_indices.shape[0]                                                       <L 161>
    var_1 = &(var_tri_indices.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    // if tri_count == 0 or sweep_samples <= 0:                                               <L 162>
    var_7 = (var_3 == var_6);
    var_5 = var_7;
    if (!var_5) {
        var_9 = (var_sweep_samples <= var_8);
        var_5 = var_5 || var_9;
    }
    if (var_5) {
        // return                                                                             <L 163>
        return;
    }
    // work_items_per_sphere = tri_count * sweep_samples                                      <L 165>
    var_10 = wp::mul(var_3, var_sweep_samples);
    // sphere_idx = tid // work_items_per_sphere                                              <L 166>
    var_11 = wp::floordiv(var_0, var_10);
    // if sphere_idx >= num_spheres:                                                          <L 167>
    var_12 = (var_11 >= var_num_spheres);
    if (var_12) {
        // return                                                                             <L 168>
        return;
    }
    // sphere_work_idx = tid % work_items_per_sphere                                          <L 170>
    var_13 = wp::mod(var_0, var_10);
    // sample_idx = sphere_work_idx // tri_count                                              <L 171>
    var_14 = wp::floordiv(var_13, var_3);
    // tri_idx = sphere_work_idx % tri_count                                                  <L 172>
    var_15 = wp::mod(var_13, var_3);
    // sphere_radius = sphere_radii[sphere_idx] + radius_margin                               <L 174>
    var_16 = wp::address(var_sphere_radii, var_11);
    var_18 = wp::load(var_16);
    var_17 = wp::add(var_18, var_radius_margin);
    // if sphere_radius <= 0.0:                                                               <L 175>
    var_20 = (var_17 <= var_19);
    if (var_20) {
        // return                                                                             <L 176>
        return;
    }
    // t1 = tri_indices[tri_idx, 0]                                                           <L 178>
    var_22 = wp::address(var_tri_indices, var_15, var_21);
    var_24 = wp::load(var_22);
    var_23 = wp::copy(var_24);
    // t2 = tri_indices[tri_idx, 1]                                                           <L 179>
    var_26 = wp::address(var_tri_indices, var_15, var_25);
    var_28 = wp::load(var_26);
    var_27 = wp::copy(var_28);
    // t3 = tri_indices[tri_idx, 2]                                                           <L 180>
    var_30 = wp::address(var_tri_indices, var_15, var_29);
    var_32 = wp::load(var_30);
    var_31 = wp::copy(var_32);
    // p1 = positions[t1]                                                                     <L 182>
    var_33 = wp::address(var_positions, var_23);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // p2 = positions[t2]                                                                     <L 183>
    var_36 = wp::address(var_positions, var_27);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    // p3 = positions[t3]                                                                     <L 184>
    var_39 = wp::address(var_positions, var_31);
    var_41 = wp::load(var_39);
    var_40 = wp::copy(var_41);
    // w1 = inv_masses[t1]                                                                    <L 186>
    var_42 = wp::address(var_inv_masses, var_23);
    var_44 = wp::load(var_42);
    var_43 = wp::copy(var_44);
    // w2 = inv_masses[t2]                                                                    <L 187>
    var_45 = wp::address(var_inv_masses, var_27);
    var_47 = wp::load(var_45);
    var_46 = wp::copy(var_47);
    // w3 = inv_masses[t3]                                                                    <L 188>
    var_48 = wp::address(var_inv_masses, var_31);
    var_50 = wp::load(var_48);
    var_49 = wp::copy(var_50);
    // weight = w1 + w2 + w3                                                                  <L 189>
    var_51 = wp::add(var_43, var_46);
    var_52 = wp::add(var_51, var_49);
    // if weight <= 0.0:                                                                      <L 190>
    var_54 = (var_52 <= var_53);
    if (var_54) {
        // return                                                                             <L 191>
        return;
    }
    // sample_factor = wp.float32(sample_idx + 1) / wp.float32(sweep_samples)                 <L 193>
    var_56 = wp::add(var_14, var_55);
    var_57 = wp::float32(var_56);
    var_58 = wp::float32(var_sweep_samples);
    var_59 = wp::div(var_57, var_58);
    // sphere_pos = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)       <L 194>
    var_60 = wp::address(var_sphere_centers_prev, var_11);
    var_61 = wp::address(var_sphere_centers, var_11);
    var_63 = wp::load(var_60);
    var_64 = wp::load(var_61);
    var_62 = wp::lerp(var_63, var_64, var_59);
    // if cull_radius > 0.0:                                                                  <L 195>
    var_66 = (var_cull_radius > var_65);
    if (var_66) {
        // centroid = (p1 + p2 + p3) / 3.0                                                    <L 196>
        var_67 = wp::add(var_34, var_37);
        var_68 = wp::add(var_67, var_40);
        var_70 = wp::div(var_68, var_69);
        // if wp.length(centroid - sphere_pos) > cull_radius:                                 <L 197>
        var_71 = wp::sub(var_70, var_62);
        var_72 = wp::length(var_71);
        var_73 = (var_72 > var_cull_radius);
        if (var_73) {
            // return                                                                         <L 198>
            return;
        }
    }
    // closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)         <L 200>
    triangle_closest_point_0(var_34, var_37, var_40, var_62, var_74, var_75, var_76);
    // to_sphere = closest_p - sphere_pos                                                     <L 201>
    var_77 = wp::sub(var_74, var_62);
    // dist = wp.length(to_sphere)                                                            <L 202>
    var_78 = wp::length(var_77);
    // if dist >= sphere_radius:                                                              <L 203>
    var_79 = (var_78 >= var_17);
    if (var_79) {
        // return                                                                             <L 204>
        return;
    }
    // penetration = sphere_radius - dist                                                     <L 206>
    var_80 = wp::sub(var_17, var_78);
    // if dist > 1e-8:                                                                        <L 207>
    var_82 = (var_78 > var_81);
    if (var_82) {
        // correction_dir = to_sphere / dist                                                  <L 208>
        var_83 = wp::div(var_77, var_78);
    }
    if (!var_82) {
        // correction_dir = _double_sided_fallback_dir(                                       <L 210>
        // p1,                                                                                <L 211>
        // p2,                                                                                <L 212>
        // p3,                                                                                <L 213>
        // sphere_pos,                                                                        <L 214>
        // sphere_centers_prev[sphere_idx] - sphere_centers[sphere_idx],                      <L 215>
        var_84 = wp::address(var_sphere_centers_prev, var_11);
        var_85 = wp::address(var_sphere_centers, var_11);
        var_87 = wp::load(var_84);
        var_88 = wp::load(var_85);
        var_86 = wp::sub(var_87, var_88);
        var_89 = _double_sided_fallback_dir_0(var_34, var_37, var_40, var_62, var_86);
    }
    var_90 = wp::where(var_82, var_83, var_89);
    // total_correction = correction_dir * penetration                                        <L 218>
    var_91 = wp::mul(var_90, var_80);
    // d1 = total_correction * (w1 / weight)                                                  <L 219>
    var_92 = wp::div(var_43, var_52);
    var_93 = wp::mul(var_91, var_92);
    // d2 = total_correction * (w2 / weight)                                                  <L 220>
    var_94 = wp::div(var_46, var_52);
    var_95 = wp::mul(var_91, var_94);
    // d3 = total_correction * (w3 / weight)                                                  <L 221>
    var_96 = wp::div(var_49, var_52);
    var_97 = wp::mul(var_91, var_96);
    // wp.atomic_add(delta_accumulator, t1, d1)                                               <L 223>
    var_98 = wp::atomic_add(var_delta_accumulator, var_23, var_93);
    // wp.atomic_add(delta_accumulator, t2, d2)                                               <L 224>
    var_99 = wp::atomic_add(var_delta_accumulator, var_27, var_95);
    // wp.atomic_add(delta_accumulator, t3, d3)                                               <L 225>
    var_100 = wp::atomic_add(var_delta_accumulator, var_31, var_97);
    // wp.atomic_add(delta_counter, t1, 1)                                                    <L 226>
    var_102 = wp::atomic_add(var_delta_counter, var_23, var_101);
    // wp.atomic_add(delta_counter, t2, 1)                                                    <L 227>
    var_104 = wp::atomic_add(var_delta_counter, var_27, var_103);
    // wp.atomic_add(delta_counter, t3, 1)                                                    <L 228>
    var_106 = wp::atomic_add(var_delta_counter, var_31, var_105);
    // wp.atomic_add(reaction_accumulator, 0, -total_correction)                              <L 229>
    var_108 = wp::neg(var_91);
    var_109 = wp::atomic_add(var_reaction_accumulator, var_107, var_108);
    // wp.atomic_add(reaction_counter, 0, 1)                                                  <L 230>
    var_112 = wp::atomic_add(var_reaction_counter, var_110, var_111);
}



void collide_triangles_vs_spheres_636d21ad_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_collide_triangles_vs_spheres_636d21ad *_wp_args,
    wp_args_collide_triangles_vs_spheres_636d21ad *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::float32> var_inv_masses = _wp_args->inv_masses;
    wp::array_t<wp::int32> var_tri_indices = _wp_args->tri_indices;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers_prev = _wp_args->sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_centers = _wp_args->sphere_centers;
    wp::array_t<wp::float32> var_sphere_radii = _wp_args->sphere_radii;
    wp::int32 var_num_spheres = _wp_args->num_spheres;
    wp::int32 var_sweep_samples = _wp_args->sweep_samples;
    wp::float32 var_radius_margin = _wp_args->radius_margin;
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
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_centers_prev = _wp_adj_args->sphere_centers_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_centers = _wp_adj_args->sphere_centers;
    wp::array_t<wp::float32> adj_sphere_radii = _wp_adj_args->sphere_radii;
    wp::int32 adj_num_spheres = _wp_adj_args->num_spheres;
    wp::int32 adj_sweep_samples = _wp_adj_args->sweep_samples;
    wp::float32 adj_radius_margin = _wp_adj_args->radius_margin;
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
    bool var_7;
    const wp::int32 var_8 = 0;
    bool var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    bool var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32 var_15;
    wp::float32* var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    const wp::int32 var_21 = 0;
    wp::int32* var_22;
    wp::int32 var_23;
    wp::int32 var_24;
    const wp::int32 var_25 = 1;
    wp::int32* var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    const wp::int32 var_29 = 2;
    wp::int32* var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32>* var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32>* var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::float32* var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32* var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32* var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    const wp::float32 var_53 = 0.0;
    bool var_54;
    const wp::int32 var_55 = 1;
    wp::int32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::vec_t<3, wp::float32>* var_60;
    wp::vec_t<3, wp::float32>* var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    const wp::float32 var_65 = 0.0;
    bool var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = 3.0;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::float32 var_72;
    bool var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::int32 var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::float32 var_78;
    bool var_79;
    wp::float32 var_80;
    const wp::float32 var_81 = 1e-08;
    bool var_82;
    wp::vec_t<3, wp::float32> var_83;
    wp::vec_t<3, wp::float32>* var_84;
    wp::vec_t<3, wp::float32>* var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::vec_t<3, wp::float32> var_87;
    wp::vec_t<3, wp::float32> var_88;
    wp::vec_t<3, wp::float32> var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::float32 var_92;
    wp::vec_t<3, wp::float32> var_93;
    wp::float32 var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::float32 var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    const wp::int32 var_101 = 1;
    wp::int32 var_102;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    const wp::int32 var_105 = 1;
    wp::int32 var_106;
    const wp::int32 var_107 = 0;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    const wp::int32 var_110 = 0;
    const wp::int32 var_111 = 1;
    wp::int32 var_112;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::shape_t adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::shape_t adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    bool adj_7 = {};
    wp::int32 adj_8 = {};
    bool adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
    bool adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    wp::int32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    bool adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::int32 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::int32 adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::vec_t<3, wp::float32> adj_34 = {};
    wp::vec_t<3, wp::float32> adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::vec_t<3, wp::float32> adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
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
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    bool adj_54 = {};
    wp::int32 adj_55 = {};
    wp::int32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::vec_t<3, wp::float32> adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::vec_t<3, wp::float32> adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::vec_t<3, wp::float32> adj_64 = {};
    wp::float32 adj_65 = {};
    bool adj_66 = {};
    wp::vec_t<3, wp::float32> adj_67 = {};
    wp::vec_t<3, wp::float32> adj_68 = {};
    wp::float32 adj_69 = {};
    wp::vec_t<3, wp::float32> adj_70 = {};
    wp::vec_t<3, wp::float32> adj_71 = {};
    wp::float32 adj_72 = {};
    bool adj_73 = {};
    wp::vec_t<3, wp::float32> adj_74 = {};
    wp::vec_t<3, wp::float32> adj_75 = {};
    wp::int32 adj_76 = {};
    wp::vec_t<3, wp::float32> adj_77 = {};
    wp::float32 adj_78 = {};
    bool adj_79 = {};
    wp::float32 adj_80 = {};
    wp::float32 adj_81 = {};
    bool adj_82 = {};
    wp::vec_t<3, wp::float32> adj_83 = {};
    wp::vec_t<3, wp::float32> adj_84 = {};
    wp::vec_t<3, wp::float32> adj_85 = {};
    wp::vec_t<3, wp::float32> adj_86 = {};
    wp::vec_t<3, wp::float32> adj_87 = {};
    wp::vec_t<3, wp::float32> adj_88 = {};
    wp::vec_t<3, wp::float32> adj_89 = {};
    wp::vec_t<3, wp::float32> adj_90 = {};
    wp::vec_t<3, wp::float32> adj_91 = {};
    wp::float32 adj_92 = {};
    wp::vec_t<3, wp::float32> adj_93 = {};
    wp::float32 adj_94 = {};
    wp::vec_t<3, wp::float32> adj_95 = {};
    wp::float32 adj_96 = {};
    wp::vec_t<3, wp::float32> adj_97 = {};
    wp::vec_t<3, wp::float32> adj_98 = {};
    wp::vec_t<3, wp::float32> adj_99 = {};
    wp::vec_t<3, wp::float32> adj_100 = {};
    wp::int32 adj_101 = {};
    wp::int32 adj_102 = {};
    wp::int32 adj_103 = {};
    wp::int32 adj_104 = {};
    wp::int32 adj_105 = {};
    wp::int32 adj_106 = {};
    wp::int32 adj_107 = {};
    wp::vec_t<3, wp::float32> adj_108 = {};
    wp::vec_t<3, wp::float32> adj_109 = {};
    wp::int32 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::int32 adj_112 = {};
    //---------
    // forward
    // def collide_triangles_vs_spheres(                                                      <L 141>
    // tid = wp.tid()                                                                         <L 160>
    var_0 = builtin_tid1d();
    // tri_count = tri_indices.shape[0]                                                       <L 161>
    var_1 = &(var_tri_indices.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    // if tri_count == 0 or sweep_samples <= 0:                                               <L 162>
    var_7 = (var_3 == var_6);
    var_5 = var_7;
    if (!var_5) {
        var_9 = (var_sweep_samples <= var_8);
        var_5 = var_5 || var_9;
    }
    if (var_5) {
        // return                                                                             <L 163>
        goto label0;
    }
    // work_items_per_sphere = tri_count * sweep_samples                                      <L 165>
    var_10 = wp::mul(var_3, var_sweep_samples);
    // sphere_idx = tid // work_items_per_sphere                                              <L 166>
    var_11 = wp::floordiv(var_0, var_10);
    // if sphere_idx >= num_spheres:                                                          <L 167>
    var_12 = (var_11 >= var_num_spheres);
    if (var_12) {
        // return                                                                             <L 168>
        goto label1;
    }
    // sphere_work_idx = tid % work_items_per_sphere                                          <L 170>
    var_13 = wp::mod(var_0, var_10);
    // sample_idx = sphere_work_idx // tri_count                                              <L 171>
    var_14 = wp::floordiv(var_13, var_3);
    // tri_idx = sphere_work_idx % tri_count                                                  <L 172>
    var_15 = wp::mod(var_13, var_3);
    // sphere_radius = sphere_radii[sphere_idx] + radius_margin                               <L 174>
    var_16 = wp::address(var_sphere_radii, var_11);
    var_18 = wp::load(var_16);
    var_17 = wp::add(var_18, var_radius_margin);
    // if sphere_radius <= 0.0:                                                               <L 175>
    var_20 = (var_17 <= var_19);
    if (var_20) {
        // return                                                                             <L 176>
        goto label2;
    }
    // t1 = tri_indices[tri_idx, 0]                                                           <L 178>
    var_22 = wp::address(var_tri_indices, var_15, var_21);
    var_24 = wp::load(var_22);
    var_23 = wp::copy(var_24);
    // t2 = tri_indices[tri_idx, 1]                                                           <L 179>
    var_26 = wp::address(var_tri_indices, var_15, var_25);
    var_28 = wp::load(var_26);
    var_27 = wp::copy(var_28);
    // t3 = tri_indices[tri_idx, 2]                                                           <L 180>
    var_30 = wp::address(var_tri_indices, var_15, var_29);
    var_32 = wp::load(var_30);
    var_31 = wp::copy(var_32);
    // p1 = positions[t1]                                                                     <L 182>
    var_33 = wp::address(var_positions, var_23);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // p2 = positions[t2]                                                                     <L 183>
    var_36 = wp::address(var_positions, var_27);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    // p3 = positions[t3]                                                                     <L 184>
    var_39 = wp::address(var_positions, var_31);
    var_41 = wp::load(var_39);
    var_40 = wp::copy(var_41);
    // w1 = inv_masses[t1]                                                                    <L 186>
    var_42 = wp::address(var_inv_masses, var_23);
    var_44 = wp::load(var_42);
    var_43 = wp::copy(var_44);
    // w2 = inv_masses[t2]                                                                    <L 187>
    var_45 = wp::address(var_inv_masses, var_27);
    var_47 = wp::load(var_45);
    var_46 = wp::copy(var_47);
    // w3 = inv_masses[t3]                                                                    <L 188>
    var_48 = wp::address(var_inv_masses, var_31);
    var_50 = wp::load(var_48);
    var_49 = wp::copy(var_50);
    // weight = w1 + w2 + w3                                                                  <L 189>
    var_51 = wp::add(var_43, var_46);
    var_52 = wp::add(var_51, var_49);
    // if weight <= 0.0:                                                                      <L 190>
    var_54 = (var_52 <= var_53);
    if (var_54) {
        // return                                                                             <L 191>
        goto label3;
    }
    // sample_factor = wp.float32(sample_idx + 1) / wp.float32(sweep_samples)                 <L 193>
    var_56 = wp::add(var_14, var_55);
    var_57 = wp::float32(var_56);
    var_58 = wp::float32(var_sweep_samples);
    var_59 = wp::div(var_57, var_58);
    // sphere_pos = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)       <L 194>
    var_60 = wp::address(var_sphere_centers_prev, var_11);
    var_61 = wp::address(var_sphere_centers, var_11);
    var_63 = wp::load(var_60);
    var_64 = wp::load(var_61);
    var_62 = wp::lerp(var_63, var_64, var_59);
    // if cull_radius > 0.0:                                                                  <L 195>
    var_66 = (var_cull_radius > var_65);
    if (var_66) {
        // centroid = (p1 + p2 + p3) / 3.0                                                    <L 196>
        var_67 = wp::add(var_34, var_37);
        var_68 = wp::add(var_67, var_40);
        var_70 = wp::div(var_68, var_69);
        // if wp.length(centroid - sphere_pos) > cull_radius:                                 <L 197>
        var_71 = wp::sub(var_70, var_62);
        var_72 = wp::length(var_71);
        var_73 = (var_72 > var_cull_radius);
        if (var_73) {
            // return                                                                         <L 198>
            goto label4;
        }
    }
    // closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)         <L 200>
    triangle_closest_point_0(var_34, var_37, var_40, var_62, var_74, var_75, var_76);
    // to_sphere = closest_p - sphere_pos                                                     <L 201>
    var_77 = wp::sub(var_74, var_62);
    // dist = wp.length(to_sphere)                                                            <L 202>
    var_78 = wp::length(var_77);
    // if dist >= sphere_radius:                                                              <L 203>
    var_79 = (var_78 >= var_17);
    if (var_79) {
        // return                                                                             <L 204>
        goto label5;
    }
    // penetration = sphere_radius - dist                                                     <L 206>
    var_80 = wp::sub(var_17, var_78);
    // if dist > 1e-8:                                                                        <L 207>
    var_82 = (var_78 > var_81);
    if (var_82) {
        // correction_dir = to_sphere / dist                                                  <L 208>
        var_83 = wp::div(var_77, var_78);
    }
    if (!var_82) {
        // correction_dir = _double_sided_fallback_dir(                                       <L 210>
        // p1,                                                                                <L 211>
        // p2,                                                                                <L 212>
        // p3,                                                                                <L 213>
        // sphere_pos,                                                                        <L 214>
        // sphere_centers_prev[sphere_idx] - sphere_centers[sphere_idx],                      <L 215>
        var_84 = wp::address(var_sphere_centers_prev, var_11);
        var_85 = wp::address(var_sphere_centers, var_11);
        var_87 = wp::load(var_84);
        var_88 = wp::load(var_85);
        var_86 = wp::sub(var_87, var_88);
        var_89 = _double_sided_fallback_dir_0(var_34, var_37, var_40, var_62, var_86);
    }
    var_90 = wp::where(var_82, var_83, var_89);
    // total_correction = correction_dir * penetration                                        <L 218>
    var_91 = wp::mul(var_90, var_80);
    // d1 = total_correction * (w1 / weight)                                                  <L 219>
    var_92 = wp::div(var_43, var_52);
    var_93 = wp::mul(var_91, var_92);
    // d2 = total_correction * (w2 / weight)                                                  <L 220>
    var_94 = wp::div(var_46, var_52);
    var_95 = wp::mul(var_91, var_94);
    // d3 = total_correction * (w3 / weight)                                                  <L 221>
    var_96 = wp::div(var_49, var_52);
    var_97 = wp::mul(var_91, var_96);
    // wp.atomic_add(delta_accumulator, t1, d1)                                               <L 223>
    // var_98 = wp::atomic_add(var_delta_accumulator, var_23, var_93);
    // wp.atomic_add(delta_accumulator, t2, d2)                                               <L 224>
    // var_99 = wp::atomic_add(var_delta_accumulator, var_27, var_95);
    // wp.atomic_add(delta_accumulator, t3, d3)                                               <L 225>
    // var_100 = wp::atomic_add(var_delta_accumulator, var_31, var_97);
    // wp.atomic_add(delta_counter, t1, 1)                                                    <L 226>
    // var_102 = wp::atomic_add(var_delta_counter, var_23, var_101);
    // wp.atomic_add(delta_counter, t2, 1)                                                    <L 227>
    // var_104 = wp::atomic_add(var_delta_counter, var_27, var_103);
    // wp.atomic_add(delta_counter, t3, 1)                                                    <L 228>
    // var_106 = wp::atomic_add(var_delta_counter, var_31, var_105);
    // wp.atomic_add(reaction_accumulator, 0, -total_correction)                              <L 229>
    var_108 = wp::neg(var_91);
    // var_109 = wp::atomic_add(var_reaction_accumulator, var_107, var_108);
    // wp.atomic_add(reaction_counter, 0, 1)                                                  <L 230>
    // var_112 = wp::atomic_add(var_reaction_counter, var_110, var_111);
    //---------
    // reverse
    wp::adj_atomic_add(var_reaction_counter, var_110, var_111, adj_reaction_counter, adj_110, adj_111, adj_112);
    // adj: wp.atomic_add(reaction_counter, 0, 1)                                             <L 230>
    wp::adj_atomic_add(var_reaction_accumulator, var_107, var_108, adj_reaction_accumulator, adj_107, adj_108, adj_109);
    wp::adj_neg(var_91, adj_91, adj_108);
    // adj: wp.atomic_add(reaction_accumulator, 0, -total_correction)                         <L 229>
    wp::adj_atomic_add(var_delta_counter, var_31, var_105, adj_delta_counter, adj_31, adj_105, adj_106);
    // adj: wp.atomic_add(delta_counter, t3, 1)                                               <L 228>
    wp::adj_atomic_add(var_delta_counter, var_27, var_103, adj_delta_counter, adj_27, adj_103, adj_104);
    // adj: wp.atomic_add(delta_counter, t2, 1)                                               <L 227>
    wp::adj_atomic_add(var_delta_counter, var_23, var_101, adj_delta_counter, adj_23, adj_101, adj_102);
    // adj: wp.atomic_add(delta_counter, t1, 1)                                               <L 226>
    wp::adj_atomic_add(var_delta_accumulator, var_31, var_97, adj_delta_accumulator, adj_31, adj_97, adj_100);
    // adj: wp.atomic_add(delta_accumulator, t3, d3)                                          <L 225>
    wp::adj_atomic_add(var_delta_accumulator, var_27, var_95, adj_delta_accumulator, adj_27, adj_95, adj_99);
    // adj: wp.atomic_add(delta_accumulator, t2, d2)                                          <L 224>
    wp::adj_atomic_add(var_delta_accumulator, var_23, var_93, adj_delta_accumulator, adj_23, adj_93, adj_98);
    // adj: wp.atomic_add(delta_accumulator, t1, d1)                                          <L 223>
    wp::adj_mul(var_91, var_96, adj_91, adj_96, adj_97);
    wp::adj_div(var_49, var_52, var_96, adj_49, adj_52, adj_96);
    // adj: d3 = total_correction * (w3 / weight)                                             <L 221>
    wp::adj_mul(var_91, var_94, adj_91, adj_94, adj_95);
    wp::adj_div(var_46, var_52, var_94, adj_46, adj_52, adj_94);
    // adj: d2 = total_correction * (w2 / weight)                                             <L 220>
    wp::adj_mul(var_91, var_92, adj_91, adj_92, adj_93);
    wp::adj_div(var_43, var_52, var_92, adj_43, adj_52, adj_92);
    // adj: d1 = total_correction * (w1 / weight)                                             <L 219>
    wp::adj_mul(var_90, var_80, adj_90, adj_80, adj_91);
    // adj: total_correction = correction_dir * penetration                                   <L 218>
    wp::adj_where(var_82, var_83, var_89, adj_82, adj_83, adj_89, adj_90);
    if (!var_82) {
        adj__double_sided_fallback_dir_0(var_34, var_37, var_40, var_62, var_86, adj_34, adj_37, adj_40, adj_62, adj_86, adj_89);
        wp::adj_sub(var_87, var_88, adj_84, adj_85, adj_86);
        wp::adj_address(var_sphere_centers, var_11, adj_sphere_centers, adj_11, adj_85);
        wp::adj_address(var_sphere_centers_prev, var_11, adj_sphere_centers_prev, adj_11, adj_84);
        // adj: sphere_centers_prev[sphere_idx] - sphere_centers[sphere_idx],                 <L 215>
        // adj: sphere_pos,                                                                   <L 214>
        // adj: p3,                                                                           <L 213>
        // adj: p2,                                                                           <L 212>
        // adj: p1,                                                                           <L 211>
        // adj: correction_dir = _double_sided_fallback_dir(                                  <L 210>
    }
    if (var_82) {
        wp::adj_div(var_77, var_78, adj_77, adj_78, adj_83);
        // adj: correction_dir = to_sphere / dist                                             <L 208>
    }
    // adj: if dist > 1e-8:                                                                   <L 207>
    wp::adj_sub(var_17, var_78, adj_17, adj_78, adj_80);
    // adj: penetration = sphere_radius - dist                                                <L 206>
    if (var_79) {
        label5:;
        // adj: return                                                                        <L 204>
    }
    // adj: if dist >= sphere_radius:                                                         <L 203>
    wp::adj_length(var_77, var_78, adj_77, adj_78);
    // adj: dist = wp.length(to_sphere)                                                       <L 202>
    wp::adj_sub(var_74, var_62, adj_74, adj_62, adj_77);
    // adj: to_sphere = closest_p - sphere_pos                                                <L 201>
    adj_triangle_closest_point_0(var_34, var_37, var_40, var_62, var_74, var_75, var_76, adj_34, adj_37, adj_40, adj_62, adj_74, adj_75, adj_76);
    // adj: closest_p, bary, feature_type = triangle_closest_point(p1, p2, p3, sphere_pos)    <L 200>
    if (var_66) {
        if (var_73) {
            label4:;
            // adj: return                                                                    <L 198>
        }
        wp::adj_length(var_71, var_72, adj_71, adj_72);
        wp::adj_sub(var_70, var_62, adj_70, adj_62, adj_71);
        // adj: if wp.length(centroid - sphere_pos) > cull_radius:                            <L 197>
        wp::adj_div(var_68, var_69, adj_68, adj_69, adj_70);
        wp::adj_add(var_67, var_40, adj_67, adj_40, adj_68);
        wp::adj_add(var_34, var_37, adj_34, adj_37, adj_67);
        // adj: centroid = (p1 + p2 + p3) / 3.0                                               <L 196>
    }
    // adj: if cull_radius > 0.0:                                                             <L 195>
    wp::adj_lerp(var_63, var_64, var_59, adj_60, adj_61, adj_59, adj_62);
    wp::adj_address(var_sphere_centers, var_11, adj_sphere_centers, adj_11, adj_61);
    wp::adj_address(var_sphere_centers_prev, var_11, adj_sphere_centers_prev, adj_11, adj_60);
    // adj: sphere_pos = wp.lerp(sphere_centers_prev[sphere_idx], sphere_centers[sphere_idx], sample_factor)  <L 194>
    wp::adj_div(var_57, var_58, var_59, adj_57, adj_58, adj_59);
    wp::adj_float32(var_sweep_samples, adj_sweep_samples, adj_58);
    wp::adj_float32(var_56, adj_56, adj_57);
    wp::adj_add(var_14, var_55, adj_14, adj_55, adj_56);
    // adj: sample_factor = wp.float32(sample_idx + 1) / wp.float32(sweep_samples)            <L 193>
    if (var_54) {
        label3:;
        // adj: return                                                                        <L 191>
    }
    // adj: if weight <= 0.0:                                                                 <L 190>
    wp::adj_add(var_51, var_49, adj_51, adj_49, adj_52);
    wp::adj_add(var_43, var_46, adj_43, adj_46, adj_51);
    // adj: weight = w1 + w2 + w3                                                             <L 189>
    wp::adj_copy(var_50, adj_48, adj_49);
    wp::adj_address(var_inv_masses, var_31, adj_inv_masses, adj_31, adj_48);
    // adj: w3 = inv_masses[t3]                                                               <L 188>
    wp::adj_copy(var_47, adj_45, adj_46);
    wp::adj_address(var_inv_masses, var_27, adj_inv_masses, adj_27, adj_45);
    // adj: w2 = inv_masses[t2]                                                               <L 187>
    wp::adj_copy(var_44, adj_42, adj_43);
    wp::adj_address(var_inv_masses, var_23, adj_inv_masses, adj_23, adj_42);
    // adj: w1 = inv_masses[t1]                                                               <L 186>
    wp::adj_copy(var_41, adj_39, adj_40);
    wp::adj_address(var_positions, var_31, adj_positions, adj_31, adj_39);
    // adj: p3 = positions[t3]                                                                <L 184>
    wp::adj_copy(var_38, adj_36, adj_37);
    wp::adj_address(var_positions, var_27, adj_positions, adj_27, adj_36);
    // adj: p2 = positions[t2]                                                                <L 183>
    wp::adj_copy(var_35, adj_33, adj_34);
    wp::adj_address(var_positions, var_23, adj_positions, adj_23, adj_33);
    // adj: p1 = positions[t1]                                                                <L 182>
    wp::adj_copy(var_32, adj_30, adj_31);
    wp::adj_address(var_tri_indices, var_15, var_29, adj_tri_indices, adj_15, adj_29, adj_30);
    // adj: t3 = tri_indices[tri_idx, 2]                                                      <L 180>
    wp::adj_copy(var_28, adj_26, adj_27);
    wp::adj_address(var_tri_indices, var_15, var_25, adj_tri_indices, adj_15, adj_25, adj_26);
    // adj: t2 = tri_indices[tri_idx, 1]                                                      <L 179>
    wp::adj_copy(var_24, adj_22, adj_23);
    wp::adj_address(var_tri_indices, var_15, var_21, adj_tri_indices, adj_15, adj_21, adj_22);
    // adj: t1 = tri_indices[tri_idx, 0]                                                      <L 178>
    if (var_20) {
        label2:;
        // adj: return                                                                        <L 176>
    }
    // adj: if sphere_radius <= 0.0:                                                          <L 175>
    wp::adj_add(var_18, var_radius_margin, adj_16, adj_radius_margin, adj_17);
    wp::adj_address(var_sphere_radii, var_11, adj_sphere_radii, adj_11, adj_16);
    // adj: sphere_radius = sphere_radii[sphere_idx] + radius_margin                          <L 174>
    wp::adj_mod(var_13, var_3, adj_13, adj_3, adj_15);
    // adj: tri_idx = sphere_work_idx % tri_count                                             <L 172>
    // adj: sample_idx = sphere_work_idx // tri_count                                         <L 171>
    wp::adj_mod(var_0, var_10, adj_0, adj_10, adj_13);
    // adj: sphere_work_idx = tid % work_items_per_sphere                                     <L 170>
    if (var_12) {
        label1:;
        // adj: return                                                                        <L 168>
    }
    // adj: if sphere_idx >= num_spheres:                                                     <L 167>
    // adj: sphere_idx = tid // work_items_per_sphere                                         <L 166>
    wp::adj_mul(var_3, var_sweep_samples, adj_3, adj_sweep_samples, adj_10);
    // adj: work_items_per_sphere = tri_count * sweep_samples                                 <L 165>
    if (var_5) {
        label0:;
        // adj: return                                                                        <L 163>
    }
    if (!var_5) {
    }
    // adj: if tri_count == 0 or sweep_samples <= 0:                                          <L 162>
    wp::adj_extract(var_4, var_2, adj_1, adj_2, adj_3);
    adj_tri_indices.shape = adj_1;
    // adj: tri_count = tri_indices.shape[0]                                                  <L 161>
    // adj: tid = wp.tid()                                                                    <L 160>
    // adj: def collide_triangles_vs_spheres(                                                 <L 141>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void collide_triangles_vs_spheres_636d21ad_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_collide_triangles_vs_spheres_636d21ad *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        collide_triangles_vs_spheres_636d21ad_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void collide_triangles_vs_spheres_636d21ad_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_collide_triangles_vs_spheres_636d21ad *_wp_args,
    wp_args_collide_triangles_vs_spheres_636d21ad *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        collide_triangles_vs_spheres_636d21ad_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
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
    // def apply_surface_vertex_truncation(                                                   <L 323>
    // tid = wp.tid()                                                                         <L 328>
    var_0 = builtin_tid1d();
    // if tid >= surface_vertex_ids.shape[0]:                                                 <L 329>
    var_1 = &(var_surface_vertex_ids.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = (var_0 >= var_3);
    if (var_5) {
        // return                                                                             <L 330>
        return;
    }
    // vertex_id = surface_vertex_ids[tid]                                                    <L 332>
    var_6 = wp::address(var_surface_vertex_ids, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]          <L 333>
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
    // def apply_surface_vertex_truncation(                                                   <L 323>
    // tid = wp.tid()                                                                         <L 328>
    var_0 = builtin_tid1d();
    // if tid >= surface_vertex_ids.shape[0]:                                                 <L 329>
    var_1 = &(var_surface_vertex_ids.shape);
    var_4 = wp::load(var_1);
    var_3 = wp::extract(var_4, var_2);
    var_5 = (var_0 >= var_3);
    if (var_5) {
        // return                                                                             <L 330>
        goto label0;
    }
    // vertex_id = surface_vertex_ids[tid]                                                    <L 332>
    var_6 = wp::address(var_surface_vertex_ids, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]          <L 333>
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
    // adj: displacements[vertex_id] = displacements[vertex_id] * truncation_t[vertex_id]     <L 333>
    wp::adj_copy(var_8, adj_6, adj_7);
    wp::adj_address(var_surface_vertex_ids, var_0, adj_surface_vertex_ids, adj_0, adj_6);
    // adj: vertex_id = surface_vertex_ids[tid]                                               <L 332>
    if (var_5) {
        label0:;
        // adj: return                                                                        <L 330>
    }
    wp::adj_extract(var_4, var_2, adj_1, adj_2, adj_3);
    adj_surface_vertex_ids.shape = adj_1;
    // adj: if tid >= surface_vertex_ids.shape[0]:                                            <L 329>
    // adj: tid = wp.tid()                                                                    <L 328>
    // adj: def apply_surface_vertex_truncation(                                              <L 323>
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

