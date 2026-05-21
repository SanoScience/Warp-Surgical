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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:20
static CUDA_CALLABLE wp::quat_t<wp::float32> _extract_rotation_0(
    wp::mat_t<3, 3, wp::float32> var_a,
    wp::quat_t<wp::float32> var_q_init,
    wp::int32 var_max_iters)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 3;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::int32 var_6 = 2;
    wp::float32 var_7;
    wp::vec_t<4, wp::float32> var_8;
    wp::float32 var_9;
    const wp::float32 var_10 = 1e-12;
    bool var_11;
    const wp::float32 var_12 = 1.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.0;
    const wp::float32 var_15 = 0.0;
    wp::vec_t<4, wp::float32> var_16;
    wp::vec_t<4, wp::float32> var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::vec_t<4, wp::float32> var_20;
    wp::vec_t<4, wp::float32> var_21;
    wp::int32 var_22;
    const wp::int32 var_23 = 0;
    bool var_24;
    const wp::int32 var_25 = 0;
    wp::int32 var_26;
    const wp::int32 var_27 = 16;
    bool var_28;
    const wp::int32 var_29 = 16;
    wp::int32 var_30;
    const wp::int32 var_31 = 0;
    const wp::int32 var_32 = 0;
    wp::float32 var_33;
    const wp::int32 var_34 = 0;
    const wp::int32 var_35 = 1;
    wp::float32 var_36;
    const wp::int32 var_37 = 0;
    const wp::int32 var_38 = 2;
    wp::float32 var_39;
    const wp::int32 var_40 = 1;
    const wp::int32 var_41 = 0;
    wp::float32 var_42;
    const wp::int32 var_43 = 1;
    const wp::int32 var_44 = 1;
    wp::float32 var_45;
    const wp::int32 var_46 = 1;
    const wp::int32 var_47 = 2;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    const wp::int32 var_50 = 0;
    wp::float32 var_51;
    const wp::int32 var_52 = 2;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    const wp::int32 var_56 = 2;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    wp::float32 var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    wp::float32 var_68;
    wp::float32 var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    const wp::int32 var_74 = 16;
    wp::range_t var_75;
    wp::int32 var_76;
    bool var_77;
    const wp::int32 var_78 = 0;
    wp::float32 var_79;
    wp::float32 var_80;
    const wp::int32 var_81 = 1;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    const wp::int32 var_85 = 2;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    const wp::int32 var_89 = 3;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    const wp::int32 var_93 = 0;
    wp::float32 var_94;
    wp::float32 var_95;
    const wp::int32 var_96 = 1;
    wp::float32 var_97;
    wp::float32 var_98;
    wp::float32 var_99;
    const wp::int32 var_100 = 2;
    wp::float32 var_101;
    wp::float32 var_102;
    wp::float32 var_103;
    const wp::int32 var_104 = 3;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    const wp::int32 var_108 = 0;
    wp::float32 var_109;
    wp::float32 var_110;
    const wp::int32 var_111 = 1;
    wp::float32 var_112;
    wp::float32 var_113;
    wp::float32 var_114;
    const wp::int32 var_115 = 2;
    wp::float32 var_116;
    wp::float32 var_117;
    wp::float32 var_118;
    const wp::int32 var_119 = 3;
    wp::float32 var_120;
    wp::float32 var_121;
    wp::float32 var_122;
    const wp::int32 var_123 = 0;
    wp::float32 var_124;
    wp::float32 var_125;
    const wp::int32 var_126 = 1;
    wp::float32 var_127;
    wp::float32 var_128;
    wp::float32 var_129;
    const wp::int32 var_130 = 2;
    wp::float32 var_131;
    wp::float32 var_132;
    wp::float32 var_133;
    const wp::int32 var_134 = 3;
    wp::float32 var_135;
    wp::float32 var_136;
    wp::float32 var_137;
    wp::vec_t<4, wp::float32> var_138;
    wp::float32 var_139;
    const wp::float32 var_140 = 1e-20;
    bool var_141;
    wp::float32 var_142;
    wp::vec_t<4, wp::float32> var_143;
    const wp::int32 var_144 = 1;
    wp::float32 var_145;
    const wp::int32 var_146 = 2;
    wp::float32 var_147;
    const wp::int32 var_148 = 3;
    wp::float32 var_149;
    const wp::int32 var_150 = 0;
    wp::float32 var_151;
    wp::quat_t<wp::float32> var_152;
    //---------
    // forward
    // def _extract_rotation(a: wp.mat33, q_init: wp.quat, max_iters: int) -> wp.quat:        <L 21>
    // q4 = wp.vec4(q_init[3], q_init[0], q_init[1], q_init[2])                               <L 23>
    var_1 = wp::extract(var_q_init, var_0);
    var_3 = wp::extract(var_q_init, var_2);
    var_5 = wp::extract(var_q_init, var_4);
    var_7 = wp::extract(var_q_init, var_6);
    var_8 = wp::vec_t<4, wp::float32>(var_1, var_3, var_5, var_7);
    // if wp.dot(q4, q4) < 1.0e-12:                                                           <L 24>
    var_9 = wp::dot(var_8, var_8);
    var_11 = (var_9 < var_10);
    if (var_11) {
        // q4 = wp.vec4(1.0, 0.0, 0.0, 0.0)                                                   <L 25>
        var_16 = wp::vec_t<4, wp::float32>(var_12, var_13, var_14, var_15);
    }
    var_17 = wp::where(var_11, var_16, var_8);
    if (!var_11) {
        // q4 = q4 / wp.sqrt(wp.dot(q4, q4))                                                  <L 27>
        var_18 = wp::dot(var_17, var_17);
        var_19 = wp::sqrt(var_18);
        var_20 = wp::div(var_17, var_19);
    }
    var_21 = wp::where(var_11, var_17, var_20);
    // iters = max_iters                                                                      <L 29>
    var_22 = wp::copy(var_max_iters);
    // if iters < 0:                                                                          <L 30>
    var_24 = (var_22 < var_23);
    if (var_24) {
        // iters = 0                                                                          <L 31>
    }
    var_26 = wp::where(var_24, var_25, var_22);
    // if iters > wp.static(MAX_ROTATION_ITERS):                                              <L 32>
    var_28 = (var_26 > var_27);
    if (var_28) {
        // iters = wp.static(MAX_ROTATION_ITERS)                                              <L 33>
    }
    var_30 = wp::where(var_28, var_29, var_26);
    // a00 = a[0, 0]                                                                          <L 35>
    var_33 = wp::extract(var_a, var_31, var_32);
    // a01 = a[0, 1]                                                                          <L 36>
    var_36 = wp::extract(var_a, var_34, var_35);
    // a02 = a[0, 2]                                                                          <L 37>
    var_39 = wp::extract(var_a, var_37, var_38);
    // a10 = a[1, 0]                                                                          <L 38>
    var_42 = wp::extract(var_a, var_40, var_41);
    // a11 = a[1, 1]                                                                          <L 39>
    var_45 = wp::extract(var_a, var_43, var_44);
    // a12 = a[1, 2]                                                                          <L 40>
    var_48 = wp::extract(var_a, var_46, var_47);
    // a20 = a[2, 0]                                                                          <L 41>
    var_51 = wp::extract(var_a, var_49, var_50);
    // a21 = a[2, 1]                                                                          <L 42>
    var_54 = wp::extract(var_a, var_52, var_53);
    // a22 = a[2, 2]                                                                          <L 43>
    var_57 = wp::extract(var_a, var_55, var_56);
    // k00 = a00 + a11 + a22                                                                  <L 45>
    var_58 = wp::add(var_33, var_45);
    var_59 = wp::add(var_58, var_57);
    // k01 = a12 - a21                                                                        <L 46>
    var_60 = wp::sub(var_48, var_54);
    // k02 = a20 - a02                                                                        <L 47>
    var_61 = wp::sub(var_51, var_39);
    // k03 = a01 - a10                                                                        <L 48>
    var_62 = wp::sub(var_36, var_42);
    // k11 = a00 - a11 - a22                                                                  <L 49>
    var_63 = wp::sub(var_33, var_45);
    var_64 = wp::sub(var_63, var_57);
    // k12 = a01 + a10                                                                        <L 50>
    var_65 = wp::add(var_36, var_42);
    // k13 = a02 + a20                                                                        <L 51>
    var_66 = wp::add(var_39, var_51);
    // k22 = -a00 + a11 - a22                                                                 <L 52>
    var_67 = wp::neg(var_33);
    var_68 = wp::add(var_67, var_45);
    var_69 = wp::sub(var_68, var_57);
    // k23 = a12 + a21                                                                        <L 53>
    var_70 = wp::add(var_48, var_54);
    // k33 = -a00 - a11 + a22                                                                 <L 54>
    var_71 = wp::neg(var_33);
    var_72 = wp::sub(var_71, var_45);
    var_73 = wp::add(var_72, var_57);
    // for it in range(wp.static(MAX_ROTATION_ITERS)):                                        <L 56>
    var_75 = wp::range(var_74);
    start_for_0:;
        if (iter_cmp(var_75) == 0) goto end_for_0;
        var_76 = wp::iter_next(var_75);
        // if it >= iters:                                                                    <L 57>
        var_77 = (var_76 >= var_30);
        if (var_77) {
            // break                                                                          <L 58>
            goto end_for_0;
        }
        // next_q4 = wp.vec4(                                                                 <L 60>
        // k00 * q4[0] + k01 * q4[1] + k02 * q4[2] + k03 * q4[3],                             <L 61>
        var_79 = wp::extract(var_21, var_78);
        var_80 = wp::mul(var_59, var_79);
        var_82 = wp::extract(var_21, var_81);
        var_83 = wp::mul(var_60, var_82);
        var_84 = wp::add(var_80, var_83);
        var_86 = wp::extract(var_21, var_85);
        var_87 = wp::mul(var_61, var_86);
        var_88 = wp::add(var_84, var_87);
        var_90 = wp::extract(var_21, var_89);
        var_91 = wp::mul(var_62, var_90);
        var_92 = wp::add(var_88, var_91);
        // k01 * q4[0] + k11 * q4[1] + k12 * q4[2] + k13 * q4[3],                             <L 62>
        var_94 = wp::extract(var_21, var_93);
        var_95 = wp::mul(var_60, var_94);
        var_97 = wp::extract(var_21, var_96);
        var_98 = wp::mul(var_64, var_97);
        var_99 = wp::add(var_95, var_98);
        var_101 = wp::extract(var_21, var_100);
        var_102 = wp::mul(var_65, var_101);
        var_103 = wp::add(var_99, var_102);
        var_105 = wp::extract(var_21, var_104);
        var_106 = wp::mul(var_66, var_105);
        var_107 = wp::add(var_103, var_106);
        // k02 * q4[0] + k12 * q4[1] + k22 * q4[2] + k23 * q4[3],                             <L 63>
        var_109 = wp::extract(var_21, var_108);
        var_110 = wp::mul(var_61, var_109);
        var_112 = wp::extract(var_21, var_111);
        var_113 = wp::mul(var_65, var_112);
        var_114 = wp::add(var_110, var_113);
        var_116 = wp::extract(var_21, var_115);
        var_117 = wp::mul(var_69, var_116);
        var_118 = wp::add(var_114, var_117);
        var_120 = wp::extract(var_21, var_119);
        var_121 = wp::mul(var_70, var_120);
        var_122 = wp::add(var_118, var_121);
        // k03 * q4[0] + k13 * q4[1] + k23 * q4[2] + k33 * q4[3],                             <L 64>
        var_124 = wp::extract(var_21, var_123);
        var_125 = wp::mul(var_62, var_124);
        var_127 = wp::extract(var_21, var_126);
        var_128 = wp::mul(var_66, var_127);
        var_129 = wp::add(var_125, var_128);
        var_131 = wp::extract(var_21, var_130);
        var_132 = wp::mul(var_70, var_131);
        var_133 = wp::add(var_129, var_132);
        var_135 = wp::extract(var_21, var_134);
        var_136 = wp::mul(var_73, var_135);
        var_137 = wp::add(var_133, var_136);
        var_138 = wp::vec_t<4, wp::float32>(var_92, var_107, var_122, var_137);
        // next_norm_sq = wp.dot(next_q4, next_q4)                                            <L 66>
        var_139 = wp::dot(var_138, var_138);
        // if next_norm_sq < 1.0e-20:                                                         <L 67>
        var_141 = (var_139 < var_140);
        if (var_141) {
            // break                                                                          <L 68>
            goto end_for_0;
        }
        // q4 = next_q4 / wp.sqrt(next_norm_sq)                                               <L 69>
        var_142 = wp::sqrt(var_139);
        var_143 = wp::div(var_138, var_142);
        wp::assign(var_21, var_143);
        goto start_for_0;
    end_for_0:;
    // return wp.quat(q4[1], q4[2], q4[3], q4[0])                                             <L 71>
    var_145 = wp::extract(var_21, var_144);
    var_147 = wp::extract(var_21, var_146);
    var_149 = wp::extract(var_21, var_148);
    var_151 = wp::extract(var_21, var_150);
    var_152 = wp::quat_t<wp::float32>(var_145, var_147, var_149, var_151);
    return var_152;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:15
static CUDA_CALLABLE wp::float32 _quat_dot_0(
    wp::quat_t<wp::float32> var_a,
    wp::quat_t<wp::float32> var_b)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    const wp::int32 var_7 = 1;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    const wp::int32 var_11 = 2;
    wp::float32 var_12;
    const wp::int32 var_13 = 2;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    const wp::int32 var_17 = 3;
    wp::float32 var_18;
    const wp::int32 var_19 = 3;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    //---------
    // forward
    // def _quat_dot(a: wp.quat, b: wp.quat) -> float:                                        <L 16>
    // return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]                           <L 17>
    var_1 = wp::extract(var_a, var_0);
    var_3 = wp::extract(var_b, var_2);
    var_4 = wp::mul(var_1, var_3);
    var_6 = wp::extract(var_a, var_5);
    var_8 = wp::extract(var_b, var_7);
    var_9 = wp::mul(var_6, var_8);
    var_10 = wp::add(var_4, var_9);
    var_12 = wp::extract(var_a, var_11);
    var_14 = wp::extract(var_b, var_13);
    var_15 = wp::mul(var_12, var_14);
    var_16 = wp::add(var_10, var_15);
    var_18 = wp::extract(var_a, var_17);
    var_20 = wp::extract(var_b, var_19);
    var_21 = wp::mul(var_18, var_20);
    var_22 = wp::add(var_16, var_21);
    return var_22;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:74
static CUDA_CALLABLE wp::float32 _support_scale_from_alpha_0(
    wp::float32 var_inv_weight,
    wp::float32 var_support_alpha)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    bool var_4;
    const wp::float32 var_5 = 1.0;
    const wp::float32 var_6 = 1.0;
    bool var_7;
    const wp::float32 var_8 = 0.5;
    bool var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    //---------
    // forward
    // def _support_scale_from_alpha(inv_weight: float, support_alpha: float) -> float:       <L 75>
    // if inv_weight <= 0.0:                                                                  <L 76>
    var_1 = (var_inv_weight <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 77>
        return var_2;
    }
    // if support_alpha <= 0.0:                                                               <L 78>
    var_4 = (var_support_alpha <= var_3);
    if (var_4) {
        // return 1.0                                                                         <L 79>
        return var_5;
    }
    // if support_alpha >= 1.0:                                                               <L 80>
    var_7 = (var_support_alpha >= var_6);
    if (var_7) {
        // return inv_weight                                                                  <L 81>
        return var_inv_weight;
    }
    // if support_alpha == 0.5:                                                               <L 82>
    var_9 = (var_support_alpha == var_8);
    if (var_9) {
        // return wp.sqrt(inv_weight)                                                         <L 83>
        var_10 = wp::sqrt(var_inv_weight);
        return var_10;
    }
    // return wp.pow(inv_weight, support_alpha)                                               <L 84>
    var_11 = wp::pow(var_inv_weight, var_support_alpha);
    return var_11;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:87
static CUDA_CALLABLE wp::float32 _uniform8_slot_sign_x_0(
    wp::int32 var_slot)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    const wp::int32 var_3 = 2;
    bool var_4;
    const wp::int32 var_5 = 5;
    bool var_6;
    const wp::int32 var_7 = 6;
    bool var_8;
    const wp::float32 var_9 = 1.0;
    const wp::float32 var_10 = -1.0;
    //---------
    // forward
    // def _uniform8_slot_sign_x(slot: int) -> float:                                         <L 88>
    // if slot == 1 or slot == 2 or slot == 5 or slot == 6:                                   <L 89>
    var_2 = (var_slot == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_slot == var_3);
        var_0 = var_0 || var_4;
    }
    if (!var_0) {
        var_6 = (var_slot == var_5);
        var_0 = var_0 || var_6;
    }
    if (!var_0) {
        var_8 = (var_slot == var_7);
        var_0 = var_0 || var_8;
    }
    if (var_0) {
        // return 1.0                                                                         <L 90>
        return var_9;
    }
    // return -1.0                                                                            <L 91>
    return var_10;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:94
static CUDA_CALLABLE wp::float32 _uniform8_slot_sign_y_0(
    wp::int32 var_slot)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 2;
    bool var_2;
    const wp::int32 var_3 = 3;
    bool var_4;
    const wp::int32 var_5 = 6;
    bool var_6;
    const wp::int32 var_7 = 7;
    bool var_8;
    const wp::float32 var_9 = 1.0;
    const wp::float32 var_10 = -1.0;
    //---------
    // forward
    // def _uniform8_slot_sign_y(slot: int) -> float:                                         <L 95>
    // if slot == 2 or slot == 3 or slot == 6 or slot == 7:                                   <L 96>
    var_2 = (var_slot == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_slot == var_3);
        var_0 = var_0 || var_4;
    }
    if (!var_0) {
        var_6 = (var_slot == var_5);
        var_0 = var_0 || var_6;
    }
    if (!var_0) {
        var_8 = (var_slot == var_7);
        var_0 = var_0 || var_8;
    }
    if (var_0) {
        // return 1.0                                                                         <L 97>
        return var_9;
    }
    // return -1.0                                                                            <L 98>
    return var_10;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:101
static CUDA_CALLABLE wp::float32 _uniform8_slot_sign_z_0(
    wp::int32 var_slot)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 4;
    bool var_2;
    const wp::int32 var_3 = 5;
    bool var_4;
    const wp::int32 var_5 = 6;
    bool var_6;
    const wp::int32 var_7 = 7;
    bool var_8;
    const wp::float32 var_9 = 1.0;
    const wp::float32 var_10 = -1.0;
    //---------
    // forward
    // def _uniform8_slot_sign_z(slot: int) -> float:                                         <L 102>
    // if slot == 4 or slot == 5 or slot == 6 or slot == 7:                                   <L 103>
    var_2 = (var_slot == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_slot == var_3);
        var_0 = var_0 || var_4;
    }
    if (!var_0) {
        var_6 = (var_slot == var_5);
        var_0 = var_0 || var_6;
    }
    if (!var_0) {
        var_8 = (var_slot == var_7);
        var_0 = var_0 || var_8;
    }
    if (var_0) {
        // return 1.0                                                                         <L 104>
        return var_9;
    }
    // return -1.0                                                                            <L 105>
    return var_10;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:994
static CUDA_CALLABLE wp::int32 _slot_uses_upper_x_0(
    wp::int32 var_slot)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 1;
    bool var_2;
    const wp::int32 var_3 = 2;
    bool var_4;
    const wp::int32 var_5 = 5;
    bool var_6;
    const wp::int32 var_7 = 6;
    bool var_8;
    const wp::int32 var_9 = 1;
    const wp::int32 var_10 = 0;
    //---------
    // forward
    // def _slot_uses_upper_x(slot: int) -> int:                                              <L 995>
    // if slot == 1 or slot == 2 or slot == 5 or slot == 6:                                   <L 996>
    var_2 = (var_slot == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_slot == var_3);
        var_0 = var_0 || var_4;
    }
    if (!var_0) {
        var_6 = (var_slot == var_5);
        var_0 = var_0 || var_6;
    }
    if (!var_0) {
        var_8 = (var_slot == var_7);
        var_0 = var_0 || var_8;
    }
    if (var_0) {
        // return 1                                                                           <L 997>
        return var_9;
    }
    // return 0                                                                               <L 998>
    return var_10;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:1001
static CUDA_CALLABLE wp::int32 _slot_uses_upper_y_0(
    wp::int32 var_slot)
{
    //---------
    // primal vars
    bool var_0;
    const wp::int32 var_1 = 2;
    bool var_2;
    const wp::int32 var_3 = 3;
    bool var_4;
    const wp::int32 var_5 = 6;
    bool var_6;
    const wp::int32 var_7 = 7;
    bool var_8;
    const wp::int32 var_9 = 1;
    const wp::int32 var_10 = 0;
    //---------
    // forward
    // def _slot_uses_upper_y(slot: int) -> int:                                              <L 1002>
    // if slot == 2 or slot == 3 or slot == 6 or slot == 7:                                   <L 1003>
    var_2 = (var_slot == var_1);
    var_0 = var_2;
    if (!var_0) {
        var_4 = (var_slot == var_3);
        var_0 = var_0 || var_4;
    }
    if (!var_0) {
        var_6 = (var_slot == var_5);
        var_0 = var_0 || var_6;
    }
    if (!var_0) {
        var_8 = (var_slot == var_7);
        var_0 = var_0 || var_8;
    }
    if (var_0) {
        // return 1                                                                           <L 1004>
        return var_9;
    }
    // return 0                                                                               <L 1005>
    return var_10;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:1008
static CUDA_CALLABLE wp::int32 _slot_uses_upper_z_0(
    wp::int32 var_slot)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 4;
    bool var_1;
    const wp::int32 var_2 = 1;
    const wp::int32 var_3 = 0;
    //---------
    // forward
    // def _slot_uses_upper_z(slot: int) -> int:                                              <L 1009>
    // if slot >= 4:                                                                          <L 1010>
    var_1 = (var_slot >= var_0);
    if (var_1) {
        // return 1                                                                           <L 1011>
        return var_2;
    }
    // return 0                                                                               <L 1012>
    return var_3;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:20
static CUDA_CALLABLE void adj__extract_rotation_0(
    wp::mat_t<3, 3, wp::float32> var_a,
    wp::quat_t<wp::float32> var_q_init,
    wp::int32 var_max_iters,
    wp::mat_t<3, 3, wp::float32> & adj_a,
    wp::quat_t<wp::float32> & adj_q_init,
    wp::int32 & adj_max_iters,
    wp::quat_t<wp::float32> & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:15
static CUDA_CALLABLE void adj__quat_dot_0(
    wp::quat_t<wp::float32> var_a,
    wp::quat_t<wp::float32> var_b,
    wp::quat_t<wp::float32> & adj_a,
    wp::quat_t<wp::float32> & adj_b,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:74
static CUDA_CALLABLE void adj__support_scale_from_alpha_0(
    wp::float32 var_inv_weight,
    wp::float32 var_support_alpha,
    wp::float32 & adj_inv_weight,
    wp::float32 & adj_support_alpha,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:87
static CUDA_CALLABLE void adj__uniform8_slot_sign_x_0(
    wp::int32 var_slot,
    wp::int32 & adj_slot,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:94
static CUDA_CALLABLE void adj__uniform8_slot_sign_y_0(
    wp::int32 var_slot,
    wp::int32 & adj_slot,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:101
static CUDA_CALLABLE void adj__uniform8_slot_sign_z_0(
    wp::int32 var_slot,
    wp::int32 & adj_slot,
    wp::float32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:994
static CUDA_CALLABLE void adj__slot_uses_upper_x_0(
    wp::int32 var_slot,
    wp::int32 & adj_slot,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:1001
static CUDA_CALLABLE void adj__slot_uses_upper_y_0(
    wp::int32 var_slot,
    wp::int32 & adj_slot,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/shape_matching.py:1008
static CUDA_CALLABLE void adj__slot_uses_upper_z_0(
    wp::int32 var_slot,
    wp::int32 & adj_slot,
    wp::int32 & adj_ret)
{
	// reverse mode disabled (module option "enable_backward" is False or no dependent kernel found with "enable_backward")
}



extern "C" __global__ void apply_shape_matching_particle_gather_uniform8_a9a23521_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_init,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_deltas,
    wp::array_t<wp::int32> var_particle_cluster_offsets,
    wp::array_t<wp::int32> var_particle_cluster_indices,
    wp::array_t<wp::int32> var_particle_cluster_member_offsets,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::float32 var_stiffness,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cluster_translations,
    wp::int32 var_include_base_delta,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd_out)
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
        const wp::float32 var_7 = 0.0;
        const wp::float32 var_8 = 0.0;
        const wp::float32 var_9 = 0.0;
        wp::vec_t<3, wp::float32> var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        bool var_17;
        wp::float32* var_18;
        const wp::float32 var_19 = 0.0;
        bool var_20;
        wp::float32 var_21;
        const wp::float32 var_22 = 0.0;
        bool var_23;
        wp::float32* var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 1;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        bool var_37;
        wp::int32* var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        wp::int32* var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        wp::int32 var_44;
        wp::float32* var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.0;
        bool var_49;
        wp::int32* var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 8;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::vec_t<3, wp::float32>* var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        const wp::int32 var_59 = 0;
        bool var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32>* var_67;
        wp::quat_t<wp::float32>* var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::quat_t<wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32>* var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        wp::vec_t<3, wp::float32>* var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::float32 var_93;
        bool var_94;
        wp::float32 var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        //---------
        // forward
        // def apply_shape_matching_particle_gather_uniform8(                                     <L 313>
        // particle_idx = wp.tid()                                                                <L 338>
        var_0 = builtin_tid1d();
        // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                         <L 340>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 341>
            continue;
        }
        // total_delta = wp.vec3(0.0, 0.0, 0.0)                                                   <L 343>
        var_10 = wp::vec_t<3, wp::float32>(var_7, var_8, var_9);
        // if include_base_delta != 0:                                                            <L 344>
        var_12 = (var_include_base_delta != var_11);
        if (var_12) {
            // total_delta += particle_deltas[particle_idx]                                       <L 345>
            var_13 = wp::address(var_particle_deltas, var_0);
            var_15 = wp::load(var_13);
            var_14 = wp::add(var_10, var_15);
        }
        var_16 = wp::where(var_12, var_14, var_10);
        // if particle_inv_mass[particle_idx] > 0.0 and stiffness > 0.0:                          <L 347>
        var_18 = wp::address(var_particle_inv_mass, var_0);
        var_21 = wp::load(var_18);
        var_20 = (var_21 > var_19);
        var_17 = var_20;
        if (var_17) {
            var_23 = (var_stiffness > var_22);
            var_17 = var_17 && var_23;
        }
        if (var_17) {
            // inv_weight = particle_cluster_inv_weights[particle_idx]                            <L 348>
            var_24 = wp::address(var_particle_cluster_inv_weights, var_0);
            var_26 = wp::load(var_24);
            var_25 = wp::copy(var_26);
            // if inv_weight > 0.0:                                                               <L 349>
            var_28 = (var_25 > var_27);
            if (var_28) {
                // cursor = particle_cluster_offsets[particle_idx]                                <L 350>
                var_29 = wp::address(var_particle_cluster_offsets, var_0);
                var_31 = wp::load(var_29);
                var_30 = wp::copy(var_31);
                // end = particle_cluster_offsets[particle_idx + 1]                               <L 351>
                var_33 = wp::add(var_0, var_32);
                var_34 = wp::address(var_particle_cluster_offsets, var_33);
                var_36 = wp::load(var_34);
                var_35 = wp::copy(var_36);
                // while cursor < end:                                                            <L 352>
        start_while_1:;
                var_37 = (var_30 < var_35);
        if ((var_37) == false) goto end_while_1;
                    // cluster_idx = particle_cluster_indices[cursor]                             <L 353>
                    var_38 = wp::address(var_particle_cluster_indices, var_30);
                    var_40 = wp::load(var_38);
                    var_39 = wp::copy(var_40);
                    // if cluster_active[cluster_idx] != 0:                                       <L 354>
                    var_41 = wp::address(var_cluster_active, var_39);
                    var_44 = wp::load(var_41);
                    var_43 = (var_44 != var_42);
                    if (var_43) {
                        // coeff = coefficients[cluster_idx]                                      <L 355>
                        var_45 = wp::address(var_coefficients, var_39);
                        var_47 = wp::load(var_45);
                        var_46 = wp::copy(var_47);
                        // if coeff > 0.0:                                                        <L 356>
                        var_49 = (var_46 > var_48);
                        if (var_49) {
                            // member_offset = particle_cluster_member_offsets[cursor]            <L 357>
                            var_50 = wp::address(var_particle_cluster_member_offsets, var_30);
                            var_52 = wp::load(var_50);
                            var_51 = wp::copy(var_52);
                            // local_idx = member_offset - cluster_idx * wp.static(UNIFORM_CLUSTER_SIZE)       <L 358>
                            var_54 = wp::mul(var_39, var_53);
                            var_55 = wp::sub(var_51, var_54);
                            // q_rel = rest_local_template[local_idx]                             <L 359>
                            var_56 = wp::address(var_rest_local_template, var_55);
                            var_58 = wp::load(var_56);
                            var_57 = wp::copy(var_58);
                            // if use_rest_local_template == 0:                                   <L 360>
                            var_60 = (var_use_rest_local_template == var_59);
                            if (var_60) {
                                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 361>
                                var_61 = wp::mul(var_55, var_cluster_count);
                                var_62 = wp::add(var_61, var_39);
                                var_63 = wp::address(var_rest_local_positions_by_slot, var_62);
                                var_65 = wp::load(var_63);
                                var_64 = wp::copy(var_65);
                            }
                            var_66 = wp::where(var_60, var_64, var_57);
                            // goal = cluster_translations[cluster_idx] + wp.quat_rotate(         <L 362>
                            var_67 = wp::address(var_cluster_translations, var_39);
                            // cluster_rotations[cluster_idx],                                    <L 363>
                            var_68 = wp::address(var_cluster_rotations, var_39);
                            // q_rel,                                                             <L 364>
                            var_70 = wp::load(var_68);
                            var_69 = wp::quat_rotate(var_70, var_66);
                            var_72 = wp::load(var_67);
                            var_71 = wp::add(var_72, var_69);
                            // total_delta += (goal - particle_q[particle_idx]) * (coeff * stiffness * inv_weight)       <L 366>
                            var_73 = wp::address(var_particle_q, var_0);
                            var_75 = wp::load(var_73);
                            var_74 = wp::sub(var_71, var_75);
                            var_76 = wp::mul(var_46, var_stiffness);
                            var_77 = wp::mul(var_76, var_25);
                            var_78 = wp::mul(var_74, var_77);
                            var_79 = wp::add(var_16, var_78);
                        }
                        var_80 = wp::where(var_49, var_79, var_16);
                    }
                    var_81 = wp::where(var_43, var_80, var_16);
                    // cursor += 1                                                                <L 367>
                    var_83 = wp::add(var_30, var_82);
                    wp::assign(var_16, var_81);
                    wp::assign(var_30, var_83);
        goto start_while_1;
        end_while_1:;
            }
        }
        // x0 = particle_q_init[particle_idx]                                                     <L 369>
        var_84 = wp::address(var_particle_q_init, var_0);
        var_86 = wp::load(var_84);
        var_85 = wp::copy(var_86);
        // xp = particle_q[particle_idx]                                                          <L 370>
        var_87 = wp::address(var_particle_q, var_0);
        var_89 = wp::load(var_87);
        var_88 = wp::copy(var_89);
        // x_new = xp + total_delta                                                               <L 371>
        var_90 = wp::add(var_88, var_16);
        // v_new = (x_new - x0) / dt                                                              <L 372>
        var_91 = wp::sub(var_90, var_85);
        var_92 = wp::div(var_91, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 374>
        var_93 = wp::length(var_92);
        // if v_new_mag > v_max:                                                                  <L 375>
        var_94 = (var_93 > var_v_max);
        if (var_94) {
            // v_new *= v_max / v_new_mag                                                         <L 376>
            var_95 = wp::div(var_v_max, var_93);
            var_96 = wp::mul(var_92, var_95);
        }
        var_97 = wp::where(var_94, var_96, var_92);
        // particle_q_out[particle_idx] = x_new                                                   <L 378>
        wp::array_store(var_particle_q_out, var_0, var_90);
        // particle_qd_out[particle_idx] = v_new                                                  <L 379>
        wp::array_store(var_particle_qd_out, var_0, var_97);
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform27_colored_gs_ef0da61e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_color_cluster_indices,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::int32 var_color_start,
    wp::float32 var_stiffness,
    wp::float32 var_support_alpha,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations)
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 27;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::float32* var_44;
        const wp::float32 var_45 = 0.0;
        bool var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        bool var_52;
        wp::float32 var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::float32 var_55 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_56;
        const wp::int32 var_57 = 27;
        wp::range_t var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32* var_65;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        bool var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32>* var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::vec_t<3, wp::float32>* var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::mat_t<3, 3, wp::float32> var_86;
        wp::mat_t<3, 3, wp::float32> var_87;
        wp::quat_t<wp::float32>* var_88;
        wp::quat_t<wp::float32> var_89;
        wp::quat_t<wp::float32> var_90;
        wp::quat_t<wp::float32> var_91;
        wp::float32 var_92;
        const wp::float32 var_93 = 0.0;
        bool var_94;
        const wp::int32 var_95 = 0;
        wp::float32 var_96;
        wp::float32 var_97;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 3;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::quat_t<wp::float32> var_107;
        wp::quat_t<wp::float32> var_108;
        const wp::int32 var_109 = 0;
        bool var_110;
        const wp::int32 var_111 = 27;
        wp::range_t var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        bool var_119;
        wp::int32* var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        wp::int32 var_123;
        const wp::int32 var_124 = 0;
        bool var_125;
        wp::float32* var_126;
        const wp::float32 var_127 = 0.0;
        bool var_128;
        wp::float32 var_129;
        wp::int32 var_130;
        wp::vec_t<3, wp::float32>* var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 0;
        bool var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::float32 var_144;
        wp::float32* var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::float32 var_149 = 0.0;
        bool var_150;
        wp::vec_t<3, wp::float32>* var_151;
        wp::vec_t<3, wp::float32>* var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform27_colored_gs(                                <L 741>
        // cluster_idx = color_cluster_indices[color_start + wp.tid()]                            <L 760>
        var_0 = builtin_tid1d();
        var_1 = wp::add(var_color_start, var_0);
        var_2 = wp::address(var_color_cluster_indices, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 762>
        var_6 = wp::address(var_cluster_active, var_3);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_5 = var_8;
        if (!var_5) {
            var_11 = (var_stiffness <= var_10);
            var_5 = var_5 || var_11;
        }
        if (var_5) {
            // return                                                                             <L 763>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 765>
        var_12 = wp::address(var_coefficients, var_3);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if coeff <= 0.0:                                                                       <L 766>
        var_16 = (var_13 <= var_15);
        if (var_16) {
            // return                                                                             <L 767>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 769>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // member_count = int(0)                                                                  <L 770>
        var_22 = wp::int(var_21);
        // dynamic_count = int(0)                                                                 <L 771>
        var_24 = wp::int(var_23);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):                            <L 773>
        var_26 = wp::range(var_25);
        start_for_2:;
            if (iter_cmp(var_26) == 0) goto end_for_2;
            var_27 = wp::iter_next(var_26);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 774>
            var_28 = wp::mul(var_27, var_cluster_count);
            var_29 = wp::add(var_28, var_3);
            var_30 = wp::address(var_indices_by_slot, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 775>
            var_33 = wp::address(var_particle_flags, var_31);
            var_36 = wp::load(var_33);
            var_35 = wp::bit_and(var_36, var_34);
            var_38 = (var_35 == var_37);
            if (var_38) {
                // continue                                                                       <L 776>
                goto start_for_2;
            }
            // center += particle_q[particle_idx]                                                 <L 777>
            var_39 = wp::address(var_particle_q, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::add(var_20, var_41);
            // member_count += 1                                                                  <L 778>
            var_43 = wp::add(var_22, var_42);
            // if particle_inv_mass[particle_idx] > 0.0:                                          <L 779>
            var_44 = wp::address(var_particle_inv_mass, var_31);
            var_47 = wp::load(var_44);
            var_46 = (var_47 > var_45);
            if (var_46) {
                // dynamic_count += 1                                                             <L 780>
                var_49 = wp::add(var_24, var_48);
            }
            var_50 = wp::where(var_46, var_49, var_24);
            wp::assign(var_20, var_40);
            wp::assign(var_22, var_43);
            wp::assign(var_24, var_50);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 782>
        var_52 = (var_22 == var_51);
        if (var_52) {
            // return                                                                             <L 783>
            continue;
        }
        // center /= float(member_count)                                                          <L 785>
        var_53 = wp::float(var_22);
        var_54 = wp::div(var_20, var_53);
        // covariance = wp.mat33(0.0)                                                             <L 787>
        var_56 = wp::mat_t<3, 3, wp::float32>(var_55);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):                            <L 788>
        var_58 = wp::range(var_57);
        start_for_5:;
            if (iter_cmp(var_58) == 0) goto end_for_5;
            var_59 = wp::iter_next(var_58);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 789>
            var_60 = wp::mul(var_59, var_cluster_count);
            var_61 = wp::add(var_60, var_3);
            var_62 = wp::address(var_indices_by_slot, var_61);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 790>
            var_65 = wp::address(var_particle_flags, var_63);
            var_68 = wp::load(var_65);
            var_67 = wp::bit_and(var_68, var_66);
            var_70 = (var_67 == var_69);
            if (var_70) {
                // continue                                                                       <L 791>
                wp::assign(var_31, var_63);
                goto start_for_5;
            }
            var_71 = wp::where(var_70, var_31, var_63);
            // x_rel = particle_q[particle_idx] - center                                          <L 792>
            var_72 = wp::address(var_particle_q, var_71);
            var_74 = wp::load(var_72);
            var_73 = wp::sub(var_74, var_54);
            // q_rel = rest_local_template[local_idx]                                             <L 793>
            var_75 = wp::address(var_rest_local_template, var_59);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // if use_rest_local_template == 0:                                                   <L 794>
            var_79 = (var_use_rest_local_template == var_78);
            if (var_79) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 795>
                var_80 = wp::mul(var_59, var_cluster_count);
                var_81 = wp::add(var_80, var_3);
                var_82 = wp::address(var_rest_local_positions_by_slot, var_81);
                var_84 = wp::load(var_82);
                var_83 = wp::copy(var_84);
            }
            var_85 = wp::where(var_79, var_83, var_76);
            // covariance += wp.outer(q_rel, x_rel)                                               <L 796>
            var_86 = wp::outer(var_85, var_73);
            var_87 = wp::add(var_56, var_86);
            wp::assign(var_31, var_71);
            wp::assign(var_56, var_87);
            goto start_for_5;
        end_for_5:;
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 798>
        var_88 = wp::address(var_cluster_rotations, var_3);
        var_90 = wp::load(var_88);
        var_89 = wp::copy(var_90);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 799>
        var_91 = _extract_rotation_0(var_56, var_89, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 800>
        var_92 = _quat_dot_0(var_91, var_89);
        var_94 = (var_92 < var_93);
        if (var_94) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 801>
            var_96 = wp::extract(var_91, var_95);
            var_97 = wp::neg(var_96);
            var_99 = wp::extract(var_91, var_98);
            var_100 = wp::neg(var_99);
            var_102 = wp::extract(var_91, var_101);
            var_103 = wp::neg(var_102);
            var_105 = wp::extract(var_91, var_104);
            var_106 = wp::neg(var_105);
            var_107 = wp::quat_t<wp::float32>(var_97, var_100, var_103, var_106);
        }
        var_108 = wp::where(var_94, var_107, var_91);
        // cluster_rotations[cluster_idx] = rotation                                              <L 803>
        wp::array_store(var_cluster_rotations, var_3, var_108);
        // if dynamic_count == 0:                                                                 <L 805>
        var_110 = (var_24 == var_109);
        if (var_110) {
            // return                                                                             <L 806>
            continue;
        }
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):                            <L 808>
        var_112 = wp::range(var_111);
        start_for_8:;
            if (iter_cmp(var_112) == 0) goto end_for_8;
            var_113 = wp::iter_next(var_112);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 809>
            var_114 = wp::mul(var_113, var_cluster_count);
            var_115 = wp::add(var_114, var_3);
            var_116 = wp::address(var_indices_by_slot, var_115);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:       <L 810>
            var_120 = wp::address(var_particle_flags, var_117);
            var_123 = wp::load(var_120);
            var_122 = wp::bit_and(var_123, var_121);
            var_125 = (var_122 == var_124);
            var_119 = var_125;
            if (!var_119) {
                var_126 = wp::address(var_particle_inv_mass, var_117);
                var_129 = wp::load(var_126);
                var_128 = (var_129 <= var_127);
                var_119 = var_119 || var_128;
            }
            if (var_119) {
                // continue                                                                       <L 811>
                wp::assign(var_31, var_117);
                goto start_for_8;
            }
            var_130 = wp::where(var_119, var_31, var_117);
            // q_rel = rest_local_template[local_idx]                                             <L 813>
            var_131 = wp::address(var_rest_local_template, var_113);
            var_133 = wp::load(var_131);
            var_132 = wp::copy(var_133);
            // if use_rest_local_template == 0:                                                   <L 814>
            var_135 = (var_use_rest_local_template == var_134);
            if (var_135) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 815>
                var_136 = wp::mul(var_113, var_cluster_count);
                var_137 = wp::add(var_136, var_3);
                var_138 = wp::address(var_rest_local_positions_by_slot, var_137);
                var_140 = wp::load(var_138);
                var_139 = wp::copy(var_140);
            }
            var_141 = wp::where(var_135, var_139, var_132);
            // goal = center + wp.quat_rotate(rotation, q_rel)                                    <L 816>
            var_142 = wp::quat_rotate(var_108, var_141);
            var_143 = wp::add(var_54, var_142);
            // particle_scale = coeff * stiffness * _support_scale_from_alpha(                    <L 817>
            var_144 = wp::mul(var_13, var_stiffness);
            // particle_cluster_inv_weights[particle_idx],                                        <L 818>
            var_145 = wp::address(var_particle_cluster_inv_weights, var_130);
            // support_alpha,                                                                     <L 819>
            var_147 = wp::load(var_145);
            var_146 = _support_scale_from_alpha_0(var_147, var_support_alpha);
            var_148 = wp::mul(var_144, var_146);
            // if particle_scale > 0.0:                                                           <L 821>
            var_150 = (var_148 > var_149);
            if (var_150) {
                // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 822>
                var_151 = wp::address(var_particle_q, var_130);
                var_152 = wp::address(var_particle_q, var_130);
                var_154 = wp::load(var_152);
                var_153 = wp::sub(var_143, var_154);
                var_155 = wp::mul(var_153, var_148);
                var_157 = wp::load(var_151);
                var_156 = wp::add(var_157, var_155);
                // particle_q[particle_idx] = x_new                                               <L 823>
                wp::array_store(var_particle_q, var_130, var_156);
            }
            wp::assign(var_31, var_130);
            wp::assign(var_85, var_141);
            goto start_for_8;
        end_for_8:;
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform8_colored_gs_12229cc9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_color_cluster_indices,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::int32 var_color_start,
    wp::float32 var_stiffness,
    wp::float32 var_support_alpha,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations)
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        wp::vec_t<3, wp::float32> var_24;
        const wp::float32 var_25 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        const wp::int32 var_31 = 8;
        wp::range_t var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32* var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        const wp::int32 var_51 = 0;
        bool var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::vec_t<3, wp::float32>* var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::mat_t<3, 3, wp::float32> var_61;
        wp::mat_t<3, 3, wp::float32> var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::float32* var_65;
        const wp::float32 var_66 = 0.0;
        bool var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        wp::float32 var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::mat_t<3, 3, wp::float32> var_76;
        wp::mat_t<3, 3, wp::float32> var_77;
        wp::quat_t<wp::float32>* var_78;
        wp::quat_t<wp::float32> var_79;
        wp::quat_t<wp::float32> var_80;
        wp::quat_t<wp::float32> var_81;
        wp::float32 var_82;
        const wp::float32 var_83 = 0.0;
        bool var_84;
        const wp::int32 var_85 = 0;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 1;
        wp::float32 var_89;
        wp::float32 var_90;
        const wp::int32 var_91 = 2;
        wp::float32 var_92;
        wp::float32 var_93;
        const wp::int32 var_94 = 3;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::quat_t<wp::float32> var_97;
        wp::quat_t<wp::float32> var_98;
        const wp::int32 var_99 = 0;
        bool var_100;
        const wp::int32 var_101 = 8;
        wp::range_t var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32* var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        bool var_109;
        wp::int32* var_110;
        const wp::int32 var_111 = 1;
        wp::int32 var_112;
        wp::int32 var_113;
        const wp::int32 var_114 = 0;
        bool var_115;
        wp::float32* var_116;
        const wp::float32 var_117 = 0.0;
        bool var_118;
        wp::float32 var_119;
        wp::int32 var_120;
        wp::vec_t<3, wp::float32>* var_121;
        wp::vec_t<3, wp::float32> var_122;
        wp::vec_t<3, wp::float32> var_123;
        const wp::int32 var_124 = 0;
        bool var_125;
        wp::int32 var_126;
        wp::int32 var_127;
        wp::vec_t<3, wp::float32>* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::float32 var_134;
        wp::float32* var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        wp::float32 var_138;
        const wp::float32 var_139 = 0.0;
        bool var_140;
        wp::vec_t<3, wp::float32>* var_141;
        wp::vec_t<3, wp::float32>* var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform8_colored_gs(                                 <L 517>
        // cluster_idx = color_cluster_indices[color_start + wp.tid()]                            <L 536>
        var_0 = builtin_tid1d();
        var_1 = wp::add(var_color_start, var_0);
        var_2 = wp::address(var_color_cluster_indices, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 538>
        var_6 = wp::address(var_cluster_active, var_3);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_5 = var_8;
        if (!var_5) {
            var_11 = (var_stiffness <= var_10);
            var_5 = var_5 || var_11;
        }
        if (var_5) {
            // return                                                                             <L 539>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 541>
        var_12 = wp::address(var_coefficients, var_3);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if coeff <= 0.0:                                                                       <L 542>
        var_16 = (var_13 <= var_15);
        if (var_16) {
            // return                                                                             <L 543>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 545>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // rest_sum = wp.vec3(0.0, 0.0, 0.0)                                                      <L 546>
        var_24 = wp::vec_t<3, wp::float32>(var_21, var_22, var_23);
        // covariance = wp.mat33(0.0)                                                             <L 547>
        var_26 = wp::mat_t<3, 3, wp::float32>(var_25);
        // member_count = int(0)                                                                  <L 548>
        var_28 = wp::int(var_27);
        // dynamic_count = int(0)                                                                 <L 549>
        var_30 = wp::int(var_29);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 551>
        var_32 = wp::range(var_31);
        start_for_2:;
            if (iter_cmp(var_32) == 0) goto end_for_2;
            var_33 = wp::iter_next(var_32);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 552>
            var_34 = wp::mul(var_33, var_cluster_count);
            var_35 = wp::add(var_34, var_3);
            var_36 = wp::address(var_indices_by_slot, var_35);
            var_38 = wp::load(var_36);
            var_37 = wp::copy(var_38);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 553>
            var_39 = wp::address(var_particle_flags, var_37);
            var_42 = wp::load(var_39);
            var_41 = wp::bit_and(var_42, var_40);
            var_44 = (var_41 == var_43);
            if (var_44) {
                // continue                                                                       <L 554>
                goto start_for_2;
            }
            // x = particle_q[particle_idx]                                                       <L 555>
            var_45 = wp::address(var_particle_q, var_37);
            var_47 = wp::load(var_45);
            var_46 = wp::copy(var_47);
            // q_rel = rest_local_template[local_idx]                                             <L 556>
            var_48 = wp::address(var_rest_local_template, var_33);
            var_50 = wp::load(var_48);
            var_49 = wp::copy(var_50);
            // if use_rest_local_template == 0:                                                   <L 557>
            var_52 = (var_use_rest_local_template == var_51);
            if (var_52) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 558>
                var_53 = wp::mul(var_33, var_cluster_count);
                var_54 = wp::add(var_53, var_3);
                var_55 = wp::address(var_rest_local_positions_by_slot, var_54);
                var_57 = wp::load(var_55);
                var_56 = wp::copy(var_57);
            }
            var_58 = wp::where(var_52, var_56, var_49);
            // center += x                                                                        <L 559>
            var_59 = wp::add(var_20, var_46);
            // rest_sum += q_rel                                                                  <L 560>
            var_60 = wp::add(var_24, var_58);
            // covariance += wp.outer(q_rel, x)                                                   <L 561>
            var_61 = wp::outer(var_58, var_46);
            var_62 = wp::add(var_26, var_61);
            // member_count += 1                                                                  <L 562>
            var_64 = wp::add(var_28, var_63);
            // if particle_inv_mass[particle_idx] > 0.0:                                          <L 563>
            var_65 = wp::address(var_particle_inv_mass, var_37);
            var_68 = wp::load(var_65);
            var_67 = (var_68 > var_66);
            if (var_67) {
                // dynamic_count += 1                                                             <L 564>
                var_70 = wp::add(var_30, var_69);
            }
            var_71 = wp::where(var_67, var_70, var_30);
            wp::assign(var_20, var_59);
            wp::assign(var_24, var_60);
            wp::assign(var_26, var_62);
            wp::assign(var_28, var_64);
            wp::assign(var_30, var_71);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 566>
        var_73 = (var_28 == var_72);
        if (var_73) {
            // return                                                                             <L 567>
            continue;
        }
        // center /= float(member_count)                                                          <L 569>
        var_74 = wp::float(var_28);
        var_75 = wp::div(var_20, var_74);
        // covariance -= wp.outer(rest_sum, center)                                               <L 570>
        var_76 = wp::outer(var_24, var_75);
        var_77 = wp::sub(var_26, var_76);
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 572>
        var_78 = wp::address(var_cluster_rotations, var_3);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 573>
        var_81 = _extract_rotation_0(var_77, var_79, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 574>
        var_82 = _quat_dot_0(var_81, var_79);
        var_84 = (var_82 < var_83);
        if (var_84) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 575>
            var_86 = wp::extract(var_81, var_85);
            var_87 = wp::neg(var_86);
            var_89 = wp::extract(var_81, var_88);
            var_90 = wp::neg(var_89);
            var_92 = wp::extract(var_81, var_91);
            var_93 = wp::neg(var_92);
            var_95 = wp::extract(var_81, var_94);
            var_96 = wp::neg(var_95);
            var_97 = wp::quat_t<wp::float32>(var_87, var_90, var_93, var_96);
        }
        var_98 = wp::where(var_84, var_97, var_81);
        // cluster_rotations[cluster_idx] = rotation                                              <L 577>
        wp::array_store(var_cluster_rotations, var_3, var_98);
        // if dynamic_count == 0:                                                                 <L 579>
        var_100 = (var_30 == var_99);
        if (var_100) {
            // return                                                                             <L 580>
            continue;
        }
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 582>
        var_102 = wp::range(var_101);
        start_for_6:;
            if (iter_cmp(var_102) == 0) goto end_for_6;
            var_103 = wp::iter_next(var_102);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 583>
            var_104 = wp::mul(var_103, var_cluster_count);
            var_105 = wp::add(var_104, var_3);
            var_106 = wp::address(var_indices_by_slot, var_105);
            var_108 = wp::load(var_106);
            var_107 = wp::copy(var_108);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:       <L 584>
            var_110 = wp::address(var_particle_flags, var_107);
            var_113 = wp::load(var_110);
            var_112 = wp::bit_and(var_113, var_111);
            var_115 = (var_112 == var_114);
            var_109 = var_115;
            if (!var_109) {
                var_116 = wp::address(var_particle_inv_mass, var_107);
                var_119 = wp::load(var_116);
                var_118 = (var_119 <= var_117);
                var_109 = var_109 || var_118;
            }
            if (var_109) {
                // continue                                                                       <L 585>
                wp::assign(var_37, var_107);
                goto start_for_6;
            }
            var_120 = wp::where(var_109, var_37, var_107);
            // q_rel = rest_local_template[local_idx]                                             <L 587>
            var_121 = wp::address(var_rest_local_template, var_103);
            var_123 = wp::load(var_121);
            var_122 = wp::copy(var_123);
            // if use_rest_local_template == 0:                                                   <L 588>
            var_125 = (var_use_rest_local_template == var_124);
            if (var_125) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 589>
                var_126 = wp::mul(var_103, var_cluster_count);
                var_127 = wp::add(var_126, var_3);
                var_128 = wp::address(var_rest_local_positions_by_slot, var_127);
                var_130 = wp::load(var_128);
                var_129 = wp::copy(var_130);
            }
            var_131 = wp::where(var_125, var_129, var_122);
            // goal = center + wp.quat_rotate(rotation, q_rel)                                    <L 590>
            var_132 = wp::quat_rotate(var_98, var_131);
            var_133 = wp::add(var_75, var_132);
            // particle_scale = coeff * stiffness * _support_scale_from_alpha(                    <L 591>
            var_134 = wp::mul(var_13, var_stiffness);
            // particle_cluster_inv_weights[particle_idx],                                        <L 592>
            var_135 = wp::address(var_particle_cluster_inv_weights, var_120);
            // support_alpha,                                                                     <L 593>
            var_137 = wp::load(var_135);
            var_136 = _support_scale_from_alpha_0(var_137, var_support_alpha);
            var_138 = wp::mul(var_134, var_136);
            // if particle_scale > 0.0:                                                           <L 595>
            var_140 = (var_138 > var_139);
            if (var_140) {
                // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 596>
                var_141 = wp::address(var_particle_q, var_120);
                var_142 = wp::address(var_particle_q, var_120);
                var_144 = wp::load(var_142);
                var_143 = wp::sub(var_133, var_144);
                var_145 = wp::mul(var_143, var_138);
                var_147 = wp::load(var_141);
                var_146 = wp::add(var_147, var_145);
                // particle_q[particle_idx] = x_new                                               <L 597>
                wp::array_store(var_particle_q, var_120, var_146);
            }
            wp::assign(var_37, var_120);
            wp::assign(var_58, var_131);
            goto start_for_6;
        end_for_6:;
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform125_389d4cf6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::float32 var_stiffness,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cluster_translations,
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
        bool var_1;
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 125;
        wp::range_t var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::float32* var_40;
        const wp::float32 var_41 = 0.0;
        bool var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::float32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        const wp::float32 var_51 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_52;
        const wp::int32 var_53 = 125;
        wp::range_t var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::int32* var_61;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        bool var_66;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32>* var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        const wp::int32 var_74 = 0;
        bool var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::mat_t<3, 3, wp::float32> var_82;
        wp::mat_t<3, 3, wp::float32> var_83;
        wp::quat_t<wp::float32>* var_84;
        wp::quat_t<wp::float32> var_85;
        wp::quat_t<wp::float32> var_86;
        wp::quat_t<wp::float32> var_87;
        wp::float32 var_88;
        const wp::float32 var_89 = 0.0;
        bool var_90;
        const wp::int32 var_91 = 0;
        wp::float32 var_92;
        wp::float32 var_93;
        const wp::int32 var_94 = 1;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::int32 var_100 = 3;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::quat_t<wp::float32> var_103;
        wp::quat_t<wp::float32> var_104;
        const wp::int32 var_105 = 0;
        bool var_106;
        const wp::int32 var_107 = 125;
        wp::range_t var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        bool var_115;
        wp::int32* var_116;
        const wp::int32 var_117 = 1;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 0;
        bool var_121;
        wp::float32* var_122;
        const wp::float32 var_123 = 0.0;
        bool var_124;
        wp::float32 var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32>* var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        const wp::int32 var_130 = 0;
        bool var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::vec_t<3, wp::float32>* var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::float32 var_140;
        wp::float32* var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        const wp::float32 var_144 = 0.0;
        bool var_145;
        wp::vec_t<3, wp::float32>* var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform125(                                          <L 827>
        // cluster_idx = wp.tid()                                                                 <L 845>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 847>
        var_2 = wp::address(var_cluster_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_1 = var_4;
        if (!var_1) {
            var_7 = (var_stiffness <= var_6);
            var_1 = var_1 || var_7;
        }
        if (var_1) {
            // return                                                                             <L 848>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 850>
        var_8 = wp::address(var_coefficients, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if coeff <= 0.0:                                                                       <L 851>
        var_12 = (var_9 <= var_11);
        if (var_12) {
            // return                                                                             <L 852>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 854>
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // member_count = int(0)                                                                  <L 855>
        var_18 = wp::int(var_17);
        // dynamic_count = int(0)                                                                 <L 856>
        var_20 = wp::int(var_19);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):                           <L 858>
        var_22 = wp::range(var_21);
        start_for_2:;
            if (iter_cmp(var_22) == 0) goto end_for_2;
            var_23 = wp::iter_next(var_22);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 859>
            var_24 = wp::mul(var_23, var_cluster_count);
            var_25 = wp::add(var_24, var_0);
            var_26 = wp::address(var_indices_by_slot, var_25);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 860>
            var_29 = wp::address(var_particle_flags, var_27);
            var_32 = wp::load(var_29);
            var_31 = wp::bit_and(var_32, var_30);
            var_34 = (var_31 == var_33);
            if (var_34) {
                // continue                                                                       <L 861>
                goto start_for_2;
            }
            // center += particle_q[particle_idx]                                                 <L 862>
            var_35 = wp::address(var_particle_q, var_27);
            var_37 = wp::load(var_35);
            var_36 = wp::add(var_16, var_37);
            // member_count += 1                                                                  <L 863>
            var_39 = wp::add(var_18, var_38);
            // if particle_inv_mass[particle_idx] > 0.0:                                          <L 864>
            var_40 = wp::address(var_particle_inv_mass, var_27);
            var_43 = wp::load(var_40);
            var_42 = (var_43 > var_41);
            if (var_42) {
                // dynamic_count += 1                                                             <L 865>
                var_45 = wp::add(var_20, var_44);
            }
            var_46 = wp::where(var_42, var_45, var_20);
            wp::assign(var_16, var_36);
            wp::assign(var_18, var_39);
            wp::assign(var_20, var_46);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 867>
        var_48 = (var_18 == var_47);
        if (var_48) {
            // return                                                                             <L 868>
            continue;
        }
        // center /= float(member_count)                                                          <L 870>
        var_49 = wp::float(var_18);
        var_50 = wp::div(var_16, var_49);
        // covariance = wp.mat33(0.0)                                                             <L 872>
        var_52 = wp::mat_t<3, 3, wp::float32>(var_51);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):                           <L 873>
        var_54 = wp::range(var_53);
        start_for_5:;
            if (iter_cmp(var_54) == 0) goto end_for_5;
            var_55 = wp::iter_next(var_54);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 874>
            var_56 = wp::mul(var_55, var_cluster_count);
            var_57 = wp::add(var_56, var_0);
            var_58 = wp::address(var_indices_by_slot, var_57);
            var_60 = wp::load(var_58);
            var_59 = wp::copy(var_60);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 875>
            var_61 = wp::address(var_particle_flags, var_59);
            var_64 = wp::load(var_61);
            var_63 = wp::bit_and(var_64, var_62);
            var_66 = (var_63 == var_65);
            if (var_66) {
                // continue                                                                       <L 876>
                wp::assign(var_27, var_59);
                goto start_for_5;
            }
            var_67 = wp::where(var_66, var_27, var_59);
            // x_rel = particle_q[particle_idx] - center                                          <L 877>
            var_68 = wp::address(var_particle_q, var_67);
            var_70 = wp::load(var_68);
            var_69 = wp::sub(var_70, var_50);
            // q_rel = rest_local_template[local_idx]                                             <L 878>
            var_71 = wp::address(var_rest_local_template, var_55);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            // if use_rest_local_template == 0:                                                   <L 879>
            var_75 = (var_use_rest_local_template == var_74);
            if (var_75) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 880>
                var_76 = wp::mul(var_55, var_cluster_count);
                var_77 = wp::add(var_76, var_0);
                var_78 = wp::address(var_rest_local_positions_by_slot, var_77);
                var_80 = wp::load(var_78);
                var_79 = wp::copy(var_80);
            }
            var_81 = wp::where(var_75, var_79, var_72);
            // covariance += wp.outer(q_rel, x_rel)                                               <L 881>
            var_82 = wp::outer(var_81, var_69);
            var_83 = wp::add(var_52, var_82);
            wp::assign(var_27, var_67);
            wp::assign(var_52, var_83);
            goto start_for_5;
        end_for_5:;
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 883>
        var_84 = wp::address(var_cluster_rotations, var_0);
        var_86 = wp::load(var_84);
        var_85 = wp::copy(var_86);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 884>
        var_87 = _extract_rotation_0(var_52, var_85, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 885>
        var_88 = _quat_dot_0(var_87, var_85);
        var_90 = (var_88 < var_89);
        if (var_90) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 886>
            var_92 = wp::extract(var_87, var_91);
            var_93 = wp::neg(var_92);
            var_95 = wp::extract(var_87, var_94);
            var_96 = wp::neg(var_95);
            var_98 = wp::extract(var_87, var_97);
            var_99 = wp::neg(var_98);
            var_101 = wp::extract(var_87, var_100);
            var_102 = wp::neg(var_101);
            var_103 = wp::quat_t<wp::float32>(var_93, var_96, var_99, var_102);
        }
        var_104 = wp::where(var_90, var_103, var_87);
        // cluster_rotations[cluster_idx] = rotation                                              <L 888>
        wp::array_store(var_cluster_rotations, var_0, var_104);
        // cluster_translations[cluster_idx] = center                                             <L 889>
        wp::array_store(var_cluster_translations, var_0, var_50);
        // if dynamic_count == 0:                                                                 <L 891>
        var_106 = (var_20 == var_105);
        if (var_106) {
            // return                                                                             <L 892>
            continue;
        }
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):                           <L 894>
        var_108 = wp::range(var_107);
        start_for_8:;
            if (iter_cmp(var_108) == 0) goto end_for_8;
            var_109 = wp::iter_next(var_108);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 895>
            var_110 = wp::mul(var_109, var_cluster_count);
            var_111 = wp::add(var_110, var_0);
            var_112 = wp::address(var_indices_by_slot, var_111);
            var_114 = wp::load(var_112);
            var_113 = wp::copy(var_114);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:       <L 896>
            var_116 = wp::address(var_particle_flags, var_113);
            var_119 = wp::load(var_116);
            var_118 = wp::bit_and(var_119, var_117);
            var_121 = (var_118 == var_120);
            var_115 = var_121;
            if (!var_115) {
                var_122 = wp::address(var_particle_inv_mass, var_113);
                var_125 = wp::load(var_122);
                var_124 = (var_125 <= var_123);
                var_115 = var_115 || var_124;
            }
            if (var_115) {
                // continue                                                                       <L 897>
                wp::assign(var_27, var_113);
                goto start_for_8;
            }
            var_126 = wp::where(var_115, var_27, var_113);
            // q_rel = rest_local_template[local_idx]                                             <L 899>
            var_127 = wp::address(var_rest_local_template, var_109);
            var_129 = wp::load(var_127);
            var_128 = wp::copy(var_129);
            // if use_rest_local_template == 0:                                                   <L 900>
            var_131 = (var_use_rest_local_template == var_130);
            if (var_131) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 901>
                var_132 = wp::mul(var_109, var_cluster_count);
                var_133 = wp::add(var_132, var_0);
                var_134 = wp::address(var_rest_local_positions_by_slot, var_133);
                var_136 = wp::load(var_134);
                var_135 = wp::copy(var_136);
            }
            var_137 = wp::where(var_131, var_135, var_128);
            // goal = center + wp.quat_rotate(rotation, q_rel)                                    <L 902>
            var_138 = wp::quat_rotate(var_104, var_137);
            var_139 = wp::add(var_50, var_138);
            // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]       <L 903>
            var_140 = wp::mul(var_9, var_stiffness);
            var_141 = wp::address(var_particle_cluster_inv_weights, var_126);
            var_143 = wp::load(var_141);
            var_142 = wp::mul(var_140, var_143);
            // if particle_scale > 0.0:                                                           <L 904>
            var_145 = (var_142 > var_144);
            if (var_145) {
                // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 905>
                var_146 = wp::address(var_particle_q, var_126);
                var_148 = wp::load(var_146);
                var_147 = wp::sub(var_139, var_148);
                var_149 = wp::mul(var_147, var_142);
                var_150 = wp::atomic_add(var_particle_deltas, var_126, var_149);
            }
            wp::assign(var_27, var_126);
            wp::assign(var_81, var_137);
            goto start_for_8;
        end_for_8:;
    }
}



extern "C" __global__ void compute_l0_runtime_active_kernel_df5f3df5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_l0_active,
    wp::array_t<wp::int32> var_l0_source_cell,
    wp::array_t<wp::int32> var_l1_cell_to_cluster,
    wp::array_t<wp::int32> var_l1_projection_active,
    wp::int32 var_has_l1_projection,
    wp::array_t<wp::int32> var_l2_cell_to_cluster,
    wp::array_t<wp::int32> var_l2_projection_active,
    wp::int32 var_has_l2_projection,
    wp::array_t<wp::int32> var_runtime_active,
    wp::array_t<wp::int32> var_sleeping_count)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        bool var_17;
        const wp::int32 var_18 = 0;
        bool var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::int32 var_26;
        wp::int32 var_27;
        bool var_28;
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        bool var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        wp::int32* var_39;
        const wp::int32 var_40 = 0;
        bool var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 0;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        const wp::int32 var_53 = 1;
        wp::int32 var_54;
        wp::int32 var_55;
        //---------
        // forward
        // def compute_l0_runtime_active_kernel(                                                  <L 1485>
        // cluster_idx = wp.tid()                                                                 <L 1497>
        var_0 = builtin_tid1d();
        // active = l0_active[cluster_idx]                                                        <L 1498>
        var_1 = wp::address(var_l0_active, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // value = active                                                                         <L 1499>
        var_4 = wp::copy(var_2);
        // slept = int(0)                                                                         <L 1500>
        var_6 = wp::int(var_5);
        // if active != 0:                                                                        <L 1502>
        var_8 = (var_2 != var_7);
        if (var_8) {
            // source_cell = l0_source_cell[cluster_idx]                                          <L 1503>
            var_9 = wp::address(var_l0_source_cell, var_0);
            var_11 = wp::load(var_9);
            var_10 = wp::copy(var_11);
            // if has_l2_projection != 0:                                                         <L 1504>
            var_13 = (var_has_l2_projection != var_12);
            if (var_13) {
                // parent_l2 = l2_cell_to_cluster[source_cell]                                    <L 1505>
                var_14 = wp::address(var_l2_cell_to_cluster, var_10);
                var_16 = wp::load(var_14);
                var_15 = wp::copy(var_16);
                // if parent_l2 >= 0 and l2_projection_active[parent_l2] != 0:                    <L 1506>
                var_19 = (var_15 >= var_18);
                var_17 = var_19;
                if (var_17) {
                    var_20 = wp::address(var_l2_projection_active, var_15);
                    var_23 = wp::load(var_20);
                    var_22 = (var_23 != var_21);
                    var_17 = var_17 && var_22;
                }
                if (var_17) {
                    // slept = int(1)                                                             <L 1507>
                    var_25 = wp::int(var_24);
                }
                var_26 = wp::where(var_17, var_25, var_6);
            }
            var_27 = wp::where(var_13, var_26, var_6);
            // if slept == 0 and has_l1_projection != 0:                                          <L 1508>
            var_30 = (var_27 == var_29);
            var_28 = var_30;
            if (var_28) {
                var_32 = (var_has_l1_projection != var_31);
                var_28 = var_28 && var_32;
            }
            if (var_28) {
                // parent_l1 = l1_cell_to_cluster[source_cell]                                    <L 1509>
                var_33 = wp::address(var_l1_cell_to_cluster, var_10);
                var_35 = wp::load(var_33);
                var_34 = wp::copy(var_35);
                // if parent_l1 >= 0 and l1_projection_active[parent_l1] != 0:                    <L 1510>
                var_38 = (var_34 >= var_37);
                var_36 = var_38;
                if (var_36) {
                    var_39 = wp::address(var_l1_projection_active, var_34);
                    var_42 = wp::load(var_39);
                    var_41 = (var_42 != var_40);
                    var_36 = var_36 && var_41;
                }
                if (var_36) {
                    // slept = int(1)                                                             <L 1511>
                    var_44 = wp::int(var_43);
                }
                var_45 = wp::where(var_36, var_44, var_27);
            }
            var_46 = wp::where(var_28, var_45, var_27);
        }
        var_47 = wp::where(var_8, var_46, var_6);
        // if slept != 0:                                                                         <L 1513>
        var_49 = (var_47 != var_48);
        if (var_49) {
            // value = int(0)                                                                     <L 1514>
            var_51 = wp::int(var_50);
            // wp.atomic_add(sleeping_count, 0, 1)                                                <L 1515>
            var_54 = wp::atomic_add(var_sleeping_count, var_52, var_53);
        }
        var_55 = wp::where(var_49, var_51, var_4);
        // runtime_active[cluster_idx] = value                                                    <L 1517>
        wp::array_store(var_runtime_active, var_0, var_55);
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform8_colored_gs_template_active_79bf4da3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_color_cluster_indices,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_color_start,
    wp::float32 var_stiffness,
    wp::float32 var_support_alpha,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations)
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::float32 var_21 = 0.0;
        const wp::float32 var_22 = 0.0;
        const wp::float32 var_23 = 0.0;
        wp::vec_t<3, wp::float32> var_24;
        const wp::float32 var_25 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::mat_t<3, 3, wp::float32> var_41;
        wp::mat_t<3, 3, wp::float32> var_42;
        const wp::int32 var_43 = 1;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32>* var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::mat_t<3, 3, wp::float32> var_57;
        wp::mat_t<3, 3, wp::float32> var_58;
        const wp::int32 var_59 = 2;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32>* var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::mat_t<3, 3, wp::float32> var_73;
        wp::mat_t<3, 3, wp::float32> var_74;
        const wp::int32 var_75 = 3;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        wp::vec_t<3, wp::float32>* var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32>* var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::mat_t<3, 3, wp::float32> var_89;
        wp::mat_t<3, 3, wp::float32> var_90;
        const wp::int32 var_91 = 4;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::int32* var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::vec_t<3, wp::float32>* var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32>* var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::mat_t<3, 3, wp::float32> var_105;
        wp::mat_t<3, 3, wp::float32> var_106;
        const wp::int32 var_107 = 5;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::vec_t<3, wp::float32>* var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32>* var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::mat_t<3, 3, wp::float32> var_121;
        wp::mat_t<3, 3, wp::float32> var_122;
        const wp::int32 var_123 = 6;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32* var_126;
        wp::int32 var_127;
        wp::int32 var_128;
        wp::vec_t<3, wp::float32>* var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32>* var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::vec_t<3, wp::float32> var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::mat_t<3, 3, wp::float32> var_137;
        wp::mat_t<3, 3, wp::float32> var_138;
        const wp::int32 var_139 = 7;
        wp::int32 var_140;
        wp::int32 var_141;
        wp::int32* var_142;
        wp::int32 var_143;
        wp::int32 var_144;
        wp::vec_t<3, wp::float32>* var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32>* var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::mat_t<3, 3, wp::float32> var_153;
        wp::mat_t<3, 3, wp::float32> var_154;
        const wp::float32 var_155 = 0.125;
        wp::vec_t<3, wp::float32> var_156;
        wp::mat_t<3, 3, wp::float32> var_157;
        wp::mat_t<3, 3, wp::float32> var_158;
        wp::quat_t<wp::float32>* var_159;
        wp::quat_t<wp::float32> var_160;
        wp::quat_t<wp::float32> var_161;
        wp::quat_t<wp::float32> var_162;
        wp::float32 var_163;
        const wp::float32 var_164 = 0.0;
        bool var_165;
        const wp::int32 var_166 = 0;
        wp::float32 var_167;
        wp::float32 var_168;
        const wp::int32 var_169 = 1;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 2;
        wp::float32 var_173;
        wp::float32 var_174;
        const wp::int32 var_175 = 3;
        wp::float32 var_176;
        wp::float32 var_177;
        wp::quat_t<wp::float32> var_178;
        wp::quat_t<wp::float32> var_179;
        const wp::int32 var_180 = 0;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::int32* var_183;
        wp::int32 var_184;
        wp::int32 var_185;
        wp::vec_t<3, wp::float32>* var_186;
        wp::vec_t<3, wp::float32> var_187;
        wp::vec_t<3, wp::float32> var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::float32 var_190;
        wp::float32* var_191;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        const wp::float32 var_195 = 0.0;
        bool var_196;
        wp::vec_t<3, wp::float32>* var_197;
        wp::vec_t<3, wp::float32>* var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::vec_t<3, wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        wp::vec_t<3, wp::float32> var_203;
        const wp::int32 var_204 = 1;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32* var_207;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::vec_t<3, wp::float32>* var_210;
        wp::vec_t<3, wp::float32> var_211;
        wp::vec_t<3, wp::float32> var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::float32 var_214;
        wp::float32* var_215;
        wp::float32 var_216;
        wp::float32 var_217;
        wp::float32 var_218;
        const wp::float32 var_219 = 0.0;
        bool var_220;
        wp::vec_t<3, wp::float32>* var_221;
        wp::vec_t<3, wp::float32>* var_222;
        wp::vec_t<3, wp::float32> var_223;
        wp::vec_t<3, wp::float32> var_224;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        const wp::int32 var_229 = 2;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::int32* var_232;
        wp::int32 var_233;
        wp::int32 var_234;
        wp::vec_t<3, wp::float32>* var_235;
        wp::vec_t<3, wp::float32> var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::vec_t<3, wp::float32> var_238;
        wp::float32 var_239;
        wp::float32* var_240;
        wp::float32 var_241;
        wp::float32 var_242;
        wp::float32 var_243;
        const wp::float32 var_244 = 0.0;
        bool var_245;
        wp::vec_t<3, wp::float32>* var_246;
        wp::vec_t<3, wp::float32>* var_247;
        wp::vec_t<3, wp::float32> var_248;
        wp::vec_t<3, wp::float32> var_249;
        wp::vec_t<3, wp::float32> var_250;
        wp::vec_t<3, wp::float32> var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<3, wp::float32> var_253;
        const wp::int32 var_254 = 3;
        wp::int32 var_255;
        wp::int32 var_256;
        wp::int32* var_257;
        wp::int32 var_258;
        wp::int32 var_259;
        wp::vec_t<3, wp::float32>* var_260;
        wp::vec_t<3, wp::float32> var_261;
        wp::vec_t<3, wp::float32> var_262;
        wp::vec_t<3, wp::float32> var_263;
        wp::float32 var_264;
        wp::float32* var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::float32 var_269 = 0.0;
        bool var_270;
        wp::vec_t<3, wp::float32>* var_271;
        wp::vec_t<3, wp::float32>* var_272;
        wp::vec_t<3, wp::float32> var_273;
        wp::vec_t<3, wp::float32> var_274;
        wp::vec_t<3, wp::float32> var_275;
        wp::vec_t<3, wp::float32> var_276;
        wp::vec_t<3, wp::float32> var_277;
        wp::vec_t<3, wp::float32> var_278;
        const wp::int32 var_279 = 4;
        wp::int32 var_280;
        wp::int32 var_281;
        wp::int32* var_282;
        wp::int32 var_283;
        wp::int32 var_284;
        wp::vec_t<3, wp::float32>* var_285;
        wp::vec_t<3, wp::float32> var_286;
        wp::vec_t<3, wp::float32> var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::float32 var_289;
        wp::float32* var_290;
        wp::float32 var_291;
        wp::float32 var_292;
        wp::float32 var_293;
        const wp::float32 var_294 = 0.0;
        bool var_295;
        wp::vec_t<3, wp::float32>* var_296;
        wp::vec_t<3, wp::float32>* var_297;
        wp::vec_t<3, wp::float32> var_298;
        wp::vec_t<3, wp::float32> var_299;
        wp::vec_t<3, wp::float32> var_300;
        wp::vec_t<3, wp::float32> var_301;
        wp::vec_t<3, wp::float32> var_302;
        wp::vec_t<3, wp::float32> var_303;
        const wp::int32 var_304 = 5;
        wp::int32 var_305;
        wp::int32 var_306;
        wp::int32* var_307;
        wp::int32 var_308;
        wp::int32 var_309;
        wp::vec_t<3, wp::float32>* var_310;
        wp::vec_t<3, wp::float32> var_311;
        wp::vec_t<3, wp::float32> var_312;
        wp::vec_t<3, wp::float32> var_313;
        wp::float32 var_314;
        wp::float32* var_315;
        wp::float32 var_316;
        wp::float32 var_317;
        wp::float32 var_318;
        const wp::float32 var_319 = 0.0;
        bool var_320;
        wp::vec_t<3, wp::float32>* var_321;
        wp::vec_t<3, wp::float32>* var_322;
        wp::vec_t<3, wp::float32> var_323;
        wp::vec_t<3, wp::float32> var_324;
        wp::vec_t<3, wp::float32> var_325;
        wp::vec_t<3, wp::float32> var_326;
        wp::vec_t<3, wp::float32> var_327;
        wp::vec_t<3, wp::float32> var_328;
        const wp::int32 var_329 = 6;
        wp::int32 var_330;
        wp::int32 var_331;
        wp::int32* var_332;
        wp::int32 var_333;
        wp::int32 var_334;
        wp::vec_t<3, wp::float32>* var_335;
        wp::vec_t<3, wp::float32> var_336;
        wp::vec_t<3, wp::float32> var_337;
        wp::vec_t<3, wp::float32> var_338;
        wp::float32 var_339;
        wp::float32* var_340;
        wp::float32 var_341;
        wp::float32 var_342;
        wp::float32 var_343;
        const wp::float32 var_344 = 0.0;
        bool var_345;
        wp::vec_t<3, wp::float32>* var_346;
        wp::vec_t<3, wp::float32>* var_347;
        wp::vec_t<3, wp::float32> var_348;
        wp::vec_t<3, wp::float32> var_349;
        wp::vec_t<3, wp::float32> var_350;
        wp::vec_t<3, wp::float32> var_351;
        wp::vec_t<3, wp::float32> var_352;
        wp::vec_t<3, wp::float32> var_353;
        const wp::int32 var_354 = 7;
        wp::int32 var_355;
        wp::int32 var_356;
        wp::int32* var_357;
        wp::int32 var_358;
        wp::int32 var_359;
        wp::vec_t<3, wp::float32>* var_360;
        wp::vec_t<3, wp::float32> var_361;
        wp::vec_t<3, wp::float32> var_362;
        wp::vec_t<3, wp::float32> var_363;
        wp::float32 var_364;
        wp::float32* var_365;
        wp::float32 var_366;
        wp::float32 var_367;
        wp::float32 var_368;
        const wp::float32 var_369 = 0.0;
        bool var_370;
        wp::vec_t<3, wp::float32>* var_371;
        wp::vec_t<3, wp::float32>* var_372;
        wp::vec_t<3, wp::float32> var_373;
        wp::vec_t<3, wp::float32> var_374;
        wp::vec_t<3, wp::float32> var_375;
        wp::vec_t<3, wp::float32> var_376;
        wp::vec_t<3, wp::float32> var_377;
        wp::vec_t<3, wp::float32> var_378;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform8_colored_gs_template_active(                 <L 601>
        // cluster_idx = color_cluster_indices[color_start + wp.tid()]                            <L 616>
        var_0 = builtin_tid1d();
        var_1 = wp::add(var_color_start, var_0);
        var_2 = wp::address(var_color_cluster_indices, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 618>
        var_6 = wp::address(var_cluster_active, var_3);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_5 = var_8;
        if (!var_5) {
            var_11 = (var_stiffness <= var_10);
            var_5 = var_5 || var_11;
        }
        if (var_5) {
            // return                                                                             <L 619>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 621>
        var_12 = wp::address(var_coefficients, var_3);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if coeff <= 0.0:                                                                       <L 622>
        var_16 = (var_13 <= var_15);
        if (var_16) {
            // return                                                                             <L 623>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 625>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // rest_sum = wp.vec3(0.0, 0.0, 0.0)                                                      <L 626>
        var_24 = wp::vec_t<3, wp::float32>(var_21, var_22, var_23);
        // covariance = wp.mat33(0.0)                                                             <L 627>
        var_26 = wp::mat_t<3, 3, wp::float32>(var_25);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 628>
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_28 = wp::mul(var_27, var_cluster_count);
        var_29 = wp::add(var_28, var_3);
        var_30 = wp::address(var_indices_by_slot, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_33 = wp::address(var_particle_q, var_31);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_36 = wp::address(var_rest_local_template, var_27);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // center += x                                                                            <L 632>
        var_39 = wp::add(var_20, var_34);
        // rest_sum += q_rel                                                                      <L 633>
        var_40 = wp::add(var_24, var_37);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_41 = wp::outer(var_37, var_34);
        var_42 = wp::add(var_26, var_41);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_44 = wp::mul(var_43, var_cluster_count);
        var_45 = wp::add(var_44, var_3);
        var_46 = wp::address(var_indices_by_slot, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_49 = wp::address(var_particle_q, var_47);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_52 = wp::address(var_rest_local_template, var_43);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // center += x                                                                            <L 632>
        var_55 = wp::add(var_39, var_50);
        // rest_sum += q_rel                                                                      <L 633>
        var_56 = wp::add(var_40, var_53);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_57 = wp::outer(var_53, var_50);
        var_58 = wp::add(var_42, var_57);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_60 = wp::mul(var_59, var_cluster_count);
        var_61 = wp::add(var_60, var_3);
        var_62 = wp::address(var_indices_by_slot, var_61);
        var_64 = wp::load(var_62);
        var_63 = wp::copy(var_64);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_65 = wp::address(var_particle_q, var_63);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_68 = wp::address(var_rest_local_template, var_59);
        var_70 = wp::load(var_68);
        var_69 = wp::copy(var_70);
        // center += x                                                                            <L 632>
        var_71 = wp::add(var_55, var_66);
        // rest_sum += q_rel                                                                      <L 633>
        var_72 = wp::add(var_56, var_69);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_73 = wp::outer(var_69, var_66);
        var_74 = wp::add(var_58, var_73);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_76 = wp::mul(var_75, var_cluster_count);
        var_77 = wp::add(var_76, var_3);
        var_78 = wp::address(var_indices_by_slot, var_77);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_81 = wp::address(var_particle_q, var_79);
        var_83 = wp::load(var_81);
        var_82 = wp::copy(var_83);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_84 = wp::address(var_rest_local_template, var_75);
        var_86 = wp::load(var_84);
        var_85 = wp::copy(var_86);
        // center += x                                                                            <L 632>
        var_87 = wp::add(var_71, var_82);
        // rest_sum += q_rel                                                                      <L 633>
        var_88 = wp::add(var_72, var_85);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_89 = wp::outer(var_85, var_82);
        var_90 = wp::add(var_74, var_89);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_92 = wp::mul(var_91, var_cluster_count);
        var_93 = wp::add(var_92, var_3);
        var_94 = wp::address(var_indices_by_slot, var_93);
        var_96 = wp::load(var_94);
        var_95 = wp::copy(var_96);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_97 = wp::address(var_particle_q, var_95);
        var_99 = wp::load(var_97);
        var_98 = wp::copy(var_99);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_100 = wp::address(var_rest_local_template, var_91);
        var_102 = wp::load(var_100);
        var_101 = wp::copy(var_102);
        // center += x                                                                            <L 632>
        var_103 = wp::add(var_87, var_98);
        // rest_sum += q_rel                                                                      <L 633>
        var_104 = wp::add(var_88, var_101);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_105 = wp::outer(var_101, var_98);
        var_106 = wp::add(var_90, var_105);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_108 = wp::mul(var_107, var_cluster_count);
        var_109 = wp::add(var_108, var_3);
        var_110 = wp::address(var_indices_by_slot, var_109);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_113 = wp::address(var_particle_q, var_111);
        var_115 = wp::load(var_113);
        var_114 = wp::copy(var_115);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_116 = wp::address(var_rest_local_template, var_107);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // center += x                                                                            <L 632>
        var_119 = wp::add(var_103, var_114);
        // rest_sum += q_rel                                                                      <L 633>
        var_120 = wp::add(var_104, var_117);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_121 = wp::outer(var_117, var_114);
        var_122 = wp::add(var_106, var_121);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_124 = wp::mul(var_123, var_cluster_count);
        var_125 = wp::add(var_124, var_3);
        var_126 = wp::address(var_indices_by_slot, var_125);
        var_128 = wp::load(var_126);
        var_127 = wp::copy(var_128);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_129 = wp::address(var_particle_q, var_127);
        var_131 = wp::load(var_129);
        var_130 = wp::copy(var_131);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_132 = wp::address(var_rest_local_template, var_123);
        var_134 = wp::load(var_132);
        var_133 = wp::copy(var_134);
        // center += x                                                                            <L 632>
        var_135 = wp::add(var_119, var_130);
        // rest_sum += q_rel                                                                      <L 633>
        var_136 = wp::add(var_120, var_133);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_137 = wp::outer(var_133, var_130);
        var_138 = wp::add(var_122, var_137);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 629>
        var_140 = wp::mul(var_139, var_cluster_count);
        var_141 = wp::add(var_140, var_3);
        var_142 = wp::address(var_indices_by_slot, var_141);
        var_144 = wp::load(var_142);
        var_143 = wp::copy(var_144);
        // x = particle_q[particle_idx]                                                           <L 630>
        var_145 = wp::address(var_particle_q, var_143);
        var_147 = wp::load(var_145);
        var_146 = wp::copy(var_147);
        // q_rel = rest_local_template[local_idx]                                                 <L 631>
        var_148 = wp::address(var_rest_local_template, var_139);
        var_150 = wp::load(var_148);
        var_149 = wp::copy(var_150);
        // center += x                                                                            <L 632>
        var_151 = wp::add(var_135, var_146);
        // rest_sum += q_rel                                                                      <L 633>
        var_152 = wp::add(var_136, var_149);
        // covariance += wp.outer(q_rel, x)                                                       <L 634>
        var_153 = wp::outer(var_149, var_146);
        var_154 = wp::add(var_138, var_153);
        // center *= 0.125                                                                        <L 636>
        var_156 = wp::mul(var_151, var_155);
        // covariance -= wp.outer(rest_sum, center)                                               <L 637>
        var_157 = wp::outer(var_152, var_156);
        var_158 = wp::sub(var_154, var_157);
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 639>
        var_159 = wp::address(var_cluster_rotations, var_3);
        var_161 = wp::load(var_159);
        var_160 = wp::copy(var_161);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 640>
        var_162 = _extract_rotation_0(var_158, var_160, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 641>
        var_163 = _quat_dot_0(var_162, var_160);
        var_165 = (var_163 < var_164);
        if (var_165) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 642>
            var_167 = wp::extract(var_162, var_166);
            var_168 = wp::neg(var_167);
            var_170 = wp::extract(var_162, var_169);
            var_171 = wp::neg(var_170);
            var_173 = wp::extract(var_162, var_172);
            var_174 = wp::neg(var_173);
            var_176 = wp::extract(var_162, var_175);
            var_177 = wp::neg(var_176);
            var_178 = wp::quat_t<wp::float32>(var_168, var_171, var_174, var_177);
        }
        var_179 = wp::where(var_165, var_178, var_162);
        // cluster_rotations[cluster_idx] = rotation                                              <L 644>
        wp::array_store(var_cluster_rotations, var_3, var_179);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 646>
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_181 = wp::mul(var_180, var_cluster_count);
        var_182 = wp::add(var_181, var_3);
        var_183 = wp::address(var_indices_by_slot, var_182);
        var_185 = wp::load(var_183);
        var_184 = wp::copy(var_185);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_186 = wp::address(var_rest_local_template, var_180);
        var_188 = wp::load(var_186);
        var_187 = wp::quat_rotate(var_179, var_188);
        var_189 = wp::add(var_156, var_187);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_190 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_191 = wp::address(var_particle_cluster_inv_weights, var_184);
        // support_alpha,                                                                         <L 651>
        var_193 = wp::load(var_191);
        var_192 = _support_scale_from_alpha_0(var_193, var_support_alpha);
        var_194 = wp::mul(var_190, var_192);
        // if particle_scale > 0.0:                                                               <L 653>
        var_196 = (var_194 > var_195);
        if (var_196) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_197 = wp::address(var_particle_q, var_184);
            var_198 = wp::address(var_particle_q, var_184);
            var_200 = wp::load(var_198);
            var_199 = wp::sub(var_189, var_200);
            var_201 = wp::mul(var_199, var_194);
            var_203 = wp::load(var_197);
            var_202 = wp::add(var_203, var_201);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_184, var_202);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_205 = wp::mul(var_204, var_cluster_count);
        var_206 = wp::add(var_205, var_3);
        var_207 = wp::address(var_indices_by_slot, var_206);
        var_209 = wp::load(var_207);
        var_208 = wp::copy(var_209);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_210 = wp::address(var_rest_local_template, var_204);
        var_212 = wp::load(var_210);
        var_211 = wp::quat_rotate(var_179, var_212);
        var_213 = wp::add(var_156, var_211);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_214 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_215 = wp::address(var_particle_cluster_inv_weights, var_208);
        // support_alpha,                                                                         <L 651>
        var_217 = wp::load(var_215);
        var_216 = _support_scale_from_alpha_0(var_217, var_support_alpha);
        var_218 = wp::mul(var_214, var_216);
        // if particle_scale > 0.0:                                                               <L 653>
        var_220 = (var_218 > var_219);
        if (var_220) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_221 = wp::address(var_particle_q, var_208);
            var_222 = wp::address(var_particle_q, var_208);
            var_224 = wp::load(var_222);
            var_223 = wp::sub(var_213, var_224);
            var_225 = wp::mul(var_223, var_218);
            var_227 = wp::load(var_221);
            var_226 = wp::add(var_227, var_225);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_208, var_226);
        }
        var_228 = wp::where(var_220, var_226, var_202);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_230 = wp::mul(var_229, var_cluster_count);
        var_231 = wp::add(var_230, var_3);
        var_232 = wp::address(var_indices_by_slot, var_231);
        var_234 = wp::load(var_232);
        var_233 = wp::copy(var_234);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_235 = wp::address(var_rest_local_template, var_229);
        var_237 = wp::load(var_235);
        var_236 = wp::quat_rotate(var_179, var_237);
        var_238 = wp::add(var_156, var_236);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_239 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_240 = wp::address(var_particle_cluster_inv_weights, var_233);
        // support_alpha,                                                                         <L 651>
        var_242 = wp::load(var_240);
        var_241 = _support_scale_from_alpha_0(var_242, var_support_alpha);
        var_243 = wp::mul(var_239, var_241);
        // if particle_scale > 0.0:                                                               <L 653>
        var_245 = (var_243 > var_244);
        if (var_245) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_246 = wp::address(var_particle_q, var_233);
            var_247 = wp::address(var_particle_q, var_233);
            var_249 = wp::load(var_247);
            var_248 = wp::sub(var_238, var_249);
            var_250 = wp::mul(var_248, var_243);
            var_252 = wp::load(var_246);
            var_251 = wp::add(var_252, var_250);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_233, var_251);
        }
        var_253 = wp::where(var_245, var_251, var_228);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_255 = wp::mul(var_254, var_cluster_count);
        var_256 = wp::add(var_255, var_3);
        var_257 = wp::address(var_indices_by_slot, var_256);
        var_259 = wp::load(var_257);
        var_258 = wp::copy(var_259);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_260 = wp::address(var_rest_local_template, var_254);
        var_262 = wp::load(var_260);
        var_261 = wp::quat_rotate(var_179, var_262);
        var_263 = wp::add(var_156, var_261);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_264 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_265 = wp::address(var_particle_cluster_inv_weights, var_258);
        // support_alpha,                                                                         <L 651>
        var_267 = wp::load(var_265);
        var_266 = _support_scale_from_alpha_0(var_267, var_support_alpha);
        var_268 = wp::mul(var_264, var_266);
        // if particle_scale > 0.0:                                                               <L 653>
        var_270 = (var_268 > var_269);
        if (var_270) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_271 = wp::address(var_particle_q, var_258);
            var_272 = wp::address(var_particle_q, var_258);
            var_274 = wp::load(var_272);
            var_273 = wp::sub(var_263, var_274);
            var_275 = wp::mul(var_273, var_268);
            var_277 = wp::load(var_271);
            var_276 = wp::add(var_277, var_275);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_258, var_276);
        }
        var_278 = wp::where(var_270, var_276, var_253);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_280 = wp::mul(var_279, var_cluster_count);
        var_281 = wp::add(var_280, var_3);
        var_282 = wp::address(var_indices_by_slot, var_281);
        var_284 = wp::load(var_282);
        var_283 = wp::copy(var_284);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_285 = wp::address(var_rest_local_template, var_279);
        var_287 = wp::load(var_285);
        var_286 = wp::quat_rotate(var_179, var_287);
        var_288 = wp::add(var_156, var_286);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_289 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_290 = wp::address(var_particle_cluster_inv_weights, var_283);
        // support_alpha,                                                                         <L 651>
        var_292 = wp::load(var_290);
        var_291 = _support_scale_from_alpha_0(var_292, var_support_alpha);
        var_293 = wp::mul(var_289, var_291);
        // if particle_scale > 0.0:                                                               <L 653>
        var_295 = (var_293 > var_294);
        if (var_295) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_296 = wp::address(var_particle_q, var_283);
            var_297 = wp::address(var_particle_q, var_283);
            var_299 = wp::load(var_297);
            var_298 = wp::sub(var_288, var_299);
            var_300 = wp::mul(var_298, var_293);
            var_302 = wp::load(var_296);
            var_301 = wp::add(var_302, var_300);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_283, var_301);
        }
        var_303 = wp::where(var_295, var_301, var_278);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_305 = wp::mul(var_304, var_cluster_count);
        var_306 = wp::add(var_305, var_3);
        var_307 = wp::address(var_indices_by_slot, var_306);
        var_309 = wp::load(var_307);
        var_308 = wp::copy(var_309);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_310 = wp::address(var_rest_local_template, var_304);
        var_312 = wp::load(var_310);
        var_311 = wp::quat_rotate(var_179, var_312);
        var_313 = wp::add(var_156, var_311);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_314 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_315 = wp::address(var_particle_cluster_inv_weights, var_308);
        // support_alpha,                                                                         <L 651>
        var_317 = wp::load(var_315);
        var_316 = _support_scale_from_alpha_0(var_317, var_support_alpha);
        var_318 = wp::mul(var_314, var_316);
        // if particle_scale > 0.0:                                                               <L 653>
        var_320 = (var_318 > var_319);
        if (var_320) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_321 = wp::address(var_particle_q, var_308);
            var_322 = wp::address(var_particle_q, var_308);
            var_324 = wp::load(var_322);
            var_323 = wp::sub(var_313, var_324);
            var_325 = wp::mul(var_323, var_318);
            var_327 = wp::load(var_321);
            var_326 = wp::add(var_327, var_325);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_308, var_326);
        }
        var_328 = wp::where(var_320, var_326, var_303);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_330 = wp::mul(var_329, var_cluster_count);
        var_331 = wp::add(var_330, var_3);
        var_332 = wp::address(var_indices_by_slot, var_331);
        var_334 = wp::load(var_332);
        var_333 = wp::copy(var_334);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_335 = wp::address(var_rest_local_template, var_329);
        var_337 = wp::load(var_335);
        var_336 = wp::quat_rotate(var_179, var_337);
        var_338 = wp::add(var_156, var_336);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_339 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_340 = wp::address(var_particle_cluster_inv_weights, var_333);
        // support_alpha,                                                                         <L 651>
        var_342 = wp::load(var_340);
        var_341 = _support_scale_from_alpha_0(var_342, var_support_alpha);
        var_343 = wp::mul(var_339, var_341);
        // if particle_scale > 0.0:                                                               <L 653>
        var_345 = (var_343 > var_344);
        if (var_345) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_346 = wp::address(var_particle_q, var_333);
            var_347 = wp::address(var_particle_q, var_333);
            var_349 = wp::load(var_347);
            var_348 = wp::sub(var_338, var_349);
            var_350 = wp::mul(var_348, var_343);
            var_352 = wp::load(var_346);
            var_351 = wp::add(var_352, var_350);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_333, var_351);
        }
        var_353 = wp::where(var_345, var_351, var_328);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 647>
        var_355 = wp::mul(var_354, var_cluster_count);
        var_356 = wp::add(var_355, var_3);
        var_357 = wp::address(var_indices_by_slot, var_356);
        var_359 = wp::load(var_357);
        var_358 = wp::copy(var_359);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 648>
        var_360 = wp::address(var_rest_local_template, var_354);
        var_362 = wp::load(var_360);
        var_361 = wp::quat_rotate(var_179, var_362);
        var_363 = wp::add(var_156, var_361);
        // particle_scale = coeff * stiffness * _support_scale_from_alpha(                        <L 649>
        var_364 = wp::mul(var_13, var_stiffness);
        // particle_cluster_inv_weights[particle_idx],                                            <L 650>
        var_365 = wp::address(var_particle_cluster_inv_weights, var_358);
        // support_alpha,                                                                         <L 651>
        var_367 = wp::load(var_365);
        var_366 = _support_scale_from_alpha_0(var_367, var_support_alpha);
        var_368 = wp::mul(var_364, var_366);
        // if particle_scale > 0.0:                                                               <L 653>
        var_370 = (var_368 > var_369);
        if (var_370) {
            // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 654>
            var_371 = wp::address(var_particle_q, var_358);
            var_372 = wp::address(var_particle_q, var_358);
            var_374 = wp::load(var_372);
            var_373 = wp::sub(var_363, var_374);
            var_375 = wp::mul(var_373, var_368);
            var_377 = wp::load(var_371);
            var_376 = wp::add(var_377, var_375);
            // particle_q[particle_idx] = x_new                                                   <L 655>
            wp::array_store(var_particle_q, var_358, var_376);
        }
        var_378 = wp::where(var_370, var_376, var_353);
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform125_colored_gs_7be93f58_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_color_cluster_indices,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::int32 var_color_start,
    wp::float32 var_stiffness,
    wp::float32 var_support_alpha,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations)
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
        wp::int32 var_1;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        const wp::float32 var_10 = 0.0;
        bool var_11;
        wp::float32* var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        const wp::float32 var_15 = 0.0;
        bool var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::int32 var_21 = 0;
        wp::int32 var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 125;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 1;
        wp::int32 var_43;
        wp::float32* var_44;
        const wp::float32 var_45 = 0.0;
        bool var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        bool var_52;
        wp::float32 var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::float32 var_55 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_56;
        const wp::int32 var_57 = 125;
        wp::range_t var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32* var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32* var_65;
        const wp::int32 var_66 = 1;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        bool var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32>* var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        wp::int32 var_80;
        wp::int32 var_81;
        wp::vec_t<3, wp::float32>* var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        wp::mat_t<3, 3, wp::float32> var_86;
        wp::mat_t<3, 3, wp::float32> var_87;
        wp::quat_t<wp::float32>* var_88;
        wp::quat_t<wp::float32> var_89;
        wp::quat_t<wp::float32> var_90;
        wp::quat_t<wp::float32> var_91;
        wp::float32 var_92;
        const wp::float32 var_93 = 0.0;
        bool var_94;
        const wp::int32 var_95 = 0;
        wp::float32 var_96;
        wp::float32 var_97;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        wp::float32 var_103;
        const wp::int32 var_104 = 3;
        wp::float32 var_105;
        wp::float32 var_106;
        wp::quat_t<wp::float32> var_107;
        wp::quat_t<wp::float32> var_108;
        const wp::int32 var_109 = 0;
        bool var_110;
        const wp::int32 var_111 = 125;
        wp::range_t var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        bool var_119;
        wp::int32* var_120;
        const wp::int32 var_121 = 1;
        wp::int32 var_122;
        wp::int32 var_123;
        const wp::int32 var_124 = 0;
        bool var_125;
        wp::float32* var_126;
        const wp::float32 var_127 = 0.0;
        bool var_128;
        wp::float32 var_129;
        wp::int32 var_130;
        wp::vec_t<3, wp::float32>* var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 0;
        bool var_135;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32>* var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::float32 var_144;
        wp::float32* var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        const wp::float32 var_149 = 0.0;
        bool var_150;
        wp::vec_t<3, wp::float32>* var_151;
        wp::vec_t<3, wp::float32>* var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform125_colored_gs(                               <L 909>
        // cluster_idx = color_cluster_indices[color_start + wp.tid()]                            <L 928>
        var_0 = builtin_tid1d();
        var_1 = wp::add(var_color_start, var_0);
        var_2 = wp::address(var_color_cluster_indices, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 930>
        var_6 = wp::address(var_cluster_active, var_3);
        var_9 = wp::load(var_6);
        var_8 = (var_9 == var_7);
        var_5 = var_8;
        if (!var_5) {
            var_11 = (var_stiffness <= var_10);
            var_5 = var_5 || var_11;
        }
        if (var_5) {
            // return                                                                             <L 931>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 933>
        var_12 = wp::address(var_coefficients, var_3);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if coeff <= 0.0:                                                                       <L 934>
        var_16 = (var_13 <= var_15);
        if (var_16) {
            // return                                                                             <L 935>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 937>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // member_count = int(0)                                                                  <L 938>
        var_22 = wp::int(var_21);
        // dynamic_count = int(0)                                                                 <L 939>
        var_24 = wp::int(var_23);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):                           <L 941>
        var_26 = wp::range(var_25);
        start_for_2:;
            if (iter_cmp(var_26) == 0) goto end_for_2;
            var_27 = wp::iter_next(var_26);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 942>
            var_28 = wp::mul(var_27, var_cluster_count);
            var_29 = wp::add(var_28, var_3);
            var_30 = wp::address(var_indices_by_slot, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 943>
            var_33 = wp::address(var_particle_flags, var_31);
            var_36 = wp::load(var_33);
            var_35 = wp::bit_and(var_36, var_34);
            var_38 = (var_35 == var_37);
            if (var_38) {
                // continue                                                                       <L 944>
                goto start_for_2;
            }
            // center += particle_q[particle_idx]                                                 <L 945>
            var_39 = wp::address(var_particle_q, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::add(var_20, var_41);
            // member_count += 1                                                                  <L 946>
            var_43 = wp::add(var_22, var_42);
            // if particle_inv_mass[particle_idx] > 0.0:                                          <L 947>
            var_44 = wp::address(var_particle_inv_mass, var_31);
            var_47 = wp::load(var_44);
            var_46 = (var_47 > var_45);
            if (var_46) {
                // dynamic_count += 1                                                             <L 948>
                var_49 = wp::add(var_24, var_48);
            }
            var_50 = wp::where(var_46, var_49, var_24);
            wp::assign(var_20, var_40);
            wp::assign(var_22, var_43);
            wp::assign(var_24, var_50);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 950>
        var_52 = (var_22 == var_51);
        if (var_52) {
            // return                                                                             <L 951>
            continue;
        }
        // center /= float(member_count)                                                          <L 953>
        var_53 = wp::float(var_22);
        var_54 = wp::div(var_20, var_53);
        // covariance = wp.mat33(0.0)                                                             <L 955>
        var_56 = wp::mat_t<3, 3, wp::float32>(var_55);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):                           <L 956>
        var_58 = wp::range(var_57);
        start_for_5:;
            if (iter_cmp(var_58) == 0) goto end_for_5;
            var_59 = wp::iter_next(var_58);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 957>
            var_60 = wp::mul(var_59, var_cluster_count);
            var_61 = wp::add(var_60, var_3);
            var_62 = wp::address(var_indices_by_slot, var_61);
            var_64 = wp::load(var_62);
            var_63 = wp::copy(var_64);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 958>
            var_65 = wp::address(var_particle_flags, var_63);
            var_68 = wp::load(var_65);
            var_67 = wp::bit_and(var_68, var_66);
            var_70 = (var_67 == var_69);
            if (var_70) {
                // continue                                                                       <L 959>
                wp::assign(var_31, var_63);
                goto start_for_5;
            }
            var_71 = wp::where(var_70, var_31, var_63);
            // x_rel = particle_q[particle_idx] - center                                          <L 960>
            var_72 = wp::address(var_particle_q, var_71);
            var_74 = wp::load(var_72);
            var_73 = wp::sub(var_74, var_54);
            // q_rel = rest_local_template[local_idx]                                             <L 961>
            var_75 = wp::address(var_rest_local_template, var_59);
            var_77 = wp::load(var_75);
            var_76 = wp::copy(var_77);
            // if use_rest_local_template == 0:                                                   <L 962>
            var_79 = (var_use_rest_local_template == var_78);
            if (var_79) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 963>
                var_80 = wp::mul(var_59, var_cluster_count);
                var_81 = wp::add(var_80, var_3);
                var_82 = wp::address(var_rest_local_positions_by_slot, var_81);
                var_84 = wp::load(var_82);
                var_83 = wp::copy(var_84);
            }
            var_85 = wp::where(var_79, var_83, var_76);
            // covariance += wp.outer(q_rel, x_rel)                                               <L 964>
            var_86 = wp::outer(var_85, var_73);
            var_87 = wp::add(var_56, var_86);
            wp::assign(var_31, var_71);
            wp::assign(var_56, var_87);
            goto start_for_5;
        end_for_5:;
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 966>
        var_88 = wp::address(var_cluster_rotations, var_3);
        var_90 = wp::load(var_88);
        var_89 = wp::copy(var_90);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 967>
        var_91 = _extract_rotation_0(var_56, var_89, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 968>
        var_92 = _quat_dot_0(var_91, var_89);
        var_94 = (var_92 < var_93);
        if (var_94) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 969>
            var_96 = wp::extract(var_91, var_95);
            var_97 = wp::neg(var_96);
            var_99 = wp::extract(var_91, var_98);
            var_100 = wp::neg(var_99);
            var_102 = wp::extract(var_91, var_101);
            var_103 = wp::neg(var_102);
            var_105 = wp::extract(var_91, var_104);
            var_106 = wp::neg(var_105);
            var_107 = wp::quat_t<wp::float32>(var_97, var_100, var_103, var_106);
        }
        var_108 = wp::where(var_94, var_107, var_91);
        // cluster_rotations[cluster_idx] = rotation                                              <L 971>
        wp::array_store(var_cluster_rotations, var_3, var_108);
        // if dynamic_count == 0:                                                                 <L 973>
        var_110 = (var_24 == var_109);
        if (var_110) {
            // return                                                                             <L 974>
            continue;
        }
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_125)):                           <L 976>
        var_112 = wp::range(var_111);
        start_for_8:;
            if (iter_cmp(var_112) == 0) goto end_for_8;
            var_113 = wp::iter_next(var_112);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 977>
            var_114 = wp::mul(var_113, var_cluster_count);
            var_115 = wp::add(var_114, var_3);
            var_116 = wp::address(var_indices_by_slot, var_115);
            var_118 = wp::load(var_116);
            var_117 = wp::copy(var_118);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:       <L 978>
            var_120 = wp::address(var_particle_flags, var_117);
            var_123 = wp::load(var_120);
            var_122 = wp::bit_and(var_123, var_121);
            var_125 = (var_122 == var_124);
            var_119 = var_125;
            if (!var_119) {
                var_126 = wp::address(var_particle_inv_mass, var_117);
                var_129 = wp::load(var_126);
                var_128 = (var_129 <= var_127);
                var_119 = var_119 || var_128;
            }
            if (var_119) {
                // continue                                                                       <L 979>
                wp::assign(var_31, var_117);
                goto start_for_8;
            }
            var_130 = wp::where(var_119, var_31, var_117);
            // q_rel = rest_local_template[local_idx]                                             <L 981>
            var_131 = wp::address(var_rest_local_template, var_113);
            var_133 = wp::load(var_131);
            var_132 = wp::copy(var_133);
            // if use_rest_local_template == 0:                                                   <L 982>
            var_135 = (var_use_rest_local_template == var_134);
            if (var_135) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 983>
                var_136 = wp::mul(var_113, var_cluster_count);
                var_137 = wp::add(var_136, var_3);
                var_138 = wp::address(var_rest_local_positions_by_slot, var_137);
                var_140 = wp::load(var_138);
                var_139 = wp::copy(var_140);
            }
            var_141 = wp::where(var_135, var_139, var_132);
            // goal = center + wp.quat_rotate(rotation, q_rel)                                    <L 984>
            var_142 = wp::quat_rotate(var_108, var_141);
            var_143 = wp::add(var_54, var_142);
            // particle_scale = coeff * stiffness * _support_scale_from_alpha(                    <L 985>
            var_144 = wp::mul(var_13, var_stiffness);
            // particle_cluster_inv_weights[particle_idx],                                        <L 986>
            var_145 = wp::address(var_particle_cluster_inv_weights, var_130);
            // support_alpha,                                                                     <L 987>
            var_147 = wp::load(var_145);
            var_146 = _support_scale_from_alpha_0(var_147, var_support_alpha);
            var_148 = wp::mul(var_144, var_146);
            // if particle_scale > 0.0:                                                           <L 989>
            var_150 = (var_148 > var_149);
            if (var_150) {
                // x_new = particle_q[particle_idx] + (goal - particle_q[particle_idx]) * particle_scale       <L 990>
                var_151 = wp::address(var_particle_q, var_130);
                var_152 = wp::address(var_particle_q, var_130);
                var_154 = wp::load(var_152);
                var_153 = wp::sub(var_143, var_154);
                var_155 = wp::mul(var_153, var_148);
                var_157 = wp::load(var_151);
                var_156 = wp::add(var_157, var_155);
                // particle_q[particle_idx] = x_new                                               <L 991>
                wp::array_store(var_particle_q, var_130, var_156);
            }
            wp::assign(var_31, var_130);
            wp::assign(var_85, var_141);
            goto start_for_8;
        end_for_8:;
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform8_template_active_9e18c4c0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::float32 var_stiffness,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cluster_translations,
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
        bool var_1;
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::float32 var_21 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32>* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::mat_t<3, 3, wp::float32> var_37;
        wp::mat_t<3, 3, wp::float32> var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32* var_42;
        wp::int32 var_43;
        wp::int32 var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::mat_t<3, 3, wp::float32> var_53;
        wp::mat_t<3, 3, wp::float32> var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::vec_t<3, wp::float32>* var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        wp::vec_t<3, wp::float32>* var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::mat_t<3, 3, wp::float32> var_69;
        wp::mat_t<3, 3, wp::float32> var_70;
        const wp::int32 var_71 = 3;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32* var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32>* var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32>* var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::mat_t<3, 3, wp::float32> var_85;
        wp::mat_t<3, 3, wp::float32> var_86;
        const wp::int32 var_87 = 4;
        wp::int32 var_88;
        wp::int32 var_89;
        wp::int32* var_90;
        wp::int32 var_91;
        wp::int32 var_92;
        wp::vec_t<3, wp::float32>* var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32>* var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::vec_t<3, wp::float32> var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::mat_t<3, 3, wp::float32> var_101;
        wp::mat_t<3, 3, wp::float32> var_102;
        const wp::int32 var_103 = 5;
        wp::int32 var_104;
        wp::int32 var_105;
        wp::int32* var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        wp::vec_t<3, wp::float32>* var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32>* var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::mat_t<3, 3, wp::float32> var_117;
        wp::mat_t<3, 3, wp::float32> var_118;
        const wp::int32 var_119 = 6;
        wp::int32 var_120;
        wp::int32 var_121;
        wp::int32* var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        wp::vec_t<3, wp::float32>* var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32>* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32> var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::mat_t<3, 3, wp::float32> var_133;
        wp::mat_t<3, 3, wp::float32> var_134;
        const wp::int32 var_135 = 7;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::int32* var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        wp::vec_t<3, wp::float32>* var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32>* var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::mat_t<3, 3, wp::float32> var_149;
        wp::mat_t<3, 3, wp::float32> var_150;
        const wp::float32 var_151 = 0.125;
        wp::vec_t<3, wp::float32> var_152;
        wp::mat_t<3, 3, wp::float32> var_153;
        wp::mat_t<3, 3, wp::float32> var_154;
        wp::quat_t<wp::float32>* var_155;
        wp::quat_t<wp::float32> var_156;
        wp::quat_t<wp::float32> var_157;
        wp::quat_t<wp::float32> var_158;
        wp::float32 var_159;
        const wp::float32 var_160 = 0.0;
        bool var_161;
        const wp::int32 var_162 = 0;
        wp::float32 var_163;
        wp::float32 var_164;
        const wp::int32 var_165 = 1;
        wp::float32 var_166;
        wp::float32 var_167;
        const wp::int32 var_168 = 2;
        wp::float32 var_169;
        wp::float32 var_170;
        const wp::int32 var_171 = 3;
        wp::float32 var_172;
        wp::float32 var_173;
        wp::quat_t<wp::float32> var_174;
        wp::quat_t<wp::float32> var_175;
        const wp::int32 var_176 = 0;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32* var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::vec_t<3, wp::float32>* var_182;
        wp::vec_t<3, wp::float32> var_183;
        wp::vec_t<3, wp::float32> var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::float32 var_186;
        wp::float32* var_187;
        wp::float32 var_188;
        wp::float32 var_189;
        const wp::float32 var_190 = 0.0;
        bool var_191;
        wp::vec_t<3, wp::float32>* var_192;
        wp::vec_t<3, wp::float32> var_193;
        wp::vec_t<3, wp::float32> var_194;
        wp::vec_t<3, wp::float32> var_195;
        wp::vec_t<3, wp::float32> var_196;
        const wp::int32 var_197 = 1;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::int32* var_200;
        wp::int32 var_201;
        wp::int32 var_202;
        wp::vec_t<3, wp::float32>* var_203;
        wp::vec_t<3, wp::float32> var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        wp::float32 var_207;
        wp::float32* var_208;
        wp::float32 var_209;
        wp::float32 var_210;
        const wp::float32 var_211 = 0.0;
        bool var_212;
        wp::vec_t<3, wp::float32>* var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::vec_t<3, wp::float32> var_215;
        wp::vec_t<3, wp::float32> var_216;
        wp::vec_t<3, wp::float32> var_217;
        const wp::int32 var_218 = 2;
        wp::int32 var_219;
        wp::int32 var_220;
        wp::int32* var_221;
        wp::int32 var_222;
        wp::int32 var_223;
        wp::vec_t<3, wp::float32>* var_224;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::float32 var_228;
        wp::float32* var_229;
        wp::float32 var_230;
        wp::float32 var_231;
        const wp::float32 var_232 = 0.0;
        bool var_233;
        wp::vec_t<3, wp::float32>* var_234;
        wp::vec_t<3, wp::float32> var_235;
        wp::vec_t<3, wp::float32> var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::vec_t<3, wp::float32> var_238;
        const wp::int32 var_239 = 3;
        wp::int32 var_240;
        wp::int32 var_241;
        wp::int32* var_242;
        wp::int32 var_243;
        wp::int32 var_244;
        wp::vec_t<3, wp::float32>* var_245;
        wp::vec_t<3, wp::float32> var_246;
        wp::vec_t<3, wp::float32> var_247;
        wp::vec_t<3, wp::float32> var_248;
        wp::float32 var_249;
        wp::float32* var_250;
        wp::float32 var_251;
        wp::float32 var_252;
        const wp::float32 var_253 = 0.0;
        bool var_254;
        wp::vec_t<3, wp::float32>* var_255;
        wp::vec_t<3, wp::float32> var_256;
        wp::vec_t<3, wp::float32> var_257;
        wp::vec_t<3, wp::float32> var_258;
        wp::vec_t<3, wp::float32> var_259;
        const wp::int32 var_260 = 4;
        wp::int32 var_261;
        wp::int32 var_262;
        wp::int32* var_263;
        wp::int32 var_264;
        wp::int32 var_265;
        wp::vec_t<3, wp::float32>* var_266;
        wp::vec_t<3, wp::float32> var_267;
        wp::vec_t<3, wp::float32> var_268;
        wp::vec_t<3, wp::float32> var_269;
        wp::float32 var_270;
        wp::float32* var_271;
        wp::float32 var_272;
        wp::float32 var_273;
        const wp::float32 var_274 = 0.0;
        bool var_275;
        wp::vec_t<3, wp::float32>* var_276;
        wp::vec_t<3, wp::float32> var_277;
        wp::vec_t<3, wp::float32> var_278;
        wp::vec_t<3, wp::float32> var_279;
        wp::vec_t<3, wp::float32> var_280;
        const wp::int32 var_281 = 5;
        wp::int32 var_282;
        wp::int32 var_283;
        wp::int32* var_284;
        wp::int32 var_285;
        wp::int32 var_286;
        wp::vec_t<3, wp::float32>* var_287;
        wp::vec_t<3, wp::float32> var_288;
        wp::vec_t<3, wp::float32> var_289;
        wp::vec_t<3, wp::float32> var_290;
        wp::float32 var_291;
        wp::float32* var_292;
        wp::float32 var_293;
        wp::float32 var_294;
        const wp::float32 var_295 = 0.0;
        bool var_296;
        wp::vec_t<3, wp::float32>* var_297;
        wp::vec_t<3, wp::float32> var_298;
        wp::vec_t<3, wp::float32> var_299;
        wp::vec_t<3, wp::float32> var_300;
        wp::vec_t<3, wp::float32> var_301;
        const wp::int32 var_302 = 6;
        wp::int32 var_303;
        wp::int32 var_304;
        wp::int32* var_305;
        wp::int32 var_306;
        wp::int32 var_307;
        wp::vec_t<3, wp::float32>* var_308;
        wp::vec_t<3, wp::float32> var_309;
        wp::vec_t<3, wp::float32> var_310;
        wp::vec_t<3, wp::float32> var_311;
        wp::float32 var_312;
        wp::float32* var_313;
        wp::float32 var_314;
        wp::float32 var_315;
        const wp::float32 var_316 = 0.0;
        bool var_317;
        wp::vec_t<3, wp::float32>* var_318;
        wp::vec_t<3, wp::float32> var_319;
        wp::vec_t<3, wp::float32> var_320;
        wp::vec_t<3, wp::float32> var_321;
        wp::vec_t<3, wp::float32> var_322;
        const wp::int32 var_323 = 7;
        wp::int32 var_324;
        wp::int32 var_325;
        wp::int32* var_326;
        wp::int32 var_327;
        wp::int32 var_328;
        wp::vec_t<3, wp::float32>* var_329;
        wp::vec_t<3, wp::float32> var_330;
        wp::vec_t<3, wp::float32> var_331;
        wp::vec_t<3, wp::float32> var_332;
        wp::float32 var_333;
        wp::float32* var_334;
        wp::float32 var_335;
        wp::float32 var_336;
        const wp::float32 var_337 = 0.0;
        bool var_338;
        wp::vec_t<3, wp::float32>* var_339;
        wp::vec_t<3, wp::float32> var_340;
        wp::vec_t<3, wp::float32> var_341;
        wp::vec_t<3, wp::float32> var_342;
        wp::vec_t<3, wp::float32> var_343;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform8_template_active(                            <L 463>
        // cluster_idx = wp.tid()                                                                 <L 477>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 479>
        var_2 = wp::address(var_cluster_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_1 = var_4;
        if (!var_1) {
            var_7 = (var_stiffness <= var_6);
            var_1 = var_1 || var_7;
        }
        if (var_1) {
            // return                                                                             <L 480>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 482>
        var_8 = wp::address(var_coefficients, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if coeff <= 0.0:                                                                       <L 483>
        var_12 = (var_9 <= var_11);
        if (var_12) {
            // return                                                                             <L 484>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 486>
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // rest_sum = wp.vec3(0.0, 0.0, 0.0)                                                      <L 487>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // covariance = wp.mat33(0.0)                                                             <L 488>
        var_22 = wp::mat_t<3, 3, wp::float32>(var_21);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 489>
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_24 = wp::mul(var_23, var_cluster_count);
        var_25 = wp::add(var_24, var_0);
        var_26 = wp::address(var_indices_by_slot, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_29 = wp::address(var_particle_q, var_27);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_32 = wp::address(var_rest_local_template, var_23);
        var_34 = wp::load(var_32);
        var_33 = wp::copy(var_34);
        // center += x                                                                            <L 493>
        var_35 = wp::add(var_16, var_30);
        // rest_sum += q_rel                                                                      <L 494>
        var_36 = wp::add(var_20, var_33);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_37 = wp::outer(var_33, var_30);
        var_38 = wp::add(var_22, var_37);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_40 = wp::mul(var_39, var_cluster_count);
        var_41 = wp::add(var_40, var_0);
        var_42 = wp::address(var_indices_by_slot, var_41);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_45 = wp::address(var_particle_q, var_43);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_48 = wp::address(var_rest_local_template, var_39);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // center += x                                                                            <L 493>
        var_51 = wp::add(var_35, var_46);
        // rest_sum += q_rel                                                                      <L 494>
        var_52 = wp::add(var_36, var_49);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_53 = wp::outer(var_49, var_46);
        var_54 = wp::add(var_38, var_53);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_56 = wp::mul(var_55, var_cluster_count);
        var_57 = wp::add(var_56, var_0);
        var_58 = wp::address(var_indices_by_slot, var_57);
        var_60 = wp::load(var_58);
        var_59 = wp::copy(var_60);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_61 = wp::address(var_particle_q, var_59);
        var_63 = wp::load(var_61);
        var_62 = wp::copy(var_63);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_64 = wp::address(var_rest_local_template, var_55);
        var_66 = wp::load(var_64);
        var_65 = wp::copy(var_66);
        // center += x                                                                            <L 493>
        var_67 = wp::add(var_51, var_62);
        // rest_sum += q_rel                                                                      <L 494>
        var_68 = wp::add(var_52, var_65);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_69 = wp::outer(var_65, var_62);
        var_70 = wp::add(var_54, var_69);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_72 = wp::mul(var_71, var_cluster_count);
        var_73 = wp::add(var_72, var_0);
        var_74 = wp::address(var_indices_by_slot, var_73);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_77 = wp::address(var_particle_q, var_75);
        var_79 = wp::load(var_77);
        var_78 = wp::copy(var_79);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_80 = wp::address(var_rest_local_template, var_71);
        var_82 = wp::load(var_80);
        var_81 = wp::copy(var_82);
        // center += x                                                                            <L 493>
        var_83 = wp::add(var_67, var_78);
        // rest_sum += q_rel                                                                      <L 494>
        var_84 = wp::add(var_68, var_81);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_85 = wp::outer(var_81, var_78);
        var_86 = wp::add(var_70, var_85);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_88 = wp::mul(var_87, var_cluster_count);
        var_89 = wp::add(var_88, var_0);
        var_90 = wp::address(var_indices_by_slot, var_89);
        var_92 = wp::load(var_90);
        var_91 = wp::copy(var_92);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_93 = wp::address(var_particle_q, var_91);
        var_95 = wp::load(var_93);
        var_94 = wp::copy(var_95);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_96 = wp::address(var_rest_local_template, var_87);
        var_98 = wp::load(var_96);
        var_97 = wp::copy(var_98);
        // center += x                                                                            <L 493>
        var_99 = wp::add(var_83, var_94);
        // rest_sum += q_rel                                                                      <L 494>
        var_100 = wp::add(var_84, var_97);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_101 = wp::outer(var_97, var_94);
        var_102 = wp::add(var_86, var_101);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_104 = wp::mul(var_103, var_cluster_count);
        var_105 = wp::add(var_104, var_0);
        var_106 = wp::address(var_indices_by_slot, var_105);
        var_108 = wp::load(var_106);
        var_107 = wp::copy(var_108);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_109 = wp::address(var_particle_q, var_107);
        var_111 = wp::load(var_109);
        var_110 = wp::copy(var_111);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_112 = wp::address(var_rest_local_template, var_103);
        var_114 = wp::load(var_112);
        var_113 = wp::copy(var_114);
        // center += x                                                                            <L 493>
        var_115 = wp::add(var_99, var_110);
        // rest_sum += q_rel                                                                      <L 494>
        var_116 = wp::add(var_100, var_113);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_117 = wp::outer(var_113, var_110);
        var_118 = wp::add(var_102, var_117);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_120 = wp::mul(var_119, var_cluster_count);
        var_121 = wp::add(var_120, var_0);
        var_122 = wp::address(var_indices_by_slot, var_121);
        var_124 = wp::load(var_122);
        var_123 = wp::copy(var_124);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_125 = wp::address(var_particle_q, var_123);
        var_127 = wp::load(var_125);
        var_126 = wp::copy(var_127);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_128 = wp::address(var_rest_local_template, var_119);
        var_130 = wp::load(var_128);
        var_129 = wp::copy(var_130);
        // center += x                                                                            <L 493>
        var_131 = wp::add(var_115, var_126);
        // rest_sum += q_rel                                                                      <L 494>
        var_132 = wp::add(var_116, var_129);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_133 = wp::outer(var_129, var_126);
        var_134 = wp::add(var_118, var_133);
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 490>
        var_136 = wp::mul(var_135, var_cluster_count);
        var_137 = wp::add(var_136, var_0);
        var_138 = wp::address(var_indices_by_slot, var_137);
        var_140 = wp::load(var_138);
        var_139 = wp::copy(var_140);
        // x = particle_q[particle_idx]                                                           <L 491>
        var_141 = wp::address(var_particle_q, var_139);
        var_143 = wp::load(var_141);
        var_142 = wp::copy(var_143);
        // q_rel = rest_local_template[local_idx]                                                 <L 492>
        var_144 = wp::address(var_rest_local_template, var_135);
        var_146 = wp::load(var_144);
        var_145 = wp::copy(var_146);
        // center += x                                                                            <L 493>
        var_147 = wp::add(var_131, var_142);
        // rest_sum += q_rel                                                                      <L 494>
        var_148 = wp::add(var_132, var_145);
        // covariance += wp.outer(q_rel, x)                                                       <L 495>
        var_149 = wp::outer(var_145, var_142);
        var_150 = wp::add(var_134, var_149);
        // center *= 0.125                                                                        <L 497>
        var_152 = wp::mul(var_147, var_151);
        // covariance -= wp.outer(rest_sum, center)                                               <L 498>
        var_153 = wp::outer(var_148, var_152);
        var_154 = wp::sub(var_150, var_153);
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 500>
        var_155 = wp::address(var_cluster_rotations, var_0);
        var_157 = wp::load(var_155);
        var_156 = wp::copy(var_157);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 501>
        var_158 = _extract_rotation_0(var_154, var_156, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 502>
        var_159 = _quat_dot_0(var_158, var_156);
        var_161 = (var_159 < var_160);
        if (var_161) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 503>
            var_163 = wp::extract(var_158, var_162);
            var_164 = wp::neg(var_163);
            var_166 = wp::extract(var_158, var_165);
            var_167 = wp::neg(var_166);
            var_169 = wp::extract(var_158, var_168);
            var_170 = wp::neg(var_169);
            var_172 = wp::extract(var_158, var_171);
            var_173 = wp::neg(var_172);
            var_174 = wp::quat_t<wp::float32>(var_164, var_167, var_170, var_173);
        }
        var_175 = wp::where(var_161, var_174, var_158);
        // cluster_rotations[cluster_idx] = rotation                                              <L 505>
        wp::array_store(var_cluster_rotations, var_0, var_175);
        // cluster_translations[cluster_idx] = center                                             <L 506>
        wp::array_store(var_cluster_translations, var_0, var_152);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 508>
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_177 = wp::mul(var_176, var_cluster_count);
        var_178 = wp::add(var_177, var_0);
        var_179 = wp::address(var_indices_by_slot, var_178);
        var_181 = wp::load(var_179);
        var_180 = wp::copy(var_181);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_182 = wp::address(var_rest_local_template, var_176);
        var_184 = wp::load(var_182);
        var_183 = wp::quat_rotate(var_175, var_184);
        var_185 = wp::add(var_152, var_183);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_186 = wp::mul(var_9, var_stiffness);
        var_187 = wp::address(var_particle_cluster_inv_weights, var_180);
        var_189 = wp::load(var_187);
        var_188 = wp::mul(var_186, var_189);
        // if particle_scale > 0.0:                                                               <L 512>
        var_191 = (var_188 > var_190);
        if (var_191) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_192 = wp::address(var_particle_q, var_180);
            var_194 = wp::load(var_192);
            var_193 = wp::sub(var_185, var_194);
            var_195 = wp::mul(var_193, var_188);
            var_196 = wp::atomic_add(var_particle_deltas, var_180, var_195);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_198 = wp::mul(var_197, var_cluster_count);
        var_199 = wp::add(var_198, var_0);
        var_200 = wp::address(var_indices_by_slot, var_199);
        var_202 = wp::load(var_200);
        var_201 = wp::copy(var_202);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_203 = wp::address(var_rest_local_template, var_197);
        var_205 = wp::load(var_203);
        var_204 = wp::quat_rotate(var_175, var_205);
        var_206 = wp::add(var_152, var_204);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_207 = wp::mul(var_9, var_stiffness);
        var_208 = wp::address(var_particle_cluster_inv_weights, var_201);
        var_210 = wp::load(var_208);
        var_209 = wp::mul(var_207, var_210);
        // if particle_scale > 0.0:                                                               <L 512>
        var_212 = (var_209 > var_211);
        if (var_212) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_213 = wp::address(var_particle_q, var_201);
            var_215 = wp::load(var_213);
            var_214 = wp::sub(var_206, var_215);
            var_216 = wp::mul(var_214, var_209);
            var_217 = wp::atomic_add(var_particle_deltas, var_201, var_216);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_219 = wp::mul(var_218, var_cluster_count);
        var_220 = wp::add(var_219, var_0);
        var_221 = wp::address(var_indices_by_slot, var_220);
        var_223 = wp::load(var_221);
        var_222 = wp::copy(var_223);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_224 = wp::address(var_rest_local_template, var_218);
        var_226 = wp::load(var_224);
        var_225 = wp::quat_rotate(var_175, var_226);
        var_227 = wp::add(var_152, var_225);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_228 = wp::mul(var_9, var_stiffness);
        var_229 = wp::address(var_particle_cluster_inv_weights, var_222);
        var_231 = wp::load(var_229);
        var_230 = wp::mul(var_228, var_231);
        // if particle_scale > 0.0:                                                               <L 512>
        var_233 = (var_230 > var_232);
        if (var_233) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_234 = wp::address(var_particle_q, var_222);
            var_236 = wp::load(var_234);
            var_235 = wp::sub(var_227, var_236);
            var_237 = wp::mul(var_235, var_230);
            var_238 = wp::atomic_add(var_particle_deltas, var_222, var_237);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_240 = wp::mul(var_239, var_cluster_count);
        var_241 = wp::add(var_240, var_0);
        var_242 = wp::address(var_indices_by_slot, var_241);
        var_244 = wp::load(var_242);
        var_243 = wp::copy(var_244);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_245 = wp::address(var_rest_local_template, var_239);
        var_247 = wp::load(var_245);
        var_246 = wp::quat_rotate(var_175, var_247);
        var_248 = wp::add(var_152, var_246);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_249 = wp::mul(var_9, var_stiffness);
        var_250 = wp::address(var_particle_cluster_inv_weights, var_243);
        var_252 = wp::load(var_250);
        var_251 = wp::mul(var_249, var_252);
        // if particle_scale > 0.0:                                                               <L 512>
        var_254 = (var_251 > var_253);
        if (var_254) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_255 = wp::address(var_particle_q, var_243);
            var_257 = wp::load(var_255);
            var_256 = wp::sub(var_248, var_257);
            var_258 = wp::mul(var_256, var_251);
            var_259 = wp::atomic_add(var_particle_deltas, var_243, var_258);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_261 = wp::mul(var_260, var_cluster_count);
        var_262 = wp::add(var_261, var_0);
        var_263 = wp::address(var_indices_by_slot, var_262);
        var_265 = wp::load(var_263);
        var_264 = wp::copy(var_265);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_266 = wp::address(var_rest_local_template, var_260);
        var_268 = wp::load(var_266);
        var_267 = wp::quat_rotate(var_175, var_268);
        var_269 = wp::add(var_152, var_267);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_270 = wp::mul(var_9, var_stiffness);
        var_271 = wp::address(var_particle_cluster_inv_weights, var_264);
        var_273 = wp::load(var_271);
        var_272 = wp::mul(var_270, var_273);
        // if particle_scale > 0.0:                                                               <L 512>
        var_275 = (var_272 > var_274);
        if (var_275) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_276 = wp::address(var_particle_q, var_264);
            var_278 = wp::load(var_276);
            var_277 = wp::sub(var_269, var_278);
            var_279 = wp::mul(var_277, var_272);
            var_280 = wp::atomic_add(var_particle_deltas, var_264, var_279);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_282 = wp::mul(var_281, var_cluster_count);
        var_283 = wp::add(var_282, var_0);
        var_284 = wp::address(var_indices_by_slot, var_283);
        var_286 = wp::load(var_284);
        var_285 = wp::copy(var_286);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_287 = wp::address(var_rest_local_template, var_281);
        var_289 = wp::load(var_287);
        var_288 = wp::quat_rotate(var_175, var_289);
        var_290 = wp::add(var_152, var_288);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_291 = wp::mul(var_9, var_stiffness);
        var_292 = wp::address(var_particle_cluster_inv_weights, var_285);
        var_294 = wp::load(var_292);
        var_293 = wp::mul(var_291, var_294);
        // if particle_scale > 0.0:                                                               <L 512>
        var_296 = (var_293 > var_295);
        if (var_296) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_297 = wp::address(var_particle_q, var_285);
            var_299 = wp::load(var_297);
            var_298 = wp::sub(var_290, var_299);
            var_300 = wp::mul(var_298, var_293);
            var_301 = wp::atomic_add(var_particle_deltas, var_285, var_300);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_303 = wp::mul(var_302, var_cluster_count);
        var_304 = wp::add(var_303, var_0);
        var_305 = wp::address(var_indices_by_slot, var_304);
        var_307 = wp::load(var_305);
        var_306 = wp::copy(var_307);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_308 = wp::address(var_rest_local_template, var_302);
        var_310 = wp::load(var_308);
        var_309 = wp::quat_rotate(var_175, var_310);
        var_311 = wp::add(var_152, var_309);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_312 = wp::mul(var_9, var_stiffness);
        var_313 = wp::address(var_particle_cluster_inv_weights, var_306);
        var_315 = wp::load(var_313);
        var_314 = wp::mul(var_312, var_315);
        // if particle_scale > 0.0:                                                               <L 512>
        var_317 = (var_314 > var_316);
        if (var_317) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_318 = wp::address(var_particle_q, var_306);
            var_320 = wp::load(var_318);
            var_319 = wp::sub(var_311, var_320);
            var_321 = wp::mul(var_319, var_314);
            var_322 = wp::atomic_add(var_particle_deltas, var_306, var_321);
        }
        // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]                <L 509>
        var_324 = wp::mul(var_323, var_cluster_count);
        var_325 = wp::add(var_324, var_0);
        var_326 = wp::address(var_indices_by_slot, var_325);
        var_328 = wp::load(var_326);
        var_327 = wp::copy(var_328);
        // goal = center + wp.quat_rotate(rotation, rest_local_template[local_idx])               <L 510>
        var_329 = wp::address(var_rest_local_template, var_323);
        var_331 = wp::load(var_329);
        var_330 = wp::quat_rotate(var_175, var_331);
        var_332 = wp::add(var_152, var_330);
        // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]        <L 511>
        var_333 = wp::mul(var_9, var_stiffness);
        var_334 = wp::address(var_particle_cluster_inv_weights, var_327);
        var_336 = wp::load(var_334);
        var_335 = wp::mul(var_333, var_336);
        // if particle_scale > 0.0:                                                               <L 512>
        var_338 = (var_335 > var_337);
        if (var_338) {
            // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 513>
            var_339 = wp::address(var_particle_q, var_327);
            var_341 = wp::load(var_339);
            var_340 = wp::sub(var_332, var_341);
            var_342 = wp::mul(var_340, var_335);
            var_343 = wp::atomic_add(var_particle_deltas, var_327, var_342);
        }
    }
}



extern "C" __global__ void compute_coarse_sleep_projection_kernel_2836d096_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_block_keys,
    wp::array_t<wp::int32> var_sleepable,
    wp::array_t<wp::int32> var_wake_mask,
    wp::int32 var_level_enabled,
    wp::array_t<wp::int32> var_projection_active,
    wp::array_t<wp::int32> var_projection_count)
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
        const wp::int32 var_1 = 0;
        wp::int32 var_2;
        bool var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        wp::int32* var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 2;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        bool var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        wp::shape_t* var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::shape_t var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        wp::shape_t* var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        wp::shape_t var_39;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 2;
        wp::int32 var_45;
        wp::shape_t var_46;
        bool var_47;
        wp::int32* var_48;
        const wp::int32 var_49 = 0;
        bool var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        bool var_57;
        const wp::int32 var_58 = 0;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        //---------
        // forward
        // def compute_coarse_sleep_projection_kernel(                                            <L 1434>
        // cluster_idx = wp.tid()                                                                 <L 1443>
        var_0 = builtin_tid1d();
        // value = int(0)                                                                         <L 1444>
        var_2 = wp::int(var_1);
        // if level_enabled != 0 and cluster_active[cluster_idx] != 0 and sleepable[cluster_idx] != 0:       <L 1445>
        var_5 = (var_level_enabled != var_4);
        var_3 = var_5;
        if (var_3) {
            var_6 = wp::address(var_cluster_active, var_0);
            var_9 = wp::load(var_6);
            var_8 = (var_9 != var_7);
            var_3 = var_3 && var_8;
        }
        if (var_3) {
            var_10 = wp::address(var_sleepable, var_0);
            var_13 = wp::load(var_10);
            var_12 = (var_13 != var_11);
            var_3 = var_3 && var_12;
        }
        if (var_3) {
            // bx = block_keys[cluster_idx, 0]                                                    <L 1446>
            var_15 = wp::address(var_block_keys, var_0, var_14);
            var_17 = wp::load(var_15);
            var_16 = wp::copy(var_17);
            // by = block_keys[cluster_idx, 1]                                                    <L 1447>
            var_19 = wp::address(var_block_keys, var_0, var_18);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // bz = block_keys[cluster_idx, 2]                                                    <L 1448>
            var_23 = wp::address(var_block_keys, var_0, var_22);
            var_25 = wp::load(var_23);
            var_24 = wp::copy(var_25);
            // if (                                                                               <L 1449>
            // bx >= 0                                                                            <L 1450>
            var_28 = (var_16 >= var_27);
            var_26 = var_28;
            if (var_26) {
                // and bx < wake_mask.shape[0]                                                    <L 1451>
                var_29 = &(var_wake_mask.shape);
                var_32 = wp::load(var_29);
                var_31 = wp::extract(var_32, var_30);
                var_33 = (var_16 < var_31);
                var_26 = var_26 && var_33;
            }
            if (var_26) {
                // and by >= 0                                                                    <L 1452>
                var_35 = (var_20 >= var_34);
                var_26 = var_26 && var_35;
            }
            if (var_26) {
                // and by < wake_mask.shape[1]                                                    <L 1453>
                var_36 = &(var_wake_mask.shape);
                var_39 = wp::load(var_36);
                var_38 = wp::extract(var_39, var_37);
                var_40 = (var_20 < var_38);
                var_26 = var_26 && var_40;
            }
            if (var_26) {
                // and bz >= 0                                                                    <L 1454>
                var_42 = (var_24 >= var_41);
                var_26 = var_26 && var_42;
            }
            if (var_26) {
                // and bz < wake_mask.shape[2]                                                    <L 1455>
                var_43 = &(var_wake_mask.shape);
                var_46 = wp::load(var_43);
                var_45 = wp::extract(var_46, var_44);
                var_47 = (var_24 < var_45);
                var_26 = var_26 && var_47;
            }
            if (var_26) {
                // and wake_mask[bx, by, bz] == 0                                                 <L 1456>
                var_48 = wp::address(var_wake_mask, var_16, var_20, var_24);
                var_51 = wp::load(var_48);
                var_50 = (var_51 == var_49);
                var_26 = var_26 && var_50;
            }
            if (var_26) {
                // value = int(1)                                                                 <L 1458>
                var_53 = wp::int(var_52);
            }
            var_54 = wp::where(var_26, var_53, var_2);
        }
        var_55 = wp::where(var_3, var_54, var_2);
        // projection_active[cluster_idx] = value                                                 <L 1460>
        wp::array_store(var_projection_active, var_0, var_55);
        // if value != 0:                                                                         <L 1461>
        var_57 = (var_55 != var_56);
        if (var_57) {
            // wp.atomic_add(projection_count, 0, 1)                                              <L 1462>
            var_60 = wp::atomic_add(var_projection_count, var_58, var_59);
        }
    }
}



extern "C" __global__ void finalize_position_update_from_q_15fc23c4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_init,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd,
    wp::array_t<wp::int32> var_particle_flags,
    wp::float32 var_dt,
    wp::float32 var_v_max)
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
        wp::vec_t<3, wp::float32>* var_7;
        wp::vec_t<3, wp::float32> var_8;
        wp::vec_t<3, wp::float32> var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::float32 var_15;
        bool var_16;
        wp::float32 var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        //---------
        // forward
        // def finalize_position_update_from_q(                                                   <L 109>
        // particle_idx = wp.tid()                                                                <L 117>
        var_0 = builtin_tid1d();
        // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                         <L 118>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 119>
            continue;
        }
        // x0 = particle_q_init[particle_idx]                                                     <L 121>
        var_7 = wp::address(var_particle_q_init, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // x_new = particle_q[particle_idx]                                                       <L 122>
        var_10 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // v_new = (x_new - x0) / dt                                                              <L 123>
        var_13 = wp::sub(var_11, var_8);
        var_14 = wp::div(var_13, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 124>
        var_15 = wp::length(var_14);
        // if v_new_mag > v_max:                                                                  <L 125>
        var_16 = (var_15 > var_v_max);
        if (var_16) {
            // v_new *= v_max / v_new_mag                                                         <L 126>
            var_17 = wp::div(var_v_max, var_15);
            var_18 = wp::mul(var_14, var_17);
            // x_new = x0 + v_new * dt                                                            <L 127>
            var_19 = wp::mul(var_18, var_dt);
            var_20 = wp::add(var_8, var_19);
            // particle_q[particle_idx] = x_new                                                   <L 128>
            wp::array_store(var_particle_q, var_0, var_20);
        }
        var_21 = wp::where(var_16, var_20, var_11);
        var_22 = wp::where(var_16, var_18, var_14);
        // particle_qd[particle_idx] = v_new                                                      <L 130>
        wp::array_store(var_particle_qd, var_0, var_22);
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform8_512c1454_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::float32 var_stiffness,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cluster_translations,
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
        bool var_1;
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::float32 var_21 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        const wp::int32 var_27 = 8;
        wp::range_t var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        wp::int32* var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        bool var_40;
        wp::vec_t<3, wp::float32>* var_41;
        wp::vec_t<3, wp::float32> var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::mat_t<3, 3, wp::float32> var_57;
        wp::mat_t<3, 3, wp::float32> var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        wp::float32* var_61;
        const wp::float32 var_62 = 0.0;
        bool var_63;
        wp::float32 var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 0;
        bool var_69;
        wp::float32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::mat_t<3, 3, wp::float32> var_72;
        wp::mat_t<3, 3, wp::float32> var_73;
        wp::quat_t<wp::float32>* var_74;
        wp::quat_t<wp::float32> var_75;
        wp::quat_t<wp::float32> var_76;
        wp::quat_t<wp::float32> var_77;
        wp::float32 var_78;
        const wp::float32 var_79 = 0.0;
        bool var_80;
        const wp::int32 var_81 = 0;
        wp::float32 var_82;
        wp::float32 var_83;
        const wp::int32 var_84 = 1;
        wp::float32 var_85;
        wp::float32 var_86;
        const wp::int32 var_87 = 2;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 3;
        wp::float32 var_91;
        wp::float32 var_92;
        wp::quat_t<wp::float32> var_93;
        wp::quat_t<wp::float32> var_94;
        const wp::int32 var_95 = 0;
        bool var_96;
        const wp::int32 var_97 = 8;
        wp::range_t var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        wp::int32 var_101;
        wp::int32* var_102;
        wp::int32 var_103;
        wp::int32 var_104;
        bool var_105;
        wp::int32* var_106;
        const wp::int32 var_107 = 1;
        wp::int32 var_108;
        wp::int32 var_109;
        const wp::int32 var_110 = 0;
        bool var_111;
        wp::float32* var_112;
        const wp::float32 var_113 = 0.0;
        bool var_114;
        wp::float32 var_115;
        wp::int32 var_116;
        wp::vec_t<3, wp::float32>* var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32> var_119;
        const wp::int32 var_120 = 0;
        bool var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::vec_t<3, wp::float32>* var_124;
        wp::vec_t<3, wp::float32> var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::float32 var_130;
        wp::float32* var_131;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::float32 var_134 = 0.0;
        bool var_135;
        wp::vec_t<3, wp::float32>* var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform8(                                            <L 383>
        // cluster_idx = wp.tid()                                                                 <L 401>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 403>
        var_2 = wp::address(var_cluster_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_1 = var_4;
        if (!var_1) {
            var_7 = (var_stiffness <= var_6);
            var_1 = var_1 || var_7;
        }
        if (var_1) {
            // return                                                                             <L 404>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 406>
        var_8 = wp::address(var_coefficients, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if coeff <= 0.0:                                                                       <L 407>
        var_12 = (var_9 <= var_11);
        if (var_12) {
            // return                                                                             <L 408>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 410>
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // rest_sum = wp.vec3(0.0, 0.0, 0.0)                                                      <L 411>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // covariance = wp.mat33(0.0)                                                             <L 412>
        var_22 = wp::mat_t<3, 3, wp::float32>(var_21);
        // member_count = int(0)                                                                  <L 413>
        var_24 = wp::int(var_23);
        // dynamic_count = int(0)                                                                 <L 414>
        var_26 = wp::int(var_25);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 416>
        var_28 = wp::range(var_27);
        start_for_2:;
            if (iter_cmp(var_28) == 0) goto end_for_2;
            var_29 = wp::iter_next(var_28);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 417>
            var_30 = wp::mul(var_29, var_cluster_count);
            var_31 = wp::add(var_30, var_0);
            var_32 = wp::address(var_indices_by_slot, var_31);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 418>
            var_35 = wp::address(var_particle_flags, var_33);
            var_38 = wp::load(var_35);
            var_37 = wp::bit_and(var_38, var_36);
            var_40 = (var_37 == var_39);
            if (var_40) {
                // continue                                                                       <L 419>
                goto start_for_2;
            }
            // x = particle_q[particle_idx]                                                       <L 420>
            var_41 = wp::address(var_particle_q, var_33);
            var_43 = wp::load(var_41);
            var_42 = wp::copy(var_43);
            // q_rel = rest_local_template[local_idx]                                             <L 421>
            var_44 = wp::address(var_rest_local_template, var_29);
            var_46 = wp::load(var_44);
            var_45 = wp::copy(var_46);
            // if use_rest_local_template == 0:                                                   <L 422>
            var_48 = (var_use_rest_local_template == var_47);
            if (var_48) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 423>
                var_49 = wp::mul(var_29, var_cluster_count);
                var_50 = wp::add(var_49, var_0);
                var_51 = wp::address(var_rest_local_positions_by_slot, var_50);
                var_53 = wp::load(var_51);
                var_52 = wp::copy(var_53);
            }
            var_54 = wp::where(var_48, var_52, var_45);
            // center += x                                                                        <L 424>
            var_55 = wp::add(var_16, var_42);
            // rest_sum += q_rel                                                                  <L 425>
            var_56 = wp::add(var_20, var_54);
            // covariance += wp.outer(q_rel, x)                                                   <L 426>
            var_57 = wp::outer(var_54, var_42);
            var_58 = wp::add(var_22, var_57);
            // member_count += 1                                                                  <L 427>
            var_60 = wp::add(var_24, var_59);
            // if particle_inv_mass[particle_idx] > 0.0:                                          <L 428>
            var_61 = wp::address(var_particle_inv_mass, var_33);
            var_64 = wp::load(var_61);
            var_63 = (var_64 > var_62);
            if (var_63) {
                // dynamic_count += 1                                                             <L 429>
                var_66 = wp::add(var_26, var_65);
            }
            var_67 = wp::where(var_63, var_66, var_26);
            wp::assign(var_16, var_55);
            wp::assign(var_20, var_56);
            wp::assign(var_22, var_58);
            wp::assign(var_24, var_60);
            wp::assign(var_26, var_67);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 431>
        var_69 = (var_24 == var_68);
        if (var_69) {
            // return                                                                             <L 432>
            continue;
        }
        // center /= float(member_count)                                                          <L 434>
        var_70 = wp::float(var_24);
        var_71 = wp::div(var_16, var_70);
        // covariance -= wp.outer(rest_sum, center)                                               <L 435>
        var_72 = wp::outer(var_20, var_71);
        var_73 = wp::sub(var_22, var_72);
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 437>
        var_74 = wp::address(var_cluster_rotations, var_0);
        var_76 = wp::load(var_74);
        var_75 = wp::copy(var_76);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 438>
        var_77 = _extract_rotation_0(var_73, var_75, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 439>
        var_78 = _quat_dot_0(var_77, var_75);
        var_80 = (var_78 < var_79);
        if (var_80) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 440>
            var_82 = wp::extract(var_77, var_81);
            var_83 = wp::neg(var_82);
            var_85 = wp::extract(var_77, var_84);
            var_86 = wp::neg(var_85);
            var_88 = wp::extract(var_77, var_87);
            var_89 = wp::neg(var_88);
            var_91 = wp::extract(var_77, var_90);
            var_92 = wp::neg(var_91);
            var_93 = wp::quat_t<wp::float32>(var_83, var_86, var_89, var_92);
        }
        var_94 = wp::where(var_80, var_93, var_77);
        // cluster_rotations[cluster_idx] = rotation                                              <L 442>
        wp::array_store(var_cluster_rotations, var_0, var_94);
        // cluster_translations[cluster_idx] = center                                             <L 443>
        wp::array_store(var_cluster_translations, var_0, var_71);
        // if dynamic_count == 0:                                                                 <L 445>
        var_96 = (var_26 == var_95);
        if (var_96) {
            // return                                                                             <L 446>
            continue;
        }
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 448>
        var_98 = wp::range(var_97);
        start_for_6:;
            if (iter_cmp(var_98) == 0) goto end_for_6;
            var_99 = wp::iter_next(var_98);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 449>
            var_100 = wp::mul(var_99, var_cluster_count);
            var_101 = wp::add(var_100, var_0);
            var_102 = wp::address(var_indices_by_slot, var_101);
            var_104 = wp::load(var_102);
            var_103 = wp::copy(var_104);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:       <L 450>
            var_106 = wp::address(var_particle_flags, var_103);
            var_109 = wp::load(var_106);
            var_108 = wp::bit_and(var_109, var_107);
            var_111 = (var_108 == var_110);
            var_105 = var_111;
            if (!var_105) {
                var_112 = wp::address(var_particle_inv_mass, var_103);
                var_115 = wp::load(var_112);
                var_114 = (var_115 <= var_113);
                var_105 = var_105 || var_114;
            }
            if (var_105) {
                // continue                                                                       <L 451>
                wp::assign(var_33, var_103);
                goto start_for_6;
            }
            var_116 = wp::where(var_105, var_33, var_103);
            // q_rel = rest_local_template[local_idx]                                             <L 453>
            var_117 = wp::address(var_rest_local_template, var_99);
            var_119 = wp::load(var_117);
            var_118 = wp::copy(var_119);
            // if use_rest_local_template == 0:                                                   <L 454>
            var_121 = (var_use_rest_local_template == var_120);
            if (var_121) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 455>
                var_122 = wp::mul(var_99, var_cluster_count);
                var_123 = wp::add(var_122, var_0);
                var_124 = wp::address(var_rest_local_positions_by_slot, var_123);
                var_126 = wp::load(var_124);
                var_125 = wp::copy(var_126);
            }
            var_127 = wp::where(var_121, var_125, var_118);
            // goal = center + wp.quat_rotate(rotation, q_rel)                                    <L 456>
            var_128 = wp::quat_rotate(var_94, var_127);
            var_129 = wp::add(var_71, var_128);
            // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]       <L 457>
            var_130 = wp::mul(var_9, var_stiffness);
            var_131 = wp::address(var_particle_cluster_inv_weights, var_116);
            var_133 = wp::load(var_131);
            var_132 = wp::mul(var_130, var_133);
            // if particle_scale > 0.0:                                                           <L 458>
            var_135 = (var_132 > var_134);
            if (var_135) {
                // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 459>
                var_136 = wp::address(var_particle_q, var_116);
                var_138 = wp::load(var_136);
                var_137 = wp::sub(var_129, var_138);
                var_139 = wp::mul(var_137, var_132);
                var_140 = wp::atomic_add(var_particle_deltas, var_116, var_139);
            }
            wp::assign(var_33, var_116);
            wp::assign(var_54, var_127);
            goto start_for_6;
        end_for_6:;
    }
}



extern "C" __global__ void solve_volume_constraints_uniform8_835d9c71_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::float32 var_stiffness,
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
        bool var_1;
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 3;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 4;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 5;
        wp::int32 var_44;
        wp::int32 var_45;
        wp::int32* var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        const wp::int32 var_49 = 6;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 7;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        bool var_61;
        wp::int32* var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::int32 var_65;
        const wp::int32 var_66 = 0;
        bool var_67;
        wp::int32* var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        wp::int32* var_74;
        const wp::int32 var_75 = 1;
        wp::int32 var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        wp::int32* var_80;
        const wp::int32 var_81 = 1;
        wp::int32 var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 0;
        bool var_85;
        wp::int32* var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        wp::int32 var_89;
        const wp::int32 var_90 = 0;
        bool var_91;
        wp::int32* var_92;
        const wp::int32 var_93 = 1;
        wp::int32 var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 0;
        bool var_97;
        wp::int32* var_98;
        const wp::int32 var_99 = 1;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 0;
        bool var_103;
        wp::int32* var_104;
        const wp::int32 var_105 = 1;
        wp::int32 var_106;
        wp::int32 var_107;
        const wp::int32 var_108 = 0;
        bool var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::vec_t<3, wp::float32> var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32>* var_113;
        wp::vec_t<3, wp::float32> var_114;
        wp::vec_t<3, wp::float32> var_115;
        wp::vec_t<3, wp::float32>* var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::vec_t<3, wp::float32> var_118;
        wp::vec_t<3, wp::float32>* var_119;
        wp::vec_t<3, wp::float32> var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32>* var_122;
        wp::vec_t<3, wp::float32> var_123;
        wp::vec_t<3, wp::float32> var_124;
        wp::vec_t<3, wp::float32>* var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32>* var_128;
        wp::vec_t<3, wp::float32> var_129;
        wp::vec_t<3, wp::float32> var_130;
        wp::vec_t<3, wp::float32>* var_131;
        wp::vec_t<3, wp::float32> var_132;
        wp::vec_t<3, wp::float32> var_133;
        wp::vec_t<3, wp::float32> var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::vec_t<3, wp::float32> var_140;
        const wp::float32 var_141 = 0.25;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        wp::vec_t<3, wp::float32> var_144;
        wp::vec_t<3, wp::float32> var_145;
        wp::vec_t<3, wp::float32> var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        const wp::float32 var_150 = 0.25;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        const wp::float32 var_159 = 0.25;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::float32 var_162;
        const wp::int32 var_163 = 0;
        wp::vec_t<3, wp::float32>* var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        const wp::int32 var_167 = 1;
        wp::vec_t<3, wp::float32>* var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::vec_t<3, wp::float32> var_170;
        const wp::int32 var_171 = 2;
        wp::vec_t<3, wp::float32>* var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::vec_t<3, wp::float32> var_174;
        const wp::int32 var_175 = 3;
        wp::vec_t<3, wp::float32>* var_176;
        wp::vec_t<3, wp::float32> var_177;
        wp::vec_t<3, wp::float32> var_178;
        const wp::int32 var_179 = 4;
        wp::vec_t<3, wp::float32>* var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::vec_t<3, wp::float32> var_182;
        const wp::int32 var_183 = 5;
        wp::vec_t<3, wp::float32>* var_184;
        wp::vec_t<3, wp::float32> var_185;
        wp::vec_t<3, wp::float32> var_186;
        const wp::int32 var_187 = 6;
        wp::vec_t<3, wp::float32>* var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        const wp::int32 var_191 = 7;
        wp::vec_t<3, wp::float32>* var_192;
        wp::vec_t<3, wp::float32> var_193;
        wp::vec_t<3, wp::float32> var_194;
        const wp::int32 var_195 = 0;
        bool var_196;
        const wp::int32 var_197 = 0;
        wp::int32 var_198;
        wp::int32 var_199;
        wp::vec_t<3, wp::float32>* var_200;
        wp::vec_t<3, wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        const wp::int32 var_203 = 1;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::vec_t<3, wp::float32>* var_206;
        wp::vec_t<3, wp::float32> var_207;
        wp::vec_t<3, wp::float32> var_208;
        const wp::int32 var_209 = 2;
        wp::int32 var_210;
        wp::int32 var_211;
        wp::vec_t<3, wp::float32>* var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::vec_t<3, wp::float32> var_214;
        const wp::int32 var_215 = 3;
        wp::int32 var_216;
        wp::int32 var_217;
        wp::vec_t<3, wp::float32>* var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::vec_t<3, wp::float32> var_220;
        const wp::int32 var_221 = 4;
        wp::int32 var_222;
        wp::int32 var_223;
        wp::vec_t<3, wp::float32>* var_224;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        const wp::int32 var_227 = 5;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::vec_t<3, wp::float32>* var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        const wp::int32 var_233 = 6;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::vec_t<3, wp::float32>* var_236;
        wp::vec_t<3, wp::float32> var_237;
        wp::vec_t<3, wp::float32> var_238;
        const wp::int32 var_239 = 7;
        wp::int32 var_240;
        wp::int32 var_241;
        wp::vec_t<3, wp::float32>* var_242;
        wp::vec_t<3, wp::float32> var_243;
        wp::vec_t<3, wp::float32> var_244;
        wp::vec_t<3, wp::float32> var_245;
        wp::vec_t<3, wp::float32> var_246;
        wp::vec_t<3, wp::float32> var_247;
        wp::vec_t<3, wp::float32> var_248;
        wp::vec_t<3, wp::float32> var_249;
        wp::vec_t<3, wp::float32> var_250;
        wp::vec_t<3, wp::float32> var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<3, wp::float32> var_253;
        wp::vec_t<3, wp::float32> var_254;
        wp::vec_t<3, wp::float32> var_255;
        wp::vec_t<3, wp::float32> var_256;
        wp::vec_t<3, wp::float32> var_257;
        wp::vec_t<3, wp::float32> var_258;
        wp::vec_t<3, wp::float32> var_259;
        const wp::float32 var_260 = 0.25;
        wp::vec_t<3, wp::float32> var_261;
        wp::vec_t<3, wp::float32> var_262;
        wp::vec_t<3, wp::float32> var_263;
        wp::vec_t<3, wp::float32> var_264;
        wp::vec_t<3, wp::float32> var_265;
        wp::vec_t<3, wp::float32> var_266;
        wp::vec_t<3, wp::float32> var_267;
        wp::vec_t<3, wp::float32> var_268;
        const wp::float32 var_269 = 0.25;
        wp::vec_t<3, wp::float32> var_270;
        wp::vec_t<3, wp::float32> var_271;
        wp::vec_t<3, wp::float32> var_272;
        wp::vec_t<3, wp::float32> var_273;
        wp::vec_t<3, wp::float32> var_274;
        wp::vec_t<3, wp::float32> var_275;
        wp::vec_t<3, wp::float32> var_276;
        wp::vec_t<3, wp::float32> var_277;
        const wp::float32 var_278 = 0.25;
        wp::vec_t<3, wp::float32> var_279;
        wp::vec_t<3, wp::float32> var_280;
        wp::float32 var_281;
        wp::float32 var_282;
        const wp::float32 var_283 = 1e-12;
        bool var_284;
        wp::vec_t<3, wp::float32> var_285;
        wp::vec_t<3, wp::float32> var_286;
        wp::vec_t<3, wp::float32> var_287;
        const wp::float32 var_288 = 0.0;
        wp::float32 var_289;
        const wp::int32 var_290 = 8;
        wp::range_t var_291;
        wp::int32 var_292;
        wp::int32 var_293;
        wp::int32 var_294;
        wp::int32* var_295;
        wp::int32 var_296;
        wp::int32 var_297;
        wp::float32* var_298;
        wp::float32* var_299;
        wp::float32 var_300;
        wp::float32 var_301;
        wp::float32 var_302;
        const wp::float32 var_303 = 0.0;
        bool var_304;
        wp::float32 var_305;
        wp::vec_t<3, wp::float32> var_306;
        wp::float32 var_307;
        wp::vec_t<3, wp::float32> var_308;
        wp::vec_t<3, wp::float32> var_309;
        wp::float32 var_310;
        wp::vec_t<3, wp::float32> var_311;
        wp::vec_t<3, wp::float32> var_312;
        const wp::float32 var_313 = 0.25;
        wp::vec_t<3, wp::float32> var_314;
        wp::float32 var_315;
        wp::float32 var_316;
        wp::float32 var_317;
        const wp::float32 var_318 = 1e-20;
        bool var_319;
        wp::float32 var_320;
        wp::float32 var_321;
        wp::float32 var_322;
        wp::float32 var_323;
        wp::float32 var_324;
        const wp::int32 var_325 = 8;
        wp::range_t var_326;
        wp::int32 var_327;
        wp::int32 var_328;
        wp::int32 var_329;
        wp::int32* var_330;
        wp::int32 var_331;
        wp::int32 var_332;
        wp::float32* var_333;
        wp::float32* var_334;
        wp::float32 var_335;
        wp::float32 var_336;
        wp::float32 var_337;
        const wp::float32 var_338 = 0.0;
        bool var_339;
        wp::int32 var_340;
        wp::float32 var_341;
        wp::float32 var_342;
        wp::vec_t<3, wp::float32> var_343;
        wp::float32 var_344;
        wp::vec_t<3, wp::float32> var_345;
        wp::vec_t<3, wp::float32> var_346;
        wp::float32 var_347;
        wp::vec_t<3, wp::float32> var_348;
        wp::vec_t<3, wp::float32> var_349;
        const wp::float32 var_350 = 0.25;
        wp::vec_t<3, wp::float32> var_351;
        wp::float32 var_352;
        wp::vec_t<3, wp::float32> var_353;
        wp::vec_t<3, wp::float32> var_354;
        //---------
        // forward
        // def solve_volume_constraints_uniform8(                                                 <L 134>
        // cluster_idx = wp.tid()                                                                 <L 149>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 151>
        var_2 = wp::address(var_cluster_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_1 = var_4;
        if (!var_1) {
            var_7 = (var_stiffness <= var_6);
            var_1 = var_1 || var_7;
        }
        if (var_1) {
            // return                                                                             <L 152>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 154>
        var_8 = wp::address(var_coefficients, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if coeff <= 0.0:                                                                       <L 155>
        var_12 = (var_9 <= var_11);
        if (var_12) {
            // return                                                                             <L 156>
            continue;
        }
        // i0 = indices_by_slot[0 * cluster_count + cluster_idx]                                  <L 158>
        var_14 = wp::mul(var_13, var_cluster_count);
        var_15 = wp::add(var_14, var_0);
        var_16 = wp::address(var_indices_by_slot, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // i1 = indices_by_slot[1 * cluster_count + cluster_idx]                                  <L 159>
        var_20 = wp::mul(var_19, var_cluster_count);
        var_21 = wp::add(var_20, var_0);
        var_22 = wp::address(var_indices_by_slot, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // i2 = indices_by_slot[2 * cluster_count + cluster_idx]                                  <L 160>
        var_26 = wp::mul(var_25, var_cluster_count);
        var_27 = wp::add(var_26, var_0);
        var_28 = wp::address(var_indices_by_slot, var_27);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // i3 = indices_by_slot[3 * cluster_count + cluster_idx]                                  <L 161>
        var_32 = wp::mul(var_31, var_cluster_count);
        var_33 = wp::add(var_32, var_0);
        var_34 = wp::address(var_indices_by_slot, var_33);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // i4 = indices_by_slot[4 * cluster_count + cluster_idx]                                  <L 162>
        var_38 = wp::mul(var_37, var_cluster_count);
        var_39 = wp::add(var_38, var_0);
        var_40 = wp::address(var_indices_by_slot, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // i5 = indices_by_slot[5 * cluster_count + cluster_idx]                                  <L 163>
        var_44 = wp::mul(var_43, var_cluster_count);
        var_45 = wp::add(var_44, var_0);
        var_46 = wp::address(var_indices_by_slot, var_45);
        var_48 = wp::load(var_46);
        var_47 = wp::copy(var_48);
        // i6 = indices_by_slot[6 * cluster_count + cluster_idx]                                  <L 164>
        var_50 = wp::mul(var_49, var_cluster_count);
        var_51 = wp::add(var_50, var_0);
        var_52 = wp::address(var_indices_by_slot, var_51);
        var_54 = wp::load(var_52);
        var_53 = wp::copy(var_54);
        // i7 = indices_by_slot[7 * cluster_count + cluster_idx]                                  <L 165>
        var_56 = wp::mul(var_55, var_cluster_count);
        var_57 = wp::add(var_56, var_0);
        var_58 = wp::address(var_indices_by_slot, var_57);
        var_60 = wp::load(var_58);
        var_59 = wp::copy(var_60);
        // if (                                                                                   <L 167>
        // (particle_flags[i0] & ParticleFlags.ACTIVE) == 0                                       <L 168>
        var_62 = wp::address(var_particle_flags, var_17);
        var_65 = wp::load(var_62);
        var_64 = wp::bit_and(var_65, var_63);
        var_67 = (var_64 == var_66);
        var_61 = var_67;
        if (!var_61) {
            // or (particle_flags[i1] & ParticleFlags.ACTIVE) == 0                                <L 169>
            var_68 = wp::address(var_particle_flags, var_23);
            var_71 = wp::load(var_68);
            var_70 = wp::bit_and(var_71, var_69);
            var_73 = (var_70 == var_72);
            var_61 = var_61 || var_73;
        }
        if (!var_61) {
            // or (particle_flags[i2] & ParticleFlags.ACTIVE) == 0                                <L 170>
            var_74 = wp::address(var_particle_flags, var_29);
            var_77 = wp::load(var_74);
            var_76 = wp::bit_and(var_77, var_75);
            var_79 = (var_76 == var_78);
            var_61 = var_61 || var_79;
        }
        if (!var_61) {
            // or (particle_flags[i3] & ParticleFlags.ACTIVE) == 0                                <L 171>
            var_80 = wp::address(var_particle_flags, var_35);
            var_83 = wp::load(var_80);
            var_82 = wp::bit_and(var_83, var_81);
            var_85 = (var_82 == var_84);
            var_61 = var_61 || var_85;
        }
        if (!var_61) {
            // or (particle_flags[i4] & ParticleFlags.ACTIVE) == 0                                <L 172>
            var_86 = wp::address(var_particle_flags, var_41);
            var_89 = wp::load(var_86);
            var_88 = wp::bit_and(var_89, var_87);
            var_91 = (var_88 == var_90);
            var_61 = var_61 || var_91;
        }
        if (!var_61) {
            // or (particle_flags[i5] & ParticleFlags.ACTIVE) == 0                                <L 173>
            var_92 = wp::address(var_particle_flags, var_47);
            var_95 = wp::load(var_92);
            var_94 = wp::bit_and(var_95, var_93);
            var_97 = (var_94 == var_96);
            var_61 = var_61 || var_97;
        }
        if (!var_61) {
            // or (particle_flags[i6] & ParticleFlags.ACTIVE) == 0                                <L 174>
            var_98 = wp::address(var_particle_flags, var_53);
            var_101 = wp::load(var_98);
            var_100 = wp::bit_and(var_101, var_99);
            var_103 = (var_100 == var_102);
            var_61 = var_61 || var_103;
        }
        if (!var_61) {
            // or (particle_flags[i7] & ParticleFlags.ACTIVE) == 0                                <L 175>
            var_104 = wp::address(var_particle_flags, var_59);
            var_107 = wp::load(var_104);
            var_106 = wp::bit_and(var_107, var_105);
            var_109 = (var_106 == var_108);
            var_61 = var_61 || var_109;
        }
        if (var_61) {
            // return                                                                             <L 177>
            continue;
        }
        // p0 = particle_q[i0]                                                                    <L 179>
        var_110 = wp::address(var_particle_q, var_17);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // p1 = particle_q[i1]                                                                    <L 180>
        var_113 = wp::address(var_particle_q, var_23);
        var_115 = wp::load(var_113);
        var_114 = wp::copy(var_115);
        // p2 = particle_q[i2]                                                                    <L 181>
        var_116 = wp::address(var_particle_q, var_29);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // p3 = particle_q[i3]                                                                    <L 182>
        var_119 = wp::address(var_particle_q, var_35);
        var_121 = wp::load(var_119);
        var_120 = wp::copy(var_121);
        // p4 = particle_q[i4]                                                                    <L 183>
        var_122 = wp::address(var_particle_q, var_41);
        var_124 = wp::load(var_122);
        var_123 = wp::copy(var_124);
        // p5 = particle_q[i5]                                                                    <L 184>
        var_125 = wp::address(var_particle_q, var_47);
        var_127 = wp::load(var_125);
        var_126 = wp::copy(var_127);
        // p6 = particle_q[i6]                                                                    <L 185>
        var_128 = wp::address(var_particle_q, var_53);
        var_130 = wp::load(var_128);
        var_129 = wp::copy(var_130);
        // p7 = particle_q[i7]                                                                    <L 186>
        var_131 = wp::address(var_particle_q, var_59);
        var_133 = wp::load(var_131);
        var_132 = wp::copy(var_133);
        // ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                            <L 188>
        var_134 = wp::sub(var_114, var_111);
        var_135 = wp::sub(var_117, var_120);
        var_136 = wp::add(var_134, var_135);
        var_137 = wp::sub(var_126, var_123);
        var_138 = wp::add(var_136, var_137);
        var_139 = wp::sub(var_129, var_132);
        var_140 = wp::add(var_138, var_139);
        var_142 = wp::mul(var_140, var_141);
        // ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                            <L 189>
        var_143 = wp::sub(var_120, var_111);
        var_144 = wp::sub(var_117, var_114);
        var_145 = wp::add(var_143, var_144);
        var_146 = wp::sub(var_132, var_123);
        var_147 = wp::add(var_145, var_146);
        var_148 = wp::sub(var_129, var_126);
        var_149 = wp::add(var_147, var_148);
        var_151 = wp::mul(var_149, var_150);
        // az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                            <L 190>
        var_152 = wp::sub(var_123, var_111);
        var_153 = wp::sub(var_126, var_114);
        var_154 = wp::add(var_152, var_153);
        var_155 = wp::sub(var_129, var_117);
        var_156 = wp::add(var_154, var_155);
        var_157 = wp::sub(var_132, var_120);
        var_158 = wp::add(var_156, var_157);
        var_160 = wp::mul(var_158, var_159);
        // volume = wp.dot(ax, wp.cross(ay, az))                                                  <L 191>
        var_161 = wp::cross(var_151, var_160);
        var_162 = wp::dot(var_142, var_161);
        // q0 = rest_local_template[0]                                                            <L 193>
        var_164 = wp::address(var_rest_local_template, var_163);
        var_166 = wp::load(var_164);
        var_165 = wp::copy(var_166);
        // q1 = rest_local_template[1]                                                            <L 194>
        var_168 = wp::address(var_rest_local_template, var_167);
        var_170 = wp::load(var_168);
        var_169 = wp::copy(var_170);
        // q2 = rest_local_template[2]                                                            <L 195>
        var_172 = wp::address(var_rest_local_template, var_171);
        var_174 = wp::load(var_172);
        var_173 = wp::copy(var_174);
        // q3 = rest_local_template[3]                                                            <L 196>
        var_176 = wp::address(var_rest_local_template, var_175);
        var_178 = wp::load(var_176);
        var_177 = wp::copy(var_178);
        // q4 = rest_local_template[4]                                                            <L 197>
        var_180 = wp::address(var_rest_local_template, var_179);
        var_182 = wp::load(var_180);
        var_181 = wp::copy(var_182);
        // q5 = rest_local_template[5]                                                            <L 198>
        var_184 = wp::address(var_rest_local_template, var_183);
        var_186 = wp::load(var_184);
        var_185 = wp::copy(var_186);
        // q6 = rest_local_template[6]                                                            <L 199>
        var_188 = wp::address(var_rest_local_template, var_187);
        var_190 = wp::load(var_188);
        var_189 = wp::copy(var_190);
        // q7 = rest_local_template[7]                                                            <L 200>
        var_192 = wp::address(var_rest_local_template, var_191);
        var_194 = wp::load(var_192);
        var_193 = wp::copy(var_194);
        // if use_rest_local_template == 0:                                                       <L 201>
        var_196 = (var_use_rest_local_template == var_195);
        if (var_196) {
            // q0 = rest_local_positions_by_slot[0 * cluster_count + cluster_idx]                 <L 202>
            var_198 = wp::mul(var_197, var_cluster_count);
            var_199 = wp::add(var_198, var_0);
            var_200 = wp::address(var_rest_local_positions_by_slot, var_199);
            var_202 = wp::load(var_200);
            var_201 = wp::copy(var_202);
            // q1 = rest_local_positions_by_slot[1 * cluster_count + cluster_idx]                 <L 203>
            var_204 = wp::mul(var_203, var_cluster_count);
            var_205 = wp::add(var_204, var_0);
            var_206 = wp::address(var_rest_local_positions_by_slot, var_205);
            var_208 = wp::load(var_206);
            var_207 = wp::copy(var_208);
            // q2 = rest_local_positions_by_slot[2 * cluster_count + cluster_idx]                 <L 204>
            var_210 = wp::mul(var_209, var_cluster_count);
            var_211 = wp::add(var_210, var_0);
            var_212 = wp::address(var_rest_local_positions_by_slot, var_211);
            var_214 = wp::load(var_212);
            var_213 = wp::copy(var_214);
            // q3 = rest_local_positions_by_slot[3 * cluster_count + cluster_idx]                 <L 205>
            var_216 = wp::mul(var_215, var_cluster_count);
            var_217 = wp::add(var_216, var_0);
            var_218 = wp::address(var_rest_local_positions_by_slot, var_217);
            var_220 = wp::load(var_218);
            var_219 = wp::copy(var_220);
            // q4 = rest_local_positions_by_slot[4 * cluster_count + cluster_idx]                 <L 206>
            var_222 = wp::mul(var_221, var_cluster_count);
            var_223 = wp::add(var_222, var_0);
            var_224 = wp::address(var_rest_local_positions_by_slot, var_223);
            var_226 = wp::load(var_224);
            var_225 = wp::copy(var_226);
            // q5 = rest_local_positions_by_slot[5 * cluster_count + cluster_idx]                 <L 207>
            var_228 = wp::mul(var_227, var_cluster_count);
            var_229 = wp::add(var_228, var_0);
            var_230 = wp::address(var_rest_local_positions_by_slot, var_229);
            var_232 = wp::load(var_230);
            var_231 = wp::copy(var_232);
            // q6 = rest_local_positions_by_slot[6 * cluster_count + cluster_idx]                 <L 208>
            var_234 = wp::mul(var_233, var_cluster_count);
            var_235 = wp::add(var_234, var_0);
            var_236 = wp::address(var_rest_local_positions_by_slot, var_235);
            var_238 = wp::load(var_236);
            var_237 = wp::copy(var_238);
            // q7 = rest_local_positions_by_slot[7 * cluster_count + cluster_idx]                 <L 209>
            var_240 = wp::mul(var_239, var_cluster_count);
            var_241 = wp::add(var_240, var_0);
            var_242 = wp::address(var_rest_local_positions_by_slot, var_241);
            var_244 = wp::load(var_242);
            var_243 = wp::copy(var_244);
        }
        var_245 = wp::where(var_196, var_201, var_165);
        var_246 = wp::where(var_196, var_207, var_169);
        var_247 = wp::where(var_196, var_213, var_173);
        var_248 = wp::where(var_196, var_219, var_177);
        var_249 = wp::where(var_196, var_225, var_181);
        var_250 = wp::where(var_196, var_231, var_185);
        var_251 = wp::where(var_196, var_237, var_189);
        var_252 = wp::where(var_196, var_243, var_193);
        // rest_ax = ((q1 - q0) + (q2 - q3) + (q5 - q4) + (q6 - q7)) * 0.25                       <L 211>
        var_253 = wp::sub(var_246, var_245);
        var_254 = wp::sub(var_247, var_248);
        var_255 = wp::add(var_253, var_254);
        var_256 = wp::sub(var_250, var_249);
        var_257 = wp::add(var_255, var_256);
        var_258 = wp::sub(var_251, var_252);
        var_259 = wp::add(var_257, var_258);
        var_261 = wp::mul(var_259, var_260);
        // rest_ay = ((q3 - q0) + (q2 - q1) + (q7 - q4) + (q6 - q5)) * 0.25                       <L 212>
        var_262 = wp::sub(var_248, var_245);
        var_263 = wp::sub(var_247, var_246);
        var_264 = wp::add(var_262, var_263);
        var_265 = wp::sub(var_252, var_249);
        var_266 = wp::add(var_264, var_265);
        var_267 = wp::sub(var_251, var_250);
        var_268 = wp::add(var_266, var_267);
        var_270 = wp::mul(var_268, var_269);
        // rest_az = ((q4 - q0) + (q5 - q1) + (q6 - q2) + (q7 - q3)) * 0.25                       <L 213>
        var_271 = wp::sub(var_249, var_245);
        var_272 = wp::sub(var_250, var_246);
        var_273 = wp::add(var_271, var_272);
        var_274 = wp::sub(var_251, var_247);
        var_275 = wp::add(var_273, var_274);
        var_276 = wp::sub(var_252, var_248);
        var_277 = wp::add(var_275, var_276);
        var_279 = wp::mul(var_277, var_278);
        // rest_volume = wp.dot(rest_ax, wp.cross(rest_ay, rest_az))                              <L 214>
        var_280 = wp::cross(var_270, var_279);
        var_281 = wp::dot(var_261, var_280);
        // if wp.abs(rest_volume) <= 1.0e-12:                                                     <L 216>
        var_282 = wp::abs(var_281);
        var_284 = (var_282 <= var_283);
        if (var_284) {
            // return                                                                             <L 217>
            continue;
        }
        // gx = wp.cross(ay, az)                                                                  <L 219>
        var_285 = wp::cross(var_151, var_160);
        // gy = wp.cross(az, ax)                                                                  <L 220>
        var_286 = wp::cross(var_160, var_142);
        // gz = wp.cross(ax, ay)                                                                  <L 221>
        var_287 = wp::cross(var_142, var_151);
        // denom = float(0.0)                                                                     <L 223>
        var_289 = wp::float(var_288);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 224>
        var_291 = wp::range(var_290);
        start_for_4:;
            if (iter_cmp(var_291) == 0) goto end_for_4;
            var_292 = wp::iter_next(var_291);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 225>
            var_293 = wp::mul(var_292, var_cluster_count);
            var_294 = wp::add(var_293, var_0);
            var_295 = wp::address(var_indices_by_slot, var_294);
            var_297 = wp::load(var_295);
            var_296 = wp::copy(var_297);
            // inv_weight = particle_inv_mass[particle_idx] * particle_cluster_inv_weights[particle_idx]       <L 226>
            var_298 = wp::address(var_particle_inv_mass, var_296);
            var_299 = wp::address(var_particle_cluster_inv_weights, var_296);
            var_301 = wp::load(var_298);
            var_302 = wp::load(var_299);
            var_300 = wp::mul(var_301, var_302);
            // if inv_weight <= 0.0:                                                              <L 227>
            var_304 = (var_300 <= var_303);
            if (var_304) {
                // continue                                                                       <L 228>
                goto start_for_4;
            }
            // grad = (                                                                           <L 229>
            // gx * _uniform8_slot_sign_x(local_idx)                                              <L 230>
            var_305 = _uniform8_slot_sign_x_0(var_292);
            var_306 = wp::mul(var_285, var_305);
            // + gy * _uniform8_slot_sign_y(local_idx)                                            <L 231>
            var_307 = _uniform8_slot_sign_y_0(var_292);
            var_308 = wp::mul(var_286, var_307);
            var_309 = wp::add(var_306, var_308);
            // + gz * _uniform8_slot_sign_z(local_idx)                                            <L 232>
            var_310 = _uniform8_slot_sign_z_0(var_292);
            var_311 = wp::mul(var_287, var_310);
            var_312 = wp::add(var_309, var_311);
            // ) * 0.25                                                                           <L 233>
            var_314 = wp::mul(var_312, var_313);
            // denom += inv_weight * wp.dot(grad, grad)                                           <L 234>
            var_315 = wp::dot(var_314, var_314);
            var_316 = wp::mul(var_300, var_315);
            var_317 = wp::add(var_289, var_316);
            wp::assign(var_289, var_317);
            goto start_for_4;
        end_for_4:;
        // if denom <= 1.0e-20:                                                                   <L 236>
        var_319 = (var_289 <= var_318);
        if (var_319) {
            // return                                                                             <L 237>
            continue;
        }
        // lagrange = -(volume - rest_volume) * coeff * stiffness / denom                         <L 239>
        var_320 = wp::sub(var_162, var_281);
        var_321 = wp::neg(var_320);
        var_322 = wp::mul(var_321, var_9);
        var_323 = wp::mul(var_322, var_stiffness);
        var_324 = wp::div(var_323, var_289);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 240>
        var_326 = wp::range(var_325);
        start_for_7:;
            if (iter_cmp(var_326) == 0) goto end_for_7;
            var_327 = wp::iter_next(var_326);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 241>
            var_328 = wp::mul(var_327, var_cluster_count);
            var_329 = wp::add(var_328, var_0);
            var_330 = wp::address(var_indices_by_slot, var_329);
            var_332 = wp::load(var_330);
            var_331 = wp::copy(var_332);
            // inv_weight = particle_inv_mass[particle_idx] * particle_cluster_inv_weights[particle_idx]       <L 242>
            var_333 = wp::address(var_particle_inv_mass, var_331);
            var_334 = wp::address(var_particle_cluster_inv_weights, var_331);
            var_336 = wp::load(var_333);
            var_337 = wp::load(var_334);
            var_335 = wp::mul(var_336, var_337);
            // if inv_weight <= 0.0:                                                              <L 243>
            var_339 = (var_335 <= var_338);
            if (var_339) {
                // continue                                                                       <L 244>
                wp::assign(var_296, var_331);
                wp::assign(var_300, var_335);
                goto start_for_7;
            }
            var_340 = wp::where(var_339, var_296, var_331);
            var_341 = wp::where(var_339, var_300, var_335);
            // grad = (                                                                           <L 245>
            // gx * _uniform8_slot_sign_x(local_idx)                                              <L 246>
            var_342 = _uniform8_slot_sign_x_0(var_327);
            var_343 = wp::mul(var_285, var_342);
            // + gy * _uniform8_slot_sign_y(local_idx)                                            <L 247>
            var_344 = _uniform8_slot_sign_y_0(var_327);
            var_345 = wp::mul(var_286, var_344);
            var_346 = wp::add(var_343, var_345);
            // + gz * _uniform8_slot_sign_z(local_idx)                                            <L 248>
            var_347 = _uniform8_slot_sign_z_0(var_327);
            var_348 = wp::mul(var_287, var_347);
            var_349 = wp::add(var_346, var_348);
            // ) * 0.25                                                                           <L 249>
            var_351 = wp::mul(var_349, var_350);
            // wp.atomic_add(particle_deltas, particle_idx, grad * (lagrange * inv_weight))       <L 250>
            var_352 = wp::mul(var_324, var_341);
            var_353 = wp::mul(var_351, var_352);
            var_354 = wp::atomic_add(var_particle_deltas, var_340, var_353);
            wp::assign(var_296, var_340);
            wp::assign(var_300, var_341);
            wp::assign(var_314, var_351);
            goto start_for_7;
        end_for_7:;
    }
}



extern "C" __global__ void compute_shape_matching_cluster_poses_uniform8_a8cd20e1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::float32 var_stiffness,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cluster_translations)
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
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        const wp::float32 var_17 = 0.0;
        const wp::float32 var_18 = 0.0;
        const wp::float32 var_19 = 0.0;
        wp::vec_t<3, wp::float32> var_20;
        const wp::float32 var_21 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_22;
        const wp::int32 var_23 = 0;
        wp::int32 var_24;
        const wp::int32 var_25 = 8;
        wp::range_t var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        wp::int32* var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::mat_t<3, 3, wp::float32> var_55;
        wp::mat_t<3, 3, wp::float32> var_56;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        bool var_60;
        wp::float32 var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::mat_t<3, 3, wp::float32> var_63;
        wp::mat_t<3, 3, wp::float32> var_64;
        wp::quat_t<wp::float32>* var_65;
        wp::quat_t<wp::float32> var_66;
        wp::quat_t<wp::float32> var_67;
        wp::quat_t<wp::float32> var_68;
        wp::float32 var_69;
        const wp::float32 var_70 = 0.0;
        bool var_71;
        const wp::int32 var_72 = 0;
        wp::float32 var_73;
        wp::float32 var_74;
        const wp::int32 var_75 = 1;
        wp::float32 var_76;
        wp::float32 var_77;
        const wp::int32 var_78 = 2;
        wp::float32 var_79;
        wp::float32 var_80;
        const wp::int32 var_81 = 3;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::quat_t<wp::float32> var_84;
        wp::quat_t<wp::float32> var_85;
        //---------
        // forward
        // def compute_shape_matching_cluster_poses_uniform8(                                     <L 254>
        // cluster_idx = wp.tid()                                                                 <L 269>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 271>
        var_2 = wp::address(var_cluster_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_1 = var_4;
        if (!var_1) {
            var_7 = (var_stiffness <= var_6);
            var_1 = var_1 || var_7;
        }
        if (var_1) {
            // return                                                                             <L 272>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 274>
        var_8 = wp::address(var_coefficients, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if coeff <= 0.0:                                                                       <L 275>
        var_12 = (var_9 <= var_11);
        if (var_12) {
            // return                                                                             <L 276>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 278>
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // rest_sum = wp.vec3(0.0, 0.0, 0.0)                                                      <L 279>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        // covariance = wp.mat33(0.0)                                                             <L 280>
        var_22 = wp::mat_t<3, 3, wp::float32>(var_21);
        // member_count = int(0)                                                                  <L 281>
        var_24 = wp::int(var_23);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE)):                               <L 283>
        var_26 = wp::range(var_25);
        start_for_2:;
            if (iter_cmp(var_26) == 0) goto end_for_2;
            var_27 = wp::iter_next(var_26);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 284>
            var_28 = wp::mul(var_27, var_cluster_count);
            var_29 = wp::add(var_28, var_0);
            var_30 = wp::address(var_indices_by_slot, var_29);
            var_32 = wp::load(var_30);
            var_31 = wp::copy(var_32);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 285>
            var_33 = wp::address(var_particle_flags, var_31);
            var_36 = wp::load(var_33);
            var_35 = wp::bit_and(var_36, var_34);
            var_38 = (var_35 == var_37);
            if (var_38) {
                // continue                                                                       <L 286>
                goto start_for_2;
            }
            // x = particle_q[particle_idx]                                                       <L 287>
            var_39 = wp::address(var_particle_q, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // q_rel = rest_local_template[local_idx]                                             <L 288>
            var_42 = wp::address(var_rest_local_template, var_27);
            var_44 = wp::load(var_42);
            var_43 = wp::copy(var_44);
            // if use_rest_local_template == 0:                                                   <L 289>
            var_46 = (var_use_rest_local_template == var_45);
            if (var_46) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 290>
                var_47 = wp::mul(var_27, var_cluster_count);
                var_48 = wp::add(var_47, var_0);
                var_49 = wp::address(var_rest_local_positions_by_slot, var_48);
                var_51 = wp::load(var_49);
                var_50 = wp::copy(var_51);
            }
            var_52 = wp::where(var_46, var_50, var_43);
            // center += x                                                                        <L 291>
            var_53 = wp::add(var_16, var_40);
            // rest_sum += q_rel                                                                  <L 292>
            var_54 = wp::add(var_20, var_52);
            // covariance += wp.outer(q_rel, x)                                                   <L 293>
            var_55 = wp::outer(var_52, var_40);
            var_56 = wp::add(var_22, var_55);
            // member_count += 1                                                                  <L 294>
            var_58 = wp::add(var_24, var_57);
            wp::assign(var_16, var_53);
            wp::assign(var_20, var_54);
            wp::assign(var_22, var_56);
            wp::assign(var_24, var_58);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 296>
        var_60 = (var_24 == var_59);
        if (var_60) {
            // return                                                                             <L 297>
            continue;
        }
        // center /= float(member_count)                                                          <L 299>
        var_61 = wp::float(var_24);
        var_62 = wp::div(var_16, var_61);
        // covariance -= wp.outer(rest_sum, center)                                               <L 301>
        var_63 = wp::outer(var_20, var_62);
        var_64 = wp::sub(var_22, var_63);
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 303>
        var_65 = wp::address(var_cluster_rotations, var_0);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 304>
        var_68 = _extract_rotation_0(var_64, var_66, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 305>
        var_69 = _quat_dot_0(var_68, var_66);
        var_71 = (var_69 < var_70);
        if (var_71) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 306>
            var_73 = wp::extract(var_68, var_72);
            var_74 = wp::neg(var_73);
            var_76 = wp::extract(var_68, var_75);
            var_77 = wp::neg(var_76);
            var_79 = wp::extract(var_68, var_78);
            var_80 = wp::neg(var_79);
            var_82 = wp::extract(var_68, var_81);
            var_83 = wp::neg(var_82);
            var_84 = wp::quat_t<wp::float32>(var_74, var_77, var_80, var_83);
        }
        var_85 = wp::where(var_71, var_84, var_68);
        // cluster_rotations[cluster_idx] = rotation                                              <L 308>
        wp::array_store(var_cluster_rotations, var_0, var_85);
        // cluster_translations[cluster_idx] = center                                             <L 309>
        wp::array_store(var_cluster_translations, var_0, var_62);
    }
}



extern "C" __global__ void prolongate_shape_matching_corrections_uniform8_529dc364_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_init,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_coarse_q_snapshot,
    wp::array_t<wp::int32> var_child_cluster,
    wp::array_t<wp::int32> var_block_keys,
    wp::array_t<wp::int32> var_node_grid_xyz,
    wp::array_t<wp::int32> var_grid_to_node,
    wp::int32 var_block_size,
    wp::int32 var_max_grid_x,
    wp::int32 var_max_grid_y,
    wp::int32 var_max_grid_z,
    wp::array_t<wp::int32> var_cluster_active,
    wp::int32 var_absolute_projection,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd_out)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        bool var_10;
        wp::int32* var_11;
        const wp::int32 var_12 = 1;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::float32* var_17;
        const wp::float32 var_18 = 0.0;
        bool var_19;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        wp::int32* var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 1;
        wp::int32* var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 2;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 1;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 2;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        wp::float32 var_61;
        wp::int32 var_62;
        wp::float32 var_63;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::int32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        const wp::float32 var_70 = 0.0;
        const wp::float32 var_71 = 0.0;
        const wp::float32 var_72 = 0.0;
        wp::vec_t<3, wp::float32> var_73;
        const wp::int32 var_74 = 1;
        wp::int32 var_75;
        bool var_76;
        const wp::int32 var_77 = 0;
        bool var_78;
        const wp::int32 var_79 = 0;
        bool var_80;
        const wp::int32 var_81 = 0;
        bool var_82;
        bool var_83;
        bool var_84;
        bool var_85;
        bool var_86;
        bool var_87;
        bool var_88;
        bool var_89;
        bool var_90;
        bool var_91;
        const wp::int32 var_92 = 0;
        wp::int32 var_93;
        wp::int32 var_94;
        const wp::int32 var_95 = 0;
        bool var_96;
        const wp::int32 var_97 = 8;
        wp::range_t var_98;
        wp::int32 var_99;
        wp::int32 var_100;
        const wp::float32 var_101 = 1.0;
        wp::float32 var_102;
        wp::int32 var_103;
        const wp::int32 var_104 = 0;
        bool var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        wp::int32 var_108;
        wp::float32 var_109;
        wp::int32 var_110;
        const wp::float32 var_111 = 1.0;
        wp::float32 var_112;
        wp::int32 var_113;
        const wp::int32 var_114 = 0;
        bool var_115;
        wp::int32 var_116;
        wp::float32 var_117;
        wp::int32 var_118;
        wp::float32 var_119;
        wp::int32 var_120;
        const wp::float32 var_121 = 1.0;
        wp::float32 var_122;
        wp::int32 var_123;
        const wp::int32 var_124 = 0;
        bool var_125;
        wp::int32 var_126;
        wp::float32 var_127;
        wp::int32 var_128;
        wp::float32 var_129;
        wp::int32* var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        bool var_133;
        const wp::int32 var_134 = 0;
        bool var_135;
        wp::int32* var_136;
        const wp::int32 var_137 = 1;
        wp::int32 var_138;
        wp::int32 var_139;
        const wp::int32 var_140 = 0;
        bool var_141;
        const wp::int32 var_142 = 0;
        wp::int32 var_143;
        wp::float32 var_144;
        wp::float32 var_145;
        const wp::int32 var_146 = 0;
        bool var_147;
        wp::vec_t<3, wp::float32>* var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        wp::vec_t<3, wp::float32> var_151;
        wp::vec_t<3, wp::float32> var_152;
        wp::vec_t<3, wp::float32>* var_153;
        wp::vec_t<3, wp::float32>* var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        wp::vec_t<3, wp::float32> var_158;
        wp::vec_t<3, wp::float32> var_159;
        wp::vec_t<3, wp::float32> var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        const wp::int32 var_163 = 0;
        bool var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<3, wp::float32> var_168;
        const wp::int32 var_169 = 1;
        wp::int32 var_170;
        wp::vec_t<3, wp::float32> var_171;
        wp::int32 var_172;
        wp::vec_t<3, wp::float32> var_173;
        wp::int32 var_174;
        wp::vec_t<3, wp::float32> var_175;
        wp::int32 var_176;
        const wp::int32 var_177 = 0;
        bool var_178;
        wp::vec_t<3, wp::float32>* var_179;
        wp::vec_t<3, wp::float32> var_180;
        wp::vec_t<3, wp::float32> var_181;
        wp::vec_t<3, wp::float32> var_182;
        wp::float32 var_183;
        bool var_184;
        wp::float32 var_185;
        wp::vec_t<3, wp::float32> var_186;
        wp::vec_t<3, wp::float32> var_187;
        //---------
        // forward
        // def prolongate_shape_matching_corrections_uniform8(                                    <L 1080>
        // particle_idx = wp.tid()                                                                <L 1102>
        var_0 = builtin_tid1d();
        // x_old = particle_q[particle_idx]                                                       <L 1103>
        var_1 = wp::address(var_particle_q, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // x_new = x_old                                                                          <L 1104>
        var_4 = wp::copy(var_2);
        // v_old = particle_qd[particle_idx]                                                      <L 1105>
        var_5 = wp::address(var_particle_qd, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // changed = int(0)                                                                       <L 1106>
        var_9 = wp::int(var_8);
        // if (                                                                                   <L 1108>
        // (particle_flags[particle_idx] & ParticleFlags.ACTIVE) != 0                             <L 1109>
        var_11 = wp::address(var_particle_flags, var_0);
        var_14 = wp::load(var_11);
        var_13 = wp::bit_and(var_14, var_12);
        var_16 = (var_13 != var_15);
        var_10 = var_16;
        if (var_10) {
            // and particle_inv_mass[particle_idx] > 0.0                                          <L 1110>
            var_17 = wp::address(var_particle_inv_mass, var_0);
            var_20 = wp::load(var_17);
            var_19 = (var_20 > var_18);
            var_10 = var_10 && var_19;
        }
        if (var_10) {
            // cluster_idx = child_cluster[particle_idx]                                          <L 1112>
            var_21 = wp::address(var_child_cluster, var_0);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // if cluster_idx >= 0 and cluster_active[cluster_idx] != 0:                          <L 1113>
            var_26 = (var_22 >= var_25);
            var_24 = var_26;
            if (var_24) {
                var_27 = wp::address(var_cluster_active, var_22);
                var_30 = wp::load(var_27);
                var_29 = (var_30 != var_28);
                var_24 = var_24 && var_29;
            }
            if (var_24) {
                // gx = node_grid_xyz[particle_idx, 0]                                            <L 1114>
                var_32 = wp::address(var_node_grid_xyz, var_0, var_31);
                var_34 = wp::load(var_32);
                var_33 = wp::copy(var_34);
                // gy = node_grid_xyz[particle_idx, 1]                                            <L 1115>
                var_36 = wp::address(var_node_grid_xyz, var_0, var_35);
                var_38 = wp::load(var_36);
                var_37 = wp::copy(var_38);
                // gz = node_grid_xyz[particle_idx, 2]                                            <L 1116>
                var_40 = wp::address(var_node_grid_xyz, var_0, var_39);
                var_42 = wp::load(var_40);
                var_41 = wp::copy(var_42);
                // lx = block_keys[cluster_idx, 0] * block_size                                   <L 1117>
                var_44 = wp::address(var_block_keys, var_22, var_43);
                var_46 = wp::load(var_44);
                var_45 = wp::mul(var_46, var_block_size);
                // ly = block_keys[cluster_idx, 1] * block_size                                   <L 1118>
                var_48 = wp::address(var_block_keys, var_22, var_47);
                var_50 = wp::load(var_48);
                var_49 = wp::mul(var_50, var_block_size);
                // lz = block_keys[cluster_idx, 2] * block_size                                   <L 1119>
                var_52 = wp::address(var_block_keys, var_22, var_51);
                var_54 = wp::load(var_52);
                var_53 = wp::mul(var_54, var_block_size);
                // ux = lx + block_size                                                           <L 1120>
                var_55 = wp::add(var_45, var_block_size);
                // uy = ly + block_size                                                           <L 1121>
                var_56 = wp::add(var_49, var_block_size);
                // uz = lz + block_size                                                           <L 1122>
                var_57 = wp::add(var_53, var_block_size);
                // fx = float(gx - lx) / float(block_size)                                        <L 1123>
                var_58 = wp::sub(var_33, var_45);
                var_59 = wp::float(var_58);
                var_60 = wp::float(var_block_size);
                var_61 = wp::div(var_59, var_60);
                // fy = float(gy - ly) / float(block_size)                                        <L 1124>
                var_62 = wp::sub(var_37, var_49);
                var_63 = wp::float(var_62);
                var_64 = wp::float(var_block_size);
                var_65 = wp::div(var_63, var_64);
                // fz = float(gz - lz) / float(block_size)                                        <L 1125>
                var_66 = wp::sub(var_41, var_53);
                var_67 = wp::float(var_66);
                var_68 = wp::float(var_block_size);
                var_69 = wp::div(var_67, var_68);
                // interpolated = wp.vec3(0.0, 0.0, 0.0)                                          <L 1126>
                var_73 = wp::vec_t<3, wp::float32>(var_70, var_71, var_72);
                // valid = int(1)                                                                 <L 1127>
                var_75 = wp::int(var_74);
                // if (                                                                           <L 1128>
                // lx < 0                                                                         <L 1129>
                var_78 = (var_45 < var_77);
                var_76 = var_78;
                if (!var_76) {
                    // or ly < 0                                                                  <L 1130>
                    var_80 = (var_49 < var_79);
                    var_76 = var_76 || var_80;
                }
                if (!var_76) {
                    // or lz < 0                                                                  <L 1131>
                    var_82 = (var_53 < var_81);
                    var_76 = var_76 || var_82;
                }
                if (!var_76) {
                    // or ux > max_grid_x                                                         <L 1132>
                    var_83 = (var_55 > var_max_grid_x);
                    var_76 = var_76 || var_83;
                }
                if (!var_76) {
                    // or uy > max_grid_y                                                         <L 1133>
                    var_84 = (var_56 > var_max_grid_y);
                    var_76 = var_76 || var_84;
                }
                if (!var_76) {
                    // or uz > max_grid_z                                                         <L 1134>
                    var_85 = (var_57 > var_max_grid_z);
                    var_76 = var_76 || var_85;
                }
                if (!var_76) {
                    // or gx < lx                                                                 <L 1135>
                    var_86 = (var_33 < var_45);
                    var_76 = var_76 || var_86;
                }
                if (!var_76) {
                    // or gy < ly                                                                 <L 1136>
                    var_87 = (var_37 < var_49);
                    var_76 = var_76 || var_87;
                }
                if (!var_76) {
                    // or gz < lz                                                                 <L 1137>
                    var_88 = (var_41 < var_53);
                    var_76 = var_76 || var_88;
                }
                if (!var_76) {
                    // or gx > ux                                                                 <L 1138>
                    var_89 = (var_33 > var_55);
                    var_76 = var_76 || var_89;
                }
                if (!var_76) {
                    // or gy > uy                                                                 <L 1139>
                    var_90 = (var_37 > var_56);
                    var_76 = var_76 || var_90;
                }
                if (!var_76) {
                    // or gz > uz                                                                 <L 1140>
                    var_91 = (var_41 > var_57);
                    var_76 = var_76 || var_91;
                }
                if (var_76) {
                    // valid = int(0)                                                             <L 1142>
                    var_93 = wp::int(var_92);
                }
                var_94 = wp::where(var_76, var_93, var_75);
                // if valid != 0:                                                                 <L 1143>
                var_96 = (var_94 != var_95);
                if (var_96) {
                    // for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):                 <L 1144>
                    var_98 = wp::range(var_97);
                    start_for_0:;
                        if (iter_cmp(var_98) == 0) goto end_for_0;
                        var_99 = wp::iter_next(var_98);
                        // px = lx                                                                <L 1145>
                        var_100 = wp::copy(var_45);
                        // wx = 1.0 - fx                                                          <L 1146>
                        var_102 = wp::sub(var_101, var_61);
                        // if _slot_uses_upper_x(parent_slot) != 0:                               <L 1147>
                        var_103 = _slot_uses_upper_x_0(var_99);
                        var_105 = (var_103 != var_104);
                        if (var_105) {
                            // px = ux                                                            <L 1148>
                            var_106 = wp::copy(var_55);
                            // wx = fx                                                            <L 1149>
                            var_107 = wp::copy(var_61);
                        }
                        var_108 = wp::where(var_105, var_106, var_100);
                        var_109 = wp::where(var_105, var_107, var_102);
                        // py = ly                                                                <L 1151>
                        var_110 = wp::copy(var_49);
                        // wy = 1.0 - fy                                                          <L 1152>
                        var_112 = wp::sub(var_111, var_65);
                        // if _slot_uses_upper_y(parent_slot) != 0:                               <L 1153>
                        var_113 = _slot_uses_upper_y_0(var_99);
                        var_115 = (var_113 != var_114);
                        if (var_115) {
                            // py = uy                                                            <L 1154>
                            var_116 = wp::copy(var_56);
                            // wy = fy                                                            <L 1155>
                            var_117 = wp::copy(var_65);
                        }
                        var_118 = wp::where(var_115, var_116, var_110);
                        var_119 = wp::where(var_115, var_117, var_112);
                        // pz = lz                                                                <L 1157>
                        var_120 = wp::copy(var_53);
                        // wz = 1.0 - fz                                                          <L 1158>
                        var_122 = wp::sub(var_121, var_69);
                        // if _slot_uses_upper_z(parent_slot) != 0:                               <L 1159>
                        var_123 = _slot_uses_upper_z_0(var_99);
                        var_125 = (var_123 != var_124);
                        if (var_125) {
                            // pz = uz                                                            <L 1160>
                            var_126 = wp::copy(var_57);
                            // wz = fz                                                            <L 1161>
                            var_127 = wp::copy(var_69);
                        }
                        var_128 = wp::where(var_125, var_126, var_120);
                        var_129 = wp::where(var_125, var_127, var_122);
                        // parent_idx = grid_to_node[px, py, pz]                                  <L 1163>
                        var_130 = wp::address(var_grid_to_node, var_108, var_118, var_128);
                        var_132 = wp::load(var_130);
                        var_131 = wp::copy(var_132);
                        // if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:       <L 1164>
                        var_135 = (var_131 < var_134);
                        var_133 = var_135;
                        if (!var_133) {
                            var_136 = wp::address(var_particle_flags, var_131);
                            var_139 = wp::load(var_136);
                            var_138 = wp::bit_and(var_139, var_137);
                            var_141 = (var_138 == var_140);
                            var_133 = var_133 || var_141;
                        }
                        if (var_133) {
                            // valid = int(0)                                                     <L 1165>
                            var_143 = wp::int(var_142);
                            // break                                                              <L 1166>
                            wp::assign(var_94, var_143);
                            goto end_for_0;
                        }
                        // weight = wx * wy * wz                                                  <L 1167>
                        var_144 = wp::mul(var_109, var_119);
                        var_145 = wp::mul(var_144, var_129);
                        // if absolute_projection != 0:                                           <L 1168>
                        var_147 = (var_absolute_projection != var_146);
                        if (var_147) {
                            // interpolated += particle_q[parent_idx] * weight                    <L 1169>
                            var_148 = wp::address(var_particle_q, var_131);
                            var_150 = wp::load(var_148);
                            var_149 = wp::mul(var_150, var_145);
                            var_151 = wp::add(var_73, var_149);
                        }
                        var_152 = wp::where(var_147, var_151, var_73);
                        if (!var_147) {
                            // interpolated += (particle_q[parent_idx] - coarse_q_snapshot[parent_idx]) * weight       <L 1171>
                            var_153 = wp::address(var_particle_q, var_131);
                            var_154 = wp::address(var_coarse_q_snapshot, var_131);
                            var_156 = wp::load(var_153);
                            var_157 = wp::load(var_154);
                            var_155 = wp::sub(var_156, var_157);
                            var_158 = wp::mul(var_155, var_145);
                            var_159 = wp::add(var_152, var_158);
                        }
                        var_160 = wp::where(var_147, var_152, var_159);
                        wp::assign(var_73, var_160);
                        goto start_for_0;
                    end_for_0:;
                }
                // if valid != 0:                                                                 <L 1172>
                var_162 = (var_94 != var_161);
                if (var_162) {
                    // if absolute_projection != 0:                                               <L 1173>
                    var_164 = (var_absolute_projection != var_163);
                    if (var_164) {
                        // x_new = interpolated                                                   <L 1174>
                        var_165 = wp::copy(var_73);
                    }
                    var_166 = wp::where(var_164, var_165, var_4);
                    if (!var_164) {
                        // x_new = x_old + interpolated                                           <L 1176>
                        var_167 = wp::add(var_2, var_73);
                    }
                    var_168 = wp::where(var_164, var_166, var_167);
                    // changed = int(1)                                                           <L 1177>
                    var_170 = wp::int(var_169);
                }
                var_171 = wp::where(var_162, var_168, var_4);
                var_172 = wp::where(var_162, var_170, var_9);
            }
            var_173 = wp::where(var_24, var_171, var_4);
            var_174 = wp::where(var_24, var_172, var_9);
        }
        var_175 = wp::where(var_10, var_173, var_4);
        var_176 = wp::where(var_10, var_174, var_9);
        // if changed == 0:                                                                       <L 1179>
        var_178 = (var_176 == var_177);
        if (var_178) {
            // particle_q_out[particle_idx] = x_old                                               <L 1180>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1181>
            wp::array_store(var_particle_qd_out, var_0, var_6);
            // return                                                                             <L 1182>
            continue;
        }
        // v_new = (x_new - particle_q_init[particle_idx]) / dt                                   <L 1184>
        var_179 = wp::address(var_particle_q_init, var_0);
        var_181 = wp::load(var_179);
        var_180 = wp::sub(var_175, var_181);
        var_182 = wp::div(var_180, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 1185>
        var_183 = wp::length(var_182);
        // if v_new_mag > v_max:                                                                  <L 1186>
        var_184 = (var_183 > var_v_max);
        if (var_184) {
            // v_new *= v_max / v_new_mag                                                         <L 1187>
            var_185 = wp::div(var_v_max, var_183);
            var_186 = wp::mul(var_182, var_185);
        }
        var_187 = wp::where(var_184, var_186, var_182);
        // particle_q_out[particle_idx] = x_new                                                   <L 1189>
        wp::array_store(var_particle_q_out, var_0, var_175);
        // particle_qd_out[particle_idx] = v_new                                                  <L 1190>
        wp::array_store(var_particle_qd_out, var_0, var_187);
    }
}



extern "C" __global__ void accumulate_runtime_cluster_counts_kernel_e9a047b9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
    wp::array_t<wp::int32> var_particle_cluster_counts)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 0;
        bool var_18;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        //---------
        // forward
        // def accumulate_runtime_cluster_counts_kernel(                                          <L 1521>
        // cluster_idx = wp.tid()                                                                 <L 1527>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0:                                                   <L 1528>
        var_1 = wp::address(var_cluster_active, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 1529>
            continue;
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 1531>
        var_5 = wp::address(var_cluster_offsets, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 1532>
        var_9 = wp::add(var_0, var_8);
        var_10 = wp::address(var_cluster_offsets, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // while cursor < end:                                                                    <L 1533>
        start_while_1:;
        var_13 = (var_6 < var_11);
        if ((var_13) == false) goto end_while_1;
            // particle_idx = cluster_indices[cursor]                                             <L 1534>
            var_14 = wp::address(var_cluster_indices, var_6);
            var_16 = wp::load(var_14);
            var_15 = wp::copy(var_16);
            // if particle_idx >= 0:                                                              <L 1535>
            var_18 = (var_15 >= var_17);
            if (var_18) {
                // wp.atomic_add(particle_cluster_counts, particle_idx, 1)                        <L 1536>
                var_20 = wp::atomic_add(var_particle_cluster_counts, var_15, var_19);
            }
            // cursor += 1                                                                        <L 1537>
            var_22 = wp::add(var_6, var_21);
            wp::assign(var_6, var_22);
        goto start_while_1;
        end_while_1:;
    }
}



extern "C" __global__ void project_shape_matching_children_uniform8_0b6b710a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_init,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_awake_l0_cluster_counts,
    wp::array_t<wp::int32> var_child_cluster,
    wp::array_t<wp::int32> var_block_keys,
    wp::array_t<wp::int32> var_node_grid_xyz,
    wp::array_t<wp::int32> var_grid_to_node,
    wp::int32 var_block_size,
    wp::int32 var_max_grid_x,
    wp::int32 var_max_grid_y,
    wp::int32 var_max_grid_z,
    wp::array_t<wp::int32> var_projection_active,
    wp::float32 var_projection_stiffness,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd_out)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        const wp::float32 var_10 = 1.0;
        bool var_11;
        const wp::float32 var_12 = 1.0;
        wp::float32 var_13;
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::float32* var_21;
        const wp::float32 var_22 = 0.0;
        bool var_23;
        wp::float32 var_24;
        wp::int32* var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        bool var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 1;
        wp::int32* var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 2;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        const wp::int32 var_51 = 0;
        wp::int32* var_52;
        wp::int32 var_53;
        wp::int32 var_54;
        const wp::int32 var_55 = 1;
        wp::int32* var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        const wp::int32 var_59 = 2;
        wp::int32* var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        wp::int32 var_63;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::int32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::int32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        const wp::float32 var_78 = 0.0;
        const wp::float32 var_79 = 0.0;
        const wp::float32 var_80 = 0.0;
        wp::vec_t<3, wp::float32> var_81;
        const wp::int32 var_82 = 1;
        wp::int32 var_83;
        bool var_84;
        const wp::int32 var_85 = 0;
        bool var_86;
        const wp::int32 var_87 = 0;
        bool var_88;
        const wp::int32 var_89 = 0;
        bool var_90;
        bool var_91;
        bool var_92;
        bool var_93;
        bool var_94;
        bool var_95;
        bool var_96;
        bool var_97;
        bool var_98;
        bool var_99;
        const wp::int32 var_100 = 0;
        wp::int32 var_101;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        bool var_104;
        const wp::int32 var_105 = 8;
        wp::range_t var_106;
        wp::int32 var_107;
        wp::int32 var_108;
        const wp::float32 var_109 = 1.0;
        wp::float32 var_110;
        wp::int32 var_111;
        const wp::int32 var_112 = 0;
        bool var_113;
        wp::int32 var_114;
        wp::float32 var_115;
        wp::int32 var_116;
        wp::float32 var_117;
        wp::int32 var_118;
        const wp::float32 var_119 = 1.0;
        wp::float32 var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 0;
        bool var_123;
        wp::int32 var_124;
        wp::float32 var_125;
        wp::int32 var_126;
        wp::float32 var_127;
        wp::int32 var_128;
        const wp::float32 var_129 = 1.0;
        wp::float32 var_130;
        wp::int32 var_131;
        const wp::int32 var_132 = 0;
        bool var_133;
        wp::int32 var_134;
        wp::float32 var_135;
        wp::int32 var_136;
        wp::float32 var_137;
        wp::int32* var_138;
        wp::int32 var_139;
        wp::int32 var_140;
        bool var_141;
        const wp::int32 var_142 = 0;
        bool var_143;
        wp::int32* var_144;
        const wp::int32 var_145 = 1;
        wp::int32 var_146;
        wp::int32 var_147;
        const wp::int32 var_148 = 0;
        bool var_149;
        const wp::int32 var_150 = 0;
        wp::int32 var_151;
        wp::vec_t<3, wp::float32>* var_152;
        wp::float32 var_153;
        wp::float32 var_154;
        wp::vec_t<3, wp::float32> var_155;
        wp::vec_t<3, wp::float32> var_156;
        wp::vec_t<3, wp::float32> var_157;
        const wp::int32 var_158 = 0;
        bool var_159;
        wp::vec_t<3, wp::float32> var_160;
        wp::vec_t<3, wp::float32> var_161;
        wp::vec_t<3, wp::float32> var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::vec_t<3, wp::float32> var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::float32 var_167;
        bool var_168;
        wp::float32 var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::vec_t<3, wp::float32> var_171;
        //---------
        // forward
        // def project_shape_matching_children_uniform8(                                          <L 1264>
        // particle_idx = wp.tid()                                                                <L 1286>
        var_0 = builtin_tid1d();
        // x_old = particle_q[particle_idx]                                                       <L 1287>
        var_1 = wp::address(var_particle_q, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // v_old = particle_qd[particle_idx]                                                      <L 1288>
        var_4 = wp::address(var_particle_qd, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // alpha = projection_stiffness                                                           <L 1290>
        var_7 = wp::copy(var_projection_stiffness);
        // if alpha <= 0.0:                                                                       <L 1291>
        var_9 = (var_7 <= var_8);
        if (var_9) {
            // particle_q_out[particle_idx] = x_old                                               <L 1292>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1293>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1294>
            continue;
        }
        // if alpha > 1.0:                                                                        <L 1295>
        var_11 = (var_7 > var_10);
        if (var_11) {
            // alpha = 1.0                                                                        <L 1296>
        }
        var_13 = wp::where(var_11, var_12, var_7);
        // if (                                                                                   <L 1298>
        // (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0                             <L 1299>
        var_15 = wp::address(var_particle_flags, var_0);
        var_18 = wp::load(var_15);
        var_17 = wp::bit_and(var_18, var_16);
        var_20 = (var_17 == var_19);
        var_14 = var_20;
        if (!var_14) {
            // or particle_inv_mass[particle_idx] <= 0.0                                          <L 1300>
            var_21 = wp::address(var_particle_inv_mass, var_0);
            var_24 = wp::load(var_21);
            var_23 = (var_24 <= var_22);
            var_14 = var_14 || var_23;
        }
        if (!var_14) {
            // or awake_l0_cluster_counts[particle_idx] > 0                                       <L 1301>
            var_25 = wp::address(var_awake_l0_cluster_counts, var_0);
            var_28 = wp::load(var_25);
            var_27 = (var_28 > var_26);
            var_14 = var_14 || var_27;
        }
        if (var_14) {
            // particle_q_out[particle_idx] = x_old                                               <L 1303>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1304>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1305>
            continue;
        }
        // cluster_idx = child_cluster[particle_idx]                                              <L 1307>
        var_29 = wp::address(var_child_cluster, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // if cluster_idx < 0 or projection_active[cluster_idx] == 0:                             <L 1308>
        var_34 = (var_30 < var_33);
        var_32 = var_34;
        if (!var_32) {
            var_35 = wp::address(var_projection_active, var_30);
            var_38 = wp::load(var_35);
            var_37 = (var_38 == var_36);
            var_32 = var_32 || var_37;
        }
        if (var_32) {
            // particle_q_out[particle_idx] = x_old                                               <L 1309>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1310>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1311>
            continue;
        }
        // gx = node_grid_xyz[particle_idx, 0]                                                    <L 1313>
        var_40 = wp::address(var_node_grid_xyz, var_0, var_39);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // gy = node_grid_xyz[particle_idx, 1]                                                    <L 1314>
        var_44 = wp::address(var_node_grid_xyz, var_0, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // gz = node_grid_xyz[particle_idx, 2]                                                    <L 1315>
        var_48 = wp::address(var_node_grid_xyz, var_0, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // lx = block_keys[cluster_idx, 0] * block_size                                           <L 1316>
        var_52 = wp::address(var_block_keys, var_30, var_51);
        var_54 = wp::load(var_52);
        var_53 = wp::mul(var_54, var_block_size);
        // ly = block_keys[cluster_idx, 1] * block_size                                           <L 1317>
        var_56 = wp::address(var_block_keys, var_30, var_55);
        var_58 = wp::load(var_56);
        var_57 = wp::mul(var_58, var_block_size);
        // lz = block_keys[cluster_idx, 2] * block_size                                           <L 1318>
        var_60 = wp::address(var_block_keys, var_30, var_59);
        var_62 = wp::load(var_60);
        var_61 = wp::mul(var_62, var_block_size);
        // ux = lx + block_size                                                                   <L 1319>
        var_63 = wp::add(var_53, var_block_size);
        // uy = ly + block_size                                                                   <L 1320>
        var_64 = wp::add(var_57, var_block_size);
        // uz = lz + block_size                                                                   <L 1321>
        var_65 = wp::add(var_61, var_block_size);
        // fx = float(gx - lx) / float(block_size)                                                <L 1322>
        var_66 = wp::sub(var_41, var_53);
        var_67 = wp::float(var_66);
        var_68 = wp::float(var_block_size);
        var_69 = wp::div(var_67, var_68);
        // fy = float(gy - ly) / float(block_size)                                                <L 1323>
        var_70 = wp::sub(var_45, var_57);
        var_71 = wp::float(var_70);
        var_72 = wp::float(var_block_size);
        var_73 = wp::div(var_71, var_72);
        // fz = float(gz - lz) / float(block_size)                                                <L 1324>
        var_74 = wp::sub(var_49, var_61);
        var_75 = wp::float(var_74);
        var_76 = wp::float(var_block_size);
        var_77 = wp::div(var_75, var_76);
        // x_target = wp.vec3(0.0, 0.0, 0.0)                                                      <L 1325>
        var_81 = wp::vec_t<3, wp::float32>(var_78, var_79, var_80);
        // valid = int(1)                                                                         <L 1326>
        var_83 = wp::int(var_82);
        // if (                                                                                   <L 1327>
        // lx < 0                                                                                 <L 1328>
        var_86 = (var_53 < var_85);
        var_84 = var_86;
        if (!var_84) {
            // or ly < 0                                                                          <L 1329>
            var_88 = (var_57 < var_87);
            var_84 = var_84 || var_88;
        }
        if (!var_84) {
            // or lz < 0                                                                          <L 1330>
            var_90 = (var_61 < var_89);
            var_84 = var_84 || var_90;
        }
        if (!var_84) {
            // or ux > max_grid_x                                                                 <L 1331>
            var_91 = (var_63 > var_max_grid_x);
            var_84 = var_84 || var_91;
        }
        if (!var_84) {
            // or uy > max_grid_y                                                                 <L 1332>
            var_92 = (var_64 > var_max_grid_y);
            var_84 = var_84 || var_92;
        }
        if (!var_84) {
            // or uz > max_grid_z                                                                 <L 1333>
            var_93 = (var_65 > var_max_grid_z);
            var_84 = var_84 || var_93;
        }
        if (!var_84) {
            // or gx < lx                                                                         <L 1334>
            var_94 = (var_41 < var_53);
            var_84 = var_84 || var_94;
        }
        if (!var_84) {
            // or gy < ly                                                                         <L 1335>
            var_95 = (var_45 < var_57);
            var_84 = var_84 || var_95;
        }
        if (!var_84) {
            // or gz < lz                                                                         <L 1336>
            var_96 = (var_49 < var_61);
            var_84 = var_84 || var_96;
        }
        if (!var_84) {
            // or gx > ux                                                                         <L 1337>
            var_97 = (var_41 > var_63);
            var_84 = var_84 || var_97;
        }
        if (!var_84) {
            // or gy > uy                                                                         <L 1338>
            var_98 = (var_45 > var_64);
            var_84 = var_84 || var_98;
        }
        if (!var_84) {
            // or gz > uz                                                                         <L 1339>
            var_99 = (var_49 > var_65);
            var_84 = var_84 || var_99;
        }
        if (var_84) {
            // valid = int(0)                                                                     <L 1341>
            var_101 = wp::int(var_100);
        }
        var_102 = wp::where(var_84, var_101, var_83);
        // if valid != 0:                                                                         <L 1342>
        var_104 = (var_102 != var_103);
        if (var_104) {
            // for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):                         <L 1343>
            var_106 = wp::range(var_105);
            start_for_3:;
                if (iter_cmp(var_106) == 0) goto end_for_3;
                var_107 = wp::iter_next(var_106);
                // px = lx                                                                        <L 1344>
                var_108 = wp::copy(var_53);
                // wx = 1.0 - fx                                                                  <L 1345>
                var_110 = wp::sub(var_109, var_69);
                // if _slot_uses_upper_x(parent_slot) != 0:                                       <L 1346>
                var_111 = _slot_uses_upper_x_0(var_107);
                var_113 = (var_111 != var_112);
                if (var_113) {
                    // px = ux                                                                    <L 1347>
                    var_114 = wp::copy(var_63);
                    // wx = fx                                                                    <L 1348>
                    var_115 = wp::copy(var_69);
                }
                var_116 = wp::where(var_113, var_114, var_108);
                var_117 = wp::where(var_113, var_115, var_110);
                // py = ly                                                                        <L 1350>
                var_118 = wp::copy(var_57);
                // wy = 1.0 - fy                                                                  <L 1351>
                var_120 = wp::sub(var_119, var_73);
                // if _slot_uses_upper_y(parent_slot) != 0:                                       <L 1352>
                var_121 = _slot_uses_upper_y_0(var_107);
                var_123 = (var_121 != var_122);
                if (var_123) {
                    // py = uy                                                                    <L 1353>
                    var_124 = wp::copy(var_64);
                    // wy = fy                                                                    <L 1354>
                    var_125 = wp::copy(var_73);
                }
                var_126 = wp::where(var_123, var_124, var_118);
                var_127 = wp::where(var_123, var_125, var_120);
                // pz = lz                                                                        <L 1356>
                var_128 = wp::copy(var_61);
                // wz = 1.0 - fz                                                                  <L 1357>
                var_130 = wp::sub(var_129, var_77);
                // if _slot_uses_upper_z(parent_slot) != 0:                                       <L 1358>
                var_131 = _slot_uses_upper_z_0(var_107);
                var_133 = (var_131 != var_132);
                if (var_133) {
                    // pz = uz                                                                    <L 1359>
                    var_134 = wp::copy(var_65);
                    // wz = fz                                                                    <L 1360>
                    var_135 = wp::copy(var_77);
                }
                var_136 = wp::where(var_133, var_134, var_128);
                var_137 = wp::where(var_133, var_135, var_130);
                // parent_idx = grid_to_node[px, py, pz]                                          <L 1362>
                var_138 = wp::address(var_grid_to_node, var_116, var_126, var_136);
                var_140 = wp::load(var_138);
                var_139 = wp::copy(var_140);
                // if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:       <L 1363>
                var_143 = (var_139 < var_142);
                var_141 = var_143;
                if (!var_141) {
                    var_144 = wp::address(var_particle_flags, var_139);
                    var_147 = wp::load(var_144);
                    var_146 = wp::bit_and(var_147, var_145);
                    var_149 = (var_146 == var_148);
                    var_141 = var_141 || var_149;
                }
                if (var_141) {
                    // valid = int(0)                                                             <L 1364>
                    var_151 = wp::int(var_150);
                    // break                                                                      <L 1365>
                    wp::assign(var_102, var_151);
                    goto end_for_3;
                }
                // x_target += particle_q[parent_idx] * (wx * wy * wz)                            <L 1366>
                var_152 = wp::address(var_particle_q, var_139);
                var_153 = wp::mul(var_117, var_127);
                var_154 = wp::mul(var_153, var_137);
                var_156 = wp::load(var_152);
                var_155 = wp::mul(var_156, var_154);
                var_157 = wp::add(var_81, var_155);
                wp::assign(var_81, var_157);
                goto start_for_3;
            end_for_3:;
        }
        // if valid == 0:                                                                         <L 1368>
        var_159 = (var_102 == var_158);
        if (var_159) {
            // particle_q_out[particle_idx] = x_old                                               <L 1369>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1370>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1371>
            continue;
        }
        // x_new = x_old + (x_target - x_old) * alpha                                             <L 1373>
        var_160 = wp::sub(var_81, var_2);
        var_161 = wp::mul(var_160, var_13);
        var_162 = wp::add(var_2, var_161);
        // v_new = (x_new - particle_q_init[particle_idx]) / dt                                   <L 1374>
        var_163 = wp::address(var_particle_q_init, var_0);
        var_165 = wp::load(var_163);
        var_164 = wp::sub(var_162, var_165);
        var_166 = wp::div(var_164, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 1375>
        var_167 = wp::length(var_166);
        // if v_new_mag > v_max:                                                                  <L 1376>
        var_168 = (var_167 > var_v_max);
        if (var_168) {
            // v_new *= v_max / v_new_mag                                                         <L 1377>
            var_169 = wp::div(var_v_max, var_167);
            var_170 = wp::mul(var_166, var_169);
        }
        var_171 = wp::where(var_168, var_170, var_166);
        // particle_q_out[particle_idx] = x_new                                                   <L 1379>
        wp::array_store(var_particle_q_out, var_0, var_162);
        // particle_qd_out[particle_idx] = v_new                                                  <L 1380>
        wp::array_store(var_particle_qd_out, var_0, var_171);
    }
}



extern "C" __global__ void mark_wake_blocks_from_deleted_cells_kernel_b4b937be_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_to_cluster,
    wp::array_t<wp::int32> var_block_keys,
    wp::int32 var_wake_halo_blocks,
    wp::array_t<wp::int32> var_wake_mask)
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        bool var_8;
        const wp::int32 var_9 = 0;
        bool var_10;
        bool var_11;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        const wp::int32 var_17 = 0;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 2;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::shape_t* var_29;
        const wp::int32 var_30 = 0;
        wp::int32 var_31;
        wp::shape_t var_32;
        wp::shape_t* var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::shape_t var_36;
        wp::shape_t* var_37;
        const wp::int32 var_38 = 2;
        wp::int32 var_39;
        wp::shape_t var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::int32 var_46;
        bool var_47;
        wp::int32 var_48;
        bool var_49;
        const wp::int32 var_50 = 0;
        bool var_51;
        bool var_52;
        wp::int32 var_53;
        bool var_54;
        wp::int32 var_55;
        bool var_56;
        const wp::int32 var_57 = 0;
        bool var_58;
        bool var_59;
        wp::int32 var_60;
        bool var_61;
        wp::int32 var_62;
        bool var_63;
        const wp::int32 var_64 = 0;
        bool var_65;
        bool var_66;
        const wp::int32 var_67 = 1;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        const wp::int32 var_72 = 1;
        wp::int32 var_73;
        //---------
        // forward
        // def mark_wake_blocks_from_deleted_cells_kernel(                                        <L 1384>
        // i = wp.tid()                                                                           <L 1393>
        var_0 = builtin_tid1d();
        // if i >= deleted_count[0]:                                                              <L 1394>
        var_2 = wp::address(var_deleted_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 1395>
            continue;
        }
        // cell_idx = deleted_cells[i]                                                            <L 1397>
        var_5 = wp::address(var_deleted_cells, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 1398>
        var_10 = (var_6 < var_9);
        var_8 = var_10;
        if (!var_8) {
            var_11 = (var_6 >= var_num_cells);
            var_8 = var_8 || var_11;
        }
        if (var_8) {
            // return                                                                             <L 1399>
            continue;
        }
        // cluster_idx = cell_to_cluster[cell_idx]                                                <L 1401>
        var_12 = wp::address(var_cell_to_cluster, var_6);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if cluster_idx < 0:                                                                    <L 1402>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 1403>
            continue;
        }
        // bx = block_keys[cluster_idx, 0]                                                        <L 1405>
        var_18 = wp::address(var_block_keys, var_13, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // by = block_keys[cluster_idx, 1]                                                        <L 1406>
        var_22 = wp::address(var_block_keys, var_13, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // bz = block_keys[cluster_idx, 2]                                                        <L 1407>
        var_26 = wp::address(var_block_keys, var_13, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // nx = wake_mask.shape[0]                                                                <L 1408>
        var_29 = &(var_wake_mask.shape);
        var_32 = wp::load(var_29);
        var_31 = wp::extract(var_32, var_30);
        // ny = wake_mask.shape[1]                                                                <L 1409>
        var_33 = &(var_wake_mask.shape);
        var_36 = wp::load(var_33);
        var_35 = wp::extract(var_36, var_34);
        // nz = wake_mask.shape[2]                                                                <L 1410>
        var_37 = &(var_wake_mask.shape);
        var_40 = wp::load(var_37);
        var_39 = wp::extract(var_40, var_38);
        // halo = wake_halo_blocks                                                                <L 1411>
        var_41 = wp::copy(var_wake_halo_blocks);
        // if halo < 0:                                                                           <L 1412>
        var_43 = (var_41 < var_42);
        if (var_43) {
            // halo = 0                                                                           <L 1413>
        }
        var_45 = wp::where(var_43, var_44, var_41);
        // dx = -halo                                                                             <L 1415>
        var_46 = wp::neg(var_45);
        // while dx <= halo:                                                                      <L 1416>
        start_while_3:;
        var_47 = (var_46 <= var_45);
        if ((var_47) == false) goto end_while_3;
            // x = bx + dx                                                                        <L 1417>
            var_48 = wp::add(var_19, var_46);
            // if x >= 0 and x < nx:                                                              <L 1418>
            var_51 = (var_48 >= var_50);
            var_49 = var_51;
            if (var_49) {
                var_52 = (var_48 < var_31);
                var_49 = var_49 && var_52;
            }
            if (var_49) {
                // dy = -halo                                                                     <L 1419>
                var_53 = wp::neg(var_45);
                // while dy <= halo:                                                              <L 1420>
        start_while_5:;
                var_54 = (var_53 <= var_45);
        if ((var_54) == false) goto end_while_5;
                    // y = by + dy                                                                <L 1421>
                    var_55 = wp::add(var_23, var_53);
                    // if y >= 0 and y < ny:                                                      <L 1422>
                    var_58 = (var_55 >= var_57);
                    var_56 = var_58;
                    if (var_56) {
                        var_59 = (var_55 < var_35);
                        var_56 = var_56 && var_59;
                    }
                    if (var_56) {
                        // dz = -halo                                                             <L 1423>
                        var_60 = wp::neg(var_45);
                        // while dz <= halo:                                                      <L 1424>
        start_while_7:;
                        var_61 = (var_60 <= var_45);
        if ((var_61) == false) goto end_while_7;
                            // z = bz + dz                                                        <L 1425>
                            var_62 = wp::add(var_27, var_60);
                            // if z >= 0 and z < nz:                                              <L 1426>
                            var_65 = (var_62 >= var_64);
                            var_63 = var_65;
                            if (var_63) {
                                var_66 = (var_62 < var_39);
                                var_63 = var_63 && var_66;
                            }
                            if (var_63) {
                                // wake_mask[x, y, z] = 1                                         <L 1427>
                                wp::array_store(var_wake_mask, var_48, var_55, var_62, var_67);
                            }
                            // dz += 1                                                            <L 1428>
                            var_69 = wp::add(var_60, var_68);
                            wp::assign(var_60, var_69);
        goto start_while_7;
        end_while_7:;
                    }
                    // dy += 1                                                                    <L 1429>
                    var_71 = wp::add(var_53, var_70);
                    wp::assign(var_53, var_71);
        goto start_while_5;
        end_while_5:;
            }
            // dx += 1                                                                            <L 1430>
            var_73 = wp::add(var_46, var_72);
            wp::assign(var_46, var_73);
        goto start_while_3;
        end_while_3:;
    }
}



extern "C" __global__ void prolongate_shape_matching_corrections_uniform8_table_1edb0e67_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_init,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_coarse_q_snapshot,
    wp::array_t<wp::int32> var_parent_indices,
    wp::array_t<wp::float32> var_parent_weights,
    wp::array_t<wp::int32> var_child_cluster,
    wp::array_t<wp::int32> var_cluster_active,
    wp::int32 var_absolute_projection,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd_out)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::vec_t<3, wp::float32>* var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        const wp::int32 var_8 = 0;
        wp::int32 var_9;
        bool var_10;
        wp::int32* var_11;
        const wp::int32 var_12 = 1;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::float32* var_17;
        const wp::float32 var_18 = 0.0;
        bool var_19;
        wp::float32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        wp::int32* var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 8;
        wp::int32 var_32;
        const wp::float32 var_33 = 0.0;
        const wp::float32 var_34 = 0.0;
        const wp::float32 var_35 = 0.0;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        const wp::int32 var_39 = 8;
        wp::range_t var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32* var_43;
        wp::int32 var_44;
        wp::int32 var_45;
        bool var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32* var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        const wp::int32 var_55 = 0;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::float32* var_58;
        wp::float32 var_59;
        wp::float32 var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32>* var_68;
        wp::vec_t<3, wp::float32>* var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        const wp::int32 var_76 = 0;
        bool var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::int32 var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::int32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 0;
        bool var_93;
        wp::vec_t<3, wp::float32>* var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::vec_t<3, wp::float32> var_96;
        wp::vec_t<3, wp::float32> var_97;
        wp::float32 var_98;
        bool var_99;
        wp::float32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::vec_t<3, wp::float32> var_102;
        //---------
        // forward
        // def prolongate_shape_matching_corrections_uniform8_table(                              <L 1016>
        // particle_idx = wp.tid()                                                                <L 1033>
        var_0 = builtin_tid1d();
        // x_old = particle_q[particle_idx]                                                       <L 1034>
        var_1 = wp::address(var_particle_q, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // x_new = x_old                                                                          <L 1035>
        var_4 = wp::copy(var_2);
        // v_old = particle_qd[particle_idx]                                                      <L 1036>
        var_5 = wp::address(var_particle_qd, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // changed = int(0)                                                                       <L 1037>
        var_9 = wp::int(var_8);
        // if (                                                                                   <L 1039>
        // (particle_flags[particle_idx] & ParticleFlags.ACTIVE) != 0                             <L 1040>
        var_11 = wp::address(var_particle_flags, var_0);
        var_14 = wp::load(var_11);
        var_13 = wp::bit_and(var_14, var_12);
        var_16 = (var_13 != var_15);
        var_10 = var_16;
        if (var_10) {
            // and particle_inv_mass[particle_idx] > 0.0                                          <L 1041>
            var_17 = wp::address(var_particle_inv_mass, var_0);
            var_20 = wp::load(var_17);
            var_19 = (var_20 > var_18);
            var_10 = var_10 && var_19;
        }
        if (var_10) {
            // cluster_idx = child_cluster[particle_idx]                                          <L 1043>
            var_21 = wp::address(var_child_cluster, var_0);
            var_23 = wp::load(var_21);
            var_22 = wp::copy(var_23);
            // if cluster_idx >= 0 and cluster_active[cluster_idx] != 0:                          <L 1044>
            var_26 = (var_22 >= var_25);
            var_24 = var_26;
            if (var_24) {
                var_27 = wp::address(var_cluster_active, var_22);
                var_30 = wp::load(var_27);
                var_29 = (var_30 != var_28);
                var_24 = var_24 && var_29;
            }
            if (var_24) {
                // base = particle_idx * wp.static(UNIFORM_CLUSTER_SIZE)                          <L 1045>
                var_32 = wp::mul(var_0, var_31);
                // interpolated = wp.vec3(0.0, 0.0, 0.0)                                          <L 1046>
                var_36 = wp::vec_t<3, wp::float32>(var_33, var_34, var_35);
                // valid = int(1)                                                                 <L 1047>
                var_38 = wp::int(var_37);
                // for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):                     <L 1048>
                var_40 = wp::range(var_39);
                start_for_0:;
                    if (iter_cmp(var_40) == 0) goto end_for_0;
                    var_41 = wp::iter_next(var_40);
                    // parent_idx = parent_indices[base + parent_slot]                            <L 1049>
                    var_42 = wp::add(var_32, var_41);
                    var_43 = wp::address(var_parent_indices, var_42);
                    var_45 = wp::load(var_43);
                    var_44 = wp::copy(var_45);
                    // if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:       <L 1050>
                    var_48 = (var_44 < var_47);
                    var_46 = var_48;
                    if (!var_46) {
                        var_49 = wp::address(var_particle_flags, var_44);
                        var_52 = wp::load(var_49);
                        var_51 = wp::bit_and(var_52, var_50);
                        var_54 = (var_51 == var_53);
                        var_46 = var_46 || var_54;
                    }
                    if (var_46) {
                        // valid = int(0)                                                         <L 1051>
                        var_56 = wp::int(var_55);
                        // break                                                                  <L 1052>
                        wp::assign(var_38, var_56);
                        goto end_for_0;
                    }
                    // weight = parent_weights[base + parent_slot]                                <L 1053>
                    var_57 = wp::add(var_32, var_41);
                    var_58 = wp::address(var_parent_weights, var_57);
                    var_60 = wp::load(var_58);
                    var_59 = wp::copy(var_60);
                    // if absolute_projection != 0:                                               <L 1054>
                    var_62 = (var_absolute_projection != var_61);
                    if (var_62) {
                        // interpolated += particle_q[parent_idx] * weight                        <L 1055>
                        var_63 = wp::address(var_particle_q, var_44);
                        var_65 = wp::load(var_63);
                        var_64 = wp::mul(var_65, var_59);
                        var_66 = wp::add(var_36, var_64);
                    }
                    var_67 = wp::where(var_62, var_66, var_36);
                    if (!var_62) {
                        // interpolated += (particle_q[parent_idx] - coarse_q_snapshot[parent_idx]) * weight       <L 1057>
                        var_68 = wp::address(var_particle_q, var_44);
                        var_69 = wp::address(var_coarse_q_snapshot, var_44);
                        var_71 = wp::load(var_68);
                        var_72 = wp::load(var_69);
                        var_70 = wp::sub(var_71, var_72);
                        var_73 = wp::mul(var_70, var_59);
                        var_74 = wp::add(var_67, var_73);
                    }
                    var_75 = wp::where(var_62, var_67, var_74);
                    wp::assign(var_36, var_75);
                    goto start_for_0;
                end_for_0:;
                // if valid != 0:                                                                 <L 1058>
                var_77 = (var_38 != var_76);
                if (var_77) {
                    // if absolute_projection != 0:                                               <L 1059>
                    var_79 = (var_absolute_projection != var_78);
                    if (var_79) {
                        // x_new = interpolated                                                   <L 1060>
                        var_80 = wp::copy(var_36);
                    }
                    var_81 = wp::where(var_79, var_80, var_4);
                    if (!var_79) {
                        // x_new = x_old + interpolated                                           <L 1062>
                        var_82 = wp::add(var_2, var_36);
                    }
                    var_83 = wp::where(var_79, var_81, var_82);
                    // changed = int(1)                                                           <L 1063>
                    var_85 = wp::int(var_84);
                }
                var_86 = wp::where(var_77, var_83, var_4);
                var_87 = wp::where(var_77, var_85, var_9);
            }
            var_88 = wp::where(var_24, var_86, var_4);
            var_89 = wp::where(var_24, var_87, var_9);
        }
        var_90 = wp::where(var_10, var_88, var_4);
        var_91 = wp::where(var_10, var_89, var_9);
        // if changed == 0:                                                                       <L 1065>
        var_93 = (var_91 == var_92);
        if (var_93) {
            // particle_q_out[particle_idx] = x_old                                               <L 1066>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1067>
            wp::array_store(var_particle_qd_out, var_0, var_6);
            // return                                                                             <L 1068>
            continue;
        }
        // v_new = (x_new - particle_q_init[particle_idx]) / dt                                   <L 1070>
        var_94 = wp::address(var_particle_q_init, var_0);
        var_96 = wp::load(var_94);
        var_95 = wp::sub(var_90, var_96);
        var_97 = wp::div(var_95, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 1071>
        var_98 = wp::length(var_97);
        // if v_new_mag > v_max:                                                                  <L 1072>
        var_99 = (var_98 > var_v_max);
        if (var_99) {
            // v_new *= v_max / v_new_mag                                                         <L 1073>
            var_100 = wp::div(var_v_max, var_98);
            var_101 = wp::mul(var_97, var_100);
        }
        var_102 = wp::where(var_99, var_101, var_97);
        // particle_q_out[particle_idx] = x_new                                                   <L 1075>
        wp::array_store(var_particle_q_out, var_0, var_90);
        // particle_qd_out[particle_idx] = v_new                                                  <L 1076>
        wp::array_store(var_particle_qd_out, var_0, var_102);
    }
}



extern "C" __global__ void project_shape_matching_children_uniform8_table_bee52543_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_init,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_awake_l0_cluster_counts,
    wp::array_t<wp::int32> var_parent_indices,
    wp::array_t<wp::float32> var_parent_weights,
    wp::array_t<wp::int32> var_child_cluster,
    wp::array_t<wp::int32> var_projection_active,
    wp::float32 var_projection_stiffness,
    wp::float32 var_dt,
    wp::float32 var_v_max,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q_out,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_qd_out)
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
        wp::vec_t<3, wp::float32>* var_1;
        wp::vec_t<3, wp::float32> var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::float32 var_7;
        const wp::float32 var_8 = 0.0;
        bool var_9;
        const wp::float32 var_10 = 1.0;
        bool var_11;
        const wp::float32 var_12 = 1.0;
        wp::float32 var_13;
        bool var_14;
        wp::int32* var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::float32* var_21;
        const wp::float32 var_22 = 0.0;
        bool var_23;
        wp::float32 var_24;
        wp::int32* var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        bool var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        const wp::int32 var_36 = 0;
        bool var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 8;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        const wp::float32 var_43 = 0.0;
        wp::vec_t<3, wp::float32> var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        const wp::int32 var_47 = 8;
        wp::range_t var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        wp::int32* var_51;
        wp::int32 var_52;
        wp::int32 var_53;
        bool var_54;
        const wp::int32 var_55 = 0;
        bool var_56;
        wp::int32* var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        const wp::int32 var_63 = 0;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32>* var_65;
        wp::int32 var_66;
        wp::float32* var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        wp::vec_t<3, wp::float32> var_74;
        wp::vec_t<3, wp::float32> var_75;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32>* var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::float32 var_81;
        bool var_82;
        wp::float32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::vec_t<3, wp::float32> var_85;
        //---------
        // forward
        // def project_shape_matching_children_uniform8_table(                                    <L 1194>
        // particle_idx = wp.tid()                                                                <L 1211>
        var_0 = builtin_tid1d();
        // x_old = particle_q[particle_idx]                                                       <L 1212>
        var_1 = wp::address(var_particle_q, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // v_old = particle_qd[particle_idx]                                                      <L 1213>
        var_4 = wp::address(var_particle_qd, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // alpha = projection_stiffness                                                           <L 1215>
        var_7 = wp::copy(var_projection_stiffness);
        // if alpha <= 0.0:                                                                       <L 1216>
        var_9 = (var_7 <= var_8);
        if (var_9) {
            // particle_q_out[particle_idx] = x_old                                               <L 1217>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1218>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1219>
            continue;
        }
        // if alpha > 1.0:                                                                        <L 1220>
        var_11 = (var_7 > var_10);
        if (var_11) {
            // alpha = 1.0                                                                        <L 1221>
        }
        var_13 = wp::where(var_11, var_12, var_7);
        // if (                                                                                   <L 1223>
        // (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0                             <L 1224>
        var_15 = wp::address(var_particle_flags, var_0);
        var_18 = wp::load(var_15);
        var_17 = wp::bit_and(var_18, var_16);
        var_20 = (var_17 == var_19);
        var_14 = var_20;
        if (!var_14) {
            // or particle_inv_mass[particle_idx] <= 0.0                                          <L 1225>
            var_21 = wp::address(var_particle_inv_mass, var_0);
            var_24 = wp::load(var_21);
            var_23 = (var_24 <= var_22);
            var_14 = var_14 || var_23;
        }
        if (!var_14) {
            // or awake_l0_cluster_counts[particle_idx] > 0                                       <L 1226>
            var_25 = wp::address(var_awake_l0_cluster_counts, var_0);
            var_28 = wp::load(var_25);
            var_27 = (var_28 > var_26);
            var_14 = var_14 || var_27;
        }
        if (var_14) {
            // particle_q_out[particle_idx] = x_old                                               <L 1228>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1229>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1230>
            continue;
        }
        // cluster_idx = child_cluster[particle_idx]                                              <L 1232>
        var_29 = wp::address(var_child_cluster, var_0);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // if cluster_idx < 0 or projection_active[cluster_idx] == 0:                             <L 1233>
        var_34 = (var_30 < var_33);
        var_32 = var_34;
        if (!var_32) {
            var_35 = wp::address(var_projection_active, var_30);
            var_38 = wp::load(var_35);
            var_37 = (var_38 == var_36);
            var_32 = var_32 || var_37;
        }
        if (var_32) {
            // particle_q_out[particle_idx] = x_old                                               <L 1234>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1235>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1236>
            continue;
        }
        // base = particle_idx * wp.static(UNIFORM_CLUSTER_SIZE)                                  <L 1238>
        var_40 = wp::mul(var_0, var_39);
        // x_target = wp.vec3(0.0, 0.0, 0.0)                                                      <L 1239>
        var_44 = wp::vec_t<3, wp::float32>(var_41, var_42, var_43);
        // valid = int(1)                                                                         <L 1240>
        var_46 = wp::int(var_45);
        // for parent_slot in range(wp.static(UNIFORM_CLUSTER_SIZE)):                             <L 1241>
        var_48 = wp::range(var_47);
        start_for_3:;
            if (iter_cmp(var_48) == 0) goto end_for_3;
            var_49 = wp::iter_next(var_48);
            // parent_idx = parent_indices[base + parent_slot]                                    <L 1242>
            var_50 = wp::add(var_40, var_49);
            var_51 = wp::address(var_parent_indices, var_50);
            var_53 = wp::load(var_51);
            var_52 = wp::copy(var_53);
            // if parent_idx < 0 or (particle_flags[parent_idx] & ParticleFlags.ACTIVE) == 0:       <L 1243>
            var_56 = (var_52 < var_55);
            var_54 = var_56;
            if (!var_54) {
                var_57 = wp::address(var_particle_flags, var_52);
                var_60 = wp::load(var_57);
                var_59 = wp::bit_and(var_60, var_58);
                var_62 = (var_59 == var_61);
                var_54 = var_54 || var_62;
            }
            if (var_54) {
                // valid = int(0)                                                                 <L 1244>
                var_64 = wp::int(var_63);
                // break                                                                          <L 1245>
                wp::assign(var_46, var_64);
                goto end_for_3;
            }
            // x_target += particle_q[parent_idx] * parent_weights[base + parent_slot]            <L 1246>
            var_65 = wp::address(var_particle_q, var_52);
            var_66 = wp::add(var_40, var_49);
            var_67 = wp::address(var_parent_weights, var_66);
            var_69 = wp::load(var_65);
            var_70 = wp::load(var_67);
            var_68 = wp::mul(var_69, var_70);
            var_71 = wp::add(var_44, var_68);
            wp::assign(var_44, var_71);
            goto start_for_3;
        end_for_3:;
        // if valid == 0:                                                                         <L 1248>
        var_73 = (var_46 == var_72);
        if (var_73) {
            // particle_q_out[particle_idx] = x_old                                               <L 1249>
            wp::array_store(var_particle_q_out, var_0, var_2);
            // particle_qd_out[particle_idx] = v_old                                              <L 1250>
            wp::array_store(var_particle_qd_out, var_0, var_5);
            // return                                                                             <L 1251>
            continue;
        }
        // x_new = x_old + (x_target - x_old) * alpha                                             <L 1253>
        var_74 = wp::sub(var_44, var_2);
        var_75 = wp::mul(var_74, var_13);
        var_76 = wp::add(var_2, var_75);
        // v_new = (x_new - particle_q_init[particle_idx]) / dt                                   <L 1254>
        var_77 = wp::address(var_particle_q_init, var_0);
        var_79 = wp::load(var_77);
        var_78 = wp::sub(var_76, var_79);
        var_80 = wp::div(var_78, var_dt);
        // v_new_mag = wp.length(v_new)                                                           <L 1255>
        var_81 = wp::length(var_80);
        // if v_new_mag > v_max:                                                                  <L 1256>
        var_82 = (var_81 > var_v_max);
        if (var_82) {
            // v_new *= v_max / v_new_mag                                                         <L 1257>
            var_83 = wp::div(var_v_max, var_81);
            var_84 = wp::mul(var_80, var_83);
        }
        var_85 = wp::where(var_82, var_84, var_80);
        // particle_q_out[particle_idx] = x_new                                                   <L 1259>
        wp::array_store(var_particle_q_out, var_0, var_76);
        // particle_qd_out[particle_idx] = v_new                                                  <L 1260>
        wp::array_store(var_particle_qd_out, var_0, var_85);
    }
}



extern "C" __global__ void mask_l1_projection_covered_by_l2_kernel_b02a0d30_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_l1_source_cell,
    wp::array_t<wp::int32> var_l2_cell_to_cluster,
    wp::array_t<wp::int32> var_l2_projection_active,
    wp::array_t<wp::int32> var_l1_projection_active,
    wp::array_t<wp::int32> var_l1_projection_count)
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
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        bool var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 0;
        const wp::int32 var_19 = 0;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        //---------
        // forward
        // def mask_l1_projection_covered_by_l2_kernel(                                           <L 1466>
        // l1_idx = wp.tid()                                                                      <L 1473>
        var_0 = builtin_tid1d();
        // if l1_projection_active[l1_idx] == 0:                                                  <L 1474>
        var_1 = wp::address(var_l1_projection_active, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 1475>
            continue;
        }
        // source_cell = l1_source_cell[l1_idx]                                                   <L 1477>
        var_5 = wp::address(var_l1_source_cell, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // parent_l2 = l2_cell_to_cluster[source_cell]                                            <L 1478>
        var_8 = wp::address(var_l2_cell_to_cluster, var_6);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if parent_l2 >= 0 and l2_projection_active[parent_l2] != 0:                            <L 1479>
        var_13 = (var_9 >= var_12);
        var_11 = var_13;
        if (var_11) {
            var_14 = wp::address(var_l2_projection_active, var_9);
            var_17 = wp::load(var_14);
            var_16 = (var_17 != var_15);
            var_11 = var_11 && var_16;
        }
        if (var_11) {
            // l1_projection_active[l1_idx] = 0                                                   <L 1480>
            wp::array_store(var_l1_projection_active, var_0, var_18);
            // wp.atomic_sub(l1_projection_count, 0, 1)                                           <L 1481>
            var_21 = wp::atomic_sub(var_l1_projection_count, var_19, var_20);
        }
    }
}



extern "C" __global__ void solve_shape_matching_clusters_uniform27_c0cd2af0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_positions_by_slot,
    wp::array_t<wp::vec_t<3, wp::float32>> var_rest_local_template,
    wp::array_t<wp::float32> var_coefficients,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::int32 var_cluster_count,
    wp::int32 var_use_rest_local_template,
    wp::float32 var_stiffness,
    wp::int32 var_rotation_iterations,
    wp::array_t<wp::quat_t<wp::float32>> var_cluster_rotations,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cluster_translations,
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
        bool var_1;
        wp::int32* var_2;
        const wp::int32 var_3 = 0;
        bool var_4;
        wp::int32 var_5;
        const wp::float32 var_6 = 0.0;
        bool var_7;
        wp::float32* var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::float32 var_11 = 0.0;
        bool var_12;
        const wp::float32 var_13 = 0.0;
        const wp::float32 var_14 = 0.0;
        const wp::float32 var_15 = 0.0;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 27;
        wp::range_t var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32* var_29;
        const wp::int32 var_30 = 1;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::int32 var_38 = 1;
        wp::int32 var_39;
        wp::float32* var_40;
        const wp::float32 var_41 = 0.0;
        bool var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::int32 var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::float32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        const wp::float32 var_51 = 0.0;
        wp::mat_t<3, 3, wp::float32> var_52;
        const wp::int32 var_53 = 27;
        wp::range_t var_54;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32* var_58;
        wp::int32 var_59;
        wp::int32 var_60;
        wp::int32* var_61;
        const wp::int32 var_62 = 1;
        wp::int32 var_63;
        wp::int32 var_64;
        const wp::int32 var_65 = 0;
        bool var_66;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32>* var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32>* var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        const wp::int32 var_74 = 0;
        bool var_75;
        wp::int32 var_76;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::mat_t<3, 3, wp::float32> var_82;
        wp::mat_t<3, 3, wp::float32> var_83;
        wp::quat_t<wp::float32>* var_84;
        wp::quat_t<wp::float32> var_85;
        wp::quat_t<wp::float32> var_86;
        wp::quat_t<wp::float32> var_87;
        wp::float32 var_88;
        const wp::float32 var_89 = 0.0;
        bool var_90;
        const wp::int32 var_91 = 0;
        wp::float32 var_92;
        wp::float32 var_93;
        const wp::int32 var_94 = 1;
        wp::float32 var_95;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        wp::float32 var_98;
        wp::float32 var_99;
        const wp::int32 var_100 = 3;
        wp::float32 var_101;
        wp::float32 var_102;
        wp::quat_t<wp::float32> var_103;
        wp::quat_t<wp::float32> var_104;
        const wp::int32 var_105 = 0;
        bool var_106;
        const wp::int32 var_107 = 27;
        wp::range_t var_108;
        wp::int32 var_109;
        wp::int32 var_110;
        wp::int32 var_111;
        wp::int32* var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        bool var_115;
        wp::int32* var_116;
        const wp::int32 var_117 = 1;
        wp::int32 var_118;
        wp::int32 var_119;
        const wp::int32 var_120 = 0;
        bool var_121;
        wp::float32* var_122;
        const wp::float32 var_123 = 0.0;
        bool var_124;
        wp::float32 var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32>* var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::vec_t<3, wp::float32> var_129;
        const wp::int32 var_130 = 0;
        bool var_131;
        wp::int32 var_132;
        wp::int32 var_133;
        wp::vec_t<3, wp::float32>* var_134;
        wp::vec_t<3, wp::float32> var_135;
        wp::vec_t<3, wp::float32> var_136;
        wp::vec_t<3, wp::float32> var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        wp::float32 var_140;
        wp::float32* var_141;
        wp::float32 var_142;
        wp::float32 var_143;
        const wp::float32 var_144 = 0.0;
        bool var_145;
        wp::vec_t<3, wp::float32>* var_146;
        wp::vec_t<3, wp::float32> var_147;
        wp::vec_t<3, wp::float32> var_148;
        wp::vec_t<3, wp::float32> var_149;
        wp::vec_t<3, wp::float32> var_150;
        //---------
        // forward
        // def solve_shape_matching_clusters_uniform27(                                           <L 659>
        // cluster_idx = wp.tid()                                                                 <L 677>
        var_0 = builtin_tid1d();
        // if cluster_active[cluster_idx] == 0 or stiffness <= 0.0:                               <L 679>
        var_2 = wp::address(var_cluster_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        var_1 = var_4;
        if (!var_1) {
            var_7 = (var_stiffness <= var_6);
            var_1 = var_1 || var_7;
        }
        if (var_1) {
            // return                                                                             <L 680>
            continue;
        }
        // coeff = coefficients[cluster_idx]                                                      <L 682>
        var_8 = wp::address(var_coefficients, var_0);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // if coeff <= 0.0:                                                                       <L 683>
        var_12 = (var_9 <= var_11);
        if (var_12) {
            // return                                                                             <L 684>
            continue;
        }
        // center = wp.vec3(0.0, 0.0, 0.0)                                                        <L 686>
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // member_count = int(0)                                                                  <L 687>
        var_18 = wp::int(var_17);
        // dynamic_count = int(0)                                                                 <L 688>
        var_20 = wp::int(var_19);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):                            <L 690>
        var_22 = wp::range(var_21);
        start_for_2:;
            if (iter_cmp(var_22) == 0) goto end_for_2;
            var_23 = wp::iter_next(var_22);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 691>
            var_24 = wp::mul(var_23, var_cluster_count);
            var_25 = wp::add(var_24, var_0);
            var_26 = wp::address(var_indices_by_slot, var_25);
            var_28 = wp::load(var_26);
            var_27 = wp::copy(var_28);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 692>
            var_29 = wp::address(var_particle_flags, var_27);
            var_32 = wp::load(var_29);
            var_31 = wp::bit_and(var_32, var_30);
            var_34 = (var_31 == var_33);
            if (var_34) {
                // continue                                                                       <L 693>
                goto start_for_2;
            }
            // center += particle_q[particle_idx]                                                 <L 694>
            var_35 = wp::address(var_particle_q, var_27);
            var_37 = wp::load(var_35);
            var_36 = wp::add(var_16, var_37);
            // member_count += 1                                                                  <L 695>
            var_39 = wp::add(var_18, var_38);
            // if particle_inv_mass[particle_idx] > 0.0:                                          <L 696>
            var_40 = wp::address(var_particle_inv_mass, var_27);
            var_43 = wp::load(var_40);
            var_42 = (var_43 > var_41);
            if (var_42) {
                // dynamic_count += 1                                                             <L 697>
                var_45 = wp::add(var_20, var_44);
            }
            var_46 = wp::where(var_42, var_45, var_20);
            wp::assign(var_16, var_36);
            wp::assign(var_18, var_39);
            wp::assign(var_20, var_46);
            goto start_for_2;
        end_for_2:;
        // if member_count == 0:                                                                  <L 699>
        var_48 = (var_18 == var_47);
        if (var_48) {
            // return                                                                             <L 700>
            continue;
        }
        // center /= float(member_count)                                                          <L 702>
        var_49 = wp::float(var_18);
        var_50 = wp::div(var_16, var_49);
        // covariance = wp.mat33(0.0)                                                             <L 704>
        var_52 = wp::mat_t<3, 3, wp::float32>(var_51);
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):                            <L 705>
        var_54 = wp::range(var_53);
        start_for_5:;
            if (iter_cmp(var_54) == 0) goto end_for_5;
            var_55 = wp::iter_next(var_54);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 706>
            var_56 = wp::mul(var_55, var_cluster_count);
            var_57 = wp::add(var_56, var_0);
            var_58 = wp::address(var_indices_by_slot, var_57);
            var_60 = wp::load(var_58);
            var_59 = wp::copy(var_60);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0:                     <L 707>
            var_61 = wp::address(var_particle_flags, var_59);
            var_64 = wp::load(var_61);
            var_63 = wp::bit_and(var_64, var_62);
            var_66 = (var_63 == var_65);
            if (var_66) {
                // continue                                                                       <L 708>
                wp::assign(var_27, var_59);
                goto start_for_5;
            }
            var_67 = wp::where(var_66, var_27, var_59);
            // x_rel = particle_q[particle_idx] - center                                          <L 709>
            var_68 = wp::address(var_particle_q, var_67);
            var_70 = wp::load(var_68);
            var_69 = wp::sub(var_70, var_50);
            // q_rel = rest_local_template[local_idx]                                             <L 710>
            var_71 = wp::address(var_rest_local_template, var_55);
            var_73 = wp::load(var_71);
            var_72 = wp::copy(var_73);
            // if use_rest_local_template == 0:                                                   <L 711>
            var_75 = (var_use_rest_local_template == var_74);
            if (var_75) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 712>
                var_76 = wp::mul(var_55, var_cluster_count);
                var_77 = wp::add(var_76, var_0);
                var_78 = wp::address(var_rest_local_positions_by_slot, var_77);
                var_80 = wp::load(var_78);
                var_79 = wp::copy(var_80);
            }
            var_81 = wp::where(var_75, var_79, var_72);
            // covariance += wp.outer(q_rel, x_rel)                                               <L 713>
            var_82 = wp::outer(var_81, var_69);
            var_83 = wp::add(var_52, var_82);
            wp::assign(var_27, var_67);
            wp::assign(var_52, var_83);
            goto start_for_5;
        end_for_5:;
        // prev_rotation = cluster_rotations[cluster_idx]                                         <L 715>
        var_84 = wp::address(var_cluster_rotations, var_0);
        var_86 = wp::load(var_84);
        var_85 = wp::copy(var_86);
        // rotation = _extract_rotation(covariance, prev_rotation, rotation_iterations)           <L 716>
        var_87 = _extract_rotation_0(var_52, var_85, var_rotation_iterations);
        // if _quat_dot(rotation, prev_rotation) < 0.0:                                           <L 717>
        var_88 = _quat_dot_0(var_87, var_85);
        var_90 = (var_88 < var_89);
        if (var_90) {
            // rotation = wp.quat(-rotation[0], -rotation[1], -rotation[2], -rotation[3])         <L 718>
            var_92 = wp::extract(var_87, var_91);
            var_93 = wp::neg(var_92);
            var_95 = wp::extract(var_87, var_94);
            var_96 = wp::neg(var_95);
            var_98 = wp::extract(var_87, var_97);
            var_99 = wp::neg(var_98);
            var_101 = wp::extract(var_87, var_100);
            var_102 = wp::neg(var_101);
            var_103 = wp::quat_t<wp::float32>(var_93, var_96, var_99, var_102);
        }
        var_104 = wp::where(var_90, var_103, var_87);
        // cluster_rotations[cluster_idx] = rotation                                              <L 720>
        wp::array_store(var_cluster_rotations, var_0, var_104);
        // cluster_translations[cluster_idx] = center                                             <L 721>
        wp::array_store(var_cluster_translations, var_0, var_50);
        // if dynamic_count == 0:                                                                 <L 723>
        var_106 = (var_20 == var_105);
        if (var_106) {
            // return                                                                             <L 724>
            continue;
        }
        // for local_idx in range(wp.static(UNIFORM_CLUSTER_SIZE_27)):                            <L 726>
        var_108 = wp::range(var_107);
        start_for_8:;
            if (iter_cmp(var_108) == 0) goto end_for_8;
            var_109 = wp::iter_next(var_108);
            // particle_idx = indices_by_slot[local_idx * cluster_count + cluster_idx]            <L 727>
            var_110 = wp::mul(var_109, var_cluster_count);
            var_111 = wp::add(var_110, var_0);
            var_112 = wp::address(var_indices_by_slot, var_111);
            var_114 = wp::load(var_112);
            var_113 = wp::copy(var_114);
            // if (particle_flags[particle_idx] & ParticleFlags.ACTIVE) == 0 or particle_inv_mass[particle_idx] <= 0.0:       <L 728>
            var_116 = wp::address(var_particle_flags, var_113);
            var_119 = wp::load(var_116);
            var_118 = wp::bit_and(var_119, var_117);
            var_121 = (var_118 == var_120);
            var_115 = var_121;
            if (!var_115) {
                var_122 = wp::address(var_particle_inv_mass, var_113);
                var_125 = wp::load(var_122);
                var_124 = (var_125 <= var_123);
                var_115 = var_115 || var_124;
            }
            if (var_115) {
                // continue                                                                       <L 729>
                wp::assign(var_27, var_113);
                goto start_for_8;
            }
            var_126 = wp::where(var_115, var_27, var_113);
            // q_rel = rest_local_template[local_idx]                                             <L 731>
            var_127 = wp::address(var_rest_local_template, var_109);
            var_129 = wp::load(var_127);
            var_128 = wp::copy(var_129);
            // if use_rest_local_template == 0:                                                   <L 732>
            var_131 = (var_use_rest_local_template == var_130);
            if (var_131) {
                // q_rel = rest_local_positions_by_slot[local_idx * cluster_count + cluster_idx]       <L 733>
                var_132 = wp::mul(var_109, var_cluster_count);
                var_133 = wp::add(var_132, var_0);
                var_134 = wp::address(var_rest_local_positions_by_slot, var_133);
                var_136 = wp::load(var_134);
                var_135 = wp::copy(var_136);
            }
            var_137 = wp::where(var_131, var_135, var_128);
            // goal = center + wp.quat_rotate(rotation, q_rel)                                    <L 734>
            var_138 = wp::quat_rotate(var_104, var_137);
            var_139 = wp::add(var_50, var_138);
            // particle_scale = coeff * stiffness * particle_cluster_inv_weights[particle_idx]       <L 735>
            var_140 = wp::mul(var_9, var_stiffness);
            var_141 = wp::address(var_particle_cluster_inv_weights, var_126);
            var_143 = wp::load(var_141);
            var_142 = wp::mul(var_140, var_143);
            // if particle_scale > 0.0:                                                           <L 736>
            var_145 = (var_142 > var_144);
            if (var_145) {
                // wp.atomic_add(particle_deltas, particle_idx, (goal - particle_q[particle_idx]) * particle_scale)       <L 737>
                var_146 = wp::address(var_particle_q, var_126);
                var_148 = wp::load(var_146);
                var_147 = wp::sub(var_139, var_148);
                var_149 = wp::mul(var_147, var_142);
                var_150 = wp::atomic_add(var_particle_deltas, var_126, var_149);
            }
            wp::assign(var_27, var_126);
            wp::assign(var_81, var_137);
            goto start_for_8;
        end_for_8:;
    }
}



extern "C" __global__ void finalize_runtime_cluster_inv_weights_kernel_e7131549_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights)
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
        wp::int32 var_2;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        bool var_5;
        const wp::float32 var_6 = 1.0;
        wp::float32 var_7;
        wp::float32 var_8;
        const wp::int32 var_9 = 0;
        const wp::float32 var_10 = 0.0;
        //---------
        // forward
        // def finalize_runtime_cluster_inv_weights_kernel(                                       <L 1541>
        // particle_idx = wp.tid()                                                                <L 1545>
        var_0 = builtin_tid1d();
        // count = particle_cluster_counts[particle_idx]                                          <L 1546>
        var_1 = wp::address(var_particle_cluster_counts, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if count > 0:                                                                          <L 1547>
        var_5 = (var_2 > var_4);
        if (var_5) {
            // particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)                    <L 1548>
            var_7 = wp::float(var_2);
            var_8 = wp::div(var_6, var_7);
            wp::array_store(var_particle_cluster_inv_weights, var_0, var_8);
        }
        if (!var_5) {
            // particle_cluster_counts[particle_idx] = 0                                          <L 1550>
            wp::array_store(var_particle_cluster_counts, var_0, var_9);
            // particle_cluster_inv_weights[particle_idx] = 0.0                                   <L 1551>
            wp::array_store(var_particle_cluster_inv_weights, var_0, var_10);
        }
    }
}

