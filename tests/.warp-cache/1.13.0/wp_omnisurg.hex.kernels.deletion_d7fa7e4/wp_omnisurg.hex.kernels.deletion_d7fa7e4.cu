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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/deletion.py:364
static CUDA_CALLABLE wp::float32 _point_segment_distance_sq_0(
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
    // def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3):                    <L 365>
    // ab = b - a                                                                             <L 366>
    var_0 = wp::sub(var_b, var_a);
    // denom = wp.dot(ab, ab)                                                                 <L 367>
    var_1 = wp::dot(var_0, var_0);
    // if denom <= 1.0e-20:                                                                   <L 368>
    var_3 = (var_1 <= var_2);
    if (var_3) {
        // d = p - a                                                                          <L 369>
        var_4 = wp::sub(var_p, var_a);
        // return wp.dot(d, d)                                                                <L 370>
        var_5 = wp::dot(var_4, var_4);
        return var_5;
    }
    // t = wp.dot(p - a, ab) / denom                                                          <L 372>
    var_6 = wp::sub(var_p, var_a);
    var_7 = wp::dot(var_6, var_0);
    var_8 = wp::div(var_7, var_1);
    // if t < 0.0:                                                                            <L 373>
    var_10 = (var_8 < var_9);
    if (var_10) {
        // t = 0.0                                                                            <L 374>
    }
    var_12 = wp::where(var_10, var_11, var_8);
    if (!var_10) {
        // elif t > 1.0:                                                                      <L 375>
        var_14 = (var_12 > var_13);
        if (var_14) {
            // t = 1.0                                                                        <L 376>
        }
        var_16 = wp::where(var_14, var_15, var_12);
    }
    var_17 = wp::where(var_10, var_12, var_16);
    // q = a + ab * t                                                                         <L 377>
    var_18 = wp::mul(var_0, var_17);
    var_19 = wp::add(var_a, var_18);
    // d = p - q                                                                              <L 378>
    var_20 = wp::sub(var_p, var_19);
    // return wp.dot(d, d)                                                                    <L 379>
    var_21 = wp::dot(var_20, var_20);
    return var_21;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/deletion.py:382
static CUDA_CALLABLE wp::float32 _point_triangle_distance_sq_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-20;
    bool var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    bool var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    bool var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    bool var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    bool var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    bool var_38;
    const wp::float32 var_39 = 0.0;
    bool var_40;
    const wp::float32 var_41 = 0.0;
    bool var_42;
    const wp::float32 var_43 = 0.0;
    bool var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    bool var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    bool var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    bool var_63;
    const wp::float32 var_64 = 0.0;
    bool var_65;
    const wp::float32 var_66 = 0.0;
    bool var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    bool var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.0;
    bool var_85;
    wp::float32 var_86;
    const wp::float32 var_87 = 0.0;
    bool var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::float32 var_100;
    const wp::float32 var_101 = 1.0;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::float32 var_112;
    //---------
    // forward
    // def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):       <L 383>
    // ab = b - a                                                                             <L 384>
    var_0 = wp::sub(var_b, var_a);
    // ac = c - a                                                                             <L 385>
    var_1 = wp::sub(var_c, var_a);
    // n = wp.cross(ab, ac)                                                                   <L 386>
    var_2 = wp::cross(var_0, var_1);
    // if wp.dot(n, n) <= 1.0e-20:                                                            <L 387>
    var_3 = wp::dot(var_2, var_2);
    var_5 = (var_3 <= var_4);
    if (var_5) {
        // d0 = _point_segment_distance_sq(p, a, b)                                           <L 388>
        var_6 = _point_segment_distance_sq_0(var_p, var_a, var_b);
        // d1 = _point_segment_distance_sq(p, b, c)                                           <L 389>
        var_7 = _point_segment_distance_sq_0(var_p, var_b, var_c);
        // d2 = _point_segment_distance_sq(p, c, a)                                           <L 390>
        var_8 = _point_segment_distance_sq_0(var_p, var_c, var_a);
        // best_dist = d0                                                                     <L 391>
        var_9 = wp::copy(var_6);
        // if d1 < best_dist:                                                                 <L 392>
        var_10 = (var_7 < var_9);
        if (var_10) {
            // best_dist = d1                                                                 <L 393>
            var_11 = wp::copy(var_7);
        }
        var_12 = wp::where(var_10, var_11, var_9);
        // if d2 < best_dist:                                                                 <L 394>
        var_13 = (var_8 < var_12);
        if (var_13) {
            // best_dist = d2                                                                 <L 395>
            var_14 = wp::copy(var_8);
        }
        var_15 = wp::where(var_13, var_14, var_12);
        // return best_dist                                                                   <L 396>
        return var_15;
    }
    // ap = p - a                                                                             <L 398>
    var_16 = wp::sub(var_p, var_a);
    // d1 = wp.dot(ab, ap)                                                                    <L 399>
    var_17 = wp::dot(var_0, var_16);
    // d2 = wp.dot(ac, ap)                                                                    <L 400>
    var_18 = wp::dot(var_1, var_16);
    // if d1 <= 0.0 and d2 <= 0.0:                                                            <L 401>
    var_21 = (var_17 <= var_20);
    var_19 = var_21;
    if (var_19) {
        var_23 = (var_18 <= var_22);
        var_19 = var_19 && var_23;
    }
    if (var_19) {
        // delta_a = p - a                                                                    <L 402>
        var_24 = wp::sub(var_p, var_a);
        // return wp.dot(delta_a, delta_a)                                                    <L 403>
        var_25 = wp::dot(var_24, var_24);
        return var_25;
    }
    // bp = p - b                                                                             <L 405>
    var_26 = wp::sub(var_p, var_b);
    // d3 = wp.dot(ab, bp)                                                                    <L 406>
    var_27 = wp::dot(var_0, var_26);
    // d4 = wp.dot(ac, bp)                                                                    <L 407>
    var_28 = wp::dot(var_1, var_26);
    // if d3 >= 0.0 and d4 <= d3:                                                             <L 408>
    var_31 = (var_27 >= var_30);
    var_29 = var_31;
    if (var_29) {
        var_32 = (var_28 <= var_27);
        var_29 = var_29 && var_32;
    }
    if (var_29) {
        // delta_b = p - b                                                                    <L 409>
        var_33 = wp::sub(var_p, var_b);
        // return wp.dot(delta_b, delta_b)                                                    <L 410>
        var_34 = wp::dot(var_33, var_33);
        return var_34;
    }
    // vc = d1 * d4 - d3 * d2                                                                 <L 412>
    var_35 = wp::mul(var_17, var_28);
    var_36 = wp::mul(var_27, var_18);
    var_37 = wp::sub(var_35, var_36);
    // if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                              <L 413>
    var_40 = (var_37 <= var_39);
    var_38 = var_40;
    if (var_38) {
        var_42 = (var_17 >= var_41);
        var_38 = var_38 && var_42;
    }
    if (var_38) {
        var_44 = (var_27 <= var_43);
        var_38 = var_38 && var_44;
    }
    if (var_38) {
        // v = d1 / (d1 - d3)                                                                 <L 414>
        var_45 = wp::sub(var_17, var_27);
        var_46 = wp::div(var_17, var_45);
        // q = a + ab * v                                                                     <L 415>
        var_47 = wp::mul(var_0, var_46);
        var_48 = wp::add(var_a, var_47);
        // delta_ab = p - q                                                                   <L 416>
        var_49 = wp::sub(var_p, var_48);
        // return wp.dot(delta_ab, delta_ab)                                                  <L 417>
        var_50 = wp::dot(var_49, var_49);
        return var_50;
    }
    // cp = p - c                                                                             <L 419>
    var_51 = wp::sub(var_p, var_c);
    // d5 = wp.dot(ab, cp)                                                                    <L 420>
    var_52 = wp::dot(var_0, var_51);
    // d6 = wp.dot(ac, cp)                                                                    <L 421>
    var_53 = wp::dot(var_1, var_51);
    // if d6 >= 0.0 and d5 <= d6:                                                             <L 422>
    var_56 = (var_53 >= var_55);
    var_54 = var_56;
    if (var_54) {
        var_57 = (var_52 <= var_53);
        var_54 = var_54 && var_57;
    }
    if (var_54) {
        // delta_c = p - c                                                                    <L 423>
        var_58 = wp::sub(var_p, var_c);
        // return wp.dot(delta_c, delta_c)                                                    <L 424>
        var_59 = wp::dot(var_58, var_58);
        return var_59;
    }
    // vb = d5 * d2 - d1 * d6                                                                 <L 426>
    var_60 = wp::mul(var_52, var_18);
    var_61 = wp::mul(var_17, var_53);
    var_62 = wp::sub(var_60, var_61);
    // if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                              <L 427>
    var_65 = (var_62 <= var_64);
    var_63 = var_65;
    if (var_63) {
        var_67 = (var_18 >= var_66);
        var_63 = var_63 && var_67;
    }
    if (var_63) {
        var_69 = (var_53 <= var_68);
        var_63 = var_63 && var_69;
    }
    if (var_63) {
        // w = d2 / (d2 - d6)                                                                 <L 428>
        var_70 = wp::sub(var_18, var_53);
        var_71 = wp::div(var_18, var_70);
        // q = a + ac * w                                                                     <L 429>
        var_72 = wp::mul(var_1, var_71);
        var_73 = wp::add(var_a, var_72);
        // delta_ac = p - q                                                                   <L 430>
        var_74 = wp::sub(var_p, var_73);
        // return wp.dot(delta_ac, delta_ac)                                                  <L 431>
        var_75 = wp::dot(var_74, var_74);
        return var_75;
    }
    var_76 = wp::where(var_63, var_73, var_48);
    // va = d3 * d6 - d5 * d4                                                                 <L 433>
    var_77 = wp::mul(var_27, var_53);
    var_78 = wp::mul(var_52, var_28);
    var_79 = wp::sub(var_77, var_78);
    // if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                                <L 434>
    var_82 = (var_79 <= var_81);
    var_80 = var_82;
    if (var_80) {
        var_83 = wp::sub(var_28, var_27);
        var_85 = (var_83 >= var_84);
        var_80 = var_80 && var_85;
    }
    if (var_80) {
        var_86 = wp::sub(var_52, var_53);
        var_88 = (var_86 >= var_87);
        var_80 = var_80 && var_88;
    }
    if (var_80) {
        // w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                            <L 435>
        var_89 = wp::sub(var_28, var_27);
        var_90 = wp::sub(var_28, var_27);
        var_91 = wp::sub(var_52, var_53);
        var_92 = wp::add(var_90, var_91);
        var_93 = wp::div(var_89, var_92);
        // q = b + (c - b) * w                                                                <L 436>
        var_94 = wp::sub(var_c, var_b);
        var_95 = wp::mul(var_94, var_93);
        var_96 = wp::add(var_b, var_95);
        // delta_bc = p - q                                                                   <L 437>
        var_97 = wp::sub(var_p, var_96);
        // return wp.dot(delta_bc, delta_bc)                                                  <L 438>
        var_98 = wp::dot(var_97, var_97);
        return var_98;
    }
    var_99 = wp::where(var_80, var_96, var_76);
    var_100 = wp::where(var_80, var_93, var_71);
    // denom = 1.0 / (va + vb + vc)                                                           <L 440>
    var_102 = wp::add(var_79, var_62);
    var_103 = wp::add(var_102, var_37);
    var_104 = wp::div(var_101, var_103);
    // v = vb * denom                                                                         <L 441>
    var_105 = wp::mul(var_62, var_104);
    // w = vc * denom                                                                         <L 442>
    var_106 = wp::mul(var_37, var_104);
    // q = a + ab * v + ac * w                                                                <L 443>
    var_107 = wp::mul(var_0, var_105);
    var_108 = wp::add(var_a, var_107);
    var_109 = wp::mul(var_1, var_106);
    var_110 = wp::add(var_108, var_109);
    // delta_face = p - q                                                                     <L 444>
    var_111 = wp::sub(var_p, var_110);
    // return wp.dot(delta_face, delta_face)                                                  <L 445>
    var_112 = wp::dot(var_111, var_111);
    return var_112;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/deletion.py:364
static CUDA_CALLABLE void adj__point_segment_distance_sq_0(
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
    // def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3):                    <L 365>
    // ab = b - a                                                                             <L 366>
    var_0 = wp::sub(var_b, var_a);
    // denom = wp.dot(ab, ab)                                                                 <L 367>
    var_1 = wp::dot(var_0, var_0);
    // if denom <= 1.0e-20:                                                                   <L 368>
    var_3 = (var_1 <= var_2);
    if (var_3) {
        // d = p - a                                                                          <L 369>
        var_4 = wp::sub(var_p, var_a);
        // return wp.dot(d, d)                                                                <L 370>
        var_5 = wp::dot(var_4, var_4);
        goto label0;
    }
    // t = wp.dot(p - a, ab) / denom                                                          <L 372>
    var_6 = wp::sub(var_p, var_a);
    var_7 = wp::dot(var_6, var_0);
    var_8 = wp::div(var_7, var_1);
    // if t < 0.0:                                                                            <L 373>
    var_10 = (var_8 < var_9);
    if (var_10) {
        // t = 0.0                                                                            <L 374>
    }
    var_12 = wp::where(var_10, var_11, var_8);
    if (!var_10) {
        // elif t > 1.0:                                                                      <L 375>
        var_14 = (var_12 > var_13);
        if (var_14) {
            // t = 1.0                                                                        <L 376>
        }
        var_16 = wp::where(var_14, var_15, var_12);
    }
    var_17 = wp::where(var_10, var_12, var_16);
    // q = a + ab * t                                                                         <L 377>
    var_18 = wp::mul(var_0, var_17);
    var_19 = wp::add(var_a, var_18);
    // d = p - q                                                                              <L 378>
    var_20 = wp::sub(var_p, var_19);
    // return wp.dot(d, d)                                                                    <L 379>
    var_21 = wp::dot(var_20, var_20);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_21 += adj_ret;
    wp::adj_dot(var_20, var_20, adj_20, adj_20, adj_21);
    // adj: return wp.dot(d, d)                                                               <L 379>
    wp::adj_sub(var_p, var_19, adj_p, adj_19, adj_20);
    // adj: d = p - q                                                                         <L 378>
    wp::adj_add(var_a, var_18, adj_a, adj_18, adj_19);
    wp::adj_mul(var_0, var_17, adj_0, adj_17, adj_18);
    // adj: q = a + ab * t                                                                    <L 377>
    wp::adj_where(var_10, var_12, var_16, adj_10, adj_12, adj_16, adj_17);
    if (!var_10) {
        wp::adj_where(var_14, var_15, var_12, adj_14, adj_15, adj_12, adj_16);
        if (var_14) {
            // adj: t = 1.0                                                                   <L 376>
        }
        // adj: elif t > 1.0:                                                                 <L 375>
    }
    wp::adj_where(var_10, var_11, var_8, adj_10, adj_11, adj_8, adj_12);
    if (var_10) {
        // adj: t = 0.0                                                                       <L 374>
    }
    // adj: if t < 0.0:                                                                       <L 373>
    wp::adj_div(var_7, var_1, var_8, adj_7, adj_1, adj_8);
    wp::adj_dot(var_6, var_0, adj_6, adj_0, adj_7);
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_6);
    // adj: t = wp.dot(p - a, ab) / denom                                                     <L 372>
    if (var_3) {
        label0:;
        adj_5 += adj_ret;
        wp::adj_dot(var_4, var_4, adj_4, adj_4, adj_5);
        // adj: return wp.dot(d, d)                                                           <L 370>
        wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_4);
        // adj: d = p - a                                                                     <L 369>
    }
    // adj: if denom <= 1.0e-20:                                                              <L 368>
    wp::adj_dot(var_0, var_0, adj_0, adj_0, adj_1);
    // adj: denom = wp.dot(ab, ab)                                                            <L 367>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: ab = b - a                                                                        <L 366>
    // adj: def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3):               <L 365>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/deletion.py:382
static CUDA_CALLABLE void adj__point_triangle_distance_sq_0(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::vec_t<3, wp::float32> var_c,
    wp::vec_t<3, wp::float32> & adj_p,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::vec_t<3, wp::float32> & adj_c,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::float32 var_3;
    const wp::float32 var_4 = 1e-20;
    bool var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    bool var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    bool var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    bool var_19;
    const wp::float32 var_20 = 0.0;
    bool var_21;
    const wp::float32 var_22 = 0.0;
    bool var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::float32 var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    bool var_29;
    const wp::float32 var_30 = 0.0;
    bool var_31;
    bool var_32;
    wp::vec_t<3, wp::float32> var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    bool var_38;
    const wp::float32 var_39 = 0.0;
    bool var_40;
    const wp::float32 var_41 = 0.0;
    bool var_42;
    const wp::float32 var_43 = 0.0;
    bool var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::vec_t<3, wp::float32> var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    bool var_54;
    const wp::float32 var_55 = 0.0;
    bool var_56;
    bool var_57;
    wp::vec_t<3, wp::float32> var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    bool var_63;
    const wp::float32 var_64 = 0.0;
    bool var_65;
    const wp::float32 var_66 = 0.0;
    bool var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    wp::vec_t<3, wp::float32> var_74;
    wp::float32 var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    wp::float32 var_79;
    bool var_80;
    const wp::float32 var_81 = 0.0;
    bool var_82;
    wp::float32 var_83;
    const wp::float32 var_84 = 0.0;
    bool var_85;
    wp::float32 var_86;
    const wp::float32 var_87 = 0.0;
    bool var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::float32 var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::float32 var_100;
    const wp::float32 var_101 = 1.0;
    wp::float32 var_102;
    wp::float32 var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::float32> var_109;
    wp::vec_t<3, wp::float32> var_110;
    wp::vec_t<3, wp::float32> var_111;
    wp::float32 var_112;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    bool adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    bool adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    bool adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::vec_t<3, wp::float32> adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    bool adj_19 = {};
    wp::float32 adj_20 = {};
    bool adj_21 = {};
    wp::float32 adj_22 = {};
    bool adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::float32 adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    bool adj_29 = {};
    wp::float32 adj_30 = {};
    bool adj_31 = {};
    bool adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::float32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    bool adj_38 = {};
    wp::float32 adj_39 = {};
    bool adj_40 = {};
    wp::float32 adj_41 = {};
    bool adj_42 = {};
    wp::float32 adj_43 = {};
    bool adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::vec_t<3, wp::float32> adj_47 = {};
    wp::vec_t<3, wp::float32> adj_48 = {};
    wp::vec_t<3, wp::float32> adj_49 = {};
    wp::float32 adj_50 = {};
    wp::vec_t<3, wp::float32> adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    bool adj_54 = {};
    wp::float32 adj_55 = {};
    bool adj_56 = {};
    bool adj_57 = {};
    wp::vec_t<3, wp::float32> adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    bool adj_63 = {};
    wp::float32 adj_64 = {};
    bool adj_65 = {};
    wp::float32 adj_66 = {};
    bool adj_67 = {};
    wp::float32 adj_68 = {};
    bool adj_69 = {};
    wp::float32 adj_70 = {};
    wp::float32 adj_71 = {};
    wp::vec_t<3, wp::float32> adj_72 = {};
    wp::vec_t<3, wp::float32> adj_73 = {};
    wp::vec_t<3, wp::float32> adj_74 = {};
    wp::float32 adj_75 = {};
    wp::vec_t<3, wp::float32> adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::float32 adj_79 = {};
    bool adj_80 = {};
    wp::float32 adj_81 = {};
    bool adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    bool adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float32 adj_87 = {};
    bool adj_88 = {};
    wp::float32 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float32 adj_91 = {};
    wp::float32 adj_92 = {};
    wp::float32 adj_93 = {};
    wp::vec_t<3, wp::float32> adj_94 = {};
    wp::vec_t<3, wp::float32> adj_95 = {};
    wp::vec_t<3, wp::float32> adj_96 = {};
    wp::vec_t<3, wp::float32> adj_97 = {};
    wp::float32 adj_98 = {};
    wp::vec_t<3, wp::float32> adj_99 = {};
    wp::float32 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float32 adj_102 = {};
    wp::float32 adj_103 = {};
    wp::float32 adj_104 = {};
    wp::float32 adj_105 = {};
    wp::float32 adj_106 = {};
    wp::vec_t<3, wp::float32> adj_107 = {};
    wp::vec_t<3, wp::float32> adj_108 = {};
    wp::vec_t<3, wp::float32> adj_109 = {};
    wp::vec_t<3, wp::float32> adj_110 = {};
    wp::vec_t<3, wp::float32> adj_111 = {};
    wp::float32 adj_112 = {};
    //---------
    // forward
    // def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):       <L 383>
    // ab = b - a                                                                             <L 384>
    var_0 = wp::sub(var_b, var_a);
    // ac = c - a                                                                             <L 385>
    var_1 = wp::sub(var_c, var_a);
    // n = wp.cross(ab, ac)                                                                   <L 386>
    var_2 = wp::cross(var_0, var_1);
    // if wp.dot(n, n) <= 1.0e-20:                                                            <L 387>
    var_3 = wp::dot(var_2, var_2);
    var_5 = (var_3 <= var_4);
    if (var_5) {
        // d0 = _point_segment_distance_sq(p, a, b)                                           <L 388>
        var_6 = _point_segment_distance_sq_0(var_p, var_a, var_b);
        // d1 = _point_segment_distance_sq(p, b, c)                                           <L 389>
        var_7 = _point_segment_distance_sq_0(var_p, var_b, var_c);
        // d2 = _point_segment_distance_sq(p, c, a)                                           <L 390>
        var_8 = _point_segment_distance_sq_0(var_p, var_c, var_a);
        // best_dist = d0                                                                     <L 391>
        var_9 = wp::copy(var_6);
        // if d1 < best_dist:                                                                 <L 392>
        var_10 = (var_7 < var_9);
        if (var_10) {
            // best_dist = d1                                                                 <L 393>
            var_11 = wp::copy(var_7);
        }
        var_12 = wp::where(var_10, var_11, var_9);
        // if d2 < best_dist:                                                                 <L 394>
        var_13 = (var_8 < var_12);
        if (var_13) {
            // best_dist = d2                                                                 <L 395>
            var_14 = wp::copy(var_8);
        }
        var_15 = wp::where(var_13, var_14, var_12);
        // return best_dist                                                                   <L 396>
        goto label0;
    }
    // ap = p - a                                                                             <L 398>
    var_16 = wp::sub(var_p, var_a);
    // d1 = wp.dot(ab, ap)                                                                    <L 399>
    var_17 = wp::dot(var_0, var_16);
    // d2 = wp.dot(ac, ap)                                                                    <L 400>
    var_18 = wp::dot(var_1, var_16);
    // if d1 <= 0.0 and d2 <= 0.0:                                                            <L 401>
    var_21 = (var_17 <= var_20);
    var_19 = var_21;
    if (var_19) {
        var_23 = (var_18 <= var_22);
        var_19 = var_19 && var_23;
    }
    if (var_19) {
        // delta_a = p - a                                                                    <L 402>
        var_24 = wp::sub(var_p, var_a);
        // return wp.dot(delta_a, delta_a)                                                    <L 403>
        var_25 = wp::dot(var_24, var_24);
        goto label1;
    }
    // bp = p - b                                                                             <L 405>
    var_26 = wp::sub(var_p, var_b);
    // d3 = wp.dot(ab, bp)                                                                    <L 406>
    var_27 = wp::dot(var_0, var_26);
    // d4 = wp.dot(ac, bp)                                                                    <L 407>
    var_28 = wp::dot(var_1, var_26);
    // if d3 >= 0.0 and d4 <= d3:                                                             <L 408>
    var_31 = (var_27 >= var_30);
    var_29 = var_31;
    if (var_29) {
        var_32 = (var_28 <= var_27);
        var_29 = var_29 && var_32;
    }
    if (var_29) {
        // delta_b = p - b                                                                    <L 409>
        var_33 = wp::sub(var_p, var_b);
        // return wp.dot(delta_b, delta_b)                                                    <L 410>
        var_34 = wp::dot(var_33, var_33);
        goto label2;
    }
    // vc = d1 * d4 - d3 * d2                                                                 <L 412>
    var_35 = wp::mul(var_17, var_28);
    var_36 = wp::mul(var_27, var_18);
    var_37 = wp::sub(var_35, var_36);
    // if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                              <L 413>
    var_40 = (var_37 <= var_39);
    var_38 = var_40;
    if (var_38) {
        var_42 = (var_17 >= var_41);
        var_38 = var_38 && var_42;
    }
    if (var_38) {
        var_44 = (var_27 <= var_43);
        var_38 = var_38 && var_44;
    }
    if (var_38) {
        // v = d1 / (d1 - d3)                                                                 <L 414>
        var_45 = wp::sub(var_17, var_27);
        var_46 = wp::div(var_17, var_45);
        // q = a + ab * v                                                                     <L 415>
        var_47 = wp::mul(var_0, var_46);
        var_48 = wp::add(var_a, var_47);
        // delta_ab = p - q                                                                   <L 416>
        var_49 = wp::sub(var_p, var_48);
        // return wp.dot(delta_ab, delta_ab)                                                  <L 417>
        var_50 = wp::dot(var_49, var_49);
        goto label3;
    }
    // cp = p - c                                                                             <L 419>
    var_51 = wp::sub(var_p, var_c);
    // d5 = wp.dot(ab, cp)                                                                    <L 420>
    var_52 = wp::dot(var_0, var_51);
    // d6 = wp.dot(ac, cp)                                                                    <L 421>
    var_53 = wp::dot(var_1, var_51);
    // if d6 >= 0.0 and d5 <= d6:                                                             <L 422>
    var_56 = (var_53 >= var_55);
    var_54 = var_56;
    if (var_54) {
        var_57 = (var_52 <= var_53);
        var_54 = var_54 && var_57;
    }
    if (var_54) {
        // delta_c = p - c                                                                    <L 423>
        var_58 = wp::sub(var_p, var_c);
        // return wp.dot(delta_c, delta_c)                                                    <L 424>
        var_59 = wp::dot(var_58, var_58);
        goto label4;
    }
    // vb = d5 * d2 - d1 * d6                                                                 <L 426>
    var_60 = wp::mul(var_52, var_18);
    var_61 = wp::mul(var_17, var_53);
    var_62 = wp::sub(var_60, var_61);
    // if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                              <L 427>
    var_65 = (var_62 <= var_64);
    var_63 = var_65;
    if (var_63) {
        var_67 = (var_18 >= var_66);
        var_63 = var_63 && var_67;
    }
    if (var_63) {
        var_69 = (var_53 <= var_68);
        var_63 = var_63 && var_69;
    }
    if (var_63) {
        // w = d2 / (d2 - d6)                                                                 <L 428>
        var_70 = wp::sub(var_18, var_53);
        var_71 = wp::div(var_18, var_70);
        // q = a + ac * w                                                                     <L 429>
        var_72 = wp::mul(var_1, var_71);
        var_73 = wp::add(var_a, var_72);
        // delta_ac = p - q                                                                   <L 430>
        var_74 = wp::sub(var_p, var_73);
        // return wp.dot(delta_ac, delta_ac)                                                  <L 431>
        var_75 = wp::dot(var_74, var_74);
        goto label5;
    }
    var_76 = wp::where(var_63, var_73, var_48);
    // va = d3 * d6 - d5 * d4                                                                 <L 433>
    var_77 = wp::mul(var_27, var_53);
    var_78 = wp::mul(var_52, var_28);
    var_79 = wp::sub(var_77, var_78);
    // if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                                <L 434>
    var_82 = (var_79 <= var_81);
    var_80 = var_82;
    if (var_80) {
        var_83 = wp::sub(var_28, var_27);
        var_85 = (var_83 >= var_84);
        var_80 = var_80 && var_85;
    }
    if (var_80) {
        var_86 = wp::sub(var_52, var_53);
        var_88 = (var_86 >= var_87);
        var_80 = var_80 && var_88;
    }
    if (var_80) {
        // w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                            <L 435>
        var_89 = wp::sub(var_28, var_27);
        var_90 = wp::sub(var_28, var_27);
        var_91 = wp::sub(var_52, var_53);
        var_92 = wp::add(var_90, var_91);
        var_93 = wp::div(var_89, var_92);
        // q = b + (c - b) * w                                                                <L 436>
        var_94 = wp::sub(var_c, var_b);
        var_95 = wp::mul(var_94, var_93);
        var_96 = wp::add(var_b, var_95);
        // delta_bc = p - q                                                                   <L 437>
        var_97 = wp::sub(var_p, var_96);
        // return wp.dot(delta_bc, delta_bc)                                                  <L 438>
        var_98 = wp::dot(var_97, var_97);
        goto label6;
    }
    var_99 = wp::where(var_80, var_96, var_76);
    var_100 = wp::where(var_80, var_93, var_71);
    // denom = 1.0 / (va + vb + vc)                                                           <L 440>
    var_102 = wp::add(var_79, var_62);
    var_103 = wp::add(var_102, var_37);
    var_104 = wp::div(var_101, var_103);
    // v = vb * denom                                                                         <L 441>
    var_105 = wp::mul(var_62, var_104);
    // w = vc * denom                                                                         <L 442>
    var_106 = wp::mul(var_37, var_104);
    // q = a + ab * v + ac * w                                                                <L 443>
    var_107 = wp::mul(var_0, var_105);
    var_108 = wp::add(var_a, var_107);
    var_109 = wp::mul(var_1, var_106);
    var_110 = wp::add(var_108, var_109);
    // delta_face = p - q                                                                     <L 444>
    var_111 = wp::sub(var_p, var_110);
    // return wp.dot(delta_face, delta_face)                                                  <L 445>
    var_112 = wp::dot(var_111, var_111);
    goto label7;
    //---------
    // reverse
    label7:;
    adj_112 += adj_ret;
    wp::adj_dot(var_111, var_111, adj_111, adj_111, adj_112);
    // adj: return wp.dot(delta_face, delta_face)                                             <L 445>
    wp::adj_sub(var_p, var_110, adj_p, adj_110, adj_111);
    // adj: delta_face = p - q                                                                <L 444>
    wp::adj_add(var_108, var_109, adj_108, adj_109, adj_110);
    wp::adj_mul(var_1, var_106, adj_1, adj_106, adj_109);
    wp::adj_add(var_a, var_107, adj_a, adj_107, adj_108);
    wp::adj_mul(var_0, var_105, adj_0, adj_105, adj_107);
    // adj: q = a + ab * v + ac * w                                                           <L 443>
    wp::adj_mul(var_37, var_104, adj_37, adj_104, adj_106);
    // adj: w = vc * denom                                                                    <L 442>
    wp::adj_mul(var_62, var_104, adj_62, adj_104, adj_105);
    // adj: v = vb * denom                                                                    <L 441>
    wp::adj_div(var_101, var_103, var_104, adj_101, adj_103, adj_104);
    wp::adj_add(var_102, var_37, adj_102, adj_37, adj_103);
    wp::adj_add(var_79, var_62, adj_79, adj_62, adj_102);
    // adj: denom = 1.0 / (va + vb + vc)                                                      <L 440>
    wp::adj_where(var_80, var_93, var_71, adj_80, adj_93, adj_71, adj_100);
    wp::adj_where(var_80, var_96, var_76, adj_80, adj_96, adj_76, adj_99);
    if (var_80) {
        label6:;
        adj_98 += adj_ret;
        wp::adj_dot(var_97, var_97, adj_97, adj_97, adj_98);
        // adj: return wp.dot(delta_bc, delta_bc)                                             <L 438>
        wp::adj_sub(var_p, var_96, adj_p, adj_96, adj_97);
        // adj: delta_bc = p - q                                                              <L 437>
        wp::adj_add(var_b, var_95, adj_b, adj_95, adj_96);
        wp::adj_mul(var_94, var_93, adj_94, adj_93, adj_95);
        wp::adj_sub(var_c, var_b, adj_c, adj_b, adj_94);
        // adj: q = b + (c - b) * w                                                           <L 436>
        wp::adj_div(var_89, var_92, var_93, adj_89, adj_92, adj_93);
        wp::adj_add(var_90, var_91, adj_90, adj_91, adj_92);
        wp::adj_sub(var_52, var_53, adj_52, adj_53, adj_91);
        wp::adj_sub(var_28, var_27, adj_28, adj_27, adj_90);
        wp::adj_sub(var_28, var_27, adj_28, adj_27, adj_89);
        // adj: w = (d4 - d3) / ((d4 - d3) + (d5 - d6))                                       <L 435>
    }
    if (var_80) {
        wp::adj_sub(var_52, var_53, adj_52, adj_53, adj_86);
    }
    if (var_80) {
        wp::adj_sub(var_28, var_27, adj_28, adj_27, adj_83);
    }
    // adj: if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:                           <L 434>
    wp::adj_sub(var_77, var_78, adj_77, adj_78, adj_79);
    wp::adj_mul(var_52, var_28, adj_52, adj_28, adj_78);
    wp::adj_mul(var_27, var_53, adj_27, adj_53, adj_77);
    // adj: va = d3 * d6 - d5 * d4                                                            <L 433>
    wp::adj_where(var_63, var_73, var_48, adj_63, adj_73, adj_48, adj_76);
    if (var_63) {
        label5:;
        adj_75 += adj_ret;
        wp::adj_dot(var_74, var_74, adj_74, adj_74, adj_75);
        // adj: return wp.dot(delta_ac, delta_ac)                                             <L 431>
        wp::adj_sub(var_p, var_73, adj_p, adj_73, adj_74);
        // adj: delta_ac = p - q                                                              <L 430>
        wp::adj_add(var_a, var_72, adj_a, adj_72, adj_73);
        wp::adj_mul(var_1, var_71, adj_1, adj_71, adj_72);
        // adj: q = a + ac * w                                                                <L 429>
        wp::adj_div(var_18, var_70, var_71, adj_18, adj_70, adj_71);
        wp::adj_sub(var_18, var_53, adj_18, adj_53, adj_70);
        // adj: w = d2 / (d2 - d6)                                                            <L 428>
    }
    if (var_63) {
    }
    if (var_63) {
    }
    // adj: if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:                                         <L 427>
    wp::adj_sub(var_60, var_61, adj_60, adj_61, adj_62);
    wp::adj_mul(var_17, var_53, adj_17, adj_53, adj_61);
    wp::adj_mul(var_52, var_18, adj_52, adj_18, adj_60);
    // adj: vb = d5 * d2 - d1 * d6                                                            <L 426>
    if (var_54) {
        label4:;
        adj_59 += adj_ret;
        wp::adj_dot(var_58, var_58, adj_58, adj_58, adj_59);
        // adj: return wp.dot(delta_c, delta_c)                                               <L 424>
        wp::adj_sub(var_p, var_c, adj_p, adj_c, adj_58);
        // adj: delta_c = p - c                                                               <L 423>
    }
    if (var_54) {
    }
    // adj: if d6 >= 0.0 and d5 <= d6:                                                        <L 422>
    wp::adj_dot(var_1, var_51, adj_1, adj_51, adj_53);
    // adj: d6 = wp.dot(ac, cp)                                                               <L 421>
    wp::adj_dot(var_0, var_51, adj_0, adj_51, adj_52);
    // adj: d5 = wp.dot(ab, cp)                                                               <L 420>
    wp::adj_sub(var_p, var_c, adj_p, adj_c, adj_51);
    // adj: cp = p - c                                                                        <L 419>
    if (var_38) {
        label3:;
        adj_50 += adj_ret;
        wp::adj_dot(var_49, var_49, adj_49, adj_49, adj_50);
        // adj: return wp.dot(delta_ab, delta_ab)                                             <L 417>
        wp::adj_sub(var_p, var_48, adj_p, adj_48, adj_49);
        // adj: delta_ab = p - q                                                              <L 416>
        wp::adj_add(var_a, var_47, adj_a, adj_47, adj_48);
        wp::adj_mul(var_0, var_46, adj_0, adj_46, adj_47);
        // adj: q = a + ab * v                                                                <L 415>
        wp::adj_div(var_17, var_45, var_46, adj_17, adj_45, adj_46);
        wp::adj_sub(var_17, var_27, adj_17, adj_27, adj_45);
        // adj: v = d1 / (d1 - d3)                                                            <L 414>
    }
    if (var_38) {
    }
    if (var_38) {
    }
    // adj: if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:                                         <L 413>
    wp::adj_sub(var_35, var_36, adj_35, adj_36, adj_37);
    wp::adj_mul(var_27, var_18, adj_27, adj_18, adj_36);
    wp::adj_mul(var_17, var_28, adj_17, adj_28, adj_35);
    // adj: vc = d1 * d4 - d3 * d2                                                            <L 412>
    if (var_29) {
        label2:;
        adj_34 += adj_ret;
        wp::adj_dot(var_33, var_33, adj_33, adj_33, adj_34);
        // adj: return wp.dot(delta_b, delta_b)                                               <L 410>
        wp::adj_sub(var_p, var_b, adj_p, adj_b, adj_33);
        // adj: delta_b = p - b                                                               <L 409>
    }
    if (var_29) {
    }
    // adj: if d3 >= 0.0 and d4 <= d3:                                                        <L 408>
    wp::adj_dot(var_1, var_26, adj_1, adj_26, adj_28);
    // adj: d4 = wp.dot(ac, bp)                                                               <L 407>
    wp::adj_dot(var_0, var_26, adj_0, adj_26, adj_27);
    // adj: d3 = wp.dot(ab, bp)                                                               <L 406>
    wp::adj_sub(var_p, var_b, adj_p, adj_b, adj_26);
    // adj: bp = p - b                                                                        <L 405>
    if (var_19) {
        label1:;
        adj_25 += adj_ret;
        wp::adj_dot(var_24, var_24, adj_24, adj_24, adj_25);
        // adj: return wp.dot(delta_a, delta_a)                                               <L 403>
        wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_24);
        // adj: delta_a = p - a                                                               <L 402>
    }
    if (var_19) {
    }
    // adj: if d1 <= 0.0 and d2 <= 0.0:                                                       <L 401>
    wp::adj_dot(var_1, var_16, adj_1, adj_16, adj_18);
    // adj: d2 = wp.dot(ac, ap)                                                               <L 400>
    wp::adj_dot(var_0, var_16, adj_0, adj_16, adj_17);
    // adj: d1 = wp.dot(ab, ap)                                                               <L 399>
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_16);
    // adj: ap = p - a                                                                        <L 398>
    if (var_5) {
        label0:;
        adj_15 += adj_ret;
        // adj: return best_dist                                                              <L 396>
        wp::adj_where(var_13, var_14, var_12, adj_13, adj_14, adj_12, adj_15);
        if (var_13) {
            wp::adj_copy(var_8, adj_8, adj_14);
            // adj: best_dist = d2                                                            <L 395>
        }
        // adj: if d2 < best_dist:                                                            <L 394>
        wp::adj_where(var_10, var_11, var_9, adj_10, adj_11, adj_9, adj_12);
        if (var_10) {
            wp::adj_copy(var_7, adj_7, adj_11);
            // adj: best_dist = d1                                                            <L 393>
        }
        // adj: if d1 < best_dist:                                                            <L 392>
        wp::adj_copy(var_6, adj_6, adj_9);
        // adj: best_dist = d0                                                                <L 391>
        adj__point_segment_distance_sq_0(var_p, var_c, var_a, adj_p, adj_c, adj_a, adj_8);
        // adj: d2 = _point_segment_distance_sq(p, c, a)                                      <L 390>
        adj__point_segment_distance_sq_0(var_p, var_b, var_c, adj_p, adj_b, adj_c, adj_7);
        // adj: d1 = _point_segment_distance_sq(p, b, c)                                      <L 389>
        adj__point_segment_distance_sq_0(var_p, var_a, var_b, adj_p, adj_a, adj_b, adj_6);
        // adj: d0 = _point_segment_distance_sq(p, a, b)                                      <L 388>
    }
    wp::adj_dot(var_2, var_2, adj_2, adj_2, adj_3);
    // adj: if wp.dot(n, n) <= 1.0e-20:                                                       <L 387>
    wp::adj_cross(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: n = wp.cross(ab, ac)                                                              <L 386>
    wp::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    // adj: ac = c - a                                                                        <L 385>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: ab = b - a                                                                        <L 384>
    // adj: def _point_triangle_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):  <L 383>
    return;
}



extern "C" __global__ void validate_node_state_kernel_b94a5c4e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_locked_node_mask,
    wp::array_t<wp::float32> var_particle_mass,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_error_count,
    wp::array_t<wp::int32> var_first_error)
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
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 2000;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        bool var_28;
        const wp::float32 var_29 = -1e-05;
        bool var_30;
        const wp::float32 var_31 = -1e-05;
        bool var_32;
        const wp::float32 var_33 = -1e-05;
        bool var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        bool var_37;
        bool var_38;
        bool var_39;
        bool var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        bool var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        bool var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        bool var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        const wp::float32 var_63 = 0.0;
        bool var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        const wp::int32 var_67 = 0;
        bool var_68;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        const wp::int32 var_74 = 0;
        //---------
        // forward
        // def validate_node_state_kernel(                                                        <L 294>
        // i = wp.tid()                                                                           <L 304>
        var_0 = builtin_tid1d();
        // support = node_support_count[i]                                                        <L 305>
        var_1 = wp::address(var_node_support_count, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // mass = node_mass[i]                                                                    <L 306>
        var_4 = wp::address(var_node_mass, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // effective_mass = particle_mass[i]                                                      <L 307>
        var_7 = wp::address(var_particle_mass, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // inv_mass = particle_inv_mass[i]                                                        <L 308>
        var_10 = wp::address(var_particle_inv_mass, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // flags = particle_flags[i]                                                              <L 309>
        var_13 = wp::address(var_particle_flags, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // locked = locked_node_mask[i]                                                           <L 310>
        var_16 = wp::address(var_locked_node_mask, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // bad = int(0)                                                                           <L 312>
        var_20 = wp::int(var_19);
        // code = int(2000 + i)                                                                   <L 313>
        var_22 = wp::add(var_21, var_0);
        var_23 = wp::int(var_22);
        // if support < 0:                                                                        <L 314>
        var_25 = (var_2 < var_24);
        if (var_25) {
            // bad = 1                                                                            <L 315>
        }
        var_27 = wp::where(var_25, var_26, var_20);
        // if mass < -1.0e-5 or effective_mass < -1.0e-5 or inv_mass < -1.0e-5:                   <L 316>
        var_30 = (var_5 < var_29);
        var_28 = var_30;
        if (!var_28) {
            var_32 = (var_8 < var_31);
            var_28 = var_28 || var_32;
        }
        if (!var_28) {
            var_34 = (var_11 < var_33);
            var_28 = var_28 || var_34;
        }
        if (var_28) {
            // bad = 1                                                                            <L 317>
        }
        var_36 = wp::where(var_28, var_35, var_27);
        // if mass != mass or effective_mass != effective_mass or inv_mass != inv_mass:           <L 318>
        var_38 = (var_5 != var_5);
        var_37 = var_38;
        if (!var_37) {
            var_39 = (var_8 != var_8);
            var_37 = var_37 || var_39;
        }
        if (!var_37) {
            var_40 = (var_11 != var_11);
            var_37 = var_37 || var_40;
        }
        if (var_37) {
            // bad = 1                                                                            <L 319>
        }
        var_42 = wp::where(var_37, var_41, var_36);
        // if support <= 0 and (flags & _ACTIVE_BIT) != 0:                                        <L 320>
        var_45 = (var_2 <= var_44);
        var_43 = var_45;
        if (var_43) {
            var_47 = wp::bit_and(var_14, var_46);
            var_49 = (var_47 != var_48);
            var_43 = var_43 && var_49;
        }
        if (var_43) {
            // bad = 1                                                                            <L 321>
        }
        var_51 = wp::where(var_43, var_50, var_42);
        // if support > 0 and (flags & _ACTIVE_BIT) == 0:                                         <L 322>
        var_54 = (var_2 > var_53);
        var_52 = var_54;
        if (var_52) {
            var_55 = wp::bit_and(var_14, var_46);
            var_57 = (var_55 == var_56);
            var_52 = var_52 && var_57;
        }
        if (var_52) {
            // bad = 1                                                                            <L 323>
        }
        var_59 = wp::where(var_52, var_58, var_51);
        // if locked != 0 and inv_mass != 0.0:                                                    <L 324>
        var_62 = (var_17 != var_61);
        var_60 = var_62;
        if (var_60) {
            var_64 = (var_11 != var_63);
            var_60 = var_60 && var_64;
        }
        if (var_60) {
            // bad = 1                                                                            <L 325>
        }
        var_66 = wp::where(var_60, var_65, var_59);
        // if bad != 0:                                                                           <L 327>
        var_68 = (var_66 != var_67);
        if (var_68) {
            // old = wp.atomic_add(error_count, 0, 1)                                             <L 328>
            var_71 = wp::atomic_add(var_error_count, var_69, var_70);
            // if old == 0:                                                                       <L 329>
            var_73 = (var_71 == var_72);
            if (var_73) {
                // first_error[0] = code                                                          <L 330>
                wp::array_store(var_first_error, var_74, var_23);
            }
        }
    }
}



extern "C" __global__ void validate_node_state_kernel_b94a5c4e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_locked_node_mask,
    wp::array_t<wp::float32> var_particle_mass,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_error_count,
    wp::array_t<wp::int32> var_first_error,
    wp::array_t<wp::int32> adj_node_support_count,
    wp::array_t<wp::float32> adj_node_mass,
    wp::array_t<wp::int32> adj_locked_node_mask,
    wp::array_t<wp::float32> adj_particle_mass,
    wp::array_t<wp::float32> adj_particle_inv_mass,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_error_count,
    wp::array_t<wp::int32> adj_first_error)
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
        wp::float32* var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32* var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 2000;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        bool var_28;
        const wp::float32 var_29 = -1e-05;
        bool var_30;
        const wp::float32 var_31 = -1e-05;
        bool var_32;
        const wp::float32 var_33 = -1e-05;
        bool var_34;
        const wp::int32 var_35 = 1;
        wp::int32 var_36;
        bool var_37;
        bool var_38;
        bool var_39;
        bool var_40;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        bool var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        bool var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        bool var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        bool var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        const wp::float32 var_63 = 0.0;
        bool var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        const wp::int32 var_67 = 0;
        bool var_68;
        const wp::int32 var_69 = 0;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        const wp::int32 var_74 = 0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
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
        bool adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        bool adj_28 = {};
        wp::float32 adj_29 = {};
        bool adj_30 = {};
        wp::float32 adj_31 = {};
        bool adj_32 = {};
        wp::float32 adj_33 = {};
        bool adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        bool adj_37 = {};
        bool adj_38 = {};
        bool adj_39 = {};
        bool adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::int32 adj_44 = {};
        bool adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        bool adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        bool adj_52 = {};
        wp::int32 adj_53 = {};
        bool adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        bool adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        bool adj_60 = {};
        wp::int32 adj_61 = {};
        bool adj_62 = {};
        wp::float32 adj_63 = {};
        bool adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        bool adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        bool adj_73 = {};
        wp::int32 adj_74 = {};
        //---------
        // forward
        // def validate_node_state_kernel(                                                        <L 294>
        // i = wp.tid()                                                                           <L 304>
        var_0 = builtin_tid1d();
        // support = node_support_count[i]                                                        <L 305>
        var_1 = wp::address(var_node_support_count, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // mass = node_mass[i]                                                                    <L 306>
        var_4 = wp::address(var_node_mass, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // effective_mass = particle_mass[i]                                                      <L 307>
        var_7 = wp::address(var_particle_mass, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // inv_mass = particle_inv_mass[i]                                                        <L 308>
        var_10 = wp::address(var_particle_inv_mass, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // flags = particle_flags[i]                                                              <L 309>
        var_13 = wp::address(var_particle_flags, var_0);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // locked = locked_node_mask[i]                                                           <L 310>
        var_16 = wp::address(var_locked_node_mask, var_0);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // bad = int(0)                                                                           <L 312>
        var_20 = wp::int(var_19);
        // code = int(2000 + i)                                                                   <L 313>
        var_22 = wp::add(var_21, var_0);
        var_23 = wp::int(var_22);
        // if support < 0:                                                                        <L 314>
        var_25 = (var_2 < var_24);
        if (var_25) {
            // bad = 1                                                                            <L 315>
        }
        var_27 = wp::where(var_25, var_26, var_20);
        // if mass < -1.0e-5 or effective_mass < -1.0e-5 or inv_mass < -1.0e-5:                   <L 316>
        var_30 = (var_5 < var_29);
        var_28 = var_30;
        if (!var_28) {
            var_32 = (var_8 < var_31);
            var_28 = var_28 || var_32;
        }
        if (!var_28) {
            var_34 = (var_11 < var_33);
            var_28 = var_28 || var_34;
        }
        if (var_28) {
            // bad = 1                                                                            <L 317>
        }
        var_36 = wp::where(var_28, var_35, var_27);
        // if mass != mass or effective_mass != effective_mass or inv_mass != inv_mass:           <L 318>
        var_38 = (var_5 != var_5);
        var_37 = var_38;
        if (!var_37) {
            var_39 = (var_8 != var_8);
            var_37 = var_37 || var_39;
        }
        if (!var_37) {
            var_40 = (var_11 != var_11);
            var_37 = var_37 || var_40;
        }
        if (var_37) {
            // bad = 1                                                                            <L 319>
        }
        var_42 = wp::where(var_37, var_41, var_36);
        // if support <= 0 and (flags & _ACTIVE_BIT) != 0:                                        <L 320>
        var_45 = (var_2 <= var_44);
        var_43 = var_45;
        if (var_43) {
            var_47 = wp::bit_and(var_14, var_46);
            var_49 = (var_47 != var_48);
            var_43 = var_43 && var_49;
        }
        if (var_43) {
            // bad = 1                                                                            <L 321>
        }
        var_51 = wp::where(var_43, var_50, var_42);
        // if support > 0 and (flags & _ACTIVE_BIT) == 0:                                         <L 322>
        var_54 = (var_2 > var_53);
        var_52 = var_54;
        if (var_52) {
            var_55 = wp::bit_and(var_14, var_46);
            var_57 = (var_55 == var_56);
            var_52 = var_52 && var_57;
        }
        if (var_52) {
            // bad = 1                                                                            <L 323>
        }
        var_59 = wp::where(var_52, var_58, var_51);
        // if locked != 0 and inv_mass != 0.0:                                                    <L 324>
        var_62 = (var_17 != var_61);
        var_60 = var_62;
        if (var_60) {
            var_64 = (var_11 != var_63);
            var_60 = var_60 && var_64;
        }
        if (var_60) {
            // bad = 1                                                                            <L 325>
        }
        var_66 = wp::where(var_60, var_65, var_59);
        // if bad != 0:                                                                           <L 327>
        var_68 = (var_66 != var_67);
        if (var_68) {
            // old = wp.atomic_add(error_count, 0, 1)                                             <L 328>
            // var_71 = wp::atomic_add(var_error_count, var_69, var_70);
            // if old == 0:                                                                       <L 329>
            var_73 = (var_71 == var_72);
            if (var_73) {
                // first_error[0] = code                                                          <L 330>
                // wp::array_store(var_first_error, var_74, var_23);
            }
        }
        //---------
        // reverse
        if (var_68) {
            if (var_73) {
                wp::adj_array_store(var_first_error, var_74, var_23, adj_first_error, adj_74, adj_23);
                // adj: first_error[0] = code                                                     <L 330>
            }
            // adj: if old == 0:                                                                  <L 329>
            wp::adj_atomic_add(var_error_count, var_69, var_70, adj_error_count, adj_69, adj_70, adj_71);
            // adj: old = wp.atomic_add(error_count, 0, 1)                                        <L 328>
        }
        // adj: if bad != 0:                                                                      <L 327>
        wp::adj_where(var_60, var_65, var_59, adj_60, adj_65, adj_59, adj_66);
        if (var_60) {
            // adj: bad = 1                                                                       <L 325>
        }
        if (var_60) {
        }
        // adj: if locked != 0 and inv_mass != 0.0:                                               <L 324>
        wp::adj_where(var_52, var_58, var_51, adj_52, adj_58, adj_51, adj_59);
        if (var_52) {
            // adj: bad = 1                                                                       <L 323>
        }
        if (var_52) {
        }
        // adj: if support > 0 and (flags & _ACTIVE_BIT) == 0:                                    <L 322>
        wp::adj_where(var_43, var_50, var_42, adj_43, adj_50, adj_42, adj_51);
        if (var_43) {
            // adj: bad = 1                                                                       <L 321>
        }
        if (var_43) {
        }
        // adj: if support <= 0 and (flags & _ACTIVE_BIT) != 0:                                   <L 320>
        wp::adj_where(var_37, var_41, var_36, adj_37, adj_41, adj_36, adj_42);
        if (var_37) {
            // adj: bad = 1                                                                       <L 319>
        }
        if (!var_37) {
        }
        if (!var_37) {
        }
        // adj: if mass != mass or effective_mass != effective_mass or inv_mass != inv_mass:      <L 318>
        wp::adj_where(var_28, var_35, var_27, adj_28, adj_35, adj_27, adj_36);
        if (var_28) {
            // adj: bad = 1                                                                       <L 317>
        }
        if (!var_28) {
        }
        if (!var_28) {
        }
        // adj: if mass < -1.0e-5 or effective_mass < -1.0e-5 or inv_mass < -1.0e-5:              <L 316>
        wp::adj_where(var_25, var_26, var_20, adj_25, adj_26, adj_20, adj_27);
        if (var_25) {
            // adj: bad = 1                                                                       <L 315>
        }
        // adj: if support < 0:                                                                   <L 314>
        wp::adj_int(var_22, adj_22, adj_23);
        wp::adj_add(var_21, var_0, adj_21, adj_0, adj_22);
        // adj: code = int(2000 + i)                                                              <L 313>
        wp::adj_int(var_19, adj_19, adj_20);
        // adj: bad = int(0)                                                                      <L 312>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_address(var_locked_node_mask, var_0, adj_locked_node_mask, adj_0, adj_16);
        // adj: locked = locked_node_mask[i]                                                      <L 310>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_13);
        // adj: flags = particle_flags[i]                                                         <L 309>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_particle_inv_mass, var_0, adj_particle_inv_mass, adj_0, adj_10);
        // adj: inv_mass = particle_inv_mass[i]                                                   <L 308>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_particle_mass, var_0, adj_particle_mass, adj_0, adj_7);
        // adj: effective_mass = particle_mass[i]                                                 <L 307>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_address(var_node_mass, var_0, adj_node_mass, adj_0, adj_4);
        // adj: mass = node_mass[i]                                                               <L 306>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_node_support_count, var_0, adj_node_support_count, adj_0, adj_1);
        // adj: support = node_support_count[i]                                                   <L 305>
        // adj: i = wp.tid()                                                                      <L 304>
        // adj: def validate_node_state_kernel(                                                   <L 294>
        continue;
    }
}



extern "C" __global__ void delete_cells_sparse_kernel_b7ec3e34_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_ids,
    wp::int32 var_candidate_capacity,
    wp::array_t<wp::int32> var_candidate_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::float32> var_cell_mass,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::array_t<wp::int32> var_deleted_total)
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
        bool var_2;
        const wp::int32 var_3 = 0;
        wp::int32* var_4;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        bool var_13;
        const wp::int32 var_14 = 1;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        bool var_18;
        const wp::int32 var_19 = 0;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        bool var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        const wp::float32 var_27 = 1.0;
        const wp::float32 var_28 = 8.0;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 0;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 2;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 3;
        wp::int32* var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 1;
        wp::int32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 4;
        wp::int32* var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::int32 var_72 = 5;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::int32 var_80 = 6;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 7;
        wp::int32* var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        //---------
        // forward
        // def delete_cells_sparse_kernel(                                                        <L 13>
        // tid = wp.tid()                                                                         <L 34>
        var_0 = builtin_tid1d();
        // if tid >= candidate_capacity or tid >= candidate_count[0]:                             <L 35>
        var_2 = (var_0 >= var_candidate_capacity);
        var_1 = var_2;
        if (!var_1) {
            var_4 = wp::address(var_candidate_count, var_3);
            var_6 = wp::load(var_4);
            var_5 = (var_0 >= var_6);
            var_1 = var_1 || var_5;
        }
        if (var_1) {
            // return                                                                             <L 36>
            continue;
        }
        // cell_idx = cell_ids[tid]                                                               <L 38>
        var_7 = wp::address(var_cell_ids, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 39>
        var_12 = (var_8 < var_11);
        var_10 = var_12;
        if (!var_10) {
            var_13 = (var_8 >= var_num_cells);
            var_10 = var_10 || var_13;
        }
        if (var_10) {
            // return                                                                             <L 40>
            continue;
        }
        // old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)                                <L 42>
        var_16 = wp::atomic_cas(var_cell_active, var_8, var_14, var_15);
        // if old_active != 1:                                                                    <L 43>
        var_18 = (var_16 != var_17);
        if (var_18) {
            // return                                                                             <L 44>
            continue;
        }
        // out_idx = wp.atomic_add(deleted_count, 0, 1)                                           <L 46>
        var_21 = wp::atomic_add(var_deleted_count, var_19, var_20);
        // if out_idx < candidate_capacity:                                                       <L 47>
        var_22 = (var_21 < var_candidate_capacity);
        if (var_22) {
            // deleted_cells[out_idx] = cell_idx                                                  <L 48>
            wp::array_store(var_deleted_cells, var_21, var_8);
        }
        // wp.atomic_add(deleted_total, 0, 1)                                                     <L 49>
        var_25 = wp::atomic_add(var_deleted_total, var_23, var_24);
        // cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)                                     <L 51>
        var_26 = wp::address(var_cell_mass, var_8);
        var_29 = wp::div(var_27, var_28);
        var_31 = wp::load(var_26);
        var_30 = wp::mul(var_31, var_29);
        // for local_node in range(8):                                                            <L 52>
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_33 = wp::address(var_cell_nodes, var_8, var_32);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_37 = wp::atomic_sub(var_node_support_count, var_34, var_36);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_38 = wp::neg(var_30);
        var_39 = wp::atomic_add(var_node_mass, var_34, var_38);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_41 = wp::address(var_cell_nodes, var_8, var_40);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_45 = wp::atomic_sub(var_node_support_count, var_42, var_44);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_46 = wp::neg(var_30);
        var_47 = wp::atomic_add(var_node_mass, var_42, var_46);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_49 = wp::address(var_cell_nodes, var_8, var_48);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_53 = wp::atomic_sub(var_node_support_count, var_50, var_52);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_54 = wp::neg(var_30);
        var_55 = wp::atomic_add(var_node_mass, var_50, var_54);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_57 = wp::address(var_cell_nodes, var_8, var_56);
        var_59 = wp::load(var_57);
        var_58 = wp::copy(var_59);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_61 = wp::atomic_sub(var_node_support_count, var_58, var_60);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_62 = wp::neg(var_30);
        var_63 = wp::atomic_add(var_node_mass, var_58, var_62);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_65 = wp::address(var_cell_nodes, var_8, var_64);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_69 = wp::atomic_sub(var_node_support_count, var_66, var_68);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_70 = wp::neg(var_30);
        var_71 = wp::atomic_add(var_node_mass, var_66, var_70);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_73 = wp::address(var_cell_nodes, var_8, var_72);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_77 = wp::atomic_sub(var_node_support_count, var_74, var_76);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_78 = wp::neg(var_30);
        var_79 = wp::atomic_add(var_node_mass, var_74, var_78);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_81 = wp::address(var_cell_nodes, var_8, var_80);
        var_83 = wp::load(var_81);
        var_82 = wp::copy(var_83);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_85 = wp::atomic_sub(var_node_support_count, var_82, var_84);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_86 = wp::neg(var_30);
        var_87 = wp::atomic_add(var_node_mass, var_82, var_86);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_89 = wp::address(var_cell_nodes, var_8, var_88);
        var_91 = wp::load(var_89);
        var_90 = wp::copy(var_91);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        var_93 = wp::atomic_sub(var_node_support_count, var_90, var_92);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_94 = wp::neg(var_30);
        var_95 = wp::atomic_add(var_node_mass, var_90, var_94);
    }
}



extern "C" __global__ void delete_cells_sparse_kernel_b7ec3e34_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_ids,
    wp::int32 var_candidate_capacity,
    wp::array_t<wp::int32> var_candidate_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::float32> var_cell_mass,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::array_t<wp::int32> var_deleted_total,
    wp::array_t<wp::int32> adj_cell_ids,
    wp::int32 adj_candidate_capacity,
    wp::array_t<wp::int32> adj_candidate_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::float32> adj_cell_mass,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_node_support_count,
    wp::array_t<wp::float32> adj_node_mass,
    wp::array_t<wp::int32> adj_deleted_cells,
    wp::array_t<wp::int32> adj_deleted_count,
    wp::array_t<wp::int32> adj_deleted_total)
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
        bool var_2;
        const wp::int32 var_3 = 0;
        wp::int32* var_4;
        bool var_5;
        wp::int32 var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        bool var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        bool var_13;
        const wp::int32 var_14 = 1;
        const wp::int32 var_15 = 0;
        wp::int32 var_16;
        const wp::int32 var_17 = 1;
        bool var_18;
        const wp::int32 var_19 = 0;
        const wp::int32 var_20 = 1;
        wp::int32 var_21;
        bool var_22;
        const wp::int32 var_23 = 0;
        const wp::int32 var_24 = 1;
        wp::int32 var_25;
        wp::float32* var_26;
        const wp::float32 var_27 = 1.0;
        const wp::float32 var_28 = 8.0;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 0;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        wp::int32* var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::int32 var_48 = 2;
        wp::int32* var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        const wp::int32 var_52 = 1;
        wp::int32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 3;
        wp::int32* var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 1;
        wp::int32 var_61;
        wp::float32 var_62;
        wp::float32 var_63;
        const wp::int32 var_64 = 4;
        wp::int32* var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        const wp::int32 var_72 = 5;
        wp::int32* var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::int32 var_80 = 6;
        wp::int32* var_81;
        wp::int32 var_82;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        wp::int32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        const wp::int32 var_88 = 7;
        wp::int32* var_89;
        wp::int32 var_90;
        wp::int32 var_91;
        const wp::int32 var_92 = 1;
        wp::int32 var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        bool adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        bool adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::int32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        //---------
        // forward
        // def delete_cells_sparse_kernel(                                                        <L 13>
        // tid = wp.tid()                                                                         <L 34>
        var_0 = builtin_tid1d();
        // if tid >= candidate_capacity or tid >= candidate_count[0]:                             <L 35>
        var_2 = (var_0 >= var_candidate_capacity);
        var_1 = var_2;
        if (!var_1) {
            var_4 = wp::address(var_candidate_count, var_3);
            var_6 = wp::load(var_4);
            var_5 = (var_0 >= var_6);
            var_1 = var_1 || var_5;
        }
        if (var_1) {
            // return                                                                             <L 36>
            goto label0;
        }
        // cell_idx = cell_ids[tid]                                                               <L 38>
        var_7 = wp::address(var_cell_ids, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 39>
        var_12 = (var_8 < var_11);
        var_10 = var_12;
        if (!var_10) {
            var_13 = (var_8 >= var_num_cells);
            var_10 = var_10 || var_13;
        }
        if (var_10) {
            // return                                                                             <L 40>
            goto label1;
        }
        // old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)                                <L 42>
        // var_16 = wp::atomic_cas(var_cell_active, var_8, var_14, var_15);
        // if old_active != 1:                                                                    <L 43>
        var_18 = (var_16 != var_17);
        if (var_18) {
            // return                                                                             <L 44>
            goto label2;
        }
        // out_idx = wp.atomic_add(deleted_count, 0, 1)                                           <L 46>
        // var_21 = wp::atomic_add(var_deleted_count, var_19, var_20);
        // if out_idx < candidate_capacity:                                                       <L 47>
        var_22 = (var_21 < var_candidate_capacity);
        if (var_22) {
            // deleted_cells[out_idx] = cell_idx                                                  <L 48>
            // wp::array_store(var_deleted_cells, var_21, var_8);
        }
        // wp.atomic_add(deleted_total, 0, 1)                                                     <L 49>
        // var_25 = wp::atomic_add(var_deleted_total, var_23, var_24);
        // cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)                                     <L 51>
        var_26 = wp::address(var_cell_mass, var_8);
        var_29 = wp::div(var_27, var_28);
        var_31 = wp::load(var_26);
        var_30 = wp::mul(var_31, var_29);
        // for local_node in range(8):                                                            <L 52>
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_33 = wp::address(var_cell_nodes, var_8, var_32);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_37 = wp::atomic_sub(var_node_support_count, var_34, var_36);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_38 = wp::neg(var_30);
        // var_39 = wp::atomic_add(var_node_mass, var_34, var_38);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_41 = wp::address(var_cell_nodes, var_8, var_40);
        var_43 = wp::load(var_41);
        var_42 = wp::copy(var_43);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_45 = wp::atomic_sub(var_node_support_count, var_42, var_44);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_46 = wp::neg(var_30);
        // var_47 = wp::atomic_add(var_node_mass, var_42, var_46);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_49 = wp::address(var_cell_nodes, var_8, var_48);
        var_51 = wp::load(var_49);
        var_50 = wp::copy(var_51);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_53 = wp::atomic_sub(var_node_support_count, var_50, var_52);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_54 = wp::neg(var_30);
        // var_55 = wp::atomic_add(var_node_mass, var_50, var_54);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_57 = wp::address(var_cell_nodes, var_8, var_56);
        var_59 = wp::load(var_57);
        var_58 = wp::copy(var_59);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_61 = wp::atomic_sub(var_node_support_count, var_58, var_60);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_62 = wp::neg(var_30);
        // var_63 = wp::atomic_add(var_node_mass, var_58, var_62);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_65 = wp::address(var_cell_nodes, var_8, var_64);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_69 = wp::atomic_sub(var_node_support_count, var_66, var_68);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_70 = wp::neg(var_30);
        // var_71 = wp::atomic_add(var_node_mass, var_66, var_70);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_73 = wp::address(var_cell_nodes, var_8, var_72);
        var_75 = wp::load(var_73);
        var_74 = wp::copy(var_75);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_77 = wp::atomic_sub(var_node_support_count, var_74, var_76);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_78 = wp::neg(var_30);
        // var_79 = wp::atomic_add(var_node_mass, var_74, var_78);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_81 = wp::address(var_cell_nodes, var_8, var_80);
        var_83 = wp::load(var_81);
        var_82 = wp::copy(var_83);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_85 = wp::atomic_sub(var_node_support_count, var_82, var_84);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_86 = wp::neg(var_30);
        // var_87 = wp::atomic_add(var_node_mass, var_82, var_86);
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 53>
        var_89 = wp::address(var_cell_nodes, var_8, var_88);
        var_91 = wp::load(var_89);
        var_90 = wp::copy(var_91);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 54>
        // var_93 = wp::atomic_sub(var_node_support_count, var_90, var_92);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 55>
        var_94 = wp::neg(var_30);
        // var_95 = wp::atomic_add(var_node_mass, var_90, var_94);
        //---------
        // reverse
        wp::adj_atomic_add(var_node_mass, var_90, var_94, adj_node_mass, adj_90, adj_94, adj_95);
        wp::adj_neg(var_30, adj_30, adj_94);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_90, var_92, adj_node_support_count, adj_90, adj_92, adj_93);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_91, adj_89, adj_90);
        wp::adj_address(var_cell_nodes, var_8, var_88, adj_cell_nodes, adj_8, adj_88, adj_89);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_82, var_86, adj_node_mass, adj_82, adj_86, adj_87);
        wp::adj_neg(var_30, adj_30, adj_86);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_82, var_84, adj_node_support_count, adj_82, adj_84, adj_85);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_83, adj_81, adj_82);
        wp::adj_address(var_cell_nodes, var_8, var_80, adj_cell_nodes, adj_8, adj_80, adj_81);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_74, var_78, adj_node_mass, adj_74, adj_78, adj_79);
        wp::adj_neg(var_30, adj_30, adj_78);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_74, var_76, adj_node_support_count, adj_74, adj_76, adj_77);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_75, adj_73, adj_74);
        wp::adj_address(var_cell_nodes, var_8, var_72, adj_cell_nodes, adj_8, adj_72, adj_73);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_66, var_70, adj_node_mass, adj_66, adj_70, adj_71);
        wp::adj_neg(var_30, adj_30, adj_70);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_66, var_68, adj_node_support_count, adj_66, adj_68, adj_69);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_67, adj_65, adj_66);
        wp::adj_address(var_cell_nodes, var_8, var_64, adj_cell_nodes, adj_8, adj_64, adj_65);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_58, var_62, adj_node_mass, adj_58, adj_62, adj_63);
        wp::adj_neg(var_30, adj_30, adj_62);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_58, var_60, adj_node_support_count, adj_58, adj_60, adj_61);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_59, adj_57, adj_58);
        wp::adj_address(var_cell_nodes, var_8, var_56, adj_cell_nodes, adj_8, adj_56, adj_57);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_50, var_54, adj_node_mass, adj_50, adj_54, adj_55);
        wp::adj_neg(var_30, adj_30, adj_54);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_50, var_52, adj_node_support_count, adj_50, adj_52, adj_53);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_51, adj_49, adj_50);
        wp::adj_address(var_cell_nodes, var_8, var_48, adj_cell_nodes, adj_8, adj_48, adj_49);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_42, var_46, adj_node_mass, adj_42, adj_46, adj_47);
        wp::adj_neg(var_30, adj_30, adj_46);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_42, var_44, adj_node_support_count, adj_42, adj_44, adj_45);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_43, adj_41, adj_42);
        wp::adj_address(var_cell_nodes, var_8, var_40, adj_cell_nodes, adj_8, adj_40, adj_41);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        wp::adj_atomic_add(var_node_mass, var_34, var_38, adj_node_mass, adj_34, adj_38, adj_39);
        wp::adj_neg(var_30, adj_30, adj_38);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 55>
        wp::adj_atomic_sub(var_node_support_count, var_34, var_36, adj_node_support_count, adj_34, adj_36, adj_37);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 54>
        wp::adj_copy(var_35, adj_33, adj_34);
        wp::adj_address(var_cell_nodes, var_8, var_32, adj_cell_nodes, adj_8, adj_32, adj_33);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 53>
        // adj: for local_node in range(8):                                                       <L 52>
        wp::adj_mul(var_31, var_29, adj_26, adj_29, adj_30);
        wp::adj_div(var_27, var_28, var_29, adj_27, adj_28, adj_29);
        wp::adj_address(var_cell_mass, var_8, adj_cell_mass, adj_8, adj_26);
        // adj: cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)                                <L 51>
        wp::adj_atomic_add(var_deleted_total, var_23, var_24, adj_deleted_total, adj_23, adj_24, adj_25);
        // adj: wp.atomic_add(deleted_total, 0, 1)                                                <L 49>
        if (var_22) {
            wp::adj_array_store(var_deleted_cells, var_21, var_8, adj_deleted_cells, adj_21, adj_8);
            // adj: deleted_cells[out_idx] = cell_idx                                             <L 48>
        }
        // adj: if out_idx < candidate_capacity:                                                  <L 47>
        wp::adj_atomic_add(var_deleted_count, var_19, var_20, adj_deleted_count, adj_19, adj_20, adj_21);
        // adj: out_idx = wp.atomic_add(deleted_count, 0, 1)                                      <L 46>
        if (var_18) {
            label2:;
            // adj: return                                                                        <L 44>
        }
        // adj: if old_active != 1:                                                               <L 43>
        // adj: old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)                           <L 42>
        if (var_10) {
            label1:;
            // adj: return                                                                        <L 40>
        }
        if (!var_10) {
        }
        // adj: if cell_idx < 0 or cell_idx >= num_cells:                                         <L 39>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_cell_ids, var_0, adj_cell_ids, adj_0, adj_7);
        // adj: cell_idx = cell_ids[tid]                                                          <L 38>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 36>
        }
        if (!var_1) {
            wp::adj_address(var_candidate_count, var_3, adj_candidate_count, adj_3, adj_4);
        }
        // adj: if tid >= candidate_capacity or tid >= candidate_count[0]:                        <L 35>
        // adj: tid = wp.tid()                                                                    <L 34>
        // adj: def delete_cells_sparse_kernel(                                                   <L 13>
        continue;
    }
}



extern "C" __global__ void deactivate_deleted_cell_clusters_kernel_2a33aef6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_to_cluster,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::int32> var_deactivated_cluster_ids,
    wp::array_t<wp::int32> var_deactivated_count)
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
        const wp::int32 var_17 = 1;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        bool var_21;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        bool var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        //---------
        // forward
        // def deactivate_deleted_cell_clusters_kernel(                                           <L 205>
        // i = wp.tid()                                                                           <L 218>
        var_0 = builtin_tid1d();
        // if i >= deleted_count[0]:                                                              <L 219>
        var_2 = wp::address(var_deleted_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 220>
            continue;
        }
        // cell_idx = deleted_cells[i]                                                            <L 222>
        var_5 = wp::address(var_deleted_cells, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 223>
        var_10 = (var_6 < var_9);
        var_8 = var_10;
        if (!var_8) {
            var_11 = (var_6 >= var_num_cells);
            var_8 = var_8 || var_11;
        }
        if (var_8) {
            // return                                                                             <L 224>
            continue;
        }
        // cluster_idx = cell_to_cluster[cell_idx]                                                <L 226>
        var_12 = wp::address(var_cell_to_cluster, var_6);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if cluster_idx < 0:                                                                    <L 227>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 228>
            continue;
        }
        // old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)                          <L 230>
        var_19 = wp::atomic_cas(var_cluster_active, var_13, var_17, var_18);
        // if old_active != 1:                                                                    <L 231>
        var_21 = (var_19 != var_20);
        if (var_21) {
            // return                                                                             <L 232>
            continue;
        }
        // out_idx = wp.atomic_add(deactivated_count, 0, 1)                                       <L 234>
        var_24 = wp::atomic_add(var_deactivated_count, var_22, var_23);
        // if out_idx < deactivated_cluster_ids.shape[0]:                                         <L 235>
        var_25 = &(var_deactivated_cluster_ids.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = (var_24 < var_27);
        if (var_29) {
            // deactivated_cluster_ids[out_idx] = cluster_idx                                     <L 236>
            wp::array_store(var_deactivated_cluster_ids, var_24, var_13);
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 238>
        var_30 = wp::address(var_cluster_offsets, var_13);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 239>
        var_34 = wp::add(var_13, var_33);
        var_35 = wp::address(var_cluster_offsets, var_34);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // while cursor < end:                                                                    <L 240>
        start_while_4:;
        var_38 = (var_31 < var_36);
        if ((var_38) == false) goto end_while_4;
            // particle_idx = cluster_indices[cursor]                                             <L 241>
            var_39 = wp::address(var_cluster_indices, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // if particle_idx >= 0:                                                              <L 242>
            var_43 = (var_40 >= var_42);
            if (var_43) {
                // wp.atomic_sub(particle_cluster_counts, particle_idx, 1)                        <L 243>
                var_45 = wp::atomic_sub(var_particle_cluster_counts, var_40, var_44);
            }
            // cursor += 1                                                                        <L 244>
            var_47 = wp::add(var_31, var_46);
            wp::assign(var_31, var_47);
        goto start_while_4;
        end_while_4:;
    }
}



extern "C" __global__ void deactivate_deleted_cell_clusters_kernel_2a33aef6_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_to_cluster,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::int32> var_deactivated_cluster_ids,
    wp::array_t<wp::int32> var_deactivated_count,
    wp::array_t<wp::int32> adj_deleted_cells,
    wp::array_t<wp::int32> adj_deleted_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_to_cluster,
    wp::array_t<wp::int32> adj_cluster_active,
    wp::array_t<wp::int32> adj_cluster_offsets,
    wp::array_t<wp::int32> adj_cluster_indices,
    wp::array_t<wp::int32> adj_particle_cluster_counts,
    wp::array_t<wp::int32> adj_deactivated_cluster_ids,
    wp::array_t<wp::int32> adj_deactivated_count)
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
        const wp::int32 var_17 = 1;
        const wp::int32 var_18 = 0;
        wp::int32 var_19;
        const wp::int32 var_20 = 1;
        bool var_21;
        const wp::int32 var_22 = 0;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::shape_t* var_25;
        const wp::int32 var_26 = 0;
        wp::int32 var_27;
        wp::shape_t var_28;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        bool var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        bool adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        bool adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        bool adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::shape_t adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::shape_t adj_28 = {};
        bool adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        bool adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        //---------
        // forward
        // def deactivate_deleted_cell_clusters_kernel(                                           <L 205>
        // i = wp.tid()                                                                           <L 218>
        var_0 = builtin_tid1d();
        // if i >= deleted_count[0]:                                                              <L 219>
        var_2 = wp::address(var_deleted_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 220>
            goto label0;
        }
        // cell_idx = deleted_cells[i]                                                            <L 222>
        var_5 = wp::address(var_deleted_cells, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 223>
        var_10 = (var_6 < var_9);
        var_8 = var_10;
        if (!var_8) {
            var_11 = (var_6 >= var_num_cells);
            var_8 = var_8 || var_11;
        }
        if (var_8) {
            // return                                                                             <L 224>
            goto label1;
        }
        // cluster_idx = cell_to_cluster[cell_idx]                                                <L 226>
        var_12 = wp::address(var_cell_to_cluster, var_6);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // if cluster_idx < 0:                                                                    <L 227>
        var_16 = (var_13 < var_15);
        if (var_16) {
            // return                                                                             <L 228>
            goto label2;
        }
        // old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)                          <L 230>
        // var_19 = wp::atomic_cas(var_cluster_active, var_13, var_17, var_18);
        // if old_active != 1:                                                                    <L 231>
        var_21 = (var_19 != var_20);
        if (var_21) {
            // return                                                                             <L 232>
            goto label3;
        }
        // out_idx = wp.atomic_add(deactivated_count, 0, 1)                                       <L 234>
        // var_24 = wp::atomic_add(var_deactivated_count, var_22, var_23);
        // if out_idx < deactivated_cluster_ids.shape[0]:                                         <L 235>
        var_25 = &(var_deactivated_cluster_ids.shape);
        var_28 = wp::load(var_25);
        var_27 = wp::extract(var_28, var_26);
        var_29 = (var_24 < var_27);
        if (var_29) {
            // deactivated_cluster_ids[out_idx] = cluster_idx                                     <L 236>
            // wp::array_store(var_deactivated_cluster_ids, var_24, var_13);
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 238>
        var_30 = wp::address(var_cluster_offsets, var_13);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 239>
        var_34 = wp::add(var_13, var_33);
        var_35 = wp::address(var_cluster_offsets, var_34);
        var_37 = wp::load(var_35);
        var_36 = wp::copy(var_37);
        // while cursor < end:                                                                    <L 240>
        //---------
        // reverse
        start_while_4:;
        var_38 = (var_31 < var_36);
        if ((var_38) == false) goto end_while_4;
        adj_39 = {};
        adj_40 = {};
        adj_41 = {};
        adj_42 = {};
        adj_43 = {};
        adj_44 = {};
        adj_45 = {};
        adj_46 = {};
        adj_47 = {};
            // particle_idx = cluster_indices[cursor]                                             <L 241>
            var_39 = wp::address(var_cluster_indices, var_31);
            var_41 = wp::load(var_39);
            var_40 = wp::copy(var_41);
            // if particle_idx >= 0:                                                              <L 242>
            var_43 = (var_40 >= var_42);
            if (var_43) {
                // wp.atomic_sub(particle_cluster_counts, particle_idx, 1)                        <L 243>
                // var_45 = wp::atomic_sub(var_particle_cluster_counts, var_40, var_44);
            }
            // cursor += 1                                                                        <L 244>
            var_47 = wp::add(var_31, var_46);
            wp::assign(var_31, var_47);
            wp::adj_assign(var_31, var_47, adj_31, adj_47);
            wp::adj_add(var_31, var_46, adj_31, adj_46, adj_47);
            // adj: cursor += 1                                                                   <L 244>
            if (var_43) {
                wp::adj_atomic_sub(var_particle_cluster_counts, var_40, var_44, adj_particle_cluster_counts, adj_40, adj_44, adj_45);
                // adj: wp.atomic_sub(particle_cluster_counts, particle_idx, 1)                   <L 243>
            }
            // adj: if particle_idx >= 0:                                                         <L 242>
            wp::adj_copy(var_41, adj_39, adj_40);
            wp::adj_address(var_cluster_indices, var_31, adj_cluster_indices, adj_31, adj_39);
            // adj: particle_idx = cluster_indices[cursor]                                        <L 241>
        goto start_while_4;
        end_while_4:;
        // adj: while cursor < end:                                                               <L 240>
        wp::adj_copy(var_37, adj_35, adj_36);
        wp::adj_address(var_cluster_offsets, var_34, adj_cluster_offsets, adj_34, adj_35);
        wp::adj_add(var_13, var_33, adj_13, adj_33, adj_34);
        // adj: end = cluster_offsets[cluster_idx + 1]                                            <L 239>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_cluster_offsets, var_13, adj_cluster_offsets, adj_13, adj_30);
        // adj: cursor = cluster_offsets[cluster_idx]                                             <L 238>
        if (var_29) {
            wp::adj_array_store(var_deactivated_cluster_ids, var_24, var_13, adj_deactivated_cluster_ids, adj_24, adj_13);
            // adj: deactivated_cluster_ids[out_idx] = cluster_idx                                <L 236>
        }
        wp::adj_extract(var_28, var_26, adj_25, adj_26, adj_27);
        adj_deactivated_cluster_ids.shape = adj_25;
        // adj: if out_idx < deactivated_cluster_ids.shape[0]:                                    <L 235>
        wp::adj_atomic_add(var_deactivated_count, var_22, var_23, adj_deactivated_count, adj_22, adj_23, adj_24);
        // adj: out_idx = wp.atomic_add(deactivated_count, 0, 1)                                  <L 234>
        if (var_21) {
            label3:;
            // adj: return                                                                        <L 232>
        }
        // adj: if old_active != 1:                                                               <L 231>
        // adj: old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)                     <L 230>
        if (var_16) {
            label2:;
            // adj: return                                                                        <L 228>
        }
        // adj: if cluster_idx < 0:                                                               <L 227>
        wp::adj_copy(var_14, adj_12, adj_13);
        wp::adj_address(var_cell_to_cluster, var_6, adj_cell_to_cluster, adj_6, adj_12);
        // adj: cluster_idx = cell_to_cluster[cell_idx]                                           <L 226>
        if (var_8) {
            label1:;
            // adj: return                                                                        <L 224>
        }
        if (!var_8) {
        }
        // adj: if cell_idx < 0 or cell_idx >= num_cells:                                         <L 223>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_deleted_cells, var_0, adj_deleted_cells, adj_0, adj_5);
        // adj: cell_idx = deleted_cells[i]                                                       <L 222>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 220>
        }
        wp::adj_address(var_deleted_count, var_1, adj_deleted_count, adj_1, adj_2);
        // adj: if i >= deleted_count[0]:                                                         <L 219>
        // adj: i = wp.tid()                                                                      <L 218>
        // adj: def deactivate_deleted_cell_clusters_kernel(                                      <L 205>
        continue;
    }
}



extern "C" __global__ void select_sphere_particle_contact_cells_kernel_33f06896_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q,
    wp::array_t<wp::int32> var_sphere_cut_enabled,
    wp::int32 var_sphere_count,
    wp::float32 var_sphere_radius,
    wp::array_t<wp::int32> var_selected_cells,
    wp::array_t<wp::int32> var_selected_count)
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
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 8;
        wp::range_t var_16;
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
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        bool var_44;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        //---------
        // forward
        // def select_sphere_particle_contact_cells_kernel(                                       <L 540>
        // c = wp.tid()                                                                           <L 558>
        var_0 = builtin_tid1d();
        // if c >= num_cells:                                                                     <L 559>
        var_1 = (var_0 >= var_num_cells);
        if (var_1) {
            // return                                                                             <L 560>
            continue;
        }
        // if cell_active[c] == 0:                                                                <L 561>
        var_2 = wp::address(var_cell_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // return                                                                             <L 562>
            continue;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 563>
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
            // return                                                                             <L 564>
            continue;
        }
        // for local_node in range(8):                                                            <L 566>
        var_16 = wp::range(var_15);
        start_for_3:;
            if (iter_cmp(var_16) == 0) goto end_for_3;
            var_17 = wp::iter_next(var_16);
            // node_idx = cell_nodes[c, local_node]                                               <L 567>
            var_18 = wp::address(var_cell_nodes, var_0, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // if (particle_flags[node_idx] & _ACTIVE_BIT) == 0:                                  <L 568>
            var_21 = wp::address(var_particle_flags, var_19);
            var_24 = wp::load(var_21);
            var_23 = wp::bit_and(var_24, var_22);
            var_26 = (var_23 == var_25);
            if (var_26) {
                // continue                                                                       <L 569>
                goto start_for_3;
            }
            // q = particle_q[node_idx]                                                           <L 570>
            var_27 = wp::address(var_particle_q, var_19);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // for sphere_idx in range(sphere_count):                                             <L 571>
            var_30 = wp::range(var_sphere_count);
            start_for_5:;
                if (iter_cmp(var_30) == 0) goto end_for_5;
                var_31 = wp::iter_next(var_30);
                // if sphere_cut_enabled[sphere_idx] == 0:                                        <L 572>
                var_32 = wp::address(var_sphere_cut_enabled, var_31);
                var_35 = wp::load(var_32);
                var_34 = (var_35 == var_33);
                if (var_34) {
                    // continue                                                                   <L 573>
                    goto start_for_5;
                }
                // radius = sphere_radius + particle_radius[node_idx]                             <L 574>
                var_36 = wp::address(var_particle_radius, var_19);
                var_38 = wp::load(var_36);
                var_37 = wp::add(var_sphere_radius, var_38);
                // delta = q - sphere_q[sphere_idx]                                               <L 575>
                var_39 = wp::address(var_sphere_q, var_31);
                var_41 = wp::load(var_39);
                var_40 = wp::sub(var_28, var_41);
                // if wp.dot(delta, delta) <= radius * radius:                                    <L 576>
                var_42 = wp::dot(var_40, var_40);
                var_43 = wp::mul(var_37, var_37);
                var_44 = (var_42 <= var_43);
                if (var_44) {
                    // out_idx = wp.atomic_add(selected_count, 0, 1)                              <L 577>
                    var_47 = wp::atomic_add(var_selected_count, var_45, var_46);
                    // selected_cells[out_idx] = c                                                <L 578>
                    wp::array_store(var_selected_cells, var_47, var_0);
                    // return                                                                     <L 579>
                    continue;
                }
                goto start_for_5;
            end_for_5:;
            goto start_for_3;
        end_for_3:;
    }
}



extern "C" __global__ void select_sphere_particle_contact_cells_kernel_33f06896_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::float32> var_particle_radius,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_sphere_q,
    wp::array_t<wp::int32> var_sphere_cut_enabled,
    wp::int32 var_sphere_count,
    wp::float32 var_sphere_radius,
    wp::array_t<wp::int32> var_selected_cells,
    wp::array_t<wp::int32> var_selected_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_cell_material,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_material_cuttable,
    wp::int32 adj_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::float32> adj_particle_radius,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_sphere_q,
    wp::array_t<wp::int32> adj_sphere_cut_enabled,
    wp::int32 adj_sphere_count,
    wp::float32 adj_sphere_radius,
    wp::array_t<wp::int32> adj_selected_cells,
    wp::array_t<wp::int32> adj_selected_count)
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
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32* var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 8;
        wp::range_t var_16;
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
        wp::range_t var_30;
        wp::int32 var_31;
        wp::int32* var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        bool var_44;
        const wp::int32 var_45 = 0;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
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
        wp::int32 adj_15 = {};
        wp::range_t adj_16 = {};
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
        wp::range_t adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        bool adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        bool adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        //---------
        // forward
        // def select_sphere_particle_contact_cells_kernel(                                       <L 540>
        // c = wp.tid()                                                                           <L 558>
        var_0 = builtin_tid1d();
        // if c >= num_cells:                                                                     <L 559>
        var_1 = (var_0 >= var_num_cells);
        if (var_1) {
            // return                                                                             <L 560>
            goto label0;
        }
        // if cell_active[c] == 0:                                                                <L 561>
        var_2 = wp::address(var_cell_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // return                                                                             <L 562>
            goto label1;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 563>
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
            // return                                                                             <L 564>
            goto label2;
        }
        // for local_node in range(8):                                                            <L 566>
        var_16 = wp::range(var_15);
        //---------
        // reverse
        var_16 = wp::iter_reverse(var_16);
        start_for_3:;
            if (iter_cmp(var_16) == 0) goto end_for_3;
            var_17 = wp::iter_next(var_16);
        	adj_18 = {};
        	adj_19 = {};
        	adj_20 = {};
        	adj_21 = {};
        	adj_22 = {};
        	adj_23 = {};
        	adj_24 = {};
        	adj_25 = {};
        	adj_26 = {};
        	adj_27 = {};
        	adj_28 = {};
        	adj_29 = {};
        	adj_30 = {};
            // node_idx = cell_nodes[c, local_node]                                               <L 567>
            var_18 = wp::address(var_cell_nodes, var_0, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // if (particle_flags[node_idx] & _ACTIVE_BIT) == 0:                                  <L 568>
            var_21 = wp::address(var_particle_flags, var_19);
            var_24 = wp::load(var_21);
            var_23 = wp::bit_and(var_24, var_22);
            var_26 = (var_23 == var_25);
            if (var_26) {
                // continue                                                                       <L 569>
                goto start_for_3;
            }
            // q = particle_q[node_idx]                                                           <L 570>
            var_27 = wp::address(var_particle_q, var_19);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // for sphere_idx in range(sphere_count):                                             <L 571>
            var_30 = wp::range(var_sphere_count);
            var_30 = wp::iter_reverse(var_30);
            start_for_5:;
                if (iter_cmp(var_30) == 0) goto end_for_5;
                var_31 = wp::iter_next(var_30);
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
            	adj_45 = {};
            	adj_46 = {};
            	adj_47 = {};
                // if sphere_cut_enabled[sphere_idx] == 0:                                        <L 572>
                var_32 = wp::address(var_sphere_cut_enabled, var_31);
                var_35 = wp::load(var_32);
                var_34 = (var_35 == var_33);
                if (var_34) {
                    // continue                                                                   <L 573>
                    goto start_for_5;
                }
                // radius = sphere_radius + particle_radius[node_idx]                             <L 574>
                var_36 = wp::address(var_particle_radius, var_19);
                var_38 = wp::load(var_36);
                var_37 = wp::add(var_sphere_radius, var_38);
                // delta = q - sphere_q[sphere_idx]                                               <L 575>
                var_39 = wp::address(var_sphere_q, var_31);
                var_41 = wp::load(var_39);
                var_40 = wp::sub(var_28, var_41);
                // if wp.dot(delta, delta) <= radius * radius:                                    <L 576>
                var_42 = wp::dot(var_40, var_40);
                var_43 = wp::mul(var_37, var_37);
                var_44 = (var_42 <= var_43);
                if (var_44) {
                    // out_idx = wp.atomic_add(selected_count, 0, 1)                              <L 577>
                    // var_47 = wp::atomic_add(var_selected_count, var_45, var_46);
                    // selected_cells[out_idx] = c                                                <L 578>
                    // wp::array_store(var_selected_cells, var_47, var_0);
                    // return                                                                     <L 579>
                    goto label7;
                }
                if (var_44) {
                    label7:;
                    // adj: return                                                                <L 579>
                    wp::adj_array_store(var_selected_cells, var_47, var_0, adj_selected_cells, adj_47, adj_0);
                    // adj: selected_cells[out_idx] = c                                           <L 578>
                    wp::adj_atomic_add(var_selected_count, var_45, var_46, adj_selected_count, adj_45, adj_46, adj_47);
                    // adj: out_idx = wp.atomic_add(selected_count, 0, 1)                         <L 577>
                }
                wp::adj_mul(var_37, var_37, adj_37, adj_37, adj_43);
                wp::adj_dot(var_40, var_40, adj_40, adj_40, adj_42);
                // adj: if wp.dot(delta, delta) <= radius * radius:                               <L 576>
                wp::adj_sub(var_28, var_41, adj_28, adj_39, adj_40);
                wp::adj_address(var_sphere_q, var_31, adj_sphere_q, adj_31, adj_39);
                // adj: delta = q - sphere_q[sphere_idx]                                          <L 575>
                wp::adj_add(var_sphere_radius, var_38, adj_sphere_radius, adj_36, adj_37);
                wp::adj_address(var_particle_radius, var_19, adj_particle_radius, adj_19, adj_36);
                // adj: radius = sphere_radius + particle_radius[node_idx]                        <L 574>
                if (var_34) {
                    // adj: continue                                                              <L 573>
                }
                wp::adj_address(var_sphere_cut_enabled, var_31, adj_sphere_cut_enabled, adj_31, adj_32);
                // adj: if sphere_cut_enabled[sphere_idx] == 0:                                   <L 572>
            	goto start_for_5;
            end_for_5:;
            wp::adj_range(var_sphere_count, adj_sphere_count, adj_30);
            // adj: for sphere_idx in range(sphere_count):                                        <L 571>
            wp::adj_copy(var_29, adj_27, adj_28);
            wp::adj_address(var_particle_q, var_19, adj_particle_q, adj_19, adj_27);
            // adj: q = particle_q[node_idx]                                                      <L 570>
            if (var_26) {
                // adj: continue                                                                  <L 569>
            }
            wp::adj_address(var_particle_flags, var_19, adj_particle_flags, adj_19, adj_21);
            // adj: if (particle_flags[node_idx] & _ACTIVE_BIT) == 0:                             <L 568>
            wp::adj_copy(var_20, adj_18, adj_19);
            wp::adj_address(var_cell_nodes, var_0, var_17, adj_cell_nodes, adj_0, adj_17, adj_18);
            // adj: node_idx = cell_nodes[c, local_node]                                          <L 567>
        	goto start_for_3;
        end_for_3:;
        wp::adj_range(var_15, adj_15, adj_16);
        // adj: for local_node in range(8):                                                       <L 566>
        if (var_6) {
            label2:;
            // adj: return                                                                        <L 564>
        }
        if (var_6) {
            wp::adj_address(var_material_cuttable, var_11, adj_material_cuttable, adj_9, adj_10);
            wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_9);
        }
        // adj: if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:         <L 563>
        if (var_4) {
            label1:;
            // adj: return                                                                        <L 562>
        }
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
        // adj: if cell_active[c] == 0:                                                           <L 561>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 560>
        }
        // adj: if c >= num_cells:                                                                <L 559>
        // adj: c = wp.tid()                                                                      <L 558>
        // adj: def select_sphere_particle_contact_cells_kernel(                                  <L 540>
        continue;
    }
}



extern "C" __global__ void validate_cluster_state_kernel_98ca18a5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::array_t<wp::int32> var_error_count,
    wp::array_t<wp::int32> var_first_error)
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
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        bool var_18;
        const wp::float32 var_19 = -1e-06;
        bool var_20;
        bool var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::shape_t* var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::shape_t var_35;
        bool var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 0;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        const wp::int32 var_55 = 4000;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        //---------
        // forward
        // def validate_cluster_state_kernel(                                                     <L 333>
        // i = wp.tid()                                                                           <L 340>
        var_0 = builtin_tid1d();
        // bad = int(0)                                                                           <L 341>
        var_2 = wp::int(var_1);
        // if i < particle_cluster_counts.shape[0]:                                               <L 343>
        var_3 = &(var_particle_cluster_counts.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = (var_0 < var_5);
        if (var_7) {
            // count = particle_cluster_counts[i]                                                 <L 344>
            var_8 = wp::address(var_particle_cluster_counts, var_0);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // inv_weight = particle_cluster_inv_weights[i]                                       <L 345>
            var_11 = wp::address(var_particle_cluster_inv_weights, var_0);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if count < 0:                                                                      <L 346>
            var_15 = (var_9 < var_14);
            if (var_15) {
                // bad = 1                                                                        <L 347>
            }
            var_17 = wp::where(var_15, var_16, var_2);
            // if inv_weight < -1.0e-6 or inv_weight != inv_weight:                               <L 348>
            var_20 = (var_12 < var_19);
            var_18 = var_20;
            if (!var_18) {
                var_21 = (var_12 != var_12);
                var_18 = var_18 || var_21;
            }
            if (var_18) {
                // bad = 1                                                                        <L 349>
            }
            var_23 = wp::where(var_18, var_22, var_17);
            // if count <= 0 and inv_weight != 0.0:                                               <L 350>
            var_26 = (var_9 <= var_25);
            var_24 = var_26;
            if (var_24) {
                var_28 = (var_12 != var_27);
                var_24 = var_24 && var_28;
            }
            if (var_24) {
                // bad = 1                                                                        <L 351>
            }
            var_30 = wp::where(var_24, var_29, var_23);
        }
        var_31 = wp::where(var_7, var_30, var_2);
        // if i < cluster_active.shape[0]:                                                        <L 353>
        var_32 = &(var_cluster_active.shape);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = (var_0 < var_34);
        if (var_36) {
            // active = cluster_active[i]                                                         <L 354>
            var_37 = wp::address(var_cluster_active, var_0);
            var_39 = wp::load(var_37);
            var_38 = wp::copy(var_39);
            // if active != 0 and active != 1:                                                    <L 355>
            var_42 = (var_38 != var_41);
            var_40 = var_42;
            if (var_40) {
                var_44 = (var_38 != var_43);
                var_40 = var_40 && var_44;
            }
            if (var_40) {
                // bad = 1                                                                        <L 356>
            }
            var_46 = wp::where(var_40, var_45, var_31);
        }
        var_47 = wp::where(var_36, var_46, var_31);
        // if bad != 0:                                                                           <L 358>
        var_49 = (var_47 != var_48);
        if (var_49) {
            // old = wp.atomic_add(error_count, 0, 1)                                             <L 359>
            var_52 = wp::atomic_add(var_error_count, var_50, var_51);
            // if old == 0:                                                                       <L 360>
            var_54 = (var_52 == var_53);
            if (var_54) {
                // first_error[0] = 4000 + i                                                      <L 361>
                var_56 = wp::add(var_55, var_0);
                wp::array_store(var_first_error, var_57, var_56);
            }
        }
    }
}



extern "C" __global__ void validate_cluster_state_kernel_98ca18a5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::array_t<wp::int32> var_error_count,
    wp::array_t<wp::int32> var_first_error,
    wp::array_t<wp::int32> adj_cluster_active,
    wp::array_t<wp::int32> adj_particle_cluster_counts,
    wp::array_t<wp::float32> adj_particle_cluster_inv_weights,
    wp::array_t<wp::int32> adj_error_count,
    wp::array_t<wp::int32> adj_first_error)
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
        wp::shape_t* var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::shape_t var_6;
        bool var_7;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::float32* var_11;
        wp::float32 var_12;
        wp::float32 var_13;
        const wp::int32 var_14 = 0;
        bool var_15;
        const wp::int32 var_16 = 1;
        wp::int32 var_17;
        bool var_18;
        const wp::float32 var_19 = -1e-06;
        bool var_20;
        bool var_21;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::float32 var_27 = 0.0;
        bool var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32 var_31;
        wp::shape_t* var_32;
        const wp::int32 var_33 = 0;
        wp::int32 var_34;
        wp::shape_t var_35;
        bool var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        const wp::int32 var_43 = 1;
        bool var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        const wp::int32 var_48 = 0;
        bool var_49;
        const wp::int32 var_50 = 0;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        const wp::int32 var_55 = 4000;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::shape_t adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::shape_t adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        bool adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        bool adj_18 = {};
        wp::float32 adj_19 = {};
        bool adj_20 = {};
        bool adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        bool adj_26 = {};
        wp::float32 adj_27 = {};
        bool adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::shape_t adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::shape_t adj_35 = {};
        bool adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        bool adj_40 = {};
        wp::int32 adj_41 = {};
        bool adj_42 = {};
        wp::int32 adj_43 = {};
        bool adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        bool adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        bool adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        //---------
        // forward
        // def validate_cluster_state_kernel(                                                     <L 333>
        // i = wp.tid()                                                                           <L 340>
        var_0 = builtin_tid1d();
        // bad = int(0)                                                                           <L 341>
        var_2 = wp::int(var_1);
        // if i < particle_cluster_counts.shape[0]:                                               <L 343>
        var_3 = &(var_particle_cluster_counts.shape);
        var_6 = wp::load(var_3);
        var_5 = wp::extract(var_6, var_4);
        var_7 = (var_0 < var_5);
        if (var_7) {
            // count = particle_cluster_counts[i]                                                 <L 344>
            var_8 = wp::address(var_particle_cluster_counts, var_0);
            var_10 = wp::load(var_8);
            var_9 = wp::copy(var_10);
            // inv_weight = particle_cluster_inv_weights[i]                                       <L 345>
            var_11 = wp::address(var_particle_cluster_inv_weights, var_0);
            var_13 = wp::load(var_11);
            var_12 = wp::copy(var_13);
            // if count < 0:                                                                      <L 346>
            var_15 = (var_9 < var_14);
            if (var_15) {
                // bad = 1                                                                        <L 347>
            }
            var_17 = wp::where(var_15, var_16, var_2);
            // if inv_weight < -1.0e-6 or inv_weight != inv_weight:                               <L 348>
            var_20 = (var_12 < var_19);
            var_18 = var_20;
            if (!var_18) {
                var_21 = (var_12 != var_12);
                var_18 = var_18 || var_21;
            }
            if (var_18) {
                // bad = 1                                                                        <L 349>
            }
            var_23 = wp::where(var_18, var_22, var_17);
            // if count <= 0 and inv_weight != 0.0:                                               <L 350>
            var_26 = (var_9 <= var_25);
            var_24 = var_26;
            if (var_24) {
                var_28 = (var_12 != var_27);
                var_24 = var_24 && var_28;
            }
            if (var_24) {
                // bad = 1                                                                        <L 351>
            }
            var_30 = wp::where(var_24, var_29, var_23);
        }
        var_31 = wp::where(var_7, var_30, var_2);
        // if i < cluster_active.shape[0]:                                                        <L 353>
        var_32 = &(var_cluster_active.shape);
        var_35 = wp::load(var_32);
        var_34 = wp::extract(var_35, var_33);
        var_36 = (var_0 < var_34);
        if (var_36) {
            // active = cluster_active[i]                                                         <L 354>
            var_37 = wp::address(var_cluster_active, var_0);
            var_39 = wp::load(var_37);
            var_38 = wp::copy(var_39);
            // if active != 0 and active != 1:                                                    <L 355>
            var_42 = (var_38 != var_41);
            var_40 = var_42;
            if (var_40) {
                var_44 = (var_38 != var_43);
                var_40 = var_40 && var_44;
            }
            if (var_40) {
                // bad = 1                                                                        <L 356>
            }
            var_46 = wp::where(var_40, var_45, var_31);
        }
        var_47 = wp::where(var_36, var_46, var_31);
        // if bad != 0:                                                                           <L 358>
        var_49 = (var_47 != var_48);
        if (var_49) {
            // old = wp.atomic_add(error_count, 0, 1)                                             <L 359>
            // var_52 = wp::atomic_add(var_error_count, var_50, var_51);
            // if old == 0:                                                                       <L 360>
            var_54 = (var_52 == var_53);
            if (var_54) {
                // first_error[0] = 4000 + i                                                      <L 361>
                var_56 = wp::add(var_55, var_0);
                // wp::array_store(var_first_error, var_57, var_56);
            }
        }
        //---------
        // reverse
        if (var_49) {
            if (var_54) {
                wp::adj_array_store(var_first_error, var_57, var_56, adj_first_error, adj_57, adj_56);
                wp::adj_add(var_55, var_0, adj_55, adj_0, adj_56);
                // adj: first_error[0] = 4000 + i                                                 <L 361>
            }
            // adj: if old == 0:                                                                  <L 360>
            wp::adj_atomic_add(var_error_count, var_50, var_51, adj_error_count, adj_50, adj_51, adj_52);
            // adj: old = wp.atomic_add(error_count, 0, 1)                                        <L 359>
        }
        // adj: if bad != 0:                                                                      <L 358>
        wp::adj_where(var_36, var_46, var_31, adj_36, adj_46, adj_31, adj_47);
        if (var_36) {
            wp::adj_where(var_40, var_45, var_31, adj_40, adj_45, adj_31, adj_46);
            if (var_40) {
                // adj: bad = 1                                                                   <L 356>
            }
            if (var_40) {
            }
            // adj: if active != 0 and active != 1:                                               <L 355>
            wp::adj_copy(var_39, adj_37, adj_38);
            wp::adj_address(var_cluster_active, var_0, adj_cluster_active, adj_0, adj_37);
            // adj: active = cluster_active[i]                                                    <L 354>
        }
        wp::adj_extract(var_35, var_33, adj_32, adj_33, adj_34);
        adj_cluster_active.shape = adj_32;
        // adj: if i < cluster_active.shape[0]:                                                   <L 353>
        wp::adj_where(var_7, var_30, var_2, adj_7, adj_30, adj_2, adj_31);
        if (var_7) {
            wp::adj_where(var_24, var_29, var_23, adj_24, adj_29, adj_23, adj_30);
            if (var_24) {
                // adj: bad = 1                                                                   <L 351>
            }
            if (var_24) {
            }
            // adj: if count <= 0 and inv_weight != 0.0:                                          <L 350>
            wp::adj_where(var_18, var_22, var_17, adj_18, adj_22, adj_17, adj_23);
            if (var_18) {
                // adj: bad = 1                                                                   <L 349>
            }
            if (!var_18) {
            }
            // adj: if inv_weight < -1.0e-6 or inv_weight != inv_weight:                          <L 348>
            wp::adj_where(var_15, var_16, var_2, adj_15, adj_16, adj_2, adj_17);
            if (var_15) {
                // adj: bad = 1                                                                   <L 347>
            }
            // adj: if count < 0:                                                                 <L 346>
            wp::adj_copy(var_13, adj_11, adj_12);
            wp::adj_address(var_particle_cluster_inv_weights, var_0, adj_particle_cluster_inv_weights, adj_0, adj_11);
            // adj: inv_weight = particle_cluster_inv_weights[i]                                  <L 345>
            wp::adj_copy(var_10, adj_8, adj_9);
            wp::adj_address(var_particle_cluster_counts, var_0, adj_particle_cluster_counts, adj_0, adj_8);
            // adj: count = particle_cluster_counts[i]                                            <L 344>
        }
        wp::adj_extract(var_6, var_4, adj_3, adj_4, adj_5);
        adj_particle_cluster_counts.shape = adj_3;
        // adj: if i < particle_cluster_counts.shape[0]:                                          <L 343>
        wp::adj_int(var_1, adj_1, adj_2);
        // adj: bad = int(0)                                                                      <L 341>
        // adj: i = wp.tid()                                                                      <L 340>
        // adj: def validate_cluster_state_kernel(                                                <L 333>
        continue;
    }
}



extern "C" __global__ void select_ray_segment_cells_kernel_6e51a20c_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_start_cell_ids,
    wp::vec_t<3, wp::float32> var_ray_dir,
    wp::float32 var_depth,
    wp::float32 var_padding,
    wp::array_t<wp::int32> var_selected_cells,
    wp::array_t<wp::int32> var_selected_count)
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
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        bool var_9;
        wp::int32* var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32 var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 0.0;
        const wp::float32 var_29 = 0.0;
        wp::vec_t<3, wp::float32> var_30;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.0;
        wp::vec_t<3, wp::float32> var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::int32* var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        const wp::int32 var_46 = 1;
        wp::int32* var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::int32* var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::int32 var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        const wp::int32 var_57 = 2;
        wp::int32* var_58;
        wp::vec_t<3, wp::float32>* var_59;
        wp::int32 var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::int32* var_63;
        wp::vec_t<3, wp::float32>* var_64;
        wp::int32 var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::int32 var_68 = 3;
        wp::int32* var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::int32* var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        const wp::int32 var_79 = 4;
        wp::int32* var_80;
        wp::vec_t<3, wp::float32>* var_81;
        wp::int32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::int32* var_85;
        wp::vec_t<3, wp::float32>* var_86;
        wp::int32 var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 5;
        wp::int32* var_91;
        wp::vec_t<3, wp::float32>* var_92;
        wp::int32 var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::int32* var_96;
        wp::vec_t<3, wp::float32>* var_97;
        wp::int32 var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        const wp::int32 var_101 = 6;
        wp::int32* var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::int32 var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::int32* var_107;
        wp::vec_t<3, wp::float32>* var_108;
        wp::int32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        const wp::int32 var_112 = 7;
        wp::int32* var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::int32* var_118;
        wp::vec_t<3, wp::float32>* var_119;
        wp::int32 var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        const wp::float32 var_123 = 0.125;
        wp::vec_t<3, wp::float32> var_124;
        const wp::float32 var_125 = 0.125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        bool var_131;
        const wp::int32 var_132 = 0;
        const wp::int32 var_133 = 1;
        wp::int32 var_134;
        //---------
        // forward
        // def select_ray_segment_cells_kernel(                                                   <L 492>
        // c = wp.tid()                                                                           <L 508>
        var_0 = builtin_tid1d();
        // if c >= num_cells:                                                                     <L 509>
        var_1 = (var_0 >= var_num_cells);
        if (var_1) {
            // return                                                                             <L 510>
            continue;
        }
        // start_cell = start_cell_ids[0]                                                         <L 512>
        var_3 = wp::address(var_start_cell_ids, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if start_cell < 0 or start_cell >= num_cells:                                          <L 513>
        var_8 = (var_4 < var_7);
        var_6 = var_8;
        if (!var_6) {
            var_9 = (var_4 >= var_num_cells);
            var_6 = var_6 || var_9;
        }
        if (var_6) {
            // return                                                                             <L 514>
            continue;
        }
        // if cell_active[start_cell] == 0:                                                       <L 515>
        var_10 = wp::address(var_cell_active, var_4);
        var_13 = wp::load(var_10);
        var_12 = (var_13 == var_11);
        if (var_12) {
            // return                                                                             <L 516>
            continue;
        }
        // if cell_active[c] == 0:                                                                <L 517>
        var_14 = wp::address(var_cell_active, var_0);
        var_17 = wp::load(var_14);
        var_16 = (var_17 == var_15);
        if (var_16) {
            // return                                                                             <L 518>
            continue;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 519>
        var_20 = (var_has_material_filter != var_19);
        var_18 = var_20;
        if (var_18) {
            var_21 = wp::address(var_cell_material, var_0);
            var_23 = wp::load(var_21);
            var_22 = wp::address(var_material_cuttable, var_23);
            var_26 = wp::load(var_22);
            var_25 = (var_26 == var_24);
            var_18 = var_18 && var_25;
        }
        if (var_18) {
            // return                                                                             <L 520>
            continue;
        }
        // start = wp.vec3(0.0, 0.0, 0.0)                                                         <L 522>
        var_30 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        // centre = wp.vec3(0.0, 0.0, 0.0)                                                        <L 523>
        var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
        // for local_node in range(8):                                                            <L 524>
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_36 = wp::address(var_cell_nodes, var_4, var_35);
        var_38 = wp::load(var_36);
        var_37 = wp::address(var_particle_q, var_38);
        var_40 = wp::load(var_37);
        var_39 = wp::add(var_30, var_40);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_41 = wp::address(var_cell_nodes, var_0, var_35);
        var_43 = wp::load(var_41);
        var_42 = wp::address(var_particle_q, var_43);
        var_45 = wp::load(var_42);
        var_44 = wp::add(var_34, var_45);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_47 = wp::address(var_cell_nodes, var_4, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::address(var_particle_q, var_49);
        var_51 = wp::load(var_48);
        var_50 = wp::add(var_39, var_51);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_52 = wp::address(var_cell_nodes, var_0, var_46);
        var_54 = wp::load(var_52);
        var_53 = wp::address(var_particle_q, var_54);
        var_56 = wp::load(var_53);
        var_55 = wp::add(var_44, var_56);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_58 = wp::address(var_cell_nodes, var_4, var_57);
        var_60 = wp::load(var_58);
        var_59 = wp::address(var_particle_q, var_60);
        var_62 = wp::load(var_59);
        var_61 = wp::add(var_50, var_62);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_63 = wp::address(var_cell_nodes, var_0, var_57);
        var_65 = wp::load(var_63);
        var_64 = wp::address(var_particle_q, var_65);
        var_67 = wp::load(var_64);
        var_66 = wp::add(var_55, var_67);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_69 = wp::address(var_cell_nodes, var_4, var_68);
        var_71 = wp::load(var_69);
        var_70 = wp::address(var_particle_q, var_71);
        var_73 = wp::load(var_70);
        var_72 = wp::add(var_61, var_73);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_74 = wp::address(var_cell_nodes, var_0, var_68);
        var_76 = wp::load(var_74);
        var_75 = wp::address(var_particle_q, var_76);
        var_78 = wp::load(var_75);
        var_77 = wp::add(var_66, var_78);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_80 = wp::address(var_cell_nodes, var_4, var_79);
        var_82 = wp::load(var_80);
        var_81 = wp::address(var_particle_q, var_82);
        var_84 = wp::load(var_81);
        var_83 = wp::add(var_72, var_84);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_85 = wp::address(var_cell_nodes, var_0, var_79);
        var_87 = wp::load(var_85);
        var_86 = wp::address(var_particle_q, var_87);
        var_89 = wp::load(var_86);
        var_88 = wp::add(var_77, var_89);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_91 = wp::address(var_cell_nodes, var_4, var_90);
        var_93 = wp::load(var_91);
        var_92 = wp::address(var_particle_q, var_93);
        var_95 = wp::load(var_92);
        var_94 = wp::add(var_83, var_95);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_96 = wp::address(var_cell_nodes, var_0, var_90);
        var_98 = wp::load(var_96);
        var_97 = wp::address(var_particle_q, var_98);
        var_100 = wp::load(var_97);
        var_99 = wp::add(var_88, var_100);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_102 = wp::address(var_cell_nodes, var_4, var_101);
        var_104 = wp::load(var_102);
        var_103 = wp::address(var_particle_q, var_104);
        var_106 = wp::load(var_103);
        var_105 = wp::add(var_94, var_106);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_107 = wp::address(var_cell_nodes, var_0, var_101);
        var_109 = wp::load(var_107);
        var_108 = wp::address(var_particle_q, var_109);
        var_111 = wp::load(var_108);
        var_110 = wp::add(var_99, var_111);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_113 = wp::address(var_cell_nodes, var_4, var_112);
        var_115 = wp::load(var_113);
        var_114 = wp::address(var_particle_q, var_115);
        var_117 = wp::load(var_114);
        var_116 = wp::add(var_105, var_117);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_118 = wp::address(var_cell_nodes, var_0, var_112);
        var_120 = wp::load(var_118);
        var_119 = wp::address(var_particle_q, var_120);
        var_122 = wp::load(var_119);
        var_121 = wp::add(var_110, var_122);
        // start *= 0.125                                                                         <L 527>
        var_124 = wp::mul(var_116, var_123);
        // centre *= 0.125                                                                        <L 528>
        var_126 = wp::mul(var_121, var_125);
        // end = start + ray_dir * depth                                                          <L 530>
        var_127 = wp::mul(var_ray_dir, var_depth);
        var_128 = wp::add(var_124, var_127);
        // dist_sq = _point_segment_distance_sq(centre, start, end)                               <L 531>
        var_129 = _point_segment_distance_sq_0(var_126, var_124, var_128);
        // if dist_sq > padding * padding:                                                        <L 532>
        var_130 = wp::mul(var_padding, var_padding);
        var_131 = (var_129 > var_130);
        if (var_131) {
            // return                                                                             <L 533>
            continue;
        }
        // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 535>
        var_134 = wp::atomic_add(var_selected_count, var_132, var_133);
        // selected_cells[out_idx] = c                                                            <L 536>
        wp::array_store(var_selected_cells, var_134, var_0);
    }
}



extern "C" __global__ void select_ray_segment_cells_kernel_6e51a20c_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_start_cell_ids,
    wp::vec_t<3, wp::float32> var_ray_dir,
    wp::float32 var_depth,
    wp::float32 var_padding,
    wp::array_t<wp::int32> var_selected_cells,
    wp::array_t<wp::int32> var_selected_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_cell_material,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_material_cuttable,
    wp::int32 adj_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_start_cell_ids,
    wp::vec_t<3, wp::float32> adj_ray_dir,
    wp::float32 adj_depth,
    wp::float32 adj_padding,
    wp::array_t<wp::int32> adj_selected_cells,
    wp::array_t<wp::int32> adj_selected_count)
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
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        bool var_9;
        wp::int32* var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        wp::int32* var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32 var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 0.0;
        const wp::float32 var_29 = 0.0;
        wp::vec_t<3, wp::float32> var_30;
        const wp::float32 var_31 = 0.0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.0;
        wp::vec_t<3, wp::float32> var_34;
        const wp::int32 var_35 = 0;
        wp::int32* var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        wp::int32* var_41;
        wp::vec_t<3, wp::float32>* var_42;
        wp::int32 var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32> var_45;
        const wp::int32 var_46 = 1;
        wp::int32* var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::int32* var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::int32 var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        const wp::int32 var_57 = 2;
        wp::int32* var_58;
        wp::vec_t<3, wp::float32>* var_59;
        wp::int32 var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::int32* var_63;
        wp::vec_t<3, wp::float32>* var_64;
        wp::int32 var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::int32 var_68 = 3;
        wp::int32* var_69;
        wp::vec_t<3, wp::float32>* var_70;
        wp::int32 var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::int32* var_74;
        wp::vec_t<3, wp::float32>* var_75;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        const wp::int32 var_79 = 4;
        wp::int32* var_80;
        wp::vec_t<3, wp::float32>* var_81;
        wp::int32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        wp::vec_t<3, wp::float32> var_84;
        wp::int32* var_85;
        wp::vec_t<3, wp::float32>* var_86;
        wp::int32 var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        const wp::int32 var_90 = 5;
        wp::int32* var_91;
        wp::vec_t<3, wp::float32>* var_92;
        wp::int32 var_93;
        wp::vec_t<3, wp::float32> var_94;
        wp::vec_t<3, wp::float32> var_95;
        wp::int32* var_96;
        wp::vec_t<3, wp::float32>* var_97;
        wp::int32 var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::vec_t<3, wp::float32> var_100;
        const wp::int32 var_101 = 6;
        wp::int32* var_102;
        wp::vec_t<3, wp::float32>* var_103;
        wp::int32 var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::int32* var_107;
        wp::vec_t<3, wp::float32>* var_108;
        wp::int32 var_109;
        wp::vec_t<3, wp::float32> var_110;
        wp::vec_t<3, wp::float32> var_111;
        const wp::int32 var_112 = 7;
        wp::int32* var_113;
        wp::vec_t<3, wp::float32>* var_114;
        wp::int32 var_115;
        wp::vec_t<3, wp::float32> var_116;
        wp::vec_t<3, wp::float32> var_117;
        wp::int32* var_118;
        wp::vec_t<3, wp::float32>* var_119;
        wp::int32 var_120;
        wp::vec_t<3, wp::float32> var_121;
        wp::vec_t<3, wp::float32> var_122;
        const wp::float32 var_123 = 0.125;
        wp::vec_t<3, wp::float32> var_124;
        const wp::float32 var_125 = 0.125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::float32 var_129;
        wp::float32 var_130;
        bool var_131;
        const wp::int32 var_132 = 0;
        const wp::int32 var_133 = 1;
        wp::int32 var_134;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        bool adj_16 = {};
        wp::int32 adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        bool adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        bool adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::int32 adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
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
        wp::int32 adj_52 = {};
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::int32 adj_54 = {};
        wp::vec_t<3, wp::float32> adj_55 = {};
        wp::vec_t<3, wp::float32> adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::vec_t<3, wp::float32> adj_59 = {};
        wp::int32 adj_60 = {};
        wp::vec_t<3, wp::float32> adj_61 = {};
        wp::vec_t<3, wp::float32> adj_62 = {};
        wp::int32 adj_63 = {};
        wp::vec_t<3, wp::float32> adj_64 = {};
        wp::int32 adj_65 = {};
        wp::vec_t<3, wp::float32> adj_66 = {};
        wp::vec_t<3, wp::float32> adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::vec_t<3, wp::float32> adj_70 = {};
        wp::int32 adj_71 = {};
        wp::vec_t<3, wp::float32> adj_72 = {};
        wp::vec_t<3, wp::float32> adj_73 = {};
        wp::int32 adj_74 = {};
        wp::vec_t<3, wp::float32> adj_75 = {};
        wp::int32 adj_76 = {};
        wp::vec_t<3, wp::float32> adj_77 = {};
        wp::vec_t<3, wp::float32> adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::vec_t<3, wp::float32> adj_81 = {};
        wp::int32 adj_82 = {};
        wp::vec_t<3, wp::float32> adj_83 = {};
        wp::vec_t<3, wp::float32> adj_84 = {};
        wp::int32 adj_85 = {};
        wp::vec_t<3, wp::float32> adj_86 = {};
        wp::int32 adj_87 = {};
        wp::vec_t<3, wp::float32> adj_88 = {};
        wp::vec_t<3, wp::float32> adj_89 = {};
        wp::int32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::vec_t<3, wp::float32> adj_92 = {};
        wp::int32 adj_93 = {};
        wp::vec_t<3, wp::float32> adj_94 = {};
        wp::vec_t<3, wp::float32> adj_95 = {};
        wp::int32 adj_96 = {};
        wp::vec_t<3, wp::float32> adj_97 = {};
        wp::int32 adj_98 = {};
        wp::vec_t<3, wp::float32> adj_99 = {};
        wp::vec_t<3, wp::float32> adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::vec_t<3, wp::float32> adj_103 = {};
        wp::int32 adj_104 = {};
        wp::vec_t<3, wp::float32> adj_105 = {};
        wp::vec_t<3, wp::float32> adj_106 = {};
        wp::int32 adj_107 = {};
        wp::vec_t<3, wp::float32> adj_108 = {};
        wp::int32 adj_109 = {};
        wp::vec_t<3, wp::float32> adj_110 = {};
        wp::vec_t<3, wp::float32> adj_111 = {};
        wp::int32 adj_112 = {};
        wp::int32 adj_113 = {};
        wp::vec_t<3, wp::float32> adj_114 = {};
        wp::int32 adj_115 = {};
        wp::vec_t<3, wp::float32> adj_116 = {};
        wp::vec_t<3, wp::float32> adj_117 = {};
        wp::int32 adj_118 = {};
        wp::vec_t<3, wp::float32> adj_119 = {};
        wp::int32 adj_120 = {};
        wp::vec_t<3, wp::float32> adj_121 = {};
        wp::vec_t<3, wp::float32> adj_122 = {};
        wp::float32 adj_123 = {};
        wp::vec_t<3, wp::float32> adj_124 = {};
        wp::float32 adj_125 = {};
        wp::vec_t<3, wp::float32> adj_126 = {};
        wp::vec_t<3, wp::float32> adj_127 = {};
        wp::vec_t<3, wp::float32> adj_128 = {};
        wp::float32 adj_129 = {};
        wp::float32 adj_130 = {};
        bool adj_131 = {};
        wp::int32 adj_132 = {};
        wp::int32 adj_133 = {};
        wp::int32 adj_134 = {};
        //---------
        // forward
        // def select_ray_segment_cells_kernel(                                                   <L 492>
        // c = wp.tid()                                                                           <L 508>
        var_0 = builtin_tid1d();
        // if c >= num_cells:                                                                     <L 509>
        var_1 = (var_0 >= var_num_cells);
        if (var_1) {
            // return                                                                             <L 510>
            goto label0;
        }
        // start_cell = start_cell_ids[0]                                                         <L 512>
        var_3 = wp::address(var_start_cell_ids, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if start_cell < 0 or start_cell >= num_cells:                                          <L 513>
        var_8 = (var_4 < var_7);
        var_6 = var_8;
        if (!var_6) {
            var_9 = (var_4 >= var_num_cells);
            var_6 = var_6 || var_9;
        }
        if (var_6) {
            // return                                                                             <L 514>
            goto label1;
        }
        // if cell_active[start_cell] == 0:                                                       <L 515>
        var_10 = wp::address(var_cell_active, var_4);
        var_13 = wp::load(var_10);
        var_12 = (var_13 == var_11);
        if (var_12) {
            // return                                                                             <L 516>
            goto label2;
        }
        // if cell_active[c] == 0:                                                                <L 517>
        var_14 = wp::address(var_cell_active, var_0);
        var_17 = wp::load(var_14);
        var_16 = (var_17 == var_15);
        if (var_16) {
            // return                                                                             <L 518>
            goto label3;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 519>
        var_20 = (var_has_material_filter != var_19);
        var_18 = var_20;
        if (var_18) {
            var_21 = wp::address(var_cell_material, var_0);
            var_23 = wp::load(var_21);
            var_22 = wp::address(var_material_cuttable, var_23);
            var_26 = wp::load(var_22);
            var_25 = (var_26 == var_24);
            var_18 = var_18 && var_25;
        }
        if (var_18) {
            // return                                                                             <L 520>
            goto label4;
        }
        // start = wp.vec3(0.0, 0.0, 0.0)                                                         <L 522>
        var_30 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        // centre = wp.vec3(0.0, 0.0, 0.0)                                                        <L 523>
        var_34 = wp::vec_t<3, wp::float32>(var_31, var_32, var_33);
        // for local_node in range(8):                                                            <L 524>
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_36 = wp::address(var_cell_nodes, var_4, var_35);
        var_38 = wp::load(var_36);
        var_37 = wp::address(var_particle_q, var_38);
        var_40 = wp::load(var_37);
        var_39 = wp::add(var_30, var_40);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_41 = wp::address(var_cell_nodes, var_0, var_35);
        var_43 = wp::load(var_41);
        var_42 = wp::address(var_particle_q, var_43);
        var_45 = wp::load(var_42);
        var_44 = wp::add(var_34, var_45);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_47 = wp::address(var_cell_nodes, var_4, var_46);
        var_49 = wp::load(var_47);
        var_48 = wp::address(var_particle_q, var_49);
        var_51 = wp::load(var_48);
        var_50 = wp::add(var_39, var_51);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_52 = wp::address(var_cell_nodes, var_0, var_46);
        var_54 = wp::load(var_52);
        var_53 = wp::address(var_particle_q, var_54);
        var_56 = wp::load(var_53);
        var_55 = wp::add(var_44, var_56);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_58 = wp::address(var_cell_nodes, var_4, var_57);
        var_60 = wp::load(var_58);
        var_59 = wp::address(var_particle_q, var_60);
        var_62 = wp::load(var_59);
        var_61 = wp::add(var_50, var_62);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_63 = wp::address(var_cell_nodes, var_0, var_57);
        var_65 = wp::load(var_63);
        var_64 = wp::address(var_particle_q, var_65);
        var_67 = wp::load(var_64);
        var_66 = wp::add(var_55, var_67);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_69 = wp::address(var_cell_nodes, var_4, var_68);
        var_71 = wp::load(var_69);
        var_70 = wp::address(var_particle_q, var_71);
        var_73 = wp::load(var_70);
        var_72 = wp::add(var_61, var_73);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_74 = wp::address(var_cell_nodes, var_0, var_68);
        var_76 = wp::load(var_74);
        var_75 = wp::address(var_particle_q, var_76);
        var_78 = wp::load(var_75);
        var_77 = wp::add(var_66, var_78);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_80 = wp::address(var_cell_nodes, var_4, var_79);
        var_82 = wp::load(var_80);
        var_81 = wp::address(var_particle_q, var_82);
        var_84 = wp::load(var_81);
        var_83 = wp::add(var_72, var_84);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_85 = wp::address(var_cell_nodes, var_0, var_79);
        var_87 = wp::load(var_85);
        var_86 = wp::address(var_particle_q, var_87);
        var_89 = wp::load(var_86);
        var_88 = wp::add(var_77, var_89);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_91 = wp::address(var_cell_nodes, var_4, var_90);
        var_93 = wp::load(var_91);
        var_92 = wp::address(var_particle_q, var_93);
        var_95 = wp::load(var_92);
        var_94 = wp::add(var_83, var_95);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_96 = wp::address(var_cell_nodes, var_0, var_90);
        var_98 = wp::load(var_96);
        var_97 = wp::address(var_particle_q, var_98);
        var_100 = wp::load(var_97);
        var_99 = wp::add(var_88, var_100);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_102 = wp::address(var_cell_nodes, var_4, var_101);
        var_104 = wp::load(var_102);
        var_103 = wp::address(var_particle_q, var_104);
        var_106 = wp::load(var_103);
        var_105 = wp::add(var_94, var_106);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_107 = wp::address(var_cell_nodes, var_0, var_101);
        var_109 = wp::load(var_107);
        var_108 = wp::address(var_particle_q, var_109);
        var_111 = wp::load(var_108);
        var_110 = wp::add(var_99, var_111);
        // start += particle_q[cell_nodes[start_cell, local_node]]                                <L 525>
        var_113 = wp::address(var_cell_nodes, var_4, var_112);
        var_115 = wp::load(var_113);
        var_114 = wp::address(var_particle_q, var_115);
        var_117 = wp::load(var_114);
        var_116 = wp::add(var_105, var_117);
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 526>
        var_118 = wp::address(var_cell_nodes, var_0, var_112);
        var_120 = wp::load(var_118);
        var_119 = wp::address(var_particle_q, var_120);
        var_122 = wp::load(var_119);
        var_121 = wp::add(var_110, var_122);
        // start *= 0.125                                                                         <L 527>
        var_124 = wp::mul(var_116, var_123);
        // centre *= 0.125                                                                        <L 528>
        var_126 = wp::mul(var_121, var_125);
        // end = start + ray_dir * depth                                                          <L 530>
        var_127 = wp::mul(var_ray_dir, var_depth);
        var_128 = wp::add(var_124, var_127);
        // dist_sq = _point_segment_distance_sq(centre, start, end)                               <L 531>
        var_129 = _point_segment_distance_sq_0(var_126, var_124, var_128);
        // if dist_sq > padding * padding:                                                        <L 532>
        var_130 = wp::mul(var_padding, var_padding);
        var_131 = (var_129 > var_130);
        if (var_131) {
            // return                                                                             <L 533>
            goto label5;
        }
        // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 535>
        // var_134 = wp::atomic_add(var_selected_count, var_132, var_133);
        // selected_cells[out_idx] = c                                                            <L 536>
        // wp::array_store(var_selected_cells, var_134, var_0);
        //---------
        // reverse
        wp::adj_array_store(var_selected_cells, var_134, var_0, adj_selected_cells, adj_134, adj_0);
        // adj: selected_cells[out_idx] = c                                                       <L 536>
        wp::adj_atomic_add(var_selected_count, var_132, var_133, adj_selected_count, adj_132, adj_133, adj_134);
        // adj: out_idx = wp.atomic_add(selected_count, 0, 1)                                     <L 535>
        if (var_131) {
            label5:;
            // adj: return                                                                        <L 533>
        }
        wp::adj_mul(var_padding, var_padding, adj_padding, adj_padding, adj_130);
        // adj: if dist_sq > padding * padding:                                                   <L 532>
        adj__point_segment_distance_sq_0(var_126, var_124, var_128, adj_126, adj_124, adj_128, adj_129);
        // adj: dist_sq = _point_segment_distance_sq(centre, start, end)                          <L 531>
        wp::adj_add(var_124, var_127, adj_124, adj_127, adj_128);
        wp::adj_mul(var_ray_dir, var_depth, adj_ray_dir, adj_depth, adj_127);
        // adj: end = start + ray_dir * depth                                                     <L 530>
        wp::adj_mul(var_121, var_125, adj_121, adj_125, adj_126);
        // adj: centre *= 0.125                                                                   <L 528>
        wp::adj_mul(var_116, var_123, adj_116, adj_123, adj_124);
        // adj: start *= 0.125                                                                    <L 527>
        wp::adj_add(var_110, var_122, adj_110, adj_119, adj_121);
        wp::adj_address(var_particle_q, var_120, adj_particle_q, adj_118, adj_119);
        wp::adj_address(var_cell_nodes, var_0, var_112, adj_cell_nodes, adj_0, adj_112, adj_118);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_105, var_117, adj_105, adj_114, adj_116);
        wp::adj_address(var_particle_q, var_115, adj_particle_q, adj_113, adj_114);
        wp::adj_address(var_cell_nodes, var_4, var_112, adj_cell_nodes, adj_4, adj_112, adj_113);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_99, var_111, adj_99, adj_108, adj_110);
        wp::adj_address(var_particle_q, var_109, adj_particle_q, adj_107, adj_108);
        wp::adj_address(var_cell_nodes, var_0, var_101, adj_cell_nodes, adj_0, adj_101, adj_107);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_94, var_106, adj_94, adj_103, adj_105);
        wp::adj_address(var_particle_q, var_104, adj_particle_q, adj_102, adj_103);
        wp::adj_address(var_cell_nodes, var_4, var_101, adj_cell_nodes, adj_4, adj_101, adj_102);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_88, var_100, adj_88, adj_97, adj_99);
        wp::adj_address(var_particle_q, var_98, adj_particle_q, adj_96, adj_97);
        wp::adj_address(var_cell_nodes, var_0, var_90, adj_cell_nodes, adj_0, adj_90, adj_96);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_83, var_95, adj_83, adj_92, adj_94);
        wp::adj_address(var_particle_q, var_93, adj_particle_q, adj_91, adj_92);
        wp::adj_address(var_cell_nodes, var_4, var_90, adj_cell_nodes, adj_4, adj_90, adj_91);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_77, var_89, adj_77, adj_86, adj_88);
        wp::adj_address(var_particle_q, var_87, adj_particle_q, adj_85, adj_86);
        wp::adj_address(var_cell_nodes, var_0, var_79, adj_cell_nodes, adj_0, adj_79, adj_85);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_72, var_84, adj_72, adj_81, adj_83);
        wp::adj_address(var_particle_q, var_82, adj_particle_q, adj_80, adj_81);
        wp::adj_address(var_cell_nodes, var_4, var_79, adj_cell_nodes, adj_4, adj_79, adj_80);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_66, var_78, adj_66, adj_75, adj_77);
        wp::adj_address(var_particle_q, var_76, adj_particle_q, adj_74, adj_75);
        wp::adj_address(var_cell_nodes, var_0, var_68, adj_cell_nodes, adj_0, adj_68, adj_74);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_61, var_73, adj_61, adj_70, adj_72);
        wp::adj_address(var_particle_q, var_71, adj_particle_q, adj_69, adj_70);
        wp::adj_address(var_cell_nodes, var_4, var_68, adj_cell_nodes, adj_4, adj_68, adj_69);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_55, var_67, adj_55, adj_64, adj_66);
        wp::adj_address(var_particle_q, var_65, adj_particle_q, adj_63, adj_64);
        wp::adj_address(var_cell_nodes, var_0, var_57, adj_cell_nodes, adj_0, adj_57, adj_63);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_50, var_62, adj_50, adj_59, adj_61);
        wp::adj_address(var_particle_q, var_60, adj_particle_q, adj_58, adj_59);
        wp::adj_address(var_cell_nodes, var_4, var_57, adj_cell_nodes, adj_4, adj_57, adj_58);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_44, var_56, adj_44, adj_53, adj_55);
        wp::adj_address(var_particle_q, var_54, adj_particle_q, adj_52, adj_53);
        wp::adj_address(var_cell_nodes, var_0, var_46, adj_cell_nodes, adj_0, adj_46, adj_52);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_39, var_51, adj_39, adj_48, adj_50);
        wp::adj_address(var_particle_q, var_49, adj_particle_q, adj_47, adj_48);
        wp::adj_address(var_cell_nodes, var_4, var_46, adj_cell_nodes, adj_4, adj_46, adj_47);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        wp::adj_add(var_34, var_45, adj_34, adj_42, adj_44);
        wp::adj_address(var_particle_q, var_43, adj_particle_q, adj_41, adj_42);
        wp::adj_address(var_cell_nodes, var_0, var_35, adj_cell_nodes, adj_0, adj_35, adj_41);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 526>
        wp::adj_add(var_30, var_40, adj_30, adj_37, adj_39);
        wp::adj_address(var_particle_q, var_38, adj_particle_q, adj_36, adj_37);
        wp::adj_address(var_cell_nodes, var_4, var_35, adj_cell_nodes, adj_4, adj_35, adj_36);
        // adj: start += particle_q[cell_nodes[start_cell, local_node]]                           <L 525>
        // adj: for local_node in range(8):                                                       <L 524>
        wp::adj_vec_t(var_31, var_32, var_33, adj_31, adj_32, adj_33, adj_34);
        // adj: centre = wp.vec3(0.0, 0.0, 0.0)                                                   <L 523>
        wp::adj_vec_t(var_27, var_28, var_29, adj_27, adj_28, adj_29, adj_30);
        // adj: start = wp.vec3(0.0, 0.0, 0.0)                                                    <L 522>
        if (var_18) {
            label4:;
            // adj: return                                                                        <L 520>
        }
        if (var_18) {
            wp::adj_address(var_material_cuttable, var_23, adj_material_cuttable, adj_21, adj_22);
            wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_21);
        }
        // adj: if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:         <L 519>
        if (var_16) {
            label3:;
            // adj: return                                                                        <L 518>
        }
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_14);
        // adj: if cell_active[c] == 0:                                                           <L 517>
        if (var_12) {
            label2:;
            // adj: return                                                                        <L 516>
        }
        wp::adj_address(var_cell_active, var_4, adj_cell_active, adj_4, adj_10);
        // adj: if cell_active[start_cell] == 0:                                                  <L 515>
        if (var_6) {
            label1:;
            // adj: return                                                                        <L 514>
        }
        if (!var_6) {
        }
        // adj: if start_cell < 0 or start_cell >= num_cells:                                     <L 513>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_start_cell_ids, var_2, adj_start_cell_ids, adj_2, adj_3);
        // adj: start_cell = start_cell_ids[0]                                                    <L 512>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 510>
        }
        // adj: if c >= num_cells:                                                                <L 509>
        // adj: c = wp.tid()                                                                      <L 508>
        // adj: def select_ray_segment_cells_kernel(                                              <L 492>
        continue;
    }
}



extern "C" __global__ void deactivate_single_deleted_cell_cluster_kernel_f780ad66_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_to_cluster,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
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
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 0;
        bool var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        const wp::int32 var_18 = 1;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        bool var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        bool var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::float32 var_44 = 1.0;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 0;
        const wp::float32 var_48 = 0.0;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        //---------
        // forward
        // def deactivate_single_deleted_cell_cluster_kernel(                                     <L 162>
        // if deleted_count[0] <= 0:                                                              <L 174>
        var_1 = wp::address(var_deleted_count, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 <= var_2);
        if (var_3) {
            // return                                                                             <L 175>
            continue;
        }
        // cell_idx = deleted_cells[0]                                                            <L 177>
        var_6 = wp::address(var_deleted_cells, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 178>
        var_11 = (var_7 < var_10);
        var_9 = var_11;
        if (!var_9) {
            var_12 = (var_7 >= var_num_cells);
            var_9 = var_9 || var_12;
        }
        if (var_9) {
            // return                                                                             <L 179>
            continue;
        }
        // cluster_idx = cell_to_cluster[cell_idx]                                                <L 181>
        var_13 = wp::address(var_cell_to_cluster, var_7);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if cluster_idx < 0:                                                                    <L 182>
        var_17 = (var_14 < var_16);
        if (var_17) {
            // return                                                                             <L 183>
            continue;
        }
        // old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)                          <L 185>
        var_20 = wp::atomic_cas(var_cluster_active, var_14, var_18, var_19);
        // if old_active != 1:                                                                    <L 186>
        var_22 = (var_20 != var_21);
        if (var_22) {
            // return                                                                             <L 187>
            continue;
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 189>
        var_23 = wp::address(var_cluster_offsets, var_14);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 190>
        var_27 = wp::add(var_14, var_26);
        var_28 = wp::address(var_cluster_offsets, var_27);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // while cursor < end:                                                                    <L 191>
        start_while_4:;
        var_31 = (var_24 < var_29);
        if ((var_31) == false) goto end_while_4;
            // particle_idx = cluster_indices[cursor]                                             <L 192>
            var_32 = wp::address(var_cluster_indices, var_24);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if particle_idx >= 0:                                                              <L 193>
            var_36 = (var_33 >= var_35);
            if (var_36) {
                // wp.atomic_sub(particle_cluster_counts, particle_idx, 1)                        <L 194>
                var_38 = wp::atomic_sub(var_particle_cluster_counts, var_33, var_37);
                // count = particle_cluster_counts[particle_idx]                                  <L 195>
                var_39 = wp::address(var_particle_cluster_counts, var_33);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // if count > 0:                                                                  <L 196>
                var_43 = (var_40 > var_42);
                if (var_43) {
                    // particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)            <L 197>
                    var_45 = wp::float(var_40);
                    var_46 = wp::div(var_44, var_45);
                    wp::array_store(var_particle_cluster_inv_weights, var_33, var_46);
                }
                if (!var_43) {
                    // particle_cluster_counts[particle_idx] = 0                                  <L 199>
                    wp::array_store(var_particle_cluster_counts, var_33, var_47);
                    // particle_cluster_inv_weights[particle_idx] = 0.0                           <L 200>
                    wp::array_store(var_particle_cluster_inv_weights, var_33, var_48);
                }
            }
            // cursor += 1                                                                        <L 201>
            var_50 = wp::add(var_24, var_49);
            wp::assign(var_24, var_50);
        goto start_while_4;
        end_while_4:;
    }
}



extern "C" __global__ void deactivate_single_deleted_cell_cluster_kernel_f780ad66_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_to_cluster,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::array_t<wp::int32> adj_deleted_cells,
    wp::array_t<wp::int32> adj_deleted_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_to_cluster,
    wp::array_t<wp::int32> adj_cluster_active,
    wp::array_t<wp::int32> adj_cluster_offsets,
    wp::array_t<wp::int32> adj_cluster_indices,
    wp::array_t<wp::int32> adj_particle_cluster_counts,
    wp::array_t<wp::float32> adj_particle_cluster_inv_weights)
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
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        const wp::int32 var_2 = 0;
        bool var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 0;
        bool var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        const wp::int32 var_18 = 1;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        bool var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 1;
        wp::int32 var_27;
        wp::int32* var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        bool var_31;
        wp::int32* var_32;
        wp::int32 var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        bool var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::float32 var_44 = 1.0;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 0;
        const wp::float32 var_48 = 0.0;
        const wp::int32 var_49 = 1;
        wp::int32 var_50;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        bool adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        bool adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        bool adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        bool adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        bool adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        //---------
        // forward
        // def deactivate_single_deleted_cell_cluster_kernel(                                     <L 162>
        // if deleted_count[0] <= 0:                                                              <L 174>
        var_1 = wp::address(var_deleted_count, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 <= var_2);
        if (var_3) {
            // return                                                                             <L 175>
            goto label0;
        }
        // cell_idx = deleted_cells[0]                                                            <L 177>
        var_6 = wp::address(var_deleted_cells, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 178>
        var_11 = (var_7 < var_10);
        var_9 = var_11;
        if (!var_9) {
            var_12 = (var_7 >= var_num_cells);
            var_9 = var_9 || var_12;
        }
        if (var_9) {
            // return                                                                             <L 179>
            goto label1;
        }
        // cluster_idx = cell_to_cluster[cell_idx]                                                <L 181>
        var_13 = wp::address(var_cell_to_cluster, var_7);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // if cluster_idx < 0:                                                                    <L 182>
        var_17 = (var_14 < var_16);
        if (var_17) {
            // return                                                                             <L 183>
            goto label2;
        }
        // old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)                          <L 185>
        // var_20 = wp::atomic_cas(var_cluster_active, var_14, var_18, var_19);
        // if old_active != 1:                                                                    <L 186>
        var_22 = (var_20 != var_21);
        if (var_22) {
            // return                                                                             <L 187>
            goto label3;
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 189>
        var_23 = wp::address(var_cluster_offsets, var_14);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 190>
        var_27 = wp::add(var_14, var_26);
        var_28 = wp::address(var_cluster_offsets, var_27);
        var_30 = wp::load(var_28);
        var_29 = wp::copy(var_30);
        // while cursor < end:                                                                    <L 191>
        //---------
        // reverse
        start_while_4:;
        var_31 = (var_24 < var_29);
        if ((var_31) == false) goto end_while_4;
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
        adj_45 = {};
        adj_46 = {};
        adj_47 = {};
        adj_48 = {};
        adj_49 = {};
        adj_50 = {};
            // particle_idx = cluster_indices[cursor]                                             <L 192>
            var_32 = wp::address(var_cluster_indices, var_24);
            var_34 = wp::load(var_32);
            var_33 = wp::copy(var_34);
            // if particle_idx >= 0:                                                              <L 193>
            var_36 = (var_33 >= var_35);
            if (var_36) {
                // wp.atomic_sub(particle_cluster_counts, particle_idx, 1)                        <L 194>
                // var_38 = wp::atomic_sub(var_particle_cluster_counts, var_33, var_37);
                // count = particle_cluster_counts[particle_idx]                                  <L 195>
                var_39 = wp::address(var_particle_cluster_counts, var_33);
                var_41 = wp::load(var_39);
                var_40 = wp::copy(var_41);
                // if count > 0:                                                                  <L 196>
                var_43 = (var_40 > var_42);
                if (var_43) {
                    // particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)            <L 197>
                    var_45 = wp::float(var_40);
                    var_46 = wp::div(var_44, var_45);
                    // wp::array_store(var_particle_cluster_inv_weights, var_33, var_46);
                }
                if (!var_43) {
                    // particle_cluster_counts[particle_idx] = 0                                  <L 199>
                    // wp::array_store(var_particle_cluster_counts, var_33, var_47);
                    // particle_cluster_inv_weights[particle_idx] = 0.0                           <L 200>
                    // wp::array_store(var_particle_cluster_inv_weights, var_33, var_48);
                }
            }
            // cursor += 1                                                                        <L 201>
            var_50 = wp::add(var_24, var_49);
            wp::assign(var_24, var_50);
            wp::adj_assign(var_24, var_50, adj_24, adj_50);
            wp::adj_add(var_24, var_49, adj_24, adj_49, adj_50);
            // adj: cursor += 1                                                                   <L 201>
            if (var_36) {
                if (!var_43) {
                    wp::adj_array_store(var_particle_cluster_inv_weights, var_33, var_48, adj_particle_cluster_inv_weights, adj_33, adj_48);
                    // adj: particle_cluster_inv_weights[particle_idx] = 0.0                      <L 200>
                    wp::adj_array_store(var_particle_cluster_counts, var_33, var_47, adj_particle_cluster_counts, adj_33, adj_47);
                    // adj: particle_cluster_counts[particle_idx] = 0                             <L 199>
                }
                if (var_43) {
                    wp::adj_array_store(var_particle_cluster_inv_weights, var_33, var_46, adj_particle_cluster_inv_weights, adj_33, adj_46);
                    wp::adj_div(var_44, var_45, var_46, adj_44, adj_45, adj_46);
                    wp::adj_float(var_40, adj_40, adj_45);
                    // adj: particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)       <L 197>
                }
                // adj: if count > 0:                                                             <L 196>
                wp::adj_copy(var_41, adj_39, adj_40);
                wp::adj_address(var_particle_cluster_counts, var_33, adj_particle_cluster_counts, adj_33, adj_39);
                // adj: count = particle_cluster_counts[particle_idx]                             <L 195>
                wp::adj_atomic_sub(var_particle_cluster_counts, var_33, var_37, adj_particle_cluster_counts, adj_33, adj_37, adj_38);
                // adj: wp.atomic_sub(particle_cluster_counts, particle_idx, 1)                   <L 194>
            }
            // adj: if particle_idx >= 0:                                                         <L 193>
            wp::adj_copy(var_34, adj_32, adj_33);
            wp::adj_address(var_cluster_indices, var_24, adj_cluster_indices, adj_24, adj_32);
            // adj: particle_idx = cluster_indices[cursor]                                        <L 192>
        goto start_while_4;
        end_while_4:;
        // adj: while cursor < end:                                                               <L 191>
        wp::adj_copy(var_30, adj_28, adj_29);
        wp::adj_address(var_cluster_offsets, var_27, adj_cluster_offsets, adj_27, adj_28);
        wp::adj_add(var_14, var_26, adj_14, adj_26, adj_27);
        // adj: end = cluster_offsets[cluster_idx + 1]                                            <L 190>
        wp::adj_copy(var_25, adj_23, adj_24);
        wp::adj_address(var_cluster_offsets, var_14, adj_cluster_offsets, adj_14, adj_23);
        // adj: cursor = cluster_offsets[cluster_idx]                                             <L 189>
        if (var_22) {
            label3:;
            // adj: return                                                                        <L 187>
        }
        // adj: if old_active != 1:                                                               <L 186>
        // adj: old_active = wp.atomic_cas(cluster_active, cluster_idx, 1, 0)                     <L 185>
        if (var_17) {
            label2:;
            // adj: return                                                                        <L 183>
        }
        // adj: if cluster_idx < 0:                                                               <L 182>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_cell_to_cluster, var_7, adj_cell_to_cluster, adj_7, adj_13);
        // adj: cluster_idx = cell_to_cluster[cell_idx]                                           <L 181>
        if (var_9) {
            label1:;
            // adj: return                                                                        <L 179>
        }
        if (!var_9) {
        }
        // adj: if cell_idx < 0 or cell_idx >= num_cells:                                         <L 178>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_deleted_cells, var_5, adj_deleted_cells, adj_5, adj_6);
        // adj: cell_idx = deleted_cells[0]                                                       <L 177>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 175>
        }
        wp::adj_address(var_deleted_count, var_0, adj_deleted_count, adj_0, adj_1);
        // adj: if deleted_count[0] <= 0:                                                         <L 174>
        // adj: def deactivate_single_deleted_cell_cluster_kernel(                                <L 162>
        continue;
    }
}



extern "C" __global__ void delete_single_cell_complete_kernel_0db01062_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_ids,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::float32> var_cell_mass,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_locked_node_mask,
    wp::array_t<wp::float32> var_particle_mass,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::array_t<wp::int32> var_deleted_total)
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
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        bool var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        bool var_7;
        const wp::int32 var_8 = 1;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        bool var_12;
        const wp::int32 var_13 = 0;
        const wp::int32 var_14 = 1;
        const wp::int32 var_15 = 0;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::float32* var_19;
        const wp::float32 var_20 = 1.0;
        const wp::float32 var_21 = 8.0;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::int32* var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32 var_49;
        const wp::float32 var_50 = 0.0;
        const wp::float32 var_51 = 0.0;
        const wp::float32 var_52 = 0.0;
        bool var_53;
        const wp::float32 var_54 = 1.0;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.0;
        const wp::float32 var_57 = 0.0;
        const wp::float32 var_58 = 0.0;
        const wp::int32 var_59 = 0;
        const wp::float32 var_60 = 0.0;
        const wp::float32 var_61 = 0.0;
        wp::int32 var_62;
        wp::int32 var_63;
        const wp::int32 var_64 = 1;
        wp::int32* var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        bool var_82;
        wp::int32 var_83;
        wp::int32* var_84;
        const wp::int32 var_85 = 0;
        bool var_86;
        wp::int32 var_87;
        const wp::float32 var_88 = 0.0;
        const wp::float32 var_89 = 0.0;
        const wp::float32 var_90 = 0.0;
        bool var_91;
        const wp::float32 var_92 = 1.0;
        wp::float32 var_93;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        const wp::float32 var_96 = 0.0;
        const wp::int32 var_97 = 0;
        const wp::float32 var_98 = 0.0;
        const wp::float32 var_99 = 0.0;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 2;
        wp::int32* var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::float32* var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 0;
        bool var_120;
        wp::int32 var_121;
        wp::int32* var_122;
        const wp::int32 var_123 = 0;
        bool var_124;
        wp::int32 var_125;
        const wp::float32 var_126 = 0.0;
        const wp::float32 var_127 = 0.0;
        const wp::float32 var_128 = 0.0;
        bool var_129;
        const wp::float32 var_130 = 1.0;
        wp::float32 var_131;
        const wp::float32 var_132 = 0.0;
        const wp::float32 var_133 = 0.0;
        const wp::float32 var_134 = 0.0;
        const wp::int32 var_135 = 0;
        const wp::float32 var_136 = 0.0;
        const wp::float32 var_137 = 0.0;
        wp::int32 var_138;
        wp::int32 var_139;
        const wp::int32 var_140 = 3;
        wp::int32* var_141;
        wp::int32 var_142;
        wp::int32 var_143;
        const wp::int32 var_144 = 1;
        wp::int32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::int32* var_148;
        wp::int32 var_149;
        wp::int32 var_150;
        wp::float32* var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::int32* var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        const wp::int32 var_157 = 0;
        bool var_158;
        wp::int32 var_159;
        wp::int32* var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        wp::int32 var_163;
        const wp::float32 var_164 = 0.0;
        const wp::float32 var_165 = 0.0;
        const wp::float32 var_166 = 0.0;
        bool var_167;
        const wp::float32 var_168 = 1.0;
        wp::float32 var_169;
        const wp::float32 var_170 = 0.0;
        const wp::float32 var_171 = 0.0;
        const wp::float32 var_172 = 0.0;
        const wp::int32 var_173 = 0;
        const wp::float32 var_174 = 0.0;
        const wp::float32 var_175 = 0.0;
        wp::int32 var_176;
        wp::int32 var_177;
        const wp::int32 var_178 = 4;
        wp::int32* var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        const wp::int32 var_182 = 1;
        wp::int32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::int32* var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        wp::float32* var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::int32* var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        const wp::int32 var_195 = 0;
        bool var_196;
        wp::int32 var_197;
        wp::int32* var_198;
        const wp::int32 var_199 = 0;
        bool var_200;
        wp::int32 var_201;
        const wp::float32 var_202 = 0.0;
        const wp::float32 var_203 = 0.0;
        const wp::float32 var_204 = 0.0;
        bool var_205;
        const wp::float32 var_206 = 1.0;
        wp::float32 var_207;
        const wp::float32 var_208 = 0.0;
        const wp::float32 var_209 = 0.0;
        const wp::float32 var_210 = 0.0;
        const wp::int32 var_211 = 0;
        const wp::float32 var_212 = 0.0;
        const wp::float32 var_213 = 0.0;
        wp::int32 var_214;
        wp::int32 var_215;
        const wp::int32 var_216 = 5;
        wp::int32* var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        const wp::int32 var_220 = 1;
        wp::int32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::int32* var_224;
        wp::int32 var_225;
        wp::int32 var_226;
        wp::float32* var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::int32* var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        const wp::int32 var_233 = 0;
        bool var_234;
        wp::int32 var_235;
        wp::int32* var_236;
        const wp::int32 var_237 = 0;
        bool var_238;
        wp::int32 var_239;
        const wp::float32 var_240 = 0.0;
        const wp::float32 var_241 = 0.0;
        const wp::float32 var_242 = 0.0;
        bool var_243;
        const wp::float32 var_244 = 1.0;
        wp::float32 var_245;
        const wp::float32 var_246 = 0.0;
        const wp::float32 var_247 = 0.0;
        const wp::float32 var_248 = 0.0;
        const wp::int32 var_249 = 0;
        const wp::float32 var_250 = 0.0;
        const wp::float32 var_251 = 0.0;
        wp::int32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 6;
        wp::int32* var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        const wp::int32 var_258 = 1;
        wp::int32 var_259;
        wp::float32 var_260;
        wp::float32 var_261;
        wp::int32* var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::float32* var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::int32* var_268;
        wp::int32 var_269;
        wp::int32 var_270;
        const wp::int32 var_271 = 0;
        bool var_272;
        wp::int32 var_273;
        wp::int32* var_274;
        const wp::int32 var_275 = 0;
        bool var_276;
        wp::int32 var_277;
        const wp::float32 var_278 = 0.0;
        const wp::float32 var_279 = 0.0;
        const wp::float32 var_280 = 0.0;
        bool var_281;
        const wp::float32 var_282 = 1.0;
        wp::float32 var_283;
        const wp::float32 var_284 = 0.0;
        const wp::float32 var_285 = 0.0;
        const wp::float32 var_286 = 0.0;
        const wp::int32 var_287 = 0;
        const wp::float32 var_288 = 0.0;
        const wp::float32 var_289 = 0.0;
        wp::int32 var_290;
        wp::int32 var_291;
        const wp::int32 var_292 = 7;
        wp::int32* var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        const wp::int32 var_296 = 1;
        wp::int32 var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        wp::int32* var_300;
        wp::int32 var_301;
        wp::int32 var_302;
        wp::float32* var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::int32* var_306;
        wp::int32 var_307;
        wp::int32 var_308;
        const wp::int32 var_309 = 0;
        bool var_310;
        wp::int32 var_311;
        wp::int32* var_312;
        const wp::int32 var_313 = 0;
        bool var_314;
        wp::int32 var_315;
        const wp::float32 var_316 = 0.0;
        const wp::float32 var_317 = 0.0;
        const wp::float32 var_318 = 0.0;
        bool var_319;
        const wp::float32 var_320 = 1.0;
        wp::float32 var_321;
        const wp::float32 var_322 = 0.0;
        const wp::float32 var_323 = 0.0;
        const wp::float32 var_324 = 0.0;
        const wp::int32 var_325 = 0;
        const wp::float32 var_326 = 0.0;
        const wp::float32 var_327 = 0.0;
        wp::int32 var_328;
        wp::int32 var_329;
        //---------
        // forward
        // def delete_single_cell_complete_kernel(                                                <L 59>
        // cell_idx = cell_ids[0]                                                                 <L 76>
        var_1 = wp::address(var_cell_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 77>
        var_6 = (var_2 < var_5);
        var_4 = var_6;
        if (!var_4) {
            var_7 = (var_2 >= var_num_cells);
            var_4 = var_4 || var_7;
        }
        if (var_4) {
            // return                                                                             <L 78>
            continue;
        }
        // old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)                                <L 80>
        var_10 = wp::atomic_cas(var_cell_active, var_2, var_8, var_9);
        // if old_active != 1:                                                                    <L 81>
        var_12 = (var_10 != var_11);
        if (var_12) {
            // return                                                                             <L 82>
            continue;
        }
        // deleted_cells[0] = cell_idx                                                            <L 84>
        wp::array_store(var_deleted_cells, var_13, var_2);
        // deleted_count[0] = 1                                                                   <L 85>
        wp::array_store(var_deleted_count, var_15, var_14);
        // wp.atomic_add(deleted_total, 0, 1)                                                     <L 86>
        var_18 = wp::atomic_add(var_deleted_total, var_16, var_17);
        // cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)                                     <L 88>
        var_19 = wp::address(var_cell_mass, var_2);
        var_22 = wp::div(var_20, var_21);
        var_24 = wp::load(var_19);
        var_23 = wp::mul(var_24, var_22);
        // for local_node in range(8):                                                            <L 89>
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_26 = wp::address(var_cell_nodes, var_2, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_30 = wp::atomic_sub(var_node_support_count, var_27, var_29);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_31 = wp::neg(var_23);
        var_32 = wp::atomic_add(var_node_mass, var_27, var_31);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_33 = wp::address(var_node_support_count, var_27);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_36 = wp::address(var_node_mass, var_27);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_39 = wp::address(var_particle_flags, var_27);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // if support > 0:                                                                        <L 97>
        var_43 = (var_34 > var_42);
        if (var_43) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_45 = wp::bit_or(var_40, var_44);
            wp::array_store(var_particle_flags, var_27, var_45);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_46 = wp::address(var_locked_node_mask, var_27);
            var_49 = wp::load(var_46);
            var_48 = (var_49 != var_47);
            if (var_48) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_27, var_50);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_27, var_51);
            }
            if (!var_48) {
                // elif mass > 0.0:                                                               <L 102>
                var_53 = (var_37 > var_52);
                if (var_53) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_27, var_37);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_55 = wp::div(var_54, var_37);
                    wp::array_store(var_particle_inv_mass, var_27, var_55);
                }
                if (!var_53) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_27, var_56);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_27, var_57);
                }
            }
        }
        if (!var_43) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_27, var_58);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_27, var_59);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_27, var_60);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_27, var_61);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_62 = wp::invert(var_44);
            var_63 = wp::bit_and(var_40, var_62);
            wp::array_store(var_particle_flags, var_27, var_63);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_65 = wp::address(var_cell_nodes, var_2, var_64);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_69 = wp::atomic_sub(var_node_support_count, var_66, var_68);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_70 = wp::neg(var_23);
        var_71 = wp::atomic_add(var_node_mass, var_66, var_70);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_72 = wp::address(var_node_support_count, var_66);
        var_74 = wp::load(var_72);
        var_73 = wp::copy(var_74);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_75 = wp::address(var_node_mass, var_66);
        var_77 = wp::load(var_75);
        var_76 = wp::copy(var_77);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_78 = wp::address(var_particle_flags, var_66);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // if support > 0:                                                                        <L 97>
        var_82 = (var_73 > var_81);
        if (var_82) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_83 = wp::bit_or(var_79, var_44);
            wp::array_store(var_particle_flags, var_66, var_83);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_84 = wp::address(var_locked_node_mask, var_66);
            var_87 = wp::load(var_84);
            var_86 = (var_87 != var_85);
            if (var_86) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_66, var_88);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_66, var_89);
            }
            if (!var_86) {
                // elif mass > 0.0:                                                               <L 102>
                var_91 = (var_76 > var_90);
                if (var_91) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_66, var_76);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_93 = wp::div(var_92, var_76);
                    wp::array_store(var_particle_inv_mass, var_66, var_93);
                }
                if (!var_91) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_66, var_94);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_66, var_95);
                }
            }
        }
        if (!var_82) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_66, var_96);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_66, var_97);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_66, var_98);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_66, var_99);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_100 = wp::invert(var_44);
            var_101 = wp::bit_and(var_79, var_100);
            wp::array_store(var_particle_flags, var_66, var_101);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_103 = wp::address(var_cell_nodes, var_2, var_102);
        var_105 = wp::load(var_103);
        var_104 = wp::copy(var_105);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_107 = wp::atomic_sub(var_node_support_count, var_104, var_106);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_108 = wp::neg(var_23);
        var_109 = wp::atomic_add(var_node_mass, var_104, var_108);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_110 = wp::address(var_node_support_count, var_104);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_113 = wp::address(var_node_mass, var_104);
        var_115 = wp::load(var_113);
        var_114 = wp::copy(var_115);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_116 = wp::address(var_particle_flags, var_104);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // if support > 0:                                                                        <L 97>
        var_120 = (var_111 > var_119);
        if (var_120) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_121 = wp::bit_or(var_117, var_44);
            wp::array_store(var_particle_flags, var_104, var_121);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_122 = wp::address(var_locked_node_mask, var_104);
            var_125 = wp::load(var_122);
            var_124 = (var_125 != var_123);
            if (var_124) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_104, var_126);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_104, var_127);
            }
            if (!var_124) {
                // elif mass > 0.0:                                                               <L 102>
                var_129 = (var_114 > var_128);
                if (var_129) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_104, var_114);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_131 = wp::div(var_130, var_114);
                    wp::array_store(var_particle_inv_mass, var_104, var_131);
                }
                if (!var_129) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_104, var_132);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_104, var_133);
                }
            }
        }
        if (!var_120) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_104, var_134);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_104, var_135);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_104, var_136);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_104, var_137);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_138 = wp::invert(var_44);
            var_139 = wp::bit_and(var_117, var_138);
            wp::array_store(var_particle_flags, var_104, var_139);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_141 = wp::address(var_cell_nodes, var_2, var_140);
        var_143 = wp::load(var_141);
        var_142 = wp::copy(var_143);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_145 = wp::atomic_sub(var_node_support_count, var_142, var_144);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_146 = wp::neg(var_23);
        var_147 = wp::atomic_add(var_node_mass, var_142, var_146);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_148 = wp::address(var_node_support_count, var_142);
        var_150 = wp::load(var_148);
        var_149 = wp::copy(var_150);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_151 = wp::address(var_node_mass, var_142);
        var_153 = wp::load(var_151);
        var_152 = wp::copy(var_153);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_154 = wp::address(var_particle_flags, var_142);
        var_156 = wp::load(var_154);
        var_155 = wp::copy(var_156);
        // if support > 0:                                                                        <L 97>
        var_158 = (var_149 > var_157);
        if (var_158) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_159 = wp::bit_or(var_155, var_44);
            wp::array_store(var_particle_flags, var_142, var_159);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_160 = wp::address(var_locked_node_mask, var_142);
            var_163 = wp::load(var_160);
            var_162 = (var_163 != var_161);
            if (var_162) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_142, var_164);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_142, var_165);
            }
            if (!var_162) {
                // elif mass > 0.0:                                                               <L 102>
                var_167 = (var_152 > var_166);
                if (var_167) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_142, var_152);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_169 = wp::div(var_168, var_152);
                    wp::array_store(var_particle_inv_mass, var_142, var_169);
                }
                if (!var_167) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_142, var_170);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_142, var_171);
                }
            }
        }
        if (!var_158) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_142, var_172);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_142, var_173);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_142, var_174);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_142, var_175);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_176 = wp::invert(var_44);
            var_177 = wp::bit_and(var_155, var_176);
            wp::array_store(var_particle_flags, var_142, var_177);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_179 = wp::address(var_cell_nodes, var_2, var_178);
        var_181 = wp::load(var_179);
        var_180 = wp::copy(var_181);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_183 = wp::atomic_sub(var_node_support_count, var_180, var_182);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_184 = wp::neg(var_23);
        var_185 = wp::atomic_add(var_node_mass, var_180, var_184);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_186 = wp::address(var_node_support_count, var_180);
        var_188 = wp::load(var_186);
        var_187 = wp::copy(var_188);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_189 = wp::address(var_node_mass, var_180);
        var_191 = wp::load(var_189);
        var_190 = wp::copy(var_191);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_192 = wp::address(var_particle_flags, var_180);
        var_194 = wp::load(var_192);
        var_193 = wp::copy(var_194);
        // if support > 0:                                                                        <L 97>
        var_196 = (var_187 > var_195);
        if (var_196) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_197 = wp::bit_or(var_193, var_44);
            wp::array_store(var_particle_flags, var_180, var_197);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_198 = wp::address(var_locked_node_mask, var_180);
            var_201 = wp::load(var_198);
            var_200 = (var_201 != var_199);
            if (var_200) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_180, var_202);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_180, var_203);
            }
            if (!var_200) {
                // elif mass > 0.0:                                                               <L 102>
                var_205 = (var_190 > var_204);
                if (var_205) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_180, var_190);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_207 = wp::div(var_206, var_190);
                    wp::array_store(var_particle_inv_mass, var_180, var_207);
                }
                if (!var_205) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_180, var_208);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_180, var_209);
                }
            }
        }
        if (!var_196) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_180, var_210);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_180, var_211);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_180, var_212);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_180, var_213);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_214 = wp::invert(var_44);
            var_215 = wp::bit_and(var_193, var_214);
            wp::array_store(var_particle_flags, var_180, var_215);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_217 = wp::address(var_cell_nodes, var_2, var_216);
        var_219 = wp::load(var_217);
        var_218 = wp::copy(var_219);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_221 = wp::atomic_sub(var_node_support_count, var_218, var_220);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_222 = wp::neg(var_23);
        var_223 = wp::atomic_add(var_node_mass, var_218, var_222);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_224 = wp::address(var_node_support_count, var_218);
        var_226 = wp::load(var_224);
        var_225 = wp::copy(var_226);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_227 = wp::address(var_node_mass, var_218);
        var_229 = wp::load(var_227);
        var_228 = wp::copy(var_229);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_230 = wp::address(var_particle_flags, var_218);
        var_232 = wp::load(var_230);
        var_231 = wp::copy(var_232);
        // if support > 0:                                                                        <L 97>
        var_234 = (var_225 > var_233);
        if (var_234) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_235 = wp::bit_or(var_231, var_44);
            wp::array_store(var_particle_flags, var_218, var_235);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_236 = wp::address(var_locked_node_mask, var_218);
            var_239 = wp::load(var_236);
            var_238 = (var_239 != var_237);
            if (var_238) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_218, var_240);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_218, var_241);
            }
            if (!var_238) {
                // elif mass > 0.0:                                                               <L 102>
                var_243 = (var_228 > var_242);
                if (var_243) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_218, var_228);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_245 = wp::div(var_244, var_228);
                    wp::array_store(var_particle_inv_mass, var_218, var_245);
                }
                if (!var_243) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_218, var_246);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_218, var_247);
                }
            }
        }
        if (!var_234) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_218, var_248);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_218, var_249);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_218, var_250);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_218, var_251);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_252 = wp::invert(var_44);
            var_253 = wp::bit_and(var_231, var_252);
            wp::array_store(var_particle_flags, var_218, var_253);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_255 = wp::address(var_cell_nodes, var_2, var_254);
        var_257 = wp::load(var_255);
        var_256 = wp::copy(var_257);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_259 = wp::atomic_sub(var_node_support_count, var_256, var_258);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_260 = wp::neg(var_23);
        var_261 = wp::atomic_add(var_node_mass, var_256, var_260);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_262 = wp::address(var_node_support_count, var_256);
        var_264 = wp::load(var_262);
        var_263 = wp::copy(var_264);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_265 = wp::address(var_node_mass, var_256);
        var_267 = wp::load(var_265);
        var_266 = wp::copy(var_267);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_268 = wp::address(var_particle_flags, var_256);
        var_270 = wp::load(var_268);
        var_269 = wp::copy(var_270);
        // if support > 0:                                                                        <L 97>
        var_272 = (var_263 > var_271);
        if (var_272) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_273 = wp::bit_or(var_269, var_44);
            wp::array_store(var_particle_flags, var_256, var_273);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_274 = wp::address(var_locked_node_mask, var_256);
            var_277 = wp::load(var_274);
            var_276 = (var_277 != var_275);
            if (var_276) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_256, var_278);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_256, var_279);
            }
            if (!var_276) {
                // elif mass > 0.0:                                                               <L 102>
                var_281 = (var_266 > var_280);
                if (var_281) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_256, var_266);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_283 = wp::div(var_282, var_266);
                    wp::array_store(var_particle_inv_mass, var_256, var_283);
                }
                if (!var_281) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_256, var_284);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_256, var_285);
                }
            }
        }
        if (!var_272) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_256, var_286);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_256, var_287);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_256, var_288);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_256, var_289);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_290 = wp::invert(var_44);
            var_291 = wp::bit_and(var_269, var_290);
            wp::array_store(var_particle_flags, var_256, var_291);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_293 = wp::address(var_cell_nodes, var_2, var_292);
        var_295 = wp::load(var_293);
        var_294 = wp::copy(var_295);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        var_297 = wp::atomic_sub(var_node_support_count, var_294, var_296);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_298 = wp::neg(var_23);
        var_299 = wp::atomic_add(var_node_mass, var_294, var_298);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_300 = wp::address(var_node_support_count, var_294);
        var_302 = wp::load(var_300);
        var_301 = wp::copy(var_302);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_303 = wp::address(var_node_mass, var_294);
        var_305 = wp::load(var_303);
        var_304 = wp::copy(var_305);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_306 = wp::address(var_particle_flags, var_294);
        var_308 = wp::load(var_306);
        var_307 = wp::copy(var_308);
        // if support > 0:                                                                        <L 97>
        var_310 = (var_301 > var_309);
        if (var_310) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_311 = wp::bit_or(var_307, var_44);
            wp::array_store(var_particle_flags, var_294, var_311);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_312 = wp::address(var_locked_node_mask, var_294);
            var_315 = wp::load(var_312);
            var_314 = (var_315 != var_313);
            if (var_314) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                wp::array_store(var_particle_mass, var_294, var_316);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                wp::array_store(var_particle_inv_mass, var_294, var_317);
            }
            if (!var_314) {
                // elif mass > 0.0:                                                               <L 102>
                var_319 = (var_304 > var_318);
                if (var_319) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    wp::array_store(var_particle_mass, var_294, var_304);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_321 = wp::div(var_320, var_304);
                    wp::array_store(var_particle_inv_mass, var_294, var_321);
                }
                if (!var_319) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    wp::array_store(var_particle_mass, var_294, var_322);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    wp::array_store(var_particle_inv_mass, var_294, var_323);
                }
            }
        }
        if (!var_310) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            wp::array_store(var_node_mass, var_294, var_324);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            wp::array_store(var_locked_node_mask, var_294, var_325);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            wp::array_store(var_particle_mass, var_294, var_326);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            wp::array_store(var_particle_inv_mass, var_294, var_327);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_328 = wp::invert(var_44);
            var_329 = wp::bit_and(var_307, var_328);
            wp::array_store(var_particle_flags, var_294, var_329);
        }
    }
}



extern "C" __global__ void delete_single_cell_complete_kernel_0db01062_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_ids,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::float32> var_cell_mass,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_locked_node_mask,
    wp::array_t<wp::float32> var_particle_mass,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::array_t<wp::int32> var_deleted_total,
    wp::array_t<wp::int32> adj_cell_ids,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::float32> adj_cell_mass,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_node_support_count,
    wp::array_t<wp::float32> adj_node_mass,
    wp::array_t<wp::int32> adj_locked_node_mask,
    wp::array_t<wp::float32> adj_particle_mass,
    wp::array_t<wp::float32> adj_particle_inv_mass,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_deleted_cells,
    wp::array_t<wp::int32> adj_deleted_count,
    wp::array_t<wp::int32> adj_deleted_total)
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
        const wp::int32 var_0 = 0;
        wp::int32* var_1;
        wp::int32 var_2;
        wp::int32 var_3;
        bool var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        bool var_7;
        const wp::int32 var_8 = 1;
        const wp::int32 var_9 = 0;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        bool var_12;
        const wp::int32 var_13 = 0;
        const wp::int32 var_14 = 1;
        const wp::int32 var_15 = 0;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        wp::float32* var_19;
        const wp::float32 var_20 = 1.0;
        const wp::float32 var_21 = 8.0;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 0;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::int32* var_33;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::float32* var_36;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::int32* var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        const wp::int32 var_42 = 0;
        bool var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::int32* var_46;
        const wp::int32 var_47 = 0;
        bool var_48;
        wp::int32 var_49;
        const wp::float32 var_50 = 0.0;
        const wp::float32 var_51 = 0.0;
        const wp::float32 var_52 = 0.0;
        bool var_53;
        const wp::float32 var_54 = 1.0;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.0;
        const wp::float32 var_57 = 0.0;
        const wp::float32 var_58 = 0.0;
        const wp::int32 var_59 = 0;
        const wp::float32 var_60 = 0.0;
        const wp::float32 var_61 = 0.0;
        wp::int32 var_62;
        wp::int32 var_63;
        const wp::int32 var_64 = 1;
        wp::int32* var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 1;
        wp::int32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::int32* var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::float32* var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::int32* var_78;
        wp::int32 var_79;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        bool var_82;
        wp::int32 var_83;
        wp::int32* var_84;
        const wp::int32 var_85 = 0;
        bool var_86;
        wp::int32 var_87;
        const wp::float32 var_88 = 0.0;
        const wp::float32 var_89 = 0.0;
        const wp::float32 var_90 = 0.0;
        bool var_91;
        const wp::float32 var_92 = 1.0;
        wp::float32 var_93;
        const wp::float32 var_94 = 0.0;
        const wp::float32 var_95 = 0.0;
        const wp::float32 var_96 = 0.0;
        const wp::int32 var_97 = 0;
        const wp::float32 var_98 = 0.0;
        const wp::float32 var_99 = 0.0;
        wp::int32 var_100;
        wp::int32 var_101;
        const wp::int32 var_102 = 2;
        wp::int32* var_103;
        wp::int32 var_104;
        wp::int32 var_105;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        wp::float32 var_108;
        wp::float32 var_109;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        wp::float32* var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::int32* var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        const wp::int32 var_119 = 0;
        bool var_120;
        wp::int32 var_121;
        wp::int32* var_122;
        const wp::int32 var_123 = 0;
        bool var_124;
        wp::int32 var_125;
        const wp::float32 var_126 = 0.0;
        const wp::float32 var_127 = 0.0;
        const wp::float32 var_128 = 0.0;
        bool var_129;
        const wp::float32 var_130 = 1.0;
        wp::float32 var_131;
        const wp::float32 var_132 = 0.0;
        const wp::float32 var_133 = 0.0;
        const wp::float32 var_134 = 0.0;
        const wp::int32 var_135 = 0;
        const wp::float32 var_136 = 0.0;
        const wp::float32 var_137 = 0.0;
        wp::int32 var_138;
        wp::int32 var_139;
        const wp::int32 var_140 = 3;
        wp::int32* var_141;
        wp::int32 var_142;
        wp::int32 var_143;
        const wp::int32 var_144 = 1;
        wp::int32 var_145;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::int32* var_148;
        wp::int32 var_149;
        wp::int32 var_150;
        wp::float32* var_151;
        wp::float32 var_152;
        wp::float32 var_153;
        wp::int32* var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        const wp::int32 var_157 = 0;
        bool var_158;
        wp::int32 var_159;
        wp::int32* var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        wp::int32 var_163;
        const wp::float32 var_164 = 0.0;
        const wp::float32 var_165 = 0.0;
        const wp::float32 var_166 = 0.0;
        bool var_167;
        const wp::float32 var_168 = 1.0;
        wp::float32 var_169;
        const wp::float32 var_170 = 0.0;
        const wp::float32 var_171 = 0.0;
        const wp::float32 var_172 = 0.0;
        const wp::int32 var_173 = 0;
        const wp::float32 var_174 = 0.0;
        const wp::float32 var_175 = 0.0;
        wp::int32 var_176;
        wp::int32 var_177;
        const wp::int32 var_178 = 4;
        wp::int32* var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        const wp::int32 var_182 = 1;
        wp::int32 var_183;
        wp::float32 var_184;
        wp::float32 var_185;
        wp::int32* var_186;
        wp::int32 var_187;
        wp::int32 var_188;
        wp::float32* var_189;
        wp::float32 var_190;
        wp::float32 var_191;
        wp::int32* var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        const wp::int32 var_195 = 0;
        bool var_196;
        wp::int32 var_197;
        wp::int32* var_198;
        const wp::int32 var_199 = 0;
        bool var_200;
        wp::int32 var_201;
        const wp::float32 var_202 = 0.0;
        const wp::float32 var_203 = 0.0;
        const wp::float32 var_204 = 0.0;
        bool var_205;
        const wp::float32 var_206 = 1.0;
        wp::float32 var_207;
        const wp::float32 var_208 = 0.0;
        const wp::float32 var_209 = 0.0;
        const wp::float32 var_210 = 0.0;
        const wp::int32 var_211 = 0;
        const wp::float32 var_212 = 0.0;
        const wp::float32 var_213 = 0.0;
        wp::int32 var_214;
        wp::int32 var_215;
        const wp::int32 var_216 = 5;
        wp::int32* var_217;
        wp::int32 var_218;
        wp::int32 var_219;
        const wp::int32 var_220 = 1;
        wp::int32 var_221;
        wp::float32 var_222;
        wp::float32 var_223;
        wp::int32* var_224;
        wp::int32 var_225;
        wp::int32 var_226;
        wp::float32* var_227;
        wp::float32 var_228;
        wp::float32 var_229;
        wp::int32* var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        const wp::int32 var_233 = 0;
        bool var_234;
        wp::int32 var_235;
        wp::int32* var_236;
        const wp::int32 var_237 = 0;
        bool var_238;
        wp::int32 var_239;
        const wp::float32 var_240 = 0.0;
        const wp::float32 var_241 = 0.0;
        const wp::float32 var_242 = 0.0;
        bool var_243;
        const wp::float32 var_244 = 1.0;
        wp::float32 var_245;
        const wp::float32 var_246 = 0.0;
        const wp::float32 var_247 = 0.0;
        const wp::float32 var_248 = 0.0;
        const wp::int32 var_249 = 0;
        const wp::float32 var_250 = 0.0;
        const wp::float32 var_251 = 0.0;
        wp::int32 var_252;
        wp::int32 var_253;
        const wp::int32 var_254 = 6;
        wp::int32* var_255;
        wp::int32 var_256;
        wp::int32 var_257;
        const wp::int32 var_258 = 1;
        wp::int32 var_259;
        wp::float32 var_260;
        wp::float32 var_261;
        wp::int32* var_262;
        wp::int32 var_263;
        wp::int32 var_264;
        wp::float32* var_265;
        wp::float32 var_266;
        wp::float32 var_267;
        wp::int32* var_268;
        wp::int32 var_269;
        wp::int32 var_270;
        const wp::int32 var_271 = 0;
        bool var_272;
        wp::int32 var_273;
        wp::int32* var_274;
        const wp::int32 var_275 = 0;
        bool var_276;
        wp::int32 var_277;
        const wp::float32 var_278 = 0.0;
        const wp::float32 var_279 = 0.0;
        const wp::float32 var_280 = 0.0;
        bool var_281;
        const wp::float32 var_282 = 1.0;
        wp::float32 var_283;
        const wp::float32 var_284 = 0.0;
        const wp::float32 var_285 = 0.0;
        const wp::float32 var_286 = 0.0;
        const wp::int32 var_287 = 0;
        const wp::float32 var_288 = 0.0;
        const wp::float32 var_289 = 0.0;
        wp::int32 var_290;
        wp::int32 var_291;
        const wp::int32 var_292 = 7;
        wp::int32* var_293;
        wp::int32 var_294;
        wp::int32 var_295;
        const wp::int32 var_296 = 1;
        wp::int32 var_297;
        wp::float32 var_298;
        wp::float32 var_299;
        wp::int32* var_300;
        wp::int32 var_301;
        wp::int32 var_302;
        wp::float32* var_303;
        wp::float32 var_304;
        wp::float32 var_305;
        wp::int32* var_306;
        wp::int32 var_307;
        wp::int32 var_308;
        const wp::int32 var_309 = 0;
        bool var_310;
        wp::int32 var_311;
        wp::int32* var_312;
        const wp::int32 var_313 = 0;
        bool var_314;
        wp::int32 var_315;
        const wp::float32 var_316 = 0.0;
        const wp::float32 var_317 = 0.0;
        const wp::float32 var_318 = 0.0;
        bool var_319;
        const wp::float32 var_320 = 1.0;
        wp::float32 var_321;
        const wp::float32 var_322 = 0.0;
        const wp::float32 var_323 = 0.0;
        const wp::float32 var_324 = 0.0;
        const wp::int32 var_325 = 0;
        const wp::float32 var_326 = 0.0;
        const wp::float32 var_327 = 0.0;
        wp::int32 var_328;
        wp::int32 var_329;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        bool adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        bool adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        bool adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        bool adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        bool adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::float32 adj_90 = {};
        bool adj_91 = {};
        wp::float32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::float32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::int32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::int32 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::float32 adj_113 = {};
        wp::float32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        bool adj_120 = {};
        wp::int32 adj_121 = {};
        wp::int32 adj_122 = {};
        wp::int32 adj_123 = {};
        bool adj_124 = {};
        wp::int32 adj_125 = {};
        wp::float32 adj_126 = {};
        wp::float32 adj_127 = {};
        wp::float32 adj_128 = {};
        bool adj_129 = {};
        wp::float32 adj_130 = {};
        wp::float32 adj_131 = {};
        wp::float32 adj_132 = {};
        wp::float32 adj_133 = {};
        wp::float32 adj_134 = {};
        wp::int32 adj_135 = {};
        wp::float32 adj_136 = {};
        wp::float32 adj_137 = {};
        wp::int32 adj_138 = {};
        wp::int32 adj_139 = {};
        wp::int32 adj_140 = {};
        wp::int32 adj_141 = {};
        wp::int32 adj_142 = {};
        wp::int32 adj_143 = {};
        wp::int32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::float32 adj_147 = {};
        wp::int32 adj_148 = {};
        wp::int32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::float32 adj_151 = {};
        wp::float32 adj_152 = {};
        wp::float32 adj_153 = {};
        wp::int32 adj_154 = {};
        wp::int32 adj_155 = {};
        wp::int32 adj_156 = {};
        wp::int32 adj_157 = {};
        bool adj_158 = {};
        wp::int32 adj_159 = {};
        wp::int32 adj_160 = {};
        wp::int32 adj_161 = {};
        bool adj_162 = {};
        wp::int32 adj_163 = {};
        wp::float32 adj_164 = {};
        wp::float32 adj_165 = {};
        wp::float32 adj_166 = {};
        bool adj_167 = {};
        wp::float32 adj_168 = {};
        wp::float32 adj_169 = {};
        wp::float32 adj_170 = {};
        wp::float32 adj_171 = {};
        wp::float32 adj_172 = {};
        wp::int32 adj_173 = {};
        wp::float32 adj_174 = {};
        wp::float32 adj_175 = {};
        wp::int32 adj_176 = {};
        wp::int32 adj_177 = {};
        wp::int32 adj_178 = {};
        wp::int32 adj_179 = {};
        wp::int32 adj_180 = {};
        wp::int32 adj_181 = {};
        wp::int32 adj_182 = {};
        wp::int32 adj_183 = {};
        wp::float32 adj_184 = {};
        wp::float32 adj_185 = {};
        wp::int32 adj_186 = {};
        wp::int32 adj_187 = {};
        wp::int32 adj_188 = {};
        wp::float32 adj_189 = {};
        wp::float32 adj_190 = {};
        wp::float32 adj_191 = {};
        wp::int32 adj_192 = {};
        wp::int32 adj_193 = {};
        wp::int32 adj_194 = {};
        wp::int32 adj_195 = {};
        bool adj_196 = {};
        wp::int32 adj_197 = {};
        wp::int32 adj_198 = {};
        wp::int32 adj_199 = {};
        bool adj_200 = {};
        wp::int32 adj_201 = {};
        wp::float32 adj_202 = {};
        wp::float32 adj_203 = {};
        wp::float32 adj_204 = {};
        bool adj_205 = {};
        wp::float32 adj_206 = {};
        wp::float32 adj_207 = {};
        wp::float32 adj_208 = {};
        wp::float32 adj_209 = {};
        wp::float32 adj_210 = {};
        wp::int32 adj_211 = {};
        wp::float32 adj_212 = {};
        wp::float32 adj_213 = {};
        wp::int32 adj_214 = {};
        wp::int32 adj_215 = {};
        wp::int32 adj_216 = {};
        wp::int32 adj_217 = {};
        wp::int32 adj_218 = {};
        wp::int32 adj_219 = {};
        wp::int32 adj_220 = {};
        wp::int32 adj_221 = {};
        wp::float32 adj_222 = {};
        wp::float32 adj_223 = {};
        wp::int32 adj_224 = {};
        wp::int32 adj_225 = {};
        wp::int32 adj_226 = {};
        wp::float32 adj_227 = {};
        wp::float32 adj_228 = {};
        wp::float32 adj_229 = {};
        wp::int32 adj_230 = {};
        wp::int32 adj_231 = {};
        wp::int32 adj_232 = {};
        wp::int32 adj_233 = {};
        bool adj_234 = {};
        wp::int32 adj_235 = {};
        wp::int32 adj_236 = {};
        wp::int32 adj_237 = {};
        bool adj_238 = {};
        wp::int32 adj_239 = {};
        wp::float32 adj_240 = {};
        wp::float32 adj_241 = {};
        wp::float32 adj_242 = {};
        bool adj_243 = {};
        wp::float32 adj_244 = {};
        wp::float32 adj_245 = {};
        wp::float32 adj_246 = {};
        wp::float32 adj_247 = {};
        wp::float32 adj_248 = {};
        wp::int32 adj_249 = {};
        wp::float32 adj_250 = {};
        wp::float32 adj_251 = {};
        wp::int32 adj_252 = {};
        wp::int32 adj_253 = {};
        wp::int32 adj_254 = {};
        wp::int32 adj_255 = {};
        wp::int32 adj_256 = {};
        wp::int32 adj_257 = {};
        wp::int32 adj_258 = {};
        wp::int32 adj_259 = {};
        wp::float32 adj_260 = {};
        wp::float32 adj_261 = {};
        wp::int32 adj_262 = {};
        wp::int32 adj_263 = {};
        wp::int32 adj_264 = {};
        wp::float32 adj_265 = {};
        wp::float32 adj_266 = {};
        wp::float32 adj_267 = {};
        wp::int32 adj_268 = {};
        wp::int32 adj_269 = {};
        wp::int32 adj_270 = {};
        wp::int32 adj_271 = {};
        bool adj_272 = {};
        wp::int32 adj_273 = {};
        wp::int32 adj_274 = {};
        wp::int32 adj_275 = {};
        bool adj_276 = {};
        wp::int32 adj_277 = {};
        wp::float32 adj_278 = {};
        wp::float32 adj_279 = {};
        wp::float32 adj_280 = {};
        bool adj_281 = {};
        wp::float32 adj_282 = {};
        wp::float32 adj_283 = {};
        wp::float32 adj_284 = {};
        wp::float32 adj_285 = {};
        wp::float32 adj_286 = {};
        wp::int32 adj_287 = {};
        wp::float32 adj_288 = {};
        wp::float32 adj_289 = {};
        wp::int32 adj_290 = {};
        wp::int32 adj_291 = {};
        wp::int32 adj_292 = {};
        wp::int32 adj_293 = {};
        wp::int32 adj_294 = {};
        wp::int32 adj_295 = {};
        wp::int32 adj_296 = {};
        wp::int32 adj_297 = {};
        wp::float32 adj_298 = {};
        wp::float32 adj_299 = {};
        wp::int32 adj_300 = {};
        wp::int32 adj_301 = {};
        wp::int32 adj_302 = {};
        wp::float32 adj_303 = {};
        wp::float32 adj_304 = {};
        wp::float32 adj_305 = {};
        wp::int32 adj_306 = {};
        wp::int32 adj_307 = {};
        wp::int32 adj_308 = {};
        wp::int32 adj_309 = {};
        bool adj_310 = {};
        wp::int32 adj_311 = {};
        wp::int32 adj_312 = {};
        wp::int32 adj_313 = {};
        bool adj_314 = {};
        wp::int32 adj_315 = {};
        wp::float32 adj_316 = {};
        wp::float32 adj_317 = {};
        wp::float32 adj_318 = {};
        bool adj_319 = {};
        wp::float32 adj_320 = {};
        wp::float32 adj_321 = {};
        wp::float32 adj_322 = {};
        wp::float32 adj_323 = {};
        wp::float32 adj_324 = {};
        wp::int32 adj_325 = {};
        wp::float32 adj_326 = {};
        wp::float32 adj_327 = {};
        wp::int32 adj_328 = {};
        wp::int32 adj_329 = {};
        //---------
        // forward
        // def delete_single_cell_complete_kernel(                                                <L 59>
        // cell_idx = cell_ids[0]                                                                 <L 76>
        var_1 = wp::address(var_cell_ids, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 77>
        var_6 = (var_2 < var_5);
        var_4 = var_6;
        if (!var_4) {
            var_7 = (var_2 >= var_num_cells);
            var_4 = var_4 || var_7;
        }
        if (var_4) {
            // return                                                                             <L 78>
            goto label0;
        }
        // old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)                                <L 80>
        // var_10 = wp::atomic_cas(var_cell_active, var_2, var_8, var_9);
        // if old_active != 1:                                                                    <L 81>
        var_12 = (var_10 != var_11);
        if (var_12) {
            // return                                                                             <L 82>
            goto label1;
        }
        // deleted_cells[0] = cell_idx                                                            <L 84>
        // wp::array_store(var_deleted_cells, var_13, var_2);
        // deleted_count[0] = 1                                                                   <L 85>
        // wp::array_store(var_deleted_count, var_15, var_14);
        // wp.atomic_add(deleted_total, 0, 1)                                                     <L 86>
        // var_18 = wp::atomic_add(var_deleted_total, var_16, var_17);
        // cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)                                     <L 88>
        var_19 = wp::address(var_cell_mass, var_2);
        var_22 = wp::div(var_20, var_21);
        var_24 = wp::load(var_19);
        var_23 = wp::mul(var_24, var_22);
        // for local_node in range(8):                                                            <L 89>
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_26 = wp::address(var_cell_nodes, var_2, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_30 = wp::atomic_sub(var_node_support_count, var_27, var_29);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_31 = wp::neg(var_23);
        // var_32 = wp::atomic_add(var_node_mass, var_27, var_31);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_33 = wp::address(var_node_support_count, var_27);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_36 = wp::address(var_node_mass, var_27);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_39 = wp::address(var_particle_flags, var_27);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // if support > 0:                                                                        <L 97>
        var_43 = (var_34 > var_42);
        if (var_43) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_45 = wp::bit_or(var_40, var_44);
            // wp::array_store(var_particle_flags, var_27, var_45);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_46 = wp::address(var_locked_node_mask, var_27);
            var_49 = wp::load(var_46);
            var_48 = (var_49 != var_47);
            if (var_48) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_27, var_50);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_27, var_51);
            }
            if (!var_48) {
                // elif mass > 0.0:                                                               <L 102>
                var_53 = (var_37 > var_52);
                if (var_53) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_27, var_37);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_55 = wp::div(var_54, var_37);
                    // wp::array_store(var_particle_inv_mass, var_27, var_55);
                }
                if (!var_53) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_27, var_56);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_27, var_57);
                }
            }
        }
        if (!var_43) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_27, var_58);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_27, var_59);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_27, var_60);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_27, var_61);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_62 = wp::invert(var_44);
            var_63 = wp::bit_and(var_40, var_62);
            // wp::array_store(var_particle_flags, var_27, var_63);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_65 = wp::address(var_cell_nodes, var_2, var_64);
        var_67 = wp::load(var_65);
        var_66 = wp::copy(var_67);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_69 = wp::atomic_sub(var_node_support_count, var_66, var_68);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_70 = wp::neg(var_23);
        // var_71 = wp::atomic_add(var_node_mass, var_66, var_70);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_72 = wp::address(var_node_support_count, var_66);
        var_74 = wp::load(var_72);
        var_73 = wp::copy(var_74);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_75 = wp::address(var_node_mass, var_66);
        var_77 = wp::load(var_75);
        var_76 = wp::copy(var_77);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_78 = wp::address(var_particle_flags, var_66);
        var_80 = wp::load(var_78);
        var_79 = wp::copy(var_80);
        // if support > 0:                                                                        <L 97>
        var_82 = (var_73 > var_81);
        if (var_82) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_83 = wp::bit_or(var_79, var_44);
            // wp::array_store(var_particle_flags, var_66, var_83);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_84 = wp::address(var_locked_node_mask, var_66);
            var_87 = wp::load(var_84);
            var_86 = (var_87 != var_85);
            if (var_86) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_66, var_88);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_66, var_89);
            }
            if (!var_86) {
                // elif mass > 0.0:                                                               <L 102>
                var_91 = (var_76 > var_90);
                if (var_91) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_66, var_76);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_93 = wp::div(var_92, var_76);
                    // wp::array_store(var_particle_inv_mass, var_66, var_93);
                }
                if (!var_91) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_66, var_94);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_66, var_95);
                }
            }
        }
        if (!var_82) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_66, var_96);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_66, var_97);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_66, var_98);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_66, var_99);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_100 = wp::invert(var_44);
            var_101 = wp::bit_and(var_79, var_100);
            // wp::array_store(var_particle_flags, var_66, var_101);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_103 = wp::address(var_cell_nodes, var_2, var_102);
        var_105 = wp::load(var_103);
        var_104 = wp::copy(var_105);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_107 = wp::atomic_sub(var_node_support_count, var_104, var_106);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_108 = wp::neg(var_23);
        // var_109 = wp::atomic_add(var_node_mass, var_104, var_108);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_110 = wp::address(var_node_support_count, var_104);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_113 = wp::address(var_node_mass, var_104);
        var_115 = wp::load(var_113);
        var_114 = wp::copy(var_115);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_116 = wp::address(var_particle_flags, var_104);
        var_118 = wp::load(var_116);
        var_117 = wp::copy(var_118);
        // if support > 0:                                                                        <L 97>
        var_120 = (var_111 > var_119);
        if (var_120) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_121 = wp::bit_or(var_117, var_44);
            // wp::array_store(var_particle_flags, var_104, var_121);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_122 = wp::address(var_locked_node_mask, var_104);
            var_125 = wp::load(var_122);
            var_124 = (var_125 != var_123);
            if (var_124) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_104, var_126);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_104, var_127);
            }
            if (!var_124) {
                // elif mass > 0.0:                                                               <L 102>
                var_129 = (var_114 > var_128);
                if (var_129) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_104, var_114);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_131 = wp::div(var_130, var_114);
                    // wp::array_store(var_particle_inv_mass, var_104, var_131);
                }
                if (!var_129) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_104, var_132);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_104, var_133);
                }
            }
        }
        if (!var_120) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_104, var_134);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_104, var_135);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_104, var_136);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_104, var_137);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_138 = wp::invert(var_44);
            var_139 = wp::bit_and(var_117, var_138);
            // wp::array_store(var_particle_flags, var_104, var_139);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_141 = wp::address(var_cell_nodes, var_2, var_140);
        var_143 = wp::load(var_141);
        var_142 = wp::copy(var_143);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_145 = wp::atomic_sub(var_node_support_count, var_142, var_144);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_146 = wp::neg(var_23);
        // var_147 = wp::atomic_add(var_node_mass, var_142, var_146);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_148 = wp::address(var_node_support_count, var_142);
        var_150 = wp::load(var_148);
        var_149 = wp::copy(var_150);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_151 = wp::address(var_node_mass, var_142);
        var_153 = wp::load(var_151);
        var_152 = wp::copy(var_153);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_154 = wp::address(var_particle_flags, var_142);
        var_156 = wp::load(var_154);
        var_155 = wp::copy(var_156);
        // if support > 0:                                                                        <L 97>
        var_158 = (var_149 > var_157);
        if (var_158) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_159 = wp::bit_or(var_155, var_44);
            // wp::array_store(var_particle_flags, var_142, var_159);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_160 = wp::address(var_locked_node_mask, var_142);
            var_163 = wp::load(var_160);
            var_162 = (var_163 != var_161);
            if (var_162) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_142, var_164);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_142, var_165);
            }
            if (!var_162) {
                // elif mass > 0.0:                                                               <L 102>
                var_167 = (var_152 > var_166);
                if (var_167) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_142, var_152);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_169 = wp::div(var_168, var_152);
                    // wp::array_store(var_particle_inv_mass, var_142, var_169);
                }
                if (!var_167) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_142, var_170);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_142, var_171);
                }
            }
        }
        if (!var_158) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_142, var_172);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_142, var_173);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_142, var_174);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_142, var_175);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_176 = wp::invert(var_44);
            var_177 = wp::bit_and(var_155, var_176);
            // wp::array_store(var_particle_flags, var_142, var_177);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_179 = wp::address(var_cell_nodes, var_2, var_178);
        var_181 = wp::load(var_179);
        var_180 = wp::copy(var_181);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_183 = wp::atomic_sub(var_node_support_count, var_180, var_182);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_184 = wp::neg(var_23);
        // var_185 = wp::atomic_add(var_node_mass, var_180, var_184);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_186 = wp::address(var_node_support_count, var_180);
        var_188 = wp::load(var_186);
        var_187 = wp::copy(var_188);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_189 = wp::address(var_node_mass, var_180);
        var_191 = wp::load(var_189);
        var_190 = wp::copy(var_191);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_192 = wp::address(var_particle_flags, var_180);
        var_194 = wp::load(var_192);
        var_193 = wp::copy(var_194);
        // if support > 0:                                                                        <L 97>
        var_196 = (var_187 > var_195);
        if (var_196) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_197 = wp::bit_or(var_193, var_44);
            // wp::array_store(var_particle_flags, var_180, var_197);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_198 = wp::address(var_locked_node_mask, var_180);
            var_201 = wp::load(var_198);
            var_200 = (var_201 != var_199);
            if (var_200) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_180, var_202);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_180, var_203);
            }
            if (!var_200) {
                // elif mass > 0.0:                                                               <L 102>
                var_205 = (var_190 > var_204);
                if (var_205) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_180, var_190);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_207 = wp::div(var_206, var_190);
                    // wp::array_store(var_particle_inv_mass, var_180, var_207);
                }
                if (!var_205) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_180, var_208);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_180, var_209);
                }
            }
        }
        if (!var_196) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_180, var_210);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_180, var_211);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_180, var_212);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_180, var_213);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_214 = wp::invert(var_44);
            var_215 = wp::bit_and(var_193, var_214);
            // wp::array_store(var_particle_flags, var_180, var_215);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_217 = wp::address(var_cell_nodes, var_2, var_216);
        var_219 = wp::load(var_217);
        var_218 = wp::copy(var_219);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_221 = wp::atomic_sub(var_node_support_count, var_218, var_220);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_222 = wp::neg(var_23);
        // var_223 = wp::atomic_add(var_node_mass, var_218, var_222);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_224 = wp::address(var_node_support_count, var_218);
        var_226 = wp::load(var_224);
        var_225 = wp::copy(var_226);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_227 = wp::address(var_node_mass, var_218);
        var_229 = wp::load(var_227);
        var_228 = wp::copy(var_229);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_230 = wp::address(var_particle_flags, var_218);
        var_232 = wp::load(var_230);
        var_231 = wp::copy(var_232);
        // if support > 0:                                                                        <L 97>
        var_234 = (var_225 > var_233);
        if (var_234) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_235 = wp::bit_or(var_231, var_44);
            // wp::array_store(var_particle_flags, var_218, var_235);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_236 = wp::address(var_locked_node_mask, var_218);
            var_239 = wp::load(var_236);
            var_238 = (var_239 != var_237);
            if (var_238) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_218, var_240);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_218, var_241);
            }
            if (!var_238) {
                // elif mass > 0.0:                                                               <L 102>
                var_243 = (var_228 > var_242);
                if (var_243) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_218, var_228);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_245 = wp::div(var_244, var_228);
                    // wp::array_store(var_particle_inv_mass, var_218, var_245);
                }
                if (!var_243) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_218, var_246);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_218, var_247);
                }
            }
        }
        if (!var_234) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_218, var_248);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_218, var_249);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_218, var_250);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_218, var_251);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_252 = wp::invert(var_44);
            var_253 = wp::bit_and(var_231, var_252);
            // wp::array_store(var_particle_flags, var_218, var_253);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_255 = wp::address(var_cell_nodes, var_2, var_254);
        var_257 = wp::load(var_255);
        var_256 = wp::copy(var_257);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_259 = wp::atomic_sub(var_node_support_count, var_256, var_258);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_260 = wp::neg(var_23);
        // var_261 = wp::atomic_add(var_node_mass, var_256, var_260);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_262 = wp::address(var_node_support_count, var_256);
        var_264 = wp::load(var_262);
        var_263 = wp::copy(var_264);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_265 = wp::address(var_node_mass, var_256);
        var_267 = wp::load(var_265);
        var_266 = wp::copy(var_267);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_268 = wp::address(var_particle_flags, var_256);
        var_270 = wp::load(var_268);
        var_269 = wp::copy(var_270);
        // if support > 0:                                                                        <L 97>
        var_272 = (var_263 > var_271);
        if (var_272) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_273 = wp::bit_or(var_269, var_44);
            // wp::array_store(var_particle_flags, var_256, var_273);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_274 = wp::address(var_locked_node_mask, var_256);
            var_277 = wp::load(var_274);
            var_276 = (var_277 != var_275);
            if (var_276) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_256, var_278);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_256, var_279);
            }
            if (!var_276) {
                // elif mass > 0.0:                                                               <L 102>
                var_281 = (var_266 > var_280);
                if (var_281) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_256, var_266);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_283 = wp::div(var_282, var_266);
                    // wp::array_store(var_particle_inv_mass, var_256, var_283);
                }
                if (!var_281) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_256, var_284);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_256, var_285);
                }
            }
        }
        if (!var_272) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_256, var_286);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_256, var_287);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_256, var_288);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_256, var_289);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_290 = wp::invert(var_44);
            var_291 = wp::bit_and(var_269, var_290);
            // wp::array_store(var_particle_flags, var_256, var_291);
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 90>
        var_293 = wp::address(var_cell_nodes, var_2, var_292);
        var_295 = wp::load(var_293);
        var_294 = wp::copy(var_295);
        // wp.atomic_sub(node_support_count, node_idx, 1)                                         <L 91>
        // var_297 = wp::atomic_sub(var_node_support_count, var_294, var_296);
        // wp.atomic_add(node_mass, node_idx, -cell_node_mass)                                    <L 92>
        var_298 = wp::neg(var_23);
        // var_299 = wp::atomic_add(var_node_mass, var_294, var_298);
        // support = node_support_count[node_idx]                                                 <L 94>
        var_300 = wp::address(var_node_support_count, var_294);
        var_302 = wp::load(var_300);
        var_301 = wp::copy(var_302);
        // mass = node_mass[node_idx]                                                             <L 95>
        var_303 = wp::address(var_node_mass, var_294);
        var_305 = wp::load(var_303);
        var_304 = wp::copy(var_305);
        // flags = particle_flags[node_idx]                                                       <L 96>
        var_306 = wp::address(var_particle_flags, var_294);
        var_308 = wp::load(var_306);
        var_307 = wp::copy(var_308);
        // if support > 0:                                                                        <L 97>
        var_310 = (var_301 > var_309);
        if (var_310) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 98>
            var_311 = wp::bit_or(var_307, var_44);
            // wp::array_store(var_particle_flags, var_294, var_311);
            // if locked_node_mask[node_idx] != 0:                                                <L 99>
            var_312 = wp::address(var_locked_node_mask, var_294);
            var_315 = wp::load(var_312);
            var_314 = (var_315 != var_313);
            if (var_314) {
                // particle_mass[node_idx] = 0.0                                                  <L 100>
                // wp::array_store(var_particle_mass, var_294, var_316);
                // particle_inv_mass[node_idx] = 0.0                                              <L 101>
                // wp::array_store(var_particle_inv_mass, var_294, var_317);
            }
            if (!var_314) {
                // elif mass > 0.0:                                                               <L 102>
                var_319 = (var_304 > var_318);
                if (var_319) {
                    // particle_mass[node_idx] = mass                                             <L 103>
                    // wp::array_store(var_particle_mass, var_294, var_304);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 104>
                    var_321 = wp::div(var_320, var_304);
                    // wp::array_store(var_particle_inv_mass, var_294, var_321);
                }
                if (!var_319) {
                    // particle_mass[node_idx] = 0.0                                              <L 106>
                    // wp::array_store(var_particle_mass, var_294, var_322);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 107>
                    // wp::array_store(var_particle_inv_mass, var_294, var_323);
                }
            }
        }
        if (!var_310) {
            // node_mass[node_idx] = 0.0                                                          <L 109>
            // wp::array_store(var_node_mass, var_294, var_324);
            // locked_node_mask[node_idx] = 0                                                     <L 110>
            // wp::array_store(var_locked_node_mask, var_294, var_325);
            // particle_mass[node_idx] = 0.0                                                      <L 111>
            // wp::array_store(var_particle_mass, var_294, var_326);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 112>
            // wp::array_store(var_particle_inv_mass, var_294, var_327);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 113>
            var_328 = wp::invert(var_44);
            var_329 = wp::bit_and(var_307, var_328);
            // wp::array_store(var_particle_flags, var_294, var_329);
        }
        //---------
        // reverse
        if (!var_310) {
            wp::adj_array_store(var_particle_flags, var_294, var_329, adj_particle_flags, adj_294, adj_329);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_294, var_327, adj_particle_inv_mass, adj_294, adj_327);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_294, var_326, adj_particle_mass, adj_294, adj_326);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_294, var_325, adj_locked_node_mask, adj_294, adj_325);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_294, var_324, adj_node_mass, adj_294, adj_324);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_310) {
            if (!var_314) {
                if (!var_319) {
                    wp::adj_array_store(var_particle_inv_mass, var_294, var_323, adj_particle_inv_mass, adj_294, adj_323);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_294, var_322, adj_particle_mass, adj_294, adj_322);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_319) {
                    wp::adj_array_store(var_particle_inv_mass, var_294, var_321, adj_particle_inv_mass, adj_294, adj_321);
                    wp::adj_div(var_320, var_304, var_321, adj_320, adj_304, adj_321);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_294, var_304, adj_particle_mass, adj_294, adj_304);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_314) {
                wp::adj_array_store(var_particle_inv_mass, var_294, var_317, adj_particle_inv_mass, adj_294, adj_317);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_294, var_316, adj_particle_mass, adj_294, adj_316);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_294, adj_locked_node_mask, adj_294, adj_312);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_294, var_311, adj_particle_flags, adj_294, adj_311);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_308, adj_306, adj_307);
        wp::adj_address(var_particle_flags, var_294, adj_particle_flags, adj_294, adj_306);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_305, adj_303, adj_304);
        wp::adj_address(var_node_mass, var_294, adj_node_mass, adj_294, adj_303);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_302, adj_300, adj_301);
        wp::adj_address(var_node_support_count, var_294, adj_node_support_count, adj_294, adj_300);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_294, var_298, adj_node_mass, adj_294, adj_298, adj_299);
        wp::adj_neg(var_23, adj_23, adj_298);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_294, var_296, adj_node_support_count, adj_294, adj_296, adj_297);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_295, adj_293, adj_294);
        wp::adj_address(var_cell_nodes, var_2, var_292, adj_cell_nodes, adj_2, adj_292, adj_293);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_272) {
            wp::adj_array_store(var_particle_flags, var_256, var_291, adj_particle_flags, adj_256, adj_291);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_256, var_289, adj_particle_inv_mass, adj_256, adj_289);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_256, var_288, adj_particle_mass, adj_256, adj_288);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_256, var_287, adj_locked_node_mask, adj_256, adj_287);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_256, var_286, adj_node_mass, adj_256, adj_286);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_272) {
            if (!var_276) {
                if (!var_281) {
                    wp::adj_array_store(var_particle_inv_mass, var_256, var_285, adj_particle_inv_mass, adj_256, adj_285);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_256, var_284, adj_particle_mass, adj_256, adj_284);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_281) {
                    wp::adj_array_store(var_particle_inv_mass, var_256, var_283, adj_particle_inv_mass, adj_256, adj_283);
                    wp::adj_div(var_282, var_266, var_283, adj_282, adj_266, adj_283);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_256, var_266, adj_particle_mass, adj_256, adj_266);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_276) {
                wp::adj_array_store(var_particle_inv_mass, var_256, var_279, adj_particle_inv_mass, adj_256, adj_279);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_256, var_278, adj_particle_mass, adj_256, adj_278);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_256, adj_locked_node_mask, adj_256, adj_274);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_256, var_273, adj_particle_flags, adj_256, adj_273);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_270, adj_268, adj_269);
        wp::adj_address(var_particle_flags, var_256, adj_particle_flags, adj_256, adj_268);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_267, adj_265, adj_266);
        wp::adj_address(var_node_mass, var_256, adj_node_mass, adj_256, adj_265);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_264, adj_262, adj_263);
        wp::adj_address(var_node_support_count, var_256, adj_node_support_count, adj_256, adj_262);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_256, var_260, adj_node_mass, adj_256, adj_260, adj_261);
        wp::adj_neg(var_23, adj_23, adj_260);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_256, var_258, adj_node_support_count, adj_256, adj_258, adj_259);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_257, adj_255, adj_256);
        wp::adj_address(var_cell_nodes, var_2, var_254, adj_cell_nodes, adj_2, adj_254, adj_255);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_234) {
            wp::adj_array_store(var_particle_flags, var_218, var_253, adj_particle_flags, adj_218, adj_253);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_218, var_251, adj_particle_inv_mass, adj_218, adj_251);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_218, var_250, adj_particle_mass, adj_218, adj_250);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_218, var_249, adj_locked_node_mask, adj_218, adj_249);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_218, var_248, adj_node_mass, adj_218, adj_248);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_234) {
            if (!var_238) {
                if (!var_243) {
                    wp::adj_array_store(var_particle_inv_mass, var_218, var_247, adj_particle_inv_mass, adj_218, adj_247);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_218, var_246, adj_particle_mass, adj_218, adj_246);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_243) {
                    wp::adj_array_store(var_particle_inv_mass, var_218, var_245, adj_particle_inv_mass, adj_218, adj_245);
                    wp::adj_div(var_244, var_228, var_245, adj_244, adj_228, adj_245);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_218, var_228, adj_particle_mass, adj_218, adj_228);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_238) {
                wp::adj_array_store(var_particle_inv_mass, var_218, var_241, adj_particle_inv_mass, adj_218, adj_241);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_218, var_240, adj_particle_mass, adj_218, adj_240);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_218, adj_locked_node_mask, adj_218, adj_236);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_218, var_235, adj_particle_flags, adj_218, adj_235);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_232, adj_230, adj_231);
        wp::adj_address(var_particle_flags, var_218, adj_particle_flags, adj_218, adj_230);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_229, adj_227, adj_228);
        wp::adj_address(var_node_mass, var_218, adj_node_mass, adj_218, adj_227);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_226, adj_224, adj_225);
        wp::adj_address(var_node_support_count, var_218, adj_node_support_count, adj_218, adj_224);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_218, var_222, adj_node_mass, adj_218, adj_222, adj_223);
        wp::adj_neg(var_23, adj_23, adj_222);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_218, var_220, adj_node_support_count, adj_218, adj_220, adj_221);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_219, adj_217, adj_218);
        wp::adj_address(var_cell_nodes, var_2, var_216, adj_cell_nodes, adj_2, adj_216, adj_217);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_196) {
            wp::adj_array_store(var_particle_flags, var_180, var_215, adj_particle_flags, adj_180, adj_215);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_180, var_213, adj_particle_inv_mass, adj_180, adj_213);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_180, var_212, adj_particle_mass, adj_180, adj_212);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_180, var_211, adj_locked_node_mask, adj_180, adj_211);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_180, var_210, adj_node_mass, adj_180, adj_210);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_196) {
            if (!var_200) {
                if (!var_205) {
                    wp::adj_array_store(var_particle_inv_mass, var_180, var_209, adj_particle_inv_mass, adj_180, adj_209);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_180, var_208, adj_particle_mass, adj_180, adj_208);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_205) {
                    wp::adj_array_store(var_particle_inv_mass, var_180, var_207, adj_particle_inv_mass, adj_180, adj_207);
                    wp::adj_div(var_206, var_190, var_207, adj_206, adj_190, adj_207);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_180, var_190, adj_particle_mass, adj_180, adj_190);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_200) {
                wp::adj_array_store(var_particle_inv_mass, var_180, var_203, adj_particle_inv_mass, adj_180, adj_203);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_180, var_202, adj_particle_mass, adj_180, adj_202);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_180, adj_locked_node_mask, adj_180, adj_198);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_180, var_197, adj_particle_flags, adj_180, adj_197);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_194, adj_192, adj_193);
        wp::adj_address(var_particle_flags, var_180, adj_particle_flags, adj_180, adj_192);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_191, adj_189, adj_190);
        wp::adj_address(var_node_mass, var_180, adj_node_mass, adj_180, adj_189);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_188, adj_186, adj_187);
        wp::adj_address(var_node_support_count, var_180, adj_node_support_count, adj_180, adj_186);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_180, var_184, adj_node_mass, adj_180, adj_184, adj_185);
        wp::adj_neg(var_23, adj_23, adj_184);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_180, var_182, adj_node_support_count, adj_180, adj_182, adj_183);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_181, adj_179, adj_180);
        wp::adj_address(var_cell_nodes, var_2, var_178, adj_cell_nodes, adj_2, adj_178, adj_179);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_158) {
            wp::adj_array_store(var_particle_flags, var_142, var_177, adj_particle_flags, adj_142, adj_177);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_142, var_175, adj_particle_inv_mass, adj_142, adj_175);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_142, var_174, adj_particle_mass, adj_142, adj_174);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_142, var_173, adj_locked_node_mask, adj_142, adj_173);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_142, var_172, adj_node_mass, adj_142, adj_172);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_158) {
            if (!var_162) {
                if (!var_167) {
                    wp::adj_array_store(var_particle_inv_mass, var_142, var_171, adj_particle_inv_mass, adj_142, adj_171);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_142, var_170, adj_particle_mass, adj_142, adj_170);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_167) {
                    wp::adj_array_store(var_particle_inv_mass, var_142, var_169, adj_particle_inv_mass, adj_142, adj_169);
                    wp::adj_div(var_168, var_152, var_169, adj_168, adj_152, adj_169);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_142, var_152, adj_particle_mass, adj_142, adj_152);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_162) {
                wp::adj_array_store(var_particle_inv_mass, var_142, var_165, adj_particle_inv_mass, adj_142, adj_165);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_142, var_164, adj_particle_mass, adj_142, adj_164);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_142, adj_locked_node_mask, adj_142, adj_160);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_142, var_159, adj_particle_flags, adj_142, adj_159);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_156, adj_154, adj_155);
        wp::adj_address(var_particle_flags, var_142, adj_particle_flags, adj_142, adj_154);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_153, adj_151, adj_152);
        wp::adj_address(var_node_mass, var_142, adj_node_mass, adj_142, adj_151);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_150, adj_148, adj_149);
        wp::adj_address(var_node_support_count, var_142, adj_node_support_count, adj_142, adj_148);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_142, var_146, adj_node_mass, adj_142, adj_146, adj_147);
        wp::adj_neg(var_23, adj_23, adj_146);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_142, var_144, adj_node_support_count, adj_142, adj_144, adj_145);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_143, adj_141, adj_142);
        wp::adj_address(var_cell_nodes, var_2, var_140, adj_cell_nodes, adj_2, adj_140, adj_141);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_120) {
            wp::adj_array_store(var_particle_flags, var_104, var_139, adj_particle_flags, adj_104, adj_139);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_104, var_137, adj_particle_inv_mass, adj_104, adj_137);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_104, var_136, adj_particle_mass, adj_104, adj_136);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_104, var_135, adj_locked_node_mask, adj_104, adj_135);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_104, var_134, adj_node_mass, adj_104, adj_134);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_120) {
            if (!var_124) {
                if (!var_129) {
                    wp::adj_array_store(var_particle_inv_mass, var_104, var_133, adj_particle_inv_mass, adj_104, adj_133);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_104, var_132, adj_particle_mass, adj_104, adj_132);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_129) {
                    wp::adj_array_store(var_particle_inv_mass, var_104, var_131, adj_particle_inv_mass, adj_104, adj_131);
                    wp::adj_div(var_130, var_114, var_131, adj_130, adj_114, adj_131);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_104, var_114, adj_particle_mass, adj_104, adj_114);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_124) {
                wp::adj_array_store(var_particle_inv_mass, var_104, var_127, adj_particle_inv_mass, adj_104, adj_127);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_104, var_126, adj_particle_mass, adj_104, adj_126);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_104, adj_locked_node_mask, adj_104, adj_122);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_104, var_121, adj_particle_flags, adj_104, adj_121);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_118, adj_116, adj_117);
        wp::adj_address(var_particle_flags, var_104, adj_particle_flags, adj_104, adj_116);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_115, adj_113, adj_114);
        wp::adj_address(var_node_mass, var_104, adj_node_mass, adj_104, adj_113);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_112, adj_110, adj_111);
        wp::adj_address(var_node_support_count, var_104, adj_node_support_count, adj_104, adj_110);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_104, var_108, adj_node_mass, adj_104, adj_108, adj_109);
        wp::adj_neg(var_23, adj_23, adj_108);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_104, var_106, adj_node_support_count, adj_104, adj_106, adj_107);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_105, adj_103, adj_104);
        wp::adj_address(var_cell_nodes, var_2, var_102, adj_cell_nodes, adj_2, adj_102, adj_103);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_82) {
            wp::adj_array_store(var_particle_flags, var_66, var_101, adj_particle_flags, adj_66, adj_101);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_66, var_99, adj_particle_inv_mass, adj_66, adj_99);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_66, var_98, adj_particle_mass, adj_66, adj_98);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_66, var_97, adj_locked_node_mask, adj_66, adj_97);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_66, var_96, adj_node_mass, adj_66, adj_96);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_82) {
            if (!var_86) {
                if (!var_91) {
                    wp::adj_array_store(var_particle_inv_mass, var_66, var_95, adj_particle_inv_mass, adj_66, adj_95);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_66, var_94, adj_particle_mass, adj_66, adj_94);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_91) {
                    wp::adj_array_store(var_particle_inv_mass, var_66, var_93, adj_particle_inv_mass, adj_66, adj_93);
                    wp::adj_div(var_92, var_76, var_93, adj_92, adj_76, adj_93);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_66, var_76, adj_particle_mass, adj_66, adj_76);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_86) {
                wp::adj_array_store(var_particle_inv_mass, var_66, var_89, adj_particle_inv_mass, adj_66, adj_89);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_66, var_88, adj_particle_mass, adj_66, adj_88);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_66, adj_locked_node_mask, adj_66, adj_84);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_66, var_83, adj_particle_flags, adj_66, adj_83);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_80, adj_78, adj_79);
        wp::adj_address(var_particle_flags, var_66, adj_particle_flags, adj_66, adj_78);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_77, adj_75, adj_76);
        wp::adj_address(var_node_mass, var_66, adj_node_mass, adj_66, adj_75);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_74, adj_72, adj_73);
        wp::adj_address(var_node_support_count, var_66, adj_node_support_count, adj_66, adj_72);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_66, var_70, adj_node_mass, adj_66, adj_70, adj_71);
        wp::adj_neg(var_23, adj_23, adj_70);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_66, var_68, adj_node_support_count, adj_66, adj_68, adj_69);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_67, adj_65, adj_66);
        wp::adj_address(var_cell_nodes, var_2, var_64, adj_cell_nodes, adj_2, adj_64, adj_65);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        if (!var_43) {
            wp::adj_array_store(var_particle_flags, var_27, var_63, adj_particle_flags, adj_27, adj_63);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 113>
            wp::adj_array_store(var_particle_inv_mass, var_27, var_61, adj_particle_inv_mass, adj_27, adj_61);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 112>
            wp::adj_array_store(var_particle_mass, var_27, var_60, adj_particle_mass, adj_27, adj_60);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 111>
            wp::adj_array_store(var_locked_node_mask, var_27, var_59, adj_locked_node_mask, adj_27, adj_59);
            // adj: locked_node_mask[node_idx] = 0                                                <L 110>
            wp::adj_array_store(var_node_mass, var_27, var_58, adj_node_mass, adj_27, adj_58);
            // adj: node_mass[node_idx] = 0.0                                                     <L 109>
        }
        if (var_43) {
            if (!var_48) {
                if (!var_53) {
                    wp::adj_array_store(var_particle_inv_mass, var_27, var_57, adj_particle_inv_mass, adj_27, adj_57);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 107>
                    wp::adj_array_store(var_particle_mass, var_27, var_56, adj_particle_mass, adj_27, adj_56);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 106>
                }
                if (var_53) {
                    wp::adj_array_store(var_particle_inv_mass, var_27, var_55, adj_particle_inv_mass, adj_27, adj_55);
                    wp::adj_div(var_54, var_37, var_55, adj_54, adj_37, adj_55);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 104>
                    wp::adj_array_store(var_particle_mass, var_27, var_37, adj_particle_mass, adj_27, adj_37);
                    // adj: particle_mass[node_idx] = mass                                        <L 103>
                }
                // adj: elif mass > 0.0:                                                          <L 102>
            }
            if (var_48) {
                wp::adj_array_store(var_particle_inv_mass, var_27, var_51, adj_particle_inv_mass, adj_27, adj_51);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 101>
                wp::adj_array_store(var_particle_mass, var_27, var_50, adj_particle_mass, adj_27, adj_50);
                // adj: particle_mass[node_idx] = 0.0                                             <L 100>
            }
            wp::adj_address(var_locked_node_mask, var_27, adj_locked_node_mask, adj_27, adj_46);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 99>
            wp::adj_array_store(var_particle_flags, var_27, var_45, adj_particle_flags, adj_27, adj_45);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 98>
        }
        // adj: if support > 0:                                                                   <L 97>
        wp::adj_copy(var_41, adj_39, adj_40);
        wp::adj_address(var_particle_flags, var_27, adj_particle_flags, adj_27, adj_39);
        // adj: flags = particle_flags[node_idx]                                                  <L 96>
        wp::adj_copy(var_38, adj_36, adj_37);
        wp::adj_address(var_node_mass, var_27, adj_node_mass, adj_27, adj_36);
        // adj: mass = node_mass[node_idx]                                                        <L 95>
        wp::adj_copy(var_35, adj_33, adj_34);
        wp::adj_address(var_node_support_count, var_27, adj_node_support_count, adj_27, adj_33);
        // adj: support = node_support_count[node_idx]                                            <L 94>
        wp::adj_atomic_add(var_node_mass, var_27, var_31, adj_node_mass, adj_27, adj_31, adj_32);
        wp::adj_neg(var_23, adj_23, adj_31);
        // adj: wp.atomic_add(node_mass, node_idx, -cell_node_mass)                               <L 92>
        wp::adj_atomic_sub(var_node_support_count, var_27, var_29, adj_node_support_count, adj_27, adj_29, adj_30);
        // adj: wp.atomic_sub(node_support_count, node_idx, 1)                                    <L 91>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_cell_nodes, var_2, var_25, adj_cell_nodes, adj_2, adj_25, adj_26);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 90>
        // adj: for local_node in range(8):                                                       <L 89>
        wp::adj_mul(var_24, var_22, adj_19, adj_22, adj_23);
        wp::adj_div(var_20, var_21, var_22, adj_20, adj_21, adj_22);
        wp::adj_address(var_cell_mass, var_2, adj_cell_mass, adj_2, adj_19);
        // adj: cell_node_mass = cell_mass[cell_idx] * (1.0 / 8.0)                                <L 88>
        wp::adj_atomic_add(var_deleted_total, var_16, var_17, adj_deleted_total, adj_16, adj_17, adj_18);
        // adj: wp.atomic_add(deleted_total, 0, 1)                                                <L 86>
        wp::adj_array_store(var_deleted_count, var_15, var_14, adj_deleted_count, adj_15, adj_14);
        // adj: deleted_count[0] = 1                                                              <L 85>
        wp::adj_array_store(var_deleted_cells, var_13, var_2, adj_deleted_cells, adj_13, adj_2);
        // adj: deleted_cells[0] = cell_idx                                                       <L 84>
        if (var_12) {
            label1:;
            // adj: return                                                                        <L 82>
        }
        // adj: if old_active != 1:                                                               <L 81>
        // adj: old_active = wp.atomic_cas(cell_active, cell_idx, 1, 0)                           <L 80>
        if (var_4) {
            label0:;
            // adj: return                                                                        <L 78>
        }
        if (!var_4) {
        }
        // adj: if cell_idx < 0 or cell_idx >= num_cells:                                         <L 77>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_cell_ids, var_0, adj_cell_ids, adj_0, adj_1);
        // adj: cell_idx = cell_ids[0]                                                            <L 76>
        // adj: def delete_single_cell_complete_kernel(                                           <L 59>
        continue;
    }
}



extern "C" __global__ void select_ray_surface_cells_kernel_b9b84ac1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::vec_t<3, wp::float32> var_ray0_origin,
    wp::vec_t<3, wp::float32> var_ray0_end,
    wp::vec_t<3, wp::float32> var_ray1_origin,
    wp::vec_t<3, wp::float32> var_ray1_end,
    wp::float32 var_padding,
    wp::array_t<wp::int32> var_selected_cells,
    wp::array_t<wp::int32> var_selected_count)
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
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        wp::int32* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        const wp::int32 var_25 = 1;
        wp::int32* var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 2;
        wp::int32* var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 3;
        wp::int32* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        const wp::int32 var_43 = 4;
        wp::int32* var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::int32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        const wp::int32 var_49 = 5;
        wp::int32* var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::int32 var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::int32 var_55 = 6;
        wp::int32* var_56;
        wp::vec_t<3, wp::float32>* var_57;
        wp::int32 var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        const wp::int32 var_61 = 7;
        wp::int32* var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        const wp::float32 var_67 = 0.125;
        wp::vec_t<3, wp::float32> var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        bool var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        bool var_76;
        const wp::int32 var_77 = 0;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
        //---------
        // forward
        // def select_ray_surface_cells_kernel(                                                   <L 449>
        // c = wp.tid()                                                                           <L 466>
        var_0 = builtin_tid1d();
        // if c >= num_cells:                                                                     <L 467>
        var_1 = (var_0 >= var_num_cells);
        if (var_1) {
            // return                                                                             <L 468>
            continue;
        }
        // if cell_active[c] == 0:                                                                <L 469>
        var_2 = wp::address(var_cell_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // return                                                                             <L 470>
            continue;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 471>
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
            // return                                                                             <L 472>
            continue;
        }
        // centre = wp.vec3(0.0, 0.0, 0.0)                                                        <L 474>
        var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
        // for local_node in range(8):                                                            <L 475>
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 476>
        var_20 = wp::address(var_cell_nodes, var_0, var_19);
        var_22 = wp::load(var_20);
        var_21 = wp::address(var_particle_q, var_22);
        var_24 = wp::load(var_21);
        var_23 = wp::add(var_18, var_24);
        var_26 = wp::address(var_cell_nodes, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::address(var_particle_q, var_28);
        var_30 = wp::load(var_27);
        var_29 = wp::add(var_23, var_30);
        var_32 = wp::address(var_cell_nodes, var_0, var_31);
        var_34 = wp::load(var_32);
        var_33 = wp::address(var_particle_q, var_34);
        var_36 = wp::load(var_33);
        var_35 = wp::add(var_29, var_36);
        var_38 = wp::address(var_cell_nodes, var_0, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::address(var_particle_q, var_40);
        var_42 = wp::load(var_39);
        var_41 = wp::add(var_35, var_42);
        var_44 = wp::address(var_cell_nodes, var_0, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::address(var_particle_q, var_46);
        var_48 = wp::load(var_45);
        var_47 = wp::add(var_41, var_48);
        var_50 = wp::address(var_cell_nodes, var_0, var_49);
        var_52 = wp::load(var_50);
        var_51 = wp::address(var_particle_q, var_52);
        var_54 = wp::load(var_51);
        var_53 = wp::add(var_47, var_54);
        var_56 = wp::address(var_cell_nodes, var_0, var_55);
        var_58 = wp::load(var_56);
        var_57 = wp::address(var_particle_q, var_58);
        var_60 = wp::load(var_57);
        var_59 = wp::add(var_53, var_60);
        var_62 = wp::address(var_cell_nodes, var_0, var_61);
        var_64 = wp::load(var_62);
        var_63 = wp::address(var_particle_q, var_64);
        var_66 = wp::load(var_63);
        var_65 = wp::add(var_59, var_66);
        // centre *= 0.125                                                                        <L 477>
        var_68 = wp::mul(var_65, var_67);
        // d0 = _point_triangle_distance_sq(centre, ray0_origin, ray1_origin, ray1_end)           <L 479>
        var_69 = _point_triangle_distance_sq_0(var_68, var_ray0_origin, var_ray1_origin, var_ray1_end);
        // d1 = _point_triangle_distance_sq(centre, ray0_origin, ray1_end, ray0_end)              <L 480>
        var_70 = _point_triangle_distance_sq_0(var_68, var_ray0_origin, var_ray1_end, var_ray0_end);
        // dist_sq = d0                                                                           <L 481>
        var_71 = wp::copy(var_69);
        // if d1 < dist_sq:                                                                       <L 482>
        var_72 = (var_70 < var_71);
        if (var_72) {
            // dist_sq = d1                                                                       <L 483>
            var_73 = wp::copy(var_70);
        }
        var_74 = wp::where(var_72, var_73, var_71);
        // if dist_sq > padding * padding:                                                        <L 484>
        var_75 = wp::mul(var_padding, var_padding);
        var_76 = (var_74 > var_75);
        if (var_76) {
            // return                                                                             <L 485>
            continue;
        }
        // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 487>
        var_79 = wp::atomic_add(var_selected_count, var_77, var_78);
        // selected_cells[out_idx] = c                                                            <L 488>
        wp::array_store(var_selected_cells, var_79, var_0);
    }
}



extern "C" __global__ void select_ray_surface_cells_kernel_b9b84ac1_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::vec_t<3, wp::float32> var_ray0_origin,
    wp::vec_t<3, wp::float32> var_ray0_end,
    wp::vec_t<3, wp::float32> var_ray1_origin,
    wp::vec_t<3, wp::float32> var_ray1_end,
    wp::float32 var_padding,
    wp::array_t<wp::int32> var_selected_cells,
    wp::array_t<wp::int32> var_selected_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_cell_material,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_material_cuttable,
    wp::int32 adj_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::vec_t<3, wp::float32> adj_ray0_origin,
    wp::vec_t<3, wp::float32> adj_ray0_end,
    wp::vec_t<3, wp::float32> adj_ray1_origin,
    wp::vec_t<3, wp::float32> adj_ray1_end,
    wp::float32 adj_padding,
    wp::array_t<wp::int32> adj_selected_cells,
    wp::array_t<wp::int32> adj_selected_count)
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
        const wp::float32 var_16 = 0.0;
        const wp::float32 var_17 = 0.0;
        wp::vec_t<3, wp::float32> var_18;
        const wp::int32 var_19 = 0;
        wp::int32* var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        const wp::int32 var_25 = 1;
        wp::int32* var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::int32 var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 2;
        wp::int32* var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 3;
        wp::int32* var_38;
        wp::vec_t<3, wp::float32>* var_39;
        wp::int32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        wp::vec_t<3, wp::float32> var_42;
        const wp::int32 var_43 = 4;
        wp::int32* var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::int32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        const wp::int32 var_49 = 5;
        wp::int32* var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::int32 var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32> var_54;
        const wp::int32 var_55 = 6;
        wp::int32* var_56;
        wp::vec_t<3, wp::float32>* var_57;
        wp::int32 var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        const wp::int32 var_61 = 7;
        wp::int32* var_62;
        wp::vec_t<3, wp::float32>* var_63;
        wp::int32 var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        const wp::float32 var_67 = 0.125;
        wp::vec_t<3, wp::float32> var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        bool var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        bool var_76;
        const wp::int32 var_77 = 0;
        const wp::int32 var_78 = 1;
        wp::int32 var_79;
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
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::int32 adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::int32 adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::int32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::vec_t<3, wp::float32> adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::int32 adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        wp::int32 adj_52 = {};
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::vec_t<3, wp::float32> adj_57 = {};
        wp::int32 adj_58 = {};
        wp::vec_t<3, wp::float32> adj_59 = {};
        wp::vec_t<3, wp::float32> adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::vec_t<3, wp::float32> adj_63 = {};
        wp::int32 adj_64 = {};
        wp::vec_t<3, wp::float32> adj_65 = {};
        wp::vec_t<3, wp::float32> adj_66 = {};
        wp::float32 adj_67 = {};
        wp::vec_t<3, wp::float32> adj_68 = {};
        wp::float32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::float32 adj_71 = {};
        bool adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        bool adj_76 = {};
        wp::int32 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        //---------
        // forward
        // def select_ray_surface_cells_kernel(                                                   <L 449>
        // c = wp.tid()                                                                           <L 466>
        var_0 = builtin_tid1d();
        // if c >= num_cells:                                                                     <L 467>
        var_1 = (var_0 >= var_num_cells);
        if (var_1) {
            // return                                                                             <L 468>
            goto label0;
        }
        // if cell_active[c] == 0:                                                                <L 469>
        var_2 = wp::address(var_cell_active, var_0);
        var_5 = wp::load(var_2);
        var_4 = (var_5 == var_3);
        if (var_4) {
            // return                                                                             <L 470>
            goto label1;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 471>
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
            // return                                                                             <L 472>
            goto label2;
        }
        // centre = wp.vec3(0.0, 0.0, 0.0)                                                        <L 474>
        var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
        // for local_node in range(8):                                                            <L 475>
        // centre += particle_q[cell_nodes[c, local_node]]                                        <L 476>
        var_20 = wp::address(var_cell_nodes, var_0, var_19);
        var_22 = wp::load(var_20);
        var_21 = wp::address(var_particle_q, var_22);
        var_24 = wp::load(var_21);
        var_23 = wp::add(var_18, var_24);
        var_26 = wp::address(var_cell_nodes, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::address(var_particle_q, var_28);
        var_30 = wp::load(var_27);
        var_29 = wp::add(var_23, var_30);
        var_32 = wp::address(var_cell_nodes, var_0, var_31);
        var_34 = wp::load(var_32);
        var_33 = wp::address(var_particle_q, var_34);
        var_36 = wp::load(var_33);
        var_35 = wp::add(var_29, var_36);
        var_38 = wp::address(var_cell_nodes, var_0, var_37);
        var_40 = wp::load(var_38);
        var_39 = wp::address(var_particle_q, var_40);
        var_42 = wp::load(var_39);
        var_41 = wp::add(var_35, var_42);
        var_44 = wp::address(var_cell_nodes, var_0, var_43);
        var_46 = wp::load(var_44);
        var_45 = wp::address(var_particle_q, var_46);
        var_48 = wp::load(var_45);
        var_47 = wp::add(var_41, var_48);
        var_50 = wp::address(var_cell_nodes, var_0, var_49);
        var_52 = wp::load(var_50);
        var_51 = wp::address(var_particle_q, var_52);
        var_54 = wp::load(var_51);
        var_53 = wp::add(var_47, var_54);
        var_56 = wp::address(var_cell_nodes, var_0, var_55);
        var_58 = wp::load(var_56);
        var_57 = wp::address(var_particle_q, var_58);
        var_60 = wp::load(var_57);
        var_59 = wp::add(var_53, var_60);
        var_62 = wp::address(var_cell_nodes, var_0, var_61);
        var_64 = wp::load(var_62);
        var_63 = wp::address(var_particle_q, var_64);
        var_66 = wp::load(var_63);
        var_65 = wp::add(var_59, var_66);
        // centre *= 0.125                                                                        <L 477>
        var_68 = wp::mul(var_65, var_67);
        // d0 = _point_triangle_distance_sq(centre, ray0_origin, ray1_origin, ray1_end)           <L 479>
        var_69 = _point_triangle_distance_sq_0(var_68, var_ray0_origin, var_ray1_origin, var_ray1_end);
        // d1 = _point_triangle_distance_sq(centre, ray0_origin, ray1_end, ray0_end)              <L 480>
        var_70 = _point_triangle_distance_sq_0(var_68, var_ray0_origin, var_ray1_end, var_ray0_end);
        // dist_sq = d0                                                                           <L 481>
        var_71 = wp::copy(var_69);
        // if d1 < dist_sq:                                                                       <L 482>
        var_72 = (var_70 < var_71);
        if (var_72) {
            // dist_sq = d1                                                                       <L 483>
            var_73 = wp::copy(var_70);
        }
        var_74 = wp::where(var_72, var_73, var_71);
        // if dist_sq > padding * padding:                                                        <L 484>
        var_75 = wp::mul(var_padding, var_padding);
        var_76 = (var_74 > var_75);
        if (var_76) {
            // return                                                                             <L 485>
            goto label3;
        }
        // out_idx = wp.atomic_add(selected_count, 0, 1)                                          <L 487>
        // var_79 = wp::atomic_add(var_selected_count, var_77, var_78);
        // selected_cells[out_idx] = c                                                            <L 488>
        // wp::array_store(var_selected_cells, var_79, var_0);
        //---------
        // reverse
        wp::adj_array_store(var_selected_cells, var_79, var_0, adj_selected_cells, adj_79, adj_0);
        // adj: selected_cells[out_idx] = c                                                       <L 488>
        wp::adj_atomic_add(var_selected_count, var_77, var_78, adj_selected_count, adj_77, adj_78, adj_79);
        // adj: out_idx = wp.atomic_add(selected_count, 0, 1)                                     <L 487>
        if (var_76) {
            label3:;
            // adj: return                                                                        <L 485>
        }
        wp::adj_mul(var_padding, var_padding, adj_padding, adj_padding, adj_75);
        // adj: if dist_sq > padding * padding:                                                   <L 484>
        wp::adj_where(var_72, var_73, var_71, adj_72, adj_73, adj_71, adj_74);
        if (var_72) {
            wp::adj_copy(var_70, adj_70, adj_73);
            // adj: dist_sq = d1                                                                  <L 483>
        }
        // adj: if d1 < dist_sq:                                                                  <L 482>
        wp::adj_copy(var_69, adj_69, adj_71);
        // adj: dist_sq = d0                                                                      <L 481>
        adj__point_triangle_distance_sq_0(var_68, var_ray0_origin, var_ray1_end, var_ray0_end, adj_68, adj_ray0_origin, adj_ray1_end, adj_ray0_end, adj_70);
        // adj: d1 = _point_triangle_distance_sq(centre, ray0_origin, ray1_end, ray0_end)         <L 480>
        adj__point_triangle_distance_sq_0(var_68, var_ray0_origin, var_ray1_origin, var_ray1_end, adj_68, adj_ray0_origin, adj_ray1_origin, adj_ray1_end, adj_69);
        // adj: d0 = _point_triangle_distance_sq(centre, ray0_origin, ray1_origin, ray1_end)      <L 479>
        wp::adj_mul(var_65, var_67, adj_65, adj_67, adj_68);
        // adj: centre *= 0.125                                                                   <L 477>
        wp::adj_add(var_59, var_66, adj_59, adj_63, adj_65);
        wp::adj_address(var_particle_q, var_64, adj_particle_q, adj_62, adj_63);
        wp::adj_address(var_cell_nodes, var_0, var_61, adj_cell_nodes, adj_0, adj_61, adj_62);
        wp::adj_add(var_53, var_60, adj_53, adj_57, adj_59);
        wp::adj_address(var_particle_q, var_58, adj_particle_q, adj_56, adj_57);
        wp::adj_address(var_cell_nodes, var_0, var_55, adj_cell_nodes, adj_0, adj_55, adj_56);
        wp::adj_add(var_47, var_54, adj_47, adj_51, adj_53);
        wp::adj_address(var_particle_q, var_52, adj_particle_q, adj_50, adj_51);
        wp::adj_address(var_cell_nodes, var_0, var_49, adj_cell_nodes, adj_0, adj_49, adj_50);
        wp::adj_add(var_41, var_48, adj_41, adj_45, adj_47);
        wp::adj_address(var_particle_q, var_46, adj_particle_q, adj_44, adj_45);
        wp::adj_address(var_cell_nodes, var_0, var_43, adj_cell_nodes, adj_0, adj_43, adj_44);
        wp::adj_add(var_35, var_42, adj_35, adj_39, adj_41);
        wp::adj_address(var_particle_q, var_40, adj_particle_q, adj_38, adj_39);
        wp::adj_address(var_cell_nodes, var_0, var_37, adj_cell_nodes, adj_0, adj_37, adj_38);
        wp::adj_add(var_29, var_36, adj_29, adj_33, adj_35);
        wp::adj_address(var_particle_q, var_34, adj_particle_q, adj_32, adj_33);
        wp::adj_address(var_cell_nodes, var_0, var_31, adj_cell_nodes, adj_0, adj_31, adj_32);
        wp::adj_add(var_23, var_30, adj_23, adj_27, adj_29);
        wp::adj_address(var_particle_q, var_28, adj_particle_q, adj_26, adj_27);
        wp::adj_address(var_cell_nodes, var_0, var_25, adj_cell_nodes, adj_0, adj_25, adj_26);
        wp::adj_add(var_18, var_24, adj_18, adj_21, adj_23);
        wp::adj_address(var_particle_q, var_22, adj_particle_q, adj_20, adj_21);
        wp::adj_address(var_cell_nodes, var_0, var_19, adj_cell_nodes, adj_0, adj_19, adj_20);
        // adj: centre += particle_q[cell_nodes[c, local_node]]                                   <L 476>
        // adj: for local_node in range(8):                                                       <L 475>
        wp::adj_vec_t(var_15, var_16, var_17, adj_15, adj_16, adj_17, adj_18);
        // adj: centre = wp.vec3(0.0, 0.0, 0.0)                                                   <L 474>
        if (var_6) {
            label2:;
            // adj: return                                                                        <L 472>
        }
        if (var_6) {
            wp::adj_address(var_material_cuttable, var_11, adj_material_cuttable, adj_9, adj_10);
            wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_9);
        }
        // adj: if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:         <L 471>
        if (var_4) {
            label1:;
            // adj: return                                                                        <L 470>
        }
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_2);
        // adj: if cell_active[c] == 0:                                                           <L 469>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 468>
        }
        // adj: if c >= num_cells:                                                                <L 467>
        // adj: c = wp.tid()                                                                      <L 466>
        // adj: def select_ray_surface_cells_kernel(                                              <L 449>
        continue;
    }
}



extern "C" __global__ void finalize_deleted_cell_nodes_kernel_5cf6ef37_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_locked_node_mask,
    wp::array_t<wp::float32> var_particle_mass,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags)
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
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 0;
        bool var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::int32* var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::int32 var_32;
        const wp::float32 var_33 = 0.0;
        const wp::float32 var_34 = 0.0;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::float32 var_37 = 1.0;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        const wp::float32 var_40 = 0.0;
        const wp::float32 var_41 = 0.0;
        const wp::int32 var_42 = 0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        wp::int32 var_45;
        wp::int32 var_46;
        //---------
        // forward
        // def finalize_deleted_cell_nodes_kernel(                                                <L 117>
        // cell_slot, local_node = wp.tid()                                                       <L 130>
        builtin_tid2d(var_0, var_1);
        // if cell_slot >= deleted_count[0]:                                                      <L 131>
        var_3 = wp::address(var_deleted_count, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 132>
            continue;
        }
        // cell_idx = deleted_cells[cell_slot]                                                    <L 134>
        var_6 = wp::address(var_deleted_cells, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 135>
        var_11 = (var_7 < var_10);
        var_9 = var_11;
        if (!var_9) {
            var_12 = (var_7 >= var_num_cells);
            var_9 = var_9 || var_12;
        }
        if (var_9) {
            // return                                                                             <L 136>
            continue;
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 138>
        var_13 = wp::address(var_cell_nodes, var_7, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // support = node_support_count[node_idx]                                                 <L 139>
        var_16 = wp::address(var_node_support_count, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // mass = node_mass[node_idx]                                                             <L 140>
        var_19 = wp::address(var_node_mass, var_14);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // flags = particle_flags[node_idx]                                                       <L 141>
        var_22 = wp::address(var_particle_flags, var_14);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // if support > 0:                                                                        <L 143>
        var_26 = (var_17 > var_25);
        if (var_26) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 144>
            var_28 = wp::bit_or(var_23, var_27);
            wp::array_store(var_particle_flags, var_14, var_28);
            // if locked_node_mask[node_idx] != 0:                                                <L 145>
            var_29 = wp::address(var_locked_node_mask, var_14);
            var_32 = wp::load(var_29);
            var_31 = (var_32 != var_30);
            if (var_31) {
                // particle_mass[node_idx] = 0.0                                                  <L 146>
                wp::array_store(var_particle_mass, var_14, var_33);
                // particle_inv_mass[node_idx] = 0.0                                              <L 147>
                wp::array_store(var_particle_inv_mass, var_14, var_34);
            }
            if (!var_31) {
                // elif mass > 0.0:                                                               <L 148>
                var_36 = (var_20 > var_35);
                if (var_36) {
                    // particle_mass[node_idx] = mass                                             <L 149>
                    wp::array_store(var_particle_mass, var_14, var_20);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 150>
                    var_38 = wp::div(var_37, var_20);
                    wp::array_store(var_particle_inv_mass, var_14, var_38);
                }
                if (!var_36) {
                    // particle_mass[node_idx] = 0.0                                              <L 152>
                    wp::array_store(var_particle_mass, var_14, var_39);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 153>
                    wp::array_store(var_particle_inv_mass, var_14, var_40);
                }
            }
        }
        if (!var_26) {
            // node_mass[node_idx] = 0.0                                                          <L 155>
            wp::array_store(var_node_mass, var_14, var_41);
            // locked_node_mask[node_idx] = 0                                                     <L 156>
            wp::array_store(var_locked_node_mask, var_14, var_42);
            // particle_mass[node_idx] = 0.0                                                      <L 157>
            wp::array_store(var_particle_mass, var_14, var_43);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 158>
            wp::array_store(var_particle_inv_mass, var_14, var_44);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 159>
            var_45 = wp::invert(var_27);
            var_46 = wp::bit_and(var_23, var_45);
            wp::array_store(var_particle_flags, var_14, var_46);
        }
    }
}



extern "C" __global__ void finalize_deleted_cell_nodes_kernel_5cf6ef37_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deleted_cells,
    wp::array_t<wp::int32> var_deleted_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_node_support_count,
    wp::array_t<wp::float32> var_node_mass,
    wp::array_t<wp::int32> var_locked_node_mask,
    wp::array_t<wp::float32> var_particle_mass,
    wp::array_t<wp::float32> var_particle_inv_mass,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> adj_deleted_cells,
    wp::array_t<wp::int32> adj_deleted_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_node_support_count,
    wp::array_t<wp::float32> adj_node_mass,
    wp::array_t<wp::int32> adj_locked_node_mask,
    wp::array_t<wp::float32> adj_particle_mass,
    wp::array_t<wp::float32> adj_particle_inv_mass,
    wp::array_t<wp::int32> adj_particle_flags)
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
        const wp::int32 var_2 = 0;
        wp::int32* var_3;
        bool var_4;
        wp::int32 var_5;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = 0;
        bool var_11;
        bool var_12;
        wp::int32* var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::int32* var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::int32 var_32;
        const wp::float32 var_33 = 0.0;
        const wp::float32 var_34 = 0.0;
        const wp::float32 var_35 = 0.0;
        bool var_36;
        const wp::float32 var_37 = 1.0;
        wp::float32 var_38;
        const wp::float32 var_39 = 0.0;
        const wp::float32 var_40 = 0.0;
        const wp::float32 var_41 = 0.0;
        const wp::int32 var_42 = 0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        wp::int32 var_45;
        wp::int32 var_46;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        bool adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        bool adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        bool adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        bool adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        bool adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        //---------
        // forward
        // def finalize_deleted_cell_nodes_kernel(                                                <L 117>
        // cell_slot, local_node = wp.tid()                                                       <L 130>
        builtin_tid2d(var_0, var_1);
        // if cell_slot >= deleted_count[0]:                                                      <L 131>
        var_3 = wp::address(var_deleted_count, var_2);
        var_5 = wp::load(var_3);
        var_4 = (var_0 >= var_5);
        if (var_4) {
            // return                                                                             <L 132>
            goto label0;
        }
        // cell_idx = deleted_cells[cell_slot]                                                    <L 134>
        var_6 = wp::address(var_deleted_cells, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // if cell_idx < 0 or cell_idx >= num_cells:                                              <L 135>
        var_11 = (var_7 < var_10);
        var_9 = var_11;
        if (!var_9) {
            var_12 = (var_7 >= var_num_cells);
            var_9 = var_9 || var_12;
        }
        if (var_9) {
            // return                                                                             <L 136>
            goto label1;
        }
        // node_idx = cell_nodes[cell_idx, local_node]                                            <L 138>
        var_13 = wp::address(var_cell_nodes, var_7, var_1);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // support = node_support_count[node_idx]                                                 <L 139>
        var_16 = wp::address(var_node_support_count, var_14);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // mass = node_mass[node_idx]                                                             <L 140>
        var_19 = wp::address(var_node_mass, var_14);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // flags = particle_flags[node_idx]                                                       <L 141>
        var_22 = wp::address(var_particle_flags, var_14);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // if support > 0:                                                                        <L 143>
        var_26 = (var_17 > var_25);
        if (var_26) {
            // particle_flags[node_idx] = flags | _ACTIVE_BIT                                     <L 144>
            var_28 = wp::bit_or(var_23, var_27);
            // wp::array_store(var_particle_flags, var_14, var_28);
            // if locked_node_mask[node_idx] != 0:                                                <L 145>
            var_29 = wp::address(var_locked_node_mask, var_14);
            var_32 = wp::load(var_29);
            var_31 = (var_32 != var_30);
            if (var_31) {
                // particle_mass[node_idx] = 0.0                                                  <L 146>
                // wp::array_store(var_particle_mass, var_14, var_33);
                // particle_inv_mass[node_idx] = 0.0                                              <L 147>
                // wp::array_store(var_particle_inv_mass, var_14, var_34);
            }
            if (!var_31) {
                // elif mass > 0.0:                                                               <L 148>
                var_36 = (var_20 > var_35);
                if (var_36) {
                    // particle_mass[node_idx] = mass                                             <L 149>
                    // wp::array_store(var_particle_mass, var_14, var_20);
                    // particle_inv_mass[node_idx] = 1.0 / mass                                   <L 150>
                    var_38 = wp::div(var_37, var_20);
                    // wp::array_store(var_particle_inv_mass, var_14, var_38);
                }
                if (!var_36) {
                    // particle_mass[node_idx] = 0.0                                              <L 152>
                    // wp::array_store(var_particle_mass, var_14, var_39);
                    // particle_inv_mass[node_idx] = 0.0                                          <L 153>
                    // wp::array_store(var_particle_inv_mass, var_14, var_40);
                }
            }
        }
        if (!var_26) {
            // node_mass[node_idx] = 0.0                                                          <L 155>
            // wp::array_store(var_node_mass, var_14, var_41);
            // locked_node_mask[node_idx] = 0                                                     <L 156>
            // wp::array_store(var_locked_node_mask, var_14, var_42);
            // particle_mass[node_idx] = 0.0                                                      <L 157>
            // wp::array_store(var_particle_mass, var_14, var_43);
            // particle_inv_mass[node_idx] = 0.0                                                  <L 158>
            // wp::array_store(var_particle_inv_mass, var_14, var_44);
            // particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                                  <L 159>
            var_45 = wp::invert(var_27);
            var_46 = wp::bit_and(var_23, var_45);
            // wp::array_store(var_particle_flags, var_14, var_46);
        }
        //---------
        // reverse
        if (!var_26) {
            wp::adj_array_store(var_particle_flags, var_14, var_46, adj_particle_flags, adj_14, adj_46);
            // adj: particle_flags[node_idx] = flags & (~_ACTIVE_BIT)                             <L 159>
            wp::adj_array_store(var_particle_inv_mass, var_14, var_44, adj_particle_inv_mass, adj_14, adj_44);
            // adj: particle_inv_mass[node_idx] = 0.0                                             <L 158>
            wp::adj_array_store(var_particle_mass, var_14, var_43, adj_particle_mass, adj_14, adj_43);
            // adj: particle_mass[node_idx] = 0.0                                                 <L 157>
            wp::adj_array_store(var_locked_node_mask, var_14, var_42, adj_locked_node_mask, adj_14, adj_42);
            // adj: locked_node_mask[node_idx] = 0                                                <L 156>
            wp::adj_array_store(var_node_mass, var_14, var_41, adj_node_mass, adj_14, adj_41);
            // adj: node_mass[node_idx] = 0.0                                                     <L 155>
        }
        if (var_26) {
            if (!var_31) {
                if (!var_36) {
                    wp::adj_array_store(var_particle_inv_mass, var_14, var_40, adj_particle_inv_mass, adj_14, adj_40);
                    // adj: particle_inv_mass[node_idx] = 0.0                                     <L 153>
                    wp::adj_array_store(var_particle_mass, var_14, var_39, adj_particle_mass, adj_14, adj_39);
                    // adj: particle_mass[node_idx] = 0.0                                         <L 152>
                }
                if (var_36) {
                    wp::adj_array_store(var_particle_inv_mass, var_14, var_38, adj_particle_inv_mass, adj_14, adj_38);
                    wp::adj_div(var_37, var_20, var_38, adj_37, adj_20, adj_38);
                    // adj: particle_inv_mass[node_idx] = 1.0 / mass                              <L 150>
                    wp::adj_array_store(var_particle_mass, var_14, var_20, adj_particle_mass, adj_14, adj_20);
                    // adj: particle_mass[node_idx] = mass                                        <L 149>
                }
                // adj: elif mass > 0.0:                                                          <L 148>
            }
            if (var_31) {
                wp::adj_array_store(var_particle_inv_mass, var_14, var_34, adj_particle_inv_mass, adj_14, adj_34);
                // adj: particle_inv_mass[node_idx] = 0.0                                         <L 147>
                wp::adj_array_store(var_particle_mass, var_14, var_33, adj_particle_mass, adj_14, adj_33);
                // adj: particle_mass[node_idx] = 0.0                                             <L 146>
            }
            wp::adj_address(var_locked_node_mask, var_14, adj_locked_node_mask, adj_14, adj_29);
            // adj: if locked_node_mask[node_idx] != 0:                                           <L 145>
            wp::adj_array_store(var_particle_flags, var_14, var_28, adj_particle_flags, adj_14, adj_28);
            // adj: particle_flags[node_idx] = flags | _ACTIVE_BIT                                <L 144>
        }
        // adj: if support > 0:                                                                   <L 143>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_particle_flags, var_14, adj_particle_flags, adj_14, adj_22);
        // adj: flags = particle_flags[node_idx]                                                  <L 141>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_address(var_node_mass, var_14, adj_node_mass, adj_14, adj_19);
        // adj: mass = node_mass[node_idx]                                                        <L 140>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_address(var_node_support_count, var_14, adj_node_support_count, adj_14, adj_16);
        // adj: support = node_support_count[node_idx]                                            <L 139>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_cell_nodes, var_7, var_1, adj_cell_nodes, adj_7, adj_1, adj_13);
        // adj: node_idx = cell_nodes[cell_idx, local_node]                                       <L 138>
        if (var_9) {
            label1:;
            // adj: return                                                                        <L 136>
        }
        if (!var_9) {
        }
        // adj: if cell_idx < 0 or cell_idx >= num_cells:                                         <L 135>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_deleted_cells, var_0, adj_deleted_cells, adj_0, adj_6);
        // adj: cell_idx = deleted_cells[cell_slot]                                               <L 134>
        if (var_4) {
            label0:;
            // adj: return                                                                        <L 132>
        }
        wp::adj_address(var_deleted_count, var_2, adj_deleted_count, adj_2, adj_3);
        // adj: if cell_slot >= deleted_count[0]:                                                 <L 131>
        // adj: cell_slot, local_node = wp.tid()                                                  <L 130>
        // adj: def finalize_deleted_cell_nodes_kernel(                                           <L 117>
        continue;
    }
}



extern "C" __global__ void finalize_deactivated_cluster_weights_kernel_0c98401a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deactivated_cluster_ids,
    wp::array_t<wp::int32> var_deactivated_count,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        bool var_3;
        wp::int32 var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        bool var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        const wp::float32 var_29 = 1.0;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 0;
        const wp::float32 var_33 = 0.0;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        //---------
        // forward
        // def finalize_deactivated_cluster_weights_kernel(                                       <L 248>
        // i = wp.tid()                                                                           <L 257>
        var_0 = builtin_tid1d();
        // if i >= deactivated_count[0]:                                                          <L 258>
        var_2 = wp::address(var_deactivated_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 259>
            continue;
        }
        // cluster_idx = deactivated_cluster_ids[i]                                               <L 261>
        var_5 = wp::address(var_deactivated_cluster_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if cluster_idx < 0:                                                                    <L 262>
        var_9 = (var_6 < var_8);
        if (var_9) {
            // return                                                                             <L 263>
            continue;
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 265>
        var_10 = wp::address(var_cluster_offsets, var_6);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 266>
        var_14 = wp::add(var_6, var_13);
        var_15 = wp::address(var_cluster_offsets, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // while cursor < end:                                                                    <L 267>
        start_while_2:;
        var_18 = (var_11 < var_16);
        if ((var_18) == false) goto end_while_2;
            // particle_idx = cluster_indices[cursor]                                             <L 268>
            var_19 = wp::address(var_cluster_indices, var_11);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // if particle_idx >= 0:                                                              <L 269>
            var_23 = (var_20 >= var_22);
            if (var_23) {
                // count = particle_cluster_counts[particle_idx]                                  <L 270>
                var_24 = wp::address(var_particle_cluster_counts, var_20);
                var_26 = wp::load(var_24);
                var_25 = wp::copy(var_26);
                // if count > 0:                                                                  <L 271>
                var_28 = (var_25 > var_27);
                if (var_28) {
                    // particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)            <L 272>
                    var_30 = wp::float(var_25);
                    var_31 = wp::div(var_29, var_30);
                    wp::array_store(var_particle_cluster_inv_weights, var_20, var_31);
                }
                if (!var_28) {
                    // particle_cluster_counts[particle_idx] = 0                                  <L 274>
                    wp::array_store(var_particle_cluster_counts, var_20, var_32);
                    // particle_cluster_inv_weights[particle_idx] = 0.0                           <L 275>
                    wp::array_store(var_particle_cluster_inv_weights, var_20, var_33);
                }
            }
            // cursor += 1                                                                        <L 276>
            var_35 = wp::add(var_11, var_34);
            wp::assign(var_11, var_35);
        goto start_while_2;
        end_while_2:;
    }
}



extern "C" __global__ void finalize_deactivated_cluster_weights_kernel_0c98401a_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_deactivated_cluster_ids,
    wp::array_t<wp::int32> var_deactivated_count,
    wp::array_t<wp::int32> var_cluster_offsets,
    wp::array_t<wp::int32> var_cluster_indices,
    wp::array_t<wp::int32> var_particle_cluster_counts,
    wp::array_t<wp::float32> var_particle_cluster_inv_weights,
    wp::array_t<wp::int32> adj_deactivated_cluster_ids,
    wp::array_t<wp::int32> adj_deactivated_count,
    wp::array_t<wp::int32> adj_cluster_offsets,
    wp::array_t<wp::int32> adj_cluster_indices,
    wp::array_t<wp::int32> adj_particle_cluster_counts,
    wp::array_t<wp::float32> adj_particle_cluster_inv_weights)
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
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 1;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        bool var_18;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        wp::int32* var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 0;
        bool var_28;
        const wp::float32 var_29 = 1.0;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 0;
        const wp::float32 var_33 = 0.0;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
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
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        bool adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        bool adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        //---------
        // forward
        // def finalize_deactivated_cluster_weights_kernel(                                       <L 248>
        // i = wp.tid()                                                                           <L 257>
        var_0 = builtin_tid1d();
        // if i >= deactivated_count[0]:                                                          <L 258>
        var_2 = wp::address(var_deactivated_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 259>
            goto label0;
        }
        // cluster_idx = deactivated_cluster_ids[i]                                               <L 261>
        var_5 = wp::address(var_deactivated_cluster_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if cluster_idx < 0:                                                                    <L 262>
        var_9 = (var_6 < var_8);
        if (var_9) {
            // return                                                                             <L 263>
            goto label1;
        }
        // cursor = cluster_offsets[cluster_idx]                                                  <L 265>
        var_10 = wp::address(var_cluster_offsets, var_6);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // end = cluster_offsets[cluster_idx + 1]                                                 <L 266>
        var_14 = wp::add(var_6, var_13);
        var_15 = wp::address(var_cluster_offsets, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // while cursor < end:                                                                    <L 267>
        //---------
        // reverse
        start_while_2:;
        var_18 = (var_11 < var_16);
        if ((var_18) == false) goto end_while_2;
        adj_19 = {};
        adj_20 = {};
        adj_21 = {};
        adj_22 = {};
        adj_23 = {};
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
            // particle_idx = cluster_indices[cursor]                                             <L 268>
            var_19 = wp::address(var_cluster_indices, var_11);
            var_21 = wp::load(var_19);
            var_20 = wp::copy(var_21);
            // if particle_idx >= 0:                                                              <L 269>
            var_23 = (var_20 >= var_22);
            if (var_23) {
                // count = particle_cluster_counts[particle_idx]                                  <L 270>
                var_24 = wp::address(var_particle_cluster_counts, var_20);
                var_26 = wp::load(var_24);
                var_25 = wp::copy(var_26);
                // if count > 0:                                                                  <L 271>
                var_28 = (var_25 > var_27);
                if (var_28) {
                    // particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)            <L 272>
                    var_30 = wp::float(var_25);
                    var_31 = wp::div(var_29, var_30);
                    // wp::array_store(var_particle_cluster_inv_weights, var_20, var_31);
                }
                if (!var_28) {
                    // particle_cluster_counts[particle_idx] = 0                                  <L 274>
                    // wp::array_store(var_particle_cluster_counts, var_20, var_32);
                    // particle_cluster_inv_weights[particle_idx] = 0.0                           <L 275>
                    // wp::array_store(var_particle_cluster_inv_weights, var_20, var_33);
                }
            }
            // cursor += 1                                                                        <L 276>
            var_35 = wp::add(var_11, var_34);
            wp::assign(var_11, var_35);
            wp::adj_assign(var_11, var_35, adj_11, adj_35);
            wp::adj_add(var_11, var_34, adj_11, adj_34, adj_35);
            // adj: cursor += 1                                                                   <L 276>
            if (var_23) {
                if (!var_28) {
                    wp::adj_array_store(var_particle_cluster_inv_weights, var_20, var_33, adj_particle_cluster_inv_weights, adj_20, adj_33);
                    // adj: particle_cluster_inv_weights[particle_idx] = 0.0                      <L 275>
                    wp::adj_array_store(var_particle_cluster_counts, var_20, var_32, adj_particle_cluster_counts, adj_20, adj_32);
                    // adj: particle_cluster_counts[particle_idx] = 0                             <L 274>
                }
                if (var_28) {
                    wp::adj_array_store(var_particle_cluster_inv_weights, var_20, var_31, adj_particle_cluster_inv_weights, adj_20, adj_31);
                    wp::adj_div(var_29, var_30, var_31, adj_29, adj_30, adj_31);
                    wp::adj_float(var_25, adj_25, adj_30);
                    // adj: particle_cluster_inv_weights[particle_idx] = 1.0 / float(count)       <L 272>
                }
                // adj: if count > 0:                                                             <L 271>
                wp::adj_copy(var_26, adj_24, adj_25);
                wp::adj_address(var_particle_cluster_counts, var_20, adj_particle_cluster_counts, adj_20, adj_24);
                // adj: count = particle_cluster_counts[particle_idx]                             <L 270>
            }
            // adj: if particle_idx >= 0:                                                         <L 269>
            wp::adj_copy(var_21, adj_19, adj_20);
            wp::adj_address(var_cluster_indices, var_11, adj_cluster_indices, adj_11, adj_19);
            // adj: particle_idx = cluster_indices[cursor]                                        <L 268>
        goto start_while_2;
        end_while_2:;
        // adj: while cursor < end:                                                               <L 267>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_address(var_cluster_offsets, var_14, adj_cluster_offsets, adj_14, adj_15);
        wp::adj_add(var_6, var_13, adj_6, adj_13, adj_14);
        // adj: end = cluster_offsets[cluster_idx + 1]                                            <L 266>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_cluster_offsets, var_6, adj_cluster_offsets, adj_6, adj_10);
        // adj: cursor = cluster_offsets[cluster_idx]                                             <L 265>
        if (var_9) {
            label1:;
            // adj: return                                                                        <L 263>
        }
        // adj: if cluster_idx < 0:                                                               <L 262>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_deactivated_cluster_ids, var_0, adj_deactivated_cluster_ids, adj_0, adj_5);
        // adj: cluster_idx = deactivated_cluster_ids[i]                                          <L 261>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 259>
        }
        wp::adj_address(var_deactivated_count, var_1, adj_deactivated_count, adj_1, adj_2);
        // adj: if i >= deactivated_count[0]:                                                     <L 258>
        // adj: i = wp.tid()                                                                      <L 257>
        // adj: def finalize_deactivated_cluster_weights_kernel(                                  <L 248>
        continue;
    }
}



extern "C" __global__ void validate_cell_active_kernel_3a16aa92_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_error_count,
    wp::array_t<wp::int32> var_first_error)
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
        bool var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 1;
        bool var_8;
        const wp::int32 var_9 = 0;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        const wp::int32 var_14 = 1000;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        //---------
        // forward
        // def validate_cell_active_kernel(                                                       <L 280>
        // i = wp.tid()                                                                           <L 285>
        var_0 = builtin_tid1d();
        // value = cell_active[i]                                                                 <L 286>
        var_1 = wp::address(var_cell_active, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if value != 0 and value != 1:                                                          <L 287>
        var_6 = (var_2 != var_5);
        var_4 = var_6;
        if (var_4) {
            var_8 = (var_2 != var_7);
            var_4 = var_4 && var_8;
        }
        if (var_4) {
            // old = wp.atomic_add(error_count, 0, 1)                                             <L 288>
            var_11 = wp::atomic_add(var_error_count, var_9, var_10);
            // if old == 0:                                                                       <L 289>
            var_13 = (var_11 == var_12);
            if (var_13) {
                // first_error[0] = 1000 + i                                                      <L 290>
                var_15 = wp::add(var_14, var_0);
                wp::array_store(var_first_error, var_16, var_15);
            }
        }
    }
}



extern "C" __global__ void validate_cell_active_kernel_3a16aa92_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_error_count,
    wp::array_t<wp::int32> var_first_error,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_error_count,
    wp::array_t<wp::int32> adj_first_error)
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
        bool var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 1;
        bool var_8;
        const wp::int32 var_9 = 0;
        const wp::int32 var_10 = 1;
        wp::int32 var_11;
        const wp::int32 var_12 = 0;
        bool var_13;
        const wp::int32 var_14 = 1000;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
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
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        //---------
        // forward
        // def validate_cell_active_kernel(                                                       <L 280>
        // i = wp.tid()                                                                           <L 285>
        var_0 = builtin_tid1d();
        // value = cell_active[i]                                                                 <L 286>
        var_1 = wp::address(var_cell_active, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if value != 0 and value != 1:                                                          <L 287>
        var_6 = (var_2 != var_5);
        var_4 = var_6;
        if (var_4) {
            var_8 = (var_2 != var_7);
            var_4 = var_4 && var_8;
        }
        if (var_4) {
            // old = wp.atomic_add(error_count, 0, 1)                                             <L 288>
            // var_11 = wp::atomic_add(var_error_count, var_9, var_10);
            // if old == 0:                                                                       <L 289>
            var_13 = (var_11 == var_12);
            if (var_13) {
                // first_error[0] = 1000 + i                                                      <L 290>
                var_15 = wp::add(var_14, var_0);
                // wp::array_store(var_first_error, var_16, var_15);
            }
        }
        //---------
        // reverse
        if (var_4) {
            if (var_13) {
                wp::adj_array_store(var_first_error, var_16, var_15, adj_first_error, adj_16, adj_15);
                wp::adj_add(var_14, var_0, adj_14, adj_0, adj_15);
                // adj: first_error[0] = 1000 + i                                                 <L 290>
            }
            // adj: if old == 0:                                                                  <L 289>
            wp::adj_atomic_add(var_error_count, var_9, var_10, adj_error_count, adj_9, adj_10, adj_11);
            // adj: old = wp.atomic_add(error_count, 0, 1)                                        <L 288>
        }
        if (var_4) {
        }
        // adj: if value != 0 and value != 1:                                                     <L 287>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_1);
        // adj: value = cell_active[i]                                                            <L 286>
        // adj: i = wp.tid()                                                                      <L 285>
        // adj: def validate_cell_active_kernel(                                                  <L 280>
        continue;
    }
}

