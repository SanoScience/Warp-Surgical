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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:845
static CUDA_CALLABLE wp::float32 _select_axis_0(
    wp::vec_t<3, wp::float32> var_uv,
    wp::int32 var_axis)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    //---------
    // forward
    // def _select_axis(uv: wp.vec3, axis: int) -> float:                                     <L 846>
    // if axis == 0:                                                                          <L 847>
    var_1 = (var_axis == var_0);
    if (var_1) {
        // return uv[0]                                                                       <L 848>
        var_3 = wp::extract(var_uv, var_2);
        return var_3;
    }
    // if axis == 1:                                                                          <L 849>
    var_5 = (var_axis == var_4);
    if (var_5) {
        // return uv[1]                                                                       <L 850>
        var_7 = wp::extract(var_uv, var_6);
        return var_7;
    }
    // return uv[2]                                                                           <L 851>
    var_9 = wp::extract(var_uv, var_8);
    return var_9;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:854
static CUDA_CALLABLE wp::float32 _scale_about_centre_0(
    wp::float32 var_v,
    wp::float32 var_s)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    //---------
    // forward
    // def _scale_about_centre(v: float, s: float) -> float:                                  <L 855>
    // return (v - 0.5) / s + 0.5                                                             <L 857>
    var_1 = wp::sub(var_v, var_0);
    var_2 = wp::div(var_1, var_s);
    var_4 = wp::add(var_2, var_3);
    return var_4;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1413
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _triangle_barycentric_0(
    wp::vec_t<2, wp::float32> var_p,
    wp::vec_t<2, wp::float32> var_a,
    wp::vec_t<2, wp::float32> var_b,
    wp::vec_t<2, wp::float32> var_c)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32> var_0;
    wp::vec_t<2, wp::float32> var_1;
    wp::vec_t<2, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 1e-08;
    bool var_16;
    const wp::float32 var_17 = 1.0;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 0.0;
    wp::vec_t<3, wp::float32> var_20;
    const wp::float32 var_21 = 1.0;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    const wp::int32 var_42 = 1;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    //---------
    // forward
    // def _triangle_barycentric(p: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.vec3:       <L 1414>
    // v0 = b - a                                                                             <L 1415>
    var_0 = wp::sub(var_b, var_a);
    // v1 = c - a                                                                             <L 1416>
    var_1 = wp::sub(var_c, var_a);
    // v2 = p - a                                                                             <L 1417>
    var_2 = wp::sub(var_p, var_a);
    // den = v0[0] * v1[1] - v1[0] * v0[1]                                                    <L 1418>
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_1, var_5);
    var_7 = wp::mul(var_4, var_6);
    var_9 = wp::extract(var_1, var_8);
    var_11 = wp::extract(var_0, var_10);
    var_12 = wp::mul(var_9, var_11);
    var_13 = wp::sub(var_7, var_12);
    // if wp.abs(den) < 1.0e-8:                                                               <L 1419>
    var_14 = wp::abs(var_13);
    var_16 = (var_14 < var_15);
    if (var_16) {
        // return wp.vec3(1.0, 0.0, 0.0)                                                      <L 1420>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        return var_20;
    }
    // inv_den = 1.0 / den                                                                    <L 1421>
    var_22 = wp::div(var_21, var_13);
    // w1 = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_den                                         <L 1422>
    var_24 = wp::extract(var_2, var_23);
    var_26 = wp::extract(var_1, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_1, var_28);
    var_31 = wp::extract(var_2, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::sub(var_27, var_32);
    var_34 = wp::mul(var_33, var_22);
    // w2 = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_den                                         <L 1423>
    var_36 = wp::extract(var_0, var_35);
    var_38 = wp::extract(var_2, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_41 = wp::extract(var_2, var_40);
    var_43 = wp::extract(var_0, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    var_46 = wp::mul(var_45, var_22);
    // w0 = 1.0 - w1 - w2                                                                     <L 1424>
    var_48 = wp::sub(var_47, var_34);
    var_49 = wp::sub(var_48, var_46);
    // return wp.vec3(w0, w1, w2)                                                             <L 1425>
    var_50 = wp::vec_t<3, wp::float32>(var_49, var_34, var_46);
    return var_50;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1428
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _clamp_barycentric_0(
    wp::vec_t<3, wp::float32> var_w)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-08;
    bool var_15;
    const wp::float32 var_16 = 1.0;
    const wp::float32 var_17 = 0.0;
    const wp::float32 var_18 = 0.0;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 1.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // forward
    // def _clamp_barycentric(w: wp.vec3) -> wp.vec3:                                         <L 1429>
    // c0 = wp.max(w[0], 0.0)                                                                 <L 1430>
    var_1 = wp::extract(var_w, var_0);
    var_3 = wp::max(var_1, var_2);
    // c1 = wp.max(w[1], 0.0)                                                                 <L 1431>
    var_5 = wp::extract(var_w, var_4);
    var_7 = wp::max(var_5, var_6);
    // c2 = wp.max(w[2], 0.0)                                                                 <L 1432>
    var_9 = wp::extract(var_w, var_8);
    var_11 = wp::max(var_9, var_10);
    // s = c0 + c1 + c2                                                                       <L 1433>
    var_12 = wp::add(var_3, var_7);
    var_13 = wp::add(var_12, var_11);
    // if s <= 1.0e-8:                                                                        <L 1434>
    var_15 = (var_13 <= var_14);
    if (var_15) {
        // return wp.vec3(1.0, 0.0, 0.0)                                                      <L 1435>
        var_19 = wp::vec_t<3, wp::float32>(var_16, var_17, var_18);
        return var_19;
    }
    // inv_s = 1.0 / s                                                                        <L 1436>
    var_21 = wp::div(var_20, var_13);
    // return wp.vec3(c0 * inv_s, c1 * inv_s, c2 * inv_s)                                     <L 1437>
    var_22 = wp::mul(var_3, var_21);
    var_23 = wp::mul(var_7, var_21);
    var_24 = wp::mul(var_11, var_21);
    var_25 = wp::vec_t<3, wp::float32>(var_22, var_23, var_24);
    return var_25;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1301
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _lerp_vec3_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::float32 var_t)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // forward
    // def _lerp_vec3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:                           <L 1302>
    // return a + (b - a) * t                                                                 <L 1303>
    var_0 = wp::sub(var_b, var_a);
    var_1 = wp::mul(var_0, var_t);
    var_2 = wp::add(var_a, var_1);
    return var_2;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1306
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _cold_warm_stress_color_0(
    wp::float32 var_value)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.02;
    const wp::float32 var_4 = 0.08;
    const wp::float32 var_5 = 0.7;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.72;
    const wp::float32 var_9 = 1.0;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = 0.05;
    const wp::float32 var_12 = 0.88;
    const wp::float32 var_13 = 0.18;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 1.0;
    const wp::float32 var_16 = 0.9;
    const wp::float32 var_17 = 0.05;
    wp::vec_t<3, wp::float32> var_18;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = 0.04;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 0.25;
    bool var_24;
    const wp::float32 var_25 = 4.0;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::float32 var_28 = 0.5;
    bool var_29;
    const wp::float32 var_30 = 0.25;
    wp::float32 var_31;
    const wp::float32 var_32 = 4.0;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::float32 var_35 = 0.75;
    bool var_36;
    const wp::float32 var_37 = 0.5;
    wp::float32 var_38;
    const wp::float32 var_39 = 4.0;
    wp::float32 var_40;
    wp::vec_t<3, wp::float32> var_41;
    const wp::float32 var_42 = 0.75;
    wp::float32 var_43;
    const wp::float32 var_44 = 4.0;
    wp::float32 var_45;
    wp::vec_t<3, wp::float32> var_46;
    //---------
    // forward
    // def _cold_warm_stress_color(value: float) -> wp.vec3:                                  <L 1307>
    // x = wp.clamp(value, 0.0, 1.0)                                                          <L 1309>
    var_2 = wp::clamp(var_value, var_0, var_1);
    // c0 = wp.vec3(0.02, 0.08, 0.70)                                                         <L 1310>
    var_6 = wp::vec_t<3, wp::float32>(var_3, var_4, var_5);
    // c1 = wp.vec3(0.00, 0.72, 1.00)                                                         <L 1311>
    var_10 = wp::vec_t<3, wp::float32>(var_7, var_8, var_9);
    // c2 = wp.vec3(0.05, 0.88, 0.18)                                                         <L 1312>
    var_14 = wp::vec_t<3, wp::float32>(var_11, var_12, var_13);
    // c3 = wp.vec3(1.00, 0.90, 0.05)                                                         <L 1313>
    var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
    // c4 = wp.vec3(1.00, 0.04, 0.00)                                                         <L 1314>
    var_22 = wp::vec_t<3, wp::float32>(var_19, var_20, var_21);
    // if x < 0.25:                                                                           <L 1315>
    var_24 = (var_2 < var_23);
    if (var_24) {
        // return _lerp_vec3(c0, c1, x * 4.0)                                                 <L 1316>
        var_26 = wp::mul(var_2, var_25);
        var_27 = _lerp_vec3_0(var_6, var_10, var_26);
        return var_27;
    }
    // if x < 0.50:                                                                           <L 1317>
    var_29 = (var_2 < var_28);
    if (var_29) {
        // return _lerp_vec3(c1, c2, (x - 0.25) * 4.0)                                        <L 1318>
        var_31 = wp::sub(var_2, var_30);
        var_33 = wp::mul(var_31, var_32);
        var_34 = _lerp_vec3_0(var_10, var_14, var_33);
        return var_34;
    }
    // if x < 0.75:                                                                           <L 1319>
    var_36 = (var_2 < var_35);
    if (var_36) {
        // return _lerp_vec3(c2, c3, (x - 0.50) * 4.0)                                        <L 1320>
        var_38 = wp::sub(var_2, var_37);
        var_40 = wp::mul(var_38, var_39);
        var_41 = _lerp_vec3_0(var_14, var_18, var_40);
        return var_41;
    }
    // return _lerp_vec3(c3, c4, (x - 0.75) * 4.0)                                            <L 1321>
    var_43 = wp::sub(var_2, var_42);
    var_45 = wp::mul(var_43, var_44);
    var_46 = _lerp_vec3_0(var_18, var_22, var_45);
    return var_46;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1674
static CUDA_CALLABLE wp::int32 _cluster_edge_slot_a_0(
    wp::int32 var_edge_idx)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    bool var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    const wp::int32 var_12 = 3;
    wp::int32 var_13;
    const wp::int32 var_14 = 4;
    bool var_15;
    const wp::int32 var_16 = 4;
    wp::int32 var_17;
    const wp::int32 var_18 = 5;
    bool var_19;
    const wp::int32 var_20 = 5;
    wp::int32 var_21;
    const wp::int32 var_22 = 6;
    bool var_23;
    const wp::int32 var_24 = 6;
    wp::int32 var_25;
    const wp::int32 var_26 = 7;
    bool var_27;
    const wp::int32 var_28 = 7;
    wp::int32 var_29;
    const wp::int32 var_30 = 9;
    bool var_31;
    const wp::int32 var_32 = 1;
    wp::int32 var_33;
    const wp::int32 var_34 = 10;
    bool var_35;
    const wp::int32 var_36 = 2;
    wp::int32 var_37;
    const wp::int32 var_38 = 11;
    bool var_39;
    const wp::int32 var_40 = 3;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    //---------
    // forward
    // def _cluster_edge_slot_a(edge_idx: int) -> int:                                        <L 1675>
    // slot = int(0)                                                                          <L 1676>
    var_1 = wp::int(var_0);
    // if edge_idx == 1:                                                                      <L 1677>
    var_3 = (var_edge_idx == var_2);
    if (var_3) {
        // slot = 1                                                                           <L 1678>
    }
    var_5 = wp::where(var_3, var_4, var_1);
    if (!var_3) {
        // elif edge_idx == 2:                                                                <L 1679>
        var_7 = (var_edge_idx == var_6);
        if (var_7) {
            // slot = 2                                                                       <L 1680>
        }
        var_9 = wp::where(var_7, var_8, var_5);
        if (!var_7) {
            // elif edge_idx == 3:                                                            <L 1681>
            var_11 = (var_edge_idx == var_10);
            if (var_11) {
                // slot = 3                                                                   <L 1682>
            }
            var_13 = wp::where(var_11, var_12, var_9);
            if (!var_11) {
                // elif edge_idx == 4:                                                        <L 1683>
                var_15 = (var_edge_idx == var_14);
                if (var_15) {
                    // slot = 4                                                               <L 1684>
                }
                var_17 = wp::where(var_15, var_16, var_13);
                if (!var_15) {
                    // elif edge_idx == 5:                                                    <L 1685>
                    var_19 = (var_edge_idx == var_18);
                    if (var_19) {
                        // slot = 5                                                           <L 1686>
                    }
                    var_21 = wp::where(var_19, var_20, var_17);
                    if (!var_19) {
                        // elif edge_idx == 6:                                                <L 1687>
                        var_23 = (var_edge_idx == var_22);
                        if (var_23) {
                            // slot = 6                                                       <L 1688>
                        }
                        var_25 = wp::where(var_23, var_24, var_21);
                        if (!var_23) {
                            // elif edge_idx == 7:                                            <L 1689>
                            var_27 = (var_edge_idx == var_26);
                            if (var_27) {
                                // slot = 7                                                   <L 1690>
                            }
                            var_29 = wp::where(var_27, var_28, var_25);
                            if (!var_27) {
                                // elif edge_idx == 9:                                        <L 1691>
                                var_31 = (var_edge_idx == var_30);
                                if (var_31) {
                                    // slot = 1                                               <L 1692>
                                }
                                var_33 = wp::where(var_31, var_32, var_29);
                                if (!var_31) {
                                    // elif edge_idx == 10:                                   <L 1693>
                                    var_35 = (var_edge_idx == var_34);
                                    if (var_35) {
                                        // slot = 2                                           <L 1694>
                                    }
                                    var_37 = wp::where(var_35, var_36, var_33);
                                    if (!var_35) {
                                        // elif edge_idx == 11:                               <L 1695>
                                        var_39 = (var_edge_idx == var_38);
                                        if (var_39) {
                                            // slot = 3                                       <L 1696>
                                        }
                                        var_41 = wp::where(var_39, var_40, var_37);
                                    }
                                    var_42 = wp::where(var_35, var_37, var_41);
                                }
                                var_43 = wp::where(var_31, var_33, var_42);
                            }
                            var_44 = wp::where(var_27, var_29, var_43);
                        }
                        var_45 = wp::where(var_23, var_25, var_44);
                    }
                    var_46 = wp::where(var_19, var_21, var_45);
                }
                var_47 = wp::where(var_15, var_17, var_46);
            }
            var_48 = wp::where(var_11, var_13, var_47);
        }
        var_49 = wp::where(var_7, var_9, var_48);
    }
    var_50 = wp::where(var_3, var_5, var_49);
    // return slot                                                                            <L 1697>
    return var_50;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1700
static CUDA_CALLABLE wp::int32 _cluster_edge_slot_b_0(
    wp::int32 var_edge_idx)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    bool var_3;
    const wp::int32 var_4 = 2;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    const wp::int32 var_14 = 4;
    bool var_15;
    const wp::int32 var_16 = 5;
    wp::int32 var_17;
    const wp::int32 var_18 = 5;
    bool var_19;
    const wp::int32 var_20 = 6;
    wp::int32 var_21;
    const wp::int32 var_22 = 6;
    bool var_23;
    const wp::int32 var_24 = 7;
    wp::int32 var_25;
    const wp::int32 var_26 = 7;
    bool var_27;
    const wp::int32 var_28 = 4;
    wp::int32 var_29;
    const wp::int32 var_30 = 8;
    bool var_31;
    const wp::int32 var_32 = 4;
    wp::int32 var_33;
    const wp::int32 var_34 = 9;
    bool var_35;
    const wp::int32 var_36 = 5;
    wp::int32 var_37;
    const wp::int32 var_38 = 10;
    bool var_39;
    const wp::int32 var_40 = 6;
    wp::int32 var_41;
    const wp::int32 var_42 = 11;
    bool var_43;
    const wp::int32 var_44 = 7;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    wp::int32 var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    //---------
    // forward
    // def _cluster_edge_slot_b(edge_idx: int) -> int:                                        <L 1701>
    // slot = int(1)                                                                          <L 1702>
    var_1 = wp::int(var_0);
    // if edge_idx == 1:                                                                      <L 1703>
    var_3 = (var_edge_idx == var_2);
    if (var_3) {
        // slot = 2                                                                           <L 1704>
    }
    var_5 = wp::where(var_3, var_4, var_1);
    if (!var_3) {
        // elif edge_idx == 2:                                                                <L 1705>
        var_7 = (var_edge_idx == var_6);
        if (var_7) {
            // slot = 3                                                                       <L 1706>
        }
        var_9 = wp::where(var_7, var_8, var_5);
        if (!var_7) {
            // elif edge_idx == 3:                                                            <L 1707>
            var_11 = (var_edge_idx == var_10);
            if (var_11) {
                // slot = 0                                                                   <L 1708>
            }
            var_13 = wp::where(var_11, var_12, var_9);
            if (!var_11) {
                // elif edge_idx == 4:                                                        <L 1709>
                var_15 = (var_edge_idx == var_14);
                if (var_15) {
                    // slot = 5                                                               <L 1710>
                }
                var_17 = wp::where(var_15, var_16, var_13);
                if (!var_15) {
                    // elif edge_idx == 5:                                                    <L 1711>
                    var_19 = (var_edge_idx == var_18);
                    if (var_19) {
                        // slot = 6                                                           <L 1712>
                    }
                    var_21 = wp::where(var_19, var_20, var_17);
                    if (!var_19) {
                        // elif edge_idx == 6:                                                <L 1713>
                        var_23 = (var_edge_idx == var_22);
                        if (var_23) {
                            // slot = 7                                                       <L 1714>
                        }
                        var_25 = wp::where(var_23, var_24, var_21);
                        if (!var_23) {
                            // elif edge_idx == 7:                                            <L 1715>
                            var_27 = (var_edge_idx == var_26);
                            if (var_27) {
                                // slot = 4                                                   <L 1716>
                            }
                            var_29 = wp::where(var_27, var_28, var_25);
                            if (!var_27) {
                                // elif edge_idx == 8:                                        <L 1717>
                                var_31 = (var_edge_idx == var_30);
                                if (var_31) {
                                    // slot = 4                                               <L 1718>
                                }
                                var_33 = wp::where(var_31, var_32, var_29);
                                if (!var_31) {
                                    // elif edge_idx == 9:                                    <L 1719>
                                    var_35 = (var_edge_idx == var_34);
                                    if (var_35) {
                                        // slot = 5                                           <L 1720>
                                    }
                                    var_37 = wp::where(var_35, var_36, var_33);
                                    if (!var_35) {
                                        // elif edge_idx == 10:                               <L 1721>
                                        var_39 = (var_edge_idx == var_38);
                                        if (var_39) {
                                            // slot = 6                                       <L 1722>
                                        }
                                        var_41 = wp::where(var_39, var_40, var_37);
                                        if (!var_39) {
                                            // elif edge_idx == 11:                           <L 1723>
                                            var_43 = (var_edge_idx == var_42);
                                            if (var_43) {
                                                // slot = 7                                   <L 1724>
                                            }
                                            var_45 = wp::where(var_43, var_44, var_41);
                                        }
                                        var_46 = wp::where(var_39, var_41, var_45);
                                    }
                                    var_47 = wp::where(var_35, var_37, var_46);
                                }
                                var_48 = wp::where(var_31, var_33, var_47);
                            }
                            var_49 = wp::where(var_27, var_29, var_48);
                        }
                        var_50 = wp::where(var_23, var_25, var_49);
                    }
                    var_51 = wp::where(var_19, var_21, var_50);
                }
                var_52 = wp::where(var_15, var_17, var_51);
            }
            var_53 = wp::where(var_11, var_13, var_52);
        }
        var_54 = wp::where(var_7, var_9, var_53);
    }
    var_55 = wp::where(var_3, var_5, var_54);
    // return slot                                                                            <L 1725>
    return var_55;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:845
static CUDA_CALLABLE void adj__select_axis_0(
    wp::vec_t<3, wp::float32> var_uv,
    wp::int32 var_axis,
    wp::vec_t<3, wp::float32> & adj_uv,
    wp::int32 & adj_axis,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    bool var_1;
    const wp::int32 var_2 = 0;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    bool adj_1 = {};
    wp::int32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::int32 adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float32 adj_9 = {};
    //---------
    // forward
    // def _select_axis(uv: wp.vec3, axis: int) -> float:                                     <L 846>
    // if axis == 0:                                                                          <L 847>
    var_1 = (var_axis == var_0);
    if (var_1) {
        // return uv[0]                                                                       <L 848>
        var_3 = wp::extract(var_uv, var_2);
        goto label0;
    }
    // if axis == 1:                                                                          <L 849>
    var_5 = (var_axis == var_4);
    if (var_5) {
        // return uv[1]                                                                       <L 850>
        var_7 = wp::extract(var_uv, var_6);
        goto label1;
    }
    // return uv[2]                                                                           <L 851>
    var_9 = wp::extract(var_uv, var_8);
    goto label2;
    //---------
    // reverse
    label2:;
    adj_9 += adj_ret;
    wp::adj_extract(var_uv, var_8, adj_uv, adj_8, adj_9);
    // adj: return uv[2]                                                                      <L 851>
    if (var_5) {
        label1:;
        adj_7 += adj_ret;
        wp::adj_extract(var_uv, var_6, adj_uv, adj_6, adj_7);
        // adj: return uv[1]                                                                  <L 850>
    }
    // adj: if axis == 1:                                                                     <L 849>
    if (var_1) {
        label0:;
        adj_3 += adj_ret;
        wp::adj_extract(var_uv, var_2, adj_uv, adj_2, adj_3);
        // adj: return uv[0]                                                                  <L 848>
    }
    // adj: if axis == 0:                                                                     <L 847>
    // adj: def _select_axis(uv: wp.vec3, axis: int) -> float:                                <L 846>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:854
static CUDA_CALLABLE void adj__scale_about_centre_0(
    wp::float32 var_v,
    wp::float32 var_s,
    wp::float32 & adj_v,
    wp::float32 & adj_s,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.5;
    wp::float32 var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.5;
    wp::float32 var_4;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    //---------
    // forward
    // def _scale_about_centre(v: float, s: float) -> float:                                  <L 855>
    // return (v - 0.5) / s + 0.5                                                             <L 857>
    var_1 = wp::sub(var_v, var_0);
    var_2 = wp::div(var_1, var_s);
    var_4 = wp::add(var_2, var_3);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_4 += adj_ret;
    wp::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    wp::adj_div(var_1, var_s, var_2, adj_1, adj_s, adj_2);
    wp::adj_sub(var_v, var_0, adj_v, adj_0, adj_1);
    // adj: return (v - 0.5) / s + 0.5                                                        <L 857>
    // adj: def _scale_about_centre(v: float, s: float) -> float:                             <L 855>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1413
static CUDA_CALLABLE void adj__triangle_barycentric_0(
    wp::vec_t<2, wp::float32> var_p,
    wp::vec_t<2, wp::float32> var_a,
    wp::vec_t<2, wp::float32> var_b,
    wp::vec_t<2, wp::float32> var_c,
    wp::vec_t<2, wp::float32> & adj_p,
    wp::vec_t<2, wp::float32> & adj_a,
    wp::vec_t<2, wp::float32> & adj_b,
    wp::vec_t<2, wp::float32> & adj_c,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<2, wp::float32> var_0;
    wp::vec_t<2, wp::float32> var_1;
    wp::vec_t<2, wp::float32> var_2;
    const wp::int32 var_3 = 0;
    wp::float32 var_4;
    const wp::int32 var_5 = 1;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::int32 var_8 = 0;
    wp::float32 var_9;
    const wp::int32 var_10 = 1;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    const wp::float32 var_15 = 1e-08;
    bool var_16;
    const wp::float32 var_17 = 1.0;
    const wp::float32 var_18 = 0.0;
    const wp::float32 var_19 = 0.0;
    wp::vec_t<3, wp::float32> var_20;
    const wp::float32 var_21 = 1.0;
    wp::float32 var_22;
    const wp::int32 var_23 = 0;
    wp::float32 var_24;
    const wp::int32 var_25 = 1;
    wp::float32 var_26;
    wp::float32 var_27;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    const wp::int32 var_30 = 1;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    const wp::int32 var_35 = 0;
    wp::float32 var_36;
    const wp::int32 var_37 = 1;
    wp::float32 var_38;
    wp::float32 var_39;
    const wp::int32 var_40 = 0;
    wp::float32 var_41;
    const wp::int32 var_42 = 1;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    const wp::float32 var_47 = 1.0;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::vec_t<3, wp::float32> var_50;
    //---------
    // dual vars
    wp::vec_t<2, wp::float32> adj_0 = {};
    wp::vec_t<2, wp::float32> adj_1 = {};
    wp::vec_t<2, wp::float32> adj_2 = {};
    wp::int32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    bool adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::int32 adj_28 = {};
    wp::float32 adj_29 = {};
    wp::int32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::int32 adj_35 = {};
    wp::float32 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::int32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::vec_t<3, wp::float32> adj_50 = {};
    //---------
    // forward
    // def _triangle_barycentric(p: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.vec3:       <L 1414>
    // v0 = b - a                                                                             <L 1415>
    var_0 = wp::sub(var_b, var_a);
    // v1 = c - a                                                                             <L 1416>
    var_1 = wp::sub(var_c, var_a);
    // v2 = p - a                                                                             <L 1417>
    var_2 = wp::sub(var_p, var_a);
    // den = v0[0] * v1[1] - v1[0] * v0[1]                                                    <L 1418>
    var_4 = wp::extract(var_0, var_3);
    var_6 = wp::extract(var_1, var_5);
    var_7 = wp::mul(var_4, var_6);
    var_9 = wp::extract(var_1, var_8);
    var_11 = wp::extract(var_0, var_10);
    var_12 = wp::mul(var_9, var_11);
    var_13 = wp::sub(var_7, var_12);
    // if wp.abs(den) < 1.0e-8:                                                               <L 1419>
    var_14 = wp::abs(var_13);
    var_16 = (var_14 < var_15);
    if (var_16) {
        // return wp.vec3(1.0, 0.0, 0.0)                                                      <L 1420>
        var_20 = wp::vec_t<3, wp::float32>(var_17, var_18, var_19);
        goto label0;
    }
    // inv_den = 1.0 / den                                                                    <L 1421>
    var_22 = wp::div(var_21, var_13);
    // w1 = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_den                                         <L 1422>
    var_24 = wp::extract(var_2, var_23);
    var_26 = wp::extract(var_1, var_25);
    var_27 = wp::mul(var_24, var_26);
    var_29 = wp::extract(var_1, var_28);
    var_31 = wp::extract(var_2, var_30);
    var_32 = wp::mul(var_29, var_31);
    var_33 = wp::sub(var_27, var_32);
    var_34 = wp::mul(var_33, var_22);
    // w2 = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_den                                         <L 1423>
    var_36 = wp::extract(var_0, var_35);
    var_38 = wp::extract(var_2, var_37);
    var_39 = wp::mul(var_36, var_38);
    var_41 = wp::extract(var_2, var_40);
    var_43 = wp::extract(var_0, var_42);
    var_44 = wp::mul(var_41, var_43);
    var_45 = wp::sub(var_39, var_44);
    var_46 = wp::mul(var_45, var_22);
    // w0 = 1.0 - w1 - w2                                                                     <L 1424>
    var_48 = wp::sub(var_47, var_34);
    var_49 = wp::sub(var_48, var_46);
    // return wp.vec3(w0, w1, w2)                                                             <L 1425>
    var_50 = wp::vec_t<3, wp::float32>(var_49, var_34, var_46);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_50 += adj_ret;
    wp::adj_vec_t(var_49, var_34, var_46, adj_49, adj_34, adj_46, adj_50);
    // adj: return wp.vec3(w0, w1, w2)                                                        <L 1425>
    wp::adj_sub(var_48, var_46, adj_48, adj_46, adj_49);
    wp::adj_sub(var_47, var_34, adj_47, adj_34, adj_48);
    // adj: w0 = 1.0 - w1 - w2                                                                <L 1424>
    wp::adj_mul(var_45, var_22, adj_45, adj_22, adj_46);
    wp::adj_sub(var_39, var_44, adj_39, adj_44, adj_45);
    wp::adj_mul(var_41, var_43, adj_41, adj_43, adj_44);
    wp::adj_extract(var_0, var_42, adj_0, adj_42, adj_43);
    wp::adj_extract(var_2, var_40, adj_2, adj_40, adj_41);
    wp::adj_mul(var_36, var_38, adj_36, adj_38, adj_39);
    wp::adj_extract(var_2, var_37, adj_2, adj_37, adj_38);
    wp::adj_extract(var_0, var_35, adj_0, adj_35, adj_36);
    // adj: w2 = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_den                                    <L 1423>
    wp::adj_mul(var_33, var_22, adj_33, adj_22, adj_34);
    wp::adj_sub(var_27, var_32, adj_27, adj_32, adj_33);
    wp::adj_mul(var_29, var_31, adj_29, adj_31, adj_32);
    wp::adj_extract(var_2, var_30, adj_2, adj_30, adj_31);
    wp::adj_extract(var_1, var_28, adj_1, adj_28, adj_29);
    wp::adj_mul(var_24, var_26, adj_24, adj_26, adj_27);
    wp::adj_extract(var_1, var_25, adj_1, adj_25, adj_26);
    wp::adj_extract(var_2, var_23, adj_2, adj_23, adj_24);
    // adj: w1 = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_den                                    <L 1422>
    wp::adj_div(var_21, var_13, var_22, adj_21, adj_13, adj_22);
    // adj: inv_den = 1.0 / den                                                               <L 1421>
    if (var_16) {
        label0:;
        adj_20 += adj_ret;
        wp::adj_vec_t(var_17, var_18, var_19, adj_17, adj_18, adj_19, adj_20);
        // adj: return wp.vec3(1.0, 0.0, 0.0)                                                 <L 1420>
    }
    wp::adj_abs(var_13, adj_13, adj_14);
    // adj: if wp.abs(den) < 1.0e-8:                                                          <L 1419>
    wp::adj_sub(var_7, var_12, adj_7, adj_12, adj_13);
    wp::adj_mul(var_9, var_11, adj_9, adj_11, adj_12);
    wp::adj_extract(var_0, var_10, adj_0, adj_10, adj_11);
    wp::adj_extract(var_1, var_8, adj_1, adj_8, adj_9);
    wp::adj_mul(var_4, var_6, adj_4, adj_6, adj_7);
    wp::adj_extract(var_1, var_5, adj_1, adj_5, adj_6);
    wp::adj_extract(var_0, var_3, adj_0, adj_3, adj_4);
    // adj: den = v0[0] * v1[1] - v1[0] * v0[1]                                               <L 1418>
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_2);
    // adj: v2 = p - a                                                                        <L 1417>
    wp::adj_sub(var_c, var_a, adj_c, adj_a, adj_1);
    // adj: v1 = c - a                                                                        <L 1416>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: v0 = b - a                                                                        <L 1415>
    // adj: def _triangle_barycentric(p: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.vec3:  <L 1414>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1428
static CUDA_CALLABLE void adj__clamp_barycentric_0(
    wp::vec_t<3, wp::float32> var_w,
    wp::vec_t<3, wp::float32> & adj_w,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::float32 var_1;
    const wp::float32 var_2 = 0.0;
    wp::float32 var_3;
    const wp::int32 var_4 = 1;
    wp::float32 var_5;
    const wp::float32 var_6 = 0.0;
    wp::float32 var_7;
    const wp::int32 var_8 = 2;
    wp::float32 var_9;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-08;
    bool var_15;
    const wp::float32 var_16 = 1.0;
    const wp::float32 var_17 = 0.0;
    const wp::float32 var_18 = 0.0;
    wp::vec_t<3, wp::float32> var_19;
    const wp::float32 var_20 = 1.0;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    bool adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    //---------
    // forward
    // def _clamp_barycentric(w: wp.vec3) -> wp.vec3:                                         <L 1429>
    // c0 = wp.max(w[0], 0.0)                                                                 <L 1430>
    var_1 = wp::extract(var_w, var_0);
    var_3 = wp::max(var_1, var_2);
    // c1 = wp.max(w[1], 0.0)                                                                 <L 1431>
    var_5 = wp::extract(var_w, var_4);
    var_7 = wp::max(var_5, var_6);
    // c2 = wp.max(w[2], 0.0)                                                                 <L 1432>
    var_9 = wp::extract(var_w, var_8);
    var_11 = wp::max(var_9, var_10);
    // s = c0 + c1 + c2                                                                       <L 1433>
    var_12 = wp::add(var_3, var_7);
    var_13 = wp::add(var_12, var_11);
    // if s <= 1.0e-8:                                                                        <L 1434>
    var_15 = (var_13 <= var_14);
    if (var_15) {
        // return wp.vec3(1.0, 0.0, 0.0)                                                      <L 1435>
        var_19 = wp::vec_t<3, wp::float32>(var_16, var_17, var_18);
        goto label0;
    }
    // inv_s = 1.0 / s                                                                        <L 1436>
    var_21 = wp::div(var_20, var_13);
    // return wp.vec3(c0 * inv_s, c1 * inv_s, c2 * inv_s)                                     <L 1437>
    var_22 = wp::mul(var_3, var_21);
    var_23 = wp::mul(var_7, var_21);
    var_24 = wp::mul(var_11, var_21);
    var_25 = wp::vec_t<3, wp::float32>(var_22, var_23, var_24);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_25 += adj_ret;
    wp::adj_vec_t(var_22, var_23, var_24, adj_22, adj_23, adj_24, adj_25);
    wp::adj_mul(var_11, var_21, adj_11, adj_21, adj_24);
    wp::adj_mul(var_7, var_21, adj_7, adj_21, adj_23);
    wp::adj_mul(var_3, var_21, adj_3, adj_21, adj_22);
    // adj: return wp.vec3(c0 * inv_s, c1 * inv_s, c2 * inv_s)                                <L 1437>
    wp::adj_div(var_20, var_13, var_21, adj_20, adj_13, adj_21);
    // adj: inv_s = 1.0 / s                                                                   <L 1436>
    if (var_15) {
        label0:;
        adj_19 += adj_ret;
        wp::adj_vec_t(var_16, var_17, var_18, adj_16, adj_17, adj_18, adj_19);
        // adj: return wp.vec3(1.0, 0.0, 0.0)                                                 <L 1435>
    }
    // adj: if s <= 1.0e-8:                                                                   <L 1434>
    wp::adj_add(var_12, var_11, adj_12, adj_11, adj_13);
    wp::adj_add(var_3, var_7, adj_3, adj_7, adj_12);
    // adj: s = c0 + c1 + c2                                                                  <L 1433>
    wp::adj_max(var_9, var_10, adj_9, adj_10, adj_11);
    wp::adj_extract(var_w, var_8, adj_w, adj_8, adj_9);
    // adj: c2 = wp.max(w[2], 0.0)                                                            <L 1432>
    wp::adj_max(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_extract(var_w, var_4, adj_w, adj_4, adj_5);
    // adj: c1 = wp.max(w[1], 0.0)                                                            <L 1431>
    wp::adj_max(var_1, var_2, adj_1, adj_2, adj_3);
    wp::adj_extract(var_w, var_0, adj_w, adj_0, adj_1);
    // adj: c0 = wp.max(w[0], 0.0)                                                            <L 1430>
    // adj: def _clamp_barycentric(w: wp.vec3) -> wp.vec3:                                    <L 1429>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1301
static CUDA_CALLABLE void adj__lerp_vec3_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::float32 var_t,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::float32 & adj_t,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::vec_t<3, wp::float32> var_2;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    //---------
    // forward
    // def _lerp_vec3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:                           <L 1302>
    // return a + (b - a) * t                                                                 <L 1303>
    var_0 = wp::sub(var_b, var_a);
    var_1 = wp::mul(var_0, var_t);
    var_2 = wp::add(var_a, var_1);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_2 += adj_ret;
    wp::adj_add(var_a, var_1, adj_a, adj_1, adj_2);
    wp::adj_mul(var_0, var_t, adj_0, adj_t, adj_1);
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: return a + (b - a) * t                                                            <L 1303>
    // adj: def _lerp_vec3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:                      <L 1302>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1306
static CUDA_CALLABLE void adj__cold_warm_stress_color_0(
    wp::float32 var_value,
    wp::float32 & adj_value,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 0.0;
    const wp::float32 var_1 = 1.0;
    wp::float32 var_2;
    const wp::float32 var_3 = 0.02;
    const wp::float32 var_4 = 0.08;
    const wp::float32 var_5 = 0.7;
    wp::vec_t<3, wp::float32> var_6;
    const wp::float32 var_7 = 0.0;
    const wp::float32 var_8 = 0.72;
    const wp::float32 var_9 = 1.0;
    wp::vec_t<3, wp::float32> var_10;
    const wp::float32 var_11 = 0.05;
    const wp::float32 var_12 = 0.88;
    const wp::float32 var_13 = 0.18;
    wp::vec_t<3, wp::float32> var_14;
    const wp::float32 var_15 = 1.0;
    const wp::float32 var_16 = 0.9;
    const wp::float32 var_17 = 0.05;
    wp::vec_t<3, wp::float32> var_18;
    const wp::float32 var_19 = 1.0;
    const wp::float32 var_20 = 0.04;
    const wp::float32 var_21 = 0.0;
    wp::vec_t<3, wp::float32> var_22;
    const wp::float32 var_23 = 0.25;
    bool var_24;
    const wp::float32 var_25 = 4.0;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    const wp::float32 var_28 = 0.5;
    bool var_29;
    const wp::float32 var_30 = 0.25;
    wp::float32 var_31;
    const wp::float32 var_32 = 4.0;
    wp::float32 var_33;
    wp::vec_t<3, wp::float32> var_34;
    const wp::float32 var_35 = 0.75;
    bool var_36;
    const wp::float32 var_37 = 0.5;
    wp::float32 var_38;
    const wp::float32 var_39 = 4.0;
    wp::float32 var_40;
    wp::vec_t<3, wp::float32> var_41;
    const wp::float32 var_42 = 0.75;
    wp::float32 var_43;
    const wp::float32 var_44 = 4.0;
    wp::float32 var_45;
    wp::vec_t<3, wp::float32> var_46;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    wp::float32 adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::vec_t<3, wp::float32> adj_22 = {};
    wp::float32 adj_23 = {};
    bool adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::float32 adj_28 = {};
    bool adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::vec_t<3, wp::float32> adj_34 = {};
    wp::float32 adj_35 = {};
    bool adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::vec_t<3, wp::float32> adj_46 = {};
    //---------
    // forward
    // def _cold_warm_stress_color(value: float) -> wp.vec3:                                  <L 1307>
    // x = wp.clamp(value, 0.0, 1.0)                                                          <L 1309>
    var_2 = wp::clamp(var_value, var_0, var_1);
    // c0 = wp.vec3(0.02, 0.08, 0.70)                                                         <L 1310>
    var_6 = wp::vec_t<3, wp::float32>(var_3, var_4, var_5);
    // c1 = wp.vec3(0.00, 0.72, 1.00)                                                         <L 1311>
    var_10 = wp::vec_t<3, wp::float32>(var_7, var_8, var_9);
    // c2 = wp.vec3(0.05, 0.88, 0.18)                                                         <L 1312>
    var_14 = wp::vec_t<3, wp::float32>(var_11, var_12, var_13);
    // c3 = wp.vec3(1.00, 0.90, 0.05)                                                         <L 1313>
    var_18 = wp::vec_t<3, wp::float32>(var_15, var_16, var_17);
    // c4 = wp.vec3(1.00, 0.04, 0.00)                                                         <L 1314>
    var_22 = wp::vec_t<3, wp::float32>(var_19, var_20, var_21);
    // if x < 0.25:                                                                           <L 1315>
    var_24 = (var_2 < var_23);
    if (var_24) {
        // return _lerp_vec3(c0, c1, x * 4.0)                                                 <L 1316>
        var_26 = wp::mul(var_2, var_25);
        var_27 = _lerp_vec3_0(var_6, var_10, var_26);
        goto label0;
    }
    // if x < 0.50:                                                                           <L 1317>
    var_29 = (var_2 < var_28);
    if (var_29) {
        // return _lerp_vec3(c1, c2, (x - 0.25) * 4.0)                                        <L 1318>
        var_31 = wp::sub(var_2, var_30);
        var_33 = wp::mul(var_31, var_32);
        var_34 = _lerp_vec3_0(var_10, var_14, var_33);
        goto label1;
    }
    // if x < 0.75:                                                                           <L 1319>
    var_36 = (var_2 < var_35);
    if (var_36) {
        // return _lerp_vec3(c2, c3, (x - 0.50) * 4.0)                                        <L 1320>
        var_38 = wp::sub(var_2, var_37);
        var_40 = wp::mul(var_38, var_39);
        var_41 = _lerp_vec3_0(var_14, var_18, var_40);
        goto label2;
    }
    // return _lerp_vec3(c3, c4, (x - 0.75) * 4.0)                                            <L 1321>
    var_43 = wp::sub(var_2, var_42);
    var_45 = wp::mul(var_43, var_44);
    var_46 = _lerp_vec3_0(var_18, var_22, var_45);
    goto label3;
    //---------
    // reverse
    label3:;
    adj_46 += adj_ret;
    adj__lerp_vec3_0(var_18, var_22, var_45, adj_18, adj_22, adj_45, adj_46);
    wp::adj_mul(var_43, var_44, adj_43, adj_44, adj_45);
    wp::adj_sub(var_2, var_42, adj_2, adj_42, adj_43);
    // adj: return _lerp_vec3(c3, c4, (x - 0.75) * 4.0)                                       <L 1321>
    if (var_36) {
        label2:;
        adj_41 += adj_ret;
        adj__lerp_vec3_0(var_14, var_18, var_40, adj_14, adj_18, adj_40, adj_41);
        wp::adj_mul(var_38, var_39, adj_38, adj_39, adj_40);
        wp::adj_sub(var_2, var_37, adj_2, adj_37, adj_38);
        // adj: return _lerp_vec3(c2, c3, (x - 0.50) * 4.0)                                   <L 1320>
    }
    // adj: if x < 0.75:                                                                      <L 1319>
    if (var_29) {
        label1:;
        adj_34 += adj_ret;
        adj__lerp_vec3_0(var_10, var_14, var_33, adj_10, adj_14, adj_33, adj_34);
        wp::adj_mul(var_31, var_32, adj_31, adj_32, adj_33);
        wp::adj_sub(var_2, var_30, adj_2, adj_30, adj_31);
        // adj: return _lerp_vec3(c1, c2, (x - 0.25) * 4.0)                                   <L 1318>
    }
    // adj: if x < 0.50:                                                                      <L 1317>
    if (var_24) {
        label0:;
        adj_27 += adj_ret;
        adj__lerp_vec3_0(var_6, var_10, var_26, adj_6, adj_10, adj_26, adj_27);
        wp::adj_mul(var_2, var_25, adj_2, adj_25, adj_26);
        // adj: return _lerp_vec3(c0, c1, x * 4.0)                                            <L 1316>
    }
    // adj: if x < 0.25:                                                                      <L 1315>
    wp::adj_vec_t(var_19, var_20, var_21, adj_19, adj_20, adj_21, adj_22);
    // adj: c4 = wp.vec3(1.00, 0.04, 0.00)                                                    <L 1314>
    wp::adj_vec_t(var_15, var_16, var_17, adj_15, adj_16, adj_17, adj_18);
    // adj: c3 = wp.vec3(1.00, 0.90, 0.05)                                                    <L 1313>
    wp::adj_vec_t(var_11, var_12, var_13, adj_11, adj_12, adj_13, adj_14);
    // adj: c2 = wp.vec3(0.05, 0.88, 0.18)                                                    <L 1312>
    wp::adj_vec_t(var_7, var_8, var_9, adj_7, adj_8, adj_9, adj_10);
    // adj: c1 = wp.vec3(0.00, 0.72, 1.00)                                                    <L 1311>
    wp::adj_vec_t(var_3, var_4, var_5, adj_3, adj_4, adj_5, adj_6);
    // adj: c0 = wp.vec3(0.02, 0.08, 0.70)                                                    <L 1310>
    wp::adj_clamp(var_value, var_0, var_1, adj_value, adj_0, adj_1, adj_2);
    // adj: x = wp.clamp(value, 0.0, 1.0)                                                     <L 1309>
    // adj: def _cold_warm_stress_color(value: float) -> wp.vec3:                             <L 1307>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1674
static CUDA_CALLABLE void adj__cluster_edge_slot_a_0(
    wp::int32 var_edge_idx,
    wp::int32 & adj_edge_idx,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    bool var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    const wp::int32 var_12 = 3;
    wp::int32 var_13;
    const wp::int32 var_14 = 4;
    bool var_15;
    const wp::int32 var_16 = 4;
    wp::int32 var_17;
    const wp::int32 var_18 = 5;
    bool var_19;
    const wp::int32 var_20 = 5;
    wp::int32 var_21;
    const wp::int32 var_22 = 6;
    bool var_23;
    const wp::int32 var_24 = 6;
    wp::int32 var_25;
    const wp::int32 var_26 = 7;
    bool var_27;
    const wp::int32 var_28 = 7;
    wp::int32 var_29;
    const wp::int32 var_30 = 9;
    bool var_31;
    const wp::int32 var_32 = 1;
    wp::int32 var_33;
    const wp::int32 var_34 = 10;
    bool var_35;
    const wp::int32 var_36 = 2;
    wp::int32 var_37;
    const wp::int32 var_38 = 11;
    bool var_39;
    const wp::int32 var_40 = 3;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    wp::int32 var_49;
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
    bool adj_7 = {};
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    bool adj_11 = {};
    wp::int32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    bool adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    bool adj_19 = {};
    wp::int32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    bool adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    bool adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::int32 adj_30 = {};
    bool adj_31 = {};
    wp::int32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::int32 adj_34 = {};
    bool adj_35 = {};
    wp::int32 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::int32 adj_38 = {};
    bool adj_39 = {};
    wp::int32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::int32 adj_43 = {};
    wp::int32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::int32 adj_48 = {};
    wp::int32 adj_49 = {};
    wp::int32 adj_50 = {};
    //---------
    // forward
    // def _cluster_edge_slot_a(edge_idx: int) -> int:                                        <L 1675>
    // slot = int(0)                                                                          <L 1676>
    var_1 = wp::int(var_0);
    // if edge_idx == 1:                                                                      <L 1677>
    var_3 = (var_edge_idx == var_2);
    if (var_3) {
        // slot = 1                                                                           <L 1678>
    }
    var_5 = wp::where(var_3, var_4, var_1);
    if (!var_3) {
        // elif edge_idx == 2:                                                                <L 1679>
        var_7 = (var_edge_idx == var_6);
        if (var_7) {
            // slot = 2                                                                       <L 1680>
        }
        var_9 = wp::where(var_7, var_8, var_5);
        if (!var_7) {
            // elif edge_idx == 3:                                                            <L 1681>
            var_11 = (var_edge_idx == var_10);
            if (var_11) {
                // slot = 3                                                                   <L 1682>
            }
            var_13 = wp::where(var_11, var_12, var_9);
            if (!var_11) {
                // elif edge_idx == 4:                                                        <L 1683>
                var_15 = (var_edge_idx == var_14);
                if (var_15) {
                    // slot = 4                                                               <L 1684>
                }
                var_17 = wp::where(var_15, var_16, var_13);
                if (!var_15) {
                    // elif edge_idx == 5:                                                    <L 1685>
                    var_19 = (var_edge_idx == var_18);
                    if (var_19) {
                        // slot = 5                                                           <L 1686>
                    }
                    var_21 = wp::where(var_19, var_20, var_17);
                    if (!var_19) {
                        // elif edge_idx == 6:                                                <L 1687>
                        var_23 = (var_edge_idx == var_22);
                        if (var_23) {
                            // slot = 6                                                       <L 1688>
                        }
                        var_25 = wp::where(var_23, var_24, var_21);
                        if (!var_23) {
                            // elif edge_idx == 7:                                            <L 1689>
                            var_27 = (var_edge_idx == var_26);
                            if (var_27) {
                                // slot = 7                                                   <L 1690>
                            }
                            var_29 = wp::where(var_27, var_28, var_25);
                            if (!var_27) {
                                // elif edge_idx == 9:                                        <L 1691>
                                var_31 = (var_edge_idx == var_30);
                                if (var_31) {
                                    // slot = 1                                               <L 1692>
                                }
                                var_33 = wp::where(var_31, var_32, var_29);
                                if (!var_31) {
                                    // elif edge_idx == 10:                                   <L 1693>
                                    var_35 = (var_edge_idx == var_34);
                                    if (var_35) {
                                        // slot = 2                                           <L 1694>
                                    }
                                    var_37 = wp::where(var_35, var_36, var_33);
                                    if (!var_35) {
                                        // elif edge_idx == 11:                               <L 1695>
                                        var_39 = (var_edge_idx == var_38);
                                        if (var_39) {
                                            // slot = 3                                       <L 1696>
                                        }
                                        var_41 = wp::where(var_39, var_40, var_37);
                                    }
                                    var_42 = wp::where(var_35, var_37, var_41);
                                }
                                var_43 = wp::where(var_31, var_33, var_42);
                            }
                            var_44 = wp::where(var_27, var_29, var_43);
                        }
                        var_45 = wp::where(var_23, var_25, var_44);
                    }
                    var_46 = wp::where(var_19, var_21, var_45);
                }
                var_47 = wp::where(var_15, var_17, var_46);
            }
            var_48 = wp::where(var_11, var_13, var_47);
        }
        var_49 = wp::where(var_7, var_9, var_48);
    }
    var_50 = wp::where(var_3, var_5, var_49);
    // return slot                                                                            <L 1697>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_50 += adj_ret;
    // adj: return slot                                                                       <L 1697>
    wp::adj_where(var_3, var_5, var_49, adj_3, adj_5, adj_49, adj_50);
    if (!var_3) {
        wp::adj_where(var_7, var_9, var_48, adj_7, adj_9, adj_48, adj_49);
        if (!var_7) {
            wp::adj_where(var_11, var_13, var_47, adj_11, adj_13, adj_47, adj_48);
            if (!var_11) {
                wp::adj_where(var_15, var_17, var_46, adj_15, adj_17, adj_46, adj_47);
                if (!var_15) {
                    wp::adj_where(var_19, var_21, var_45, adj_19, adj_21, adj_45, adj_46);
                    if (!var_19) {
                        wp::adj_where(var_23, var_25, var_44, adj_23, adj_25, adj_44, adj_45);
                        if (!var_23) {
                            wp::adj_where(var_27, var_29, var_43, adj_27, adj_29, adj_43, adj_44);
                            if (!var_27) {
                                wp::adj_where(var_31, var_33, var_42, adj_31, adj_33, adj_42, adj_43);
                                if (!var_31) {
                                    wp::adj_where(var_35, var_37, var_41, adj_35, adj_37, adj_41, adj_42);
                                    if (!var_35) {
                                        wp::adj_where(var_39, var_40, var_37, adj_39, adj_40, adj_37, adj_41);
                                        if (var_39) {
                                            // adj: slot = 3                                  <L 1696>
                                        }
                                        // adj: elif edge_idx == 11:                          <L 1695>
                                    }
                                    wp::adj_where(var_35, var_36, var_33, adj_35, adj_36, adj_33, adj_37);
                                    if (var_35) {
                                        // adj: slot = 2                                      <L 1694>
                                    }
                                    // adj: elif edge_idx == 10:                              <L 1693>
                                }
                                wp::adj_where(var_31, var_32, var_29, adj_31, adj_32, adj_29, adj_33);
                                if (var_31) {
                                    // adj: slot = 1                                          <L 1692>
                                }
                                // adj: elif edge_idx == 9:                                   <L 1691>
                            }
                            wp::adj_where(var_27, var_28, var_25, adj_27, adj_28, adj_25, adj_29);
                            if (var_27) {
                                // adj: slot = 7                                              <L 1690>
                            }
                            // adj: elif edge_idx == 7:                                       <L 1689>
                        }
                        wp::adj_where(var_23, var_24, var_21, adj_23, adj_24, adj_21, adj_25);
                        if (var_23) {
                            // adj: slot = 6                                                  <L 1688>
                        }
                        // adj: elif edge_idx == 6:                                           <L 1687>
                    }
                    wp::adj_where(var_19, var_20, var_17, adj_19, adj_20, adj_17, adj_21);
                    if (var_19) {
                        // adj: slot = 5                                                      <L 1686>
                    }
                    // adj: elif edge_idx == 5:                                               <L 1685>
                }
                wp::adj_where(var_15, var_16, var_13, adj_15, adj_16, adj_13, adj_17);
                if (var_15) {
                    // adj: slot = 4                                                          <L 1684>
                }
                // adj: elif edge_idx == 4:                                                   <L 1683>
            }
            wp::adj_where(var_11, var_12, var_9, adj_11, adj_12, adj_9, adj_13);
            if (var_11) {
                // adj: slot = 3                                                              <L 1682>
            }
            // adj: elif edge_idx == 3:                                                       <L 1681>
        }
        wp::adj_where(var_7, var_8, var_5, adj_7, adj_8, adj_5, adj_9);
        if (var_7) {
            // adj: slot = 2                                                                  <L 1680>
        }
        // adj: elif edge_idx == 2:                                                           <L 1679>
    }
    wp::adj_where(var_3, var_4, var_1, adj_3, adj_4, adj_1, adj_5);
    if (var_3) {
        // adj: slot = 1                                                                      <L 1678>
    }
    // adj: if edge_idx == 1:                                                                 <L 1677>
    wp::adj_int(var_0, adj_0, adj_1);
    // adj: slot = int(0)                                                                     <L 1676>
    // adj: def _cluster_edge_slot_a(edge_idx: int) -> int:                                   <L 1675>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/render.py:1700
static CUDA_CALLABLE void adj__cluster_edge_slot_b_0(
    wp::int32 var_edge_idx,
    wp::int32 & adj_edge_idx,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 1;
    wp::int32 var_1;
    const wp::int32 var_2 = 1;
    bool var_3;
    const wp::int32 var_4 = 2;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    bool var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    const wp::int32 var_10 = 3;
    bool var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    const wp::int32 var_14 = 4;
    bool var_15;
    const wp::int32 var_16 = 5;
    wp::int32 var_17;
    const wp::int32 var_18 = 5;
    bool var_19;
    const wp::int32 var_20 = 6;
    wp::int32 var_21;
    const wp::int32 var_22 = 6;
    bool var_23;
    const wp::int32 var_24 = 7;
    wp::int32 var_25;
    const wp::int32 var_26 = 7;
    bool var_27;
    const wp::int32 var_28 = 4;
    wp::int32 var_29;
    const wp::int32 var_30 = 8;
    bool var_31;
    const wp::int32 var_32 = 4;
    wp::int32 var_33;
    const wp::int32 var_34 = 9;
    bool var_35;
    const wp::int32 var_36 = 5;
    wp::int32 var_37;
    const wp::int32 var_38 = 10;
    bool var_39;
    const wp::int32 var_40 = 6;
    wp::int32 var_41;
    const wp::int32 var_42 = 11;
    bool var_43;
    const wp::int32 var_44 = 7;
    wp::int32 var_45;
    wp::int32 var_46;
    wp::int32 var_47;
    wp::int32 var_48;
    wp::int32 var_49;
    wp::int32 var_50;
    wp::int32 var_51;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    bool adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    bool adj_7 = {};
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    bool adj_11 = {};
    wp::int32 adj_12 = {};
    wp::int32 adj_13 = {};
    wp::int32 adj_14 = {};
    bool adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    bool adj_19 = {};
    wp::int32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    bool adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    bool adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::int32 adj_30 = {};
    bool adj_31 = {};
    wp::int32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::int32 adj_34 = {};
    bool adj_35 = {};
    wp::int32 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::int32 adj_38 = {};
    bool adj_39 = {};
    wp::int32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::int32 adj_42 = {};
    bool adj_43 = {};
    wp::int32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::int32 adj_48 = {};
    wp::int32 adj_49 = {};
    wp::int32 adj_50 = {};
    wp::int32 adj_51 = {};
    wp::int32 adj_52 = {};
    wp::int32 adj_53 = {};
    wp::int32 adj_54 = {};
    wp::int32 adj_55 = {};
    //---------
    // forward
    // def _cluster_edge_slot_b(edge_idx: int) -> int:                                        <L 1701>
    // slot = int(1)                                                                          <L 1702>
    var_1 = wp::int(var_0);
    // if edge_idx == 1:                                                                      <L 1703>
    var_3 = (var_edge_idx == var_2);
    if (var_3) {
        // slot = 2                                                                           <L 1704>
    }
    var_5 = wp::where(var_3, var_4, var_1);
    if (!var_3) {
        // elif edge_idx == 2:                                                                <L 1705>
        var_7 = (var_edge_idx == var_6);
        if (var_7) {
            // slot = 3                                                                       <L 1706>
        }
        var_9 = wp::where(var_7, var_8, var_5);
        if (!var_7) {
            // elif edge_idx == 3:                                                            <L 1707>
            var_11 = (var_edge_idx == var_10);
            if (var_11) {
                // slot = 0                                                                   <L 1708>
            }
            var_13 = wp::where(var_11, var_12, var_9);
            if (!var_11) {
                // elif edge_idx == 4:                                                        <L 1709>
                var_15 = (var_edge_idx == var_14);
                if (var_15) {
                    // slot = 5                                                               <L 1710>
                }
                var_17 = wp::where(var_15, var_16, var_13);
                if (!var_15) {
                    // elif edge_idx == 5:                                                    <L 1711>
                    var_19 = (var_edge_idx == var_18);
                    if (var_19) {
                        // slot = 6                                                           <L 1712>
                    }
                    var_21 = wp::where(var_19, var_20, var_17);
                    if (!var_19) {
                        // elif edge_idx == 6:                                                <L 1713>
                        var_23 = (var_edge_idx == var_22);
                        if (var_23) {
                            // slot = 7                                                       <L 1714>
                        }
                        var_25 = wp::where(var_23, var_24, var_21);
                        if (!var_23) {
                            // elif edge_idx == 7:                                            <L 1715>
                            var_27 = (var_edge_idx == var_26);
                            if (var_27) {
                                // slot = 4                                                   <L 1716>
                            }
                            var_29 = wp::where(var_27, var_28, var_25);
                            if (!var_27) {
                                // elif edge_idx == 8:                                        <L 1717>
                                var_31 = (var_edge_idx == var_30);
                                if (var_31) {
                                    // slot = 4                                               <L 1718>
                                }
                                var_33 = wp::where(var_31, var_32, var_29);
                                if (!var_31) {
                                    // elif edge_idx == 9:                                    <L 1719>
                                    var_35 = (var_edge_idx == var_34);
                                    if (var_35) {
                                        // slot = 5                                           <L 1720>
                                    }
                                    var_37 = wp::where(var_35, var_36, var_33);
                                    if (!var_35) {
                                        // elif edge_idx == 10:                               <L 1721>
                                        var_39 = (var_edge_idx == var_38);
                                        if (var_39) {
                                            // slot = 6                                       <L 1722>
                                        }
                                        var_41 = wp::where(var_39, var_40, var_37);
                                        if (!var_39) {
                                            // elif edge_idx == 11:                           <L 1723>
                                            var_43 = (var_edge_idx == var_42);
                                            if (var_43) {
                                                // slot = 7                                   <L 1724>
                                            }
                                            var_45 = wp::where(var_43, var_44, var_41);
                                        }
                                        var_46 = wp::where(var_39, var_41, var_45);
                                    }
                                    var_47 = wp::where(var_35, var_37, var_46);
                                }
                                var_48 = wp::where(var_31, var_33, var_47);
                            }
                            var_49 = wp::where(var_27, var_29, var_48);
                        }
                        var_50 = wp::where(var_23, var_25, var_49);
                    }
                    var_51 = wp::where(var_19, var_21, var_50);
                }
                var_52 = wp::where(var_15, var_17, var_51);
            }
            var_53 = wp::where(var_11, var_13, var_52);
        }
        var_54 = wp::where(var_7, var_9, var_53);
    }
    var_55 = wp::where(var_3, var_5, var_54);
    // return slot                                                                            <L 1725>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_55 += adj_ret;
    // adj: return slot                                                                       <L 1725>
    wp::adj_where(var_3, var_5, var_54, adj_3, adj_5, adj_54, adj_55);
    if (!var_3) {
        wp::adj_where(var_7, var_9, var_53, adj_7, adj_9, adj_53, adj_54);
        if (!var_7) {
            wp::adj_where(var_11, var_13, var_52, adj_11, adj_13, adj_52, adj_53);
            if (!var_11) {
                wp::adj_where(var_15, var_17, var_51, adj_15, adj_17, adj_51, adj_52);
                if (!var_15) {
                    wp::adj_where(var_19, var_21, var_50, adj_19, adj_21, adj_50, adj_51);
                    if (!var_19) {
                        wp::adj_where(var_23, var_25, var_49, adj_23, adj_25, adj_49, adj_50);
                        if (!var_23) {
                            wp::adj_where(var_27, var_29, var_48, adj_27, adj_29, adj_48, adj_49);
                            if (!var_27) {
                                wp::adj_where(var_31, var_33, var_47, adj_31, adj_33, adj_47, adj_48);
                                if (!var_31) {
                                    wp::adj_where(var_35, var_37, var_46, adj_35, adj_37, adj_46, adj_47);
                                    if (!var_35) {
                                        wp::adj_where(var_39, var_41, var_45, adj_39, adj_41, adj_45, adj_46);
                                        if (!var_39) {
                                            wp::adj_where(var_43, var_44, var_41, adj_43, adj_44, adj_41, adj_45);
                                            if (var_43) {
                                                // adj: slot = 7                              <L 1724>
                                            }
                                            // adj: elif edge_idx == 11:                      <L 1723>
                                        }
                                        wp::adj_where(var_39, var_40, var_37, adj_39, adj_40, adj_37, adj_41);
                                        if (var_39) {
                                            // adj: slot = 6                                  <L 1722>
                                        }
                                        // adj: elif edge_idx == 10:                          <L 1721>
                                    }
                                    wp::adj_where(var_35, var_36, var_33, adj_35, adj_36, adj_33, adj_37);
                                    if (var_35) {
                                        // adj: slot = 5                                      <L 1720>
                                    }
                                    // adj: elif edge_idx == 9:                               <L 1719>
                                }
                                wp::adj_where(var_31, var_32, var_29, adj_31, adj_32, adj_29, adj_33);
                                if (var_31) {
                                    // adj: slot = 4                                          <L 1718>
                                }
                                // adj: elif edge_idx == 8:                                   <L 1717>
                            }
                            wp::adj_where(var_27, var_28, var_25, adj_27, adj_28, adj_25, adj_29);
                            if (var_27) {
                                // adj: slot = 4                                              <L 1716>
                            }
                            // adj: elif edge_idx == 7:                                       <L 1715>
                        }
                        wp::adj_where(var_23, var_24, var_21, adj_23, adj_24, adj_21, adj_25);
                        if (var_23) {
                            // adj: slot = 7                                                  <L 1714>
                        }
                        // adj: elif edge_idx == 6:                                           <L 1713>
                    }
                    wp::adj_where(var_19, var_20, var_17, adj_19, adj_20, adj_17, adj_21);
                    if (var_19) {
                        // adj: slot = 6                                                      <L 1712>
                    }
                    // adj: elif edge_idx == 5:                                               <L 1711>
                }
                wp::adj_where(var_15, var_16, var_13, adj_15, adj_16, adj_13, adj_17);
                if (var_15) {
                    // adj: slot = 5                                                          <L 1710>
                }
                // adj: elif edge_idx == 4:                                                   <L 1709>
            }
            wp::adj_where(var_11, var_12, var_9, adj_11, adj_12, adj_9, adj_13);
            if (var_11) {
                // adj: slot = 0                                                              <L 1708>
            }
            // adj: elif edge_idx == 3:                                                       <L 1707>
        }
        wp::adj_where(var_7, var_8, var_5, adj_7, adj_8, adj_5, adj_9);
        if (var_7) {
            // adj: slot = 3                                                                  <L 1706>
        }
        // adj: elif edge_idx == 2:                                                           <L 1705>
    }
    wp::adj_where(var_3, var_4, var_1, adj_3, adj_4, adj_1, adj_5);
    if (var_3) {
        // adj: slot = 2                                                                      <L 1704>
    }
    // adj: if edge_idx == 1:                                                                 <L 1703>
    wp::adj_int(var_0, adj_0, adj_1);
    // adj: slot = int(1)                                                                     <L 1702>
    // adj: def _cluster_edge_slot_b(edge_idx: int) -> int:                                   <L 1701>
    return;
}



extern "C" __global__ void _expand_triangle_normals_kernel_fcdd309b_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_normals,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_normals)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        //---------
        // forward
        // def _expand_triangle_normals_kernel(                                                   <L 1069>
        // t = wp.tid()                                                                           <L 1076>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1077>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1078>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1079>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1080>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1081>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 1082>
        var_15 = wp::mul(var_0, var_14);
        // flat_normals[base + 0] = vertex_normals[v0]                                            <L 1083>
        var_16 = wp::address(var_vertex_normals, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        wp::array_store(var_flat_normals, var_18, var_19);
        // flat_normals[base + 1] = vertex_normals[v1]                                            <L 1084>
        var_20 = wp::address(var_vertex_normals, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        wp::array_store(var_flat_normals, var_22, var_23);
        // flat_normals[base + 2] = vertex_normals[v2]                                            <L 1085>
        var_24 = wp::address(var_vertex_normals, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        wp::array_store(var_flat_normals, var_26, var_27);
    }
}



extern "C" __global__ void _expand_triangle_normals_kernel_fcdd309b_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_normals,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_normals,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_normals,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_normals)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        //---------
        // forward
        // def _expand_triangle_normals_kernel(                                                   <L 1069>
        // t = wp.tid()                                                                           <L 1076>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1077>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1078>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1079>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1080>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1081>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 1082>
        var_15 = wp::mul(var_0, var_14);
        // flat_normals[base + 0] = vertex_normals[v0]                                            <L 1083>
        var_16 = wp::address(var_vertex_normals, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        // wp::array_store(var_flat_normals, var_18, var_19);
        // flat_normals[base + 1] = vertex_normals[v1]                                            <L 1084>
        var_20 = wp::address(var_vertex_normals, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        // wp::array_store(var_flat_normals, var_22, var_23);
        // flat_normals[base + 2] = vertex_normals[v2]                                            <L 1085>
        var_24 = wp::address(var_vertex_normals, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        // wp::array_store(var_flat_normals, var_26, var_27);
        //---------
        // reverse
        wp::adj_array_store(var_flat_normals, var_26, var_27, adj_flat_normals, adj_26, adj_24);
        wp::adj_add(var_15, var_25, adj_15, adj_25, adj_26);
        wp::adj_address(var_vertex_normals, var_12, adj_vertex_normals, adj_12, adj_24);
        // adj: flat_normals[base + 2] = vertex_normals[v2]                                       <L 1085>
        wp::adj_array_store(var_flat_normals, var_22, var_23, adj_flat_normals, adj_22, adj_20);
        wp::adj_add(var_15, var_21, adj_15, adj_21, adj_22);
        wp::adj_address(var_vertex_normals, var_8, adj_vertex_normals, adj_8, adj_20);
        // adj: flat_normals[base + 1] = vertex_normals[v1]                                       <L 1084>
        wp::adj_array_store(var_flat_normals, var_18, var_19, adj_flat_normals, adj_18, adj_16);
        wp::adj_add(var_15, var_17, adj_15, adj_17, adj_18);
        wp::adj_address(var_vertex_normals, var_4, adj_vertex_normals, adj_4, adj_16);
        // adj: flat_normals[base + 0] = vertex_normals[v0]                                       <L 1083>
        wp::adj_mul(var_0, var_14, adj_0, adj_14, adj_15);
        // adj: base = t * 3                                                                      <L 1082>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 1081>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 1080>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 1079>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1078>
        }
        // adj: if t >= num_triangles:                                                            <L 1077>
        // adj: t = wp.tid()                                                                      <L 1076>
        // adj: def _expand_triangle_normals_kernel(                                              <L 1069>
        continue;
    }
}



extern "C" __global__ void _compute_flat_triangle_normals_kernel_d5c8e95d_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_normals)
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
        const wp::int32 var_2 = 3;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::int32 var_14 = 2;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 1e-16;
        bool var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 1.0;
        const wp::float32 var_29 = 0.0;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 0;
        wp::int32 var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        const wp::int32 var_36 = 2;
        wp::int32 var_37;
        //---------
        // forward
        // def _compute_flat_triangle_normals_kernel(                                             <L 1089>
        // t = wp.tid()                                                                           <L 1095>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1096>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1097>
            continue;
        }
        // base = t * 3                                                                           <L 1098>
        var_3 = wp::mul(var_0, var_2);
        // p0 = flat_pos[base + 0]                                                                <L 1099>
        var_5 = wp::add(var_3, var_4);
        var_6 = wp::address(var_flat_pos, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // p1 = flat_pos[base + 1]                                                                <L 1100>
        var_10 = wp::add(var_3, var_9);
        var_11 = wp::address(var_flat_pos, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // p2 = flat_pos[base + 2]                                                                <L 1101>
        var_15 = wp::add(var_3, var_14);
        var_16 = wp::address(var_flat_pos, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // n = wp.cross(p1 - p0, p2 - p0)                                                         <L 1102>
        var_19 = wp::sub(var_12, var_7);
        var_20 = wp::sub(var_17, var_7);
        var_21 = wp::cross(var_19, var_20);
        // if wp.dot(n, n) > 1.0e-16:                                                             <L 1103>
        var_22 = wp::dot(var_21, var_21);
        var_24 = (var_22 > var_23);
        if (var_24) {
            // n = wp.normalize(n)                                                                <L 1104>
            var_25 = wp::normalize(var_21);
        }
        var_26 = wp::where(var_24, var_25, var_21);
        if (!var_24) {
            // n = wp.vec3(0.0, 1.0, 0.0)                                                         <L 1106>
            var_30 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        }
        var_31 = wp::where(var_24, var_26, var_30);
        // flat_normals[base + 0] = n                                                             <L 1107>
        var_33 = wp::add(var_3, var_32);
        wp::array_store(var_flat_normals, var_33, var_31);
        // flat_normals[base + 1] = n                                                             <L 1108>
        var_35 = wp::add(var_3, var_34);
        wp::array_store(var_flat_normals, var_35, var_31);
        // flat_normals[base + 2] = n                                                             <L 1109>
        var_37 = wp::add(var_3, var_36);
        wp::array_store(var_flat_normals, var_37, var_31);
    }
}



extern "C" __global__ void _compute_flat_triangle_normals_kernel_d5c8e95d_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_normals,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_pos,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_normals)
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
        const wp::int32 var_2 = 3;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::vec_t<3, wp::float32> var_13;
        const wp::int32 var_14 = 2;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::float32 var_22;
        const wp::float32 var_23 = 1e-16;
        bool var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        const wp::float32 var_27 = 0.0;
        const wp::float32 var_28 = 1.0;
        const wp::float32 var_29 = 0.0;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        const wp::int32 var_32 = 0;
        wp::int32 var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        const wp::int32 var_36 = 2;
        wp::int32 var_37;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        bool adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::float32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        //---------
        // forward
        // def _compute_flat_triangle_normals_kernel(                                             <L 1089>
        // t = wp.tid()                                                                           <L 1095>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1096>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1097>
            goto label0;
        }
        // base = t * 3                                                                           <L 1098>
        var_3 = wp::mul(var_0, var_2);
        // p0 = flat_pos[base + 0]                                                                <L 1099>
        var_5 = wp::add(var_3, var_4);
        var_6 = wp::address(var_flat_pos, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // p1 = flat_pos[base + 1]                                                                <L 1100>
        var_10 = wp::add(var_3, var_9);
        var_11 = wp::address(var_flat_pos, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // p2 = flat_pos[base + 2]                                                                <L 1101>
        var_15 = wp::add(var_3, var_14);
        var_16 = wp::address(var_flat_pos, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // n = wp.cross(p1 - p0, p2 - p0)                                                         <L 1102>
        var_19 = wp::sub(var_12, var_7);
        var_20 = wp::sub(var_17, var_7);
        var_21 = wp::cross(var_19, var_20);
        // if wp.dot(n, n) > 1.0e-16:                                                             <L 1103>
        var_22 = wp::dot(var_21, var_21);
        var_24 = (var_22 > var_23);
        if (var_24) {
            // n = wp.normalize(n)                                                                <L 1104>
            var_25 = wp::normalize(var_21);
        }
        var_26 = wp::where(var_24, var_25, var_21);
        if (!var_24) {
            // n = wp.vec3(0.0, 1.0, 0.0)                                                         <L 1106>
            var_30 = wp::vec_t<3, wp::float32>(var_27, var_28, var_29);
        }
        var_31 = wp::where(var_24, var_26, var_30);
        // flat_normals[base + 0] = n                                                             <L 1107>
        var_33 = wp::add(var_3, var_32);
        // wp::array_store(var_flat_normals, var_33, var_31);
        // flat_normals[base + 1] = n                                                             <L 1108>
        var_35 = wp::add(var_3, var_34);
        // wp::array_store(var_flat_normals, var_35, var_31);
        // flat_normals[base + 2] = n                                                             <L 1109>
        var_37 = wp::add(var_3, var_36);
        // wp::array_store(var_flat_normals, var_37, var_31);
        //---------
        // reverse
        wp::adj_array_store(var_flat_normals, var_37, var_31, adj_flat_normals, adj_37, adj_31);
        wp::adj_add(var_3, var_36, adj_3, adj_36, adj_37);
        // adj: flat_normals[base + 2] = n                                                        <L 1109>
        wp::adj_array_store(var_flat_normals, var_35, var_31, adj_flat_normals, adj_35, adj_31);
        wp::adj_add(var_3, var_34, adj_3, adj_34, adj_35);
        // adj: flat_normals[base + 1] = n                                                        <L 1108>
        wp::adj_array_store(var_flat_normals, var_33, var_31, adj_flat_normals, adj_33, adj_31);
        wp::adj_add(var_3, var_32, adj_3, adj_32, adj_33);
        // adj: flat_normals[base + 0] = n                                                        <L 1107>
        wp::adj_where(var_24, var_26, var_30, adj_24, adj_26, adj_30, adj_31);
        if (!var_24) {
            wp::adj_vec_t(var_27, var_28, var_29, adj_27, adj_28, adj_29, adj_30);
            // adj: n = wp.vec3(0.0, 1.0, 0.0)                                                    <L 1106>
        }
        wp::adj_where(var_24, var_25, var_21, adj_24, adj_25, adj_21, adj_26);
        if (var_24) {
            wp::adj_normalize(var_21, var_25, adj_21, adj_25);
            // adj: n = wp.normalize(n)                                                           <L 1104>
        }
        wp::adj_dot(var_21, var_21, adj_21, adj_21, adj_22);
        // adj: if wp.dot(n, n) > 1.0e-16:                                                        <L 1103>
        wp::adj_cross(var_19, var_20, adj_19, adj_20, adj_21);
        wp::adj_sub(var_17, var_7, adj_17, adj_7, adj_20);
        wp::adj_sub(var_12, var_7, adj_12, adj_7, adj_19);
        // adj: n = wp.cross(p1 - p0, p2 - p0)                                                    <L 1102>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_address(var_flat_pos, var_15, adj_flat_pos, adj_15, adj_16);
        wp::adj_add(var_3, var_14, adj_3, adj_14, adj_15);
        // adj: p2 = flat_pos[base + 2]                                                           <L 1101>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_flat_pos, var_10, adj_flat_pos, adj_10, adj_11);
        wp::adj_add(var_3, var_9, adj_3, adj_9, adj_10);
        // adj: p1 = flat_pos[base + 1]                                                           <L 1100>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_flat_pos, var_5, adj_flat_pos, adj_5, adj_6);
        wp::adj_add(var_3, var_4, adj_3, adj_4, adj_5);
        // adj: p0 = flat_pos[base + 0]                                                           <L 1099>
        wp::adj_mul(var_0, var_2, adj_0, adj_2, adj_3);
        // adj: base = t * 3                                                                      <L 1098>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1097>
        }
        // adj: if t >= num_triangles:                                                            <L 1096>
        // adj: t = wp.tid()                                                                      <L 1095>
        // adj: def _compute_flat_triangle_normals_kernel(                                        <L 1089>
        continue;
    }
}



extern "C" __global__ void _compute_triangle_tile_uvs_kernel_1ea906ca_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<2, wp::float32>> var_flat_uvs)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::float32 var_6;
        wp::int32 var_7;
        wp::float32 var_8;
        const wp::int32 var_9 = 1;
        bool var_10;
        const wp::float32 var_11 = 0.5;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.5;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 1.0;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<2, wp::float32> var_21;
        const wp::int32 var_22 = 3;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        const wp::int32 var_26 = 3;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        const wp::int32 var_30 = 3;
        wp::int32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32 var_33;
        const wp::float32 var_34 = 0.5;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.5;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::float32 var_40 = 0.5;
        wp::float32 var_41;
        const wp::float32 var_42 = 0.5;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.5;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.5;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 1.0;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::vec_t<2, wp::float32> var_56;
        const wp::int32 var_57 = 3;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::int32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::float32 var_63 = 1.0;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::vec_t<2, wp::float32> var_67;
        const wp::int32 var_68 = 3;
        wp::int32 var_69;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::float32 var_74 = 1.0;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::vec_t<2, wp::float32> var_78;
        const wp::int32 var_79 = 3;
        wp::int32 var_80;
        const wp::int32 var_81 = 2;
        wp::int32 var_82;
        //---------
        // forward
        // def _compute_triangle_tile_uvs_kernel(                                                 <L 1325>
        // t = wp.tid()                                                                           <L 1333>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1334>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1335>
            continue;
        }
        // tiles_per_row = atlas_width / tile_size                                                <L 1336>
        var_2 = wp::div(var_atlas_width, var_tile_size);
        // tile_x = t % tiles_per_row                                                             <L 1337>
        var_3 = wp::mod(var_0, var_2);
        // tile_y = t / tiles_per_row                                                             <L 1338>
        var_4 = wp::div(var_0, var_2);
        // x0 = float(tile_x * tile_size)                                                         <L 1339>
        var_5 = wp::mul(var_3, var_tile_size);
        var_6 = wp::float(var_5);
        // y0 = float(tile_y * tile_size)                                                         <L 1340>
        var_7 = wp::mul(var_4, var_tile_size);
        var_8 = wp::float(var_7);
        // if tile_size <= 1:                                                                     <L 1341>
        var_10 = (var_tile_size <= var_9);
        if (var_10) {
            // cx = x0 + 0.5                                                                      <L 1342>
            var_12 = wp::add(var_6, var_11);
            // cy = y0 + 0.5                                                                      <L 1343>
            var_14 = wp::add(var_8, var_13);
            // uv = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))            <L 1344>
            var_15 = wp::float(var_atlas_width);
            var_16 = wp::div(var_12, var_15);
            var_18 = wp::float(var_atlas_height);
            var_19 = wp::div(var_14, var_18);
            var_20 = wp::sub(var_17, var_19);
            var_21 = wp::vec_t<2, wp::float32>(var_16, var_20);
            // flat_uvs[t * 3 + 0] = uv                                                           <L 1345>
            var_23 = wp::mul(var_0, var_22);
            var_25 = wp::add(var_23, var_24);
            wp::array_store(var_flat_uvs, var_25, var_21);
            // flat_uvs[t * 3 + 1] = uv                                                           <L 1346>
            var_27 = wp::mul(var_0, var_26);
            var_29 = wp::add(var_27, var_28);
            wp::array_store(var_flat_uvs, var_29, var_21);
            // flat_uvs[t * 3 + 2] = uv                                                           <L 1347>
            var_31 = wp::mul(var_0, var_30);
            var_33 = wp::add(var_31, var_32);
            wp::array_store(var_flat_uvs, var_33, var_21);
            // return                                                                             <L 1348>
            continue;
        }
        // ax = x0 + 0.5                                                                          <L 1349>
        var_35 = wp::add(var_6, var_34);
        // ay = y0 + 0.5                                                                          <L 1350>
        var_37 = wp::add(var_8, var_36);
        // bx = x0 + float(tile_size) - 0.5                                                       <L 1351>
        var_38 = wp::float(var_tile_size);
        var_39 = wp::add(var_6, var_38);
        var_41 = wp::sub(var_39, var_40);
        // by = y0 + 0.5                                                                          <L 1352>
        var_43 = wp::add(var_8, var_42);
        // cx = x0 + 0.5                                                                          <L 1353>
        var_45 = wp::add(var_6, var_44);
        // cy = y0 + float(tile_size) - 0.5                                                       <L 1354>
        var_46 = wp::float(var_tile_size);
        var_47 = wp::add(var_8, var_46);
        var_49 = wp::sub(var_47, var_48);
        // flat_uvs[t * 3 + 0] = wp.vec2(ax / float(atlas_width), 1.0 - (ay / float(atlas_height)))       <L 1358>
        var_50 = wp::float(var_atlas_width);
        var_51 = wp::div(var_35, var_50);
        var_53 = wp::float(var_atlas_height);
        var_54 = wp::div(var_37, var_53);
        var_55 = wp::sub(var_52, var_54);
        var_56 = wp::vec_t<2, wp::float32>(var_51, var_55);
        var_58 = wp::mul(var_0, var_57);
        var_60 = wp::add(var_58, var_59);
        wp::array_store(var_flat_uvs, var_60, var_56);
        // flat_uvs[t * 3 + 1] = wp.vec2(bx / float(atlas_width), 1.0 - (by / float(atlas_height)))       <L 1359>
        var_61 = wp::float(var_atlas_width);
        var_62 = wp::div(var_41, var_61);
        var_64 = wp::float(var_atlas_height);
        var_65 = wp::div(var_43, var_64);
        var_66 = wp::sub(var_63, var_65);
        var_67 = wp::vec_t<2, wp::float32>(var_62, var_66);
        var_69 = wp::mul(var_0, var_68);
        var_71 = wp::add(var_69, var_70);
        wp::array_store(var_flat_uvs, var_71, var_67);
        // flat_uvs[t * 3 + 2] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))       <L 1360>
        var_72 = wp::float(var_atlas_width);
        var_73 = wp::div(var_45, var_72);
        var_75 = wp::float(var_atlas_height);
        var_76 = wp::div(var_49, var_75);
        var_77 = wp::sub(var_74, var_76);
        var_78 = wp::vec_t<2, wp::float32>(var_73, var_77);
        var_80 = wp::mul(var_0, var_79);
        var_82 = wp::add(var_80, var_81);
        wp::array_store(var_flat_uvs, var_82, var_78);
    }
}



extern "C" __global__ void _compute_triangle_tile_uvs_kernel_1ea906ca_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<2, wp::float32>> var_flat_uvs,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_flat_uvs)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::float32 var_6;
        wp::int32 var_7;
        wp::float32 var_8;
        const wp::int32 var_9 = 1;
        bool var_10;
        const wp::float32 var_11 = 0.5;
        wp::float32 var_12;
        const wp::float32 var_13 = 0.5;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 1.0;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<2, wp::float32> var_21;
        const wp::int32 var_22 = 3;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        wp::int32 var_25;
        const wp::int32 var_26 = 3;
        wp::int32 var_27;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        const wp::int32 var_30 = 3;
        wp::int32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32 var_33;
        const wp::float32 var_34 = 0.5;
        wp::float32 var_35;
        const wp::float32 var_36 = 0.5;
        wp::float32 var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::float32 var_40 = 0.5;
        wp::float32 var_41;
        const wp::float32 var_42 = 0.5;
        wp::float32 var_43;
        const wp::float32 var_44 = 0.5;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.5;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 1.0;
        wp::float32 var_53;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::vec_t<2, wp::float32> var_56;
        const wp::int32 var_57 = 3;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::int32 var_60;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::float32 var_63 = 1.0;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::vec_t<2, wp::float32> var_67;
        const wp::int32 var_68 = 3;
        wp::int32 var_69;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::float32 var_74 = 1.0;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::vec_t<2, wp::float32> var_78;
        const wp::int32 var_79 = 3;
        wp::int32 var_80;
        const wp::int32 var_81 = 2;
        wp::int32 var_82;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::int32 adj_9 = {};
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
        wp::vec_t<2, wp::float32> adj_21 = {};
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
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
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
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::vec_t<2, wp::float32> adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::vec_t<2, wp::float32> adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::vec_t<2, wp::float32> adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        //---------
        // forward
        // def _compute_triangle_tile_uvs_kernel(                                                 <L 1325>
        // t = wp.tid()                                                                           <L 1333>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1334>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1335>
            goto label0;
        }
        // tiles_per_row = atlas_width / tile_size                                                <L 1336>
        var_2 = wp::div(var_atlas_width, var_tile_size);
        // tile_x = t % tiles_per_row                                                             <L 1337>
        var_3 = wp::mod(var_0, var_2);
        // tile_y = t / tiles_per_row                                                             <L 1338>
        var_4 = wp::div(var_0, var_2);
        // x0 = float(tile_x * tile_size)                                                         <L 1339>
        var_5 = wp::mul(var_3, var_tile_size);
        var_6 = wp::float(var_5);
        // y0 = float(tile_y * tile_size)                                                         <L 1340>
        var_7 = wp::mul(var_4, var_tile_size);
        var_8 = wp::float(var_7);
        // if tile_size <= 1:                                                                     <L 1341>
        var_10 = (var_tile_size <= var_9);
        if (var_10) {
            // cx = x0 + 0.5                                                                      <L 1342>
            var_12 = wp::add(var_6, var_11);
            // cy = y0 + 0.5                                                                      <L 1343>
            var_14 = wp::add(var_8, var_13);
            // uv = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))            <L 1344>
            var_15 = wp::float(var_atlas_width);
            var_16 = wp::div(var_12, var_15);
            var_18 = wp::float(var_atlas_height);
            var_19 = wp::div(var_14, var_18);
            var_20 = wp::sub(var_17, var_19);
            var_21 = wp::vec_t<2, wp::float32>(var_16, var_20);
            // flat_uvs[t * 3 + 0] = uv                                                           <L 1345>
            var_23 = wp::mul(var_0, var_22);
            var_25 = wp::add(var_23, var_24);
            // wp::array_store(var_flat_uvs, var_25, var_21);
            // flat_uvs[t * 3 + 1] = uv                                                           <L 1346>
            var_27 = wp::mul(var_0, var_26);
            var_29 = wp::add(var_27, var_28);
            // wp::array_store(var_flat_uvs, var_29, var_21);
            // flat_uvs[t * 3 + 2] = uv                                                           <L 1347>
            var_31 = wp::mul(var_0, var_30);
            var_33 = wp::add(var_31, var_32);
            // wp::array_store(var_flat_uvs, var_33, var_21);
            // return                                                                             <L 1348>
            goto label1;
        }
        // ax = x0 + 0.5                                                                          <L 1349>
        var_35 = wp::add(var_6, var_34);
        // ay = y0 + 0.5                                                                          <L 1350>
        var_37 = wp::add(var_8, var_36);
        // bx = x0 + float(tile_size) - 0.5                                                       <L 1351>
        var_38 = wp::float(var_tile_size);
        var_39 = wp::add(var_6, var_38);
        var_41 = wp::sub(var_39, var_40);
        // by = y0 + 0.5                                                                          <L 1352>
        var_43 = wp::add(var_8, var_42);
        // cx = x0 + 0.5                                                                          <L 1353>
        var_45 = wp::add(var_6, var_44);
        // cy = y0 + float(tile_size) - 0.5                                                       <L 1354>
        var_46 = wp::float(var_tile_size);
        var_47 = wp::add(var_8, var_46);
        var_49 = wp::sub(var_47, var_48);
        // flat_uvs[t * 3 + 0] = wp.vec2(ax / float(atlas_width), 1.0 - (ay / float(atlas_height)))       <L 1358>
        var_50 = wp::float(var_atlas_width);
        var_51 = wp::div(var_35, var_50);
        var_53 = wp::float(var_atlas_height);
        var_54 = wp::div(var_37, var_53);
        var_55 = wp::sub(var_52, var_54);
        var_56 = wp::vec_t<2, wp::float32>(var_51, var_55);
        var_58 = wp::mul(var_0, var_57);
        var_60 = wp::add(var_58, var_59);
        // wp::array_store(var_flat_uvs, var_60, var_56);
        // flat_uvs[t * 3 + 1] = wp.vec2(bx / float(atlas_width), 1.0 - (by / float(atlas_height)))       <L 1359>
        var_61 = wp::float(var_atlas_width);
        var_62 = wp::div(var_41, var_61);
        var_64 = wp::float(var_atlas_height);
        var_65 = wp::div(var_43, var_64);
        var_66 = wp::sub(var_63, var_65);
        var_67 = wp::vec_t<2, wp::float32>(var_62, var_66);
        var_69 = wp::mul(var_0, var_68);
        var_71 = wp::add(var_69, var_70);
        // wp::array_store(var_flat_uvs, var_71, var_67);
        // flat_uvs[t * 3 + 2] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))       <L 1360>
        var_72 = wp::float(var_atlas_width);
        var_73 = wp::div(var_45, var_72);
        var_75 = wp::float(var_atlas_height);
        var_76 = wp::div(var_49, var_75);
        var_77 = wp::sub(var_74, var_76);
        var_78 = wp::vec_t<2, wp::float32>(var_73, var_77);
        var_80 = wp::mul(var_0, var_79);
        var_82 = wp::add(var_80, var_81);
        // wp::array_store(var_flat_uvs, var_82, var_78);
        //---------
        // reverse
        wp::adj_array_store(var_flat_uvs, var_82, var_78, adj_flat_uvs, adj_82, adj_78);
        wp::adj_add(var_80, var_81, adj_80, adj_81, adj_82);
        wp::adj_mul(var_0, var_79, adj_0, adj_79, adj_80);
        wp::adj_vec_t(var_73, var_77, adj_73, adj_77, adj_78);
        wp::adj_sub(var_74, var_76, adj_74, adj_76, adj_77);
        wp::adj_div(var_49, var_75, var_76, adj_49, adj_75, adj_76);
        wp::adj_float(var_atlas_height, adj_atlas_height, adj_75);
        wp::adj_div(var_45, var_72, var_73, adj_45, adj_72, adj_73);
        wp::adj_float(var_atlas_width, adj_atlas_width, adj_72);
        // adj: flat_uvs[t * 3 + 2] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))  <L 1360>
        wp::adj_array_store(var_flat_uvs, var_71, var_67, adj_flat_uvs, adj_71, adj_67);
        wp::adj_add(var_69, var_70, adj_69, adj_70, adj_71);
        wp::adj_mul(var_0, var_68, adj_0, adj_68, adj_69);
        wp::adj_vec_t(var_62, var_66, adj_62, adj_66, adj_67);
        wp::adj_sub(var_63, var_65, adj_63, adj_65, adj_66);
        wp::adj_div(var_43, var_64, var_65, adj_43, adj_64, adj_65);
        wp::adj_float(var_atlas_height, adj_atlas_height, adj_64);
        wp::adj_div(var_41, var_61, var_62, adj_41, adj_61, adj_62);
        wp::adj_float(var_atlas_width, adj_atlas_width, adj_61);
        // adj: flat_uvs[t * 3 + 1] = wp.vec2(bx / float(atlas_width), 1.0 - (by / float(atlas_height)))  <L 1359>
        wp::adj_array_store(var_flat_uvs, var_60, var_56, adj_flat_uvs, adj_60, adj_56);
        wp::adj_add(var_58, var_59, adj_58, adj_59, adj_60);
        wp::adj_mul(var_0, var_57, adj_0, adj_57, adj_58);
        wp::adj_vec_t(var_51, var_55, adj_51, adj_55, adj_56);
        wp::adj_sub(var_52, var_54, adj_52, adj_54, adj_55);
        wp::adj_div(var_37, var_53, var_54, adj_37, adj_53, adj_54);
        wp::adj_float(var_atlas_height, adj_atlas_height, adj_53);
        wp::adj_div(var_35, var_50, var_51, adj_35, adj_50, adj_51);
        wp::adj_float(var_atlas_width, adj_atlas_width, adj_50);
        // adj: flat_uvs[t * 3 + 0] = wp.vec2(ax / float(atlas_width), 1.0 - (ay / float(atlas_height)))  <L 1358>
        wp::adj_sub(var_47, var_48, adj_47, adj_48, adj_49);
        wp::adj_add(var_8, var_46, adj_8, adj_46, adj_47);
        wp::adj_float(var_tile_size, adj_tile_size, adj_46);
        // adj: cy = y0 + float(tile_size) - 0.5                                                  <L 1354>
        wp::adj_add(var_6, var_44, adj_6, adj_44, adj_45);
        // adj: cx = x0 + 0.5                                                                     <L 1353>
        wp::adj_add(var_8, var_42, adj_8, adj_42, adj_43);
        // adj: by = y0 + 0.5                                                                     <L 1352>
        wp::adj_sub(var_39, var_40, adj_39, adj_40, adj_41);
        wp::adj_add(var_6, var_38, adj_6, adj_38, adj_39);
        wp::adj_float(var_tile_size, adj_tile_size, adj_38);
        // adj: bx = x0 + float(tile_size) - 0.5                                                  <L 1351>
        wp::adj_add(var_8, var_36, adj_8, adj_36, adj_37);
        // adj: ay = y0 + 0.5                                                                     <L 1350>
        wp::adj_add(var_6, var_34, adj_6, adj_34, adj_35);
        // adj: ax = x0 + 0.5                                                                     <L 1349>
        if (var_10) {
            label1:;
            // adj: return                                                                        <L 1348>
            wp::adj_array_store(var_flat_uvs, var_33, var_21, adj_flat_uvs, adj_33, adj_21);
            wp::adj_add(var_31, var_32, adj_31, adj_32, adj_33);
            wp::adj_mul(var_0, var_30, adj_0, adj_30, adj_31);
            // adj: flat_uvs[t * 3 + 2] = uv                                                      <L 1347>
            wp::adj_array_store(var_flat_uvs, var_29, var_21, adj_flat_uvs, adj_29, adj_21);
            wp::adj_add(var_27, var_28, adj_27, adj_28, adj_29);
            wp::adj_mul(var_0, var_26, adj_0, adj_26, adj_27);
            // adj: flat_uvs[t * 3 + 1] = uv                                                      <L 1346>
            wp::adj_array_store(var_flat_uvs, var_25, var_21, adj_flat_uvs, adj_25, adj_21);
            wp::adj_add(var_23, var_24, adj_23, adj_24, adj_25);
            wp::adj_mul(var_0, var_22, adj_0, adj_22, adj_23);
            // adj: flat_uvs[t * 3 + 0] = uv                                                      <L 1345>
            wp::adj_vec_t(var_16, var_20, adj_16, adj_20, adj_21);
            wp::adj_sub(var_17, var_19, adj_17, adj_19, adj_20);
            wp::adj_div(var_14, var_18, var_19, adj_14, adj_18, adj_19);
            wp::adj_float(var_atlas_height, adj_atlas_height, adj_18);
            wp::adj_div(var_12, var_15, var_16, adj_12, adj_15, adj_16);
            wp::adj_float(var_atlas_width, adj_atlas_width, adj_15);
            // adj: uv = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))       <L 1344>
            wp::adj_add(var_8, var_13, adj_8, adj_13, adj_14);
            // adj: cy = y0 + 0.5                                                                 <L 1343>
            wp::adj_add(var_6, var_11, adj_6, adj_11, adj_12);
            // adj: cx = x0 + 0.5                                                                 <L 1342>
        }
        // adj: if tile_size <= 1:                                                                <L 1341>
        wp::adj_float(var_7, adj_7, adj_8);
        wp::adj_mul(var_4, var_tile_size, adj_4, adj_tile_size, adj_7);
        // adj: y0 = float(tile_y * tile_size)                                                    <L 1340>
        wp::adj_float(var_5, adj_5, adj_6);
        wp::adj_mul(var_3, var_tile_size, adj_3, adj_tile_size, adj_5);
        // adj: x0 = float(tile_x * tile_size)                                                    <L 1339>
        wp::adj_div(var_0, var_2, var_4, adj_0, adj_2, adj_4);
        // adj: tile_y = t / tiles_per_row                                                        <L 1338>
        wp::adj_mod(var_0, var_2, adj_0, adj_2, adj_3);
        // adj: tile_x = t % tiles_per_row                                                        <L 1337>
        wp::adj_div(var_atlas_width, var_tile_size, var_2, adj_atlas_width, adj_tile_size, adj_2);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1336>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1335>
        }
        // adj: if t >= num_triangles:                                                            <L 1334>
        // adj: t = wp.tid()                                                                      <L 1333>
        // adj: def _compute_triangle_tile_uvs_kernel(                                            <L 1325>
        continue;
    }
}



extern "C" __global__ void _sample_triangle_texture_kernel_81affedf_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_centroid_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_rgb)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 0;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        //---------
        // forward
        // def _sample_triangle_texture_kernel(                                                   <L 1198>
        // t = wp.tid()                                                                           <L 1217>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1218>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1219>
            continue;
        }
        // uv = tri_centroid_uv3[t]                                                               <L 1220>
        var_2 = wp::address(var_tri_centroid_uv3, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 1221>
        var_5 = _select_axis_0(var_3, var_src_x);
        var_6 = _scale_about_centre_0(var_5, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 1222>
        var_7 = _select_axis_0(var_3, var_src_y);
        var_8 = _scale_about_centre_0(var_7, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 1223>
        var_9 = _select_axis_0(var_3, var_src_z);
        var_10 = _scale_about_centre_0(var_9, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 1224>
        var_11 = wp::float(var_tex_nx);
        var_12 = wp::mul(var_6, var_11);
        var_13 = wp::int(var_12);
        var_16 = wp::sub(var_tex_nx, var_15);
        var_17 = wp::clamp(var_13, var_14, var_16);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 1225>
        var_18 = wp::float(var_tex_ny);
        var_19 = wp::mul(var_8, var_18);
        var_20 = wp::int(var_19);
        var_23 = wp::sub(var_tex_ny, var_22);
        var_24 = wp::clamp(var_20, var_21, var_23);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 1226>
        var_25 = wp::float(var_tex_nz);
        var_26 = wp::mul(var_10, var_25);
        var_27 = wp::int(var_26);
        var_30 = wp::sub(var_tex_nz, var_29);
        var_31 = wp::clamp(var_27, var_28, var_30);
        // if flip_x != 0:                                                                        <L 1227>
        var_33 = (var_flip_x != var_32);
        if (var_33) {
            // ix = (tex_nx - 1) - ix                                                             <L 1228>
            var_35 = wp::sub(var_tex_nx, var_34);
            var_36 = wp::sub(var_35, var_17);
        }
        var_37 = wp::where(var_33, var_36, var_17);
        // if flip_y != 0:                                                                        <L 1229>
        var_39 = (var_flip_y != var_38);
        if (var_39) {
            // iy = (tex_ny - 1) - iy                                                             <L 1230>
            var_41 = wp::sub(var_tex_ny, var_40);
            var_42 = wp::sub(var_41, var_24);
        }
        var_43 = wp::where(var_39, var_42, var_24);
        // if flip_z != 0:                                                                        <L 1231>
        var_45 = (var_flip_z != var_44);
        if (var_45) {
            // iz = (tex_nz - 1) - iz                                                             <L 1232>
            var_47 = wp::sub(var_tex_nz, var_46);
            var_48 = wp::sub(var_47, var_31);
        }
        var_49 = wp::where(var_45, var_48, var_31);
        // tri_rgb[t] = texture[ix, iy, iz]                                                       <L 1233>
        var_50 = wp::address(var_texture, var_37, var_43, var_49);
        var_51 = wp::load(var_50);
        wp::array_store(var_tri_rgb, var_0, var_51);
    }
}



extern "C" __global__ void _sample_triangle_texture_kernel_81affedf_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_centroid_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_tri_centroid_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_texture,
    wp::int32 adj_tex_nx,
    wp::int32 adj_tex_ny,
    wp::int32 adj_tex_nz,
    wp::int32 adj_src_x,
    wp::int32 adj_src_y,
    wp::int32 adj_src_z,
    wp::int32 adj_flip_x,
    wp::int32 adj_flip_y,
    wp::int32 adj_flip_z,
    wp::float32 adj_scale_x,
    wp::float32 adj_scale_y,
    wp::float32 adj_scale_z,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_tri_rgb)
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
        wp::vec_t<3, wp::float32>* var_2;
        wp::vec_t<3, wp::float32> var_3;
        wp::vec_t<3, wp::float32> var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::float32 var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        const wp::int32 var_28 = 0;
        const wp::int32 var_29 = 1;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        const wp::int32 var_34 = 1;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
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
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        bool adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        bool adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        bool adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::vec_t<3, wp::float32> adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        //---------
        // forward
        // def _sample_triangle_texture_kernel(                                                   <L 1198>
        // t = wp.tid()                                                                           <L 1217>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1218>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1219>
            goto label0;
        }
        // uv = tri_centroid_uv3[t]                                                               <L 1220>
        var_2 = wp::address(var_tri_centroid_uv3, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 1221>
        var_5 = _select_axis_0(var_3, var_src_x);
        var_6 = _scale_about_centre_0(var_5, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 1222>
        var_7 = _select_axis_0(var_3, var_src_y);
        var_8 = _scale_about_centre_0(var_7, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 1223>
        var_9 = _select_axis_0(var_3, var_src_z);
        var_10 = _scale_about_centre_0(var_9, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 1224>
        var_11 = wp::float(var_tex_nx);
        var_12 = wp::mul(var_6, var_11);
        var_13 = wp::int(var_12);
        var_16 = wp::sub(var_tex_nx, var_15);
        var_17 = wp::clamp(var_13, var_14, var_16);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 1225>
        var_18 = wp::float(var_tex_ny);
        var_19 = wp::mul(var_8, var_18);
        var_20 = wp::int(var_19);
        var_23 = wp::sub(var_tex_ny, var_22);
        var_24 = wp::clamp(var_20, var_21, var_23);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 1226>
        var_25 = wp::float(var_tex_nz);
        var_26 = wp::mul(var_10, var_25);
        var_27 = wp::int(var_26);
        var_30 = wp::sub(var_tex_nz, var_29);
        var_31 = wp::clamp(var_27, var_28, var_30);
        // if flip_x != 0:                                                                        <L 1227>
        var_33 = (var_flip_x != var_32);
        if (var_33) {
            // ix = (tex_nx - 1) - ix                                                             <L 1228>
            var_35 = wp::sub(var_tex_nx, var_34);
            var_36 = wp::sub(var_35, var_17);
        }
        var_37 = wp::where(var_33, var_36, var_17);
        // if flip_y != 0:                                                                        <L 1229>
        var_39 = (var_flip_y != var_38);
        if (var_39) {
            // iy = (tex_ny - 1) - iy                                                             <L 1230>
            var_41 = wp::sub(var_tex_ny, var_40);
            var_42 = wp::sub(var_41, var_24);
        }
        var_43 = wp::where(var_39, var_42, var_24);
        // if flip_z != 0:                                                                        <L 1231>
        var_45 = (var_flip_z != var_44);
        if (var_45) {
            // iz = (tex_nz - 1) - iz                                                             <L 1232>
            var_47 = wp::sub(var_tex_nz, var_46);
            var_48 = wp::sub(var_47, var_31);
        }
        var_49 = wp::where(var_45, var_48, var_31);
        // tri_rgb[t] = texture[ix, iy, iz]                                                       <L 1233>
        var_50 = wp::address(var_texture, var_37, var_43, var_49);
        var_51 = wp::load(var_50);
        // wp::array_store(var_tri_rgb, var_0, var_51);
        //---------
        // reverse
        wp::adj_array_store(var_tri_rgb, var_0, var_51, adj_tri_rgb, adj_0, adj_50);
        wp::adj_address(var_texture, var_37, var_43, var_49, adj_texture, adj_37, adj_43, adj_49, adj_50);
        // adj: tri_rgb[t] = texture[ix, iy, iz]                                                  <L 1233>
        wp::adj_where(var_45, var_48, var_31, adj_45, adj_48, adj_31, adj_49);
        if (var_45) {
            wp::adj_sub(var_47, var_31, adj_47, adj_31, adj_48);
            wp::adj_sub(var_tex_nz, var_46, adj_tex_nz, adj_46, adj_47);
            // adj: iz = (tex_nz - 1) - iz                                                        <L 1232>
        }
        // adj: if flip_z != 0:                                                                   <L 1231>
        wp::adj_where(var_39, var_42, var_24, adj_39, adj_42, adj_24, adj_43);
        if (var_39) {
            wp::adj_sub(var_41, var_24, adj_41, adj_24, adj_42);
            wp::adj_sub(var_tex_ny, var_40, adj_tex_ny, adj_40, adj_41);
            // adj: iy = (tex_ny - 1) - iy                                                        <L 1230>
        }
        // adj: if flip_y != 0:                                                                   <L 1229>
        wp::adj_where(var_33, var_36, var_17, adj_33, adj_36, adj_17, adj_37);
        if (var_33) {
            wp::adj_sub(var_35, var_17, adj_35, adj_17, adj_36);
            wp::adj_sub(var_tex_nx, var_34, adj_tex_nx, adj_34, adj_35);
            // adj: ix = (tex_nx - 1) - ix                                                        <L 1228>
        }
        // adj: if flip_x != 0:                                                                   <L 1227>
        wp::adj_clamp(var_27, var_28, var_30, adj_27, adj_28, adj_30, adj_31);
        wp::adj_sub(var_tex_nz, var_29, adj_tex_nz, adj_29, adj_30);
        wp::adj_int(var_26, adj_26, adj_27);
        wp::adj_mul(var_10, var_25, adj_10, adj_25, adj_26);
        wp::adj_float(var_tex_nz, adj_tex_nz, adj_25);
        // adj: iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                          <L 1226>
        wp::adj_clamp(var_20, var_21, var_23, adj_20, adj_21, adj_23, adj_24);
        wp::adj_sub(var_tex_ny, var_22, adj_tex_ny, adj_22, adj_23);
        wp::adj_int(var_19, adj_19, adj_20);
        wp::adj_mul(var_8, var_18, adj_8, adj_18, adj_19);
        wp::adj_float(var_tex_ny, adj_tex_ny, adj_18);
        // adj: iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                          <L 1225>
        wp::adj_clamp(var_13, var_14, var_16, adj_13, adj_14, adj_16, adj_17);
        wp::adj_sub(var_tex_nx, var_15, adj_tex_nx, adj_15, adj_16);
        wp::adj_int(var_12, adj_12, adj_13);
        wp::adj_mul(var_6, var_11, adj_6, adj_11, adj_12);
        wp::adj_float(var_tex_nx, adj_tex_nx, adj_11);
        // adj: ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                          <L 1224>
        adj__scale_about_centre_0(var_9, var_scale_z, adj_9, adj_scale_z, adj_10);
        adj__select_axis_0(var_3, var_src_z, adj_3, adj_src_z, adj_9);
        // adj: tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                     <L 1223>
        adj__scale_about_centre_0(var_7, var_scale_y, adj_7, adj_scale_y, adj_8);
        adj__select_axis_0(var_3, var_src_y, adj_3, adj_src_y, adj_7);
        // adj: tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                     <L 1222>
        adj__scale_about_centre_0(var_5, var_scale_x, adj_5, adj_scale_x, adj_6);
        adj__select_axis_0(var_3, var_src_x, adj_3, adj_src_x, adj_5);
        // adj: tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                     <L 1221>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_tri_centroid_uv3, var_0, adj_tri_centroid_uv3, adj_0, adj_2);
        // adj: uv = tri_centroid_uv3[t]                                                          <L 1220>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1219>
        }
        // adj: if t >= num_triangles:                                                            <L 1218>
        // adj: t = wp.tid()                                                                      <L 1217>
        // adj: def _sample_triangle_texture_kernel(                                              <L 1198>
        continue;
    }
}



extern "C" __global__ void _flatten_triangle_indices_kernel_192a7807_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::int32 var_num_triangles,
    wp::array_t<wp::int32> var_flat_indices)
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
        const wp::int32 var_2 = 3;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 1;
        wp::int32* var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 2;
        wp::int32* var_15;
        const wp::int32 var_16 = 2;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // forward
        // def _flatten_triangle_indices_kernel(                                                  <L 1029>
        // t = wp.tid()                                                                           <L 1035>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1036>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1037>
            continue;
        }
        // base = t * 3                                                                           <L 1038>
        var_3 = wp::mul(var_0, var_2);
        // flat_indices[base + 0] = tri_indices[t, 0]                                             <L 1039>
        var_5 = wp::address(var_tri_indices, var_0, var_4);
        var_7 = wp::add(var_3, var_6);
        var_8 = wp::load(var_5);
        wp::array_store(var_flat_indices, var_7, var_8);
        // flat_indices[base + 1] = tri_indices[t, 1]                                             <L 1040>
        var_10 = wp::address(var_tri_indices, var_0, var_9);
        var_12 = wp::add(var_3, var_11);
        var_13 = wp::load(var_10);
        wp::array_store(var_flat_indices, var_12, var_13);
        // flat_indices[base + 2] = tri_indices[t, 2]                                             <L 1041>
        var_15 = wp::address(var_tri_indices, var_0, var_14);
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::load(var_15);
        wp::array_store(var_flat_indices, var_17, var_18);
    }
}



extern "C" __global__ void _flatten_triangle_indices_kernel_192a7807_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::int32 var_num_triangles,
    wp::array_t<wp::int32> var_flat_indices,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::int32> adj_flat_indices)
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
        const wp::int32 var_2 = 3;
        wp::int32 var_3;
        const wp::int32 var_4 = 0;
        wp::int32* var_5;
        const wp::int32 var_6 = 0;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 1;
        wp::int32* var_10;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 2;
        wp::int32* var_15;
        const wp::int32 var_16 = 2;
        wp::int32 var_17;
        wp::int32 var_18;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::int32 adj_18 = {};
        //---------
        // forward
        // def _flatten_triangle_indices_kernel(                                                  <L 1029>
        // t = wp.tid()                                                                           <L 1035>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1036>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1037>
            goto label0;
        }
        // base = t * 3                                                                           <L 1038>
        var_3 = wp::mul(var_0, var_2);
        // flat_indices[base + 0] = tri_indices[t, 0]                                             <L 1039>
        var_5 = wp::address(var_tri_indices, var_0, var_4);
        var_7 = wp::add(var_3, var_6);
        var_8 = wp::load(var_5);
        // wp::array_store(var_flat_indices, var_7, var_8);
        // flat_indices[base + 1] = tri_indices[t, 1]                                             <L 1040>
        var_10 = wp::address(var_tri_indices, var_0, var_9);
        var_12 = wp::add(var_3, var_11);
        var_13 = wp::load(var_10);
        // wp::array_store(var_flat_indices, var_12, var_13);
        // flat_indices[base + 2] = tri_indices[t, 2]                                             <L 1041>
        var_15 = wp::address(var_tri_indices, var_0, var_14);
        var_17 = wp::add(var_3, var_16);
        var_18 = wp::load(var_15);
        // wp::array_store(var_flat_indices, var_17, var_18);
        //---------
        // reverse
        wp::adj_array_store(var_flat_indices, var_17, var_18, adj_flat_indices, adj_17, adj_15);
        wp::adj_add(var_3, var_16, adj_3, adj_16, adj_17);
        wp::adj_address(var_tri_indices, var_0, var_14, adj_tri_indices, adj_0, adj_14, adj_15);
        // adj: flat_indices[base + 2] = tri_indices[t, 2]                                        <L 1041>
        wp::adj_array_store(var_flat_indices, var_12, var_13, adj_flat_indices, adj_12, adj_10);
        wp::adj_add(var_3, var_11, adj_3, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_9, adj_tri_indices, adj_0, adj_9, adj_10);
        // adj: flat_indices[base + 1] = tri_indices[t, 1]                                        <L 1040>
        wp::adj_array_store(var_flat_indices, var_7, var_8, adj_flat_indices, adj_7, adj_5);
        wp::adj_add(var_3, var_6, adj_3, adj_6, adj_7);
        wp::adj_address(var_tri_indices, var_0, var_4, adj_tri_indices, adj_0, adj_4, adj_5);
        // adj: flat_indices[base + 0] = tri_indices[t, 0]                                        <L 1039>
        wp::adj_mul(var_0, var_2, adj_0, adj_2, adj_3);
        // adj: base = t * 3                                                                      <L 1038>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1037>
        }
        // adj: if t >= num_triangles:                                                            <L 1036>
        // adj: t = wp.tid()                                                                      <L 1035>
        // adj: def _flatten_triangle_indices_kernel(                                             <L 1029>
        continue;
    }
}



extern "C" __global__ void _compute_vertex_atlas_uvs_kernel_685ffcd1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_vertices,
    wp::array_t<wp::vec_t<2, wp::float32>> var_out_uvs)
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
        const wp::float32 var_2 = 0.0;
        const wp::float32 var_3 = 0.0;
        wp::vec_t<2, wp::float32> var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 2;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::float32 var_20 = 1.0;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::vec_t<2, wp::float32> var_24;
        //---------
        // forward
        // def _compute_vertex_atlas_uvs_kernel(                                                  <L 1364>
        // i = wp.tid()                                                                           <L 1377>
        var_0 = builtin_tid1d();
        // if i >= num_vertices:                                                                  <L 1378>
        var_1 = (var_0 >= var_num_vertices);
        if (var_1) {
            // out_uvs[i] = wp.vec2(0.0, 0.0)                                                     <L 1379>
            var_4 = wp::vec_t<2, wp::float32>(var_2, var_3);
            wp::array_store(var_out_uvs, var_0, var_4);
            // return                                                                             <L 1380>
            continue;
        }
        // tiles_per_row = atlas_width / tile_size                                                <L 1381>
        var_5 = wp::div(var_atlas_width, var_tile_size);
        // tile_x = i % tiles_per_row                                                             <L 1382>
        var_6 = wp::mod(var_0, var_5);
        // tile_y = i / tiles_per_row                                                             <L 1383>
        var_7 = wp::div(var_0, var_5);
        // cx = float(tile_x * tile_size + tile_size / 2)                                         <L 1384>
        var_8 = wp::mul(var_6, var_tile_size);
        var_10 = wp::div(var_tile_size, var_9);
        var_11 = wp::add(var_8, var_10);
        var_12 = wp::float(var_11);
        // cy = float(tile_y * tile_size + tile_size / 2)                                         <L 1385>
        var_13 = wp::mul(var_7, var_tile_size);
        var_15 = wp::div(var_tile_size, var_14);
        var_16 = wp::add(var_13, var_15);
        var_17 = wp::float(var_16);
        // out_uvs[i] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))        <L 1386>
        var_18 = wp::float(var_atlas_width);
        var_19 = wp::div(var_12, var_18);
        var_21 = wp::float(var_atlas_height);
        var_22 = wp::div(var_17, var_21);
        var_23 = wp::sub(var_20, var_22);
        var_24 = wp::vec_t<2, wp::float32>(var_19, var_23);
        wp::array_store(var_out_uvs, var_0, var_24);
    }
}



extern "C" __global__ void _compute_vertex_atlas_uvs_kernel_685ffcd1_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_vertices,
    wp::array_t<wp::vec_t<2, wp::float32>> var_out_uvs,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_vertices,
    wp::array_t<wp::vec_t<2, wp::float32>> adj_out_uvs)
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
        const wp::float32 var_2 = 0.0;
        const wp::float32 var_3 = 0.0;
        wp::vec_t<2, wp::float32> var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::float32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 2;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::float32 var_20 = 1.0;
        wp::float32 var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        wp::vec_t<2, wp::float32> var_24;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::float32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::vec_t<2, wp::float32> adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::vec_t<2, wp::float32> adj_24 = {};
        //---------
        // forward
        // def _compute_vertex_atlas_uvs_kernel(                                                  <L 1364>
        // i = wp.tid()                                                                           <L 1377>
        var_0 = builtin_tid1d();
        // if i >= num_vertices:                                                                  <L 1378>
        var_1 = (var_0 >= var_num_vertices);
        if (var_1) {
            // out_uvs[i] = wp.vec2(0.0, 0.0)                                                     <L 1379>
            var_4 = wp::vec_t<2, wp::float32>(var_2, var_3);
            // wp::array_store(var_out_uvs, var_0, var_4);
            // return                                                                             <L 1380>
            goto label0;
        }
        // tiles_per_row = atlas_width / tile_size                                                <L 1381>
        var_5 = wp::div(var_atlas_width, var_tile_size);
        // tile_x = i % tiles_per_row                                                             <L 1382>
        var_6 = wp::mod(var_0, var_5);
        // tile_y = i / tiles_per_row                                                             <L 1383>
        var_7 = wp::div(var_0, var_5);
        // cx = float(tile_x * tile_size + tile_size / 2)                                         <L 1384>
        var_8 = wp::mul(var_6, var_tile_size);
        var_10 = wp::div(var_tile_size, var_9);
        var_11 = wp::add(var_8, var_10);
        var_12 = wp::float(var_11);
        // cy = float(tile_y * tile_size + tile_size / 2)                                         <L 1385>
        var_13 = wp::mul(var_7, var_tile_size);
        var_15 = wp::div(var_tile_size, var_14);
        var_16 = wp::add(var_13, var_15);
        var_17 = wp::float(var_16);
        // out_uvs[i] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))        <L 1386>
        var_18 = wp::float(var_atlas_width);
        var_19 = wp::div(var_12, var_18);
        var_21 = wp::float(var_atlas_height);
        var_22 = wp::div(var_17, var_21);
        var_23 = wp::sub(var_20, var_22);
        var_24 = wp::vec_t<2, wp::float32>(var_19, var_23);
        // wp::array_store(var_out_uvs, var_0, var_24);
        //---------
        // reverse
        wp::adj_array_store(var_out_uvs, var_0, var_24, adj_out_uvs, adj_0, adj_24);
        wp::adj_vec_t(var_19, var_23, adj_19, adj_23, adj_24);
        wp::adj_sub(var_20, var_22, adj_20, adj_22, adj_23);
        wp::adj_div(var_17, var_21, var_22, adj_17, adj_21, adj_22);
        wp::adj_float(var_atlas_height, adj_atlas_height, adj_21);
        wp::adj_div(var_12, var_18, var_19, adj_12, adj_18, adj_19);
        wp::adj_float(var_atlas_width, adj_atlas_width, adj_18);
        // adj: out_uvs[i] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))   <L 1386>
        wp::adj_float(var_16, adj_16, adj_17);
        wp::adj_add(var_13, var_15, adj_13, adj_15, adj_16);
        wp::adj_div(var_tile_size, var_14, var_15, adj_tile_size, adj_14, adj_15);
        wp::adj_mul(var_7, var_tile_size, adj_7, adj_tile_size, adj_13);
        // adj: cy = float(tile_y * tile_size + tile_size / 2)                                    <L 1385>
        wp::adj_float(var_11, adj_11, adj_12);
        wp::adj_add(var_8, var_10, adj_8, adj_10, adj_11);
        wp::adj_div(var_tile_size, var_9, var_10, adj_tile_size, adj_9, adj_10);
        wp::adj_mul(var_6, var_tile_size, adj_6, adj_tile_size, adj_8);
        // adj: cx = float(tile_x * tile_size + tile_size / 2)                                    <L 1384>
        wp::adj_div(var_0, var_5, var_7, adj_0, adj_5, adj_7);
        // adj: tile_y = i / tiles_per_row                                                        <L 1383>
        wp::adj_mod(var_0, var_5, adj_0, adj_5, adj_6);
        // adj: tile_x = i % tiles_per_row                                                        <L 1382>
        wp::adj_div(var_atlas_width, var_tile_size, var_5, adj_atlas_width, adj_tile_size, adj_5);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1381>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1380>
            wp::adj_array_store(var_out_uvs, var_0, var_4, adj_out_uvs, adj_0, adj_4);
            wp::adj_vec_t(var_2, var_3, adj_2, adj_3, adj_4);
            // adj: out_uvs[i] = wp.vec2(0.0, 0.0)                                                <L 1379>
        }
        // adj: if i >= num_vertices:                                                             <L 1378>
        // adj: i = wp.tid()                                                                      <L 1377>
        // adj: def _compute_vertex_atlas_uvs_kernel(                                             <L 1364>
        continue;
    }
}



extern "C" __global__ void _fill_triangle_atlas_kernel_fe618cd5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_rgb,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // forward
        // def _fill_triangle_atlas_kernel(                                                       <L 1237>
        // tx, ty, sub = wp.tid()                                                                 <L 1246>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1247>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1248>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1249>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1250>
            continue;
        }
        // px = sub % tile_size                                                                   <L 1251>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1252>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1253>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1254>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1255>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1256>
            continue;
        }
        // atlas_rgb[y, x] = tri_rgb[tid]                                                         <L 1257>
        var_16 = wp::address(var_tri_rgb, var_5);
        var_17 = wp::load(var_16);
        wp::array_store(var_atlas_rgb, var_12, var_10, var_17);
    }
}



extern "C" __global__ void _fill_triangle_atlas_kernel_fe618cd5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_rgb,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_tri_rgb,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        //---------
        // forward
        // def _fill_triangle_atlas_kernel(                                                       <L 1237>
        // tx, ty, sub = wp.tid()                                                                 <L 1246>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1247>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1248>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1249>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1250>
            goto label0;
        }
        // px = sub % tile_size                                                                   <L 1251>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1252>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1253>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1254>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1255>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1256>
            goto label1;
        }
        // atlas_rgb[y, x] = tri_rgb[tid]                                                         <L 1257>
        var_16 = wp::address(var_tri_rgb, var_5);
        var_17 = wp::load(var_16);
        // wp::array_store(var_atlas_rgb, var_12, var_10, var_17);
        //---------
        // reverse
        wp::adj_array_store(var_atlas_rgb, var_12, var_10, var_17, adj_atlas_rgb, adj_12, adj_10, adj_16);
        wp::adj_address(var_tri_rgb, var_5, adj_tri_rgb, adj_5, adj_16);
        // adj: atlas_rgb[y, x] = tri_rgb[tid]                                                    <L 1257>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1256>
        }
        if (!var_13) {
        }
        // adj: if x >= atlas_width or y >= atlas_height:                                         <L 1255>
        wp::adj_add(var_11, var_8, adj_11, adj_8, adj_12);
        wp::adj_mul(var_1, var_tile_size, adj_1, adj_tile_size, adj_11);
        // adj: y = ty * tile_size + py                                                           <L 1254>
        wp::adj_add(var_9, var_7, adj_9, adj_7, adj_10);
        wp::adj_mul(var_0, var_tile_size, adj_0, adj_tile_size, adj_9);
        // adj: x = tx * tile_size + px                                                           <L 1253>
        wp::adj_div(var_2, var_tile_size, var_8, adj_2, adj_tile_size, adj_8);
        // adj: py = sub / tile_size                                                              <L 1252>
        wp::adj_mod(var_2, var_tile_size, adj_2, adj_tile_size, adj_7);
        // adj: px = sub % tile_size                                                              <L 1251>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1250>
        }
        // adj: if tid >= num_triangles:                                                          <L 1249>
        wp::adj_add(var_4, var_0, adj_4, adj_0, adj_5);
        wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
        // adj: tid = ty * tiles_per_row + tx                                                     <L 1248>
        wp::adj_div(var_atlas_width, var_tile_size, var_3, adj_atlas_width, adj_tile_size, adj_3);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1247>
        // adj: tx, ty, sub = wp.tid()                                                            <L 1246>
        // adj: def _fill_triangle_atlas_kernel(                                                  <L 1237>
        continue;
    }
}



extern "C" __global__ void _fill_atlas_tiles_kernel_aab98030_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_rgb,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_vertices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // forward
        // def _fill_atlas_tiles_kernel(                                                          <L 1390>
        // tx, ty, sub = wp.tid()                                                                 <L 1399>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1400>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // vid = ty * tiles_per_row + tx                                                          <L 1401>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if vid >= num_vertices:                                                                <L 1402>
        var_6 = (var_5 >= var_num_vertices);
        if (var_6) {
            // return                                                                             <L 1403>
            continue;
        }
        // px = sub % tile_size                                                                   <L 1404>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1405>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1406>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1407>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1408>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1409>
            continue;
        }
        // atlas_rgb[y, x] = vertex_rgb[vid]                                                      <L 1410>
        var_16 = wp::address(var_vertex_rgb, var_5);
        var_17 = wp::load(var_16);
        wp::array_store(var_atlas_rgb, var_12, var_10, var_17);
    }
}



extern "C" __global__ void _fill_atlas_tiles_kernel_aab98030_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_rgb,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_vertices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_rgb,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_vertices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        //---------
        // forward
        // def _fill_atlas_tiles_kernel(                                                          <L 1390>
        // tx, ty, sub = wp.tid()                                                                 <L 1399>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1400>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // vid = ty * tiles_per_row + tx                                                          <L 1401>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if vid >= num_vertices:                                                                <L 1402>
        var_6 = (var_5 >= var_num_vertices);
        if (var_6) {
            // return                                                                             <L 1403>
            goto label0;
        }
        // px = sub % tile_size                                                                   <L 1404>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1405>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1406>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1407>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1408>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1409>
            goto label1;
        }
        // atlas_rgb[y, x] = vertex_rgb[vid]                                                      <L 1410>
        var_16 = wp::address(var_vertex_rgb, var_5);
        var_17 = wp::load(var_16);
        // wp::array_store(var_atlas_rgb, var_12, var_10, var_17);
        //---------
        // reverse
        wp::adj_array_store(var_atlas_rgb, var_12, var_10, var_17, adj_atlas_rgb, adj_12, adj_10, adj_16);
        wp::adj_address(var_vertex_rgb, var_5, adj_vertex_rgb, adj_5, adj_16);
        // adj: atlas_rgb[y, x] = vertex_rgb[vid]                                                 <L 1410>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1409>
        }
        if (!var_13) {
        }
        // adj: if x >= atlas_width or y >= atlas_height:                                         <L 1408>
        wp::adj_add(var_11, var_8, adj_11, adj_8, adj_12);
        wp::adj_mul(var_1, var_tile_size, adj_1, adj_tile_size, adj_11);
        // adj: y = ty * tile_size + py                                                           <L 1407>
        wp::adj_add(var_9, var_7, adj_9, adj_7, adj_10);
        wp::adj_mul(var_0, var_tile_size, adj_0, adj_tile_size, adj_9);
        // adj: x = tx * tile_size + px                                                           <L 1406>
        wp::adj_div(var_2, var_tile_size, var_8, adj_2, adj_tile_size, adj_8);
        // adj: py = sub / tile_size                                                              <L 1405>
        wp::adj_mod(var_2, var_tile_size, adj_2, adj_tile_size, adj_7);
        // adj: px = sub % tile_size                                                              <L 1404>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1403>
        }
        // adj: if vid >= num_vertices:                                                           <L 1402>
        wp::adj_add(var_4, var_0, adj_4, adj_0, adj_5);
        wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
        // adj: vid = ty * tiles_per_row + tx                                                     <L 1401>
        wp::adj_div(var_atlas_width, var_tile_size, var_3, adj_atlas_width, adj_tile_size, adj_3);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1400>
        // adj: tx, ty, sub = wp.tid()                                                            <L 1399>
        // adj: def _fill_atlas_tiles_kernel(                                                     <L 1390>
        continue;
    }
}



extern "C" __global__ void _fill_triangle_stress_atlas_kernel_973685e6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::float32 var_color_scale,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        wp::int32* var_17;
        const wp::int32 var_18 = 6;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32* var_25;
        const wp::int32 var_26 = 6;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32* var_33;
        const wp::int32 var_34 = 6;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        bool var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 1.0;
        const wp::float32 var_45 = 3.0;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.5;
        const wp::float32 var_49 = 0.5;
        wp::vec_t<2, wp::float32> var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.5;
        wp::float32 var_53;
        const wp::float32 var_54 = 0.5;
        wp::vec_t<2, wp::float32> var_55;
        const wp::float32 var_56 = 0.5;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.5;
        wp::float32 var_59;
        wp::vec_t<2, wp::float32> var_60;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.5;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.5;
        wp::float32 var_66;
        wp::vec_t<2, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 1;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        //---------
        // forward
        // def _fill_triangle_stress_atlas_kernel(                                                <L 1441>
        // tx, ty, sub = wp.tid()                                                                 <L 1452>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1453>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1454>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1455>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1456>
            continue;
        }
        // px = sub % tile_size                                                                   <L 1458>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1459>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1460>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1461>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1462>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1463>
            continue;
        }
        // s0 = cell_stretch[tri_indices[tid, 0] / 6]                                             <L 1465>
        var_17 = wp::address(var_tri_indices, var_5, var_16);
        var_20 = wp::load(var_17);
        var_19 = wp::div(var_20, var_18);
        var_21 = wp::address(var_cell_stretch, var_19);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // s1 = cell_stretch[tri_indices[tid, 1] / 6]                                             <L 1466>
        var_25 = wp::address(var_tri_indices, var_5, var_24);
        var_28 = wp::load(var_25);
        var_27 = wp::div(var_28, var_26);
        var_29 = wp::address(var_cell_stretch, var_27);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // s2 = cell_stretch[tri_indices[tid, 2] / 6]                                             <L 1467>
        var_33 = wp::address(var_tri_indices, var_5, var_32);
        var_36 = wp::load(var_33);
        var_35 = wp::div(var_36, var_34);
        var_37 = wp::address(var_cell_stretch, var_35);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // if tile_size <= 1:                                                                     <L 1468>
        var_41 = (var_tile_size <= var_40);
        if (var_41) {
            // stress = (s0 + s1 + s2) * (1.0 / 3.0)                                              <L 1469>
            var_42 = wp::add(var_22, var_30);
            var_43 = wp::add(var_42, var_38);
            var_46 = wp::div(var_44, var_45);
            var_47 = wp::mul(var_43, var_46);
        }
        if (!var_41) {
            // a = wp.vec2(0.5, 0.5)                                                              <L 1471>
            var_50 = wp::vec_t<2, wp::float32>(var_48, var_49);
            // b = wp.vec2(float(tile_size) - 0.5, 0.5)                                           <L 1472>
            var_51 = wp::float(var_tile_size);
            var_53 = wp::sub(var_51, var_52);
            var_55 = wp::vec_t<2, wp::float32>(var_53, var_54);
            // c = wp.vec2(0.5, float(tile_size) - 0.5)                                           <L 1473>
            var_57 = wp::float(var_tile_size);
            var_59 = wp::sub(var_57, var_58);
            var_60 = wp::vec_t<2, wp::float32>(var_56, var_59);
            // p = wp.vec2(float(px) + 0.5, float(py) + 0.5)                                      <L 1474>
            var_61 = wp::float(var_7);
            var_63 = wp::add(var_61, var_62);
            var_64 = wp::float(var_8);
            var_66 = wp::add(var_64, var_65);
            var_67 = wp::vec_t<2, wp::float32>(var_63, var_66);
            // bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))                       <L 1475>
            var_68 = _triangle_barycentric_0(var_67, var_50, var_55, var_60);
            var_69 = _clamp_barycentric_0(var_68);
            // stress = s0 * bary[0] + s1 * bary[1] + s2 * bary[2]                                <L 1476>
            var_71 = wp::extract(var_69, var_70);
            var_72 = wp::mul(var_22, var_71);
            var_74 = wp::extract(var_69, var_73);
            var_75 = wp::mul(var_30, var_74);
            var_76 = wp::add(var_72, var_75);
            var_78 = wp::extract(var_69, var_77);
            var_79 = wp::mul(var_38, var_78);
            var_80 = wp::add(var_76, var_79);
        }
        var_81 = wp::where(var_41, var_47, var_80);
        // atlas_rgb[y, x] = _cold_warm_stress_color(stress * color_scale)                        <L 1477>
        var_82 = wp::mul(var_81, var_color_scale);
        var_83 = _cold_warm_stress_color_0(var_82);
        wp::array_store(var_atlas_rgb, var_12, var_10, var_83);
    }
}



extern "C" __global__ void _fill_triangle_stress_atlas_kernel_973685e6_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::float32 var_color_scale,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::float32> adj_cell_stretch,
    wp::float32 adj_color_scale,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        wp::int32* var_17;
        const wp::int32 var_18 = 6;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::float32* var_21;
        wp::float32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 1;
        wp::int32* var_25;
        const wp::int32 var_26 = 6;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::float32* var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32* var_33;
        const wp::int32 var_34 = 6;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::int32 var_40 = 1;
        bool var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::float32 var_44 = 1.0;
        const wp::float32 var_45 = 3.0;
        wp::float32 var_46;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.5;
        const wp::float32 var_49 = 0.5;
        wp::vec_t<2, wp::float32> var_50;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.5;
        wp::float32 var_53;
        const wp::float32 var_54 = 0.5;
        wp::vec_t<2, wp::float32> var_55;
        const wp::float32 var_56 = 0.5;
        wp::float32 var_57;
        const wp::float32 var_58 = 0.5;
        wp::float32 var_59;
        wp::vec_t<2, wp::float32> var_60;
        wp::float32 var_61;
        const wp::float32 var_62 = 0.5;
        wp::float32 var_63;
        wp::float32 var_64;
        const wp::float32 var_65 = 0.5;
        wp::float32 var_66;
        wp::vec_t<2, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 1;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        wp::vec_t<3, wp::float32> var_83;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        bool adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::float32 adj_49 = {};
        wp::vec_t<2, wp::float32> adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::vec_t<2, wp::float32> adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::vec_t<2, wp::float32> adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::vec_t<2, wp::float32> adj_67 = {};
        wp::vec_t<3, wp::float32> adj_68 = {};
        wp::vec_t<3, wp::float32> adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::vec_t<3, wp::float32> adj_83 = {};
        //---------
        // forward
        // def _fill_triangle_stress_atlas_kernel(                                                <L 1441>
        // tx, ty, sub = wp.tid()                                                                 <L 1452>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1453>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1454>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1455>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1456>
            goto label0;
        }
        // px = sub % tile_size                                                                   <L 1458>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1459>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1460>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1461>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1462>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1463>
            goto label1;
        }
        // s0 = cell_stretch[tri_indices[tid, 0] / 6]                                             <L 1465>
        var_17 = wp::address(var_tri_indices, var_5, var_16);
        var_20 = wp::load(var_17);
        var_19 = wp::div(var_20, var_18);
        var_21 = wp::address(var_cell_stretch, var_19);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // s1 = cell_stretch[tri_indices[tid, 1] / 6]                                             <L 1466>
        var_25 = wp::address(var_tri_indices, var_5, var_24);
        var_28 = wp::load(var_25);
        var_27 = wp::div(var_28, var_26);
        var_29 = wp::address(var_cell_stretch, var_27);
        var_31 = wp::load(var_29);
        var_30 = wp::copy(var_31);
        // s2 = cell_stretch[tri_indices[tid, 2] / 6]                                             <L 1467>
        var_33 = wp::address(var_tri_indices, var_5, var_32);
        var_36 = wp::load(var_33);
        var_35 = wp::div(var_36, var_34);
        var_37 = wp::address(var_cell_stretch, var_35);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // if tile_size <= 1:                                                                     <L 1468>
        var_41 = (var_tile_size <= var_40);
        if (var_41) {
            // stress = (s0 + s1 + s2) * (1.0 / 3.0)                                              <L 1469>
            var_42 = wp::add(var_22, var_30);
            var_43 = wp::add(var_42, var_38);
            var_46 = wp::div(var_44, var_45);
            var_47 = wp::mul(var_43, var_46);
        }
        if (!var_41) {
            // a = wp.vec2(0.5, 0.5)                                                              <L 1471>
            var_50 = wp::vec_t<2, wp::float32>(var_48, var_49);
            // b = wp.vec2(float(tile_size) - 0.5, 0.5)                                           <L 1472>
            var_51 = wp::float(var_tile_size);
            var_53 = wp::sub(var_51, var_52);
            var_55 = wp::vec_t<2, wp::float32>(var_53, var_54);
            // c = wp.vec2(0.5, float(tile_size) - 0.5)                                           <L 1473>
            var_57 = wp::float(var_tile_size);
            var_59 = wp::sub(var_57, var_58);
            var_60 = wp::vec_t<2, wp::float32>(var_56, var_59);
            // p = wp.vec2(float(px) + 0.5, float(py) + 0.5)                                      <L 1474>
            var_61 = wp::float(var_7);
            var_63 = wp::add(var_61, var_62);
            var_64 = wp::float(var_8);
            var_66 = wp::add(var_64, var_65);
            var_67 = wp::vec_t<2, wp::float32>(var_63, var_66);
            // bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))                       <L 1475>
            var_68 = _triangle_barycentric_0(var_67, var_50, var_55, var_60);
            var_69 = _clamp_barycentric_0(var_68);
            // stress = s0 * bary[0] + s1 * bary[1] + s2 * bary[2]                                <L 1476>
            var_71 = wp::extract(var_69, var_70);
            var_72 = wp::mul(var_22, var_71);
            var_74 = wp::extract(var_69, var_73);
            var_75 = wp::mul(var_30, var_74);
            var_76 = wp::add(var_72, var_75);
            var_78 = wp::extract(var_69, var_77);
            var_79 = wp::mul(var_38, var_78);
            var_80 = wp::add(var_76, var_79);
        }
        var_81 = wp::where(var_41, var_47, var_80);
        // atlas_rgb[y, x] = _cold_warm_stress_color(stress * color_scale)                        <L 1477>
        var_82 = wp::mul(var_81, var_color_scale);
        var_83 = _cold_warm_stress_color_0(var_82);
        // wp::array_store(var_atlas_rgb, var_12, var_10, var_83);
        //---------
        // reverse
        wp::adj_array_store(var_atlas_rgb, var_12, var_10, var_83, adj_atlas_rgb, adj_12, adj_10, adj_83);
        adj__cold_warm_stress_color_0(var_82, adj_82, adj_83);
        wp::adj_mul(var_81, var_color_scale, adj_81, adj_color_scale, adj_82);
        // adj: atlas_rgb[y, x] = _cold_warm_stress_color(stress * color_scale)                   <L 1477>
        wp::adj_where(var_41, var_47, var_80, adj_41, adj_47, adj_80, adj_81);
        if (!var_41) {
            wp::adj_add(var_76, var_79, adj_76, adj_79, adj_80);
            wp::adj_mul(var_38, var_78, adj_38, adj_78, adj_79);
            wp::adj_extract(var_69, var_77, adj_69, adj_77, adj_78);
            wp::adj_add(var_72, var_75, adj_72, adj_75, adj_76);
            wp::adj_mul(var_30, var_74, adj_30, adj_74, adj_75);
            wp::adj_extract(var_69, var_73, adj_69, adj_73, adj_74);
            wp::adj_mul(var_22, var_71, adj_22, adj_71, adj_72);
            wp::adj_extract(var_69, var_70, adj_69, adj_70, adj_71);
            // adj: stress = s0 * bary[0] + s1 * bary[1] + s2 * bary[2]                           <L 1476>
            adj__clamp_barycentric_0(var_68, adj_68, adj_69);
            adj__triangle_barycentric_0(var_67, var_50, var_55, var_60, adj_67, adj_50, adj_55, adj_60, adj_68);
            // adj: bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))                  <L 1475>
            wp::adj_vec_t(var_63, var_66, adj_63, adj_66, adj_67);
            wp::adj_add(var_64, var_65, adj_64, adj_65, adj_66);
            wp::adj_float(var_8, adj_8, adj_64);
            wp::adj_add(var_61, var_62, adj_61, adj_62, adj_63);
            wp::adj_float(var_7, adj_7, adj_61);
            // adj: p = wp.vec2(float(px) + 0.5, float(py) + 0.5)                                 <L 1474>
            wp::adj_vec_t(var_56, var_59, adj_56, adj_59, adj_60);
            wp::adj_sub(var_57, var_58, adj_57, adj_58, adj_59);
            wp::adj_float(var_tile_size, adj_tile_size, adj_57);
            // adj: c = wp.vec2(0.5, float(tile_size) - 0.5)                                      <L 1473>
            wp::adj_vec_t(var_53, var_54, adj_53, adj_54, adj_55);
            wp::adj_sub(var_51, var_52, adj_51, adj_52, adj_53);
            wp::adj_float(var_tile_size, adj_tile_size, adj_51);
            // adj: b = wp.vec2(float(tile_size) - 0.5, 0.5)                                      <L 1472>
            wp::adj_vec_t(var_48, var_49, adj_48, adj_49, adj_50);
            // adj: a = wp.vec2(0.5, 0.5)                                                         <L 1471>
        }
        if (var_41) {
            wp::adj_mul(var_43, var_46, adj_43, adj_46, adj_47);
            wp::adj_div(var_44, var_45, var_46, adj_44, adj_45, adj_46);
            wp::adj_add(var_42, var_38, adj_42, adj_38, adj_43);
            wp::adj_add(var_22, var_30, adj_22, adj_30, adj_42);
            // adj: stress = (s0 + s1 + s2) * (1.0 / 3.0)                                         <L 1469>
        }
        // adj: if tile_size <= 1:                                                                <L 1468>
        wp::adj_copy(var_39, adj_37, adj_38);
        wp::adj_address(var_cell_stretch, var_35, adj_cell_stretch, adj_35, adj_37);
        wp::adj_div(var_36, var_34, var_35, adj_33, adj_34, adj_35);
        wp::adj_address(var_tri_indices, var_5, var_32, adj_tri_indices, adj_5, adj_32, adj_33);
        // adj: s2 = cell_stretch[tri_indices[tid, 2] / 6]                                        <L 1467>
        wp::adj_copy(var_31, adj_29, adj_30);
        wp::adj_address(var_cell_stretch, var_27, adj_cell_stretch, adj_27, adj_29);
        wp::adj_div(var_28, var_26, var_27, adj_25, adj_26, adj_27);
        wp::adj_address(var_tri_indices, var_5, var_24, adj_tri_indices, adj_5, adj_24, adj_25);
        // adj: s1 = cell_stretch[tri_indices[tid, 1] / 6]                                        <L 1466>
        wp::adj_copy(var_23, adj_21, adj_22);
        wp::adj_address(var_cell_stretch, var_19, adj_cell_stretch, adj_19, adj_21);
        wp::adj_div(var_20, var_18, var_19, adj_17, adj_18, adj_19);
        wp::adj_address(var_tri_indices, var_5, var_16, adj_tri_indices, adj_5, adj_16, adj_17);
        // adj: s0 = cell_stretch[tri_indices[tid, 0] / 6]                                        <L 1465>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1463>
        }
        if (!var_13) {
        }
        // adj: if x >= atlas_width or y >= atlas_height:                                         <L 1462>
        wp::adj_add(var_11, var_8, adj_11, adj_8, adj_12);
        wp::adj_mul(var_1, var_tile_size, adj_1, adj_tile_size, adj_11);
        // adj: y = ty * tile_size + py                                                           <L 1461>
        wp::adj_add(var_9, var_7, adj_9, adj_7, adj_10);
        wp::adj_mul(var_0, var_tile_size, adj_0, adj_tile_size, adj_9);
        // adj: x = tx * tile_size + px                                                           <L 1460>
        wp::adj_div(var_2, var_tile_size, var_8, adj_2, adj_tile_size, adj_8);
        // adj: py = sub / tile_size                                                              <L 1459>
        wp::adj_mod(var_2, var_tile_size, adj_2, adj_tile_size, adj_7);
        // adj: px = sub % tile_size                                                              <L 1458>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1456>
        }
        // adj: if tid >= num_triangles:                                                          <L 1455>
        wp::adj_add(var_4, var_0, adj_4, adj_0, adj_5);
        wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
        // adj: tid = ty * tiles_per_row + tx                                                     <L 1454>
        wp::adj_div(var_atlas_width, var_tile_size, var_3, adj_atlas_width, adj_tile_size, adj_3);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1453>
        // adj: tx, ty, sub = wp.tid()                                                            <L 1452>
        // adj: def _fill_triangle_stress_atlas_kernel(                                           <L 1441>
        continue;
    }
}



extern "C" __global__ void _gather_cryo_colored_particles_kernel_a370a66f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::float32 var_inv_grid_nx,
    wp::float32 var_inv_grid_ny,
    wp::float32 var_inv_grid_nz,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        wp::int32* var_15;
        wp::float32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32* var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 2;
        wp::int32* var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        bool var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        bool var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        const wp::int32 var_75 = 0;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32>* var_80;
        wp::vec_t<3, wp::float32> var_81;
        //---------
        // forward
        // def _gather_cryo_colored_particles_kernel(                                             <L 1598>
        // i = wp.tid()                                                                           <L 1630>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 1631>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 1632>
            continue;
        }
        // if particle_q[i][2] > cut_z:                                                           <L 1633>
        var_9 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = (var_11 > var_cut_z);
        if (var_13) {
            // return                                                                             <L 1634>
            continue;
        }
        // gx = float(particle_grid_xyz[i, 0]) * inv_grid_nx                                      <L 1635>
        var_15 = wp::address(var_particle_grid_xyz, var_0, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::float(var_17);
        var_18 = wp::mul(var_16, var_inv_grid_nx);
        // gy = float(particle_grid_xyz[i, 1]) * inv_grid_ny                                      <L 1636>
        var_20 = wp::address(var_particle_grid_xyz, var_0, var_19);
        var_22 = wp::load(var_20);
        var_21 = wp::float(var_22);
        var_23 = wp::mul(var_21, var_inv_grid_ny);
        // gz = float(particle_grid_xyz[i, 2]) * inv_grid_nz                                      <L 1637>
        var_25 = wp::address(var_particle_grid_xyz, var_0, var_24);
        var_27 = wp::load(var_25);
        var_26 = wp::float(var_27);
        var_28 = wp::mul(var_26, var_inv_grid_nz);
        // uv = wp.vec3(gx, gy, gz)                                                               <L 1638>
        var_29 = wp::vec_t<3, wp::float32>(var_18, var_23, var_28);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 1639>
        var_30 = _select_axis_0(var_29, var_src_x);
        var_31 = _scale_about_centre_0(var_30, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 1640>
        var_32 = _select_axis_0(var_29, var_src_y);
        var_33 = _scale_about_centre_0(var_32, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 1641>
        var_34 = _select_axis_0(var_29, var_src_z);
        var_35 = _scale_about_centre_0(var_34, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 1642>
        var_36 = wp::float(var_tex_nx);
        var_37 = wp::mul(var_31, var_36);
        var_38 = wp::int(var_37);
        var_41 = wp::sub(var_tex_nx, var_40);
        var_42 = wp::clamp(var_38, var_39, var_41);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 1643>
        var_43 = wp::float(var_tex_ny);
        var_44 = wp::mul(var_33, var_43);
        var_45 = wp::int(var_44);
        var_48 = wp::sub(var_tex_ny, var_47);
        var_49 = wp::clamp(var_45, var_46, var_48);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 1644>
        var_50 = wp::float(var_tex_nz);
        var_51 = wp::mul(var_35, var_50);
        var_52 = wp::int(var_51);
        var_55 = wp::sub(var_tex_nz, var_54);
        var_56 = wp::clamp(var_52, var_53, var_55);
        // if flip_x != 0:                                                                        <L 1645>
        var_58 = (var_flip_x != var_57);
        if (var_58) {
            // ix = (tex_nx - 1) - ix                                                             <L 1646>
            var_60 = wp::sub(var_tex_nx, var_59);
            var_61 = wp::sub(var_60, var_42);
        }
        var_62 = wp::where(var_58, var_61, var_42);
        // if flip_y != 0:                                                                        <L 1647>
        var_64 = (var_flip_y != var_63);
        if (var_64) {
            // iy = (tex_ny - 1) - iy                                                             <L 1648>
            var_66 = wp::sub(var_tex_ny, var_65);
            var_67 = wp::sub(var_66, var_49);
        }
        var_68 = wp::where(var_64, var_67, var_49);
        // if flip_z != 0:                                                                        <L 1649>
        var_70 = (var_flip_z != var_69);
        if (var_70) {
            // iz = (tex_nz - 1) - iz                                                             <L 1650>
            var_72 = wp::sub(var_tex_nz, var_71);
            var_73 = wp::sub(var_72, var_56);
        }
        var_74 = wp::where(var_70, var_73, var_56);
        // idx = wp.atomic_add(counter, 0, 1)                                                     <L 1651>
        var_77 = wp::atomic_add(var_counter, var_75, var_76);
        // out_points[idx] = particle_q[i]                                                        <L 1652>
        var_78 = wp::address(var_particle_q, var_0);
        var_79 = wp::load(var_78);
        wp::array_store(var_out_points, var_77, var_79);
        // out_colors[idx] = texture[ix, iy, iz]                                                  <L 1653>
        var_80 = wp::address(var_texture, var_62, var_68, var_74);
        var_81 = wp::load(var_80);
        wp::array_store(var_out_colors, var_77, var_81);
    }
}



extern "C" __global__ void _gather_cryo_colored_particles_kernel_a370a66f_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::float32 var_inv_grid_nx,
    wp::float32 var_inv_grid_ny,
    wp::float32 var_inv_grid_nz,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_particle_grid_xyz,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_texture,
    wp::float32 adj_inv_grid_nx,
    wp::float32 adj_inv_grid_ny,
    wp::float32 adj_inv_grid_nz,
    wp::int32 adj_tex_nx,
    wp::int32 adj_tex_ny,
    wp::int32 adj_tex_nz,
    wp::int32 adj_src_x,
    wp::int32 adj_src_y,
    wp::int32 adj_src_z,
    wp::int32 adj_flip_x,
    wp::int32 adj_flip_y,
    wp::int32 adj_flip_z,
    wp::float32 adj_scale_x,
    wp::float32 adj_scale_y,
    wp::float32 adj_scale_z,
    wp::float32 adj_cut_z,
    wp::array_t<wp::int32> adj_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_colors)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        wp::int32* var_15;
        wp::float32 var_16;
        wp::int32 var_17;
        wp::float32 var_18;
        const wp::int32 var_19 = 1;
        wp::int32* var_20;
        wp::float32 var_21;
        wp::int32 var_22;
        wp::float32 var_23;
        const wp::int32 var_24 = 2;
        wp::int32* var_25;
        wp::float32 var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        wp::float32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        const wp::int32 var_40 = 1;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::int32 var_45;
        const wp::int32 var_46 = 0;
        const wp::int32 var_47 = 1;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::int32 var_52;
        const wp::int32 var_53 = 0;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        const wp::int32 var_57 = 0;
        bool var_58;
        const wp::int32 var_59 = 1;
        wp::int32 var_60;
        wp::int32 var_61;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        bool var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        bool var_70;
        const wp::int32 var_71 = 1;
        wp::int32 var_72;
        wp::int32 var_73;
        wp::int32 var_74;
        const wp::int32 var_75 = 0;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        wp::vec_t<3, wp::float32>* var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32>* var_80;
        wp::vec_t<3, wp::float32> var_81;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        bool adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::int32 adj_63 = {};
        bool adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        bool adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::vec_t<3, wp::float32> adj_78 = {};
        wp::vec_t<3, wp::float32> adj_79 = {};
        wp::vec_t<3, wp::float32> adj_80 = {};
        wp::vec_t<3, wp::float32> adj_81 = {};
        //---------
        // forward
        // def _gather_cryo_colored_particles_kernel(                                             <L 1598>
        // i = wp.tid()                                                                           <L 1630>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 1631>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 1632>
            goto label0;
        }
        // if particle_q[i][2] > cut_z:                                                           <L 1633>
        var_9 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = (var_11 > var_cut_z);
        if (var_13) {
            // return                                                                             <L 1634>
            goto label1;
        }
        // gx = float(particle_grid_xyz[i, 0]) * inv_grid_nx                                      <L 1635>
        var_15 = wp::address(var_particle_grid_xyz, var_0, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::float(var_17);
        var_18 = wp::mul(var_16, var_inv_grid_nx);
        // gy = float(particle_grid_xyz[i, 1]) * inv_grid_ny                                      <L 1636>
        var_20 = wp::address(var_particle_grid_xyz, var_0, var_19);
        var_22 = wp::load(var_20);
        var_21 = wp::float(var_22);
        var_23 = wp::mul(var_21, var_inv_grid_ny);
        // gz = float(particle_grid_xyz[i, 2]) * inv_grid_nz                                      <L 1637>
        var_25 = wp::address(var_particle_grid_xyz, var_0, var_24);
        var_27 = wp::load(var_25);
        var_26 = wp::float(var_27);
        var_28 = wp::mul(var_26, var_inv_grid_nz);
        // uv = wp.vec3(gx, gy, gz)                                                               <L 1638>
        var_29 = wp::vec_t<3, wp::float32>(var_18, var_23, var_28);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 1639>
        var_30 = _select_axis_0(var_29, var_src_x);
        var_31 = _scale_about_centre_0(var_30, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 1640>
        var_32 = _select_axis_0(var_29, var_src_y);
        var_33 = _scale_about_centre_0(var_32, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 1641>
        var_34 = _select_axis_0(var_29, var_src_z);
        var_35 = _scale_about_centre_0(var_34, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 1642>
        var_36 = wp::float(var_tex_nx);
        var_37 = wp::mul(var_31, var_36);
        var_38 = wp::int(var_37);
        var_41 = wp::sub(var_tex_nx, var_40);
        var_42 = wp::clamp(var_38, var_39, var_41);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 1643>
        var_43 = wp::float(var_tex_ny);
        var_44 = wp::mul(var_33, var_43);
        var_45 = wp::int(var_44);
        var_48 = wp::sub(var_tex_ny, var_47);
        var_49 = wp::clamp(var_45, var_46, var_48);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 1644>
        var_50 = wp::float(var_tex_nz);
        var_51 = wp::mul(var_35, var_50);
        var_52 = wp::int(var_51);
        var_55 = wp::sub(var_tex_nz, var_54);
        var_56 = wp::clamp(var_52, var_53, var_55);
        // if flip_x != 0:                                                                        <L 1645>
        var_58 = (var_flip_x != var_57);
        if (var_58) {
            // ix = (tex_nx - 1) - ix                                                             <L 1646>
            var_60 = wp::sub(var_tex_nx, var_59);
            var_61 = wp::sub(var_60, var_42);
        }
        var_62 = wp::where(var_58, var_61, var_42);
        // if flip_y != 0:                                                                        <L 1647>
        var_64 = (var_flip_y != var_63);
        if (var_64) {
            // iy = (tex_ny - 1) - iy                                                             <L 1648>
            var_66 = wp::sub(var_tex_ny, var_65);
            var_67 = wp::sub(var_66, var_49);
        }
        var_68 = wp::where(var_64, var_67, var_49);
        // if flip_z != 0:                                                                        <L 1649>
        var_70 = (var_flip_z != var_69);
        if (var_70) {
            // iz = (tex_nz - 1) - iz                                                             <L 1650>
            var_72 = wp::sub(var_tex_nz, var_71);
            var_73 = wp::sub(var_72, var_56);
        }
        var_74 = wp::where(var_70, var_73, var_56);
        // idx = wp.atomic_add(counter, 0, 1)                                                     <L 1651>
        // var_77 = wp::atomic_add(var_counter, var_75, var_76);
        // out_points[idx] = particle_q[i]                                                        <L 1652>
        var_78 = wp::address(var_particle_q, var_0);
        var_79 = wp::load(var_78);
        // wp::array_store(var_out_points, var_77, var_79);
        // out_colors[idx] = texture[ix, iy, iz]                                                  <L 1653>
        var_80 = wp::address(var_texture, var_62, var_68, var_74);
        var_81 = wp::load(var_80);
        // wp::array_store(var_out_colors, var_77, var_81);
        //---------
        // reverse
        wp::adj_array_store(var_out_colors, var_77, var_81, adj_out_colors, adj_77, adj_80);
        wp::adj_address(var_texture, var_62, var_68, var_74, adj_texture, adj_62, adj_68, adj_74, adj_80);
        // adj: out_colors[idx] = texture[ix, iy, iz]                                             <L 1653>
        wp::adj_array_store(var_out_points, var_77, var_79, adj_out_points, adj_77, adj_78);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_78);
        // adj: out_points[idx] = particle_q[i]                                                   <L 1652>
        wp::adj_atomic_add(var_counter, var_75, var_76, adj_counter, adj_75, adj_76, adj_77);
        // adj: idx = wp.atomic_add(counter, 0, 1)                                                <L 1651>
        wp::adj_where(var_70, var_73, var_56, adj_70, adj_73, adj_56, adj_74);
        if (var_70) {
            wp::adj_sub(var_72, var_56, adj_72, adj_56, adj_73);
            wp::adj_sub(var_tex_nz, var_71, adj_tex_nz, adj_71, adj_72);
            // adj: iz = (tex_nz - 1) - iz                                                        <L 1650>
        }
        // adj: if flip_z != 0:                                                                   <L 1649>
        wp::adj_where(var_64, var_67, var_49, adj_64, adj_67, adj_49, adj_68);
        if (var_64) {
            wp::adj_sub(var_66, var_49, adj_66, adj_49, adj_67);
            wp::adj_sub(var_tex_ny, var_65, adj_tex_ny, adj_65, adj_66);
            // adj: iy = (tex_ny - 1) - iy                                                        <L 1648>
        }
        // adj: if flip_y != 0:                                                                   <L 1647>
        wp::adj_where(var_58, var_61, var_42, adj_58, adj_61, adj_42, adj_62);
        if (var_58) {
            wp::adj_sub(var_60, var_42, adj_60, adj_42, adj_61);
            wp::adj_sub(var_tex_nx, var_59, adj_tex_nx, adj_59, adj_60);
            // adj: ix = (tex_nx - 1) - ix                                                        <L 1646>
        }
        // adj: if flip_x != 0:                                                                   <L 1645>
        wp::adj_clamp(var_52, var_53, var_55, adj_52, adj_53, adj_55, adj_56);
        wp::adj_sub(var_tex_nz, var_54, adj_tex_nz, adj_54, adj_55);
        wp::adj_int(var_51, adj_51, adj_52);
        wp::adj_mul(var_35, var_50, adj_35, adj_50, adj_51);
        wp::adj_float(var_tex_nz, adj_tex_nz, adj_50);
        // adj: iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                          <L 1644>
        wp::adj_clamp(var_45, var_46, var_48, adj_45, adj_46, adj_48, adj_49);
        wp::adj_sub(var_tex_ny, var_47, adj_tex_ny, adj_47, adj_48);
        wp::adj_int(var_44, adj_44, adj_45);
        wp::adj_mul(var_33, var_43, adj_33, adj_43, adj_44);
        wp::adj_float(var_tex_ny, adj_tex_ny, adj_43);
        // adj: iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                          <L 1643>
        wp::adj_clamp(var_38, var_39, var_41, adj_38, adj_39, adj_41, adj_42);
        wp::adj_sub(var_tex_nx, var_40, adj_tex_nx, adj_40, adj_41);
        wp::adj_int(var_37, adj_37, adj_38);
        wp::adj_mul(var_31, var_36, adj_31, adj_36, adj_37);
        wp::adj_float(var_tex_nx, adj_tex_nx, adj_36);
        // adj: ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                          <L 1642>
        adj__scale_about_centre_0(var_34, var_scale_z, adj_34, adj_scale_z, adj_35);
        adj__select_axis_0(var_29, var_src_z, adj_29, adj_src_z, adj_34);
        // adj: tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                     <L 1641>
        adj__scale_about_centre_0(var_32, var_scale_y, adj_32, adj_scale_y, adj_33);
        adj__select_axis_0(var_29, var_src_y, adj_29, adj_src_y, adj_32);
        // adj: tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                     <L 1640>
        adj__scale_about_centre_0(var_30, var_scale_x, adj_30, adj_scale_x, adj_31);
        adj__select_axis_0(var_29, var_src_x, adj_29, adj_src_x, adj_30);
        // adj: tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                     <L 1639>
        wp::adj_vec_t(var_18, var_23, var_28, adj_18, adj_23, adj_28, adj_29);
        // adj: uv = wp.vec3(gx, gy, gz)                                                          <L 1638>
        wp::adj_mul(var_26, var_inv_grid_nz, adj_26, adj_inv_grid_nz, adj_28);
        wp::adj_float(var_27, adj_25, adj_26);
        wp::adj_address(var_particle_grid_xyz, var_0, var_24, adj_particle_grid_xyz, adj_0, adj_24, adj_25);
        // adj: gz = float(particle_grid_xyz[i, 2]) * inv_grid_nz                                 <L 1637>
        wp::adj_mul(var_21, var_inv_grid_ny, adj_21, adj_inv_grid_ny, adj_23);
        wp::adj_float(var_22, adj_20, adj_21);
        wp::adj_address(var_particle_grid_xyz, var_0, var_19, adj_particle_grid_xyz, adj_0, adj_19, adj_20);
        // adj: gy = float(particle_grid_xyz[i, 1]) * inv_grid_ny                                 <L 1636>
        wp::adj_mul(var_16, var_inv_grid_nx, adj_16, adj_inv_grid_nx, adj_18);
        wp::adj_float(var_17, adj_15, adj_16);
        wp::adj_address(var_particle_grid_xyz, var_0, var_14, adj_particle_grid_xyz, adj_0, adj_14, adj_15);
        // adj: gx = float(particle_grid_xyz[i, 0]) * inv_grid_nx                                 <L 1635>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1634>
        }
        wp::adj_extract(var_12, var_10, adj_9, adj_10, adj_11);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_9);
        // adj: if particle_q[i][2] > cut_z:                                                      <L 1633>
        if (var_8) {
            label0:;
            // adj: return                                                                        <L 1632>
        }
        wp::adj_int32(var_3, adj_3, adj_4);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                     <L 1631>
        // adj: i = wp.tid()                                                                      <L 1630>
        // adj: def _gather_cryo_colored_particles_kernel(                                        <L 1598>
        continue;
    }
}



extern "C" __global__ void _bake_triangle_texture_kernel_28b457dc_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32>* var_22;
        const wp::int32 var_23 = 3;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 3;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::float32 var_38 = 1.0;
        const wp::float32 var_39 = 3.0;
        wp::float32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::float32 var_42 = 0.5;
        const wp::float32 var_43 = 0.5;
        wp::vec_t<2, wp::float32> var_44;
        wp::float32 var_45;
        const wp::float32 var_46 = 0.5;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.5;
        wp::vec_t<2, wp::float32> var_49;
        const wp::float32 var_50 = 0.5;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.5;
        wp::float32 var_53;
        wp::vec_t<2, wp::float32> var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.5;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 0.5;
        wp::float32 var_60;
        wp::vec_t<2, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32>* var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        const wp::int32 var_73 = 3;
        wp::int32 var_74;
        const wp::int32 var_75 = 1;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32>* var_77;
        const wp::int32 var_78 = 1;
        wp::float32 var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        const wp::int32 var_83 = 3;
        wp::int32 var_84;
        const wp::int32 var_85 = 2;
        wp::int32 var_86;
        wp::vec_t<3, wp::float32>* var_87;
        const wp::int32 var_88 = 2;
        wp::float32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        const wp::int32 var_104 = 1;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::int32 var_109;
        const wp::int32 var_110 = 0;
        const wp::int32 var_111 = 1;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::int32 var_116;
        const wp::int32 var_117 = 0;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        wp::int32 var_120;
        const wp::int32 var_121 = 0;
        bool var_122;
        const wp::int32 var_123 = 1;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 0;
        bool var_128;
        const wp::int32 var_129 = 1;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        const wp::int32 var_133 = 0;
        bool var_134;
        const wp::int32 var_135 = 1;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        wp::vec_t<3, wp::float32>* var_139;
        wp::vec_t<3, wp::float32> var_140;
        //---------
        // forward
        // def _bake_triangle_texture_kernel(                                                     <L 1481>
        // tx, ty, sub = wp.tid()                                                                 <L 1503>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1504>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1505>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1506>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1507>
            continue;
        }
        // px = sub % tile_size                                                                   <L 1509>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1510>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1511>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1512>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1513>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1514>
            continue;
        }
        // if tile_size <= 1:                                                                     <L 1516>
        var_17 = (var_tile_size <= var_16);
        if (var_17) {
            // uv = (                                                                             <L 1517>
            // flat_vertex_uv3[tid * 3 + 0]                                                       <L 1518>
            var_19 = wp::mul(var_5, var_18);
            var_21 = wp::add(var_19, var_20);
            var_22 = wp::address(var_flat_vertex_uv3, var_21);
            // + flat_vertex_uv3[tid * 3 + 1]                                                     <L 1519>
            var_24 = wp::mul(var_5, var_23);
            var_26 = wp::add(var_24, var_25);
            var_27 = wp::address(var_flat_vertex_uv3, var_26);
            var_29 = wp::load(var_22);
            var_30 = wp::load(var_27);
            var_28 = wp::add(var_29, var_30);
            // + flat_vertex_uv3[tid * 3 + 2]                                                     <L 1520>
            var_32 = wp::mul(var_5, var_31);
            var_34 = wp::add(var_32, var_33);
            var_35 = wp::address(var_flat_vertex_uv3, var_34);
            var_37 = wp::load(var_35);
            var_36 = wp::add(var_28, var_37);
            // ) * (1.0 / 3.0)                                                                    <L 1521>
            var_40 = wp::div(var_38, var_39);
            var_41 = wp::mul(var_36, var_40);
        }
        if (!var_17) {
            // a = wp.vec2(0.5, 0.5)                                                              <L 1523>
            var_44 = wp::vec_t<2, wp::float32>(var_42, var_43);
            // b = wp.vec2(float(tile_size) - 0.5, 0.5)                                           <L 1524>
            var_45 = wp::float(var_tile_size);
            var_47 = wp::sub(var_45, var_46);
            var_49 = wp::vec_t<2, wp::float32>(var_47, var_48);
            // c = wp.vec2(0.5, float(tile_size) - 0.5)                                           <L 1525>
            var_51 = wp::float(var_tile_size);
            var_53 = wp::sub(var_51, var_52);
            var_54 = wp::vec_t<2, wp::float32>(var_50, var_53);
            // p = wp.vec2(float(px) + 0.5, float(py) + 0.5)                                      <L 1526>
            var_55 = wp::float(var_7);
            var_57 = wp::add(var_55, var_56);
            var_58 = wp::float(var_8);
            var_60 = wp::add(var_58, var_59);
            var_61 = wp::vec_t<2, wp::float32>(var_57, var_60);
            // bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))                       <L 1527>
            var_62 = _triangle_barycentric_0(var_61, var_44, var_49, var_54);
            var_63 = _clamp_barycentric_0(var_62);
            // uv = (                                                                             <L 1528>
            // flat_vertex_uv3[tid * 3 + 0] * bary[0]                                             <L 1529>
            var_65 = wp::mul(var_5, var_64);
            var_67 = wp::add(var_65, var_66);
            var_68 = wp::address(var_flat_vertex_uv3, var_67);
            var_70 = wp::extract(var_63, var_69);
            var_72 = wp::load(var_68);
            var_71 = wp::mul(var_72, var_70);
            // + flat_vertex_uv3[tid * 3 + 1] * bary[1]                                           <L 1530>
            var_74 = wp::mul(var_5, var_73);
            var_76 = wp::add(var_74, var_75);
            var_77 = wp::address(var_flat_vertex_uv3, var_76);
            var_79 = wp::extract(var_63, var_78);
            var_81 = wp::load(var_77);
            var_80 = wp::mul(var_81, var_79);
            var_82 = wp::add(var_71, var_80);
            // + flat_vertex_uv3[tid * 3 + 2] * bary[2]                                           <L 1531>
            var_84 = wp::mul(var_5, var_83);
            var_86 = wp::add(var_84, var_85);
            var_87 = wp::address(var_flat_vertex_uv3, var_86);
            var_89 = wp::extract(var_63, var_88);
            var_91 = wp::load(var_87);
            var_90 = wp::mul(var_91, var_89);
            var_92 = wp::add(var_82, var_90);
        }
        var_93 = wp::where(var_17, var_41, var_92);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 1534>
        var_94 = _select_axis_0(var_93, var_src_x);
        var_95 = _scale_about_centre_0(var_94, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 1535>
        var_96 = _select_axis_0(var_93, var_src_y);
        var_97 = _scale_about_centre_0(var_96, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 1536>
        var_98 = _select_axis_0(var_93, var_src_z);
        var_99 = _scale_about_centre_0(var_98, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 1537>
        var_100 = wp::float(var_tex_nx);
        var_101 = wp::mul(var_95, var_100);
        var_102 = wp::int(var_101);
        var_105 = wp::sub(var_tex_nx, var_104);
        var_106 = wp::clamp(var_102, var_103, var_105);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 1538>
        var_107 = wp::float(var_tex_ny);
        var_108 = wp::mul(var_97, var_107);
        var_109 = wp::int(var_108);
        var_112 = wp::sub(var_tex_ny, var_111);
        var_113 = wp::clamp(var_109, var_110, var_112);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 1539>
        var_114 = wp::float(var_tex_nz);
        var_115 = wp::mul(var_99, var_114);
        var_116 = wp::int(var_115);
        var_119 = wp::sub(var_tex_nz, var_118);
        var_120 = wp::clamp(var_116, var_117, var_119);
        // if flip_x != 0:                                                                        <L 1540>
        var_122 = (var_flip_x != var_121);
        if (var_122) {
            // ix = (tex_nx - 1) - ix                                                             <L 1541>
            var_124 = wp::sub(var_tex_nx, var_123);
            var_125 = wp::sub(var_124, var_106);
        }
        var_126 = wp::where(var_122, var_125, var_106);
        // if flip_y != 0:                                                                        <L 1542>
        var_128 = (var_flip_y != var_127);
        if (var_128) {
            // iy = (tex_ny - 1) - iy                                                             <L 1543>
            var_130 = wp::sub(var_tex_ny, var_129);
            var_131 = wp::sub(var_130, var_113);
        }
        var_132 = wp::where(var_128, var_131, var_113);
        // if flip_z != 0:                                                                        <L 1544>
        var_134 = (var_flip_z != var_133);
        if (var_134) {
            // iz = (tex_nz - 1) - iz                                                             <L 1545>
            var_136 = wp::sub(var_tex_nz, var_135);
            var_137 = wp::sub(var_136, var_120);
        }
        var_138 = wp::where(var_134, var_137, var_120);
        // atlas_rgb[y, x] = texture[ix, iy, iz]                                                  <L 1546>
        var_139 = wp::address(var_texture, var_126, var_132, var_138);
        var_140 = wp::load(var_139);
        wp::array_store(var_atlas_rgb, var_12, var_10, var_140);
    }
}



extern "C" __global__ void _bake_triangle_texture_kernel_28b457dc_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_texture,
    wp::int32 adj_tex_nx,
    wp::int32 adj_tex_ny,
    wp::int32 adj_tex_nz,
    wp::int32 adj_src_x,
    wp::int32 adj_src_y,
    wp::int32 adj_src_z,
    wp::int32 adj_flip_x,
    wp::int32 adj_flip_y,
    wp::int32 adj_flip_z,
    wp::float32 adj_scale_x,
    wp::float32 adj_scale_y,
    wp::float32 adj_scale_z,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        const wp::int32 var_16 = 1;
        bool var_17;
        const wp::int32 var_18 = 3;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32>* var_22;
        const wp::int32 var_23 = 3;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32>* var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 3;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32>* var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::float32 var_38 = 1.0;
        const wp::float32 var_39 = 3.0;
        wp::float32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::float32 var_42 = 0.5;
        const wp::float32 var_43 = 0.5;
        wp::vec_t<2, wp::float32> var_44;
        wp::float32 var_45;
        const wp::float32 var_46 = 0.5;
        wp::float32 var_47;
        const wp::float32 var_48 = 0.5;
        wp::vec_t<2, wp::float32> var_49;
        const wp::float32 var_50 = 0.5;
        wp::float32 var_51;
        const wp::float32 var_52 = 0.5;
        wp::float32 var_53;
        wp::vec_t<2, wp::float32> var_54;
        wp::float32 var_55;
        const wp::float32 var_56 = 0.5;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 0.5;
        wp::float32 var_60;
        wp::vec_t<2, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 3;
        wp::int32 var_65;
        const wp::int32 var_66 = 0;
        wp::int32 var_67;
        wp::vec_t<3, wp::float32>* var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        const wp::int32 var_73 = 3;
        wp::int32 var_74;
        const wp::int32 var_75 = 1;
        wp::int32 var_76;
        wp::vec_t<3, wp::float32>* var_77;
        const wp::int32 var_78 = 1;
        wp::float32 var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        const wp::int32 var_83 = 3;
        wp::int32 var_84;
        const wp::int32 var_85 = 2;
        wp::int32 var_86;
        wp::vec_t<3, wp::float32>* var_87;
        const wp::int32 var_88 = 2;
        wp::float32 var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        wp::float32 var_94;
        wp::float32 var_95;
        wp::float32 var_96;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        const wp::int32 var_104 = 1;
        wp::int32 var_105;
        wp::int32 var_106;
        wp::float32 var_107;
        wp::float32 var_108;
        wp::int32 var_109;
        const wp::int32 var_110 = 0;
        const wp::int32 var_111 = 1;
        wp::int32 var_112;
        wp::int32 var_113;
        wp::float32 var_114;
        wp::float32 var_115;
        wp::int32 var_116;
        const wp::int32 var_117 = 0;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        wp::int32 var_120;
        const wp::int32 var_121 = 0;
        bool var_122;
        const wp::int32 var_123 = 1;
        wp::int32 var_124;
        wp::int32 var_125;
        wp::int32 var_126;
        const wp::int32 var_127 = 0;
        bool var_128;
        const wp::int32 var_129 = 1;
        wp::int32 var_130;
        wp::int32 var_131;
        wp::int32 var_132;
        const wp::int32 var_133 = 0;
        bool var_134;
        const wp::int32 var_135 = 1;
        wp::int32 var_136;
        wp::int32 var_137;
        wp::int32 var_138;
        wp::vec_t<3, wp::float32>* var_139;
        wp::vec_t<3, wp::float32> var_140;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
        wp::int32 adj_16 = {};
        bool adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::float32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::vec_t<2, wp::float32> adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::float32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::vec_t<2, wp::float32> adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::vec_t<2, wp::float32> adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::vec_t<2, wp::float32> adj_61 = {};
        wp::vec_t<3, wp::float32> adj_62 = {};
        wp::vec_t<3, wp::float32> adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::vec_t<3, wp::float32> adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::vec_t<3, wp::float32> adj_71 = {};
        wp::vec_t<3, wp::float32> adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::vec_t<3, wp::float32> adj_77 = {};
        wp::int32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::vec_t<3, wp::float32> adj_80 = {};
        wp::vec_t<3, wp::float32> adj_81 = {};
        wp::vec_t<3, wp::float32> adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::vec_t<3, wp::float32> adj_87 = {};
        wp::int32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::vec_t<3, wp::float32> adj_90 = {};
        wp::vec_t<3, wp::float32> adj_91 = {};
        wp::vec_t<3, wp::float32> adj_92 = {};
        wp::vec_t<3, wp::float32> adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::float32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::float32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::int32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::int32 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::int32 adj_113 = {};
        wp::float32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::int32 adj_120 = {};
        wp::int32 adj_121 = {};
        bool adj_122 = {};
        wp::int32 adj_123 = {};
        wp::int32 adj_124 = {};
        wp::int32 adj_125 = {};
        wp::int32 adj_126 = {};
        wp::int32 adj_127 = {};
        bool adj_128 = {};
        wp::int32 adj_129 = {};
        wp::int32 adj_130 = {};
        wp::int32 adj_131 = {};
        wp::int32 adj_132 = {};
        wp::int32 adj_133 = {};
        bool adj_134 = {};
        wp::int32 adj_135 = {};
        wp::int32 adj_136 = {};
        wp::int32 adj_137 = {};
        wp::int32 adj_138 = {};
        wp::vec_t<3, wp::float32> adj_139 = {};
        wp::vec_t<3, wp::float32> adj_140 = {};
        //---------
        // forward
        // def _bake_triangle_texture_kernel(                                                     <L 1481>
        // tx, ty, sub = wp.tid()                                                                 <L 1503>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1504>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1505>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1506>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1507>
            goto label0;
        }
        // px = sub % tile_size                                                                   <L 1509>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1510>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1511>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1512>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1513>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1514>
            goto label1;
        }
        // if tile_size <= 1:                                                                     <L 1516>
        var_17 = (var_tile_size <= var_16);
        if (var_17) {
            // uv = (                                                                             <L 1517>
            // flat_vertex_uv3[tid * 3 + 0]                                                       <L 1518>
            var_19 = wp::mul(var_5, var_18);
            var_21 = wp::add(var_19, var_20);
            var_22 = wp::address(var_flat_vertex_uv3, var_21);
            // + flat_vertex_uv3[tid * 3 + 1]                                                     <L 1519>
            var_24 = wp::mul(var_5, var_23);
            var_26 = wp::add(var_24, var_25);
            var_27 = wp::address(var_flat_vertex_uv3, var_26);
            var_29 = wp::load(var_22);
            var_30 = wp::load(var_27);
            var_28 = wp::add(var_29, var_30);
            // + flat_vertex_uv3[tid * 3 + 2]                                                     <L 1520>
            var_32 = wp::mul(var_5, var_31);
            var_34 = wp::add(var_32, var_33);
            var_35 = wp::address(var_flat_vertex_uv3, var_34);
            var_37 = wp::load(var_35);
            var_36 = wp::add(var_28, var_37);
            // ) * (1.0 / 3.0)                                                                    <L 1521>
            var_40 = wp::div(var_38, var_39);
            var_41 = wp::mul(var_36, var_40);
        }
        if (!var_17) {
            // a = wp.vec2(0.5, 0.5)                                                              <L 1523>
            var_44 = wp::vec_t<2, wp::float32>(var_42, var_43);
            // b = wp.vec2(float(tile_size) - 0.5, 0.5)                                           <L 1524>
            var_45 = wp::float(var_tile_size);
            var_47 = wp::sub(var_45, var_46);
            var_49 = wp::vec_t<2, wp::float32>(var_47, var_48);
            // c = wp.vec2(0.5, float(tile_size) - 0.5)                                           <L 1525>
            var_51 = wp::float(var_tile_size);
            var_53 = wp::sub(var_51, var_52);
            var_54 = wp::vec_t<2, wp::float32>(var_50, var_53);
            // p = wp.vec2(float(px) + 0.5, float(py) + 0.5)                                      <L 1526>
            var_55 = wp::float(var_7);
            var_57 = wp::add(var_55, var_56);
            var_58 = wp::float(var_8);
            var_60 = wp::add(var_58, var_59);
            var_61 = wp::vec_t<2, wp::float32>(var_57, var_60);
            // bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))                       <L 1527>
            var_62 = _triangle_barycentric_0(var_61, var_44, var_49, var_54);
            var_63 = _clamp_barycentric_0(var_62);
            // uv = (                                                                             <L 1528>
            // flat_vertex_uv3[tid * 3 + 0] * bary[0]                                             <L 1529>
            var_65 = wp::mul(var_5, var_64);
            var_67 = wp::add(var_65, var_66);
            var_68 = wp::address(var_flat_vertex_uv3, var_67);
            var_70 = wp::extract(var_63, var_69);
            var_72 = wp::load(var_68);
            var_71 = wp::mul(var_72, var_70);
            // + flat_vertex_uv3[tid * 3 + 1] * bary[1]                                           <L 1530>
            var_74 = wp::mul(var_5, var_73);
            var_76 = wp::add(var_74, var_75);
            var_77 = wp::address(var_flat_vertex_uv3, var_76);
            var_79 = wp::extract(var_63, var_78);
            var_81 = wp::load(var_77);
            var_80 = wp::mul(var_81, var_79);
            var_82 = wp::add(var_71, var_80);
            // + flat_vertex_uv3[tid * 3 + 2] * bary[2]                                           <L 1531>
            var_84 = wp::mul(var_5, var_83);
            var_86 = wp::add(var_84, var_85);
            var_87 = wp::address(var_flat_vertex_uv3, var_86);
            var_89 = wp::extract(var_63, var_88);
            var_91 = wp::load(var_87);
            var_90 = wp::mul(var_91, var_89);
            var_92 = wp::add(var_82, var_90);
        }
        var_93 = wp::where(var_17, var_41, var_92);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 1534>
        var_94 = _select_axis_0(var_93, var_src_x);
        var_95 = _scale_about_centre_0(var_94, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 1535>
        var_96 = _select_axis_0(var_93, var_src_y);
        var_97 = _scale_about_centre_0(var_96, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 1536>
        var_98 = _select_axis_0(var_93, var_src_z);
        var_99 = _scale_about_centre_0(var_98, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 1537>
        var_100 = wp::float(var_tex_nx);
        var_101 = wp::mul(var_95, var_100);
        var_102 = wp::int(var_101);
        var_105 = wp::sub(var_tex_nx, var_104);
        var_106 = wp::clamp(var_102, var_103, var_105);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 1538>
        var_107 = wp::float(var_tex_ny);
        var_108 = wp::mul(var_97, var_107);
        var_109 = wp::int(var_108);
        var_112 = wp::sub(var_tex_ny, var_111);
        var_113 = wp::clamp(var_109, var_110, var_112);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 1539>
        var_114 = wp::float(var_tex_nz);
        var_115 = wp::mul(var_99, var_114);
        var_116 = wp::int(var_115);
        var_119 = wp::sub(var_tex_nz, var_118);
        var_120 = wp::clamp(var_116, var_117, var_119);
        // if flip_x != 0:                                                                        <L 1540>
        var_122 = (var_flip_x != var_121);
        if (var_122) {
            // ix = (tex_nx - 1) - ix                                                             <L 1541>
            var_124 = wp::sub(var_tex_nx, var_123);
            var_125 = wp::sub(var_124, var_106);
        }
        var_126 = wp::where(var_122, var_125, var_106);
        // if flip_y != 0:                                                                        <L 1542>
        var_128 = (var_flip_y != var_127);
        if (var_128) {
            // iy = (tex_ny - 1) - iy                                                             <L 1543>
            var_130 = wp::sub(var_tex_ny, var_129);
            var_131 = wp::sub(var_130, var_113);
        }
        var_132 = wp::where(var_128, var_131, var_113);
        // if flip_z != 0:                                                                        <L 1544>
        var_134 = (var_flip_z != var_133);
        if (var_134) {
            // iz = (tex_nz - 1) - iz                                                             <L 1545>
            var_136 = wp::sub(var_tex_nz, var_135);
            var_137 = wp::sub(var_136, var_120);
        }
        var_138 = wp::where(var_134, var_137, var_120);
        // atlas_rgb[y, x] = texture[ix, iy, iz]                                                  <L 1546>
        var_139 = wp::address(var_texture, var_126, var_132, var_138);
        var_140 = wp::load(var_139);
        // wp::array_store(var_atlas_rgb, var_12, var_10, var_140);
        //---------
        // reverse
        wp::adj_array_store(var_atlas_rgb, var_12, var_10, var_140, adj_atlas_rgb, adj_12, adj_10, adj_139);
        wp::adj_address(var_texture, var_126, var_132, var_138, adj_texture, adj_126, adj_132, adj_138, adj_139);
        // adj: atlas_rgb[y, x] = texture[ix, iy, iz]                                             <L 1546>
        wp::adj_where(var_134, var_137, var_120, adj_134, adj_137, adj_120, adj_138);
        if (var_134) {
            wp::adj_sub(var_136, var_120, adj_136, adj_120, adj_137);
            wp::adj_sub(var_tex_nz, var_135, adj_tex_nz, adj_135, adj_136);
            // adj: iz = (tex_nz - 1) - iz                                                        <L 1545>
        }
        // adj: if flip_z != 0:                                                                   <L 1544>
        wp::adj_where(var_128, var_131, var_113, adj_128, adj_131, adj_113, adj_132);
        if (var_128) {
            wp::adj_sub(var_130, var_113, adj_130, adj_113, adj_131);
            wp::adj_sub(var_tex_ny, var_129, adj_tex_ny, adj_129, adj_130);
            // adj: iy = (tex_ny - 1) - iy                                                        <L 1543>
        }
        // adj: if flip_y != 0:                                                                   <L 1542>
        wp::adj_where(var_122, var_125, var_106, adj_122, adj_125, adj_106, adj_126);
        if (var_122) {
            wp::adj_sub(var_124, var_106, adj_124, adj_106, adj_125);
            wp::adj_sub(var_tex_nx, var_123, adj_tex_nx, adj_123, adj_124);
            // adj: ix = (tex_nx - 1) - ix                                                        <L 1541>
        }
        // adj: if flip_x != 0:                                                                   <L 1540>
        wp::adj_clamp(var_116, var_117, var_119, adj_116, adj_117, adj_119, adj_120);
        wp::adj_sub(var_tex_nz, var_118, adj_tex_nz, adj_118, adj_119);
        wp::adj_int(var_115, adj_115, adj_116);
        wp::adj_mul(var_99, var_114, adj_99, adj_114, adj_115);
        wp::adj_float(var_tex_nz, adj_tex_nz, adj_114);
        // adj: iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                          <L 1539>
        wp::adj_clamp(var_109, var_110, var_112, adj_109, adj_110, adj_112, adj_113);
        wp::adj_sub(var_tex_ny, var_111, adj_tex_ny, adj_111, adj_112);
        wp::adj_int(var_108, adj_108, adj_109);
        wp::adj_mul(var_97, var_107, adj_97, adj_107, adj_108);
        wp::adj_float(var_tex_ny, adj_tex_ny, adj_107);
        // adj: iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                          <L 1538>
        wp::adj_clamp(var_102, var_103, var_105, adj_102, adj_103, adj_105, adj_106);
        wp::adj_sub(var_tex_nx, var_104, adj_tex_nx, adj_104, adj_105);
        wp::adj_int(var_101, adj_101, adj_102);
        wp::adj_mul(var_95, var_100, adj_95, adj_100, adj_101);
        wp::adj_float(var_tex_nx, adj_tex_nx, adj_100);
        // adj: ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                          <L 1537>
        adj__scale_about_centre_0(var_98, var_scale_z, adj_98, adj_scale_z, adj_99);
        adj__select_axis_0(var_93, var_src_z, adj_93, adj_src_z, adj_98);
        // adj: tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                     <L 1536>
        adj__scale_about_centre_0(var_96, var_scale_y, adj_96, adj_scale_y, adj_97);
        adj__select_axis_0(var_93, var_src_y, adj_93, adj_src_y, adj_96);
        // adj: tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                     <L 1535>
        adj__scale_about_centre_0(var_94, var_scale_x, adj_94, adj_scale_x, adj_95);
        adj__select_axis_0(var_93, var_src_x, adj_93, adj_src_x, adj_94);
        // adj: tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                     <L 1534>
        wp::adj_where(var_17, var_41, var_92, adj_17, adj_41, adj_92, adj_93);
        if (!var_17) {
            wp::adj_add(var_82, var_90, adj_82, adj_90, adj_92);
            wp::adj_mul(var_91, var_89, adj_87, adj_89, adj_90);
            wp::adj_extract(var_63, var_88, adj_63, adj_88, adj_89);
            wp::adj_address(var_flat_vertex_uv3, var_86, adj_flat_vertex_uv3, adj_86, adj_87);
            wp::adj_add(var_84, var_85, adj_84, adj_85, adj_86);
            wp::adj_mul(var_5, var_83, adj_5, adj_83, adj_84);
            // adj: + flat_vertex_uv3[tid * 3 + 2] * bary[2]                                      <L 1531>
            wp::adj_add(var_71, var_80, adj_71, adj_80, adj_82);
            wp::adj_mul(var_81, var_79, adj_77, adj_79, adj_80);
            wp::adj_extract(var_63, var_78, adj_63, adj_78, adj_79);
            wp::adj_address(var_flat_vertex_uv3, var_76, adj_flat_vertex_uv3, adj_76, adj_77);
            wp::adj_add(var_74, var_75, adj_74, adj_75, adj_76);
            wp::adj_mul(var_5, var_73, adj_5, adj_73, adj_74);
            // adj: + flat_vertex_uv3[tid * 3 + 1] * bary[1]                                      <L 1530>
            wp::adj_mul(var_72, var_70, adj_68, adj_70, adj_71);
            wp::adj_extract(var_63, var_69, adj_63, adj_69, adj_70);
            wp::adj_address(var_flat_vertex_uv3, var_67, adj_flat_vertex_uv3, adj_67, adj_68);
            wp::adj_add(var_65, var_66, adj_65, adj_66, adj_67);
            wp::adj_mul(var_5, var_64, adj_5, adj_64, adj_65);
            // adj: flat_vertex_uv3[tid * 3 + 0] * bary[0]                                        <L 1529>
            // adj: uv = (                                                                        <L 1528>
            adj__clamp_barycentric_0(var_62, adj_62, adj_63);
            adj__triangle_barycentric_0(var_61, var_44, var_49, var_54, adj_61, adj_44, adj_49, adj_54, adj_62);
            // adj: bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))                  <L 1527>
            wp::adj_vec_t(var_57, var_60, adj_57, adj_60, adj_61);
            wp::adj_add(var_58, var_59, adj_58, adj_59, adj_60);
            wp::adj_float(var_8, adj_8, adj_58);
            wp::adj_add(var_55, var_56, adj_55, adj_56, adj_57);
            wp::adj_float(var_7, adj_7, adj_55);
            // adj: p = wp.vec2(float(px) + 0.5, float(py) + 0.5)                                 <L 1526>
            wp::adj_vec_t(var_50, var_53, adj_50, adj_53, adj_54);
            wp::adj_sub(var_51, var_52, adj_51, adj_52, adj_53);
            wp::adj_float(var_tile_size, adj_tile_size, adj_51);
            // adj: c = wp.vec2(0.5, float(tile_size) - 0.5)                                      <L 1525>
            wp::adj_vec_t(var_47, var_48, adj_47, adj_48, adj_49);
            wp::adj_sub(var_45, var_46, adj_45, adj_46, adj_47);
            wp::adj_float(var_tile_size, adj_tile_size, adj_45);
            // adj: b = wp.vec2(float(tile_size) - 0.5, 0.5)                                      <L 1524>
            wp::adj_vec_t(var_42, var_43, adj_42, adj_43, adj_44);
            // adj: a = wp.vec2(0.5, 0.5)                                                         <L 1523>
        }
        if (var_17) {
            wp::adj_mul(var_36, var_40, adj_36, adj_40, adj_41);
            wp::adj_div(var_38, var_39, var_40, adj_38, adj_39, adj_40);
            // adj: ) * (1.0 / 3.0)                                                               <L 1521>
            wp::adj_add(var_28, var_37, adj_28, adj_35, adj_36);
            wp::adj_address(var_flat_vertex_uv3, var_34, adj_flat_vertex_uv3, adj_34, adj_35);
            wp::adj_add(var_32, var_33, adj_32, adj_33, adj_34);
            wp::adj_mul(var_5, var_31, adj_5, adj_31, adj_32);
            // adj: + flat_vertex_uv3[tid * 3 + 2]                                                <L 1520>
            wp::adj_add(var_29, var_30, adj_22, adj_27, adj_28);
            wp::adj_address(var_flat_vertex_uv3, var_26, adj_flat_vertex_uv3, adj_26, adj_27);
            wp::adj_add(var_24, var_25, adj_24, adj_25, adj_26);
            wp::adj_mul(var_5, var_23, adj_5, adj_23, adj_24);
            // adj: + flat_vertex_uv3[tid * 3 + 1]                                                <L 1519>
            wp::adj_address(var_flat_vertex_uv3, var_21, adj_flat_vertex_uv3, adj_21, adj_22);
            wp::adj_add(var_19, var_20, adj_19, adj_20, adj_21);
            wp::adj_mul(var_5, var_18, adj_5, adj_18, adj_19);
            // adj: flat_vertex_uv3[tid * 3 + 0]                                                  <L 1518>
            // adj: uv = (                                                                        <L 1517>
        }
        // adj: if tile_size <= 1:                                                                <L 1516>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1514>
        }
        if (!var_13) {
        }
        // adj: if x >= atlas_width or y >= atlas_height:                                         <L 1513>
        wp::adj_add(var_11, var_8, adj_11, adj_8, adj_12);
        wp::adj_mul(var_1, var_tile_size, adj_1, adj_tile_size, adj_11);
        // adj: y = ty * tile_size + py                                                           <L 1512>
        wp::adj_add(var_9, var_7, adj_9, adj_7, adj_10);
        wp::adj_mul(var_0, var_tile_size, adj_0, adj_tile_size, adj_9);
        // adj: x = tx * tile_size + px                                                           <L 1511>
        wp::adj_div(var_2, var_tile_size, var_8, adj_2, adj_tile_size, adj_8);
        // adj: py = sub / tile_size                                                              <L 1510>
        wp::adj_mod(var_2, var_tile_size, adj_2, adj_tile_size, adj_7);
        // adj: px = sub % tile_size                                                              <L 1509>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1507>
        }
        // adj: if tid >= num_triangles:                                                          <L 1506>
        wp::adj_add(var_4, var_0, adj_4, adj_0, adj_5);
        wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
        // adj: tid = ty * tiles_per_row + tx                                                     <L 1505>
        wp::adj_div(var_atlas_width, var_tile_size, var_3, adj_atlas_width, adj_tile_size, adj_3);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1504>
        // adj: tx, ty, sub = wp.tid()                                                            <L 1503>
        // adj: def _bake_triangle_texture_kernel(                                                <L 1481>
        continue;
    }
}



extern "C" __global__ void _gather_active_particles_kernel_955344fd_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::vec_t<3, wp::float32>> var_material_colors,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::int32* var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        //---------
        // forward
        // def _gather_active_particles_kernel(                                                   <L 1550>
        // i = wp.tid()                                                                           <L 1565>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 1566>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 1567>
            continue;
        }
        // if particle_q[i][2] > cut_z:                                                           <L 1568>
        var_9 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = (var_11 > var_cut_z);
        if (var_13) {
            // return                                                                             <L 1569>
            continue;
        }
        // idx = wp.atomic_add(counter, 0, 1)                                                     <L 1570>
        var_16 = wp::atomic_add(var_counter, var_14, var_15);
        // out_points[idx] = particle_q[i]                                                        <L 1571>
        var_17 = wp::address(var_particle_q, var_0);
        var_18 = wp::load(var_17);
        wp::array_store(var_out_points, var_16, var_18);
        // out_colors[idx] = material_colors[particle_material[i]]                                <L 1572>
        var_19 = wp::address(var_particle_material, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::address(var_material_colors, var_21);
        var_22 = wp::load(var_20);
        wp::array_store(var_out_colors, var_16, var_22);
    }
}



extern "C" __global__ void _gather_active_particles_kernel_955344fd_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::vec_t<3, wp::float32>> var_material_colors,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_particle_material,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_material_colors,
    wp::float32 adj_cut_z,
    wp::array_t<wp::int32> adj_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_colors)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::int32* var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::int32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::int32 adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        //---------
        // forward
        // def _gather_active_particles_kernel(                                                   <L 1550>
        // i = wp.tid()                                                                           <L 1565>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 1566>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 1567>
            goto label0;
        }
        // if particle_q[i][2] > cut_z:                                                           <L 1568>
        var_9 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = (var_11 > var_cut_z);
        if (var_13) {
            // return                                                                             <L 1569>
            goto label1;
        }
        // idx = wp.atomic_add(counter, 0, 1)                                                     <L 1570>
        // var_16 = wp::atomic_add(var_counter, var_14, var_15);
        // out_points[idx] = particle_q[i]                                                        <L 1571>
        var_17 = wp::address(var_particle_q, var_0);
        var_18 = wp::load(var_17);
        // wp::array_store(var_out_points, var_16, var_18);
        // out_colors[idx] = material_colors[particle_material[i]]                                <L 1572>
        var_19 = wp::address(var_particle_material, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::address(var_material_colors, var_21);
        var_22 = wp::load(var_20);
        // wp::array_store(var_out_colors, var_16, var_22);
        //---------
        // reverse
        wp::adj_array_store(var_out_colors, var_16, var_22, adj_out_colors, adj_16, adj_20);
        wp::adj_address(var_material_colors, var_21, adj_material_colors, adj_19, adj_20);
        wp::adj_address(var_particle_material, var_0, adj_particle_material, adj_0, adj_19);
        // adj: out_colors[idx] = material_colors[particle_material[i]]                           <L 1572>
        wp::adj_array_store(var_out_points, var_16, var_18, adj_out_points, adj_16, adj_17);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_17);
        // adj: out_points[idx] = particle_q[i]                                                   <L 1571>
        wp::adj_atomic_add(var_counter, var_14, var_15, adj_counter, adj_14, adj_15, adj_16);
        // adj: idx = wp.atomic_add(counter, 0, 1)                                                <L 1570>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1569>
        }
        wp::adj_extract(var_12, var_10, adj_9, adj_10, adj_11);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_9);
        // adj: if particle_q[i][2] > cut_z:                                                      <L 1568>
        if (var_8) {
            label0:;
            // adj: return                                                                        <L 1567>
        }
        wp::adj_int32(var_3, adj_3, adj_4);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                     <L 1566>
        // adj: i = wp.tid()                                                                      <L 1565>
        // adj: def _gather_active_particles_kernel(                                              <L 1550>
        continue;
    }
}



extern "C" __global__ void _expand_triangle_vertices_with_procedural_coord_kernel_0963c69a_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_procedural_coord,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_procedural_coord)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        const wp::int32 var_37 = 2;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32>* var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        const wp::int32 var_49 = 2;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32> var_51;
        //---------
        // forward
        // def _expand_triangle_vertices_with_procedural_coord_kernel(                            <L 959>
        // t = wp.tid()                                                                           <L 970>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 971>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 972>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 973>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 974>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 975>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 976>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 977>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 978>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 979>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        wp::array_store(var_flat_pos, var_26, var_27);
        // flat_uv3[base + 0] = vertex_uv3[v0]                                                    <L 980>
        var_28 = wp::address(var_vertex_uv3, var_4);
        var_30 = wp::add(var_15, var_29);
        var_31 = wp::load(var_28);
        wp::array_store(var_flat_uv3, var_30, var_31);
        // flat_uv3[base + 1] = vertex_uv3[v1]                                                    <L 981>
        var_32 = wp::address(var_vertex_uv3, var_8);
        var_34 = wp::add(var_15, var_33);
        var_35 = wp::load(var_32);
        wp::array_store(var_flat_uv3, var_34, var_35);
        // flat_uv3[base + 2] = vertex_uv3[v2]                                                    <L 982>
        var_36 = wp::address(var_vertex_uv3, var_12);
        var_38 = wp::add(var_15, var_37);
        var_39 = wp::load(var_36);
        wp::array_store(var_flat_uv3, var_38, var_39);
        // flat_procedural_coord[base + 0] = procedural_coord[v0]                                 <L 983>
        var_40 = wp::address(var_procedural_coord, var_4);
        var_42 = wp::add(var_15, var_41);
        var_43 = wp::load(var_40);
        wp::array_store(var_flat_procedural_coord, var_42, var_43);
        // flat_procedural_coord[base + 1] = procedural_coord[v1]                                 <L 984>
        var_44 = wp::address(var_procedural_coord, var_8);
        var_46 = wp::add(var_15, var_45);
        var_47 = wp::load(var_44);
        wp::array_store(var_flat_procedural_coord, var_46, var_47);
        // flat_procedural_coord[base + 2] = procedural_coord[v2]                                 <L 985>
        var_48 = wp::address(var_procedural_coord, var_12);
        var_50 = wp::add(var_15, var_49);
        var_51 = wp::load(var_48);
        wp::array_store(var_flat_procedural_coord, var_50, var_51);
    }
}



extern "C" __global__ void _expand_triangle_vertices_with_procedural_coord_kernel_0963c69a_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_procedural_coord,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_procedural_coord,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_procedural_coord,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_procedural_coord)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        const wp::int32 var_37 = 2;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32>* var_40;
        const wp::int32 var_41 = 0;
        wp::int32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        const wp::int32 var_49 = 2;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32> var_51;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        //---------
        // forward
        // def _expand_triangle_vertices_with_procedural_coord_kernel(                            <L 959>
        // t = wp.tid()                                                                           <L 970>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 971>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 972>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 973>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 974>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 975>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 976>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 977>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        // wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 978>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        // wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 979>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        // wp::array_store(var_flat_pos, var_26, var_27);
        // flat_uv3[base + 0] = vertex_uv3[v0]                                                    <L 980>
        var_28 = wp::address(var_vertex_uv3, var_4);
        var_30 = wp::add(var_15, var_29);
        var_31 = wp::load(var_28);
        // wp::array_store(var_flat_uv3, var_30, var_31);
        // flat_uv3[base + 1] = vertex_uv3[v1]                                                    <L 981>
        var_32 = wp::address(var_vertex_uv3, var_8);
        var_34 = wp::add(var_15, var_33);
        var_35 = wp::load(var_32);
        // wp::array_store(var_flat_uv3, var_34, var_35);
        // flat_uv3[base + 2] = vertex_uv3[v2]                                                    <L 982>
        var_36 = wp::address(var_vertex_uv3, var_12);
        var_38 = wp::add(var_15, var_37);
        var_39 = wp::load(var_36);
        // wp::array_store(var_flat_uv3, var_38, var_39);
        // flat_procedural_coord[base + 0] = procedural_coord[v0]                                 <L 983>
        var_40 = wp::address(var_procedural_coord, var_4);
        var_42 = wp::add(var_15, var_41);
        var_43 = wp::load(var_40);
        // wp::array_store(var_flat_procedural_coord, var_42, var_43);
        // flat_procedural_coord[base + 1] = procedural_coord[v1]                                 <L 984>
        var_44 = wp::address(var_procedural_coord, var_8);
        var_46 = wp::add(var_15, var_45);
        var_47 = wp::load(var_44);
        // wp::array_store(var_flat_procedural_coord, var_46, var_47);
        // flat_procedural_coord[base + 2] = procedural_coord[v2]                                 <L 985>
        var_48 = wp::address(var_procedural_coord, var_12);
        var_50 = wp::add(var_15, var_49);
        var_51 = wp::load(var_48);
        // wp::array_store(var_flat_procedural_coord, var_50, var_51);
        //---------
        // reverse
        wp::adj_array_store(var_flat_procedural_coord, var_50, var_51, adj_flat_procedural_coord, adj_50, adj_48);
        wp::adj_add(var_15, var_49, adj_15, adj_49, adj_50);
        wp::adj_address(var_procedural_coord, var_12, adj_procedural_coord, adj_12, adj_48);
        // adj: flat_procedural_coord[base + 2] = procedural_coord[v2]                            <L 985>
        wp::adj_array_store(var_flat_procedural_coord, var_46, var_47, adj_flat_procedural_coord, adj_46, adj_44);
        wp::adj_add(var_15, var_45, adj_15, adj_45, adj_46);
        wp::adj_address(var_procedural_coord, var_8, adj_procedural_coord, adj_8, adj_44);
        // adj: flat_procedural_coord[base + 1] = procedural_coord[v1]                            <L 984>
        wp::adj_array_store(var_flat_procedural_coord, var_42, var_43, adj_flat_procedural_coord, adj_42, adj_40);
        wp::adj_add(var_15, var_41, adj_15, adj_41, adj_42);
        wp::adj_address(var_procedural_coord, var_4, adj_procedural_coord, adj_4, adj_40);
        // adj: flat_procedural_coord[base + 0] = procedural_coord[v0]                            <L 983>
        wp::adj_array_store(var_flat_uv3, var_38, var_39, adj_flat_uv3, adj_38, adj_36);
        wp::adj_add(var_15, var_37, adj_15, adj_37, adj_38);
        wp::adj_address(var_vertex_uv3, var_12, adj_vertex_uv3, adj_12, adj_36);
        // adj: flat_uv3[base + 2] = vertex_uv3[v2]                                               <L 982>
        wp::adj_array_store(var_flat_uv3, var_34, var_35, adj_flat_uv3, adj_34, adj_32);
        wp::adj_add(var_15, var_33, adj_15, adj_33, adj_34);
        wp::adj_address(var_vertex_uv3, var_8, adj_vertex_uv3, adj_8, adj_32);
        // adj: flat_uv3[base + 1] = vertex_uv3[v1]                                               <L 981>
        wp::adj_array_store(var_flat_uv3, var_30, var_31, adj_flat_uv3, adj_30, adj_28);
        wp::adj_add(var_15, var_29, adj_15, adj_29, adj_30);
        wp::adj_address(var_vertex_uv3, var_4, adj_vertex_uv3, adj_4, adj_28);
        // adj: flat_uv3[base + 0] = vertex_uv3[v0]                                               <L 980>
        wp::adj_array_store(var_flat_pos, var_26, var_27, adj_flat_pos, adj_26, adj_24);
        wp::adj_add(var_15, var_25, adj_15, adj_25, adj_26);
        wp::adj_address(var_vertex_pos, var_12, adj_vertex_pos, adj_12, adj_24);
        // adj: flat_pos[base + 2] = vertex_pos[v2]                                               <L 979>
        wp::adj_array_store(var_flat_pos, var_22, var_23, adj_flat_pos, adj_22, adj_20);
        wp::adj_add(var_15, var_21, adj_15, adj_21, adj_22);
        wp::adj_address(var_vertex_pos, var_8, adj_vertex_pos, adj_8, adj_20);
        // adj: flat_pos[base + 1] = vertex_pos[v1]                                               <L 978>
        wp::adj_array_store(var_flat_pos, var_18, var_19, adj_flat_pos, adj_18, adj_16);
        wp::adj_add(var_15, var_17, adj_15, adj_17, adj_18);
        wp::adj_address(var_vertex_pos, var_4, adj_vertex_pos, adj_4, adj_16);
        // adj: flat_pos[base + 0] = vertex_pos[v0]                                               <L 977>
        wp::adj_mul(var_0, var_14, adj_0, adj_14, adj_15);
        // adj: base = t * 3                                                                      <L 976>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 975>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 974>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 973>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 972>
        }
        // adj: if t >= num_triangles:                                                            <L 971>
        // adj: t = wp.tid()                                                                      <L 970>
        // adj: def _expand_triangle_vertices_with_procedural_coord_kernel(                       <L 959>
        continue;
    }
}



extern "C" __global__ void _expand_triangle_material_state_kernel_62443aa8_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> var_particle_material,
    wp::int32 var_num_triangles,
    wp::array_t<wp::int32> var_flat_material_id,
    wp::array_t<wp::vec_t<4, wp::float32>> var_flat_state_rgba)
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
        const wp::int32 var_14 = 6;
        wp::int32 var_15;
        const wp::int32 var_16 = 6;
        wp::int32 var_17;
        const wp::int32 var_18 = 6;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        bool var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        bool var_34;
        bool var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 3;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        wp::vec_t<4, wp::float32> var_45;
        const wp::int32 var_46 = 0;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        const wp::int32 var_50 = 2;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        wp::int32 var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        const wp::int32 var_56 = 2;
        wp::int32 var_57;
        //---------
        // forward
        // def _expand_triangle_material_state_kernel(                                            <L 1113>
        // t = wp.tid()                                                                           <L 1121>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1122>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1123>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1124>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1125>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1126>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // p0 = v0 / 6                                                                            <L 1127>
        var_15 = wp::div(var_4, var_14);
        // p1 = v1 / 6                                                                            <L 1128>
        var_17 = wp::div(var_8, var_16);
        // p2 = v2 / 6                                                                            <L 1129>
        var_19 = wp::div(var_12, var_18);
        // m0 = particle_material[p0]                                                             <L 1130>
        var_20 = wp::address(var_particle_material, var_15);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // m1 = particle_material[p1]                                                             <L 1131>
        var_23 = wp::address(var_particle_material, var_17);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // m2 = particle_material[p2]                                                             <L 1132>
        var_26 = wp::address(var_particle_material, var_19);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // mat = m0                                                                               <L 1134>
        var_29 = wp::copy(var_21);
        // if m1 == m2:                                                                           <L 1135>
        var_30 = (var_24 == var_27);
        if (var_30) {
            // mat = m1                                                                           <L 1136>
            var_31 = wp::copy(var_24);
        }
        var_32 = wp::where(var_30, var_31, var_29);
        if (!var_30) {
            // elif m0 == m1 or m0 == m2:                                                         <L 1137>
            var_34 = (var_21 == var_24);
            var_33 = var_34;
            if (!var_33) {
                var_35 = (var_21 == var_27);
                var_33 = var_33 || var_35;
            }
            if (var_33) {
                // mat = m0                                                                       <L 1138>
                var_36 = wp::copy(var_21);
            }
            var_37 = wp::where(var_33, var_36, var_32);
        }
        var_38 = wp::where(var_30, var_32, var_37);
        // base = t * 3                                                                           <L 1140>
        var_40 = wp::mul(var_0, var_39);
        // zero = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                     <L 1141>
        var_45 = wp::vec_t<4, wp::float32>(var_41, var_42, var_43, var_44);
        // flat_material_id[base + 0] = mat                                                       <L 1142>
        var_47 = wp::add(var_40, var_46);
        wp::array_store(var_flat_material_id, var_47, var_38);
        // flat_material_id[base + 1] = mat                                                       <L 1143>
        var_49 = wp::add(var_40, var_48);
        wp::array_store(var_flat_material_id, var_49, var_38);
        // flat_material_id[base + 2] = mat                                                       <L 1144>
        var_51 = wp::add(var_40, var_50);
        wp::array_store(var_flat_material_id, var_51, var_38);
        // flat_state_rgba[base + 0] = zero                                                       <L 1145>
        var_53 = wp::add(var_40, var_52);
        wp::array_store(var_flat_state_rgba, var_53, var_45);
        // flat_state_rgba[base + 1] = zero                                                       <L 1146>
        var_55 = wp::add(var_40, var_54);
        wp::array_store(var_flat_state_rgba, var_55, var_45);
        // flat_state_rgba[base + 2] = zero                                                       <L 1147>
        var_57 = wp::add(var_40, var_56);
        wp::array_store(var_flat_state_rgba, var_57, var_45);
    }
}



extern "C" __global__ void _expand_triangle_material_state_kernel_62443aa8_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> var_particle_material,
    wp::int32 var_num_triangles,
    wp::array_t<wp::int32> var_flat_material_id,
    wp::array_t<wp::vec_t<4, wp::float32>> var_flat_state_rgba,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::int32> adj_particle_material,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::int32> adj_flat_material_id,
    wp::array_t<wp::vec_t<4, wp::float32>> adj_flat_state_rgba)
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
        const wp::int32 var_14 = 6;
        wp::int32 var_15;
        const wp::int32 var_16 = 6;
        wp::int32 var_17;
        const wp::int32 var_18 = 6;
        wp::int32 var_19;
        wp::int32* var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        wp::int32* var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        bool var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        bool var_33;
        bool var_34;
        bool var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        const wp::int32 var_39 = 3;
        wp::int32 var_40;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        const wp::float32 var_43 = 0.0;
        const wp::float32 var_44 = 0.0;
        wp::vec_t<4, wp::float32> var_45;
        const wp::int32 var_46 = 0;
        wp::int32 var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        const wp::int32 var_50 = 2;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        wp::int32 var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        const wp::int32 var_56 = 2;
        wp::int32 var_57;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        bool adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        bool adj_33 = {};
        bool adj_34 = {};
        bool adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::vec_t<4, wp::float32> adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        //---------
        // forward
        // def _expand_triangle_material_state_kernel(                                            <L 1113>
        // t = wp.tid()                                                                           <L 1121>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1122>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1123>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1124>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1125>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1126>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // p0 = v0 / 6                                                                            <L 1127>
        var_15 = wp::div(var_4, var_14);
        // p1 = v1 / 6                                                                            <L 1128>
        var_17 = wp::div(var_8, var_16);
        // p2 = v2 / 6                                                                            <L 1129>
        var_19 = wp::div(var_12, var_18);
        // m0 = particle_material[p0]                                                             <L 1130>
        var_20 = wp::address(var_particle_material, var_15);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // m1 = particle_material[p1]                                                             <L 1131>
        var_23 = wp::address(var_particle_material, var_17);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // m2 = particle_material[p2]                                                             <L 1132>
        var_26 = wp::address(var_particle_material, var_19);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // mat = m0                                                                               <L 1134>
        var_29 = wp::copy(var_21);
        // if m1 == m2:                                                                           <L 1135>
        var_30 = (var_24 == var_27);
        if (var_30) {
            // mat = m1                                                                           <L 1136>
            var_31 = wp::copy(var_24);
        }
        var_32 = wp::where(var_30, var_31, var_29);
        if (!var_30) {
            // elif m0 == m1 or m0 == m2:                                                         <L 1137>
            var_34 = (var_21 == var_24);
            var_33 = var_34;
            if (!var_33) {
                var_35 = (var_21 == var_27);
                var_33 = var_33 || var_35;
            }
            if (var_33) {
                // mat = m0                                                                       <L 1138>
                var_36 = wp::copy(var_21);
            }
            var_37 = wp::where(var_33, var_36, var_32);
        }
        var_38 = wp::where(var_30, var_32, var_37);
        // base = t * 3                                                                           <L 1140>
        var_40 = wp::mul(var_0, var_39);
        // zero = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                     <L 1141>
        var_45 = wp::vec_t<4, wp::float32>(var_41, var_42, var_43, var_44);
        // flat_material_id[base + 0] = mat                                                       <L 1142>
        var_47 = wp::add(var_40, var_46);
        // wp::array_store(var_flat_material_id, var_47, var_38);
        // flat_material_id[base + 1] = mat                                                       <L 1143>
        var_49 = wp::add(var_40, var_48);
        // wp::array_store(var_flat_material_id, var_49, var_38);
        // flat_material_id[base + 2] = mat                                                       <L 1144>
        var_51 = wp::add(var_40, var_50);
        // wp::array_store(var_flat_material_id, var_51, var_38);
        // flat_state_rgba[base + 0] = zero                                                       <L 1145>
        var_53 = wp::add(var_40, var_52);
        // wp::array_store(var_flat_state_rgba, var_53, var_45);
        // flat_state_rgba[base + 1] = zero                                                       <L 1146>
        var_55 = wp::add(var_40, var_54);
        // wp::array_store(var_flat_state_rgba, var_55, var_45);
        // flat_state_rgba[base + 2] = zero                                                       <L 1147>
        var_57 = wp::add(var_40, var_56);
        // wp::array_store(var_flat_state_rgba, var_57, var_45);
        //---------
        // reverse
        wp::adj_array_store(var_flat_state_rgba, var_57, var_45, adj_flat_state_rgba, adj_57, adj_45);
        wp::adj_add(var_40, var_56, adj_40, adj_56, adj_57);
        // adj: flat_state_rgba[base + 2] = zero                                                  <L 1147>
        wp::adj_array_store(var_flat_state_rgba, var_55, var_45, adj_flat_state_rgba, adj_55, adj_45);
        wp::adj_add(var_40, var_54, adj_40, adj_54, adj_55);
        // adj: flat_state_rgba[base + 1] = zero                                                  <L 1146>
        wp::adj_array_store(var_flat_state_rgba, var_53, var_45, adj_flat_state_rgba, adj_53, adj_45);
        wp::adj_add(var_40, var_52, adj_40, adj_52, adj_53);
        // adj: flat_state_rgba[base + 0] = zero                                                  <L 1145>
        wp::adj_array_store(var_flat_material_id, var_51, var_38, adj_flat_material_id, adj_51, adj_38);
        wp::adj_add(var_40, var_50, adj_40, adj_50, adj_51);
        // adj: flat_material_id[base + 2] = mat                                                  <L 1144>
        wp::adj_array_store(var_flat_material_id, var_49, var_38, adj_flat_material_id, adj_49, adj_38);
        wp::adj_add(var_40, var_48, adj_40, adj_48, adj_49);
        // adj: flat_material_id[base + 1] = mat                                                  <L 1143>
        wp::adj_array_store(var_flat_material_id, var_47, var_38, adj_flat_material_id, adj_47, adj_38);
        wp::adj_add(var_40, var_46, adj_40, adj_46, adj_47);
        // adj: flat_material_id[base + 0] = mat                                                  <L 1142>
        wp::adj_vec_t(var_41, var_42, var_43, var_44, adj_41, adj_42, adj_43, adj_44, adj_45);
        // adj: zero = wp.vec4(0.0, 0.0, 0.0, 0.0)                                                <L 1141>
        wp::adj_mul(var_0, var_39, adj_0, adj_39, adj_40);
        // adj: base = t * 3                                                                      <L 1140>
        wp::adj_where(var_30, var_32, var_37, adj_30, adj_32, adj_37, adj_38);
        if (!var_30) {
            wp::adj_where(var_33, var_36, var_32, adj_33, adj_36, adj_32, adj_37);
            if (var_33) {
                wp::adj_copy(var_21, adj_21, adj_36);
                // adj: mat = m0                                                                  <L 1138>
            }
            if (!var_33) {
            }
            // adj: elif m0 == m1 or m0 == m2:                                                    <L 1137>
        }
        wp::adj_where(var_30, var_31, var_29, adj_30, adj_31, adj_29, adj_32);
        if (var_30) {
            wp::adj_copy(var_24, adj_24, adj_31);
            // adj: mat = m1                                                                      <L 1136>
        }
        // adj: if m1 == m2:                                                                      <L 1135>
        wp::adj_copy(var_21, adj_21, adj_29);
        // adj: mat = m0                                                                          <L 1134>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_particle_material, var_19, adj_particle_material, adj_19, adj_26);
        // adj: m2 = particle_material[p2]                                                        <L 1132>
        wp::adj_copy(var_25, adj_23, adj_24);
        wp::adj_address(var_particle_material, var_17, adj_particle_material, adj_17, adj_23);
        // adj: m1 = particle_material[p1]                                                        <L 1131>
        wp::adj_copy(var_22, adj_20, adj_21);
        wp::adj_address(var_particle_material, var_15, adj_particle_material, adj_15, adj_20);
        // adj: m0 = particle_material[p0]                                                        <L 1130>
        wp::adj_div(var_12, var_18, var_19, adj_12, adj_18, adj_19);
        // adj: p2 = v2 / 6                                                                       <L 1129>
        wp::adj_div(var_8, var_16, var_17, adj_8, adj_16, adj_17);
        // adj: p1 = v1 / 6                                                                       <L 1128>
        wp::adj_div(var_4, var_14, var_15, adj_4, adj_14, adj_15);
        // adj: p0 = v0 / 6                                                                       <L 1127>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 1126>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 1125>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 1124>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1123>
        }
        // adj: if t >= num_triangles:                                                            <L 1122>
        // adj: t = wp.tid()                                                                      <L 1121>
        // adj: def _expand_triangle_material_state_kernel(                                       <L 1113>
        continue;
    }
}



extern "C" __global__ void _gather_cluster_box_edges_kernel_76909458_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::int32 var_num_clusters,
    wp::array_t<wp::vec_t<3, wp::float32>> var_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ends)
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
        const wp::int32 var_1 = 12;
        wp::int32 var_2;
        const wp::int32 var_3 = 12;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        const wp::float32 var_40 = 0.0;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        wp::vec_t<3, wp::float32> var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32>* var_52;
        wp::vec_t<3, wp::float32> var_53;
        //---------
        // forward
        // def _gather_cluster_box_edges_kernel(                                                  <L 1729>
        // tid = wp.tid()                                                                         <L 1738>
        var_0 = builtin_tid1d();
        // cluster_idx = tid // 12                                                                <L 1739>
        var_2 = wp::floordiv(var_0, var_1);
        // edge_idx = tid - cluster_idx * 12                                                      <L 1740>
        var_4 = wp::mul(var_2, var_3);
        var_5 = wp::sub(var_0, var_4);
        // slot_a = _cluster_edge_slot_a(edge_idx)                                                <L 1742>
        var_6 = _cluster_edge_slot_a_0(var_5);
        // slot_b = _cluster_edge_slot_b(edge_idx)                                                <L 1743>
        var_7 = _cluster_edge_slot_b_0(var_5);
        // particle_a = indices_by_slot[slot_a * num_clusters + cluster_idx]                      <L 1744>
        var_8 = wp::mul(var_6, var_num_clusters);
        var_9 = wp::add(var_8, var_2);
        var_10 = wp::address(var_indices_by_slot, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // particle_b = indices_by_slot[slot_b * num_clusters + cluster_idx]                      <L 1745>
        var_13 = wp::mul(var_7, var_num_clusters);
        var_14 = wp::add(var_13, var_2);
        var_15 = wp::address(var_indices_by_slot, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // active_bit = wp.int32(ParticleFlags.ACTIVE)                                            <L 1747>
        var_20 = wp::int32(var_19);
        // if (                                                                                   <L 1748>
        // cluster_active[cluster_idx] == 0                                                       <L 1749>
        var_22 = wp::address(var_cluster_active, var_2);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        var_21 = var_24;
        if (!var_21) {
            // or particle_a < 0                                                                  <L 1750>
            var_27 = (var_11 < var_26);
            var_21 = var_21 || var_27;
        }
        if (!var_21) {
            // or particle_b < 0                                                                  <L 1751>
            var_29 = (var_16 < var_28);
            var_21 = var_21 || var_29;
        }
        if (!var_21) {
            // or (particle_flags[particle_a] & active_bit) == 0                                  <L 1752>
            var_30 = wp::address(var_particle_flags, var_11);
            var_32 = wp::load(var_30);
            var_31 = wp::bit_and(var_32, var_20);
            var_34 = (var_31 == var_33);
            var_21 = var_21 || var_34;
        }
        if (!var_21) {
            // or (particle_flags[particle_b] & active_bit) == 0                                  <L 1753>
            var_35 = wp::address(var_particle_flags, var_16);
            var_37 = wp::load(var_35);
            var_36 = wp::bit_and(var_37, var_20);
            var_39 = (var_36 == var_38);
            var_21 = var_21 || var_39;
        }
        if (var_21) {
            // p = wp.vec3(0.0, 0.0, 0.0)                                                         <L 1755>
            var_43 = wp::vec_t<3, wp::float32>(var_40, var_41, var_42);
            // if particle_a >= 0:                                                                <L 1756>
            var_45 = (var_11 >= var_44);
            if (var_45) {
                // p = particle_q[particle_a]                                                     <L 1757>
                var_46 = wp::address(var_particle_q, var_11);
                var_48 = wp::load(var_46);
                var_47 = wp::copy(var_48);
            }
            var_49 = wp::where(var_45, var_47, var_43);
            // starts[tid] = p                                                                    <L 1758>
            wp::array_store(var_starts, var_0, var_49);
            // ends[tid] = p                                                                      <L 1759>
            wp::array_store(var_ends, var_0, var_49);
        }
        if (!var_21) {
            // starts[tid] = particle_q[particle_a]                                               <L 1761>
            var_50 = wp::address(var_particle_q, var_11);
            var_51 = wp::load(var_50);
            wp::array_store(var_starts, var_0, var_51);
            // ends[tid] = particle_q[particle_b]                                                 <L 1762>
            var_52 = wp::address(var_particle_q, var_16);
            var_53 = wp::load(var_52);
            wp::array_store(var_ends, var_0, var_53);
        }
    }
}



extern "C" __global__ void _gather_cluster_box_edges_kernel_76909458_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices_by_slot,
    wp::array_t<wp::int32> var_cluster_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::int32 var_num_clusters,
    wp::array_t<wp::vec_t<3, wp::float32>> var_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ends,
    wp::array_t<wp::int32> adj_indices_by_slot,
    wp::array_t<wp::int32> adj_cluster_active,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::int32 adj_num_clusters,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_ends)
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
        const wp::int32 var_1 = 12;
        wp::int32 var_2;
        const wp::int32 var_3 = 12;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 1;
        const wp::int32 var_19 = 1;
        wp::int32 var_20;
        bool var_21;
        wp::int32* var_22;
        const wp::int32 var_23 = 0;
        bool var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 0;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        wp::int32* var_30;
        wp::int32 var_31;
        wp::int32 var_32;
        const wp::int32 var_33 = 0;
        bool var_34;
        wp::int32* var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        bool var_39;
        const wp::float32 var_40 = 0.0;
        const wp::float32 var_41 = 0.0;
        const wp::float32 var_42 = 0.0;
        wp::vec_t<3, wp::float32> var_43;
        const wp::int32 var_44 = 0;
        bool var_45;
        wp::vec_t<3, wp::float32>* var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32> var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32>* var_52;
        wp::vec_t<3, wp::float32> var_53;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        bool adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        bool adj_27 = {};
        wp::int32 adj_28 = {};
        bool adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        bool adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        bool adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        wp::int32 adj_44 = {};
        bool adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::vec_t<3, wp::float32> adj_47 = {};
        wp::vec_t<3, wp::float32> adj_48 = {};
        wp::vec_t<3, wp::float32> adj_49 = {};
        wp::vec_t<3, wp::float32> adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        wp::vec_t<3, wp::float32> adj_52 = {};
        wp::vec_t<3, wp::float32> adj_53 = {};
        //---------
        // forward
        // def _gather_cluster_box_edges_kernel(                                                  <L 1729>
        // tid = wp.tid()                                                                         <L 1738>
        var_0 = builtin_tid1d();
        // cluster_idx = tid // 12                                                                <L 1739>
        var_2 = wp::floordiv(var_0, var_1);
        // edge_idx = tid - cluster_idx * 12                                                      <L 1740>
        var_4 = wp::mul(var_2, var_3);
        var_5 = wp::sub(var_0, var_4);
        // slot_a = _cluster_edge_slot_a(edge_idx)                                                <L 1742>
        var_6 = _cluster_edge_slot_a_0(var_5);
        // slot_b = _cluster_edge_slot_b(edge_idx)                                                <L 1743>
        var_7 = _cluster_edge_slot_b_0(var_5);
        // particle_a = indices_by_slot[slot_a * num_clusters + cluster_idx]                      <L 1744>
        var_8 = wp::mul(var_6, var_num_clusters);
        var_9 = wp::add(var_8, var_2);
        var_10 = wp::address(var_indices_by_slot, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // particle_b = indices_by_slot[slot_b * num_clusters + cluster_idx]                      <L 1745>
        var_13 = wp::mul(var_7, var_num_clusters);
        var_14 = wp::add(var_13, var_2);
        var_15 = wp::address(var_indices_by_slot, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // active_bit = wp.int32(ParticleFlags.ACTIVE)                                            <L 1747>
        var_20 = wp::int32(var_19);
        // if (                                                                                   <L 1748>
        // cluster_active[cluster_idx] == 0                                                       <L 1749>
        var_22 = wp::address(var_cluster_active, var_2);
        var_25 = wp::load(var_22);
        var_24 = (var_25 == var_23);
        var_21 = var_24;
        if (!var_21) {
            // or particle_a < 0                                                                  <L 1750>
            var_27 = (var_11 < var_26);
            var_21 = var_21 || var_27;
        }
        if (!var_21) {
            // or particle_b < 0                                                                  <L 1751>
            var_29 = (var_16 < var_28);
            var_21 = var_21 || var_29;
        }
        if (!var_21) {
            // or (particle_flags[particle_a] & active_bit) == 0                                  <L 1752>
            var_30 = wp::address(var_particle_flags, var_11);
            var_32 = wp::load(var_30);
            var_31 = wp::bit_and(var_32, var_20);
            var_34 = (var_31 == var_33);
            var_21 = var_21 || var_34;
        }
        if (!var_21) {
            // or (particle_flags[particle_b] & active_bit) == 0                                  <L 1753>
            var_35 = wp::address(var_particle_flags, var_16);
            var_37 = wp::load(var_35);
            var_36 = wp::bit_and(var_37, var_20);
            var_39 = (var_36 == var_38);
            var_21 = var_21 || var_39;
        }
        if (var_21) {
            // p = wp.vec3(0.0, 0.0, 0.0)                                                         <L 1755>
            var_43 = wp::vec_t<3, wp::float32>(var_40, var_41, var_42);
            // if particle_a >= 0:                                                                <L 1756>
            var_45 = (var_11 >= var_44);
            if (var_45) {
                // p = particle_q[particle_a]                                                     <L 1757>
                var_46 = wp::address(var_particle_q, var_11);
                var_48 = wp::load(var_46);
                var_47 = wp::copy(var_48);
            }
            var_49 = wp::where(var_45, var_47, var_43);
            // starts[tid] = p                                                                    <L 1758>
            // wp::array_store(var_starts, var_0, var_49);
            // ends[tid] = p                                                                      <L 1759>
            // wp::array_store(var_ends, var_0, var_49);
        }
        if (!var_21) {
            // starts[tid] = particle_q[particle_a]                                               <L 1761>
            var_50 = wp::address(var_particle_q, var_11);
            var_51 = wp::load(var_50);
            // wp::array_store(var_starts, var_0, var_51);
            // ends[tid] = particle_q[particle_b]                                                 <L 1762>
            var_52 = wp::address(var_particle_q, var_16);
            var_53 = wp::load(var_52);
            // wp::array_store(var_ends, var_0, var_53);
        }
        //---------
        // reverse
        if (!var_21) {
            wp::adj_array_store(var_ends, var_0, var_53, adj_ends, adj_0, adj_52);
            wp::adj_address(var_particle_q, var_16, adj_particle_q, adj_16, adj_52);
            // adj: ends[tid] = particle_q[particle_b]                                            <L 1762>
            wp::adj_array_store(var_starts, var_0, var_51, adj_starts, adj_0, adj_50);
            wp::adj_address(var_particle_q, var_11, adj_particle_q, adj_11, adj_50);
            // adj: starts[tid] = particle_q[particle_a]                                          <L 1761>
        }
        if (var_21) {
            wp::adj_array_store(var_ends, var_0, var_49, adj_ends, adj_0, adj_49);
            // adj: ends[tid] = p                                                                 <L 1759>
            wp::adj_array_store(var_starts, var_0, var_49, adj_starts, adj_0, adj_49);
            // adj: starts[tid] = p                                                               <L 1758>
            wp::adj_where(var_45, var_47, var_43, adj_45, adj_47, adj_43, adj_49);
            if (var_45) {
                wp::adj_copy(var_48, adj_46, adj_47);
                wp::adj_address(var_particle_q, var_11, adj_particle_q, adj_11, adj_46);
                // adj: p = particle_q[particle_a]                                                <L 1757>
            }
            // adj: if particle_a >= 0:                                                           <L 1756>
            wp::adj_vec_t(var_40, var_41, var_42, adj_40, adj_41, adj_42, adj_43);
            // adj: p = wp.vec3(0.0, 0.0, 0.0)                                                    <L 1755>
        }
        if (!var_21) {
            wp::adj_address(var_particle_flags, var_16, adj_particle_flags, adj_16, adj_35);
            // adj: or (particle_flags[particle_b] & active_bit) == 0                             <L 1753>
        }
        if (!var_21) {
            wp::adj_address(var_particle_flags, var_11, adj_particle_flags, adj_11, adj_30);
            // adj: or (particle_flags[particle_a] & active_bit) == 0                             <L 1752>
        }
        if (!var_21) {
            // adj: or particle_b < 0                                                             <L 1751>
        }
        if (!var_21) {
            // adj: or particle_a < 0                                                             <L 1750>
        }
        wp::adj_address(var_cluster_active, var_2, adj_cluster_active, adj_2, adj_22);
        // adj: cluster_active[cluster_idx] == 0                                                  <L 1749>
        // adj: if (                                                                              <L 1748>
        wp::adj_int32(var_19, adj_19, adj_20);
        // adj: active_bit = wp.int32(ParticleFlags.ACTIVE)                                       <L 1747>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_address(var_indices_by_slot, var_14, adj_indices_by_slot, adj_14, adj_15);
        wp::adj_add(var_13, var_2, adj_13, adj_2, adj_14);
        wp::adj_mul(var_7, var_num_clusters, adj_7, adj_num_clusters, adj_13);
        // adj: particle_b = indices_by_slot[slot_b * num_clusters + cluster_idx]                 <L 1745>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_indices_by_slot, var_9, adj_indices_by_slot, adj_9, adj_10);
        wp::adj_add(var_8, var_2, adj_8, adj_2, adj_9);
        wp::adj_mul(var_6, var_num_clusters, adj_6, adj_num_clusters, adj_8);
        // adj: particle_a = indices_by_slot[slot_a * num_clusters + cluster_idx]                 <L 1744>
        adj__cluster_edge_slot_b_0(var_5, adj_5, adj_7);
        // adj: slot_b = _cluster_edge_slot_b(edge_idx)                                           <L 1743>
        adj__cluster_edge_slot_a_0(var_5, adj_5, adj_6);
        // adj: slot_a = _cluster_edge_slot_a(edge_idx)                                           <L 1742>
        wp::adj_sub(var_0, var_4, adj_0, adj_4, adj_5);
        wp::adj_mul(var_2, var_3, adj_2, adj_3, adj_4);
        // adj: edge_idx = tid - cluster_idx * 12                                                 <L 1740>
        // adj: cluster_idx = tid // 12                                                           <L 1739>
        // adj: tid = wp.tid()                                                                    <L 1738>
        // adj: def _gather_cluster_box_edges_kernel(                                             <L 1729>
        continue;
    }
}



extern "C" __global__ void _expand_triangle_positions_kernel_61ba38f6_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        //---------
        // forward
        // def _expand_triangle_positions_kernel(                                                 <L 1045>
        // t = wp.tid()                                                                           <L 1056>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1057>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1058>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1059>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1060>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1061>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 1062>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 1063>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 1064>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 1065>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        wp::array_store(var_flat_pos, var_26, var_27);
    }
}



extern "C" __global__ void _expand_triangle_positions_kernel_61ba38f6_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_pos)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        //---------
        // forward
        // def _expand_triangle_positions_kernel(                                                 <L 1045>
        // t = wp.tid()                                                                           <L 1056>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1057>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1058>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1059>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1060>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1061>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 1062>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 1063>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        // wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 1064>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        // wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 1065>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        // wp::array_store(var_flat_pos, var_26, var_27);
        //---------
        // reverse
        wp::adj_array_store(var_flat_pos, var_26, var_27, adj_flat_pos, adj_26, adj_24);
        wp::adj_add(var_15, var_25, adj_15, adj_25, adj_26);
        wp::adj_address(var_vertex_pos, var_12, adj_vertex_pos, adj_12, adj_24);
        // adj: flat_pos[base + 2] = vertex_pos[v2]                                               <L 1065>
        wp::adj_array_store(var_flat_pos, var_22, var_23, adj_flat_pos, adj_22, adj_20);
        wp::adj_add(var_15, var_21, adj_15, adj_21, adj_22);
        wp::adj_address(var_vertex_pos, var_8, adj_vertex_pos, adj_8, adj_20);
        // adj: flat_pos[base + 1] = vertex_pos[v1]                                               <L 1064>
        wp::adj_array_store(var_flat_pos, var_18, var_19, adj_flat_pos, adj_18, adj_16);
        wp::adj_add(var_15, var_17, adj_15, adj_17, adj_18);
        wp::adj_address(var_vertex_pos, var_4, adj_vertex_pos, adj_4, adj_16);
        // adj: flat_pos[base + 0] = vertex_pos[v0]                                               <L 1063>
        wp::adj_mul(var_0, var_14, adj_0, adj_14, adj_15);
        // adj: base = t * 3                                                                      <L 1062>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 1061>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 1060>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 1059>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1058>
        }
        // adj: if t >= num_triangles:                                                            <L 1057>
        // adj: t = wp.tid()                                                                      <L 1056>
        // adj: def _expand_triangle_positions_kernel(                                            <L 1045>
        continue;
    }
}



extern "C" __global__ void _accumulate_vertex_neighbours_kernel_a0c73fab_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_neighbour_sum,
    wp::array_t<wp::int32> var_neighbour_degree)
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
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        const wp::int32 var_31 = 2;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        //---------
        // forward
        // def _accumulate_vertex_neighbours_kernel(                                              <L 1151>
        // t = wp.tid()                                                                           <L 1159>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1160>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1161>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1162>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1163>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1164>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // p0 = vertex_pos[v0]                                                                    <L 1165>
        var_14 = wp::address(var_vertex_pos, var_4);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // p1 = vertex_pos[v1]                                                                    <L 1166>
        var_17 = wp::address(var_vertex_pos, var_8);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // p2 = vertex_pos[v2]                                                                    <L 1167>
        var_20 = wp::address(var_vertex_pos, var_12);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // wp.atomic_add(neighbour_sum, v0, p1 + p2)                                              <L 1169>
        var_23 = wp::add(var_18, var_21);
        var_24 = wp::atomic_add(var_neighbour_sum, var_4, var_23);
        // wp.atomic_add(neighbour_sum, v1, p0 + p2)                                              <L 1170>
        var_25 = wp::add(var_15, var_21);
        var_26 = wp::atomic_add(var_neighbour_sum, var_8, var_25);
        // wp.atomic_add(neighbour_sum, v2, p0 + p1)                                              <L 1171>
        var_27 = wp::add(var_15, var_18);
        var_28 = wp::atomic_add(var_neighbour_sum, var_12, var_27);
        // wp.atomic_add(neighbour_degree, v0, 2)                                                 <L 1173>
        var_30 = wp::atomic_add(var_neighbour_degree, var_4, var_29);
        // wp.atomic_add(neighbour_degree, v1, 2)                                                 <L 1174>
        var_32 = wp::atomic_add(var_neighbour_degree, var_8, var_31);
        // wp.atomic_add(neighbour_degree, v2, 2)                                                 <L 1175>
        var_34 = wp::atomic_add(var_neighbour_degree, var_12, var_33);
    }
}



extern "C" __global__ void _accumulate_vertex_neighbours_kernel_a0c73fab_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_neighbour_sum,
    wp::array_t<wp::int32> var_neighbour_degree,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_neighbour_sum,
    wp::array_t<wp::int32> adj_neighbour_degree)
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
        wp::vec_t<3, wp::float32>* var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32> var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        const wp::int32 var_31 = 2;
        wp::int32 var_32;
        const wp::int32 var_33 = 2;
        wp::int32 var_34;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
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
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        //---------
        // forward
        // def _accumulate_vertex_neighbours_kernel(                                              <L 1151>
        // t = wp.tid()                                                                           <L 1159>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1160>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1161>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1162>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1163>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1164>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // p0 = vertex_pos[v0]                                                                    <L 1165>
        var_14 = wp::address(var_vertex_pos, var_4);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // p1 = vertex_pos[v1]                                                                    <L 1166>
        var_17 = wp::address(var_vertex_pos, var_8);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // p2 = vertex_pos[v2]                                                                    <L 1167>
        var_20 = wp::address(var_vertex_pos, var_12);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // wp.atomic_add(neighbour_sum, v0, p1 + p2)                                              <L 1169>
        var_23 = wp::add(var_18, var_21);
        // var_24 = wp::atomic_add(var_neighbour_sum, var_4, var_23);
        // wp.atomic_add(neighbour_sum, v1, p0 + p2)                                              <L 1170>
        var_25 = wp::add(var_15, var_21);
        // var_26 = wp::atomic_add(var_neighbour_sum, var_8, var_25);
        // wp.atomic_add(neighbour_sum, v2, p0 + p1)                                              <L 1171>
        var_27 = wp::add(var_15, var_18);
        // var_28 = wp::atomic_add(var_neighbour_sum, var_12, var_27);
        // wp.atomic_add(neighbour_degree, v0, 2)                                                 <L 1173>
        // var_30 = wp::atomic_add(var_neighbour_degree, var_4, var_29);
        // wp.atomic_add(neighbour_degree, v1, 2)                                                 <L 1174>
        // var_32 = wp::atomic_add(var_neighbour_degree, var_8, var_31);
        // wp.atomic_add(neighbour_degree, v2, 2)                                                 <L 1175>
        // var_34 = wp::atomic_add(var_neighbour_degree, var_12, var_33);
        //---------
        // reverse
        wp::adj_atomic_add(var_neighbour_degree, var_12, var_33, adj_neighbour_degree, adj_12, adj_33, adj_34);
        // adj: wp.atomic_add(neighbour_degree, v2, 2)                                            <L 1175>
        wp::adj_atomic_add(var_neighbour_degree, var_8, var_31, adj_neighbour_degree, adj_8, adj_31, adj_32);
        // adj: wp.atomic_add(neighbour_degree, v1, 2)                                            <L 1174>
        wp::adj_atomic_add(var_neighbour_degree, var_4, var_29, adj_neighbour_degree, adj_4, adj_29, adj_30);
        // adj: wp.atomic_add(neighbour_degree, v0, 2)                                            <L 1173>
        wp::adj_atomic_add(var_neighbour_sum, var_12, var_27, adj_neighbour_sum, adj_12, adj_27, adj_28);
        wp::adj_add(var_15, var_18, adj_15, adj_18, adj_27);
        // adj: wp.atomic_add(neighbour_sum, v2, p0 + p1)                                         <L 1171>
        wp::adj_atomic_add(var_neighbour_sum, var_8, var_25, adj_neighbour_sum, adj_8, adj_25, adj_26);
        wp::adj_add(var_15, var_21, adj_15, adj_21, adj_25);
        // adj: wp.atomic_add(neighbour_sum, v1, p0 + p2)                                         <L 1170>
        wp::adj_atomic_add(var_neighbour_sum, var_4, var_23, adj_neighbour_sum, adj_4, adj_23, adj_24);
        wp::adj_add(var_18, var_21, adj_18, adj_21, adj_23);
        // adj: wp.atomic_add(neighbour_sum, v0, p1 + p2)                                         <L 1169>
        wp::adj_copy(var_22, adj_20, adj_21);
        wp::adj_address(var_vertex_pos, var_12, adj_vertex_pos, adj_12, adj_20);
        // adj: p2 = vertex_pos[v2]                                                               <L 1167>
        wp::adj_copy(var_19, adj_17, adj_18);
        wp::adj_address(var_vertex_pos, var_8, adj_vertex_pos, adj_8, adj_17);
        // adj: p1 = vertex_pos[v1]                                                               <L 1166>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_vertex_pos, var_4, adj_vertex_pos, adj_4, adj_14);
        // adj: p0 = vertex_pos[v0]                                                               <L 1165>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 1164>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 1163>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 1162>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1161>
        }
        // adj: if t >= num_triangles:                                                            <L 1160>
        // adj: t = wp.tid()                                                                      <L 1159>
        // adj: def _accumulate_vertex_neighbours_kernel(                                         <L 1151>
        continue;
    }
}



extern "C" __global__ void _fill_triangle_material_atlas_kernel_f0803b03_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::vec_t<3, wp::float32>> var_material_colors,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        wp::int32* var_17;
        const wp::int32 var_18 = 6;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32* var_22;
        const wp::int32 var_23 = 6;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        wp::int32* var_27;
        const wp::int32 var_28 = 6;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        bool var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        bool var_45;
        bool var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        //---------
        // forward
        // def _fill_triangle_material_atlas_kernel(                                              <L 1261>
        // tx, ty, sub = wp.tid()                                                                 <L 1272>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1273>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1274>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1275>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1276>
            continue;
        }
        // px = sub % tile_size                                                                   <L 1278>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1279>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1280>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1281>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1282>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1283>
            continue;
        }
        // p0 = tri_indices[tid, 0] / 6                                                           <L 1285>
        var_17 = wp::address(var_tri_indices, var_5, var_16);
        var_20 = wp::load(var_17);
        var_19 = wp::div(var_20, var_18);
        // p1 = tri_indices[tid, 1] / 6                                                           <L 1286>
        var_22 = wp::address(var_tri_indices, var_5, var_21);
        var_25 = wp::load(var_22);
        var_24 = wp::div(var_25, var_23);
        // p2 = tri_indices[tid, 2] / 6                                                           <L 1287>
        var_27 = wp::address(var_tri_indices, var_5, var_26);
        var_30 = wp::load(var_27);
        var_29 = wp::div(var_30, var_28);
        // m0 = particle_material[p0]                                                             <L 1288>
        var_31 = wp::address(var_particle_material, var_19);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // m1 = particle_material[p1]                                                             <L 1289>
        var_34 = wp::address(var_particle_material, var_24);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // m2 = particle_material[p2]                                                             <L 1290>
        var_37 = wp::address(var_particle_material, var_29);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // mat = m0                                                                               <L 1292>
        var_40 = wp::copy(var_32);
        // if m1 == m2:                                                                           <L 1293>
        var_41 = (var_35 == var_38);
        if (var_41) {
            // mat = m1                                                                           <L 1294>
            var_42 = wp::copy(var_35);
        }
        var_43 = wp::where(var_41, var_42, var_40);
        if (!var_41) {
            // elif m0 == m1 or m0 == m2:                                                         <L 1295>
            var_45 = (var_32 == var_35);
            var_44 = var_45;
            if (!var_44) {
                var_46 = (var_32 == var_38);
                var_44 = var_44 || var_46;
            }
            if (var_44) {
                // mat = m0                                                                       <L 1296>
                var_47 = wp::copy(var_32);
            }
            var_48 = wp::where(var_44, var_47, var_43);
        }
        var_49 = wp::where(var_41, var_43, var_48);
        // atlas_rgb[y, x] = material_colors[mat]                                                 <L 1298>
        var_50 = wp::address(var_material_colors, var_49);
        var_51 = wp::load(var_50);
        wp::array_store(var_atlas_rgb, var_12, var_10, var_51);
    }
}



extern "C" __global__ void _fill_triangle_material_atlas_kernel_f0803b03_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::vec_t<3, wp::float32>> var_material_colors,
    wp::int32 var_atlas_width,
    wp::int32 var_atlas_height,
    wp::int32 var_tile_size,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_atlas_rgb,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::int32> adj_particle_material,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_material_colors,
    wp::int32 adj_atlas_width,
    wp::int32 adj_atlas_height,
    wp::int32 adj_tile_size,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_atlas_rgb)
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
        wp::int32 var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        bool var_13;
        bool var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        wp::int32* var_17;
        const wp::int32 var_18 = 6;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 1;
        wp::int32* var_22;
        const wp::int32 var_23 = 6;
        wp::int32 var_24;
        wp::int32 var_25;
        const wp::int32 var_26 = 2;
        wp::int32* var_27;
        const wp::int32 var_28 = 6;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32* var_31;
        wp::int32 var_32;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        bool var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        bool var_45;
        bool var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        wp::vec_t<3, wp::float32>* var_50;
        wp::vec_t<3, wp::float32> var_51;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        bool adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        bool adj_13 = {};
        bool adj_14 = {};
        bool adj_15 = {};
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
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        bool adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        bool adj_44 = {};
        bool adj_45 = {};
        bool adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::vec_t<3, wp::float32> adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        //---------
        // forward
        // def _fill_triangle_material_atlas_kernel(                                              <L 1261>
        // tx, ty, sub = wp.tid()                                                                 <L 1272>
        builtin_tid3d(var_0, var_1, var_2);
        // tiles_per_row = atlas_width / tile_size                                                <L 1273>
        var_3 = wp::div(var_atlas_width, var_tile_size);
        // tid = ty * tiles_per_row + tx                                                          <L 1274>
        var_4 = wp::mul(var_1, var_3);
        var_5 = wp::add(var_4, var_0);
        // if tid >= num_triangles:                                                               <L 1275>
        var_6 = (var_5 >= var_num_triangles);
        if (var_6) {
            // return                                                                             <L 1276>
            goto label0;
        }
        // px = sub % tile_size                                                                   <L 1278>
        var_7 = wp::mod(var_2, var_tile_size);
        // py = sub / tile_size                                                                   <L 1279>
        var_8 = wp::div(var_2, var_tile_size);
        // x = tx * tile_size + px                                                                <L 1280>
        var_9 = wp::mul(var_0, var_tile_size);
        var_10 = wp::add(var_9, var_7);
        // y = ty * tile_size + py                                                                <L 1281>
        var_11 = wp::mul(var_1, var_tile_size);
        var_12 = wp::add(var_11, var_8);
        // if x >= atlas_width or y >= atlas_height:                                              <L 1282>
        var_14 = (var_10 >= var_atlas_width);
        var_13 = var_14;
        if (!var_13) {
            var_15 = (var_12 >= var_atlas_height);
            var_13 = var_13 || var_15;
        }
        if (var_13) {
            // return                                                                             <L 1283>
            goto label1;
        }
        // p0 = tri_indices[tid, 0] / 6                                                           <L 1285>
        var_17 = wp::address(var_tri_indices, var_5, var_16);
        var_20 = wp::load(var_17);
        var_19 = wp::div(var_20, var_18);
        // p1 = tri_indices[tid, 1] / 6                                                           <L 1286>
        var_22 = wp::address(var_tri_indices, var_5, var_21);
        var_25 = wp::load(var_22);
        var_24 = wp::div(var_25, var_23);
        // p2 = tri_indices[tid, 2] / 6                                                           <L 1287>
        var_27 = wp::address(var_tri_indices, var_5, var_26);
        var_30 = wp::load(var_27);
        var_29 = wp::div(var_30, var_28);
        // m0 = particle_material[p0]                                                             <L 1288>
        var_31 = wp::address(var_particle_material, var_19);
        var_33 = wp::load(var_31);
        var_32 = wp::copy(var_33);
        // m1 = particle_material[p1]                                                             <L 1289>
        var_34 = wp::address(var_particle_material, var_24);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // m2 = particle_material[p2]                                                             <L 1290>
        var_37 = wp::address(var_particle_material, var_29);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // mat = m0                                                                               <L 1292>
        var_40 = wp::copy(var_32);
        // if m1 == m2:                                                                           <L 1293>
        var_41 = (var_35 == var_38);
        if (var_41) {
            // mat = m1                                                                           <L 1294>
            var_42 = wp::copy(var_35);
        }
        var_43 = wp::where(var_41, var_42, var_40);
        if (!var_41) {
            // elif m0 == m1 or m0 == m2:                                                         <L 1295>
            var_45 = (var_32 == var_35);
            var_44 = var_45;
            if (!var_44) {
                var_46 = (var_32 == var_38);
                var_44 = var_44 || var_46;
            }
            if (var_44) {
                // mat = m0                                                                       <L 1296>
                var_47 = wp::copy(var_32);
            }
            var_48 = wp::where(var_44, var_47, var_43);
        }
        var_49 = wp::where(var_41, var_43, var_48);
        // atlas_rgb[y, x] = material_colors[mat]                                                 <L 1298>
        var_50 = wp::address(var_material_colors, var_49);
        var_51 = wp::load(var_50);
        // wp::array_store(var_atlas_rgb, var_12, var_10, var_51);
        //---------
        // reverse
        wp::adj_array_store(var_atlas_rgb, var_12, var_10, var_51, adj_atlas_rgb, adj_12, adj_10, adj_50);
        wp::adj_address(var_material_colors, var_49, adj_material_colors, adj_49, adj_50);
        // adj: atlas_rgb[y, x] = material_colors[mat]                                            <L 1298>
        wp::adj_where(var_41, var_43, var_48, adj_41, adj_43, adj_48, adj_49);
        if (!var_41) {
            wp::adj_where(var_44, var_47, var_43, adj_44, adj_47, adj_43, adj_48);
            if (var_44) {
                wp::adj_copy(var_32, adj_32, adj_47);
                // adj: mat = m0                                                                  <L 1296>
            }
            if (!var_44) {
            }
            // adj: elif m0 == m1 or m0 == m2:                                                    <L 1295>
        }
        wp::adj_where(var_41, var_42, var_40, adj_41, adj_42, adj_40, adj_43);
        if (var_41) {
            wp::adj_copy(var_35, adj_35, adj_42);
            // adj: mat = m1                                                                      <L 1294>
        }
        // adj: if m1 == m2:                                                                      <L 1293>
        wp::adj_copy(var_32, adj_32, adj_40);
        // adj: mat = m0                                                                          <L 1292>
        wp::adj_copy(var_39, adj_37, adj_38);
        wp::adj_address(var_particle_material, var_29, adj_particle_material, adj_29, adj_37);
        // adj: m2 = particle_material[p2]                                                        <L 1290>
        wp::adj_copy(var_36, adj_34, adj_35);
        wp::adj_address(var_particle_material, var_24, adj_particle_material, adj_24, adj_34);
        // adj: m1 = particle_material[p1]                                                        <L 1289>
        wp::adj_copy(var_33, adj_31, adj_32);
        wp::adj_address(var_particle_material, var_19, adj_particle_material, adj_19, adj_31);
        // adj: m0 = particle_material[p0]                                                        <L 1288>
        wp::adj_div(var_30, var_28, var_29, adj_27, adj_28, adj_29);
        wp::adj_address(var_tri_indices, var_5, var_26, adj_tri_indices, adj_5, adj_26, adj_27);
        // adj: p2 = tri_indices[tid, 2] / 6                                                      <L 1287>
        wp::adj_div(var_25, var_23, var_24, adj_22, adj_23, adj_24);
        wp::adj_address(var_tri_indices, var_5, var_21, adj_tri_indices, adj_5, adj_21, adj_22);
        // adj: p1 = tri_indices[tid, 1] / 6                                                      <L 1286>
        wp::adj_div(var_20, var_18, var_19, adj_17, adj_18, adj_19);
        wp::adj_address(var_tri_indices, var_5, var_16, adj_tri_indices, adj_5, adj_16, adj_17);
        // adj: p0 = tri_indices[tid, 0] / 6                                                      <L 1285>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1283>
        }
        if (!var_13) {
        }
        // adj: if x >= atlas_width or y >= atlas_height:                                         <L 1282>
        wp::adj_add(var_11, var_8, adj_11, adj_8, adj_12);
        wp::adj_mul(var_1, var_tile_size, adj_1, adj_tile_size, adj_11);
        // adj: y = ty * tile_size + py                                                           <L 1281>
        wp::adj_add(var_9, var_7, adj_9, adj_7, adj_10);
        wp::adj_mul(var_0, var_tile_size, adj_0, adj_tile_size, adj_9);
        // adj: x = tx * tile_size + px                                                           <L 1280>
        wp::adj_div(var_2, var_tile_size, var_8, adj_2, adj_tile_size, adj_8);
        // adj: py = sub / tile_size                                                              <L 1279>
        wp::adj_mod(var_2, var_tile_size, adj_2, adj_tile_size, adj_7);
        // adj: px = sub % tile_size                                                              <L 1278>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 1276>
        }
        // adj: if tid >= num_triangles:                                                          <L 1275>
        wp::adj_add(var_4, var_0, adj_4, adj_0, adj_5);
        wp::adj_mul(var_1, var_3, adj_1, adj_3, adj_4);
        // adj: tid = ty * tiles_per_row + tx                                                     <L 1274>
        wp::adj_div(var_atlas_width, var_tile_size, var_3, adj_atlas_width, adj_tile_size, adj_3);
        // adj: tiles_per_row = atlas_width / tile_size                                           <L 1273>
        // adj: tx, ty, sub = wp.tid()                                                            <L 1272>
        // adj: def _fill_triangle_material_atlas_kernel(                                         <L 1261>
        continue;
    }
}



extern "C" __global__ void _gather_stress_colored_particles_kernel_a01b6dde_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::float32 var_color_scale,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        //---------
        // forward
        // def _gather_stress_colored_particles_kernel(                                           <L 1576>
        // i = wp.tid()                                                                           <L 1587>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 1588>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 1589>
            continue;
        }
        // if particle_q[i][2] > cut_z:                                                           <L 1590>
        var_9 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = (var_11 > var_cut_z);
        if (var_13) {
            // return                                                                             <L 1591>
            continue;
        }
        // idx = wp.atomic_add(counter, 0, 1)                                                     <L 1592>
        var_16 = wp::atomic_add(var_counter, var_14, var_15);
        // out_points[idx] = particle_q[i]                                                        <L 1593>
        var_17 = wp::address(var_particle_q, var_0);
        var_18 = wp::load(var_17);
        wp::array_store(var_out_points, var_16, var_18);
        // out_colors[idx] = _cold_warm_stress_color(cell_stretch[i] * color_scale)               <L 1594>
        var_19 = wp::address(var_cell_stretch, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::mul(var_21, var_color_scale);
        var_22 = _cold_warm_stress_color_0(var_20);
        wp::array_store(var_out_colors, var_16, var_22);
    }
}



extern "C" __global__ void _gather_stress_colored_particles_kernel_a01b6dde_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::float32 var_color_scale,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_colors,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::float32> adj_cell_stretch,
    wp::float32 adj_color_scale,
    wp::float32 adj_cut_z,
    wp::array_t<wp::int32> adj_counter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_points,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_colors)
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
        const wp::int32 var_3 = 1;
        wp::int32 var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::int32 var_10 = 2;
        wp::float32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        bool var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 1;
        wp::int32 var_16;
        wp::vec_t<3, wp::float32>* var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::float32* var_19;
        wp::float32 var_20;
        wp::float32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        bool adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        //---------
        // forward
        // def _gather_stress_colored_particles_kernel(                                           <L 1576>
        // i = wp.tid()                                                                           <L 1587>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 1588>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 1589>
            goto label0;
        }
        // if particle_q[i][2] > cut_z:                                                           <L 1590>
        var_9 = wp::address(var_particle_q, var_0);
        var_12 = wp::load(var_9);
        var_11 = wp::extract(var_12, var_10);
        var_13 = (var_11 > var_cut_z);
        if (var_13) {
            // return                                                                             <L 1591>
            goto label1;
        }
        // idx = wp.atomic_add(counter, 0, 1)                                                     <L 1592>
        // var_16 = wp::atomic_add(var_counter, var_14, var_15);
        // out_points[idx] = particle_q[i]                                                        <L 1593>
        var_17 = wp::address(var_particle_q, var_0);
        var_18 = wp::load(var_17);
        // wp::array_store(var_out_points, var_16, var_18);
        // out_colors[idx] = _cold_warm_stress_color(cell_stretch[i] * color_scale)               <L 1594>
        var_19 = wp::address(var_cell_stretch, var_0);
        var_21 = wp::load(var_19);
        var_20 = wp::mul(var_21, var_color_scale);
        var_22 = _cold_warm_stress_color_0(var_20);
        // wp::array_store(var_out_colors, var_16, var_22);
        //---------
        // reverse
        wp::adj_array_store(var_out_colors, var_16, var_22, adj_out_colors, adj_16, adj_22);
        adj__cold_warm_stress_color_0(var_20, adj_20, adj_22);
        wp::adj_mul(var_21, var_color_scale, adj_19, adj_color_scale, adj_20);
        wp::adj_address(var_cell_stretch, var_0, adj_cell_stretch, adj_0, adj_19);
        // adj: out_colors[idx] = _cold_warm_stress_color(cell_stretch[i] * color_scale)          <L 1594>
        wp::adj_array_store(var_out_points, var_16, var_18, adj_out_points, adj_16, adj_17);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_17);
        // adj: out_points[idx] = particle_q[i]                                                   <L 1593>
        wp::adj_atomic_add(var_counter, var_14, var_15, adj_counter, adj_14, adj_15, adj_16);
        // adj: idx = wp.atomic_add(counter, 0, 1)                                                <L 1592>
        if (var_13) {
            label1:;
            // adj: return                                                                        <L 1591>
        }
        wp::adj_extract(var_12, var_10, adj_9, adj_10, adj_11);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_9);
        // adj: if particle_q[i][2] > cut_z:                                                      <L 1590>
        if (var_8) {
            label0:;
            // adj: return                                                                        <L 1589>
        }
        wp::adj_int32(var_3, adj_3, adj_4);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                     <L 1588>
        // adj: i = wp.tid()                                                                      <L 1587>
        // adj: def _gather_stress_colored_particles_kernel(                                      <L 1576>
        continue;
    }
}



extern "C" __global__ void _apply_laplacian_step_kernel_e67fdfd4_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_neighbour_sum,
    wp::array_t<wp::int32> var_neighbour_degree,
    wp::float32 var_coeff,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_pos)
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
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::float32 var_10 = 1.0;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // forward
        // def _apply_laplacian_step_kernel(                                                      <L 1179>
        // vid = wp.tid()                                                                         <L 1187>
        var_0 = builtin_tid1d();
        // degree = neighbour_degree[vid]                                                         <L 1188>
        var_1 = wp::address(var_neighbour_degree, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // p = src_pos[vid]                                                                       <L 1189>
        var_4 = wp::address(var_src_pos, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if degree <= 0:                                                                        <L 1190>
        var_8 = (var_2 <= var_7);
        if (var_8) {
            // dst_pos[vid] = p                                                                   <L 1191>
            wp::array_store(var_dst_pos, var_0, var_5);
            // return                                                                             <L 1192>
            continue;
        }
        // avg = neighbour_sum[vid] * (1.0 / float(degree))                                       <L 1193>
        var_9 = wp::address(var_neighbour_sum, var_0);
        var_11 = wp::float(var_2);
        var_12 = wp::div(var_10, var_11);
        var_14 = wp::load(var_9);
        var_13 = wp::mul(var_14, var_12);
        // dst_pos[vid] = p + (avg - p) * coeff                                                   <L 1194>
        var_15 = wp::sub(var_13, var_5);
        var_16 = wp::mul(var_15, var_coeff);
        var_17 = wp::add(var_5, var_16);
        wp::array_store(var_dst_pos, var_0, var_17);
    }
}



extern "C" __global__ void _apply_laplacian_step_kernel_e67fdfd4_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_src_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_neighbour_sum,
    wp::array_t<wp::int32> var_neighbour_degree,
    wp::float32 var_coeff,
    wp::array_t<wp::vec_t<3, wp::float32>> var_dst_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_src_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_neighbour_sum,
    wp::array_t<wp::int32> adj_neighbour_degree,
    wp::float32 adj_coeff,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dst_pos)
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
        wp::vec_t<3, wp::float32>* var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        const wp::float32 var_10 = 1.0;
        wp::float32 var_11;
        wp::float32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::vec_t<3, wp::float32> adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::float32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::vec_t<3, wp::float32> adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        //---------
        // forward
        // def _apply_laplacian_step_kernel(                                                      <L 1179>
        // vid = wp.tid()                                                                         <L 1187>
        var_0 = builtin_tid1d();
        // degree = neighbour_degree[vid]                                                         <L 1188>
        var_1 = wp::address(var_neighbour_degree, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // p = src_pos[vid]                                                                       <L 1189>
        var_4 = wp::address(var_src_pos, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::copy(var_6);
        // if degree <= 0:                                                                        <L 1190>
        var_8 = (var_2 <= var_7);
        if (var_8) {
            // dst_pos[vid] = p                                                                   <L 1191>
            // wp::array_store(var_dst_pos, var_0, var_5);
            // return                                                                             <L 1192>
            goto label0;
        }
        // avg = neighbour_sum[vid] * (1.0 / float(degree))                                       <L 1193>
        var_9 = wp::address(var_neighbour_sum, var_0);
        var_11 = wp::float(var_2);
        var_12 = wp::div(var_10, var_11);
        var_14 = wp::load(var_9);
        var_13 = wp::mul(var_14, var_12);
        // dst_pos[vid] = p + (avg - p) * coeff                                                   <L 1194>
        var_15 = wp::sub(var_13, var_5);
        var_16 = wp::mul(var_15, var_coeff);
        var_17 = wp::add(var_5, var_16);
        // wp::array_store(var_dst_pos, var_0, var_17);
        //---------
        // reverse
        wp::adj_array_store(var_dst_pos, var_0, var_17, adj_dst_pos, adj_0, adj_17);
        wp::adj_add(var_5, var_16, adj_5, adj_16, adj_17);
        wp::adj_mul(var_15, var_coeff, adj_15, adj_coeff, adj_16);
        wp::adj_sub(var_13, var_5, adj_13, adj_5, adj_15);
        // adj: dst_pos[vid] = p + (avg - p) * coeff                                              <L 1194>
        wp::adj_mul(var_14, var_12, adj_9, adj_12, adj_13);
        wp::adj_div(var_10, var_11, var_12, adj_10, adj_11, adj_12);
        wp::adj_float(var_2, adj_2, adj_11);
        wp::adj_address(var_neighbour_sum, var_0, adj_neighbour_sum, adj_0, adj_9);
        // adj: avg = neighbour_sum[vid] * (1.0 / float(degree))                                  <L 1193>
        if (var_8) {
            label0:;
            // adj: return                                                                        <L 1192>
            wp::adj_array_store(var_dst_pos, var_0, var_5, adj_dst_pos, adj_0, adj_5);
            // adj: dst_pos[vid] = p                                                              <L 1191>
        }
        // adj: if degree <= 0:                                                                   <L 1190>
        wp::adj_copy(var_6, adj_4, adj_5);
        wp::adj_address(var_src_pos, var_0, adj_src_pos, adj_0, adj_4);
        // adj: p = src_pos[vid]                                                                  <L 1189>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_neighbour_degree, var_0, adj_neighbour_degree, adj_0, adj_1);
        // adj: degree = neighbour_degree[vid]                                                    <L 1188>
        // adj: vid = wp.tid()                                                                    <L 1187>
        // adj: def _apply_laplacian_step_kernel(                                                 <L 1179>
        continue;
    }
}



extern "C" __global__ void _sample_3d_texture_kernel_aaa8c598_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_rgb)
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
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::vec_t<3, wp::float32> var_50;
        //---------
        // forward
        // def _sample_3d_texture_kernel(                                                         <L 861>
        // i = wp.tid()                                                                           <L 887>
        var_0 = builtin_tid1d();
        // uv = vertex_uv3[i]                                                                     <L 888>
        var_1 = wp::address(var_vertex_uv3, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 889>
        var_4 = _select_axis_0(var_2, var_src_x);
        var_5 = _scale_about_centre_0(var_4, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 890>
        var_6 = _select_axis_0(var_2, var_src_y);
        var_7 = _scale_about_centre_0(var_6, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 891>
        var_8 = _select_axis_0(var_2, var_src_z);
        var_9 = _scale_about_centre_0(var_8, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 892>
        var_10 = wp::float(var_tex_nx);
        var_11 = wp::mul(var_5, var_10);
        var_12 = wp::int(var_11);
        var_15 = wp::sub(var_tex_nx, var_14);
        var_16 = wp::clamp(var_12, var_13, var_15);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 893>
        var_17 = wp::float(var_tex_ny);
        var_18 = wp::mul(var_7, var_17);
        var_19 = wp::int(var_18);
        var_22 = wp::sub(var_tex_ny, var_21);
        var_23 = wp::clamp(var_19, var_20, var_22);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 894>
        var_24 = wp::float(var_tex_nz);
        var_25 = wp::mul(var_9, var_24);
        var_26 = wp::int(var_25);
        var_29 = wp::sub(var_tex_nz, var_28);
        var_30 = wp::clamp(var_26, var_27, var_29);
        // if flip_x != 0:                                                                        <L 895>
        var_32 = (var_flip_x != var_31);
        if (var_32) {
            // ix = (tex_nx - 1) - ix                                                             <L 896>
            var_34 = wp::sub(var_tex_nx, var_33);
            var_35 = wp::sub(var_34, var_16);
        }
        var_36 = wp::where(var_32, var_35, var_16);
        // if flip_y != 0:                                                                        <L 897>
        var_38 = (var_flip_y != var_37);
        if (var_38) {
            // iy = (tex_ny - 1) - iy                                                             <L 898>
            var_40 = wp::sub(var_tex_ny, var_39);
            var_41 = wp::sub(var_40, var_23);
        }
        var_42 = wp::where(var_38, var_41, var_23);
        // if flip_z != 0:                                                                        <L 899>
        var_44 = (var_flip_z != var_43);
        if (var_44) {
            // iz = (tex_nz - 1) - iz                                                             <L 900>
            var_46 = wp::sub(var_tex_nz, var_45);
            var_47 = wp::sub(var_46, var_30);
        }
        var_48 = wp::where(var_44, var_47, var_30);
        // out_rgb[i] = texture[ix, iy, iz]                                                       <L 901>
        var_49 = wp::address(var_texture, var_36, var_42, var_48);
        var_50 = wp::load(var_49);
        wp::array_store(var_out_rgb, var_0, var_50);
    }
}



extern "C" __global__ void _sample_3d_texture_kernel_aaa8c598_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_texture,
    wp::int32 var_tex_nx,
    wp::int32 var_tex_ny,
    wp::int32 var_tex_nz,
    wp::int32 var_src_x,
    wp::int32 var_src_y,
    wp::int32 var_src_z,
    wp::int32 var_flip_x,
    wp::int32 var_flip_y,
    wp::int32 var_flip_z,
    wp::float32 var_scale_x,
    wp::float32 var_scale_y,
    wp::float32 var_scale_z,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_rgb,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_texture,
    wp::int32 adj_tex_nx,
    wp::int32 adj_tex_ny,
    wp::int32 adj_tex_nz,
    wp::int32 adj_src_x,
    wp::int32 adj_src_y,
    wp::int32 adj_src_z,
    wp::int32 adj_flip_x,
    wp::int32 adj_flip_y,
    wp::int32 adj_flip_z,
    wp::float32 adj_scale_x,
    wp::float32 adj_scale_y,
    wp::float32 adj_scale_z,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_rgb)
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
        wp::float32 var_4;
        wp::float32 var_5;
        wp::float32 var_6;
        wp::float32 var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32 var_16;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::int32 var_19;
        const wp::int32 var_20 = 0;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32 var_23;
        wp::float32 var_24;
        wp::float32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 0;
        const wp::int32 var_28 = 1;
        wp::int32 var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        const wp::int32 var_39 = 1;
        wp::int32 var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::vec_t<3, wp::float32> var_50;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3, wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::float32 adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        bool adj_32 = {};
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
        wp::int32 adj_43 = {};
        bool adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::vec_t<3, wp::float32> adj_49 = {};
        wp::vec_t<3, wp::float32> adj_50 = {};
        //---------
        // forward
        // def _sample_3d_texture_kernel(                                                         <L 861>
        // i = wp.tid()                                                                           <L 887>
        var_0 = builtin_tid1d();
        // uv = vertex_uv3[i]                                                                     <L 888>
        var_1 = wp::address(var_vertex_uv3, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                          <L 889>
        var_4 = _select_axis_0(var_2, var_src_x);
        var_5 = _scale_about_centre_0(var_4, var_scale_x);
        // tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                          <L 890>
        var_6 = _select_axis_0(var_2, var_src_y);
        var_7 = _scale_about_centre_0(var_6, var_scale_y);
        // tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                          <L 891>
        var_8 = _select_axis_0(var_2, var_src_z);
        var_9 = _scale_about_centre_0(var_8, var_scale_z);
        // ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                               <L 892>
        var_10 = wp::float(var_tex_nx);
        var_11 = wp::mul(var_5, var_10);
        var_12 = wp::int(var_11);
        var_15 = wp::sub(var_tex_nx, var_14);
        var_16 = wp::clamp(var_12, var_13, var_15);
        // iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                               <L 893>
        var_17 = wp::float(var_tex_ny);
        var_18 = wp::mul(var_7, var_17);
        var_19 = wp::int(var_18);
        var_22 = wp::sub(var_tex_ny, var_21);
        var_23 = wp::clamp(var_19, var_20, var_22);
        // iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                               <L 894>
        var_24 = wp::float(var_tex_nz);
        var_25 = wp::mul(var_9, var_24);
        var_26 = wp::int(var_25);
        var_29 = wp::sub(var_tex_nz, var_28);
        var_30 = wp::clamp(var_26, var_27, var_29);
        // if flip_x != 0:                                                                        <L 895>
        var_32 = (var_flip_x != var_31);
        if (var_32) {
            // ix = (tex_nx - 1) - ix                                                             <L 896>
            var_34 = wp::sub(var_tex_nx, var_33);
            var_35 = wp::sub(var_34, var_16);
        }
        var_36 = wp::where(var_32, var_35, var_16);
        // if flip_y != 0:                                                                        <L 897>
        var_38 = (var_flip_y != var_37);
        if (var_38) {
            // iy = (tex_ny - 1) - iy                                                             <L 898>
            var_40 = wp::sub(var_tex_ny, var_39);
            var_41 = wp::sub(var_40, var_23);
        }
        var_42 = wp::where(var_38, var_41, var_23);
        // if flip_z != 0:                                                                        <L 899>
        var_44 = (var_flip_z != var_43);
        if (var_44) {
            // iz = (tex_nz - 1) - iz                                                             <L 900>
            var_46 = wp::sub(var_tex_nz, var_45);
            var_47 = wp::sub(var_46, var_30);
        }
        var_48 = wp::where(var_44, var_47, var_30);
        // out_rgb[i] = texture[ix, iy, iz]                                                       <L 901>
        var_49 = wp::address(var_texture, var_36, var_42, var_48);
        var_50 = wp::load(var_49);
        // wp::array_store(var_out_rgb, var_0, var_50);
        //---------
        // reverse
        wp::adj_array_store(var_out_rgb, var_0, var_50, adj_out_rgb, adj_0, adj_49);
        wp::adj_address(var_texture, var_36, var_42, var_48, adj_texture, adj_36, adj_42, adj_48, adj_49);
        // adj: out_rgb[i] = texture[ix, iy, iz]                                                  <L 901>
        wp::adj_where(var_44, var_47, var_30, adj_44, adj_47, adj_30, adj_48);
        if (var_44) {
            wp::adj_sub(var_46, var_30, adj_46, adj_30, adj_47);
            wp::adj_sub(var_tex_nz, var_45, adj_tex_nz, adj_45, adj_46);
            // adj: iz = (tex_nz - 1) - iz                                                        <L 900>
        }
        // adj: if flip_z != 0:                                                                   <L 899>
        wp::adj_where(var_38, var_41, var_23, adj_38, adj_41, adj_23, adj_42);
        if (var_38) {
            wp::adj_sub(var_40, var_23, adj_40, adj_23, adj_41);
            wp::adj_sub(var_tex_ny, var_39, adj_tex_ny, adj_39, adj_40);
            // adj: iy = (tex_ny - 1) - iy                                                        <L 898>
        }
        // adj: if flip_y != 0:                                                                   <L 897>
        wp::adj_where(var_32, var_35, var_16, adj_32, adj_35, adj_16, adj_36);
        if (var_32) {
            wp::adj_sub(var_34, var_16, adj_34, adj_16, adj_35);
            wp::adj_sub(var_tex_nx, var_33, adj_tex_nx, adj_33, adj_34);
            // adj: ix = (tex_nx - 1) - ix                                                        <L 896>
        }
        // adj: if flip_x != 0:                                                                   <L 895>
        wp::adj_clamp(var_26, var_27, var_29, adj_26, adj_27, adj_29, adj_30);
        wp::adj_sub(var_tex_nz, var_28, adj_tex_nz, adj_28, adj_29);
        wp::adj_int(var_25, adj_25, adj_26);
        wp::adj_mul(var_9, var_24, adj_9, adj_24, adj_25);
        wp::adj_float(var_tex_nz, adj_tex_nz, adj_24);
        // adj: iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)                          <L 894>
        wp::adj_clamp(var_19, var_20, var_22, adj_19, adj_20, adj_22, adj_23);
        wp::adj_sub(var_tex_ny, var_21, adj_tex_ny, adj_21, adj_22);
        wp::adj_int(var_18, adj_18, adj_19);
        wp::adj_mul(var_7, var_17, adj_7, adj_17, adj_18);
        wp::adj_float(var_tex_ny, adj_tex_ny, adj_17);
        // adj: iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)                          <L 893>
        wp::adj_clamp(var_12, var_13, var_15, adj_12, adj_13, adj_15, adj_16);
        wp::adj_sub(var_tex_nx, var_14, adj_tex_nx, adj_14, adj_15);
        wp::adj_int(var_11, adj_11, adj_12);
        wp::adj_mul(var_5, var_10, adj_5, adj_10, adj_11);
        wp::adj_float(var_tex_nx, adj_tex_nx, adj_10);
        // adj: ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)                          <L 892>
        adj__scale_about_centre_0(var_8, var_scale_z, adj_8, adj_scale_z, adj_9);
        adj__select_axis_0(var_2, var_src_z, adj_2, adj_src_z, adj_8);
        // adj: tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)                     <L 891>
        adj__scale_about_centre_0(var_6, var_scale_y, adj_6, adj_scale_y, adj_7);
        adj__select_axis_0(var_2, var_src_y, adj_2, adj_src_y, adj_6);
        // adj: tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)                     <L 890>
        adj__scale_about_centre_0(var_4, var_scale_x, adj_4, adj_scale_x, adj_5);
        adj__select_axis_0(var_2, var_src_x, adj_2, adj_src_x, adj_4);
        // adj: tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)                     <L 889>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_vertex_uv3, var_0, adj_vertex_uv3, adj_0, adj_1);
        // adj: uv = vertex_uv3[i]                                                                <L 888>
        // adj: i = wp.tid()                                                                      <L 887>
        // adj: def _sample_3d_texture_kernel(                                                    <L 861>
        continue;
    }
}



extern "C" __global__ void _expand_triangle_vertices_majority_uv3_kernel_6ad7a9a3_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_uv3,
    wp::array_t<wp::int32> var_particle_material,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_uv3)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        const wp::int32 var_28 = 6;
        wp::int32 var_29;
        const wp::int32 var_30 = 6;
        wp::int32 var_31;
        const wp::int32 var_32 = 6;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        bool var_47;
        bool var_48;
        bool var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        //---------
        // forward
        // def _expand_triangle_vertices_majority_uv3_kernel(                                     <L 989>
        // t = wp.tid()                                                                           <L 999>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1000>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1001>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1002>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1003>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1004>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 1005>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 1006>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 1007>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 1008>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        wp::array_store(var_flat_pos, var_26, var_27);
        // p0 = v0 / 6                                                                            <L 1010>
        var_29 = wp::div(var_4, var_28);
        // p1 = v1 / 6                                                                            <L 1011>
        var_31 = wp::div(var_8, var_30);
        // p2 = v2 / 6                                                                            <L 1012>
        var_33 = wp::div(var_12, var_32);
        // m0 = particle_material[p0]                                                             <L 1013>
        var_34 = wp::address(var_particle_material, var_29);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // m1 = particle_material[p1]                                                             <L 1014>
        var_37 = wp::address(var_particle_material, var_31);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // m2 = particle_material[p2]                                                             <L 1015>
        var_40 = wp::address(var_particle_material, var_33);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // chosen = p0                                                                            <L 1017>
        var_43 = wp::copy(var_29);
        // if m1 == m2:                                                                           <L 1018>
        var_44 = (var_38 == var_41);
        if (var_44) {
            // chosen = p1                                                                        <L 1019>
            var_45 = wp::copy(var_31);
        }
        var_46 = wp::where(var_44, var_45, var_43);
        if (!var_44) {
            // elif m0 == m1 or m0 == m2:                                                         <L 1020>
            var_48 = (var_35 == var_38);
            var_47 = var_48;
            if (!var_47) {
                var_49 = (var_35 == var_41);
                var_47 = var_47 || var_49;
            }
            if (var_47) {
                // chosen = p0                                                                    <L 1021>
                var_50 = wp::copy(var_29);
            }
            var_51 = wp::where(var_47, var_50, var_46);
        }
        var_52 = wp::where(var_44, var_46, var_51);
        // uv = particle_uv3[chosen]                                                              <L 1022>
        var_53 = wp::address(var_particle_uv3, var_52);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // flat_uv3[base + 0] = uv                                                                <L 1023>
        var_57 = wp::add(var_15, var_56);
        wp::array_store(var_flat_uv3, var_57, var_54);
        // flat_uv3[base + 1] = uv                                                                <L 1024>
        var_59 = wp::add(var_15, var_58);
        wp::array_store(var_flat_uv3, var_59, var_54);
        // flat_uv3[base + 2] = uv                                                                <L 1025>
        var_61 = wp::add(var_15, var_60);
        wp::array_store(var_flat_uv3, var_61, var_54);
    }
}



extern "C" __global__ void _expand_triangle_vertices_majority_uv3_kernel_6ad7a9a3_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_uv3,
    wp::array_t<wp::int32> var_particle_material,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_uv3,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_uv3,
    wp::array_t<wp::int32> adj_particle_material,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_uv3)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        const wp::int32 var_28 = 6;
        wp::int32 var_29;
        const wp::int32 var_30 = 6;
        wp::int32 var_31;
        const wp::int32 var_32 = 6;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32* var_40;
        wp::int32 var_41;
        wp::int32 var_42;
        wp::int32 var_43;
        bool var_44;
        wp::int32 var_45;
        wp::int32 var_46;
        bool var_47;
        bool var_48;
        bool var_49;
        wp::int32 var_50;
        wp::int32 var_51;
        wp::int32 var_52;
        wp::vec_t<3, wp::float32>* var_53;
        wp::vec_t<3, wp::float32> var_54;
        wp::vec_t<3, wp::float32> var_55;
        const wp::int32 var_56 = 0;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        const wp::int32 var_60 = 2;
        wp::int32 var_61;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::int32 adj_43 = {};
        bool adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        bool adj_47 = {};
        bool adj_48 = {};
        bool adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        wp::vec_t<3, wp::float32> adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        //---------
        // forward
        // def _expand_triangle_vertices_majority_uv3_kernel(                                     <L 989>
        // t = wp.tid()                                                                           <L 999>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 1000>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 1001>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 1002>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 1003>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 1004>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 1005>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 1006>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        // wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 1007>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        // wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 1008>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        // wp::array_store(var_flat_pos, var_26, var_27);
        // p0 = v0 / 6                                                                            <L 1010>
        var_29 = wp::div(var_4, var_28);
        // p1 = v1 / 6                                                                            <L 1011>
        var_31 = wp::div(var_8, var_30);
        // p2 = v2 / 6                                                                            <L 1012>
        var_33 = wp::div(var_12, var_32);
        // m0 = particle_material[p0]                                                             <L 1013>
        var_34 = wp::address(var_particle_material, var_29);
        var_36 = wp::load(var_34);
        var_35 = wp::copy(var_36);
        // m1 = particle_material[p1]                                                             <L 1014>
        var_37 = wp::address(var_particle_material, var_31);
        var_39 = wp::load(var_37);
        var_38 = wp::copy(var_39);
        // m2 = particle_material[p2]                                                             <L 1015>
        var_40 = wp::address(var_particle_material, var_33);
        var_42 = wp::load(var_40);
        var_41 = wp::copy(var_42);
        // chosen = p0                                                                            <L 1017>
        var_43 = wp::copy(var_29);
        // if m1 == m2:                                                                           <L 1018>
        var_44 = (var_38 == var_41);
        if (var_44) {
            // chosen = p1                                                                        <L 1019>
            var_45 = wp::copy(var_31);
        }
        var_46 = wp::where(var_44, var_45, var_43);
        if (!var_44) {
            // elif m0 == m1 or m0 == m2:                                                         <L 1020>
            var_48 = (var_35 == var_38);
            var_47 = var_48;
            if (!var_47) {
                var_49 = (var_35 == var_41);
                var_47 = var_47 || var_49;
            }
            if (var_47) {
                // chosen = p0                                                                    <L 1021>
                var_50 = wp::copy(var_29);
            }
            var_51 = wp::where(var_47, var_50, var_46);
        }
        var_52 = wp::where(var_44, var_46, var_51);
        // uv = particle_uv3[chosen]                                                              <L 1022>
        var_53 = wp::address(var_particle_uv3, var_52);
        var_55 = wp::load(var_53);
        var_54 = wp::copy(var_55);
        // flat_uv3[base + 0] = uv                                                                <L 1023>
        var_57 = wp::add(var_15, var_56);
        // wp::array_store(var_flat_uv3, var_57, var_54);
        // flat_uv3[base + 1] = uv                                                                <L 1024>
        var_59 = wp::add(var_15, var_58);
        // wp::array_store(var_flat_uv3, var_59, var_54);
        // flat_uv3[base + 2] = uv                                                                <L 1025>
        var_61 = wp::add(var_15, var_60);
        // wp::array_store(var_flat_uv3, var_61, var_54);
        //---------
        // reverse
        wp::adj_array_store(var_flat_uv3, var_61, var_54, adj_flat_uv3, adj_61, adj_54);
        wp::adj_add(var_15, var_60, adj_15, adj_60, adj_61);
        // adj: flat_uv3[base + 2] = uv                                                           <L 1025>
        wp::adj_array_store(var_flat_uv3, var_59, var_54, adj_flat_uv3, adj_59, adj_54);
        wp::adj_add(var_15, var_58, adj_15, adj_58, adj_59);
        // adj: flat_uv3[base + 1] = uv                                                           <L 1024>
        wp::adj_array_store(var_flat_uv3, var_57, var_54, adj_flat_uv3, adj_57, adj_54);
        wp::adj_add(var_15, var_56, adj_15, adj_56, adj_57);
        // adj: flat_uv3[base + 0] = uv                                                           <L 1023>
        wp::adj_copy(var_55, adj_53, adj_54);
        wp::adj_address(var_particle_uv3, var_52, adj_particle_uv3, adj_52, adj_53);
        // adj: uv = particle_uv3[chosen]                                                         <L 1022>
        wp::adj_where(var_44, var_46, var_51, adj_44, adj_46, adj_51, adj_52);
        if (!var_44) {
            wp::adj_where(var_47, var_50, var_46, adj_47, adj_50, adj_46, adj_51);
            if (var_47) {
                wp::adj_copy(var_29, adj_29, adj_50);
                // adj: chosen = p0                                                               <L 1021>
            }
            if (!var_47) {
            }
            // adj: elif m0 == m1 or m0 == m2:                                                    <L 1020>
        }
        wp::adj_where(var_44, var_45, var_43, adj_44, adj_45, adj_43, adj_46);
        if (var_44) {
            wp::adj_copy(var_31, adj_31, adj_45);
            // adj: chosen = p1                                                                   <L 1019>
        }
        // adj: if m1 == m2:                                                                      <L 1018>
        wp::adj_copy(var_29, adj_29, adj_43);
        // adj: chosen = p0                                                                       <L 1017>
        wp::adj_copy(var_42, adj_40, adj_41);
        wp::adj_address(var_particle_material, var_33, adj_particle_material, adj_33, adj_40);
        // adj: m2 = particle_material[p2]                                                        <L 1015>
        wp::adj_copy(var_39, adj_37, adj_38);
        wp::adj_address(var_particle_material, var_31, adj_particle_material, adj_31, adj_37);
        // adj: m1 = particle_material[p1]                                                        <L 1014>
        wp::adj_copy(var_36, adj_34, adj_35);
        wp::adj_address(var_particle_material, var_29, adj_particle_material, adj_29, adj_34);
        // adj: m0 = particle_material[p0]                                                        <L 1013>
        wp::adj_div(var_12, var_32, var_33, adj_12, adj_32, adj_33);
        // adj: p2 = v2 / 6                                                                       <L 1012>
        wp::adj_div(var_8, var_30, var_31, adj_8, adj_30, adj_31);
        // adj: p1 = v1 / 6                                                                       <L 1011>
        wp::adj_div(var_4, var_28, var_29, adj_4, adj_28, adj_29);
        // adj: p0 = v0 / 6                                                                       <L 1010>
        wp::adj_array_store(var_flat_pos, var_26, var_27, adj_flat_pos, adj_26, adj_24);
        wp::adj_add(var_15, var_25, adj_15, adj_25, adj_26);
        wp::adj_address(var_vertex_pos, var_12, adj_vertex_pos, adj_12, adj_24);
        // adj: flat_pos[base + 2] = vertex_pos[v2]                                               <L 1008>
        wp::adj_array_store(var_flat_pos, var_22, var_23, adj_flat_pos, adj_22, adj_20);
        wp::adj_add(var_15, var_21, adj_15, adj_21, adj_22);
        wp::adj_address(var_vertex_pos, var_8, adj_vertex_pos, adj_8, adj_20);
        // adj: flat_pos[base + 1] = vertex_pos[v1]                                               <L 1007>
        wp::adj_array_store(var_flat_pos, var_18, var_19, adj_flat_pos, adj_18, adj_16);
        wp::adj_add(var_15, var_17, adj_15, adj_17, adj_18);
        wp::adj_address(var_vertex_pos, var_4, adj_vertex_pos, adj_4, adj_16);
        // adj: flat_pos[base + 0] = vertex_pos[v0]                                               <L 1006>
        wp::adj_mul(var_0, var_14, adj_0, adj_14, adj_15);
        // adj: base = t * 3                                                                      <L 1005>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 1004>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 1003>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 1002>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 1001>
        }
        // adj: if t >= num_triangles:                                                            <L 1000>
        // adj: t = wp.tid()                                                                      <L 999>
        // adj: def _expand_triangle_vertices_majority_uv3_kernel(                                <L 989>
        continue;
    }
}



extern "C" __global__ void _expand_triangles_kernel_3dcbeaf2_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_centroid_uv3)
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
        wp::vec_t<3, wp::float32>* var_14;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 3;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        const wp::int32 var_27 = 3;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        const wp::float32 var_40 = 1.0;
        const wp::float32 var_41 = 3.0;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        //---------
        // forward
        // def _expand_triangles_kernel(                                                          <L 905>
        // t = wp.tid()                                                                           <L 921>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 922>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 923>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 924>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 925>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 926>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // flat_pos[t * 3 + 0] = vertex_pos[v0]                                                   <L 927>
        var_14 = wp::address(var_vertex_pos, var_4);
        var_16 = wp::mul(var_0, var_15);
        var_18 = wp::add(var_16, var_17);
        var_19 = wp::load(var_14);
        wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[t * 3 + 1] = vertex_pos[v1]                                                   <L 928>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::mul(var_0, var_21);
        var_24 = wp::add(var_22, var_23);
        var_25 = wp::load(var_20);
        wp::array_store(var_flat_pos, var_24, var_25);
        // flat_pos[t * 3 + 2] = vertex_pos[v2]                                                   <L 929>
        var_26 = wp::address(var_vertex_pos, var_12);
        var_28 = wp::mul(var_0, var_27);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::load(var_26);
        wp::array_store(var_flat_pos, var_30, var_31);
        // tri_centroid_uv3[t] = (vertex_uv3[v0] + vertex_uv3[v1] + vertex_uv3[v2]) * (1.0 / 3.0)       <L 930>
        var_32 = wp::address(var_vertex_uv3, var_4);
        var_33 = wp::address(var_vertex_uv3, var_8);
        var_35 = wp::load(var_32);
        var_36 = wp::load(var_33);
        var_34 = wp::add(var_35, var_36);
        var_37 = wp::address(var_vertex_uv3, var_12);
        var_39 = wp::load(var_37);
        var_38 = wp::add(var_34, var_39);
        var_42 = wp::div(var_40, var_41);
        var_43 = wp::mul(var_38, var_42);
        wp::array_store(var_tri_centroid_uv3, var_0, var_43);
    }
}



extern "C" __global__ void _expand_triangles_kernel_3dcbeaf2_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tri_centroid_uv3,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_uv3,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_tri_centroid_uv3)
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
        wp::vec_t<3, wp::float32>* var_14;
        const wp::int32 var_15 = 3;
        wp::int32 var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 3;
        wp::int32 var_22;
        const wp::int32 var_23 = 1;
        wp::int32 var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        const wp::int32 var_27 = 3;
        wp::int32 var_28;
        const wp::int32 var_29 = 2;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        wp::vec_t<3, wp::float32>* var_33;
        wp::vec_t<3, wp::float32> var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32>* var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        const wp::float32 var_40 = 1.0;
        const wp::float32 var_41 = 3.0;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        //---------
        // forward
        // def _expand_triangles_kernel(                                                          <L 905>
        // t = wp.tid()                                                                           <L 921>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 922>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 923>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 924>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 925>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 926>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // flat_pos[t * 3 + 0] = vertex_pos[v0]                                                   <L 927>
        var_14 = wp::address(var_vertex_pos, var_4);
        var_16 = wp::mul(var_0, var_15);
        var_18 = wp::add(var_16, var_17);
        var_19 = wp::load(var_14);
        // wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[t * 3 + 1] = vertex_pos[v1]                                                   <L 928>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::mul(var_0, var_21);
        var_24 = wp::add(var_22, var_23);
        var_25 = wp::load(var_20);
        // wp::array_store(var_flat_pos, var_24, var_25);
        // flat_pos[t * 3 + 2] = vertex_pos[v2]                                                   <L 929>
        var_26 = wp::address(var_vertex_pos, var_12);
        var_28 = wp::mul(var_0, var_27);
        var_30 = wp::add(var_28, var_29);
        var_31 = wp::load(var_26);
        // wp::array_store(var_flat_pos, var_30, var_31);
        // tri_centroid_uv3[t] = (vertex_uv3[v0] + vertex_uv3[v1] + vertex_uv3[v2]) * (1.0 / 3.0)       <L 930>
        var_32 = wp::address(var_vertex_uv3, var_4);
        var_33 = wp::address(var_vertex_uv3, var_8);
        var_35 = wp::load(var_32);
        var_36 = wp::load(var_33);
        var_34 = wp::add(var_35, var_36);
        var_37 = wp::address(var_vertex_uv3, var_12);
        var_39 = wp::load(var_37);
        var_38 = wp::add(var_34, var_39);
        var_42 = wp::div(var_40, var_41);
        var_43 = wp::mul(var_38, var_42);
        // wp::array_store(var_tri_centroid_uv3, var_0, var_43);
        //---------
        // reverse
        wp::adj_array_store(var_tri_centroid_uv3, var_0, var_43, adj_tri_centroid_uv3, adj_0, adj_43);
        wp::adj_mul(var_38, var_42, adj_38, adj_42, adj_43);
        wp::adj_div(var_40, var_41, var_42, adj_40, adj_41, adj_42);
        wp::adj_add(var_34, var_39, adj_34, adj_37, adj_38);
        wp::adj_address(var_vertex_uv3, var_12, adj_vertex_uv3, adj_12, adj_37);
        wp::adj_add(var_35, var_36, adj_32, adj_33, adj_34);
        wp::adj_address(var_vertex_uv3, var_8, adj_vertex_uv3, adj_8, adj_33);
        wp::adj_address(var_vertex_uv3, var_4, adj_vertex_uv3, adj_4, adj_32);
        // adj: tri_centroid_uv3[t] = (vertex_uv3[v0] + vertex_uv3[v1] + vertex_uv3[v2]) * (1.0 / 3.0)  <L 930>
        wp::adj_array_store(var_flat_pos, var_30, var_31, adj_flat_pos, adj_30, adj_26);
        wp::adj_add(var_28, var_29, adj_28, adj_29, adj_30);
        wp::adj_mul(var_0, var_27, adj_0, adj_27, adj_28);
        wp::adj_address(var_vertex_pos, var_12, adj_vertex_pos, adj_12, adj_26);
        // adj: flat_pos[t * 3 + 2] = vertex_pos[v2]                                              <L 929>
        wp::adj_array_store(var_flat_pos, var_24, var_25, adj_flat_pos, adj_24, adj_20);
        wp::adj_add(var_22, var_23, adj_22, adj_23, adj_24);
        wp::adj_mul(var_0, var_21, adj_0, adj_21, adj_22);
        wp::adj_address(var_vertex_pos, var_8, adj_vertex_pos, adj_8, adj_20);
        // adj: flat_pos[t * 3 + 1] = vertex_pos[v1]                                              <L 928>
        wp::adj_array_store(var_flat_pos, var_18, var_19, adj_flat_pos, adj_18, adj_14);
        wp::adj_add(var_16, var_17, adj_16, adj_17, adj_18);
        wp::adj_mul(var_0, var_15, adj_0, adj_15, adj_16);
        wp::adj_address(var_vertex_pos, var_4, adj_vertex_pos, adj_4, adj_14);
        // adj: flat_pos[t * 3 + 0] = vertex_pos[v0]                                              <L 927>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 926>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 925>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 924>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 923>
        }
        // adj: if t >= num_triangles:                                                            <L 922>
        // adj: t = wp.tid()                                                                      <L 921>
        // adj: def _expand_triangles_kernel(                                                     <L 905>
        continue;
    }
}



extern "C" __global__ void _gather_grab_constraint_lines_kernel_5dd2aa98_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grab_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::vec_t<3, wp::float32> var_pull_target,
    wp::array_t<wp::vec_t<3, wp::float32>> var_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ends)
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
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        //---------
        // forward
        // def _gather_grab_constraint_lines_kernel(                                              <L 1657>
        // tid = wp.tid()                                                                         <L 1664>
        var_0 = builtin_tid1d();
        // particle_idx = grab_indices[tid]                                                       <L 1665>
        var_1 = wp::address(var_grab_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if particle_idx < 0:                                                                   <L 1666>
        var_5 = (var_2 < var_4);
        if (var_5) {
            // starts[tid] = pull_target                                                          <L 1667>
            wp::array_store(var_starts, var_0, var_pull_target);
            // ends[tid] = pull_target                                                            <L 1668>
            wp::array_store(var_ends, var_0, var_pull_target);
            // return                                                                             <L 1669>
            continue;
        }
        // starts[tid] = pull_target                                                              <L 1670>
        wp::array_store(var_starts, var_0, var_pull_target);
        // ends[tid] = particle_q[particle_idx]                                                   <L 1671>
        var_6 = wp::address(var_particle_q, var_2);
        var_7 = wp::load(var_6);
        wp::array_store(var_ends, var_0, var_7);
    }
}



extern "C" __global__ void _gather_grab_constraint_lines_kernel_5dd2aa98_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grab_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::vec_t<3, wp::float32> var_pull_target,
    wp::array_t<wp::vec_t<3, wp::float32>> var_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ends,
    wp::array_t<wp::int32> adj_grab_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::vec_t<3, wp::float32> adj_pull_target,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_ends)
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
        wp::vec_t<3, wp::float32>* var_6;
        wp::vec_t<3, wp::float32> var_7;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        bool adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        //---------
        // forward
        // def _gather_grab_constraint_lines_kernel(                                              <L 1657>
        // tid = wp.tid()                                                                         <L 1664>
        var_0 = builtin_tid1d();
        // particle_idx = grab_indices[tid]                                                       <L 1665>
        var_1 = wp::address(var_grab_indices, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if particle_idx < 0:                                                                   <L 1666>
        var_5 = (var_2 < var_4);
        if (var_5) {
            // starts[tid] = pull_target                                                          <L 1667>
            // wp::array_store(var_starts, var_0, var_pull_target);
            // ends[tid] = pull_target                                                            <L 1668>
            // wp::array_store(var_ends, var_0, var_pull_target);
            // return                                                                             <L 1669>
            goto label0;
        }
        // starts[tid] = pull_target                                                              <L 1670>
        // wp::array_store(var_starts, var_0, var_pull_target);
        // ends[tid] = particle_q[particle_idx]                                                   <L 1671>
        var_6 = wp::address(var_particle_q, var_2);
        var_7 = wp::load(var_6);
        // wp::array_store(var_ends, var_0, var_7);
        //---------
        // reverse
        wp::adj_array_store(var_ends, var_0, var_7, adj_ends, adj_0, adj_6);
        wp::adj_address(var_particle_q, var_2, adj_particle_q, adj_2, adj_6);
        // adj: ends[tid] = particle_q[particle_idx]                                              <L 1671>
        wp::adj_array_store(var_starts, var_0, var_pull_target, adj_starts, adj_0, adj_pull_target);
        // adj: starts[tid] = pull_target                                                         <L 1670>
        if (var_5) {
            label0:;
            // adj: return                                                                        <L 1669>
            wp::adj_array_store(var_ends, var_0, var_pull_target, adj_ends, adj_0, adj_pull_target);
            // adj: ends[tid] = pull_target                                                       <L 1668>
            wp::adj_array_store(var_starts, var_0, var_pull_target, adj_starts, adj_0, adj_pull_target);
            // adj: starts[tid] = pull_target                                                     <L 1667>
        }
        // adj: if particle_idx < 0:                                                              <L 1666>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_grab_indices, var_0, adj_grab_indices, adj_0, adj_1);
        // adj: particle_idx = grab_indices[tid]                                                  <L 1665>
        // adj: tid = wp.tid()                                                                    <L 1664>
        // adj: def _gather_grab_constraint_lines_kernel(                                         <L 1657>
        continue;
    }
}



extern "C" __global__ void _expand_triangle_vertices_kernel_4cabe833_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_uv3)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        const wp::int32 var_37 = 2;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        //---------
        // forward
        // def _expand_triangle_vertices_kernel(                                                  <L 934>
        // t = wp.tid()                                                                           <L 943>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 944>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 945>
            continue;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 946>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 947>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 948>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 949>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 950>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 951>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 952>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        wp::array_store(var_flat_pos, var_26, var_27);
        // flat_uv3[base + 0] = vertex_uv3[v0]                                                    <L 953>
        var_28 = wp::address(var_vertex_uv3, var_4);
        var_30 = wp::add(var_15, var_29);
        var_31 = wp::load(var_28);
        wp::array_store(var_flat_uv3, var_30, var_31);
        // flat_uv3[base + 1] = vertex_uv3[v1]                                                    <L 954>
        var_32 = wp::address(var_vertex_uv3, var_8);
        var_34 = wp::add(var_15, var_33);
        var_35 = wp::load(var_32);
        wp::array_store(var_flat_uv3, var_34, var_35);
        // flat_uv3[base + 2] = vertex_uv3[v2]                                                    <L 955>
        var_36 = wp::address(var_vertex_uv3, var_12);
        var_38 = wp::add(var_15, var_37);
        var_39 = wp::load(var_36);
        wp::array_store(var_flat_uv3, var_38, var_39);
    }
}



extern "C" __global__ void _expand_triangle_vertices_kernel_4cabe833_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::int32 var_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> var_flat_uv3,
    wp::array_t<wp::int32> adj_tri_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_uv3,
    wp::int32 adj_num_triangles,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_flat_uv3)
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
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 0;
        wp::int32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32>* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32>* var_24;
        const wp::int32 var_25 = 2;
        wp::int32 var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32>* var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32>* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32>* var_36;
        const wp::int32 var_37 = 2;
        wp::int32 var_38;
        wp::vec_t<3, wp::float32> var_39;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        bool adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
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
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::int32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        //---------
        // forward
        // def _expand_triangle_vertices_kernel(                                                  <L 934>
        // t = wp.tid()                                                                           <L 943>
        var_0 = builtin_tid1d();
        // if t >= num_triangles:                                                                 <L 944>
        var_1 = (var_0 >= var_num_triangles);
        if (var_1) {
            // return                                                                             <L 945>
            goto label0;
        }
        // v0 = tri_indices[t, 0]                                                                 <L 946>
        var_3 = wp::address(var_tri_indices, var_0, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // v1 = tri_indices[t, 1]                                                                 <L 947>
        var_7 = wp::address(var_tri_indices, var_0, var_6);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // v2 = tri_indices[t, 2]                                                                 <L 948>
        var_11 = wp::address(var_tri_indices, var_0, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // base = t * 3                                                                           <L 949>
        var_15 = wp::mul(var_0, var_14);
        // flat_pos[base + 0] = vertex_pos[v0]                                                    <L 950>
        var_16 = wp::address(var_vertex_pos, var_4);
        var_18 = wp::add(var_15, var_17);
        var_19 = wp::load(var_16);
        // wp::array_store(var_flat_pos, var_18, var_19);
        // flat_pos[base + 1] = vertex_pos[v1]                                                    <L 951>
        var_20 = wp::address(var_vertex_pos, var_8);
        var_22 = wp::add(var_15, var_21);
        var_23 = wp::load(var_20);
        // wp::array_store(var_flat_pos, var_22, var_23);
        // flat_pos[base + 2] = vertex_pos[v2]                                                    <L 952>
        var_24 = wp::address(var_vertex_pos, var_12);
        var_26 = wp::add(var_15, var_25);
        var_27 = wp::load(var_24);
        // wp::array_store(var_flat_pos, var_26, var_27);
        // flat_uv3[base + 0] = vertex_uv3[v0]                                                    <L 953>
        var_28 = wp::address(var_vertex_uv3, var_4);
        var_30 = wp::add(var_15, var_29);
        var_31 = wp::load(var_28);
        // wp::array_store(var_flat_uv3, var_30, var_31);
        // flat_uv3[base + 1] = vertex_uv3[v1]                                                    <L 954>
        var_32 = wp::address(var_vertex_uv3, var_8);
        var_34 = wp::add(var_15, var_33);
        var_35 = wp::load(var_32);
        // wp::array_store(var_flat_uv3, var_34, var_35);
        // flat_uv3[base + 2] = vertex_uv3[v2]                                                    <L 955>
        var_36 = wp::address(var_vertex_uv3, var_12);
        var_38 = wp::add(var_15, var_37);
        var_39 = wp::load(var_36);
        // wp::array_store(var_flat_uv3, var_38, var_39);
        //---------
        // reverse
        wp::adj_array_store(var_flat_uv3, var_38, var_39, adj_flat_uv3, adj_38, adj_36);
        wp::adj_add(var_15, var_37, adj_15, adj_37, adj_38);
        wp::adj_address(var_vertex_uv3, var_12, adj_vertex_uv3, adj_12, adj_36);
        // adj: flat_uv3[base + 2] = vertex_uv3[v2]                                               <L 955>
        wp::adj_array_store(var_flat_uv3, var_34, var_35, adj_flat_uv3, adj_34, adj_32);
        wp::adj_add(var_15, var_33, adj_15, adj_33, adj_34);
        wp::adj_address(var_vertex_uv3, var_8, adj_vertex_uv3, adj_8, adj_32);
        // adj: flat_uv3[base + 1] = vertex_uv3[v1]                                               <L 954>
        wp::adj_array_store(var_flat_uv3, var_30, var_31, adj_flat_uv3, adj_30, adj_28);
        wp::adj_add(var_15, var_29, adj_15, adj_29, adj_30);
        wp::adj_address(var_vertex_uv3, var_4, adj_vertex_uv3, adj_4, adj_28);
        // adj: flat_uv3[base + 0] = vertex_uv3[v0]                                               <L 953>
        wp::adj_array_store(var_flat_pos, var_26, var_27, adj_flat_pos, adj_26, adj_24);
        wp::adj_add(var_15, var_25, adj_15, adj_25, adj_26);
        wp::adj_address(var_vertex_pos, var_12, adj_vertex_pos, adj_12, adj_24);
        // adj: flat_pos[base + 2] = vertex_pos[v2]                                               <L 952>
        wp::adj_array_store(var_flat_pos, var_22, var_23, adj_flat_pos, adj_22, adj_20);
        wp::adj_add(var_15, var_21, adj_15, adj_21, adj_22);
        wp::adj_address(var_vertex_pos, var_8, adj_vertex_pos, adj_8, adj_20);
        // adj: flat_pos[base + 1] = vertex_pos[v1]                                               <L 951>
        wp::adj_array_store(var_flat_pos, var_18, var_19, adj_flat_pos, adj_18, adj_16);
        wp::adj_add(var_15, var_17, adj_15, adj_17, adj_18);
        wp::adj_address(var_vertex_pos, var_4, adj_vertex_pos, adj_4, adj_16);
        // adj: flat_pos[base + 0] = vertex_pos[v0]                                               <L 950>
        wp::adj_mul(var_0, var_14, adj_0, adj_14, adj_15);
        // adj: base = t * 3                                                                      <L 949>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_tri_indices, var_0, var_10, adj_tri_indices, adj_0, adj_10, adj_11);
        // adj: v2 = tri_indices[t, 2]                                                            <L 948>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_tri_indices, var_0, var_6, adj_tri_indices, adj_0, adj_6, adj_7);
        // adj: v1 = tri_indices[t, 1]                                                            <L 947>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_tri_indices, var_0, var_2, adj_tri_indices, adj_0, adj_2, adj_3);
        // adj: v0 = tri_indices[t, 0]                                                            <L 946>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 945>
        }
        // adj: if t >= num_triangles:                                                            <L 944>
        // adj: t = wp.tid()                                                                      <L 943>
        // adj: def _expand_triangle_vertices_kernel(                                             <L 934>
        continue;
    }
}

