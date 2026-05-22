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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_render.py:29
static CUDA_CALLABLE wp::mat_t<3, 3, wp::float32> _orthonormalize_0(
    wp::vec_t<3, wp::float32> var_ax,
    wp::vec_t<3, wp::float32> var_ay,
    wp::vec_t<3, wp::float32> var_az)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-12;
    bool var_2;
    const wp::float32 var_3 = 1.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-12;
    bool var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.9;
    bool var_20;
    const wp::float32 var_21 = 1.0;
    const wp::float32 var_22 = 0.0;
    const wp::float32 var_23 = 0.0;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::float32 var_30 = 0.0;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = 0.0;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    wp::float32 var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    wp::mat_t<3, 3, wp::float32> var_63;
    //---------
    // forward
    // def _orthonormalize(ax: wp.vec3, ay: wp.vec3, az: wp.vec3) -> wp.mat33:                <L 30>
    // nx = wp.length(ax)                                                                     <L 33>
    var_0 = wp::length(var_ax);
    // if nx < 1.0e-12:                                                                       <L 34>
    var_2 = (var_0 < var_1);
    if (var_2) {
        // ax = wp.vec3(1.0, 0.0, 0.0)                                                        <L 35>
        var_6 = wp::vec_t<3, wp::float32>(var_3, var_4, var_5);
    }
    var_7 = wp::where(var_2, var_6, var_ax);
    if (!var_2) {
        // ax = ax / nx                                                                       <L 37>
        var_8 = wp::div(var_7, var_0);
    }
    var_9 = wp::where(var_2, var_7, var_8);
    // ay = ay - ax * wp.dot(ax, ay)                                                          <L 38>
    var_10 = wp::dot(var_9, var_ay);
    var_11 = wp::mul(var_9, var_10);
    var_12 = wp::sub(var_ay, var_11);
    // ny = wp.length(ay)                                                                     <L 39>
    var_13 = wp::length(var_12);
    // if ny < 1.0e-12:                                                                       <L 40>
    var_15 = (var_13 < var_14);
    if (var_15) {
        // if wp.abs(ax[0]) < 0.9:                                                            <L 43>
        var_17 = wp::extract(var_9, var_16);
        var_18 = wp::abs(var_17);
        var_20 = (var_18 < var_19);
        if (var_20) {
            // ay = wp.vec3(1.0, 0.0, 0.0) - ax * ax[0]                                       <L 44>
            var_24 = wp::vec_t<3, wp::float32>(var_21, var_22, var_23);
            var_26 = wp::extract(var_9, var_25);
            var_27 = wp::mul(var_9, var_26);
            var_28 = wp::sub(var_24, var_27);
        }
        var_29 = wp::where(var_20, var_28, var_12);
        if (!var_20) {
            // ay = wp.vec3(0.0, 1.0, 0.0) - ax * ax[1]                                       <L 46>
            var_33 = wp::vec_t<3, wp::float32>(var_30, var_31, var_32);
            var_35 = wp::extract(var_9, var_34);
            var_36 = wp::mul(var_9, var_35);
            var_37 = wp::sub(var_33, var_36);
        }
        var_38 = wp::where(var_20, var_29, var_37);
        // ay = ay / wp.length(ay)                                                            <L 47>
        var_39 = wp::length(var_38);
        var_40 = wp::div(var_38, var_39);
    }
    var_41 = wp::where(var_15, var_40, var_12);
    if (!var_15) {
        // ay = ay / ny                                                                       <L 49>
        var_42 = wp::div(var_41, var_13);
    }
    var_43 = wp::where(var_15, var_41, var_42);
    // az = wp.cross(ax, ay)                                                                  <L 50>
    var_44 = wp::cross(var_9, var_43);
    // return wp.mat33(                                                                       <L 53>
    // ax[0], ax[1], ax[2],                                                                   <L 54>
    var_46 = wp::extract(var_9, var_45);
    var_48 = wp::extract(var_9, var_47);
    var_50 = wp::extract(var_9, var_49);
    // ay[0], ay[1], ay[2],                                                                   <L 55>
    var_52 = wp::extract(var_43, var_51);
    var_54 = wp::extract(var_43, var_53);
    var_56 = wp::extract(var_43, var_55);
    // az[0], az[1], az[2],                                                                   <L 56>
    var_58 = wp::extract(var_44, var_57);
    var_60 = wp::extract(var_44, var_59);
    var_62 = wp::extract(var_44, var_61);
    var_63 = wp::mat_t<3, 3, wp::float32>(var_46, var_48, var_50, var_52, var_54, var_56, var_58, var_60, var_62);
    return var_63;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_render.py:60
static CUDA_CALLABLE wp::float32 _constraint_abs_strain_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::float32 var_rest_length)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-12;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 1.0;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // forward
    // def _constraint_abs_strain(a: wp.vec3, b: wp.vec3, rest_length: float) -> float:       <L 61>
    // if rest_length <= 1.0e-12:                                                             <L 62>
    var_1 = (var_rest_length <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 63>
        return var_2;
    }
    // return wp.abs((wp.length(b - a) / rest_length) - 1.0)                                  <L 64>
    var_3 = wp::sub(var_b, var_a);
    var_4 = wp::length(var_3);
    var_5 = wp::div(var_4, var_rest_length);
    var_7 = wp::sub(var_5, var_6);
    var_8 = wp::abs(var_7);
    return var_8;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_render.py:67
static CUDA_CALLABLE wp::float32 _cell_max_abs_strain_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3,
    wp::vec_t<3, wp::float32> var_p4,
    wp::vec_t<3, wp::float32> var_p5,
    wp::vec_t<3, wp::float32> var_p6,
    wp::vec_t<3, wp::float32> var_p7,
    wp::float32 var_voxel_size)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1.4142135623730951;
    wp::float32 var_2;
    const wp::float32 var_3 = 1.7320508075688772;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    //---------
    // forward
    // def _cell_max_abs_strain(                                                              <L 68>
    // edge = voxel_size                                                                      <L 79>
    var_0 = wp::copy(var_voxel_size);
    // face = voxel_size * 1.4142135623730951                                                 <L 80>
    var_2 = wp::mul(var_voxel_size, var_1);
    // body = voxel_size * 1.7320508075688772                                                 <L 81>
    var_4 = wp::mul(var_voxel_size, var_3);
    // s = float(0.0)                                                                         <L 82>
    var_6 = wp::float(var_5);
    // s = wp.max(s, _constraint_abs_strain(p0, p1, edge))                                    <L 84>
    var_7 = _constraint_abs_strain_0(var_p0, var_p1, var_0);
    var_8 = wp::max(var_6, var_7);
    // s = wp.max(s, _constraint_abs_strain(p1, p2, edge))                                    <L 85>
    var_9 = _constraint_abs_strain_0(var_p1, var_p2, var_0);
    var_10 = wp::max(var_8, var_9);
    // s = wp.max(s, _constraint_abs_strain(p2, p3, edge))                                    <L 86>
    var_11 = _constraint_abs_strain_0(var_p2, var_p3, var_0);
    var_12 = wp::max(var_10, var_11);
    // s = wp.max(s, _constraint_abs_strain(p3, p0, edge))                                    <L 87>
    var_13 = _constraint_abs_strain_0(var_p3, var_p0, var_0);
    var_14 = wp::max(var_12, var_13);
    // s = wp.max(s, _constraint_abs_strain(p4, p5, edge))                                    <L 88>
    var_15 = _constraint_abs_strain_0(var_p4, var_p5, var_0);
    var_16 = wp::max(var_14, var_15);
    // s = wp.max(s, _constraint_abs_strain(p5, p6, edge))                                    <L 89>
    var_17 = _constraint_abs_strain_0(var_p5, var_p6, var_0);
    var_18 = wp::max(var_16, var_17);
    // s = wp.max(s, _constraint_abs_strain(p6, p7, edge))                                    <L 90>
    var_19 = _constraint_abs_strain_0(var_p6, var_p7, var_0);
    var_20 = wp::max(var_18, var_19);
    // s = wp.max(s, _constraint_abs_strain(p7, p4, edge))                                    <L 91>
    var_21 = _constraint_abs_strain_0(var_p7, var_p4, var_0);
    var_22 = wp::max(var_20, var_21);
    // s = wp.max(s, _constraint_abs_strain(p0, p4, edge))                                    <L 92>
    var_23 = _constraint_abs_strain_0(var_p0, var_p4, var_0);
    var_24 = wp::max(var_22, var_23);
    // s = wp.max(s, _constraint_abs_strain(p1, p5, edge))                                    <L 93>
    var_25 = _constraint_abs_strain_0(var_p1, var_p5, var_0);
    var_26 = wp::max(var_24, var_25);
    // s = wp.max(s, _constraint_abs_strain(p2, p6, edge))                                    <L 94>
    var_27 = _constraint_abs_strain_0(var_p2, var_p6, var_0);
    var_28 = wp::max(var_26, var_27);
    // s = wp.max(s, _constraint_abs_strain(p3, p7, edge))                                    <L 95>
    var_29 = _constraint_abs_strain_0(var_p3, var_p7, var_0);
    var_30 = wp::max(var_28, var_29);
    // s = wp.max(s, _constraint_abs_strain(p0, p2, face))                                    <L 97>
    var_31 = _constraint_abs_strain_0(var_p0, var_p2, var_2);
    var_32 = wp::max(var_30, var_31);
    // s = wp.max(s, _constraint_abs_strain(p1, p3, face))                                    <L 98>
    var_33 = _constraint_abs_strain_0(var_p1, var_p3, var_2);
    var_34 = wp::max(var_32, var_33);
    // s = wp.max(s, _constraint_abs_strain(p4, p6, face))                                    <L 99>
    var_35 = _constraint_abs_strain_0(var_p4, var_p6, var_2);
    var_36 = wp::max(var_34, var_35);
    // s = wp.max(s, _constraint_abs_strain(p5, p7, face))                                    <L 100>
    var_37 = _constraint_abs_strain_0(var_p5, var_p7, var_2);
    var_38 = wp::max(var_36, var_37);
    // s = wp.max(s, _constraint_abs_strain(p0, p5, face))                                    <L 101>
    var_39 = _constraint_abs_strain_0(var_p0, var_p5, var_2);
    var_40 = wp::max(var_38, var_39);
    // s = wp.max(s, _constraint_abs_strain(p1, p4, face))                                    <L 102>
    var_41 = _constraint_abs_strain_0(var_p1, var_p4, var_2);
    var_42 = wp::max(var_40, var_41);
    // s = wp.max(s, _constraint_abs_strain(p3, p6, face))                                    <L 103>
    var_43 = _constraint_abs_strain_0(var_p3, var_p6, var_2);
    var_44 = wp::max(var_42, var_43);
    // s = wp.max(s, _constraint_abs_strain(p2, p7, face))                                    <L 104>
    var_45 = _constraint_abs_strain_0(var_p2, var_p7, var_2);
    var_46 = wp::max(var_44, var_45);
    // s = wp.max(s, _constraint_abs_strain(p0, p7, face))                                    <L 105>
    var_47 = _constraint_abs_strain_0(var_p0, var_p7, var_2);
    var_48 = wp::max(var_46, var_47);
    // s = wp.max(s, _constraint_abs_strain(p3, p4, face))                                    <L 106>
    var_49 = _constraint_abs_strain_0(var_p3, var_p4, var_2);
    var_50 = wp::max(var_48, var_49);
    // s = wp.max(s, _constraint_abs_strain(p1, p6, face))                                    <L 107>
    var_51 = _constraint_abs_strain_0(var_p1, var_p6, var_2);
    var_52 = wp::max(var_50, var_51);
    // s = wp.max(s, _constraint_abs_strain(p2, p5, face))                                    <L 108>
    var_53 = _constraint_abs_strain_0(var_p2, var_p5, var_2);
    var_54 = wp::max(var_52, var_53);
    // s = wp.max(s, _constraint_abs_strain(p0, p6, body))                                    <L 110>
    var_55 = _constraint_abs_strain_0(var_p0, var_p6, var_4);
    var_56 = wp::max(var_54, var_55);
    // s = wp.max(s, _constraint_abs_strain(p1, p7, body))                                    <L 111>
    var_57 = _constraint_abs_strain_0(var_p1, var_p7, var_4);
    var_58 = wp::max(var_56, var_57);
    // s = wp.max(s, _constraint_abs_strain(p2, p4, body))                                    <L 112>
    var_59 = _constraint_abs_strain_0(var_p2, var_p4, var_4);
    var_60 = wp::max(var_58, var_59);
    // s = wp.max(s, _constraint_abs_strain(p3, p5, body))                                    <L 113>
    var_61 = _constraint_abs_strain_0(var_p3, var_p5, var_4);
    var_62 = wp::max(var_60, var_61);
    // return s                                                                               <L 114>
    return var_62;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_render.py:29
static CUDA_CALLABLE void adj__orthonormalize_0(
    wp::vec_t<3, wp::float32> var_ax,
    wp::vec_t<3, wp::float32> var_ay,
    wp::vec_t<3, wp::float32> var_az,
    wp::vec_t<3, wp::float32> & adj_ax,
    wp::vec_t<3, wp::float32> & adj_ay,
    wp::vec_t<3, wp::float32> & adj_az,
    wp::mat_t<3, 3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-12;
    bool var_2;
    const wp::float32 var_3 = 1.0;
    const wp::float32 var_4 = 0.0;
    const wp::float32 var_5 = 0.0;
    wp::vec_t<3, wp::float32> var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::float32 var_10;
    wp::vec_t<3, wp::float32> var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::float32 var_13;
    const wp::float32 var_14 = 1e-12;
    bool var_15;
    const wp::int32 var_16 = 0;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.9;
    bool var_20;
    const wp::float32 var_21 = 1.0;
    const wp::float32 var_22 = 0.0;
    const wp::float32 var_23 = 0.0;
    wp::vec_t<3, wp::float32> var_24;
    const wp::int32 var_25 = 0;
    wp::float32 var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    const wp::float32 var_30 = 0.0;
    const wp::float32 var_31 = 1.0;
    const wp::float32 var_32 = 0.0;
    wp::vec_t<3, wp::float32> var_33;
    const wp::int32 var_34 = 1;
    wp::float32 var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::float32 var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32> var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::vec_t<3, wp::float32> var_44;
    const wp::int32 var_45 = 0;
    wp::float32 var_46;
    const wp::int32 var_47 = 1;
    wp::float32 var_48;
    const wp::int32 var_49 = 2;
    wp::float32 var_50;
    const wp::int32 var_51 = 0;
    wp::float32 var_52;
    const wp::int32 var_53 = 1;
    wp::float32 var_54;
    const wp::int32 var_55 = 2;
    wp::float32 var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    const wp::int32 var_59 = 1;
    wp::float32 var_60;
    const wp::int32 var_61 = 2;
    wp::float32 var_62;
    wp::mat_t<3, 3, wp::float32> var_63;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    bool adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::float32 adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    bool adj_15 = {};
    wp::int32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    bool adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::int32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    wp::float32 adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::float32 adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    wp::vec_t<3, wp::float32> adj_43 = {};
    wp::vec_t<3, wp::float32> adj_44 = {};
    wp::int32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::int32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::int32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::int32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::int32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::int32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::int32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::int32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::mat_t<3, 3, wp::float32> adj_63 = {};
    //---------
    // forward
    // def _orthonormalize(ax: wp.vec3, ay: wp.vec3, az: wp.vec3) -> wp.mat33:                <L 30>
    // nx = wp.length(ax)                                                                     <L 33>
    var_0 = wp::length(var_ax);
    // if nx < 1.0e-12:                                                                       <L 34>
    var_2 = (var_0 < var_1);
    if (var_2) {
        // ax = wp.vec3(1.0, 0.0, 0.0)                                                        <L 35>
        var_6 = wp::vec_t<3, wp::float32>(var_3, var_4, var_5);
    }
    var_7 = wp::where(var_2, var_6, var_ax);
    if (!var_2) {
        // ax = ax / nx                                                                       <L 37>
        var_8 = wp::div(var_7, var_0);
    }
    var_9 = wp::where(var_2, var_7, var_8);
    // ay = ay - ax * wp.dot(ax, ay)                                                          <L 38>
    var_10 = wp::dot(var_9, var_ay);
    var_11 = wp::mul(var_9, var_10);
    var_12 = wp::sub(var_ay, var_11);
    // ny = wp.length(ay)                                                                     <L 39>
    var_13 = wp::length(var_12);
    // if ny < 1.0e-12:                                                                       <L 40>
    var_15 = (var_13 < var_14);
    if (var_15) {
        // if wp.abs(ax[0]) < 0.9:                                                            <L 43>
        var_17 = wp::extract(var_9, var_16);
        var_18 = wp::abs(var_17);
        var_20 = (var_18 < var_19);
        if (var_20) {
            // ay = wp.vec3(1.0, 0.0, 0.0) - ax * ax[0]                                       <L 44>
            var_24 = wp::vec_t<3, wp::float32>(var_21, var_22, var_23);
            var_26 = wp::extract(var_9, var_25);
            var_27 = wp::mul(var_9, var_26);
            var_28 = wp::sub(var_24, var_27);
        }
        var_29 = wp::where(var_20, var_28, var_12);
        if (!var_20) {
            // ay = wp.vec3(0.0, 1.0, 0.0) - ax * ax[1]                                       <L 46>
            var_33 = wp::vec_t<3, wp::float32>(var_30, var_31, var_32);
            var_35 = wp::extract(var_9, var_34);
            var_36 = wp::mul(var_9, var_35);
            var_37 = wp::sub(var_33, var_36);
        }
        var_38 = wp::where(var_20, var_29, var_37);
        // ay = ay / wp.length(ay)                                                            <L 47>
        var_39 = wp::length(var_38);
        var_40 = wp::div(var_38, var_39);
    }
    var_41 = wp::where(var_15, var_40, var_12);
    if (!var_15) {
        // ay = ay / ny                                                                       <L 49>
        var_42 = wp::div(var_41, var_13);
    }
    var_43 = wp::where(var_15, var_41, var_42);
    // az = wp.cross(ax, ay)                                                                  <L 50>
    var_44 = wp::cross(var_9, var_43);
    // return wp.mat33(                                                                       <L 53>
    // ax[0], ax[1], ax[2],                                                                   <L 54>
    var_46 = wp::extract(var_9, var_45);
    var_48 = wp::extract(var_9, var_47);
    var_50 = wp::extract(var_9, var_49);
    // ay[0], ay[1], ay[2],                                                                   <L 55>
    var_52 = wp::extract(var_43, var_51);
    var_54 = wp::extract(var_43, var_53);
    var_56 = wp::extract(var_43, var_55);
    // az[0], az[1], az[2],                                                                   <L 56>
    var_58 = wp::extract(var_44, var_57);
    var_60 = wp::extract(var_44, var_59);
    var_62 = wp::extract(var_44, var_61);
    var_63 = wp::mat_t<3, 3, wp::float32>(var_46, var_48, var_50, var_52, var_54, var_56, var_58, var_60, var_62);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_63 += adj_ret;
    wp::adj_mat_t(var_46, var_48, var_50, var_52, var_54, var_56, var_58, var_60, var_62, adj_46, adj_48, adj_50, adj_52, adj_54, adj_56, adj_58, adj_60, adj_62, adj_63);
    wp::adj_extract(var_44, var_61, adj_44, adj_61, adj_62);
    wp::adj_extract(var_44, var_59, adj_44, adj_59, adj_60);
    wp::adj_extract(var_44, var_57, adj_44, adj_57, adj_58);
    // adj: az[0], az[1], az[2],                                                              <L 56>
    wp::adj_extract(var_43, var_55, adj_43, adj_55, adj_56);
    wp::adj_extract(var_43, var_53, adj_43, adj_53, adj_54);
    wp::adj_extract(var_43, var_51, adj_43, adj_51, adj_52);
    // adj: ay[0], ay[1], ay[2],                                                              <L 55>
    wp::adj_extract(var_9, var_49, adj_9, adj_49, adj_50);
    wp::adj_extract(var_9, var_47, adj_9, adj_47, adj_48);
    wp::adj_extract(var_9, var_45, adj_9, adj_45, adj_46);
    // adj: ax[0], ax[1], ax[2],                                                              <L 54>
    // adj: return wp.mat33(                                                                  <L 53>
    wp::adj_cross(var_9, var_43, adj_9, adj_43, adj_44);
    // adj: az = wp.cross(ax, ay)                                                             <L 50>
    wp::adj_where(var_15, var_41, var_42, adj_15, adj_41, adj_42, adj_43);
    if (!var_15) {
        wp::adj_div(var_41, var_13, adj_41, adj_13, adj_42);
        // adj: ay = ay / ny                                                                  <L 49>
    }
    wp::adj_where(var_15, var_40, var_12, adj_15, adj_40, adj_12, adj_41);
    if (var_15) {
        wp::adj_div(var_38, var_39, adj_38, adj_39, adj_40);
        wp::adj_length(var_38, var_39, adj_38, adj_39);
        // adj: ay = ay / wp.length(ay)                                                       <L 47>
        wp::adj_where(var_20, var_29, var_37, adj_20, adj_29, adj_37, adj_38);
        if (!var_20) {
            wp::adj_sub(var_33, var_36, adj_33, adj_36, adj_37);
            wp::adj_mul(var_9, var_35, adj_9, adj_35, adj_36);
            wp::adj_extract(var_9, var_34, adj_9, adj_34, adj_35);
            wp::adj_vec_t(var_30, var_31, var_32, adj_30, adj_31, adj_32, adj_33);
            // adj: ay = wp.vec3(0.0, 1.0, 0.0) - ax * ax[1]                                  <L 46>
        }
        wp::adj_where(var_20, var_28, var_12, adj_20, adj_28, adj_12, adj_29);
        if (var_20) {
            wp::adj_sub(var_24, var_27, adj_24, adj_27, adj_28);
            wp::adj_mul(var_9, var_26, adj_9, adj_26, adj_27);
            wp::adj_extract(var_9, var_25, adj_9, adj_25, adj_26);
            wp::adj_vec_t(var_21, var_22, var_23, adj_21, adj_22, adj_23, adj_24);
            // adj: ay = wp.vec3(1.0, 0.0, 0.0) - ax * ax[0]                                  <L 44>
        }
        wp::adj_abs(var_17, adj_17, adj_18);
        wp::adj_extract(var_9, var_16, adj_9, adj_16, adj_17);
        // adj: if wp.abs(ax[0]) < 0.9:                                                       <L 43>
    }
    // adj: if ny < 1.0e-12:                                                                  <L 40>
    wp::adj_length(var_12, var_13, adj_12, adj_13);
    // adj: ny = wp.length(ay)                                                                <L 39>
    wp::adj_sub(var_ay, var_11, adj_ay, adj_11, adj_12);
    wp::adj_mul(var_9, var_10, adj_9, adj_10, adj_11);
    wp::adj_dot(var_9, var_ay, adj_9, adj_ay, adj_10);
    // adj: ay = ay - ax * wp.dot(ax, ay)                                                     <L 38>
    wp::adj_where(var_2, var_7, var_8, adj_2, adj_7, adj_8, adj_9);
    if (!var_2) {
        wp::adj_div(var_7, var_0, adj_7, adj_0, adj_8);
        // adj: ax = ax / nx                                                                  <L 37>
    }
    wp::adj_where(var_2, var_6, var_ax, adj_2, adj_6, adj_ax, adj_7);
    if (var_2) {
        wp::adj_vec_t(var_3, var_4, var_5, adj_3, adj_4, adj_5, adj_6);
        // adj: ax = wp.vec3(1.0, 0.0, 0.0)                                                   <L 35>
    }
    // adj: if nx < 1.0e-12:                                                                  <L 34>
    wp::adj_length(var_ax, var_0, adj_ax, adj_0);
    // adj: nx = wp.length(ax)                                                                <L 33>
    // adj: def _orthonormalize(ax: wp.vec3, ay: wp.vec3, az: wp.vec3) -> wp.mat33:           <L 30>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_render.py:60
static CUDA_CALLABLE void adj__constraint_abs_strain_0(
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b,
    wp::float32 var_rest_length,
    wp::vec_t<3, wp::float32> & adj_a,
    wp::vec_t<3, wp::float32> & adj_b,
    wp::float32 & adj_rest_length,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    const wp::float32 var_0 = 1e-12;
    bool var_1;
    const wp::float32 var_2 = 0.0;
    wp::vec_t<3, wp::float32> var_3;
    wp::float32 var_4;
    wp::float32 var_5;
    const wp::float32 var_6 = 1.0;
    wp::float32 var_7;
    wp::float32 var_8;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    bool adj_1 = {};
    wp::float32 adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    //---------
    // forward
    // def _constraint_abs_strain(a: wp.vec3, b: wp.vec3, rest_length: float) -> float:       <L 61>
    // if rest_length <= 1.0e-12:                                                             <L 62>
    var_1 = (var_rest_length <= var_0);
    if (var_1) {
        // return 0.0                                                                         <L 63>
        goto label0;
    }
    // return wp.abs((wp.length(b - a) / rest_length) - 1.0)                                  <L 64>
    var_3 = wp::sub(var_b, var_a);
    var_4 = wp::length(var_3);
    var_5 = wp::div(var_4, var_rest_length);
    var_7 = wp::sub(var_5, var_6);
    var_8 = wp::abs(var_7);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_8 += adj_ret;
    wp::adj_abs(var_7, adj_7, adj_8);
    wp::adj_sub(var_5, var_6, adj_5, adj_6, adj_7);
    wp::adj_div(var_4, var_rest_length, var_5, adj_4, adj_rest_length, adj_5);
    wp::adj_length(var_3, var_4, adj_3, adj_4);
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_3);
    // adj: return wp.abs((wp.length(b - a) / rest_length) - 1.0)                             <L 64>
    if (var_1) {
        label0:;
        adj_2 += adj_ret;
        // adj: return 0.0                                                                    <L 63>
    }
    // adj: if rest_length <= 1.0e-12:                                                        <L 62>
    // adj: def _constraint_abs_strain(a: wp.vec3, b: wp.vec3, rest_length: float) -> float:  <L 61>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/cell_render.py:67
static CUDA_CALLABLE void adj__cell_max_abs_strain_0(
    wp::vec_t<3, wp::float32> var_p0,
    wp::vec_t<3, wp::float32> var_p1,
    wp::vec_t<3, wp::float32> var_p2,
    wp::vec_t<3, wp::float32> var_p3,
    wp::vec_t<3, wp::float32> var_p4,
    wp::vec_t<3, wp::float32> var_p5,
    wp::vec_t<3, wp::float32> var_p6,
    wp::vec_t<3, wp::float32> var_p7,
    wp::float32 var_voxel_size,
    wp::vec_t<3, wp::float32> & adj_p0,
    wp::vec_t<3, wp::float32> & adj_p1,
    wp::vec_t<3, wp::float32> & adj_p2,
    wp::vec_t<3, wp::float32> & adj_p3,
    wp::vec_t<3, wp::float32> & adj_p4,
    wp::vec_t<3, wp::float32> & adj_p5,
    wp::vec_t<3, wp::float32> & adj_p6,
    wp::vec_t<3, wp::float32> & adj_p7,
    wp::float32 & adj_voxel_size,
    wp::float32 & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1.4142135623730951;
    wp::float32 var_2;
    const wp::float32 var_3 = 1.7320508075688772;
    wp::float32 var_4;
    const wp::float32 var_5 = 0.0;
    wp::float32 var_6;
    wp::float32 var_7;
    wp::float32 var_8;
    wp::float32 var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    wp::float32 var_12;
    wp::float32 var_13;
    wp::float32 var_14;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    wp::float32 var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    wp::float32 var_46;
    wp::float32 var_47;
    wp::float32 var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    wp::float32 var_52;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
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
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
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
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    //---------
    // forward
    // def _cell_max_abs_strain(                                                              <L 68>
    // edge = voxel_size                                                                      <L 79>
    var_0 = wp::copy(var_voxel_size);
    // face = voxel_size * 1.4142135623730951                                                 <L 80>
    var_2 = wp::mul(var_voxel_size, var_1);
    // body = voxel_size * 1.7320508075688772                                                 <L 81>
    var_4 = wp::mul(var_voxel_size, var_3);
    // s = float(0.0)                                                                         <L 82>
    var_6 = wp::float(var_5);
    // s = wp.max(s, _constraint_abs_strain(p0, p1, edge))                                    <L 84>
    var_7 = _constraint_abs_strain_0(var_p0, var_p1, var_0);
    var_8 = wp::max(var_6, var_7);
    // s = wp.max(s, _constraint_abs_strain(p1, p2, edge))                                    <L 85>
    var_9 = _constraint_abs_strain_0(var_p1, var_p2, var_0);
    var_10 = wp::max(var_8, var_9);
    // s = wp.max(s, _constraint_abs_strain(p2, p3, edge))                                    <L 86>
    var_11 = _constraint_abs_strain_0(var_p2, var_p3, var_0);
    var_12 = wp::max(var_10, var_11);
    // s = wp.max(s, _constraint_abs_strain(p3, p0, edge))                                    <L 87>
    var_13 = _constraint_abs_strain_0(var_p3, var_p0, var_0);
    var_14 = wp::max(var_12, var_13);
    // s = wp.max(s, _constraint_abs_strain(p4, p5, edge))                                    <L 88>
    var_15 = _constraint_abs_strain_0(var_p4, var_p5, var_0);
    var_16 = wp::max(var_14, var_15);
    // s = wp.max(s, _constraint_abs_strain(p5, p6, edge))                                    <L 89>
    var_17 = _constraint_abs_strain_0(var_p5, var_p6, var_0);
    var_18 = wp::max(var_16, var_17);
    // s = wp.max(s, _constraint_abs_strain(p6, p7, edge))                                    <L 90>
    var_19 = _constraint_abs_strain_0(var_p6, var_p7, var_0);
    var_20 = wp::max(var_18, var_19);
    // s = wp.max(s, _constraint_abs_strain(p7, p4, edge))                                    <L 91>
    var_21 = _constraint_abs_strain_0(var_p7, var_p4, var_0);
    var_22 = wp::max(var_20, var_21);
    // s = wp.max(s, _constraint_abs_strain(p0, p4, edge))                                    <L 92>
    var_23 = _constraint_abs_strain_0(var_p0, var_p4, var_0);
    var_24 = wp::max(var_22, var_23);
    // s = wp.max(s, _constraint_abs_strain(p1, p5, edge))                                    <L 93>
    var_25 = _constraint_abs_strain_0(var_p1, var_p5, var_0);
    var_26 = wp::max(var_24, var_25);
    // s = wp.max(s, _constraint_abs_strain(p2, p6, edge))                                    <L 94>
    var_27 = _constraint_abs_strain_0(var_p2, var_p6, var_0);
    var_28 = wp::max(var_26, var_27);
    // s = wp.max(s, _constraint_abs_strain(p3, p7, edge))                                    <L 95>
    var_29 = _constraint_abs_strain_0(var_p3, var_p7, var_0);
    var_30 = wp::max(var_28, var_29);
    // s = wp.max(s, _constraint_abs_strain(p0, p2, face))                                    <L 97>
    var_31 = _constraint_abs_strain_0(var_p0, var_p2, var_2);
    var_32 = wp::max(var_30, var_31);
    // s = wp.max(s, _constraint_abs_strain(p1, p3, face))                                    <L 98>
    var_33 = _constraint_abs_strain_0(var_p1, var_p3, var_2);
    var_34 = wp::max(var_32, var_33);
    // s = wp.max(s, _constraint_abs_strain(p4, p6, face))                                    <L 99>
    var_35 = _constraint_abs_strain_0(var_p4, var_p6, var_2);
    var_36 = wp::max(var_34, var_35);
    // s = wp.max(s, _constraint_abs_strain(p5, p7, face))                                    <L 100>
    var_37 = _constraint_abs_strain_0(var_p5, var_p7, var_2);
    var_38 = wp::max(var_36, var_37);
    // s = wp.max(s, _constraint_abs_strain(p0, p5, face))                                    <L 101>
    var_39 = _constraint_abs_strain_0(var_p0, var_p5, var_2);
    var_40 = wp::max(var_38, var_39);
    // s = wp.max(s, _constraint_abs_strain(p1, p4, face))                                    <L 102>
    var_41 = _constraint_abs_strain_0(var_p1, var_p4, var_2);
    var_42 = wp::max(var_40, var_41);
    // s = wp.max(s, _constraint_abs_strain(p3, p6, face))                                    <L 103>
    var_43 = _constraint_abs_strain_0(var_p3, var_p6, var_2);
    var_44 = wp::max(var_42, var_43);
    // s = wp.max(s, _constraint_abs_strain(p2, p7, face))                                    <L 104>
    var_45 = _constraint_abs_strain_0(var_p2, var_p7, var_2);
    var_46 = wp::max(var_44, var_45);
    // s = wp.max(s, _constraint_abs_strain(p0, p7, face))                                    <L 105>
    var_47 = _constraint_abs_strain_0(var_p0, var_p7, var_2);
    var_48 = wp::max(var_46, var_47);
    // s = wp.max(s, _constraint_abs_strain(p3, p4, face))                                    <L 106>
    var_49 = _constraint_abs_strain_0(var_p3, var_p4, var_2);
    var_50 = wp::max(var_48, var_49);
    // s = wp.max(s, _constraint_abs_strain(p1, p6, face))                                    <L 107>
    var_51 = _constraint_abs_strain_0(var_p1, var_p6, var_2);
    var_52 = wp::max(var_50, var_51);
    // s = wp.max(s, _constraint_abs_strain(p2, p5, face))                                    <L 108>
    var_53 = _constraint_abs_strain_0(var_p2, var_p5, var_2);
    var_54 = wp::max(var_52, var_53);
    // s = wp.max(s, _constraint_abs_strain(p0, p6, body))                                    <L 110>
    var_55 = _constraint_abs_strain_0(var_p0, var_p6, var_4);
    var_56 = wp::max(var_54, var_55);
    // s = wp.max(s, _constraint_abs_strain(p1, p7, body))                                    <L 111>
    var_57 = _constraint_abs_strain_0(var_p1, var_p7, var_4);
    var_58 = wp::max(var_56, var_57);
    // s = wp.max(s, _constraint_abs_strain(p2, p4, body))                                    <L 112>
    var_59 = _constraint_abs_strain_0(var_p2, var_p4, var_4);
    var_60 = wp::max(var_58, var_59);
    // s = wp.max(s, _constraint_abs_strain(p3, p5, body))                                    <L 113>
    var_61 = _constraint_abs_strain_0(var_p3, var_p5, var_4);
    var_62 = wp::max(var_60, var_61);
    // return s                                                                               <L 114>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_62 += adj_ret;
    // adj: return s                                                                          <L 114>
    wp::adj_max(var_60, var_61, adj_60, adj_61, adj_62);
    adj__constraint_abs_strain_0(var_p3, var_p5, var_4, adj_p3, adj_p5, adj_4, adj_61);
    // adj: s = wp.max(s, _constraint_abs_strain(p3, p5, body))                               <L 113>
    wp::adj_max(var_58, var_59, adj_58, adj_59, adj_60);
    adj__constraint_abs_strain_0(var_p2, var_p4, var_4, adj_p2, adj_p4, adj_4, adj_59);
    // adj: s = wp.max(s, _constraint_abs_strain(p2, p4, body))                               <L 112>
    wp::adj_max(var_56, var_57, adj_56, adj_57, adj_58);
    adj__constraint_abs_strain_0(var_p1, var_p7, var_4, adj_p1, adj_p7, adj_4, adj_57);
    // adj: s = wp.max(s, _constraint_abs_strain(p1, p7, body))                               <L 111>
    wp::adj_max(var_54, var_55, adj_54, adj_55, adj_56);
    adj__constraint_abs_strain_0(var_p0, var_p6, var_4, adj_p0, adj_p6, adj_4, adj_55);
    // adj: s = wp.max(s, _constraint_abs_strain(p0, p6, body))                               <L 110>
    wp::adj_max(var_52, var_53, adj_52, adj_53, adj_54);
    adj__constraint_abs_strain_0(var_p2, var_p5, var_2, adj_p2, adj_p5, adj_2, adj_53);
    // adj: s = wp.max(s, _constraint_abs_strain(p2, p5, face))                               <L 108>
    wp::adj_max(var_50, var_51, adj_50, adj_51, adj_52);
    adj__constraint_abs_strain_0(var_p1, var_p6, var_2, adj_p1, adj_p6, adj_2, adj_51);
    // adj: s = wp.max(s, _constraint_abs_strain(p1, p6, face))                               <L 107>
    wp::adj_max(var_48, var_49, adj_48, adj_49, adj_50);
    adj__constraint_abs_strain_0(var_p3, var_p4, var_2, adj_p3, adj_p4, adj_2, adj_49);
    // adj: s = wp.max(s, _constraint_abs_strain(p3, p4, face))                               <L 106>
    wp::adj_max(var_46, var_47, adj_46, adj_47, adj_48);
    adj__constraint_abs_strain_0(var_p0, var_p7, var_2, adj_p0, adj_p7, adj_2, adj_47);
    // adj: s = wp.max(s, _constraint_abs_strain(p0, p7, face))                               <L 105>
    wp::adj_max(var_44, var_45, adj_44, adj_45, adj_46);
    adj__constraint_abs_strain_0(var_p2, var_p7, var_2, adj_p2, adj_p7, adj_2, adj_45);
    // adj: s = wp.max(s, _constraint_abs_strain(p2, p7, face))                               <L 104>
    wp::adj_max(var_42, var_43, adj_42, adj_43, adj_44);
    adj__constraint_abs_strain_0(var_p3, var_p6, var_2, adj_p3, adj_p6, adj_2, adj_43);
    // adj: s = wp.max(s, _constraint_abs_strain(p3, p6, face))                               <L 103>
    wp::adj_max(var_40, var_41, adj_40, adj_41, adj_42);
    adj__constraint_abs_strain_0(var_p1, var_p4, var_2, adj_p1, adj_p4, adj_2, adj_41);
    // adj: s = wp.max(s, _constraint_abs_strain(p1, p4, face))                               <L 102>
    wp::adj_max(var_38, var_39, adj_38, adj_39, adj_40);
    adj__constraint_abs_strain_0(var_p0, var_p5, var_2, adj_p0, adj_p5, adj_2, adj_39);
    // adj: s = wp.max(s, _constraint_abs_strain(p0, p5, face))                               <L 101>
    wp::adj_max(var_36, var_37, adj_36, adj_37, adj_38);
    adj__constraint_abs_strain_0(var_p5, var_p7, var_2, adj_p5, adj_p7, adj_2, adj_37);
    // adj: s = wp.max(s, _constraint_abs_strain(p5, p7, face))                               <L 100>
    wp::adj_max(var_34, var_35, adj_34, adj_35, adj_36);
    adj__constraint_abs_strain_0(var_p4, var_p6, var_2, adj_p4, adj_p6, adj_2, adj_35);
    // adj: s = wp.max(s, _constraint_abs_strain(p4, p6, face))                               <L 99>
    wp::adj_max(var_32, var_33, adj_32, adj_33, adj_34);
    adj__constraint_abs_strain_0(var_p1, var_p3, var_2, adj_p1, adj_p3, adj_2, adj_33);
    // adj: s = wp.max(s, _constraint_abs_strain(p1, p3, face))                               <L 98>
    wp::adj_max(var_30, var_31, adj_30, adj_31, adj_32);
    adj__constraint_abs_strain_0(var_p0, var_p2, var_2, adj_p0, adj_p2, adj_2, adj_31);
    // adj: s = wp.max(s, _constraint_abs_strain(p0, p2, face))                               <L 97>
    wp::adj_max(var_28, var_29, adj_28, adj_29, adj_30);
    adj__constraint_abs_strain_0(var_p3, var_p7, var_0, adj_p3, adj_p7, adj_0, adj_29);
    // adj: s = wp.max(s, _constraint_abs_strain(p3, p7, edge))                               <L 95>
    wp::adj_max(var_26, var_27, adj_26, adj_27, adj_28);
    adj__constraint_abs_strain_0(var_p2, var_p6, var_0, adj_p2, adj_p6, adj_0, adj_27);
    // adj: s = wp.max(s, _constraint_abs_strain(p2, p6, edge))                               <L 94>
    wp::adj_max(var_24, var_25, adj_24, adj_25, adj_26);
    adj__constraint_abs_strain_0(var_p1, var_p5, var_0, adj_p1, adj_p5, adj_0, adj_25);
    // adj: s = wp.max(s, _constraint_abs_strain(p1, p5, edge))                               <L 93>
    wp::adj_max(var_22, var_23, adj_22, adj_23, adj_24);
    adj__constraint_abs_strain_0(var_p0, var_p4, var_0, adj_p0, adj_p4, adj_0, adj_23);
    // adj: s = wp.max(s, _constraint_abs_strain(p0, p4, edge))                               <L 92>
    wp::adj_max(var_20, var_21, adj_20, adj_21, adj_22);
    adj__constraint_abs_strain_0(var_p7, var_p4, var_0, adj_p7, adj_p4, adj_0, adj_21);
    // adj: s = wp.max(s, _constraint_abs_strain(p7, p4, edge))                               <L 91>
    wp::adj_max(var_18, var_19, adj_18, adj_19, adj_20);
    adj__constraint_abs_strain_0(var_p6, var_p7, var_0, adj_p6, adj_p7, adj_0, adj_19);
    // adj: s = wp.max(s, _constraint_abs_strain(p6, p7, edge))                               <L 90>
    wp::adj_max(var_16, var_17, adj_16, adj_17, adj_18);
    adj__constraint_abs_strain_0(var_p5, var_p6, var_0, adj_p5, adj_p6, adj_0, adj_17);
    // adj: s = wp.max(s, _constraint_abs_strain(p5, p6, edge))                               <L 89>
    wp::adj_max(var_14, var_15, adj_14, adj_15, adj_16);
    adj__constraint_abs_strain_0(var_p4, var_p5, var_0, adj_p4, adj_p5, adj_0, adj_15);
    // adj: s = wp.max(s, _constraint_abs_strain(p4, p5, edge))                               <L 88>
    wp::adj_max(var_12, var_13, adj_12, adj_13, adj_14);
    adj__constraint_abs_strain_0(var_p3, var_p0, var_0, adj_p3, adj_p0, adj_0, adj_13);
    // adj: s = wp.max(s, _constraint_abs_strain(p3, p0, edge))                               <L 87>
    wp::adj_max(var_10, var_11, adj_10, adj_11, adj_12);
    adj__constraint_abs_strain_0(var_p2, var_p3, var_0, adj_p2, adj_p3, adj_0, adj_11);
    // adj: s = wp.max(s, _constraint_abs_strain(p2, p3, edge))                               <L 86>
    wp::adj_max(var_8, var_9, adj_8, adj_9, adj_10);
    adj__constraint_abs_strain_0(var_p1, var_p2, var_0, adj_p1, adj_p2, adj_0, adj_9);
    // adj: s = wp.max(s, _constraint_abs_strain(p1, p2, edge))                               <L 85>
    wp::adj_max(var_6, var_7, adj_6, adj_7, adj_8);
    adj__constraint_abs_strain_0(var_p0, var_p1, var_0, adj_p0, adj_p1, adj_0, adj_7);
    // adj: s = wp.max(s, _constraint_abs_strain(p0, p1, edge))                               <L 84>
    wp::adj_float(var_5, adj_5, adj_6);
    // adj: s = float(0.0)                                                                    <L 82>
    wp::adj_mul(var_voxel_size, var_3, adj_voxel_size, adj_3, adj_4);
    // adj: body = voxel_size * 1.7320508075688772                                            <L 81>
    wp::adj_mul(var_voxel_size, var_1, adj_voxel_size, adj_1, adj_2);
    // adj: face = voxel_size * 1.4142135623730951                                            <L 80>
    wp::adj_copy(var_voxel_size, adj_voxel_size, adj_0);
    // adj: edge = voxel_size                                                                 <L 79>
    // adj: def _cell_max_abs_strain(                                                         <L 68>
    return;
}



extern "C" __global__ void build_cell_aabbs_kernel_778498da_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max)
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
        wp::vec_t<3, wp::float32>* var_3;
        wp::int32 var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        const wp::int32 var_9 = 1;
        wp::int32* var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 1;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 2;
        wp::float32 var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 0;
        wp::float32 var_32;
        const wp::int32 var_33 = 0;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 1;
        wp::float32 var_37;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 2;
        wp::float32 var_42;
        const wp::int32 var_43 = 2;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 2;
        wp::int32* var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        const wp::int32 var_53 = 0;
        wp::float32 var_54;
        const wp::int32 var_55 = 0;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 1;
        wp::float32 var_59;
        const wp::int32 var_60 = 1;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 2;
        wp::float32 var_64;
        const wp::int32 var_65 = 2;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3, wp::float32> var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        const wp::int32 var_71 = 0;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 1;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 2;
        wp::float32 var_80;
        const wp::int32 var_81 = 2;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        const wp::int32 var_85 = 3;
        wp::int32* var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::int32 var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        const wp::int32 var_91 = 0;
        wp::float32 var_92;
        const wp::int32 var_93 = 0;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::int32 var_96 = 1;
        wp::float32 var_97;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        const wp::int32 var_103 = 2;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::vec_t<3, wp::float32> var_106;
        const wp::int32 var_107 = 0;
        wp::float32 var_108;
        const wp::int32 var_109 = 0;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 1;
        wp::float32 var_113;
        const wp::int32 var_114 = 1;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::vec_t<3, wp::float32> var_122;
        const wp::int32 var_123 = 4;
        wp::int32* var_124;
        wp::vec_t<3, wp::float32>* var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        const wp::int32 var_129 = 0;
        wp::float32 var_130;
        const wp::int32 var_131 = 0;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::int32 var_134 = 1;
        wp::float32 var_135;
        const wp::int32 var_136 = 1;
        wp::float32 var_137;
        wp::float32 var_138;
        const wp::int32 var_139 = 2;
        wp::float32 var_140;
        const wp::int32 var_141 = 2;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::vec_t<3, wp::float32> var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        const wp::int32 var_147 = 0;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 1;
        wp::float32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        const wp::int32 var_157 = 2;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::vec_t<3, wp::float32> var_160;
        const wp::int32 var_161 = 5;
        wp::int32* var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::int32 var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        const wp::int32 var_167 = 0;
        wp::float32 var_168;
        const wp::int32 var_169 = 0;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 1;
        wp::float32 var_173;
        const wp::int32 var_174 = 1;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 2;
        wp::float32 var_178;
        const wp::int32 var_179 = 2;
        wp::float32 var_180;
        wp::float32 var_181;
        wp::vec_t<3, wp::float32> var_182;
        const wp::int32 var_183 = 0;
        wp::float32 var_184;
        const wp::int32 var_185 = 0;
        wp::float32 var_186;
        wp::float32 var_187;
        const wp::int32 var_188 = 1;
        wp::float32 var_189;
        const wp::int32 var_190 = 1;
        wp::float32 var_191;
        wp::float32 var_192;
        const wp::int32 var_193 = 2;
        wp::float32 var_194;
        const wp::int32 var_195 = 2;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::vec_t<3, wp::float32> var_198;
        const wp::int32 var_199 = 6;
        wp::int32* var_200;
        wp::vec_t<3, wp::float32>* var_201;
        wp::int32 var_202;
        wp::vec_t<3, wp::float32> var_203;
        wp::vec_t<3, wp::float32> var_204;
        const wp::int32 var_205 = 0;
        wp::float32 var_206;
        const wp::int32 var_207 = 0;
        wp::float32 var_208;
        wp::float32 var_209;
        const wp::int32 var_210 = 1;
        wp::float32 var_211;
        const wp::int32 var_212 = 1;
        wp::float32 var_213;
        wp::float32 var_214;
        const wp::int32 var_215 = 2;
        wp::float32 var_216;
        const wp::int32 var_217 = 2;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::vec_t<3, wp::float32> var_220;
        const wp::int32 var_221 = 0;
        wp::float32 var_222;
        const wp::int32 var_223 = 0;
        wp::float32 var_224;
        wp::float32 var_225;
        const wp::int32 var_226 = 1;
        wp::float32 var_227;
        const wp::int32 var_228 = 1;
        wp::float32 var_229;
        wp::float32 var_230;
        const wp::int32 var_231 = 2;
        wp::float32 var_232;
        const wp::int32 var_233 = 2;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::vec_t<3, wp::float32> var_236;
        const wp::int32 var_237 = 7;
        wp::int32* var_238;
        wp::vec_t<3, wp::float32>* var_239;
        wp::int32 var_240;
        wp::vec_t<3, wp::float32> var_241;
        wp::vec_t<3, wp::float32> var_242;
        const wp::int32 var_243 = 0;
        wp::float32 var_244;
        const wp::int32 var_245 = 0;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 1;
        wp::float32 var_249;
        const wp::int32 var_250 = 1;
        wp::float32 var_251;
        wp::float32 var_252;
        const wp::int32 var_253 = 2;
        wp::float32 var_254;
        const wp::int32 var_255 = 2;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::vec_t<3, wp::float32> var_258;
        const wp::int32 var_259 = 0;
        wp::float32 var_260;
        const wp::int32 var_261 = 0;
        wp::float32 var_262;
        wp::float32 var_263;
        const wp::int32 var_264 = 1;
        wp::float32 var_265;
        const wp::int32 var_266 = 1;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::int32 var_269 = 2;
        wp::float32 var_270;
        const wp::int32 var_271 = 2;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::vec_t<3, wp::float32> var_274;
        //---------
        // forward
        // def build_cell_aabbs_kernel(                                                           <L 285>
        // c = wp.tid()                                                                           <L 298>
        var_0 = builtin_tid1d();
        // p = particle_q[cell_nodes[c, 0]]                                                       <L 299>
        var_2 = wp::address(var_cell_nodes, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_particle_q, var_4);
        var_6 = wp::load(var_3);
        var_5 = wp::copy(var_6);
        // lo = p                                                                                 <L 300>
        var_7 = wp::copy(var_5);
        // hi = p                                                                                 <L 301>
        var_8 = wp::copy(var_5);
        // for k in range(1, 8):                                                                  <L 302>
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_10 = wp::address(var_cell_nodes, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_particle_q, var_12);
        var_14 = wp::load(var_11);
        var_13 = wp::copy(var_14);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_16 = wp::extract(var_7, var_15);
        var_18 = wp::extract(var_13, var_17);
        var_19 = wp::min(var_16, var_18);
        var_21 = wp::extract(var_7, var_20);
        var_23 = wp::extract(var_13, var_22);
        var_24 = wp::min(var_21, var_23);
        var_26 = wp::extract(var_7, var_25);
        var_28 = wp::extract(var_13, var_27);
        var_29 = wp::min(var_26, var_28);
        var_30 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_32 = wp::extract(var_8, var_31);
        var_34 = wp::extract(var_13, var_33);
        var_35 = wp::max(var_32, var_34);
        var_37 = wp::extract(var_8, var_36);
        var_39 = wp::extract(var_13, var_38);
        var_40 = wp::max(var_37, var_39);
        var_42 = wp::extract(var_8, var_41);
        var_44 = wp::extract(var_13, var_43);
        var_45 = wp::max(var_42, var_44);
        var_46 = wp::vec_t<3, wp::float32>(var_35, var_40, var_45);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_48 = wp::address(var_cell_nodes, var_0, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::address(var_particle_q, var_50);
        var_52 = wp::load(var_49);
        var_51 = wp::copy(var_52);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_54 = wp::extract(var_30, var_53);
        var_56 = wp::extract(var_51, var_55);
        var_57 = wp::min(var_54, var_56);
        var_59 = wp::extract(var_30, var_58);
        var_61 = wp::extract(var_51, var_60);
        var_62 = wp::min(var_59, var_61);
        var_64 = wp::extract(var_30, var_63);
        var_66 = wp::extract(var_51, var_65);
        var_67 = wp::min(var_64, var_66);
        var_68 = wp::vec_t<3, wp::float32>(var_57, var_62, var_67);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_70 = wp::extract(var_46, var_69);
        var_72 = wp::extract(var_51, var_71);
        var_73 = wp::max(var_70, var_72);
        var_75 = wp::extract(var_46, var_74);
        var_77 = wp::extract(var_51, var_76);
        var_78 = wp::max(var_75, var_77);
        var_80 = wp::extract(var_46, var_79);
        var_82 = wp::extract(var_51, var_81);
        var_83 = wp::max(var_80, var_82);
        var_84 = wp::vec_t<3, wp::float32>(var_73, var_78, var_83);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_86 = wp::address(var_cell_nodes, var_0, var_85);
        var_88 = wp::load(var_86);
        var_87 = wp::address(var_particle_q, var_88);
        var_90 = wp::load(var_87);
        var_89 = wp::copy(var_90);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_92 = wp::extract(var_68, var_91);
        var_94 = wp::extract(var_89, var_93);
        var_95 = wp::min(var_92, var_94);
        var_97 = wp::extract(var_68, var_96);
        var_99 = wp::extract(var_89, var_98);
        var_100 = wp::min(var_97, var_99);
        var_102 = wp::extract(var_68, var_101);
        var_104 = wp::extract(var_89, var_103);
        var_105 = wp::min(var_102, var_104);
        var_106 = wp::vec_t<3, wp::float32>(var_95, var_100, var_105);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_108 = wp::extract(var_84, var_107);
        var_110 = wp::extract(var_89, var_109);
        var_111 = wp::max(var_108, var_110);
        var_113 = wp::extract(var_84, var_112);
        var_115 = wp::extract(var_89, var_114);
        var_116 = wp::max(var_113, var_115);
        var_118 = wp::extract(var_84, var_117);
        var_120 = wp::extract(var_89, var_119);
        var_121 = wp::max(var_118, var_120);
        var_122 = wp::vec_t<3, wp::float32>(var_111, var_116, var_121);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_124 = wp::address(var_cell_nodes, var_0, var_123);
        var_126 = wp::load(var_124);
        var_125 = wp::address(var_particle_q, var_126);
        var_128 = wp::load(var_125);
        var_127 = wp::copy(var_128);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_130 = wp::extract(var_106, var_129);
        var_132 = wp::extract(var_127, var_131);
        var_133 = wp::min(var_130, var_132);
        var_135 = wp::extract(var_106, var_134);
        var_137 = wp::extract(var_127, var_136);
        var_138 = wp::min(var_135, var_137);
        var_140 = wp::extract(var_106, var_139);
        var_142 = wp::extract(var_127, var_141);
        var_143 = wp::min(var_140, var_142);
        var_144 = wp::vec_t<3, wp::float32>(var_133, var_138, var_143);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_146 = wp::extract(var_122, var_145);
        var_148 = wp::extract(var_127, var_147);
        var_149 = wp::max(var_146, var_148);
        var_151 = wp::extract(var_122, var_150);
        var_153 = wp::extract(var_127, var_152);
        var_154 = wp::max(var_151, var_153);
        var_156 = wp::extract(var_122, var_155);
        var_158 = wp::extract(var_127, var_157);
        var_159 = wp::max(var_156, var_158);
        var_160 = wp::vec_t<3, wp::float32>(var_149, var_154, var_159);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_162 = wp::address(var_cell_nodes, var_0, var_161);
        var_164 = wp::load(var_162);
        var_163 = wp::address(var_particle_q, var_164);
        var_166 = wp::load(var_163);
        var_165 = wp::copy(var_166);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_168 = wp::extract(var_144, var_167);
        var_170 = wp::extract(var_165, var_169);
        var_171 = wp::min(var_168, var_170);
        var_173 = wp::extract(var_144, var_172);
        var_175 = wp::extract(var_165, var_174);
        var_176 = wp::min(var_173, var_175);
        var_178 = wp::extract(var_144, var_177);
        var_180 = wp::extract(var_165, var_179);
        var_181 = wp::min(var_178, var_180);
        var_182 = wp::vec_t<3, wp::float32>(var_171, var_176, var_181);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_184 = wp::extract(var_160, var_183);
        var_186 = wp::extract(var_165, var_185);
        var_187 = wp::max(var_184, var_186);
        var_189 = wp::extract(var_160, var_188);
        var_191 = wp::extract(var_165, var_190);
        var_192 = wp::max(var_189, var_191);
        var_194 = wp::extract(var_160, var_193);
        var_196 = wp::extract(var_165, var_195);
        var_197 = wp::max(var_194, var_196);
        var_198 = wp::vec_t<3, wp::float32>(var_187, var_192, var_197);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_200 = wp::address(var_cell_nodes, var_0, var_199);
        var_202 = wp::load(var_200);
        var_201 = wp::address(var_particle_q, var_202);
        var_204 = wp::load(var_201);
        var_203 = wp::copy(var_204);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_206 = wp::extract(var_182, var_205);
        var_208 = wp::extract(var_203, var_207);
        var_209 = wp::min(var_206, var_208);
        var_211 = wp::extract(var_182, var_210);
        var_213 = wp::extract(var_203, var_212);
        var_214 = wp::min(var_211, var_213);
        var_216 = wp::extract(var_182, var_215);
        var_218 = wp::extract(var_203, var_217);
        var_219 = wp::min(var_216, var_218);
        var_220 = wp::vec_t<3, wp::float32>(var_209, var_214, var_219);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_222 = wp::extract(var_198, var_221);
        var_224 = wp::extract(var_203, var_223);
        var_225 = wp::max(var_222, var_224);
        var_227 = wp::extract(var_198, var_226);
        var_229 = wp::extract(var_203, var_228);
        var_230 = wp::max(var_227, var_229);
        var_232 = wp::extract(var_198, var_231);
        var_234 = wp::extract(var_203, var_233);
        var_235 = wp::max(var_232, var_234);
        var_236 = wp::vec_t<3, wp::float32>(var_225, var_230, var_235);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_238 = wp::address(var_cell_nodes, var_0, var_237);
        var_240 = wp::load(var_238);
        var_239 = wp::address(var_particle_q, var_240);
        var_242 = wp::load(var_239);
        var_241 = wp::copy(var_242);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_244 = wp::extract(var_220, var_243);
        var_246 = wp::extract(var_241, var_245);
        var_247 = wp::min(var_244, var_246);
        var_249 = wp::extract(var_220, var_248);
        var_251 = wp::extract(var_241, var_250);
        var_252 = wp::min(var_249, var_251);
        var_254 = wp::extract(var_220, var_253);
        var_256 = wp::extract(var_241, var_255);
        var_257 = wp::min(var_254, var_256);
        var_258 = wp::vec_t<3, wp::float32>(var_247, var_252, var_257);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_260 = wp::extract(var_236, var_259);
        var_262 = wp::extract(var_241, var_261);
        var_263 = wp::max(var_260, var_262);
        var_265 = wp::extract(var_236, var_264);
        var_267 = wp::extract(var_241, var_266);
        var_268 = wp::max(var_265, var_267);
        var_270 = wp::extract(var_236, var_269);
        var_272 = wp::extract(var_241, var_271);
        var_273 = wp::max(var_270, var_272);
        var_274 = wp::vec_t<3, wp::float32>(var_263, var_268, var_273);
        // aabb_min[c] = lo                                                                       <L 306>
        wp::array_store(var_aabb_min, var_0, var_258);
        // aabb_max[c] = hi                                                                       <L 307>
        wp::array_store(var_aabb_max, var_0, var_274);
    }
}



extern "C" __global__ void build_cell_aabbs_kernel_778498da_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_max)
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
        wp::vec_t<3, wp::float32>* var_3;
        wp::int32 var_4;
        wp::vec_t<3, wp::float32> var_5;
        wp::vec_t<3, wp::float32> var_6;
        wp::vec_t<3, wp::float32> var_7;
        wp::vec_t<3, wp::float32> var_8;
        const wp::int32 var_9 = 1;
        wp::int32* var_10;
        wp::vec_t<3, wp::float32>* var_11;
        wp::int32 var_12;
        wp::vec_t<3, wp::float32> var_13;
        wp::vec_t<3, wp::float32> var_14;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        const wp::int32 var_17 = 0;
        wp::float32 var_18;
        wp::float32 var_19;
        const wp::int32 var_20 = 1;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        wp::float32 var_24;
        const wp::int32 var_25 = 2;
        wp::float32 var_26;
        const wp::int32 var_27 = 2;
        wp::float32 var_28;
        wp::float32 var_29;
        wp::vec_t<3, wp::float32> var_30;
        const wp::int32 var_31 = 0;
        wp::float32 var_32;
        const wp::int32 var_33 = 0;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 1;
        wp::float32 var_37;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 2;
        wp::float32 var_42;
        const wp::int32 var_43 = 2;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 2;
        wp::int32* var_48;
        wp::vec_t<3, wp::float32>* var_49;
        wp::int32 var_50;
        wp::vec_t<3, wp::float32> var_51;
        wp::vec_t<3, wp::float32> var_52;
        const wp::int32 var_53 = 0;
        wp::float32 var_54;
        const wp::int32 var_55 = 0;
        wp::float32 var_56;
        wp::float32 var_57;
        const wp::int32 var_58 = 1;
        wp::float32 var_59;
        const wp::int32 var_60 = 1;
        wp::float32 var_61;
        wp::float32 var_62;
        const wp::int32 var_63 = 2;
        wp::float32 var_64;
        const wp::int32 var_65 = 2;
        wp::float32 var_66;
        wp::float32 var_67;
        wp::vec_t<3, wp::float32> var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        const wp::int32 var_71 = 0;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 1;
        wp::float32 var_75;
        const wp::int32 var_76 = 1;
        wp::float32 var_77;
        wp::float32 var_78;
        const wp::int32 var_79 = 2;
        wp::float32 var_80;
        const wp::int32 var_81 = 2;
        wp::float32 var_82;
        wp::float32 var_83;
        wp::vec_t<3, wp::float32> var_84;
        const wp::int32 var_85 = 3;
        wp::int32* var_86;
        wp::vec_t<3, wp::float32>* var_87;
        wp::int32 var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        const wp::int32 var_91 = 0;
        wp::float32 var_92;
        const wp::int32 var_93 = 0;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::int32 var_96 = 1;
        wp::float32 var_97;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        wp::float32 var_100;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        const wp::int32 var_103 = 2;
        wp::float32 var_104;
        wp::float32 var_105;
        wp::vec_t<3, wp::float32> var_106;
        const wp::int32 var_107 = 0;
        wp::float32 var_108;
        const wp::int32 var_109 = 0;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 1;
        wp::float32 var_113;
        const wp::int32 var_114 = 1;
        wp::float32 var_115;
        wp::float32 var_116;
        const wp::int32 var_117 = 2;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        wp::float32 var_120;
        wp::float32 var_121;
        wp::vec_t<3, wp::float32> var_122;
        const wp::int32 var_123 = 4;
        wp::int32* var_124;
        wp::vec_t<3, wp::float32>* var_125;
        wp::int32 var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        const wp::int32 var_129 = 0;
        wp::float32 var_130;
        const wp::int32 var_131 = 0;
        wp::float32 var_132;
        wp::float32 var_133;
        const wp::int32 var_134 = 1;
        wp::float32 var_135;
        const wp::int32 var_136 = 1;
        wp::float32 var_137;
        wp::float32 var_138;
        const wp::int32 var_139 = 2;
        wp::float32 var_140;
        const wp::int32 var_141 = 2;
        wp::float32 var_142;
        wp::float32 var_143;
        wp::vec_t<3, wp::float32> var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        const wp::int32 var_147 = 0;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 1;
        wp::float32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 2;
        wp::float32 var_156;
        const wp::int32 var_157 = 2;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::vec_t<3, wp::float32> var_160;
        const wp::int32 var_161 = 5;
        wp::int32* var_162;
        wp::vec_t<3, wp::float32>* var_163;
        wp::int32 var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        const wp::int32 var_167 = 0;
        wp::float32 var_168;
        const wp::int32 var_169 = 0;
        wp::float32 var_170;
        wp::float32 var_171;
        const wp::int32 var_172 = 1;
        wp::float32 var_173;
        const wp::int32 var_174 = 1;
        wp::float32 var_175;
        wp::float32 var_176;
        const wp::int32 var_177 = 2;
        wp::float32 var_178;
        const wp::int32 var_179 = 2;
        wp::float32 var_180;
        wp::float32 var_181;
        wp::vec_t<3, wp::float32> var_182;
        const wp::int32 var_183 = 0;
        wp::float32 var_184;
        const wp::int32 var_185 = 0;
        wp::float32 var_186;
        wp::float32 var_187;
        const wp::int32 var_188 = 1;
        wp::float32 var_189;
        const wp::int32 var_190 = 1;
        wp::float32 var_191;
        wp::float32 var_192;
        const wp::int32 var_193 = 2;
        wp::float32 var_194;
        const wp::int32 var_195 = 2;
        wp::float32 var_196;
        wp::float32 var_197;
        wp::vec_t<3, wp::float32> var_198;
        const wp::int32 var_199 = 6;
        wp::int32* var_200;
        wp::vec_t<3, wp::float32>* var_201;
        wp::int32 var_202;
        wp::vec_t<3, wp::float32> var_203;
        wp::vec_t<3, wp::float32> var_204;
        const wp::int32 var_205 = 0;
        wp::float32 var_206;
        const wp::int32 var_207 = 0;
        wp::float32 var_208;
        wp::float32 var_209;
        const wp::int32 var_210 = 1;
        wp::float32 var_211;
        const wp::int32 var_212 = 1;
        wp::float32 var_213;
        wp::float32 var_214;
        const wp::int32 var_215 = 2;
        wp::float32 var_216;
        const wp::int32 var_217 = 2;
        wp::float32 var_218;
        wp::float32 var_219;
        wp::vec_t<3, wp::float32> var_220;
        const wp::int32 var_221 = 0;
        wp::float32 var_222;
        const wp::int32 var_223 = 0;
        wp::float32 var_224;
        wp::float32 var_225;
        const wp::int32 var_226 = 1;
        wp::float32 var_227;
        const wp::int32 var_228 = 1;
        wp::float32 var_229;
        wp::float32 var_230;
        const wp::int32 var_231 = 2;
        wp::float32 var_232;
        const wp::int32 var_233 = 2;
        wp::float32 var_234;
        wp::float32 var_235;
        wp::vec_t<3, wp::float32> var_236;
        const wp::int32 var_237 = 7;
        wp::int32* var_238;
        wp::vec_t<3, wp::float32>* var_239;
        wp::int32 var_240;
        wp::vec_t<3, wp::float32> var_241;
        wp::vec_t<3, wp::float32> var_242;
        const wp::int32 var_243 = 0;
        wp::float32 var_244;
        const wp::int32 var_245 = 0;
        wp::float32 var_246;
        wp::float32 var_247;
        const wp::int32 var_248 = 1;
        wp::float32 var_249;
        const wp::int32 var_250 = 1;
        wp::float32 var_251;
        wp::float32 var_252;
        const wp::int32 var_253 = 2;
        wp::float32 var_254;
        const wp::int32 var_255 = 2;
        wp::float32 var_256;
        wp::float32 var_257;
        wp::vec_t<3, wp::float32> var_258;
        const wp::int32 var_259 = 0;
        wp::float32 var_260;
        const wp::int32 var_261 = 0;
        wp::float32 var_262;
        wp::float32 var_263;
        const wp::int32 var_264 = 1;
        wp::float32 var_265;
        const wp::int32 var_266 = 1;
        wp::float32 var_267;
        wp::float32 var_268;
        const wp::int32 var_269 = 2;
        wp::float32 var_270;
        const wp::int32 var_271 = 2;
        wp::float32 var_272;
        wp::float32 var_273;
        wp::vec_t<3, wp::float32> var_274;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        wp::int32 adj_4 = {};
        wp::vec_t<3, wp::float32> adj_5 = {};
        wp::vec_t<3, wp::float32> adj_6 = {};
        wp::vec_t<3, wp::float32> adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::int32 adj_12 = {};
        wp::vec_t<3, wp::float32> adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::int32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::vec_t<3, wp::float32> adj_49 = {};
        wp::int32 adj_50 = {};
        wp::vec_t<3, wp::float32> adj_51 = {};
        wp::vec_t<3, wp::float32> adj_52 = {};
        wp::int32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::float32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::int32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::vec_t<3, wp::float32> adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::vec_t<3, wp::float32> adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::vec_t<3, wp::float32> adj_87 = {};
        wp::int32 adj_88 = {};
        wp::vec_t<3, wp::float32> adj_89 = {};
        wp::vec_t<3, wp::float32> adj_90 = {};
        wp::int32 adj_91 = {};
        wp::float32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::float32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::vec_t<3, wp::float32> adj_106 = {};
        wp::int32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::int32 adj_109 = {};
        wp::float32 adj_110 = {};
        wp::float32 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::float32 adj_113 = {};
        wp::int32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::float32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::float32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::float32 adj_120 = {};
        wp::float32 adj_121 = {};
        wp::vec_t<3, wp::float32> adj_122 = {};
        wp::int32 adj_123 = {};
        wp::int32 adj_124 = {};
        wp::vec_t<3, wp::float32> adj_125 = {};
        wp::int32 adj_126 = {};
        wp::vec_t<3, wp::float32> adj_127 = {};
        wp::vec_t<3, wp::float32> adj_128 = {};
        wp::int32 adj_129 = {};
        wp::float32 adj_130 = {};
        wp::int32 adj_131 = {};
        wp::float32 adj_132 = {};
        wp::float32 adj_133 = {};
        wp::int32 adj_134 = {};
        wp::float32 adj_135 = {};
        wp::int32 adj_136 = {};
        wp::float32 adj_137 = {};
        wp::float32 adj_138 = {};
        wp::int32 adj_139 = {};
        wp::float32 adj_140 = {};
        wp::int32 adj_141 = {};
        wp::float32 adj_142 = {};
        wp::float32 adj_143 = {};
        wp::vec_t<3, wp::float32> adj_144 = {};
        wp::int32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::int32 adj_147 = {};
        wp::float32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::float32 adj_151 = {};
        wp::int32 adj_152 = {};
        wp::float32 adj_153 = {};
        wp::float32 adj_154 = {};
        wp::int32 adj_155 = {};
        wp::float32 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::float32 adj_158 = {};
        wp::float32 adj_159 = {};
        wp::vec_t<3, wp::float32> adj_160 = {};
        wp::int32 adj_161 = {};
        wp::int32 adj_162 = {};
        wp::vec_t<3, wp::float32> adj_163 = {};
        wp::int32 adj_164 = {};
        wp::vec_t<3, wp::float32> adj_165 = {};
        wp::vec_t<3, wp::float32> adj_166 = {};
        wp::int32 adj_167 = {};
        wp::float32 adj_168 = {};
        wp::int32 adj_169 = {};
        wp::float32 adj_170 = {};
        wp::float32 adj_171 = {};
        wp::int32 adj_172 = {};
        wp::float32 adj_173 = {};
        wp::int32 adj_174 = {};
        wp::float32 adj_175 = {};
        wp::float32 adj_176 = {};
        wp::int32 adj_177 = {};
        wp::float32 adj_178 = {};
        wp::int32 adj_179 = {};
        wp::float32 adj_180 = {};
        wp::float32 adj_181 = {};
        wp::vec_t<3, wp::float32> adj_182 = {};
        wp::int32 adj_183 = {};
        wp::float32 adj_184 = {};
        wp::int32 adj_185 = {};
        wp::float32 adj_186 = {};
        wp::float32 adj_187 = {};
        wp::int32 adj_188 = {};
        wp::float32 adj_189 = {};
        wp::int32 adj_190 = {};
        wp::float32 adj_191 = {};
        wp::float32 adj_192 = {};
        wp::int32 adj_193 = {};
        wp::float32 adj_194 = {};
        wp::int32 adj_195 = {};
        wp::float32 adj_196 = {};
        wp::float32 adj_197 = {};
        wp::vec_t<3, wp::float32> adj_198 = {};
        wp::int32 adj_199 = {};
        wp::int32 adj_200 = {};
        wp::vec_t<3, wp::float32> adj_201 = {};
        wp::int32 adj_202 = {};
        wp::vec_t<3, wp::float32> adj_203 = {};
        wp::vec_t<3, wp::float32> adj_204 = {};
        wp::int32 adj_205 = {};
        wp::float32 adj_206 = {};
        wp::int32 adj_207 = {};
        wp::float32 adj_208 = {};
        wp::float32 adj_209 = {};
        wp::int32 adj_210 = {};
        wp::float32 adj_211 = {};
        wp::int32 adj_212 = {};
        wp::float32 adj_213 = {};
        wp::float32 adj_214 = {};
        wp::int32 adj_215 = {};
        wp::float32 adj_216 = {};
        wp::int32 adj_217 = {};
        wp::float32 adj_218 = {};
        wp::float32 adj_219 = {};
        wp::vec_t<3, wp::float32> adj_220 = {};
        wp::int32 adj_221 = {};
        wp::float32 adj_222 = {};
        wp::int32 adj_223 = {};
        wp::float32 adj_224 = {};
        wp::float32 adj_225 = {};
        wp::int32 adj_226 = {};
        wp::float32 adj_227 = {};
        wp::int32 adj_228 = {};
        wp::float32 adj_229 = {};
        wp::float32 adj_230 = {};
        wp::int32 adj_231 = {};
        wp::float32 adj_232 = {};
        wp::int32 adj_233 = {};
        wp::float32 adj_234 = {};
        wp::float32 adj_235 = {};
        wp::vec_t<3, wp::float32> adj_236 = {};
        wp::int32 adj_237 = {};
        wp::int32 adj_238 = {};
        wp::vec_t<3, wp::float32> adj_239 = {};
        wp::int32 adj_240 = {};
        wp::vec_t<3, wp::float32> adj_241 = {};
        wp::vec_t<3, wp::float32> adj_242 = {};
        wp::int32 adj_243 = {};
        wp::float32 adj_244 = {};
        wp::int32 adj_245 = {};
        wp::float32 adj_246 = {};
        wp::float32 adj_247 = {};
        wp::int32 adj_248 = {};
        wp::float32 adj_249 = {};
        wp::int32 adj_250 = {};
        wp::float32 adj_251 = {};
        wp::float32 adj_252 = {};
        wp::int32 adj_253 = {};
        wp::float32 adj_254 = {};
        wp::int32 adj_255 = {};
        wp::float32 adj_256 = {};
        wp::float32 adj_257 = {};
        wp::vec_t<3, wp::float32> adj_258 = {};
        wp::int32 adj_259 = {};
        wp::float32 adj_260 = {};
        wp::int32 adj_261 = {};
        wp::float32 adj_262 = {};
        wp::float32 adj_263 = {};
        wp::int32 adj_264 = {};
        wp::float32 adj_265 = {};
        wp::int32 adj_266 = {};
        wp::float32 adj_267 = {};
        wp::float32 adj_268 = {};
        wp::int32 adj_269 = {};
        wp::float32 adj_270 = {};
        wp::int32 adj_271 = {};
        wp::float32 adj_272 = {};
        wp::float32 adj_273 = {};
        wp::vec_t<3, wp::float32> adj_274 = {};
        //---------
        // forward
        // def build_cell_aabbs_kernel(                                                           <L 285>
        // c = wp.tid()                                                                           <L 298>
        var_0 = builtin_tid1d();
        // p = particle_q[cell_nodes[c, 0]]                                                       <L 299>
        var_2 = wp::address(var_cell_nodes, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::address(var_particle_q, var_4);
        var_6 = wp::load(var_3);
        var_5 = wp::copy(var_6);
        // lo = p                                                                                 <L 300>
        var_7 = wp::copy(var_5);
        // hi = p                                                                                 <L 301>
        var_8 = wp::copy(var_5);
        // for k in range(1, 8):                                                                  <L 302>
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_10 = wp::address(var_cell_nodes, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::address(var_particle_q, var_12);
        var_14 = wp::load(var_11);
        var_13 = wp::copy(var_14);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_16 = wp::extract(var_7, var_15);
        var_18 = wp::extract(var_13, var_17);
        var_19 = wp::min(var_16, var_18);
        var_21 = wp::extract(var_7, var_20);
        var_23 = wp::extract(var_13, var_22);
        var_24 = wp::min(var_21, var_23);
        var_26 = wp::extract(var_7, var_25);
        var_28 = wp::extract(var_13, var_27);
        var_29 = wp::min(var_26, var_28);
        var_30 = wp::vec_t<3, wp::float32>(var_19, var_24, var_29);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_32 = wp::extract(var_8, var_31);
        var_34 = wp::extract(var_13, var_33);
        var_35 = wp::max(var_32, var_34);
        var_37 = wp::extract(var_8, var_36);
        var_39 = wp::extract(var_13, var_38);
        var_40 = wp::max(var_37, var_39);
        var_42 = wp::extract(var_8, var_41);
        var_44 = wp::extract(var_13, var_43);
        var_45 = wp::max(var_42, var_44);
        var_46 = wp::vec_t<3, wp::float32>(var_35, var_40, var_45);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_48 = wp::address(var_cell_nodes, var_0, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::address(var_particle_q, var_50);
        var_52 = wp::load(var_49);
        var_51 = wp::copy(var_52);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_54 = wp::extract(var_30, var_53);
        var_56 = wp::extract(var_51, var_55);
        var_57 = wp::min(var_54, var_56);
        var_59 = wp::extract(var_30, var_58);
        var_61 = wp::extract(var_51, var_60);
        var_62 = wp::min(var_59, var_61);
        var_64 = wp::extract(var_30, var_63);
        var_66 = wp::extract(var_51, var_65);
        var_67 = wp::min(var_64, var_66);
        var_68 = wp::vec_t<3, wp::float32>(var_57, var_62, var_67);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_70 = wp::extract(var_46, var_69);
        var_72 = wp::extract(var_51, var_71);
        var_73 = wp::max(var_70, var_72);
        var_75 = wp::extract(var_46, var_74);
        var_77 = wp::extract(var_51, var_76);
        var_78 = wp::max(var_75, var_77);
        var_80 = wp::extract(var_46, var_79);
        var_82 = wp::extract(var_51, var_81);
        var_83 = wp::max(var_80, var_82);
        var_84 = wp::vec_t<3, wp::float32>(var_73, var_78, var_83);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_86 = wp::address(var_cell_nodes, var_0, var_85);
        var_88 = wp::load(var_86);
        var_87 = wp::address(var_particle_q, var_88);
        var_90 = wp::load(var_87);
        var_89 = wp::copy(var_90);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_92 = wp::extract(var_68, var_91);
        var_94 = wp::extract(var_89, var_93);
        var_95 = wp::min(var_92, var_94);
        var_97 = wp::extract(var_68, var_96);
        var_99 = wp::extract(var_89, var_98);
        var_100 = wp::min(var_97, var_99);
        var_102 = wp::extract(var_68, var_101);
        var_104 = wp::extract(var_89, var_103);
        var_105 = wp::min(var_102, var_104);
        var_106 = wp::vec_t<3, wp::float32>(var_95, var_100, var_105);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_108 = wp::extract(var_84, var_107);
        var_110 = wp::extract(var_89, var_109);
        var_111 = wp::max(var_108, var_110);
        var_113 = wp::extract(var_84, var_112);
        var_115 = wp::extract(var_89, var_114);
        var_116 = wp::max(var_113, var_115);
        var_118 = wp::extract(var_84, var_117);
        var_120 = wp::extract(var_89, var_119);
        var_121 = wp::max(var_118, var_120);
        var_122 = wp::vec_t<3, wp::float32>(var_111, var_116, var_121);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_124 = wp::address(var_cell_nodes, var_0, var_123);
        var_126 = wp::load(var_124);
        var_125 = wp::address(var_particle_q, var_126);
        var_128 = wp::load(var_125);
        var_127 = wp::copy(var_128);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_130 = wp::extract(var_106, var_129);
        var_132 = wp::extract(var_127, var_131);
        var_133 = wp::min(var_130, var_132);
        var_135 = wp::extract(var_106, var_134);
        var_137 = wp::extract(var_127, var_136);
        var_138 = wp::min(var_135, var_137);
        var_140 = wp::extract(var_106, var_139);
        var_142 = wp::extract(var_127, var_141);
        var_143 = wp::min(var_140, var_142);
        var_144 = wp::vec_t<3, wp::float32>(var_133, var_138, var_143);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_146 = wp::extract(var_122, var_145);
        var_148 = wp::extract(var_127, var_147);
        var_149 = wp::max(var_146, var_148);
        var_151 = wp::extract(var_122, var_150);
        var_153 = wp::extract(var_127, var_152);
        var_154 = wp::max(var_151, var_153);
        var_156 = wp::extract(var_122, var_155);
        var_158 = wp::extract(var_127, var_157);
        var_159 = wp::max(var_156, var_158);
        var_160 = wp::vec_t<3, wp::float32>(var_149, var_154, var_159);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_162 = wp::address(var_cell_nodes, var_0, var_161);
        var_164 = wp::load(var_162);
        var_163 = wp::address(var_particle_q, var_164);
        var_166 = wp::load(var_163);
        var_165 = wp::copy(var_166);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_168 = wp::extract(var_144, var_167);
        var_170 = wp::extract(var_165, var_169);
        var_171 = wp::min(var_168, var_170);
        var_173 = wp::extract(var_144, var_172);
        var_175 = wp::extract(var_165, var_174);
        var_176 = wp::min(var_173, var_175);
        var_178 = wp::extract(var_144, var_177);
        var_180 = wp::extract(var_165, var_179);
        var_181 = wp::min(var_178, var_180);
        var_182 = wp::vec_t<3, wp::float32>(var_171, var_176, var_181);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_184 = wp::extract(var_160, var_183);
        var_186 = wp::extract(var_165, var_185);
        var_187 = wp::max(var_184, var_186);
        var_189 = wp::extract(var_160, var_188);
        var_191 = wp::extract(var_165, var_190);
        var_192 = wp::max(var_189, var_191);
        var_194 = wp::extract(var_160, var_193);
        var_196 = wp::extract(var_165, var_195);
        var_197 = wp::max(var_194, var_196);
        var_198 = wp::vec_t<3, wp::float32>(var_187, var_192, var_197);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_200 = wp::address(var_cell_nodes, var_0, var_199);
        var_202 = wp::load(var_200);
        var_201 = wp::address(var_particle_q, var_202);
        var_204 = wp::load(var_201);
        var_203 = wp::copy(var_204);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_206 = wp::extract(var_182, var_205);
        var_208 = wp::extract(var_203, var_207);
        var_209 = wp::min(var_206, var_208);
        var_211 = wp::extract(var_182, var_210);
        var_213 = wp::extract(var_203, var_212);
        var_214 = wp::min(var_211, var_213);
        var_216 = wp::extract(var_182, var_215);
        var_218 = wp::extract(var_203, var_217);
        var_219 = wp::min(var_216, var_218);
        var_220 = wp::vec_t<3, wp::float32>(var_209, var_214, var_219);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_222 = wp::extract(var_198, var_221);
        var_224 = wp::extract(var_203, var_223);
        var_225 = wp::max(var_222, var_224);
        var_227 = wp::extract(var_198, var_226);
        var_229 = wp::extract(var_203, var_228);
        var_230 = wp::max(var_227, var_229);
        var_232 = wp::extract(var_198, var_231);
        var_234 = wp::extract(var_203, var_233);
        var_235 = wp::max(var_232, var_234);
        var_236 = wp::vec_t<3, wp::float32>(var_225, var_230, var_235);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 303>
        var_238 = wp::address(var_cell_nodes, var_0, var_237);
        var_240 = wp::load(var_238);
        var_239 = wp::address(var_particle_q, var_240);
        var_242 = wp::load(var_239);
        var_241 = wp::copy(var_242);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 304>
        var_244 = wp::extract(var_220, var_243);
        var_246 = wp::extract(var_241, var_245);
        var_247 = wp::min(var_244, var_246);
        var_249 = wp::extract(var_220, var_248);
        var_251 = wp::extract(var_241, var_250);
        var_252 = wp::min(var_249, var_251);
        var_254 = wp::extract(var_220, var_253);
        var_256 = wp::extract(var_241, var_255);
        var_257 = wp::min(var_254, var_256);
        var_258 = wp::vec_t<3, wp::float32>(var_247, var_252, var_257);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 305>
        var_260 = wp::extract(var_236, var_259);
        var_262 = wp::extract(var_241, var_261);
        var_263 = wp::max(var_260, var_262);
        var_265 = wp::extract(var_236, var_264);
        var_267 = wp::extract(var_241, var_266);
        var_268 = wp::max(var_265, var_267);
        var_270 = wp::extract(var_236, var_269);
        var_272 = wp::extract(var_241, var_271);
        var_273 = wp::max(var_270, var_272);
        var_274 = wp::vec_t<3, wp::float32>(var_263, var_268, var_273);
        // aabb_min[c] = lo                                                                       <L 306>
        // wp::array_store(var_aabb_min, var_0, var_258);
        // aabb_max[c] = hi                                                                       <L 307>
        // wp::array_store(var_aabb_max, var_0, var_274);
        //---------
        // reverse
        wp::adj_array_store(var_aabb_max, var_0, var_274, adj_aabb_max, adj_0, adj_274);
        // adj: aabb_max[c] = hi                                                                  <L 307>
        wp::adj_array_store(var_aabb_min, var_0, var_258, adj_aabb_min, adj_0, adj_258);
        // adj: aabb_min[c] = lo                                                                  <L 306>
        wp::adj_vec_t(var_263, var_268, var_273, adj_263, adj_268, adj_273, adj_274);
        wp::adj_max(var_270, var_272, adj_270, adj_272, adj_273);
        wp::adj_extract(var_241, var_271, adj_241, adj_271, adj_272);
        wp::adj_extract(var_236, var_269, adj_236, adj_269, adj_270);
        wp::adj_max(var_265, var_267, adj_265, adj_267, adj_268);
        wp::adj_extract(var_241, var_266, adj_241, adj_266, adj_267);
        wp::adj_extract(var_236, var_264, adj_236, adj_264, adj_265);
        wp::adj_max(var_260, var_262, adj_260, adj_262, adj_263);
        wp::adj_extract(var_241, var_261, adj_241, adj_261, adj_262);
        wp::adj_extract(var_236, var_259, adj_236, adj_259, adj_260);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_247, var_252, var_257, adj_247, adj_252, adj_257, adj_258);
        wp::adj_min(var_254, var_256, adj_254, adj_256, adj_257);
        wp::adj_extract(var_241, var_255, adj_241, adj_255, adj_256);
        wp::adj_extract(var_220, var_253, adj_220, adj_253, adj_254);
        wp::adj_min(var_249, var_251, adj_249, adj_251, adj_252);
        wp::adj_extract(var_241, var_250, adj_241, adj_250, adj_251);
        wp::adj_extract(var_220, var_248, adj_220, adj_248, adj_249);
        wp::adj_min(var_244, var_246, adj_244, adj_246, adj_247);
        wp::adj_extract(var_241, var_245, adj_241, adj_245, adj_246);
        wp::adj_extract(var_220, var_243, adj_220, adj_243, adj_244);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_242, adj_239, adj_241);
        wp::adj_address(var_particle_q, var_240, adj_particle_q, adj_238, adj_239);
        wp::adj_address(var_cell_nodes, var_0, var_237, adj_cell_nodes, adj_0, adj_237, adj_238);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        wp::adj_vec_t(var_225, var_230, var_235, adj_225, adj_230, adj_235, adj_236);
        wp::adj_max(var_232, var_234, adj_232, adj_234, adj_235);
        wp::adj_extract(var_203, var_233, adj_203, adj_233, adj_234);
        wp::adj_extract(var_198, var_231, adj_198, adj_231, adj_232);
        wp::adj_max(var_227, var_229, adj_227, adj_229, adj_230);
        wp::adj_extract(var_203, var_228, adj_203, adj_228, adj_229);
        wp::adj_extract(var_198, var_226, adj_198, adj_226, adj_227);
        wp::adj_max(var_222, var_224, adj_222, adj_224, adj_225);
        wp::adj_extract(var_203, var_223, adj_203, adj_223, adj_224);
        wp::adj_extract(var_198, var_221, adj_198, adj_221, adj_222);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_209, var_214, var_219, adj_209, adj_214, adj_219, adj_220);
        wp::adj_min(var_216, var_218, adj_216, adj_218, adj_219);
        wp::adj_extract(var_203, var_217, adj_203, adj_217, adj_218);
        wp::adj_extract(var_182, var_215, adj_182, adj_215, adj_216);
        wp::adj_min(var_211, var_213, adj_211, adj_213, adj_214);
        wp::adj_extract(var_203, var_212, adj_203, adj_212, adj_213);
        wp::adj_extract(var_182, var_210, adj_182, adj_210, adj_211);
        wp::adj_min(var_206, var_208, adj_206, adj_208, adj_209);
        wp::adj_extract(var_203, var_207, adj_203, adj_207, adj_208);
        wp::adj_extract(var_182, var_205, adj_182, adj_205, adj_206);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_204, adj_201, adj_203);
        wp::adj_address(var_particle_q, var_202, adj_particle_q, adj_200, adj_201);
        wp::adj_address(var_cell_nodes, var_0, var_199, adj_cell_nodes, adj_0, adj_199, adj_200);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        wp::adj_vec_t(var_187, var_192, var_197, adj_187, adj_192, adj_197, adj_198);
        wp::adj_max(var_194, var_196, adj_194, adj_196, adj_197);
        wp::adj_extract(var_165, var_195, adj_165, adj_195, adj_196);
        wp::adj_extract(var_160, var_193, adj_160, adj_193, adj_194);
        wp::adj_max(var_189, var_191, adj_189, adj_191, adj_192);
        wp::adj_extract(var_165, var_190, adj_165, adj_190, adj_191);
        wp::adj_extract(var_160, var_188, adj_160, adj_188, adj_189);
        wp::adj_max(var_184, var_186, adj_184, adj_186, adj_187);
        wp::adj_extract(var_165, var_185, adj_165, adj_185, adj_186);
        wp::adj_extract(var_160, var_183, adj_160, adj_183, adj_184);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_171, var_176, var_181, adj_171, adj_176, adj_181, adj_182);
        wp::adj_min(var_178, var_180, adj_178, adj_180, adj_181);
        wp::adj_extract(var_165, var_179, adj_165, adj_179, adj_180);
        wp::adj_extract(var_144, var_177, adj_144, adj_177, adj_178);
        wp::adj_min(var_173, var_175, adj_173, adj_175, adj_176);
        wp::adj_extract(var_165, var_174, adj_165, adj_174, adj_175);
        wp::adj_extract(var_144, var_172, adj_144, adj_172, adj_173);
        wp::adj_min(var_168, var_170, adj_168, adj_170, adj_171);
        wp::adj_extract(var_165, var_169, adj_165, adj_169, adj_170);
        wp::adj_extract(var_144, var_167, adj_144, adj_167, adj_168);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_166, adj_163, adj_165);
        wp::adj_address(var_particle_q, var_164, adj_particle_q, adj_162, adj_163);
        wp::adj_address(var_cell_nodes, var_0, var_161, adj_cell_nodes, adj_0, adj_161, adj_162);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        wp::adj_vec_t(var_149, var_154, var_159, adj_149, adj_154, adj_159, adj_160);
        wp::adj_max(var_156, var_158, adj_156, adj_158, adj_159);
        wp::adj_extract(var_127, var_157, adj_127, adj_157, adj_158);
        wp::adj_extract(var_122, var_155, adj_122, adj_155, adj_156);
        wp::adj_max(var_151, var_153, adj_151, adj_153, adj_154);
        wp::adj_extract(var_127, var_152, adj_127, adj_152, adj_153);
        wp::adj_extract(var_122, var_150, adj_122, adj_150, adj_151);
        wp::adj_max(var_146, var_148, adj_146, adj_148, adj_149);
        wp::adj_extract(var_127, var_147, adj_127, adj_147, adj_148);
        wp::adj_extract(var_122, var_145, adj_122, adj_145, adj_146);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_133, var_138, var_143, adj_133, adj_138, adj_143, adj_144);
        wp::adj_min(var_140, var_142, adj_140, adj_142, adj_143);
        wp::adj_extract(var_127, var_141, adj_127, adj_141, adj_142);
        wp::adj_extract(var_106, var_139, adj_106, adj_139, adj_140);
        wp::adj_min(var_135, var_137, adj_135, adj_137, adj_138);
        wp::adj_extract(var_127, var_136, adj_127, adj_136, adj_137);
        wp::adj_extract(var_106, var_134, adj_106, adj_134, adj_135);
        wp::adj_min(var_130, var_132, adj_130, adj_132, adj_133);
        wp::adj_extract(var_127, var_131, adj_127, adj_131, adj_132);
        wp::adj_extract(var_106, var_129, adj_106, adj_129, adj_130);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_128, adj_125, adj_127);
        wp::adj_address(var_particle_q, var_126, adj_particle_q, adj_124, adj_125);
        wp::adj_address(var_cell_nodes, var_0, var_123, adj_cell_nodes, adj_0, adj_123, adj_124);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        wp::adj_vec_t(var_111, var_116, var_121, adj_111, adj_116, adj_121, adj_122);
        wp::adj_max(var_118, var_120, adj_118, adj_120, adj_121);
        wp::adj_extract(var_89, var_119, adj_89, adj_119, adj_120);
        wp::adj_extract(var_84, var_117, adj_84, adj_117, adj_118);
        wp::adj_max(var_113, var_115, adj_113, adj_115, adj_116);
        wp::adj_extract(var_89, var_114, adj_89, adj_114, adj_115);
        wp::adj_extract(var_84, var_112, adj_84, adj_112, adj_113);
        wp::adj_max(var_108, var_110, adj_108, adj_110, adj_111);
        wp::adj_extract(var_89, var_109, adj_89, adj_109, adj_110);
        wp::adj_extract(var_84, var_107, adj_84, adj_107, adj_108);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_95, var_100, var_105, adj_95, adj_100, adj_105, adj_106);
        wp::adj_min(var_102, var_104, adj_102, adj_104, adj_105);
        wp::adj_extract(var_89, var_103, adj_89, adj_103, adj_104);
        wp::adj_extract(var_68, var_101, adj_68, adj_101, adj_102);
        wp::adj_min(var_97, var_99, adj_97, adj_99, adj_100);
        wp::adj_extract(var_89, var_98, adj_89, adj_98, adj_99);
        wp::adj_extract(var_68, var_96, adj_68, adj_96, adj_97);
        wp::adj_min(var_92, var_94, adj_92, adj_94, adj_95);
        wp::adj_extract(var_89, var_93, adj_89, adj_93, adj_94);
        wp::adj_extract(var_68, var_91, adj_68, adj_91, adj_92);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_90, adj_87, adj_89);
        wp::adj_address(var_particle_q, var_88, adj_particle_q, adj_86, adj_87);
        wp::adj_address(var_cell_nodes, var_0, var_85, adj_cell_nodes, adj_0, adj_85, adj_86);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        wp::adj_vec_t(var_73, var_78, var_83, adj_73, adj_78, adj_83, adj_84);
        wp::adj_max(var_80, var_82, adj_80, adj_82, adj_83);
        wp::adj_extract(var_51, var_81, adj_51, adj_81, adj_82);
        wp::adj_extract(var_46, var_79, adj_46, adj_79, adj_80);
        wp::adj_max(var_75, var_77, adj_75, adj_77, adj_78);
        wp::adj_extract(var_51, var_76, adj_51, adj_76, adj_77);
        wp::adj_extract(var_46, var_74, adj_46, adj_74, adj_75);
        wp::adj_max(var_70, var_72, adj_70, adj_72, adj_73);
        wp::adj_extract(var_51, var_71, adj_51, adj_71, adj_72);
        wp::adj_extract(var_46, var_69, adj_46, adj_69, adj_70);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_57, var_62, var_67, adj_57, adj_62, adj_67, adj_68);
        wp::adj_min(var_64, var_66, adj_64, adj_66, adj_67);
        wp::adj_extract(var_51, var_65, adj_51, adj_65, adj_66);
        wp::adj_extract(var_30, var_63, adj_30, adj_63, adj_64);
        wp::adj_min(var_59, var_61, adj_59, adj_61, adj_62);
        wp::adj_extract(var_51, var_60, adj_51, adj_60, adj_61);
        wp::adj_extract(var_30, var_58, adj_30, adj_58, adj_59);
        wp::adj_min(var_54, var_56, adj_54, adj_56, adj_57);
        wp::adj_extract(var_51, var_55, adj_51, adj_55, adj_56);
        wp::adj_extract(var_30, var_53, adj_30, adj_53, adj_54);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_52, adj_49, adj_51);
        wp::adj_address(var_particle_q, var_50, adj_particle_q, adj_48, adj_49);
        wp::adj_address(var_cell_nodes, var_0, var_47, adj_cell_nodes, adj_0, adj_47, adj_48);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        wp::adj_vec_t(var_35, var_40, var_45, adj_35, adj_40, adj_45, adj_46);
        wp::adj_max(var_42, var_44, adj_42, adj_44, adj_45);
        wp::adj_extract(var_13, var_43, adj_13, adj_43, adj_44);
        wp::adj_extract(var_8, var_41, adj_8, adj_41, adj_42);
        wp::adj_max(var_37, var_39, adj_37, adj_39, adj_40);
        wp::adj_extract(var_13, var_38, adj_13, adj_38, adj_39);
        wp::adj_extract(var_8, var_36, adj_8, adj_36, adj_37);
        wp::adj_max(var_32, var_34, adj_32, adj_34, adj_35);
        wp::adj_extract(var_13, var_33, adj_13, adj_33, adj_34);
        wp::adj_extract(var_8, var_31, adj_8, adj_31, adj_32);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 305>
        wp::adj_vec_t(var_19, var_24, var_29, adj_19, adj_24, adj_29, adj_30);
        wp::adj_min(var_26, var_28, adj_26, adj_28, adj_29);
        wp::adj_extract(var_13, var_27, adj_13, adj_27, adj_28);
        wp::adj_extract(var_7, var_25, adj_7, adj_25, adj_26);
        wp::adj_min(var_21, var_23, adj_21, adj_23, adj_24);
        wp::adj_extract(var_13, var_22, adj_13, adj_22, adj_23);
        wp::adj_extract(var_7, var_20, adj_7, adj_20, adj_21);
        wp::adj_min(var_16, var_18, adj_16, adj_18, adj_19);
        wp::adj_extract(var_13, var_17, adj_13, adj_17, adj_18);
        wp::adj_extract(var_7, var_15, adj_7, adj_15, adj_16);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 304>
        wp::adj_copy(var_14, adj_11, adj_13);
        wp::adj_address(var_particle_q, var_12, adj_particle_q, adj_10, adj_11);
        wp::adj_address(var_cell_nodes, var_0, var_9, adj_cell_nodes, adj_0, adj_9, adj_10);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 303>
        // adj: for k in range(1, 8):                                                             <L 302>
        wp::adj_copy(var_5, adj_5, adj_8);
        // adj: hi = p                                                                            <L 301>
        wp::adj_copy(var_5, adj_5, adj_7);
        // adj: lo = p                                                                            <L 300>
        wp::adj_copy(var_6, adj_3, adj_5);
        wp::adj_address(var_particle_q, var_4, adj_particle_q, adj_2, adj_3);
        wp::adj_address(var_cell_nodes, var_0, var_1, adj_cell_nodes, adj_0, adj_1, adj_2);
        // adj: p = particle_q[cell_nodes[c, 0]]                                                  <L 299>
        // adj: c = wp.tid()                                                                      <L 298>
        // adj: def build_cell_aabbs_kernel(                                                      <L 285>
        continue;
    }
}



extern "C" __global__ void build_dirty_cell_aabbs_kernel_fc34d272_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dirty_cell_ids,
    wp::array_t<wp::int32> var_dirty_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max)
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
        const wp::int32 var_12 = 0;
        wp::int32* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        const wp::int32 var_20 = 1;
        wp::int32* var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::int32 var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        const wp::int32 var_28 = 0;
        wp::float32 var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 1;
        wp::float32 var_32;
        const wp::int32 var_33 = 1;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 2;
        wp::float32 var_37;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 0;
        wp::float32 var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 1;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 2;
        wp::float32 var_53;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        const wp::int32 var_58 = 2;
        wp::int32* var_59;
        wp::vec_t<3, wp::float32>* var_60;
        wp::int32 var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        const wp::int32 var_66 = 0;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::float32 var_70;
        const wp::int32 var_71 = 1;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 2;
        wp::float32 var_75;
        const wp::int32 var_76 = 2;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        const wp::int32 var_80 = 0;
        wp::float32 var_81;
        const wp::int32 var_82 = 0;
        wp::float32 var_83;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 2;
        wp::float32 var_91;
        const wp::int32 var_92 = 2;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::vec_t<3, wp::float32> var_95;
        const wp::int32 var_96 = 3;
        wp::int32* var_97;
        wp::vec_t<3, wp::float32>* var_98;
        wp::int32 var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        const wp::int32 var_102 = 0;
        wp::float32 var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        wp::float32 var_106;
        const wp::int32 var_107 = 1;
        wp::float32 var_108;
        const wp::int32 var_109 = 1;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 2;
        wp::float32 var_113;
        const wp::int32 var_114 = 2;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::int32 var_118 = 0;
        wp::float32 var_119;
        const wp::int32 var_120 = 0;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::int32 var_123 = 1;
        wp::float32 var_124;
        const wp::int32 var_125 = 1;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 2;
        wp::float32 var_129;
        const wp::int32 var_130 = 2;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 4;
        wp::int32* var_135;
        wp::vec_t<3, wp::float32>* var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        const wp::int32 var_140 = 0;
        wp::float32 var_141;
        const wp::int32 var_142 = 0;
        wp::float32 var_143;
        wp::float32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        const wp::int32 var_152 = 2;
        wp::float32 var_153;
        wp::float32 var_154;
        wp::vec_t<3, wp::float32> var_155;
        const wp::int32 var_156 = 0;
        wp::float32 var_157;
        const wp::int32 var_158 = 0;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 1;
        wp::float32 var_162;
        const wp::int32 var_163 = 1;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 2;
        wp::float32 var_167;
        const wp::int32 var_168 = 2;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::vec_t<3, wp::float32> var_171;
        const wp::int32 var_172 = 5;
        wp::int32* var_173;
        wp::vec_t<3, wp::float32>* var_174;
        wp::int32 var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::vec_t<3, wp::float32> var_177;
        const wp::int32 var_178 = 0;
        wp::float32 var_179;
        const wp::int32 var_180 = 0;
        wp::float32 var_181;
        wp::float32 var_182;
        const wp::int32 var_183 = 1;
        wp::float32 var_184;
        const wp::int32 var_185 = 1;
        wp::float32 var_186;
        wp::float32 var_187;
        const wp::int32 var_188 = 2;
        wp::float32 var_189;
        const wp::int32 var_190 = 2;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::vec_t<3, wp::float32> var_193;
        const wp::int32 var_194 = 0;
        wp::float32 var_195;
        const wp::int32 var_196 = 0;
        wp::float32 var_197;
        wp::float32 var_198;
        const wp::int32 var_199 = 1;
        wp::float32 var_200;
        const wp::int32 var_201 = 1;
        wp::float32 var_202;
        wp::float32 var_203;
        const wp::int32 var_204 = 2;
        wp::float32 var_205;
        const wp::int32 var_206 = 2;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::vec_t<3, wp::float32> var_209;
        const wp::int32 var_210 = 6;
        wp::int32* var_211;
        wp::vec_t<3, wp::float32>* var_212;
        wp::int32 var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::vec_t<3, wp::float32> var_215;
        const wp::int32 var_216 = 0;
        wp::float32 var_217;
        const wp::int32 var_218 = 0;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::int32 var_221 = 1;
        wp::float32 var_222;
        const wp::int32 var_223 = 1;
        wp::float32 var_224;
        wp::float32 var_225;
        const wp::int32 var_226 = 2;
        wp::float32 var_227;
        const wp::int32 var_228 = 2;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::vec_t<3, wp::float32> var_231;
        const wp::int32 var_232 = 0;
        wp::float32 var_233;
        const wp::int32 var_234 = 0;
        wp::float32 var_235;
        wp::float32 var_236;
        const wp::int32 var_237 = 1;
        wp::float32 var_238;
        const wp::int32 var_239 = 1;
        wp::float32 var_240;
        wp::float32 var_241;
        const wp::int32 var_242 = 2;
        wp::float32 var_243;
        const wp::int32 var_244 = 2;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::vec_t<3, wp::float32> var_247;
        const wp::int32 var_248 = 7;
        wp::int32* var_249;
        wp::vec_t<3, wp::float32>* var_250;
        wp::int32 var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<3, wp::float32> var_253;
        const wp::int32 var_254 = 0;
        wp::float32 var_255;
        const wp::int32 var_256 = 0;
        wp::float32 var_257;
        wp::float32 var_258;
        const wp::int32 var_259 = 1;
        wp::float32 var_260;
        const wp::int32 var_261 = 1;
        wp::float32 var_262;
        wp::float32 var_263;
        const wp::int32 var_264 = 2;
        wp::float32 var_265;
        const wp::int32 var_266 = 2;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::vec_t<3, wp::float32> var_269;
        const wp::int32 var_270 = 0;
        wp::float32 var_271;
        const wp::int32 var_272 = 0;
        wp::float32 var_273;
        wp::float32 var_274;
        const wp::int32 var_275 = 1;
        wp::float32 var_276;
        const wp::int32 var_277 = 1;
        wp::float32 var_278;
        wp::float32 var_279;
        const wp::int32 var_280 = 2;
        wp::float32 var_281;
        const wp::int32 var_282 = 2;
        wp::float32 var_283;
        wp::float32 var_284;
        wp::vec_t<3, wp::float32> var_285;
        //---------
        // forward
        // def build_dirty_cell_aabbs_kernel(                                                     <L 311>
        // i = wp.tid()                                                                           <L 321>
        var_0 = builtin_tid1d();
        // if i >= dirty_count[0]:                                                                <L 322>
        var_2 = wp::address(var_dirty_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 323>
            continue;
        }
        // c = dirty_cell_ids[i]                                                                  <L 324>
        var_5 = wp::address(var_dirty_cell_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if c < 0 or c >= num_cells:                                                            <L 325>
        var_10 = (var_6 < var_9);
        var_8 = var_10;
        if (!var_8) {
            var_11 = (var_6 >= var_num_cells);
            var_8 = var_8 || var_11;
        }
        if (var_8) {
            // return                                                                             <L 326>
            continue;
        }
        // p = particle_q[cell_nodes[c, 0]]                                                       <L 328>
        var_13 = wp::address(var_cell_nodes, var_6, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_particle_q, var_15);
        var_17 = wp::load(var_14);
        var_16 = wp::copy(var_17);
        // lo = p                                                                                 <L 329>
        var_18 = wp::copy(var_16);
        // hi = p                                                                                 <L 330>
        var_19 = wp::copy(var_16);
        // for k in range(1, 8):                                                                  <L 331>
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_21 = wp::address(var_cell_nodes, var_6, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::address(var_particle_q, var_23);
        var_25 = wp::load(var_22);
        var_24 = wp::copy(var_25);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_27 = wp::extract(var_18, var_26);
        var_29 = wp::extract(var_24, var_28);
        var_30 = wp::min(var_27, var_29);
        var_32 = wp::extract(var_18, var_31);
        var_34 = wp::extract(var_24, var_33);
        var_35 = wp::min(var_32, var_34);
        var_37 = wp::extract(var_18, var_36);
        var_39 = wp::extract(var_24, var_38);
        var_40 = wp::min(var_37, var_39);
        var_41 = wp::vec_t<3, wp::float32>(var_30, var_35, var_40);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_43 = wp::extract(var_19, var_42);
        var_45 = wp::extract(var_24, var_44);
        var_46 = wp::max(var_43, var_45);
        var_48 = wp::extract(var_19, var_47);
        var_50 = wp::extract(var_24, var_49);
        var_51 = wp::max(var_48, var_50);
        var_53 = wp::extract(var_19, var_52);
        var_55 = wp::extract(var_24, var_54);
        var_56 = wp::max(var_53, var_55);
        var_57 = wp::vec_t<3, wp::float32>(var_46, var_51, var_56);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_59 = wp::address(var_cell_nodes, var_6, var_58);
        var_61 = wp::load(var_59);
        var_60 = wp::address(var_particle_q, var_61);
        var_63 = wp::load(var_60);
        var_62 = wp::copy(var_63);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_65 = wp::extract(var_41, var_64);
        var_67 = wp::extract(var_62, var_66);
        var_68 = wp::min(var_65, var_67);
        var_70 = wp::extract(var_41, var_69);
        var_72 = wp::extract(var_62, var_71);
        var_73 = wp::min(var_70, var_72);
        var_75 = wp::extract(var_41, var_74);
        var_77 = wp::extract(var_62, var_76);
        var_78 = wp::min(var_75, var_77);
        var_79 = wp::vec_t<3, wp::float32>(var_68, var_73, var_78);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_81 = wp::extract(var_57, var_80);
        var_83 = wp::extract(var_62, var_82);
        var_84 = wp::max(var_81, var_83);
        var_86 = wp::extract(var_57, var_85);
        var_88 = wp::extract(var_62, var_87);
        var_89 = wp::max(var_86, var_88);
        var_91 = wp::extract(var_57, var_90);
        var_93 = wp::extract(var_62, var_92);
        var_94 = wp::max(var_91, var_93);
        var_95 = wp::vec_t<3, wp::float32>(var_84, var_89, var_94);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_97 = wp::address(var_cell_nodes, var_6, var_96);
        var_99 = wp::load(var_97);
        var_98 = wp::address(var_particle_q, var_99);
        var_101 = wp::load(var_98);
        var_100 = wp::copy(var_101);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_103 = wp::extract(var_79, var_102);
        var_105 = wp::extract(var_100, var_104);
        var_106 = wp::min(var_103, var_105);
        var_108 = wp::extract(var_79, var_107);
        var_110 = wp::extract(var_100, var_109);
        var_111 = wp::min(var_108, var_110);
        var_113 = wp::extract(var_79, var_112);
        var_115 = wp::extract(var_100, var_114);
        var_116 = wp::min(var_113, var_115);
        var_117 = wp::vec_t<3, wp::float32>(var_106, var_111, var_116);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_119 = wp::extract(var_95, var_118);
        var_121 = wp::extract(var_100, var_120);
        var_122 = wp::max(var_119, var_121);
        var_124 = wp::extract(var_95, var_123);
        var_126 = wp::extract(var_100, var_125);
        var_127 = wp::max(var_124, var_126);
        var_129 = wp::extract(var_95, var_128);
        var_131 = wp::extract(var_100, var_130);
        var_132 = wp::max(var_129, var_131);
        var_133 = wp::vec_t<3, wp::float32>(var_122, var_127, var_132);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_135 = wp::address(var_cell_nodes, var_6, var_134);
        var_137 = wp::load(var_135);
        var_136 = wp::address(var_particle_q, var_137);
        var_139 = wp::load(var_136);
        var_138 = wp::copy(var_139);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_141 = wp::extract(var_117, var_140);
        var_143 = wp::extract(var_138, var_142);
        var_144 = wp::min(var_141, var_143);
        var_146 = wp::extract(var_117, var_145);
        var_148 = wp::extract(var_138, var_147);
        var_149 = wp::min(var_146, var_148);
        var_151 = wp::extract(var_117, var_150);
        var_153 = wp::extract(var_138, var_152);
        var_154 = wp::min(var_151, var_153);
        var_155 = wp::vec_t<3, wp::float32>(var_144, var_149, var_154);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_157 = wp::extract(var_133, var_156);
        var_159 = wp::extract(var_138, var_158);
        var_160 = wp::max(var_157, var_159);
        var_162 = wp::extract(var_133, var_161);
        var_164 = wp::extract(var_138, var_163);
        var_165 = wp::max(var_162, var_164);
        var_167 = wp::extract(var_133, var_166);
        var_169 = wp::extract(var_138, var_168);
        var_170 = wp::max(var_167, var_169);
        var_171 = wp::vec_t<3, wp::float32>(var_160, var_165, var_170);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_173 = wp::address(var_cell_nodes, var_6, var_172);
        var_175 = wp::load(var_173);
        var_174 = wp::address(var_particle_q, var_175);
        var_177 = wp::load(var_174);
        var_176 = wp::copy(var_177);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_179 = wp::extract(var_155, var_178);
        var_181 = wp::extract(var_176, var_180);
        var_182 = wp::min(var_179, var_181);
        var_184 = wp::extract(var_155, var_183);
        var_186 = wp::extract(var_176, var_185);
        var_187 = wp::min(var_184, var_186);
        var_189 = wp::extract(var_155, var_188);
        var_191 = wp::extract(var_176, var_190);
        var_192 = wp::min(var_189, var_191);
        var_193 = wp::vec_t<3, wp::float32>(var_182, var_187, var_192);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_195 = wp::extract(var_171, var_194);
        var_197 = wp::extract(var_176, var_196);
        var_198 = wp::max(var_195, var_197);
        var_200 = wp::extract(var_171, var_199);
        var_202 = wp::extract(var_176, var_201);
        var_203 = wp::max(var_200, var_202);
        var_205 = wp::extract(var_171, var_204);
        var_207 = wp::extract(var_176, var_206);
        var_208 = wp::max(var_205, var_207);
        var_209 = wp::vec_t<3, wp::float32>(var_198, var_203, var_208);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_211 = wp::address(var_cell_nodes, var_6, var_210);
        var_213 = wp::load(var_211);
        var_212 = wp::address(var_particle_q, var_213);
        var_215 = wp::load(var_212);
        var_214 = wp::copy(var_215);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_217 = wp::extract(var_193, var_216);
        var_219 = wp::extract(var_214, var_218);
        var_220 = wp::min(var_217, var_219);
        var_222 = wp::extract(var_193, var_221);
        var_224 = wp::extract(var_214, var_223);
        var_225 = wp::min(var_222, var_224);
        var_227 = wp::extract(var_193, var_226);
        var_229 = wp::extract(var_214, var_228);
        var_230 = wp::min(var_227, var_229);
        var_231 = wp::vec_t<3, wp::float32>(var_220, var_225, var_230);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_233 = wp::extract(var_209, var_232);
        var_235 = wp::extract(var_214, var_234);
        var_236 = wp::max(var_233, var_235);
        var_238 = wp::extract(var_209, var_237);
        var_240 = wp::extract(var_214, var_239);
        var_241 = wp::max(var_238, var_240);
        var_243 = wp::extract(var_209, var_242);
        var_245 = wp::extract(var_214, var_244);
        var_246 = wp::max(var_243, var_245);
        var_247 = wp::vec_t<3, wp::float32>(var_236, var_241, var_246);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_249 = wp::address(var_cell_nodes, var_6, var_248);
        var_251 = wp::load(var_249);
        var_250 = wp::address(var_particle_q, var_251);
        var_253 = wp::load(var_250);
        var_252 = wp::copy(var_253);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_255 = wp::extract(var_231, var_254);
        var_257 = wp::extract(var_252, var_256);
        var_258 = wp::min(var_255, var_257);
        var_260 = wp::extract(var_231, var_259);
        var_262 = wp::extract(var_252, var_261);
        var_263 = wp::min(var_260, var_262);
        var_265 = wp::extract(var_231, var_264);
        var_267 = wp::extract(var_252, var_266);
        var_268 = wp::min(var_265, var_267);
        var_269 = wp::vec_t<3, wp::float32>(var_258, var_263, var_268);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_271 = wp::extract(var_247, var_270);
        var_273 = wp::extract(var_252, var_272);
        var_274 = wp::max(var_271, var_273);
        var_276 = wp::extract(var_247, var_275);
        var_278 = wp::extract(var_252, var_277);
        var_279 = wp::max(var_276, var_278);
        var_281 = wp::extract(var_247, var_280);
        var_283 = wp::extract(var_252, var_282);
        var_284 = wp::max(var_281, var_283);
        var_285 = wp::vec_t<3, wp::float32>(var_274, var_279, var_284);
        // aabb_min[c] = lo                                                                       <L 335>
        wp::array_store(var_aabb_min, var_6, var_269);
        // aabb_max[c] = hi                                                                       <L 336>
        wp::array_store(var_aabb_max, var_6, var_285);
    }
}



extern "C" __global__ void build_dirty_cell_aabbs_kernel_fc34d272_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dirty_cell_ids,
    wp::array_t<wp::int32> var_dirty_count,
    wp::int32 var_num_cells,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max,
    wp::array_t<wp::int32> adj_dirty_cell_ids,
    wp::array_t<wp::int32> adj_dirty_count,
    wp::int32 adj_num_cells,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_max)
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
        const wp::int32 var_12 = 0;
        wp::int32* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32> var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32> var_19;
        const wp::int32 var_20 = 1;
        wp::int32* var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::int32 var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        const wp::int32 var_26 = 0;
        wp::float32 var_27;
        const wp::int32 var_28 = 0;
        wp::float32 var_29;
        wp::float32 var_30;
        const wp::int32 var_31 = 1;
        wp::float32 var_32;
        const wp::int32 var_33 = 1;
        wp::float32 var_34;
        wp::float32 var_35;
        const wp::int32 var_36 = 2;
        wp::float32 var_37;
        const wp::int32 var_38 = 2;
        wp::float32 var_39;
        wp::float32 var_40;
        wp::vec_t<3, wp::float32> var_41;
        const wp::int32 var_42 = 0;
        wp::float32 var_43;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        wp::float32 var_46;
        const wp::int32 var_47 = 1;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        wp::float32 var_50;
        wp::float32 var_51;
        const wp::int32 var_52 = 2;
        wp::float32 var_53;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::vec_t<3, wp::float32> var_57;
        const wp::int32 var_58 = 2;
        wp::int32* var_59;
        wp::vec_t<3, wp::float32>* var_60;
        wp::int32 var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        const wp::int32 var_66 = 0;
        wp::float32 var_67;
        wp::float32 var_68;
        const wp::int32 var_69 = 1;
        wp::float32 var_70;
        const wp::int32 var_71 = 1;
        wp::float32 var_72;
        wp::float32 var_73;
        const wp::int32 var_74 = 2;
        wp::float32 var_75;
        const wp::int32 var_76 = 2;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        const wp::int32 var_80 = 0;
        wp::float32 var_81;
        const wp::int32 var_82 = 0;
        wp::float32 var_83;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::float32 var_88;
        wp::float32 var_89;
        const wp::int32 var_90 = 2;
        wp::float32 var_91;
        const wp::int32 var_92 = 2;
        wp::float32 var_93;
        wp::float32 var_94;
        wp::vec_t<3, wp::float32> var_95;
        const wp::int32 var_96 = 3;
        wp::int32* var_97;
        wp::vec_t<3, wp::float32>* var_98;
        wp::int32 var_99;
        wp::vec_t<3, wp::float32> var_100;
        wp::vec_t<3, wp::float32> var_101;
        const wp::int32 var_102 = 0;
        wp::float32 var_103;
        const wp::int32 var_104 = 0;
        wp::float32 var_105;
        wp::float32 var_106;
        const wp::int32 var_107 = 1;
        wp::float32 var_108;
        const wp::int32 var_109 = 1;
        wp::float32 var_110;
        wp::float32 var_111;
        const wp::int32 var_112 = 2;
        wp::float32 var_113;
        const wp::int32 var_114 = 2;
        wp::float32 var_115;
        wp::float32 var_116;
        wp::vec_t<3, wp::float32> var_117;
        const wp::int32 var_118 = 0;
        wp::float32 var_119;
        const wp::int32 var_120 = 0;
        wp::float32 var_121;
        wp::float32 var_122;
        const wp::int32 var_123 = 1;
        wp::float32 var_124;
        const wp::int32 var_125 = 1;
        wp::float32 var_126;
        wp::float32 var_127;
        const wp::int32 var_128 = 2;
        wp::float32 var_129;
        const wp::int32 var_130 = 2;
        wp::float32 var_131;
        wp::float32 var_132;
        wp::vec_t<3, wp::float32> var_133;
        const wp::int32 var_134 = 4;
        wp::int32* var_135;
        wp::vec_t<3, wp::float32>* var_136;
        wp::int32 var_137;
        wp::vec_t<3, wp::float32> var_138;
        wp::vec_t<3, wp::float32> var_139;
        const wp::int32 var_140 = 0;
        wp::float32 var_141;
        const wp::int32 var_142 = 0;
        wp::float32 var_143;
        wp::float32 var_144;
        const wp::int32 var_145 = 1;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 2;
        wp::float32 var_151;
        const wp::int32 var_152 = 2;
        wp::float32 var_153;
        wp::float32 var_154;
        wp::vec_t<3, wp::float32> var_155;
        const wp::int32 var_156 = 0;
        wp::float32 var_157;
        const wp::int32 var_158 = 0;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 1;
        wp::float32 var_162;
        const wp::int32 var_163 = 1;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 2;
        wp::float32 var_167;
        const wp::int32 var_168 = 2;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::vec_t<3, wp::float32> var_171;
        const wp::int32 var_172 = 5;
        wp::int32* var_173;
        wp::vec_t<3, wp::float32>* var_174;
        wp::int32 var_175;
        wp::vec_t<3, wp::float32> var_176;
        wp::vec_t<3, wp::float32> var_177;
        const wp::int32 var_178 = 0;
        wp::float32 var_179;
        const wp::int32 var_180 = 0;
        wp::float32 var_181;
        wp::float32 var_182;
        const wp::int32 var_183 = 1;
        wp::float32 var_184;
        const wp::int32 var_185 = 1;
        wp::float32 var_186;
        wp::float32 var_187;
        const wp::int32 var_188 = 2;
        wp::float32 var_189;
        const wp::int32 var_190 = 2;
        wp::float32 var_191;
        wp::float32 var_192;
        wp::vec_t<3, wp::float32> var_193;
        const wp::int32 var_194 = 0;
        wp::float32 var_195;
        const wp::int32 var_196 = 0;
        wp::float32 var_197;
        wp::float32 var_198;
        const wp::int32 var_199 = 1;
        wp::float32 var_200;
        const wp::int32 var_201 = 1;
        wp::float32 var_202;
        wp::float32 var_203;
        const wp::int32 var_204 = 2;
        wp::float32 var_205;
        const wp::int32 var_206 = 2;
        wp::float32 var_207;
        wp::float32 var_208;
        wp::vec_t<3, wp::float32> var_209;
        const wp::int32 var_210 = 6;
        wp::int32* var_211;
        wp::vec_t<3, wp::float32>* var_212;
        wp::int32 var_213;
        wp::vec_t<3, wp::float32> var_214;
        wp::vec_t<3, wp::float32> var_215;
        const wp::int32 var_216 = 0;
        wp::float32 var_217;
        const wp::int32 var_218 = 0;
        wp::float32 var_219;
        wp::float32 var_220;
        const wp::int32 var_221 = 1;
        wp::float32 var_222;
        const wp::int32 var_223 = 1;
        wp::float32 var_224;
        wp::float32 var_225;
        const wp::int32 var_226 = 2;
        wp::float32 var_227;
        const wp::int32 var_228 = 2;
        wp::float32 var_229;
        wp::float32 var_230;
        wp::vec_t<3, wp::float32> var_231;
        const wp::int32 var_232 = 0;
        wp::float32 var_233;
        const wp::int32 var_234 = 0;
        wp::float32 var_235;
        wp::float32 var_236;
        const wp::int32 var_237 = 1;
        wp::float32 var_238;
        const wp::int32 var_239 = 1;
        wp::float32 var_240;
        wp::float32 var_241;
        const wp::int32 var_242 = 2;
        wp::float32 var_243;
        const wp::int32 var_244 = 2;
        wp::float32 var_245;
        wp::float32 var_246;
        wp::vec_t<3, wp::float32> var_247;
        const wp::int32 var_248 = 7;
        wp::int32* var_249;
        wp::vec_t<3, wp::float32>* var_250;
        wp::int32 var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<3, wp::float32> var_253;
        const wp::int32 var_254 = 0;
        wp::float32 var_255;
        const wp::int32 var_256 = 0;
        wp::float32 var_257;
        wp::float32 var_258;
        const wp::int32 var_259 = 1;
        wp::float32 var_260;
        const wp::int32 var_261 = 1;
        wp::float32 var_262;
        wp::float32 var_263;
        const wp::int32 var_264 = 2;
        wp::float32 var_265;
        const wp::int32 var_266 = 2;
        wp::float32 var_267;
        wp::float32 var_268;
        wp::vec_t<3, wp::float32> var_269;
        const wp::int32 var_270 = 0;
        wp::float32 var_271;
        const wp::int32 var_272 = 0;
        wp::float32 var_273;
        wp::float32 var_274;
        const wp::int32 var_275 = 1;
        wp::float32 var_276;
        const wp::int32 var_277 = 1;
        wp::float32 var_278;
        wp::float32 var_279;
        const wp::int32 var_280 = 2;
        wp::float32 var_281;
        const wp::int32 var_282 = 2;
        wp::float32 var_283;
        wp::float32 var_284;
        wp::vec_t<3, wp::float32> var_285;
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
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::int32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::vec_t<3, wp::float32> adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::int32 adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::int32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::float32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::float32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::vec_t<3, wp::float32> adj_41 = {};
        wp::int32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::vec_t<3, wp::float32> adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::vec_t<3, wp::float32> adj_60 = {};
        wp::int32 adj_61 = {};
        wp::vec_t<3, wp::float32> adj_62 = {};
        wp::vec_t<3, wp::float32> adj_63 = {};
        wp::int32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::vec_t<3, wp::float32> adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::int32 adj_90 = {};
        wp::float32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::float32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::vec_t<3, wp::float32> adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::vec_t<3, wp::float32> adj_98 = {};
        wp::int32 adj_99 = {};
        wp::vec_t<3, wp::float32> adj_100 = {};
        wp::vec_t<3, wp::float32> adj_101 = {};
        wp::int32 adj_102 = {};
        wp::float32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::float32 adj_105 = {};
        wp::float32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::float32 adj_108 = {};
        wp::int32 adj_109 = {};
        wp::float32 adj_110 = {};
        wp::float32 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::float32 adj_113 = {};
        wp::int32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::float32 adj_116 = {};
        wp::vec_t<3, wp::float32> adj_117 = {};
        wp::int32 adj_118 = {};
        wp::float32 adj_119 = {};
        wp::int32 adj_120 = {};
        wp::float32 adj_121 = {};
        wp::float32 adj_122 = {};
        wp::int32 adj_123 = {};
        wp::float32 adj_124 = {};
        wp::int32 adj_125 = {};
        wp::float32 adj_126 = {};
        wp::float32 adj_127 = {};
        wp::int32 adj_128 = {};
        wp::float32 adj_129 = {};
        wp::int32 adj_130 = {};
        wp::float32 adj_131 = {};
        wp::float32 adj_132 = {};
        wp::vec_t<3, wp::float32> adj_133 = {};
        wp::int32 adj_134 = {};
        wp::int32 adj_135 = {};
        wp::vec_t<3, wp::float32> adj_136 = {};
        wp::int32 adj_137 = {};
        wp::vec_t<3, wp::float32> adj_138 = {};
        wp::vec_t<3, wp::float32> adj_139 = {};
        wp::int32 adj_140 = {};
        wp::float32 adj_141 = {};
        wp::int32 adj_142 = {};
        wp::float32 adj_143 = {};
        wp::float32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::int32 adj_147 = {};
        wp::float32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::float32 adj_151 = {};
        wp::int32 adj_152 = {};
        wp::float32 adj_153 = {};
        wp::float32 adj_154 = {};
        wp::vec_t<3, wp::float32> adj_155 = {};
        wp::int32 adj_156 = {};
        wp::float32 adj_157 = {};
        wp::int32 adj_158 = {};
        wp::float32 adj_159 = {};
        wp::float32 adj_160 = {};
        wp::int32 adj_161 = {};
        wp::float32 adj_162 = {};
        wp::int32 adj_163 = {};
        wp::float32 adj_164 = {};
        wp::float32 adj_165 = {};
        wp::int32 adj_166 = {};
        wp::float32 adj_167 = {};
        wp::int32 adj_168 = {};
        wp::float32 adj_169 = {};
        wp::float32 adj_170 = {};
        wp::vec_t<3, wp::float32> adj_171 = {};
        wp::int32 adj_172 = {};
        wp::int32 adj_173 = {};
        wp::vec_t<3, wp::float32> adj_174 = {};
        wp::int32 adj_175 = {};
        wp::vec_t<3, wp::float32> adj_176 = {};
        wp::vec_t<3, wp::float32> adj_177 = {};
        wp::int32 adj_178 = {};
        wp::float32 adj_179 = {};
        wp::int32 adj_180 = {};
        wp::float32 adj_181 = {};
        wp::float32 adj_182 = {};
        wp::int32 adj_183 = {};
        wp::float32 adj_184 = {};
        wp::int32 adj_185 = {};
        wp::float32 adj_186 = {};
        wp::float32 adj_187 = {};
        wp::int32 adj_188 = {};
        wp::float32 adj_189 = {};
        wp::int32 adj_190 = {};
        wp::float32 adj_191 = {};
        wp::float32 adj_192 = {};
        wp::vec_t<3, wp::float32> adj_193 = {};
        wp::int32 adj_194 = {};
        wp::float32 adj_195 = {};
        wp::int32 adj_196 = {};
        wp::float32 adj_197 = {};
        wp::float32 adj_198 = {};
        wp::int32 adj_199 = {};
        wp::float32 adj_200 = {};
        wp::int32 adj_201 = {};
        wp::float32 adj_202 = {};
        wp::float32 adj_203 = {};
        wp::int32 adj_204 = {};
        wp::float32 adj_205 = {};
        wp::int32 adj_206 = {};
        wp::float32 adj_207 = {};
        wp::float32 adj_208 = {};
        wp::vec_t<3, wp::float32> adj_209 = {};
        wp::int32 adj_210 = {};
        wp::int32 adj_211 = {};
        wp::vec_t<3, wp::float32> adj_212 = {};
        wp::int32 adj_213 = {};
        wp::vec_t<3, wp::float32> adj_214 = {};
        wp::vec_t<3, wp::float32> adj_215 = {};
        wp::int32 adj_216 = {};
        wp::float32 adj_217 = {};
        wp::int32 adj_218 = {};
        wp::float32 adj_219 = {};
        wp::float32 adj_220 = {};
        wp::int32 adj_221 = {};
        wp::float32 adj_222 = {};
        wp::int32 adj_223 = {};
        wp::float32 adj_224 = {};
        wp::float32 adj_225 = {};
        wp::int32 adj_226 = {};
        wp::float32 adj_227 = {};
        wp::int32 adj_228 = {};
        wp::float32 adj_229 = {};
        wp::float32 adj_230 = {};
        wp::vec_t<3, wp::float32> adj_231 = {};
        wp::int32 adj_232 = {};
        wp::float32 adj_233 = {};
        wp::int32 adj_234 = {};
        wp::float32 adj_235 = {};
        wp::float32 adj_236 = {};
        wp::int32 adj_237 = {};
        wp::float32 adj_238 = {};
        wp::int32 adj_239 = {};
        wp::float32 adj_240 = {};
        wp::float32 adj_241 = {};
        wp::int32 adj_242 = {};
        wp::float32 adj_243 = {};
        wp::int32 adj_244 = {};
        wp::float32 adj_245 = {};
        wp::float32 adj_246 = {};
        wp::vec_t<3, wp::float32> adj_247 = {};
        wp::int32 adj_248 = {};
        wp::int32 adj_249 = {};
        wp::vec_t<3, wp::float32> adj_250 = {};
        wp::int32 adj_251 = {};
        wp::vec_t<3, wp::float32> adj_252 = {};
        wp::vec_t<3, wp::float32> adj_253 = {};
        wp::int32 adj_254 = {};
        wp::float32 adj_255 = {};
        wp::int32 adj_256 = {};
        wp::float32 adj_257 = {};
        wp::float32 adj_258 = {};
        wp::int32 adj_259 = {};
        wp::float32 adj_260 = {};
        wp::int32 adj_261 = {};
        wp::float32 adj_262 = {};
        wp::float32 adj_263 = {};
        wp::int32 adj_264 = {};
        wp::float32 adj_265 = {};
        wp::int32 adj_266 = {};
        wp::float32 adj_267 = {};
        wp::float32 adj_268 = {};
        wp::vec_t<3, wp::float32> adj_269 = {};
        wp::int32 adj_270 = {};
        wp::float32 adj_271 = {};
        wp::int32 adj_272 = {};
        wp::float32 adj_273 = {};
        wp::float32 adj_274 = {};
        wp::int32 adj_275 = {};
        wp::float32 adj_276 = {};
        wp::int32 adj_277 = {};
        wp::float32 adj_278 = {};
        wp::float32 adj_279 = {};
        wp::int32 adj_280 = {};
        wp::float32 adj_281 = {};
        wp::int32 adj_282 = {};
        wp::float32 adj_283 = {};
        wp::float32 adj_284 = {};
        wp::vec_t<3, wp::float32> adj_285 = {};
        //---------
        // forward
        // def build_dirty_cell_aabbs_kernel(                                                     <L 311>
        // i = wp.tid()                                                                           <L 321>
        var_0 = builtin_tid1d();
        // if i >= dirty_count[0]:                                                                <L 322>
        var_2 = wp::address(var_dirty_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 323>
            goto label0;
        }
        // c = dirty_cell_ids[i]                                                                  <L 324>
        var_5 = wp::address(var_dirty_cell_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if c < 0 or c >= num_cells:                                                            <L 325>
        var_10 = (var_6 < var_9);
        var_8 = var_10;
        if (!var_8) {
            var_11 = (var_6 >= var_num_cells);
            var_8 = var_8 || var_11;
        }
        if (var_8) {
            // return                                                                             <L 326>
            goto label1;
        }
        // p = particle_q[cell_nodes[c, 0]]                                                       <L 328>
        var_13 = wp::address(var_cell_nodes, var_6, var_12);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_particle_q, var_15);
        var_17 = wp::load(var_14);
        var_16 = wp::copy(var_17);
        // lo = p                                                                                 <L 329>
        var_18 = wp::copy(var_16);
        // hi = p                                                                                 <L 330>
        var_19 = wp::copy(var_16);
        // for k in range(1, 8):                                                                  <L 331>
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_21 = wp::address(var_cell_nodes, var_6, var_20);
        var_23 = wp::load(var_21);
        var_22 = wp::address(var_particle_q, var_23);
        var_25 = wp::load(var_22);
        var_24 = wp::copy(var_25);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_27 = wp::extract(var_18, var_26);
        var_29 = wp::extract(var_24, var_28);
        var_30 = wp::min(var_27, var_29);
        var_32 = wp::extract(var_18, var_31);
        var_34 = wp::extract(var_24, var_33);
        var_35 = wp::min(var_32, var_34);
        var_37 = wp::extract(var_18, var_36);
        var_39 = wp::extract(var_24, var_38);
        var_40 = wp::min(var_37, var_39);
        var_41 = wp::vec_t<3, wp::float32>(var_30, var_35, var_40);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_43 = wp::extract(var_19, var_42);
        var_45 = wp::extract(var_24, var_44);
        var_46 = wp::max(var_43, var_45);
        var_48 = wp::extract(var_19, var_47);
        var_50 = wp::extract(var_24, var_49);
        var_51 = wp::max(var_48, var_50);
        var_53 = wp::extract(var_19, var_52);
        var_55 = wp::extract(var_24, var_54);
        var_56 = wp::max(var_53, var_55);
        var_57 = wp::vec_t<3, wp::float32>(var_46, var_51, var_56);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_59 = wp::address(var_cell_nodes, var_6, var_58);
        var_61 = wp::load(var_59);
        var_60 = wp::address(var_particle_q, var_61);
        var_63 = wp::load(var_60);
        var_62 = wp::copy(var_63);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_65 = wp::extract(var_41, var_64);
        var_67 = wp::extract(var_62, var_66);
        var_68 = wp::min(var_65, var_67);
        var_70 = wp::extract(var_41, var_69);
        var_72 = wp::extract(var_62, var_71);
        var_73 = wp::min(var_70, var_72);
        var_75 = wp::extract(var_41, var_74);
        var_77 = wp::extract(var_62, var_76);
        var_78 = wp::min(var_75, var_77);
        var_79 = wp::vec_t<3, wp::float32>(var_68, var_73, var_78);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_81 = wp::extract(var_57, var_80);
        var_83 = wp::extract(var_62, var_82);
        var_84 = wp::max(var_81, var_83);
        var_86 = wp::extract(var_57, var_85);
        var_88 = wp::extract(var_62, var_87);
        var_89 = wp::max(var_86, var_88);
        var_91 = wp::extract(var_57, var_90);
        var_93 = wp::extract(var_62, var_92);
        var_94 = wp::max(var_91, var_93);
        var_95 = wp::vec_t<3, wp::float32>(var_84, var_89, var_94);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_97 = wp::address(var_cell_nodes, var_6, var_96);
        var_99 = wp::load(var_97);
        var_98 = wp::address(var_particle_q, var_99);
        var_101 = wp::load(var_98);
        var_100 = wp::copy(var_101);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_103 = wp::extract(var_79, var_102);
        var_105 = wp::extract(var_100, var_104);
        var_106 = wp::min(var_103, var_105);
        var_108 = wp::extract(var_79, var_107);
        var_110 = wp::extract(var_100, var_109);
        var_111 = wp::min(var_108, var_110);
        var_113 = wp::extract(var_79, var_112);
        var_115 = wp::extract(var_100, var_114);
        var_116 = wp::min(var_113, var_115);
        var_117 = wp::vec_t<3, wp::float32>(var_106, var_111, var_116);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_119 = wp::extract(var_95, var_118);
        var_121 = wp::extract(var_100, var_120);
        var_122 = wp::max(var_119, var_121);
        var_124 = wp::extract(var_95, var_123);
        var_126 = wp::extract(var_100, var_125);
        var_127 = wp::max(var_124, var_126);
        var_129 = wp::extract(var_95, var_128);
        var_131 = wp::extract(var_100, var_130);
        var_132 = wp::max(var_129, var_131);
        var_133 = wp::vec_t<3, wp::float32>(var_122, var_127, var_132);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_135 = wp::address(var_cell_nodes, var_6, var_134);
        var_137 = wp::load(var_135);
        var_136 = wp::address(var_particle_q, var_137);
        var_139 = wp::load(var_136);
        var_138 = wp::copy(var_139);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_141 = wp::extract(var_117, var_140);
        var_143 = wp::extract(var_138, var_142);
        var_144 = wp::min(var_141, var_143);
        var_146 = wp::extract(var_117, var_145);
        var_148 = wp::extract(var_138, var_147);
        var_149 = wp::min(var_146, var_148);
        var_151 = wp::extract(var_117, var_150);
        var_153 = wp::extract(var_138, var_152);
        var_154 = wp::min(var_151, var_153);
        var_155 = wp::vec_t<3, wp::float32>(var_144, var_149, var_154);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_157 = wp::extract(var_133, var_156);
        var_159 = wp::extract(var_138, var_158);
        var_160 = wp::max(var_157, var_159);
        var_162 = wp::extract(var_133, var_161);
        var_164 = wp::extract(var_138, var_163);
        var_165 = wp::max(var_162, var_164);
        var_167 = wp::extract(var_133, var_166);
        var_169 = wp::extract(var_138, var_168);
        var_170 = wp::max(var_167, var_169);
        var_171 = wp::vec_t<3, wp::float32>(var_160, var_165, var_170);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_173 = wp::address(var_cell_nodes, var_6, var_172);
        var_175 = wp::load(var_173);
        var_174 = wp::address(var_particle_q, var_175);
        var_177 = wp::load(var_174);
        var_176 = wp::copy(var_177);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_179 = wp::extract(var_155, var_178);
        var_181 = wp::extract(var_176, var_180);
        var_182 = wp::min(var_179, var_181);
        var_184 = wp::extract(var_155, var_183);
        var_186 = wp::extract(var_176, var_185);
        var_187 = wp::min(var_184, var_186);
        var_189 = wp::extract(var_155, var_188);
        var_191 = wp::extract(var_176, var_190);
        var_192 = wp::min(var_189, var_191);
        var_193 = wp::vec_t<3, wp::float32>(var_182, var_187, var_192);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_195 = wp::extract(var_171, var_194);
        var_197 = wp::extract(var_176, var_196);
        var_198 = wp::max(var_195, var_197);
        var_200 = wp::extract(var_171, var_199);
        var_202 = wp::extract(var_176, var_201);
        var_203 = wp::max(var_200, var_202);
        var_205 = wp::extract(var_171, var_204);
        var_207 = wp::extract(var_176, var_206);
        var_208 = wp::max(var_205, var_207);
        var_209 = wp::vec_t<3, wp::float32>(var_198, var_203, var_208);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_211 = wp::address(var_cell_nodes, var_6, var_210);
        var_213 = wp::load(var_211);
        var_212 = wp::address(var_particle_q, var_213);
        var_215 = wp::load(var_212);
        var_214 = wp::copy(var_215);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_217 = wp::extract(var_193, var_216);
        var_219 = wp::extract(var_214, var_218);
        var_220 = wp::min(var_217, var_219);
        var_222 = wp::extract(var_193, var_221);
        var_224 = wp::extract(var_214, var_223);
        var_225 = wp::min(var_222, var_224);
        var_227 = wp::extract(var_193, var_226);
        var_229 = wp::extract(var_214, var_228);
        var_230 = wp::min(var_227, var_229);
        var_231 = wp::vec_t<3, wp::float32>(var_220, var_225, var_230);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_233 = wp::extract(var_209, var_232);
        var_235 = wp::extract(var_214, var_234);
        var_236 = wp::max(var_233, var_235);
        var_238 = wp::extract(var_209, var_237);
        var_240 = wp::extract(var_214, var_239);
        var_241 = wp::max(var_238, var_240);
        var_243 = wp::extract(var_209, var_242);
        var_245 = wp::extract(var_214, var_244);
        var_246 = wp::max(var_243, var_245);
        var_247 = wp::vec_t<3, wp::float32>(var_236, var_241, var_246);
        // p = particle_q[cell_nodes[c, k]]                                                       <L 332>
        var_249 = wp::address(var_cell_nodes, var_6, var_248);
        var_251 = wp::load(var_249);
        var_250 = wp::address(var_particle_q, var_251);
        var_253 = wp::load(var_250);
        var_252 = wp::copy(var_253);
        // lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))            <L 333>
        var_255 = wp::extract(var_231, var_254);
        var_257 = wp::extract(var_252, var_256);
        var_258 = wp::min(var_255, var_257);
        var_260 = wp::extract(var_231, var_259);
        var_262 = wp::extract(var_252, var_261);
        var_263 = wp::min(var_260, var_262);
        var_265 = wp::extract(var_231, var_264);
        var_267 = wp::extract(var_252, var_266);
        var_268 = wp::min(var_265, var_267);
        var_269 = wp::vec_t<3, wp::float32>(var_258, var_263, var_268);
        // hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))            <L 334>
        var_271 = wp::extract(var_247, var_270);
        var_273 = wp::extract(var_252, var_272);
        var_274 = wp::max(var_271, var_273);
        var_276 = wp::extract(var_247, var_275);
        var_278 = wp::extract(var_252, var_277);
        var_279 = wp::max(var_276, var_278);
        var_281 = wp::extract(var_247, var_280);
        var_283 = wp::extract(var_252, var_282);
        var_284 = wp::max(var_281, var_283);
        var_285 = wp::vec_t<3, wp::float32>(var_274, var_279, var_284);
        // aabb_min[c] = lo                                                                       <L 335>
        // wp::array_store(var_aabb_min, var_6, var_269);
        // aabb_max[c] = hi                                                                       <L 336>
        // wp::array_store(var_aabb_max, var_6, var_285);
        //---------
        // reverse
        wp::adj_array_store(var_aabb_max, var_6, var_285, adj_aabb_max, adj_6, adj_285);
        // adj: aabb_max[c] = hi                                                                  <L 336>
        wp::adj_array_store(var_aabb_min, var_6, var_269, adj_aabb_min, adj_6, adj_269);
        // adj: aabb_min[c] = lo                                                                  <L 335>
        wp::adj_vec_t(var_274, var_279, var_284, adj_274, adj_279, adj_284, adj_285);
        wp::adj_max(var_281, var_283, adj_281, adj_283, adj_284);
        wp::adj_extract(var_252, var_282, adj_252, adj_282, adj_283);
        wp::adj_extract(var_247, var_280, adj_247, adj_280, adj_281);
        wp::adj_max(var_276, var_278, adj_276, adj_278, adj_279);
        wp::adj_extract(var_252, var_277, adj_252, adj_277, adj_278);
        wp::adj_extract(var_247, var_275, adj_247, adj_275, adj_276);
        wp::adj_max(var_271, var_273, adj_271, adj_273, adj_274);
        wp::adj_extract(var_252, var_272, adj_252, adj_272, adj_273);
        wp::adj_extract(var_247, var_270, adj_247, adj_270, adj_271);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_258, var_263, var_268, adj_258, adj_263, adj_268, adj_269);
        wp::adj_min(var_265, var_267, adj_265, adj_267, adj_268);
        wp::adj_extract(var_252, var_266, adj_252, adj_266, adj_267);
        wp::adj_extract(var_231, var_264, adj_231, adj_264, adj_265);
        wp::adj_min(var_260, var_262, adj_260, adj_262, adj_263);
        wp::adj_extract(var_252, var_261, adj_252, adj_261, adj_262);
        wp::adj_extract(var_231, var_259, adj_231, adj_259, adj_260);
        wp::adj_min(var_255, var_257, adj_255, adj_257, adj_258);
        wp::adj_extract(var_252, var_256, adj_252, adj_256, adj_257);
        wp::adj_extract(var_231, var_254, adj_231, adj_254, adj_255);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_253, adj_250, adj_252);
        wp::adj_address(var_particle_q, var_251, adj_particle_q, adj_249, adj_250);
        wp::adj_address(var_cell_nodes, var_6, var_248, adj_cell_nodes, adj_6, adj_248, adj_249);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        wp::adj_vec_t(var_236, var_241, var_246, adj_236, adj_241, adj_246, adj_247);
        wp::adj_max(var_243, var_245, adj_243, adj_245, adj_246);
        wp::adj_extract(var_214, var_244, adj_214, adj_244, adj_245);
        wp::adj_extract(var_209, var_242, adj_209, adj_242, adj_243);
        wp::adj_max(var_238, var_240, adj_238, adj_240, adj_241);
        wp::adj_extract(var_214, var_239, adj_214, adj_239, adj_240);
        wp::adj_extract(var_209, var_237, adj_209, adj_237, adj_238);
        wp::adj_max(var_233, var_235, adj_233, adj_235, adj_236);
        wp::adj_extract(var_214, var_234, adj_214, adj_234, adj_235);
        wp::adj_extract(var_209, var_232, adj_209, adj_232, adj_233);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_220, var_225, var_230, adj_220, adj_225, adj_230, adj_231);
        wp::adj_min(var_227, var_229, adj_227, adj_229, adj_230);
        wp::adj_extract(var_214, var_228, adj_214, adj_228, adj_229);
        wp::adj_extract(var_193, var_226, adj_193, adj_226, adj_227);
        wp::adj_min(var_222, var_224, adj_222, adj_224, adj_225);
        wp::adj_extract(var_214, var_223, adj_214, adj_223, adj_224);
        wp::adj_extract(var_193, var_221, adj_193, adj_221, adj_222);
        wp::adj_min(var_217, var_219, adj_217, adj_219, adj_220);
        wp::adj_extract(var_214, var_218, adj_214, adj_218, adj_219);
        wp::adj_extract(var_193, var_216, adj_193, adj_216, adj_217);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_215, adj_212, adj_214);
        wp::adj_address(var_particle_q, var_213, adj_particle_q, adj_211, adj_212);
        wp::adj_address(var_cell_nodes, var_6, var_210, adj_cell_nodes, adj_6, adj_210, adj_211);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        wp::adj_vec_t(var_198, var_203, var_208, adj_198, adj_203, adj_208, adj_209);
        wp::adj_max(var_205, var_207, adj_205, adj_207, adj_208);
        wp::adj_extract(var_176, var_206, adj_176, adj_206, adj_207);
        wp::adj_extract(var_171, var_204, adj_171, adj_204, adj_205);
        wp::adj_max(var_200, var_202, adj_200, adj_202, adj_203);
        wp::adj_extract(var_176, var_201, adj_176, adj_201, adj_202);
        wp::adj_extract(var_171, var_199, adj_171, adj_199, adj_200);
        wp::adj_max(var_195, var_197, adj_195, adj_197, adj_198);
        wp::adj_extract(var_176, var_196, adj_176, adj_196, adj_197);
        wp::adj_extract(var_171, var_194, adj_171, adj_194, adj_195);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_182, var_187, var_192, adj_182, adj_187, adj_192, adj_193);
        wp::adj_min(var_189, var_191, adj_189, adj_191, adj_192);
        wp::adj_extract(var_176, var_190, adj_176, adj_190, adj_191);
        wp::adj_extract(var_155, var_188, adj_155, adj_188, adj_189);
        wp::adj_min(var_184, var_186, adj_184, adj_186, adj_187);
        wp::adj_extract(var_176, var_185, adj_176, adj_185, adj_186);
        wp::adj_extract(var_155, var_183, adj_155, adj_183, adj_184);
        wp::adj_min(var_179, var_181, adj_179, adj_181, adj_182);
        wp::adj_extract(var_176, var_180, adj_176, adj_180, adj_181);
        wp::adj_extract(var_155, var_178, adj_155, adj_178, adj_179);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_177, adj_174, adj_176);
        wp::adj_address(var_particle_q, var_175, adj_particle_q, adj_173, adj_174);
        wp::adj_address(var_cell_nodes, var_6, var_172, adj_cell_nodes, adj_6, adj_172, adj_173);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        wp::adj_vec_t(var_160, var_165, var_170, adj_160, adj_165, adj_170, adj_171);
        wp::adj_max(var_167, var_169, adj_167, adj_169, adj_170);
        wp::adj_extract(var_138, var_168, adj_138, adj_168, adj_169);
        wp::adj_extract(var_133, var_166, adj_133, adj_166, adj_167);
        wp::adj_max(var_162, var_164, adj_162, adj_164, adj_165);
        wp::adj_extract(var_138, var_163, adj_138, adj_163, adj_164);
        wp::adj_extract(var_133, var_161, adj_133, adj_161, adj_162);
        wp::adj_max(var_157, var_159, adj_157, adj_159, adj_160);
        wp::adj_extract(var_138, var_158, adj_138, adj_158, adj_159);
        wp::adj_extract(var_133, var_156, adj_133, adj_156, adj_157);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_144, var_149, var_154, adj_144, adj_149, adj_154, adj_155);
        wp::adj_min(var_151, var_153, adj_151, adj_153, adj_154);
        wp::adj_extract(var_138, var_152, adj_138, adj_152, adj_153);
        wp::adj_extract(var_117, var_150, adj_117, adj_150, adj_151);
        wp::adj_min(var_146, var_148, adj_146, adj_148, adj_149);
        wp::adj_extract(var_138, var_147, adj_138, adj_147, adj_148);
        wp::adj_extract(var_117, var_145, adj_117, adj_145, adj_146);
        wp::adj_min(var_141, var_143, adj_141, adj_143, adj_144);
        wp::adj_extract(var_138, var_142, adj_138, adj_142, adj_143);
        wp::adj_extract(var_117, var_140, adj_117, adj_140, adj_141);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_139, adj_136, adj_138);
        wp::adj_address(var_particle_q, var_137, adj_particle_q, adj_135, adj_136);
        wp::adj_address(var_cell_nodes, var_6, var_134, adj_cell_nodes, adj_6, adj_134, adj_135);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        wp::adj_vec_t(var_122, var_127, var_132, adj_122, adj_127, adj_132, adj_133);
        wp::adj_max(var_129, var_131, adj_129, adj_131, adj_132);
        wp::adj_extract(var_100, var_130, adj_100, adj_130, adj_131);
        wp::adj_extract(var_95, var_128, adj_95, adj_128, adj_129);
        wp::adj_max(var_124, var_126, adj_124, adj_126, adj_127);
        wp::adj_extract(var_100, var_125, adj_100, adj_125, adj_126);
        wp::adj_extract(var_95, var_123, adj_95, adj_123, adj_124);
        wp::adj_max(var_119, var_121, adj_119, adj_121, adj_122);
        wp::adj_extract(var_100, var_120, adj_100, adj_120, adj_121);
        wp::adj_extract(var_95, var_118, adj_95, adj_118, adj_119);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_106, var_111, var_116, adj_106, adj_111, adj_116, adj_117);
        wp::adj_min(var_113, var_115, adj_113, adj_115, adj_116);
        wp::adj_extract(var_100, var_114, adj_100, adj_114, adj_115);
        wp::adj_extract(var_79, var_112, adj_79, adj_112, adj_113);
        wp::adj_min(var_108, var_110, adj_108, adj_110, adj_111);
        wp::adj_extract(var_100, var_109, adj_100, adj_109, adj_110);
        wp::adj_extract(var_79, var_107, adj_79, adj_107, adj_108);
        wp::adj_min(var_103, var_105, adj_103, adj_105, adj_106);
        wp::adj_extract(var_100, var_104, adj_100, adj_104, adj_105);
        wp::adj_extract(var_79, var_102, adj_79, adj_102, adj_103);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_101, adj_98, adj_100);
        wp::adj_address(var_particle_q, var_99, adj_particle_q, adj_97, adj_98);
        wp::adj_address(var_cell_nodes, var_6, var_96, adj_cell_nodes, adj_6, adj_96, adj_97);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        wp::adj_vec_t(var_84, var_89, var_94, adj_84, adj_89, adj_94, adj_95);
        wp::adj_max(var_91, var_93, adj_91, adj_93, adj_94);
        wp::adj_extract(var_62, var_92, adj_62, adj_92, adj_93);
        wp::adj_extract(var_57, var_90, adj_57, adj_90, adj_91);
        wp::adj_max(var_86, var_88, adj_86, adj_88, adj_89);
        wp::adj_extract(var_62, var_87, adj_62, adj_87, adj_88);
        wp::adj_extract(var_57, var_85, adj_57, adj_85, adj_86);
        wp::adj_max(var_81, var_83, adj_81, adj_83, adj_84);
        wp::adj_extract(var_62, var_82, adj_62, adj_82, adj_83);
        wp::adj_extract(var_57, var_80, adj_57, adj_80, adj_81);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_68, var_73, var_78, adj_68, adj_73, adj_78, adj_79);
        wp::adj_min(var_75, var_77, adj_75, adj_77, adj_78);
        wp::adj_extract(var_62, var_76, adj_62, adj_76, adj_77);
        wp::adj_extract(var_41, var_74, adj_41, adj_74, adj_75);
        wp::adj_min(var_70, var_72, adj_70, adj_72, adj_73);
        wp::adj_extract(var_62, var_71, adj_62, adj_71, adj_72);
        wp::adj_extract(var_41, var_69, adj_41, adj_69, adj_70);
        wp::adj_min(var_65, var_67, adj_65, adj_67, adj_68);
        wp::adj_extract(var_62, var_66, adj_62, adj_66, adj_67);
        wp::adj_extract(var_41, var_64, adj_41, adj_64, adj_65);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_63, adj_60, adj_62);
        wp::adj_address(var_particle_q, var_61, adj_particle_q, adj_59, adj_60);
        wp::adj_address(var_cell_nodes, var_6, var_58, adj_cell_nodes, adj_6, adj_58, adj_59);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        wp::adj_vec_t(var_46, var_51, var_56, adj_46, adj_51, adj_56, adj_57);
        wp::adj_max(var_53, var_55, adj_53, adj_55, adj_56);
        wp::adj_extract(var_24, var_54, adj_24, adj_54, adj_55);
        wp::adj_extract(var_19, var_52, adj_19, adj_52, adj_53);
        wp::adj_max(var_48, var_50, adj_48, adj_50, adj_51);
        wp::adj_extract(var_24, var_49, adj_24, adj_49, adj_50);
        wp::adj_extract(var_19, var_47, adj_19, adj_47, adj_48);
        wp::adj_max(var_43, var_45, adj_43, adj_45, adj_46);
        wp::adj_extract(var_24, var_44, adj_24, adj_44, adj_45);
        wp::adj_extract(var_19, var_42, adj_19, adj_42, adj_43);
        // adj: hi = wp.vec3(wp.max(hi[0], p[0]), wp.max(hi[1], p[1]), wp.max(hi[2], p[2]))       <L 334>
        wp::adj_vec_t(var_30, var_35, var_40, adj_30, adj_35, adj_40, adj_41);
        wp::adj_min(var_37, var_39, adj_37, adj_39, adj_40);
        wp::adj_extract(var_24, var_38, adj_24, adj_38, adj_39);
        wp::adj_extract(var_18, var_36, adj_18, adj_36, adj_37);
        wp::adj_min(var_32, var_34, adj_32, adj_34, adj_35);
        wp::adj_extract(var_24, var_33, adj_24, adj_33, adj_34);
        wp::adj_extract(var_18, var_31, adj_18, adj_31, adj_32);
        wp::adj_min(var_27, var_29, adj_27, adj_29, adj_30);
        wp::adj_extract(var_24, var_28, adj_24, adj_28, adj_29);
        wp::adj_extract(var_18, var_26, adj_18, adj_26, adj_27);
        // adj: lo = wp.vec3(wp.min(lo[0], p[0]), wp.min(lo[1], p[1]), wp.min(lo[2], p[2]))       <L 333>
        wp::adj_copy(var_25, adj_22, adj_24);
        wp::adj_address(var_particle_q, var_23, adj_particle_q, adj_21, adj_22);
        wp::adj_address(var_cell_nodes, var_6, var_20, adj_cell_nodes, adj_6, adj_20, adj_21);
        // adj: p = particle_q[cell_nodes[c, k]]                                                  <L 332>
        // adj: for k in range(1, 8):                                                             <L 331>
        wp::adj_copy(var_16, adj_16, adj_19);
        // adj: hi = p                                                                            <L 330>
        wp::adj_copy(var_16, adj_16, adj_18);
        // adj: lo = p                                                                            <L 329>
        wp::adj_copy(var_17, adj_14, adj_16);
        wp::adj_address(var_particle_q, var_15, adj_particle_q, adj_13, adj_14);
        wp::adj_address(var_cell_nodes, var_6, var_12, adj_cell_nodes, adj_6, adj_12, adj_13);
        // adj: p = particle_q[cell_nodes[c, 0]]                                                  <L 328>
        if (var_8) {
            label1:;
            // adj: return                                                                        <L 326>
        }
        if (!var_8) {
        }
        // adj: if c < 0 or c >= num_cells:                                                       <L 325>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_dirty_cell_ids, var_0, adj_dirty_cell_ids, adj_0, adj_5);
        // adj: c = dirty_cell_ids[i]                                                             <L 324>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 323>
        }
        wp::adj_address(var_dirty_count, var_1, adj_dirty_count, adj_1, adj_2);
        // adj: if i >= dirty_count[0]:                                                           <L 322>
        // adj: i = wp.tid()                                                                      <L 321>
        // adj: def build_dirty_cell_aabbs_kernel(                                                <L 311>
        continue;
    }
}



extern "C" __global__ void fill_hover_edges_kernel_b656a990_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::int32 var_hover_cell,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_edge_pairs,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::int32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::int32* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32> var_16;
        //---------
        // forward
        // def fill_hover_edges_kernel(                                                           <L 540>
        // i = wp.tid()                                                                           <L 549>
        var_0 = builtin_tid1d();
        // a = edge_pairs[i, 0]                                                                   <L 550>
        var_2 = wp::address(var_edge_pairs, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // b = edge_pairs[i, 1]                                                                   <L 551>
        var_6 = wp::address(var_edge_pairs, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // starts[i] = particle_q[cell_nodes[hover_cell, a]]                                      <L 552>
        var_9 = wp::address(var_cell_nodes, var_hover_cell, var_3);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_particle_q, var_11);
        var_12 = wp::load(var_10);
        wp::array_store(var_starts, var_0, var_12);
        // ends[i] = particle_q[cell_nodes[hover_cell, b]]                                        <L 553>
        var_13 = wp::address(var_cell_nodes, var_hover_cell, var_7);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_particle_q, var_15);
        var_16 = wp::load(var_14);
        wp::array_store(var_ends, var_0, var_16);
    }
}



extern "C" __global__ void fill_hover_edges_kernel_b656a990_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::int32 var_hover_cell,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_edge_pairs,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::vec_t<3, wp::float32>> var_starts,
    wp::array_t<wp::vec_t<3, wp::float32>> var_ends,
    wp::int32 adj_hover_cell,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_edge_pairs,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
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
        const wp::int32 var_1 = 0;
        wp::int32* var_2;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::vec_t<3, wp::float32>* var_10;
        wp::int32 var_11;
        wp::vec_t<3, wp::float32> var_12;
        wp::int32* var_13;
        wp::vec_t<3, wp::float32>* var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32> var_16;
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
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::int32 adj_11 = {};
        wp::vec_t<3, wp::float32> adj_12 = {};
        wp::int32 adj_13 = {};
        wp::vec_t<3, wp::float32> adj_14 = {};
        wp::int32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        //---------
        // forward
        // def fill_hover_edges_kernel(                                                           <L 540>
        // i = wp.tid()                                                                           <L 549>
        var_0 = builtin_tid1d();
        // a = edge_pairs[i, 0]                                                                   <L 550>
        var_2 = wp::address(var_edge_pairs, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // b = edge_pairs[i, 1]                                                                   <L 551>
        var_6 = wp::address(var_edge_pairs, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // starts[i] = particle_q[cell_nodes[hover_cell, a]]                                      <L 552>
        var_9 = wp::address(var_cell_nodes, var_hover_cell, var_3);
        var_11 = wp::load(var_9);
        var_10 = wp::address(var_particle_q, var_11);
        var_12 = wp::load(var_10);
        // wp::array_store(var_starts, var_0, var_12);
        // ends[i] = particle_q[cell_nodes[hover_cell, b]]                                        <L 553>
        var_13 = wp::address(var_cell_nodes, var_hover_cell, var_7);
        var_15 = wp::load(var_13);
        var_14 = wp::address(var_particle_q, var_15);
        var_16 = wp::load(var_14);
        // wp::array_store(var_ends, var_0, var_16);
        //---------
        // reverse
        wp::adj_array_store(var_ends, var_0, var_16, adj_ends, adj_0, adj_14);
        wp::adj_address(var_particle_q, var_15, adj_particle_q, adj_13, adj_14);
        wp::adj_address(var_cell_nodes, var_hover_cell, var_7, adj_cell_nodes, adj_hover_cell, adj_7, adj_13);
        // adj: ends[i] = particle_q[cell_nodes[hover_cell, b]]                                   <L 553>
        wp::adj_array_store(var_starts, var_0, var_12, adj_starts, adj_0, adj_10);
        wp::adj_address(var_particle_q, var_11, adj_particle_q, adj_9, adj_10);
        wp::adj_address(var_cell_nodes, var_hover_cell, var_3, adj_cell_nodes, adj_hover_cell, adj_3, adj_9);
        // adj: starts[i] = particle_q[cell_nodes[hover_cell, a]]                                 <L 552>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_edge_pairs, var_0, var_5, adj_edge_pairs, adj_0, adj_5, adj_6);
        // adj: b = edge_pairs[i, 1]                                                              <L 551>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_edge_pairs, var_0, var_1, adj_edge_pairs, adj_0, adj_1, adj_2);
        // adj: a = edge_pairs[i, 0]                                                              <L 550>
        // adj: i = wp.tid()                                                                      <L 549>
        // adj: def fill_hover_edges_kernel(                                                      <L 540>
        continue;
    }
}



extern "C" __global__ void update_cell_render_state_and_aabbs_kernel_563a3e07_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::float32 var_voxel_size,
    wp::int32 var_compute_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cell_center_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell_orientation,
    wp::array_t<wp::int32> var_cell_render_flags,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max)
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
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 4;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 5;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 6;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 7;
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
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        const wp::int32 var_57 = 0;
        wp::float32 var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::int32 var_62 = 0;
        wp::float32 var_63;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 0;
        wp::float32 var_69;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        const wp::int32 var_75 = 0;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::int32 var_80 = 1;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        wp::float32 var_83;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        const wp::int32 var_93 = 1;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::int32 var_96 = 1;
        wp::float32 var_97;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        const wp::int32 var_103 = 2;
        wp::float32 var_104;
        const wp::int32 var_105 = 2;
        wp::float32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 2;
        wp::float32 var_109;
        const wp::int32 var_110 = 2;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        const wp::int32 var_114 = 2;
        wp::float32 var_115;
        const wp::int32 var_116 = 2;
        wp::float32 var_117;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        wp::float32 var_120;
        const wp::int32 var_121 = 2;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::vec_t<3, wp::float32> var_126;
        const wp::int32 var_127 = 0;
        wp::float32 var_128;
        const wp::int32 var_129 = 0;
        wp::float32 var_130;
        wp::float32 var_131;
        const wp::int32 var_132 = 0;
        wp::float32 var_133;
        const wp::int32 var_134 = 0;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        wp::float32 var_139;
        const wp::int32 var_140 = 0;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 1;
        wp::float32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 1;
        wp::float32 var_156;
        const wp::int32 var_157 = 1;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 1;
        wp::float32 var_162;
        const wp::int32 var_163 = 1;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 1;
        wp::float32 var_167;
        const wp::int32 var_168 = 1;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        const wp::int32 var_173 = 2;
        wp::float32 var_174;
        const wp::int32 var_175 = 2;
        wp::float32 var_176;
        wp::float32 var_177;
        const wp::int32 var_178 = 2;
        wp::float32 var_179;
        const wp::int32 var_180 = 2;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        const wp::int32 var_184 = 2;
        wp::float32 var_185;
        const wp::int32 var_186 = 2;
        wp::float32 var_187;
        wp::float32 var_188;
        const wp::int32 var_189 = 2;
        wp::float32 var_190;
        const wp::int32 var_191 = 2;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::vec_t<3, wp::float32> var_196;
        wp::vec_t<3, wp::float32> var_197;
        wp::vec_t<3, wp::float32> var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::vec_t<3, wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        wp::vec_t<3, wp::float32> var_203;
        const wp::float32 var_204 = 1.0;
        const wp::float32 var_205 = 8.0;
        wp::float32 var_206;
        wp::vec_t<3, wp::float32> var_207;
        wp::vec_t<3, wp::float32> var_208;
        wp::vec_t<3, wp::float32> var_209;
        wp::vec_t<3, wp::float32> var_210;
        wp::vec_t<3, wp::float32> var_211;
        wp::vec_t<3, wp::float32> var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::vec_t<3, wp::float32> var_214;
        const wp::float32 var_215 = 0.25;
        wp::vec_t<3, wp::float32> var_216;
        wp::vec_t<3, wp::float32> var_217;
        wp::vec_t<3, wp::float32> var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::vec_t<3, wp::float32> var_220;
        wp::vec_t<3, wp::float32> var_221;
        wp::vec_t<3, wp::float32> var_222;
        wp::vec_t<3, wp::float32> var_223;
        const wp::float32 var_224 = 0.25;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        const wp::float32 var_233 = 0.25;
        wp::vec_t<3, wp::float32> var_234;
        wp::mat_t<3, 3, wp::float32> var_235;
        wp::int32* var_236;
        const wp::int32 var_237 = 0;
        bool var_238;
        wp::int32 var_239;
        const wp::int32 var_240 = 1;
        const wp::int32 var_241 = 1;
        wp::int32 var_242;
        const wp::int32 var_243 = 0;
        bool var_244;
        wp::float32 var_245;
        const wp::int32 var_246 = 0;
        wp::int32 var_247;
        const wp::int32 var_248 = 0;
        bool var_249;
        const wp::float32 var_250 = 0.0;
        //---------
        // forward
        // def update_cell_render_state_and_aabbs_kernel(                                         <L 198>
        // c = wp.tid()                                                                           <L 212>
        var_0 = builtin_tid1d();
        // n0 = cell_nodes[c, 0]                                                                  <L 213>
        var_2 = wp::address(var_cell_nodes, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // n1 = cell_nodes[c, 1]                                                                  <L 214>
        var_6 = wp::address(var_cell_nodes, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // n2 = cell_nodes[c, 2]                                                                  <L 215>
        var_10 = wp::address(var_cell_nodes, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // n3 = cell_nodes[c, 3]                                                                  <L 216>
        var_14 = wp::address(var_cell_nodes, var_0, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // n4 = cell_nodes[c, 4]                                                                  <L 217>
        var_18 = wp::address(var_cell_nodes, var_0, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // n5 = cell_nodes[c, 5]                                                                  <L 218>
        var_22 = wp::address(var_cell_nodes, var_0, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // n6 = cell_nodes[c, 6]                                                                  <L 219>
        var_26 = wp::address(var_cell_nodes, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // n7 = cell_nodes[c, 7]                                                                  <L 220>
        var_30 = wp::address(var_cell_nodes, var_0, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // p0 = particle_q[n0]                                                                    <L 222>
        var_33 = wp::address(var_particle_q, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // p1 = particle_q[n1]                                                                    <L 223>
        var_36 = wp::address(var_particle_q, var_7);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // p2 = particle_q[n2]                                                                    <L 224>
        var_39 = wp::address(var_particle_q, var_11);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // p3 = particle_q[n3]                                                                    <L 225>
        var_42 = wp::address(var_particle_q, var_15);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // p4 = particle_q[n4]                                                                    <L 226>
        var_45 = wp::address(var_particle_q, var_19);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // p5 = particle_q[n5]                                                                    <L 227>
        var_48 = wp::address(var_particle_q, var_23);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // p6 = particle_q[n6]                                                                    <L 228>
        var_51 = wp::address(var_particle_q, var_27);
        var_53 = wp::load(var_51);
        var_52 = wp::copy(var_53);
        // p7 = particle_q[n7]                                                                    <L 229>
        var_54 = wp::address(var_particle_q, var_31);
        var_56 = wp::load(var_54);
        var_55 = wp::copy(var_56);
        // lo = wp.vec3(                                                                          <L 231>
        // wp.min(wp.min(wp.min(p0[0], p1[0]), wp.min(p2[0], p3[0])), wp.min(wp.min(p4[0], p5[0]), wp.min(p6[0], p7[0]))),       <L 232>
        var_58 = wp::extract(var_34, var_57);
        var_60 = wp::extract(var_37, var_59);
        var_61 = wp::min(var_58, var_60);
        var_63 = wp::extract(var_40, var_62);
        var_65 = wp::extract(var_43, var_64);
        var_66 = wp::min(var_63, var_65);
        var_67 = wp::min(var_61, var_66);
        var_69 = wp::extract(var_46, var_68);
        var_71 = wp::extract(var_49, var_70);
        var_72 = wp::min(var_69, var_71);
        var_74 = wp::extract(var_52, var_73);
        var_76 = wp::extract(var_55, var_75);
        var_77 = wp::min(var_74, var_76);
        var_78 = wp::min(var_72, var_77);
        var_79 = wp::min(var_67, var_78);
        // wp.min(wp.min(wp.min(p0[1], p1[1]), wp.min(p2[1], p3[1])), wp.min(wp.min(p4[1], p5[1]), wp.min(p6[1], p7[1]))),       <L 233>
        var_81 = wp::extract(var_34, var_80);
        var_83 = wp::extract(var_37, var_82);
        var_84 = wp::min(var_81, var_83);
        var_86 = wp::extract(var_40, var_85);
        var_88 = wp::extract(var_43, var_87);
        var_89 = wp::min(var_86, var_88);
        var_90 = wp::min(var_84, var_89);
        var_92 = wp::extract(var_46, var_91);
        var_94 = wp::extract(var_49, var_93);
        var_95 = wp::min(var_92, var_94);
        var_97 = wp::extract(var_52, var_96);
        var_99 = wp::extract(var_55, var_98);
        var_100 = wp::min(var_97, var_99);
        var_101 = wp::min(var_95, var_100);
        var_102 = wp::min(var_90, var_101);
        // wp.min(wp.min(wp.min(p0[2], p1[2]), wp.min(p2[2], p3[2])), wp.min(wp.min(p4[2], p5[2]), wp.min(p6[2], p7[2]))),       <L 234>
        var_104 = wp::extract(var_34, var_103);
        var_106 = wp::extract(var_37, var_105);
        var_107 = wp::min(var_104, var_106);
        var_109 = wp::extract(var_40, var_108);
        var_111 = wp::extract(var_43, var_110);
        var_112 = wp::min(var_109, var_111);
        var_113 = wp::min(var_107, var_112);
        var_115 = wp::extract(var_46, var_114);
        var_117 = wp::extract(var_49, var_116);
        var_118 = wp::min(var_115, var_117);
        var_120 = wp::extract(var_52, var_119);
        var_122 = wp::extract(var_55, var_121);
        var_123 = wp::min(var_120, var_122);
        var_124 = wp::min(var_118, var_123);
        var_125 = wp::min(var_113, var_124);
        var_126 = wp::vec_t<3, wp::float32>(var_79, var_102, var_125);
        // hi = wp.vec3(                                                                          <L 236>
        // wp.max(wp.max(wp.max(p0[0], p1[0]), wp.max(p2[0], p3[0])), wp.max(wp.max(p4[0], p5[0]), wp.max(p6[0], p7[0]))),       <L 237>
        var_128 = wp::extract(var_34, var_127);
        var_130 = wp::extract(var_37, var_129);
        var_131 = wp::max(var_128, var_130);
        var_133 = wp::extract(var_40, var_132);
        var_135 = wp::extract(var_43, var_134);
        var_136 = wp::max(var_133, var_135);
        var_137 = wp::max(var_131, var_136);
        var_139 = wp::extract(var_46, var_138);
        var_141 = wp::extract(var_49, var_140);
        var_142 = wp::max(var_139, var_141);
        var_144 = wp::extract(var_52, var_143);
        var_146 = wp::extract(var_55, var_145);
        var_147 = wp::max(var_144, var_146);
        var_148 = wp::max(var_142, var_147);
        var_149 = wp::max(var_137, var_148);
        // wp.max(wp.max(wp.max(p0[1], p1[1]), wp.max(p2[1], p3[1])), wp.max(wp.max(p4[1], p5[1]), wp.max(p6[1], p7[1]))),       <L 238>
        var_151 = wp::extract(var_34, var_150);
        var_153 = wp::extract(var_37, var_152);
        var_154 = wp::max(var_151, var_153);
        var_156 = wp::extract(var_40, var_155);
        var_158 = wp::extract(var_43, var_157);
        var_159 = wp::max(var_156, var_158);
        var_160 = wp::max(var_154, var_159);
        var_162 = wp::extract(var_46, var_161);
        var_164 = wp::extract(var_49, var_163);
        var_165 = wp::max(var_162, var_164);
        var_167 = wp::extract(var_52, var_166);
        var_169 = wp::extract(var_55, var_168);
        var_170 = wp::max(var_167, var_169);
        var_171 = wp::max(var_165, var_170);
        var_172 = wp::max(var_160, var_171);
        // wp.max(wp.max(wp.max(p0[2], p1[2]), wp.max(p2[2], p3[2])), wp.max(wp.max(p4[2], p5[2]), wp.max(p6[2], p7[2]))),       <L 239>
        var_174 = wp::extract(var_34, var_173);
        var_176 = wp::extract(var_37, var_175);
        var_177 = wp::max(var_174, var_176);
        var_179 = wp::extract(var_40, var_178);
        var_181 = wp::extract(var_43, var_180);
        var_182 = wp::max(var_179, var_181);
        var_183 = wp::max(var_177, var_182);
        var_185 = wp::extract(var_46, var_184);
        var_187 = wp::extract(var_49, var_186);
        var_188 = wp::max(var_185, var_187);
        var_190 = wp::extract(var_52, var_189);
        var_192 = wp::extract(var_55, var_191);
        var_193 = wp::max(var_190, var_192);
        var_194 = wp::max(var_188, var_193);
        var_195 = wp::max(var_183, var_194);
        var_196 = wp::vec_t<3, wp::float32>(var_149, var_172, var_195);
        // aabb_min[c] = lo                                                                       <L 241>
        wp::array_store(var_aabb_min, var_0, var_126);
        // aabb_max[c] = hi                                                                       <L 242>
        wp::array_store(var_aabb_max, var_0, var_196);
        // cell_center_q[c] = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)               <L 244>
        var_197 = wp::add(var_34, var_37);
        var_198 = wp::add(var_197, var_40);
        var_199 = wp::add(var_198, var_43);
        var_200 = wp::add(var_199, var_46);
        var_201 = wp::add(var_200, var_49);
        var_202 = wp::add(var_201, var_52);
        var_203 = wp::add(var_202, var_55);
        var_206 = wp::div(var_204, var_205);
        var_207 = wp::mul(var_203, var_206);
        wp::array_store(var_cell_center_q, var_0, var_207);
        // ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                            <L 246>
        var_208 = wp::sub(var_37, var_34);
        var_209 = wp::sub(var_40, var_43);
        var_210 = wp::add(var_208, var_209);
        var_211 = wp::sub(var_49, var_46);
        var_212 = wp::add(var_210, var_211);
        var_213 = wp::sub(var_52, var_55);
        var_214 = wp::add(var_212, var_213);
        var_216 = wp::mul(var_214, var_215);
        // ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                            <L 247>
        var_217 = wp::sub(var_43, var_34);
        var_218 = wp::sub(var_40, var_37);
        var_219 = wp::add(var_217, var_218);
        var_220 = wp::sub(var_55, var_46);
        var_221 = wp::add(var_219, var_220);
        var_222 = wp::sub(var_52, var_49);
        var_223 = wp::add(var_221, var_222);
        var_225 = wp::mul(var_223, var_224);
        // az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                            <L 248>
        var_226 = wp::sub(var_46, var_34);
        var_227 = wp::sub(var_49, var_37);
        var_228 = wp::add(var_226, var_227);
        var_229 = wp::sub(var_52, var_40);
        var_230 = wp::add(var_228, var_229);
        var_231 = wp::sub(var_55, var_43);
        var_232 = wp::add(var_230, var_231);
        var_234 = wp::mul(var_232, var_233);
        // cell_orientation[c] = _orthonormalize(ax, ay, az)                                      <L 249>
        var_235 = _orthonormalize_0(var_216, var_225, var_234);
        wp::array_store(var_cell_orientation, var_0, var_235);
        // if cell_active[c] != 0:                                                                <L 251>
        var_236 = wp::address(var_cell_active, var_0);
        var_239 = wp::load(var_236);
        var_238 = (var_239 != var_237);
        if (var_238) {
            // cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)                              <L 252>
            var_242 = wp::int32(var_241);
            wp::array_store(var_cell_render_flags, var_0, var_242);
            // if compute_stretch != 0:                                                           <L 253>
            var_244 = (var_compute_stretch != var_243);
            if (var_244) {
                // cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)       <L 254>
                var_245 = _cell_max_abs_strain_0(var_34, var_37, var_40, var_43, var_46, var_49, var_52, var_55, var_voxel_size);
                wp::array_store(var_cell_stretch, var_0, var_245);
            }
        }
        if (!var_238) {
            // cell_render_flags[c] = wp.int32(0)                                                 <L 256>
            var_247 = wp::int32(var_246);
            wp::array_store(var_cell_render_flags, var_0, var_247);
            // if compute_stretch != 0:                                                           <L 257>
            var_249 = (var_compute_stretch != var_248);
            if (var_249) {
                // cell_stretch[c] = 0.0                                                          <L 258>
                wp::array_store(var_cell_stretch, var_0, var_250);
            }
        }
    }
}



extern "C" __global__ void update_cell_render_state_and_aabbs_kernel_563a3e07_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::float32 var_voxel_size,
    wp::int32 var_compute_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cell_center_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell_orientation,
    wp::array_t<wp::int32> var_cell_render_flags,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::float32 adj_voxel_size,
    wp::int32 adj_compute_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_cell_center_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_cell_orientation,
    wp::array_t<wp::int32> adj_cell_render_flags,
    wp::array_t<wp::float32> adj_cell_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_max)
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
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 4;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 5;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 6;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 7;
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
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        const wp::int32 var_57 = 0;
        wp::float32 var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        wp::float32 var_61;
        const wp::int32 var_62 = 0;
        wp::float32 var_63;
        const wp::int32 var_64 = 0;
        wp::float32 var_65;
        wp::float32 var_66;
        wp::float32 var_67;
        const wp::int32 var_68 = 0;
        wp::float32 var_69;
        const wp::int32 var_70 = 0;
        wp::float32 var_71;
        wp::float32 var_72;
        const wp::int32 var_73 = 0;
        wp::float32 var_74;
        const wp::int32 var_75 = 0;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        const wp::int32 var_80 = 1;
        wp::float32 var_81;
        const wp::int32 var_82 = 1;
        wp::float32 var_83;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        wp::float32 var_86;
        const wp::int32 var_87 = 1;
        wp::float32 var_88;
        wp::float32 var_89;
        wp::float32 var_90;
        const wp::int32 var_91 = 1;
        wp::float32 var_92;
        const wp::int32 var_93 = 1;
        wp::float32 var_94;
        wp::float32 var_95;
        const wp::int32 var_96 = 1;
        wp::float32 var_97;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        wp::float32 var_100;
        wp::float32 var_101;
        wp::float32 var_102;
        const wp::int32 var_103 = 2;
        wp::float32 var_104;
        const wp::int32 var_105 = 2;
        wp::float32 var_106;
        wp::float32 var_107;
        const wp::int32 var_108 = 2;
        wp::float32 var_109;
        const wp::int32 var_110 = 2;
        wp::float32 var_111;
        wp::float32 var_112;
        wp::float32 var_113;
        const wp::int32 var_114 = 2;
        wp::float32 var_115;
        const wp::int32 var_116 = 2;
        wp::float32 var_117;
        wp::float32 var_118;
        const wp::int32 var_119 = 2;
        wp::float32 var_120;
        const wp::int32 var_121 = 2;
        wp::float32 var_122;
        wp::float32 var_123;
        wp::float32 var_124;
        wp::float32 var_125;
        wp::vec_t<3, wp::float32> var_126;
        const wp::int32 var_127 = 0;
        wp::float32 var_128;
        const wp::int32 var_129 = 0;
        wp::float32 var_130;
        wp::float32 var_131;
        const wp::int32 var_132 = 0;
        wp::float32 var_133;
        const wp::int32 var_134 = 0;
        wp::float32 var_135;
        wp::float32 var_136;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        wp::float32 var_139;
        const wp::int32 var_140 = 0;
        wp::float32 var_141;
        wp::float32 var_142;
        const wp::int32 var_143 = 0;
        wp::float32 var_144;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        wp::float32 var_147;
        wp::float32 var_148;
        wp::float32 var_149;
        const wp::int32 var_150 = 1;
        wp::float32 var_151;
        const wp::int32 var_152 = 1;
        wp::float32 var_153;
        wp::float32 var_154;
        const wp::int32 var_155 = 1;
        wp::float32 var_156;
        const wp::int32 var_157 = 1;
        wp::float32 var_158;
        wp::float32 var_159;
        wp::float32 var_160;
        const wp::int32 var_161 = 1;
        wp::float32 var_162;
        const wp::int32 var_163 = 1;
        wp::float32 var_164;
        wp::float32 var_165;
        const wp::int32 var_166 = 1;
        wp::float32 var_167;
        const wp::int32 var_168 = 1;
        wp::float32 var_169;
        wp::float32 var_170;
        wp::float32 var_171;
        wp::float32 var_172;
        const wp::int32 var_173 = 2;
        wp::float32 var_174;
        const wp::int32 var_175 = 2;
        wp::float32 var_176;
        wp::float32 var_177;
        const wp::int32 var_178 = 2;
        wp::float32 var_179;
        const wp::int32 var_180 = 2;
        wp::float32 var_181;
        wp::float32 var_182;
        wp::float32 var_183;
        const wp::int32 var_184 = 2;
        wp::float32 var_185;
        const wp::int32 var_186 = 2;
        wp::float32 var_187;
        wp::float32 var_188;
        const wp::int32 var_189 = 2;
        wp::float32 var_190;
        const wp::int32 var_191 = 2;
        wp::float32 var_192;
        wp::float32 var_193;
        wp::float32 var_194;
        wp::float32 var_195;
        wp::vec_t<3, wp::float32> var_196;
        wp::vec_t<3, wp::float32> var_197;
        wp::vec_t<3, wp::float32> var_198;
        wp::vec_t<3, wp::float32> var_199;
        wp::vec_t<3, wp::float32> var_200;
        wp::vec_t<3, wp::float32> var_201;
        wp::vec_t<3, wp::float32> var_202;
        wp::vec_t<3, wp::float32> var_203;
        const wp::float32 var_204 = 1.0;
        const wp::float32 var_205 = 8.0;
        wp::float32 var_206;
        wp::vec_t<3, wp::float32> var_207;
        wp::vec_t<3, wp::float32> var_208;
        wp::vec_t<3, wp::float32> var_209;
        wp::vec_t<3, wp::float32> var_210;
        wp::vec_t<3, wp::float32> var_211;
        wp::vec_t<3, wp::float32> var_212;
        wp::vec_t<3, wp::float32> var_213;
        wp::vec_t<3, wp::float32> var_214;
        const wp::float32 var_215 = 0.25;
        wp::vec_t<3, wp::float32> var_216;
        wp::vec_t<3, wp::float32> var_217;
        wp::vec_t<3, wp::float32> var_218;
        wp::vec_t<3, wp::float32> var_219;
        wp::vec_t<3, wp::float32> var_220;
        wp::vec_t<3, wp::float32> var_221;
        wp::vec_t<3, wp::float32> var_222;
        wp::vec_t<3, wp::float32> var_223;
        const wp::float32 var_224 = 0.25;
        wp::vec_t<3, wp::float32> var_225;
        wp::vec_t<3, wp::float32> var_226;
        wp::vec_t<3, wp::float32> var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        const wp::float32 var_233 = 0.25;
        wp::vec_t<3, wp::float32> var_234;
        wp::mat_t<3, 3, wp::float32> var_235;
        wp::int32* var_236;
        const wp::int32 var_237 = 0;
        bool var_238;
        wp::int32 var_239;
        const wp::int32 var_240 = 1;
        const wp::int32 var_241 = 1;
        wp::int32 var_242;
        const wp::int32 var_243 = 0;
        bool var_244;
        wp::float32 var_245;
        const wp::int32 var_246 = 0;
        wp::int32 var_247;
        const wp::int32 var_248 = 0;
        bool var_249;
        const wp::float32 var_250 = 0.0;
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
        wp::vec_t<3, wp::float32> adj_42 = {};
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
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        wp::vec_t<3, wp::float32> adj_55 = {};
        wp::vec_t<3, wp::float32> adj_56 = {};
        wp::int32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::float32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::float32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::float32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::float32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::float32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::float32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::float32 adj_100 = {};
        wp::float32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::float32 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::float32 adj_106 = {};
        wp::float32 adj_107 = {};
        wp::int32 adj_108 = {};
        wp::float32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::float32 adj_111 = {};
        wp::float32 adj_112 = {};
        wp::float32 adj_113 = {};
        wp::int32 adj_114 = {};
        wp::float32 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::float32 adj_117 = {};
        wp::float32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::float32 adj_120 = {};
        wp::int32 adj_121 = {};
        wp::float32 adj_122 = {};
        wp::float32 adj_123 = {};
        wp::float32 adj_124 = {};
        wp::float32 adj_125 = {};
        wp::vec_t<3, wp::float32> adj_126 = {};
        wp::int32 adj_127 = {};
        wp::float32 adj_128 = {};
        wp::int32 adj_129 = {};
        wp::float32 adj_130 = {};
        wp::float32 adj_131 = {};
        wp::int32 adj_132 = {};
        wp::float32 adj_133 = {};
        wp::int32 adj_134 = {};
        wp::float32 adj_135 = {};
        wp::float32 adj_136 = {};
        wp::float32 adj_137 = {};
        wp::int32 adj_138 = {};
        wp::float32 adj_139 = {};
        wp::int32 adj_140 = {};
        wp::float32 adj_141 = {};
        wp::float32 adj_142 = {};
        wp::int32 adj_143 = {};
        wp::float32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::float32 adj_147 = {};
        wp::float32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::float32 adj_151 = {};
        wp::int32 adj_152 = {};
        wp::float32 adj_153 = {};
        wp::float32 adj_154 = {};
        wp::int32 adj_155 = {};
        wp::float32 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::float32 adj_158 = {};
        wp::float32 adj_159 = {};
        wp::float32 adj_160 = {};
        wp::int32 adj_161 = {};
        wp::float32 adj_162 = {};
        wp::int32 adj_163 = {};
        wp::float32 adj_164 = {};
        wp::float32 adj_165 = {};
        wp::int32 adj_166 = {};
        wp::float32 adj_167 = {};
        wp::int32 adj_168 = {};
        wp::float32 adj_169 = {};
        wp::float32 adj_170 = {};
        wp::float32 adj_171 = {};
        wp::float32 adj_172 = {};
        wp::int32 adj_173 = {};
        wp::float32 adj_174 = {};
        wp::int32 adj_175 = {};
        wp::float32 adj_176 = {};
        wp::float32 adj_177 = {};
        wp::int32 adj_178 = {};
        wp::float32 adj_179 = {};
        wp::int32 adj_180 = {};
        wp::float32 adj_181 = {};
        wp::float32 adj_182 = {};
        wp::float32 adj_183 = {};
        wp::int32 adj_184 = {};
        wp::float32 adj_185 = {};
        wp::int32 adj_186 = {};
        wp::float32 adj_187 = {};
        wp::float32 adj_188 = {};
        wp::int32 adj_189 = {};
        wp::float32 adj_190 = {};
        wp::int32 adj_191 = {};
        wp::float32 adj_192 = {};
        wp::float32 adj_193 = {};
        wp::float32 adj_194 = {};
        wp::float32 adj_195 = {};
        wp::vec_t<3, wp::float32> adj_196 = {};
        wp::vec_t<3, wp::float32> adj_197 = {};
        wp::vec_t<3, wp::float32> adj_198 = {};
        wp::vec_t<3, wp::float32> adj_199 = {};
        wp::vec_t<3, wp::float32> adj_200 = {};
        wp::vec_t<3, wp::float32> adj_201 = {};
        wp::vec_t<3, wp::float32> adj_202 = {};
        wp::vec_t<3, wp::float32> adj_203 = {};
        wp::float32 adj_204 = {};
        wp::float32 adj_205 = {};
        wp::float32 adj_206 = {};
        wp::vec_t<3, wp::float32> adj_207 = {};
        wp::vec_t<3, wp::float32> adj_208 = {};
        wp::vec_t<3, wp::float32> adj_209 = {};
        wp::vec_t<3, wp::float32> adj_210 = {};
        wp::vec_t<3, wp::float32> adj_211 = {};
        wp::vec_t<3, wp::float32> adj_212 = {};
        wp::vec_t<3, wp::float32> adj_213 = {};
        wp::vec_t<3, wp::float32> adj_214 = {};
        wp::float32 adj_215 = {};
        wp::vec_t<3, wp::float32> adj_216 = {};
        wp::vec_t<3, wp::float32> adj_217 = {};
        wp::vec_t<3, wp::float32> adj_218 = {};
        wp::vec_t<3, wp::float32> adj_219 = {};
        wp::vec_t<3, wp::float32> adj_220 = {};
        wp::vec_t<3, wp::float32> adj_221 = {};
        wp::vec_t<3, wp::float32> adj_222 = {};
        wp::vec_t<3, wp::float32> adj_223 = {};
        wp::float32 adj_224 = {};
        wp::vec_t<3, wp::float32> adj_225 = {};
        wp::vec_t<3, wp::float32> adj_226 = {};
        wp::vec_t<3, wp::float32> adj_227 = {};
        wp::vec_t<3, wp::float32> adj_228 = {};
        wp::vec_t<3, wp::float32> adj_229 = {};
        wp::vec_t<3, wp::float32> adj_230 = {};
        wp::vec_t<3, wp::float32> adj_231 = {};
        wp::vec_t<3, wp::float32> adj_232 = {};
        wp::float32 adj_233 = {};
        wp::vec_t<3, wp::float32> adj_234 = {};
        wp::mat_t<3, 3, wp::float32> adj_235 = {};
        wp::int32 adj_236 = {};
        wp::int32 adj_237 = {};
        bool adj_238 = {};
        wp::int32 adj_239 = {};
        wp::int32 adj_240 = {};
        wp::int32 adj_241 = {};
        wp::int32 adj_242 = {};
        wp::int32 adj_243 = {};
        bool adj_244 = {};
        wp::float32 adj_245 = {};
        wp::int32 adj_246 = {};
        wp::int32 adj_247 = {};
        wp::int32 adj_248 = {};
        bool adj_249 = {};
        wp::float32 adj_250 = {};
        //---------
        // forward
        // def update_cell_render_state_and_aabbs_kernel(                                         <L 198>
        // c = wp.tid()                                                                           <L 212>
        var_0 = builtin_tid1d();
        // n0 = cell_nodes[c, 0]                                                                  <L 213>
        var_2 = wp::address(var_cell_nodes, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // n1 = cell_nodes[c, 1]                                                                  <L 214>
        var_6 = wp::address(var_cell_nodes, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // n2 = cell_nodes[c, 2]                                                                  <L 215>
        var_10 = wp::address(var_cell_nodes, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // n3 = cell_nodes[c, 3]                                                                  <L 216>
        var_14 = wp::address(var_cell_nodes, var_0, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // n4 = cell_nodes[c, 4]                                                                  <L 217>
        var_18 = wp::address(var_cell_nodes, var_0, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // n5 = cell_nodes[c, 5]                                                                  <L 218>
        var_22 = wp::address(var_cell_nodes, var_0, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // n6 = cell_nodes[c, 6]                                                                  <L 219>
        var_26 = wp::address(var_cell_nodes, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // n7 = cell_nodes[c, 7]                                                                  <L 220>
        var_30 = wp::address(var_cell_nodes, var_0, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // p0 = particle_q[n0]                                                                    <L 222>
        var_33 = wp::address(var_particle_q, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // p1 = particle_q[n1]                                                                    <L 223>
        var_36 = wp::address(var_particle_q, var_7);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // p2 = particle_q[n2]                                                                    <L 224>
        var_39 = wp::address(var_particle_q, var_11);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // p3 = particle_q[n3]                                                                    <L 225>
        var_42 = wp::address(var_particle_q, var_15);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // p4 = particle_q[n4]                                                                    <L 226>
        var_45 = wp::address(var_particle_q, var_19);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // p5 = particle_q[n5]                                                                    <L 227>
        var_48 = wp::address(var_particle_q, var_23);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // p6 = particle_q[n6]                                                                    <L 228>
        var_51 = wp::address(var_particle_q, var_27);
        var_53 = wp::load(var_51);
        var_52 = wp::copy(var_53);
        // p7 = particle_q[n7]                                                                    <L 229>
        var_54 = wp::address(var_particle_q, var_31);
        var_56 = wp::load(var_54);
        var_55 = wp::copy(var_56);
        // lo = wp.vec3(                                                                          <L 231>
        // wp.min(wp.min(wp.min(p0[0], p1[0]), wp.min(p2[0], p3[0])), wp.min(wp.min(p4[0], p5[0]), wp.min(p6[0], p7[0]))),       <L 232>
        var_58 = wp::extract(var_34, var_57);
        var_60 = wp::extract(var_37, var_59);
        var_61 = wp::min(var_58, var_60);
        var_63 = wp::extract(var_40, var_62);
        var_65 = wp::extract(var_43, var_64);
        var_66 = wp::min(var_63, var_65);
        var_67 = wp::min(var_61, var_66);
        var_69 = wp::extract(var_46, var_68);
        var_71 = wp::extract(var_49, var_70);
        var_72 = wp::min(var_69, var_71);
        var_74 = wp::extract(var_52, var_73);
        var_76 = wp::extract(var_55, var_75);
        var_77 = wp::min(var_74, var_76);
        var_78 = wp::min(var_72, var_77);
        var_79 = wp::min(var_67, var_78);
        // wp.min(wp.min(wp.min(p0[1], p1[1]), wp.min(p2[1], p3[1])), wp.min(wp.min(p4[1], p5[1]), wp.min(p6[1], p7[1]))),       <L 233>
        var_81 = wp::extract(var_34, var_80);
        var_83 = wp::extract(var_37, var_82);
        var_84 = wp::min(var_81, var_83);
        var_86 = wp::extract(var_40, var_85);
        var_88 = wp::extract(var_43, var_87);
        var_89 = wp::min(var_86, var_88);
        var_90 = wp::min(var_84, var_89);
        var_92 = wp::extract(var_46, var_91);
        var_94 = wp::extract(var_49, var_93);
        var_95 = wp::min(var_92, var_94);
        var_97 = wp::extract(var_52, var_96);
        var_99 = wp::extract(var_55, var_98);
        var_100 = wp::min(var_97, var_99);
        var_101 = wp::min(var_95, var_100);
        var_102 = wp::min(var_90, var_101);
        // wp.min(wp.min(wp.min(p0[2], p1[2]), wp.min(p2[2], p3[2])), wp.min(wp.min(p4[2], p5[2]), wp.min(p6[2], p7[2]))),       <L 234>
        var_104 = wp::extract(var_34, var_103);
        var_106 = wp::extract(var_37, var_105);
        var_107 = wp::min(var_104, var_106);
        var_109 = wp::extract(var_40, var_108);
        var_111 = wp::extract(var_43, var_110);
        var_112 = wp::min(var_109, var_111);
        var_113 = wp::min(var_107, var_112);
        var_115 = wp::extract(var_46, var_114);
        var_117 = wp::extract(var_49, var_116);
        var_118 = wp::min(var_115, var_117);
        var_120 = wp::extract(var_52, var_119);
        var_122 = wp::extract(var_55, var_121);
        var_123 = wp::min(var_120, var_122);
        var_124 = wp::min(var_118, var_123);
        var_125 = wp::min(var_113, var_124);
        var_126 = wp::vec_t<3, wp::float32>(var_79, var_102, var_125);
        // hi = wp.vec3(                                                                          <L 236>
        // wp.max(wp.max(wp.max(p0[0], p1[0]), wp.max(p2[0], p3[0])), wp.max(wp.max(p4[0], p5[0]), wp.max(p6[0], p7[0]))),       <L 237>
        var_128 = wp::extract(var_34, var_127);
        var_130 = wp::extract(var_37, var_129);
        var_131 = wp::max(var_128, var_130);
        var_133 = wp::extract(var_40, var_132);
        var_135 = wp::extract(var_43, var_134);
        var_136 = wp::max(var_133, var_135);
        var_137 = wp::max(var_131, var_136);
        var_139 = wp::extract(var_46, var_138);
        var_141 = wp::extract(var_49, var_140);
        var_142 = wp::max(var_139, var_141);
        var_144 = wp::extract(var_52, var_143);
        var_146 = wp::extract(var_55, var_145);
        var_147 = wp::max(var_144, var_146);
        var_148 = wp::max(var_142, var_147);
        var_149 = wp::max(var_137, var_148);
        // wp.max(wp.max(wp.max(p0[1], p1[1]), wp.max(p2[1], p3[1])), wp.max(wp.max(p4[1], p5[1]), wp.max(p6[1], p7[1]))),       <L 238>
        var_151 = wp::extract(var_34, var_150);
        var_153 = wp::extract(var_37, var_152);
        var_154 = wp::max(var_151, var_153);
        var_156 = wp::extract(var_40, var_155);
        var_158 = wp::extract(var_43, var_157);
        var_159 = wp::max(var_156, var_158);
        var_160 = wp::max(var_154, var_159);
        var_162 = wp::extract(var_46, var_161);
        var_164 = wp::extract(var_49, var_163);
        var_165 = wp::max(var_162, var_164);
        var_167 = wp::extract(var_52, var_166);
        var_169 = wp::extract(var_55, var_168);
        var_170 = wp::max(var_167, var_169);
        var_171 = wp::max(var_165, var_170);
        var_172 = wp::max(var_160, var_171);
        // wp.max(wp.max(wp.max(p0[2], p1[2]), wp.max(p2[2], p3[2])), wp.max(wp.max(p4[2], p5[2]), wp.max(p6[2], p7[2]))),       <L 239>
        var_174 = wp::extract(var_34, var_173);
        var_176 = wp::extract(var_37, var_175);
        var_177 = wp::max(var_174, var_176);
        var_179 = wp::extract(var_40, var_178);
        var_181 = wp::extract(var_43, var_180);
        var_182 = wp::max(var_179, var_181);
        var_183 = wp::max(var_177, var_182);
        var_185 = wp::extract(var_46, var_184);
        var_187 = wp::extract(var_49, var_186);
        var_188 = wp::max(var_185, var_187);
        var_190 = wp::extract(var_52, var_189);
        var_192 = wp::extract(var_55, var_191);
        var_193 = wp::max(var_190, var_192);
        var_194 = wp::max(var_188, var_193);
        var_195 = wp::max(var_183, var_194);
        var_196 = wp::vec_t<3, wp::float32>(var_149, var_172, var_195);
        // aabb_min[c] = lo                                                                       <L 241>
        // wp::array_store(var_aabb_min, var_0, var_126);
        // aabb_max[c] = hi                                                                       <L 242>
        // wp::array_store(var_aabb_max, var_0, var_196);
        // cell_center_q[c] = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)               <L 244>
        var_197 = wp::add(var_34, var_37);
        var_198 = wp::add(var_197, var_40);
        var_199 = wp::add(var_198, var_43);
        var_200 = wp::add(var_199, var_46);
        var_201 = wp::add(var_200, var_49);
        var_202 = wp::add(var_201, var_52);
        var_203 = wp::add(var_202, var_55);
        var_206 = wp::div(var_204, var_205);
        var_207 = wp::mul(var_203, var_206);
        // wp::array_store(var_cell_center_q, var_0, var_207);
        // ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                            <L 246>
        var_208 = wp::sub(var_37, var_34);
        var_209 = wp::sub(var_40, var_43);
        var_210 = wp::add(var_208, var_209);
        var_211 = wp::sub(var_49, var_46);
        var_212 = wp::add(var_210, var_211);
        var_213 = wp::sub(var_52, var_55);
        var_214 = wp::add(var_212, var_213);
        var_216 = wp::mul(var_214, var_215);
        // ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                            <L 247>
        var_217 = wp::sub(var_43, var_34);
        var_218 = wp::sub(var_40, var_37);
        var_219 = wp::add(var_217, var_218);
        var_220 = wp::sub(var_55, var_46);
        var_221 = wp::add(var_219, var_220);
        var_222 = wp::sub(var_52, var_49);
        var_223 = wp::add(var_221, var_222);
        var_225 = wp::mul(var_223, var_224);
        // az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                            <L 248>
        var_226 = wp::sub(var_46, var_34);
        var_227 = wp::sub(var_49, var_37);
        var_228 = wp::add(var_226, var_227);
        var_229 = wp::sub(var_52, var_40);
        var_230 = wp::add(var_228, var_229);
        var_231 = wp::sub(var_55, var_43);
        var_232 = wp::add(var_230, var_231);
        var_234 = wp::mul(var_232, var_233);
        // cell_orientation[c] = _orthonormalize(ax, ay, az)                                      <L 249>
        var_235 = _orthonormalize_0(var_216, var_225, var_234);
        // wp::array_store(var_cell_orientation, var_0, var_235);
        // if cell_active[c] != 0:                                                                <L 251>
        var_236 = wp::address(var_cell_active, var_0);
        var_239 = wp::load(var_236);
        var_238 = (var_239 != var_237);
        if (var_238) {
            // cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)                              <L 252>
            var_242 = wp::int32(var_241);
            // wp::array_store(var_cell_render_flags, var_0, var_242);
            // if compute_stretch != 0:                                                           <L 253>
            var_244 = (var_compute_stretch != var_243);
            if (var_244) {
                // cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)       <L 254>
                var_245 = _cell_max_abs_strain_0(var_34, var_37, var_40, var_43, var_46, var_49, var_52, var_55, var_voxel_size);
                // wp::array_store(var_cell_stretch, var_0, var_245);
            }
        }
        if (!var_238) {
            // cell_render_flags[c] = wp.int32(0)                                                 <L 256>
            var_247 = wp::int32(var_246);
            // wp::array_store(var_cell_render_flags, var_0, var_247);
            // if compute_stretch != 0:                                                           <L 257>
            var_249 = (var_compute_stretch != var_248);
            if (var_249) {
                // cell_stretch[c] = 0.0                                                          <L 258>
                // wp::array_store(var_cell_stretch, var_0, var_250);
            }
        }
        //---------
        // reverse
        if (!var_238) {
            if (var_249) {
                wp::adj_array_store(var_cell_stretch, var_0, var_250, adj_cell_stretch, adj_0, adj_250);
                // adj: cell_stretch[c] = 0.0                                                     <L 258>
            }
            // adj: if compute_stretch != 0:                                                      <L 257>
            wp::adj_array_store(var_cell_render_flags, var_0, var_247, adj_cell_render_flags, adj_0, adj_247);
            wp::adj_int32(var_246, adj_246, adj_247);
            // adj: cell_render_flags[c] = wp.int32(0)                                            <L 256>
        }
        if (var_238) {
            if (var_244) {
                wp::adj_array_store(var_cell_stretch, var_0, var_245, adj_cell_stretch, adj_0, adj_245);
                adj__cell_max_abs_strain_0(var_34, var_37, var_40, var_43, var_46, var_49, var_52, var_55, var_voxel_size, adj_34, adj_37, adj_40, adj_43, adj_46, adj_49, adj_52, adj_55, adj_voxel_size, adj_245);
                // adj: cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)  <L 254>
            }
            // adj: if compute_stretch != 0:                                                      <L 253>
            wp::adj_array_store(var_cell_render_flags, var_0, var_242, adj_cell_render_flags, adj_0, adj_242);
            wp::adj_int32(var_241, adj_241, adj_242);
            // adj: cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)                         <L 252>
        }
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_236);
        // adj: if cell_active[c] != 0:                                                           <L 251>
        wp::adj_array_store(var_cell_orientation, var_0, var_235, adj_cell_orientation, adj_0, adj_235);
        adj__orthonormalize_0(var_216, var_225, var_234, adj_216, adj_225, adj_234, adj_235);
        // adj: cell_orientation[c] = _orthonormalize(ax, ay, az)                                 <L 249>
        wp::adj_mul(var_232, var_233, adj_232, adj_233, adj_234);
        wp::adj_add(var_230, var_231, adj_230, adj_231, adj_232);
        wp::adj_sub(var_55, var_43, adj_55, adj_43, adj_231);
        wp::adj_add(var_228, var_229, adj_228, adj_229, adj_230);
        wp::adj_sub(var_52, var_40, adj_52, adj_40, adj_229);
        wp::adj_add(var_226, var_227, adj_226, adj_227, adj_228);
        wp::adj_sub(var_49, var_37, adj_49, adj_37, adj_227);
        wp::adj_sub(var_46, var_34, adj_46, adj_34, adj_226);
        // adj: az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                       <L 248>
        wp::adj_mul(var_223, var_224, adj_223, adj_224, adj_225);
        wp::adj_add(var_221, var_222, adj_221, adj_222, adj_223);
        wp::adj_sub(var_52, var_49, adj_52, adj_49, adj_222);
        wp::adj_add(var_219, var_220, adj_219, adj_220, adj_221);
        wp::adj_sub(var_55, var_46, adj_55, adj_46, adj_220);
        wp::adj_add(var_217, var_218, adj_217, adj_218, adj_219);
        wp::adj_sub(var_40, var_37, adj_40, adj_37, adj_218);
        wp::adj_sub(var_43, var_34, adj_43, adj_34, adj_217);
        // adj: ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                       <L 247>
        wp::adj_mul(var_214, var_215, adj_214, adj_215, adj_216);
        wp::adj_add(var_212, var_213, adj_212, adj_213, adj_214);
        wp::adj_sub(var_52, var_55, adj_52, adj_55, adj_213);
        wp::adj_add(var_210, var_211, adj_210, adj_211, adj_212);
        wp::adj_sub(var_49, var_46, adj_49, adj_46, adj_211);
        wp::adj_add(var_208, var_209, adj_208, adj_209, adj_210);
        wp::adj_sub(var_40, var_43, adj_40, adj_43, adj_209);
        wp::adj_sub(var_37, var_34, adj_37, adj_34, adj_208);
        // adj: ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                       <L 246>
        wp::adj_array_store(var_cell_center_q, var_0, var_207, adj_cell_center_q, adj_0, adj_207);
        wp::adj_mul(var_203, var_206, adj_203, adj_206, adj_207);
        wp::adj_div(var_204, var_205, var_206, adj_204, adj_205, adj_206);
        wp::adj_add(var_202, var_55, adj_202, adj_55, adj_203);
        wp::adj_add(var_201, var_52, adj_201, adj_52, adj_202);
        wp::adj_add(var_200, var_49, adj_200, adj_49, adj_201);
        wp::adj_add(var_199, var_46, adj_199, adj_46, adj_200);
        wp::adj_add(var_198, var_43, adj_198, adj_43, adj_199);
        wp::adj_add(var_197, var_40, adj_197, adj_40, adj_198);
        wp::adj_add(var_34, var_37, adj_34, adj_37, adj_197);
        // adj: cell_center_q[c] = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)          <L 244>
        wp::adj_array_store(var_aabb_max, var_0, var_196, adj_aabb_max, adj_0, adj_196);
        // adj: aabb_max[c] = hi                                                                  <L 242>
        wp::adj_array_store(var_aabb_min, var_0, var_126, adj_aabb_min, adj_0, adj_126);
        // adj: aabb_min[c] = lo                                                                  <L 241>
        wp::adj_vec_t(var_149, var_172, var_195, adj_149, adj_172, adj_195, adj_196);
        wp::adj_max(var_183, var_194, adj_183, adj_194, adj_195);
        wp::adj_max(var_188, var_193, adj_188, adj_193, adj_194);
        wp::adj_max(var_190, var_192, adj_190, adj_192, adj_193);
        wp::adj_extract(var_55, var_191, adj_55, adj_191, adj_192);
        wp::adj_extract(var_52, var_189, adj_52, adj_189, adj_190);
        wp::adj_max(var_185, var_187, adj_185, adj_187, adj_188);
        wp::adj_extract(var_49, var_186, adj_49, adj_186, adj_187);
        wp::adj_extract(var_46, var_184, adj_46, adj_184, adj_185);
        wp::adj_max(var_177, var_182, adj_177, adj_182, adj_183);
        wp::adj_max(var_179, var_181, adj_179, adj_181, adj_182);
        wp::adj_extract(var_43, var_180, adj_43, adj_180, adj_181);
        wp::adj_extract(var_40, var_178, adj_40, adj_178, adj_179);
        wp::adj_max(var_174, var_176, adj_174, adj_176, adj_177);
        wp::adj_extract(var_37, var_175, adj_37, adj_175, adj_176);
        wp::adj_extract(var_34, var_173, adj_34, adj_173, adj_174);
        // adj: wp.max(wp.max(wp.max(p0[2], p1[2]), wp.max(p2[2], p3[2])), wp.max(wp.max(p4[2], p5[2]), wp.max(p6[2], p7[2]))),  <L 239>
        wp::adj_max(var_160, var_171, adj_160, adj_171, adj_172);
        wp::adj_max(var_165, var_170, adj_165, adj_170, adj_171);
        wp::adj_max(var_167, var_169, adj_167, adj_169, adj_170);
        wp::adj_extract(var_55, var_168, adj_55, adj_168, adj_169);
        wp::adj_extract(var_52, var_166, adj_52, adj_166, adj_167);
        wp::adj_max(var_162, var_164, adj_162, adj_164, adj_165);
        wp::adj_extract(var_49, var_163, adj_49, adj_163, adj_164);
        wp::adj_extract(var_46, var_161, adj_46, adj_161, adj_162);
        wp::adj_max(var_154, var_159, adj_154, adj_159, adj_160);
        wp::adj_max(var_156, var_158, adj_156, adj_158, adj_159);
        wp::adj_extract(var_43, var_157, adj_43, adj_157, adj_158);
        wp::adj_extract(var_40, var_155, adj_40, adj_155, adj_156);
        wp::adj_max(var_151, var_153, adj_151, adj_153, adj_154);
        wp::adj_extract(var_37, var_152, adj_37, adj_152, adj_153);
        wp::adj_extract(var_34, var_150, adj_34, adj_150, adj_151);
        // adj: wp.max(wp.max(wp.max(p0[1], p1[1]), wp.max(p2[1], p3[1])), wp.max(wp.max(p4[1], p5[1]), wp.max(p6[1], p7[1]))),  <L 238>
        wp::adj_max(var_137, var_148, adj_137, adj_148, adj_149);
        wp::adj_max(var_142, var_147, adj_142, adj_147, adj_148);
        wp::adj_max(var_144, var_146, adj_144, adj_146, adj_147);
        wp::adj_extract(var_55, var_145, adj_55, adj_145, adj_146);
        wp::adj_extract(var_52, var_143, adj_52, adj_143, adj_144);
        wp::adj_max(var_139, var_141, adj_139, adj_141, adj_142);
        wp::adj_extract(var_49, var_140, adj_49, adj_140, adj_141);
        wp::adj_extract(var_46, var_138, adj_46, adj_138, adj_139);
        wp::adj_max(var_131, var_136, adj_131, adj_136, adj_137);
        wp::adj_max(var_133, var_135, adj_133, adj_135, adj_136);
        wp::adj_extract(var_43, var_134, adj_43, adj_134, adj_135);
        wp::adj_extract(var_40, var_132, adj_40, adj_132, adj_133);
        wp::adj_max(var_128, var_130, adj_128, adj_130, adj_131);
        wp::adj_extract(var_37, var_129, adj_37, adj_129, adj_130);
        wp::adj_extract(var_34, var_127, adj_34, adj_127, adj_128);
        // adj: wp.max(wp.max(wp.max(p0[0], p1[0]), wp.max(p2[0], p3[0])), wp.max(wp.max(p4[0], p5[0]), wp.max(p6[0], p7[0]))),  <L 237>
        // adj: hi = wp.vec3(                                                                     <L 236>
        wp::adj_vec_t(var_79, var_102, var_125, adj_79, adj_102, adj_125, adj_126);
        wp::adj_min(var_113, var_124, adj_113, adj_124, adj_125);
        wp::adj_min(var_118, var_123, adj_118, adj_123, adj_124);
        wp::adj_min(var_120, var_122, adj_120, adj_122, adj_123);
        wp::adj_extract(var_55, var_121, adj_55, adj_121, adj_122);
        wp::adj_extract(var_52, var_119, adj_52, adj_119, adj_120);
        wp::adj_min(var_115, var_117, adj_115, adj_117, adj_118);
        wp::adj_extract(var_49, var_116, adj_49, adj_116, adj_117);
        wp::adj_extract(var_46, var_114, adj_46, adj_114, adj_115);
        wp::adj_min(var_107, var_112, adj_107, adj_112, adj_113);
        wp::adj_min(var_109, var_111, adj_109, adj_111, adj_112);
        wp::adj_extract(var_43, var_110, adj_43, adj_110, adj_111);
        wp::adj_extract(var_40, var_108, adj_40, adj_108, adj_109);
        wp::adj_min(var_104, var_106, adj_104, adj_106, adj_107);
        wp::adj_extract(var_37, var_105, adj_37, adj_105, adj_106);
        wp::adj_extract(var_34, var_103, adj_34, adj_103, adj_104);
        // adj: wp.min(wp.min(wp.min(p0[2], p1[2]), wp.min(p2[2], p3[2])), wp.min(wp.min(p4[2], p5[2]), wp.min(p6[2], p7[2]))),  <L 234>
        wp::adj_min(var_90, var_101, adj_90, adj_101, adj_102);
        wp::adj_min(var_95, var_100, adj_95, adj_100, adj_101);
        wp::adj_min(var_97, var_99, adj_97, adj_99, adj_100);
        wp::adj_extract(var_55, var_98, adj_55, adj_98, adj_99);
        wp::adj_extract(var_52, var_96, adj_52, adj_96, adj_97);
        wp::adj_min(var_92, var_94, adj_92, adj_94, adj_95);
        wp::adj_extract(var_49, var_93, adj_49, adj_93, adj_94);
        wp::adj_extract(var_46, var_91, adj_46, adj_91, adj_92);
        wp::adj_min(var_84, var_89, adj_84, adj_89, adj_90);
        wp::adj_min(var_86, var_88, adj_86, adj_88, adj_89);
        wp::adj_extract(var_43, var_87, adj_43, adj_87, adj_88);
        wp::adj_extract(var_40, var_85, adj_40, adj_85, adj_86);
        wp::adj_min(var_81, var_83, adj_81, adj_83, adj_84);
        wp::adj_extract(var_37, var_82, adj_37, adj_82, adj_83);
        wp::adj_extract(var_34, var_80, adj_34, adj_80, adj_81);
        // adj: wp.min(wp.min(wp.min(p0[1], p1[1]), wp.min(p2[1], p3[1])), wp.min(wp.min(p4[1], p5[1]), wp.min(p6[1], p7[1]))),  <L 233>
        wp::adj_min(var_67, var_78, adj_67, adj_78, adj_79);
        wp::adj_min(var_72, var_77, adj_72, adj_77, adj_78);
        wp::adj_min(var_74, var_76, adj_74, adj_76, adj_77);
        wp::adj_extract(var_55, var_75, adj_55, adj_75, adj_76);
        wp::adj_extract(var_52, var_73, adj_52, adj_73, adj_74);
        wp::adj_min(var_69, var_71, adj_69, adj_71, adj_72);
        wp::adj_extract(var_49, var_70, adj_49, adj_70, adj_71);
        wp::adj_extract(var_46, var_68, adj_46, adj_68, adj_69);
        wp::adj_min(var_61, var_66, adj_61, adj_66, adj_67);
        wp::adj_min(var_63, var_65, adj_63, adj_65, adj_66);
        wp::adj_extract(var_43, var_64, adj_43, adj_64, adj_65);
        wp::adj_extract(var_40, var_62, adj_40, adj_62, adj_63);
        wp::adj_min(var_58, var_60, adj_58, adj_60, adj_61);
        wp::adj_extract(var_37, var_59, adj_37, adj_59, adj_60);
        wp::adj_extract(var_34, var_57, adj_34, adj_57, adj_58);
        // adj: wp.min(wp.min(wp.min(p0[0], p1[0]), wp.min(p2[0], p3[0])), wp.min(wp.min(p4[0], p5[0]), wp.min(p6[0], p7[0]))),  <L 232>
        // adj: lo = wp.vec3(                                                                     <L 231>
        wp::adj_copy(var_56, adj_54, adj_55);
        wp::adj_address(var_particle_q, var_31, adj_particle_q, adj_31, adj_54);
        // adj: p7 = particle_q[n7]                                                               <L 229>
        wp::adj_copy(var_53, adj_51, adj_52);
        wp::adj_address(var_particle_q, var_27, adj_particle_q, adj_27, adj_51);
        // adj: p6 = particle_q[n6]                                                               <L 228>
        wp::adj_copy(var_50, adj_48, adj_49);
        wp::adj_address(var_particle_q, var_23, adj_particle_q, adj_23, adj_48);
        // adj: p5 = particle_q[n5]                                                               <L 227>
        wp::adj_copy(var_47, adj_45, adj_46);
        wp::adj_address(var_particle_q, var_19, adj_particle_q, adj_19, adj_45);
        // adj: p4 = particle_q[n4]                                                               <L 226>
        wp::adj_copy(var_44, adj_42, adj_43);
        wp::adj_address(var_particle_q, var_15, adj_particle_q, adj_15, adj_42);
        // adj: p3 = particle_q[n3]                                                               <L 225>
        wp::adj_copy(var_41, adj_39, adj_40);
        wp::adj_address(var_particle_q, var_11, adj_particle_q, adj_11, adj_39);
        // adj: p2 = particle_q[n2]                                                               <L 224>
        wp::adj_copy(var_38, adj_36, adj_37);
        wp::adj_address(var_particle_q, var_7, adj_particle_q, adj_7, adj_36);
        // adj: p1 = particle_q[n1]                                                               <L 223>
        wp::adj_copy(var_35, adj_33, adj_34);
        wp::adj_address(var_particle_q, var_3, adj_particle_q, adj_3, adj_33);
        // adj: p0 = particle_q[n0]                                                               <L 222>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_cell_nodes, var_0, var_29, adj_cell_nodes, adj_0, adj_29, adj_30);
        // adj: n7 = cell_nodes[c, 7]                                                             <L 220>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_cell_nodes, var_0, var_25, adj_cell_nodes, adj_0, adj_25, adj_26);
        // adj: n6 = cell_nodes[c, 6]                                                             <L 219>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_cell_nodes, var_0, var_21, adj_cell_nodes, adj_0, adj_21, adj_22);
        // adj: n5 = cell_nodes[c, 5]                                                             <L 218>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_cell_nodes, var_0, var_17, adj_cell_nodes, adj_0, adj_17, adj_18);
        // adj: n4 = cell_nodes[c, 4]                                                             <L 217>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_cell_nodes, var_0, var_13, adj_cell_nodes, adj_0, adj_13, adj_14);
        // adj: n3 = cell_nodes[c, 3]                                                             <L 216>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_cell_nodes, var_0, var_9, adj_cell_nodes, adj_0, adj_9, adj_10);
        // adj: n2 = cell_nodes[c, 2]                                                             <L 215>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_cell_nodes, var_0, var_5, adj_cell_nodes, adj_0, adj_5, adj_6);
        // adj: n1 = cell_nodes[c, 1]                                                             <L 214>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_cell_nodes, var_0, var_1, adj_cell_nodes, adj_0, adj_1, adj_2);
        // adj: n0 = cell_nodes[c, 0]                                                             <L 213>
        // adj: c = wp.tid()                                                                      <L 212>
        // adj: def update_cell_render_state_and_aabbs_kernel(                                    <L 198>
        continue;
    }
}



extern "C" __global__ void pick_ray_cell_t_kernel_5605843f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max,
    wp::vec_t<3, wp::float32> var_ray_origin,
    wp::vec_t<3, wp::float32> var_ray_dir,
    wp::array_t<wp::float32> var_cell_ray_t,
    wp::array_t<wp::float32> var_min_t)
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
        const wp::float32 var_5 = 1e+30;
        wp::float32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32 var_15;
        const wp::float32 var_16 = 1e+30;
        wp::float32 var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        const wp::float32 var_24 = -1e+30;
        wp::float32 var_25;
        const wp::float32 var_26 = 1e+30;
        wp::float32 var_27;
        const wp::int32 var_28 = 0;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 1e-08;
        bool var_35;
        bool var_36;
        bool var_37;
        bool var_38;
        const wp::float32 var_39 = 1e+30;
        wp::float32 var_40;
        const wp::float32 var_41 = 1.0;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 1e-08;
        bool var_60;
        bool var_61;
        bool var_62;
        bool var_63;
        const wp::float32 var_64 = 1e+30;
        wp::float32 var_65;
        const wp::float32 var_66 = 1.0;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 2;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::float32 var_89 = 1e-08;
        bool var_90;
        bool var_91;
        bool var_92;
        bool var_93;
        const wp::float32 var_94 = 1e+30;
        wp::float32 var_95;
        const wp::float32 var_96 = 1.0;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
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
        const wp::float32 var_113 = 0.0;
        wp::float32 var_114;
        bool var_115;
        const wp::float32 var_116 = 1e+30;
        wp::float32 var_117;
        const wp::int32 var_118 = 0;
        wp::float32 var_119;
        //---------
        // forward
        // def pick_ray_cell_t_kernel(                                                            <L 340>
        // c = wp.tid()                                                                           <L 353>
        var_0 = builtin_tid1d();
        // if cell_active[c] == 0:                                                                <L 354>
        var_1 = wp::address(var_cell_active, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // cell_ray_t[c] = float(1.0e30)                                                      <L 355>
            var_6 = wp::float(var_5);
            wp::array_store(var_cell_ray_t, var_0, var_6);
            // return                                                                             <L 356>
            continue;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 357>
        var_9 = (var_has_material_filter != var_8);
        var_7 = var_9;
        if (var_7) {
            var_10 = wp::address(var_cell_material, var_0);
            var_12 = wp::load(var_10);
            var_11 = wp::address(var_material_cuttable, var_12);
            var_15 = wp::load(var_11);
            var_14 = (var_15 == var_13);
            var_7 = var_7 && var_14;
        }
        if (var_7) {
            // cell_ray_t[c] = float(1.0e30)                                                      <L 358>
            var_17 = wp::float(var_16);
            wp::array_store(var_cell_ray_t, var_0, var_17);
            // return                                                                             <L 359>
            continue;
        }
        // lo = aabb_min[c]                                                                       <L 361>
        var_18 = wp::address(var_aabb_min, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // hi = aabb_max[c]                                                                       <L 362>
        var_21 = wp::address(var_aabb_max, var_0);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // t_near = float(-1.0e30)                                                                <L 363>
        var_25 = wp::float(var_24);
        // t_far = float(1.0e30)                                                                  <L 364>
        var_27 = wp::float(var_26);
        // for axis in range(3):                                                                  <L 368>
        // d = ray_dir[axis]                                                                      <L 369>
        var_29 = wp::extract(var_ray_dir, var_28);
        // o = ray_origin[axis]                                                                   <L 370>
        var_30 = wp::extract(var_ray_origin, var_28);
        // amin = lo[axis]                                                                        <L 371>
        var_31 = wp::extract(var_19, var_28);
        // amax = hi[axis]                                                                        <L 372>
        var_32 = wp::extract(var_22, var_28);
        // if wp.abs(d) < 1.0e-8:                                                                 <L 373>
        var_33 = wp::abs(var_29);
        var_35 = (var_33 < var_34);
        if (var_35) {
            // if o < amin or o > amax:                                                           <L 374>
            var_37 = (var_30 < var_31);
            var_36 = var_37;
            if (!var_36) {
                var_38 = (var_30 > var_32);
                var_36 = var_36 || var_38;
            }
            if (var_36) {
                // cell_ray_t[c] = float(1.0e30)                                                  <L 375>
                var_40 = wp::float(var_39);
                wp::array_store(var_cell_ray_t, var_0, var_40);
                // return                                                                         <L 376>
                continue;
            }
        }
        if (!var_35) {
            // inv_d = 1.0 / d                                                                    <L 378>
            var_42 = wp::div(var_41, var_29);
            // t0 = (amin - o) * inv_d                                                            <L 379>
            var_43 = wp::sub(var_31, var_30);
            var_44 = wp::mul(var_43, var_42);
            // t1 = (amax - o) * inv_d                                                            <L 380>
            var_45 = wp::sub(var_32, var_30);
            var_46 = wp::mul(var_45, var_42);
            // axis_near = wp.min(t0, t1)                                                         <L 381>
            var_47 = wp::min(var_44, var_46);
            // axis_far = wp.max(t0, t1)                                                          <L 382>
            var_48 = wp::max(var_44, var_46);
            // t_near = wp.max(t_near, axis_near)                                                 <L 383>
            var_49 = wp::max(var_25, var_47);
            // t_far = wp.min(t_far, axis_far)                                                    <L 384>
            var_50 = wp::min(var_27, var_48);
        }
        var_51 = wp::where(var_35, var_25, var_49);
        var_52 = wp::where(var_35, var_27, var_50);
        // d = ray_dir[axis]                                                                      <L 369>
        var_54 = wp::extract(var_ray_dir, var_53);
        // o = ray_origin[axis]                                                                   <L 370>
        var_55 = wp::extract(var_ray_origin, var_53);
        // amin = lo[axis]                                                                        <L 371>
        var_56 = wp::extract(var_19, var_53);
        // amax = hi[axis]                                                                        <L 372>
        var_57 = wp::extract(var_22, var_53);
        // if wp.abs(d) < 1.0e-8:                                                                 <L 373>
        var_58 = wp::abs(var_54);
        var_60 = (var_58 < var_59);
        if (var_60) {
            // if o < amin or o > amax:                                                           <L 374>
            var_62 = (var_55 < var_56);
            var_61 = var_62;
            if (!var_61) {
                var_63 = (var_55 > var_57);
                var_61 = var_61 || var_63;
            }
            if (var_61) {
                // cell_ray_t[c] = float(1.0e30)                                                  <L 375>
                var_65 = wp::float(var_64);
                wp::array_store(var_cell_ray_t, var_0, var_65);
                // return                                                                         <L 376>
                continue;
            }
        }
        if (!var_60) {
            // inv_d = 1.0 / d                                                                    <L 378>
            var_67 = wp::div(var_66, var_54);
            // t0 = (amin - o) * inv_d                                                            <L 379>
            var_68 = wp::sub(var_56, var_55);
            var_69 = wp::mul(var_68, var_67);
            // t1 = (amax - o) * inv_d                                                            <L 380>
            var_70 = wp::sub(var_57, var_55);
            var_71 = wp::mul(var_70, var_67);
            // axis_near = wp.min(t0, t1)                                                         <L 381>
            var_72 = wp::min(var_69, var_71);
            // axis_far = wp.max(t0, t1)                                                          <L 382>
            var_73 = wp::max(var_69, var_71);
            // t_near = wp.max(t_near, axis_near)                                                 <L 383>
            var_74 = wp::max(var_51, var_72);
            // t_far = wp.min(t_far, axis_far)                                                    <L 384>
            var_75 = wp::min(var_52, var_73);
        }
        var_76 = wp::where(var_60, var_51, var_74);
        var_77 = wp::where(var_60, var_52, var_75);
        var_78 = wp::where(var_60, var_42, var_67);
        var_79 = wp::where(var_60, var_44, var_69);
        var_80 = wp::where(var_60, var_46, var_71);
        var_81 = wp::where(var_60, var_47, var_72);
        var_82 = wp::where(var_60, var_48, var_73);
        // d = ray_dir[axis]                                                                      <L 369>
        var_84 = wp::extract(var_ray_dir, var_83);
        // o = ray_origin[axis]                                                                   <L 370>
        var_85 = wp::extract(var_ray_origin, var_83);
        // amin = lo[axis]                                                                        <L 371>
        var_86 = wp::extract(var_19, var_83);
        // amax = hi[axis]                                                                        <L 372>
        var_87 = wp::extract(var_22, var_83);
        // if wp.abs(d) < 1.0e-8:                                                                 <L 373>
        var_88 = wp::abs(var_84);
        var_90 = (var_88 < var_89);
        if (var_90) {
            // if o < amin or o > amax:                                                           <L 374>
            var_92 = (var_85 < var_86);
            var_91 = var_92;
            if (!var_91) {
                var_93 = (var_85 > var_87);
                var_91 = var_91 || var_93;
            }
            if (var_91) {
                // cell_ray_t[c] = float(1.0e30)                                                  <L 375>
                var_95 = wp::float(var_94);
                wp::array_store(var_cell_ray_t, var_0, var_95);
                // return                                                                         <L 376>
                continue;
            }
        }
        if (!var_90) {
            // inv_d = 1.0 / d                                                                    <L 378>
            var_97 = wp::div(var_96, var_84);
            // t0 = (amin - o) * inv_d                                                            <L 379>
            var_98 = wp::sub(var_86, var_85);
            var_99 = wp::mul(var_98, var_97);
            // t1 = (amax - o) * inv_d                                                            <L 380>
            var_100 = wp::sub(var_87, var_85);
            var_101 = wp::mul(var_100, var_97);
            // axis_near = wp.min(t0, t1)                                                         <L 381>
            var_102 = wp::min(var_99, var_101);
            // axis_far = wp.max(t0, t1)                                                          <L 382>
            var_103 = wp::max(var_99, var_101);
            // t_near = wp.max(t_near, axis_near)                                                 <L 383>
            var_104 = wp::max(var_76, var_102);
            // t_far = wp.min(t_far, axis_far)                                                    <L 384>
            var_105 = wp::min(var_77, var_103);
        }
        var_106 = wp::where(var_90, var_76, var_104);
        var_107 = wp::where(var_90, var_77, var_105);
        var_108 = wp::where(var_90, var_78, var_97);
        var_109 = wp::where(var_90, var_79, var_99);
        var_110 = wp::where(var_90, var_80, var_101);
        var_111 = wp::where(var_90, var_81, var_102);
        var_112 = wp::where(var_90, var_82, var_103);
        // t_hit = wp.max(t_near, 0.0)                                                            <L 386>
        var_114 = wp::max(var_106, var_113);
        // if t_far < t_hit:                                                                      <L 387>
        var_115 = (var_107 < var_114);
        if (var_115) {
            // cell_ray_t[c] = float(1.0e30)                                                      <L 388>
            var_117 = wp::float(var_116);
            wp::array_store(var_cell_ray_t, var_0, var_117);
            // return                                                                             <L 389>
            continue;
        }
        // cell_ray_t[c] = t_hit                                                                  <L 391>
        wp::array_store(var_cell_ray_t, var_0, var_114);
        // wp.atomic_min(min_t, 0, t_hit)                                                         <L 392>
        var_119 = wp::atomic_min(var_min_t, var_118, var_114);
    }
}



extern "C" __global__ void pick_ray_cell_t_kernel_5605843f_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::int32> var_cell_material,
    wp::array_t<wp::int32> var_material_cuttable,
    wp::int32 var_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> var_aabb_max,
    wp::vec_t<3, wp::float32> var_ray_origin,
    wp::vec_t<3, wp::float32> var_ray_dir,
    wp::array_t<wp::float32> var_cell_ray_t,
    wp::array_t<wp::float32> var_min_t,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::int32> adj_cell_material,
    wp::array_t<wp::int32> adj_material_cuttable,
    wp::int32 adj_has_material_filter,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_min,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_aabb_max,
    wp::vec_t<3, wp::float32> adj_ray_origin,
    wp::vec_t<3, wp::float32> adj_ray_dir,
    wp::array_t<wp::float32> adj_cell_ray_t,
    wp::array_t<wp::float32> adj_min_t)
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
        const wp::float32 var_5 = 1e+30;
        wp::float32 var_6;
        bool var_7;
        const wp::int32 var_8 = 0;
        bool var_9;
        wp::int32* var_10;
        wp::int32* var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        bool var_14;
        wp::int32 var_15;
        const wp::float32 var_16 = 1e+30;
        wp::float32 var_17;
        wp::vec_t<3, wp::float32>* var_18;
        wp::vec_t<3, wp::float32> var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        const wp::float32 var_24 = -1e+30;
        wp::float32 var_25;
        const wp::float32 var_26 = 1e+30;
        wp::float32 var_27;
        const wp::int32 var_28 = 0;
        wp::float32 var_29;
        wp::float32 var_30;
        wp::float32 var_31;
        wp::float32 var_32;
        wp::float32 var_33;
        const wp::float32 var_34 = 1e-08;
        bool var_35;
        bool var_36;
        bool var_37;
        bool var_38;
        const wp::float32 var_39 = 1e+30;
        wp::float32 var_40;
        const wp::float32 var_41 = 1.0;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        wp::float32 var_47;
        wp::float32 var_48;
        wp::float32 var_49;
        wp::float32 var_50;
        wp::float32 var_51;
        wp::float32 var_52;
        const wp::int32 var_53 = 1;
        wp::float32 var_54;
        wp::float32 var_55;
        wp::float32 var_56;
        wp::float32 var_57;
        wp::float32 var_58;
        const wp::float32 var_59 = 1e-08;
        bool var_60;
        bool var_61;
        bool var_62;
        bool var_63;
        const wp::float32 var_64 = 1e+30;
        wp::float32 var_65;
        const wp::float32 var_66 = 1.0;
        wp::float32 var_67;
        wp::float32 var_68;
        wp::float32 var_69;
        wp::float32 var_70;
        wp::float32 var_71;
        wp::float32 var_72;
        wp::float32 var_73;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::float32 var_76;
        wp::float32 var_77;
        wp::float32 var_78;
        wp::float32 var_79;
        wp::float32 var_80;
        wp::float32 var_81;
        wp::float32 var_82;
        const wp::int32 var_83 = 2;
        wp::float32 var_84;
        wp::float32 var_85;
        wp::float32 var_86;
        wp::float32 var_87;
        wp::float32 var_88;
        const wp::float32 var_89 = 1e-08;
        bool var_90;
        bool var_91;
        bool var_92;
        bool var_93;
        const wp::float32 var_94 = 1e+30;
        wp::float32 var_95;
        const wp::float32 var_96 = 1.0;
        wp::float32 var_97;
        wp::float32 var_98;
        wp::float32 var_99;
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
        const wp::float32 var_113 = 0.0;
        wp::float32 var_114;
        bool var_115;
        const wp::float32 var_116 = 1e+30;
        wp::float32 var_117;
        const wp::int32 var_118 = 0;
        wp::float32 var_119;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        bool adj_3 = {};
        wp::int32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        bool adj_7 = {};
        wp::int32 adj_8 = {};
        bool adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        bool adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::vec_t<3, wp::float32> adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::float32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::float32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        bool adj_35 = {};
        bool adj_36 = {};
        bool adj_37 = {};
        bool adj_38 = {};
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
        wp::int32 adj_53 = {};
        wp::float32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::float32 adj_56 = {};
        wp::float32 adj_57 = {};
        wp::float32 adj_58 = {};
        wp::float32 adj_59 = {};
        bool adj_60 = {};
        bool adj_61 = {};
        bool adj_62 = {};
        bool adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::float32 adj_67 = {};
        wp::float32 adj_68 = {};
        wp::float32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::float32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::float32 adj_76 = {};
        wp::float32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::float32 adj_79 = {};
        wp::float32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::float32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::float32 adj_85 = {};
        wp::float32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::float32 adj_89 = {};
        bool adj_90 = {};
        bool adj_91 = {};
        bool adj_92 = {};
        bool adj_93 = {};
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
        wp::int32 adj_118 = {};
        wp::float32 adj_119 = {};
        //---------
        // forward
        // def pick_ray_cell_t_kernel(                                                            <L 340>
        // c = wp.tid()                                                                           <L 353>
        var_0 = builtin_tid1d();
        // if cell_active[c] == 0:                                                                <L 354>
        var_1 = wp::address(var_cell_active, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // cell_ray_t[c] = float(1.0e30)                                                      <L 355>
            var_6 = wp::float(var_5);
            // wp::array_store(var_cell_ray_t, var_0, var_6);
            // return                                                                             <L 356>
            goto label0;
        }
        // if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:              <L 357>
        var_9 = (var_has_material_filter != var_8);
        var_7 = var_9;
        if (var_7) {
            var_10 = wp::address(var_cell_material, var_0);
            var_12 = wp::load(var_10);
            var_11 = wp::address(var_material_cuttable, var_12);
            var_15 = wp::load(var_11);
            var_14 = (var_15 == var_13);
            var_7 = var_7 && var_14;
        }
        if (var_7) {
            // cell_ray_t[c] = float(1.0e30)                                                      <L 358>
            var_17 = wp::float(var_16);
            // wp::array_store(var_cell_ray_t, var_0, var_17);
            // return                                                                             <L 359>
            goto label1;
        }
        // lo = aabb_min[c]                                                                       <L 361>
        var_18 = wp::address(var_aabb_min, var_0);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // hi = aabb_max[c]                                                                       <L 362>
        var_21 = wp::address(var_aabb_max, var_0);
        var_23 = wp::load(var_21);
        var_22 = wp::copy(var_23);
        // t_near = float(-1.0e30)                                                                <L 363>
        var_25 = wp::float(var_24);
        // t_far = float(1.0e30)                                                                  <L 364>
        var_27 = wp::float(var_26);
        // for axis in range(3):                                                                  <L 368>
        // d = ray_dir[axis]                                                                      <L 369>
        var_29 = wp::extract(var_ray_dir, var_28);
        // o = ray_origin[axis]                                                                   <L 370>
        var_30 = wp::extract(var_ray_origin, var_28);
        // amin = lo[axis]                                                                        <L 371>
        var_31 = wp::extract(var_19, var_28);
        // amax = hi[axis]                                                                        <L 372>
        var_32 = wp::extract(var_22, var_28);
        // if wp.abs(d) < 1.0e-8:                                                                 <L 373>
        var_33 = wp::abs(var_29);
        var_35 = (var_33 < var_34);
        if (var_35) {
            // if o < amin or o > amax:                                                           <L 374>
            var_37 = (var_30 < var_31);
            var_36 = var_37;
            if (!var_36) {
                var_38 = (var_30 > var_32);
                var_36 = var_36 || var_38;
            }
            if (var_36) {
                // cell_ray_t[c] = float(1.0e30)                                                  <L 375>
                var_40 = wp::float(var_39);
                // wp::array_store(var_cell_ray_t, var_0, var_40);
                // return                                                                         <L 376>
                goto label2;
            }
        }
        if (!var_35) {
            // inv_d = 1.0 / d                                                                    <L 378>
            var_42 = wp::div(var_41, var_29);
            // t0 = (amin - o) * inv_d                                                            <L 379>
            var_43 = wp::sub(var_31, var_30);
            var_44 = wp::mul(var_43, var_42);
            // t1 = (amax - o) * inv_d                                                            <L 380>
            var_45 = wp::sub(var_32, var_30);
            var_46 = wp::mul(var_45, var_42);
            // axis_near = wp.min(t0, t1)                                                         <L 381>
            var_47 = wp::min(var_44, var_46);
            // axis_far = wp.max(t0, t1)                                                          <L 382>
            var_48 = wp::max(var_44, var_46);
            // t_near = wp.max(t_near, axis_near)                                                 <L 383>
            var_49 = wp::max(var_25, var_47);
            // t_far = wp.min(t_far, axis_far)                                                    <L 384>
            var_50 = wp::min(var_27, var_48);
        }
        var_51 = wp::where(var_35, var_25, var_49);
        var_52 = wp::where(var_35, var_27, var_50);
        // d = ray_dir[axis]                                                                      <L 369>
        var_54 = wp::extract(var_ray_dir, var_53);
        // o = ray_origin[axis]                                                                   <L 370>
        var_55 = wp::extract(var_ray_origin, var_53);
        // amin = lo[axis]                                                                        <L 371>
        var_56 = wp::extract(var_19, var_53);
        // amax = hi[axis]                                                                        <L 372>
        var_57 = wp::extract(var_22, var_53);
        // if wp.abs(d) < 1.0e-8:                                                                 <L 373>
        var_58 = wp::abs(var_54);
        var_60 = (var_58 < var_59);
        if (var_60) {
            // if o < amin or o > amax:                                                           <L 374>
            var_62 = (var_55 < var_56);
            var_61 = var_62;
            if (!var_61) {
                var_63 = (var_55 > var_57);
                var_61 = var_61 || var_63;
            }
            if (var_61) {
                // cell_ray_t[c] = float(1.0e30)                                                  <L 375>
                var_65 = wp::float(var_64);
                // wp::array_store(var_cell_ray_t, var_0, var_65);
                // return                                                                         <L 376>
                goto label3;
            }
        }
        if (!var_60) {
            // inv_d = 1.0 / d                                                                    <L 378>
            var_67 = wp::div(var_66, var_54);
            // t0 = (amin - o) * inv_d                                                            <L 379>
            var_68 = wp::sub(var_56, var_55);
            var_69 = wp::mul(var_68, var_67);
            // t1 = (amax - o) * inv_d                                                            <L 380>
            var_70 = wp::sub(var_57, var_55);
            var_71 = wp::mul(var_70, var_67);
            // axis_near = wp.min(t0, t1)                                                         <L 381>
            var_72 = wp::min(var_69, var_71);
            // axis_far = wp.max(t0, t1)                                                          <L 382>
            var_73 = wp::max(var_69, var_71);
            // t_near = wp.max(t_near, axis_near)                                                 <L 383>
            var_74 = wp::max(var_51, var_72);
            // t_far = wp.min(t_far, axis_far)                                                    <L 384>
            var_75 = wp::min(var_52, var_73);
        }
        var_76 = wp::where(var_60, var_51, var_74);
        var_77 = wp::where(var_60, var_52, var_75);
        var_78 = wp::where(var_60, var_42, var_67);
        var_79 = wp::where(var_60, var_44, var_69);
        var_80 = wp::where(var_60, var_46, var_71);
        var_81 = wp::where(var_60, var_47, var_72);
        var_82 = wp::where(var_60, var_48, var_73);
        // d = ray_dir[axis]                                                                      <L 369>
        var_84 = wp::extract(var_ray_dir, var_83);
        // o = ray_origin[axis]                                                                   <L 370>
        var_85 = wp::extract(var_ray_origin, var_83);
        // amin = lo[axis]                                                                        <L 371>
        var_86 = wp::extract(var_19, var_83);
        // amax = hi[axis]                                                                        <L 372>
        var_87 = wp::extract(var_22, var_83);
        // if wp.abs(d) < 1.0e-8:                                                                 <L 373>
        var_88 = wp::abs(var_84);
        var_90 = (var_88 < var_89);
        if (var_90) {
            // if o < amin or o > amax:                                                           <L 374>
            var_92 = (var_85 < var_86);
            var_91 = var_92;
            if (!var_91) {
                var_93 = (var_85 > var_87);
                var_91 = var_91 || var_93;
            }
            if (var_91) {
                // cell_ray_t[c] = float(1.0e30)                                                  <L 375>
                var_95 = wp::float(var_94);
                // wp::array_store(var_cell_ray_t, var_0, var_95);
                // return                                                                         <L 376>
                goto label4;
            }
        }
        if (!var_90) {
            // inv_d = 1.0 / d                                                                    <L 378>
            var_97 = wp::div(var_96, var_84);
            // t0 = (amin - o) * inv_d                                                            <L 379>
            var_98 = wp::sub(var_86, var_85);
            var_99 = wp::mul(var_98, var_97);
            // t1 = (amax - o) * inv_d                                                            <L 380>
            var_100 = wp::sub(var_87, var_85);
            var_101 = wp::mul(var_100, var_97);
            // axis_near = wp.min(t0, t1)                                                         <L 381>
            var_102 = wp::min(var_99, var_101);
            // axis_far = wp.max(t0, t1)                                                          <L 382>
            var_103 = wp::max(var_99, var_101);
            // t_near = wp.max(t_near, axis_near)                                                 <L 383>
            var_104 = wp::max(var_76, var_102);
            // t_far = wp.min(t_far, axis_far)                                                    <L 384>
            var_105 = wp::min(var_77, var_103);
        }
        var_106 = wp::where(var_90, var_76, var_104);
        var_107 = wp::where(var_90, var_77, var_105);
        var_108 = wp::where(var_90, var_78, var_97);
        var_109 = wp::where(var_90, var_79, var_99);
        var_110 = wp::where(var_90, var_80, var_101);
        var_111 = wp::where(var_90, var_81, var_102);
        var_112 = wp::where(var_90, var_82, var_103);
        // t_hit = wp.max(t_near, 0.0)                                                            <L 386>
        var_114 = wp::max(var_106, var_113);
        // if t_far < t_hit:                                                                      <L 387>
        var_115 = (var_107 < var_114);
        if (var_115) {
            // cell_ray_t[c] = float(1.0e30)                                                      <L 388>
            var_117 = wp::float(var_116);
            // wp::array_store(var_cell_ray_t, var_0, var_117);
            // return                                                                             <L 389>
            goto label5;
        }
        // cell_ray_t[c] = t_hit                                                                  <L 391>
        // wp::array_store(var_cell_ray_t, var_0, var_114);
        // wp.atomic_min(min_t, 0, t_hit)                                                         <L 392>
        // var_119 = wp::atomic_min(var_min_t, var_118, var_114);
        //---------
        // reverse
        wp::adj_atomic_min(var_min_t, var_118, var_114, adj_min_t, adj_118, adj_114, adj_119);
        // adj: wp.atomic_min(min_t, 0, t_hit)                                                    <L 392>
        wp::adj_array_store(var_cell_ray_t, var_0, var_114, adj_cell_ray_t, adj_0, adj_114);
        // adj: cell_ray_t[c] = t_hit                                                             <L 391>
        if (var_115) {
            label5:;
            // adj: return                                                                        <L 389>
            wp::adj_array_store(var_cell_ray_t, var_0, var_117, adj_cell_ray_t, adj_0, adj_117);
            wp::adj_float(var_116, adj_116, adj_117);
            // adj: cell_ray_t[c] = float(1.0e30)                                                 <L 388>
        }
        // adj: if t_far < t_hit:                                                                 <L 387>
        wp::adj_max(var_106, var_113, adj_106, adj_113, adj_114);
        // adj: t_hit = wp.max(t_near, 0.0)                                                       <L 386>
        wp::adj_where(var_90, var_82, var_103, adj_90, adj_82, adj_103, adj_112);
        wp::adj_where(var_90, var_81, var_102, adj_90, adj_81, adj_102, adj_111);
        wp::adj_where(var_90, var_80, var_101, adj_90, adj_80, adj_101, adj_110);
        wp::adj_where(var_90, var_79, var_99, adj_90, adj_79, adj_99, adj_109);
        wp::adj_where(var_90, var_78, var_97, adj_90, adj_78, adj_97, adj_108);
        wp::adj_where(var_90, var_77, var_105, adj_90, adj_77, adj_105, adj_107);
        wp::adj_where(var_90, var_76, var_104, adj_90, adj_76, adj_104, adj_106);
        if (!var_90) {
            wp::adj_min(var_77, var_103, adj_77, adj_103, adj_105);
            // adj: t_far = wp.min(t_far, axis_far)                                               <L 384>
            wp::adj_max(var_76, var_102, adj_76, adj_102, adj_104);
            // adj: t_near = wp.max(t_near, axis_near)                                            <L 383>
            wp::adj_max(var_99, var_101, adj_99, adj_101, adj_103);
            // adj: axis_far = wp.max(t0, t1)                                                     <L 382>
            wp::adj_min(var_99, var_101, adj_99, adj_101, adj_102);
            // adj: axis_near = wp.min(t0, t1)                                                    <L 381>
            wp::adj_mul(var_100, var_97, adj_100, adj_97, adj_101);
            wp::adj_sub(var_87, var_85, adj_87, adj_85, adj_100);
            // adj: t1 = (amax - o) * inv_d                                                       <L 380>
            wp::adj_mul(var_98, var_97, adj_98, adj_97, adj_99);
            wp::adj_sub(var_86, var_85, adj_86, adj_85, adj_98);
            // adj: t0 = (amin - o) * inv_d                                                       <L 379>
            wp::adj_div(var_96, var_84, var_97, adj_96, adj_84, adj_97);
            // adj: inv_d = 1.0 / d                                                               <L 378>
        }
        if (var_90) {
            if (var_91) {
                label4:;
                // adj: return                                                                    <L 376>
                wp::adj_array_store(var_cell_ray_t, var_0, var_95, adj_cell_ray_t, adj_0, adj_95);
                wp::adj_float(var_94, adj_94, adj_95);
                // adj: cell_ray_t[c] = float(1.0e30)                                             <L 375>
            }
            if (!var_91) {
            }
            // adj: if o < amin or o > amax:                                                      <L 374>
        }
        wp::adj_abs(var_84, adj_84, adj_88);
        // adj: if wp.abs(d) < 1.0e-8:                                                            <L 373>
        wp::adj_extract(var_22, var_83, adj_22, adj_83, adj_87);
        // adj: amax = hi[axis]                                                                   <L 372>
        wp::adj_extract(var_19, var_83, adj_19, adj_83, adj_86);
        // adj: amin = lo[axis]                                                                   <L 371>
        wp::adj_extract(var_ray_origin, var_83, adj_ray_origin, adj_83, adj_85);
        // adj: o = ray_origin[axis]                                                              <L 370>
        wp::adj_extract(var_ray_dir, var_83, adj_ray_dir, adj_83, adj_84);
        // adj: d = ray_dir[axis]                                                                 <L 369>
        wp::adj_where(var_60, var_48, var_73, adj_60, adj_48, adj_73, adj_82);
        wp::adj_where(var_60, var_47, var_72, adj_60, adj_47, adj_72, adj_81);
        wp::adj_where(var_60, var_46, var_71, adj_60, adj_46, adj_71, adj_80);
        wp::adj_where(var_60, var_44, var_69, adj_60, adj_44, adj_69, adj_79);
        wp::adj_where(var_60, var_42, var_67, adj_60, adj_42, adj_67, adj_78);
        wp::adj_where(var_60, var_52, var_75, adj_60, adj_52, adj_75, adj_77);
        wp::adj_where(var_60, var_51, var_74, adj_60, adj_51, adj_74, adj_76);
        if (!var_60) {
            wp::adj_min(var_52, var_73, adj_52, adj_73, adj_75);
            // adj: t_far = wp.min(t_far, axis_far)                                               <L 384>
            wp::adj_max(var_51, var_72, adj_51, adj_72, adj_74);
            // adj: t_near = wp.max(t_near, axis_near)                                            <L 383>
            wp::adj_max(var_69, var_71, adj_69, adj_71, adj_73);
            // adj: axis_far = wp.max(t0, t1)                                                     <L 382>
            wp::adj_min(var_69, var_71, adj_69, adj_71, adj_72);
            // adj: axis_near = wp.min(t0, t1)                                                    <L 381>
            wp::adj_mul(var_70, var_67, adj_70, adj_67, adj_71);
            wp::adj_sub(var_57, var_55, adj_57, adj_55, adj_70);
            // adj: t1 = (amax - o) * inv_d                                                       <L 380>
            wp::adj_mul(var_68, var_67, adj_68, adj_67, adj_69);
            wp::adj_sub(var_56, var_55, adj_56, adj_55, adj_68);
            // adj: t0 = (amin - o) * inv_d                                                       <L 379>
            wp::adj_div(var_66, var_54, var_67, adj_66, adj_54, adj_67);
            // adj: inv_d = 1.0 / d                                                               <L 378>
        }
        if (var_60) {
            if (var_61) {
                label3:;
                // adj: return                                                                    <L 376>
                wp::adj_array_store(var_cell_ray_t, var_0, var_65, adj_cell_ray_t, adj_0, adj_65);
                wp::adj_float(var_64, adj_64, adj_65);
                // adj: cell_ray_t[c] = float(1.0e30)                                             <L 375>
            }
            if (!var_61) {
            }
            // adj: if o < amin or o > amax:                                                      <L 374>
        }
        wp::adj_abs(var_54, adj_54, adj_58);
        // adj: if wp.abs(d) < 1.0e-8:                                                            <L 373>
        wp::adj_extract(var_22, var_53, adj_22, adj_53, adj_57);
        // adj: amax = hi[axis]                                                                   <L 372>
        wp::adj_extract(var_19, var_53, adj_19, adj_53, adj_56);
        // adj: amin = lo[axis]                                                                   <L 371>
        wp::adj_extract(var_ray_origin, var_53, adj_ray_origin, adj_53, adj_55);
        // adj: o = ray_origin[axis]                                                              <L 370>
        wp::adj_extract(var_ray_dir, var_53, adj_ray_dir, adj_53, adj_54);
        // adj: d = ray_dir[axis]                                                                 <L 369>
        wp::adj_where(var_35, var_27, var_50, adj_35, adj_27, adj_50, adj_52);
        wp::adj_where(var_35, var_25, var_49, adj_35, adj_25, adj_49, adj_51);
        if (!var_35) {
            wp::adj_min(var_27, var_48, adj_27, adj_48, adj_50);
            // adj: t_far = wp.min(t_far, axis_far)                                               <L 384>
            wp::adj_max(var_25, var_47, adj_25, adj_47, adj_49);
            // adj: t_near = wp.max(t_near, axis_near)                                            <L 383>
            wp::adj_max(var_44, var_46, adj_44, adj_46, adj_48);
            // adj: axis_far = wp.max(t0, t1)                                                     <L 382>
            wp::adj_min(var_44, var_46, adj_44, adj_46, adj_47);
            // adj: axis_near = wp.min(t0, t1)                                                    <L 381>
            wp::adj_mul(var_45, var_42, adj_45, adj_42, adj_46);
            wp::adj_sub(var_32, var_30, adj_32, adj_30, adj_45);
            // adj: t1 = (amax - o) * inv_d                                                       <L 380>
            wp::adj_mul(var_43, var_42, adj_43, adj_42, adj_44);
            wp::adj_sub(var_31, var_30, adj_31, adj_30, adj_43);
            // adj: t0 = (amin - o) * inv_d                                                       <L 379>
            wp::adj_div(var_41, var_29, var_42, adj_41, adj_29, adj_42);
            // adj: inv_d = 1.0 / d                                                               <L 378>
        }
        if (var_35) {
            if (var_36) {
                label2:;
                // adj: return                                                                    <L 376>
                wp::adj_array_store(var_cell_ray_t, var_0, var_40, adj_cell_ray_t, adj_0, adj_40);
                wp::adj_float(var_39, adj_39, adj_40);
                // adj: cell_ray_t[c] = float(1.0e30)                                             <L 375>
            }
            if (!var_36) {
            }
            // adj: if o < amin or o > amax:                                                      <L 374>
        }
        wp::adj_abs(var_29, adj_29, adj_33);
        // adj: if wp.abs(d) < 1.0e-8:                                                            <L 373>
        wp::adj_extract(var_22, var_28, adj_22, adj_28, adj_32);
        // adj: amax = hi[axis]                                                                   <L 372>
        wp::adj_extract(var_19, var_28, adj_19, adj_28, adj_31);
        // adj: amin = lo[axis]                                                                   <L 371>
        wp::adj_extract(var_ray_origin, var_28, adj_ray_origin, adj_28, adj_30);
        // adj: o = ray_origin[axis]                                                              <L 370>
        wp::adj_extract(var_ray_dir, var_28, adj_ray_dir, adj_28, adj_29);
        // adj: d = ray_dir[axis]                                                                 <L 369>
        // adj: for axis in range(3):                                                             <L 368>
        wp::adj_float(var_26, adj_26, adj_27);
        // adj: t_far = float(1.0e30)                                                             <L 364>
        wp::adj_float(var_24, adj_24, adj_25);
        // adj: t_near = float(-1.0e30)                                                           <L 363>
        wp::adj_copy(var_23, adj_21, adj_22);
        wp::adj_address(var_aabb_max, var_0, adj_aabb_max, adj_0, adj_21);
        // adj: hi = aabb_max[c]                                                                  <L 362>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_aabb_min, var_0, adj_aabb_min, adj_0, adj_18);
        // adj: lo = aabb_min[c]                                                                  <L 361>
        if (var_7) {
            label1:;
            // adj: return                                                                        <L 359>
            wp::adj_array_store(var_cell_ray_t, var_0, var_17, adj_cell_ray_t, adj_0, adj_17);
            wp::adj_float(var_16, adj_16, adj_17);
            // adj: cell_ray_t[c] = float(1.0e30)                                                 <L 358>
        }
        if (var_7) {
            wp::adj_address(var_material_cuttable, var_12, adj_material_cuttable, adj_10, adj_11);
            wp::adj_address(var_cell_material, var_0, adj_cell_material, adj_0, adj_10);
        }
        // adj: if has_material_filter != 0 and material_cuttable[cell_material[c]] == 0:         <L 357>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 356>
            wp::adj_array_store(var_cell_ray_t, var_0, var_6, adj_cell_ray_t, adj_0, adj_6);
            wp::adj_float(var_5, adj_5, adj_6);
            // adj: cell_ray_t[c] = float(1.0e30)                                                 <L 355>
        }
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_1);
        // adj: if cell_active[c] == 0:                                                           <L 354>
        // adj: c = wp.tid()                                                                      <L 353>
        // adj: def pick_ray_cell_t_kernel(                                                       <L 340>
        continue;
    }
}



extern "C" __global__ void pick_ray_cell_match_kernel_660cd973_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_cell_ray_t,
    wp::array_t<wp::float32> var_min_t,
    wp::float32 var_sentinel,
    wp::array_t<wp::int32> var_hit_cell)
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
        wp::float32* var_2;
        bool var_3;
        wp::float32 var_4;
        wp::float32* var_5;
        const wp::int32 var_6 = 0;
        wp::float32* var_7;
        bool var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        //---------
        // forward
        // def pick_ray_cell_match_kernel(                                                        <L 396>
        // c = wp.tid()                                                                           <L 403>
        var_0 = builtin_tid1d();
        // if min_t[0] >= sentinel:                                                               <L 404>
        var_2 = wp::address(var_min_t, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_4 >= var_sentinel);
        if (var_3) {
            // return                                                                             <L 405>
            continue;
        }
        // if cell_ray_t[c] == min_t[0]:                                                          <L 406>
        var_5 = wp::address(var_cell_ray_t, var_0);
        var_7 = wp::address(var_min_t, var_6);
        var_9 = wp::load(var_5);
        var_10 = wp::load(var_7);
        var_8 = (var_9 == var_10);
        if (var_8) {
            // wp.atomic_min(hit_cell, 0, c)                                                      <L 407>
            var_12 = wp::atomic_min(var_hit_cell, var_11, var_0);
        }
    }
}



extern "C" __global__ void pick_ray_cell_match_kernel_660cd973_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_cell_ray_t,
    wp::array_t<wp::float32> var_min_t,
    wp::float32 var_sentinel,
    wp::array_t<wp::int32> var_hit_cell,
    wp::array_t<wp::float32> adj_cell_ray_t,
    wp::array_t<wp::float32> adj_min_t,
    wp::float32 adj_sentinel,
    wp::array_t<wp::int32> adj_hit_cell)
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
        wp::float32* var_2;
        bool var_3;
        wp::float32 var_4;
        wp::float32* var_5;
        const wp::int32 var_6 = 0;
        wp::float32* var_7;
        bool var_8;
        wp::float32 var_9;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        wp::int32 var_12;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::float32 adj_2 = {};
        bool adj_3 = {};
        wp::float32 adj_4 = {};
        wp::float32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        bool adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        //---------
        // forward
        // def pick_ray_cell_match_kernel(                                                        <L 396>
        // c = wp.tid()                                                                           <L 403>
        var_0 = builtin_tid1d();
        // if min_t[0] >= sentinel:                                                               <L 404>
        var_2 = wp::address(var_min_t, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_4 >= var_sentinel);
        if (var_3) {
            // return                                                                             <L 405>
            goto label0;
        }
        // if cell_ray_t[c] == min_t[0]:                                                          <L 406>
        var_5 = wp::address(var_cell_ray_t, var_0);
        var_7 = wp::address(var_min_t, var_6);
        var_9 = wp::load(var_5);
        var_10 = wp::load(var_7);
        var_8 = (var_9 == var_10);
        if (var_8) {
            // wp.atomic_min(hit_cell, 0, c)                                                      <L 407>
            // var_12 = wp::atomic_min(var_hit_cell, var_11, var_0);
        }
        //---------
        // reverse
        if (var_8) {
            wp::adj_atomic_min(var_hit_cell, var_11, var_0, adj_hit_cell, adj_11, adj_0, adj_12);
            // adj: wp.atomic_min(hit_cell, 0, c)                                                 <L 407>
        }
        wp::adj_address(var_min_t, var_6, adj_min_t, adj_6, adj_7);
        wp::adj_address(var_cell_ray_t, var_0, adj_cell_ray_t, adj_0, adj_5);
        // adj: if cell_ray_t[c] == min_t[0]:                                                     <L 406>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 405>
        }
        wp::adj_address(var_min_t, var_1, adj_min_t, adj_1, adj_2);
        // adj: if min_t[0] >= sentinel:                                                          <L 404>
        // adj: c = wp.tid()                                                                      <L 403>
        // adj: def pick_ray_cell_match_kernel(                                                   <L 396>
        continue;
    }
}



extern "C" __global__ void update_cell_render_state_kernel_dc3acac2_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::float32 var_voxel_size,
    wp::int32 var_compute_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cell_center_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell_orientation,
    wp::array_t<wp::int32> var_cell_render_flags,
    wp::array_t<wp::float32> var_cell_stretch)
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
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 4;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 5;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 6;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 7;
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
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::float32 var_64 = 1.0;
        const wp::float32 var_65 = 8.0;
        wp::float32 var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        const wp::float32 var_75 = 0.25;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        const wp::float32 var_84 = 0.25;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        const wp::float32 var_93 = 0.25;
        wp::vec_t<3, wp::float32> var_94;
        wp::mat_t<3, 3, wp::float32> var_95;
        wp::int32* var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 1;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        bool var_104;
        wp::float32 var_105;
        const wp::int32 var_106 = 0;
        wp::int32 var_107;
        const wp::int32 var_108 = 0;
        bool var_109;
        const wp::float32 var_110 = 0.0;
        //---------
        // forward
        // def update_cell_render_state_kernel(                                                   <L 118>
        // c = wp.tid()                                                                           <L 143>
        var_0 = builtin_tid1d();
        // n0 = cell_nodes[c, 0]                                                                  <L 144>
        var_2 = wp::address(var_cell_nodes, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // n1 = cell_nodes[c, 1]                                                                  <L 145>
        var_6 = wp::address(var_cell_nodes, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // n2 = cell_nodes[c, 2]                                                                  <L 146>
        var_10 = wp::address(var_cell_nodes, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // n3 = cell_nodes[c, 3]                                                                  <L 147>
        var_14 = wp::address(var_cell_nodes, var_0, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // n4 = cell_nodes[c, 4]                                                                  <L 148>
        var_18 = wp::address(var_cell_nodes, var_0, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // n5 = cell_nodes[c, 5]                                                                  <L 149>
        var_22 = wp::address(var_cell_nodes, var_0, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // n6 = cell_nodes[c, 6]                                                                  <L 150>
        var_26 = wp::address(var_cell_nodes, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // n7 = cell_nodes[c, 7]                                                                  <L 151>
        var_30 = wp::address(var_cell_nodes, var_0, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // p0 = particle_q[n0]                                                                    <L 153>
        var_33 = wp::address(var_particle_q, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // p1 = particle_q[n1]                                                                    <L 154>
        var_36 = wp::address(var_particle_q, var_7);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // p2 = particle_q[n2]                                                                    <L 155>
        var_39 = wp::address(var_particle_q, var_11);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // p3 = particle_q[n3]                                                                    <L 156>
        var_42 = wp::address(var_particle_q, var_15);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // p4 = particle_q[n4]                                                                    <L 157>
        var_45 = wp::address(var_particle_q, var_19);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // p5 = particle_q[n5]                                                                    <L 158>
        var_48 = wp::address(var_particle_q, var_23);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // p6 = particle_q[n6]                                                                    <L 159>
        var_51 = wp::address(var_particle_q, var_27);
        var_53 = wp::load(var_51);
        var_52 = wp::copy(var_53);
        // p7 = particle_q[n7]                                                                    <L 160>
        var_54 = wp::address(var_particle_q, var_31);
        var_56 = wp::load(var_54);
        var_55 = wp::copy(var_56);
        // centre = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)                         <L 162>
        var_57 = wp::add(var_34, var_37);
        var_58 = wp::add(var_57, var_40);
        var_59 = wp::add(var_58, var_43);
        var_60 = wp::add(var_59, var_46);
        var_61 = wp::add(var_60, var_49);
        var_62 = wp::add(var_61, var_52);
        var_63 = wp::add(var_62, var_55);
        var_66 = wp::div(var_64, var_65);
        var_67 = wp::mul(var_63, var_66);
        // cell_center_q[c] = centre                                                              <L 163>
        wp::array_store(var_cell_center_q, var_0, var_67);
        // ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                            <L 165>
        var_68 = wp::sub(var_37, var_34);
        var_69 = wp::sub(var_40, var_43);
        var_70 = wp::add(var_68, var_69);
        var_71 = wp::sub(var_49, var_46);
        var_72 = wp::add(var_70, var_71);
        var_73 = wp::sub(var_52, var_55);
        var_74 = wp::add(var_72, var_73);
        var_76 = wp::mul(var_74, var_75);
        // ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                            <L 166>
        var_77 = wp::sub(var_43, var_34);
        var_78 = wp::sub(var_40, var_37);
        var_79 = wp::add(var_77, var_78);
        var_80 = wp::sub(var_55, var_46);
        var_81 = wp::add(var_79, var_80);
        var_82 = wp::sub(var_52, var_49);
        var_83 = wp::add(var_81, var_82);
        var_85 = wp::mul(var_83, var_84);
        // az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                            <L 167>
        var_86 = wp::sub(var_46, var_34);
        var_87 = wp::sub(var_49, var_37);
        var_88 = wp::add(var_86, var_87);
        var_89 = wp::sub(var_52, var_40);
        var_90 = wp::add(var_88, var_89);
        var_91 = wp::sub(var_55, var_43);
        var_92 = wp::add(var_90, var_91);
        var_94 = wp::mul(var_92, var_93);
        // cell_orientation[c] = _orthonormalize(ax, ay, az)                                      <L 169>
        var_95 = _orthonormalize_0(var_76, var_85, var_94);
        wp::array_store(var_cell_orientation, var_0, var_95);
        // if cell_active[c] != 0:                                                                <L 171>
        var_96 = wp::address(var_cell_active, var_0);
        var_99 = wp::load(var_96);
        var_98 = (var_99 != var_97);
        if (var_98) {
            // cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)                              <L 172>
            var_102 = wp::int32(var_101);
            wp::array_store(var_cell_render_flags, var_0, var_102);
            // if compute_stretch != 0:                                                           <L 173>
            var_104 = (var_compute_stretch != var_103);
            if (var_104) {
                // cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)       <L 174>
                var_105 = _cell_max_abs_strain_0(var_34, var_37, var_40, var_43, var_46, var_49, var_52, var_55, var_voxel_size);
                wp::array_store(var_cell_stretch, var_0, var_105);
            }
        }
        if (!var_98) {
            // cell_render_flags[c] = wp.int32(0)                                                 <L 176>
            var_107 = wp::int32(var_106);
            wp::array_store(var_cell_render_flags, var_0, var_107);
            // if compute_stretch != 0:                                                           <L 177>
            var_109 = (var_compute_stretch != var_108);
            if (var_109) {
                // cell_stretch[c] = 0.0                                                          <L 178>
                wp::array_store(var_cell_stretch, var_0, var_110);
            }
        }
    }
}



extern "C" __global__ void update_cell_render_state_kernel_dc3acac2_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_cell_nodes,
    wp::array_t<wp::int32> var_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::float32 var_voxel_size,
    wp::int32 var_compute_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> var_cell_center_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_cell_orientation,
    wp::array_t<wp::int32> var_cell_render_flags,
    wp::array_t<wp::float32> var_cell_stretch,
    wp::array_t<wp::int32> adj_cell_nodes,
    wp::array_t<wp::int32> adj_cell_active,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::float32 adj_voxel_size,
    wp::int32 adj_compute_stretch,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_cell_center_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_cell_orientation,
    wp::array_t<wp::int32> adj_cell_render_flags,
    wp::array_t<wp::float32> adj_cell_stretch)
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
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::int32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32* var_14;
        wp::int32 var_15;
        wp::int32 var_16;
        const wp::int32 var_17 = 4;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 5;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 6;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 7;
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
        wp::vec_t<3, wp::float32>* var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32> var_44;
        wp::vec_t<3, wp::float32>* var_45;
        wp::vec_t<3, wp::float32> var_46;
        wp::vec_t<3, wp::float32> var_47;
        wp::vec_t<3, wp::float32>* var_48;
        wp::vec_t<3, wp::float32> var_49;
        wp::vec_t<3, wp::float32> var_50;
        wp::vec_t<3, wp::float32>* var_51;
        wp::vec_t<3, wp::float32> var_52;
        wp::vec_t<3, wp::float32> var_53;
        wp::vec_t<3, wp::float32>* var_54;
        wp::vec_t<3, wp::float32> var_55;
        wp::vec_t<3, wp::float32> var_56;
        wp::vec_t<3, wp::float32> var_57;
        wp::vec_t<3, wp::float32> var_58;
        wp::vec_t<3, wp::float32> var_59;
        wp::vec_t<3, wp::float32> var_60;
        wp::vec_t<3, wp::float32> var_61;
        wp::vec_t<3, wp::float32> var_62;
        wp::vec_t<3, wp::float32> var_63;
        const wp::float32 var_64 = 1.0;
        const wp::float32 var_65 = 8.0;
        wp::float32 var_66;
        wp::vec_t<3, wp::float32> var_67;
        wp::vec_t<3, wp::float32> var_68;
        wp::vec_t<3, wp::float32> var_69;
        wp::vec_t<3, wp::float32> var_70;
        wp::vec_t<3, wp::float32> var_71;
        wp::vec_t<3, wp::float32> var_72;
        wp::vec_t<3, wp::float32> var_73;
        wp::vec_t<3, wp::float32> var_74;
        const wp::float32 var_75 = 0.25;
        wp::vec_t<3, wp::float32> var_76;
        wp::vec_t<3, wp::float32> var_77;
        wp::vec_t<3, wp::float32> var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        wp::vec_t<3, wp::float32> var_82;
        wp::vec_t<3, wp::float32> var_83;
        const wp::float32 var_84 = 0.25;
        wp::vec_t<3, wp::float32> var_85;
        wp::vec_t<3, wp::float32> var_86;
        wp::vec_t<3, wp::float32> var_87;
        wp::vec_t<3, wp::float32> var_88;
        wp::vec_t<3, wp::float32> var_89;
        wp::vec_t<3, wp::float32> var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        const wp::float32 var_93 = 0.25;
        wp::vec_t<3, wp::float32> var_94;
        wp::mat_t<3, 3, wp::float32> var_95;
        wp::int32* var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 1;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        bool var_104;
        wp::float32 var_105;
        const wp::int32 var_106 = 0;
        wp::int32 var_107;
        const wp::int32 var_108 = 0;
        bool var_109;
        const wp::float32 var_110 = 0.0;
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
        wp::vec_t<3, wp::float32> adj_42 = {};
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
        wp::vec_t<3, wp::float32> adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        wp::vec_t<3, wp::float32> adj_55 = {};
        wp::vec_t<3, wp::float32> adj_56 = {};
        wp::vec_t<3, wp::float32> adj_57 = {};
        wp::vec_t<3, wp::float32> adj_58 = {};
        wp::vec_t<3, wp::float32> adj_59 = {};
        wp::vec_t<3, wp::float32> adj_60 = {};
        wp::vec_t<3, wp::float32> adj_61 = {};
        wp::vec_t<3, wp::float32> adj_62 = {};
        wp::vec_t<3, wp::float32> adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::vec_t<3, wp::float32> adj_67 = {};
        wp::vec_t<3, wp::float32> adj_68 = {};
        wp::vec_t<3, wp::float32> adj_69 = {};
        wp::vec_t<3, wp::float32> adj_70 = {};
        wp::vec_t<3, wp::float32> adj_71 = {};
        wp::vec_t<3, wp::float32> adj_72 = {};
        wp::vec_t<3, wp::float32> adj_73 = {};
        wp::vec_t<3, wp::float32> adj_74 = {};
        wp::float32 adj_75 = {};
        wp::vec_t<3, wp::float32> adj_76 = {};
        wp::vec_t<3, wp::float32> adj_77 = {};
        wp::vec_t<3, wp::float32> adj_78 = {};
        wp::vec_t<3, wp::float32> adj_79 = {};
        wp::vec_t<3, wp::float32> adj_80 = {};
        wp::vec_t<3, wp::float32> adj_81 = {};
        wp::vec_t<3, wp::float32> adj_82 = {};
        wp::vec_t<3, wp::float32> adj_83 = {};
        wp::float32 adj_84 = {};
        wp::vec_t<3, wp::float32> adj_85 = {};
        wp::vec_t<3, wp::float32> adj_86 = {};
        wp::vec_t<3, wp::float32> adj_87 = {};
        wp::vec_t<3, wp::float32> adj_88 = {};
        wp::vec_t<3, wp::float32> adj_89 = {};
        wp::vec_t<3, wp::float32> adj_90 = {};
        wp::vec_t<3, wp::float32> adj_91 = {};
        wp::vec_t<3, wp::float32> adj_92 = {};
        wp::float32 adj_93 = {};
        wp::vec_t<3, wp::float32> adj_94 = {};
        wp::mat_t<3, 3, wp::float32> adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        bool adj_98 = {};
        wp::int32 adj_99 = {};
        wp::int32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        bool adj_104 = {};
        wp::float32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::int32 adj_108 = {};
        bool adj_109 = {};
        wp::float32 adj_110 = {};
        //---------
        // forward
        // def update_cell_render_state_kernel(                                                   <L 118>
        // c = wp.tid()                                                                           <L 143>
        var_0 = builtin_tid1d();
        // n0 = cell_nodes[c, 0]                                                                  <L 144>
        var_2 = wp::address(var_cell_nodes, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // n1 = cell_nodes[c, 1]                                                                  <L 145>
        var_6 = wp::address(var_cell_nodes, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // n2 = cell_nodes[c, 2]                                                                  <L 146>
        var_10 = wp::address(var_cell_nodes, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // n3 = cell_nodes[c, 3]                                                                  <L 147>
        var_14 = wp::address(var_cell_nodes, var_0, var_13);
        var_16 = wp::load(var_14);
        var_15 = wp::copy(var_16);
        // n4 = cell_nodes[c, 4]                                                                  <L 148>
        var_18 = wp::address(var_cell_nodes, var_0, var_17);
        var_20 = wp::load(var_18);
        var_19 = wp::copy(var_20);
        // n5 = cell_nodes[c, 5]                                                                  <L 149>
        var_22 = wp::address(var_cell_nodes, var_0, var_21);
        var_24 = wp::load(var_22);
        var_23 = wp::copy(var_24);
        // n6 = cell_nodes[c, 6]                                                                  <L 150>
        var_26 = wp::address(var_cell_nodes, var_0, var_25);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // n7 = cell_nodes[c, 7]                                                                  <L 151>
        var_30 = wp::address(var_cell_nodes, var_0, var_29);
        var_32 = wp::load(var_30);
        var_31 = wp::copy(var_32);
        // p0 = particle_q[n0]                                                                    <L 153>
        var_33 = wp::address(var_particle_q, var_3);
        var_35 = wp::load(var_33);
        var_34 = wp::copy(var_35);
        // p1 = particle_q[n1]                                                                    <L 154>
        var_36 = wp::address(var_particle_q, var_7);
        var_38 = wp::load(var_36);
        var_37 = wp::copy(var_38);
        // p2 = particle_q[n2]                                                                    <L 155>
        var_39 = wp::address(var_particle_q, var_11);
        var_41 = wp::load(var_39);
        var_40 = wp::copy(var_41);
        // p3 = particle_q[n3]                                                                    <L 156>
        var_42 = wp::address(var_particle_q, var_15);
        var_44 = wp::load(var_42);
        var_43 = wp::copy(var_44);
        // p4 = particle_q[n4]                                                                    <L 157>
        var_45 = wp::address(var_particle_q, var_19);
        var_47 = wp::load(var_45);
        var_46 = wp::copy(var_47);
        // p5 = particle_q[n5]                                                                    <L 158>
        var_48 = wp::address(var_particle_q, var_23);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // p6 = particle_q[n6]                                                                    <L 159>
        var_51 = wp::address(var_particle_q, var_27);
        var_53 = wp::load(var_51);
        var_52 = wp::copy(var_53);
        // p7 = particle_q[n7]                                                                    <L 160>
        var_54 = wp::address(var_particle_q, var_31);
        var_56 = wp::load(var_54);
        var_55 = wp::copy(var_56);
        // centre = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)                         <L 162>
        var_57 = wp::add(var_34, var_37);
        var_58 = wp::add(var_57, var_40);
        var_59 = wp::add(var_58, var_43);
        var_60 = wp::add(var_59, var_46);
        var_61 = wp::add(var_60, var_49);
        var_62 = wp::add(var_61, var_52);
        var_63 = wp::add(var_62, var_55);
        var_66 = wp::div(var_64, var_65);
        var_67 = wp::mul(var_63, var_66);
        // cell_center_q[c] = centre                                                              <L 163>
        // wp::array_store(var_cell_center_q, var_0, var_67);
        // ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                            <L 165>
        var_68 = wp::sub(var_37, var_34);
        var_69 = wp::sub(var_40, var_43);
        var_70 = wp::add(var_68, var_69);
        var_71 = wp::sub(var_49, var_46);
        var_72 = wp::add(var_70, var_71);
        var_73 = wp::sub(var_52, var_55);
        var_74 = wp::add(var_72, var_73);
        var_76 = wp::mul(var_74, var_75);
        // ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                            <L 166>
        var_77 = wp::sub(var_43, var_34);
        var_78 = wp::sub(var_40, var_37);
        var_79 = wp::add(var_77, var_78);
        var_80 = wp::sub(var_55, var_46);
        var_81 = wp::add(var_79, var_80);
        var_82 = wp::sub(var_52, var_49);
        var_83 = wp::add(var_81, var_82);
        var_85 = wp::mul(var_83, var_84);
        // az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                            <L 167>
        var_86 = wp::sub(var_46, var_34);
        var_87 = wp::sub(var_49, var_37);
        var_88 = wp::add(var_86, var_87);
        var_89 = wp::sub(var_52, var_40);
        var_90 = wp::add(var_88, var_89);
        var_91 = wp::sub(var_55, var_43);
        var_92 = wp::add(var_90, var_91);
        var_94 = wp::mul(var_92, var_93);
        // cell_orientation[c] = _orthonormalize(ax, ay, az)                                      <L 169>
        var_95 = _orthonormalize_0(var_76, var_85, var_94);
        // wp::array_store(var_cell_orientation, var_0, var_95);
        // if cell_active[c] != 0:                                                                <L 171>
        var_96 = wp::address(var_cell_active, var_0);
        var_99 = wp::load(var_96);
        var_98 = (var_99 != var_97);
        if (var_98) {
            // cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)                              <L 172>
            var_102 = wp::int32(var_101);
            // wp::array_store(var_cell_render_flags, var_0, var_102);
            // if compute_stretch != 0:                                                           <L 173>
            var_104 = (var_compute_stretch != var_103);
            if (var_104) {
                // cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)       <L 174>
                var_105 = _cell_max_abs_strain_0(var_34, var_37, var_40, var_43, var_46, var_49, var_52, var_55, var_voxel_size);
                // wp::array_store(var_cell_stretch, var_0, var_105);
            }
        }
        if (!var_98) {
            // cell_render_flags[c] = wp.int32(0)                                                 <L 176>
            var_107 = wp::int32(var_106);
            // wp::array_store(var_cell_render_flags, var_0, var_107);
            // if compute_stretch != 0:                                                           <L 177>
            var_109 = (var_compute_stretch != var_108);
            if (var_109) {
                // cell_stretch[c] = 0.0                                                          <L 178>
                // wp::array_store(var_cell_stretch, var_0, var_110);
            }
        }
        //---------
        // reverse
        if (!var_98) {
            if (var_109) {
                wp::adj_array_store(var_cell_stretch, var_0, var_110, adj_cell_stretch, adj_0, adj_110);
                // adj: cell_stretch[c] = 0.0                                                     <L 178>
            }
            // adj: if compute_stretch != 0:                                                      <L 177>
            wp::adj_array_store(var_cell_render_flags, var_0, var_107, adj_cell_render_flags, adj_0, adj_107);
            wp::adj_int32(var_106, adj_106, adj_107);
            // adj: cell_render_flags[c] = wp.int32(0)                                            <L 176>
        }
        if (var_98) {
            if (var_104) {
                wp::adj_array_store(var_cell_stretch, var_0, var_105, adj_cell_stretch, adj_0, adj_105);
                adj__cell_max_abs_strain_0(var_34, var_37, var_40, var_43, var_46, var_49, var_52, var_55, var_voxel_size, adj_34, adj_37, adj_40, adj_43, adj_46, adj_49, adj_52, adj_55, adj_voxel_size, adj_105);
                // adj: cell_stretch[c] = _cell_max_abs_strain(p0, p1, p2, p3, p4, p5, p6, p7, voxel_size)  <L 174>
            }
            // adj: if compute_stretch != 0:                                                      <L 173>
            wp::adj_array_store(var_cell_render_flags, var_0, var_102, adj_cell_render_flags, adj_0, adj_102);
            wp::adj_int32(var_101, adj_101, adj_102);
            // adj: cell_render_flags[c] = wp.int32(ParticleFlags.ACTIVE)                         <L 172>
        }
        wp::adj_address(var_cell_active, var_0, adj_cell_active, adj_0, adj_96);
        // adj: if cell_active[c] != 0:                                                           <L 171>
        wp::adj_array_store(var_cell_orientation, var_0, var_95, adj_cell_orientation, adj_0, adj_95);
        adj__orthonormalize_0(var_76, var_85, var_94, adj_76, adj_85, adj_94, adj_95);
        // adj: cell_orientation[c] = _orthonormalize(ax, ay, az)                                 <L 169>
        wp::adj_mul(var_92, var_93, adj_92, adj_93, adj_94);
        wp::adj_add(var_90, var_91, adj_90, adj_91, adj_92);
        wp::adj_sub(var_55, var_43, adj_55, adj_43, adj_91);
        wp::adj_add(var_88, var_89, adj_88, adj_89, adj_90);
        wp::adj_sub(var_52, var_40, adj_52, adj_40, adj_89);
        wp::adj_add(var_86, var_87, adj_86, adj_87, adj_88);
        wp::adj_sub(var_49, var_37, adj_49, adj_37, adj_87);
        wp::adj_sub(var_46, var_34, adj_46, adj_34, adj_86);
        // adj: az = ((p4 - p0) + (p5 - p1) + (p6 - p2) + (p7 - p3)) * 0.25                       <L 167>
        wp::adj_mul(var_83, var_84, adj_83, adj_84, adj_85);
        wp::adj_add(var_81, var_82, adj_81, adj_82, adj_83);
        wp::adj_sub(var_52, var_49, adj_52, adj_49, adj_82);
        wp::adj_add(var_79, var_80, adj_79, adj_80, adj_81);
        wp::adj_sub(var_55, var_46, adj_55, adj_46, adj_80);
        wp::adj_add(var_77, var_78, adj_77, adj_78, adj_79);
        wp::adj_sub(var_40, var_37, adj_40, adj_37, adj_78);
        wp::adj_sub(var_43, var_34, adj_43, adj_34, adj_77);
        // adj: ay = ((p3 - p0) + (p2 - p1) + (p7 - p4) + (p6 - p5)) * 0.25                       <L 166>
        wp::adj_mul(var_74, var_75, adj_74, adj_75, adj_76);
        wp::adj_add(var_72, var_73, adj_72, adj_73, adj_74);
        wp::adj_sub(var_52, var_55, adj_52, adj_55, adj_73);
        wp::adj_add(var_70, var_71, adj_70, adj_71, adj_72);
        wp::adj_sub(var_49, var_46, adj_49, adj_46, adj_71);
        wp::adj_add(var_68, var_69, adj_68, adj_69, adj_70);
        wp::adj_sub(var_40, var_43, adj_40, adj_43, adj_69);
        wp::adj_sub(var_37, var_34, adj_37, adj_34, adj_68);
        // adj: ax = ((p1 - p0) + (p2 - p3) + (p5 - p4) + (p6 - p7)) * 0.25                       <L 165>
        wp::adj_array_store(var_cell_center_q, var_0, var_67, adj_cell_center_q, adj_0, adj_67);
        // adj: cell_center_q[c] = centre                                                         <L 163>
        wp::adj_mul(var_63, var_66, adj_63, adj_66, adj_67);
        wp::adj_div(var_64, var_65, var_66, adj_64, adj_65, adj_66);
        wp::adj_add(var_62, var_55, adj_62, adj_55, adj_63);
        wp::adj_add(var_61, var_52, adj_61, adj_52, adj_62);
        wp::adj_add(var_60, var_49, adj_60, adj_49, adj_61);
        wp::adj_add(var_59, var_46, adj_59, adj_46, adj_60);
        wp::adj_add(var_58, var_43, adj_58, adj_43, adj_59);
        wp::adj_add(var_57, var_40, adj_57, adj_40, adj_58);
        wp::adj_add(var_34, var_37, adj_34, adj_37, adj_57);
        // adj: centre = (p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7) * (1.0 / 8.0)                    <L 162>
        wp::adj_copy(var_56, adj_54, adj_55);
        wp::adj_address(var_particle_q, var_31, adj_particle_q, adj_31, adj_54);
        // adj: p7 = particle_q[n7]                                                               <L 160>
        wp::adj_copy(var_53, adj_51, adj_52);
        wp::adj_address(var_particle_q, var_27, adj_particle_q, adj_27, adj_51);
        // adj: p6 = particle_q[n6]                                                               <L 159>
        wp::adj_copy(var_50, adj_48, adj_49);
        wp::adj_address(var_particle_q, var_23, adj_particle_q, adj_23, adj_48);
        // adj: p5 = particle_q[n5]                                                               <L 158>
        wp::adj_copy(var_47, adj_45, adj_46);
        wp::adj_address(var_particle_q, var_19, adj_particle_q, adj_19, adj_45);
        // adj: p4 = particle_q[n4]                                                               <L 157>
        wp::adj_copy(var_44, adj_42, adj_43);
        wp::adj_address(var_particle_q, var_15, adj_particle_q, adj_15, adj_42);
        // adj: p3 = particle_q[n3]                                                               <L 156>
        wp::adj_copy(var_41, adj_39, adj_40);
        wp::adj_address(var_particle_q, var_11, adj_particle_q, adj_11, adj_39);
        // adj: p2 = particle_q[n2]                                                               <L 155>
        wp::adj_copy(var_38, adj_36, adj_37);
        wp::adj_address(var_particle_q, var_7, adj_particle_q, adj_7, adj_36);
        // adj: p1 = particle_q[n1]                                                               <L 154>
        wp::adj_copy(var_35, adj_33, adj_34);
        wp::adj_address(var_particle_q, var_3, adj_particle_q, adj_3, adj_33);
        // adj: p0 = particle_q[n0]                                                               <L 153>
        wp::adj_copy(var_32, adj_30, adj_31);
        wp::adj_address(var_cell_nodes, var_0, var_29, adj_cell_nodes, adj_0, adj_29, adj_30);
        // adj: n7 = cell_nodes[c, 7]                                                             <L 151>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_cell_nodes, var_0, var_25, adj_cell_nodes, adj_0, adj_25, adj_26);
        // adj: n6 = cell_nodes[c, 6]                                                             <L 150>
        wp::adj_copy(var_24, adj_22, adj_23);
        wp::adj_address(var_cell_nodes, var_0, var_21, adj_cell_nodes, adj_0, adj_21, adj_22);
        // adj: n5 = cell_nodes[c, 5]                                                             <L 149>
        wp::adj_copy(var_20, adj_18, adj_19);
        wp::adj_address(var_cell_nodes, var_0, var_17, adj_cell_nodes, adj_0, adj_17, adj_18);
        // adj: n4 = cell_nodes[c, 4]                                                             <L 148>
        wp::adj_copy(var_16, adj_14, adj_15);
        wp::adj_address(var_cell_nodes, var_0, var_13, adj_cell_nodes, adj_0, adj_13, adj_14);
        // adj: n3 = cell_nodes[c, 3]                                                             <L 147>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_cell_nodes, var_0, var_9, adj_cell_nodes, adj_0, adj_9, adj_10);
        // adj: n2 = cell_nodes[c, 2]                                                             <L 146>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_cell_nodes, var_0, var_5, adj_cell_nodes, adj_0, adj_5, adj_6);
        // adj: n1 = cell_nodes[c, 1]                                                             <L 145>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_cell_nodes, var_0, var_1, adj_cell_nodes, adj_0, adj_1, adj_2);
        // adj: n0 = cell_nodes[c, 0]                                                             <L 144>
        // adj: c = wp.tid()                                                                      <L 143>
        // adj: def update_cell_render_state_kernel(                                              <L 118>
        continue;
    }
}

