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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/heat.py:33
static CUDA_CALLABLE wp::float32 _point_segment_distance_sq_2(
    wp::vec_t<3, wp::float32> var_p,
    wp::vec_t<3, wp::float32> var_a,
    wp::vec_t<3, wp::float32> var_b)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1e-12;
    bool var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 1.0;
    bool var_13;
    const wp::float32 var_14 = 1.0;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    //---------
    // forward
    // def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:           <L 34>
    // ab = b - a                                                                             <L 36>
    var_0 = wp::sub(var_b, var_a);
    // ap = p - a                                                                             <L 37>
    var_1 = wp::sub(var_p, var_a);
    // denom = wp.dot(ab, ab)                                                                 <L 38>
    var_2 = wp::dot(var_0, var_0);
    // if denom < 1.0e-12:                                                                    <L 39>
    var_4 = (var_2 < var_3);
    if (var_4) {
        // return wp.dot(ap, ap)                                                              <L 40>
        var_5 = wp::dot(var_1, var_1);
        return var_5;
    }
    // t = wp.dot(ap, ab) / denom                                                             <L 41>
    var_6 = wp::dot(var_1, var_0);
    var_7 = wp::div(var_6, var_2);
    // if t < 0.0:                                                                            <L 42>
    var_9 = (var_7 < var_8);
    if (var_9) {
        // t = 0.0                                                                            <L 43>
    }
    var_11 = wp::where(var_9, var_10, var_7);
    if (!var_9) {
        // elif t > 1.0:                                                                      <L 44>
        var_13 = (var_11 > var_12);
        if (var_13) {
            // t = 1.0                                                                        <L 45>
        }
        var_15 = wp::where(var_13, var_14, var_11);
    }
    var_16 = wp::where(var_9, var_11, var_15);
    // closest = a + ab * t                                                                   <L 46>
    var_17 = wp::mul(var_0, var_16);
    var_18 = wp::add(var_a, var_17);
    // d = p - closest                                                                        <L 47>
    var_19 = wp::sub(var_p, var_18);
    // return wp.dot(d, d)                                                                    <L 48>
    var_20 = wp::dot(var_19, var_19);
    return var_20;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/heat.py:33
static CUDA_CALLABLE void adj__point_segment_distance_sq_2(
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
    wp::vec_t<3, wp::float32> var_1;
    wp::float32 var_2;
    const wp::float32 var_3 = 1e-12;
    bool var_4;
    wp::float32 var_5;
    wp::float32 var_6;
    wp::float32 var_7;
    const wp::float32 var_8 = 0.0;
    bool var_9;
    const wp::float32 var_10 = 0.0;
    wp::float32 var_11;
    const wp::float32 var_12 = 1.0;
    bool var_13;
    const wp::float32 var_14 = 1.0;
    wp::float32 var_15;
    wp::float32 var_16;
    wp::vec_t<3, wp::float32> var_17;
    wp::vec_t<3, wp::float32> var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::float32 var_20;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    bool adj_4 = {};
    wp::float32 adj_5 = {};
    wp::float32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::float32 adj_8 = {};
    bool adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::float32 adj_12 = {};
    bool adj_13 = {};
    wp::float32 adj_14 = {};
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::vec_t<3, wp::float32> adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::float32 adj_20 = {};
    //---------
    // forward
    // def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:           <L 34>
    // ab = b - a                                                                             <L 36>
    var_0 = wp::sub(var_b, var_a);
    // ap = p - a                                                                             <L 37>
    var_1 = wp::sub(var_p, var_a);
    // denom = wp.dot(ab, ab)                                                                 <L 38>
    var_2 = wp::dot(var_0, var_0);
    // if denom < 1.0e-12:                                                                    <L 39>
    var_4 = (var_2 < var_3);
    if (var_4) {
        // return wp.dot(ap, ap)                                                              <L 40>
        var_5 = wp::dot(var_1, var_1);
        goto label0;
    }
    // t = wp.dot(ap, ab) / denom                                                             <L 41>
    var_6 = wp::dot(var_1, var_0);
    var_7 = wp::div(var_6, var_2);
    // if t < 0.0:                                                                            <L 42>
    var_9 = (var_7 < var_8);
    if (var_9) {
        // t = 0.0                                                                            <L 43>
    }
    var_11 = wp::where(var_9, var_10, var_7);
    if (!var_9) {
        // elif t > 1.0:                                                                      <L 44>
        var_13 = (var_11 > var_12);
        if (var_13) {
            // t = 1.0                                                                        <L 45>
        }
        var_15 = wp::where(var_13, var_14, var_11);
    }
    var_16 = wp::where(var_9, var_11, var_15);
    // closest = a + ab * t                                                                   <L 46>
    var_17 = wp::mul(var_0, var_16);
    var_18 = wp::add(var_a, var_17);
    // d = p - closest                                                                        <L 47>
    var_19 = wp::sub(var_p, var_18);
    // return wp.dot(d, d)                                                                    <L 48>
    var_20 = wp::dot(var_19, var_19);
    goto label1;
    //---------
    // reverse
    label1:;
    adj_20 += adj_ret;
    wp::adj_dot(var_19, var_19, adj_19, adj_19, adj_20);
    // adj: return wp.dot(d, d)                                                               <L 48>
    wp::adj_sub(var_p, var_18, adj_p, adj_18, adj_19);
    // adj: d = p - closest                                                                   <L 47>
    wp::adj_add(var_a, var_17, adj_a, adj_17, adj_18);
    wp::adj_mul(var_0, var_16, adj_0, adj_16, adj_17);
    // adj: closest = a + ab * t                                                              <L 46>
    wp::adj_where(var_9, var_11, var_15, adj_9, adj_11, adj_15, adj_16);
    if (!var_9) {
        wp::adj_where(var_13, var_14, var_11, adj_13, adj_14, adj_11, adj_15);
        if (var_13) {
            // adj: t = 1.0                                                                   <L 45>
        }
        // adj: elif t > 1.0:                                                                 <L 44>
    }
    wp::adj_where(var_9, var_10, var_7, adj_9, adj_10, adj_7, adj_11);
    if (var_9) {
        // adj: t = 0.0                                                                       <L 43>
    }
    // adj: if t < 0.0:                                                                       <L 42>
    wp::adj_div(var_6, var_2, var_7, adj_6, adj_2, adj_7);
    wp::adj_dot(var_1, var_0, adj_1, adj_0, adj_6);
    // adj: t = wp.dot(ap, ab) / denom                                                        <L 41>
    if (var_4) {
        label0:;
        adj_5 += adj_ret;
        wp::adj_dot(var_1, var_1, adj_1, adj_1, adj_5);
        // adj: return wp.dot(ap, ap)                                                         <L 40>
    }
    // adj: if denom < 1.0e-12:                                                               <L 39>
    wp::adj_dot(var_0, var_0, adj_0, adj_0, adj_2);
    // adj: denom = wp.dot(ab, ab)                                                            <L 38>
    wp::adj_sub(var_p, var_a, adj_p, adj_a, adj_1);
    // adj: ap = p - a                                                                        <L 37>
    wp::adj_sub(var_b, var_a, adj_b, adj_a, adj_0);
    // adj: ab = b - a                                                                        <L 36>
    // adj: def _point_segment_distance_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:      <L 34>
    return;
}



extern "C" __global__ void apply_damage_kernel_7d9960f5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::float32> var_material_resistance,
    wp::array_t<wp::float32> var_particle_heat,
    wp::array_t<wp::float32> var_particle_burnt,
    wp::float32 var_fulguration,
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
        wp::int32* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 15.0;
        bool var_11;
        wp::float32* var_12;
        const wp::float32 var_13 = 0.04;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.05;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::float32 var_21 = 1.0;
        bool var_22;
        const wp::float32 var_23 = 1.0;
        wp::float32 var_24;
        wp::int32* var_25;
        wp::float32* var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        const wp::float32 var_31 = 0.0;
        bool var_32;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        //---------
        // forward
        // def apply_damage_kernel(                                                               <L 120>
        // i = wp.tid()                                                                           <L 134>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & _ACTIVE_BIT) == 0:                                             <L 135>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 136>
            continue;
        }
        // h = particle_heat[i]                                                                   <L 137>
        var_7 = wp::address(var_particle_heat, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if h > _BURN_HEAT_THRESHOLD:                                                           <L 139>
        var_11 = (var_8 > var_10);
        if (var_11) {
            // b = particle_burnt[i] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration       <L 140>
            var_12 = wp::address(var_particle_burnt, var_0);
            var_14 = wp::sub(var_8, var_10);
            var_15 = wp::mul(var_13, var_14);
            var_17 = wp::add(var_15, var_16);
            var_18 = wp::mul(var_17, var_fulguration);
            var_20 = wp::load(var_12);
            var_19 = wp::add(var_20, var_18);
            // if b > 1.0:                                                                        <L 141>
            var_22 = (var_19 > var_21);
            if (var_22) {
                // b = 1.0                                                                        <L 142>
            }
            var_24 = wp::where(var_22, var_23, var_19);
            // particle_burnt[i] = b                                                              <L 143>
            wp::array_store(var_particle_burnt, var_0, var_24);
        }
        // r = material_resistance[particle_material[i]]                                          <L 144>
        var_25 = wp::address(var_particle_material, var_0);
        var_27 = wp::load(var_25);
        var_26 = wp::address(var_material_resistance, var_27);
        var_29 = wp::load(var_26);
        var_28 = wp::copy(var_29);
        // if r > 0.0 and h > r:                                                                  <L 145>
        var_32 = (var_28 > var_31);
        var_30 = var_32;
        if (var_30) {
            var_33 = (var_8 > var_28);
            var_30 = var_30 && var_33;
        }
        if (var_30) {
            // particle_flags[i] = particle_flags[i] & (~_ACTIVE_BIT)                             <L 146>
            var_34 = wp::address(var_particle_flags, var_0);
            var_35 = wp::invert(var_2);
            var_37 = wp::load(var_34);
            var_36 = wp::bit_and(var_37, var_35);
            wp::array_store(var_particle_flags, var_0, var_36);
        }
    }
}



extern "C" __global__ void apply_damage_kernel_7d9960f5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::float32> var_material_resistance,
    wp::array_t<wp::float32> var_particle_heat,
    wp::array_t<wp::float32> var_particle_burnt,
    wp::float32 var_fulguration,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> adj_particle_material,
    wp::array_t<wp::float32> adj_material_resistance,
    wp::array_t<wp::float32> adj_particle_heat,
    wp::array_t<wp::float32> adj_particle_burnt,
    wp::float32 adj_fulguration,
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
        wp::int32* var_1;
        const wp::int32 var_2 = 1;
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        wp::float32* var_7;
        wp::float32 var_8;
        wp::float32 var_9;
        const wp::float32 var_10 = 15.0;
        bool var_11;
        wp::float32* var_12;
        const wp::float32 var_13 = 0.04;
        wp::float32 var_14;
        wp::float32 var_15;
        const wp::float32 var_16 = 0.05;
        wp::float32 var_17;
        wp::float32 var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        const wp::float32 var_21 = 1.0;
        bool var_22;
        const wp::float32 var_23 = 1.0;
        wp::float32 var_24;
        wp::int32* var_25;
        wp::float32* var_26;
        wp::int32 var_27;
        wp::float32 var_28;
        wp::float32 var_29;
        bool var_30;
        const wp::float32 var_31 = 0.0;
        bool var_32;
        bool var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
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
        bool adj_11 = {};
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
        bool adj_22 = {};
        wp::float32 adj_23 = {};
        wp::float32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        bool adj_30 = {};
        wp::float32 adj_31 = {};
        bool adj_32 = {};
        bool adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        //---------
        // forward
        // def apply_damage_kernel(                                                               <L 120>
        // i = wp.tid()                                                                           <L 134>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & _ACTIVE_BIT) == 0:                                             <L 135>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // return                                                                             <L 136>
            goto label0;
        }
        // h = particle_heat[i]                                                                   <L 137>
        var_7 = wp::address(var_particle_heat, var_0);
        var_9 = wp::load(var_7);
        var_8 = wp::copy(var_9);
        // if h > _BURN_HEAT_THRESHOLD:                                                           <L 139>
        var_11 = (var_8 > var_10);
        if (var_11) {
            // b = particle_burnt[i] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration       <L 140>
            var_12 = wp::address(var_particle_burnt, var_0);
            var_14 = wp::sub(var_8, var_10);
            var_15 = wp::mul(var_13, var_14);
            var_17 = wp::add(var_15, var_16);
            var_18 = wp::mul(var_17, var_fulguration);
            var_20 = wp::load(var_12);
            var_19 = wp::add(var_20, var_18);
            // if b > 1.0:                                                                        <L 141>
            var_22 = (var_19 > var_21);
            if (var_22) {
                // b = 1.0                                                                        <L 142>
            }
            var_24 = wp::where(var_22, var_23, var_19);
            // particle_burnt[i] = b                                                              <L 143>
            // wp::array_store(var_particle_burnt, var_0, var_24);
        }
        // r = material_resistance[particle_material[i]]                                          <L 144>
        var_25 = wp::address(var_particle_material, var_0);
        var_27 = wp::load(var_25);
        var_26 = wp::address(var_material_resistance, var_27);
        var_29 = wp::load(var_26);
        var_28 = wp::copy(var_29);
        // if r > 0.0 and h > r:                                                                  <L 145>
        var_32 = (var_28 > var_31);
        var_30 = var_32;
        if (var_30) {
            var_33 = (var_8 > var_28);
            var_30 = var_30 && var_33;
        }
        if (var_30) {
            // particle_flags[i] = particle_flags[i] & (~_ACTIVE_BIT)                             <L 146>
            var_34 = wp::address(var_particle_flags, var_0);
            var_35 = wp::invert(var_2);
            var_37 = wp::load(var_34);
            var_36 = wp::bit_and(var_37, var_35);
            // wp::array_store(var_particle_flags, var_0, var_36);
        }
        //---------
        // reverse
        if (var_30) {
            wp::adj_array_store(var_particle_flags, var_0, var_36, adj_particle_flags, adj_0, adj_36);
            wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_34);
            // adj: particle_flags[i] = particle_flags[i] & (~_ACTIVE_BIT)                        <L 146>
        }
        if (var_30) {
        }
        // adj: if r > 0.0 and h > r:                                                             <L 145>
        wp::adj_copy(var_29, adj_26, adj_28);
        wp::adj_address(var_material_resistance, var_27, adj_material_resistance, adj_25, adj_26);
        wp::adj_address(var_particle_material, var_0, adj_particle_material, adj_0, adj_25);
        // adj: r = material_resistance[particle_material[i]]                                     <L 144>
        if (var_11) {
            wp::adj_array_store(var_particle_burnt, var_0, var_24, adj_particle_burnt, adj_0, adj_24);
            // adj: particle_burnt[i] = b                                                         <L 143>
            wp::adj_where(var_22, var_23, var_19, adj_22, adj_23, adj_19, adj_24);
            if (var_22) {
                // adj: b = 1.0                                                                   <L 142>
            }
            // adj: if b > 1.0:                                                                   <L 141>
            wp::adj_add(var_20, var_18, adj_12, adj_18, adj_19);
            wp::adj_mul(var_17, var_fulguration, adj_17, adj_fulguration, adj_18);
            wp::adj_add(var_15, var_16, adj_15, adj_16, adj_17);
            wp::adj_mul(var_13, var_14, adj_13, adj_14, adj_15);
            wp::adj_sub(var_8, var_10, adj_8, adj_10, adj_14);
            wp::adj_address(var_particle_burnt, var_0, adj_particle_burnt, adj_0, adj_12);
            // adj: b = particle_burnt[i] + (_BURN_SLOPE * (h - _BURN_HEAT_THRESHOLD) + _BURN_BIAS) * fulguration  <L 140>
        }
        // adj: if h > _BURN_HEAT_THRESHOLD:                                                      <L 139>
        wp::adj_copy(var_9, adj_7, adj_8);
        wp::adj_address(var_particle_heat, var_0, adj_particle_heat, adj_0, adj_7);
        // adj: h = particle_heat[i]                                                              <L 137>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 136>
        }
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[i] & _ACTIVE_BIT) == 0:                                        <L 135>
        // adj: i = wp.tid()                                                                      <L 134>
        // adj: def apply_damage_kernel(                                                          <L 120>
        continue;
    }
}



extern "C" __global__ void disable_cut_springs_kernel_fb8e02e0_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_spring_indices,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::float32> var_material_stiffness_scale,
    wp::array_t<wp::float32> var_spring_stiffness_base,
    wp::array_t<wp::int32> var_spring_enabled,
    wp::array_t<wp::float32> var_spring_stiffness)
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
        const wp::int32 var_5 = 2;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 2;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        bool var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.5;
        wp::int32* var_34;
        wp::float32* var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::float32* var_38;
        wp::int32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
        //---------
        // forward
        // def disable_cut_springs_kernel(                                                        <L 150>
        // tid = wp.tid()                                                                         <L 173>
        var_0 = builtin_tid1d();
        // if spring_enabled[tid] == 0:                                                           <L 174>
        var_1 = wp::address(var_spring_enabled, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 175>
            continue;
        }
        // i = spring_indices[tid * 2 + 0]                                                        <L 176>
        var_6 = wp::mul(var_0, var_5);
        var_8 = wp::add(var_6, var_7);
        var_9 = wp::address(var_spring_indices, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // j = spring_indices[tid * 2 + 1]                                                        <L 177>
        var_13 = wp::mul(var_0, var_12);
        var_15 = wp::add(var_13, var_14);
        var_16 = wp::address(var_spring_indices, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if (particle_flags[i] & _ACTIVE_BIT) == 0 or (particle_flags[j] & _ACTIVE_BIT) == 0:       <L 178>
        var_20 = wp::address(var_particle_flags, var_10);
        var_23 = wp::load(var_20);
        var_22 = wp::bit_and(var_23, var_21);
        var_25 = (var_22 == var_24);
        var_19 = var_25;
        if (!var_19) {
            var_26 = wp::address(var_particle_flags, var_17);
            var_28 = wp::load(var_26);
            var_27 = wp::bit_and(var_28, var_21);
            var_30 = (var_27 == var_29);
            var_19 = var_19 || var_30;
        }
        if (var_19) {
            // spring_enabled[tid] = 0                                                            <L 179>
            wp::array_store(var_spring_enabled, var_0, var_31);
            // spring_stiffness[tid] = 0.0                                                        <L 180>
            wp::array_store(var_spring_stiffness, var_0, var_32);
            // return                                                                             <L 181>
            continue;
        }
        // scale = 0.5 * (                                                                        <L 182>
        // material_stiffness_scale[particle_material[i]]                                         <L 183>
        var_34 = wp::address(var_particle_material, var_10);
        var_36 = wp::load(var_34);
        var_35 = wp::address(var_material_stiffness_scale, var_36);
        // + material_stiffness_scale[particle_material[j]]                                       <L 184>
        var_37 = wp::address(var_particle_material, var_17);
        var_39 = wp::load(var_37);
        var_38 = wp::address(var_material_stiffness_scale, var_39);
        var_41 = wp::load(var_35);
        var_42 = wp::load(var_38);
        var_40 = wp::add(var_41, var_42);
        var_43 = wp::mul(var_33, var_40);
        // spring_stiffness[tid] = spring_stiffness_base[tid] * scale                             <L 186>
        var_44 = wp::address(var_spring_stiffness_base, var_0);
        var_46 = wp::load(var_44);
        var_45 = wp::mul(var_46, var_43);
        wp::array_store(var_spring_stiffness, var_0, var_45);
    }
}



extern "C" __global__ void disable_cut_springs_kernel_fb8e02e0_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_spring_indices,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::float32> var_material_stiffness_scale,
    wp::array_t<wp::float32> var_spring_stiffness_base,
    wp::array_t<wp::int32> var_spring_enabled,
    wp::array_t<wp::float32> var_spring_stiffness,
    wp::array_t<wp::int32> adj_spring_indices,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_particle_material,
    wp::array_t<wp::float32> adj_material_stiffness_scale,
    wp::array_t<wp::float32> adj_spring_stiffness_base,
    wp::array_t<wp::int32> adj_spring_enabled,
    wp::array_t<wp::float32> adj_spring_stiffness)
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
        const wp::int32 var_5 = 2;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        wp::int32 var_8;
        wp::int32* var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        const wp::int32 var_12 = 2;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        bool var_19;
        wp::int32* var_20;
        const wp::int32 var_21 = 1;
        wp::int32 var_22;
        wp::int32 var_23;
        const wp::int32 var_24 = 0;
        bool var_25;
        wp::int32* var_26;
        wp::int32 var_27;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        bool var_30;
        const wp::int32 var_31 = 0;
        const wp::float32 var_32 = 0.0;
        const wp::float32 var_33 = 0.5;
        wp::int32* var_34;
        wp::float32* var_35;
        wp::int32 var_36;
        wp::int32* var_37;
        wp::float32* var_38;
        wp::int32 var_39;
        wp::float32 var_40;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32* var_44;
        wp::float32 var_45;
        wp::float32 var_46;
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
        bool adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        bool adj_25 = {};
        wp::int32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        bool adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::float32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::float32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::float32 adj_46 = {};
        //---------
        // forward
        // def disable_cut_springs_kernel(                                                        <L 150>
        // tid = wp.tid()                                                                         <L 173>
        var_0 = builtin_tid1d();
        // if spring_enabled[tid] == 0:                                                           <L 174>
        var_1 = wp::address(var_spring_enabled, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // return                                                                             <L 175>
            goto label0;
        }
        // i = spring_indices[tid * 2 + 0]                                                        <L 176>
        var_6 = wp::mul(var_0, var_5);
        var_8 = wp::add(var_6, var_7);
        var_9 = wp::address(var_spring_indices, var_8);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // j = spring_indices[tid * 2 + 1]                                                        <L 177>
        var_13 = wp::mul(var_0, var_12);
        var_15 = wp::add(var_13, var_14);
        var_16 = wp::address(var_spring_indices, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // if (particle_flags[i] & _ACTIVE_BIT) == 0 or (particle_flags[j] & _ACTIVE_BIT) == 0:       <L 178>
        var_20 = wp::address(var_particle_flags, var_10);
        var_23 = wp::load(var_20);
        var_22 = wp::bit_and(var_23, var_21);
        var_25 = (var_22 == var_24);
        var_19 = var_25;
        if (!var_19) {
            var_26 = wp::address(var_particle_flags, var_17);
            var_28 = wp::load(var_26);
            var_27 = wp::bit_and(var_28, var_21);
            var_30 = (var_27 == var_29);
            var_19 = var_19 || var_30;
        }
        if (var_19) {
            // spring_enabled[tid] = 0                                                            <L 179>
            // wp::array_store(var_spring_enabled, var_0, var_31);
            // spring_stiffness[tid] = 0.0                                                        <L 180>
            // wp::array_store(var_spring_stiffness, var_0, var_32);
            // return                                                                             <L 181>
            goto label1;
        }
        // scale = 0.5 * (                                                                        <L 182>
        // material_stiffness_scale[particle_material[i]]                                         <L 183>
        var_34 = wp::address(var_particle_material, var_10);
        var_36 = wp::load(var_34);
        var_35 = wp::address(var_material_stiffness_scale, var_36);
        // + material_stiffness_scale[particle_material[j]]                                       <L 184>
        var_37 = wp::address(var_particle_material, var_17);
        var_39 = wp::load(var_37);
        var_38 = wp::address(var_material_stiffness_scale, var_39);
        var_41 = wp::load(var_35);
        var_42 = wp::load(var_38);
        var_40 = wp::add(var_41, var_42);
        var_43 = wp::mul(var_33, var_40);
        // spring_stiffness[tid] = spring_stiffness_base[tid] * scale                             <L 186>
        var_44 = wp::address(var_spring_stiffness_base, var_0);
        var_46 = wp::load(var_44);
        var_45 = wp::mul(var_46, var_43);
        // wp::array_store(var_spring_stiffness, var_0, var_45);
        //---------
        // reverse
        wp::adj_array_store(var_spring_stiffness, var_0, var_45, adj_spring_stiffness, adj_0, adj_45);
        wp::adj_mul(var_46, var_43, adj_44, adj_43, adj_45);
        wp::adj_address(var_spring_stiffness_base, var_0, adj_spring_stiffness_base, adj_0, adj_44);
        // adj: spring_stiffness[tid] = spring_stiffness_base[tid] * scale                        <L 186>
        wp::adj_mul(var_33, var_40, adj_33, adj_40, adj_43);
        wp::adj_add(var_41, var_42, adj_35, adj_38, adj_40);
        wp::adj_address(var_material_stiffness_scale, var_39, adj_material_stiffness_scale, adj_37, adj_38);
        wp::adj_address(var_particle_material, var_17, adj_particle_material, adj_17, adj_37);
        // adj: + material_stiffness_scale[particle_material[j]]                                  <L 184>
        wp::adj_address(var_material_stiffness_scale, var_36, adj_material_stiffness_scale, adj_34, adj_35);
        wp::adj_address(var_particle_material, var_10, adj_particle_material, adj_10, adj_34);
        // adj: material_stiffness_scale[particle_material[i]]                                    <L 183>
        // adj: scale = 0.5 * (                                                                   <L 182>
        if (var_19) {
            label1:;
            // adj: return                                                                        <L 181>
            wp::adj_array_store(var_spring_stiffness, var_0, var_32, adj_spring_stiffness, adj_0, adj_32);
            // adj: spring_stiffness[tid] = 0.0                                                   <L 180>
            wp::adj_array_store(var_spring_enabled, var_0, var_31, adj_spring_enabled, adj_0, adj_31);
            // adj: spring_enabled[tid] = 0                                                       <L 179>
        }
        if (!var_19) {
            wp::adj_address(var_particle_flags, var_17, adj_particle_flags, adj_17, adj_26);
        }
        wp::adj_address(var_particle_flags, var_10, adj_particle_flags, adj_10, adj_20);
        // adj: if (particle_flags[i] & _ACTIVE_BIT) == 0 or (particle_flags[j] & _ACTIVE_BIT) == 0:  <L 178>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_address(var_spring_indices, var_15, adj_spring_indices, adj_15, adj_16);
        wp::adj_add(var_13, var_14, adj_13, adj_14, adj_15);
        wp::adj_mul(var_0, var_12, adj_0, adj_12, adj_13);
        // adj: j = spring_indices[tid * 2 + 1]                                                   <L 177>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_spring_indices, var_8, adj_spring_indices, adj_8, adj_9);
        wp::adj_add(var_6, var_7, adj_6, adj_7, adj_8);
        wp::adj_mul(var_0, var_5, adj_0, adj_5, adj_6);
        // adj: i = spring_indices[tid * 2 + 0]                                                   <L 176>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 175>
        }
        wp::adj_address(var_spring_enabled, var_0, adj_spring_enabled, adj_0, adj_1);
        // adj: if spring_enabled[tid] == 0:                                                      <L 174>
        // adj: tid = wp.tid()                                                                    <L 173>
        // adj: def disable_cut_springs_kernel(                                                   <L 150>
        continue;
    }
}



extern "C" __global__ void heat_apply_kernel_ba9b8892_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tool_p0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tool_p1,
    wp::array_t<wp::float32> var_tool_radius,
    wp::array_t<wp::int32> var_tool_role_electrode,
    wp::int32 var_num_segments,
    wp::float32 var_tool_power,
    wp::int32 var_tool_active,
    wp::array_t<wp::float32> var_particle_heat)
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
        bool var_2;
        wp::int32* var_3;
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::range_t var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::float32 var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::float32 var_26;
        bool var_27;
        wp::float32* var_28;
        const wp::float32 var_29 = 0.9;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.1;
        wp::float32 var_33;
        wp::float32 var_34;
        //---------
        // forward
        // def heat_apply_kernel(                                                                 <L 52>
        // i = wp.tid()                                                                           <L 70>
        var_0 = builtin_tid1d();
        // if tool_active == 0:                                                                   <L 71>
        var_2 = (var_tool_active == var_1);
        if (var_2) {
            // return                                                                             <L 72>
            continue;
        }
        // if (particle_flags[i] & _ACTIVE_BIT) == 0:                                             <L 73>
        var_3 = wp::address(var_particle_flags, var_0);
        var_6 = wp::load(var_3);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 74>
            continue;
        }
        // p = particle_q[i]                                                                      <L 75>
        var_9 = wp::address(var_particle_q, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // for s in range(num_segments):                                                          <L 76>
        var_12 = wp::range(var_num_segments);
        start_for_2:;
            if (iter_cmp(var_12) == 0) goto end_for_2;
            var_13 = wp::iter_next(var_12);
            // if tool_role_electrode[s] == 0:                                                    <L 77>
            var_14 = wp::address(var_tool_role_electrode, var_13);
            var_17 = wp::load(var_14);
            var_16 = (var_17 == var_15);
            if (var_16) {
                // continue                                                                       <L 78>
                goto start_for_2;
            }
            // r = tool_radius[s]                                                                 <L 79>
            var_18 = wp::address(var_tool_radius, var_13);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // d2 = _point_segment_distance_sq(p, tool_p0[s], tool_p1[s])                         <L 80>
            var_21 = wp::address(var_tool_p0, var_13);
            var_22 = wp::address(var_tool_p1, var_13);
            var_24 = wp::load(var_21);
            var_25 = wp::load(var_22);
            var_23 = _point_segment_distance_sq_2(var_10, var_24, var_25);
            // if d2 <= r * r:                                                                    <L 81>
            var_26 = wp::mul(var_19, var_19);
            var_27 = (var_23 <= var_26);
            if (var_27) {
                // particle_heat[i] = particle_heat[i] * _HEAT_INJECT_RETAIN + tool_power * _HEAT_INJECT_NEW       <L 82>
                var_28 = wp::address(var_particle_heat, var_0);
                var_31 = wp::load(var_28);
                var_30 = wp::mul(var_31, var_29);
                var_33 = wp::mul(var_tool_power, var_32);
                var_34 = wp::add(var_30, var_33);
                wp::array_store(var_particle_heat, var_0, var_34);
                // return                                                                         <L 83>
                continue;
            }
            goto start_for_2;
        end_for_2:;
    }
}



extern "C" __global__ void heat_apply_kernel_ba9b8892_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tool_p0,
    wp::array_t<wp::vec_t<3, wp::float32>> var_tool_p1,
    wp::array_t<wp::float32> var_tool_radius,
    wp::array_t<wp::int32> var_tool_role_electrode,
    wp::int32 var_num_segments,
    wp::float32 var_tool_power,
    wp::int32 var_tool_active,
    wp::array_t<wp::float32> var_particle_heat,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_tool_p0,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_tool_p1,
    wp::array_t<wp::float32> adj_tool_radius,
    wp::array_t<wp::int32> adj_tool_role_electrode,
    wp::int32 adj_num_segments,
    wp::float32 adj_tool_power,
    wp::int32 adj_tool_active,
    wp::array_t<wp::float32> adj_particle_heat)
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
        bool var_2;
        wp::int32* var_3;
        const wp::int32 var_4 = 1;
        wp::int32 var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::vec_t<3, wp::float32>* var_9;
        wp::vec_t<3, wp::float32> var_10;
        wp::vec_t<3, wp::float32> var_11;
        wp::range_t var_12;
        wp::int32 var_13;
        wp::int32* var_14;
        const wp::int32 var_15 = 0;
        bool var_16;
        wp::int32 var_17;
        wp::float32* var_18;
        wp::float32 var_19;
        wp::float32 var_20;
        wp::vec_t<3, wp::float32>* var_21;
        wp::vec_t<3, wp::float32>* var_22;
        wp::float32 var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::float32 var_26;
        bool var_27;
        wp::float32* var_28;
        const wp::float32 var_29 = 0.9;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::float32 var_32 = 0.1;
        wp::float32 var_33;
        wp::float32 var_34;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        bool adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::int32 adj_7 = {};
        bool adj_8 = {};
        wp::vec_t<3, wp::float32> adj_9 = {};
        wp::vec_t<3, wp::float32> adj_10 = {};
        wp::vec_t<3, wp::float32> adj_11 = {};
        wp::range_t adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        bool adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::float32 adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::float32 adj_26 = {};
        bool adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::float32 adj_34 = {};
        //---------
        // forward
        // def heat_apply_kernel(                                                                 <L 52>
        // i = wp.tid()                                                                           <L 70>
        var_0 = builtin_tid1d();
        // if tool_active == 0:                                                                   <L 71>
        var_2 = (var_tool_active == var_1);
        if (var_2) {
            // return                                                                             <L 72>
            goto label0;
        }
        // if (particle_flags[i] & _ACTIVE_BIT) == 0:                                             <L 73>
        var_3 = wp::address(var_particle_flags, var_0);
        var_6 = wp::load(var_3);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // return                                                                             <L 74>
            goto label1;
        }
        // p = particle_q[i]                                                                      <L 75>
        var_9 = wp::address(var_particle_q, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // for s in range(num_segments):                                                          <L 76>
        var_12 = wp::range(var_num_segments);
        //---------
        // reverse
        var_12 = wp::iter_reverse(var_12);
        start_for_2:;
            if (iter_cmp(var_12) == 0) goto end_for_2;
            var_13 = wp::iter_next(var_12);
        	adj_14 = {};
        	adj_15 = {};
        	adj_16 = {};
        	adj_17 = {};
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
        	adj_31 = {};
        	adj_32 = {};
        	adj_33 = {};
        	adj_34 = {};
            // if tool_role_electrode[s] == 0:                                                    <L 77>
            var_14 = wp::address(var_tool_role_electrode, var_13);
            var_17 = wp::load(var_14);
            var_16 = (var_17 == var_15);
            if (var_16) {
                // continue                                                                       <L 78>
                goto start_for_2;
            }
            // r = tool_radius[s]                                                                 <L 79>
            var_18 = wp::address(var_tool_radius, var_13);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // d2 = _point_segment_distance_sq(p, tool_p0[s], tool_p1[s])                         <L 80>
            var_21 = wp::address(var_tool_p0, var_13);
            var_22 = wp::address(var_tool_p1, var_13);
            var_24 = wp::load(var_21);
            var_25 = wp::load(var_22);
            var_23 = _point_segment_distance_sq_2(var_10, var_24, var_25);
            // if d2 <= r * r:                                                                    <L 81>
            var_26 = wp::mul(var_19, var_19);
            var_27 = (var_23 <= var_26);
            if (var_27) {
                // particle_heat[i] = particle_heat[i] * _HEAT_INJECT_RETAIN + tool_power * _HEAT_INJECT_NEW       <L 82>
                var_28 = wp::address(var_particle_heat, var_0);
                var_31 = wp::load(var_28);
                var_30 = wp::mul(var_31, var_29);
                var_33 = wp::mul(var_tool_power, var_32);
                var_34 = wp::add(var_30, var_33);
                // wp::array_store(var_particle_heat, var_0, var_34);
                // return                                                                         <L 83>
                goto label4;
            }
            if (var_27) {
                label4:;
                // adj: return                                                                    <L 83>
                wp::adj_array_store(var_particle_heat, var_0, var_34, adj_particle_heat, adj_0, adj_34);
                wp::adj_add(var_30, var_33, adj_30, adj_33, adj_34);
                wp::adj_mul(var_tool_power, var_32, adj_tool_power, adj_32, adj_33);
                wp::adj_mul(var_31, var_29, adj_28, adj_29, adj_30);
                wp::adj_address(var_particle_heat, var_0, adj_particle_heat, adj_0, adj_28);
                // adj: particle_heat[i] = particle_heat[i] * _HEAT_INJECT_RETAIN + tool_power * _HEAT_INJECT_NEW  <L 82>
            }
            wp::adj_mul(var_19, var_19, adj_19, adj_19, adj_26);
            // adj: if d2 <= r * r:                                                               <L 81>
            adj__point_segment_distance_sq_2(var_10, var_24, var_25, adj_10, adj_21, adj_22, adj_23);
            wp::adj_address(var_tool_p1, var_13, adj_tool_p1, adj_13, adj_22);
            wp::adj_address(var_tool_p0, var_13, adj_tool_p0, adj_13, adj_21);
            // adj: d2 = _point_segment_distance_sq(p, tool_p0[s], tool_p1[s])                    <L 80>
            wp::adj_copy(var_20, adj_18, adj_19);
            wp::adj_address(var_tool_radius, var_13, adj_tool_radius, adj_13, adj_18);
            // adj: r = tool_radius[s]                                                            <L 79>
            if (var_16) {
                // adj: continue                                                                  <L 78>
            }
            wp::adj_address(var_tool_role_electrode, var_13, adj_tool_role_electrode, adj_13, adj_14);
            // adj: if tool_role_electrode[s] == 0:                                               <L 77>
        	goto start_for_2;
        end_for_2:;
        wp::adj_range(var_num_segments, adj_num_segments, adj_12);
        // adj: for s in range(num_segments):                                                     <L 76>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_9);
        // adj: p = particle_q[i]                                                                 <L 75>
        if (var_8) {
            label1:;
            // adj: return                                                                        <L 74>
        }
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_3);
        // adj: if (particle_flags[i] & _ACTIVE_BIT) == 0:                                        <L 73>
        if (var_2) {
            label0:;
            // adj: return                                                                        <L 72>
        }
        // adj: if tool_active == 0:                                                              <L 71>
        // adj: i = wp.tid()                                                                      <L 70>
        // adj: def heat_apply_kernel(                                                            <L 52>
        continue;
    }
}



extern "C" __global__ void heat_diffuse_kernel_99dd82de_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_neighbors,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::float32> var_material_conductivity,
    wp::array_t<wp::float32> var_heat_in,
    wp::float32 var_dt,
    wp::array_t<wp::float32> var_heat_out)
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
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::float32* var_13;
        wp::int32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 0.0;
        wp::float32 var_18;
        const wp::int32 var_19 = 6;
        wp::range_t var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::int32* var_32;
        wp::float32* var_33;
        wp::int32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::float32 var_40 = 0.5;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.98;
        wp::float32 var_46;
        const wp::float32 var_47 = 0.16;
        wp::float32 var_48;
        wp::float32 var_49;
        //---------
        // forward
        // def heat_diffuse_kernel(                                                               <L 87>
        // i = wp.tid()                                                                           <L 101>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & _ACTIVE_BIT) == 0:                                             <L 102>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // heat_out[i] = heat_in[i]                                                           <L 103>
            var_7 = wp::address(var_heat_in, var_0);
            var_8 = wp::load(var_7);
            wp::array_store(var_heat_out, var_0, var_8);
            // return                                                                             <L 104>
            continue;
        }
        // h = heat_in[i]                                                                         <L 105>
        var_9 = wp::address(var_heat_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // c_i = material_conductivity[particle_material[i]]                                      <L 106>
        var_12 = wp::address(var_particle_material, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::address(var_material_conductivity, var_14);
        var_16 = wp::load(var_13);
        var_15 = wp::copy(var_16);
        // accum = float(0.0)                                                                     <L 107>
        var_18 = wp::float(var_17);
        // for d in range(6):                                                                     <L 108>
        var_20 = wp::range(var_19);
        start_for_1:;
            if (iter_cmp(var_20) == 0) goto end_for_1;
            var_21 = wp::iter_next(var_20);
            // nb = particle_neighbors[i, d]                                                      <L 109>
            var_22 = wp::address(var_particle_neighbors, var_0, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // if nb < 0:                                                                         <L 110>
            var_26 = (var_23 < var_25);
            if (var_26) {
                // continue                                                                       <L 111>
                goto start_for_1;
            }
            // if (particle_flags[nb] & _ACTIVE_BIT) == 0:                                        <L 112>
            var_27 = wp::address(var_particle_flags, var_23);
            var_29 = wp::load(var_27);
            var_28 = wp::bit_and(var_29, var_2);
            var_31 = (var_28 == var_30);
            if (var_31) {
                // continue                                                                       <L 113>
                goto start_for_1;
            }
            // c_n = material_conductivity[particle_material[nb]]                                 <L 114>
            var_32 = wp::address(var_particle_material, var_23);
            var_34 = wp::load(var_32);
            var_33 = wp::address(var_material_conductivity, var_34);
            var_36 = wp::load(var_33);
            var_35 = wp::copy(var_36);
            // accum += (heat_in[nb] - h) * 0.5 * (c_n + c_i)                                     <L 115>
            var_37 = wp::address(var_heat_in, var_23);
            var_39 = wp::load(var_37);
            var_38 = wp::sub(var_39, var_10);
            var_41 = wp::mul(var_38, var_40);
            var_42 = wp::add(var_35, var_15);
            var_43 = wp::mul(var_41, var_42);
            var_44 = wp::add(var_18, var_43);
            wp::assign(var_18, var_44);
            goto start_for_1;
        end_for_1:;
        // heat_out[i] = h * _HEAT_RETAIN + accum * _HEAT_DIFFUSE                                 <L 116>
        var_46 = wp::mul(var_10, var_45);
        var_48 = wp::mul(var_18, var_47);
        var_49 = wp::add(var_46, var_48);
        wp::array_store(var_heat_out, var_0, var_49);
    }
}



extern "C" __global__ void heat_diffuse_kernel_99dd82de_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_neighbors,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::float32> var_material_conductivity,
    wp::array_t<wp::float32> var_heat_in,
    wp::float32 var_dt,
    wp::array_t<wp::float32> var_heat_out,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_particle_neighbors,
    wp::array_t<wp::int32> adj_particle_material,
    wp::array_t<wp::float32> adj_material_conductivity,
    wp::array_t<wp::float32> adj_heat_in,
    wp::float32 adj_dt,
    wp::array_t<wp::float32> adj_heat_out)
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
        wp::float32 var_8;
        wp::float32* var_9;
        wp::float32 var_10;
        wp::float32 var_11;
        wp::int32* var_12;
        wp::float32* var_13;
        wp::int32 var_14;
        wp::float32 var_15;
        wp::float32 var_16;
        const wp::float32 var_17 = 0.0;
        wp::float32 var_18;
        const wp::int32 var_19 = 6;
        wp::range_t var_20;
        wp::int32 var_21;
        wp::int32* var_22;
        wp::int32 var_23;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 0;
        bool var_31;
        wp::int32* var_32;
        wp::float32* var_33;
        wp::int32 var_34;
        wp::float32 var_35;
        wp::float32 var_36;
        wp::float32* var_37;
        wp::float32 var_38;
        wp::float32 var_39;
        const wp::float32 var_40 = 0.5;
        wp::float32 var_41;
        wp::float32 var_42;
        wp::float32 var_43;
        wp::float32 var_44;
        const wp::float32 var_45 = 0.98;
        wp::float32 var_46;
        const wp::float32 var_47 = 0.16;
        wp::float32 var_48;
        wp::float32 var_49;
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
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::float32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::range_t adj_20 = {};
        wp::int32 adj_21 = {};
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
        wp::int32 adj_34 = {};
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
        //---------
        // forward
        // def heat_diffuse_kernel(                                                               <L 87>
        // i = wp.tid()                                                                           <L 101>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & _ACTIVE_BIT) == 0:                                             <L 102>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::load(var_1);
        var_3 = wp::bit_and(var_4, var_2);
        var_6 = (var_3 == var_5);
        if (var_6) {
            // heat_out[i] = heat_in[i]                                                           <L 103>
            var_7 = wp::address(var_heat_in, var_0);
            var_8 = wp::load(var_7);
            // wp::array_store(var_heat_out, var_0, var_8);
            // return                                                                             <L 104>
            goto label0;
        }
        // h = heat_in[i]                                                                         <L 105>
        var_9 = wp::address(var_heat_in, var_0);
        var_11 = wp::load(var_9);
        var_10 = wp::copy(var_11);
        // c_i = material_conductivity[particle_material[i]]                                      <L 106>
        var_12 = wp::address(var_particle_material, var_0);
        var_14 = wp::load(var_12);
        var_13 = wp::address(var_material_conductivity, var_14);
        var_16 = wp::load(var_13);
        var_15 = wp::copy(var_16);
        // accum = float(0.0)                                                                     <L 107>
        var_18 = wp::float(var_17);
        // for d in range(6):                                                                     <L 108>
        var_20 = wp::range(var_19);
        // heat_out[i] = h * _HEAT_RETAIN + accum * _HEAT_DIFFUSE                                 <L 116>
        var_46 = wp::mul(var_10, var_45);
        var_48 = wp::mul(var_18, var_47);
        var_49 = wp::add(var_46, var_48);
        // wp::array_store(var_heat_out, var_0, var_49);
        //---------
        // reverse
        wp::adj_array_store(var_heat_out, var_0, var_49, adj_heat_out, adj_0, adj_49);
        wp::adj_add(var_46, var_48, adj_46, adj_48, adj_49);
        wp::adj_mul(var_18, var_47, adj_18, adj_47, adj_48);
        wp::adj_mul(var_10, var_45, adj_10, adj_45, adj_46);
        // adj: heat_out[i] = h * _HEAT_RETAIN + accum * _HEAT_DIFFUSE                            <L 116>
        var_20 = wp::iter_reverse(var_20);
        start_for_1:;
            if (iter_cmp(var_20) == 0) goto end_for_1;
            var_21 = wp::iter_next(var_20);
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
        	adj_36 = {};
        	adj_37 = {};
        	adj_38 = {};
        	adj_39 = {};
        	adj_40 = {};
        	adj_41 = {};
        	adj_42 = {};
        	adj_43 = {};
        	adj_44 = {};
            // nb = particle_neighbors[i, d]                                                      <L 109>
            var_22 = wp::address(var_particle_neighbors, var_0, var_21);
            var_24 = wp::load(var_22);
            var_23 = wp::copy(var_24);
            // if nb < 0:                                                                         <L 110>
            var_26 = (var_23 < var_25);
            if (var_26) {
                // continue                                                                       <L 111>
                goto start_for_1;
            }
            // if (particle_flags[nb] & _ACTIVE_BIT) == 0:                                        <L 112>
            var_27 = wp::address(var_particle_flags, var_23);
            var_29 = wp::load(var_27);
            var_28 = wp::bit_and(var_29, var_2);
            var_31 = (var_28 == var_30);
            if (var_31) {
                // continue                                                                       <L 113>
                goto start_for_1;
            }
            // c_n = material_conductivity[particle_material[nb]]                                 <L 114>
            var_32 = wp::address(var_particle_material, var_23);
            var_34 = wp::load(var_32);
            var_33 = wp::address(var_material_conductivity, var_34);
            var_36 = wp::load(var_33);
            var_35 = wp::copy(var_36);
            // accum += (heat_in[nb] - h) * 0.5 * (c_n + c_i)                                     <L 115>
            var_37 = wp::address(var_heat_in, var_23);
            var_39 = wp::load(var_37);
            var_38 = wp::sub(var_39, var_10);
            var_41 = wp::mul(var_38, var_40);
            var_42 = wp::add(var_35, var_15);
            var_43 = wp::mul(var_41, var_42);
            var_44 = wp::add(var_18, var_43);
            wp::assign(var_18, var_44);
            wp::adj_assign(var_18, var_44, adj_18, adj_44);
            wp::adj_add(var_18, var_43, adj_18, adj_43, adj_44);
            wp::adj_mul(var_41, var_42, adj_41, adj_42, adj_43);
            wp::adj_add(var_35, var_15, adj_35, adj_15, adj_42);
            wp::adj_mul(var_38, var_40, adj_38, adj_40, adj_41);
            wp::adj_sub(var_39, var_10, adj_37, adj_10, adj_38);
            wp::adj_address(var_heat_in, var_23, adj_heat_in, adj_23, adj_37);
            // adj: accum += (heat_in[nb] - h) * 0.5 * (c_n + c_i)                                <L 115>
            wp::adj_copy(var_36, adj_33, adj_35);
            wp::adj_address(var_material_conductivity, var_34, adj_material_conductivity, adj_32, adj_33);
            wp::adj_address(var_particle_material, var_23, adj_particle_material, adj_23, adj_32);
            // adj: c_n = material_conductivity[particle_material[nb]]                            <L 114>
            if (var_31) {
                // adj: continue                                                                  <L 113>
            }
            wp::adj_address(var_particle_flags, var_23, adj_particle_flags, adj_23, adj_27);
            // adj: if (particle_flags[nb] & _ACTIVE_BIT) == 0:                                   <L 112>
            if (var_26) {
                // adj: continue                                                                  <L 111>
            }
            // adj: if nb < 0:                                                                    <L 110>
            wp::adj_copy(var_24, adj_22, adj_23);
            wp::adj_address(var_particle_neighbors, var_0, var_21, adj_particle_neighbors, adj_0, adj_21, adj_22);
            // adj: nb = particle_neighbors[i, d]                                                 <L 109>
        	goto start_for_1;
        end_for_1:;
        wp::adj_range(var_19, adj_19, adj_20);
        // adj: for d in range(6):                                                                <L 108>
        wp::adj_float(var_17, adj_17, adj_18);
        // adj: accum = float(0.0)                                                                <L 107>
        wp::adj_copy(var_16, adj_13, adj_15);
        wp::adj_address(var_material_conductivity, var_14, adj_material_conductivity, adj_12, adj_13);
        wp::adj_address(var_particle_material, var_0, adj_particle_material, adj_0, adj_12);
        // adj: c_i = material_conductivity[particle_material[i]]                                 <L 106>
        wp::adj_copy(var_11, adj_9, adj_10);
        wp::adj_address(var_heat_in, var_0, adj_heat_in, adj_0, adj_9);
        // adj: h = heat_in[i]                                                                    <L 105>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 104>
            wp::adj_array_store(var_heat_out, var_0, var_8, adj_heat_out, adj_0, adj_7);
            wp::adj_address(var_heat_in, var_0, adj_heat_in, adj_0, adj_7);
            // adj: heat_out[i] = heat_in[i]                                                      <L 103>
        }
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[i] & _ACTIVE_BIT) == 0:                                        <L 102>
        // adj: i = wp.tid()                                                                      <L 101>
        // adj: def heat_diffuse_kernel(                                                          <L 87>
        continue;
    }
}

