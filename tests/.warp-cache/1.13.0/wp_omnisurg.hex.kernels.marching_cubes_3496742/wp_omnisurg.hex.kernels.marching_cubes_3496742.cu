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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:72
static CUDA_CALLABLE wp::int32 _corner_particle_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_c)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
    //---------
    // forward
    // def _corner_particle(                                                                  <L 73>
    // ox = corner_offsets[c, 0]                                                              <L 78>
    var_1 = wp::address(var_corner_offsets, var_c, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // oy = corner_offsets[c, 1]                                                              <L 79>
    var_5 = wp::address(var_corner_offsets, var_c, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // oz = corner_offsets[c, 2]                                                              <L 80>
    var_9 = wp::address(var_corner_offsets, var_c, var_8);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // return grid_to_particle[cx + ox, cy + oy, cz + oz]                                     <L 81>
    var_12 = wp::add(var_cx, var_2);
    var_13 = wp::add(var_cy, var_6);
    var_14 = wp::add(var_cz, var_10);
    var_15 = wp::address(var_grid_to_particle, var_12, var_13, var_14);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    return var_16;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:84
static CUDA_CALLABLE wp::int32 _corner_active_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_c)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::int32* var_4;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 1;
    //---------
    // forward
    // def _corner_active(                                                                    <L 85>
    // p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, c)                  <L 91>
    var_0 = _corner_particle_0(var_grid_to_particle, var_corner_offsets, var_cx, var_cy, var_cz, var_c);
    // if p < 0:                                                                              <L 92>
    var_2 = (var_0 < var_1);
    if (var_2) {
        // return 0                                                                           <L 93>
        return var_3;
    }
    // if (particle_flags[p] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 94>
    var_4 = wp::address(var_particle_flags, var_0);
    var_7 = wp::int32(var_6);
    var_9 = wp::load(var_4);
    var_8 = wp::bit_and(var_9, var_7);
    var_11 = (var_8 == var_10);
    if (var_11) {
        // return 0                                                                           <L 95>
        return var_12;
    }
    // return 1                                                                               <L 96>
    return var_13;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:261
static CUDA_CALLABLE wp::int32 _edge_to_vertex_id_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_edge)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    bool var_10;
    const wp::int32 var_11 = 0;
    bool var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::int32 var_15 = -1;
    bool var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    const wp::int32 var_21 = -1;
    wp::int32 var_22;
    wp::int32* var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    wp::int32 var_28;
    const wp::int32 var_29 = 1;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    const wp::int32 var_36 = -1;
    const wp::int32 var_37 = 6;
    wp::int32 var_38;
    wp::int32 var_39;
    //---------
    // forward
    // def _edge_to_vertex_id(                                                                <L 262>
    // a = edge_corners[edge, 0]                                                              <L 276>
    var_1 = wp::address(var_edge_corners, var_edge, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // b = edge_corners[edge, 1]                                                              <L 277>
    var_5 = wp::address(var_edge_corners, var_edge, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // a_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, a)       <L 278>
    var_8 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_2);
    // b_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, b)       <L 279>
    var_9 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_6);
    // if a_active == 0 and b_active == 0:                                                    <L 283>
    var_12 = (var_8 == var_11);
    var_10 = var_12;
    if (var_10) {
        var_14 = (var_9 == var_13);
        var_10 = var_10 && var_14;
    }
    if (var_10) {
        // return -1                                                                          <L 284>
        return var_15;
    }
    // if a_active != 0 and b_active != 0:                                                    <L 285>
    var_18 = (var_8 != var_17);
    var_16 = var_18;
    if (var_16) {
        var_20 = (var_9 != var_19);
        var_16 = var_16 && var_20;
    }
    if (var_16) {
        // return -1                                                                          <L 286>
        return var_21;
    }
    // active_corner = a                                                                      <L 287>
    var_22 = wp::copy(var_2);
    // d = edge_base_dir[edge]                                                                <L 288>
    var_23 = wp::address(var_edge_base_dir, var_edge);
    var_25 = wp::load(var_23);
    var_24 = wp::copy(var_25);
    // if a_active == 0:                                                                      <L 289>
    var_27 = (var_8 == var_26);
    if (var_27) {
        // active_corner = b                                                                  <L 290>
        var_28 = wp::copy(var_6);
        // d = d - 1                                                                          <L 291>
        var_30 = wp::sub(var_24, var_29);
    }
    var_31 = wp::where(var_27, var_28, var_22);
    var_32 = wp::where(var_27, var_30, var_24);
    // p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, active_corner)       <L 292>
    var_33 = _corner_particle_0(var_grid_to_particle, var_corner_offsets, var_cx, var_cy, var_cz, var_31);
    // if p < 0:                                                                              <L 293>
    var_35 = (var_33 < var_34);
    if (var_35) {
        // return -1                                                                          <L 294>
        return var_36;
    }
    // return p * 6 + d                                                                       <L 295>
    var_38 = wp::mul(var_33, var_37);
    var_39 = wp::add(var_38, var_32);
    return var_39;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:160
static CUDA_CALLABLE wp::int32 _cube_flat_id_0(
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    //---------
    // forward
    // def _cube_flat_id(cx: int, cy: int, cz: int, ny_cells: int, nz_cells: int) -> int:       <L 161>
    // return (cx * ny_cells + cy) * nz_cells + cz                                            <L 162>
    var_0 = wp::mul(var_cx, var_ny_cells);
    var_1 = wp::add(var_0, var_cy);
    var_2 = wp::mul(var_1, var_nz_cells);
    var_3 = wp::add(var_2, var_cz);
    return var_3;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:144
static CUDA_CALLABLE wp::int32 _compute_cube_case_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::int32 var_15 = 1;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 2;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    bool var_23;
    const wp::int32 var_24 = 1;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    const wp::int32 var_29 = 3;
    wp::int32 var_30;
    const wp::int32 var_31 = 0;
    bool var_32;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 4;
    wp::int32 var_39;
    const wp::int32 var_40 = 0;
    bool var_41;
    const wp::int32 var_42 = 1;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    const wp::int32 var_47 = 5;
    wp::int32 var_48;
    const wp::int32 var_49 = 0;
    bool var_50;
    const wp::int32 var_51 = 1;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 6;
    wp::int32 var_57;
    const wp::int32 var_58 = 0;
    bool var_59;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    const wp::int32 var_65 = 7;
    wp::int32 var_66;
    const wp::int32 var_67 = 0;
    bool var_68;
    const wp::int32 var_69 = 1;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::int32 var_73;
    //---------
    // forward
    // def _compute_cube_case(                                                                <L 145>
    // mask = int(0)                                                                          <L 153>
    var_1 = wp::int(var_0);
    // for c in range(8):                                                                     <L 154>
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_3 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_2);
    var_5 = (var_3 != var_4);
    if (var_5) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_7 = wp::int(var_6);
        var_8 = wp::lshift(var_7, var_2);
        var_9 = wp::bit_or(var_1, var_8);
    }
    var_10 = wp::where(var_5, var_9, var_1);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_12 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_11);
    var_14 = (var_12 != var_13);
    if (var_14) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_16 = wp::int(var_15);
        var_17 = wp::lshift(var_16, var_11);
        var_18 = wp::bit_or(var_10, var_17);
    }
    var_19 = wp::where(var_14, var_18, var_10);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_21 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_20);
    var_23 = (var_21 != var_22);
    if (var_23) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_25 = wp::int(var_24);
        var_26 = wp::lshift(var_25, var_20);
        var_27 = wp::bit_or(var_19, var_26);
    }
    var_28 = wp::where(var_23, var_27, var_19);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_30 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_29);
    var_32 = (var_30 != var_31);
    if (var_32) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_34 = wp::int(var_33);
        var_35 = wp::lshift(var_34, var_29);
        var_36 = wp::bit_or(var_28, var_35);
    }
    var_37 = wp::where(var_32, var_36, var_28);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_39 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_38);
    var_41 = (var_39 != var_40);
    if (var_41) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_43 = wp::int(var_42);
        var_44 = wp::lshift(var_43, var_38);
        var_45 = wp::bit_or(var_37, var_44);
    }
    var_46 = wp::where(var_41, var_45, var_37);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_48 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_47);
    var_50 = (var_48 != var_49);
    if (var_50) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_52 = wp::int(var_51);
        var_53 = wp::lshift(var_52, var_47);
        var_54 = wp::bit_or(var_46, var_53);
    }
    var_55 = wp::where(var_50, var_54, var_46);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_57 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_56);
    var_59 = (var_57 != var_58);
    if (var_59) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_61 = wp::int(var_60);
        var_62 = wp::lshift(var_61, var_56);
        var_63 = wp::bit_or(var_55, var_62);
    }
    var_64 = wp::where(var_59, var_63, var_55);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_66 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_65);
    var_68 = (var_66 != var_67);
    if (var_68) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_70 = wp::int(var_69);
        var_71 = wp::lshift(var_70, var_65);
        var_72 = wp::bit_or(var_64, var_71);
    }
    var_73 = wp::where(var_68, var_72, var_64);
    // return mask                                                                            <L 157>
    return var_73;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:165
static CUDA_CALLABLE void _clear_fixed_slot_0(
    wp::int32 var_slot,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = -1;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 0;
    const wp::int32 var_7 = 2;
    //---------
    // forward
    // def _clear_fixed_slot(                                                                 <L 166>
    // slot_active[slot] = 0                                                                  <L 172>
    wp::array_store(var_slot_active, var_slot, var_0);
    // slot_to_compact[slot] = -1                                                             <L 173>
    wp::array_store(var_slot_to_compact, var_slot, var_1);
    // slot_tri_indices[slot, 0] = 0                                                          <L 174>
    wp::array_store(var_slot_tri_indices, var_slot, var_3, var_2);
    // slot_tri_indices[slot, 1] = 0                                                          <L 175>
    wp::array_store(var_slot_tri_indices, var_slot, var_5, var_4);
    // slot_tri_indices[slot, 2] = 0                                                          <L 176>
    wp::array_store(var_slot_tri_indices, var_slot, var_7, var_6);
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:298
static CUDA_CALLABLE wp::int32 _write_cube_fixed_slots_0(
    wp::int32 var_cube_flat,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_case,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 5;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    const wp::int32 var_10 = 4;
    wp::int32 var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    bool var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 255;
    bool var_18;
    const wp::int32 var_19 = 0;
    const wp::int32 var_20 = 3;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 0;
    bool var_28;
    const wp::int32 var_29 = 3;
    wp::int32 var_30;
    const wp::int32 var_31 = 1;
    wp::int32 var_32;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 3;
    wp::int32 var_37;
    const wp::int32 var_38 = 2;
    wp::int32 var_39;
    wp::int32* var_40;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    bool var_46;
    const wp::int32 var_47 = 0;
    bool var_48;
    const wp::int32 var_49 = 0;
    bool var_50;
    const wp::int32 var_51 = 0;
    bool var_52;
    bool var_53;
    wp::int32 var_54;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 1;
    const wp::int32 var_57 = 2;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 1;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    const wp::int32 var_63 = 1;
    const wp::int32 var_64 = 3;
    wp::int32 var_65;
    const wp::int32 var_66 = 0;
    wp::int32 var_67;
    wp::int32* var_68;
    wp::int32 var_69;
    wp::int32 var_70;
    const wp::int32 var_71 = 0;
    bool var_72;
    const wp::int32 var_73 = 3;
    wp::int32 var_74;
    const wp::int32 var_75 = 1;
    wp::int32 var_76;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    const wp::int32 var_80 = 3;
    wp::int32 var_81;
    const wp::int32 var_82 = 2;
    wp::int32 var_83;
    wp::int32* var_84;
    wp::int32 var_85;
    wp::int32 var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    bool var_90;
    const wp::int32 var_91 = 0;
    bool var_92;
    const wp::int32 var_93 = 0;
    bool var_94;
    const wp::int32 var_95 = 0;
    bool var_96;
    bool var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 0;
    const wp::int32 var_100 = 1;
    const wp::int32 var_101 = 2;
    const wp::int32 var_102 = 1;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::int32 var_106;
    wp::int32 var_107;
    wp::int32 var_108;
    wp::int32 var_109;
    wp::int32 var_110;
    wp::int32 var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    const wp::int32 var_114 = 2;
    const wp::int32 var_115 = 3;
    wp::int32 var_116;
    const wp::int32 var_117 = 0;
    wp::int32 var_118;
    wp::int32* var_119;
    wp::int32 var_120;
    wp::int32 var_121;
    const wp::int32 var_122 = 0;
    bool var_123;
    const wp::int32 var_124 = 3;
    wp::int32 var_125;
    const wp::int32 var_126 = 1;
    wp::int32 var_127;
    wp::int32* var_128;
    wp::int32 var_129;
    wp::int32 var_130;
    const wp::int32 var_131 = 3;
    wp::int32 var_132;
    const wp::int32 var_133 = 2;
    wp::int32 var_134;
    wp::int32* var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::int32 var_140;
    bool var_141;
    const wp::int32 var_142 = 0;
    bool var_143;
    const wp::int32 var_144 = 0;
    bool var_145;
    const wp::int32 var_146 = 0;
    bool var_147;
    bool var_148;
    wp::int32 var_149;
    const wp::int32 var_150 = 0;
    const wp::int32 var_151 = 1;
    const wp::int32 var_152 = 2;
    const wp::int32 var_153 = 1;
    const wp::int32 var_154 = 1;
    wp::int32 var_155;
    wp::int32 var_156;
    wp::int32 var_157;
    wp::int32 var_158;
    wp::int32 var_159;
    wp::int32 var_160;
    wp::int32 var_161;
    wp::int32 var_162;
    wp::int32 var_163;
    wp::int32 var_164;
    const wp::int32 var_165 = 3;
    const wp::int32 var_166 = 3;
    wp::int32 var_167;
    const wp::int32 var_168 = 0;
    wp::int32 var_169;
    wp::int32* var_170;
    wp::int32 var_171;
    wp::int32 var_172;
    const wp::int32 var_173 = 0;
    bool var_174;
    const wp::int32 var_175 = 3;
    wp::int32 var_176;
    const wp::int32 var_177 = 1;
    wp::int32 var_178;
    wp::int32* var_179;
    wp::int32 var_180;
    wp::int32 var_181;
    const wp::int32 var_182 = 3;
    wp::int32 var_183;
    const wp::int32 var_184 = 2;
    wp::int32 var_185;
    wp::int32* var_186;
    wp::int32 var_187;
    wp::int32 var_188;
    wp::int32 var_189;
    wp::int32 var_190;
    wp::int32 var_191;
    bool var_192;
    const wp::int32 var_193 = 0;
    bool var_194;
    const wp::int32 var_195 = 0;
    bool var_196;
    const wp::int32 var_197 = 0;
    bool var_198;
    bool var_199;
    wp::int32 var_200;
    const wp::int32 var_201 = 0;
    const wp::int32 var_202 = 1;
    const wp::int32 var_203 = 2;
    const wp::int32 var_204 = 1;
    const wp::int32 var_205 = 1;
    wp::int32 var_206;
    wp::int32 var_207;
    wp::int32 var_208;
    wp::int32 var_209;
    wp::int32 var_210;
    wp::int32 var_211;
    wp::int32 var_212;
    wp::int32 var_213;
    wp::int32 var_214;
    wp::int32 var_215;
    const wp::int32 var_216 = 4;
    const wp::int32 var_217 = 3;
    wp::int32 var_218;
    const wp::int32 var_219 = 0;
    wp::int32 var_220;
    wp::int32* var_221;
    wp::int32 var_222;
    wp::int32 var_223;
    const wp::int32 var_224 = 0;
    bool var_225;
    const wp::int32 var_226 = 3;
    wp::int32 var_227;
    const wp::int32 var_228 = 1;
    wp::int32 var_229;
    wp::int32* var_230;
    wp::int32 var_231;
    wp::int32 var_232;
    const wp::int32 var_233 = 3;
    wp::int32 var_234;
    const wp::int32 var_235 = 2;
    wp::int32 var_236;
    wp::int32* var_237;
    wp::int32 var_238;
    wp::int32 var_239;
    wp::int32 var_240;
    wp::int32 var_241;
    wp::int32 var_242;
    bool var_243;
    const wp::int32 var_244 = 0;
    bool var_245;
    const wp::int32 var_246 = 0;
    bool var_247;
    const wp::int32 var_248 = 0;
    bool var_249;
    bool var_250;
    wp::int32 var_251;
    const wp::int32 var_252 = 0;
    const wp::int32 var_253 = 1;
    const wp::int32 var_254 = 2;
    const wp::int32 var_255 = 1;
    const wp::int32 var_256 = 1;
    wp::int32 var_257;
    wp::int32 var_258;
    wp::int32 var_259;
    wp::int32 var_260;
    wp::int32 var_261;
    wp::int32 var_262;
    wp::int32 var_263;
    wp::int32 var_264;
    wp::int32 var_265;
    wp::int32 var_266;
    wp::int32 var_267;
    //---------
    // forward
    // def _write_cube_fixed_slots(                                                           <L 299>
    // base = cube_flat * MC_MAX_TRIS_PER_CASE                                                <L 316>
    var_1 = wp::mul(var_cube_flat, var_0);
    // for local in range(MC_MAX_TRIS_PER_CASE):                                              <L 317>
    // _clear_fixed_slot(base + local, slot_tri_indices, slot_active, slot_to_compact)        <L 318>
    var_3 = wp::add(var_1, var_2);
    _clear_fixed_slot_0(var_3, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_5 = wp::add(var_1, var_4);
    _clear_fixed_slot_0(var_5, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_7 = wp::add(var_1, var_6);
    _clear_fixed_slot_0(var_7, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_9 = wp::add(var_1, var_8);
    _clear_fixed_slot_0(var_9, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_11 = wp::add(var_1, var_10);
    _clear_fixed_slot_0(var_11, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    // out_count = int(0)                                                                     <L 320>
    var_13 = wp::int(var_12);
    // if case != 0 and case != 255:                                                          <L 321>
    var_16 = (var_case != var_15);
    var_14 = var_16;
    if (var_14) {
        var_18 = (var_case != var_17);
        var_14 = var_14 && var_18;
    }
    if (var_14) {
        // for t in range(MC_MAX_TRIS_PER_CASE):                                              <L 322>
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_21 = wp::mul(var_19, var_20);
        var_23 = wp::add(var_21, var_22);
        var_24 = wp::address(var_case_triangles, var_case, var_23);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // if e0 >= 0:                                                                        <L 324>
        var_28 = (var_25 >= var_27);
        if (var_28) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_30 = wp::mul(var_19, var_29);
            var_32 = wp::add(var_30, var_31);
            var_33 = wp::address(var_case_triangles, var_case, var_32);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_37 = wp::mul(var_19, var_36);
            var_39 = wp::add(var_37, var_38);
            var_40 = wp::address(var_case_triangles, var_case, var_39);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_43 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_25);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_44 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_34);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_45 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_41);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_48 = (var_43 >= var_47);
            var_46 = var_48;
            if (var_46) {
                var_50 = (var_44 >= var_49);
                var_46 = var_46 && var_50;
            }
            if (var_46) {
                var_52 = (var_45 >= var_51);
                var_46 = var_46 && var_52;
            }
            if (var_46) {
                var_53 = (var_13 < var_0);
                var_46 = var_46 && var_53;
            }
            if (var_46) {
                // slot = base + out_count                                                    <L 361>
                var_54 = wp::add(var_1, var_13);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                wp::array_store(var_slot_tri_indices, var_54, var_55, var_43);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                wp::array_store(var_slot_tri_indices, var_54, var_56, var_44);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                wp::array_store(var_slot_tri_indices, var_54, var_57, var_45);
                // slot_active[slot] = 1                                                      <L 365>
                wp::array_store(var_slot_active, var_54, var_58);
                // out_count = out_count + 1                                                  <L 366>
                var_60 = wp::add(var_13, var_59);
            }
            var_61 = wp::where(var_46, var_60, var_13);
        }
        var_62 = wp::where(var_28, var_61, var_13);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_65 = wp::mul(var_63, var_64);
        var_67 = wp::add(var_65, var_66);
        var_68 = wp::address(var_case_triangles, var_case, var_67);
        var_70 = wp::load(var_68);
        var_69 = wp::copy(var_70);
        // if e0 >= 0:                                                                        <L 324>
        var_72 = (var_69 >= var_71);
        if (var_72) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_74 = wp::mul(var_63, var_73);
            var_76 = wp::add(var_74, var_75);
            var_77 = wp::address(var_case_triangles, var_case, var_76);
            var_79 = wp::load(var_77);
            var_78 = wp::copy(var_79);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_81 = wp::mul(var_63, var_80);
            var_83 = wp::add(var_81, var_82);
            var_84 = wp::address(var_case_triangles, var_case, var_83);
            var_86 = wp::load(var_84);
            var_85 = wp::copy(var_86);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_87 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_69);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_88 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_78);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_89 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_85);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_92 = (var_87 >= var_91);
            var_90 = var_92;
            if (var_90) {
                var_94 = (var_88 >= var_93);
                var_90 = var_90 && var_94;
            }
            if (var_90) {
                var_96 = (var_89 >= var_95);
                var_90 = var_90 && var_96;
            }
            if (var_90) {
                var_97 = (var_62 < var_0);
                var_90 = var_90 && var_97;
            }
            if (var_90) {
                // slot = base + out_count                                                    <L 361>
                var_98 = wp::add(var_1, var_62);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                wp::array_store(var_slot_tri_indices, var_98, var_99, var_87);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                wp::array_store(var_slot_tri_indices, var_98, var_100, var_88);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                wp::array_store(var_slot_tri_indices, var_98, var_101, var_89);
                // slot_active[slot] = 1                                                      <L 365>
                wp::array_store(var_slot_active, var_98, var_102);
                // out_count = out_count + 1                                                  <L 366>
                var_104 = wp::add(var_62, var_103);
            }
            var_105 = wp::where(var_90, var_104, var_62);
            var_106 = wp::where(var_90, var_98, var_54);
        }
        var_107 = wp::where(var_72, var_105, var_62);
        var_108 = wp::where(var_72, var_78, var_34);
        var_109 = wp::where(var_72, var_85, var_41);
        var_110 = wp::where(var_72, var_87, var_43);
        var_111 = wp::where(var_72, var_88, var_44);
        var_112 = wp::where(var_72, var_89, var_45);
        var_113 = wp::where(var_72, var_106, var_54);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_116 = wp::mul(var_114, var_115);
        var_118 = wp::add(var_116, var_117);
        var_119 = wp::address(var_case_triangles, var_case, var_118);
        var_121 = wp::load(var_119);
        var_120 = wp::copy(var_121);
        // if e0 >= 0:                                                                        <L 324>
        var_123 = (var_120 >= var_122);
        if (var_123) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_125 = wp::mul(var_114, var_124);
            var_127 = wp::add(var_125, var_126);
            var_128 = wp::address(var_case_triangles, var_case, var_127);
            var_130 = wp::load(var_128);
            var_129 = wp::copy(var_130);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_132 = wp::mul(var_114, var_131);
            var_134 = wp::add(var_132, var_133);
            var_135 = wp::address(var_case_triangles, var_case, var_134);
            var_137 = wp::load(var_135);
            var_136 = wp::copy(var_137);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_138 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_120);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_139 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_129);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_140 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_136);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_143 = (var_138 >= var_142);
            var_141 = var_143;
            if (var_141) {
                var_145 = (var_139 >= var_144);
                var_141 = var_141 && var_145;
            }
            if (var_141) {
                var_147 = (var_140 >= var_146);
                var_141 = var_141 && var_147;
            }
            if (var_141) {
                var_148 = (var_107 < var_0);
                var_141 = var_141 && var_148;
            }
            if (var_141) {
                // slot = base + out_count                                                    <L 361>
                var_149 = wp::add(var_1, var_107);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                wp::array_store(var_slot_tri_indices, var_149, var_150, var_138);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                wp::array_store(var_slot_tri_indices, var_149, var_151, var_139);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                wp::array_store(var_slot_tri_indices, var_149, var_152, var_140);
                // slot_active[slot] = 1                                                      <L 365>
                wp::array_store(var_slot_active, var_149, var_153);
                // out_count = out_count + 1                                                  <L 366>
                var_155 = wp::add(var_107, var_154);
            }
            var_156 = wp::where(var_141, var_155, var_107);
            var_157 = wp::where(var_141, var_149, var_113);
        }
        var_158 = wp::where(var_123, var_156, var_107);
        var_159 = wp::where(var_123, var_129, var_108);
        var_160 = wp::where(var_123, var_136, var_109);
        var_161 = wp::where(var_123, var_138, var_110);
        var_162 = wp::where(var_123, var_139, var_111);
        var_163 = wp::where(var_123, var_140, var_112);
        var_164 = wp::where(var_123, var_157, var_113);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_167 = wp::mul(var_165, var_166);
        var_169 = wp::add(var_167, var_168);
        var_170 = wp::address(var_case_triangles, var_case, var_169);
        var_172 = wp::load(var_170);
        var_171 = wp::copy(var_172);
        // if e0 >= 0:                                                                        <L 324>
        var_174 = (var_171 >= var_173);
        if (var_174) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_176 = wp::mul(var_165, var_175);
            var_178 = wp::add(var_176, var_177);
            var_179 = wp::address(var_case_triangles, var_case, var_178);
            var_181 = wp::load(var_179);
            var_180 = wp::copy(var_181);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_183 = wp::mul(var_165, var_182);
            var_185 = wp::add(var_183, var_184);
            var_186 = wp::address(var_case_triangles, var_case, var_185);
            var_188 = wp::load(var_186);
            var_187 = wp::copy(var_188);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_189 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_171);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_190 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_180);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_191 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_187);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_194 = (var_189 >= var_193);
            var_192 = var_194;
            if (var_192) {
                var_196 = (var_190 >= var_195);
                var_192 = var_192 && var_196;
            }
            if (var_192) {
                var_198 = (var_191 >= var_197);
                var_192 = var_192 && var_198;
            }
            if (var_192) {
                var_199 = (var_158 < var_0);
                var_192 = var_192 && var_199;
            }
            if (var_192) {
                // slot = base + out_count                                                    <L 361>
                var_200 = wp::add(var_1, var_158);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                wp::array_store(var_slot_tri_indices, var_200, var_201, var_189);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                wp::array_store(var_slot_tri_indices, var_200, var_202, var_190);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                wp::array_store(var_slot_tri_indices, var_200, var_203, var_191);
                // slot_active[slot] = 1                                                      <L 365>
                wp::array_store(var_slot_active, var_200, var_204);
                // out_count = out_count + 1                                                  <L 366>
                var_206 = wp::add(var_158, var_205);
            }
            var_207 = wp::where(var_192, var_206, var_158);
            var_208 = wp::where(var_192, var_200, var_164);
        }
        var_209 = wp::where(var_174, var_207, var_158);
        var_210 = wp::where(var_174, var_180, var_159);
        var_211 = wp::where(var_174, var_187, var_160);
        var_212 = wp::where(var_174, var_189, var_161);
        var_213 = wp::where(var_174, var_190, var_162);
        var_214 = wp::where(var_174, var_191, var_163);
        var_215 = wp::where(var_174, var_208, var_164);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_218 = wp::mul(var_216, var_217);
        var_220 = wp::add(var_218, var_219);
        var_221 = wp::address(var_case_triangles, var_case, var_220);
        var_223 = wp::load(var_221);
        var_222 = wp::copy(var_223);
        // if e0 >= 0:                                                                        <L 324>
        var_225 = (var_222 >= var_224);
        if (var_225) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_227 = wp::mul(var_216, var_226);
            var_229 = wp::add(var_227, var_228);
            var_230 = wp::address(var_case_triangles, var_case, var_229);
            var_232 = wp::load(var_230);
            var_231 = wp::copy(var_232);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_234 = wp::mul(var_216, var_233);
            var_236 = wp::add(var_234, var_235);
            var_237 = wp::address(var_case_triangles, var_case, var_236);
            var_239 = wp::load(var_237);
            var_238 = wp::copy(var_239);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_240 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_222);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_241 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_231);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_242 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_238);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_245 = (var_240 >= var_244);
            var_243 = var_245;
            if (var_243) {
                var_247 = (var_241 >= var_246);
                var_243 = var_243 && var_247;
            }
            if (var_243) {
                var_249 = (var_242 >= var_248);
                var_243 = var_243 && var_249;
            }
            if (var_243) {
                var_250 = (var_209 < var_0);
                var_243 = var_243 && var_250;
            }
            if (var_243) {
                // slot = base + out_count                                                    <L 361>
                var_251 = wp::add(var_1, var_209);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                wp::array_store(var_slot_tri_indices, var_251, var_252, var_240);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                wp::array_store(var_slot_tri_indices, var_251, var_253, var_241);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                wp::array_store(var_slot_tri_indices, var_251, var_254, var_242);
                // slot_active[slot] = 1                                                      <L 365>
                wp::array_store(var_slot_active, var_251, var_255);
                // out_count = out_count + 1                                                  <L 366>
                var_257 = wp::add(var_209, var_256);
            }
            var_258 = wp::where(var_243, var_257, var_209);
            var_259 = wp::where(var_243, var_251, var_215);
        }
        var_260 = wp::where(var_225, var_258, var_209);
        var_261 = wp::where(var_225, var_231, var_210);
        var_262 = wp::where(var_225, var_238, var_211);
        var_263 = wp::where(var_225, var_240, var_212);
        var_264 = wp::where(var_225, var_241, var_213);
        var_265 = wp::where(var_225, var_242, var_214);
        var_266 = wp::where(var_225, var_259, var_215);
    }
    var_267 = wp::where(var_14, var_260, var_13);
    // cube_tri_counts[cube_flat] = out_count                                                 <L 368>
    wp::array_store(var_cube_tri_counts, var_cube_flat, var_267);
    // return out_count                                                                       <L 369>
    return var_267;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:72
static CUDA_CALLABLE void adj__corner_particle_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_c,
    wp::array_t<wp::int32> & adj_grid_to_particle,
    wp::array_t<wp::int32> & adj_corner_offsets,
    wp::int32 & adj_cx,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_c,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32* var_9;
    wp::int32 var_10;
    wp::int32 var_11;
    wp::int32 var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::int32* var_15;
    wp::int32 var_16;
    wp::int32 var_17;
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
    //---------
    // forward
    // def _corner_particle(                                                                  <L 73>
    // ox = corner_offsets[c, 0]                                                              <L 78>
    var_1 = wp::address(var_corner_offsets, var_c, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // oy = corner_offsets[c, 1]                                                              <L 79>
    var_5 = wp::address(var_corner_offsets, var_c, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // oz = corner_offsets[c, 2]                                                              <L 80>
    var_9 = wp::address(var_corner_offsets, var_c, var_8);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // return grid_to_particle[cx + ox, cy + oy, cz + oz]                                     <L 81>
    var_12 = wp::add(var_cx, var_2);
    var_13 = wp::add(var_cy, var_6);
    var_14 = wp::add(var_cz, var_10);
    var_15 = wp::address(var_grid_to_particle, var_12, var_13, var_14);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_16 += adj_ret;
    wp::adj_copy(var_17, adj_15, adj_16);
    wp::adj_address(var_grid_to_particle, var_12, var_13, var_14, adj_grid_to_particle, adj_12, adj_13, adj_14, adj_15);
    wp::adj_add(var_cz, var_10, adj_cz, adj_10, adj_14);
    wp::adj_add(var_cy, var_6, adj_cy, adj_6, adj_13);
    wp::adj_add(var_cx, var_2, adj_cx, adj_2, adj_12);
    // adj: return grid_to_particle[cx + ox, cy + oy, cz + oz]                                <L 81>
    wp::adj_copy(var_11, adj_9, adj_10);
    wp::adj_address(var_corner_offsets, var_c, var_8, adj_corner_offsets, adj_c, adj_8, adj_9);
    // adj: oz = corner_offsets[c, 2]                                                         <L 80>
    wp::adj_copy(var_7, adj_5, adj_6);
    wp::adj_address(var_corner_offsets, var_c, var_4, adj_corner_offsets, adj_c, adj_4, adj_5);
    // adj: oy = corner_offsets[c, 1]                                                         <L 79>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_corner_offsets, var_c, var_0, adj_corner_offsets, adj_c, adj_0, adj_1);
    // adj: ox = corner_offsets[c, 0]                                                         <L 78>
    // adj: def _corner_particle(                                                             <L 73>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:84
static CUDA_CALLABLE void adj__corner_active_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_c,
    wp::array_t<wp::int32> & adj_grid_to_particle,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::array_t<wp::int32> & adj_corner_offsets,
    wp::int32 & adj_cx,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_c,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 0;
    bool var_2;
    const wp::int32 var_3 = 0;
    wp::int32* var_4;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    const wp::int32 var_10 = 0;
    bool var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 1;
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
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    bool adj_11 = {};
    wp::int32 adj_12 = {};
    wp::int32 adj_13 = {};
    //---------
    // forward
    // def _corner_active(                                                                    <L 85>
    // p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, c)                  <L 91>
    var_0 = _corner_particle_0(var_grid_to_particle, var_corner_offsets, var_cx, var_cy, var_cz, var_c);
    // if p < 0:                                                                              <L 92>
    var_2 = (var_0 < var_1);
    if (var_2) {
        // return 0                                                                           <L 93>
        goto label0;
    }
    // if (particle_flags[p] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 94>
    var_4 = wp::address(var_particle_flags, var_0);
    var_7 = wp::int32(var_6);
    var_9 = wp::load(var_4);
    var_8 = wp::bit_and(var_9, var_7);
    var_11 = (var_8 == var_10);
    if (var_11) {
        // return 0                                                                           <L 95>
        goto label1;
    }
    // return 1                                                                               <L 96>
    goto label2;
    //---------
    // reverse
    label2:;
    adj_13 += adj_ret;
    // adj: return 1                                                                          <L 96>
    if (var_11) {
        label1:;
        adj_12 += adj_ret;
        // adj: return 0                                                                      <L 95>
    }
    wp::adj_int32(var_6, adj_6, adj_7);
    wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_4);
    // adj: if (particle_flags[p] & wp.int32(ParticleFlags.ACTIVE)) == 0:                     <L 94>
    if (var_2) {
        label0:;
        adj_3 += adj_ret;
        // adj: return 0                                                                      <L 93>
    }
    // adj: if p < 0:                                                                         <L 92>
    adj__corner_particle_0(var_grid_to_particle, var_corner_offsets, var_cx, var_cy, var_cz, var_c, adj_grid_to_particle, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_c, adj_0);
    // adj: p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, c)             <L 91>
    // adj: def _corner_active(                                                               <L 85>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:261
static CUDA_CALLABLE void adj__edge_to_vertex_id_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_edge,
    wp::array_t<wp::int32> & adj_grid_to_particle,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::array_t<wp::int32> & adj_corner_offsets,
    wp::array_t<wp::int32> & adj_edge_corners,
    wp::array_t<wp::int32> & adj_edge_base_dir,
    wp::int32 & adj_cx,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_edge,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32* var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    bool var_10;
    const wp::int32 var_11 = 0;
    bool var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::int32 var_15 = -1;
    bool var_16;
    const wp::int32 var_17 = 0;
    bool var_18;
    const wp::int32 var_19 = 0;
    bool var_20;
    const wp::int32 var_21 = -1;
    wp::int32 var_22;
    wp::int32* var_23;
    wp::int32 var_24;
    wp::int32 var_25;
    const wp::int32 var_26 = 0;
    bool var_27;
    wp::int32 var_28;
    const wp::int32 var_29 = 1;
    wp::int32 var_30;
    wp::int32 var_31;
    wp::int32 var_32;
    wp::int32 var_33;
    const wp::int32 var_34 = 0;
    bool var_35;
    const wp::int32 var_36 = -1;
    const wp::int32 var_37 = 6;
    wp::int32 var_38;
    wp::int32 var_39;
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
    bool adj_10 = {};
    wp::int32 adj_11 = {};
    bool adj_12 = {};
    wp::int32 adj_13 = {};
    bool adj_14 = {};
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
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    bool adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::int32 adj_30 = {};
    wp::int32 adj_31 = {};
    wp::int32 adj_32 = {};
    wp::int32 adj_33 = {};
    wp::int32 adj_34 = {};
    bool adj_35 = {};
    wp::int32 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::int32 adj_38 = {};
    wp::int32 adj_39 = {};
    //---------
    // forward
    // def _edge_to_vertex_id(                                                                <L 262>
    // a = edge_corners[edge, 0]                                                              <L 276>
    var_1 = wp::address(var_edge_corners, var_edge, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // b = edge_corners[edge, 1]                                                              <L 277>
    var_5 = wp::address(var_edge_corners, var_edge, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // a_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, a)       <L 278>
    var_8 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_2);
    // b_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, b)       <L 279>
    var_9 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_6);
    // if a_active == 0 and b_active == 0:                                                    <L 283>
    var_12 = (var_8 == var_11);
    var_10 = var_12;
    if (var_10) {
        var_14 = (var_9 == var_13);
        var_10 = var_10 && var_14;
    }
    if (var_10) {
        // return -1                                                                          <L 284>
        goto label0;
    }
    // if a_active != 0 and b_active != 0:                                                    <L 285>
    var_18 = (var_8 != var_17);
    var_16 = var_18;
    if (var_16) {
        var_20 = (var_9 != var_19);
        var_16 = var_16 && var_20;
    }
    if (var_16) {
        // return -1                                                                          <L 286>
        goto label1;
    }
    // active_corner = a                                                                      <L 287>
    var_22 = wp::copy(var_2);
    // d = edge_base_dir[edge]                                                                <L 288>
    var_23 = wp::address(var_edge_base_dir, var_edge);
    var_25 = wp::load(var_23);
    var_24 = wp::copy(var_25);
    // if a_active == 0:                                                                      <L 289>
    var_27 = (var_8 == var_26);
    if (var_27) {
        // active_corner = b                                                                  <L 290>
        var_28 = wp::copy(var_6);
        // d = d - 1                                                                          <L 291>
        var_30 = wp::sub(var_24, var_29);
    }
    var_31 = wp::where(var_27, var_28, var_22);
    var_32 = wp::where(var_27, var_30, var_24);
    // p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, active_corner)       <L 292>
    var_33 = _corner_particle_0(var_grid_to_particle, var_corner_offsets, var_cx, var_cy, var_cz, var_31);
    // if p < 0:                                                                              <L 293>
    var_35 = (var_33 < var_34);
    if (var_35) {
        // return -1                                                                          <L 294>
        goto label2;
    }
    // return p * 6 + d                                                                       <L 295>
    var_38 = wp::mul(var_33, var_37);
    var_39 = wp::add(var_38, var_32);
    goto label3;
    //---------
    // reverse
    label3:;
    adj_39 += adj_ret;
    wp::adj_add(var_38, var_32, adj_38, adj_32, adj_39);
    wp::adj_mul(var_33, var_37, adj_33, adj_37, adj_38);
    // adj: return p * 6 + d                                                                  <L 295>
    if (var_35) {
        label2:;
        adj_36 += adj_ret;
        // adj: return -1                                                                     <L 294>
    }
    // adj: if p < 0:                                                                         <L 293>
    adj__corner_particle_0(var_grid_to_particle, var_corner_offsets, var_cx, var_cy, var_cz, var_31, adj_grid_to_particle, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_31, adj_33);
    // adj: p = _corner_particle(grid_to_particle, corner_offsets, cx, cy, cz, active_corner)  <L 292>
    wp::adj_where(var_27, var_30, var_24, adj_27, adj_30, adj_24, adj_32);
    wp::adj_where(var_27, var_28, var_22, adj_27, adj_28, adj_22, adj_31);
    if (var_27) {
        wp::adj_sub(var_24, var_29, adj_24, adj_29, adj_30);
        // adj: d = d - 1                                                                     <L 291>
        wp::adj_copy(var_6, adj_6, adj_28);
        // adj: active_corner = b                                                             <L 290>
    }
    // adj: if a_active == 0:                                                                 <L 289>
    wp::adj_copy(var_25, adj_23, adj_24);
    wp::adj_address(var_edge_base_dir, var_edge, adj_edge_base_dir, adj_edge, adj_23);
    // adj: d = edge_base_dir[edge]                                                           <L 288>
    wp::adj_copy(var_2, adj_2, adj_22);
    // adj: active_corner = a                                                                 <L 287>
    if (var_16) {
        label1:;
        adj_21 += adj_ret;
        // adj: return -1                                                                     <L 286>
    }
    if (var_16) {
    }
    // adj: if a_active != 0 and b_active != 0:                                               <L 285>
    if (var_10) {
        label0:;
        adj_15 += adj_ret;
        // adj: return -1                                                                     <L 284>
    }
    if (var_10) {
    }
    // adj: if a_active == 0 and b_active == 0:                                               <L 283>
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_6, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_6, adj_9);
    // adj: b_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, b)  <L 279>
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_2, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_2, adj_8);
    // adj: a_active = _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, a)  <L 278>
    wp::adj_copy(var_7, adj_5, adj_6);
    wp::adj_address(var_edge_corners, var_edge, var_4, adj_edge_corners, adj_edge, adj_4, adj_5);
    // adj: b = edge_corners[edge, 1]                                                         <L 277>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_edge_corners, var_edge, var_0, adj_edge_corners, adj_edge, adj_0, adj_1);
    // adj: a = edge_corners[edge, 0]                                                         <L 276>
    // adj: def _edge_to_vertex_id(                                                           <L 262>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:160
static CUDA_CALLABLE void adj__cube_flat_id_0(
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::int32 & adj_cx,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_ny_cells,
    wp::int32 & adj_nz_cells,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::int32 var_2;
    wp::int32 var_3;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    //---------
    // forward
    // def _cube_flat_id(cx: int, cy: int, cz: int, ny_cells: int, nz_cells: int) -> int:       <L 161>
    // return (cx * ny_cells + cy) * nz_cells + cz                                            <L 162>
    var_0 = wp::mul(var_cx, var_ny_cells);
    var_1 = wp::add(var_0, var_cy);
    var_2 = wp::mul(var_1, var_nz_cells);
    var_3 = wp::add(var_2, var_cz);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_3 += adj_ret;
    wp::adj_add(var_2, var_cz, adj_2, adj_cz, adj_3);
    wp::adj_mul(var_1, var_nz_cells, adj_1, adj_nz_cells, adj_2);
    wp::adj_add(var_0, var_cy, adj_0, adj_cy, adj_1);
    wp::adj_mul(var_cx, var_ny_cells, adj_cx, adj_ny_cells, adj_0);
    // adj: return (cx * ny_cells + cy) * nz_cells + cz                                       <L 162>
    // adj: def _cube_flat_id(cx: int, cy: int, cz: int, ny_cells: int, nz_cells: int) -> int:  <L 161>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:144
static CUDA_CALLABLE void adj__compute_cube_case_0(
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::array_t<wp::int32> & adj_grid_to_particle,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::array_t<wp::int32> & adj_corner_offsets,
    wp::int32 & adj_cx,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 0;
    bool var_5;
    const wp::int32 var_6 = 1;
    wp::int32 var_7;
    wp::int32 var_8;
    wp::int32 var_9;
    wp::int32 var_10;
    const wp::int32 var_11 = 1;
    wp::int32 var_12;
    const wp::int32 var_13 = 0;
    bool var_14;
    const wp::int32 var_15 = 1;
    wp::int32 var_16;
    wp::int32 var_17;
    wp::int32 var_18;
    wp::int32 var_19;
    const wp::int32 var_20 = 2;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    bool var_23;
    const wp::int32 var_24 = 1;
    wp::int32 var_25;
    wp::int32 var_26;
    wp::int32 var_27;
    wp::int32 var_28;
    const wp::int32 var_29 = 3;
    wp::int32 var_30;
    const wp::int32 var_31 = 0;
    bool var_32;
    const wp::int32 var_33 = 1;
    wp::int32 var_34;
    wp::int32 var_35;
    wp::int32 var_36;
    wp::int32 var_37;
    const wp::int32 var_38 = 4;
    wp::int32 var_39;
    const wp::int32 var_40 = 0;
    bool var_41;
    const wp::int32 var_42 = 1;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    wp::int32 var_46;
    const wp::int32 var_47 = 5;
    wp::int32 var_48;
    const wp::int32 var_49 = 0;
    bool var_50;
    const wp::int32 var_51 = 1;
    wp::int32 var_52;
    wp::int32 var_53;
    wp::int32 var_54;
    wp::int32 var_55;
    const wp::int32 var_56 = 6;
    wp::int32 var_57;
    const wp::int32 var_58 = 0;
    bool var_59;
    const wp::int32 var_60 = 1;
    wp::int32 var_61;
    wp::int32 var_62;
    wp::int32 var_63;
    wp::int32 var_64;
    const wp::int32 var_65 = 7;
    wp::int32 var_66;
    const wp::int32 var_67 = 0;
    bool var_68;
    const wp::int32 var_69 = 1;
    wp::int32 var_70;
    wp::int32 var_71;
    wp::int32 var_72;
    wp::int32 var_73;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    bool adj_5 = {};
    wp::int32 adj_6 = {};
    wp::int32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::int32 adj_9 = {};
    wp::int32 adj_10 = {};
    wp::int32 adj_11 = {};
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
    bool adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
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
    wp::int32 adj_38 = {};
    wp::int32 adj_39 = {};
    wp::int32 adj_40 = {};
    bool adj_41 = {};
    wp::int32 adj_42 = {};
    wp::int32 adj_43 = {};
    wp::int32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::int32 adj_48 = {};
    wp::int32 adj_49 = {};
    bool adj_50 = {};
    wp::int32 adj_51 = {};
    wp::int32 adj_52 = {};
    wp::int32 adj_53 = {};
    wp::int32 adj_54 = {};
    wp::int32 adj_55 = {};
    wp::int32 adj_56 = {};
    wp::int32 adj_57 = {};
    wp::int32 adj_58 = {};
    bool adj_59 = {};
    wp::int32 adj_60 = {};
    wp::int32 adj_61 = {};
    wp::int32 adj_62 = {};
    wp::int32 adj_63 = {};
    wp::int32 adj_64 = {};
    wp::int32 adj_65 = {};
    wp::int32 adj_66 = {};
    wp::int32 adj_67 = {};
    bool adj_68 = {};
    wp::int32 adj_69 = {};
    wp::int32 adj_70 = {};
    wp::int32 adj_71 = {};
    wp::int32 adj_72 = {};
    wp::int32 adj_73 = {};
    //---------
    // forward
    // def _compute_cube_case(                                                                <L 145>
    // mask = int(0)                                                                          <L 153>
    var_1 = wp::int(var_0);
    // for c in range(8):                                                                     <L 154>
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_3 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_2);
    var_5 = (var_3 != var_4);
    if (var_5) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_7 = wp::int(var_6);
        var_8 = wp::lshift(var_7, var_2);
        var_9 = wp::bit_or(var_1, var_8);
    }
    var_10 = wp::where(var_5, var_9, var_1);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_12 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_11);
    var_14 = (var_12 != var_13);
    if (var_14) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_16 = wp::int(var_15);
        var_17 = wp::lshift(var_16, var_11);
        var_18 = wp::bit_or(var_10, var_17);
    }
    var_19 = wp::where(var_14, var_18, var_10);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_21 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_20);
    var_23 = (var_21 != var_22);
    if (var_23) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_25 = wp::int(var_24);
        var_26 = wp::lshift(var_25, var_20);
        var_27 = wp::bit_or(var_19, var_26);
    }
    var_28 = wp::where(var_23, var_27, var_19);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_30 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_29);
    var_32 = (var_30 != var_31);
    if (var_32) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_34 = wp::int(var_33);
        var_35 = wp::lshift(var_34, var_29);
        var_36 = wp::bit_or(var_28, var_35);
    }
    var_37 = wp::where(var_32, var_36, var_28);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_39 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_38);
    var_41 = (var_39 != var_40);
    if (var_41) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_43 = wp::int(var_42);
        var_44 = wp::lshift(var_43, var_38);
        var_45 = wp::bit_or(var_37, var_44);
    }
    var_46 = wp::where(var_41, var_45, var_37);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_48 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_47);
    var_50 = (var_48 != var_49);
    if (var_50) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_52 = wp::int(var_51);
        var_53 = wp::lshift(var_52, var_47);
        var_54 = wp::bit_or(var_46, var_53);
    }
    var_55 = wp::where(var_50, var_54, var_46);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_57 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_56);
    var_59 = (var_57 != var_58);
    if (var_59) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_61 = wp::int(var_60);
        var_62 = wp::lshift(var_61, var_56);
        var_63 = wp::bit_or(var_55, var_62);
    }
    var_64 = wp::where(var_59, var_63, var_55);
    // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 155>
    var_66 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_65);
    var_68 = (var_66 != var_67);
    if (var_68) {
        // mask = mask | (int(1) << c)                                                        <L 156>
        var_70 = wp::int(var_69);
        var_71 = wp::lshift(var_70, var_65);
        var_72 = wp::bit_or(var_64, var_71);
    }
    var_73 = wp::where(var_68, var_72, var_64);
    // return mask                                                                            <L 157>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_73 += adj_ret;
    // adj: return mask                                                                       <L 157>
    wp::adj_where(var_68, var_72, var_64, adj_68, adj_72, adj_64, adj_73);
    if (var_68) {
        wp::adj_int(var_69, adj_69, adj_70);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_65, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_65, adj_66);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_59, var_63, var_55, adj_59, adj_63, adj_55, adj_64);
    if (var_59) {
        wp::adj_int(var_60, adj_60, adj_61);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_56, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_56, adj_57);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_50, var_54, var_46, adj_50, adj_54, adj_46, adj_55);
    if (var_50) {
        wp::adj_int(var_51, adj_51, adj_52);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_47, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_47, adj_48);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_41, var_45, var_37, adj_41, adj_45, adj_37, adj_46);
    if (var_41) {
        wp::adj_int(var_42, adj_42, adj_43);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_38, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_38, adj_39);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_32, var_36, var_28, adj_32, adj_36, adj_28, adj_37);
    if (var_32) {
        wp::adj_int(var_33, adj_33, adj_34);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_29, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_29, adj_30);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_23, var_27, var_19, adj_23, adj_27, adj_19, adj_28);
    if (var_23) {
        wp::adj_int(var_24, adj_24, adj_25);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_20, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_20, adj_21);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_14, var_18, var_10, adj_14, adj_18, adj_10, adj_19);
    if (var_14) {
        wp::adj_int(var_15, adj_15, adj_16);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_11, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_11, adj_12);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    wp::adj_where(var_5, var_9, var_1, adj_5, adj_9, adj_1, adj_10);
    if (var_5) {
        wp::adj_int(var_6, adj_6, adj_7);
        // adj: mask = mask | (int(1) << c)                                                   <L 156>
    }
    adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_cx, var_cy, var_cz, var_2, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_cx, adj_cy, adj_cz, adj_2, adj_3);
    // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 155>
    // adj: for c in range(8):                                                                <L 154>
    wp::adj_int(var_0, adj_0, adj_1);
    // adj: mask = int(0)                                                                     <L 153>
    // adj: def _compute_cube_case(                                                           <L 145>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:165
static CUDA_CALLABLE void adj__clear_fixed_slot_0(
    wp::int32 var_slot,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::int32 & adj_slot,
    wp::array_t<wp::int32> & adj_slot_tri_indices,
    wp::array_t<wp::int32> & adj_slot_active,
    wp::array_t<wp::int32> & adj_slot_to_compact)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 0;
    const wp::int32 var_1 = -1;
    const wp::int32 var_2 = 0;
    const wp::int32 var_3 = 0;
    const wp::int32 var_4 = 0;
    const wp::int32 var_5 = 1;
    const wp::int32 var_6 = 0;
    const wp::int32 var_7 = 2;
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
    //---------
    // forward
    // def _clear_fixed_slot(                                                                 <L 166>
    // slot_active[slot] = 0                                                                  <L 172>
    // wp::array_store(var_slot_active, var_slot, var_0);
    // slot_to_compact[slot] = -1                                                             <L 173>
    // wp::array_store(var_slot_to_compact, var_slot, var_1);
    // slot_tri_indices[slot, 0] = 0                                                          <L 174>
    // wp::array_store(var_slot_tri_indices, var_slot, var_3, var_2);
    // slot_tri_indices[slot, 1] = 0                                                          <L 175>
    // wp::array_store(var_slot_tri_indices, var_slot, var_5, var_4);
    // slot_tri_indices[slot, 2] = 0                                                          <L 176>
    // wp::array_store(var_slot_tri_indices, var_slot, var_7, var_6);
    //---------
    // reverse
    wp::adj_array_store(var_slot_tri_indices, var_slot, var_7, var_6, adj_slot_tri_indices, adj_slot, adj_7, adj_6);
    // adj: slot_tri_indices[slot, 2] = 0                                                     <L 176>
    wp::adj_array_store(var_slot_tri_indices, var_slot, var_5, var_4, adj_slot_tri_indices, adj_slot, adj_5, adj_4);
    // adj: slot_tri_indices[slot, 1] = 0                                                     <L 175>
    wp::adj_array_store(var_slot_tri_indices, var_slot, var_3, var_2, adj_slot_tri_indices, adj_slot, adj_3, adj_2);
    // adj: slot_tri_indices[slot, 0] = 0                                                     <L 174>
    wp::adj_array_store(var_slot_to_compact, var_slot, var_1, adj_slot_to_compact, adj_slot, adj_1);
    // adj: slot_to_compact[slot] = -1                                                        <L 173>
    wp::adj_array_store(var_slot_active, var_slot, var_0, adj_slot_active, adj_slot, adj_0);
    // adj: slot_active[slot] = 0                                                             <L 172>
    // adj: def _clear_fixed_slot(                                                            <L 166>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/marching_cubes.py:298
static CUDA_CALLABLE void adj__write_cube_fixed_slots_0(
    wp::int32 var_cube_flat,
    wp::int32 var_cx,
    wp::int32 var_cy,
    wp::int32 var_cz,
    wp::int32 var_case,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::int32 & adj_cube_flat,
    wp::int32 & adj_cx,
    wp::int32 & adj_cy,
    wp::int32 & adj_cz,
    wp::int32 & adj_case,
    wp::array_t<wp::int32> & adj_grid_to_particle,
    wp::array_t<wp::int32> & adj_particle_flags,
    wp::array_t<wp::int32> & adj_corner_offsets,
    wp::array_t<wp::int32> & adj_edge_corners,
    wp::array_t<wp::int32> & adj_edge_base_dir,
    wp::array_t<wp::int32> & adj_case_triangles,
    wp::array_t<wp::int32> & adj_cube_tri_counts,
    wp::array_t<wp::int32> & adj_slot_tri_indices,
    wp::array_t<wp::int32> & adj_slot_active,
    wp::array_t<wp::int32> & adj_slot_to_compact,
    wp::int32 & adj_ret)
{
    //---------
    // primal vars
    const wp::int32 var_0 = 5;
    wp::int32 var_1;
    const wp::int32 var_2 = 0;
    wp::int32 var_3;
    const wp::int32 var_4 = 1;
    wp::int32 var_5;
    const wp::int32 var_6 = 2;
    wp::int32 var_7;
    const wp::int32 var_8 = 3;
    wp::int32 var_9;
    const wp::int32 var_10 = 4;
    wp::int32 var_11;
    const wp::int32 var_12 = 0;
    wp::int32 var_13;
    bool var_14;
    const wp::int32 var_15 = 0;
    bool var_16;
    const wp::int32 var_17 = 255;
    bool var_18;
    const wp::int32 var_19 = 0;
    const wp::int32 var_20 = 3;
    wp::int32 var_21;
    const wp::int32 var_22 = 0;
    wp::int32 var_23;
    wp::int32* var_24;
    wp::int32 var_25;
    wp::int32 var_26;
    const wp::int32 var_27 = 0;
    bool var_28;
    const wp::int32 var_29 = 3;
    wp::int32 var_30;
    const wp::int32 var_31 = 1;
    wp::int32 var_32;
    wp::int32* var_33;
    wp::int32 var_34;
    wp::int32 var_35;
    const wp::int32 var_36 = 3;
    wp::int32 var_37;
    const wp::int32 var_38 = 2;
    wp::int32 var_39;
    wp::int32* var_40;
    wp::int32 var_41;
    wp::int32 var_42;
    wp::int32 var_43;
    wp::int32 var_44;
    wp::int32 var_45;
    bool var_46;
    const wp::int32 var_47 = 0;
    bool var_48;
    const wp::int32 var_49 = 0;
    bool var_50;
    const wp::int32 var_51 = 0;
    bool var_52;
    bool var_53;
    wp::int32 var_54;
    const wp::int32 var_55 = 0;
    const wp::int32 var_56 = 1;
    const wp::int32 var_57 = 2;
    const wp::int32 var_58 = 1;
    const wp::int32 var_59 = 1;
    wp::int32 var_60;
    wp::int32 var_61;
    wp::int32 var_62;
    const wp::int32 var_63 = 1;
    const wp::int32 var_64 = 3;
    wp::int32 var_65;
    const wp::int32 var_66 = 0;
    wp::int32 var_67;
    wp::int32* var_68;
    wp::int32 var_69;
    wp::int32 var_70;
    const wp::int32 var_71 = 0;
    bool var_72;
    const wp::int32 var_73 = 3;
    wp::int32 var_74;
    const wp::int32 var_75 = 1;
    wp::int32 var_76;
    wp::int32* var_77;
    wp::int32 var_78;
    wp::int32 var_79;
    const wp::int32 var_80 = 3;
    wp::int32 var_81;
    const wp::int32 var_82 = 2;
    wp::int32 var_83;
    wp::int32* var_84;
    wp::int32 var_85;
    wp::int32 var_86;
    wp::int32 var_87;
    wp::int32 var_88;
    wp::int32 var_89;
    bool var_90;
    const wp::int32 var_91 = 0;
    bool var_92;
    const wp::int32 var_93 = 0;
    bool var_94;
    const wp::int32 var_95 = 0;
    bool var_96;
    bool var_97;
    wp::int32 var_98;
    const wp::int32 var_99 = 0;
    const wp::int32 var_100 = 1;
    const wp::int32 var_101 = 2;
    const wp::int32 var_102 = 1;
    const wp::int32 var_103 = 1;
    wp::int32 var_104;
    wp::int32 var_105;
    wp::int32 var_106;
    wp::int32 var_107;
    wp::int32 var_108;
    wp::int32 var_109;
    wp::int32 var_110;
    wp::int32 var_111;
    wp::int32 var_112;
    wp::int32 var_113;
    const wp::int32 var_114 = 2;
    const wp::int32 var_115 = 3;
    wp::int32 var_116;
    const wp::int32 var_117 = 0;
    wp::int32 var_118;
    wp::int32* var_119;
    wp::int32 var_120;
    wp::int32 var_121;
    const wp::int32 var_122 = 0;
    bool var_123;
    const wp::int32 var_124 = 3;
    wp::int32 var_125;
    const wp::int32 var_126 = 1;
    wp::int32 var_127;
    wp::int32* var_128;
    wp::int32 var_129;
    wp::int32 var_130;
    const wp::int32 var_131 = 3;
    wp::int32 var_132;
    const wp::int32 var_133 = 2;
    wp::int32 var_134;
    wp::int32* var_135;
    wp::int32 var_136;
    wp::int32 var_137;
    wp::int32 var_138;
    wp::int32 var_139;
    wp::int32 var_140;
    bool var_141;
    const wp::int32 var_142 = 0;
    bool var_143;
    const wp::int32 var_144 = 0;
    bool var_145;
    const wp::int32 var_146 = 0;
    bool var_147;
    bool var_148;
    wp::int32 var_149;
    const wp::int32 var_150 = 0;
    const wp::int32 var_151 = 1;
    const wp::int32 var_152 = 2;
    const wp::int32 var_153 = 1;
    const wp::int32 var_154 = 1;
    wp::int32 var_155;
    wp::int32 var_156;
    wp::int32 var_157;
    wp::int32 var_158;
    wp::int32 var_159;
    wp::int32 var_160;
    wp::int32 var_161;
    wp::int32 var_162;
    wp::int32 var_163;
    wp::int32 var_164;
    const wp::int32 var_165 = 3;
    const wp::int32 var_166 = 3;
    wp::int32 var_167;
    const wp::int32 var_168 = 0;
    wp::int32 var_169;
    wp::int32* var_170;
    wp::int32 var_171;
    wp::int32 var_172;
    const wp::int32 var_173 = 0;
    bool var_174;
    const wp::int32 var_175 = 3;
    wp::int32 var_176;
    const wp::int32 var_177 = 1;
    wp::int32 var_178;
    wp::int32* var_179;
    wp::int32 var_180;
    wp::int32 var_181;
    const wp::int32 var_182 = 3;
    wp::int32 var_183;
    const wp::int32 var_184 = 2;
    wp::int32 var_185;
    wp::int32* var_186;
    wp::int32 var_187;
    wp::int32 var_188;
    wp::int32 var_189;
    wp::int32 var_190;
    wp::int32 var_191;
    bool var_192;
    const wp::int32 var_193 = 0;
    bool var_194;
    const wp::int32 var_195 = 0;
    bool var_196;
    const wp::int32 var_197 = 0;
    bool var_198;
    bool var_199;
    wp::int32 var_200;
    const wp::int32 var_201 = 0;
    const wp::int32 var_202 = 1;
    const wp::int32 var_203 = 2;
    const wp::int32 var_204 = 1;
    const wp::int32 var_205 = 1;
    wp::int32 var_206;
    wp::int32 var_207;
    wp::int32 var_208;
    wp::int32 var_209;
    wp::int32 var_210;
    wp::int32 var_211;
    wp::int32 var_212;
    wp::int32 var_213;
    wp::int32 var_214;
    wp::int32 var_215;
    const wp::int32 var_216 = 4;
    const wp::int32 var_217 = 3;
    wp::int32 var_218;
    const wp::int32 var_219 = 0;
    wp::int32 var_220;
    wp::int32* var_221;
    wp::int32 var_222;
    wp::int32 var_223;
    const wp::int32 var_224 = 0;
    bool var_225;
    const wp::int32 var_226 = 3;
    wp::int32 var_227;
    const wp::int32 var_228 = 1;
    wp::int32 var_229;
    wp::int32* var_230;
    wp::int32 var_231;
    wp::int32 var_232;
    const wp::int32 var_233 = 3;
    wp::int32 var_234;
    const wp::int32 var_235 = 2;
    wp::int32 var_236;
    wp::int32* var_237;
    wp::int32 var_238;
    wp::int32 var_239;
    wp::int32 var_240;
    wp::int32 var_241;
    wp::int32 var_242;
    bool var_243;
    const wp::int32 var_244 = 0;
    bool var_245;
    const wp::int32 var_246 = 0;
    bool var_247;
    const wp::int32 var_248 = 0;
    bool var_249;
    bool var_250;
    wp::int32 var_251;
    const wp::int32 var_252 = 0;
    const wp::int32 var_253 = 1;
    const wp::int32 var_254 = 2;
    const wp::int32 var_255 = 1;
    const wp::int32 var_256 = 1;
    wp::int32 var_257;
    wp::int32 var_258;
    wp::int32 var_259;
    wp::int32 var_260;
    wp::int32 var_261;
    wp::int32 var_262;
    wp::int32 var_263;
    wp::int32 var_264;
    wp::int32 var_265;
    wp::int32 var_266;
    wp::int32 var_267;
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
    bool adj_14 = {};
    wp::int32 adj_15 = {};
    bool adj_16 = {};
    wp::int32 adj_17 = {};
    bool adj_18 = {};
    wp::int32 adj_19 = {};
    wp::int32 adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::int32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::int32 adj_25 = {};
    wp::int32 adj_26 = {};
    wp::int32 adj_27 = {};
    bool adj_28 = {};
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
    wp::int32 adj_44 = {};
    wp::int32 adj_45 = {};
    bool adj_46 = {};
    wp::int32 adj_47 = {};
    bool adj_48 = {};
    wp::int32 adj_49 = {};
    bool adj_50 = {};
    wp::int32 adj_51 = {};
    bool adj_52 = {};
    bool adj_53 = {};
    wp::int32 adj_54 = {};
    wp::int32 adj_55 = {};
    wp::int32 adj_56 = {};
    wp::int32 adj_57 = {};
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
    wp::int32 adj_74 = {};
    wp::int32 adj_75 = {};
    wp::int32 adj_76 = {};
    wp::int32 adj_77 = {};
    wp::int32 adj_78 = {};
    wp::int32 adj_79 = {};
    wp::int32 adj_80 = {};
    wp::int32 adj_81 = {};
    wp::int32 adj_82 = {};
    wp::int32 adj_83 = {};
    wp::int32 adj_84 = {};
    wp::int32 adj_85 = {};
    wp::int32 adj_86 = {};
    wp::int32 adj_87 = {};
    wp::int32 adj_88 = {};
    wp::int32 adj_89 = {};
    bool adj_90 = {};
    wp::int32 adj_91 = {};
    bool adj_92 = {};
    wp::int32 adj_93 = {};
    bool adj_94 = {};
    wp::int32 adj_95 = {};
    bool adj_96 = {};
    bool adj_97 = {};
    wp::int32 adj_98 = {};
    wp::int32 adj_99 = {};
    wp::int32 adj_100 = {};
    wp::int32 adj_101 = {};
    wp::int32 adj_102 = {};
    wp::int32 adj_103 = {};
    wp::int32 adj_104 = {};
    wp::int32 adj_105 = {};
    wp::int32 adj_106 = {};
    wp::int32 adj_107 = {};
    wp::int32 adj_108 = {};
    wp::int32 adj_109 = {};
    wp::int32 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::int32 adj_112 = {};
    wp::int32 adj_113 = {};
    wp::int32 adj_114 = {};
    wp::int32 adj_115 = {};
    wp::int32 adj_116 = {};
    wp::int32 adj_117 = {};
    wp::int32 adj_118 = {};
    wp::int32 adj_119 = {};
    wp::int32 adj_120 = {};
    wp::int32 adj_121 = {};
    wp::int32 adj_122 = {};
    bool adj_123 = {};
    wp::int32 adj_124 = {};
    wp::int32 adj_125 = {};
    wp::int32 adj_126 = {};
    wp::int32 adj_127 = {};
    wp::int32 adj_128 = {};
    wp::int32 adj_129 = {};
    wp::int32 adj_130 = {};
    wp::int32 adj_131 = {};
    wp::int32 adj_132 = {};
    wp::int32 adj_133 = {};
    wp::int32 adj_134 = {};
    wp::int32 adj_135 = {};
    wp::int32 adj_136 = {};
    wp::int32 adj_137 = {};
    wp::int32 adj_138 = {};
    wp::int32 adj_139 = {};
    wp::int32 adj_140 = {};
    bool adj_141 = {};
    wp::int32 adj_142 = {};
    bool adj_143 = {};
    wp::int32 adj_144 = {};
    bool adj_145 = {};
    wp::int32 adj_146 = {};
    bool adj_147 = {};
    bool adj_148 = {};
    wp::int32 adj_149 = {};
    wp::int32 adj_150 = {};
    wp::int32 adj_151 = {};
    wp::int32 adj_152 = {};
    wp::int32 adj_153 = {};
    wp::int32 adj_154 = {};
    wp::int32 adj_155 = {};
    wp::int32 adj_156 = {};
    wp::int32 adj_157 = {};
    wp::int32 adj_158 = {};
    wp::int32 adj_159 = {};
    wp::int32 adj_160 = {};
    wp::int32 adj_161 = {};
    wp::int32 adj_162 = {};
    wp::int32 adj_163 = {};
    wp::int32 adj_164 = {};
    wp::int32 adj_165 = {};
    wp::int32 adj_166 = {};
    wp::int32 adj_167 = {};
    wp::int32 adj_168 = {};
    wp::int32 adj_169 = {};
    wp::int32 adj_170 = {};
    wp::int32 adj_171 = {};
    wp::int32 adj_172 = {};
    wp::int32 adj_173 = {};
    bool adj_174 = {};
    wp::int32 adj_175 = {};
    wp::int32 adj_176 = {};
    wp::int32 adj_177 = {};
    wp::int32 adj_178 = {};
    wp::int32 adj_179 = {};
    wp::int32 adj_180 = {};
    wp::int32 adj_181 = {};
    wp::int32 adj_182 = {};
    wp::int32 adj_183 = {};
    wp::int32 adj_184 = {};
    wp::int32 adj_185 = {};
    wp::int32 adj_186 = {};
    wp::int32 adj_187 = {};
    wp::int32 adj_188 = {};
    wp::int32 adj_189 = {};
    wp::int32 adj_190 = {};
    wp::int32 adj_191 = {};
    bool adj_192 = {};
    wp::int32 adj_193 = {};
    bool adj_194 = {};
    wp::int32 adj_195 = {};
    bool adj_196 = {};
    wp::int32 adj_197 = {};
    bool adj_198 = {};
    bool adj_199 = {};
    wp::int32 adj_200 = {};
    wp::int32 adj_201 = {};
    wp::int32 adj_202 = {};
    wp::int32 adj_203 = {};
    wp::int32 adj_204 = {};
    wp::int32 adj_205 = {};
    wp::int32 adj_206 = {};
    wp::int32 adj_207 = {};
    wp::int32 adj_208 = {};
    wp::int32 adj_209 = {};
    wp::int32 adj_210 = {};
    wp::int32 adj_211 = {};
    wp::int32 adj_212 = {};
    wp::int32 adj_213 = {};
    wp::int32 adj_214 = {};
    wp::int32 adj_215 = {};
    wp::int32 adj_216 = {};
    wp::int32 adj_217 = {};
    wp::int32 adj_218 = {};
    wp::int32 adj_219 = {};
    wp::int32 adj_220 = {};
    wp::int32 adj_221 = {};
    wp::int32 adj_222 = {};
    wp::int32 adj_223 = {};
    wp::int32 adj_224 = {};
    bool adj_225 = {};
    wp::int32 adj_226 = {};
    wp::int32 adj_227 = {};
    wp::int32 adj_228 = {};
    wp::int32 adj_229 = {};
    wp::int32 adj_230 = {};
    wp::int32 adj_231 = {};
    wp::int32 adj_232 = {};
    wp::int32 adj_233 = {};
    wp::int32 adj_234 = {};
    wp::int32 adj_235 = {};
    wp::int32 adj_236 = {};
    wp::int32 adj_237 = {};
    wp::int32 adj_238 = {};
    wp::int32 adj_239 = {};
    wp::int32 adj_240 = {};
    wp::int32 adj_241 = {};
    wp::int32 adj_242 = {};
    bool adj_243 = {};
    wp::int32 adj_244 = {};
    bool adj_245 = {};
    wp::int32 adj_246 = {};
    bool adj_247 = {};
    wp::int32 adj_248 = {};
    bool adj_249 = {};
    bool adj_250 = {};
    wp::int32 adj_251 = {};
    wp::int32 adj_252 = {};
    wp::int32 adj_253 = {};
    wp::int32 adj_254 = {};
    wp::int32 adj_255 = {};
    wp::int32 adj_256 = {};
    wp::int32 adj_257 = {};
    wp::int32 adj_258 = {};
    wp::int32 adj_259 = {};
    wp::int32 adj_260 = {};
    wp::int32 adj_261 = {};
    wp::int32 adj_262 = {};
    wp::int32 adj_263 = {};
    wp::int32 adj_264 = {};
    wp::int32 adj_265 = {};
    wp::int32 adj_266 = {};
    wp::int32 adj_267 = {};
    //---------
    // forward
    // def _write_cube_fixed_slots(                                                           <L 299>
    // base = cube_flat * MC_MAX_TRIS_PER_CASE                                                <L 316>
    var_1 = wp::mul(var_cube_flat, var_0);
    // for local in range(MC_MAX_TRIS_PER_CASE):                                              <L 317>
    // _clear_fixed_slot(base + local, slot_tri_indices, slot_active, slot_to_compact)        <L 318>
    var_3 = wp::add(var_1, var_2);
    _clear_fixed_slot_0(var_3, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_5 = wp::add(var_1, var_4);
    _clear_fixed_slot_0(var_5, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_7 = wp::add(var_1, var_6);
    _clear_fixed_slot_0(var_7, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_9 = wp::add(var_1, var_8);
    _clear_fixed_slot_0(var_9, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    var_11 = wp::add(var_1, var_10);
    _clear_fixed_slot_0(var_11, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    // out_count = int(0)                                                                     <L 320>
    var_13 = wp::int(var_12);
    // if case != 0 and case != 255:                                                          <L 321>
    var_16 = (var_case != var_15);
    var_14 = var_16;
    if (var_14) {
        var_18 = (var_case != var_17);
        var_14 = var_14 && var_18;
    }
    if (var_14) {
        // for t in range(MC_MAX_TRIS_PER_CASE):                                              <L 322>
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_21 = wp::mul(var_19, var_20);
        var_23 = wp::add(var_21, var_22);
        var_24 = wp::address(var_case_triangles, var_case, var_23);
        var_26 = wp::load(var_24);
        var_25 = wp::copy(var_26);
        // if e0 >= 0:                                                                        <L 324>
        var_28 = (var_25 >= var_27);
        if (var_28) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_30 = wp::mul(var_19, var_29);
            var_32 = wp::add(var_30, var_31);
            var_33 = wp::address(var_case_triangles, var_case, var_32);
            var_35 = wp::load(var_33);
            var_34 = wp::copy(var_35);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_37 = wp::mul(var_19, var_36);
            var_39 = wp::add(var_37, var_38);
            var_40 = wp::address(var_case_triangles, var_case, var_39);
            var_42 = wp::load(var_40);
            var_41 = wp::copy(var_42);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_43 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_25);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_44 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_34);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_45 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_41);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_48 = (var_43 >= var_47);
            var_46 = var_48;
            if (var_46) {
                var_50 = (var_44 >= var_49);
                var_46 = var_46 && var_50;
            }
            if (var_46) {
                var_52 = (var_45 >= var_51);
                var_46 = var_46 && var_52;
            }
            if (var_46) {
                var_53 = (var_13 < var_0);
                var_46 = var_46 && var_53;
            }
            if (var_46) {
                // slot = base + out_count                                                    <L 361>
                var_54 = wp::add(var_1, var_13);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                // wp::array_store(var_slot_tri_indices, var_54, var_55, var_43);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                // wp::array_store(var_slot_tri_indices, var_54, var_56, var_44);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                // wp::array_store(var_slot_tri_indices, var_54, var_57, var_45);
                // slot_active[slot] = 1                                                      <L 365>
                // wp::array_store(var_slot_active, var_54, var_58);
                // out_count = out_count + 1                                                  <L 366>
                var_60 = wp::add(var_13, var_59);
            }
            var_61 = wp::where(var_46, var_60, var_13);
        }
        var_62 = wp::where(var_28, var_61, var_13);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_65 = wp::mul(var_63, var_64);
        var_67 = wp::add(var_65, var_66);
        var_68 = wp::address(var_case_triangles, var_case, var_67);
        var_70 = wp::load(var_68);
        var_69 = wp::copy(var_70);
        // if e0 >= 0:                                                                        <L 324>
        var_72 = (var_69 >= var_71);
        if (var_72) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_74 = wp::mul(var_63, var_73);
            var_76 = wp::add(var_74, var_75);
            var_77 = wp::address(var_case_triangles, var_case, var_76);
            var_79 = wp::load(var_77);
            var_78 = wp::copy(var_79);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_81 = wp::mul(var_63, var_80);
            var_83 = wp::add(var_81, var_82);
            var_84 = wp::address(var_case_triangles, var_case, var_83);
            var_86 = wp::load(var_84);
            var_85 = wp::copy(var_86);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_87 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_69);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_88 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_78);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_89 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_85);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_92 = (var_87 >= var_91);
            var_90 = var_92;
            if (var_90) {
                var_94 = (var_88 >= var_93);
                var_90 = var_90 && var_94;
            }
            if (var_90) {
                var_96 = (var_89 >= var_95);
                var_90 = var_90 && var_96;
            }
            if (var_90) {
                var_97 = (var_62 < var_0);
                var_90 = var_90 && var_97;
            }
            if (var_90) {
                // slot = base + out_count                                                    <L 361>
                var_98 = wp::add(var_1, var_62);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                // wp::array_store(var_slot_tri_indices, var_98, var_99, var_87);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                // wp::array_store(var_slot_tri_indices, var_98, var_100, var_88);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                // wp::array_store(var_slot_tri_indices, var_98, var_101, var_89);
                // slot_active[slot] = 1                                                      <L 365>
                // wp::array_store(var_slot_active, var_98, var_102);
                // out_count = out_count + 1                                                  <L 366>
                var_104 = wp::add(var_62, var_103);
            }
            var_105 = wp::where(var_90, var_104, var_62);
            var_106 = wp::where(var_90, var_98, var_54);
        }
        var_107 = wp::where(var_72, var_105, var_62);
        var_108 = wp::where(var_72, var_78, var_34);
        var_109 = wp::where(var_72, var_85, var_41);
        var_110 = wp::where(var_72, var_87, var_43);
        var_111 = wp::where(var_72, var_88, var_44);
        var_112 = wp::where(var_72, var_89, var_45);
        var_113 = wp::where(var_72, var_106, var_54);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_116 = wp::mul(var_114, var_115);
        var_118 = wp::add(var_116, var_117);
        var_119 = wp::address(var_case_triangles, var_case, var_118);
        var_121 = wp::load(var_119);
        var_120 = wp::copy(var_121);
        // if e0 >= 0:                                                                        <L 324>
        var_123 = (var_120 >= var_122);
        if (var_123) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_125 = wp::mul(var_114, var_124);
            var_127 = wp::add(var_125, var_126);
            var_128 = wp::address(var_case_triangles, var_case, var_127);
            var_130 = wp::load(var_128);
            var_129 = wp::copy(var_130);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_132 = wp::mul(var_114, var_131);
            var_134 = wp::add(var_132, var_133);
            var_135 = wp::address(var_case_triangles, var_case, var_134);
            var_137 = wp::load(var_135);
            var_136 = wp::copy(var_137);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_138 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_120);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_139 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_129);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_140 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_136);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_143 = (var_138 >= var_142);
            var_141 = var_143;
            if (var_141) {
                var_145 = (var_139 >= var_144);
                var_141 = var_141 && var_145;
            }
            if (var_141) {
                var_147 = (var_140 >= var_146);
                var_141 = var_141 && var_147;
            }
            if (var_141) {
                var_148 = (var_107 < var_0);
                var_141 = var_141 && var_148;
            }
            if (var_141) {
                // slot = base + out_count                                                    <L 361>
                var_149 = wp::add(var_1, var_107);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                // wp::array_store(var_slot_tri_indices, var_149, var_150, var_138);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                // wp::array_store(var_slot_tri_indices, var_149, var_151, var_139);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                // wp::array_store(var_slot_tri_indices, var_149, var_152, var_140);
                // slot_active[slot] = 1                                                      <L 365>
                // wp::array_store(var_slot_active, var_149, var_153);
                // out_count = out_count + 1                                                  <L 366>
                var_155 = wp::add(var_107, var_154);
            }
            var_156 = wp::where(var_141, var_155, var_107);
            var_157 = wp::where(var_141, var_149, var_113);
        }
        var_158 = wp::where(var_123, var_156, var_107);
        var_159 = wp::where(var_123, var_129, var_108);
        var_160 = wp::where(var_123, var_136, var_109);
        var_161 = wp::where(var_123, var_138, var_110);
        var_162 = wp::where(var_123, var_139, var_111);
        var_163 = wp::where(var_123, var_140, var_112);
        var_164 = wp::where(var_123, var_157, var_113);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_167 = wp::mul(var_165, var_166);
        var_169 = wp::add(var_167, var_168);
        var_170 = wp::address(var_case_triangles, var_case, var_169);
        var_172 = wp::load(var_170);
        var_171 = wp::copy(var_172);
        // if e0 >= 0:                                                                        <L 324>
        var_174 = (var_171 >= var_173);
        if (var_174) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_176 = wp::mul(var_165, var_175);
            var_178 = wp::add(var_176, var_177);
            var_179 = wp::address(var_case_triangles, var_case, var_178);
            var_181 = wp::load(var_179);
            var_180 = wp::copy(var_181);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_183 = wp::mul(var_165, var_182);
            var_185 = wp::add(var_183, var_184);
            var_186 = wp::address(var_case_triangles, var_case, var_185);
            var_188 = wp::load(var_186);
            var_187 = wp::copy(var_188);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_189 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_171);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_190 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_180);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_191 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_187);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_194 = (var_189 >= var_193);
            var_192 = var_194;
            if (var_192) {
                var_196 = (var_190 >= var_195);
                var_192 = var_192 && var_196;
            }
            if (var_192) {
                var_198 = (var_191 >= var_197);
                var_192 = var_192 && var_198;
            }
            if (var_192) {
                var_199 = (var_158 < var_0);
                var_192 = var_192 && var_199;
            }
            if (var_192) {
                // slot = base + out_count                                                    <L 361>
                var_200 = wp::add(var_1, var_158);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                // wp::array_store(var_slot_tri_indices, var_200, var_201, var_189);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                // wp::array_store(var_slot_tri_indices, var_200, var_202, var_190);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                // wp::array_store(var_slot_tri_indices, var_200, var_203, var_191);
                // slot_active[slot] = 1                                                      <L 365>
                // wp::array_store(var_slot_active, var_200, var_204);
                // out_count = out_count + 1                                                  <L 366>
                var_206 = wp::add(var_158, var_205);
            }
            var_207 = wp::where(var_192, var_206, var_158);
            var_208 = wp::where(var_192, var_200, var_164);
        }
        var_209 = wp::where(var_174, var_207, var_158);
        var_210 = wp::where(var_174, var_180, var_159);
        var_211 = wp::where(var_174, var_187, var_160);
        var_212 = wp::where(var_174, var_189, var_161);
        var_213 = wp::where(var_174, var_190, var_162);
        var_214 = wp::where(var_174, var_191, var_163);
        var_215 = wp::where(var_174, var_208, var_164);
        // e0 = case_triangles[case, t * 3 + 0]                                               <L 323>
        var_218 = wp::mul(var_216, var_217);
        var_220 = wp::add(var_218, var_219);
        var_221 = wp::address(var_case_triangles, var_case, var_220);
        var_223 = wp::load(var_221);
        var_222 = wp::copy(var_223);
        // if e0 >= 0:                                                                        <L 324>
        var_225 = (var_222 >= var_224);
        if (var_225) {
            // e1 = case_triangles[case, t * 3 + 1]                                           <L 325>
            var_227 = wp::mul(var_216, var_226);
            var_229 = wp::add(var_227, var_228);
            var_230 = wp::address(var_case_triangles, var_case, var_229);
            var_232 = wp::load(var_230);
            var_231 = wp::copy(var_232);
            // e2 = case_triangles[case, t * 3 + 2]                                           <L 326>
            var_234 = wp::mul(var_216, var_233);
            var_236 = wp::add(var_234, var_235);
            var_237 = wp::address(var_case_triangles, var_case, var_236);
            var_239 = wp::load(var_237);
            var_238 = wp::copy(var_239);
            // v0 = _edge_to_vertex_id(                                                       <L 327>
            // grid_to_particle,                                                              <L 328>
            // particle_flags,                                                                <L 329>
            // corner_offsets,                                                                <L 330>
            // edge_corners,                                                                  <L 331>
            // edge_base_dir,                                                                 <L 332>
            // cx,                                                                            <L 333>
            // cy,                                                                            <L 334>
            // cz,                                                                            <L 335>
            // e0,                                                                            <L 336>
            var_240 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_222);
            // v1 = _edge_to_vertex_id(                                                       <L 338>
            // grid_to_particle,                                                              <L 339>
            // particle_flags,                                                                <L 340>
            // corner_offsets,                                                                <L 341>
            // edge_corners,                                                                  <L 342>
            // edge_base_dir,                                                                 <L 343>
            // cx,                                                                            <L 344>
            // cy,                                                                            <L 345>
            // cz,                                                                            <L 346>
            // e1,                                                                            <L 347>
            var_241 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_231);
            // v2 = _edge_to_vertex_id(                                                       <L 349>
            // grid_to_particle,                                                              <L 350>
            // particle_flags,                                                                <L 351>
            // corner_offsets,                                                                <L 352>
            // edge_corners,                                                                  <L 353>
            // edge_base_dir,                                                                 <L 354>
            // cx,                                                                            <L 355>
            // cy,                                                                            <L 356>
            // cz,                                                                            <L 357>
            // e2,                                                                            <L 358>
            var_242 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_238);
            // if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:       <L 360>
            var_245 = (var_240 >= var_244);
            var_243 = var_245;
            if (var_243) {
                var_247 = (var_241 >= var_246);
                var_243 = var_243 && var_247;
            }
            if (var_243) {
                var_249 = (var_242 >= var_248);
                var_243 = var_243 && var_249;
            }
            if (var_243) {
                var_250 = (var_209 < var_0);
                var_243 = var_243 && var_250;
            }
            if (var_243) {
                // slot = base + out_count                                                    <L 361>
                var_251 = wp::add(var_1, var_209);
                // slot_tri_indices[slot, 0] = v0                                             <L 362>
                // wp::array_store(var_slot_tri_indices, var_251, var_252, var_240);
                // slot_tri_indices[slot, 1] = v1                                             <L 363>
                // wp::array_store(var_slot_tri_indices, var_251, var_253, var_241);
                // slot_tri_indices[slot, 2] = v2                                             <L 364>
                // wp::array_store(var_slot_tri_indices, var_251, var_254, var_242);
                // slot_active[slot] = 1                                                      <L 365>
                // wp::array_store(var_slot_active, var_251, var_255);
                // out_count = out_count + 1                                                  <L 366>
                var_257 = wp::add(var_209, var_256);
            }
            var_258 = wp::where(var_243, var_257, var_209);
            var_259 = wp::where(var_243, var_251, var_215);
        }
        var_260 = wp::where(var_225, var_258, var_209);
        var_261 = wp::where(var_225, var_231, var_210);
        var_262 = wp::where(var_225, var_238, var_211);
        var_263 = wp::where(var_225, var_240, var_212);
        var_264 = wp::where(var_225, var_241, var_213);
        var_265 = wp::where(var_225, var_242, var_214);
        var_266 = wp::where(var_225, var_259, var_215);
    }
    var_267 = wp::where(var_14, var_260, var_13);
    // cube_tri_counts[cube_flat] = out_count                                                 <L 368>
    // wp::array_store(var_cube_tri_counts, var_cube_flat, var_267);
    // return out_count                                                                       <L 369>
    goto label0;
    //---------
    // reverse
    label0:;
    adj_267 += adj_ret;
    // adj: return out_count                                                                  <L 369>
    wp::adj_array_store(var_cube_tri_counts, var_cube_flat, var_267, adj_cube_tri_counts, adj_cube_flat, adj_267);
    // adj: cube_tri_counts[cube_flat] = out_count                                            <L 368>
    wp::adj_where(var_14, var_260, var_13, adj_14, adj_260, adj_13, adj_267);
    if (var_14) {
        wp::adj_where(var_225, var_259, var_215, adj_225, adj_259, adj_215, adj_266);
        wp::adj_where(var_225, var_242, var_214, adj_225, adj_242, adj_214, adj_265);
        wp::adj_where(var_225, var_241, var_213, adj_225, adj_241, adj_213, adj_264);
        wp::adj_where(var_225, var_240, var_212, adj_225, adj_240, adj_212, adj_263);
        wp::adj_where(var_225, var_238, var_211, adj_225, adj_238, adj_211, adj_262);
        wp::adj_where(var_225, var_231, var_210, adj_225, adj_231, adj_210, adj_261);
        wp::adj_where(var_225, var_258, var_209, adj_225, adj_258, adj_209, adj_260);
        if (var_225) {
            wp::adj_where(var_243, var_251, var_215, adj_243, adj_251, adj_215, adj_259);
            wp::adj_where(var_243, var_257, var_209, adj_243, adj_257, adj_209, adj_258);
            if (var_243) {
                wp::adj_add(var_209, var_256, adj_209, adj_256, adj_257);
                // adj: out_count = out_count + 1                                             <L 366>
                wp::adj_array_store(var_slot_active, var_251, var_255, adj_slot_active, adj_251, adj_255);
                // adj: slot_active[slot] = 1                                                 <L 365>
                wp::adj_array_store(var_slot_tri_indices, var_251, var_254, var_242, adj_slot_tri_indices, adj_251, adj_254, adj_242);
                // adj: slot_tri_indices[slot, 2] = v2                                        <L 364>
                wp::adj_array_store(var_slot_tri_indices, var_251, var_253, var_241, adj_slot_tri_indices, adj_251, adj_253, adj_241);
                // adj: slot_tri_indices[slot, 1] = v1                                        <L 363>
                wp::adj_array_store(var_slot_tri_indices, var_251, var_252, var_240, adj_slot_tri_indices, adj_251, adj_252, adj_240);
                // adj: slot_tri_indices[slot, 0] = v0                                        <L 362>
                wp::adj_add(var_1, var_209, adj_1, adj_209, adj_251);
                // adj: slot = base + out_count                                               <L 361>
            }
            if (var_243) {
            }
            if (var_243) {
            }
            if (var_243) {
            }
            // adj: if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:  <L 360>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_238, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_238, adj_242);
            // adj: e2,                                                                       <L 358>
            // adj: cz,                                                                       <L 357>
            // adj: cy,                                                                       <L 356>
            // adj: cx,                                                                       <L 355>
            // adj: edge_base_dir,                                                            <L 354>
            // adj: edge_corners,                                                             <L 353>
            // adj: corner_offsets,                                                           <L 352>
            // adj: particle_flags,                                                           <L 351>
            // adj: grid_to_particle,                                                         <L 350>
            // adj: v2 = _edge_to_vertex_id(                                                  <L 349>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_231, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_231, adj_241);
            // adj: e1,                                                                       <L 347>
            // adj: cz,                                                                       <L 346>
            // adj: cy,                                                                       <L 345>
            // adj: cx,                                                                       <L 344>
            // adj: edge_base_dir,                                                            <L 343>
            // adj: edge_corners,                                                             <L 342>
            // adj: corner_offsets,                                                           <L 341>
            // adj: particle_flags,                                                           <L 340>
            // adj: grid_to_particle,                                                         <L 339>
            // adj: v1 = _edge_to_vertex_id(                                                  <L 338>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_222, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_222, adj_240);
            // adj: e0,                                                                       <L 336>
            // adj: cz,                                                                       <L 335>
            // adj: cy,                                                                       <L 334>
            // adj: cx,                                                                       <L 333>
            // adj: edge_base_dir,                                                            <L 332>
            // adj: edge_corners,                                                             <L 331>
            // adj: corner_offsets,                                                           <L 330>
            // adj: particle_flags,                                                           <L 329>
            // adj: grid_to_particle,                                                         <L 328>
            // adj: v0 = _edge_to_vertex_id(                                                  <L 327>
            wp::adj_copy(var_239, adj_237, adj_238);
            wp::adj_address(var_case_triangles, var_case, var_236, adj_case_triangles, adj_case, adj_236, adj_237);
            wp::adj_add(var_234, var_235, adj_234, adj_235, adj_236);
            wp::adj_mul(var_216, var_233, adj_216, adj_233, adj_234);
            // adj: e2 = case_triangles[case, t * 3 + 2]                                      <L 326>
            wp::adj_copy(var_232, adj_230, adj_231);
            wp::adj_address(var_case_triangles, var_case, var_229, adj_case_triangles, adj_case, adj_229, adj_230);
            wp::adj_add(var_227, var_228, adj_227, adj_228, adj_229);
            wp::adj_mul(var_216, var_226, adj_216, adj_226, adj_227);
            // adj: e1 = case_triangles[case, t * 3 + 1]                                      <L 325>
        }
        // adj: if e0 >= 0:                                                                   <L 324>
        wp::adj_copy(var_223, adj_221, adj_222);
        wp::adj_address(var_case_triangles, var_case, var_220, adj_case_triangles, adj_case, adj_220, adj_221);
        wp::adj_add(var_218, var_219, adj_218, adj_219, adj_220);
        wp::adj_mul(var_216, var_217, adj_216, adj_217, adj_218);
        // adj: e0 = case_triangles[case, t * 3 + 0]                                          <L 323>
        wp::adj_where(var_174, var_208, var_164, adj_174, adj_208, adj_164, adj_215);
        wp::adj_where(var_174, var_191, var_163, adj_174, adj_191, adj_163, adj_214);
        wp::adj_where(var_174, var_190, var_162, adj_174, adj_190, adj_162, adj_213);
        wp::adj_where(var_174, var_189, var_161, adj_174, adj_189, adj_161, adj_212);
        wp::adj_where(var_174, var_187, var_160, adj_174, adj_187, adj_160, adj_211);
        wp::adj_where(var_174, var_180, var_159, adj_174, adj_180, adj_159, adj_210);
        wp::adj_where(var_174, var_207, var_158, adj_174, adj_207, adj_158, adj_209);
        if (var_174) {
            wp::adj_where(var_192, var_200, var_164, adj_192, adj_200, adj_164, adj_208);
            wp::adj_where(var_192, var_206, var_158, adj_192, adj_206, adj_158, adj_207);
            if (var_192) {
                wp::adj_add(var_158, var_205, adj_158, adj_205, adj_206);
                // adj: out_count = out_count + 1                                             <L 366>
                wp::adj_array_store(var_slot_active, var_200, var_204, adj_slot_active, adj_200, adj_204);
                // adj: slot_active[slot] = 1                                                 <L 365>
                wp::adj_array_store(var_slot_tri_indices, var_200, var_203, var_191, adj_slot_tri_indices, adj_200, adj_203, adj_191);
                // adj: slot_tri_indices[slot, 2] = v2                                        <L 364>
                wp::adj_array_store(var_slot_tri_indices, var_200, var_202, var_190, adj_slot_tri_indices, adj_200, adj_202, adj_190);
                // adj: slot_tri_indices[slot, 1] = v1                                        <L 363>
                wp::adj_array_store(var_slot_tri_indices, var_200, var_201, var_189, adj_slot_tri_indices, adj_200, adj_201, adj_189);
                // adj: slot_tri_indices[slot, 0] = v0                                        <L 362>
                wp::adj_add(var_1, var_158, adj_1, adj_158, adj_200);
                // adj: slot = base + out_count                                               <L 361>
            }
            if (var_192) {
            }
            if (var_192) {
            }
            if (var_192) {
            }
            // adj: if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:  <L 360>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_187, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_187, adj_191);
            // adj: e2,                                                                       <L 358>
            // adj: cz,                                                                       <L 357>
            // adj: cy,                                                                       <L 356>
            // adj: cx,                                                                       <L 355>
            // adj: edge_base_dir,                                                            <L 354>
            // adj: edge_corners,                                                             <L 353>
            // adj: corner_offsets,                                                           <L 352>
            // adj: particle_flags,                                                           <L 351>
            // adj: grid_to_particle,                                                         <L 350>
            // adj: v2 = _edge_to_vertex_id(                                                  <L 349>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_180, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_180, adj_190);
            // adj: e1,                                                                       <L 347>
            // adj: cz,                                                                       <L 346>
            // adj: cy,                                                                       <L 345>
            // adj: cx,                                                                       <L 344>
            // adj: edge_base_dir,                                                            <L 343>
            // adj: edge_corners,                                                             <L 342>
            // adj: corner_offsets,                                                           <L 341>
            // adj: particle_flags,                                                           <L 340>
            // adj: grid_to_particle,                                                         <L 339>
            // adj: v1 = _edge_to_vertex_id(                                                  <L 338>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_171, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_171, adj_189);
            // adj: e0,                                                                       <L 336>
            // adj: cz,                                                                       <L 335>
            // adj: cy,                                                                       <L 334>
            // adj: cx,                                                                       <L 333>
            // adj: edge_base_dir,                                                            <L 332>
            // adj: edge_corners,                                                             <L 331>
            // adj: corner_offsets,                                                           <L 330>
            // adj: particle_flags,                                                           <L 329>
            // adj: grid_to_particle,                                                         <L 328>
            // adj: v0 = _edge_to_vertex_id(                                                  <L 327>
            wp::adj_copy(var_188, adj_186, adj_187);
            wp::adj_address(var_case_triangles, var_case, var_185, adj_case_triangles, adj_case, adj_185, adj_186);
            wp::adj_add(var_183, var_184, adj_183, adj_184, adj_185);
            wp::adj_mul(var_165, var_182, adj_165, adj_182, adj_183);
            // adj: e2 = case_triangles[case, t * 3 + 2]                                      <L 326>
            wp::adj_copy(var_181, adj_179, adj_180);
            wp::adj_address(var_case_triangles, var_case, var_178, adj_case_triangles, adj_case, adj_178, adj_179);
            wp::adj_add(var_176, var_177, adj_176, adj_177, adj_178);
            wp::adj_mul(var_165, var_175, adj_165, adj_175, adj_176);
            // adj: e1 = case_triangles[case, t * 3 + 1]                                      <L 325>
        }
        // adj: if e0 >= 0:                                                                   <L 324>
        wp::adj_copy(var_172, adj_170, adj_171);
        wp::adj_address(var_case_triangles, var_case, var_169, adj_case_triangles, adj_case, adj_169, adj_170);
        wp::adj_add(var_167, var_168, adj_167, adj_168, adj_169);
        wp::adj_mul(var_165, var_166, adj_165, adj_166, adj_167);
        // adj: e0 = case_triangles[case, t * 3 + 0]                                          <L 323>
        wp::adj_where(var_123, var_157, var_113, adj_123, adj_157, adj_113, adj_164);
        wp::adj_where(var_123, var_140, var_112, adj_123, adj_140, adj_112, adj_163);
        wp::adj_where(var_123, var_139, var_111, adj_123, adj_139, adj_111, adj_162);
        wp::adj_where(var_123, var_138, var_110, adj_123, adj_138, adj_110, adj_161);
        wp::adj_where(var_123, var_136, var_109, adj_123, adj_136, adj_109, adj_160);
        wp::adj_where(var_123, var_129, var_108, adj_123, adj_129, adj_108, adj_159);
        wp::adj_where(var_123, var_156, var_107, adj_123, adj_156, adj_107, adj_158);
        if (var_123) {
            wp::adj_where(var_141, var_149, var_113, adj_141, adj_149, adj_113, adj_157);
            wp::adj_where(var_141, var_155, var_107, adj_141, adj_155, adj_107, adj_156);
            if (var_141) {
                wp::adj_add(var_107, var_154, adj_107, adj_154, adj_155);
                // adj: out_count = out_count + 1                                             <L 366>
                wp::adj_array_store(var_slot_active, var_149, var_153, adj_slot_active, adj_149, adj_153);
                // adj: slot_active[slot] = 1                                                 <L 365>
                wp::adj_array_store(var_slot_tri_indices, var_149, var_152, var_140, adj_slot_tri_indices, adj_149, adj_152, adj_140);
                // adj: slot_tri_indices[slot, 2] = v2                                        <L 364>
                wp::adj_array_store(var_slot_tri_indices, var_149, var_151, var_139, adj_slot_tri_indices, adj_149, adj_151, adj_139);
                // adj: slot_tri_indices[slot, 1] = v1                                        <L 363>
                wp::adj_array_store(var_slot_tri_indices, var_149, var_150, var_138, adj_slot_tri_indices, adj_149, adj_150, adj_138);
                // adj: slot_tri_indices[slot, 0] = v0                                        <L 362>
                wp::adj_add(var_1, var_107, adj_1, adj_107, adj_149);
                // adj: slot = base + out_count                                               <L 361>
            }
            if (var_141) {
            }
            if (var_141) {
            }
            if (var_141) {
            }
            // adj: if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:  <L 360>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_136, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_136, adj_140);
            // adj: e2,                                                                       <L 358>
            // adj: cz,                                                                       <L 357>
            // adj: cy,                                                                       <L 356>
            // adj: cx,                                                                       <L 355>
            // adj: edge_base_dir,                                                            <L 354>
            // adj: edge_corners,                                                             <L 353>
            // adj: corner_offsets,                                                           <L 352>
            // adj: particle_flags,                                                           <L 351>
            // adj: grid_to_particle,                                                         <L 350>
            // adj: v2 = _edge_to_vertex_id(                                                  <L 349>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_129, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_129, adj_139);
            // adj: e1,                                                                       <L 347>
            // adj: cz,                                                                       <L 346>
            // adj: cy,                                                                       <L 345>
            // adj: cx,                                                                       <L 344>
            // adj: edge_base_dir,                                                            <L 343>
            // adj: edge_corners,                                                             <L 342>
            // adj: corner_offsets,                                                           <L 341>
            // adj: particle_flags,                                                           <L 340>
            // adj: grid_to_particle,                                                         <L 339>
            // adj: v1 = _edge_to_vertex_id(                                                  <L 338>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_120, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_120, adj_138);
            // adj: e0,                                                                       <L 336>
            // adj: cz,                                                                       <L 335>
            // adj: cy,                                                                       <L 334>
            // adj: cx,                                                                       <L 333>
            // adj: edge_base_dir,                                                            <L 332>
            // adj: edge_corners,                                                             <L 331>
            // adj: corner_offsets,                                                           <L 330>
            // adj: particle_flags,                                                           <L 329>
            // adj: grid_to_particle,                                                         <L 328>
            // adj: v0 = _edge_to_vertex_id(                                                  <L 327>
            wp::adj_copy(var_137, adj_135, adj_136);
            wp::adj_address(var_case_triangles, var_case, var_134, adj_case_triangles, adj_case, adj_134, adj_135);
            wp::adj_add(var_132, var_133, adj_132, adj_133, adj_134);
            wp::adj_mul(var_114, var_131, adj_114, adj_131, adj_132);
            // adj: e2 = case_triangles[case, t * 3 + 2]                                      <L 326>
            wp::adj_copy(var_130, adj_128, adj_129);
            wp::adj_address(var_case_triangles, var_case, var_127, adj_case_triangles, adj_case, adj_127, adj_128);
            wp::adj_add(var_125, var_126, adj_125, adj_126, adj_127);
            wp::adj_mul(var_114, var_124, adj_114, adj_124, adj_125);
            // adj: e1 = case_triangles[case, t * 3 + 1]                                      <L 325>
        }
        // adj: if e0 >= 0:                                                                   <L 324>
        wp::adj_copy(var_121, adj_119, adj_120);
        wp::adj_address(var_case_triangles, var_case, var_118, adj_case_triangles, adj_case, adj_118, adj_119);
        wp::adj_add(var_116, var_117, adj_116, adj_117, adj_118);
        wp::adj_mul(var_114, var_115, adj_114, adj_115, adj_116);
        // adj: e0 = case_triangles[case, t * 3 + 0]                                          <L 323>
        wp::adj_where(var_72, var_106, var_54, adj_72, adj_106, adj_54, adj_113);
        wp::adj_where(var_72, var_89, var_45, adj_72, adj_89, adj_45, adj_112);
        wp::adj_where(var_72, var_88, var_44, adj_72, adj_88, adj_44, adj_111);
        wp::adj_where(var_72, var_87, var_43, adj_72, adj_87, adj_43, adj_110);
        wp::adj_where(var_72, var_85, var_41, adj_72, adj_85, adj_41, adj_109);
        wp::adj_where(var_72, var_78, var_34, adj_72, adj_78, adj_34, adj_108);
        wp::adj_where(var_72, var_105, var_62, adj_72, adj_105, adj_62, adj_107);
        if (var_72) {
            wp::adj_where(var_90, var_98, var_54, adj_90, adj_98, adj_54, adj_106);
            wp::adj_where(var_90, var_104, var_62, adj_90, adj_104, adj_62, adj_105);
            if (var_90) {
                wp::adj_add(var_62, var_103, adj_62, adj_103, adj_104);
                // adj: out_count = out_count + 1                                             <L 366>
                wp::adj_array_store(var_slot_active, var_98, var_102, adj_slot_active, adj_98, adj_102);
                // adj: slot_active[slot] = 1                                                 <L 365>
                wp::adj_array_store(var_slot_tri_indices, var_98, var_101, var_89, adj_slot_tri_indices, adj_98, adj_101, adj_89);
                // adj: slot_tri_indices[slot, 2] = v2                                        <L 364>
                wp::adj_array_store(var_slot_tri_indices, var_98, var_100, var_88, adj_slot_tri_indices, adj_98, adj_100, adj_88);
                // adj: slot_tri_indices[slot, 1] = v1                                        <L 363>
                wp::adj_array_store(var_slot_tri_indices, var_98, var_99, var_87, adj_slot_tri_indices, adj_98, adj_99, adj_87);
                // adj: slot_tri_indices[slot, 0] = v0                                        <L 362>
                wp::adj_add(var_1, var_62, adj_1, adj_62, adj_98);
                // adj: slot = base + out_count                                               <L 361>
            }
            if (var_90) {
            }
            if (var_90) {
            }
            if (var_90) {
            }
            // adj: if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:  <L 360>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_85, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_85, adj_89);
            // adj: e2,                                                                       <L 358>
            // adj: cz,                                                                       <L 357>
            // adj: cy,                                                                       <L 356>
            // adj: cx,                                                                       <L 355>
            // adj: edge_base_dir,                                                            <L 354>
            // adj: edge_corners,                                                             <L 353>
            // adj: corner_offsets,                                                           <L 352>
            // adj: particle_flags,                                                           <L 351>
            // adj: grid_to_particle,                                                         <L 350>
            // adj: v2 = _edge_to_vertex_id(                                                  <L 349>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_78, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_78, adj_88);
            // adj: e1,                                                                       <L 347>
            // adj: cz,                                                                       <L 346>
            // adj: cy,                                                                       <L 345>
            // adj: cx,                                                                       <L 344>
            // adj: edge_base_dir,                                                            <L 343>
            // adj: edge_corners,                                                             <L 342>
            // adj: corner_offsets,                                                           <L 341>
            // adj: particle_flags,                                                           <L 340>
            // adj: grid_to_particle,                                                         <L 339>
            // adj: v1 = _edge_to_vertex_id(                                                  <L 338>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_69, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_69, adj_87);
            // adj: e0,                                                                       <L 336>
            // adj: cz,                                                                       <L 335>
            // adj: cy,                                                                       <L 334>
            // adj: cx,                                                                       <L 333>
            // adj: edge_base_dir,                                                            <L 332>
            // adj: edge_corners,                                                             <L 331>
            // adj: corner_offsets,                                                           <L 330>
            // adj: particle_flags,                                                           <L 329>
            // adj: grid_to_particle,                                                         <L 328>
            // adj: v0 = _edge_to_vertex_id(                                                  <L 327>
            wp::adj_copy(var_86, adj_84, adj_85);
            wp::adj_address(var_case_triangles, var_case, var_83, adj_case_triangles, adj_case, adj_83, adj_84);
            wp::adj_add(var_81, var_82, adj_81, adj_82, adj_83);
            wp::adj_mul(var_63, var_80, adj_63, adj_80, adj_81);
            // adj: e2 = case_triangles[case, t * 3 + 2]                                      <L 326>
            wp::adj_copy(var_79, adj_77, adj_78);
            wp::adj_address(var_case_triangles, var_case, var_76, adj_case_triangles, adj_case, adj_76, adj_77);
            wp::adj_add(var_74, var_75, adj_74, adj_75, adj_76);
            wp::adj_mul(var_63, var_73, adj_63, adj_73, adj_74);
            // adj: e1 = case_triangles[case, t * 3 + 1]                                      <L 325>
        }
        // adj: if e0 >= 0:                                                                   <L 324>
        wp::adj_copy(var_70, adj_68, adj_69);
        wp::adj_address(var_case_triangles, var_case, var_67, adj_case_triangles, adj_case, adj_67, adj_68);
        wp::adj_add(var_65, var_66, adj_65, adj_66, adj_67);
        wp::adj_mul(var_63, var_64, adj_63, adj_64, adj_65);
        // adj: e0 = case_triangles[case, t * 3 + 0]                                          <L 323>
        wp::adj_where(var_28, var_61, var_13, adj_28, adj_61, adj_13, adj_62);
        if (var_28) {
            wp::adj_where(var_46, var_60, var_13, adj_46, adj_60, adj_13, adj_61);
            if (var_46) {
                wp::adj_add(var_13, var_59, adj_13, adj_59, adj_60);
                // adj: out_count = out_count + 1                                             <L 366>
                wp::adj_array_store(var_slot_active, var_54, var_58, adj_slot_active, adj_54, adj_58);
                // adj: slot_active[slot] = 1                                                 <L 365>
                wp::adj_array_store(var_slot_tri_indices, var_54, var_57, var_45, adj_slot_tri_indices, adj_54, adj_57, adj_45);
                // adj: slot_tri_indices[slot, 2] = v2                                        <L 364>
                wp::adj_array_store(var_slot_tri_indices, var_54, var_56, var_44, adj_slot_tri_indices, adj_54, adj_56, adj_44);
                // adj: slot_tri_indices[slot, 1] = v1                                        <L 363>
                wp::adj_array_store(var_slot_tri_indices, var_54, var_55, var_43, adj_slot_tri_indices, adj_54, adj_55, adj_43);
                // adj: slot_tri_indices[slot, 0] = v0                                        <L 362>
                wp::adj_add(var_1, var_13, adj_1, adj_13, adj_54);
                // adj: slot = base + out_count                                               <L 361>
            }
            if (var_46) {
            }
            if (var_46) {
            }
            if (var_46) {
            }
            // adj: if v0 >= 0 and v1 >= 0 and v2 >= 0 and out_count < MC_MAX_TRIS_PER_CASE:  <L 360>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_41, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_41, adj_45);
            // adj: e2,                                                                       <L 358>
            // adj: cz,                                                                       <L 357>
            // adj: cy,                                                                       <L 356>
            // adj: cx,                                                                       <L 355>
            // adj: edge_base_dir,                                                            <L 354>
            // adj: edge_corners,                                                             <L 353>
            // adj: corner_offsets,                                                           <L 352>
            // adj: particle_flags,                                                           <L 351>
            // adj: grid_to_particle,                                                         <L 350>
            // adj: v2 = _edge_to_vertex_id(                                                  <L 349>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_34, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_34, adj_44);
            // adj: e1,                                                                       <L 347>
            // adj: cz,                                                                       <L 346>
            // adj: cy,                                                                       <L 345>
            // adj: cx,                                                                       <L 344>
            // adj: edge_base_dir,                                                            <L 343>
            // adj: edge_corners,                                                             <L 342>
            // adj: corner_offsets,                                                           <L 341>
            // adj: particle_flags,                                                           <L 340>
            // adj: grid_to_particle,                                                         <L 339>
            // adj: v1 = _edge_to_vertex_id(                                                  <L 338>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_cx, var_cy, var_cz, var_25, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_cx, adj_cy, adj_cz, adj_25, adj_43);
            // adj: e0,                                                                       <L 336>
            // adj: cz,                                                                       <L 335>
            // adj: cy,                                                                       <L 334>
            // adj: cx,                                                                       <L 333>
            // adj: edge_base_dir,                                                            <L 332>
            // adj: edge_corners,                                                             <L 331>
            // adj: corner_offsets,                                                           <L 330>
            // adj: particle_flags,                                                           <L 329>
            // adj: grid_to_particle,                                                         <L 328>
            // adj: v0 = _edge_to_vertex_id(                                                  <L 327>
            wp::adj_copy(var_42, adj_40, adj_41);
            wp::adj_address(var_case_triangles, var_case, var_39, adj_case_triangles, adj_case, adj_39, adj_40);
            wp::adj_add(var_37, var_38, adj_37, adj_38, adj_39);
            wp::adj_mul(var_19, var_36, adj_19, adj_36, adj_37);
            // adj: e2 = case_triangles[case, t * 3 + 2]                                      <L 326>
            wp::adj_copy(var_35, adj_33, adj_34);
            wp::adj_address(var_case_triangles, var_case, var_32, adj_case_triangles, adj_case, adj_32, adj_33);
            wp::adj_add(var_30, var_31, adj_30, adj_31, adj_32);
            wp::adj_mul(var_19, var_29, adj_19, adj_29, adj_30);
            // adj: e1 = case_triangles[case, t * 3 + 1]                                      <L 325>
        }
        // adj: if e0 >= 0:                                                                   <L 324>
        wp::adj_copy(var_26, adj_24, adj_25);
        wp::adj_address(var_case_triangles, var_case, var_23, adj_case_triangles, adj_case, adj_23, adj_24);
        wp::adj_add(var_21, var_22, adj_21, adj_22, adj_23);
        wp::adj_mul(var_19, var_20, adj_19, adj_20, adj_21);
        // adj: e0 = case_triangles[case, t * 3 + 0]                                          <L 323>
        // adj: for t in range(MC_MAX_TRIS_PER_CASE):                                         <L 322>
    }
    if (var_14) {
    }
    // adj: if case != 0 and case != 255:                                                     <L 321>
    wp::adj_int(var_12, adj_12, adj_13);
    // adj: out_count = int(0)                                                                <L 320>
    adj__clear_fixed_slot_0(var_11, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_11, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact);
    wp::adj_add(var_1, var_10, adj_1, adj_10, adj_11);
    adj__clear_fixed_slot_0(var_9, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_9, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact);
    wp::adj_add(var_1, var_8, adj_1, adj_8, adj_9);
    adj__clear_fixed_slot_0(var_7, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_7, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact);
    wp::adj_add(var_1, var_6, adj_1, adj_6, adj_7);
    adj__clear_fixed_slot_0(var_5, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_5, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact);
    wp::adj_add(var_1, var_4, adj_1, adj_4, adj_5);
    adj__clear_fixed_slot_0(var_3, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_3, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact);
    wp::adj_add(var_1, var_2, adj_1, adj_2, adj_3);
    // adj: _clear_fixed_slot(base + local, slot_tri_indices, slot_active, slot_to_compact)   <L 318>
    // adj: for local in range(MC_MAX_TRIS_PER_CASE):                                         <L 317>
    wp::adj_mul(var_cube_flat, var_0, adj_cube_flat, adj_0, adj_1);
    // adj: base = cube_flat * MC_MAX_TRIS_PER_CASE                                           <L 316>
    // adj: def _write_cube_fixed_slots(                                                      <L 299>
    return;
}



extern "C" __global__ void compute_vertex_positions_kernel_10e5750e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_particle_orientation,
    wp::float32 var_mc_factor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos)
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
        wp::mat_t<3, 3, wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::mat_t<3, 3, wp::float32> var_4;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        wp::vec_t<3, wp::float32> var_8;
        const wp::float32 var_9 = 1.0;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::int32 var_13 = 0;
        const wp::int32 var_14 = 0;
        wp::float32 var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 1;
        wp::float32 var_18;
        const wp::int32 var_19 = 0;
        const wp::int32 var_20 = 2;
        wp::float32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        const wp::float32 var_23 = -1.0;
        wp::vec_t<3, wp::float32> var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        const wp::int32 var_28 = 0;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        const wp::int32 var_31 = 0;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 2;
        wp::float32 var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::float32 var_38 = 1.0;
        wp::vec_t<3, wp::float32> var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 2;
        bool var_42;
        const wp::int32 var_43 = 1;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::int32 var_46 = 1;
        const wp::int32 var_47 = 1;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        const wp::float32 var_53 = -1.0;
        wp::vec_t<3, wp::float32> var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        const wp::int32 var_58 = 1;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        const wp::int32 var_62 = 1;
        wp::float32 var_63;
        const wp::int32 var_64 = 1;
        const wp::int32 var_65 = 2;
        wp::float32 var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::float32 var_68 = 1.0;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        const wp::int32 var_71 = 4;
        bool var_72;
        const wp::int32 var_73 = 2;
        const wp::int32 var_74 = 0;
        wp::float32 var_75;
        const wp::int32 var_76 = 2;
        const wp::int32 var_77 = 1;
        wp::float32 var_78;
        const wp::int32 var_79 = 2;
        const wp::int32 var_80 = 2;
        wp::float32 var_81;
        wp::vec_t<3, wp::float32> var_82;
        const wp::float32 var_83 = -1.0;
        wp::vec_t<3, wp::float32> var_84;
        wp::float32 var_85;
        const wp::int32 var_86 = 2;
        const wp::int32 var_87 = 0;
        wp::float32 var_88;
        const wp::int32 var_89 = 2;
        const wp::int32 var_90 = 1;
        wp::float32 var_91;
        const wp::int32 var_92 = 2;
        const wp::int32 var_93 = 2;
        wp::float32 var_94;
        wp::vec_t<3, wp::float32> var_95;
        const wp::float32 var_96 = 1.0;
        wp::vec_t<3, wp::float32> var_97;
        wp::float32 var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::float32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::float32 var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::float32 var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::float32 var_106;
        const wp::int32 var_107 = 6;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::float32 var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        //---------
        // forward
        // def compute_vertex_positions_kernel(                                                   <L 216>
        // p, d = wp.tid()                                                                        <L 234>
        builtin_tid2d(var_0, var_1);
        // frame = particle_orientation[p]                                                        <L 235>
        var_2 = wp::address(var_particle_orientation, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // ax = wp.vec3(0.0, 0.0, 0.0)                                                            <L 237>
        var_8 = wp::vec_t<3, wp::float32>(var_5, var_6, var_7);
        // sign = float(1.0)                                                                      <L 238>
        var_10 = wp::float(var_9);
        // if d == 0:                                                                             <L 239>
        var_12 = (var_1 == var_11);
        if (var_12) {
            // ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                                <L 240>
            var_15 = wp::extract(var_3, var_13, var_14);
            var_18 = wp::extract(var_3, var_16, var_17);
            var_21 = wp::extract(var_3, var_19, var_20);
            var_22 = wp::vec_t<3, wp::float32>(var_15, var_18, var_21);
            // sign = -1.0                                                                        <L 241>
        }
        var_24 = wp::where(var_12, var_22, var_8);
        var_25 = wp::where(var_12, var_23, var_10);
        if (!var_12) {
            // elif d == 1:                                                                       <L 242>
            var_27 = (var_1 == var_26);
            if (var_27) {
                // ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                            <L 243>
                var_30 = wp::extract(var_3, var_28, var_29);
                var_33 = wp::extract(var_3, var_31, var_32);
                var_36 = wp::extract(var_3, var_34, var_35);
                var_37 = wp::vec_t<3, wp::float32>(var_30, var_33, var_36);
                // sign = 1.0                                                                     <L 244>
            }
            var_39 = wp::where(var_27, var_37, var_24);
            var_40 = wp::where(var_27, var_38, var_25);
            if (!var_27) {
                // elif d == 2:                                                                   <L 245>
                var_42 = (var_1 == var_41);
                if (var_42) {
                    // ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])                        <L 246>
                    var_45 = wp::extract(var_3, var_43, var_44);
                    var_48 = wp::extract(var_3, var_46, var_47);
                    var_51 = wp::extract(var_3, var_49, var_50);
                    var_52 = wp::vec_t<3, wp::float32>(var_45, var_48, var_51);
                    // sign = -1.0                                                                <L 247>
                }
                var_54 = wp::where(var_42, var_52, var_39);
                var_55 = wp::where(var_42, var_53, var_40);
                if (!var_42) {
                    // elif d == 3:                                                               <L 248>
                    var_57 = (var_1 == var_56);
                    if (var_57) {
                        // ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])                    <L 249>
                        var_60 = wp::extract(var_3, var_58, var_59);
                        var_63 = wp::extract(var_3, var_61, var_62);
                        var_66 = wp::extract(var_3, var_64, var_65);
                        var_67 = wp::vec_t<3, wp::float32>(var_60, var_63, var_66);
                        // sign = 1.0                                                             <L 250>
                    }
                    var_69 = wp::where(var_57, var_67, var_54);
                    var_70 = wp::where(var_57, var_68, var_55);
                    if (!var_57) {
                        // elif d == 4:                                                           <L 251>
                        var_72 = (var_1 == var_71);
                        if (var_72) {
                            // ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])                <L 252>
                            var_75 = wp::extract(var_3, var_73, var_74);
                            var_78 = wp::extract(var_3, var_76, var_77);
                            var_81 = wp::extract(var_3, var_79, var_80);
                            var_82 = wp::vec_t<3, wp::float32>(var_75, var_78, var_81);
                            // sign = -1.0                                                        <L 253>
                        }
                        var_84 = wp::where(var_72, var_82, var_69);
                        var_85 = wp::where(var_72, var_83, var_70);
                        if (!var_72) {
                            // ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])                <L 255>
                            var_88 = wp::extract(var_3, var_86, var_87);
                            var_91 = wp::extract(var_3, var_89, var_90);
                            var_94 = wp::extract(var_3, var_92, var_93);
                            var_95 = wp::vec_t<3, wp::float32>(var_88, var_91, var_94);
                            // sign = 1.0                                                         <L 256>
                        }
                        var_97 = wp::where(var_72, var_84, var_95);
                        var_98 = wp::where(var_72, var_85, var_96);
                    }
                    var_99 = wp::where(var_57, var_69, var_97);
                    var_100 = wp::where(var_57, var_70, var_98);
                }
                var_101 = wp::where(var_42, var_54, var_99);
                var_102 = wp::where(var_42, var_55, var_100);
            }
            var_103 = wp::where(var_27, var_39, var_101);
            var_104 = wp::where(var_27, var_40, var_102);
        }
        var_105 = wp::where(var_12, var_24, var_103);
        var_106 = wp::where(var_12, var_25, var_104);
        // vid = p * 6 + d                                                                        <L 257>
        var_108 = wp::mul(var_0, var_107);
        var_109 = wp::add(var_108, var_1);
        // vertex_pos[vid] = particle_q[p] + ax * (sign * mc_factor)                              <L 258>
        var_110 = wp::address(var_particle_q, var_0);
        var_111 = wp::mul(var_106, var_mc_factor);
        var_112 = wp::mul(var_105, var_111);
        var_114 = wp::load(var_110);
        var_113 = wp::add(var_114, var_112);
        wp::array_store(var_vertex_pos, var_109, var_113);
    }
}



extern "C" __global__ void compute_vertex_positions_kernel_10e5750e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_particle_orientation,
    wp::float32 var_mc_factor,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_pos,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_particle_orientation,
    wp::float32 adj_mc_factor,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_pos)
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
        wp::mat_t<3, 3, wp::float32>* var_2;
        wp::mat_t<3, 3, wp::float32> var_3;
        wp::mat_t<3, 3, wp::float32> var_4;
        const wp::float32 var_5 = 0.0;
        const wp::float32 var_6 = 0.0;
        const wp::float32 var_7 = 0.0;
        wp::vec_t<3, wp::float32> var_8;
        const wp::float32 var_9 = 1.0;
        wp::float32 var_10;
        const wp::int32 var_11 = 0;
        bool var_12;
        const wp::int32 var_13 = 0;
        const wp::int32 var_14 = 0;
        wp::float32 var_15;
        const wp::int32 var_16 = 0;
        const wp::int32 var_17 = 1;
        wp::float32 var_18;
        const wp::int32 var_19 = 0;
        const wp::int32 var_20 = 2;
        wp::float32 var_21;
        wp::vec_t<3, wp::float32> var_22;
        const wp::float32 var_23 = -1.0;
        wp::vec_t<3, wp::float32> var_24;
        wp::float32 var_25;
        const wp::int32 var_26 = 1;
        bool var_27;
        const wp::int32 var_28 = 0;
        const wp::int32 var_29 = 0;
        wp::float32 var_30;
        const wp::int32 var_31 = 0;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 0;
        const wp::int32 var_35 = 2;
        wp::float32 var_36;
        wp::vec_t<3, wp::float32> var_37;
        const wp::float32 var_38 = 1.0;
        wp::vec_t<3, wp::float32> var_39;
        wp::float32 var_40;
        const wp::int32 var_41 = 2;
        bool var_42;
        const wp::int32 var_43 = 1;
        const wp::int32 var_44 = 0;
        wp::float32 var_45;
        const wp::int32 var_46 = 1;
        const wp::int32 var_47 = 1;
        wp::float32 var_48;
        const wp::int32 var_49 = 1;
        const wp::int32 var_50 = 2;
        wp::float32 var_51;
        wp::vec_t<3, wp::float32> var_52;
        const wp::float32 var_53 = -1.0;
        wp::vec_t<3, wp::float32> var_54;
        wp::float32 var_55;
        const wp::int32 var_56 = 3;
        bool var_57;
        const wp::int32 var_58 = 1;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        const wp::int32 var_62 = 1;
        wp::float32 var_63;
        const wp::int32 var_64 = 1;
        const wp::int32 var_65 = 2;
        wp::float32 var_66;
        wp::vec_t<3, wp::float32> var_67;
        const wp::float32 var_68 = 1.0;
        wp::vec_t<3, wp::float32> var_69;
        wp::float32 var_70;
        const wp::int32 var_71 = 4;
        bool var_72;
        const wp::int32 var_73 = 2;
        const wp::int32 var_74 = 0;
        wp::float32 var_75;
        const wp::int32 var_76 = 2;
        const wp::int32 var_77 = 1;
        wp::float32 var_78;
        const wp::int32 var_79 = 2;
        const wp::int32 var_80 = 2;
        wp::float32 var_81;
        wp::vec_t<3, wp::float32> var_82;
        const wp::float32 var_83 = -1.0;
        wp::vec_t<3, wp::float32> var_84;
        wp::float32 var_85;
        const wp::int32 var_86 = 2;
        const wp::int32 var_87 = 0;
        wp::float32 var_88;
        const wp::int32 var_89 = 2;
        const wp::int32 var_90 = 1;
        wp::float32 var_91;
        const wp::int32 var_92 = 2;
        const wp::int32 var_93 = 2;
        wp::float32 var_94;
        wp::vec_t<3, wp::float32> var_95;
        const wp::float32 var_96 = 1.0;
        wp::vec_t<3, wp::float32> var_97;
        wp::float32 var_98;
        wp::vec_t<3, wp::float32> var_99;
        wp::float32 var_100;
        wp::vec_t<3, wp::float32> var_101;
        wp::float32 var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::float32 var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::float32 var_106;
        const wp::int32 var_107 = 6;
        wp::int32 var_108;
        wp::int32 var_109;
        wp::vec_t<3, wp::float32>* var_110;
        wp::float32 var_111;
        wp::vec_t<3, wp::float32> var_112;
        wp::vec_t<3, wp::float32> var_113;
        wp::vec_t<3, wp::float32> var_114;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::mat_t<3, 3, wp::float32> adj_2 = {};
        wp::mat_t<3, 3, wp::float32> adj_3 = {};
        wp::mat_t<3, 3, wp::float32> adj_4 = {};
        wp::float32 adj_5 = {};
        wp::float32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::vec_t<3, wp::float32> adj_8 = {};
        wp::float32 adj_9 = {};
        wp::float32 adj_10 = {};
        wp::int32 adj_11 = {};
        bool adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::float32 adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::float32 adj_25 = {};
        wp::int32 adj_26 = {};
        bool adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::float32 adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::float32 adj_40 = {};
        wp::int32 adj_41 = {};
        bool adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::float32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::vec_t<3, wp::float32> adj_52 = {};
        wp::float32 adj_53 = {};
        wp::vec_t<3, wp::float32> adj_54 = {};
        wp::float32 adj_55 = {};
        wp::int32 adj_56 = {};
        bool adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::float32 adj_66 = {};
        wp::vec_t<3, wp::float32> adj_67 = {};
        wp::float32 adj_68 = {};
        wp::vec_t<3, wp::float32> adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        bool adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::float32 adj_81 = {};
        wp::vec_t<3, wp::float32> adj_82 = {};
        wp::float32 adj_83 = {};
        wp::vec_t<3, wp::float32> adj_84 = {};
        wp::float32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::float32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::int32 adj_90 = {};
        wp::float32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::float32 adj_94 = {};
        wp::vec_t<3, wp::float32> adj_95 = {};
        wp::float32 adj_96 = {};
        wp::vec_t<3, wp::float32> adj_97 = {};
        wp::float32 adj_98 = {};
        wp::vec_t<3, wp::float32> adj_99 = {};
        wp::float32 adj_100 = {};
        wp::vec_t<3, wp::float32> adj_101 = {};
        wp::float32 adj_102 = {};
        wp::vec_t<3, wp::float32> adj_103 = {};
        wp::float32 adj_104 = {};
        wp::vec_t<3, wp::float32> adj_105 = {};
        wp::float32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::int32 adj_108 = {};
        wp::int32 adj_109 = {};
        wp::vec_t<3, wp::float32> adj_110 = {};
        wp::float32 adj_111 = {};
        wp::vec_t<3, wp::float32> adj_112 = {};
        wp::vec_t<3, wp::float32> adj_113 = {};
        wp::vec_t<3, wp::float32> adj_114 = {};
        //---------
        // forward
        // def compute_vertex_positions_kernel(                                                   <L 216>
        // p, d = wp.tid()                                                                        <L 234>
        builtin_tid2d(var_0, var_1);
        // frame = particle_orientation[p]                                                        <L 235>
        var_2 = wp::address(var_particle_orientation, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // ax = wp.vec3(0.0, 0.0, 0.0)                                                            <L 237>
        var_8 = wp::vec_t<3, wp::float32>(var_5, var_6, var_7);
        // sign = float(1.0)                                                                      <L 238>
        var_10 = wp::float(var_9);
        // if d == 0:                                                                             <L 239>
        var_12 = (var_1 == var_11);
        if (var_12) {
            // ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                                <L 240>
            var_15 = wp::extract(var_3, var_13, var_14);
            var_18 = wp::extract(var_3, var_16, var_17);
            var_21 = wp::extract(var_3, var_19, var_20);
            var_22 = wp::vec_t<3, wp::float32>(var_15, var_18, var_21);
            // sign = -1.0                                                                        <L 241>
        }
        var_24 = wp::where(var_12, var_22, var_8);
        var_25 = wp::where(var_12, var_23, var_10);
        if (!var_12) {
            // elif d == 1:                                                                       <L 242>
            var_27 = (var_1 == var_26);
            if (var_27) {
                // ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                            <L 243>
                var_30 = wp::extract(var_3, var_28, var_29);
                var_33 = wp::extract(var_3, var_31, var_32);
                var_36 = wp::extract(var_3, var_34, var_35);
                var_37 = wp::vec_t<3, wp::float32>(var_30, var_33, var_36);
                // sign = 1.0                                                                     <L 244>
            }
            var_39 = wp::where(var_27, var_37, var_24);
            var_40 = wp::where(var_27, var_38, var_25);
            if (!var_27) {
                // elif d == 2:                                                                   <L 245>
                var_42 = (var_1 == var_41);
                if (var_42) {
                    // ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])                        <L 246>
                    var_45 = wp::extract(var_3, var_43, var_44);
                    var_48 = wp::extract(var_3, var_46, var_47);
                    var_51 = wp::extract(var_3, var_49, var_50);
                    var_52 = wp::vec_t<3, wp::float32>(var_45, var_48, var_51);
                    // sign = -1.0                                                                <L 247>
                }
                var_54 = wp::where(var_42, var_52, var_39);
                var_55 = wp::where(var_42, var_53, var_40);
                if (!var_42) {
                    // elif d == 3:                                                               <L 248>
                    var_57 = (var_1 == var_56);
                    if (var_57) {
                        // ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])                    <L 249>
                        var_60 = wp::extract(var_3, var_58, var_59);
                        var_63 = wp::extract(var_3, var_61, var_62);
                        var_66 = wp::extract(var_3, var_64, var_65);
                        var_67 = wp::vec_t<3, wp::float32>(var_60, var_63, var_66);
                        // sign = 1.0                                                             <L 250>
                    }
                    var_69 = wp::where(var_57, var_67, var_54);
                    var_70 = wp::where(var_57, var_68, var_55);
                    if (!var_57) {
                        // elif d == 4:                                                           <L 251>
                        var_72 = (var_1 == var_71);
                        if (var_72) {
                            // ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])                <L 252>
                            var_75 = wp::extract(var_3, var_73, var_74);
                            var_78 = wp::extract(var_3, var_76, var_77);
                            var_81 = wp::extract(var_3, var_79, var_80);
                            var_82 = wp::vec_t<3, wp::float32>(var_75, var_78, var_81);
                            // sign = -1.0                                                        <L 253>
                        }
                        var_84 = wp::where(var_72, var_82, var_69);
                        var_85 = wp::where(var_72, var_83, var_70);
                        if (!var_72) {
                            // ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])                <L 255>
                            var_88 = wp::extract(var_3, var_86, var_87);
                            var_91 = wp::extract(var_3, var_89, var_90);
                            var_94 = wp::extract(var_3, var_92, var_93);
                            var_95 = wp::vec_t<3, wp::float32>(var_88, var_91, var_94);
                            // sign = 1.0                                                         <L 256>
                        }
                        var_97 = wp::where(var_72, var_84, var_95);
                        var_98 = wp::where(var_72, var_85, var_96);
                    }
                    var_99 = wp::where(var_57, var_69, var_97);
                    var_100 = wp::where(var_57, var_70, var_98);
                }
                var_101 = wp::where(var_42, var_54, var_99);
                var_102 = wp::where(var_42, var_55, var_100);
            }
            var_103 = wp::where(var_27, var_39, var_101);
            var_104 = wp::where(var_27, var_40, var_102);
        }
        var_105 = wp::where(var_12, var_24, var_103);
        var_106 = wp::where(var_12, var_25, var_104);
        // vid = p * 6 + d                                                                        <L 257>
        var_108 = wp::mul(var_0, var_107);
        var_109 = wp::add(var_108, var_1);
        // vertex_pos[vid] = particle_q[p] + ax * (sign * mc_factor)                              <L 258>
        var_110 = wp::address(var_particle_q, var_0);
        var_111 = wp::mul(var_106, var_mc_factor);
        var_112 = wp::mul(var_105, var_111);
        var_114 = wp::load(var_110);
        var_113 = wp::add(var_114, var_112);
        // wp::array_store(var_vertex_pos, var_109, var_113);
        //---------
        // reverse
        wp::adj_array_store(var_vertex_pos, var_109, var_113, adj_vertex_pos, adj_109, adj_113);
        wp::adj_add(var_114, var_112, adj_110, adj_112, adj_113);
        wp::adj_mul(var_105, var_111, adj_105, adj_111, adj_112);
        wp::adj_mul(var_106, var_mc_factor, adj_106, adj_mc_factor, adj_111);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_110);
        // adj: vertex_pos[vid] = particle_q[p] + ax * (sign * mc_factor)                         <L 258>
        wp::adj_add(var_108, var_1, adj_108, adj_1, adj_109);
        wp::adj_mul(var_0, var_107, adj_0, adj_107, adj_108);
        // adj: vid = p * 6 + d                                                                   <L 257>
        wp::adj_where(var_12, var_25, var_104, adj_12, adj_25, adj_104, adj_106);
        wp::adj_where(var_12, var_24, var_103, adj_12, adj_24, adj_103, adj_105);
        if (!var_12) {
            wp::adj_where(var_27, var_40, var_102, adj_27, adj_40, adj_102, adj_104);
            wp::adj_where(var_27, var_39, var_101, adj_27, adj_39, adj_101, adj_103);
            if (!var_27) {
                wp::adj_where(var_42, var_55, var_100, adj_42, adj_55, adj_100, adj_102);
                wp::adj_where(var_42, var_54, var_99, adj_42, adj_54, adj_99, adj_101);
                if (!var_42) {
                    wp::adj_where(var_57, var_70, var_98, adj_57, adj_70, adj_98, adj_100);
                    wp::adj_where(var_57, var_69, var_97, adj_57, adj_69, adj_97, adj_99);
                    if (!var_57) {
                        wp::adj_where(var_72, var_85, var_96, adj_72, adj_85, adj_96, adj_98);
                        wp::adj_where(var_72, var_84, var_95, adj_72, adj_84, adj_95, adj_97);
                        if (!var_72) {
                            // adj: sign = 1.0                                                    <L 256>
                            wp::adj_vec_t(var_88, var_91, var_94, adj_88, adj_91, adj_94, adj_95);
                            wp::adj_extract(var_3, var_92, var_93, adj_3, adj_92, adj_93, adj_94);
                            wp::adj_extract(var_3, var_89, var_90, adj_3, adj_89, adj_90, adj_91);
                            wp::adj_extract(var_3, var_86, var_87, adj_3, adj_86, adj_87, adj_88);
                            // adj: ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])           <L 255>
                        }
                        wp::adj_where(var_72, var_83, var_70, adj_72, adj_83, adj_70, adj_85);
                        wp::adj_where(var_72, var_82, var_69, adj_72, adj_82, adj_69, adj_84);
                        if (var_72) {
                            // adj: sign = -1.0                                                   <L 253>
                            wp::adj_vec_t(var_75, var_78, var_81, adj_75, adj_78, adj_81, adj_82);
                            wp::adj_extract(var_3, var_79, var_80, adj_3, adj_79, adj_80, adj_81);
                            wp::adj_extract(var_3, var_76, var_77, adj_3, adj_76, adj_77, adj_78);
                            wp::adj_extract(var_3, var_73, var_74, adj_3, adj_73, adj_74, adj_75);
                            // adj: ax = wp.vec3(frame[2, 0], frame[2, 1], frame[2, 2])           <L 252>
                        }
                        // adj: elif d == 4:                                                      <L 251>
                    }
                    wp::adj_where(var_57, var_68, var_55, adj_57, adj_68, adj_55, adj_70);
                    wp::adj_where(var_57, var_67, var_54, adj_57, adj_67, adj_54, adj_69);
                    if (var_57) {
                        // adj: sign = 1.0                                                        <L 250>
                        wp::adj_vec_t(var_60, var_63, var_66, adj_60, adj_63, adj_66, adj_67);
                        wp::adj_extract(var_3, var_64, var_65, adj_3, adj_64, adj_65, adj_66);
                        wp::adj_extract(var_3, var_61, var_62, adj_3, adj_61, adj_62, adj_63);
                        wp::adj_extract(var_3, var_58, var_59, adj_3, adj_58, adj_59, adj_60);
                        // adj: ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])               <L 249>
                    }
                    // adj: elif d == 3:                                                          <L 248>
                }
                wp::adj_where(var_42, var_53, var_40, adj_42, adj_53, adj_40, adj_55);
                wp::adj_where(var_42, var_52, var_39, adj_42, adj_52, adj_39, adj_54);
                if (var_42) {
                    // adj: sign = -1.0                                                           <L 247>
                    wp::adj_vec_t(var_45, var_48, var_51, adj_45, adj_48, adj_51, adj_52);
                    wp::adj_extract(var_3, var_49, var_50, adj_3, adj_49, adj_50, adj_51);
                    wp::adj_extract(var_3, var_46, var_47, adj_3, adj_46, adj_47, adj_48);
                    wp::adj_extract(var_3, var_43, var_44, adj_3, adj_43, adj_44, adj_45);
                    // adj: ax = wp.vec3(frame[1, 0], frame[1, 1], frame[1, 2])                   <L 246>
                }
                // adj: elif d == 2:                                                              <L 245>
            }
            wp::adj_where(var_27, var_38, var_25, adj_27, adj_38, adj_25, adj_40);
            wp::adj_where(var_27, var_37, var_24, adj_27, adj_37, adj_24, adj_39);
            if (var_27) {
                // adj: sign = 1.0                                                                <L 244>
                wp::adj_vec_t(var_30, var_33, var_36, adj_30, adj_33, adj_36, adj_37);
                wp::adj_extract(var_3, var_34, var_35, adj_3, adj_34, adj_35, adj_36);
                wp::adj_extract(var_3, var_31, var_32, adj_3, adj_31, adj_32, adj_33);
                wp::adj_extract(var_3, var_28, var_29, adj_3, adj_28, adj_29, adj_30);
                // adj: ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                       <L 243>
            }
            // adj: elif d == 1:                                                                  <L 242>
        }
        wp::adj_where(var_12, var_23, var_10, adj_12, adj_23, adj_10, adj_25);
        wp::adj_where(var_12, var_22, var_8, adj_12, adj_22, adj_8, adj_24);
        if (var_12) {
            // adj: sign = -1.0                                                                   <L 241>
            wp::adj_vec_t(var_15, var_18, var_21, adj_15, adj_18, adj_21, adj_22);
            wp::adj_extract(var_3, var_19, var_20, adj_3, adj_19, adj_20, adj_21);
            wp::adj_extract(var_3, var_16, var_17, adj_3, adj_16, adj_17, adj_18);
            wp::adj_extract(var_3, var_13, var_14, adj_3, adj_13, adj_14, adj_15);
            // adj: ax = wp.vec3(frame[0, 0], frame[0, 1], frame[0, 2])                           <L 240>
        }
        // adj: if d == 0:                                                                        <L 239>
        wp::adj_float(var_9, adj_9, adj_10);
        // adj: sign = float(1.0)                                                                 <L 238>
        wp::adj_vec_t(var_5, var_6, var_7, adj_5, adj_6, adj_7, adj_8);
        // adj: ax = wp.vec3(0.0, 0.0, 0.0)                                                       <L 237>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_particle_orientation, var_0, adj_particle_orientation, adj_0, adj_2);
        // adj: frame = particle_orientation[p]                                                   <L 235>
        // adj: p, d = wp.tid()                                                                   <L 234>
        // adj: def compute_vertex_positions_kernel(                                              <L 216>
        continue;
    }
}



extern "C" __global__ void compute_visible_flags_kernel_6ef67016_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::int32> var_material_visible,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_visible_flags)
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
        wp::int32* var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 2;
        wp::float32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        bool var_20;
        const wp::int32 var_21 = 1;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        //---------
        // forward
        // def compute_visible_flags_kernel(                                                      <L 100>
        // i = wp.tid()                                                                           <L 114>
        var_0 = builtin_tid1d();
        // f = particle_flags[i]                                                                  <L 115>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if material_visible[particle_material[i]] == 0:                                        <L 116>
        var_4 = wp::address(var_particle_material, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_material_visible, var_6);
        var_9 = wp::load(var_5);
        var_8 = (var_9 == var_7);
        if (var_8) {
            // f = f & (~wp.int32(ParticleFlags.ACTIVE))                                          <L 117>
            var_12 = wp::int32(var_11);
            var_13 = wp::invert(var_12);
            var_14 = wp::bit_and(var_2, var_13);
        }
        var_15 = wp::where(var_8, var_14, var_2);
        // if particle_q[i][2] > cut_z:                                                           <L 118>
        var_16 = wp::address(var_particle_q, var_0);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = (var_18 > var_cut_z);
        if (var_20) {
            // f = f & (~wp.int32(ParticleFlags.ACTIVE))                                          <L 119>
            var_23 = wp::int32(var_22);
            var_24 = wp::invert(var_23);
            var_25 = wp::bit_and(var_15, var_24);
        }
        var_26 = wp::where(var_20, var_25, var_15);
        // visible_flags[i] = f                                                                   <L 120>
        wp::array_store(var_visible_flags, var_0, var_26);
    }
}



extern "C" __global__ void compute_visible_flags_kernel_6ef67016_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_material,
    wp::array_t<wp::int32> var_material_visible,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::float32 var_cut_z,
    wp::array_t<wp::int32> var_visible_flags,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_particle_material,
    wp::array_t<wp::int32> adj_material_visible,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::float32 adj_cut_z,
    wp::array_t<wp::int32> adj_visible_flags)
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
        wp::int32* var_4;
        wp::int32* var_5;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 1;
        const wp::int32 var_11 = 1;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        wp::int32 var_15;
        wp::vec_t<3, wp::float32>* var_16;
        const wp::int32 var_17 = 2;
        wp::float32 var_18;
        wp::vec_t<3, wp::float32> var_19;
        bool var_20;
        const wp::int32 var_21 = 1;
        const wp::int32 var_22 = 1;
        wp::int32 var_23;
        wp::int32 var_24;
        wp::int32 var_25;
        wp::int32 var_26;
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
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::float32 adj_18 = {};
        wp::vec_t<3, wp::float32> adj_19 = {};
        bool adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        //---------
        // forward
        // def compute_visible_flags_kernel(                                                      <L 100>
        // i = wp.tid()                                                                           <L 114>
        var_0 = builtin_tid1d();
        // f = particle_flags[i]                                                                  <L 115>
        var_1 = wp::address(var_particle_flags, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // if material_visible[particle_material[i]] == 0:                                        <L 116>
        var_4 = wp::address(var_particle_material, var_0);
        var_6 = wp::load(var_4);
        var_5 = wp::address(var_material_visible, var_6);
        var_9 = wp::load(var_5);
        var_8 = (var_9 == var_7);
        if (var_8) {
            // f = f & (~wp.int32(ParticleFlags.ACTIVE))                                          <L 117>
            var_12 = wp::int32(var_11);
            var_13 = wp::invert(var_12);
            var_14 = wp::bit_and(var_2, var_13);
        }
        var_15 = wp::where(var_8, var_14, var_2);
        // if particle_q[i][2] > cut_z:                                                           <L 118>
        var_16 = wp::address(var_particle_q, var_0);
        var_19 = wp::load(var_16);
        var_18 = wp::extract(var_19, var_17);
        var_20 = (var_18 > var_cut_z);
        if (var_20) {
            // f = f & (~wp.int32(ParticleFlags.ACTIVE))                                          <L 119>
            var_23 = wp::int32(var_22);
            var_24 = wp::invert(var_23);
            var_25 = wp::bit_and(var_15, var_24);
        }
        var_26 = wp::where(var_20, var_25, var_15);
        // visible_flags[i] = f                                                                   <L 120>
        // wp::array_store(var_visible_flags, var_0, var_26);
        //---------
        // reverse
        wp::adj_array_store(var_visible_flags, var_0, var_26, adj_visible_flags, adj_0, adj_26);
        // adj: visible_flags[i] = f                                                              <L 120>
        wp::adj_where(var_20, var_25, var_15, adj_20, adj_25, adj_15, adj_26);
        if (var_20) {
            wp::adj_int32(var_22, adj_22, adj_23);
            // adj: f = f & (~wp.int32(ParticleFlags.ACTIVE))                                     <L 119>
        }
        wp::adj_extract(var_19, var_17, adj_16, adj_17, adj_18);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_16);
        // adj: if particle_q[i][2] > cut_z:                                                      <L 118>
        wp::adj_where(var_8, var_14, var_2, adj_8, adj_14, adj_2, adj_15);
        if (var_8) {
            wp::adj_int32(var_11, adj_11, adj_12);
            // adj: f = f & (~wp.int32(ParticleFlags.ACTIVE))                                     <L 117>
        }
        wp::adj_address(var_material_visible, var_6, adj_material_visible, adj_4, adj_5);
        wp::adj_address(var_particle_material, var_0, adj_particle_material, adj_0, adj_4);
        // adj: if material_visible[particle_material[i]] == 0:                                   <L 116>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: f = particle_flags[i]                                                             <L 115>
        // adj: i = wp.tid()                                                                      <L 114>
        // adj: def compute_visible_flags_kernel(                                                 <L 100>
        continue;
    }
}



extern "C" __global__ void bake_vertex_uv3_kernel_cf72b964_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::float32 var_inv_grid_nx,
    wp::float32 var_inv_grid_ny,
    wp::float32 var_inv_grid_nz,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3)
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
        wp::float32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::float32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::float32 var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 6;
        wp::int32 var_18;
        const wp::float32 var_19 = 0.5;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        const wp::int32 var_24 = 2;
        wp::float32 var_25;
        wp::vec_t<3, wp::float32> var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        const wp::float32 var_29 = 0.5;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::float32 var_40;
        const wp::float32 var_41 = 0.5;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 0;
        wp::float32 var_50;
        const wp::float32 var_51 = 0.5;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        const wp::int32 var_57 = 3;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        wp::float32 var_62;
        const wp::float32 var_63 = 0.5;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::vec_t<3, wp::float32> var_66;
        const wp::int32 var_67 = 4;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        const wp::int32 var_71 = 1;
        wp::float32 var_72;
        const wp::float32 var_73 = 0.5;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::vec_t<3, wp::float32> var_76;
        const wp::int32 var_77 = 5;
        wp::int32 var_78;
        //---------
        // forward
        // def bake_vertex_uv3_kernel(                                                            <L 180>
        // p = wp.tid()                                                                           <L 196>
        var_0 = builtin_tid1d();
        // gx = float(particle_grid_xyz[p, 0])                                                    <L 197>
        var_2 = wp::address(var_particle_grid_xyz, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::float(var_4);
        // gy = float(particle_grid_xyz[p, 1])                                                    <L 198>
        var_6 = wp::address(var_particle_grid_xyz, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::float(var_8);
        // gz = float(particle_grid_xyz[p, 2])                                                    <L 199>
        var_10 = wp::address(var_particle_grid_xyz, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::float(var_12);
        // centre = wp.vec3(                                                                      <L 200>
        // gx * inv_grid_nx,                                                                      <L 201>
        var_13 = wp::mul(var_3, var_inv_grid_nx);
        // gy * inv_grid_ny,                                                                      <L 202>
        var_14 = wp::mul(var_7, var_inv_grid_ny);
        // gz * inv_grid_nz,                                                                      <L 203>
        var_15 = wp::mul(var_11, var_inv_grid_nz);
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // particle_uv3[p] = centre                                                               <L 205>
        wp::array_store(var_particle_uv3, var_0, var_16);
        // base = p * 6                                                                           <L 206>
        var_18 = wp::mul(var_0, var_17);
        // vertex_uv3[base + 0] = wp.vec3((gx - 0.5) * inv_grid_nx, centre[1], centre[2])         <L 207>
        var_20 = wp::sub(var_3, var_19);
        var_21 = wp::mul(var_20, var_inv_grid_nx);
        var_23 = wp::extract(var_16, var_22);
        var_25 = wp::extract(var_16, var_24);
        var_26 = wp::vec_t<3, wp::float32>(var_21, var_23, var_25);
        var_28 = wp::add(var_18, var_27);
        wp::array_store(var_vertex_uv3, var_28, var_26);
        // vertex_uv3[base + 1] = wp.vec3((gx + 0.5) * inv_grid_nx, centre[1], centre[2])         <L 208>
        var_30 = wp::add(var_3, var_29);
        var_31 = wp::mul(var_30, var_inv_grid_nx);
        var_33 = wp::extract(var_16, var_32);
        var_35 = wp::extract(var_16, var_34);
        var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
        var_38 = wp::add(var_18, var_37);
        wp::array_store(var_vertex_uv3, var_38, var_36);
        // vertex_uv3[base + 2] = wp.vec3(centre[0], (gy - 0.5) * inv_grid_ny, centre[2])         <L 209>
        var_40 = wp::extract(var_16, var_39);
        var_42 = wp::sub(var_7, var_41);
        var_43 = wp::mul(var_42, var_inv_grid_ny);
        var_45 = wp::extract(var_16, var_44);
        var_46 = wp::vec_t<3, wp::float32>(var_40, var_43, var_45);
        var_48 = wp::add(var_18, var_47);
        wp::array_store(var_vertex_uv3, var_48, var_46);
        // vertex_uv3[base + 3] = wp.vec3(centre[0], (gy + 0.5) * inv_grid_ny, centre[2])         <L 210>
        var_50 = wp::extract(var_16, var_49);
        var_52 = wp::add(var_7, var_51);
        var_53 = wp::mul(var_52, var_inv_grid_ny);
        var_55 = wp::extract(var_16, var_54);
        var_56 = wp::vec_t<3, wp::float32>(var_50, var_53, var_55);
        var_58 = wp::add(var_18, var_57);
        wp::array_store(var_vertex_uv3, var_58, var_56);
        // vertex_uv3[base + 4] = wp.vec3(centre[0], centre[1], (gz - 0.5) * inv_grid_nz)         <L 211>
        var_60 = wp::extract(var_16, var_59);
        var_62 = wp::extract(var_16, var_61);
        var_64 = wp::sub(var_11, var_63);
        var_65 = wp::mul(var_64, var_inv_grid_nz);
        var_66 = wp::vec_t<3, wp::float32>(var_60, var_62, var_65);
        var_68 = wp::add(var_18, var_67);
        wp::array_store(var_vertex_uv3, var_68, var_66);
        // vertex_uv3[base + 5] = wp.vec3(centre[0], centre[1], (gz + 0.5) * inv_grid_nz)         <L 212>
        var_70 = wp::extract(var_16, var_69);
        var_72 = wp::extract(var_16, var_71);
        var_74 = wp::add(var_11, var_73);
        var_75 = wp::mul(var_74, var_inv_grid_nz);
        var_76 = wp::vec_t<3, wp::float32>(var_70, var_72, var_75);
        var_78 = wp::add(var_18, var_77);
        wp::array_store(var_vertex_uv3, var_78, var_76);
    }
}



extern "C" __global__ void bake_vertex_uv3_kernel_cf72b964_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::float32 var_inv_grid_nx,
    wp::float32 var_inv_grid_ny,
    wp::float32 var_inv_grid_nz,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertex_uv3,
    wp::array_t<wp::int32> adj_particle_grid_xyz,
    wp::float32 adj_inv_grid_nx,
    wp::float32 adj_inv_grid_ny,
    wp::float32 adj_inv_grid_nz,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_uv3,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertex_uv3)
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
        wp::float32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 1;
        wp::int32* var_6;
        wp::float32 var_7;
        wp::int32 var_8;
        const wp::int32 var_9 = 2;
        wp::int32* var_10;
        wp::float32 var_11;
        wp::int32 var_12;
        wp::float32 var_13;
        wp::float32 var_14;
        wp::float32 var_15;
        wp::vec_t<3, wp::float32> var_16;
        const wp::int32 var_17 = 6;
        wp::int32 var_18;
        const wp::float32 var_19 = 0.5;
        wp::float32 var_20;
        wp::float32 var_21;
        const wp::int32 var_22 = 1;
        wp::float32 var_23;
        const wp::int32 var_24 = 2;
        wp::float32 var_25;
        wp::vec_t<3, wp::float32> var_26;
        const wp::int32 var_27 = 0;
        wp::int32 var_28;
        const wp::float32 var_29 = 0.5;
        wp::float32 var_30;
        wp::float32 var_31;
        const wp::int32 var_32 = 1;
        wp::float32 var_33;
        const wp::int32 var_34 = 2;
        wp::float32 var_35;
        wp::vec_t<3, wp::float32> var_36;
        const wp::int32 var_37 = 1;
        wp::int32 var_38;
        const wp::int32 var_39 = 0;
        wp::float32 var_40;
        const wp::float32 var_41 = 0.5;
        wp::float32 var_42;
        wp::float32 var_43;
        const wp::int32 var_44 = 2;
        wp::float32 var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 0;
        wp::float32 var_50;
        const wp::float32 var_51 = 0.5;
        wp::float32 var_52;
        wp::float32 var_53;
        const wp::int32 var_54 = 2;
        wp::float32 var_55;
        wp::vec_t<3, wp::float32> var_56;
        const wp::int32 var_57 = 3;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::float32 var_60;
        const wp::int32 var_61 = 1;
        wp::float32 var_62;
        const wp::float32 var_63 = 0.5;
        wp::float32 var_64;
        wp::float32 var_65;
        wp::vec_t<3, wp::float32> var_66;
        const wp::int32 var_67 = 4;
        wp::int32 var_68;
        const wp::int32 var_69 = 0;
        wp::float32 var_70;
        const wp::int32 var_71 = 1;
        wp::float32 var_72;
        const wp::float32 var_73 = 0.5;
        wp::float32 var_74;
        wp::float32 var_75;
        wp::vec_t<3, wp::float32> var_76;
        const wp::int32 var_77 = 5;
        wp::int32 var_78;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::float32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        wp::float32 adj_7 = {};
        wp::int32 adj_8 = {};
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::float32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::float32 adj_13 = {};
        wp::float32 adj_14 = {};
        wp::float32 adj_15 = {};
        wp::vec_t<3, wp::float32> adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::float32 adj_20 = {};
        wp::float32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::float32 adj_23 = {};
        wp::int32 adj_24 = {};
        wp::float32 adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::float32 adj_30 = {};
        wp::float32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::float32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::float32 adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        wp::float32 adj_40 = {};
        wp::float32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::float32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::float32 adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::float32 adj_50 = {};
        wp::float32 adj_51 = {};
        wp::float32 adj_52 = {};
        wp::float32 adj_53 = {};
        wp::int32 adj_54 = {};
        wp::float32 adj_55 = {};
        wp::vec_t<3, wp::float32> adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::float32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::float32 adj_62 = {};
        wp::float32 adj_63 = {};
        wp::float32 adj_64 = {};
        wp::float32 adj_65 = {};
        wp::vec_t<3, wp::float32> adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::float32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::float32 adj_73 = {};
        wp::float32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::vec_t<3, wp::float32> adj_76 = {};
        wp::int32 adj_77 = {};
        wp::int32 adj_78 = {};
        //---------
        // forward
        // def bake_vertex_uv3_kernel(                                                            <L 180>
        // p = wp.tid()                                                                           <L 196>
        var_0 = builtin_tid1d();
        // gx = float(particle_grid_xyz[p, 0])                                                    <L 197>
        var_2 = wp::address(var_particle_grid_xyz, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::float(var_4);
        // gy = float(particle_grid_xyz[p, 1])                                                    <L 198>
        var_6 = wp::address(var_particle_grid_xyz, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::float(var_8);
        // gz = float(particle_grid_xyz[p, 2])                                                    <L 199>
        var_10 = wp::address(var_particle_grid_xyz, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::float(var_12);
        // centre = wp.vec3(                                                                      <L 200>
        // gx * inv_grid_nx,                                                                      <L 201>
        var_13 = wp::mul(var_3, var_inv_grid_nx);
        // gy * inv_grid_ny,                                                                      <L 202>
        var_14 = wp::mul(var_7, var_inv_grid_ny);
        // gz * inv_grid_nz,                                                                      <L 203>
        var_15 = wp::mul(var_11, var_inv_grid_nz);
        var_16 = wp::vec_t<3, wp::float32>(var_13, var_14, var_15);
        // particle_uv3[p] = centre                                                               <L 205>
        // wp::array_store(var_particle_uv3, var_0, var_16);
        // base = p * 6                                                                           <L 206>
        var_18 = wp::mul(var_0, var_17);
        // vertex_uv3[base + 0] = wp.vec3((gx - 0.5) * inv_grid_nx, centre[1], centre[2])         <L 207>
        var_20 = wp::sub(var_3, var_19);
        var_21 = wp::mul(var_20, var_inv_grid_nx);
        var_23 = wp::extract(var_16, var_22);
        var_25 = wp::extract(var_16, var_24);
        var_26 = wp::vec_t<3, wp::float32>(var_21, var_23, var_25);
        var_28 = wp::add(var_18, var_27);
        // wp::array_store(var_vertex_uv3, var_28, var_26);
        // vertex_uv3[base + 1] = wp.vec3((gx + 0.5) * inv_grid_nx, centre[1], centre[2])         <L 208>
        var_30 = wp::add(var_3, var_29);
        var_31 = wp::mul(var_30, var_inv_grid_nx);
        var_33 = wp::extract(var_16, var_32);
        var_35 = wp::extract(var_16, var_34);
        var_36 = wp::vec_t<3, wp::float32>(var_31, var_33, var_35);
        var_38 = wp::add(var_18, var_37);
        // wp::array_store(var_vertex_uv3, var_38, var_36);
        // vertex_uv3[base + 2] = wp.vec3(centre[0], (gy - 0.5) * inv_grid_ny, centre[2])         <L 209>
        var_40 = wp::extract(var_16, var_39);
        var_42 = wp::sub(var_7, var_41);
        var_43 = wp::mul(var_42, var_inv_grid_ny);
        var_45 = wp::extract(var_16, var_44);
        var_46 = wp::vec_t<3, wp::float32>(var_40, var_43, var_45);
        var_48 = wp::add(var_18, var_47);
        // wp::array_store(var_vertex_uv3, var_48, var_46);
        // vertex_uv3[base + 3] = wp.vec3(centre[0], (gy + 0.5) * inv_grid_ny, centre[2])         <L 210>
        var_50 = wp::extract(var_16, var_49);
        var_52 = wp::add(var_7, var_51);
        var_53 = wp::mul(var_52, var_inv_grid_ny);
        var_55 = wp::extract(var_16, var_54);
        var_56 = wp::vec_t<3, wp::float32>(var_50, var_53, var_55);
        var_58 = wp::add(var_18, var_57);
        // wp::array_store(var_vertex_uv3, var_58, var_56);
        // vertex_uv3[base + 4] = wp.vec3(centre[0], centre[1], (gz - 0.5) * inv_grid_nz)         <L 211>
        var_60 = wp::extract(var_16, var_59);
        var_62 = wp::extract(var_16, var_61);
        var_64 = wp::sub(var_11, var_63);
        var_65 = wp::mul(var_64, var_inv_grid_nz);
        var_66 = wp::vec_t<3, wp::float32>(var_60, var_62, var_65);
        var_68 = wp::add(var_18, var_67);
        // wp::array_store(var_vertex_uv3, var_68, var_66);
        // vertex_uv3[base + 5] = wp.vec3(centre[0], centre[1], (gz + 0.5) * inv_grid_nz)         <L 212>
        var_70 = wp::extract(var_16, var_69);
        var_72 = wp::extract(var_16, var_71);
        var_74 = wp::add(var_11, var_73);
        var_75 = wp::mul(var_74, var_inv_grid_nz);
        var_76 = wp::vec_t<3, wp::float32>(var_70, var_72, var_75);
        var_78 = wp::add(var_18, var_77);
        // wp::array_store(var_vertex_uv3, var_78, var_76);
        //---------
        // reverse
        wp::adj_array_store(var_vertex_uv3, var_78, var_76, adj_vertex_uv3, adj_78, adj_76);
        wp::adj_add(var_18, var_77, adj_18, adj_77, adj_78);
        wp::adj_vec_t(var_70, var_72, var_75, adj_70, adj_72, adj_75, adj_76);
        wp::adj_mul(var_74, var_inv_grid_nz, adj_74, adj_inv_grid_nz, adj_75);
        wp::adj_add(var_11, var_73, adj_11, adj_73, adj_74);
        wp::adj_extract(var_16, var_71, adj_16, adj_71, adj_72);
        wp::adj_extract(var_16, var_69, adj_16, adj_69, adj_70);
        // adj: vertex_uv3[base + 5] = wp.vec3(centre[0], centre[1], (gz + 0.5) * inv_grid_nz)    <L 212>
        wp::adj_array_store(var_vertex_uv3, var_68, var_66, adj_vertex_uv3, adj_68, adj_66);
        wp::adj_add(var_18, var_67, adj_18, adj_67, adj_68);
        wp::adj_vec_t(var_60, var_62, var_65, adj_60, adj_62, adj_65, adj_66);
        wp::adj_mul(var_64, var_inv_grid_nz, adj_64, adj_inv_grid_nz, adj_65);
        wp::adj_sub(var_11, var_63, adj_11, adj_63, adj_64);
        wp::adj_extract(var_16, var_61, adj_16, adj_61, adj_62);
        wp::adj_extract(var_16, var_59, adj_16, adj_59, adj_60);
        // adj: vertex_uv3[base + 4] = wp.vec3(centre[0], centre[1], (gz - 0.5) * inv_grid_nz)    <L 211>
        wp::adj_array_store(var_vertex_uv3, var_58, var_56, adj_vertex_uv3, adj_58, adj_56);
        wp::adj_add(var_18, var_57, adj_18, adj_57, adj_58);
        wp::adj_vec_t(var_50, var_53, var_55, adj_50, adj_53, adj_55, adj_56);
        wp::adj_extract(var_16, var_54, adj_16, adj_54, adj_55);
        wp::adj_mul(var_52, var_inv_grid_ny, adj_52, adj_inv_grid_ny, adj_53);
        wp::adj_add(var_7, var_51, adj_7, adj_51, adj_52);
        wp::adj_extract(var_16, var_49, adj_16, adj_49, adj_50);
        // adj: vertex_uv3[base + 3] = wp.vec3(centre[0], (gy + 0.5) * inv_grid_ny, centre[2])    <L 210>
        wp::adj_array_store(var_vertex_uv3, var_48, var_46, adj_vertex_uv3, adj_48, adj_46);
        wp::adj_add(var_18, var_47, adj_18, adj_47, adj_48);
        wp::adj_vec_t(var_40, var_43, var_45, adj_40, adj_43, adj_45, adj_46);
        wp::adj_extract(var_16, var_44, adj_16, adj_44, adj_45);
        wp::adj_mul(var_42, var_inv_grid_ny, adj_42, adj_inv_grid_ny, adj_43);
        wp::adj_sub(var_7, var_41, adj_7, adj_41, adj_42);
        wp::adj_extract(var_16, var_39, adj_16, adj_39, adj_40);
        // adj: vertex_uv3[base + 2] = wp.vec3(centre[0], (gy - 0.5) * inv_grid_ny, centre[2])    <L 209>
        wp::adj_array_store(var_vertex_uv3, var_38, var_36, adj_vertex_uv3, adj_38, adj_36);
        wp::adj_add(var_18, var_37, adj_18, adj_37, adj_38);
        wp::adj_vec_t(var_31, var_33, var_35, adj_31, adj_33, adj_35, adj_36);
        wp::adj_extract(var_16, var_34, adj_16, adj_34, adj_35);
        wp::adj_extract(var_16, var_32, adj_16, adj_32, adj_33);
        wp::adj_mul(var_30, var_inv_grid_nx, adj_30, adj_inv_grid_nx, adj_31);
        wp::adj_add(var_3, var_29, adj_3, adj_29, adj_30);
        // adj: vertex_uv3[base + 1] = wp.vec3((gx + 0.5) * inv_grid_nx, centre[1], centre[2])    <L 208>
        wp::adj_array_store(var_vertex_uv3, var_28, var_26, adj_vertex_uv3, adj_28, adj_26);
        wp::adj_add(var_18, var_27, adj_18, adj_27, adj_28);
        wp::adj_vec_t(var_21, var_23, var_25, adj_21, adj_23, adj_25, adj_26);
        wp::adj_extract(var_16, var_24, adj_16, adj_24, adj_25);
        wp::adj_extract(var_16, var_22, adj_16, adj_22, adj_23);
        wp::adj_mul(var_20, var_inv_grid_nx, adj_20, adj_inv_grid_nx, adj_21);
        wp::adj_sub(var_3, var_19, adj_3, adj_19, adj_20);
        // adj: vertex_uv3[base + 0] = wp.vec3((gx - 0.5) * inv_grid_nx, centre[1], centre[2])    <L 207>
        wp::adj_mul(var_0, var_17, adj_0, adj_17, adj_18);
        // adj: base = p * 6                                                                      <L 206>
        wp::adj_array_store(var_particle_uv3, var_0, var_16, adj_particle_uv3, adj_0, adj_16);
        // adj: particle_uv3[p] = centre                                                          <L 205>
        wp::adj_vec_t(var_13, var_14, var_15, adj_13, adj_14, adj_15, adj_16);
        wp::adj_mul(var_11, var_inv_grid_nz, adj_11, adj_inv_grid_nz, adj_15);
        // adj: gz * inv_grid_nz,                                                                 <L 203>
        wp::adj_mul(var_7, var_inv_grid_ny, adj_7, adj_inv_grid_ny, adj_14);
        // adj: gy * inv_grid_ny,                                                                 <L 202>
        wp::adj_mul(var_3, var_inv_grid_nx, adj_3, adj_inv_grid_nx, adj_13);
        // adj: gx * inv_grid_nx,                                                                 <L 201>
        // adj: centre = wp.vec3(                                                                 <L 200>
        wp::adj_float(var_12, adj_10, adj_11);
        wp::adj_address(var_particle_grid_xyz, var_0, var_9, adj_particle_grid_xyz, adj_0, adj_9, adj_10);
        // adj: gz = float(particle_grid_xyz[p, 2])                                               <L 199>
        wp::adj_float(var_8, adj_6, adj_7);
        wp::adj_address(var_particle_grid_xyz, var_0, var_5, adj_particle_grid_xyz, adj_0, adj_5, adj_6);
        // adj: gy = float(particle_grid_xyz[p, 1])                                               <L 198>
        wp::adj_float(var_4, adj_2, adj_3);
        wp::adj_address(var_particle_grid_xyz, var_0, var_1, adj_particle_grid_xyz, adj_0, adj_1, adj_2);
        // adj: gx = float(particle_grid_xyz[p, 0])                                               <L 197>
        // adj: p = wp.tid()                                                                      <L 196>
        // adj: def bake_vertex_uv3_kernel(                                                       <L 180>
        continue;
    }
}



extern "C" __global__ void emit_triangles_kernel_5330e039_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::int32 var_max_triangles,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        const wp::int32 var_9 = 255;
        bool var_10;
        const wp::int32 var_11 = 5;
        wp::range_t var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        const wp::int32 var_23 = 3;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 3;
        wp::int32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        const wp::int32 var_47 = 0;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        bool var_50;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = 1;
        const wp::int32 var_53 = 2;
        //---------
        // forward
        // def emit_triangles_kernel(                                                             <L 373>
        // cx, cy, cz = wp.tid()                                                                  <L 386>
        builtin_tid3d(var_0, var_1, var_2);
        // case = cube_cases[cx, cy, cz]                                                          <L 387>
        var_3 = wp::address(var_cube_cases, var_0, var_1, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if case == 0 or case == 255:                                                           <L 388>
        var_8 = (var_4 == var_7);
        var_6 = var_8;
        if (!var_6) {
            var_10 = (var_4 == var_9);
            var_6 = var_6 || var_10;
        }
        if (var_6) {
            // return                                                                             <L 389>
            continue;
        }
        // for t in range(5):                                                                     <L 393>
        var_12 = wp::range(var_11);
        start_for_1:;
            if (iter_cmp(var_12) == 0) goto end_for_1;
            var_13 = wp::iter_next(var_12);
            // e0 = case_triangles[case, t * 3 + 0]                                               <L 394>
            var_15 = wp::mul(var_13, var_14);
            var_17 = wp::add(var_15, var_16);
            var_18 = wp::address(var_case_triangles, var_4, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // if e0 < 0:                                                                         <L 395>
            var_22 = (var_19 < var_21);
            if (var_22) {
                // return                                                                         <L 396>
                continue;
            }
            // e1 = case_triangles[case, t * 3 + 1]                                               <L 397>
            var_24 = wp::mul(var_13, var_23);
            var_26 = wp::add(var_24, var_25);
            var_27 = wp::address(var_case_triangles, var_4, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // e2 = case_triangles[case, t * 3 + 2]                                               <L 398>
            var_31 = wp::mul(var_13, var_30);
            var_33 = wp::add(var_31, var_32);
            var_34 = wp::address(var_case_triangles, var_4, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // v0 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e0)       <L 399>
            var_37 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_19);
            // v1 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e1)       <L 400>
            var_38 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_28);
            // v2 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e2)       <L 401>
            var_39 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_35);
            // if v0 < 0 or v1 < 0 or v2 < 0:                                                     <L 402>
            var_42 = (var_37 < var_41);
            var_40 = var_42;
            if (!var_40) {
                var_44 = (var_38 < var_43);
                var_40 = var_40 || var_44;
            }
            if (!var_40) {
                var_46 = (var_39 < var_45);
                var_40 = var_40 || var_46;
            }
            if (var_40) {
                // continue                                                                       <L 403>
                goto start_for_1;
            }
            // idx = wp.atomic_add(tri_count, 0, 1)                                               <L 404>
            var_49 = wp::atomic_add(var_tri_count, var_47, var_48);
            // if idx < max_triangles:                                                            <L 405>
            var_50 = (var_49 < var_max_triangles);
            if (var_50) {
                // tri_indices[idx, 0] = v0                                                       <L 406>
                wp::array_store(var_tri_indices, var_49, var_51, var_37);
                // tri_indices[idx, 1] = v1                                                       <L 407>
                wp::array_store(var_tri_indices, var_49, var_52, var_38);
                // tri_indices[idx, 2] = v2                                                       <L 408>
                wp::array_store(var_tri_indices, var_49, var_53, var_39);
            }
            goto start_for_1;
        end_for_1:;
    }
}



extern "C" __global__ void emit_triangles_kernel_5330e039_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::int32 var_max_triangles,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> adj_grid_to_particle,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_cube_cases,
    wp::array_t<wp::int32> adj_corner_offsets,
    wp::array_t<wp::int32> adj_edge_corners,
    wp::array_t<wp::int32> adj_edge_base_dir,
    wp::array_t<wp::int32> adj_case_triangles,
    wp::int32 adj_max_triangles,
    wp::array_t<wp::int32> adj_tri_count,
    wp::array_t<wp::int32> adj_tri_indices)
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
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        bool var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        const wp::int32 var_9 = 255;
        bool var_10;
        const wp::int32 var_11 = 5;
        wp::range_t var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 3;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        wp::int32 var_17;
        wp::int32* var_18;
        wp::int32 var_19;
        wp::int32 var_20;
        const wp::int32 var_21 = 0;
        bool var_22;
        const wp::int32 var_23 = 3;
        wp::int32 var_24;
        const wp::int32 var_25 = 1;
        wp::int32 var_26;
        wp::int32* var_27;
        wp::int32 var_28;
        wp::int32 var_29;
        const wp::int32 var_30 = 3;
        wp::int32 var_31;
        const wp::int32 var_32 = 2;
        wp::int32 var_33;
        wp::int32* var_34;
        wp::int32 var_35;
        wp::int32 var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        bool var_40;
        const wp::int32 var_41 = 0;
        bool var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 0;
        bool var_46;
        const wp::int32 var_47 = 0;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        bool var_50;
        const wp::int32 var_51 = 0;
        const wp::int32 var_52 = 1;
        const wp::int32 var_53 = 2;
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
        bool adj_8 = {};
        wp::int32 adj_9 = {};
        bool adj_10 = {};
        wp::int32 adj_11 = {};
        wp::range_t adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        wp::int32 adj_17 = {};
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
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::int32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::int32 adj_39 = {};
        bool adj_40 = {};
        wp::int32 adj_41 = {};
        bool adj_42 = {};
        wp::int32 adj_43 = {};
        bool adj_44 = {};
        wp::int32 adj_45 = {};
        bool adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        bool adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        //---------
        // forward
        // def emit_triangles_kernel(                                                             <L 373>
        // cx, cy, cz = wp.tid()                                                                  <L 386>
        builtin_tid3d(var_0, var_1, var_2);
        // case = cube_cases[cx, cy, cz]                                                          <L 387>
        var_3 = wp::address(var_cube_cases, var_0, var_1, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // if case == 0 or case == 255:                                                           <L 388>
        var_8 = (var_4 == var_7);
        var_6 = var_8;
        if (!var_6) {
            var_10 = (var_4 == var_9);
            var_6 = var_6 || var_10;
        }
        if (var_6) {
            // return                                                                             <L 389>
            goto label0;
        }
        // for t in range(5):                                                                     <L 393>
        var_12 = wp::range(var_11);
        //---------
        // reverse
        var_12 = wp::iter_reverse(var_12);
        start_for_1:;
            if (iter_cmp(var_12) == 0) goto end_for_1;
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
            // e0 = case_triangles[case, t * 3 + 0]                                               <L 394>
            var_15 = wp::mul(var_13, var_14);
            var_17 = wp::add(var_15, var_16);
            var_18 = wp::address(var_case_triangles, var_4, var_17);
            var_20 = wp::load(var_18);
            var_19 = wp::copy(var_20);
            // if e0 < 0:                                                                         <L 395>
            var_22 = (var_19 < var_21);
            if (var_22) {
                // return                                                                         <L 396>
                goto label3;
            }
            // e1 = case_triangles[case, t * 3 + 1]                                               <L 397>
            var_24 = wp::mul(var_13, var_23);
            var_26 = wp::add(var_24, var_25);
            var_27 = wp::address(var_case_triangles, var_4, var_26);
            var_29 = wp::load(var_27);
            var_28 = wp::copy(var_29);
            // e2 = case_triangles[case, t * 3 + 2]                                               <L 398>
            var_31 = wp::mul(var_13, var_30);
            var_33 = wp::add(var_31, var_32);
            var_34 = wp::address(var_case_triangles, var_4, var_33);
            var_36 = wp::load(var_34);
            var_35 = wp::copy(var_36);
            // v0 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e0)       <L 399>
            var_37 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_19);
            // v1 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e1)       <L 400>
            var_38 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_28);
            // v2 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e2)       <L 401>
            var_39 = _edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_35);
            // if v0 < 0 or v1 < 0 or v2 < 0:                                                     <L 402>
            var_42 = (var_37 < var_41);
            var_40 = var_42;
            if (!var_40) {
                var_44 = (var_38 < var_43);
                var_40 = var_40 || var_44;
            }
            if (!var_40) {
                var_46 = (var_39 < var_45);
                var_40 = var_40 || var_46;
            }
            if (var_40) {
                // continue                                                                       <L 403>
                goto start_for_1;
            }
            // idx = wp.atomic_add(tri_count, 0, 1)                                               <L 404>
            // var_49 = wp::atomic_add(var_tri_count, var_47, var_48);
            // if idx < max_triangles:                                                            <L 405>
            var_50 = (var_49 < var_max_triangles);
            if (var_50) {
                // tri_indices[idx, 0] = v0                                                       <L 406>
                // wp::array_store(var_tri_indices, var_49, var_51, var_37);
                // tri_indices[idx, 1] = v1                                                       <L 407>
                // wp::array_store(var_tri_indices, var_49, var_52, var_38);
                // tri_indices[idx, 2] = v2                                                       <L 408>
                // wp::array_store(var_tri_indices, var_49, var_53, var_39);
            }
            if (var_50) {
                wp::adj_array_store(var_tri_indices, var_49, var_53, var_39, adj_tri_indices, adj_49, adj_53, adj_39);
                // adj: tri_indices[idx, 2] = v2                                                  <L 408>
                wp::adj_array_store(var_tri_indices, var_49, var_52, var_38, adj_tri_indices, adj_49, adj_52, adj_38);
                // adj: tri_indices[idx, 1] = v1                                                  <L 407>
                wp::adj_array_store(var_tri_indices, var_49, var_51, var_37, adj_tri_indices, adj_49, adj_51, adj_37);
                // adj: tri_indices[idx, 0] = v0                                                  <L 406>
            }
            // adj: if idx < max_triangles:                                                       <L 405>
            wp::adj_atomic_add(var_tri_count, var_47, var_48, adj_tri_count, adj_47, adj_48, adj_49);
            // adj: idx = wp.atomic_add(tri_count, 0, 1)                                          <L 404>
            if (var_40) {
                // adj: continue                                                                  <L 403>
            }
            if (!var_40) {
            }
            if (!var_40) {
            }
            // adj: if v0 < 0 or v1 < 0 or v2 < 0:                                                <L 402>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_35, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_0, adj_1, adj_2, adj_35, adj_39);
            // adj: v2 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e2)  <L 401>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_28, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_0, adj_1, adj_2, adj_28, adj_38);
            // adj: v1 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e1)  <L 400>
            adj__edge_to_vertex_id_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_0, var_1, var_2, var_19, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_0, adj_1, adj_2, adj_19, adj_37);
            // adj: v0 = _edge_to_vertex_id(grid_to_particle, particle_flags, corner_offsets, edge_corners, edge_base_dir, cx, cy, cz, e0)  <L 399>
            wp::adj_copy(var_36, adj_34, adj_35);
            wp::adj_address(var_case_triangles, var_4, var_33, adj_case_triangles, adj_4, adj_33, adj_34);
            wp::adj_add(var_31, var_32, adj_31, adj_32, adj_33);
            wp::adj_mul(var_13, var_30, adj_13, adj_30, adj_31);
            // adj: e2 = case_triangles[case, t * 3 + 2]                                          <L 398>
            wp::adj_copy(var_29, adj_27, adj_28);
            wp::adj_address(var_case_triangles, var_4, var_26, adj_case_triangles, adj_4, adj_26, adj_27);
            wp::adj_add(var_24, var_25, adj_24, adj_25, adj_26);
            wp::adj_mul(var_13, var_23, adj_13, adj_23, adj_24);
            // adj: e1 = case_triangles[case, t * 3 + 1]                                          <L 397>
            if (var_22) {
                label3:;
                // adj: return                                                                    <L 396>
            }
            // adj: if e0 < 0:                                                                    <L 395>
            wp::adj_copy(var_20, adj_18, adj_19);
            wp::adj_address(var_case_triangles, var_4, var_17, adj_case_triangles, adj_4, adj_17, adj_18);
            wp::adj_add(var_15, var_16, adj_15, adj_16, adj_17);
            wp::adj_mul(var_13, var_14, adj_13, adj_14, adj_15);
            // adj: e0 = case_triangles[case, t * 3 + 0]                                          <L 394>
        	goto start_for_1;
        end_for_1:;
        wp::adj_range(var_11, adj_11, adj_12);
        // adj: for t in range(5):                                                                <L 393>
        if (var_6) {
            label0:;
            // adj: return                                                                        <L 389>
        }
        if (!var_6) {
        }
        // adj: if case == 0 or case == 255:                                                      <L 388>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_cube_cases, var_0, var_1, var_2, adj_cube_cases, adj_0, adj_1, adj_2, adj_3);
        // adj: case = cube_cases[cx, cy, cz]                                                     <L 387>
        // adj: cx, cy, cz = wp.tid()                                                             <L 386>
        // adj: def emit_triangles_kernel(                                                        <L 373>
        continue;
    }
}



extern "C" __global__ void mark_dirty_cubes_from_particles_device_count_kernel_42d6e7ae_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_ids,
    wp::array_t<wp::int32> var_particle_count,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::int32 var_nx_cells,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::int32 var_stamp,
    wp::array_t<wp::int32> var_dirty_marks,
    wp::array_t<wp::int32> var_dirty_count,
    wp::array_t<wp::int32> var_dirty_cube_ids)
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
        const wp::int32 var_10 = 0;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 2;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        bool var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        bool var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        bool var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        bool var_42;
        const wp::int32 var_43 = 0;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::shape_t* var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        wp::shape_t var_49;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        bool var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        bool var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        bool var_59;
        const wp::int32 var_60 = 0;
        const wp::int32 var_61 = 1;
        wp::int32 var_62;
        wp::shape_t* var_63;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        wp::shape_t var_66;
        bool var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 1;
        wp::int32 var_73;
        bool var_74;
        const wp::int32 var_75 = 0;
        bool var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        bool var_80;
        const wp::int32 var_81 = 0;
        bool var_82;
        bool var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        bool var_86;
        const wp::int32 var_87 = 0;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        wp::shape_t* var_90;
        const wp::int32 var_91 = 0;
        wp::int32 var_92;
        wp::shape_t var_93;
        bool var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        wp::int32 var_98;
        const wp::int32 var_99 = 1;
        wp::int32 var_100;
        bool var_101;
        const wp::int32 var_102 = 0;
        bool var_103;
        bool var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        bool var_107;
        const wp::int32 var_108 = 0;
        const wp::int32 var_109 = 1;
        wp::int32 var_110;
        wp::shape_t* var_111;
        const wp::int32 var_112 = 0;
        wp::int32 var_113;
        wp::shape_t var_114;
        bool var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        const wp::int32 var_125 = 1;
        wp::int32 var_126;
        bool var_127;
        const wp::int32 var_128 = 0;
        bool var_129;
        bool var_130;
        const wp::int32 var_131 = 0;
        wp::int32 var_132;
        bool var_133;
        const wp::int32 var_134 = 0;
        bool var_135;
        bool var_136;
        const wp::int32 var_137 = 0;
        wp::int32 var_138;
        bool var_139;
        const wp::int32 var_140 = 0;
        bool var_141;
        bool var_142;
        wp::int32 var_143;
        wp::int32 var_144;
        bool var_145;
        const wp::int32 var_146 = 0;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        wp::shape_t* var_149;
        const wp::int32 var_150 = 0;
        wp::int32 var_151;
        wp::shape_t var_152;
        bool var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        const wp::int32 var_158 = 1;
        wp::int32 var_159;
        bool var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        bool var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        bool var_166;
        const wp::int32 var_167 = 0;
        const wp::int32 var_168 = 1;
        wp::int32 var_169;
        wp::shape_t* var_170;
        const wp::int32 var_171 = 0;
        wp::int32 var_172;
        wp::shape_t var_173;
        bool var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::int32 var_183;
        const wp::int32 var_184 = 1;
        wp::int32 var_185;
        bool var_186;
        const wp::int32 var_187 = 0;
        bool var_188;
        bool var_189;
        const wp::int32 var_190 = 0;
        wp::int32 var_191;
        bool var_192;
        const wp::int32 var_193 = 0;
        bool var_194;
        bool var_195;
        wp::int32 var_196;
        wp::int32 var_197;
        bool var_198;
        const wp::int32 var_199 = 0;
        const wp::int32 var_200 = 1;
        wp::int32 var_201;
        wp::shape_t* var_202;
        const wp::int32 var_203 = 0;
        wp::int32 var_204;
        wp::shape_t var_205;
        bool var_206;
        wp::int32 var_207;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        const wp::int32 var_211 = 1;
        wp::int32 var_212;
        bool var_213;
        const wp::int32 var_214 = 0;
        bool var_215;
        bool var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        bool var_219;
        const wp::int32 var_220 = 0;
        const wp::int32 var_221 = 1;
        wp::int32 var_222;
        wp::shape_t* var_223;
        const wp::int32 var_224 = 0;
        wp::int32 var_225;
        wp::shape_t var_226;
        bool var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::int32 var_233;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::int32 var_241;
        wp::int32 var_242;
        wp::int32 var_243;
        //---------
        // forward
        // def mark_dirty_cubes_from_particles_device_count_kernel(                               <L 517>
        // i = wp.tid()                                                                           <L 530>
        var_0 = builtin_tid1d();
        // if i >= particle_count[0]:                                                             <L 531>
        var_2 = wp::address(var_particle_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 532>
            continue;
        }
        // p = particle_ids[i]                                                                    <L 533>
        var_5 = wp::address(var_particle_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if p < 0:                                                                              <L 534>
        var_9 = (var_6 < var_8);
        if (var_9) {
            // return                                                                             <L 535>
            continue;
        }
        // gx = particle_grid_xyz[p, 0]                                                           <L 536>
        var_11 = wp::address(var_particle_grid_xyz, var_6, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // gy = particle_grid_xyz[p, 1]                                                           <L 537>
        var_15 = wp::address(var_particle_grid_xyz, var_6, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // gz = particle_grid_xyz[p, 2]                                                           <L 538>
        var_19 = wp::address(var_particle_grid_xyz, var_6, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // for ox in range(2):                                                                    <L 539>
        // cx = gx - ox                                                                           <L 540>
        var_23 = wp::sub(var_12, var_22);
        // if cx >= 0 and cx < nx_cells:                                                          <L 541>
        var_26 = (var_23 >= var_25);
        var_24 = var_26;
        if (var_24) {
            var_27 = (var_23 < var_nx_cells);
            var_24 = var_24 && var_27;
        }
        if (var_24) {
            // for oy in range(2):                                                                <L 542>
            // cy = gy - oy                                                                       <L 543>
            var_29 = wp::sub(var_16, var_28);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_32 = (var_29 >= var_31);
            var_30 = var_32;
            if (var_30) {
                var_33 = (var_29 < var_ny_cells);
                var_30 = var_30 && var_33;
            }
            if (var_30) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_35 = wp::sub(var_20, var_34);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_38 = (var_35 >= var_37);
                var_36 = var_38;
                if (var_36) {
                    var_39 = (var_35 < var_nz_cells);
                    var_36 = var_36 && var_39;
                }
                if (var_36) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_40 = _cube_flat_id_0(var_23, var_29, var_35, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_41 = wp::atomic_exch(var_dirty_marks, var_40, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_42 = (var_41 != var_stamp);
                    if (var_42) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_45 = wp::atomic_add(var_dirty_count, var_43, var_44);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_46 = &(var_dirty_cube_ids.shape);
                        var_49 = wp::load(var_46);
                        var_48 = wp::extract(var_49, var_47);
                        var_50 = (var_45 < var_48);
                        if (var_50) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_45, var_40);
                        }
                    }
                }
                // cz = gz - oz                                                                   <L 546>
                var_52 = wp::sub(var_20, var_51);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_55 = (var_52 >= var_54);
                var_53 = var_55;
                if (var_53) {
                    var_56 = (var_52 < var_nz_cells);
                    var_53 = var_53 && var_56;
                }
                if (var_53) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_57 = _cube_flat_id_0(var_23, var_29, var_52, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_58 = wp::atomic_exch(var_dirty_marks, var_57, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_59 = (var_58 != var_stamp);
                    if (var_59) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_62 = wp::atomic_add(var_dirty_count, var_60, var_61);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_63 = &(var_dirty_cube_ids.shape);
                        var_66 = wp::load(var_63);
                        var_65 = wp::extract(var_66, var_64);
                        var_67 = (var_62 < var_65);
                        if (var_67) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_62, var_57);
                        }
                    }
                    var_68 = wp::where(var_59, var_62, var_45);
                }
                var_69 = wp::where(var_53, var_57, var_40);
                var_70 = wp::where(var_53, var_58, var_41);
                var_71 = wp::where(var_53, var_68, var_45);
            }
            // cy = gy - oy                                                                       <L 543>
            var_73 = wp::sub(var_16, var_72);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_76 = (var_73 >= var_75);
            var_74 = var_76;
            if (var_74) {
                var_77 = (var_73 < var_ny_cells);
                var_74 = var_74 && var_77;
            }
            if (var_74) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_79 = wp::sub(var_20, var_78);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_82 = (var_79 >= var_81);
                var_80 = var_82;
                if (var_80) {
                    var_83 = (var_79 < var_nz_cells);
                    var_80 = var_80 && var_83;
                }
                if (var_80) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_84 = _cube_flat_id_0(var_23, var_73, var_79, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_85 = wp::atomic_exch(var_dirty_marks, var_84, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_86 = (var_85 != var_stamp);
                    if (var_86) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_89 = wp::atomic_add(var_dirty_count, var_87, var_88);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_90 = &(var_dirty_cube_ids.shape);
                        var_93 = wp::load(var_90);
                        var_92 = wp::extract(var_93, var_91);
                        var_94 = (var_89 < var_92);
                        if (var_94) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_89, var_84);
                        }
                    }
                    var_95 = wp::where(var_86, var_89, var_71);
                }
                var_96 = wp::where(var_80, var_84, var_69);
                var_97 = wp::where(var_80, var_85, var_70);
                var_98 = wp::where(var_80, var_95, var_71);
                // cz = gz - oz                                                                   <L 546>
                var_100 = wp::sub(var_20, var_99);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_103 = (var_100 >= var_102);
                var_101 = var_103;
                if (var_101) {
                    var_104 = (var_100 < var_nz_cells);
                    var_101 = var_101 && var_104;
                }
                if (var_101) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_105 = _cube_flat_id_0(var_23, var_73, var_100, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_106 = wp::atomic_exch(var_dirty_marks, var_105, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_107 = (var_106 != var_stamp);
                    if (var_107) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_110 = wp::atomic_add(var_dirty_count, var_108, var_109);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_111 = &(var_dirty_cube_ids.shape);
                        var_114 = wp::load(var_111);
                        var_113 = wp::extract(var_114, var_112);
                        var_115 = (var_110 < var_113);
                        if (var_115) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_110, var_105);
                        }
                    }
                    var_116 = wp::where(var_107, var_110, var_98);
                }
                var_117 = wp::where(var_101, var_105, var_96);
                var_118 = wp::where(var_101, var_106, var_97);
                var_119 = wp::where(var_101, var_116, var_98);
            }
            var_120 = wp::where(var_74, var_99, var_51);
            var_121 = wp::where(var_74, var_100, var_52);
            var_122 = wp::where(var_74, var_117, var_69);
            var_123 = wp::where(var_74, var_118, var_70);
            var_124 = wp::where(var_74, var_119, var_71);
        }
        // cx = gx - ox                                                                           <L 540>
        var_126 = wp::sub(var_12, var_125);
        // if cx >= 0 and cx < nx_cells:                                                          <L 541>
        var_129 = (var_126 >= var_128);
        var_127 = var_129;
        if (var_127) {
            var_130 = (var_126 < var_nx_cells);
            var_127 = var_127 && var_130;
        }
        if (var_127) {
            // for oy in range(2):                                                                <L 542>
            // cy = gy - oy                                                                       <L 543>
            var_132 = wp::sub(var_16, var_131);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_135 = (var_132 >= var_134);
            var_133 = var_135;
            if (var_133) {
                var_136 = (var_132 < var_ny_cells);
                var_133 = var_133 && var_136;
            }
            if (var_133) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_138 = wp::sub(var_20, var_137);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_141 = (var_138 >= var_140);
                var_139 = var_141;
                if (var_139) {
                    var_142 = (var_138 < var_nz_cells);
                    var_139 = var_139 && var_142;
                }
                if (var_139) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_143 = _cube_flat_id_0(var_126, var_132, var_138, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_144 = wp::atomic_exch(var_dirty_marks, var_143, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_145 = (var_144 != var_stamp);
                    if (var_145) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_148 = wp::atomic_add(var_dirty_count, var_146, var_147);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_149 = &(var_dirty_cube_ids.shape);
                        var_152 = wp::load(var_149);
                        var_151 = wp::extract(var_152, var_150);
                        var_153 = (var_148 < var_151);
                        if (var_153) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_148, var_143);
                        }
                    }
                    var_154 = wp::where(var_145, var_148, var_124);
                }
                var_155 = wp::where(var_139, var_143, var_122);
                var_156 = wp::where(var_139, var_144, var_123);
                var_157 = wp::where(var_139, var_154, var_124);
                // cz = gz - oz                                                                   <L 546>
                var_159 = wp::sub(var_20, var_158);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_162 = (var_159 >= var_161);
                var_160 = var_162;
                if (var_160) {
                    var_163 = (var_159 < var_nz_cells);
                    var_160 = var_160 && var_163;
                }
                if (var_160) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_164 = _cube_flat_id_0(var_126, var_132, var_159, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_165 = wp::atomic_exch(var_dirty_marks, var_164, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_166 = (var_165 != var_stamp);
                    if (var_166) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_169 = wp::atomic_add(var_dirty_count, var_167, var_168);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_170 = &(var_dirty_cube_ids.shape);
                        var_173 = wp::load(var_170);
                        var_172 = wp::extract(var_173, var_171);
                        var_174 = (var_169 < var_172);
                        if (var_174) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_169, var_164);
                        }
                    }
                    var_175 = wp::where(var_166, var_169, var_157);
                }
                var_176 = wp::where(var_160, var_164, var_155);
                var_177 = wp::where(var_160, var_165, var_156);
                var_178 = wp::where(var_160, var_175, var_157);
            }
            var_179 = wp::where(var_133, var_158, var_120);
            var_180 = wp::where(var_133, var_159, var_121);
            var_181 = wp::where(var_133, var_176, var_122);
            var_182 = wp::where(var_133, var_177, var_123);
            var_183 = wp::where(var_133, var_178, var_124);
            // cy = gy - oy                                                                       <L 543>
            var_185 = wp::sub(var_16, var_184);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_188 = (var_185 >= var_187);
            var_186 = var_188;
            if (var_186) {
                var_189 = (var_185 < var_ny_cells);
                var_186 = var_186 && var_189;
            }
            if (var_186) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_191 = wp::sub(var_20, var_190);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_194 = (var_191 >= var_193);
                var_192 = var_194;
                if (var_192) {
                    var_195 = (var_191 < var_nz_cells);
                    var_192 = var_192 && var_195;
                }
                if (var_192) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_196 = _cube_flat_id_0(var_126, var_185, var_191, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_197 = wp::atomic_exch(var_dirty_marks, var_196, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_198 = (var_197 != var_stamp);
                    if (var_198) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_201 = wp::atomic_add(var_dirty_count, var_199, var_200);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_202 = &(var_dirty_cube_ids.shape);
                        var_205 = wp::load(var_202);
                        var_204 = wp::extract(var_205, var_203);
                        var_206 = (var_201 < var_204);
                        if (var_206) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_201, var_196);
                        }
                    }
                    var_207 = wp::where(var_198, var_201, var_183);
                }
                var_208 = wp::where(var_192, var_196, var_181);
                var_209 = wp::where(var_192, var_197, var_182);
                var_210 = wp::where(var_192, var_207, var_183);
                // cz = gz - oz                                                                   <L 546>
                var_212 = wp::sub(var_20, var_211);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_215 = (var_212 >= var_214);
                var_213 = var_215;
                if (var_213) {
                    var_216 = (var_212 < var_nz_cells);
                    var_213 = var_213 && var_216;
                }
                if (var_213) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_217 = _cube_flat_id_0(var_126, var_185, var_212, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    var_218 = wp::atomic_exch(var_dirty_marks, var_217, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_219 = (var_218 != var_stamp);
                    if (var_219) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        var_222 = wp::atomic_add(var_dirty_count, var_220, var_221);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_223 = &(var_dirty_cube_ids.shape);
                        var_226 = wp::load(var_223);
                        var_225 = wp::extract(var_226, var_224);
                        var_227 = (var_222 < var_225);
                        if (var_227) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            wp::array_store(var_dirty_cube_ids, var_222, var_217);
                        }
                    }
                    var_228 = wp::where(var_219, var_222, var_210);
                }
                var_229 = wp::where(var_213, var_217, var_208);
                var_230 = wp::where(var_213, var_218, var_209);
                var_231 = wp::where(var_213, var_228, var_210);
            }
            var_232 = wp::where(var_186, var_211, var_179);
            var_233 = wp::where(var_186, var_212, var_180);
            var_234 = wp::where(var_186, var_229, var_181);
            var_235 = wp::where(var_186, var_230, var_182);
            var_236 = wp::where(var_186, var_231, var_183);
        }
        var_237 = wp::where(var_127, var_184, var_72);
        var_238 = wp::where(var_127, var_185, var_73);
        var_239 = wp::where(var_127, var_232, var_120);
        var_240 = wp::where(var_127, var_233, var_121);
        var_241 = wp::where(var_127, var_234, var_122);
        var_242 = wp::where(var_127, var_235, var_123);
        var_243 = wp::where(var_127, var_236, var_124);
    }
}



extern "C" __global__ void mark_dirty_cubes_from_particles_device_count_kernel_42d6e7ae_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_ids,
    wp::array_t<wp::int32> var_particle_count,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::int32 var_nx_cells,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::int32 var_stamp,
    wp::array_t<wp::int32> var_dirty_marks,
    wp::array_t<wp::int32> var_dirty_count,
    wp::array_t<wp::int32> var_dirty_cube_ids,
    wp::array_t<wp::int32> adj_particle_ids,
    wp::array_t<wp::int32> adj_particle_count,
    wp::array_t<wp::int32> adj_particle_grid_xyz,
    wp::int32 adj_nx_cells,
    wp::int32 adj_ny_cells,
    wp::int32 adj_nz_cells,
    wp::int32 adj_stamp,
    wp::array_t<wp::int32> adj_dirty_marks,
    wp::array_t<wp::int32> adj_dirty_count,
    wp::array_t<wp::int32> adj_dirty_cube_ids)
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
        const wp::int32 var_10 = 0;
        wp::int32* var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32* var_15;
        wp::int32 var_16;
        wp::int32 var_17;
        const wp::int32 var_18 = 2;
        wp::int32* var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        const wp::int32 var_22 = 0;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        wp::int32 var_29;
        bool var_30;
        const wp::int32 var_31 = 0;
        bool var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        wp::int32 var_35;
        bool var_36;
        const wp::int32 var_37 = 0;
        bool var_38;
        bool var_39;
        wp::int32 var_40;
        wp::int32 var_41;
        bool var_42;
        const wp::int32 var_43 = 0;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        wp::shape_t* var_46;
        const wp::int32 var_47 = 0;
        wp::int32 var_48;
        wp::shape_t var_49;
        bool var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        bool var_53;
        const wp::int32 var_54 = 0;
        bool var_55;
        bool var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        bool var_59;
        const wp::int32 var_60 = 0;
        const wp::int32 var_61 = 1;
        wp::int32 var_62;
        wp::shape_t* var_63;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        wp::shape_t var_66;
        bool var_67;
        wp::int32 var_68;
        wp::int32 var_69;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 1;
        wp::int32 var_73;
        bool var_74;
        const wp::int32 var_75 = 0;
        bool var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        wp::int32 var_79;
        bool var_80;
        const wp::int32 var_81 = 0;
        bool var_82;
        bool var_83;
        wp::int32 var_84;
        wp::int32 var_85;
        bool var_86;
        const wp::int32 var_87 = 0;
        const wp::int32 var_88 = 1;
        wp::int32 var_89;
        wp::shape_t* var_90;
        const wp::int32 var_91 = 0;
        wp::int32 var_92;
        wp::shape_t var_93;
        bool var_94;
        wp::int32 var_95;
        wp::int32 var_96;
        wp::int32 var_97;
        wp::int32 var_98;
        const wp::int32 var_99 = 1;
        wp::int32 var_100;
        bool var_101;
        const wp::int32 var_102 = 0;
        bool var_103;
        bool var_104;
        wp::int32 var_105;
        wp::int32 var_106;
        bool var_107;
        const wp::int32 var_108 = 0;
        const wp::int32 var_109 = 1;
        wp::int32 var_110;
        wp::shape_t* var_111;
        const wp::int32 var_112 = 0;
        wp::int32 var_113;
        wp::shape_t var_114;
        bool var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        wp::int32 var_122;
        wp::int32 var_123;
        wp::int32 var_124;
        const wp::int32 var_125 = 1;
        wp::int32 var_126;
        bool var_127;
        const wp::int32 var_128 = 0;
        bool var_129;
        bool var_130;
        const wp::int32 var_131 = 0;
        wp::int32 var_132;
        bool var_133;
        const wp::int32 var_134 = 0;
        bool var_135;
        bool var_136;
        const wp::int32 var_137 = 0;
        wp::int32 var_138;
        bool var_139;
        const wp::int32 var_140 = 0;
        bool var_141;
        bool var_142;
        wp::int32 var_143;
        wp::int32 var_144;
        bool var_145;
        const wp::int32 var_146 = 0;
        const wp::int32 var_147 = 1;
        wp::int32 var_148;
        wp::shape_t* var_149;
        const wp::int32 var_150 = 0;
        wp::int32 var_151;
        wp::shape_t var_152;
        bool var_153;
        wp::int32 var_154;
        wp::int32 var_155;
        wp::int32 var_156;
        wp::int32 var_157;
        const wp::int32 var_158 = 1;
        wp::int32 var_159;
        bool var_160;
        const wp::int32 var_161 = 0;
        bool var_162;
        bool var_163;
        wp::int32 var_164;
        wp::int32 var_165;
        bool var_166;
        const wp::int32 var_167 = 0;
        const wp::int32 var_168 = 1;
        wp::int32 var_169;
        wp::shape_t* var_170;
        const wp::int32 var_171 = 0;
        wp::int32 var_172;
        wp::shape_t var_173;
        bool var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        wp::int32 var_181;
        wp::int32 var_182;
        wp::int32 var_183;
        const wp::int32 var_184 = 1;
        wp::int32 var_185;
        bool var_186;
        const wp::int32 var_187 = 0;
        bool var_188;
        bool var_189;
        const wp::int32 var_190 = 0;
        wp::int32 var_191;
        bool var_192;
        const wp::int32 var_193 = 0;
        bool var_194;
        bool var_195;
        wp::int32 var_196;
        wp::int32 var_197;
        bool var_198;
        const wp::int32 var_199 = 0;
        const wp::int32 var_200 = 1;
        wp::int32 var_201;
        wp::shape_t* var_202;
        const wp::int32 var_203 = 0;
        wp::int32 var_204;
        wp::shape_t var_205;
        bool var_206;
        wp::int32 var_207;
        wp::int32 var_208;
        wp::int32 var_209;
        wp::int32 var_210;
        const wp::int32 var_211 = 1;
        wp::int32 var_212;
        bool var_213;
        const wp::int32 var_214 = 0;
        bool var_215;
        bool var_216;
        wp::int32 var_217;
        wp::int32 var_218;
        bool var_219;
        const wp::int32 var_220 = 0;
        const wp::int32 var_221 = 1;
        wp::int32 var_222;
        wp::shape_t* var_223;
        const wp::int32 var_224 = 0;
        wp::int32 var_225;
        wp::shape_t var_226;
        bool var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::int32 var_233;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        wp::int32 var_241;
        wp::int32 var_242;
        wp::int32 var_243;
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
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        wp::int32 adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        bool adj_26 = {};
        bool adj_27 = {};
        wp::int32 adj_28 = {};
        wp::int32 adj_29 = {};
        bool adj_30 = {};
        wp::int32 adj_31 = {};
        bool adj_32 = {};
        bool adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        bool adj_36 = {};
        wp::int32 adj_37 = {};
        bool adj_38 = {};
        bool adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        bool adj_42 = {};
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::shape_t adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::shape_t adj_49 = {};
        bool adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        bool adj_53 = {};
        wp::int32 adj_54 = {};
        bool adj_55 = {};
        bool adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        bool adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::shape_t adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::shape_t adj_66 = {};
        bool adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        bool adj_74 = {};
        wp::int32 adj_75 = {};
        bool adj_76 = {};
        bool adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        bool adj_80 = {};
        wp::int32 adj_81 = {};
        bool adj_82 = {};
        bool adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        bool adj_86 = {};
        wp::int32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::shape_t adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::shape_t adj_93 = {};
        bool adj_94 = {};
        wp::int32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::int32 adj_99 = {};
        wp::int32 adj_100 = {};
        bool adj_101 = {};
        wp::int32 adj_102 = {};
        bool adj_103 = {};
        bool adj_104 = {};
        wp::int32 adj_105 = {};
        wp::int32 adj_106 = {};
        bool adj_107 = {};
        wp::int32 adj_108 = {};
        wp::int32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::shape_t adj_111 = {};
        wp::int32 adj_112 = {};
        wp::int32 adj_113 = {};
        wp::shape_t adj_114 = {};
        bool adj_115 = {};
        wp::int32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::int32 adj_120 = {};
        wp::int32 adj_121 = {};
        wp::int32 adj_122 = {};
        wp::int32 adj_123 = {};
        wp::int32 adj_124 = {};
        wp::int32 adj_125 = {};
        wp::int32 adj_126 = {};
        bool adj_127 = {};
        wp::int32 adj_128 = {};
        bool adj_129 = {};
        bool adj_130 = {};
        wp::int32 adj_131 = {};
        wp::int32 adj_132 = {};
        bool adj_133 = {};
        wp::int32 adj_134 = {};
        bool adj_135 = {};
        bool adj_136 = {};
        wp::int32 adj_137 = {};
        wp::int32 adj_138 = {};
        bool adj_139 = {};
        wp::int32 adj_140 = {};
        bool adj_141 = {};
        bool adj_142 = {};
        wp::int32 adj_143 = {};
        wp::int32 adj_144 = {};
        bool adj_145 = {};
        wp::int32 adj_146 = {};
        wp::int32 adj_147 = {};
        wp::int32 adj_148 = {};
        wp::shape_t adj_149 = {};
        wp::int32 adj_150 = {};
        wp::int32 adj_151 = {};
        wp::shape_t adj_152 = {};
        bool adj_153 = {};
        wp::int32 adj_154 = {};
        wp::int32 adj_155 = {};
        wp::int32 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::int32 adj_158 = {};
        wp::int32 adj_159 = {};
        bool adj_160 = {};
        wp::int32 adj_161 = {};
        bool adj_162 = {};
        bool adj_163 = {};
        wp::int32 adj_164 = {};
        wp::int32 adj_165 = {};
        bool adj_166 = {};
        wp::int32 adj_167 = {};
        wp::int32 adj_168 = {};
        wp::int32 adj_169 = {};
        wp::shape_t adj_170 = {};
        wp::int32 adj_171 = {};
        wp::int32 adj_172 = {};
        wp::shape_t adj_173 = {};
        bool adj_174 = {};
        wp::int32 adj_175 = {};
        wp::int32 adj_176 = {};
        wp::int32 adj_177 = {};
        wp::int32 adj_178 = {};
        wp::int32 adj_179 = {};
        wp::int32 adj_180 = {};
        wp::int32 adj_181 = {};
        wp::int32 adj_182 = {};
        wp::int32 adj_183 = {};
        wp::int32 adj_184 = {};
        wp::int32 adj_185 = {};
        bool adj_186 = {};
        wp::int32 adj_187 = {};
        bool adj_188 = {};
        bool adj_189 = {};
        wp::int32 adj_190 = {};
        wp::int32 adj_191 = {};
        bool adj_192 = {};
        wp::int32 adj_193 = {};
        bool adj_194 = {};
        bool adj_195 = {};
        wp::int32 adj_196 = {};
        wp::int32 adj_197 = {};
        bool adj_198 = {};
        wp::int32 adj_199 = {};
        wp::int32 adj_200 = {};
        wp::int32 adj_201 = {};
        wp::shape_t adj_202 = {};
        wp::int32 adj_203 = {};
        wp::int32 adj_204 = {};
        wp::shape_t adj_205 = {};
        bool adj_206 = {};
        wp::int32 adj_207 = {};
        wp::int32 adj_208 = {};
        wp::int32 adj_209 = {};
        wp::int32 adj_210 = {};
        wp::int32 adj_211 = {};
        wp::int32 adj_212 = {};
        bool adj_213 = {};
        wp::int32 adj_214 = {};
        bool adj_215 = {};
        bool adj_216 = {};
        wp::int32 adj_217 = {};
        wp::int32 adj_218 = {};
        bool adj_219 = {};
        wp::int32 adj_220 = {};
        wp::int32 adj_221 = {};
        wp::int32 adj_222 = {};
        wp::shape_t adj_223 = {};
        wp::int32 adj_224 = {};
        wp::int32 adj_225 = {};
        wp::shape_t adj_226 = {};
        bool adj_227 = {};
        wp::int32 adj_228 = {};
        wp::int32 adj_229 = {};
        wp::int32 adj_230 = {};
        wp::int32 adj_231 = {};
        wp::int32 adj_232 = {};
        wp::int32 adj_233 = {};
        wp::int32 adj_234 = {};
        wp::int32 adj_235 = {};
        wp::int32 adj_236 = {};
        wp::int32 adj_237 = {};
        wp::int32 adj_238 = {};
        wp::int32 adj_239 = {};
        wp::int32 adj_240 = {};
        wp::int32 adj_241 = {};
        wp::int32 adj_242 = {};
        wp::int32 adj_243 = {};
        //---------
        // forward
        // def mark_dirty_cubes_from_particles_device_count_kernel(                               <L 517>
        // i = wp.tid()                                                                           <L 530>
        var_0 = builtin_tid1d();
        // if i >= particle_count[0]:                                                             <L 531>
        var_2 = wp::address(var_particle_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 532>
            goto label0;
        }
        // p = particle_ids[i]                                                                    <L 533>
        var_5 = wp::address(var_particle_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // if p < 0:                                                                              <L 534>
        var_9 = (var_6 < var_8);
        if (var_9) {
            // return                                                                             <L 535>
            goto label1;
        }
        // gx = particle_grid_xyz[p, 0]                                                           <L 536>
        var_11 = wp::address(var_particle_grid_xyz, var_6, var_10);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // gy = particle_grid_xyz[p, 1]                                                           <L 537>
        var_15 = wp::address(var_particle_grid_xyz, var_6, var_14);
        var_17 = wp::load(var_15);
        var_16 = wp::copy(var_17);
        // gz = particle_grid_xyz[p, 2]                                                           <L 538>
        var_19 = wp::address(var_particle_grid_xyz, var_6, var_18);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // for ox in range(2):                                                                    <L 539>
        // cx = gx - ox                                                                           <L 540>
        var_23 = wp::sub(var_12, var_22);
        // if cx >= 0 and cx < nx_cells:                                                          <L 541>
        var_26 = (var_23 >= var_25);
        var_24 = var_26;
        if (var_24) {
            var_27 = (var_23 < var_nx_cells);
            var_24 = var_24 && var_27;
        }
        if (var_24) {
            // for oy in range(2):                                                                <L 542>
            // cy = gy - oy                                                                       <L 543>
            var_29 = wp::sub(var_16, var_28);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_32 = (var_29 >= var_31);
            var_30 = var_32;
            if (var_30) {
                var_33 = (var_29 < var_ny_cells);
                var_30 = var_30 && var_33;
            }
            if (var_30) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_35 = wp::sub(var_20, var_34);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_38 = (var_35 >= var_37);
                var_36 = var_38;
                if (var_36) {
                    var_39 = (var_35 < var_nz_cells);
                    var_36 = var_36 && var_39;
                }
                if (var_36) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_40 = _cube_flat_id_0(var_23, var_29, var_35, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_41 = wp::atomic_exch(var_dirty_marks, var_40, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_42 = (var_41 != var_stamp);
                    if (var_42) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_45 = wp::atomic_add(var_dirty_count, var_43, var_44);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_46 = &(var_dirty_cube_ids.shape);
                        var_49 = wp::load(var_46);
                        var_48 = wp::extract(var_49, var_47);
                        var_50 = (var_45 < var_48);
                        if (var_50) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_45, var_40);
                        }
                    }
                }
                // cz = gz - oz                                                                   <L 546>
                var_52 = wp::sub(var_20, var_51);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_55 = (var_52 >= var_54);
                var_53 = var_55;
                if (var_53) {
                    var_56 = (var_52 < var_nz_cells);
                    var_53 = var_53 && var_56;
                }
                if (var_53) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_57 = _cube_flat_id_0(var_23, var_29, var_52, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_58 = wp::atomic_exch(var_dirty_marks, var_57, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_59 = (var_58 != var_stamp);
                    if (var_59) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_62 = wp::atomic_add(var_dirty_count, var_60, var_61);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_63 = &(var_dirty_cube_ids.shape);
                        var_66 = wp::load(var_63);
                        var_65 = wp::extract(var_66, var_64);
                        var_67 = (var_62 < var_65);
                        if (var_67) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_62, var_57);
                        }
                    }
                    var_68 = wp::where(var_59, var_62, var_45);
                }
                var_69 = wp::where(var_53, var_57, var_40);
                var_70 = wp::where(var_53, var_58, var_41);
                var_71 = wp::where(var_53, var_68, var_45);
            }
            // cy = gy - oy                                                                       <L 543>
            var_73 = wp::sub(var_16, var_72);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_76 = (var_73 >= var_75);
            var_74 = var_76;
            if (var_74) {
                var_77 = (var_73 < var_ny_cells);
                var_74 = var_74 && var_77;
            }
            if (var_74) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_79 = wp::sub(var_20, var_78);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_82 = (var_79 >= var_81);
                var_80 = var_82;
                if (var_80) {
                    var_83 = (var_79 < var_nz_cells);
                    var_80 = var_80 && var_83;
                }
                if (var_80) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_84 = _cube_flat_id_0(var_23, var_73, var_79, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_85 = wp::atomic_exch(var_dirty_marks, var_84, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_86 = (var_85 != var_stamp);
                    if (var_86) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_89 = wp::atomic_add(var_dirty_count, var_87, var_88);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_90 = &(var_dirty_cube_ids.shape);
                        var_93 = wp::load(var_90);
                        var_92 = wp::extract(var_93, var_91);
                        var_94 = (var_89 < var_92);
                        if (var_94) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_89, var_84);
                        }
                    }
                    var_95 = wp::where(var_86, var_89, var_71);
                }
                var_96 = wp::where(var_80, var_84, var_69);
                var_97 = wp::where(var_80, var_85, var_70);
                var_98 = wp::where(var_80, var_95, var_71);
                // cz = gz - oz                                                                   <L 546>
                var_100 = wp::sub(var_20, var_99);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_103 = (var_100 >= var_102);
                var_101 = var_103;
                if (var_101) {
                    var_104 = (var_100 < var_nz_cells);
                    var_101 = var_101 && var_104;
                }
                if (var_101) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_105 = _cube_flat_id_0(var_23, var_73, var_100, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_106 = wp::atomic_exch(var_dirty_marks, var_105, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_107 = (var_106 != var_stamp);
                    if (var_107) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_110 = wp::atomic_add(var_dirty_count, var_108, var_109);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_111 = &(var_dirty_cube_ids.shape);
                        var_114 = wp::load(var_111);
                        var_113 = wp::extract(var_114, var_112);
                        var_115 = (var_110 < var_113);
                        if (var_115) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_110, var_105);
                        }
                    }
                    var_116 = wp::where(var_107, var_110, var_98);
                }
                var_117 = wp::where(var_101, var_105, var_96);
                var_118 = wp::where(var_101, var_106, var_97);
                var_119 = wp::where(var_101, var_116, var_98);
            }
            var_120 = wp::where(var_74, var_99, var_51);
            var_121 = wp::where(var_74, var_100, var_52);
            var_122 = wp::where(var_74, var_117, var_69);
            var_123 = wp::where(var_74, var_118, var_70);
            var_124 = wp::where(var_74, var_119, var_71);
        }
        // cx = gx - ox                                                                           <L 540>
        var_126 = wp::sub(var_12, var_125);
        // if cx >= 0 and cx < nx_cells:                                                          <L 541>
        var_129 = (var_126 >= var_128);
        var_127 = var_129;
        if (var_127) {
            var_130 = (var_126 < var_nx_cells);
            var_127 = var_127 && var_130;
        }
        if (var_127) {
            // for oy in range(2):                                                                <L 542>
            // cy = gy - oy                                                                       <L 543>
            var_132 = wp::sub(var_16, var_131);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_135 = (var_132 >= var_134);
            var_133 = var_135;
            if (var_133) {
                var_136 = (var_132 < var_ny_cells);
                var_133 = var_133 && var_136;
            }
            if (var_133) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_138 = wp::sub(var_20, var_137);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_141 = (var_138 >= var_140);
                var_139 = var_141;
                if (var_139) {
                    var_142 = (var_138 < var_nz_cells);
                    var_139 = var_139 && var_142;
                }
                if (var_139) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_143 = _cube_flat_id_0(var_126, var_132, var_138, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_144 = wp::atomic_exch(var_dirty_marks, var_143, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_145 = (var_144 != var_stamp);
                    if (var_145) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_148 = wp::atomic_add(var_dirty_count, var_146, var_147);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_149 = &(var_dirty_cube_ids.shape);
                        var_152 = wp::load(var_149);
                        var_151 = wp::extract(var_152, var_150);
                        var_153 = (var_148 < var_151);
                        if (var_153) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_148, var_143);
                        }
                    }
                    var_154 = wp::where(var_145, var_148, var_124);
                }
                var_155 = wp::where(var_139, var_143, var_122);
                var_156 = wp::where(var_139, var_144, var_123);
                var_157 = wp::where(var_139, var_154, var_124);
                // cz = gz - oz                                                                   <L 546>
                var_159 = wp::sub(var_20, var_158);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_162 = (var_159 >= var_161);
                var_160 = var_162;
                if (var_160) {
                    var_163 = (var_159 < var_nz_cells);
                    var_160 = var_160 && var_163;
                }
                if (var_160) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_164 = _cube_flat_id_0(var_126, var_132, var_159, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_165 = wp::atomic_exch(var_dirty_marks, var_164, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_166 = (var_165 != var_stamp);
                    if (var_166) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_169 = wp::atomic_add(var_dirty_count, var_167, var_168);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_170 = &(var_dirty_cube_ids.shape);
                        var_173 = wp::load(var_170);
                        var_172 = wp::extract(var_173, var_171);
                        var_174 = (var_169 < var_172);
                        if (var_174) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_169, var_164);
                        }
                    }
                    var_175 = wp::where(var_166, var_169, var_157);
                }
                var_176 = wp::where(var_160, var_164, var_155);
                var_177 = wp::where(var_160, var_165, var_156);
                var_178 = wp::where(var_160, var_175, var_157);
            }
            var_179 = wp::where(var_133, var_158, var_120);
            var_180 = wp::where(var_133, var_159, var_121);
            var_181 = wp::where(var_133, var_176, var_122);
            var_182 = wp::where(var_133, var_177, var_123);
            var_183 = wp::where(var_133, var_178, var_124);
            // cy = gy - oy                                                                       <L 543>
            var_185 = wp::sub(var_16, var_184);
            // if cy >= 0 and cy < ny_cells:                                                      <L 544>
            var_188 = (var_185 >= var_187);
            var_186 = var_188;
            if (var_186) {
                var_189 = (var_185 < var_ny_cells);
                var_186 = var_186 && var_189;
            }
            if (var_186) {
                // for oz in range(2):                                                            <L 545>
                // cz = gz - oz                                                                   <L 546>
                var_191 = wp::sub(var_20, var_190);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_194 = (var_191 >= var_193);
                var_192 = var_194;
                if (var_192) {
                    var_195 = (var_191 < var_nz_cells);
                    var_192 = var_192 && var_195;
                }
                if (var_192) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_196 = _cube_flat_id_0(var_126, var_185, var_191, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_197 = wp::atomic_exch(var_dirty_marks, var_196, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_198 = (var_197 != var_stamp);
                    if (var_198) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_201 = wp::atomic_add(var_dirty_count, var_199, var_200);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_202 = &(var_dirty_cube_ids.shape);
                        var_205 = wp::load(var_202);
                        var_204 = wp::extract(var_205, var_203);
                        var_206 = (var_201 < var_204);
                        if (var_206) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_201, var_196);
                        }
                    }
                    var_207 = wp::where(var_198, var_201, var_183);
                }
                var_208 = wp::where(var_192, var_196, var_181);
                var_209 = wp::where(var_192, var_197, var_182);
                var_210 = wp::where(var_192, var_207, var_183);
                // cz = gz - oz                                                                   <L 546>
                var_212 = wp::sub(var_20, var_211);
                // if cz >= 0 and cz < nz_cells:                                                  <L 547>
                var_215 = (var_212 >= var_214);
                var_213 = var_215;
                if (var_213) {
                    var_216 = (var_212 < var_nz_cells);
                    var_213 = var_213 && var_216;
                }
                if (var_213) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 548>
                    var_217 = _cube_flat_id_0(var_126, var_185, var_212, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 549>
                    // var_218 = wp::atomic_exch(var_dirty_marks, var_217, var_stamp);
                    // if old != stamp:                                                           <L 550>
                    var_219 = (var_218 != var_stamp);
                    if (var_219) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 551>
                        // var_222 = wp::atomic_add(var_dirty_count, var_220, var_221);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 552>
                        var_223 = &(var_dirty_cube_ids.shape);
                        var_226 = wp::load(var_223);
                        var_225 = wp::extract(var_226, var_224);
                        var_227 = (var_222 < var_225);
                        if (var_227) {
                            // dirty_cube_ids[dst] = flat                                         <L 553>
                            // wp::array_store(var_dirty_cube_ids, var_222, var_217);
                        }
                    }
                    var_228 = wp::where(var_219, var_222, var_210);
                }
                var_229 = wp::where(var_213, var_217, var_208);
                var_230 = wp::where(var_213, var_218, var_209);
                var_231 = wp::where(var_213, var_228, var_210);
            }
            var_232 = wp::where(var_186, var_211, var_179);
            var_233 = wp::where(var_186, var_212, var_180);
            var_234 = wp::where(var_186, var_229, var_181);
            var_235 = wp::where(var_186, var_230, var_182);
            var_236 = wp::where(var_186, var_231, var_183);
        }
        var_237 = wp::where(var_127, var_184, var_72);
        var_238 = wp::where(var_127, var_185, var_73);
        var_239 = wp::where(var_127, var_232, var_120);
        var_240 = wp::where(var_127, var_233, var_121);
        var_241 = wp::where(var_127, var_234, var_122);
        var_242 = wp::where(var_127, var_235, var_123);
        var_243 = wp::where(var_127, var_236, var_124);
        //---------
        // reverse
        wp::adj_where(var_127, var_236, var_124, adj_127, adj_236, adj_124, adj_243);
        wp::adj_where(var_127, var_235, var_123, adj_127, adj_235, adj_123, adj_242);
        wp::adj_where(var_127, var_234, var_122, adj_127, adj_234, adj_122, adj_241);
        wp::adj_where(var_127, var_233, var_121, adj_127, adj_233, adj_121, adj_240);
        wp::adj_where(var_127, var_232, var_120, adj_127, adj_232, adj_120, adj_239);
        wp::adj_where(var_127, var_185, var_73, adj_127, adj_185, adj_73, adj_238);
        wp::adj_where(var_127, var_184, var_72, adj_127, adj_184, adj_72, adj_237);
        if (var_127) {
            wp::adj_where(var_186, var_231, var_183, adj_186, adj_231, adj_183, adj_236);
            wp::adj_where(var_186, var_230, var_182, adj_186, adj_230, adj_182, adj_235);
            wp::adj_where(var_186, var_229, var_181, adj_186, adj_229, adj_181, adj_234);
            wp::adj_where(var_186, var_212, var_180, adj_186, adj_212, adj_180, adj_233);
            wp::adj_where(var_186, var_211, var_179, adj_186, adj_211, adj_179, adj_232);
            if (var_186) {
                wp::adj_where(var_213, var_228, var_210, adj_213, adj_228, adj_210, adj_231);
                wp::adj_where(var_213, var_218, var_209, adj_213, adj_218, adj_209, adj_230);
                wp::adj_where(var_213, var_217, var_208, adj_213, adj_217, adj_208, adj_229);
                if (var_213) {
                    wp::adj_where(var_219, var_222, var_210, adj_219, adj_222, adj_210, adj_228);
                    if (var_219) {
                        if (var_227) {
                            wp::adj_array_store(var_dirty_cube_ids, var_222, var_217, adj_dirty_cube_ids, adj_222, adj_217);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_226, var_224, adj_223, adj_224, adj_225);
                        adj_dirty_cube_ids.shape = adj_223;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_220, var_221, adj_dirty_count, adj_220, adj_221, adj_222);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_126, var_185, var_212, var_ny_cells, var_nz_cells, adj_126, adj_185, adj_212, adj_ny_cells, adj_nz_cells, adj_217);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_213) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_211, adj_20, adj_211, adj_212);
                // adj: cz = gz - oz                                                              <L 546>
                wp::adj_where(var_192, var_207, var_183, adj_192, adj_207, adj_183, adj_210);
                wp::adj_where(var_192, var_197, var_182, adj_192, adj_197, adj_182, adj_209);
                wp::adj_where(var_192, var_196, var_181, adj_192, adj_196, adj_181, adj_208);
                if (var_192) {
                    wp::adj_where(var_198, var_201, var_183, adj_198, adj_201, adj_183, adj_207);
                    if (var_198) {
                        if (var_206) {
                            wp::adj_array_store(var_dirty_cube_ids, var_201, var_196, adj_dirty_cube_ids, adj_201, adj_196);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_205, var_203, adj_202, adj_203, adj_204);
                        adj_dirty_cube_ids.shape = adj_202;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_199, var_200, adj_dirty_count, adj_199, adj_200, adj_201);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_126, var_185, var_191, var_ny_cells, var_nz_cells, adj_126, adj_185, adj_191, adj_ny_cells, adj_nz_cells, adj_196);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_192) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_190, adj_20, adj_190, adj_191);
                // adj: cz = gz - oz                                                              <L 546>
                // adj: for oz in range(2):                                                       <L 545>
            }
            if (var_186) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 544>
            wp::adj_sub(var_16, var_184, adj_16, adj_184, adj_185);
            // adj: cy = gy - oy                                                                  <L 543>
            wp::adj_where(var_133, var_178, var_124, adj_133, adj_178, adj_124, adj_183);
            wp::adj_where(var_133, var_177, var_123, adj_133, adj_177, adj_123, adj_182);
            wp::adj_where(var_133, var_176, var_122, adj_133, adj_176, adj_122, adj_181);
            wp::adj_where(var_133, var_159, var_121, adj_133, adj_159, adj_121, adj_180);
            wp::adj_where(var_133, var_158, var_120, adj_133, adj_158, adj_120, adj_179);
            if (var_133) {
                wp::adj_where(var_160, var_175, var_157, adj_160, adj_175, adj_157, adj_178);
                wp::adj_where(var_160, var_165, var_156, adj_160, adj_165, adj_156, adj_177);
                wp::adj_where(var_160, var_164, var_155, adj_160, adj_164, adj_155, adj_176);
                if (var_160) {
                    wp::adj_where(var_166, var_169, var_157, adj_166, adj_169, adj_157, adj_175);
                    if (var_166) {
                        if (var_174) {
                            wp::adj_array_store(var_dirty_cube_ids, var_169, var_164, adj_dirty_cube_ids, adj_169, adj_164);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_173, var_171, adj_170, adj_171, adj_172);
                        adj_dirty_cube_ids.shape = adj_170;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_167, var_168, adj_dirty_count, adj_167, adj_168, adj_169);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_126, var_132, var_159, var_ny_cells, var_nz_cells, adj_126, adj_132, adj_159, adj_ny_cells, adj_nz_cells, adj_164);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_160) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_158, adj_20, adj_158, adj_159);
                // adj: cz = gz - oz                                                              <L 546>
                wp::adj_where(var_139, var_154, var_124, adj_139, adj_154, adj_124, adj_157);
                wp::adj_where(var_139, var_144, var_123, adj_139, adj_144, adj_123, adj_156);
                wp::adj_where(var_139, var_143, var_122, adj_139, adj_143, adj_122, adj_155);
                if (var_139) {
                    wp::adj_where(var_145, var_148, var_124, adj_145, adj_148, adj_124, adj_154);
                    if (var_145) {
                        if (var_153) {
                            wp::adj_array_store(var_dirty_cube_ids, var_148, var_143, adj_dirty_cube_ids, adj_148, adj_143);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_152, var_150, adj_149, adj_150, adj_151);
                        adj_dirty_cube_ids.shape = adj_149;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_146, var_147, adj_dirty_count, adj_146, adj_147, adj_148);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_126, var_132, var_138, var_ny_cells, var_nz_cells, adj_126, adj_132, adj_138, adj_ny_cells, adj_nz_cells, adj_143);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_139) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_137, adj_20, adj_137, adj_138);
                // adj: cz = gz - oz                                                              <L 546>
                // adj: for oz in range(2):                                                       <L 545>
            }
            if (var_133) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 544>
            wp::adj_sub(var_16, var_131, adj_16, adj_131, adj_132);
            // adj: cy = gy - oy                                                                  <L 543>
            // adj: for oy in range(2):                                                           <L 542>
        }
        if (var_127) {
        }
        // adj: if cx >= 0 and cx < nx_cells:                                                     <L 541>
        wp::adj_sub(var_12, var_125, adj_12, adj_125, adj_126);
        // adj: cx = gx - ox                                                                      <L 540>
        if (var_24) {
            wp::adj_where(var_74, var_119, var_71, adj_74, adj_119, adj_71, adj_124);
            wp::adj_where(var_74, var_118, var_70, adj_74, adj_118, adj_70, adj_123);
            wp::adj_where(var_74, var_117, var_69, adj_74, adj_117, adj_69, adj_122);
            wp::adj_where(var_74, var_100, var_52, adj_74, adj_100, adj_52, adj_121);
            wp::adj_where(var_74, var_99, var_51, adj_74, adj_99, adj_51, adj_120);
            if (var_74) {
                wp::adj_where(var_101, var_116, var_98, adj_101, adj_116, adj_98, adj_119);
                wp::adj_where(var_101, var_106, var_97, adj_101, adj_106, adj_97, adj_118);
                wp::adj_where(var_101, var_105, var_96, adj_101, adj_105, adj_96, adj_117);
                if (var_101) {
                    wp::adj_where(var_107, var_110, var_98, adj_107, adj_110, adj_98, adj_116);
                    if (var_107) {
                        if (var_115) {
                            wp::adj_array_store(var_dirty_cube_ids, var_110, var_105, adj_dirty_cube_ids, adj_110, adj_105);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_114, var_112, adj_111, adj_112, adj_113);
                        adj_dirty_cube_ids.shape = adj_111;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_108, var_109, adj_dirty_count, adj_108, adj_109, adj_110);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_23, var_73, var_100, var_ny_cells, var_nz_cells, adj_23, adj_73, adj_100, adj_ny_cells, adj_nz_cells, adj_105);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_101) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_99, adj_20, adj_99, adj_100);
                // adj: cz = gz - oz                                                              <L 546>
                wp::adj_where(var_80, var_95, var_71, adj_80, adj_95, adj_71, adj_98);
                wp::adj_where(var_80, var_85, var_70, adj_80, adj_85, adj_70, adj_97);
                wp::adj_where(var_80, var_84, var_69, adj_80, adj_84, adj_69, adj_96);
                if (var_80) {
                    wp::adj_where(var_86, var_89, var_71, adj_86, adj_89, adj_71, adj_95);
                    if (var_86) {
                        if (var_94) {
                            wp::adj_array_store(var_dirty_cube_ids, var_89, var_84, adj_dirty_cube_ids, adj_89, adj_84);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_93, var_91, adj_90, adj_91, adj_92);
                        adj_dirty_cube_ids.shape = adj_90;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_87, var_88, adj_dirty_count, adj_87, adj_88, adj_89);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_23, var_73, var_79, var_ny_cells, var_nz_cells, adj_23, adj_73, adj_79, adj_ny_cells, adj_nz_cells, adj_84);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_80) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_78, adj_20, adj_78, adj_79);
                // adj: cz = gz - oz                                                              <L 546>
                // adj: for oz in range(2):                                                       <L 545>
            }
            if (var_74) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 544>
            wp::adj_sub(var_16, var_72, adj_16, adj_72, adj_73);
            // adj: cy = gy - oy                                                                  <L 543>
            if (var_30) {
                wp::adj_where(var_53, var_68, var_45, adj_53, adj_68, adj_45, adj_71);
                wp::adj_where(var_53, var_58, var_41, adj_53, adj_58, adj_41, adj_70);
                wp::adj_where(var_53, var_57, var_40, adj_53, adj_57, adj_40, adj_69);
                if (var_53) {
                    wp::adj_where(var_59, var_62, var_45, adj_59, adj_62, adj_45, adj_68);
                    if (var_59) {
                        if (var_67) {
                            wp::adj_array_store(var_dirty_cube_ids, var_62, var_57, adj_dirty_cube_ids, adj_62, adj_57);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_66, var_64, adj_63, adj_64, adj_65);
                        adj_dirty_cube_ids.shape = adj_63;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_60, var_61, adj_dirty_count, adj_60, adj_61, adj_62);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_23, var_29, var_52, var_ny_cells, var_nz_cells, adj_23, adj_29, adj_52, adj_ny_cells, adj_nz_cells, adj_57);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_53) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_51, adj_20, adj_51, adj_52);
                // adj: cz = gz - oz                                                              <L 546>
                if (var_36) {
                    if (var_42) {
                        if (var_50) {
                            wp::adj_array_store(var_dirty_cube_ids, var_45, var_40, adj_dirty_cube_ids, adj_45, adj_40);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 553>
                        }
                        wp::adj_extract(var_49, var_47, adj_46, adj_47, adj_48);
                        adj_dirty_cube_ids.shape = adj_46;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 552>
                        wp::adj_atomic_add(var_dirty_count, var_43, var_44, adj_dirty_count, adj_43, adj_44, adj_45);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 551>
                    }
                    // adj: if old != stamp:                                                      <L 550>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 549>
                    adj__cube_flat_id_0(var_23, var_29, var_35, var_ny_cells, var_nz_cells, adj_23, adj_29, adj_35, adj_ny_cells, adj_nz_cells, adj_40);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 548>
                }
                if (var_36) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 547>
                wp::adj_sub(var_20, var_34, adj_20, adj_34, adj_35);
                // adj: cz = gz - oz                                                              <L 546>
                // adj: for oz in range(2):                                                       <L 545>
            }
            if (var_30) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 544>
            wp::adj_sub(var_16, var_28, adj_16, adj_28, adj_29);
            // adj: cy = gy - oy                                                                  <L 543>
            // adj: for oy in range(2):                                                           <L 542>
        }
        if (var_24) {
        }
        // adj: if cx >= 0 and cx < nx_cells:                                                     <L 541>
        wp::adj_sub(var_12, var_22, adj_12, adj_22, adj_23);
        // adj: cx = gx - ox                                                                      <L 540>
        // adj: for ox in range(2):                                                               <L 539>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_address(var_particle_grid_xyz, var_6, var_18, adj_particle_grid_xyz, adj_6, adj_18, adj_19);
        // adj: gz = particle_grid_xyz[p, 2]                                                      <L 538>
        wp::adj_copy(var_17, adj_15, adj_16);
        wp::adj_address(var_particle_grid_xyz, var_6, var_14, adj_particle_grid_xyz, adj_6, adj_14, adj_15);
        // adj: gy = particle_grid_xyz[p, 1]                                                      <L 537>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_particle_grid_xyz, var_6, var_10, adj_particle_grid_xyz, adj_6, adj_10, adj_11);
        // adj: gx = particle_grid_xyz[p, 0]                                                      <L 536>
        if (var_9) {
            label1:;
            // adj: return                                                                        <L 535>
        }
        // adj: if p < 0:                                                                         <L 534>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_particle_ids, var_0, adj_particle_ids, adj_0, adj_5);
        // adj: p = particle_ids[i]                                                               <L 533>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 532>
        }
        wp::adj_address(var_particle_count, var_1, adj_particle_count, adj_1, adj_2);
        // adj: if i >= particle_count[0]:                                                        <L 531>
        // adj: i = wp.tid()                                                                      <L 530>
        // adj: def mark_dirty_cubes_from_particles_device_count_kernel(                          <L 517>
        continue;
    }
}



extern "C" __global__ void mark_dirty_cubes_from_particles_kernel_462257c1_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_ids,
    wp::int32 var_particle_count,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::int32 var_nx_cells,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::int32 var_stamp,
    wp::array_t<wp::int32> var_dirty_marks,
    wp::array_t<wp::int32> var_dirty_count,
    wp::array_t<wp::int32> var_dirty_cube_ids)
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
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 0;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        bool var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        bool var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        bool var_39;
        const wp::int32 var_40 = 0;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        bool var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        bool var_50;
        const wp::int32 var_51 = 0;
        bool var_52;
        bool var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        bool var_56;
        const wp::int32 var_57 = 0;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::shape_t* var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::shape_t var_63;
        bool var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        bool var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        bool var_74;
        const wp::int32 var_75 = 0;
        wp::int32 var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        bool var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        bool var_83;
        const wp::int32 var_84 = 0;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        wp::shape_t* var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        wp::shape_t var_90;
        bool var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 1;
        wp::int32 var_97;
        bool var_98;
        const wp::int32 var_99 = 0;
        bool var_100;
        bool var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        bool var_104;
        const wp::int32 var_105 = 0;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        wp::shape_t* var_108;
        const wp::int32 var_109 = 0;
        wp::int32 var_110;
        wp::shape_t var_111;
        bool var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 1;
        wp::int32 var_123;
        bool var_124;
        const wp::int32 var_125 = 0;
        bool var_126;
        bool var_127;
        const wp::int32 var_128 = 0;
        wp::int32 var_129;
        bool var_130;
        const wp::int32 var_131 = 0;
        bool var_132;
        bool var_133;
        const wp::int32 var_134 = 0;
        wp::int32 var_135;
        bool var_136;
        const wp::int32 var_137 = 0;
        bool var_138;
        bool var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        bool var_142;
        const wp::int32 var_143 = 0;
        const wp::int32 var_144 = 1;
        wp::int32 var_145;
        wp::shape_t* var_146;
        const wp::int32 var_147 = 0;
        wp::int32 var_148;
        wp::shape_t var_149;
        bool var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32 var_154;
        const wp::int32 var_155 = 1;
        wp::int32 var_156;
        bool var_157;
        const wp::int32 var_158 = 0;
        bool var_159;
        bool var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        bool var_163;
        const wp::int32 var_164 = 0;
        const wp::int32 var_165 = 1;
        wp::int32 var_166;
        wp::shape_t* var_167;
        const wp::int32 var_168 = 0;
        wp::int32 var_169;
        wp::shape_t var_170;
        bool var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        bool var_183;
        const wp::int32 var_184 = 0;
        bool var_185;
        bool var_186;
        const wp::int32 var_187 = 0;
        wp::int32 var_188;
        bool var_189;
        const wp::int32 var_190 = 0;
        bool var_191;
        bool var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        bool var_195;
        const wp::int32 var_196 = 0;
        const wp::int32 var_197 = 1;
        wp::int32 var_198;
        wp::shape_t* var_199;
        const wp::int32 var_200 = 0;
        wp::int32 var_201;
        wp::shape_t var_202;
        bool var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        const wp::int32 var_208 = 1;
        wp::int32 var_209;
        bool var_210;
        const wp::int32 var_211 = 0;
        bool var_212;
        bool var_213;
        wp::int32 var_214;
        wp::int32 var_215;
        bool var_216;
        const wp::int32 var_217 = 0;
        const wp::int32 var_218 = 1;
        wp::int32 var_219;
        wp::shape_t* var_220;
        const wp::int32 var_221 = 0;
        wp::int32 var_222;
        wp::shape_t var_223;
        bool var_224;
        wp::int32 var_225;
        wp::int32 var_226;
        wp::int32 var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::int32 var_233;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
        //---------
        // forward
        // def mark_dirty_cubes_from_particles_kernel(                                            <L 477>
        // i = wp.tid()                                                                           <L 490>
        var_0 = builtin_tid1d();
        // if i >= particle_count:                                                                <L 491>
        var_1 = (var_0 >= var_particle_count);
        if (var_1) {
            // return                                                                             <L 492>
            continue;
        }
        // p = particle_ids[i]                                                                    <L 493>
        var_2 = wp::address(var_particle_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if p < 0:                                                                              <L 494>
        var_6 = (var_3 < var_5);
        if (var_6) {
            // return                                                                             <L 495>
            continue;
        }
        // gx = particle_grid_xyz[p, 0]                                                           <L 496>
        var_8 = wp::address(var_particle_grid_xyz, var_3, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // gy = particle_grid_xyz[p, 1]                                                           <L 497>
        var_12 = wp::address(var_particle_grid_xyz, var_3, var_11);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // gz = particle_grid_xyz[p, 2]                                                           <L 498>
        var_16 = wp::address(var_particle_grid_xyz, var_3, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for ox in range(2):                                                                    <L 499>
        // cx = gx - ox                                                                           <L 500>
        var_20 = wp::sub(var_9, var_19);
        // if cx >= 0 and cx < nx_cells:                                                          <L 501>
        var_23 = (var_20 >= var_22);
        var_21 = var_23;
        if (var_21) {
            var_24 = (var_20 < var_nx_cells);
            var_21 = var_21 && var_24;
        }
        if (var_21) {
            // for oy in range(2):                                                                <L 502>
            // cy = gy - oy                                                                       <L 503>
            var_26 = wp::sub(var_13, var_25);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_29 = (var_26 >= var_28);
            var_27 = var_29;
            if (var_27) {
                var_30 = (var_26 < var_ny_cells);
                var_27 = var_27 && var_30;
            }
            if (var_27) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_32 = wp::sub(var_17, var_31);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_35 = (var_32 >= var_34);
                var_33 = var_35;
                if (var_33) {
                    var_36 = (var_32 < var_nz_cells);
                    var_33 = var_33 && var_36;
                }
                if (var_33) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_37 = _cube_flat_id_0(var_20, var_26, var_32, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_38 = wp::atomic_exch(var_dirty_marks, var_37, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_39 = (var_38 != var_stamp);
                    if (var_39) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_42 = wp::atomic_add(var_dirty_count, var_40, var_41);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_43 = &(var_dirty_cube_ids.shape);
                        var_46 = wp::load(var_43);
                        var_45 = wp::extract(var_46, var_44);
                        var_47 = (var_42 < var_45);
                        if (var_47) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_42, var_37);
                        }
                    }
                }
                // cz = gz - oz                                                                   <L 506>
                var_49 = wp::sub(var_17, var_48);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_52 = (var_49 >= var_51);
                var_50 = var_52;
                if (var_50) {
                    var_53 = (var_49 < var_nz_cells);
                    var_50 = var_50 && var_53;
                }
                if (var_50) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_54 = _cube_flat_id_0(var_20, var_26, var_49, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_55 = wp::atomic_exch(var_dirty_marks, var_54, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_56 = (var_55 != var_stamp);
                    if (var_56) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_59 = wp::atomic_add(var_dirty_count, var_57, var_58);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_60 = &(var_dirty_cube_ids.shape);
                        var_63 = wp::load(var_60);
                        var_62 = wp::extract(var_63, var_61);
                        var_64 = (var_59 < var_62);
                        if (var_64) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_59, var_54);
                        }
                    }
                    var_65 = wp::where(var_56, var_59, var_42);
                }
                var_66 = wp::where(var_50, var_54, var_37);
                var_67 = wp::where(var_50, var_55, var_38);
                var_68 = wp::where(var_50, var_65, var_42);
            }
            // cy = gy - oy                                                                       <L 503>
            var_70 = wp::sub(var_13, var_69);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_73 = (var_70 >= var_72);
            var_71 = var_73;
            if (var_71) {
                var_74 = (var_70 < var_ny_cells);
                var_71 = var_71 && var_74;
            }
            if (var_71) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_76 = wp::sub(var_17, var_75);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_79 = (var_76 >= var_78);
                var_77 = var_79;
                if (var_77) {
                    var_80 = (var_76 < var_nz_cells);
                    var_77 = var_77 && var_80;
                }
                if (var_77) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_81 = _cube_flat_id_0(var_20, var_70, var_76, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_82 = wp::atomic_exch(var_dirty_marks, var_81, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_83 = (var_82 != var_stamp);
                    if (var_83) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_86 = wp::atomic_add(var_dirty_count, var_84, var_85);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_87 = &(var_dirty_cube_ids.shape);
                        var_90 = wp::load(var_87);
                        var_89 = wp::extract(var_90, var_88);
                        var_91 = (var_86 < var_89);
                        if (var_91) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_86, var_81);
                        }
                    }
                    var_92 = wp::where(var_83, var_86, var_68);
                }
                var_93 = wp::where(var_77, var_81, var_66);
                var_94 = wp::where(var_77, var_82, var_67);
                var_95 = wp::where(var_77, var_92, var_68);
                // cz = gz - oz                                                                   <L 506>
                var_97 = wp::sub(var_17, var_96);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_100 = (var_97 >= var_99);
                var_98 = var_100;
                if (var_98) {
                    var_101 = (var_97 < var_nz_cells);
                    var_98 = var_98 && var_101;
                }
                if (var_98) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_102 = _cube_flat_id_0(var_20, var_70, var_97, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_103 = wp::atomic_exch(var_dirty_marks, var_102, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_104 = (var_103 != var_stamp);
                    if (var_104) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_107 = wp::atomic_add(var_dirty_count, var_105, var_106);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_108 = &(var_dirty_cube_ids.shape);
                        var_111 = wp::load(var_108);
                        var_110 = wp::extract(var_111, var_109);
                        var_112 = (var_107 < var_110);
                        if (var_112) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_107, var_102);
                        }
                    }
                    var_113 = wp::where(var_104, var_107, var_95);
                }
                var_114 = wp::where(var_98, var_102, var_93);
                var_115 = wp::where(var_98, var_103, var_94);
                var_116 = wp::where(var_98, var_113, var_95);
            }
            var_117 = wp::where(var_71, var_96, var_48);
            var_118 = wp::where(var_71, var_97, var_49);
            var_119 = wp::where(var_71, var_114, var_66);
            var_120 = wp::where(var_71, var_115, var_67);
            var_121 = wp::where(var_71, var_116, var_68);
        }
        // cx = gx - ox                                                                           <L 500>
        var_123 = wp::sub(var_9, var_122);
        // if cx >= 0 and cx < nx_cells:                                                          <L 501>
        var_126 = (var_123 >= var_125);
        var_124 = var_126;
        if (var_124) {
            var_127 = (var_123 < var_nx_cells);
            var_124 = var_124 && var_127;
        }
        if (var_124) {
            // for oy in range(2):                                                                <L 502>
            // cy = gy - oy                                                                       <L 503>
            var_129 = wp::sub(var_13, var_128);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_132 = (var_129 >= var_131);
            var_130 = var_132;
            if (var_130) {
                var_133 = (var_129 < var_ny_cells);
                var_130 = var_130 && var_133;
            }
            if (var_130) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_135 = wp::sub(var_17, var_134);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_138 = (var_135 >= var_137);
                var_136 = var_138;
                if (var_136) {
                    var_139 = (var_135 < var_nz_cells);
                    var_136 = var_136 && var_139;
                }
                if (var_136) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_140 = _cube_flat_id_0(var_123, var_129, var_135, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_141 = wp::atomic_exch(var_dirty_marks, var_140, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_142 = (var_141 != var_stamp);
                    if (var_142) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_145 = wp::atomic_add(var_dirty_count, var_143, var_144);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_146 = &(var_dirty_cube_ids.shape);
                        var_149 = wp::load(var_146);
                        var_148 = wp::extract(var_149, var_147);
                        var_150 = (var_145 < var_148);
                        if (var_150) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_145, var_140);
                        }
                    }
                    var_151 = wp::where(var_142, var_145, var_121);
                }
                var_152 = wp::where(var_136, var_140, var_119);
                var_153 = wp::where(var_136, var_141, var_120);
                var_154 = wp::where(var_136, var_151, var_121);
                // cz = gz - oz                                                                   <L 506>
                var_156 = wp::sub(var_17, var_155);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_159 = (var_156 >= var_158);
                var_157 = var_159;
                if (var_157) {
                    var_160 = (var_156 < var_nz_cells);
                    var_157 = var_157 && var_160;
                }
                if (var_157) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_161 = _cube_flat_id_0(var_123, var_129, var_156, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_162 = wp::atomic_exch(var_dirty_marks, var_161, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_163 = (var_162 != var_stamp);
                    if (var_163) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_166 = wp::atomic_add(var_dirty_count, var_164, var_165);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_167 = &(var_dirty_cube_ids.shape);
                        var_170 = wp::load(var_167);
                        var_169 = wp::extract(var_170, var_168);
                        var_171 = (var_166 < var_169);
                        if (var_171) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_166, var_161);
                        }
                    }
                    var_172 = wp::where(var_163, var_166, var_154);
                }
                var_173 = wp::where(var_157, var_161, var_152);
                var_174 = wp::where(var_157, var_162, var_153);
                var_175 = wp::where(var_157, var_172, var_154);
            }
            var_176 = wp::where(var_130, var_155, var_117);
            var_177 = wp::where(var_130, var_156, var_118);
            var_178 = wp::where(var_130, var_173, var_119);
            var_179 = wp::where(var_130, var_174, var_120);
            var_180 = wp::where(var_130, var_175, var_121);
            // cy = gy - oy                                                                       <L 503>
            var_182 = wp::sub(var_13, var_181);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_185 = (var_182 >= var_184);
            var_183 = var_185;
            if (var_183) {
                var_186 = (var_182 < var_ny_cells);
                var_183 = var_183 && var_186;
            }
            if (var_183) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_188 = wp::sub(var_17, var_187);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_191 = (var_188 >= var_190);
                var_189 = var_191;
                if (var_189) {
                    var_192 = (var_188 < var_nz_cells);
                    var_189 = var_189 && var_192;
                }
                if (var_189) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_193 = _cube_flat_id_0(var_123, var_182, var_188, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_194 = wp::atomic_exch(var_dirty_marks, var_193, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_195 = (var_194 != var_stamp);
                    if (var_195) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_198 = wp::atomic_add(var_dirty_count, var_196, var_197);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_199 = &(var_dirty_cube_ids.shape);
                        var_202 = wp::load(var_199);
                        var_201 = wp::extract(var_202, var_200);
                        var_203 = (var_198 < var_201);
                        if (var_203) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_198, var_193);
                        }
                    }
                    var_204 = wp::where(var_195, var_198, var_180);
                }
                var_205 = wp::where(var_189, var_193, var_178);
                var_206 = wp::where(var_189, var_194, var_179);
                var_207 = wp::where(var_189, var_204, var_180);
                // cz = gz - oz                                                                   <L 506>
                var_209 = wp::sub(var_17, var_208);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_212 = (var_209 >= var_211);
                var_210 = var_212;
                if (var_210) {
                    var_213 = (var_209 < var_nz_cells);
                    var_210 = var_210 && var_213;
                }
                if (var_210) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_214 = _cube_flat_id_0(var_123, var_182, var_209, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    var_215 = wp::atomic_exch(var_dirty_marks, var_214, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_216 = (var_215 != var_stamp);
                    if (var_216) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        var_219 = wp::atomic_add(var_dirty_count, var_217, var_218);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_220 = &(var_dirty_cube_ids.shape);
                        var_223 = wp::load(var_220);
                        var_222 = wp::extract(var_223, var_221);
                        var_224 = (var_219 < var_222);
                        if (var_224) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            wp::array_store(var_dirty_cube_ids, var_219, var_214);
                        }
                    }
                    var_225 = wp::where(var_216, var_219, var_207);
                }
                var_226 = wp::where(var_210, var_214, var_205);
                var_227 = wp::where(var_210, var_215, var_206);
                var_228 = wp::where(var_210, var_225, var_207);
            }
            var_229 = wp::where(var_183, var_208, var_176);
            var_230 = wp::where(var_183, var_209, var_177);
            var_231 = wp::where(var_183, var_226, var_178);
            var_232 = wp::where(var_183, var_227, var_179);
            var_233 = wp::where(var_183, var_228, var_180);
        }
        var_234 = wp::where(var_124, var_181, var_69);
        var_235 = wp::where(var_124, var_182, var_70);
        var_236 = wp::where(var_124, var_229, var_117);
        var_237 = wp::where(var_124, var_230, var_118);
        var_238 = wp::where(var_124, var_231, var_119);
        var_239 = wp::where(var_124, var_232, var_120);
        var_240 = wp::where(var_124, var_233, var_121);
    }
}



extern "C" __global__ void mark_dirty_cubes_from_particles_kernel_462257c1_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_particle_ids,
    wp::int32 var_particle_count,
    wp::array_t<wp::int32> var_particle_grid_xyz,
    wp::int32 var_nx_cells,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::int32 var_stamp,
    wp::array_t<wp::int32> var_dirty_marks,
    wp::array_t<wp::int32> var_dirty_count,
    wp::array_t<wp::int32> var_dirty_cube_ids,
    wp::array_t<wp::int32> adj_particle_ids,
    wp::int32 adj_particle_count,
    wp::array_t<wp::int32> adj_particle_grid_xyz,
    wp::int32 adj_nx_cells,
    wp::int32 adj_ny_cells,
    wp::int32 adj_nz_cells,
    wp::int32 adj_stamp,
    wp::array_t<wp::int32> adj_dirty_marks,
    wp::array_t<wp::int32> adj_dirty_count,
    wp::array_t<wp::int32> adj_dirty_cube_ids)
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
        wp::int32 var_3;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        bool var_6;
        const wp::int32 var_7 = 0;
        wp::int32* var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        const wp::int32 var_11 = 1;
        wp::int32* var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32* var_16;
        wp::int32 var_17;
        wp::int32 var_18;
        const wp::int32 var_19 = 0;
        wp::int32 var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        wp::int32 var_26;
        bool var_27;
        const wp::int32 var_28 = 0;
        bool var_29;
        bool var_30;
        const wp::int32 var_31 = 0;
        wp::int32 var_32;
        bool var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        bool var_36;
        wp::int32 var_37;
        wp::int32 var_38;
        bool var_39;
        const wp::int32 var_40 = 0;
        const wp::int32 var_41 = 1;
        wp::int32 var_42;
        wp::shape_t* var_43;
        const wp::int32 var_44 = 0;
        wp::int32 var_45;
        wp::shape_t var_46;
        bool var_47;
        const wp::int32 var_48 = 1;
        wp::int32 var_49;
        bool var_50;
        const wp::int32 var_51 = 0;
        bool var_52;
        bool var_53;
        wp::int32 var_54;
        wp::int32 var_55;
        bool var_56;
        const wp::int32 var_57 = 0;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        wp::shape_t* var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        wp::shape_t var_63;
        bool var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        wp::int32 var_68;
        const wp::int32 var_69 = 1;
        wp::int32 var_70;
        bool var_71;
        const wp::int32 var_72 = 0;
        bool var_73;
        bool var_74;
        const wp::int32 var_75 = 0;
        wp::int32 var_76;
        bool var_77;
        const wp::int32 var_78 = 0;
        bool var_79;
        bool var_80;
        wp::int32 var_81;
        wp::int32 var_82;
        bool var_83;
        const wp::int32 var_84 = 0;
        const wp::int32 var_85 = 1;
        wp::int32 var_86;
        wp::shape_t* var_87;
        const wp::int32 var_88 = 0;
        wp::int32 var_89;
        wp::shape_t var_90;
        bool var_91;
        wp::int32 var_92;
        wp::int32 var_93;
        wp::int32 var_94;
        wp::int32 var_95;
        const wp::int32 var_96 = 1;
        wp::int32 var_97;
        bool var_98;
        const wp::int32 var_99 = 0;
        bool var_100;
        bool var_101;
        wp::int32 var_102;
        wp::int32 var_103;
        bool var_104;
        const wp::int32 var_105 = 0;
        const wp::int32 var_106 = 1;
        wp::int32 var_107;
        wp::shape_t* var_108;
        const wp::int32 var_109 = 0;
        wp::int32 var_110;
        wp::shape_t var_111;
        bool var_112;
        wp::int32 var_113;
        wp::int32 var_114;
        wp::int32 var_115;
        wp::int32 var_116;
        wp::int32 var_117;
        wp::int32 var_118;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 1;
        wp::int32 var_123;
        bool var_124;
        const wp::int32 var_125 = 0;
        bool var_126;
        bool var_127;
        const wp::int32 var_128 = 0;
        wp::int32 var_129;
        bool var_130;
        const wp::int32 var_131 = 0;
        bool var_132;
        bool var_133;
        const wp::int32 var_134 = 0;
        wp::int32 var_135;
        bool var_136;
        const wp::int32 var_137 = 0;
        bool var_138;
        bool var_139;
        wp::int32 var_140;
        wp::int32 var_141;
        bool var_142;
        const wp::int32 var_143 = 0;
        const wp::int32 var_144 = 1;
        wp::int32 var_145;
        wp::shape_t* var_146;
        const wp::int32 var_147 = 0;
        wp::int32 var_148;
        wp::shape_t var_149;
        bool var_150;
        wp::int32 var_151;
        wp::int32 var_152;
        wp::int32 var_153;
        wp::int32 var_154;
        const wp::int32 var_155 = 1;
        wp::int32 var_156;
        bool var_157;
        const wp::int32 var_158 = 0;
        bool var_159;
        bool var_160;
        wp::int32 var_161;
        wp::int32 var_162;
        bool var_163;
        const wp::int32 var_164 = 0;
        const wp::int32 var_165 = 1;
        wp::int32 var_166;
        wp::shape_t* var_167;
        const wp::int32 var_168 = 0;
        wp::int32 var_169;
        wp::shape_t var_170;
        bool var_171;
        wp::int32 var_172;
        wp::int32 var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        wp::int32 var_176;
        wp::int32 var_177;
        wp::int32 var_178;
        wp::int32 var_179;
        wp::int32 var_180;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        bool var_183;
        const wp::int32 var_184 = 0;
        bool var_185;
        bool var_186;
        const wp::int32 var_187 = 0;
        wp::int32 var_188;
        bool var_189;
        const wp::int32 var_190 = 0;
        bool var_191;
        bool var_192;
        wp::int32 var_193;
        wp::int32 var_194;
        bool var_195;
        const wp::int32 var_196 = 0;
        const wp::int32 var_197 = 1;
        wp::int32 var_198;
        wp::shape_t* var_199;
        const wp::int32 var_200 = 0;
        wp::int32 var_201;
        wp::shape_t var_202;
        bool var_203;
        wp::int32 var_204;
        wp::int32 var_205;
        wp::int32 var_206;
        wp::int32 var_207;
        const wp::int32 var_208 = 1;
        wp::int32 var_209;
        bool var_210;
        const wp::int32 var_211 = 0;
        bool var_212;
        bool var_213;
        wp::int32 var_214;
        wp::int32 var_215;
        bool var_216;
        const wp::int32 var_217 = 0;
        const wp::int32 var_218 = 1;
        wp::int32 var_219;
        wp::shape_t* var_220;
        const wp::int32 var_221 = 0;
        wp::int32 var_222;
        wp::shape_t var_223;
        bool var_224;
        wp::int32 var_225;
        wp::int32 var_226;
        wp::int32 var_227;
        wp::int32 var_228;
        wp::int32 var_229;
        wp::int32 var_230;
        wp::int32 var_231;
        wp::int32 var_232;
        wp::int32 var_233;
        wp::int32 var_234;
        wp::int32 var_235;
        wp::int32 var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        wp::int32 var_239;
        wp::int32 var_240;
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
        bool adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
        bool adj_27 = {};
        wp::int32 adj_28 = {};
        bool adj_29 = {};
        bool adj_30 = {};
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        bool adj_33 = {};
        wp::int32 adj_34 = {};
        bool adj_35 = {};
        bool adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        bool adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::int32 adj_42 = {};
        wp::shape_t adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::shape_t adj_46 = {};
        bool adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        bool adj_50 = {};
        wp::int32 adj_51 = {};
        bool adj_52 = {};
        bool adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        bool adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::shape_t adj_60 = {};
        wp::int32 adj_61 = {};
        wp::int32 adj_62 = {};
        wp::shape_t adj_63 = {};
        bool adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        bool adj_71 = {};
        wp::int32 adj_72 = {};
        bool adj_73 = {};
        bool adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        bool adj_77 = {};
        wp::int32 adj_78 = {};
        bool adj_79 = {};
        bool adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        bool adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::shape_t adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::shape_t adj_90 = {};
        bool adj_91 = {};
        wp::int32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        bool adj_98 = {};
        wp::int32 adj_99 = {};
        bool adj_100 = {};
        bool adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        bool adj_104 = {};
        wp::int32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::shape_t adj_108 = {};
        wp::int32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::shape_t adj_111 = {};
        bool adj_112 = {};
        wp::int32 adj_113 = {};
        wp::int32 adj_114 = {};
        wp::int32 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::int32 adj_120 = {};
        wp::int32 adj_121 = {};
        wp::int32 adj_122 = {};
        wp::int32 adj_123 = {};
        bool adj_124 = {};
        wp::int32 adj_125 = {};
        bool adj_126 = {};
        bool adj_127 = {};
        wp::int32 adj_128 = {};
        wp::int32 adj_129 = {};
        bool adj_130 = {};
        wp::int32 adj_131 = {};
        bool adj_132 = {};
        bool adj_133 = {};
        wp::int32 adj_134 = {};
        wp::int32 adj_135 = {};
        bool adj_136 = {};
        wp::int32 adj_137 = {};
        bool adj_138 = {};
        bool adj_139 = {};
        wp::int32 adj_140 = {};
        wp::int32 adj_141 = {};
        bool adj_142 = {};
        wp::int32 adj_143 = {};
        wp::int32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::shape_t adj_146 = {};
        wp::int32 adj_147 = {};
        wp::int32 adj_148 = {};
        wp::shape_t adj_149 = {};
        bool adj_150 = {};
        wp::int32 adj_151 = {};
        wp::int32 adj_152 = {};
        wp::int32 adj_153 = {};
        wp::int32 adj_154 = {};
        wp::int32 adj_155 = {};
        wp::int32 adj_156 = {};
        bool adj_157 = {};
        wp::int32 adj_158 = {};
        bool adj_159 = {};
        bool adj_160 = {};
        wp::int32 adj_161 = {};
        wp::int32 adj_162 = {};
        bool adj_163 = {};
        wp::int32 adj_164 = {};
        wp::int32 adj_165 = {};
        wp::int32 adj_166 = {};
        wp::shape_t adj_167 = {};
        wp::int32 adj_168 = {};
        wp::int32 adj_169 = {};
        wp::shape_t adj_170 = {};
        bool adj_171 = {};
        wp::int32 adj_172 = {};
        wp::int32 adj_173 = {};
        wp::int32 adj_174 = {};
        wp::int32 adj_175 = {};
        wp::int32 adj_176 = {};
        wp::int32 adj_177 = {};
        wp::int32 adj_178 = {};
        wp::int32 adj_179 = {};
        wp::int32 adj_180 = {};
        wp::int32 adj_181 = {};
        wp::int32 adj_182 = {};
        bool adj_183 = {};
        wp::int32 adj_184 = {};
        bool adj_185 = {};
        bool adj_186 = {};
        wp::int32 adj_187 = {};
        wp::int32 adj_188 = {};
        bool adj_189 = {};
        wp::int32 adj_190 = {};
        bool adj_191 = {};
        bool adj_192 = {};
        wp::int32 adj_193 = {};
        wp::int32 adj_194 = {};
        bool adj_195 = {};
        wp::int32 adj_196 = {};
        wp::int32 adj_197 = {};
        wp::int32 adj_198 = {};
        wp::shape_t adj_199 = {};
        wp::int32 adj_200 = {};
        wp::int32 adj_201 = {};
        wp::shape_t adj_202 = {};
        bool adj_203 = {};
        wp::int32 adj_204 = {};
        wp::int32 adj_205 = {};
        wp::int32 adj_206 = {};
        wp::int32 adj_207 = {};
        wp::int32 adj_208 = {};
        wp::int32 adj_209 = {};
        bool adj_210 = {};
        wp::int32 adj_211 = {};
        bool adj_212 = {};
        bool adj_213 = {};
        wp::int32 adj_214 = {};
        wp::int32 adj_215 = {};
        bool adj_216 = {};
        wp::int32 adj_217 = {};
        wp::int32 adj_218 = {};
        wp::int32 adj_219 = {};
        wp::shape_t adj_220 = {};
        wp::int32 adj_221 = {};
        wp::int32 adj_222 = {};
        wp::shape_t adj_223 = {};
        bool adj_224 = {};
        wp::int32 adj_225 = {};
        wp::int32 adj_226 = {};
        wp::int32 adj_227 = {};
        wp::int32 adj_228 = {};
        wp::int32 adj_229 = {};
        wp::int32 adj_230 = {};
        wp::int32 adj_231 = {};
        wp::int32 adj_232 = {};
        wp::int32 adj_233 = {};
        wp::int32 adj_234 = {};
        wp::int32 adj_235 = {};
        wp::int32 adj_236 = {};
        wp::int32 adj_237 = {};
        wp::int32 adj_238 = {};
        wp::int32 adj_239 = {};
        wp::int32 adj_240 = {};
        //---------
        // forward
        // def mark_dirty_cubes_from_particles_kernel(                                            <L 477>
        // i = wp.tid()                                                                           <L 490>
        var_0 = builtin_tid1d();
        // if i >= particle_count:                                                                <L 491>
        var_1 = (var_0 >= var_particle_count);
        if (var_1) {
            // return                                                                             <L 492>
            goto label0;
        }
        // p = particle_ids[i]                                                                    <L 493>
        var_2 = wp::address(var_particle_ids, var_0);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // if p < 0:                                                                              <L 494>
        var_6 = (var_3 < var_5);
        if (var_6) {
            // return                                                                             <L 495>
            goto label1;
        }
        // gx = particle_grid_xyz[p, 0]                                                           <L 496>
        var_8 = wp::address(var_particle_grid_xyz, var_3, var_7);
        var_10 = wp::load(var_8);
        var_9 = wp::copy(var_10);
        // gy = particle_grid_xyz[p, 1]                                                           <L 497>
        var_12 = wp::address(var_particle_grid_xyz, var_3, var_11);
        var_14 = wp::load(var_12);
        var_13 = wp::copy(var_14);
        // gz = particle_grid_xyz[p, 2]                                                           <L 498>
        var_16 = wp::address(var_particle_grid_xyz, var_3, var_15);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // for ox in range(2):                                                                    <L 499>
        // cx = gx - ox                                                                           <L 500>
        var_20 = wp::sub(var_9, var_19);
        // if cx >= 0 and cx < nx_cells:                                                          <L 501>
        var_23 = (var_20 >= var_22);
        var_21 = var_23;
        if (var_21) {
            var_24 = (var_20 < var_nx_cells);
            var_21 = var_21 && var_24;
        }
        if (var_21) {
            // for oy in range(2):                                                                <L 502>
            // cy = gy - oy                                                                       <L 503>
            var_26 = wp::sub(var_13, var_25);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_29 = (var_26 >= var_28);
            var_27 = var_29;
            if (var_27) {
                var_30 = (var_26 < var_ny_cells);
                var_27 = var_27 && var_30;
            }
            if (var_27) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_32 = wp::sub(var_17, var_31);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_35 = (var_32 >= var_34);
                var_33 = var_35;
                if (var_33) {
                    var_36 = (var_32 < var_nz_cells);
                    var_33 = var_33 && var_36;
                }
                if (var_33) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_37 = _cube_flat_id_0(var_20, var_26, var_32, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_38 = wp::atomic_exch(var_dirty_marks, var_37, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_39 = (var_38 != var_stamp);
                    if (var_39) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_42 = wp::atomic_add(var_dirty_count, var_40, var_41);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_43 = &(var_dirty_cube_ids.shape);
                        var_46 = wp::load(var_43);
                        var_45 = wp::extract(var_46, var_44);
                        var_47 = (var_42 < var_45);
                        if (var_47) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_42, var_37);
                        }
                    }
                }
                // cz = gz - oz                                                                   <L 506>
                var_49 = wp::sub(var_17, var_48);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_52 = (var_49 >= var_51);
                var_50 = var_52;
                if (var_50) {
                    var_53 = (var_49 < var_nz_cells);
                    var_50 = var_50 && var_53;
                }
                if (var_50) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_54 = _cube_flat_id_0(var_20, var_26, var_49, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_55 = wp::atomic_exch(var_dirty_marks, var_54, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_56 = (var_55 != var_stamp);
                    if (var_56) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_59 = wp::atomic_add(var_dirty_count, var_57, var_58);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_60 = &(var_dirty_cube_ids.shape);
                        var_63 = wp::load(var_60);
                        var_62 = wp::extract(var_63, var_61);
                        var_64 = (var_59 < var_62);
                        if (var_64) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_59, var_54);
                        }
                    }
                    var_65 = wp::where(var_56, var_59, var_42);
                }
                var_66 = wp::where(var_50, var_54, var_37);
                var_67 = wp::where(var_50, var_55, var_38);
                var_68 = wp::where(var_50, var_65, var_42);
            }
            // cy = gy - oy                                                                       <L 503>
            var_70 = wp::sub(var_13, var_69);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_73 = (var_70 >= var_72);
            var_71 = var_73;
            if (var_71) {
                var_74 = (var_70 < var_ny_cells);
                var_71 = var_71 && var_74;
            }
            if (var_71) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_76 = wp::sub(var_17, var_75);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_79 = (var_76 >= var_78);
                var_77 = var_79;
                if (var_77) {
                    var_80 = (var_76 < var_nz_cells);
                    var_77 = var_77 && var_80;
                }
                if (var_77) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_81 = _cube_flat_id_0(var_20, var_70, var_76, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_82 = wp::atomic_exch(var_dirty_marks, var_81, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_83 = (var_82 != var_stamp);
                    if (var_83) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_86 = wp::atomic_add(var_dirty_count, var_84, var_85);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_87 = &(var_dirty_cube_ids.shape);
                        var_90 = wp::load(var_87);
                        var_89 = wp::extract(var_90, var_88);
                        var_91 = (var_86 < var_89);
                        if (var_91) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_86, var_81);
                        }
                    }
                    var_92 = wp::where(var_83, var_86, var_68);
                }
                var_93 = wp::where(var_77, var_81, var_66);
                var_94 = wp::where(var_77, var_82, var_67);
                var_95 = wp::where(var_77, var_92, var_68);
                // cz = gz - oz                                                                   <L 506>
                var_97 = wp::sub(var_17, var_96);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_100 = (var_97 >= var_99);
                var_98 = var_100;
                if (var_98) {
                    var_101 = (var_97 < var_nz_cells);
                    var_98 = var_98 && var_101;
                }
                if (var_98) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_102 = _cube_flat_id_0(var_20, var_70, var_97, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_103 = wp::atomic_exch(var_dirty_marks, var_102, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_104 = (var_103 != var_stamp);
                    if (var_104) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_107 = wp::atomic_add(var_dirty_count, var_105, var_106);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_108 = &(var_dirty_cube_ids.shape);
                        var_111 = wp::load(var_108);
                        var_110 = wp::extract(var_111, var_109);
                        var_112 = (var_107 < var_110);
                        if (var_112) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_107, var_102);
                        }
                    }
                    var_113 = wp::where(var_104, var_107, var_95);
                }
                var_114 = wp::where(var_98, var_102, var_93);
                var_115 = wp::where(var_98, var_103, var_94);
                var_116 = wp::where(var_98, var_113, var_95);
            }
            var_117 = wp::where(var_71, var_96, var_48);
            var_118 = wp::where(var_71, var_97, var_49);
            var_119 = wp::where(var_71, var_114, var_66);
            var_120 = wp::where(var_71, var_115, var_67);
            var_121 = wp::where(var_71, var_116, var_68);
        }
        // cx = gx - ox                                                                           <L 500>
        var_123 = wp::sub(var_9, var_122);
        // if cx >= 0 and cx < nx_cells:                                                          <L 501>
        var_126 = (var_123 >= var_125);
        var_124 = var_126;
        if (var_124) {
            var_127 = (var_123 < var_nx_cells);
            var_124 = var_124 && var_127;
        }
        if (var_124) {
            // for oy in range(2):                                                                <L 502>
            // cy = gy - oy                                                                       <L 503>
            var_129 = wp::sub(var_13, var_128);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_132 = (var_129 >= var_131);
            var_130 = var_132;
            if (var_130) {
                var_133 = (var_129 < var_ny_cells);
                var_130 = var_130 && var_133;
            }
            if (var_130) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_135 = wp::sub(var_17, var_134);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_138 = (var_135 >= var_137);
                var_136 = var_138;
                if (var_136) {
                    var_139 = (var_135 < var_nz_cells);
                    var_136 = var_136 && var_139;
                }
                if (var_136) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_140 = _cube_flat_id_0(var_123, var_129, var_135, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_141 = wp::atomic_exch(var_dirty_marks, var_140, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_142 = (var_141 != var_stamp);
                    if (var_142) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_145 = wp::atomic_add(var_dirty_count, var_143, var_144);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_146 = &(var_dirty_cube_ids.shape);
                        var_149 = wp::load(var_146);
                        var_148 = wp::extract(var_149, var_147);
                        var_150 = (var_145 < var_148);
                        if (var_150) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_145, var_140);
                        }
                    }
                    var_151 = wp::where(var_142, var_145, var_121);
                }
                var_152 = wp::where(var_136, var_140, var_119);
                var_153 = wp::where(var_136, var_141, var_120);
                var_154 = wp::where(var_136, var_151, var_121);
                // cz = gz - oz                                                                   <L 506>
                var_156 = wp::sub(var_17, var_155);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_159 = (var_156 >= var_158);
                var_157 = var_159;
                if (var_157) {
                    var_160 = (var_156 < var_nz_cells);
                    var_157 = var_157 && var_160;
                }
                if (var_157) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_161 = _cube_flat_id_0(var_123, var_129, var_156, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_162 = wp::atomic_exch(var_dirty_marks, var_161, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_163 = (var_162 != var_stamp);
                    if (var_163) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_166 = wp::atomic_add(var_dirty_count, var_164, var_165);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_167 = &(var_dirty_cube_ids.shape);
                        var_170 = wp::load(var_167);
                        var_169 = wp::extract(var_170, var_168);
                        var_171 = (var_166 < var_169);
                        if (var_171) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_166, var_161);
                        }
                    }
                    var_172 = wp::where(var_163, var_166, var_154);
                }
                var_173 = wp::where(var_157, var_161, var_152);
                var_174 = wp::where(var_157, var_162, var_153);
                var_175 = wp::where(var_157, var_172, var_154);
            }
            var_176 = wp::where(var_130, var_155, var_117);
            var_177 = wp::where(var_130, var_156, var_118);
            var_178 = wp::where(var_130, var_173, var_119);
            var_179 = wp::where(var_130, var_174, var_120);
            var_180 = wp::where(var_130, var_175, var_121);
            // cy = gy - oy                                                                       <L 503>
            var_182 = wp::sub(var_13, var_181);
            // if cy >= 0 and cy < ny_cells:                                                      <L 504>
            var_185 = (var_182 >= var_184);
            var_183 = var_185;
            if (var_183) {
                var_186 = (var_182 < var_ny_cells);
                var_183 = var_183 && var_186;
            }
            if (var_183) {
                // for oz in range(2):                                                            <L 505>
                // cz = gz - oz                                                                   <L 506>
                var_188 = wp::sub(var_17, var_187);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_191 = (var_188 >= var_190);
                var_189 = var_191;
                if (var_189) {
                    var_192 = (var_188 < var_nz_cells);
                    var_189 = var_189 && var_192;
                }
                if (var_189) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_193 = _cube_flat_id_0(var_123, var_182, var_188, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_194 = wp::atomic_exch(var_dirty_marks, var_193, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_195 = (var_194 != var_stamp);
                    if (var_195) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_198 = wp::atomic_add(var_dirty_count, var_196, var_197);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_199 = &(var_dirty_cube_ids.shape);
                        var_202 = wp::load(var_199);
                        var_201 = wp::extract(var_202, var_200);
                        var_203 = (var_198 < var_201);
                        if (var_203) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_198, var_193);
                        }
                    }
                    var_204 = wp::where(var_195, var_198, var_180);
                }
                var_205 = wp::where(var_189, var_193, var_178);
                var_206 = wp::where(var_189, var_194, var_179);
                var_207 = wp::where(var_189, var_204, var_180);
                // cz = gz - oz                                                                   <L 506>
                var_209 = wp::sub(var_17, var_208);
                // if cz >= 0 and cz < nz_cells:                                                  <L 507>
                var_212 = (var_209 >= var_211);
                var_210 = var_212;
                if (var_210) {
                    var_213 = (var_209 < var_nz_cells);
                    var_210 = var_210 && var_213;
                }
                if (var_210) {
                    // flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                       <L 508>
                    var_214 = _cube_flat_id_0(var_123, var_182, var_209, var_ny_cells, var_nz_cells);
                    // old = wp.atomic_exch(dirty_marks, flat, stamp)                             <L 509>
                    // var_215 = wp::atomic_exch(var_dirty_marks, var_214, var_stamp);
                    // if old != stamp:                                                           <L 510>
                    var_216 = (var_215 != var_stamp);
                    if (var_216) {
                        // dst = wp.atomic_add(dirty_count, 0, 1)                                 <L 511>
                        // var_219 = wp::atomic_add(var_dirty_count, var_217, var_218);
                        // if dst < dirty_cube_ids.shape[0]:                                      <L 512>
                        var_220 = &(var_dirty_cube_ids.shape);
                        var_223 = wp::load(var_220);
                        var_222 = wp::extract(var_223, var_221);
                        var_224 = (var_219 < var_222);
                        if (var_224) {
                            // dirty_cube_ids[dst] = flat                                         <L 513>
                            // wp::array_store(var_dirty_cube_ids, var_219, var_214);
                        }
                    }
                    var_225 = wp::where(var_216, var_219, var_207);
                }
                var_226 = wp::where(var_210, var_214, var_205);
                var_227 = wp::where(var_210, var_215, var_206);
                var_228 = wp::where(var_210, var_225, var_207);
            }
            var_229 = wp::where(var_183, var_208, var_176);
            var_230 = wp::where(var_183, var_209, var_177);
            var_231 = wp::where(var_183, var_226, var_178);
            var_232 = wp::where(var_183, var_227, var_179);
            var_233 = wp::where(var_183, var_228, var_180);
        }
        var_234 = wp::where(var_124, var_181, var_69);
        var_235 = wp::where(var_124, var_182, var_70);
        var_236 = wp::where(var_124, var_229, var_117);
        var_237 = wp::where(var_124, var_230, var_118);
        var_238 = wp::where(var_124, var_231, var_119);
        var_239 = wp::where(var_124, var_232, var_120);
        var_240 = wp::where(var_124, var_233, var_121);
        //---------
        // reverse
        wp::adj_where(var_124, var_233, var_121, adj_124, adj_233, adj_121, adj_240);
        wp::adj_where(var_124, var_232, var_120, adj_124, adj_232, adj_120, adj_239);
        wp::adj_where(var_124, var_231, var_119, adj_124, adj_231, adj_119, adj_238);
        wp::adj_where(var_124, var_230, var_118, adj_124, adj_230, adj_118, adj_237);
        wp::adj_where(var_124, var_229, var_117, adj_124, adj_229, adj_117, adj_236);
        wp::adj_where(var_124, var_182, var_70, adj_124, adj_182, adj_70, adj_235);
        wp::adj_where(var_124, var_181, var_69, adj_124, adj_181, adj_69, adj_234);
        if (var_124) {
            wp::adj_where(var_183, var_228, var_180, adj_183, adj_228, adj_180, adj_233);
            wp::adj_where(var_183, var_227, var_179, adj_183, adj_227, adj_179, adj_232);
            wp::adj_where(var_183, var_226, var_178, adj_183, adj_226, adj_178, adj_231);
            wp::adj_where(var_183, var_209, var_177, adj_183, adj_209, adj_177, adj_230);
            wp::adj_where(var_183, var_208, var_176, adj_183, adj_208, adj_176, adj_229);
            if (var_183) {
                wp::adj_where(var_210, var_225, var_207, adj_210, adj_225, adj_207, adj_228);
                wp::adj_where(var_210, var_215, var_206, adj_210, adj_215, adj_206, adj_227);
                wp::adj_where(var_210, var_214, var_205, adj_210, adj_214, adj_205, adj_226);
                if (var_210) {
                    wp::adj_where(var_216, var_219, var_207, adj_216, adj_219, adj_207, adj_225);
                    if (var_216) {
                        if (var_224) {
                            wp::adj_array_store(var_dirty_cube_ids, var_219, var_214, adj_dirty_cube_ids, adj_219, adj_214);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_223, var_221, adj_220, adj_221, adj_222);
                        adj_dirty_cube_ids.shape = adj_220;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_217, var_218, adj_dirty_count, adj_217, adj_218, adj_219);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_123, var_182, var_209, var_ny_cells, var_nz_cells, adj_123, adj_182, adj_209, adj_ny_cells, adj_nz_cells, adj_214);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_210) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_208, adj_17, adj_208, adj_209);
                // adj: cz = gz - oz                                                              <L 506>
                wp::adj_where(var_189, var_204, var_180, adj_189, adj_204, adj_180, adj_207);
                wp::adj_where(var_189, var_194, var_179, adj_189, adj_194, adj_179, adj_206);
                wp::adj_where(var_189, var_193, var_178, adj_189, adj_193, adj_178, adj_205);
                if (var_189) {
                    wp::adj_where(var_195, var_198, var_180, adj_195, adj_198, adj_180, adj_204);
                    if (var_195) {
                        if (var_203) {
                            wp::adj_array_store(var_dirty_cube_ids, var_198, var_193, adj_dirty_cube_ids, adj_198, adj_193);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_202, var_200, adj_199, adj_200, adj_201);
                        adj_dirty_cube_ids.shape = adj_199;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_196, var_197, adj_dirty_count, adj_196, adj_197, adj_198);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_123, var_182, var_188, var_ny_cells, var_nz_cells, adj_123, adj_182, adj_188, adj_ny_cells, adj_nz_cells, adj_193);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_189) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_187, adj_17, adj_187, adj_188);
                // adj: cz = gz - oz                                                              <L 506>
                // adj: for oz in range(2):                                                       <L 505>
            }
            if (var_183) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 504>
            wp::adj_sub(var_13, var_181, adj_13, adj_181, adj_182);
            // adj: cy = gy - oy                                                                  <L 503>
            wp::adj_where(var_130, var_175, var_121, adj_130, adj_175, adj_121, adj_180);
            wp::adj_where(var_130, var_174, var_120, adj_130, adj_174, adj_120, adj_179);
            wp::adj_where(var_130, var_173, var_119, adj_130, adj_173, adj_119, adj_178);
            wp::adj_where(var_130, var_156, var_118, adj_130, adj_156, adj_118, adj_177);
            wp::adj_where(var_130, var_155, var_117, adj_130, adj_155, adj_117, adj_176);
            if (var_130) {
                wp::adj_where(var_157, var_172, var_154, adj_157, adj_172, adj_154, adj_175);
                wp::adj_where(var_157, var_162, var_153, adj_157, adj_162, adj_153, adj_174);
                wp::adj_where(var_157, var_161, var_152, adj_157, adj_161, adj_152, adj_173);
                if (var_157) {
                    wp::adj_where(var_163, var_166, var_154, adj_163, adj_166, adj_154, adj_172);
                    if (var_163) {
                        if (var_171) {
                            wp::adj_array_store(var_dirty_cube_ids, var_166, var_161, adj_dirty_cube_ids, adj_166, adj_161);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_170, var_168, adj_167, adj_168, adj_169);
                        adj_dirty_cube_ids.shape = adj_167;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_164, var_165, adj_dirty_count, adj_164, adj_165, adj_166);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_123, var_129, var_156, var_ny_cells, var_nz_cells, adj_123, adj_129, adj_156, adj_ny_cells, adj_nz_cells, adj_161);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_157) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_155, adj_17, adj_155, adj_156);
                // adj: cz = gz - oz                                                              <L 506>
                wp::adj_where(var_136, var_151, var_121, adj_136, adj_151, adj_121, adj_154);
                wp::adj_where(var_136, var_141, var_120, adj_136, adj_141, adj_120, adj_153);
                wp::adj_where(var_136, var_140, var_119, adj_136, adj_140, adj_119, adj_152);
                if (var_136) {
                    wp::adj_where(var_142, var_145, var_121, adj_142, adj_145, adj_121, adj_151);
                    if (var_142) {
                        if (var_150) {
                            wp::adj_array_store(var_dirty_cube_ids, var_145, var_140, adj_dirty_cube_ids, adj_145, adj_140);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_149, var_147, adj_146, adj_147, adj_148);
                        adj_dirty_cube_ids.shape = adj_146;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_143, var_144, adj_dirty_count, adj_143, adj_144, adj_145);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_123, var_129, var_135, var_ny_cells, var_nz_cells, adj_123, adj_129, adj_135, adj_ny_cells, adj_nz_cells, adj_140);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_136) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_134, adj_17, adj_134, adj_135);
                // adj: cz = gz - oz                                                              <L 506>
                // adj: for oz in range(2):                                                       <L 505>
            }
            if (var_130) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 504>
            wp::adj_sub(var_13, var_128, adj_13, adj_128, adj_129);
            // adj: cy = gy - oy                                                                  <L 503>
            // adj: for oy in range(2):                                                           <L 502>
        }
        if (var_124) {
        }
        // adj: if cx >= 0 and cx < nx_cells:                                                     <L 501>
        wp::adj_sub(var_9, var_122, adj_9, adj_122, adj_123);
        // adj: cx = gx - ox                                                                      <L 500>
        if (var_21) {
            wp::adj_where(var_71, var_116, var_68, adj_71, adj_116, adj_68, adj_121);
            wp::adj_where(var_71, var_115, var_67, adj_71, adj_115, adj_67, adj_120);
            wp::adj_where(var_71, var_114, var_66, adj_71, adj_114, adj_66, adj_119);
            wp::adj_where(var_71, var_97, var_49, adj_71, adj_97, adj_49, adj_118);
            wp::adj_where(var_71, var_96, var_48, adj_71, adj_96, adj_48, adj_117);
            if (var_71) {
                wp::adj_where(var_98, var_113, var_95, adj_98, adj_113, adj_95, adj_116);
                wp::adj_where(var_98, var_103, var_94, adj_98, adj_103, adj_94, adj_115);
                wp::adj_where(var_98, var_102, var_93, adj_98, adj_102, adj_93, adj_114);
                if (var_98) {
                    wp::adj_where(var_104, var_107, var_95, adj_104, adj_107, adj_95, adj_113);
                    if (var_104) {
                        if (var_112) {
                            wp::adj_array_store(var_dirty_cube_ids, var_107, var_102, adj_dirty_cube_ids, adj_107, adj_102);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_111, var_109, adj_108, adj_109, adj_110);
                        adj_dirty_cube_ids.shape = adj_108;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_105, var_106, adj_dirty_count, adj_105, adj_106, adj_107);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_20, var_70, var_97, var_ny_cells, var_nz_cells, adj_20, adj_70, adj_97, adj_ny_cells, adj_nz_cells, adj_102);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_98) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_96, adj_17, adj_96, adj_97);
                // adj: cz = gz - oz                                                              <L 506>
                wp::adj_where(var_77, var_92, var_68, adj_77, adj_92, adj_68, adj_95);
                wp::adj_where(var_77, var_82, var_67, adj_77, adj_82, adj_67, adj_94);
                wp::adj_where(var_77, var_81, var_66, adj_77, adj_81, adj_66, adj_93);
                if (var_77) {
                    wp::adj_where(var_83, var_86, var_68, adj_83, adj_86, adj_68, adj_92);
                    if (var_83) {
                        if (var_91) {
                            wp::adj_array_store(var_dirty_cube_ids, var_86, var_81, adj_dirty_cube_ids, adj_86, adj_81);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_90, var_88, adj_87, adj_88, adj_89);
                        adj_dirty_cube_ids.shape = adj_87;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_84, var_85, adj_dirty_count, adj_84, adj_85, adj_86);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_20, var_70, var_76, var_ny_cells, var_nz_cells, adj_20, adj_70, adj_76, adj_ny_cells, adj_nz_cells, adj_81);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_77) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_75, adj_17, adj_75, adj_76);
                // adj: cz = gz - oz                                                              <L 506>
                // adj: for oz in range(2):                                                       <L 505>
            }
            if (var_71) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 504>
            wp::adj_sub(var_13, var_69, adj_13, adj_69, adj_70);
            // adj: cy = gy - oy                                                                  <L 503>
            if (var_27) {
                wp::adj_where(var_50, var_65, var_42, adj_50, adj_65, adj_42, adj_68);
                wp::adj_where(var_50, var_55, var_38, adj_50, adj_55, adj_38, adj_67);
                wp::adj_where(var_50, var_54, var_37, adj_50, adj_54, adj_37, adj_66);
                if (var_50) {
                    wp::adj_where(var_56, var_59, var_42, adj_56, adj_59, adj_42, adj_65);
                    if (var_56) {
                        if (var_64) {
                            wp::adj_array_store(var_dirty_cube_ids, var_59, var_54, adj_dirty_cube_ids, adj_59, adj_54);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_63, var_61, adj_60, adj_61, adj_62);
                        adj_dirty_cube_ids.shape = adj_60;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_57, var_58, adj_dirty_count, adj_57, adj_58, adj_59);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_20, var_26, var_49, var_ny_cells, var_nz_cells, adj_20, adj_26, adj_49, adj_ny_cells, adj_nz_cells, adj_54);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_50) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_48, adj_17, adj_48, adj_49);
                // adj: cz = gz - oz                                                              <L 506>
                if (var_33) {
                    if (var_39) {
                        if (var_47) {
                            wp::adj_array_store(var_dirty_cube_ids, var_42, var_37, adj_dirty_cube_ids, adj_42, adj_37);
                            // adj: dirty_cube_ids[dst] = flat                                    <L 513>
                        }
                        wp::adj_extract(var_46, var_44, adj_43, adj_44, adj_45);
                        adj_dirty_cube_ids.shape = adj_43;
                        // adj: if dst < dirty_cube_ids.shape[0]:                                 <L 512>
                        wp::adj_atomic_add(var_dirty_count, var_40, var_41, adj_dirty_count, adj_40, adj_41, adj_42);
                        // adj: dst = wp.atomic_add(dirty_count, 0, 1)                            <L 511>
                    }
                    // adj: if old != stamp:                                                      <L 510>
                    // adj: old = wp.atomic_exch(dirty_marks, flat, stamp)                        <L 509>
                    adj__cube_flat_id_0(var_20, var_26, var_32, var_ny_cells, var_nz_cells, adj_20, adj_26, adj_32, adj_ny_cells, adj_nz_cells, adj_37);
                    // adj: flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                  <L 508>
                }
                if (var_33) {
                }
                // adj: if cz >= 0 and cz < nz_cells:                                             <L 507>
                wp::adj_sub(var_17, var_31, adj_17, adj_31, adj_32);
                // adj: cz = gz - oz                                                              <L 506>
                // adj: for oz in range(2):                                                       <L 505>
            }
            if (var_27) {
            }
            // adj: if cy >= 0 and cy < ny_cells:                                                 <L 504>
            wp::adj_sub(var_13, var_25, adj_13, adj_25, adj_26);
            // adj: cy = gy - oy                                                                  <L 503>
            // adj: for oy in range(2):                                                           <L 502>
        }
        if (var_21) {
        }
        // adj: if cx >= 0 and cx < nx_cells:                                                     <L 501>
        wp::adj_sub(var_9, var_19, adj_9, adj_19, adj_20);
        // adj: cx = gx - ox                                                                      <L 500>
        // adj: for ox in range(2):                                                               <L 499>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_address(var_particle_grid_xyz, var_3, var_15, adj_particle_grid_xyz, adj_3, adj_15, adj_16);
        // adj: gz = particle_grid_xyz[p, 2]                                                      <L 498>
        wp::adj_copy(var_14, adj_12, adj_13);
        wp::adj_address(var_particle_grid_xyz, var_3, var_11, adj_particle_grid_xyz, adj_3, adj_11, adj_12);
        // adj: gy = particle_grid_xyz[p, 1]                                                      <L 497>
        wp::adj_copy(var_10, adj_8, adj_9);
        wp::adj_address(var_particle_grid_xyz, var_3, var_7, adj_particle_grid_xyz, adj_3, adj_7, adj_8);
        // adj: gx = particle_grid_xyz[p, 0]                                                      <L 496>
        if (var_6) {
            label1:;
            // adj: return                                                                        <L 495>
        }
        // adj: if p < 0:                                                                         <L 494>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_particle_ids, var_0, adj_particle_ids, adj_0, adj_2);
        // adj: p = particle_ids[i]                                                               <L 493>
        if (var_1) {
            label0:;
            // adj: return                                                                        <L 492>
        }
        // adj: if i >= particle_count:                                                           <L 491>
        // adj: i = wp.tid()                                                                      <L 490>
        // adj: def mark_dirty_cubes_from_particles_kernel(                                       <L 477>
        continue;
    }
}



extern "C" __global__ void compute_cube_cases_kernel_3dbe880f_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_cube_cases)
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
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 2;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 3;
        wp::int32 var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 4;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 5;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        const wp::int32 var_59 = 6;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 7;
        wp::int32 var_69;
        const wp::int32 var_70 = 0;
        bool var_71;
        const wp::int32 var_72 = 1;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::int32 var_76;
        //---------
        // forward
        // def compute_cube_cases_kernel(                                                         <L 124>
        // cx, cy, cz = wp.tid()                                                                  <L 136>
        builtin_tid3d(var_0, var_1, var_2);
        // mask = int(0)                                                                          <L 137>
        var_4 = wp::int(var_3);
        // for c in range(8):                                                                     <L 138>
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_6 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_5);
        var_8 = (var_6 != var_7);
        if (var_8) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_10 = wp::int(var_9);
            var_11 = wp::lshift(var_10, var_5);
            var_12 = wp::bit_or(var_4, var_11);
        }
        var_13 = wp::where(var_8, var_12, var_4);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_15 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_14);
        var_17 = (var_15 != var_16);
        if (var_17) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_19 = wp::int(var_18);
            var_20 = wp::lshift(var_19, var_14);
            var_21 = wp::bit_or(var_13, var_20);
        }
        var_22 = wp::where(var_17, var_21, var_13);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_24 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_23);
        var_26 = (var_24 != var_25);
        if (var_26) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_28 = wp::int(var_27);
            var_29 = wp::lshift(var_28, var_23);
            var_30 = wp::bit_or(var_22, var_29);
        }
        var_31 = wp::where(var_26, var_30, var_22);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_33 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_32);
        var_35 = (var_33 != var_34);
        if (var_35) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_37 = wp::int(var_36);
            var_38 = wp::lshift(var_37, var_32);
            var_39 = wp::bit_or(var_31, var_38);
        }
        var_40 = wp::where(var_35, var_39, var_31);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_42 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_41);
        var_44 = (var_42 != var_43);
        if (var_44) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_46 = wp::int(var_45);
            var_47 = wp::lshift(var_46, var_41);
            var_48 = wp::bit_or(var_40, var_47);
        }
        var_49 = wp::where(var_44, var_48, var_40);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_51 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_50);
        var_53 = (var_51 != var_52);
        if (var_53) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_55 = wp::int(var_54);
            var_56 = wp::lshift(var_55, var_50);
            var_57 = wp::bit_or(var_49, var_56);
        }
        var_58 = wp::where(var_53, var_57, var_49);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_60 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_59);
        var_62 = (var_60 != var_61);
        if (var_62) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_64 = wp::int(var_63);
            var_65 = wp::lshift(var_64, var_59);
            var_66 = wp::bit_or(var_58, var_65);
        }
        var_67 = wp::where(var_62, var_66, var_58);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_69 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_68);
        var_71 = (var_69 != var_70);
        if (var_71) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_73 = wp::int(var_72);
            var_74 = wp::lshift(var_73, var_68);
            var_75 = wp::bit_or(var_67, var_74);
        }
        var_76 = wp::where(var_71, var_75, var_67);
        // cube_cases[cx, cy, cz] = mask                                                          <L 141>
        wp::array_store(var_cube_cases, var_0, var_1, var_2, var_76);
    }
}



extern "C" __global__ void compute_cube_cases_kernel_3dbe880f_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> adj_grid_to_particle,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_corner_offsets,
    wp::array_t<wp::int32> adj_cube_cases)
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
        const wp::int32 var_3 = 0;
        wp::int32 var_4;
        const wp::int32 var_5 = 0;
        wp::int32 var_6;
        const wp::int32 var_7 = 0;
        bool var_8;
        const wp::int32 var_9 = 1;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        const wp::int32 var_14 = 1;
        wp::int32 var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        const wp::int32 var_18 = 1;
        wp::int32 var_19;
        wp::int32 var_20;
        wp::int32 var_21;
        wp::int32 var_22;
        const wp::int32 var_23 = 2;
        wp::int32 var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 1;
        wp::int32 var_28;
        wp::int32 var_29;
        wp::int32 var_30;
        wp::int32 var_31;
        const wp::int32 var_32 = 3;
        wp::int32 var_33;
        const wp::int32 var_34 = 0;
        bool var_35;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        wp::int32 var_40;
        const wp::int32 var_41 = 4;
        wp::int32 var_42;
        const wp::int32 var_43 = 0;
        bool var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        wp::int32 var_47;
        wp::int32 var_48;
        wp::int32 var_49;
        const wp::int32 var_50 = 5;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = 1;
        wp::int32 var_55;
        wp::int32 var_56;
        wp::int32 var_57;
        wp::int32 var_58;
        const wp::int32 var_59 = 6;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        bool var_62;
        const wp::int32 var_63 = 1;
        wp::int32 var_64;
        wp::int32 var_65;
        wp::int32 var_66;
        wp::int32 var_67;
        const wp::int32 var_68 = 7;
        wp::int32 var_69;
        const wp::int32 var_70 = 0;
        bool var_71;
        const wp::int32 var_72 = 1;
        wp::int32 var_73;
        wp::int32 var_74;
        wp::int32 var_75;
        wp::int32 var_76;
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
        wp::int32 adj_9 = {};
        wp::int32 adj_10 = {};
        wp::int32 adj_11 = {};
        wp::int32 adj_12 = {};
        wp::int32 adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::int32 adj_16 = {};
        bool adj_17 = {};
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
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
        wp::int32 adj_31 = {};
        wp::int32 adj_32 = {};
        wp::int32 adj_33 = {};
        wp::int32 adj_34 = {};
        bool adj_35 = {};
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
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        bool adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        wp::int32 adj_61 = {};
        bool adj_62 = {};
        wp::int32 adj_63 = {};
        wp::int32 adj_64 = {};
        wp::int32 adj_65 = {};
        wp::int32 adj_66 = {};
        wp::int32 adj_67 = {};
        wp::int32 adj_68 = {};
        wp::int32 adj_69 = {};
        wp::int32 adj_70 = {};
        bool adj_71 = {};
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        wp::int32 adj_76 = {};
        //---------
        // forward
        // def compute_cube_cases_kernel(                                                         <L 124>
        // cx, cy, cz = wp.tid()                                                                  <L 136>
        builtin_tid3d(var_0, var_1, var_2);
        // mask = int(0)                                                                          <L 137>
        var_4 = wp::int(var_3);
        // for c in range(8):                                                                     <L 138>
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_6 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_5);
        var_8 = (var_6 != var_7);
        if (var_8) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_10 = wp::int(var_9);
            var_11 = wp::lshift(var_10, var_5);
            var_12 = wp::bit_or(var_4, var_11);
        }
        var_13 = wp::where(var_8, var_12, var_4);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_15 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_14);
        var_17 = (var_15 != var_16);
        if (var_17) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_19 = wp::int(var_18);
            var_20 = wp::lshift(var_19, var_14);
            var_21 = wp::bit_or(var_13, var_20);
        }
        var_22 = wp::where(var_17, var_21, var_13);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_24 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_23);
        var_26 = (var_24 != var_25);
        if (var_26) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_28 = wp::int(var_27);
            var_29 = wp::lshift(var_28, var_23);
            var_30 = wp::bit_or(var_22, var_29);
        }
        var_31 = wp::where(var_26, var_30, var_22);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_33 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_32);
        var_35 = (var_33 != var_34);
        if (var_35) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_37 = wp::int(var_36);
            var_38 = wp::lshift(var_37, var_32);
            var_39 = wp::bit_or(var_31, var_38);
        }
        var_40 = wp::where(var_35, var_39, var_31);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_42 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_41);
        var_44 = (var_42 != var_43);
        if (var_44) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_46 = wp::int(var_45);
            var_47 = wp::lshift(var_46, var_41);
            var_48 = wp::bit_or(var_40, var_47);
        }
        var_49 = wp::where(var_44, var_48, var_40);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_51 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_50);
        var_53 = (var_51 != var_52);
        if (var_53) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_55 = wp::int(var_54);
            var_56 = wp::lshift(var_55, var_50);
            var_57 = wp::bit_or(var_49, var_56);
        }
        var_58 = wp::where(var_53, var_57, var_49);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_60 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_59);
        var_62 = (var_60 != var_61);
        if (var_62) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_64 = wp::int(var_63);
            var_65 = wp::lshift(var_64, var_59);
            var_66 = wp::bit_or(var_58, var_65);
        }
        var_67 = wp::where(var_62, var_66, var_58);
        // if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:       <L 139>
        var_69 = _corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_68);
        var_71 = (var_69 != var_70);
        if (var_71) {
            // mask = mask | (int(1) << c)                                                        <L 140>
            var_73 = wp::int(var_72);
            var_74 = wp::lshift(var_73, var_68);
            var_75 = wp::bit_or(var_67, var_74);
        }
        var_76 = wp::where(var_71, var_75, var_67);
        // cube_cases[cx, cy, cz] = mask                                                          <L 141>
        // wp::array_store(var_cube_cases, var_0, var_1, var_2, var_76);
        //---------
        // reverse
        wp::adj_array_store(var_cube_cases, var_0, var_1, var_2, var_76, adj_cube_cases, adj_0, adj_1, adj_2, adj_76);
        // adj: cube_cases[cx, cy, cz] = mask                                                     <L 141>
        wp::adj_where(var_71, var_75, var_67, adj_71, adj_75, adj_67, adj_76);
        if (var_71) {
            wp::adj_int(var_72, adj_72, adj_73);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_68, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_68, adj_69);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_62, var_66, var_58, adj_62, adj_66, adj_58, adj_67);
        if (var_62) {
            wp::adj_int(var_63, adj_63, adj_64);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_59, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_59, adj_60);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_53, var_57, var_49, adj_53, adj_57, adj_49, adj_58);
        if (var_53) {
            wp::adj_int(var_54, adj_54, adj_55);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_50, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_50, adj_51);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_44, var_48, var_40, adj_44, adj_48, adj_40, adj_49);
        if (var_44) {
            wp::adj_int(var_45, adj_45, adj_46);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_41, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_41, adj_42);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_35, var_39, var_31, adj_35, adj_39, adj_31, adj_40);
        if (var_35) {
            wp::adj_int(var_36, adj_36, adj_37);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_32, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_32, adj_33);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_26, var_30, var_22, adj_26, adj_30, adj_22, adj_31);
        if (var_26) {
            wp::adj_int(var_27, adj_27, adj_28);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_23, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_23, adj_24);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_17, var_21, var_13, adj_17, adj_21, adj_13, adj_22);
        if (var_17) {
            wp::adj_int(var_18, adj_18, adj_19);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_14, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_14, adj_15);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        wp::adj_where(var_8, var_12, var_4, adj_8, adj_12, adj_4, adj_13);
        if (var_8) {
            wp::adj_int(var_9, adj_9, adj_10);
            // adj: mask = mask | (int(1) << c)                                                   <L 140>
        }
        adj__corner_active_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_0, var_1, var_2, var_5, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_0, adj_1, adj_2, adj_5, adj_6);
        // adj: if _corner_active(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz, c) != 0:  <L 139>
        // adj: for c in range(8):                                                                <L 138>
        wp::adj_int(var_3, adj_3, adj_4);
        // adj: mask = int(0)                                                                     <L 137>
        // adj: cx, cy, cz = wp.tid()                                                             <L 136>
        // adj: def compute_cube_cases_kernel(                                                    <L 124>
        continue;
    }
}



extern "C" __global__ void reemit_dirty_cube_slots_kernel_dc0efb8e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dirty_cube_ids,
    wp::array_t<wp::int32> var_dirty_count,
    wp::int32 var_nx_cells,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> var_compact_to_slot,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices)
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
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        bool var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 5;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        wp::int32* var_39;
        const wp::int32 var_40 = 0;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32* var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        const wp::int32 var_46 = 2;
        wp::int32* var_47;
        const wp::int32 var_48 = 2;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        wp::int32* var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::int32* var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32* var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        const wp::int32 var_67 = 2;
        wp::int32* var_68;
        const wp::int32 var_69 = 2;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 2;
        wp::int32 var_73;
        wp::int32* var_74;
        const wp::int32 var_75 = 0;
        bool var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 0;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        wp::int32* var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32* var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        const wp::int32 var_89 = 2;
        wp::int32* var_90;
        const wp::int32 var_91 = 2;
        wp::int32 var_92;
        wp::int32 var_93;
        const wp::int32 var_94 = 3;
        wp::int32 var_95;
        wp::int32* var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 0;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        wp::int32* var_104;
        const wp::int32 var_105 = 0;
        wp::int32 var_106;
        const wp::int32 var_107 = 1;
        wp::int32* var_108;
        const wp::int32 var_109 = 1;
        wp::int32 var_110;
        const wp::int32 var_111 = 2;
        wp::int32* var_112;
        const wp::int32 var_113 = 2;
        wp::int32 var_114;
        wp::int32 var_115;
        const wp::int32 var_116 = 4;
        wp::int32 var_117;
        wp::int32* var_118;
        const wp::int32 var_119 = 0;
        bool var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 0;
        const wp::int32 var_123 = 1;
        wp::int32 var_124;
        const wp::int32 var_125 = 0;
        wp::int32* var_126;
        const wp::int32 var_127 = 0;
        wp::int32 var_128;
        const wp::int32 var_129 = 1;
        wp::int32* var_130;
        const wp::int32 var_131 = 1;
        wp::int32 var_132;
        const wp::int32 var_133 = 2;
        wp::int32* var_134;
        const wp::int32 var_135 = 2;
        wp::int32 var_136;
        wp::int32 var_137;
        //---------
        // forward
        // def reemit_dirty_cube_slots_kernel(                                                    <L 598>
        // i = wp.tid()                                                                           <L 620>
        var_0 = builtin_tid1d();
        // if i >= dirty_count[0]:                                                                <L 621>
        var_2 = wp::address(var_dirty_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 622>
            continue;
        }
        // cube_flat = dirty_cube_ids[i]                                                          <L 624>
        var_5 = wp::address(var_dirty_cube_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // plane = ny_cells * nz_cells                                                            <L 625>
        var_8 = wp::mul(var_ny_cells, var_nz_cells);
        // cx = cube_flat / plane                                                                 <L 626>
        var_9 = wp::div(var_6, var_8);
        // rem = cube_flat - cx * plane                                                           <L 627>
        var_10 = wp::mul(var_9, var_8);
        var_11 = wp::sub(var_6, var_10);
        // cy = rem / nz_cells                                                                    <L 628>
        var_12 = wp::div(var_11, var_nz_cells);
        // cz = rem - cy * nz_cells                                                               <L 629>
        var_13 = wp::mul(var_12, var_nz_cells);
        var_14 = wp::sub(var_11, var_13);
        // if cx < 0 or cx >= nx_cells or cy < 0 or cy >= ny_cells or cz < 0 or cz >= nz_cells:       <L 630>
        var_17 = (var_9 < var_16);
        var_15 = var_17;
        if (!var_15) {
            var_18 = (var_9 >= var_nx_cells);
            var_15 = var_15 || var_18;
        }
        if (!var_15) {
            var_20 = (var_12 < var_19);
            var_15 = var_15 || var_20;
        }
        if (!var_15) {
            var_21 = (var_12 >= var_ny_cells);
            var_15 = var_15 || var_21;
        }
        if (!var_15) {
            var_23 = (var_14 < var_22);
            var_15 = var_15 || var_23;
        }
        if (!var_15) {
            var_24 = (var_14 >= var_nz_cells);
            var_15 = var_15 || var_24;
        }
        if (var_15) {
            // return                                                                             <L 631>
            continue;
        }
        // case = _compute_cube_case(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz)       <L 633>
        var_25 = _compute_cube_case_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_9, var_12, var_14);
        // cube_cases[cx, cy, cz] = case                                                          <L 634>
        wp::array_store(var_cube_cases, var_9, var_12, var_14, var_25);
        // _write_cube_fixed_slots(                                                               <L 635>
        // cube_flat,                                                                             <L 636>
        // cx,                                                                                    <L 637>
        // cy,                                                                                    <L 638>
        // cz,                                                                                    <L 639>
        // case,                                                                                  <L 640>
        // grid_to_particle,                                                                      <L 641>
        // particle_flags,                                                                        <L 642>
        // corner_offsets,                                                                        <L 643>
        // edge_corners,                                                                          <L 644>
        // edge_base_dir,                                                                         <L 645>
        // case_triangles,                                                                        <L 646>
        // cube_tri_counts,                                                                       <L 647>
        // slot_tri_indices,                                                                      <L 648>
        // slot_active,                                                                           <L 649>
        // slot_to_compact,                                                                       <L 650>
        var_26 = _write_cube_fixed_slots_0(var_6, var_9, var_12, var_14, var_25, var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_case_triangles, var_cube_tri_counts, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
        // base = cube_flat * MC_MAX_TRIS_PER_CASE                                                <L 653>
        var_28 = wp::mul(var_6, var_27);
        // for local in range(MC_MAX_TRIS_PER_CASE):                                              <L 654>
        // slot = base + local                                                                    <L 655>
        var_30 = wp::add(var_28, var_29);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_31 = wp::address(var_slot_active, var_30);
        var_34 = wp::load(var_31);
        var_33 = (var_34 != var_32);
        if (var_33) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            var_37 = wp::atomic_add(var_tri_count, var_35, var_36);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_39 = wp::address(var_slot_tri_indices, var_30, var_38);
            var_41 = wp::load(var_39);
            wp::array_store(var_tri_indices, var_37, var_40, var_41);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_43 = wp::address(var_slot_tri_indices, var_30, var_42);
            var_45 = wp::load(var_43);
            wp::array_store(var_tri_indices, var_37, var_44, var_45);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_47 = wp::address(var_slot_tri_indices, var_30, var_46);
            var_49 = wp::load(var_47);
            wp::array_store(var_tri_indices, var_37, var_48, var_49);
            // slot_to_compact[slot] = dst                                                        <L 661>
            wp::array_store(var_slot_to_compact, var_30, var_37);
            // compact_to_slot[dst] = slot                                                        <L 662>
            wp::array_store(var_compact_to_slot, var_37, var_30);
        }
        // slot = base + local                                                                    <L 655>
        var_51 = wp::add(var_28, var_50);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_52 = wp::address(var_slot_active, var_51);
        var_55 = wp::load(var_52);
        var_54 = (var_55 != var_53);
        if (var_54) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            var_58 = wp::atomic_add(var_tri_count, var_56, var_57);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_60 = wp::address(var_slot_tri_indices, var_51, var_59);
            var_62 = wp::load(var_60);
            wp::array_store(var_tri_indices, var_58, var_61, var_62);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_64 = wp::address(var_slot_tri_indices, var_51, var_63);
            var_66 = wp::load(var_64);
            wp::array_store(var_tri_indices, var_58, var_65, var_66);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_68 = wp::address(var_slot_tri_indices, var_51, var_67);
            var_70 = wp::load(var_68);
            wp::array_store(var_tri_indices, var_58, var_69, var_70);
            // slot_to_compact[slot] = dst                                                        <L 661>
            wp::array_store(var_slot_to_compact, var_51, var_58);
            // compact_to_slot[dst] = slot                                                        <L 662>
            wp::array_store(var_compact_to_slot, var_58, var_51);
        }
        var_71 = wp::where(var_54, var_58, var_37);
        // slot = base + local                                                                    <L 655>
        var_73 = wp::add(var_28, var_72);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_74 = wp::address(var_slot_active, var_73);
        var_77 = wp::load(var_74);
        var_76 = (var_77 != var_75);
        if (var_76) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            var_80 = wp::atomic_add(var_tri_count, var_78, var_79);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_82 = wp::address(var_slot_tri_indices, var_73, var_81);
            var_84 = wp::load(var_82);
            wp::array_store(var_tri_indices, var_80, var_83, var_84);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_86 = wp::address(var_slot_tri_indices, var_73, var_85);
            var_88 = wp::load(var_86);
            wp::array_store(var_tri_indices, var_80, var_87, var_88);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_90 = wp::address(var_slot_tri_indices, var_73, var_89);
            var_92 = wp::load(var_90);
            wp::array_store(var_tri_indices, var_80, var_91, var_92);
            // slot_to_compact[slot] = dst                                                        <L 661>
            wp::array_store(var_slot_to_compact, var_73, var_80);
            // compact_to_slot[dst] = slot                                                        <L 662>
            wp::array_store(var_compact_to_slot, var_80, var_73);
        }
        var_93 = wp::where(var_76, var_80, var_71);
        // slot = base + local                                                                    <L 655>
        var_95 = wp::add(var_28, var_94);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_96 = wp::address(var_slot_active, var_95);
        var_99 = wp::load(var_96);
        var_98 = (var_99 != var_97);
        if (var_98) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            var_102 = wp::atomic_add(var_tri_count, var_100, var_101);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_104 = wp::address(var_slot_tri_indices, var_95, var_103);
            var_106 = wp::load(var_104);
            wp::array_store(var_tri_indices, var_102, var_105, var_106);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_108 = wp::address(var_slot_tri_indices, var_95, var_107);
            var_110 = wp::load(var_108);
            wp::array_store(var_tri_indices, var_102, var_109, var_110);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_112 = wp::address(var_slot_tri_indices, var_95, var_111);
            var_114 = wp::load(var_112);
            wp::array_store(var_tri_indices, var_102, var_113, var_114);
            // slot_to_compact[slot] = dst                                                        <L 661>
            wp::array_store(var_slot_to_compact, var_95, var_102);
            // compact_to_slot[dst] = slot                                                        <L 662>
            wp::array_store(var_compact_to_slot, var_102, var_95);
        }
        var_115 = wp::where(var_98, var_102, var_93);
        // slot = base + local                                                                    <L 655>
        var_117 = wp::add(var_28, var_116);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_118 = wp::address(var_slot_active, var_117);
        var_121 = wp::load(var_118);
        var_120 = (var_121 != var_119);
        if (var_120) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            var_124 = wp::atomic_add(var_tri_count, var_122, var_123);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_126 = wp::address(var_slot_tri_indices, var_117, var_125);
            var_128 = wp::load(var_126);
            wp::array_store(var_tri_indices, var_124, var_127, var_128);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_130 = wp::address(var_slot_tri_indices, var_117, var_129);
            var_132 = wp::load(var_130);
            wp::array_store(var_tri_indices, var_124, var_131, var_132);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_134 = wp::address(var_slot_tri_indices, var_117, var_133);
            var_136 = wp::load(var_134);
            wp::array_store(var_tri_indices, var_124, var_135, var_136);
            // slot_to_compact[slot] = dst                                                        <L 661>
            wp::array_store(var_slot_to_compact, var_117, var_124);
            // compact_to_slot[dst] = slot                                                        <L 662>
            wp::array_store(var_compact_to_slot, var_124, var_117);
        }
        var_137 = wp::where(var_120, var_124, var_115);
    }
}



extern "C" __global__ void reemit_dirty_cube_slots_kernel_dc0efb8e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dirty_cube_ids,
    wp::array_t<wp::int32> var_dirty_count,
    wp::int32 var_nx_cells,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> var_compact_to_slot,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> adj_dirty_cube_ids,
    wp::array_t<wp::int32> adj_dirty_count,
    wp::int32 adj_nx_cells,
    wp::int32 adj_ny_cells,
    wp::int32 adj_nz_cells,
    wp::array_t<wp::int32> adj_grid_to_particle,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_cube_cases,
    wp::array_t<wp::int32> adj_corner_offsets,
    wp::array_t<wp::int32> adj_edge_corners,
    wp::array_t<wp::int32> adj_edge_base_dir,
    wp::array_t<wp::int32> adj_case_triangles,
    wp::array_t<wp::int32> adj_cube_tri_counts,
    wp::array_t<wp::int32> adj_slot_tri_indices,
    wp::array_t<wp::int32> adj_slot_active,
    wp::array_t<wp::int32> adj_slot_to_compact,
    wp::array_t<wp::int32> adj_compact_to_slot,
    wp::array_t<wp::int32> adj_tri_count,
    wp::array_t<wp::int32> adj_tri_indices)
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
        wp::int32 var_8;
        wp::int32 var_9;
        wp::int32 var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        wp::int32 var_13;
        wp::int32 var_14;
        bool var_15;
        const wp::int32 var_16 = 0;
        bool var_17;
        bool var_18;
        const wp::int32 var_19 = 0;
        bool var_20;
        bool var_21;
        const wp::int32 var_22 = 0;
        bool var_23;
        bool var_24;
        wp::int32 var_25;
        wp::int32 var_26;
        const wp::int32 var_27 = 5;
        wp::int32 var_28;
        const wp::int32 var_29 = 0;
        wp::int32 var_30;
        wp::int32* var_31;
        const wp::int32 var_32 = 0;
        bool var_33;
        wp::int32 var_34;
        const wp::int32 var_35 = 0;
        const wp::int32 var_36 = 1;
        wp::int32 var_37;
        const wp::int32 var_38 = 0;
        wp::int32* var_39;
        const wp::int32 var_40 = 0;
        wp::int32 var_41;
        const wp::int32 var_42 = 1;
        wp::int32* var_43;
        const wp::int32 var_44 = 1;
        wp::int32 var_45;
        const wp::int32 var_46 = 2;
        wp::int32* var_47;
        const wp::int32 var_48 = 2;
        wp::int32 var_49;
        const wp::int32 var_50 = 1;
        wp::int32 var_51;
        wp::int32* var_52;
        const wp::int32 var_53 = 0;
        bool var_54;
        wp::int32 var_55;
        const wp::int32 var_56 = 0;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        const wp::int32 var_59 = 0;
        wp::int32* var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        const wp::int32 var_63 = 1;
        wp::int32* var_64;
        const wp::int32 var_65 = 1;
        wp::int32 var_66;
        const wp::int32 var_67 = 2;
        wp::int32* var_68;
        const wp::int32 var_69 = 2;
        wp::int32 var_70;
        wp::int32 var_71;
        const wp::int32 var_72 = 2;
        wp::int32 var_73;
        wp::int32* var_74;
        const wp::int32 var_75 = 0;
        bool var_76;
        wp::int32 var_77;
        const wp::int32 var_78 = 0;
        const wp::int32 var_79 = 1;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        wp::int32* var_82;
        const wp::int32 var_83 = 0;
        wp::int32 var_84;
        const wp::int32 var_85 = 1;
        wp::int32* var_86;
        const wp::int32 var_87 = 1;
        wp::int32 var_88;
        const wp::int32 var_89 = 2;
        wp::int32* var_90;
        const wp::int32 var_91 = 2;
        wp::int32 var_92;
        wp::int32 var_93;
        const wp::int32 var_94 = 3;
        wp::int32 var_95;
        wp::int32* var_96;
        const wp::int32 var_97 = 0;
        bool var_98;
        wp::int32 var_99;
        const wp::int32 var_100 = 0;
        const wp::int32 var_101 = 1;
        wp::int32 var_102;
        const wp::int32 var_103 = 0;
        wp::int32* var_104;
        const wp::int32 var_105 = 0;
        wp::int32 var_106;
        const wp::int32 var_107 = 1;
        wp::int32* var_108;
        const wp::int32 var_109 = 1;
        wp::int32 var_110;
        const wp::int32 var_111 = 2;
        wp::int32* var_112;
        const wp::int32 var_113 = 2;
        wp::int32 var_114;
        wp::int32 var_115;
        const wp::int32 var_116 = 4;
        wp::int32 var_117;
        wp::int32* var_118;
        const wp::int32 var_119 = 0;
        bool var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 0;
        const wp::int32 var_123 = 1;
        wp::int32 var_124;
        const wp::int32 var_125 = 0;
        wp::int32* var_126;
        const wp::int32 var_127 = 0;
        wp::int32 var_128;
        const wp::int32 var_129 = 1;
        wp::int32* var_130;
        const wp::int32 var_131 = 1;
        wp::int32 var_132;
        const wp::int32 var_133 = 2;
        wp::int32* var_134;
        const wp::int32 var_135 = 2;
        wp::int32 var_136;
        wp::int32 var_137;
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
        bool adj_15 = {};
        wp::int32 adj_16 = {};
        bool adj_17 = {};
        bool adj_18 = {};
        wp::int32 adj_19 = {};
        bool adj_20 = {};
        bool adj_21 = {};
        wp::int32 adj_22 = {};
        bool adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        wp::int32 adj_26 = {};
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
        wp::int32 adj_39 = {};
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
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        wp::int32 adj_53 = {};
        bool adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
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
        wp::int32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::int32 adj_75 = {};
        bool adj_76 = {};
        wp::int32 adj_77 = {};
        wp::int32 adj_78 = {};
        wp::int32 adj_79 = {};
        wp::int32 adj_80 = {};
        wp::int32 adj_81 = {};
        wp::int32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::int32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::int32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::int32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
        bool adj_98 = {};
        wp::int32 adj_99 = {};
        wp::int32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::int32 adj_102 = {};
        wp::int32 adj_103 = {};
        wp::int32 adj_104 = {};
        wp::int32 adj_105 = {};
        wp::int32 adj_106 = {};
        wp::int32 adj_107 = {};
        wp::int32 adj_108 = {};
        wp::int32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::int32 adj_111 = {};
        wp::int32 adj_112 = {};
        wp::int32 adj_113 = {};
        wp::int32 adj_114 = {};
        wp::int32 adj_115 = {};
        wp::int32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        bool adj_120 = {};
        wp::int32 adj_121 = {};
        wp::int32 adj_122 = {};
        wp::int32 adj_123 = {};
        wp::int32 adj_124 = {};
        wp::int32 adj_125 = {};
        wp::int32 adj_126 = {};
        wp::int32 adj_127 = {};
        wp::int32 adj_128 = {};
        wp::int32 adj_129 = {};
        wp::int32 adj_130 = {};
        wp::int32 adj_131 = {};
        wp::int32 adj_132 = {};
        wp::int32 adj_133 = {};
        wp::int32 adj_134 = {};
        wp::int32 adj_135 = {};
        wp::int32 adj_136 = {};
        wp::int32 adj_137 = {};
        //---------
        // forward
        // def reemit_dirty_cube_slots_kernel(                                                    <L 598>
        // i = wp.tid()                                                                           <L 620>
        var_0 = builtin_tid1d();
        // if i >= dirty_count[0]:                                                                <L 621>
        var_2 = wp::address(var_dirty_count, var_1);
        var_4 = wp::load(var_2);
        var_3 = (var_0 >= var_4);
        if (var_3) {
            // return                                                                             <L 622>
            goto label0;
        }
        // cube_flat = dirty_cube_ids[i]                                                          <L 624>
        var_5 = wp::address(var_dirty_cube_ids, var_0);
        var_7 = wp::load(var_5);
        var_6 = wp::copy(var_7);
        // plane = ny_cells * nz_cells                                                            <L 625>
        var_8 = wp::mul(var_ny_cells, var_nz_cells);
        // cx = cube_flat / plane                                                                 <L 626>
        var_9 = wp::div(var_6, var_8);
        // rem = cube_flat - cx * plane                                                           <L 627>
        var_10 = wp::mul(var_9, var_8);
        var_11 = wp::sub(var_6, var_10);
        // cy = rem / nz_cells                                                                    <L 628>
        var_12 = wp::div(var_11, var_nz_cells);
        // cz = rem - cy * nz_cells                                                               <L 629>
        var_13 = wp::mul(var_12, var_nz_cells);
        var_14 = wp::sub(var_11, var_13);
        // if cx < 0 or cx >= nx_cells or cy < 0 or cy >= ny_cells or cz < 0 or cz >= nz_cells:       <L 630>
        var_17 = (var_9 < var_16);
        var_15 = var_17;
        if (!var_15) {
            var_18 = (var_9 >= var_nx_cells);
            var_15 = var_15 || var_18;
        }
        if (!var_15) {
            var_20 = (var_12 < var_19);
            var_15 = var_15 || var_20;
        }
        if (!var_15) {
            var_21 = (var_12 >= var_ny_cells);
            var_15 = var_15 || var_21;
        }
        if (!var_15) {
            var_23 = (var_14 < var_22);
            var_15 = var_15 || var_23;
        }
        if (!var_15) {
            var_24 = (var_14 >= var_nz_cells);
            var_15 = var_15 || var_24;
        }
        if (var_15) {
            // return                                                                             <L 631>
            goto label1;
        }
        // case = _compute_cube_case(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz)       <L 633>
        var_25 = _compute_cube_case_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_9, var_12, var_14);
        // cube_cases[cx, cy, cz] = case                                                          <L 634>
        // wp::array_store(var_cube_cases, var_9, var_12, var_14, var_25);
        // _write_cube_fixed_slots(                                                               <L 635>
        // cube_flat,                                                                             <L 636>
        // cx,                                                                                    <L 637>
        // cy,                                                                                    <L 638>
        // cz,                                                                                    <L 639>
        // case,                                                                                  <L 640>
        // grid_to_particle,                                                                      <L 641>
        // particle_flags,                                                                        <L 642>
        // corner_offsets,                                                                        <L 643>
        // edge_corners,                                                                          <L 644>
        // edge_base_dir,                                                                         <L 645>
        // case_triangles,                                                                        <L 646>
        // cube_tri_counts,                                                                       <L 647>
        // slot_tri_indices,                                                                      <L 648>
        // slot_active,                                                                           <L 649>
        // slot_to_compact,                                                                       <L 650>
        var_26 = _write_cube_fixed_slots_0(var_6, var_9, var_12, var_14, var_25, var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_case_triangles, var_cube_tri_counts, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
        // base = cube_flat * MC_MAX_TRIS_PER_CASE                                                <L 653>
        var_28 = wp::mul(var_6, var_27);
        // for local in range(MC_MAX_TRIS_PER_CASE):                                              <L 654>
        // slot = base + local                                                                    <L 655>
        var_30 = wp::add(var_28, var_29);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_31 = wp::address(var_slot_active, var_30);
        var_34 = wp::load(var_31);
        var_33 = (var_34 != var_32);
        if (var_33) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            // var_37 = wp::atomic_add(var_tri_count, var_35, var_36);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_39 = wp::address(var_slot_tri_indices, var_30, var_38);
            var_41 = wp::load(var_39);
            // wp::array_store(var_tri_indices, var_37, var_40, var_41);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_43 = wp::address(var_slot_tri_indices, var_30, var_42);
            var_45 = wp::load(var_43);
            // wp::array_store(var_tri_indices, var_37, var_44, var_45);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_47 = wp::address(var_slot_tri_indices, var_30, var_46);
            var_49 = wp::load(var_47);
            // wp::array_store(var_tri_indices, var_37, var_48, var_49);
            // slot_to_compact[slot] = dst                                                        <L 661>
            // wp::array_store(var_slot_to_compact, var_30, var_37);
            // compact_to_slot[dst] = slot                                                        <L 662>
            // wp::array_store(var_compact_to_slot, var_37, var_30);
        }
        // slot = base + local                                                                    <L 655>
        var_51 = wp::add(var_28, var_50);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_52 = wp::address(var_slot_active, var_51);
        var_55 = wp::load(var_52);
        var_54 = (var_55 != var_53);
        if (var_54) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            // var_58 = wp::atomic_add(var_tri_count, var_56, var_57);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_60 = wp::address(var_slot_tri_indices, var_51, var_59);
            var_62 = wp::load(var_60);
            // wp::array_store(var_tri_indices, var_58, var_61, var_62);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_64 = wp::address(var_slot_tri_indices, var_51, var_63);
            var_66 = wp::load(var_64);
            // wp::array_store(var_tri_indices, var_58, var_65, var_66);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_68 = wp::address(var_slot_tri_indices, var_51, var_67);
            var_70 = wp::load(var_68);
            // wp::array_store(var_tri_indices, var_58, var_69, var_70);
            // slot_to_compact[slot] = dst                                                        <L 661>
            // wp::array_store(var_slot_to_compact, var_51, var_58);
            // compact_to_slot[dst] = slot                                                        <L 662>
            // wp::array_store(var_compact_to_slot, var_58, var_51);
        }
        var_71 = wp::where(var_54, var_58, var_37);
        // slot = base + local                                                                    <L 655>
        var_73 = wp::add(var_28, var_72);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_74 = wp::address(var_slot_active, var_73);
        var_77 = wp::load(var_74);
        var_76 = (var_77 != var_75);
        if (var_76) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            // var_80 = wp::atomic_add(var_tri_count, var_78, var_79);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_82 = wp::address(var_slot_tri_indices, var_73, var_81);
            var_84 = wp::load(var_82);
            // wp::array_store(var_tri_indices, var_80, var_83, var_84);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_86 = wp::address(var_slot_tri_indices, var_73, var_85);
            var_88 = wp::load(var_86);
            // wp::array_store(var_tri_indices, var_80, var_87, var_88);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_90 = wp::address(var_slot_tri_indices, var_73, var_89);
            var_92 = wp::load(var_90);
            // wp::array_store(var_tri_indices, var_80, var_91, var_92);
            // slot_to_compact[slot] = dst                                                        <L 661>
            // wp::array_store(var_slot_to_compact, var_73, var_80);
            // compact_to_slot[dst] = slot                                                        <L 662>
            // wp::array_store(var_compact_to_slot, var_80, var_73);
        }
        var_93 = wp::where(var_76, var_80, var_71);
        // slot = base + local                                                                    <L 655>
        var_95 = wp::add(var_28, var_94);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_96 = wp::address(var_slot_active, var_95);
        var_99 = wp::load(var_96);
        var_98 = (var_99 != var_97);
        if (var_98) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            // var_102 = wp::atomic_add(var_tri_count, var_100, var_101);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_104 = wp::address(var_slot_tri_indices, var_95, var_103);
            var_106 = wp::load(var_104);
            // wp::array_store(var_tri_indices, var_102, var_105, var_106);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_108 = wp::address(var_slot_tri_indices, var_95, var_107);
            var_110 = wp::load(var_108);
            // wp::array_store(var_tri_indices, var_102, var_109, var_110);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_112 = wp::address(var_slot_tri_indices, var_95, var_111);
            var_114 = wp::load(var_112);
            // wp::array_store(var_tri_indices, var_102, var_113, var_114);
            // slot_to_compact[slot] = dst                                                        <L 661>
            // wp::array_store(var_slot_to_compact, var_95, var_102);
            // compact_to_slot[dst] = slot                                                        <L 662>
            // wp::array_store(var_compact_to_slot, var_102, var_95);
        }
        var_115 = wp::where(var_98, var_102, var_93);
        // slot = base + local                                                                    <L 655>
        var_117 = wp::add(var_28, var_116);
        // if slot_active[slot] != 0:                                                             <L 656>
        var_118 = wp::address(var_slot_active, var_117);
        var_121 = wp::load(var_118);
        var_120 = (var_121 != var_119);
        if (var_120) {
            // dst = wp.atomic_add(tri_count, 0, 1)                                               <L 657>
            // var_124 = wp::atomic_add(var_tri_count, var_122, var_123);
            // tri_indices[dst, 0] = slot_tri_indices[slot, 0]                                    <L 658>
            var_126 = wp::address(var_slot_tri_indices, var_117, var_125);
            var_128 = wp::load(var_126);
            // wp::array_store(var_tri_indices, var_124, var_127, var_128);
            // tri_indices[dst, 1] = slot_tri_indices[slot, 1]                                    <L 659>
            var_130 = wp::address(var_slot_tri_indices, var_117, var_129);
            var_132 = wp::load(var_130);
            // wp::array_store(var_tri_indices, var_124, var_131, var_132);
            // tri_indices[dst, 2] = slot_tri_indices[slot, 2]                                    <L 660>
            var_134 = wp::address(var_slot_tri_indices, var_117, var_133);
            var_136 = wp::load(var_134);
            // wp::array_store(var_tri_indices, var_124, var_135, var_136);
            // slot_to_compact[slot] = dst                                                        <L 661>
            // wp::array_store(var_slot_to_compact, var_117, var_124);
            // compact_to_slot[dst] = slot                                                        <L 662>
            // wp::array_store(var_compact_to_slot, var_124, var_117);
        }
        var_137 = wp::where(var_120, var_124, var_115);
        //---------
        // reverse
        wp::adj_where(var_120, var_124, var_115, adj_120, adj_124, adj_115, adj_137);
        if (var_120) {
            wp::adj_array_store(var_compact_to_slot, var_124, var_117, adj_compact_to_slot, adj_124, adj_117);
            // adj: compact_to_slot[dst] = slot                                                   <L 662>
            wp::adj_array_store(var_slot_to_compact, var_117, var_124, adj_slot_to_compact, adj_117, adj_124);
            // adj: slot_to_compact[slot] = dst                                                   <L 661>
            wp::adj_array_store(var_tri_indices, var_124, var_135, var_136, adj_tri_indices, adj_124, adj_135, adj_134);
            wp::adj_address(var_slot_tri_indices, var_117, var_133, adj_slot_tri_indices, adj_117, adj_133, adj_134);
            // adj: tri_indices[dst, 2] = slot_tri_indices[slot, 2]                               <L 660>
            wp::adj_array_store(var_tri_indices, var_124, var_131, var_132, adj_tri_indices, adj_124, adj_131, adj_130);
            wp::adj_address(var_slot_tri_indices, var_117, var_129, adj_slot_tri_indices, adj_117, adj_129, adj_130);
            // adj: tri_indices[dst, 1] = slot_tri_indices[slot, 1]                               <L 659>
            wp::adj_array_store(var_tri_indices, var_124, var_127, var_128, adj_tri_indices, adj_124, adj_127, adj_126);
            wp::adj_address(var_slot_tri_indices, var_117, var_125, adj_slot_tri_indices, adj_117, adj_125, adj_126);
            // adj: tri_indices[dst, 0] = slot_tri_indices[slot, 0]                               <L 658>
            wp::adj_atomic_add(var_tri_count, var_122, var_123, adj_tri_count, adj_122, adj_123, adj_124);
            // adj: dst = wp.atomic_add(tri_count, 0, 1)                                          <L 657>
        }
        wp::adj_address(var_slot_active, var_117, adj_slot_active, adj_117, adj_118);
        // adj: if slot_active[slot] != 0:                                                        <L 656>
        wp::adj_add(var_28, var_116, adj_28, adj_116, adj_117);
        // adj: slot = base + local                                                               <L 655>
        wp::adj_where(var_98, var_102, var_93, adj_98, adj_102, adj_93, adj_115);
        if (var_98) {
            wp::adj_array_store(var_compact_to_slot, var_102, var_95, adj_compact_to_slot, adj_102, adj_95);
            // adj: compact_to_slot[dst] = slot                                                   <L 662>
            wp::adj_array_store(var_slot_to_compact, var_95, var_102, adj_slot_to_compact, adj_95, adj_102);
            // adj: slot_to_compact[slot] = dst                                                   <L 661>
            wp::adj_array_store(var_tri_indices, var_102, var_113, var_114, adj_tri_indices, adj_102, adj_113, adj_112);
            wp::adj_address(var_slot_tri_indices, var_95, var_111, adj_slot_tri_indices, adj_95, adj_111, adj_112);
            // adj: tri_indices[dst, 2] = slot_tri_indices[slot, 2]                               <L 660>
            wp::adj_array_store(var_tri_indices, var_102, var_109, var_110, adj_tri_indices, adj_102, adj_109, adj_108);
            wp::adj_address(var_slot_tri_indices, var_95, var_107, adj_slot_tri_indices, adj_95, adj_107, adj_108);
            // adj: tri_indices[dst, 1] = slot_tri_indices[slot, 1]                               <L 659>
            wp::adj_array_store(var_tri_indices, var_102, var_105, var_106, adj_tri_indices, adj_102, adj_105, adj_104);
            wp::adj_address(var_slot_tri_indices, var_95, var_103, adj_slot_tri_indices, adj_95, adj_103, adj_104);
            // adj: tri_indices[dst, 0] = slot_tri_indices[slot, 0]                               <L 658>
            wp::adj_atomic_add(var_tri_count, var_100, var_101, adj_tri_count, adj_100, adj_101, adj_102);
            // adj: dst = wp.atomic_add(tri_count, 0, 1)                                          <L 657>
        }
        wp::adj_address(var_slot_active, var_95, adj_slot_active, adj_95, adj_96);
        // adj: if slot_active[slot] != 0:                                                        <L 656>
        wp::adj_add(var_28, var_94, adj_28, adj_94, adj_95);
        // adj: slot = base + local                                                               <L 655>
        wp::adj_where(var_76, var_80, var_71, adj_76, adj_80, adj_71, adj_93);
        if (var_76) {
            wp::adj_array_store(var_compact_to_slot, var_80, var_73, adj_compact_to_slot, adj_80, adj_73);
            // adj: compact_to_slot[dst] = slot                                                   <L 662>
            wp::adj_array_store(var_slot_to_compact, var_73, var_80, adj_slot_to_compact, adj_73, adj_80);
            // adj: slot_to_compact[slot] = dst                                                   <L 661>
            wp::adj_array_store(var_tri_indices, var_80, var_91, var_92, adj_tri_indices, adj_80, adj_91, adj_90);
            wp::adj_address(var_slot_tri_indices, var_73, var_89, adj_slot_tri_indices, adj_73, adj_89, adj_90);
            // adj: tri_indices[dst, 2] = slot_tri_indices[slot, 2]                               <L 660>
            wp::adj_array_store(var_tri_indices, var_80, var_87, var_88, adj_tri_indices, adj_80, adj_87, adj_86);
            wp::adj_address(var_slot_tri_indices, var_73, var_85, adj_slot_tri_indices, adj_73, adj_85, adj_86);
            // adj: tri_indices[dst, 1] = slot_tri_indices[slot, 1]                               <L 659>
            wp::adj_array_store(var_tri_indices, var_80, var_83, var_84, adj_tri_indices, adj_80, adj_83, adj_82);
            wp::adj_address(var_slot_tri_indices, var_73, var_81, adj_slot_tri_indices, adj_73, adj_81, adj_82);
            // adj: tri_indices[dst, 0] = slot_tri_indices[slot, 0]                               <L 658>
            wp::adj_atomic_add(var_tri_count, var_78, var_79, adj_tri_count, adj_78, adj_79, adj_80);
            // adj: dst = wp.atomic_add(tri_count, 0, 1)                                          <L 657>
        }
        wp::adj_address(var_slot_active, var_73, adj_slot_active, adj_73, adj_74);
        // adj: if slot_active[slot] != 0:                                                        <L 656>
        wp::adj_add(var_28, var_72, adj_28, adj_72, adj_73);
        // adj: slot = base + local                                                               <L 655>
        wp::adj_where(var_54, var_58, var_37, adj_54, adj_58, adj_37, adj_71);
        if (var_54) {
            wp::adj_array_store(var_compact_to_slot, var_58, var_51, adj_compact_to_slot, adj_58, adj_51);
            // adj: compact_to_slot[dst] = slot                                                   <L 662>
            wp::adj_array_store(var_slot_to_compact, var_51, var_58, adj_slot_to_compact, adj_51, adj_58);
            // adj: slot_to_compact[slot] = dst                                                   <L 661>
            wp::adj_array_store(var_tri_indices, var_58, var_69, var_70, adj_tri_indices, adj_58, adj_69, adj_68);
            wp::adj_address(var_slot_tri_indices, var_51, var_67, adj_slot_tri_indices, adj_51, adj_67, adj_68);
            // adj: tri_indices[dst, 2] = slot_tri_indices[slot, 2]                               <L 660>
            wp::adj_array_store(var_tri_indices, var_58, var_65, var_66, adj_tri_indices, adj_58, adj_65, adj_64);
            wp::adj_address(var_slot_tri_indices, var_51, var_63, adj_slot_tri_indices, adj_51, adj_63, adj_64);
            // adj: tri_indices[dst, 1] = slot_tri_indices[slot, 1]                               <L 659>
            wp::adj_array_store(var_tri_indices, var_58, var_61, var_62, adj_tri_indices, adj_58, adj_61, adj_60);
            wp::adj_address(var_slot_tri_indices, var_51, var_59, adj_slot_tri_indices, adj_51, adj_59, adj_60);
            // adj: tri_indices[dst, 0] = slot_tri_indices[slot, 0]                               <L 658>
            wp::adj_atomic_add(var_tri_count, var_56, var_57, adj_tri_count, adj_56, adj_57, adj_58);
            // adj: dst = wp.atomic_add(tri_count, 0, 1)                                          <L 657>
        }
        wp::adj_address(var_slot_active, var_51, adj_slot_active, adj_51, adj_52);
        // adj: if slot_active[slot] != 0:                                                        <L 656>
        wp::adj_add(var_28, var_50, adj_28, adj_50, adj_51);
        // adj: slot = base + local                                                               <L 655>
        if (var_33) {
            wp::adj_array_store(var_compact_to_slot, var_37, var_30, adj_compact_to_slot, adj_37, adj_30);
            // adj: compact_to_slot[dst] = slot                                                   <L 662>
            wp::adj_array_store(var_slot_to_compact, var_30, var_37, adj_slot_to_compact, adj_30, adj_37);
            // adj: slot_to_compact[slot] = dst                                                   <L 661>
            wp::adj_array_store(var_tri_indices, var_37, var_48, var_49, adj_tri_indices, adj_37, adj_48, adj_47);
            wp::adj_address(var_slot_tri_indices, var_30, var_46, adj_slot_tri_indices, adj_30, adj_46, adj_47);
            // adj: tri_indices[dst, 2] = slot_tri_indices[slot, 2]                               <L 660>
            wp::adj_array_store(var_tri_indices, var_37, var_44, var_45, adj_tri_indices, adj_37, adj_44, adj_43);
            wp::adj_address(var_slot_tri_indices, var_30, var_42, adj_slot_tri_indices, adj_30, adj_42, adj_43);
            // adj: tri_indices[dst, 1] = slot_tri_indices[slot, 1]                               <L 659>
            wp::adj_array_store(var_tri_indices, var_37, var_40, var_41, adj_tri_indices, adj_37, adj_40, adj_39);
            wp::adj_address(var_slot_tri_indices, var_30, var_38, adj_slot_tri_indices, adj_30, adj_38, adj_39);
            // adj: tri_indices[dst, 0] = slot_tri_indices[slot, 0]                               <L 658>
            wp::adj_atomic_add(var_tri_count, var_35, var_36, adj_tri_count, adj_35, adj_36, adj_37);
            // adj: dst = wp.atomic_add(tri_count, 0, 1)                                          <L 657>
        }
        wp::adj_address(var_slot_active, var_30, adj_slot_active, adj_30, adj_31);
        // adj: if slot_active[slot] != 0:                                                        <L 656>
        wp::adj_add(var_28, var_29, adj_28, adj_29, adj_30);
        // adj: slot = base + local                                                               <L 655>
        // adj: for local in range(MC_MAX_TRIS_PER_CASE):                                         <L 654>
        wp::adj_mul(var_6, var_27, adj_6, adj_27, adj_28);
        // adj: base = cube_flat * MC_MAX_TRIS_PER_CASE                                           <L 653>
        adj__write_cube_fixed_slots_0(var_6, var_9, var_12, var_14, var_25, var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_case_triangles, var_cube_tri_counts, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_6, adj_9, adj_12, adj_14, adj_25, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_case_triangles, adj_cube_tri_counts, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact, adj_26);
        // adj: slot_to_compact,                                                                  <L 650>
        // adj: slot_active,                                                                      <L 649>
        // adj: slot_tri_indices,                                                                 <L 648>
        // adj: cube_tri_counts,                                                                  <L 647>
        // adj: case_triangles,                                                                   <L 646>
        // adj: edge_base_dir,                                                                    <L 645>
        // adj: edge_corners,                                                                     <L 644>
        // adj: corner_offsets,                                                                   <L 643>
        // adj: particle_flags,                                                                   <L 642>
        // adj: grid_to_particle,                                                                 <L 641>
        // adj: case,                                                                             <L 640>
        // adj: cz,                                                                               <L 639>
        // adj: cy,                                                                               <L 638>
        // adj: cx,                                                                               <L 637>
        // adj: cube_flat,                                                                        <L 636>
        // adj: _write_cube_fixed_slots(                                                          <L 635>
        wp::adj_array_store(var_cube_cases, var_9, var_12, var_14, var_25, adj_cube_cases, adj_9, adj_12, adj_14, adj_25);
        // adj: cube_cases[cx, cy, cz] = case                                                     <L 634>
        adj__compute_cube_case_0(var_grid_to_particle, var_particle_flags, var_corner_offsets, var_9, var_12, var_14, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_9, adj_12, adj_14, adj_25);
        // adj: case = _compute_cube_case(grid_to_particle, particle_flags, corner_offsets, cx, cy, cz)  <L 633>
        if (var_15) {
            label1:;
            // adj: return                                                                        <L 631>
        }
        if (!var_15) {
        }
        if (!var_15) {
        }
        if (!var_15) {
        }
        if (!var_15) {
        }
        if (!var_15) {
        }
        // adj: if cx < 0 or cx >= nx_cells or cy < 0 or cy >= ny_cells or cz < 0 or cz >= nz_cells:  <L 630>
        wp::adj_sub(var_11, var_13, adj_11, adj_13, adj_14);
        wp::adj_mul(var_12, var_nz_cells, adj_12, adj_nz_cells, adj_13);
        // adj: cz = rem - cy * nz_cells                                                          <L 629>
        wp::adj_div(var_11, var_nz_cells, var_12, adj_11, adj_nz_cells, adj_12);
        // adj: cy = rem / nz_cells                                                               <L 628>
        wp::adj_sub(var_6, var_10, adj_6, adj_10, adj_11);
        wp::adj_mul(var_9, var_8, adj_9, adj_8, adj_10);
        // adj: rem = cube_flat - cx * plane                                                      <L 627>
        wp::adj_div(var_6, var_8, var_9, adj_6, adj_8, adj_9);
        // adj: cx = cube_flat / plane                                                            <L 626>
        wp::adj_mul(var_ny_cells, var_nz_cells, adj_ny_cells, adj_nz_cells, adj_8);
        // adj: plane = ny_cells * nz_cells                                                       <L 625>
        wp::adj_copy(var_7, adj_5, adj_6);
        wp::adj_address(var_dirty_cube_ids, var_0, adj_dirty_cube_ids, adj_0, adj_5);
        // adj: cube_flat = dirty_cube_ids[i]                                                     <L 624>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 622>
        }
        wp::adj_address(var_dirty_count, var_1, adj_dirty_count, adj_1, adj_2);
        // adj: if i >= dirty_count[0]:                                                           <L 621>
        // adj: i = wp.tid()                                                                      <L 620>
        // adj: def reemit_dirty_cube_slots_kernel(                                               <L 598>
        continue;
    }
}



extern "C" __global__ void emit_fixed_slots_kernel_5273bb5e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact)
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
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        //---------
        // forward
        // def emit_fixed_slots_kernel(                                                           <L 412>
        // cx, cy, cz = wp.tid()                                                                  <L 428>
        builtin_tid3d(var_0, var_1, var_2);
        // cube_flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                              <L 429>
        var_3 = _cube_flat_id_0(var_0, var_1, var_2, var_ny_cells, var_nz_cells);
        // _write_cube_fixed_slots(                                                               <L 430>
        // cube_flat,                                                                             <L 431>
        // cx,                                                                                    <L 432>
        // cy,                                                                                    <L 433>
        // cz,                                                                                    <L 434>
        // cube_cases[cx, cy, cz],                                                                <L 435>
        var_4 = wp::address(var_cube_cases, var_0, var_1, var_2);
        // grid_to_particle,                                                                      <L 436>
        // particle_flags,                                                                        <L 437>
        // corner_offsets,                                                                        <L 438>
        // edge_corners,                                                                          <L 439>
        // edge_base_dir,                                                                         <L 440>
        // case_triangles,                                                                        <L 441>
        // cube_tri_counts,                                                                       <L 442>
        // slot_tri_indices,                                                                      <L 443>
        // slot_active,                                                                           <L 444>
        // slot_to_compact,                                                                       <L 445>
        var_6 = wp::load(var_4);
        var_5 = _write_cube_fixed_slots_0(var_3, var_0, var_1, var_2, var_6, var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_case_triangles, var_cube_tri_counts, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
    }
}



extern "C" __global__ void emit_fixed_slots_kernel_5273bb5e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_grid_to_particle,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_cube_cases,
    wp::array_t<wp::int32> var_corner_offsets,
    wp::array_t<wp::int32> var_edge_corners,
    wp::array_t<wp::int32> var_edge_base_dir,
    wp::array_t<wp::int32> var_case_triangles,
    wp::int32 var_ny_cells,
    wp::int32 var_nz_cells,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> adj_grid_to_particle,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_cube_cases,
    wp::array_t<wp::int32> adj_corner_offsets,
    wp::array_t<wp::int32> adj_edge_corners,
    wp::array_t<wp::int32> adj_edge_base_dir,
    wp::array_t<wp::int32> adj_case_triangles,
    wp::int32 adj_ny_cells,
    wp::int32 adj_nz_cells,
    wp::array_t<wp::int32> adj_cube_tri_counts,
    wp::array_t<wp::int32> adj_slot_tri_indices,
    wp::array_t<wp::int32> adj_slot_active,
    wp::array_t<wp::int32> adj_slot_to_compact)
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
        wp::int32* var_4;
        wp::int32 var_5;
        wp::int32 var_6;
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::int32 adj_1 = {};
        wp::int32 adj_2 = {};
        wp::int32 adj_3 = {};
        wp::int32 adj_4 = {};
        wp::int32 adj_5 = {};
        wp::int32 adj_6 = {};
        //---------
        // forward
        // def emit_fixed_slots_kernel(                                                           <L 412>
        // cx, cy, cz = wp.tid()                                                                  <L 428>
        builtin_tid3d(var_0, var_1, var_2);
        // cube_flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                              <L 429>
        var_3 = _cube_flat_id_0(var_0, var_1, var_2, var_ny_cells, var_nz_cells);
        // _write_cube_fixed_slots(                                                               <L 430>
        // cube_flat,                                                                             <L 431>
        // cx,                                                                                    <L 432>
        // cy,                                                                                    <L 433>
        // cz,                                                                                    <L 434>
        // cube_cases[cx, cy, cz],                                                                <L 435>
        var_4 = wp::address(var_cube_cases, var_0, var_1, var_2);
        // grid_to_particle,                                                                      <L 436>
        // particle_flags,                                                                        <L 437>
        // corner_offsets,                                                                        <L 438>
        // edge_corners,                                                                          <L 439>
        // edge_base_dir,                                                                         <L 440>
        // case_triangles,                                                                        <L 441>
        // cube_tri_counts,                                                                       <L 442>
        // slot_tri_indices,                                                                      <L 443>
        // slot_active,                                                                           <L 444>
        // slot_to_compact,                                                                       <L 445>
        var_6 = wp::load(var_4);
        var_5 = _write_cube_fixed_slots_0(var_3, var_0, var_1, var_2, var_6, var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_case_triangles, var_cube_tri_counts, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
        //---------
        // reverse
        adj__write_cube_fixed_slots_0(var_3, var_0, var_1, var_2, var_6, var_grid_to_particle, var_particle_flags, var_corner_offsets, var_edge_corners, var_edge_base_dir, var_case_triangles, var_cube_tri_counts, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_3, adj_0, adj_1, adj_2, adj_4, adj_grid_to_particle, adj_particle_flags, adj_corner_offsets, adj_edge_corners, adj_edge_base_dir, adj_case_triangles, adj_cube_tri_counts, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact, adj_5);
        // adj: slot_to_compact,                                                                  <L 445>
        // adj: slot_active,                                                                      <L 444>
        // adj: slot_tri_indices,                                                                 <L 443>
        // adj: cube_tri_counts,                                                                  <L 442>
        // adj: case_triangles,                                                                   <L 441>
        // adj: edge_base_dir,                                                                    <L 440>
        // adj: edge_corners,                                                                     <L 439>
        // adj: corner_offsets,                                                                   <L 438>
        // adj: particle_flags,                                                                   <L 437>
        // adj: grid_to_particle,                                                                 <L 436>
        wp::adj_address(var_cube_cases, var_0, var_1, var_2, adj_cube_cases, adj_0, adj_1, adj_2, adj_4);
        // adj: cube_cases[cx, cy, cz],                                                           <L 435>
        // adj: cz,                                                                               <L 434>
        // adj: cy,                                                                               <L 433>
        // adj: cx,                                                                               <L 432>
        // adj: cube_flat,                                                                        <L 431>
        // adj: _write_cube_fixed_slots(                                                          <L 430>
        adj__cube_flat_id_0(var_0, var_1, var_2, var_ny_cells, var_nz_cells, adj_0, adj_1, adj_2, adj_ny_cells, adj_nz_cells, adj_3);
        // adj: cube_flat = _cube_flat_id(cx, cy, cz, ny_cells, nz_cells)                         <L 429>
        // adj: cx, cy, cz = wp.tid()                                                             <L 428>
        // adj: def emit_fixed_slots_kernel(                                                      <L 412>
        continue;
    }
}



extern "C" __global__ void compact_fixed_slots_kernel_a2dbade5_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> var_compact_to_slot,
    wp::int32 var_max_triangles,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices)
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
        const wp::int32 var_5 = -1;
        const wp::int32 var_6 = 0;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = -1;
        const wp::int32 var_11 = 0;
        wp::int32* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32* var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        wp::int32* var_20;
        const wp::int32 var_21 = 2;
        wp::int32 var_22;
        //---------
        // forward
        // def compact_fixed_slots_kernel(                                                        <L 450>
        // slot = wp.tid()                                                                        <L 460>
        var_0 = builtin_tid1d();
        // if slot_active[slot] == 0:                                                             <L 461>
        var_1 = wp::address(var_slot_active, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // slot_to_compact[slot] = -1                                                         <L 462>
            wp::array_store(var_slot_to_compact, var_0, var_5);
            // return                                                                             <L 463>
            continue;
        }
        // idx = wp.atomic_add(tri_count, 0, 1)                                                   <L 465>
        var_8 = wp::atomic_add(var_tri_count, var_6, var_7);
        // if idx >= max_triangles:                                                               <L 466>
        var_9 = (var_8 >= var_max_triangles);
        if (var_9) {
            // slot_to_compact[slot] = -1                                                         <L 467>
            wp::array_store(var_slot_to_compact, var_0, var_10);
            // return                                                                             <L 468>
            continue;
        }
        // tri_indices[idx, 0] = slot_tri_indices[slot, 0]                                        <L 469>
        var_12 = wp::address(var_slot_tri_indices, var_0, var_11);
        var_14 = wp::load(var_12);
        wp::array_store(var_tri_indices, var_8, var_13, var_14);
        // tri_indices[idx, 1] = slot_tri_indices[slot, 1]                                        <L 470>
        var_16 = wp::address(var_slot_tri_indices, var_0, var_15);
        var_18 = wp::load(var_16);
        wp::array_store(var_tri_indices, var_8, var_17, var_18);
        // tri_indices[idx, 2] = slot_tri_indices[slot, 2]                                        <L 471>
        var_20 = wp::address(var_slot_tri_indices, var_0, var_19);
        var_22 = wp::load(var_20);
        wp::array_store(var_tri_indices, var_8, var_21, var_22);
        // slot_to_compact[slot] = idx                                                            <L 472>
        wp::array_store(var_slot_to_compact, var_0, var_8);
        // compact_to_slot[idx] = slot                                                            <L 473>
        wp::array_store(var_compact_to_slot, var_8, var_0);
    }
}



extern "C" __global__ void compact_fixed_slots_kernel_a2dbade5_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> var_compact_to_slot,
    wp::int32 var_max_triangles,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> adj_slot_tri_indices,
    wp::array_t<wp::int32> adj_slot_active,
    wp::array_t<wp::int32> adj_slot_to_compact,
    wp::array_t<wp::int32> adj_compact_to_slot,
    wp::int32 adj_max_triangles,
    wp::array_t<wp::int32> adj_tri_count,
    wp::array_t<wp::int32> adj_tri_indices)
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
        const wp::int32 var_5 = -1;
        const wp::int32 var_6 = 0;
        const wp::int32 var_7 = 1;
        wp::int32 var_8;
        bool var_9;
        const wp::int32 var_10 = -1;
        const wp::int32 var_11 = 0;
        wp::int32* var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        const wp::int32 var_15 = 1;
        wp::int32* var_16;
        const wp::int32 var_17 = 1;
        wp::int32 var_18;
        const wp::int32 var_19 = 2;
        wp::int32* var_20;
        const wp::int32 var_21 = 2;
        wp::int32 var_22;
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
        wp::int32 adj_18 = {};
        wp::int32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::int32 adj_22 = {};
        //---------
        // forward
        // def compact_fixed_slots_kernel(                                                        <L 450>
        // slot = wp.tid()                                                                        <L 460>
        var_0 = builtin_tid1d();
        // if slot_active[slot] == 0:                                                             <L 461>
        var_1 = wp::address(var_slot_active, var_0);
        var_4 = wp::load(var_1);
        var_3 = (var_4 == var_2);
        if (var_3) {
            // slot_to_compact[slot] = -1                                                         <L 462>
            // wp::array_store(var_slot_to_compact, var_0, var_5);
            // return                                                                             <L 463>
            goto label0;
        }
        // idx = wp.atomic_add(tri_count, 0, 1)                                                   <L 465>
        // var_8 = wp::atomic_add(var_tri_count, var_6, var_7);
        // if idx >= max_triangles:                                                               <L 466>
        var_9 = (var_8 >= var_max_triangles);
        if (var_9) {
            // slot_to_compact[slot] = -1                                                         <L 467>
            // wp::array_store(var_slot_to_compact, var_0, var_10);
            // return                                                                             <L 468>
            goto label1;
        }
        // tri_indices[idx, 0] = slot_tri_indices[slot, 0]                                        <L 469>
        var_12 = wp::address(var_slot_tri_indices, var_0, var_11);
        var_14 = wp::load(var_12);
        // wp::array_store(var_tri_indices, var_8, var_13, var_14);
        // tri_indices[idx, 1] = slot_tri_indices[slot, 1]                                        <L 470>
        var_16 = wp::address(var_slot_tri_indices, var_0, var_15);
        var_18 = wp::load(var_16);
        // wp::array_store(var_tri_indices, var_8, var_17, var_18);
        // tri_indices[idx, 2] = slot_tri_indices[slot, 2]                                        <L 471>
        var_20 = wp::address(var_slot_tri_indices, var_0, var_19);
        var_22 = wp::load(var_20);
        // wp::array_store(var_tri_indices, var_8, var_21, var_22);
        // slot_to_compact[slot] = idx                                                            <L 472>
        // wp::array_store(var_slot_to_compact, var_0, var_8);
        // compact_to_slot[idx] = slot                                                            <L 473>
        // wp::array_store(var_compact_to_slot, var_8, var_0);
        //---------
        // reverse
        wp::adj_array_store(var_compact_to_slot, var_8, var_0, adj_compact_to_slot, adj_8, adj_0);
        // adj: compact_to_slot[idx] = slot                                                       <L 473>
        wp::adj_array_store(var_slot_to_compact, var_0, var_8, adj_slot_to_compact, adj_0, adj_8);
        // adj: slot_to_compact[slot] = idx                                                       <L 472>
        wp::adj_array_store(var_tri_indices, var_8, var_21, var_22, adj_tri_indices, adj_8, adj_21, adj_20);
        wp::adj_address(var_slot_tri_indices, var_0, var_19, adj_slot_tri_indices, adj_0, adj_19, adj_20);
        // adj: tri_indices[idx, 2] = slot_tri_indices[slot, 2]                                   <L 471>
        wp::adj_array_store(var_tri_indices, var_8, var_17, var_18, adj_tri_indices, adj_8, adj_17, adj_16);
        wp::adj_address(var_slot_tri_indices, var_0, var_15, adj_slot_tri_indices, adj_0, adj_15, adj_16);
        // adj: tri_indices[idx, 1] = slot_tri_indices[slot, 1]                                   <L 470>
        wp::adj_array_store(var_tri_indices, var_8, var_13, var_14, adj_tri_indices, adj_8, adj_13, adj_12);
        wp::adj_address(var_slot_tri_indices, var_0, var_11, adj_slot_tri_indices, adj_0, adj_11, adj_12);
        // adj: tri_indices[idx, 0] = slot_tri_indices[slot, 0]                                   <L 469>
        if (var_9) {
            label1:;
            // adj: return                                                                        <L 468>
            wp::adj_array_store(var_slot_to_compact, var_0, var_10, adj_slot_to_compact, adj_0, adj_10);
            // adj: slot_to_compact[slot] = -1                                                    <L 467>
        }
        // adj: if idx >= max_triangles:                                                          <L 466>
        wp::adj_atomic_add(var_tri_count, var_6, var_7, adj_tri_count, adj_6, adj_7, adj_8);
        // adj: idx = wp.atomic_add(tri_count, 0, 1)                                              <L 465>
        if (var_3) {
            label0:;
            // adj: return                                                                        <L 463>
            wp::adj_array_store(var_slot_to_compact, var_0, var_5, adj_slot_to_compact, adj_0, adj_5);
            // adj: slot_to_compact[slot] = -1                                                    <L 462>
        }
        wp::adj_address(var_slot_active, var_0, adj_slot_active, adj_0, adj_1);
        // adj: if slot_active[slot] == 0:                                                        <L 461>
        // adj: slot = wp.tid()                                                                   <L 460>
        // adj: def compact_fixed_slots_kernel(                                                   <L 450>
        continue;
    }
}



extern "C" __global__ void remove_dirty_cube_slots_kernel_104641af_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dirty_cube_ids,
    wp::array_t<wp::int32> var_dirty_count,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> var_compact_to_slot,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices)
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
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        const wp::int32 var_11 = 5;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        bool var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 0;
        bool var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 0;
        wp::int32* var_28;
        bool var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        wp::int32* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32 var_35;
        bool var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        wp::int32* var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32* var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        const wp::int32 var_48 = 2;
        wp::int32* var_49;
        const wp::int32 var_50 = 2;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = -1;
        const wp::int32 var_55 = 0;
        const wp::int32 var_56 = 1;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
        //---------
        // forward
        // def remove_dirty_cube_slots_kernel(                                                    <L 557>
        // n = dirty_count[0]                                                                     <L 569>
        var_1 = wp::address(var_dirty_count, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // i = int(0)                                                                             <L 570>
        var_5 = wp::int(var_4);
        // while i < n:                                                                           <L 571>
        start_while_0:;
        var_6 = (var_5 < var_2);
        if ((var_6) == false) goto end_while_0;
            // cube_flat = dirty_cube_ids[i]                                                      <L 572>
            var_7 = wp::address(var_dirty_cube_ids, var_5);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // cube_tri_counts[cube_flat] = 0                                                     <L 573>
            wp::array_store(var_cube_tri_counts, var_8, var_10);
            // base = cube_flat * MC_MAX_TRIS_PER_CASE                                            <L 574>
            var_12 = wp::mul(var_8, var_11);
            // local = int(0)                                                                     <L 575>
            var_14 = wp::int(var_13);
            // while local < MC_MAX_TRIS_PER_CASE:                                                <L 576>
        start_while_2:;
            var_15 = (var_14 < var_11);
        if ((var_15) == false) goto end_while_2;
                // slot = base + local                                                            <L 577>
                var_16 = wp::add(var_12, var_14);
                // if slot_active[slot] != 0:                                                     <L 578>
                var_17 = wp::address(var_slot_active, var_16);
                var_20 = wp::load(var_17);
                var_19 = (var_20 != var_18);
                if (var_19) {
                    // compact_idx = slot_to_compact[slot]                                        <L 579>
                    var_21 = wp::address(var_slot_to_compact, var_16);
                    var_23 = wp::load(var_21);
                    var_22 = wp::copy(var_23);
                    // if compact_idx >= 0 and compact_idx < tri_count[0]:                        <L 580>
                    var_26 = (var_22 >= var_25);
                    var_24 = var_26;
                    if (var_24) {
                        var_28 = wp::address(var_tri_count, var_27);
                        var_30 = wp::load(var_28);
                        var_29 = (var_22 < var_30);
                        var_24 = var_24 && var_29;
                    }
                    if (var_24) {
                        // last_idx = tri_count[0] - 1                                            <L 581>
                        var_32 = wp::address(var_tri_count, var_31);
                        var_35 = wp::load(var_32);
                        var_34 = wp::sub(var_35, var_33);
                        // if compact_idx != last_idx:                                            <L 582>
                        var_36 = (var_22 != var_34);
                        if (var_36) {
                            // moved_slot = compact_to_slot[last_idx]                             <L 583>
                            var_37 = wp::address(var_compact_to_slot, var_34);
                            var_39 = wp::load(var_37);
                            var_38 = wp::copy(var_39);
                            // tri_indices[compact_idx, 0] = tri_indices[last_idx, 0]             <L 584>
                            var_41 = wp::address(var_tri_indices, var_34, var_40);
                            var_43 = wp::load(var_41);
                            wp::array_store(var_tri_indices, var_22, var_42, var_43);
                            // tri_indices[compact_idx, 1] = tri_indices[last_idx, 1]             <L 585>
                            var_45 = wp::address(var_tri_indices, var_34, var_44);
                            var_47 = wp::load(var_45);
                            wp::array_store(var_tri_indices, var_22, var_46, var_47);
                            // tri_indices[compact_idx, 2] = tri_indices[last_idx, 2]             <L 586>
                            var_49 = wp::address(var_tri_indices, var_34, var_48);
                            var_51 = wp::load(var_49);
                            wp::array_store(var_tri_indices, var_22, var_50, var_51);
                            // compact_to_slot[compact_idx] = moved_slot                          <L 587>
                            wp::array_store(var_compact_to_slot, var_22, var_38);
                            // if moved_slot >= 0:                                                <L 588>
                            var_53 = (var_38 >= var_52);
                            if (var_53) {
                                // slot_to_compact[moved_slot] = compact_idx                      <L 589>
                                wp::array_store(var_slot_to_compact, var_38, var_22);
                            }
                        }
                        // compact_to_slot[last_idx] = -1                                         <L 590>
                        wp::array_store(var_compact_to_slot, var_34, var_54);
                        // tri_count[0] = last_idx                                                <L 591>
                        wp::array_store(var_tri_count, var_55, var_34);
                    }
                    // _clear_fixed_slot(slot, slot_tri_indices, slot_active, slot_to_compact)       <L 592>
                    _clear_fixed_slot_0(var_16, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
                }
                // local = local + 1                                                              <L 593>
                var_57 = wp::add(var_14, var_56);
                wp::assign(var_14, var_57);
        goto start_while_2;
        end_while_2:;
            // i = i + 1                                                                          <L 594>
            var_59 = wp::add(var_5, var_58);
            wp::assign(var_5, var_59);
        goto start_while_0;
        end_while_0:;
    }
}



extern "C" __global__ void remove_dirty_cube_slots_kernel_104641af_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_dirty_cube_ids,
    wp::array_t<wp::int32> var_dirty_count,
    wp::array_t<wp::int32> var_cube_tri_counts,
    wp::array_t<wp::int32> var_slot_tri_indices,
    wp::array_t<wp::int32> var_slot_active,
    wp::array_t<wp::int32> var_slot_to_compact,
    wp::array_t<wp::int32> var_compact_to_slot,
    wp::array_t<wp::int32> var_tri_count,
    wp::array_t<wp::int32> var_tri_indices,
    wp::array_t<wp::int32> adj_dirty_cube_ids,
    wp::array_t<wp::int32> adj_dirty_count,
    wp::array_t<wp::int32> adj_cube_tri_counts,
    wp::array_t<wp::int32> adj_slot_tri_indices,
    wp::array_t<wp::int32> adj_slot_active,
    wp::array_t<wp::int32> adj_slot_to_compact,
    wp::array_t<wp::int32> adj_compact_to_slot,
    wp::array_t<wp::int32> adj_tri_count,
    wp::array_t<wp::int32> adj_tri_indices)
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
        const wp::int32 var_4 = 0;
        wp::int32 var_5;
        bool var_6;
        wp::int32* var_7;
        wp::int32 var_8;
        wp::int32 var_9;
        const wp::int32 var_10 = 0;
        const wp::int32 var_11 = 5;
        wp::int32 var_12;
        const wp::int32 var_13 = 0;
        wp::int32 var_14;
        bool var_15;
        wp::int32 var_16;
        wp::int32* var_17;
        const wp::int32 var_18 = 0;
        bool var_19;
        wp::int32 var_20;
        wp::int32* var_21;
        wp::int32 var_22;
        wp::int32 var_23;
        bool var_24;
        const wp::int32 var_25 = 0;
        bool var_26;
        const wp::int32 var_27 = 0;
        wp::int32* var_28;
        bool var_29;
        wp::int32 var_30;
        const wp::int32 var_31 = 0;
        wp::int32* var_32;
        const wp::int32 var_33 = 1;
        wp::int32 var_34;
        wp::int32 var_35;
        bool var_36;
        wp::int32* var_37;
        wp::int32 var_38;
        wp::int32 var_39;
        const wp::int32 var_40 = 0;
        wp::int32* var_41;
        const wp::int32 var_42 = 0;
        wp::int32 var_43;
        const wp::int32 var_44 = 1;
        wp::int32* var_45;
        const wp::int32 var_46 = 1;
        wp::int32 var_47;
        const wp::int32 var_48 = 2;
        wp::int32* var_49;
        const wp::int32 var_50 = 2;
        wp::int32 var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        const wp::int32 var_54 = -1;
        const wp::int32 var_55 = 0;
        const wp::int32 var_56 = 1;
        wp::int32 var_57;
        const wp::int32 var_58 = 1;
        wp::int32 var_59;
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
        wp::int32 adj_23 = {};
        bool adj_24 = {};
        wp::int32 adj_25 = {};
        bool adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        bool adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
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
        wp::int32 adj_43 = {};
        wp::int32 adj_44 = {};
        wp::int32 adj_45 = {};
        wp::int32 adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        wp::int32 adj_51 = {};
        wp::int32 adj_52 = {};
        bool adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        //---------
        // forward
        // def remove_dirty_cube_slots_kernel(                                                    <L 557>
        // n = dirty_count[0]                                                                     <L 569>
        var_1 = wp::address(var_dirty_count, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::copy(var_3);
        // i = int(0)                                                                             <L 570>
        var_5 = wp::int(var_4);
        // while i < n:                                                                           <L 571>
        //---------
        // reverse
        start_while_0:;
        var_6 = (var_5 < var_2);
        if ((var_6) == false) goto end_while_0;
        adj_7 = {};
        adj_8 = {};
        adj_9 = {};
        adj_10 = {};
        adj_11 = {};
        adj_12 = {};
        adj_13 = {};
        adj_14 = {};
        adj_58 = {};
        adj_59 = {};
            // cube_flat = dirty_cube_ids[i]                                                      <L 572>
            var_7 = wp::address(var_dirty_cube_ids, var_5);
            var_9 = wp::load(var_7);
            var_8 = wp::copy(var_9);
            // cube_tri_counts[cube_flat] = 0                                                     <L 573>
            // wp::array_store(var_cube_tri_counts, var_8, var_10);
            // base = cube_flat * MC_MAX_TRIS_PER_CASE                                            <L 574>
            var_12 = wp::mul(var_8, var_11);
            // local = int(0)                                                                     <L 575>
            var_14 = wp::int(var_13);
            // while local < MC_MAX_TRIS_PER_CASE:                                                <L 576>
            // i = i + 1                                                                          <L 594>
            var_59 = wp::add(var_5, var_58);
            wp::assign(var_5, var_59);
            wp::adj_assign(var_5, var_59, adj_5, adj_59);
            wp::adj_add(var_5, var_58, adj_5, adj_58, adj_59);
            // adj: i = i + 1                                                                     <L 594>
        start_while_2:;
            var_15 = (var_14 < var_11);
        if ((var_15) == false) goto end_while_2;
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
                // slot = base + local                                                            <L 577>
                var_16 = wp::add(var_12, var_14);
                // if slot_active[slot] != 0:                                                     <L 578>
                var_17 = wp::address(var_slot_active, var_16);
                var_20 = wp::load(var_17);
                var_19 = (var_20 != var_18);
                if (var_19) {
                    // compact_idx = slot_to_compact[slot]                                        <L 579>
                    var_21 = wp::address(var_slot_to_compact, var_16);
                    var_23 = wp::load(var_21);
                    var_22 = wp::copy(var_23);
                    // if compact_idx >= 0 and compact_idx < tri_count[0]:                        <L 580>
                    var_26 = (var_22 >= var_25);
                    var_24 = var_26;
                    if (var_24) {
                        var_28 = wp::address(var_tri_count, var_27);
                        var_30 = wp::load(var_28);
                        var_29 = (var_22 < var_30);
                        var_24 = var_24 && var_29;
                    }
                    if (var_24) {
                        // last_idx = tri_count[0] - 1                                            <L 581>
                        var_32 = wp::address(var_tri_count, var_31);
                        var_35 = wp::load(var_32);
                        var_34 = wp::sub(var_35, var_33);
                        // if compact_idx != last_idx:                                            <L 582>
                        var_36 = (var_22 != var_34);
                        if (var_36) {
                            // moved_slot = compact_to_slot[last_idx]                             <L 583>
                            var_37 = wp::address(var_compact_to_slot, var_34);
                            var_39 = wp::load(var_37);
                            var_38 = wp::copy(var_39);
                            // tri_indices[compact_idx, 0] = tri_indices[last_idx, 0]             <L 584>
                            var_41 = wp::address(var_tri_indices, var_34, var_40);
                            var_43 = wp::load(var_41);
                            // wp::array_store(var_tri_indices, var_22, var_42, var_43);
                            // tri_indices[compact_idx, 1] = tri_indices[last_idx, 1]             <L 585>
                            var_45 = wp::address(var_tri_indices, var_34, var_44);
                            var_47 = wp::load(var_45);
                            // wp::array_store(var_tri_indices, var_22, var_46, var_47);
                            // tri_indices[compact_idx, 2] = tri_indices[last_idx, 2]             <L 586>
                            var_49 = wp::address(var_tri_indices, var_34, var_48);
                            var_51 = wp::load(var_49);
                            // wp::array_store(var_tri_indices, var_22, var_50, var_51);
                            // compact_to_slot[compact_idx] = moved_slot                          <L 587>
                            // wp::array_store(var_compact_to_slot, var_22, var_38);
                            // if moved_slot >= 0:                                                <L 588>
                            var_53 = (var_38 >= var_52);
                            if (var_53) {
                                // slot_to_compact[moved_slot] = compact_idx                      <L 589>
                                // wp::array_store(var_slot_to_compact, var_38, var_22);
                            }
                        }
                        // compact_to_slot[last_idx] = -1                                         <L 590>
                        // wp::array_store(var_compact_to_slot, var_34, var_54);
                        // tri_count[0] = last_idx                                                <L 591>
                        // wp::array_store(var_tri_count, var_55, var_34);
                    }
                    // _clear_fixed_slot(slot, slot_tri_indices, slot_active, slot_to_compact)       <L 592>
                    _clear_fixed_slot_0(var_16, var_slot_tri_indices, var_slot_active, var_slot_to_compact);
                }
                // local = local + 1                                                              <L 593>
                var_57 = wp::add(var_14, var_56);
                wp::assign(var_14, var_57);
                wp::adj_assign(var_14, var_57, adj_14, adj_57);
                wp::adj_add(var_14, var_56, adj_14, adj_56, adj_57);
                // adj: local = local + 1                                                         <L 593>
                if (var_19) {
                    adj__clear_fixed_slot_0(var_16, var_slot_tri_indices, var_slot_active, var_slot_to_compact, adj_16, adj_slot_tri_indices, adj_slot_active, adj_slot_to_compact);
                    // adj: _clear_fixed_slot(slot, slot_tri_indices, slot_active, slot_to_compact)  <L 592>
                    if (var_24) {
                        wp::adj_array_store(var_tri_count, var_55, var_34, adj_tri_count, adj_55, adj_34);
                        // adj: tri_count[0] = last_idx                                           <L 591>
                        wp::adj_array_store(var_compact_to_slot, var_34, var_54, adj_compact_to_slot, adj_34, adj_54);
                        // adj: compact_to_slot[last_idx] = -1                                    <L 590>
                        if (var_36) {
                            if (var_53) {
                                wp::adj_array_store(var_slot_to_compact, var_38, var_22, adj_slot_to_compact, adj_38, adj_22);
                                // adj: slot_to_compact[moved_slot] = compact_idx                 <L 589>
                            }
                            // adj: if moved_slot >= 0:                                           <L 588>
                            wp::adj_array_store(var_compact_to_slot, var_22, var_38, adj_compact_to_slot, adj_22, adj_38);
                            // adj: compact_to_slot[compact_idx] = moved_slot                     <L 587>
                            wp::adj_array_store(var_tri_indices, var_22, var_50, var_51, adj_tri_indices, adj_22, adj_50, adj_49);
                            wp::adj_address(var_tri_indices, var_34, var_48, adj_tri_indices, adj_34, adj_48, adj_49);
                            // adj: tri_indices[compact_idx, 2] = tri_indices[last_idx, 2]        <L 586>
                            wp::adj_array_store(var_tri_indices, var_22, var_46, var_47, adj_tri_indices, adj_22, adj_46, adj_45);
                            wp::adj_address(var_tri_indices, var_34, var_44, adj_tri_indices, adj_34, adj_44, adj_45);
                            // adj: tri_indices[compact_idx, 1] = tri_indices[last_idx, 1]        <L 585>
                            wp::adj_array_store(var_tri_indices, var_22, var_42, var_43, adj_tri_indices, adj_22, adj_42, adj_41);
                            wp::adj_address(var_tri_indices, var_34, var_40, adj_tri_indices, adj_34, adj_40, adj_41);
                            // adj: tri_indices[compact_idx, 0] = tri_indices[last_idx, 0]        <L 584>
                            wp::adj_copy(var_39, adj_37, adj_38);
                            wp::adj_address(var_compact_to_slot, var_34, adj_compact_to_slot, adj_34, adj_37);
                            // adj: moved_slot = compact_to_slot[last_idx]                        <L 583>
                        }
                        // adj: if compact_idx != last_idx:                                       <L 582>
                        wp::adj_sub(var_35, var_33, adj_32, adj_33, adj_34);
                        wp::adj_address(var_tri_count, var_31, adj_tri_count, adj_31, adj_32);
                        // adj: last_idx = tri_count[0] - 1                                       <L 581>
                    }
                    if (var_24) {
                        wp::adj_address(var_tri_count, var_27, adj_tri_count, adj_27, adj_28);
                    }
                    // adj: if compact_idx >= 0 and compact_idx < tri_count[0]:                   <L 580>
                    wp::adj_copy(var_23, adj_21, adj_22);
                    wp::adj_address(var_slot_to_compact, var_16, adj_slot_to_compact, adj_16, adj_21);
                    // adj: compact_idx = slot_to_compact[slot]                                   <L 579>
                }
                wp::adj_address(var_slot_active, var_16, adj_slot_active, adj_16, adj_17);
                // adj: if slot_active[slot] != 0:                                                <L 578>
                wp::adj_add(var_12, var_14, adj_12, adj_14, adj_16);
                // adj: slot = base + local                                                       <L 577>
        goto start_while_2;
        end_while_2:;
            // adj: while local < MC_MAX_TRIS_PER_CASE:                                           <L 576>
            wp::adj_int(var_13, adj_13, adj_14);
            // adj: local = int(0)                                                                <L 575>
            wp::adj_mul(var_8, var_11, adj_8, adj_11, adj_12);
            // adj: base = cube_flat * MC_MAX_TRIS_PER_CASE                                       <L 574>
            wp::adj_array_store(var_cube_tri_counts, var_8, var_10, adj_cube_tri_counts, adj_8, adj_10);
            // adj: cube_tri_counts[cube_flat] = 0                                                <L 573>
            wp::adj_copy(var_9, adj_7, adj_8);
            wp::adj_address(var_dirty_cube_ids, var_5, adj_dirty_cube_ids, adj_5, adj_7);
            // adj: cube_flat = dirty_cube_ids[i]                                                 <L 572>
        goto start_while_0;
        end_while_0:;
        // adj: while i < n:                                                                      <L 571>
        wp::adj_int(var_4, adj_4, adj_5);
        // adj: i = int(0)                                                                        <L 570>
        wp::adj_copy(var_3, adj_1, adj_2);
        wp::adj_address(var_dirty_count, var_0, adj_dirty_count, adj_0, adj_1);
        // adj: n = dirty_count[0]                                                                <L 569>
        // adj: def remove_dirty_cube_slots_kernel(                                               <L 557>
        continue;
    }
}

