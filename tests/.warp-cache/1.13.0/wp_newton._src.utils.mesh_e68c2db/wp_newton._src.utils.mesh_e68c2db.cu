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



extern "C" __global__ void normalize_vertex_normals_cf8dd499_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals)
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
        //---------
        // forward
        // def normalize_vertex_normals(normals: wp.array[wp.vec3]):                              <L 39>
        // tid = wp.tid()                                                                         <L 41>
        var_0 = builtin_tid1d();
        // normals[tid] = wp.normalize(normals[tid])                                              <L 42>
        var_1 = wp::address(var_normals, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::normalize(var_3);
        wp::array_store(var_normals, var_0, var_2);
    }
}



extern "C" __global__ void normalize_vertex_normals_cf8dd499_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normals)
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
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::vec_t<3, wp::float32> adj_1 = {};
        wp::vec_t<3, wp::float32> adj_2 = {};
        wp::vec_t<3, wp::float32> adj_3 = {};
        //---------
        // forward
        // def normalize_vertex_normals(normals: wp.array[wp.vec3]):                              <L 39>
        // tid = wp.tid()                                                                         <L 41>
        var_0 = builtin_tid1d();
        // normals[tid] = wp.normalize(normals[tid])                                              <L 42>
        var_1 = wp::address(var_normals, var_0);
        var_3 = wp::load(var_1);
        var_2 = wp::normalize(var_3);
        // wp::array_store(var_normals, var_0, var_2);
        //---------
        // reverse
        wp::adj_array_store(var_normals, var_0, var_2, adj_normals, adj_0, adj_2);
        wp::adj_normalize(var_3, var_2, adj_1, adj_2);
        wp::adj_address(var_normals, var_0, adj_normals, adj_0, adj_1);
        // adj: normals[tid] = wp.normalize(normals[tid])                                         <L 42>
        // adj: tid = wp.tid()                                                                    <L 41>
        // adj: def normalize_vertex_normals(normals: wp.array[wp.vec3]):                         <L 39>
        continue;
    }
}



extern "C" __global__ void solidify_mesh_kernel_ea083d06_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_thickness,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_vertices,
    wp::array_t<wp::int32> var_out_indices)
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
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::float32* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::float32 var_28;
        wp::float32* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        const wp::int32 var_41 = 2;
        wp::int32 var_42;
        const wp::int32 var_43 = 2;
        wp::int32 var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 2;
        wp::int32 var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 2;
        wp::int32 var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        const wp::int32 var_59 = 8;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        const wp::int32 var_66 = 1;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        const wp::int32 var_69 = 2;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        const wp::int32 var_72 = 0;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        const wp::int32 var_75 = 1;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        const wp::int32 var_78 = 2;
        const wp::int32 var_79 = 2;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = 2;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        const wp::int32 var_85 = 2;
        wp::int32 var_86;
        const wp::int32 var_87 = 2;
        const wp::int32 var_88 = 3;
        wp::int32 var_89;
        const wp::int32 var_90 = 0;
        const wp::int32 var_91 = 3;
        wp::int32 var_92;
        const wp::int32 var_93 = 1;
        const wp::int32 var_94 = 3;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        const wp::int32 var_97 = 4;
        wp::int32 var_98;
        const wp::int32 var_99 = 0;
        const wp::int32 var_100 = 4;
        wp::int32 var_101;
        const wp::int32 var_102 = 1;
        const wp::int32 var_103 = 4;
        wp::int32 var_104;
        const wp::int32 var_105 = 2;
        const wp::int32 var_106 = 5;
        wp::int32 var_107;
        const wp::int32 var_108 = 0;
        const wp::int32 var_109 = 5;
        wp::int32 var_110;
        const wp::int32 var_111 = 1;
        const wp::int32 var_112 = 5;
        wp::int32 var_113;
        const wp::int32 var_114 = 2;
        const wp::int32 var_115 = 6;
        wp::int32 var_116;
        const wp::int32 var_117 = 0;
        const wp::int32 var_118 = 6;
        wp::int32 var_119;
        const wp::int32 var_120 = 1;
        const wp::int32 var_121 = 6;
        wp::int32 var_122;
        const wp::int32 var_123 = 2;
        const wp::int32 var_124 = 7;
        wp::int32 var_125;
        const wp::int32 var_126 = 0;
        const wp::int32 var_127 = 7;
        wp::int32 var_128;
        const wp::int32 var_129 = 1;
        const wp::int32 var_130 = 7;
        wp::int32 var_131;
        const wp::int32 var_132 = 2;
        //---------
        // forward
        // def solidify_mesh_kernel(                                                              <L 1274>
        // tid = wp.tid()                                                                         <L 1298>
        var_0 = builtin_tid1d();
        // i = indices[tid, 0]                                                                    <L 1299>
        var_2 = wp::address(var_indices, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // j = indices[tid, 1]                                                                    <L 1300>
        var_6 = wp::address(var_indices, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // k = indices[tid, 2]                                                                    <L 1301>
        var_10 = wp::address(var_indices, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // vi = vertices[i]                                                                       <L 1303>
        var_13 = wp::address(var_vertices, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // vj = vertices[j]                                                                       <L 1304>
        var_16 = wp::address(var_vertices, var_7);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // vk = vertices[k]                                                                       <L 1305>
        var_19 = wp::address(var_vertices, var_11);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // normal = wp.normalize(wp.cross(vj - vi, vk - vi))                                      <L 1307>
        var_22 = wp::sub(var_17, var_14);
        var_23 = wp::sub(var_20, var_14);
        var_24 = wp::cross(var_22, var_23);
        var_25 = wp::normalize(var_24);
        // ti = normal * thickness[i]                                                             <L 1308>
        var_26 = wp::address(var_thickness, var_3);
        var_28 = wp::load(var_26);
        var_27 = wp::mul(var_25, var_28);
        // tj = normal * thickness[j]                                                             <L 1309>
        var_29 = wp::address(var_thickness, var_7);
        var_31 = wp::load(var_29);
        var_30 = wp::mul(var_25, var_31);
        // tk = normal * thickness[k]                                                             <L 1310>
        var_32 = wp::address(var_thickness, var_11);
        var_34 = wp::load(var_32);
        var_33 = wp::mul(var_25, var_34);
        // vi0 = vi + ti                                                                          <L 1313>
        var_35 = wp::add(var_14, var_27);
        // vi1 = vi - ti                                                                          <L 1314>
        var_36 = wp::sub(var_14, var_27);
        // vj0 = vj + tj                                                                          <L 1315>
        var_37 = wp::add(var_17, var_30);
        // vj1 = vj - tj                                                                          <L 1316>
        var_38 = wp::sub(var_17, var_30);
        // vk0 = vk + tk                                                                          <L 1317>
        var_39 = wp::add(var_20, var_33);
        // vk1 = vk - tk                                                                          <L 1318>
        var_40 = wp::sub(var_20, var_33);
        // i0 = i * 2                                                                             <L 1320>
        var_42 = wp::mul(var_3, var_41);
        // i1 = i * 2 + 1                                                                         <L 1321>
        var_44 = wp::mul(var_3, var_43);
        var_46 = wp::add(var_44, var_45);
        // j0 = j * 2                                                                             <L 1322>
        var_48 = wp::mul(var_7, var_47);
        // j1 = j * 2 + 1                                                                         <L 1323>
        var_50 = wp::mul(var_7, var_49);
        var_52 = wp::add(var_50, var_51);
        // k0 = k * 2                                                                             <L 1324>
        var_54 = wp::mul(var_11, var_53);
        // k1 = k * 2 + 1                                                                         <L 1325>
        var_56 = wp::mul(var_11, var_55);
        var_58 = wp::add(var_56, var_57);
        // out_vertices[i0] = vi0                                                                 <L 1327>
        wp::array_store(var_out_vertices, var_42, var_35);
        // out_vertices[i1] = vi1                                                                 <L 1328>
        wp::array_store(var_out_vertices, var_46, var_36);
        // out_vertices[j0] = vj0                                                                 <L 1329>
        wp::array_store(var_out_vertices, var_48, var_37);
        // out_vertices[j1] = vj1                                                                 <L 1330>
        wp::array_store(var_out_vertices, var_52, var_38);
        // out_vertices[k0] = vk0                                                                 <L 1331>
        wp::array_store(var_out_vertices, var_54, var_39);
        // out_vertices[k1] = vk1                                                                 <L 1332>
        wp::array_store(var_out_vertices, var_58, var_40);
        // oid = tid * 8                                                                          <L 1334>
        var_60 = wp::mul(var_0, var_59);
        // out_indices[oid + 0, 0] = i0                                                           <L 1335>
        var_62 = wp::add(var_60, var_61);
        wp::array_store(var_out_indices, var_62, var_63, var_42);
        // out_indices[oid + 0, 1] = j0                                                           <L 1336>
        var_65 = wp::add(var_60, var_64);
        wp::array_store(var_out_indices, var_65, var_66, var_48);
        // out_indices[oid + 0, 2] = k0                                                           <L 1337>
        var_68 = wp::add(var_60, var_67);
        wp::array_store(var_out_indices, var_68, var_69, var_54);
        // out_indices[oid + 1, 0] = j0                                                           <L 1338>
        var_71 = wp::add(var_60, var_70);
        wp::array_store(var_out_indices, var_71, var_72, var_48);
        // out_indices[oid + 1, 1] = k1                                                           <L 1339>
        var_74 = wp::add(var_60, var_73);
        wp::array_store(var_out_indices, var_74, var_75, var_58);
        // out_indices[oid + 1, 2] = k0                                                           <L 1340>
        var_77 = wp::add(var_60, var_76);
        wp::array_store(var_out_indices, var_77, var_78, var_54);
        // out_indices[oid + 2, 0] = j0                                                           <L 1341>
        var_80 = wp::add(var_60, var_79);
        wp::array_store(var_out_indices, var_80, var_81, var_48);
        // out_indices[oid + 2, 1] = j1                                                           <L 1342>
        var_83 = wp::add(var_60, var_82);
        wp::array_store(var_out_indices, var_83, var_84, var_52);
        // out_indices[oid + 2, 2] = k1                                                           <L 1343>
        var_86 = wp::add(var_60, var_85);
        wp::array_store(var_out_indices, var_86, var_87, var_58);
        // out_indices[oid + 3, 0] = j0                                                           <L 1344>
        var_89 = wp::add(var_60, var_88);
        wp::array_store(var_out_indices, var_89, var_90, var_48);
        // out_indices[oid + 3, 1] = i1                                                           <L 1345>
        var_92 = wp::add(var_60, var_91);
        wp::array_store(var_out_indices, var_92, var_93, var_46);
        // out_indices[oid + 3, 2] = j1                                                           <L 1346>
        var_95 = wp::add(var_60, var_94);
        wp::array_store(var_out_indices, var_95, var_96, var_52);
        // out_indices[oid + 4, 0] = j0                                                           <L 1347>
        var_98 = wp::add(var_60, var_97);
        wp::array_store(var_out_indices, var_98, var_99, var_48);
        // out_indices[oid + 4, 1] = i0                                                           <L 1348>
        var_101 = wp::add(var_60, var_100);
        wp::array_store(var_out_indices, var_101, var_102, var_42);
        // out_indices[oid + 4, 2] = i1                                                           <L 1349>
        var_104 = wp::add(var_60, var_103);
        wp::array_store(var_out_indices, var_104, var_105, var_46);
        // out_indices[oid + 5, 0] = j1                                                           <L 1350>
        var_107 = wp::add(var_60, var_106);
        wp::array_store(var_out_indices, var_107, var_108, var_52);
        // out_indices[oid + 5, 1] = i1                                                           <L 1351>
        var_110 = wp::add(var_60, var_109);
        wp::array_store(var_out_indices, var_110, var_111, var_46);
        // out_indices[oid + 5, 2] = k1                                                           <L 1352>
        var_113 = wp::add(var_60, var_112);
        wp::array_store(var_out_indices, var_113, var_114, var_58);
        // out_indices[oid + 6, 0] = i1                                                           <L 1353>
        var_116 = wp::add(var_60, var_115);
        wp::array_store(var_out_indices, var_116, var_117, var_46);
        // out_indices[oid + 6, 1] = i0                                                           <L 1354>
        var_119 = wp::add(var_60, var_118);
        wp::array_store(var_out_indices, var_119, var_120, var_42);
        // out_indices[oid + 6, 2] = k0                                                           <L 1355>
        var_122 = wp::add(var_60, var_121);
        wp::array_store(var_out_indices, var_122, var_123, var_54);
        // out_indices[oid + 7, 0] = i1                                                           <L 1356>
        var_125 = wp::add(var_60, var_124);
        wp::array_store(var_out_indices, var_125, var_126, var_46);
        // out_indices[oid + 7, 1] = k0                                                           <L 1357>
        var_128 = wp::add(var_60, var_127);
        wp::array_store(var_out_indices, var_128, var_129, var_54);
        // out_indices[oid + 7, 2] = k1                                                           <L 1358>
        var_131 = wp::add(var_60, var_130);
        wp::array_store(var_out_indices, var_131, var_132, var_58);
    }
}



extern "C" __global__ void solidify_mesh_kernel_ea083d06_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_vertices,
    wp::array_t<wp::float32> var_thickness,
    wp::array_t<wp::vec_t<3, wp::float32>> var_out_vertices,
    wp::array_t<wp::int32> var_out_indices,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_vertices,
    wp::array_t<wp::float32> adj_thickness,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_out_vertices,
    wp::array_t<wp::int32> adj_out_indices)
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
        wp::vec_t<3, wp::float32>* var_13;
        wp::vec_t<3, wp::float32> var_14;
        wp::vec_t<3, wp::float32> var_15;
        wp::vec_t<3, wp::float32>* var_16;
        wp::vec_t<3, wp::float32> var_17;
        wp::vec_t<3, wp::float32> var_18;
        wp::vec_t<3, wp::float32>* var_19;
        wp::vec_t<3, wp::float32> var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32> var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::float32* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::float32 var_28;
        wp::float32* var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::float32 var_31;
        wp::float32* var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::float32 var_34;
        wp::vec_t<3, wp::float32> var_35;
        wp::vec_t<3, wp::float32> var_36;
        wp::vec_t<3, wp::float32> var_37;
        wp::vec_t<3, wp::float32> var_38;
        wp::vec_t<3, wp::float32> var_39;
        wp::vec_t<3, wp::float32> var_40;
        const wp::int32 var_41 = 2;
        wp::int32 var_42;
        const wp::int32 var_43 = 2;
        wp::int32 var_44;
        const wp::int32 var_45 = 1;
        wp::int32 var_46;
        const wp::int32 var_47 = 2;
        wp::int32 var_48;
        const wp::int32 var_49 = 2;
        wp::int32 var_50;
        const wp::int32 var_51 = 1;
        wp::int32 var_52;
        const wp::int32 var_53 = 2;
        wp::int32 var_54;
        const wp::int32 var_55 = 2;
        wp::int32 var_56;
        const wp::int32 var_57 = 1;
        wp::int32 var_58;
        const wp::int32 var_59 = 8;
        wp::int32 var_60;
        const wp::int32 var_61 = 0;
        wp::int32 var_62;
        const wp::int32 var_63 = 0;
        const wp::int32 var_64 = 0;
        wp::int32 var_65;
        const wp::int32 var_66 = 1;
        const wp::int32 var_67 = 0;
        wp::int32 var_68;
        const wp::int32 var_69 = 2;
        const wp::int32 var_70 = 1;
        wp::int32 var_71;
        const wp::int32 var_72 = 0;
        const wp::int32 var_73 = 1;
        wp::int32 var_74;
        const wp::int32 var_75 = 1;
        const wp::int32 var_76 = 1;
        wp::int32 var_77;
        const wp::int32 var_78 = 2;
        const wp::int32 var_79 = 2;
        wp::int32 var_80;
        const wp::int32 var_81 = 0;
        const wp::int32 var_82 = 2;
        wp::int32 var_83;
        const wp::int32 var_84 = 1;
        const wp::int32 var_85 = 2;
        wp::int32 var_86;
        const wp::int32 var_87 = 2;
        const wp::int32 var_88 = 3;
        wp::int32 var_89;
        const wp::int32 var_90 = 0;
        const wp::int32 var_91 = 3;
        wp::int32 var_92;
        const wp::int32 var_93 = 1;
        const wp::int32 var_94 = 3;
        wp::int32 var_95;
        const wp::int32 var_96 = 2;
        const wp::int32 var_97 = 4;
        wp::int32 var_98;
        const wp::int32 var_99 = 0;
        const wp::int32 var_100 = 4;
        wp::int32 var_101;
        const wp::int32 var_102 = 1;
        const wp::int32 var_103 = 4;
        wp::int32 var_104;
        const wp::int32 var_105 = 2;
        const wp::int32 var_106 = 5;
        wp::int32 var_107;
        const wp::int32 var_108 = 0;
        const wp::int32 var_109 = 5;
        wp::int32 var_110;
        const wp::int32 var_111 = 1;
        const wp::int32 var_112 = 5;
        wp::int32 var_113;
        const wp::int32 var_114 = 2;
        const wp::int32 var_115 = 6;
        wp::int32 var_116;
        const wp::int32 var_117 = 0;
        const wp::int32 var_118 = 6;
        wp::int32 var_119;
        const wp::int32 var_120 = 1;
        const wp::int32 var_121 = 6;
        wp::int32 var_122;
        const wp::int32 var_123 = 2;
        const wp::int32 var_124 = 7;
        wp::int32 var_125;
        const wp::int32 var_126 = 0;
        const wp::int32 var_127 = 7;
        wp::int32 var_128;
        const wp::int32 var_129 = 1;
        const wp::int32 var_130 = 7;
        wp::int32 var_131;
        const wp::int32 var_132 = 2;
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
        wp::vec_t<3, wp::float32> adj_13 = {};
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
        wp::float32 adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::float32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::float32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::float32 adj_34 = {};
        wp::vec_t<3, wp::float32> adj_35 = {};
        wp::vec_t<3, wp::float32> adj_36 = {};
        wp::vec_t<3, wp::float32> adj_37 = {};
        wp::vec_t<3, wp::float32> adj_38 = {};
        wp::vec_t<3, wp::float32> adj_39 = {};
        wp::vec_t<3, wp::float32> adj_40 = {};
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
        wp::int32 adj_72 = {};
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
        wp::int32 adj_90 = {};
        wp::int32 adj_91 = {};
        wp::int32 adj_92 = {};
        wp::int32 adj_93 = {};
        wp::int32 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::int32 adj_96 = {};
        wp::int32 adj_97 = {};
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
        //---------
        // forward
        // def solidify_mesh_kernel(                                                              <L 1274>
        // tid = wp.tid()                                                                         <L 1298>
        var_0 = builtin_tid1d();
        // i = indices[tid, 0]                                                                    <L 1299>
        var_2 = wp::address(var_indices, var_0, var_1);
        var_4 = wp::load(var_2);
        var_3 = wp::copy(var_4);
        // j = indices[tid, 1]                                                                    <L 1300>
        var_6 = wp::address(var_indices, var_0, var_5);
        var_8 = wp::load(var_6);
        var_7 = wp::copy(var_8);
        // k = indices[tid, 2]                                                                    <L 1301>
        var_10 = wp::address(var_indices, var_0, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // vi = vertices[i]                                                                       <L 1303>
        var_13 = wp::address(var_vertices, var_3);
        var_15 = wp::load(var_13);
        var_14 = wp::copy(var_15);
        // vj = vertices[j]                                                                       <L 1304>
        var_16 = wp::address(var_vertices, var_7);
        var_18 = wp::load(var_16);
        var_17 = wp::copy(var_18);
        // vk = vertices[k]                                                                       <L 1305>
        var_19 = wp::address(var_vertices, var_11);
        var_21 = wp::load(var_19);
        var_20 = wp::copy(var_21);
        // normal = wp.normalize(wp.cross(vj - vi, vk - vi))                                      <L 1307>
        var_22 = wp::sub(var_17, var_14);
        var_23 = wp::sub(var_20, var_14);
        var_24 = wp::cross(var_22, var_23);
        var_25 = wp::normalize(var_24);
        // ti = normal * thickness[i]                                                             <L 1308>
        var_26 = wp::address(var_thickness, var_3);
        var_28 = wp::load(var_26);
        var_27 = wp::mul(var_25, var_28);
        // tj = normal * thickness[j]                                                             <L 1309>
        var_29 = wp::address(var_thickness, var_7);
        var_31 = wp::load(var_29);
        var_30 = wp::mul(var_25, var_31);
        // tk = normal * thickness[k]                                                             <L 1310>
        var_32 = wp::address(var_thickness, var_11);
        var_34 = wp::load(var_32);
        var_33 = wp::mul(var_25, var_34);
        // vi0 = vi + ti                                                                          <L 1313>
        var_35 = wp::add(var_14, var_27);
        // vi1 = vi - ti                                                                          <L 1314>
        var_36 = wp::sub(var_14, var_27);
        // vj0 = vj + tj                                                                          <L 1315>
        var_37 = wp::add(var_17, var_30);
        // vj1 = vj - tj                                                                          <L 1316>
        var_38 = wp::sub(var_17, var_30);
        // vk0 = vk + tk                                                                          <L 1317>
        var_39 = wp::add(var_20, var_33);
        // vk1 = vk - tk                                                                          <L 1318>
        var_40 = wp::sub(var_20, var_33);
        // i0 = i * 2                                                                             <L 1320>
        var_42 = wp::mul(var_3, var_41);
        // i1 = i * 2 + 1                                                                         <L 1321>
        var_44 = wp::mul(var_3, var_43);
        var_46 = wp::add(var_44, var_45);
        // j0 = j * 2                                                                             <L 1322>
        var_48 = wp::mul(var_7, var_47);
        // j1 = j * 2 + 1                                                                         <L 1323>
        var_50 = wp::mul(var_7, var_49);
        var_52 = wp::add(var_50, var_51);
        // k0 = k * 2                                                                             <L 1324>
        var_54 = wp::mul(var_11, var_53);
        // k1 = k * 2 + 1                                                                         <L 1325>
        var_56 = wp::mul(var_11, var_55);
        var_58 = wp::add(var_56, var_57);
        // out_vertices[i0] = vi0                                                                 <L 1327>
        // wp::array_store(var_out_vertices, var_42, var_35);
        // out_vertices[i1] = vi1                                                                 <L 1328>
        // wp::array_store(var_out_vertices, var_46, var_36);
        // out_vertices[j0] = vj0                                                                 <L 1329>
        // wp::array_store(var_out_vertices, var_48, var_37);
        // out_vertices[j1] = vj1                                                                 <L 1330>
        // wp::array_store(var_out_vertices, var_52, var_38);
        // out_vertices[k0] = vk0                                                                 <L 1331>
        // wp::array_store(var_out_vertices, var_54, var_39);
        // out_vertices[k1] = vk1                                                                 <L 1332>
        // wp::array_store(var_out_vertices, var_58, var_40);
        // oid = tid * 8                                                                          <L 1334>
        var_60 = wp::mul(var_0, var_59);
        // out_indices[oid + 0, 0] = i0                                                           <L 1335>
        var_62 = wp::add(var_60, var_61);
        // wp::array_store(var_out_indices, var_62, var_63, var_42);
        // out_indices[oid + 0, 1] = j0                                                           <L 1336>
        var_65 = wp::add(var_60, var_64);
        // wp::array_store(var_out_indices, var_65, var_66, var_48);
        // out_indices[oid + 0, 2] = k0                                                           <L 1337>
        var_68 = wp::add(var_60, var_67);
        // wp::array_store(var_out_indices, var_68, var_69, var_54);
        // out_indices[oid + 1, 0] = j0                                                           <L 1338>
        var_71 = wp::add(var_60, var_70);
        // wp::array_store(var_out_indices, var_71, var_72, var_48);
        // out_indices[oid + 1, 1] = k1                                                           <L 1339>
        var_74 = wp::add(var_60, var_73);
        // wp::array_store(var_out_indices, var_74, var_75, var_58);
        // out_indices[oid + 1, 2] = k0                                                           <L 1340>
        var_77 = wp::add(var_60, var_76);
        // wp::array_store(var_out_indices, var_77, var_78, var_54);
        // out_indices[oid + 2, 0] = j0                                                           <L 1341>
        var_80 = wp::add(var_60, var_79);
        // wp::array_store(var_out_indices, var_80, var_81, var_48);
        // out_indices[oid + 2, 1] = j1                                                           <L 1342>
        var_83 = wp::add(var_60, var_82);
        // wp::array_store(var_out_indices, var_83, var_84, var_52);
        // out_indices[oid + 2, 2] = k1                                                           <L 1343>
        var_86 = wp::add(var_60, var_85);
        // wp::array_store(var_out_indices, var_86, var_87, var_58);
        // out_indices[oid + 3, 0] = j0                                                           <L 1344>
        var_89 = wp::add(var_60, var_88);
        // wp::array_store(var_out_indices, var_89, var_90, var_48);
        // out_indices[oid + 3, 1] = i1                                                           <L 1345>
        var_92 = wp::add(var_60, var_91);
        // wp::array_store(var_out_indices, var_92, var_93, var_46);
        // out_indices[oid + 3, 2] = j1                                                           <L 1346>
        var_95 = wp::add(var_60, var_94);
        // wp::array_store(var_out_indices, var_95, var_96, var_52);
        // out_indices[oid + 4, 0] = j0                                                           <L 1347>
        var_98 = wp::add(var_60, var_97);
        // wp::array_store(var_out_indices, var_98, var_99, var_48);
        // out_indices[oid + 4, 1] = i0                                                           <L 1348>
        var_101 = wp::add(var_60, var_100);
        // wp::array_store(var_out_indices, var_101, var_102, var_42);
        // out_indices[oid + 4, 2] = i1                                                           <L 1349>
        var_104 = wp::add(var_60, var_103);
        // wp::array_store(var_out_indices, var_104, var_105, var_46);
        // out_indices[oid + 5, 0] = j1                                                           <L 1350>
        var_107 = wp::add(var_60, var_106);
        // wp::array_store(var_out_indices, var_107, var_108, var_52);
        // out_indices[oid + 5, 1] = i1                                                           <L 1351>
        var_110 = wp::add(var_60, var_109);
        // wp::array_store(var_out_indices, var_110, var_111, var_46);
        // out_indices[oid + 5, 2] = k1                                                           <L 1352>
        var_113 = wp::add(var_60, var_112);
        // wp::array_store(var_out_indices, var_113, var_114, var_58);
        // out_indices[oid + 6, 0] = i1                                                           <L 1353>
        var_116 = wp::add(var_60, var_115);
        // wp::array_store(var_out_indices, var_116, var_117, var_46);
        // out_indices[oid + 6, 1] = i0                                                           <L 1354>
        var_119 = wp::add(var_60, var_118);
        // wp::array_store(var_out_indices, var_119, var_120, var_42);
        // out_indices[oid + 6, 2] = k0                                                           <L 1355>
        var_122 = wp::add(var_60, var_121);
        // wp::array_store(var_out_indices, var_122, var_123, var_54);
        // out_indices[oid + 7, 0] = i1                                                           <L 1356>
        var_125 = wp::add(var_60, var_124);
        // wp::array_store(var_out_indices, var_125, var_126, var_46);
        // out_indices[oid + 7, 1] = k0                                                           <L 1357>
        var_128 = wp::add(var_60, var_127);
        // wp::array_store(var_out_indices, var_128, var_129, var_54);
        // out_indices[oid + 7, 2] = k1                                                           <L 1358>
        var_131 = wp::add(var_60, var_130);
        // wp::array_store(var_out_indices, var_131, var_132, var_58);
        //---------
        // reverse
        wp::adj_array_store(var_out_indices, var_131, var_132, var_58, adj_out_indices, adj_131, adj_132, adj_58);
        wp::adj_add(var_60, var_130, adj_60, adj_130, adj_131);
        // adj: out_indices[oid + 7, 2] = k1                                                      <L 1358>
        wp::adj_array_store(var_out_indices, var_128, var_129, var_54, adj_out_indices, adj_128, adj_129, adj_54);
        wp::adj_add(var_60, var_127, adj_60, adj_127, adj_128);
        // adj: out_indices[oid + 7, 1] = k0                                                      <L 1357>
        wp::adj_array_store(var_out_indices, var_125, var_126, var_46, adj_out_indices, adj_125, adj_126, adj_46);
        wp::adj_add(var_60, var_124, adj_60, adj_124, adj_125);
        // adj: out_indices[oid + 7, 0] = i1                                                      <L 1356>
        wp::adj_array_store(var_out_indices, var_122, var_123, var_54, adj_out_indices, adj_122, adj_123, adj_54);
        wp::adj_add(var_60, var_121, adj_60, adj_121, adj_122);
        // adj: out_indices[oid + 6, 2] = k0                                                      <L 1355>
        wp::adj_array_store(var_out_indices, var_119, var_120, var_42, adj_out_indices, adj_119, adj_120, adj_42);
        wp::adj_add(var_60, var_118, adj_60, adj_118, adj_119);
        // adj: out_indices[oid + 6, 1] = i0                                                      <L 1354>
        wp::adj_array_store(var_out_indices, var_116, var_117, var_46, adj_out_indices, adj_116, adj_117, adj_46);
        wp::adj_add(var_60, var_115, adj_60, adj_115, adj_116);
        // adj: out_indices[oid + 6, 0] = i1                                                      <L 1353>
        wp::adj_array_store(var_out_indices, var_113, var_114, var_58, adj_out_indices, adj_113, adj_114, adj_58);
        wp::adj_add(var_60, var_112, adj_60, adj_112, adj_113);
        // adj: out_indices[oid + 5, 2] = k1                                                      <L 1352>
        wp::adj_array_store(var_out_indices, var_110, var_111, var_46, adj_out_indices, adj_110, adj_111, adj_46);
        wp::adj_add(var_60, var_109, adj_60, adj_109, adj_110);
        // adj: out_indices[oid + 5, 1] = i1                                                      <L 1351>
        wp::adj_array_store(var_out_indices, var_107, var_108, var_52, adj_out_indices, adj_107, adj_108, adj_52);
        wp::adj_add(var_60, var_106, adj_60, adj_106, adj_107);
        // adj: out_indices[oid + 5, 0] = j1                                                      <L 1350>
        wp::adj_array_store(var_out_indices, var_104, var_105, var_46, adj_out_indices, adj_104, adj_105, adj_46);
        wp::adj_add(var_60, var_103, adj_60, adj_103, adj_104);
        // adj: out_indices[oid + 4, 2] = i1                                                      <L 1349>
        wp::adj_array_store(var_out_indices, var_101, var_102, var_42, adj_out_indices, adj_101, adj_102, adj_42);
        wp::adj_add(var_60, var_100, adj_60, adj_100, adj_101);
        // adj: out_indices[oid + 4, 1] = i0                                                      <L 1348>
        wp::adj_array_store(var_out_indices, var_98, var_99, var_48, adj_out_indices, adj_98, adj_99, adj_48);
        wp::adj_add(var_60, var_97, adj_60, adj_97, adj_98);
        // adj: out_indices[oid + 4, 0] = j0                                                      <L 1347>
        wp::adj_array_store(var_out_indices, var_95, var_96, var_52, adj_out_indices, adj_95, adj_96, adj_52);
        wp::adj_add(var_60, var_94, adj_60, adj_94, adj_95);
        // adj: out_indices[oid + 3, 2] = j1                                                      <L 1346>
        wp::adj_array_store(var_out_indices, var_92, var_93, var_46, adj_out_indices, adj_92, adj_93, adj_46);
        wp::adj_add(var_60, var_91, adj_60, adj_91, adj_92);
        // adj: out_indices[oid + 3, 1] = i1                                                      <L 1345>
        wp::adj_array_store(var_out_indices, var_89, var_90, var_48, adj_out_indices, adj_89, adj_90, adj_48);
        wp::adj_add(var_60, var_88, adj_60, adj_88, adj_89);
        // adj: out_indices[oid + 3, 0] = j0                                                      <L 1344>
        wp::adj_array_store(var_out_indices, var_86, var_87, var_58, adj_out_indices, adj_86, adj_87, adj_58);
        wp::adj_add(var_60, var_85, adj_60, adj_85, adj_86);
        // adj: out_indices[oid + 2, 2] = k1                                                      <L 1343>
        wp::adj_array_store(var_out_indices, var_83, var_84, var_52, adj_out_indices, adj_83, adj_84, adj_52);
        wp::adj_add(var_60, var_82, adj_60, adj_82, adj_83);
        // adj: out_indices[oid + 2, 1] = j1                                                      <L 1342>
        wp::adj_array_store(var_out_indices, var_80, var_81, var_48, adj_out_indices, adj_80, adj_81, adj_48);
        wp::adj_add(var_60, var_79, adj_60, adj_79, adj_80);
        // adj: out_indices[oid + 2, 0] = j0                                                      <L 1341>
        wp::adj_array_store(var_out_indices, var_77, var_78, var_54, adj_out_indices, adj_77, adj_78, adj_54);
        wp::adj_add(var_60, var_76, adj_60, adj_76, adj_77);
        // adj: out_indices[oid + 1, 2] = k0                                                      <L 1340>
        wp::adj_array_store(var_out_indices, var_74, var_75, var_58, adj_out_indices, adj_74, adj_75, adj_58);
        wp::adj_add(var_60, var_73, adj_60, adj_73, adj_74);
        // adj: out_indices[oid + 1, 1] = k1                                                      <L 1339>
        wp::adj_array_store(var_out_indices, var_71, var_72, var_48, adj_out_indices, adj_71, adj_72, adj_48);
        wp::adj_add(var_60, var_70, adj_60, adj_70, adj_71);
        // adj: out_indices[oid + 1, 0] = j0                                                      <L 1338>
        wp::adj_array_store(var_out_indices, var_68, var_69, var_54, adj_out_indices, adj_68, adj_69, adj_54);
        wp::adj_add(var_60, var_67, adj_60, adj_67, adj_68);
        // adj: out_indices[oid + 0, 2] = k0                                                      <L 1337>
        wp::adj_array_store(var_out_indices, var_65, var_66, var_48, adj_out_indices, adj_65, adj_66, adj_48);
        wp::adj_add(var_60, var_64, adj_60, adj_64, adj_65);
        // adj: out_indices[oid + 0, 1] = j0                                                      <L 1336>
        wp::adj_array_store(var_out_indices, var_62, var_63, var_42, adj_out_indices, adj_62, adj_63, adj_42);
        wp::adj_add(var_60, var_61, adj_60, adj_61, adj_62);
        // adj: out_indices[oid + 0, 0] = i0                                                      <L 1335>
        wp::adj_mul(var_0, var_59, adj_0, adj_59, adj_60);
        // adj: oid = tid * 8                                                                     <L 1334>
        wp::adj_array_store(var_out_vertices, var_58, var_40, adj_out_vertices, adj_58, adj_40);
        // adj: out_vertices[k1] = vk1                                                            <L 1332>
        wp::adj_array_store(var_out_vertices, var_54, var_39, adj_out_vertices, adj_54, adj_39);
        // adj: out_vertices[k0] = vk0                                                            <L 1331>
        wp::adj_array_store(var_out_vertices, var_52, var_38, adj_out_vertices, adj_52, adj_38);
        // adj: out_vertices[j1] = vj1                                                            <L 1330>
        wp::adj_array_store(var_out_vertices, var_48, var_37, adj_out_vertices, adj_48, adj_37);
        // adj: out_vertices[j0] = vj0                                                            <L 1329>
        wp::adj_array_store(var_out_vertices, var_46, var_36, adj_out_vertices, adj_46, adj_36);
        // adj: out_vertices[i1] = vi1                                                            <L 1328>
        wp::adj_array_store(var_out_vertices, var_42, var_35, adj_out_vertices, adj_42, adj_35);
        // adj: out_vertices[i0] = vi0                                                            <L 1327>
        wp::adj_add(var_56, var_57, adj_56, adj_57, adj_58);
        wp::adj_mul(var_11, var_55, adj_11, adj_55, adj_56);
        // adj: k1 = k * 2 + 1                                                                    <L 1325>
        wp::adj_mul(var_11, var_53, adj_11, adj_53, adj_54);
        // adj: k0 = k * 2                                                                        <L 1324>
        wp::adj_add(var_50, var_51, adj_50, adj_51, adj_52);
        wp::adj_mul(var_7, var_49, adj_7, adj_49, adj_50);
        // adj: j1 = j * 2 + 1                                                                    <L 1323>
        wp::adj_mul(var_7, var_47, adj_7, adj_47, adj_48);
        // adj: j0 = j * 2                                                                        <L 1322>
        wp::adj_add(var_44, var_45, adj_44, adj_45, adj_46);
        wp::adj_mul(var_3, var_43, adj_3, adj_43, adj_44);
        // adj: i1 = i * 2 + 1                                                                    <L 1321>
        wp::adj_mul(var_3, var_41, adj_3, adj_41, adj_42);
        // adj: i0 = i * 2                                                                        <L 1320>
        wp::adj_sub(var_20, var_33, adj_20, adj_33, adj_40);
        // adj: vk1 = vk - tk                                                                     <L 1318>
        wp::adj_add(var_20, var_33, adj_20, adj_33, adj_39);
        // adj: vk0 = vk + tk                                                                     <L 1317>
        wp::adj_sub(var_17, var_30, adj_17, adj_30, adj_38);
        // adj: vj1 = vj - tj                                                                     <L 1316>
        wp::adj_add(var_17, var_30, adj_17, adj_30, adj_37);
        // adj: vj0 = vj + tj                                                                     <L 1315>
        wp::adj_sub(var_14, var_27, adj_14, adj_27, adj_36);
        // adj: vi1 = vi - ti                                                                     <L 1314>
        wp::adj_add(var_14, var_27, adj_14, adj_27, adj_35);
        // adj: vi0 = vi + ti                                                                     <L 1313>
        wp::adj_mul(var_25, var_34, adj_25, adj_32, adj_33);
        wp::adj_address(var_thickness, var_11, adj_thickness, adj_11, adj_32);
        // adj: tk = normal * thickness[k]                                                        <L 1310>
        wp::adj_mul(var_25, var_31, adj_25, adj_29, adj_30);
        wp::adj_address(var_thickness, var_7, adj_thickness, adj_7, adj_29);
        // adj: tj = normal * thickness[j]                                                        <L 1309>
        wp::adj_mul(var_25, var_28, adj_25, adj_26, adj_27);
        wp::adj_address(var_thickness, var_3, adj_thickness, adj_3, adj_26);
        // adj: ti = normal * thickness[i]                                                        <L 1308>
        wp::adj_normalize(var_24, var_25, adj_24, adj_25);
        wp::adj_cross(var_22, var_23, adj_22, adj_23, adj_24);
        wp::adj_sub(var_20, var_14, adj_20, adj_14, adj_23);
        wp::adj_sub(var_17, var_14, adj_17, adj_14, adj_22);
        // adj: normal = wp.normalize(wp.cross(vj - vi, vk - vi))                                 <L 1307>
        wp::adj_copy(var_21, adj_19, adj_20);
        wp::adj_address(var_vertices, var_11, adj_vertices, adj_11, adj_19);
        // adj: vk = vertices[k]                                                                  <L 1305>
        wp::adj_copy(var_18, adj_16, adj_17);
        wp::adj_address(var_vertices, var_7, adj_vertices, adj_7, adj_16);
        // adj: vj = vertices[j]                                                                  <L 1304>
        wp::adj_copy(var_15, adj_13, adj_14);
        wp::adj_address(var_vertices, var_3, adj_vertices, adj_3, adj_13);
        // adj: vi = vertices[i]                                                                  <L 1303>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_indices, var_0, var_9, adj_indices, adj_0, adj_9, adj_10);
        // adj: k = indices[tid, 2]                                                               <L 1301>
        wp::adj_copy(var_8, adj_6, adj_7);
        wp::adj_address(var_indices, var_0, var_5, adj_indices, adj_0, adj_5, adj_6);
        // adj: j = indices[tid, 1]                                                               <L 1300>
        wp::adj_copy(var_4, adj_2, adj_3);
        wp::adj_address(var_indices, var_0, var_1, adj_indices, adj_0, adj_1, adj_2);
        // adj: i = indices[tid, 0]                                                               <L 1299>
        // adj: tid = wp.tid()                                                                    <L 1298>
        // adj: def solidify_mesh_kernel(                                                         <L 1274>
        continue;
    }
}



extern "C" __global__ void accumulate_vertex_normals_120e66d9_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_points,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals)
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
        const wp::int32 var_1 = 3;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 3;
        wp::int32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
        //---------
        // forward
        // def accumulate_vertex_normals(                                                         <L 18>
        // face = wp.tid()                                                                        <L 25>
        var_0 = builtin_tid1d();
        // i0 = indices[face * 3]                                                                 <L 26>
        var_2 = wp::mul(var_0, var_1);
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // i1 = indices[face * 3 + 1]                                                             <L 27>
        var_7 = wp::mul(var_0, var_6);
        var_9 = wp::add(var_7, var_8);
        var_10 = wp::address(var_indices, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // i2 = indices[face * 3 + 2]                                                             <L 28>
        var_14 = wp::mul(var_0, var_13);
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::address(var_indices, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // v0 = points[i0]                                                                        <L 29>
        var_20 = wp::address(var_points, var_4);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // v1 = points[i1]                                                                        <L 30>
        var_23 = wp::address(var_points, var_11);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // v2 = points[i2]                                                                        <L 31>
        var_26 = wp::address(var_points, var_18);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // normal = wp.cross(v1 - v0, v2 - v0)                                                    <L 32>
        var_29 = wp::sub(var_24, var_21);
        var_30 = wp::sub(var_27, var_21);
        var_31 = wp::cross(var_29, var_30);
        // wp.atomic_add(normals, i0, normal)                                                     <L 33>
        var_32 = wp::atomic_add(var_normals, var_4, var_31);
        // wp.atomic_add(normals, i1, normal)                                                     <L 34>
        var_33 = wp::atomic_add(var_normals, var_11, var_31);
        // wp.atomic_add(normals, i2, normal)                                                     <L 35>
        var_34 = wp::atomic_add(var_normals, var_18, var_31);
    }
}



extern "C" __global__ void accumulate_vertex_normals_120e66d9_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_points,
    wp::array_t<wp::int32> var_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> var_normals,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_points,
    wp::array_t<wp::int32> adj_indices,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_normals)
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
        const wp::int32 var_1 = 3;
        wp::int32 var_2;
        wp::int32* var_3;
        wp::int32 var_4;
        wp::int32 var_5;
        const wp::int32 var_6 = 3;
        wp::int32 var_7;
        const wp::int32 var_8 = 1;
        wp::int32 var_9;
        wp::int32* var_10;
        wp::int32 var_11;
        wp::int32 var_12;
        const wp::int32 var_13 = 3;
        wp::int32 var_14;
        const wp::int32 var_15 = 2;
        wp::int32 var_16;
        wp::int32* var_17;
        wp::int32 var_18;
        wp::int32 var_19;
        wp::vec_t<3, wp::float32>* var_20;
        wp::vec_t<3, wp::float32> var_21;
        wp::vec_t<3, wp::float32> var_22;
        wp::vec_t<3, wp::float32>* var_23;
        wp::vec_t<3, wp::float32> var_24;
        wp::vec_t<3, wp::float32> var_25;
        wp::vec_t<3, wp::float32>* var_26;
        wp::vec_t<3, wp::float32> var_27;
        wp::vec_t<3, wp::float32> var_28;
        wp::vec_t<3, wp::float32> var_29;
        wp::vec_t<3, wp::float32> var_30;
        wp::vec_t<3, wp::float32> var_31;
        wp::vec_t<3, wp::float32> var_32;
        wp::vec_t<3, wp::float32> var_33;
        wp::vec_t<3, wp::float32> var_34;
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
        wp::vec_t<3, wp::float32> adj_20 = {};
        wp::vec_t<3, wp::float32> adj_21 = {};
        wp::vec_t<3, wp::float32> adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::vec_t<3, wp::float32> adj_24 = {};
        wp::vec_t<3, wp::float32> adj_25 = {};
        wp::vec_t<3, wp::float32> adj_26 = {};
        wp::vec_t<3, wp::float32> adj_27 = {};
        wp::vec_t<3, wp::float32> adj_28 = {};
        wp::vec_t<3, wp::float32> adj_29 = {};
        wp::vec_t<3, wp::float32> adj_30 = {};
        wp::vec_t<3, wp::float32> adj_31 = {};
        wp::vec_t<3, wp::float32> adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::vec_t<3, wp::float32> adj_34 = {};
        //---------
        // forward
        // def accumulate_vertex_normals(                                                         <L 18>
        // face = wp.tid()                                                                        <L 25>
        var_0 = builtin_tid1d();
        // i0 = indices[face * 3]                                                                 <L 26>
        var_2 = wp::mul(var_0, var_1);
        var_3 = wp::address(var_indices, var_2);
        var_5 = wp::load(var_3);
        var_4 = wp::copy(var_5);
        // i1 = indices[face * 3 + 1]                                                             <L 27>
        var_7 = wp::mul(var_0, var_6);
        var_9 = wp::add(var_7, var_8);
        var_10 = wp::address(var_indices, var_9);
        var_12 = wp::load(var_10);
        var_11 = wp::copy(var_12);
        // i2 = indices[face * 3 + 2]                                                             <L 28>
        var_14 = wp::mul(var_0, var_13);
        var_16 = wp::add(var_14, var_15);
        var_17 = wp::address(var_indices, var_16);
        var_19 = wp::load(var_17);
        var_18 = wp::copy(var_19);
        // v0 = points[i0]                                                                        <L 29>
        var_20 = wp::address(var_points, var_4);
        var_22 = wp::load(var_20);
        var_21 = wp::copy(var_22);
        // v1 = points[i1]                                                                        <L 30>
        var_23 = wp::address(var_points, var_11);
        var_25 = wp::load(var_23);
        var_24 = wp::copy(var_25);
        // v2 = points[i2]                                                                        <L 31>
        var_26 = wp::address(var_points, var_18);
        var_28 = wp::load(var_26);
        var_27 = wp::copy(var_28);
        // normal = wp.cross(v1 - v0, v2 - v0)                                                    <L 32>
        var_29 = wp::sub(var_24, var_21);
        var_30 = wp::sub(var_27, var_21);
        var_31 = wp::cross(var_29, var_30);
        // wp.atomic_add(normals, i0, normal)                                                     <L 33>
        // var_32 = wp::atomic_add(var_normals, var_4, var_31);
        // wp.atomic_add(normals, i1, normal)                                                     <L 34>
        // var_33 = wp::atomic_add(var_normals, var_11, var_31);
        // wp.atomic_add(normals, i2, normal)                                                     <L 35>
        // var_34 = wp::atomic_add(var_normals, var_18, var_31);
        //---------
        // reverse
        wp::adj_atomic_add(var_normals, var_18, var_31, adj_normals, adj_18, adj_31, adj_34);
        // adj: wp.atomic_add(normals, i2, normal)                                                <L 35>
        wp::adj_atomic_add(var_normals, var_11, var_31, adj_normals, adj_11, adj_31, adj_33);
        // adj: wp.atomic_add(normals, i1, normal)                                                <L 34>
        wp::adj_atomic_add(var_normals, var_4, var_31, adj_normals, adj_4, adj_31, adj_32);
        // adj: wp.atomic_add(normals, i0, normal)                                                <L 33>
        wp::adj_cross(var_29, var_30, adj_29, adj_30, adj_31);
        wp::adj_sub(var_27, var_21, adj_27, adj_21, adj_30);
        wp::adj_sub(var_24, var_21, adj_24, adj_21, adj_29);
        // adj: normal = wp.cross(v1 - v0, v2 - v0)                                               <L 32>
        wp::adj_copy(var_28, adj_26, adj_27);
        wp::adj_address(var_points, var_18, adj_points, adj_18, adj_26);
        // adj: v2 = points[i2]                                                                   <L 31>
        wp::adj_copy(var_25, adj_23, adj_24);
        wp::adj_address(var_points, var_11, adj_points, adj_11, adj_23);
        // adj: v1 = points[i1]                                                                   <L 30>
        wp::adj_copy(var_22, adj_20, adj_21);
        wp::adj_address(var_points, var_4, adj_points, adj_4, adj_20);
        // adj: v0 = points[i0]                                                                   <L 29>
        wp::adj_copy(var_19, adj_17, adj_18);
        wp::adj_address(var_indices, var_16, adj_indices, adj_16, adj_17);
        wp::adj_add(var_14, var_15, adj_14, adj_15, adj_16);
        wp::adj_mul(var_0, var_13, adj_0, adj_13, adj_14);
        // adj: i2 = indices[face * 3 + 2]                                                        <L 28>
        wp::adj_copy(var_12, adj_10, adj_11);
        wp::adj_address(var_indices, var_9, adj_indices, adj_9, adj_10);
        wp::adj_add(var_7, var_8, adj_7, adj_8, adj_9);
        wp::adj_mul(var_0, var_6, adj_0, adj_6, adj_7);
        // adj: i1 = indices[face * 3 + 1]                                                        <L 27>
        wp::adj_copy(var_5, adj_3, adj_4);
        wp::adj_address(var_indices, var_2, adj_indices, adj_2, adj_3);
        wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
        // adj: i0 = indices[face * 3]                                                            <L 26>
        // adj: face = wp.tid()                                                                   <L 25>
        // adj: def accumulate_vertex_normals(                                                    <L 18>
        continue;
    }
}

