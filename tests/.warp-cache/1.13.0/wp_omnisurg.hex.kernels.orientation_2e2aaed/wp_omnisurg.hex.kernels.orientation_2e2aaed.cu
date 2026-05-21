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


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/orientation.py:30
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _safe_normalize_0(
    wp::vec_t<3, wp::float32> var_v)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-06;
    bool var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _safe_normalize(v: wp.vec3) -> wp.vec3:                                            <L 31>
    // n = wp.length(v)                                                                       <L 32>
    var_0 = wp::length(var_v);
    // if n > 1.0e-6:                                                                         <L 33>
    var_2 = (var_0 > var_1);
    if (var_2) {
        // return v / n                                                                       <L 34>
        var_3 = wp::div(var_v, var_0);
        return var_3;
    }
    // return v                                                                               <L 35>
    return var_v;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/orientation.py:38
static CUDA_CALLABLE wp::vec_t<3, wp::float32> _update_axis_0(
    wp::vec_t<3, wp::float32> var_orie_axis,
    wp::float32 var_sign,
    wp::vec_t<3, wp::float32> var_diff,
    wp::float32 var_fA)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // forward
    // def _update_axis(orie_axis: wp.vec3, sign: float, diff: wp.vec3, fA: float) -> wp.vec3:       <L 39>
    // return orie_axis + _safe_normalize(diff) * (sign * fA)                                 <L 40>
    var_0 = _safe_normalize_0(var_diff);
    var_1 = wp::mul(var_sign, var_fA);
    var_2 = wp::mul(var_0, var_1);
    var_3 = wp::add(var_orie_axis, var_2);
    return var_3;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/orientation.py:30
static CUDA_CALLABLE void adj__safe_normalize_0(
    wp::vec_t<3, wp::float32> var_v,
    wp::vec_t<3, wp::float32> & adj_v,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::float32 var_0;
    const wp::float32 var_1 = 1e-06;
    bool var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // dual vars
    wp::float32 adj_0 = {};
    wp::float32 adj_1 = {};
    bool adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    //---------
    // forward
    // def _safe_normalize(v: wp.vec3) -> wp.vec3:                                            <L 31>
    // n = wp.length(v)                                                                       <L 32>
    var_0 = wp::length(var_v);
    // if n > 1.0e-6:                                                                         <L 33>
    var_2 = (var_0 > var_1);
    if (var_2) {
        // return v / n                                                                       <L 34>
        var_3 = wp::div(var_v, var_0);
        goto label0;
    }
    // return v                                                                               <L 35>
    goto label1;
    //---------
    // reverse
    label1:;
    adj_v += adj_ret;
    // adj: return v                                                                          <L 35>
    if (var_2) {
        label0:;
        adj_3 += adj_ret;
        wp::adj_div(var_v, var_0, adj_v, adj_0, adj_3);
        // adj: return v / n                                                                  <L 34>
    }
    // adj: if n > 1.0e-6:                                                                    <L 33>
    wp::adj_length(var_v, var_0, adj_v, adj_0);
    // adj: n = wp.length(v)                                                                  <L 32>
    // adj: def _safe_normalize(v: wp.vec3) -> wp.vec3:                                       <L 31>
    return;
}


// /home/pkorzeniowsk/Projects/warp-surgical/warp-surgical-dev/omnisurg/hex/kernels/orientation.py:38
static CUDA_CALLABLE void adj__update_axis_0(
    wp::vec_t<3, wp::float32> var_orie_axis,
    wp::float32 var_sign,
    wp::vec_t<3, wp::float32> var_diff,
    wp::float32 var_fA,
    wp::vec_t<3, wp::float32> & adj_orie_axis,
    wp::float32 & adj_sign,
    wp::vec_t<3, wp::float32> & adj_diff,
    wp::float32 & adj_fA,
    wp::vec_t<3, wp::float32> & adj_ret)
{
    //---------
    // primal vars
    wp::vec_t<3, wp::float32> var_0;
    wp::float32 var_1;
    wp::vec_t<3, wp::float32> var_2;
    wp::vec_t<3, wp::float32> var_3;
    //---------
    // dual vars
    wp::vec_t<3, wp::float32> adj_0 = {};
    wp::float32 adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    //---------
    // forward
    // def _update_axis(orie_axis: wp.vec3, sign: float, diff: wp.vec3, fA: float) -> wp.vec3:       <L 39>
    // return orie_axis + _safe_normalize(diff) * (sign * fA)                                 <L 40>
    var_0 = _safe_normalize_0(var_diff);
    var_1 = wp::mul(var_sign, var_fA);
    var_2 = wp::mul(var_0, var_1);
    var_3 = wp::add(var_orie_axis, var_2);
    goto label0;
    //---------
    // reverse
    label0:;
    adj_3 += adj_ret;
    wp::adj_add(var_orie_axis, var_2, adj_orie_axis, adj_2, adj_3);
    wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    wp::adj_mul(var_sign, var_fA, adj_sign, adj_fA, adj_1);
    adj__safe_normalize_0(var_diff, adj_diff, adj_0);
    // adj: return orie_axis + _safe_normalize(diff) * (sign * fA)                            <L 40>
    // adj: def _update_axis(orie_axis: wp.vec3, sign: float, diff: wp.vec3, fA: float) -> wp.vec3:  <L 39>
    return;
}



extern "C" __global__ void initialize_orientations_kernel_c6dc710e_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_orientation_out)
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
        const wp::mat_t<3, 3, wp::float32> var_1 = wp::initializer_array<9,wp::float32>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        //---------
        // forward
        // def initialize_orientations_kernel(orientation_out: wp.array(dtype=wp.mat33)):         <L 44>
        // i = wp.tid()                                                                           <L 46>
        var_0 = builtin_tid1d();
        // orientation_out[i] = IDENTITY_MAT33                                                    <L 47>
        wp::array_store(var_orientation_out, var_0, var_1);
    }
}



extern "C" __global__ void initialize_orientations_kernel_c6dc710e_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_orientation_out,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_orientation_out)
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
        const wp::mat_t<3, 3, wp::float32> var_1 = wp::initializer_array<9,wp::float32>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        //---------
        // dual vars
        wp::int32 adj_0 = {};
        wp::mat_t<3, 3, wp::float32> adj_1 = {};
        //---------
        // forward
        // def initialize_orientations_kernel(orientation_out: wp.array(dtype=wp.mat33)):         <L 44>
        // i = wp.tid()                                                                           <L 46>
        var_0 = builtin_tid1d();
        // orientation_out[i] = IDENTITY_MAT33                                                    <L 47>
        // wp::array_store(var_orientation_out, var_0, var_1);
        //---------
        // reverse
        wp::adj_array_store(var_orientation_out, var_0, var_1, adj_orientation_out, adj_0, adj_1);
        // adj: orientation_out[i] = IDENTITY_MAT33                                               <L 47>
        // adj: i = wp.tid()                                                                      <L 46>
        // adj: def initialize_orientations_kernel(orientation_out: wp.array(dtype=wp.mat33)):    <L 44>
        continue;
    }
}



extern "C" __global__ void update_orientation_kernel_419bec28_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_neighbors,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_orientation_in,
    wp::float32 var_fA,
    wp::float32 var_fB,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_orientation_out)
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
        wp::mat_t<3, 3, wp::float32>* var_9;
        wp::mat_t<3, 3, wp::float32> var_10;
        wp::mat_t<3, 3, wp::float32>* var_11;
        wp::mat_t<3, 3, wp::float32> var_12;
        wp::mat_t<3, 3, wp::float32> var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 1;
        wp::float32 var_19;
        const wp::int32 var_20 = 0;
        const wp::int32 var_21 = 2;
        wp::float32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        const wp::int32 var_24 = 1;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        const wp::int32 var_28 = 1;
        wp::float32 var_29;
        const wp::int32 var_30 = 1;
        const wp::int32 var_31 = 2;
        wp::float32 var_32;
        wp::vec_t<3, wp::float32> var_33;
        const wp::int32 var_34 = 2;
        const wp::int32 var_35 = 0;
        wp::float32 var_36;
        const wp::int32 var_37 = 2;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 2;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 0;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        bool var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        wp::int32* var_54;
        const wp::int32 var_55 = 1;
        const wp::int32 var_56 = 1;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 0;
        bool var_61;
        const wp::float32 var_62 = -1.0;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::mat_t<3, 3, wp::float32>* var_67;
        wp::mat_t<3, 3, wp::float32> var_68;
        wp::mat_t<3, 3, wp::float32> var_69;
        const wp::int32 var_70 = 0;
        const wp::int32 var_71 = 0;
        wp::float32 var_72;
        const wp::int32 var_73 = 0;
        const wp::int32 var_74 = 1;
        wp::float32 var_75;
        const wp::int32 var_76 = 0;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        const wp::int32 var_82 = 1;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        const wp::int32 var_86 = 1;
        wp::float32 var_87;
        const wp::int32 var_88 = 1;
        const wp::int32 var_89 = 2;
        wp::float32 var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        const wp::int32 var_94 = 2;
        const wp::int32 var_95 = 0;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        const wp::int32 var_100 = 2;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        const wp::int32 var_109 = 1;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        bool var_113;
        const wp::int32 var_114 = 0;
        bool var_115;
        wp::int32* var_116;
        const wp::int32 var_117 = 1;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 0;
        bool var_123;
        const wp::float32 var_124 = 1.0;
        wp::vec_t<3, wp::float32>* var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::mat_t<3, 3, wp::float32>* var_129;
        wp::mat_t<3, 3, wp::float32> var_130;
        wp::mat_t<3, 3, wp::float32> var_131;
        const wp::int32 var_132 = 0;
        const wp::int32 var_133 = 0;
        wp::float32 var_134;
        const wp::int32 var_135 = 0;
        const wp::int32 var_136 = 1;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        const wp::int32 var_139 = 2;
        wp::float32 var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        const wp::int32 var_144 = 1;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        const wp::int32 var_148 = 1;
        wp::float32 var_149;
        const wp::int32 var_150 = 1;
        const wp::int32 var_151 = 2;
        wp::float32 var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        const wp::int32 var_156 = 2;
        const wp::int32 var_157 = 0;
        wp::float32 var_158;
        const wp::int32 var_159 = 2;
        const wp::int32 var_160 = 1;
        wp::float32 var_161;
        const wp::int32 var_162 = 2;
        const wp::int32 var_163 = 2;
        wp::float32 var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::mat_t<3, 3, wp::float32> var_171;
        const wp::int32 var_172 = 2;
        wp::int32* var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        bool var_176;
        const wp::int32 var_177 = 0;
        bool var_178;
        wp::int32* var_179;
        const wp::int32 var_180 = 1;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        const wp::int32 var_185 = 0;
        bool var_186;
        const wp::float32 var_187 = -1.0;
        wp::vec_t<3, wp::float32>* var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::vec_t<3, wp::float32> var_191;
        wp::mat_t<3, 3, wp::float32>* var_192;
        wp::mat_t<3, 3, wp::float32> var_193;
        wp::mat_t<3, 3, wp::float32> var_194;
        const wp::int32 var_195 = 0;
        const wp::int32 var_196 = 0;
        wp::float32 var_197;
        const wp::int32 var_198 = 0;
        const wp::int32 var_199 = 1;
        wp::float32 var_200;
        const wp::int32 var_201 = 0;
        const wp::int32 var_202 = 2;
        wp::float32 var_203;
        wp::vec_t<3, wp::float32> var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        const wp::int32 var_207 = 1;
        const wp::int32 var_208 = 0;
        wp::float32 var_209;
        const wp::int32 var_210 = 1;
        const wp::int32 var_211 = 1;
        wp::float32 var_212;
        const wp::int32 var_213 = 1;
        const wp::int32 var_214 = 2;
        wp::float32 var_215;
        wp::vec_t<3, wp::float32> var_216;
        wp::vec_t<3, wp::float32> var_217;
        wp::vec_t<3, wp::float32> var_218;
        const wp::int32 var_219 = 2;
        const wp::int32 var_220 = 0;
        wp::float32 var_221;
        const wp::int32 var_222 = 2;
        const wp::int32 var_223 = 1;
        wp::float32 var_224;
        const wp::int32 var_225 = 2;
        const wp::int32 var_226 = 2;
        wp::float32 var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        wp::vec_t<3, wp::float32> var_233;
        wp::mat_t<3, 3, wp::float32> var_234;
        const wp::int32 var_235 = 3;
        wp::int32* var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        bool var_239;
        const wp::int32 var_240 = 0;
        bool var_241;
        wp::int32* var_242;
        const wp::int32 var_243 = 1;
        const wp::int32 var_244 = 1;
        wp::int32 var_245;
        wp::int32 var_246;
        wp::int32 var_247;
        const wp::int32 var_248 = 0;
        bool var_249;
        const wp::float32 var_250 = 1.0;
        wp::vec_t<3, wp::float32>* var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<3, wp::float32> var_253;
        wp::vec_t<3, wp::float32> var_254;
        wp::mat_t<3, 3, wp::float32>* var_255;
        wp::mat_t<3, 3, wp::float32> var_256;
        wp::mat_t<3, 3, wp::float32> var_257;
        const wp::int32 var_258 = 0;
        const wp::int32 var_259 = 0;
        wp::float32 var_260;
        const wp::int32 var_261 = 0;
        const wp::int32 var_262 = 1;
        wp::float32 var_263;
        const wp::int32 var_264 = 0;
        const wp::int32 var_265 = 2;
        wp::float32 var_266;
        wp::vec_t<3, wp::float32> var_267;
        wp::vec_t<3, wp::float32> var_268;
        wp::vec_t<3, wp::float32> var_269;
        const wp::int32 var_270 = 1;
        const wp::int32 var_271 = 0;
        wp::float32 var_272;
        const wp::int32 var_273 = 1;
        const wp::int32 var_274 = 1;
        wp::float32 var_275;
        const wp::int32 var_276 = 1;
        const wp::int32 var_277 = 2;
        wp::float32 var_278;
        wp::vec_t<3, wp::float32> var_279;
        wp::vec_t<3, wp::float32> var_280;
        wp::vec_t<3, wp::float32> var_281;
        const wp::int32 var_282 = 2;
        const wp::int32 var_283 = 0;
        wp::float32 var_284;
        const wp::int32 var_285 = 2;
        const wp::int32 var_286 = 1;
        wp::float32 var_287;
        const wp::int32 var_288 = 2;
        const wp::int32 var_289 = 2;
        wp::float32 var_290;
        wp::vec_t<3, wp::float32> var_291;
        wp::vec_t<3, wp::float32> var_292;
        wp::vec_t<3, wp::float32> var_293;
        wp::vec_t<3, wp::float32> var_294;
        wp::vec_t<3, wp::float32> var_295;
        wp::vec_t<3, wp::float32> var_296;
        wp::mat_t<3, 3, wp::float32> var_297;
        const wp::int32 var_298 = 4;
        wp::int32* var_299;
        wp::int32 var_300;
        wp::int32 var_301;
        bool var_302;
        const wp::int32 var_303 = 0;
        bool var_304;
        wp::int32* var_305;
        const wp::int32 var_306 = 1;
        const wp::int32 var_307 = 1;
        wp::int32 var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        const wp::int32 var_311 = 0;
        bool var_312;
        const wp::float32 var_313 = -1.0;
        wp::vec_t<3, wp::float32>* var_314;
        wp::vec_t<3, wp::float32> var_315;
        wp::vec_t<3, wp::float32> var_316;
        wp::vec_t<3, wp::float32> var_317;
        wp::mat_t<3, 3, wp::float32>* var_318;
        wp::mat_t<3, 3, wp::float32> var_319;
        wp::mat_t<3, 3, wp::float32> var_320;
        const wp::int32 var_321 = 0;
        const wp::int32 var_322 = 0;
        wp::float32 var_323;
        const wp::int32 var_324 = 0;
        const wp::int32 var_325 = 1;
        wp::float32 var_326;
        const wp::int32 var_327 = 0;
        const wp::int32 var_328 = 2;
        wp::float32 var_329;
        wp::vec_t<3, wp::float32> var_330;
        wp::vec_t<3, wp::float32> var_331;
        wp::vec_t<3, wp::float32> var_332;
        const wp::int32 var_333 = 1;
        const wp::int32 var_334 = 0;
        wp::float32 var_335;
        const wp::int32 var_336 = 1;
        const wp::int32 var_337 = 1;
        wp::float32 var_338;
        const wp::int32 var_339 = 1;
        const wp::int32 var_340 = 2;
        wp::float32 var_341;
        wp::vec_t<3, wp::float32> var_342;
        wp::vec_t<3, wp::float32> var_343;
        wp::vec_t<3, wp::float32> var_344;
        const wp::int32 var_345 = 2;
        const wp::int32 var_346 = 0;
        wp::float32 var_347;
        const wp::int32 var_348 = 2;
        const wp::int32 var_349 = 1;
        wp::float32 var_350;
        const wp::int32 var_351 = 2;
        const wp::int32 var_352 = 2;
        wp::float32 var_353;
        wp::vec_t<3, wp::float32> var_354;
        wp::vec_t<3, wp::float32> var_355;
        wp::vec_t<3, wp::float32> var_356;
        wp::vec_t<3, wp::float32> var_357;
        wp::vec_t<3, wp::float32> var_358;
        wp::vec_t<3, wp::float32> var_359;
        wp::mat_t<3, 3, wp::float32> var_360;
        const wp::int32 var_361 = 5;
        wp::int32* var_362;
        wp::int32 var_363;
        wp::int32 var_364;
        bool var_365;
        const wp::int32 var_366 = 0;
        bool var_367;
        wp::int32* var_368;
        const wp::int32 var_369 = 1;
        const wp::int32 var_370 = 1;
        wp::int32 var_371;
        wp::int32 var_372;
        wp::int32 var_373;
        const wp::int32 var_374 = 0;
        bool var_375;
        const wp::float32 var_376 = 1.0;
        wp::vec_t<3, wp::float32>* var_377;
        wp::vec_t<3, wp::float32> var_378;
        wp::vec_t<3, wp::float32> var_379;
        wp::vec_t<3, wp::float32> var_380;
        wp::mat_t<3, 3, wp::float32>* var_381;
        wp::mat_t<3, 3, wp::float32> var_382;
        wp::mat_t<3, 3, wp::float32> var_383;
        const wp::int32 var_384 = 0;
        const wp::int32 var_385 = 0;
        wp::float32 var_386;
        const wp::int32 var_387 = 0;
        const wp::int32 var_388 = 1;
        wp::float32 var_389;
        const wp::int32 var_390 = 0;
        const wp::int32 var_391 = 2;
        wp::float32 var_392;
        wp::vec_t<3, wp::float32> var_393;
        wp::vec_t<3, wp::float32> var_394;
        wp::vec_t<3, wp::float32> var_395;
        const wp::int32 var_396 = 1;
        const wp::int32 var_397 = 0;
        wp::float32 var_398;
        const wp::int32 var_399 = 1;
        const wp::int32 var_400 = 1;
        wp::float32 var_401;
        const wp::int32 var_402 = 1;
        const wp::int32 var_403 = 2;
        wp::float32 var_404;
        wp::vec_t<3, wp::float32> var_405;
        wp::vec_t<3, wp::float32> var_406;
        wp::vec_t<3, wp::float32> var_407;
        const wp::int32 var_408 = 2;
        const wp::int32 var_409 = 0;
        wp::float32 var_410;
        const wp::int32 var_411 = 2;
        const wp::int32 var_412 = 1;
        wp::float32 var_413;
        const wp::int32 var_414 = 2;
        const wp::int32 var_415 = 2;
        wp::float32 var_416;
        wp::vec_t<3, wp::float32> var_417;
        wp::vec_t<3, wp::float32> var_418;
        wp::vec_t<3, wp::float32> var_419;
        wp::vec_t<3, wp::float32> var_420;
        wp::vec_t<3, wp::float32> var_421;
        wp::vec_t<3, wp::float32> var_422;
        wp::mat_t<3, 3, wp::float32> var_423;
        wp::vec_t<3, wp::float32> var_424;
        wp::vec_t<3, wp::float32> var_425;
        wp::vec_t<3, wp::float32> var_426;
        const wp::int32 var_427 = 0;
        wp::float32 var_428;
        const wp::int32 var_429 = 1;
        wp::float32 var_430;
        const wp::int32 var_431 = 2;
        wp::float32 var_432;
        const wp::int32 var_433 = 0;
        wp::float32 var_434;
        const wp::int32 var_435 = 1;
        wp::float32 var_436;
        const wp::int32 var_437 = 2;
        wp::float32 var_438;
        const wp::int32 var_439 = 0;
        wp::float32 var_440;
        const wp::int32 var_441 = 1;
        wp::float32 var_442;
        const wp::int32 var_443 = 2;
        wp::float32 var_444;
        wp::mat_t<3, 3, wp::float32> var_445;
        //---------
        // forward
        // def update_orientation_kernel(                                                         <L 51>
        // i = wp.tid()                                                                           <L 67>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 68>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // orientation_out[i] = orientation_in[i]                                             <L 69>
            var_9 = wp::address(var_orientation_in, var_0);
            var_10 = wp::load(var_9);
            wp::array_store(var_orientation_out, var_0, var_10);
            // return                                                                             <L 70>
            continue;
        }
        // prev = orientation_in[i]                                                               <L 72>
        var_11 = wp::address(var_orientation_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // orie_x = wp.vec3(prev[0, 0], prev[0, 1], prev[0, 2])                                   <L 73>
        var_16 = wp::extract(var_12, var_14, var_15);
        var_19 = wp::extract(var_12, var_17, var_18);
        var_22 = wp::extract(var_12, var_20, var_21);
        var_23 = wp::vec_t<3, wp::float32>(var_16, var_19, var_22);
        // orie_y = wp.vec3(prev[1, 0], prev[1, 1], prev[1, 2])                                   <L 74>
        var_26 = wp::extract(var_12, var_24, var_25);
        var_29 = wp::extract(var_12, var_27, var_28);
        var_32 = wp::extract(var_12, var_30, var_31);
        var_33 = wp::vec_t<3, wp::float32>(var_26, var_29, var_32);
        // orie_z = wp.vec3(prev[2, 0], prev[2, 1], prev[2, 2])                                   <L 75>
        var_36 = wp::extract(var_12, var_34, var_35);
        var_39 = wp::extract(var_12, var_37, var_38);
        var_42 = wp::extract(var_12, var_40, var_41);
        var_43 = wp::vec_t<3, wp::float32>(var_36, var_39, var_42);
        // pos_i = particle_q[i]                                                                  <L 76>
        var_44 = wp::address(var_particle_q, var_0);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // n = particle_neighbors[i, 0]  # -X                                                     <L 80>
        var_48 = wp::address(var_particle_neighbors, var_0, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 81>
        var_53 = (var_49 >= var_52);
        var_51 = var_53;
        if (var_51) {
            var_54 = wp::address(var_particle_flags, var_49);
            var_57 = wp::int32(var_56);
            var_59 = wp::load(var_54);
            var_58 = wp::bit_and(var_59, var_57);
            var_61 = (var_58 != var_60);
            var_51 = var_51 && var_61;
        }
        if (var_51) {
            // orie_x = _update_axis(orie_x, -1.0, particle_q[n] - pos_i, fA)                     <L 82>
            var_63 = wp::address(var_particle_q, var_49);
            var_65 = wp::load(var_63);
            var_64 = wp::sub(var_65, var_45);
            var_66 = _update_axis_0(var_23, var_62, var_64, var_fA);
            // nb = orientation_in[n]                                                             <L 83>
            var_67 = wp::address(var_orientation_in, var_49);
            var_69 = wp::load(var_67);
            var_68 = wp::copy(var_69);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 84>
            var_72 = wp::extract(var_68, var_70, var_71);
            var_75 = wp::extract(var_68, var_73, var_74);
            var_78 = wp::extract(var_68, var_76, var_77);
            var_79 = wp::vec_t<3, wp::float32>(var_72, var_75, var_78);
            var_80 = wp::mul(var_79, var_fB);
            var_81 = wp::add(var_66, var_80);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 85>
            var_84 = wp::extract(var_68, var_82, var_83);
            var_87 = wp::extract(var_68, var_85, var_86);
            var_90 = wp::extract(var_68, var_88, var_89);
            var_91 = wp::vec_t<3, wp::float32>(var_84, var_87, var_90);
            var_92 = wp::mul(var_91, var_fB);
            var_93 = wp::add(var_33, var_92);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 86>
            var_96 = wp::extract(var_68, var_94, var_95);
            var_99 = wp::extract(var_68, var_97, var_98);
            var_102 = wp::extract(var_68, var_100, var_101);
            var_103 = wp::vec_t<3, wp::float32>(var_96, var_99, var_102);
            var_104 = wp::mul(var_103, var_fB);
            var_105 = wp::add(var_43, var_104);
        }
        var_106 = wp::where(var_51, var_81, var_23);
        var_107 = wp::where(var_51, var_93, var_33);
        var_108 = wp::where(var_51, var_105, var_43);
        // n = particle_neighbors[i, 1]  # +X                                                     <L 88>
        var_110 = wp::address(var_particle_neighbors, var_0, var_109);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 89>
        var_115 = (var_111 >= var_114);
        var_113 = var_115;
        if (var_113) {
            var_116 = wp::address(var_particle_flags, var_111);
            var_119 = wp::int32(var_118);
            var_121 = wp::load(var_116);
            var_120 = wp::bit_and(var_121, var_119);
            var_123 = (var_120 != var_122);
            var_113 = var_113 && var_123;
        }
        if (var_113) {
            // orie_x = _update_axis(orie_x, 1.0, particle_q[n] - pos_i, fA)                      <L 90>
            var_125 = wp::address(var_particle_q, var_111);
            var_127 = wp::load(var_125);
            var_126 = wp::sub(var_127, var_45);
            var_128 = _update_axis_0(var_106, var_124, var_126, var_fA);
            // nb = orientation_in[n]                                                             <L 91>
            var_129 = wp::address(var_orientation_in, var_111);
            var_131 = wp::load(var_129);
            var_130 = wp::copy(var_131);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 92>
            var_134 = wp::extract(var_130, var_132, var_133);
            var_137 = wp::extract(var_130, var_135, var_136);
            var_140 = wp::extract(var_130, var_138, var_139);
            var_141 = wp::vec_t<3, wp::float32>(var_134, var_137, var_140);
            var_142 = wp::mul(var_141, var_fB);
            var_143 = wp::add(var_128, var_142);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 93>
            var_146 = wp::extract(var_130, var_144, var_145);
            var_149 = wp::extract(var_130, var_147, var_148);
            var_152 = wp::extract(var_130, var_150, var_151);
            var_153 = wp::vec_t<3, wp::float32>(var_146, var_149, var_152);
            var_154 = wp::mul(var_153, var_fB);
            var_155 = wp::add(var_107, var_154);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 94>
            var_158 = wp::extract(var_130, var_156, var_157);
            var_161 = wp::extract(var_130, var_159, var_160);
            var_164 = wp::extract(var_130, var_162, var_163);
            var_165 = wp::vec_t<3, wp::float32>(var_158, var_161, var_164);
            var_166 = wp::mul(var_165, var_fB);
            var_167 = wp::add(var_108, var_166);
        }
        var_168 = wp::where(var_113, var_143, var_106);
        var_169 = wp::where(var_113, var_155, var_107);
        var_170 = wp::where(var_113, var_167, var_108);
        var_171 = wp::where(var_113, var_130, var_68);
        // n = particle_neighbors[i, 2]  # -Y                                                     <L 96>
        var_173 = wp::address(var_particle_neighbors, var_0, var_172);
        var_175 = wp::load(var_173);
        var_174 = wp::copy(var_175);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 97>
        var_178 = (var_174 >= var_177);
        var_176 = var_178;
        if (var_176) {
            var_179 = wp::address(var_particle_flags, var_174);
            var_182 = wp::int32(var_181);
            var_184 = wp::load(var_179);
            var_183 = wp::bit_and(var_184, var_182);
            var_186 = (var_183 != var_185);
            var_176 = var_176 && var_186;
        }
        if (var_176) {
            // orie_y = _update_axis(orie_y, -1.0, particle_q[n] - pos_i, fA)                     <L 98>
            var_188 = wp::address(var_particle_q, var_174);
            var_190 = wp::load(var_188);
            var_189 = wp::sub(var_190, var_45);
            var_191 = _update_axis_0(var_169, var_187, var_189, var_fA);
            // nb = orientation_in[n]                                                             <L 99>
            var_192 = wp::address(var_orientation_in, var_174);
            var_194 = wp::load(var_192);
            var_193 = wp::copy(var_194);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 100>
            var_197 = wp::extract(var_193, var_195, var_196);
            var_200 = wp::extract(var_193, var_198, var_199);
            var_203 = wp::extract(var_193, var_201, var_202);
            var_204 = wp::vec_t<3, wp::float32>(var_197, var_200, var_203);
            var_205 = wp::mul(var_204, var_fB);
            var_206 = wp::add(var_168, var_205);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 101>
            var_209 = wp::extract(var_193, var_207, var_208);
            var_212 = wp::extract(var_193, var_210, var_211);
            var_215 = wp::extract(var_193, var_213, var_214);
            var_216 = wp::vec_t<3, wp::float32>(var_209, var_212, var_215);
            var_217 = wp::mul(var_216, var_fB);
            var_218 = wp::add(var_191, var_217);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 102>
            var_221 = wp::extract(var_193, var_219, var_220);
            var_224 = wp::extract(var_193, var_222, var_223);
            var_227 = wp::extract(var_193, var_225, var_226);
            var_228 = wp::vec_t<3, wp::float32>(var_221, var_224, var_227);
            var_229 = wp::mul(var_228, var_fB);
            var_230 = wp::add(var_170, var_229);
        }
        var_231 = wp::where(var_176, var_206, var_168);
        var_232 = wp::where(var_176, var_218, var_169);
        var_233 = wp::where(var_176, var_230, var_170);
        var_234 = wp::where(var_176, var_193, var_171);
        // n = particle_neighbors[i, 3]  # +Y                                                     <L 104>
        var_236 = wp::address(var_particle_neighbors, var_0, var_235);
        var_238 = wp::load(var_236);
        var_237 = wp::copy(var_238);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 105>
        var_241 = (var_237 >= var_240);
        var_239 = var_241;
        if (var_239) {
            var_242 = wp::address(var_particle_flags, var_237);
            var_245 = wp::int32(var_244);
            var_247 = wp::load(var_242);
            var_246 = wp::bit_and(var_247, var_245);
            var_249 = (var_246 != var_248);
            var_239 = var_239 && var_249;
        }
        if (var_239) {
            // orie_y = _update_axis(orie_y, 1.0, particle_q[n] - pos_i, fA)                      <L 106>
            var_251 = wp::address(var_particle_q, var_237);
            var_253 = wp::load(var_251);
            var_252 = wp::sub(var_253, var_45);
            var_254 = _update_axis_0(var_232, var_250, var_252, var_fA);
            // nb = orientation_in[n]                                                             <L 107>
            var_255 = wp::address(var_orientation_in, var_237);
            var_257 = wp::load(var_255);
            var_256 = wp::copy(var_257);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 108>
            var_260 = wp::extract(var_256, var_258, var_259);
            var_263 = wp::extract(var_256, var_261, var_262);
            var_266 = wp::extract(var_256, var_264, var_265);
            var_267 = wp::vec_t<3, wp::float32>(var_260, var_263, var_266);
            var_268 = wp::mul(var_267, var_fB);
            var_269 = wp::add(var_231, var_268);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 109>
            var_272 = wp::extract(var_256, var_270, var_271);
            var_275 = wp::extract(var_256, var_273, var_274);
            var_278 = wp::extract(var_256, var_276, var_277);
            var_279 = wp::vec_t<3, wp::float32>(var_272, var_275, var_278);
            var_280 = wp::mul(var_279, var_fB);
            var_281 = wp::add(var_254, var_280);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 110>
            var_284 = wp::extract(var_256, var_282, var_283);
            var_287 = wp::extract(var_256, var_285, var_286);
            var_290 = wp::extract(var_256, var_288, var_289);
            var_291 = wp::vec_t<3, wp::float32>(var_284, var_287, var_290);
            var_292 = wp::mul(var_291, var_fB);
            var_293 = wp::add(var_233, var_292);
        }
        var_294 = wp::where(var_239, var_269, var_231);
        var_295 = wp::where(var_239, var_281, var_232);
        var_296 = wp::where(var_239, var_293, var_233);
        var_297 = wp::where(var_239, var_256, var_234);
        // n = particle_neighbors[i, 4]  # -Z                                                     <L 112>
        var_299 = wp::address(var_particle_neighbors, var_0, var_298);
        var_301 = wp::load(var_299);
        var_300 = wp::copy(var_301);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 113>
        var_304 = (var_300 >= var_303);
        var_302 = var_304;
        if (var_302) {
            var_305 = wp::address(var_particle_flags, var_300);
            var_308 = wp::int32(var_307);
            var_310 = wp::load(var_305);
            var_309 = wp::bit_and(var_310, var_308);
            var_312 = (var_309 != var_311);
            var_302 = var_302 && var_312;
        }
        if (var_302) {
            // orie_z = _update_axis(orie_z, -1.0, particle_q[n] - pos_i, fA)                     <L 114>
            var_314 = wp::address(var_particle_q, var_300);
            var_316 = wp::load(var_314);
            var_315 = wp::sub(var_316, var_45);
            var_317 = _update_axis_0(var_296, var_313, var_315, var_fA);
            // nb = orientation_in[n]                                                             <L 115>
            var_318 = wp::address(var_orientation_in, var_300);
            var_320 = wp::load(var_318);
            var_319 = wp::copy(var_320);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 116>
            var_323 = wp::extract(var_319, var_321, var_322);
            var_326 = wp::extract(var_319, var_324, var_325);
            var_329 = wp::extract(var_319, var_327, var_328);
            var_330 = wp::vec_t<3, wp::float32>(var_323, var_326, var_329);
            var_331 = wp::mul(var_330, var_fB);
            var_332 = wp::add(var_294, var_331);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 117>
            var_335 = wp::extract(var_319, var_333, var_334);
            var_338 = wp::extract(var_319, var_336, var_337);
            var_341 = wp::extract(var_319, var_339, var_340);
            var_342 = wp::vec_t<3, wp::float32>(var_335, var_338, var_341);
            var_343 = wp::mul(var_342, var_fB);
            var_344 = wp::add(var_295, var_343);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 118>
            var_347 = wp::extract(var_319, var_345, var_346);
            var_350 = wp::extract(var_319, var_348, var_349);
            var_353 = wp::extract(var_319, var_351, var_352);
            var_354 = wp::vec_t<3, wp::float32>(var_347, var_350, var_353);
            var_355 = wp::mul(var_354, var_fB);
            var_356 = wp::add(var_317, var_355);
        }
        var_357 = wp::where(var_302, var_332, var_294);
        var_358 = wp::where(var_302, var_344, var_295);
        var_359 = wp::where(var_302, var_356, var_296);
        var_360 = wp::where(var_302, var_319, var_297);
        // n = particle_neighbors[i, 5]  # +Z                                                     <L 120>
        var_362 = wp::address(var_particle_neighbors, var_0, var_361);
        var_364 = wp::load(var_362);
        var_363 = wp::copy(var_364);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 121>
        var_367 = (var_363 >= var_366);
        var_365 = var_367;
        if (var_365) {
            var_368 = wp::address(var_particle_flags, var_363);
            var_371 = wp::int32(var_370);
            var_373 = wp::load(var_368);
            var_372 = wp::bit_and(var_373, var_371);
            var_375 = (var_372 != var_374);
            var_365 = var_365 && var_375;
        }
        if (var_365) {
            // orie_z = _update_axis(orie_z, 1.0, particle_q[n] - pos_i, fA)                      <L 122>
            var_377 = wp::address(var_particle_q, var_363);
            var_379 = wp::load(var_377);
            var_378 = wp::sub(var_379, var_45);
            var_380 = _update_axis_0(var_359, var_376, var_378, var_fA);
            // nb = orientation_in[n]                                                             <L 123>
            var_381 = wp::address(var_orientation_in, var_363);
            var_383 = wp::load(var_381);
            var_382 = wp::copy(var_383);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 124>
            var_386 = wp::extract(var_382, var_384, var_385);
            var_389 = wp::extract(var_382, var_387, var_388);
            var_392 = wp::extract(var_382, var_390, var_391);
            var_393 = wp::vec_t<3, wp::float32>(var_386, var_389, var_392);
            var_394 = wp::mul(var_393, var_fB);
            var_395 = wp::add(var_357, var_394);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 125>
            var_398 = wp::extract(var_382, var_396, var_397);
            var_401 = wp::extract(var_382, var_399, var_400);
            var_404 = wp::extract(var_382, var_402, var_403);
            var_405 = wp::vec_t<3, wp::float32>(var_398, var_401, var_404);
            var_406 = wp::mul(var_405, var_fB);
            var_407 = wp::add(var_358, var_406);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 126>
            var_410 = wp::extract(var_382, var_408, var_409);
            var_413 = wp::extract(var_382, var_411, var_412);
            var_416 = wp::extract(var_382, var_414, var_415);
            var_417 = wp::vec_t<3, wp::float32>(var_410, var_413, var_416);
            var_418 = wp::mul(var_417, var_fB);
            var_419 = wp::add(var_380, var_418);
        }
        var_420 = wp::where(var_365, var_395, var_357);
        var_421 = wp::where(var_365, var_407, var_358);
        var_422 = wp::where(var_365, var_419, var_359);
        var_423 = wp::where(var_365, var_382, var_360);
        // orie_x = _safe_normalize(orie_x)                                                       <L 128>
        var_424 = _safe_normalize_0(var_420);
        // orie_y = _safe_normalize(orie_y)                                                       <L 129>
        var_425 = _safe_normalize_0(var_421);
        // orie_z = _safe_normalize(orie_z)                                                       <L 130>
        var_426 = _safe_normalize_0(var_422);
        // orientation_out[i] = wp.mat33(                                                         <L 132>
        // orie_x[0], orie_x[1], orie_x[2],                                                       <L 133>
        var_428 = wp::extract(var_424, var_427);
        var_430 = wp::extract(var_424, var_429);
        var_432 = wp::extract(var_424, var_431);
        // orie_y[0], orie_y[1], orie_y[2],                                                       <L 134>
        var_434 = wp::extract(var_425, var_433);
        var_436 = wp::extract(var_425, var_435);
        var_438 = wp::extract(var_425, var_437);
        // orie_z[0], orie_z[1], orie_z[2],                                                       <L 135>
        var_440 = wp::extract(var_426, var_439);
        var_442 = wp::extract(var_426, var_441);
        var_444 = wp::extract(var_426, var_443);
        var_445 = wp::mat_t<3, 3, wp::float32>(var_428, var_430, var_432, var_434, var_436, var_438, var_440, var_442, var_444);
        // orientation_out[i] = wp.mat33(                                                         <L 132>
        wp::array_store(var_orientation_out, var_0, var_445);
    }
}



extern "C" __global__ void update_orientation_kernel_419bec28_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::vec_t<3, wp::float32>> var_particle_q,
    wp::array_t<wp::int32> var_particle_flags,
    wp::array_t<wp::int32> var_particle_neighbors,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_orientation_in,
    wp::float32 var_fA,
    wp::float32 var_fB,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_orientation_out,
    wp::array_t<wp::vec_t<3, wp::float32>> adj_particle_q,
    wp::array_t<wp::int32> adj_particle_flags,
    wp::array_t<wp::int32> adj_particle_neighbors,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_orientation_in,
    wp::float32 adj_fA,
    wp::float32 adj_fB,
    wp::array_t<wp::mat_t<3, 3, wp::float32>> adj_orientation_out)
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
        wp::mat_t<3, 3, wp::float32>* var_9;
        wp::mat_t<3, 3, wp::float32> var_10;
        wp::mat_t<3, 3, wp::float32>* var_11;
        wp::mat_t<3, 3, wp::float32> var_12;
        wp::mat_t<3, 3, wp::float32> var_13;
        const wp::int32 var_14 = 0;
        const wp::int32 var_15 = 0;
        wp::float32 var_16;
        const wp::int32 var_17 = 0;
        const wp::int32 var_18 = 1;
        wp::float32 var_19;
        const wp::int32 var_20 = 0;
        const wp::int32 var_21 = 2;
        wp::float32 var_22;
        wp::vec_t<3, wp::float32> var_23;
        const wp::int32 var_24 = 1;
        const wp::int32 var_25 = 0;
        wp::float32 var_26;
        const wp::int32 var_27 = 1;
        const wp::int32 var_28 = 1;
        wp::float32 var_29;
        const wp::int32 var_30 = 1;
        const wp::int32 var_31 = 2;
        wp::float32 var_32;
        wp::vec_t<3, wp::float32> var_33;
        const wp::int32 var_34 = 2;
        const wp::int32 var_35 = 0;
        wp::float32 var_36;
        const wp::int32 var_37 = 2;
        const wp::int32 var_38 = 1;
        wp::float32 var_39;
        const wp::int32 var_40 = 2;
        const wp::int32 var_41 = 2;
        wp::float32 var_42;
        wp::vec_t<3, wp::float32> var_43;
        wp::vec_t<3, wp::float32>* var_44;
        wp::vec_t<3, wp::float32> var_45;
        wp::vec_t<3, wp::float32> var_46;
        const wp::int32 var_47 = 0;
        wp::int32* var_48;
        wp::int32 var_49;
        wp::int32 var_50;
        bool var_51;
        const wp::int32 var_52 = 0;
        bool var_53;
        wp::int32* var_54;
        const wp::int32 var_55 = 1;
        const wp::int32 var_56 = 1;
        wp::int32 var_57;
        wp::int32 var_58;
        wp::int32 var_59;
        const wp::int32 var_60 = 0;
        bool var_61;
        const wp::float32 var_62 = -1.0;
        wp::vec_t<3, wp::float32>* var_63;
        wp::vec_t<3, wp::float32> var_64;
        wp::vec_t<3, wp::float32> var_65;
        wp::vec_t<3, wp::float32> var_66;
        wp::mat_t<3, 3, wp::float32>* var_67;
        wp::mat_t<3, 3, wp::float32> var_68;
        wp::mat_t<3, 3, wp::float32> var_69;
        const wp::int32 var_70 = 0;
        const wp::int32 var_71 = 0;
        wp::float32 var_72;
        const wp::int32 var_73 = 0;
        const wp::int32 var_74 = 1;
        wp::float32 var_75;
        const wp::int32 var_76 = 0;
        const wp::int32 var_77 = 2;
        wp::float32 var_78;
        wp::vec_t<3, wp::float32> var_79;
        wp::vec_t<3, wp::float32> var_80;
        wp::vec_t<3, wp::float32> var_81;
        const wp::int32 var_82 = 1;
        const wp::int32 var_83 = 0;
        wp::float32 var_84;
        const wp::int32 var_85 = 1;
        const wp::int32 var_86 = 1;
        wp::float32 var_87;
        const wp::int32 var_88 = 1;
        const wp::int32 var_89 = 2;
        wp::float32 var_90;
        wp::vec_t<3, wp::float32> var_91;
        wp::vec_t<3, wp::float32> var_92;
        wp::vec_t<3, wp::float32> var_93;
        const wp::int32 var_94 = 2;
        const wp::int32 var_95 = 0;
        wp::float32 var_96;
        const wp::int32 var_97 = 2;
        const wp::int32 var_98 = 1;
        wp::float32 var_99;
        const wp::int32 var_100 = 2;
        const wp::int32 var_101 = 2;
        wp::float32 var_102;
        wp::vec_t<3, wp::float32> var_103;
        wp::vec_t<3, wp::float32> var_104;
        wp::vec_t<3, wp::float32> var_105;
        wp::vec_t<3, wp::float32> var_106;
        wp::vec_t<3, wp::float32> var_107;
        wp::vec_t<3, wp::float32> var_108;
        const wp::int32 var_109 = 1;
        wp::int32* var_110;
        wp::int32 var_111;
        wp::int32 var_112;
        bool var_113;
        const wp::int32 var_114 = 0;
        bool var_115;
        wp::int32* var_116;
        const wp::int32 var_117 = 1;
        const wp::int32 var_118 = 1;
        wp::int32 var_119;
        wp::int32 var_120;
        wp::int32 var_121;
        const wp::int32 var_122 = 0;
        bool var_123;
        const wp::float32 var_124 = 1.0;
        wp::vec_t<3, wp::float32>* var_125;
        wp::vec_t<3, wp::float32> var_126;
        wp::vec_t<3, wp::float32> var_127;
        wp::vec_t<3, wp::float32> var_128;
        wp::mat_t<3, 3, wp::float32>* var_129;
        wp::mat_t<3, 3, wp::float32> var_130;
        wp::mat_t<3, 3, wp::float32> var_131;
        const wp::int32 var_132 = 0;
        const wp::int32 var_133 = 0;
        wp::float32 var_134;
        const wp::int32 var_135 = 0;
        const wp::int32 var_136 = 1;
        wp::float32 var_137;
        const wp::int32 var_138 = 0;
        const wp::int32 var_139 = 2;
        wp::float32 var_140;
        wp::vec_t<3, wp::float32> var_141;
        wp::vec_t<3, wp::float32> var_142;
        wp::vec_t<3, wp::float32> var_143;
        const wp::int32 var_144 = 1;
        const wp::int32 var_145 = 0;
        wp::float32 var_146;
        const wp::int32 var_147 = 1;
        const wp::int32 var_148 = 1;
        wp::float32 var_149;
        const wp::int32 var_150 = 1;
        const wp::int32 var_151 = 2;
        wp::float32 var_152;
        wp::vec_t<3, wp::float32> var_153;
        wp::vec_t<3, wp::float32> var_154;
        wp::vec_t<3, wp::float32> var_155;
        const wp::int32 var_156 = 2;
        const wp::int32 var_157 = 0;
        wp::float32 var_158;
        const wp::int32 var_159 = 2;
        const wp::int32 var_160 = 1;
        wp::float32 var_161;
        const wp::int32 var_162 = 2;
        const wp::int32 var_163 = 2;
        wp::float32 var_164;
        wp::vec_t<3, wp::float32> var_165;
        wp::vec_t<3, wp::float32> var_166;
        wp::vec_t<3, wp::float32> var_167;
        wp::vec_t<3, wp::float32> var_168;
        wp::vec_t<3, wp::float32> var_169;
        wp::vec_t<3, wp::float32> var_170;
        wp::mat_t<3, 3, wp::float32> var_171;
        const wp::int32 var_172 = 2;
        wp::int32* var_173;
        wp::int32 var_174;
        wp::int32 var_175;
        bool var_176;
        const wp::int32 var_177 = 0;
        bool var_178;
        wp::int32* var_179;
        const wp::int32 var_180 = 1;
        const wp::int32 var_181 = 1;
        wp::int32 var_182;
        wp::int32 var_183;
        wp::int32 var_184;
        const wp::int32 var_185 = 0;
        bool var_186;
        const wp::float32 var_187 = -1.0;
        wp::vec_t<3, wp::float32>* var_188;
        wp::vec_t<3, wp::float32> var_189;
        wp::vec_t<3, wp::float32> var_190;
        wp::vec_t<3, wp::float32> var_191;
        wp::mat_t<3, 3, wp::float32>* var_192;
        wp::mat_t<3, 3, wp::float32> var_193;
        wp::mat_t<3, 3, wp::float32> var_194;
        const wp::int32 var_195 = 0;
        const wp::int32 var_196 = 0;
        wp::float32 var_197;
        const wp::int32 var_198 = 0;
        const wp::int32 var_199 = 1;
        wp::float32 var_200;
        const wp::int32 var_201 = 0;
        const wp::int32 var_202 = 2;
        wp::float32 var_203;
        wp::vec_t<3, wp::float32> var_204;
        wp::vec_t<3, wp::float32> var_205;
        wp::vec_t<3, wp::float32> var_206;
        const wp::int32 var_207 = 1;
        const wp::int32 var_208 = 0;
        wp::float32 var_209;
        const wp::int32 var_210 = 1;
        const wp::int32 var_211 = 1;
        wp::float32 var_212;
        const wp::int32 var_213 = 1;
        const wp::int32 var_214 = 2;
        wp::float32 var_215;
        wp::vec_t<3, wp::float32> var_216;
        wp::vec_t<3, wp::float32> var_217;
        wp::vec_t<3, wp::float32> var_218;
        const wp::int32 var_219 = 2;
        const wp::int32 var_220 = 0;
        wp::float32 var_221;
        const wp::int32 var_222 = 2;
        const wp::int32 var_223 = 1;
        wp::float32 var_224;
        const wp::int32 var_225 = 2;
        const wp::int32 var_226 = 2;
        wp::float32 var_227;
        wp::vec_t<3, wp::float32> var_228;
        wp::vec_t<3, wp::float32> var_229;
        wp::vec_t<3, wp::float32> var_230;
        wp::vec_t<3, wp::float32> var_231;
        wp::vec_t<3, wp::float32> var_232;
        wp::vec_t<3, wp::float32> var_233;
        wp::mat_t<3, 3, wp::float32> var_234;
        const wp::int32 var_235 = 3;
        wp::int32* var_236;
        wp::int32 var_237;
        wp::int32 var_238;
        bool var_239;
        const wp::int32 var_240 = 0;
        bool var_241;
        wp::int32* var_242;
        const wp::int32 var_243 = 1;
        const wp::int32 var_244 = 1;
        wp::int32 var_245;
        wp::int32 var_246;
        wp::int32 var_247;
        const wp::int32 var_248 = 0;
        bool var_249;
        const wp::float32 var_250 = 1.0;
        wp::vec_t<3, wp::float32>* var_251;
        wp::vec_t<3, wp::float32> var_252;
        wp::vec_t<3, wp::float32> var_253;
        wp::vec_t<3, wp::float32> var_254;
        wp::mat_t<3, 3, wp::float32>* var_255;
        wp::mat_t<3, 3, wp::float32> var_256;
        wp::mat_t<3, 3, wp::float32> var_257;
        const wp::int32 var_258 = 0;
        const wp::int32 var_259 = 0;
        wp::float32 var_260;
        const wp::int32 var_261 = 0;
        const wp::int32 var_262 = 1;
        wp::float32 var_263;
        const wp::int32 var_264 = 0;
        const wp::int32 var_265 = 2;
        wp::float32 var_266;
        wp::vec_t<3, wp::float32> var_267;
        wp::vec_t<3, wp::float32> var_268;
        wp::vec_t<3, wp::float32> var_269;
        const wp::int32 var_270 = 1;
        const wp::int32 var_271 = 0;
        wp::float32 var_272;
        const wp::int32 var_273 = 1;
        const wp::int32 var_274 = 1;
        wp::float32 var_275;
        const wp::int32 var_276 = 1;
        const wp::int32 var_277 = 2;
        wp::float32 var_278;
        wp::vec_t<3, wp::float32> var_279;
        wp::vec_t<3, wp::float32> var_280;
        wp::vec_t<3, wp::float32> var_281;
        const wp::int32 var_282 = 2;
        const wp::int32 var_283 = 0;
        wp::float32 var_284;
        const wp::int32 var_285 = 2;
        const wp::int32 var_286 = 1;
        wp::float32 var_287;
        const wp::int32 var_288 = 2;
        const wp::int32 var_289 = 2;
        wp::float32 var_290;
        wp::vec_t<3, wp::float32> var_291;
        wp::vec_t<3, wp::float32> var_292;
        wp::vec_t<3, wp::float32> var_293;
        wp::vec_t<3, wp::float32> var_294;
        wp::vec_t<3, wp::float32> var_295;
        wp::vec_t<3, wp::float32> var_296;
        wp::mat_t<3, 3, wp::float32> var_297;
        const wp::int32 var_298 = 4;
        wp::int32* var_299;
        wp::int32 var_300;
        wp::int32 var_301;
        bool var_302;
        const wp::int32 var_303 = 0;
        bool var_304;
        wp::int32* var_305;
        const wp::int32 var_306 = 1;
        const wp::int32 var_307 = 1;
        wp::int32 var_308;
        wp::int32 var_309;
        wp::int32 var_310;
        const wp::int32 var_311 = 0;
        bool var_312;
        const wp::float32 var_313 = -1.0;
        wp::vec_t<3, wp::float32>* var_314;
        wp::vec_t<3, wp::float32> var_315;
        wp::vec_t<3, wp::float32> var_316;
        wp::vec_t<3, wp::float32> var_317;
        wp::mat_t<3, 3, wp::float32>* var_318;
        wp::mat_t<3, 3, wp::float32> var_319;
        wp::mat_t<3, 3, wp::float32> var_320;
        const wp::int32 var_321 = 0;
        const wp::int32 var_322 = 0;
        wp::float32 var_323;
        const wp::int32 var_324 = 0;
        const wp::int32 var_325 = 1;
        wp::float32 var_326;
        const wp::int32 var_327 = 0;
        const wp::int32 var_328 = 2;
        wp::float32 var_329;
        wp::vec_t<3, wp::float32> var_330;
        wp::vec_t<3, wp::float32> var_331;
        wp::vec_t<3, wp::float32> var_332;
        const wp::int32 var_333 = 1;
        const wp::int32 var_334 = 0;
        wp::float32 var_335;
        const wp::int32 var_336 = 1;
        const wp::int32 var_337 = 1;
        wp::float32 var_338;
        const wp::int32 var_339 = 1;
        const wp::int32 var_340 = 2;
        wp::float32 var_341;
        wp::vec_t<3, wp::float32> var_342;
        wp::vec_t<3, wp::float32> var_343;
        wp::vec_t<3, wp::float32> var_344;
        const wp::int32 var_345 = 2;
        const wp::int32 var_346 = 0;
        wp::float32 var_347;
        const wp::int32 var_348 = 2;
        const wp::int32 var_349 = 1;
        wp::float32 var_350;
        const wp::int32 var_351 = 2;
        const wp::int32 var_352 = 2;
        wp::float32 var_353;
        wp::vec_t<3, wp::float32> var_354;
        wp::vec_t<3, wp::float32> var_355;
        wp::vec_t<3, wp::float32> var_356;
        wp::vec_t<3, wp::float32> var_357;
        wp::vec_t<3, wp::float32> var_358;
        wp::vec_t<3, wp::float32> var_359;
        wp::mat_t<3, 3, wp::float32> var_360;
        const wp::int32 var_361 = 5;
        wp::int32* var_362;
        wp::int32 var_363;
        wp::int32 var_364;
        bool var_365;
        const wp::int32 var_366 = 0;
        bool var_367;
        wp::int32* var_368;
        const wp::int32 var_369 = 1;
        const wp::int32 var_370 = 1;
        wp::int32 var_371;
        wp::int32 var_372;
        wp::int32 var_373;
        const wp::int32 var_374 = 0;
        bool var_375;
        const wp::float32 var_376 = 1.0;
        wp::vec_t<3, wp::float32>* var_377;
        wp::vec_t<3, wp::float32> var_378;
        wp::vec_t<3, wp::float32> var_379;
        wp::vec_t<3, wp::float32> var_380;
        wp::mat_t<3, 3, wp::float32>* var_381;
        wp::mat_t<3, 3, wp::float32> var_382;
        wp::mat_t<3, 3, wp::float32> var_383;
        const wp::int32 var_384 = 0;
        const wp::int32 var_385 = 0;
        wp::float32 var_386;
        const wp::int32 var_387 = 0;
        const wp::int32 var_388 = 1;
        wp::float32 var_389;
        const wp::int32 var_390 = 0;
        const wp::int32 var_391 = 2;
        wp::float32 var_392;
        wp::vec_t<3, wp::float32> var_393;
        wp::vec_t<3, wp::float32> var_394;
        wp::vec_t<3, wp::float32> var_395;
        const wp::int32 var_396 = 1;
        const wp::int32 var_397 = 0;
        wp::float32 var_398;
        const wp::int32 var_399 = 1;
        const wp::int32 var_400 = 1;
        wp::float32 var_401;
        const wp::int32 var_402 = 1;
        const wp::int32 var_403 = 2;
        wp::float32 var_404;
        wp::vec_t<3, wp::float32> var_405;
        wp::vec_t<3, wp::float32> var_406;
        wp::vec_t<3, wp::float32> var_407;
        const wp::int32 var_408 = 2;
        const wp::int32 var_409 = 0;
        wp::float32 var_410;
        const wp::int32 var_411 = 2;
        const wp::int32 var_412 = 1;
        wp::float32 var_413;
        const wp::int32 var_414 = 2;
        const wp::int32 var_415 = 2;
        wp::float32 var_416;
        wp::vec_t<3, wp::float32> var_417;
        wp::vec_t<3, wp::float32> var_418;
        wp::vec_t<3, wp::float32> var_419;
        wp::vec_t<3, wp::float32> var_420;
        wp::vec_t<3, wp::float32> var_421;
        wp::vec_t<3, wp::float32> var_422;
        wp::mat_t<3, 3, wp::float32> var_423;
        wp::vec_t<3, wp::float32> var_424;
        wp::vec_t<3, wp::float32> var_425;
        wp::vec_t<3, wp::float32> var_426;
        const wp::int32 var_427 = 0;
        wp::float32 var_428;
        const wp::int32 var_429 = 1;
        wp::float32 var_430;
        const wp::int32 var_431 = 2;
        wp::float32 var_432;
        const wp::int32 var_433 = 0;
        wp::float32 var_434;
        const wp::int32 var_435 = 1;
        wp::float32 var_436;
        const wp::int32 var_437 = 2;
        wp::float32 var_438;
        const wp::int32 var_439 = 0;
        wp::float32 var_440;
        const wp::int32 var_441 = 1;
        wp::float32 var_442;
        const wp::int32 var_443 = 2;
        wp::float32 var_444;
        wp::mat_t<3, 3, wp::float32> var_445;
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
        wp::mat_t<3, 3, wp::float32> adj_9 = {};
        wp::mat_t<3, 3, wp::float32> adj_10 = {};
        wp::mat_t<3, 3, wp::float32> adj_11 = {};
        wp::mat_t<3, 3, wp::float32> adj_12 = {};
        wp::mat_t<3, 3, wp::float32> adj_13 = {};
        wp::int32 adj_14 = {};
        wp::int32 adj_15 = {};
        wp::float32 adj_16 = {};
        wp::int32 adj_17 = {};
        wp::int32 adj_18 = {};
        wp::float32 adj_19 = {};
        wp::int32 adj_20 = {};
        wp::int32 adj_21 = {};
        wp::float32 adj_22 = {};
        wp::vec_t<3, wp::float32> adj_23 = {};
        wp::int32 adj_24 = {};
        wp::int32 adj_25 = {};
        wp::float32 adj_26 = {};
        wp::int32 adj_27 = {};
        wp::int32 adj_28 = {};
        wp::float32 adj_29 = {};
        wp::int32 adj_30 = {};
        wp::int32 adj_31 = {};
        wp::float32 adj_32 = {};
        wp::vec_t<3, wp::float32> adj_33 = {};
        wp::int32 adj_34 = {};
        wp::int32 adj_35 = {};
        wp::float32 adj_36 = {};
        wp::int32 adj_37 = {};
        wp::int32 adj_38 = {};
        wp::float32 adj_39 = {};
        wp::int32 adj_40 = {};
        wp::int32 adj_41 = {};
        wp::float32 adj_42 = {};
        wp::vec_t<3, wp::float32> adj_43 = {};
        wp::vec_t<3, wp::float32> adj_44 = {};
        wp::vec_t<3, wp::float32> adj_45 = {};
        wp::vec_t<3, wp::float32> adj_46 = {};
        wp::int32 adj_47 = {};
        wp::int32 adj_48 = {};
        wp::int32 adj_49 = {};
        wp::int32 adj_50 = {};
        bool adj_51 = {};
        wp::int32 adj_52 = {};
        bool adj_53 = {};
        wp::int32 adj_54 = {};
        wp::int32 adj_55 = {};
        wp::int32 adj_56 = {};
        wp::int32 adj_57 = {};
        wp::int32 adj_58 = {};
        wp::int32 adj_59 = {};
        wp::int32 adj_60 = {};
        bool adj_61 = {};
        wp::float32 adj_62 = {};
        wp::vec_t<3, wp::float32> adj_63 = {};
        wp::vec_t<3, wp::float32> adj_64 = {};
        wp::vec_t<3, wp::float32> adj_65 = {};
        wp::vec_t<3, wp::float32> adj_66 = {};
        wp::mat_t<3, 3, wp::float32> adj_67 = {};
        wp::mat_t<3, 3, wp::float32> adj_68 = {};
        wp::mat_t<3, 3, wp::float32> adj_69 = {};
        wp::int32 adj_70 = {};
        wp::int32 adj_71 = {};
        wp::float32 adj_72 = {};
        wp::int32 adj_73 = {};
        wp::int32 adj_74 = {};
        wp::float32 adj_75 = {};
        wp::int32 adj_76 = {};
        wp::int32 adj_77 = {};
        wp::float32 adj_78 = {};
        wp::vec_t<3, wp::float32> adj_79 = {};
        wp::vec_t<3, wp::float32> adj_80 = {};
        wp::vec_t<3, wp::float32> adj_81 = {};
        wp::int32 adj_82 = {};
        wp::int32 adj_83 = {};
        wp::float32 adj_84 = {};
        wp::int32 adj_85 = {};
        wp::int32 adj_86 = {};
        wp::float32 adj_87 = {};
        wp::int32 adj_88 = {};
        wp::int32 adj_89 = {};
        wp::float32 adj_90 = {};
        wp::vec_t<3, wp::float32> adj_91 = {};
        wp::vec_t<3, wp::float32> adj_92 = {};
        wp::vec_t<3, wp::float32> adj_93 = {};
        wp::int32 adj_94 = {};
        wp::int32 adj_95 = {};
        wp::float32 adj_96 = {};
        wp::int32 adj_97 = {};
        wp::int32 adj_98 = {};
        wp::float32 adj_99 = {};
        wp::int32 adj_100 = {};
        wp::int32 adj_101 = {};
        wp::float32 adj_102 = {};
        wp::vec_t<3, wp::float32> adj_103 = {};
        wp::vec_t<3, wp::float32> adj_104 = {};
        wp::vec_t<3, wp::float32> adj_105 = {};
        wp::vec_t<3, wp::float32> adj_106 = {};
        wp::vec_t<3, wp::float32> adj_107 = {};
        wp::vec_t<3, wp::float32> adj_108 = {};
        wp::int32 adj_109 = {};
        wp::int32 adj_110 = {};
        wp::int32 adj_111 = {};
        wp::int32 adj_112 = {};
        bool adj_113 = {};
        wp::int32 adj_114 = {};
        bool adj_115 = {};
        wp::int32 adj_116 = {};
        wp::int32 adj_117 = {};
        wp::int32 adj_118 = {};
        wp::int32 adj_119 = {};
        wp::int32 adj_120 = {};
        wp::int32 adj_121 = {};
        wp::int32 adj_122 = {};
        bool adj_123 = {};
        wp::float32 adj_124 = {};
        wp::vec_t<3, wp::float32> adj_125 = {};
        wp::vec_t<3, wp::float32> adj_126 = {};
        wp::vec_t<3, wp::float32> adj_127 = {};
        wp::vec_t<3, wp::float32> adj_128 = {};
        wp::mat_t<3, 3, wp::float32> adj_129 = {};
        wp::mat_t<3, 3, wp::float32> adj_130 = {};
        wp::mat_t<3, 3, wp::float32> adj_131 = {};
        wp::int32 adj_132 = {};
        wp::int32 adj_133 = {};
        wp::float32 adj_134 = {};
        wp::int32 adj_135 = {};
        wp::int32 adj_136 = {};
        wp::float32 adj_137 = {};
        wp::int32 adj_138 = {};
        wp::int32 adj_139 = {};
        wp::float32 adj_140 = {};
        wp::vec_t<3, wp::float32> adj_141 = {};
        wp::vec_t<3, wp::float32> adj_142 = {};
        wp::vec_t<3, wp::float32> adj_143 = {};
        wp::int32 adj_144 = {};
        wp::int32 adj_145 = {};
        wp::float32 adj_146 = {};
        wp::int32 adj_147 = {};
        wp::int32 adj_148 = {};
        wp::float32 adj_149 = {};
        wp::int32 adj_150 = {};
        wp::int32 adj_151 = {};
        wp::float32 adj_152 = {};
        wp::vec_t<3, wp::float32> adj_153 = {};
        wp::vec_t<3, wp::float32> adj_154 = {};
        wp::vec_t<3, wp::float32> adj_155 = {};
        wp::int32 adj_156 = {};
        wp::int32 adj_157 = {};
        wp::float32 adj_158 = {};
        wp::int32 adj_159 = {};
        wp::int32 adj_160 = {};
        wp::float32 adj_161 = {};
        wp::int32 adj_162 = {};
        wp::int32 adj_163 = {};
        wp::float32 adj_164 = {};
        wp::vec_t<3, wp::float32> adj_165 = {};
        wp::vec_t<3, wp::float32> adj_166 = {};
        wp::vec_t<3, wp::float32> adj_167 = {};
        wp::vec_t<3, wp::float32> adj_168 = {};
        wp::vec_t<3, wp::float32> adj_169 = {};
        wp::vec_t<3, wp::float32> adj_170 = {};
        wp::mat_t<3, 3, wp::float32> adj_171 = {};
        wp::int32 adj_172 = {};
        wp::int32 adj_173 = {};
        wp::int32 adj_174 = {};
        wp::int32 adj_175 = {};
        bool adj_176 = {};
        wp::int32 adj_177 = {};
        bool adj_178 = {};
        wp::int32 adj_179 = {};
        wp::int32 adj_180 = {};
        wp::int32 adj_181 = {};
        wp::int32 adj_182 = {};
        wp::int32 adj_183 = {};
        wp::int32 adj_184 = {};
        wp::int32 adj_185 = {};
        bool adj_186 = {};
        wp::float32 adj_187 = {};
        wp::vec_t<3, wp::float32> adj_188 = {};
        wp::vec_t<3, wp::float32> adj_189 = {};
        wp::vec_t<3, wp::float32> adj_190 = {};
        wp::vec_t<3, wp::float32> adj_191 = {};
        wp::mat_t<3, 3, wp::float32> adj_192 = {};
        wp::mat_t<3, 3, wp::float32> adj_193 = {};
        wp::mat_t<3, 3, wp::float32> adj_194 = {};
        wp::int32 adj_195 = {};
        wp::int32 adj_196 = {};
        wp::float32 adj_197 = {};
        wp::int32 adj_198 = {};
        wp::int32 adj_199 = {};
        wp::float32 adj_200 = {};
        wp::int32 adj_201 = {};
        wp::int32 adj_202 = {};
        wp::float32 adj_203 = {};
        wp::vec_t<3, wp::float32> adj_204 = {};
        wp::vec_t<3, wp::float32> adj_205 = {};
        wp::vec_t<3, wp::float32> adj_206 = {};
        wp::int32 adj_207 = {};
        wp::int32 adj_208 = {};
        wp::float32 adj_209 = {};
        wp::int32 adj_210 = {};
        wp::int32 adj_211 = {};
        wp::float32 adj_212 = {};
        wp::int32 adj_213 = {};
        wp::int32 adj_214 = {};
        wp::float32 adj_215 = {};
        wp::vec_t<3, wp::float32> adj_216 = {};
        wp::vec_t<3, wp::float32> adj_217 = {};
        wp::vec_t<3, wp::float32> adj_218 = {};
        wp::int32 adj_219 = {};
        wp::int32 adj_220 = {};
        wp::float32 adj_221 = {};
        wp::int32 adj_222 = {};
        wp::int32 adj_223 = {};
        wp::float32 adj_224 = {};
        wp::int32 adj_225 = {};
        wp::int32 adj_226 = {};
        wp::float32 adj_227 = {};
        wp::vec_t<3, wp::float32> adj_228 = {};
        wp::vec_t<3, wp::float32> adj_229 = {};
        wp::vec_t<3, wp::float32> adj_230 = {};
        wp::vec_t<3, wp::float32> adj_231 = {};
        wp::vec_t<3, wp::float32> adj_232 = {};
        wp::vec_t<3, wp::float32> adj_233 = {};
        wp::mat_t<3, 3, wp::float32> adj_234 = {};
        wp::int32 adj_235 = {};
        wp::int32 adj_236 = {};
        wp::int32 adj_237 = {};
        wp::int32 adj_238 = {};
        bool adj_239 = {};
        wp::int32 adj_240 = {};
        bool adj_241 = {};
        wp::int32 adj_242 = {};
        wp::int32 adj_243 = {};
        wp::int32 adj_244 = {};
        wp::int32 adj_245 = {};
        wp::int32 adj_246 = {};
        wp::int32 adj_247 = {};
        wp::int32 adj_248 = {};
        bool adj_249 = {};
        wp::float32 adj_250 = {};
        wp::vec_t<3, wp::float32> adj_251 = {};
        wp::vec_t<3, wp::float32> adj_252 = {};
        wp::vec_t<3, wp::float32> adj_253 = {};
        wp::vec_t<3, wp::float32> adj_254 = {};
        wp::mat_t<3, 3, wp::float32> adj_255 = {};
        wp::mat_t<3, 3, wp::float32> adj_256 = {};
        wp::mat_t<3, 3, wp::float32> adj_257 = {};
        wp::int32 adj_258 = {};
        wp::int32 adj_259 = {};
        wp::float32 adj_260 = {};
        wp::int32 adj_261 = {};
        wp::int32 adj_262 = {};
        wp::float32 adj_263 = {};
        wp::int32 adj_264 = {};
        wp::int32 adj_265 = {};
        wp::float32 adj_266 = {};
        wp::vec_t<3, wp::float32> adj_267 = {};
        wp::vec_t<3, wp::float32> adj_268 = {};
        wp::vec_t<3, wp::float32> adj_269 = {};
        wp::int32 adj_270 = {};
        wp::int32 adj_271 = {};
        wp::float32 adj_272 = {};
        wp::int32 adj_273 = {};
        wp::int32 adj_274 = {};
        wp::float32 adj_275 = {};
        wp::int32 adj_276 = {};
        wp::int32 adj_277 = {};
        wp::float32 adj_278 = {};
        wp::vec_t<3, wp::float32> adj_279 = {};
        wp::vec_t<3, wp::float32> adj_280 = {};
        wp::vec_t<3, wp::float32> adj_281 = {};
        wp::int32 adj_282 = {};
        wp::int32 adj_283 = {};
        wp::float32 adj_284 = {};
        wp::int32 adj_285 = {};
        wp::int32 adj_286 = {};
        wp::float32 adj_287 = {};
        wp::int32 adj_288 = {};
        wp::int32 adj_289 = {};
        wp::float32 adj_290 = {};
        wp::vec_t<3, wp::float32> adj_291 = {};
        wp::vec_t<3, wp::float32> adj_292 = {};
        wp::vec_t<3, wp::float32> adj_293 = {};
        wp::vec_t<3, wp::float32> adj_294 = {};
        wp::vec_t<3, wp::float32> adj_295 = {};
        wp::vec_t<3, wp::float32> adj_296 = {};
        wp::mat_t<3, 3, wp::float32> adj_297 = {};
        wp::int32 adj_298 = {};
        wp::int32 adj_299 = {};
        wp::int32 adj_300 = {};
        wp::int32 adj_301 = {};
        bool adj_302 = {};
        wp::int32 adj_303 = {};
        bool adj_304 = {};
        wp::int32 adj_305 = {};
        wp::int32 adj_306 = {};
        wp::int32 adj_307 = {};
        wp::int32 adj_308 = {};
        wp::int32 adj_309 = {};
        wp::int32 adj_310 = {};
        wp::int32 adj_311 = {};
        bool adj_312 = {};
        wp::float32 adj_313 = {};
        wp::vec_t<3, wp::float32> adj_314 = {};
        wp::vec_t<3, wp::float32> adj_315 = {};
        wp::vec_t<3, wp::float32> adj_316 = {};
        wp::vec_t<3, wp::float32> adj_317 = {};
        wp::mat_t<3, 3, wp::float32> adj_318 = {};
        wp::mat_t<3, 3, wp::float32> adj_319 = {};
        wp::mat_t<3, 3, wp::float32> adj_320 = {};
        wp::int32 adj_321 = {};
        wp::int32 adj_322 = {};
        wp::float32 adj_323 = {};
        wp::int32 adj_324 = {};
        wp::int32 adj_325 = {};
        wp::float32 adj_326 = {};
        wp::int32 adj_327 = {};
        wp::int32 adj_328 = {};
        wp::float32 adj_329 = {};
        wp::vec_t<3, wp::float32> adj_330 = {};
        wp::vec_t<3, wp::float32> adj_331 = {};
        wp::vec_t<3, wp::float32> adj_332 = {};
        wp::int32 adj_333 = {};
        wp::int32 adj_334 = {};
        wp::float32 adj_335 = {};
        wp::int32 adj_336 = {};
        wp::int32 adj_337 = {};
        wp::float32 adj_338 = {};
        wp::int32 adj_339 = {};
        wp::int32 adj_340 = {};
        wp::float32 adj_341 = {};
        wp::vec_t<3, wp::float32> adj_342 = {};
        wp::vec_t<3, wp::float32> adj_343 = {};
        wp::vec_t<3, wp::float32> adj_344 = {};
        wp::int32 adj_345 = {};
        wp::int32 adj_346 = {};
        wp::float32 adj_347 = {};
        wp::int32 adj_348 = {};
        wp::int32 adj_349 = {};
        wp::float32 adj_350 = {};
        wp::int32 adj_351 = {};
        wp::int32 adj_352 = {};
        wp::float32 adj_353 = {};
        wp::vec_t<3, wp::float32> adj_354 = {};
        wp::vec_t<3, wp::float32> adj_355 = {};
        wp::vec_t<3, wp::float32> adj_356 = {};
        wp::vec_t<3, wp::float32> adj_357 = {};
        wp::vec_t<3, wp::float32> adj_358 = {};
        wp::vec_t<3, wp::float32> adj_359 = {};
        wp::mat_t<3, 3, wp::float32> adj_360 = {};
        wp::int32 adj_361 = {};
        wp::int32 adj_362 = {};
        wp::int32 adj_363 = {};
        wp::int32 adj_364 = {};
        bool adj_365 = {};
        wp::int32 adj_366 = {};
        bool adj_367 = {};
        wp::int32 adj_368 = {};
        wp::int32 adj_369 = {};
        wp::int32 adj_370 = {};
        wp::int32 adj_371 = {};
        wp::int32 adj_372 = {};
        wp::int32 adj_373 = {};
        wp::int32 adj_374 = {};
        bool adj_375 = {};
        wp::float32 adj_376 = {};
        wp::vec_t<3, wp::float32> adj_377 = {};
        wp::vec_t<3, wp::float32> adj_378 = {};
        wp::vec_t<3, wp::float32> adj_379 = {};
        wp::vec_t<3, wp::float32> adj_380 = {};
        wp::mat_t<3, 3, wp::float32> adj_381 = {};
        wp::mat_t<3, 3, wp::float32> adj_382 = {};
        wp::mat_t<3, 3, wp::float32> adj_383 = {};
        wp::int32 adj_384 = {};
        wp::int32 adj_385 = {};
        wp::float32 adj_386 = {};
        wp::int32 adj_387 = {};
        wp::int32 adj_388 = {};
        wp::float32 adj_389 = {};
        wp::int32 adj_390 = {};
        wp::int32 adj_391 = {};
        wp::float32 adj_392 = {};
        wp::vec_t<3, wp::float32> adj_393 = {};
        wp::vec_t<3, wp::float32> adj_394 = {};
        wp::vec_t<3, wp::float32> adj_395 = {};
        wp::int32 adj_396 = {};
        wp::int32 adj_397 = {};
        wp::float32 adj_398 = {};
        wp::int32 adj_399 = {};
        wp::int32 adj_400 = {};
        wp::float32 adj_401 = {};
        wp::int32 adj_402 = {};
        wp::int32 adj_403 = {};
        wp::float32 adj_404 = {};
        wp::vec_t<3, wp::float32> adj_405 = {};
        wp::vec_t<3, wp::float32> adj_406 = {};
        wp::vec_t<3, wp::float32> adj_407 = {};
        wp::int32 adj_408 = {};
        wp::int32 adj_409 = {};
        wp::float32 adj_410 = {};
        wp::int32 adj_411 = {};
        wp::int32 adj_412 = {};
        wp::float32 adj_413 = {};
        wp::int32 adj_414 = {};
        wp::int32 adj_415 = {};
        wp::float32 adj_416 = {};
        wp::vec_t<3, wp::float32> adj_417 = {};
        wp::vec_t<3, wp::float32> adj_418 = {};
        wp::vec_t<3, wp::float32> adj_419 = {};
        wp::vec_t<3, wp::float32> adj_420 = {};
        wp::vec_t<3, wp::float32> adj_421 = {};
        wp::vec_t<3, wp::float32> adj_422 = {};
        wp::mat_t<3, 3, wp::float32> adj_423 = {};
        wp::vec_t<3, wp::float32> adj_424 = {};
        wp::vec_t<3, wp::float32> adj_425 = {};
        wp::vec_t<3, wp::float32> adj_426 = {};
        wp::int32 adj_427 = {};
        wp::float32 adj_428 = {};
        wp::int32 adj_429 = {};
        wp::float32 adj_430 = {};
        wp::int32 adj_431 = {};
        wp::float32 adj_432 = {};
        wp::int32 adj_433 = {};
        wp::float32 adj_434 = {};
        wp::int32 adj_435 = {};
        wp::float32 adj_436 = {};
        wp::int32 adj_437 = {};
        wp::float32 adj_438 = {};
        wp::int32 adj_439 = {};
        wp::float32 adj_440 = {};
        wp::int32 adj_441 = {};
        wp::float32 adj_442 = {};
        wp::int32 adj_443 = {};
        wp::float32 adj_444 = {};
        wp::mat_t<3, 3, wp::float32> adj_445 = {};
        //---------
        // forward
        // def update_orientation_kernel(                                                         <L 51>
        // i = wp.tid()                                                                           <L 67>
        var_0 = builtin_tid1d();
        // if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                          <L 68>
        var_1 = wp::address(var_particle_flags, var_0);
        var_4 = wp::int32(var_3);
        var_6 = wp::load(var_1);
        var_5 = wp::bit_and(var_6, var_4);
        var_8 = (var_5 == var_7);
        if (var_8) {
            // orientation_out[i] = orientation_in[i]                                             <L 69>
            var_9 = wp::address(var_orientation_in, var_0);
            var_10 = wp::load(var_9);
            // wp::array_store(var_orientation_out, var_0, var_10);
            // return                                                                             <L 70>
            goto label0;
        }
        // prev = orientation_in[i]                                                               <L 72>
        var_11 = wp::address(var_orientation_in, var_0);
        var_13 = wp::load(var_11);
        var_12 = wp::copy(var_13);
        // orie_x = wp.vec3(prev[0, 0], prev[0, 1], prev[0, 2])                                   <L 73>
        var_16 = wp::extract(var_12, var_14, var_15);
        var_19 = wp::extract(var_12, var_17, var_18);
        var_22 = wp::extract(var_12, var_20, var_21);
        var_23 = wp::vec_t<3, wp::float32>(var_16, var_19, var_22);
        // orie_y = wp.vec3(prev[1, 0], prev[1, 1], prev[1, 2])                                   <L 74>
        var_26 = wp::extract(var_12, var_24, var_25);
        var_29 = wp::extract(var_12, var_27, var_28);
        var_32 = wp::extract(var_12, var_30, var_31);
        var_33 = wp::vec_t<3, wp::float32>(var_26, var_29, var_32);
        // orie_z = wp.vec3(prev[2, 0], prev[2, 1], prev[2, 2])                                   <L 75>
        var_36 = wp::extract(var_12, var_34, var_35);
        var_39 = wp::extract(var_12, var_37, var_38);
        var_42 = wp::extract(var_12, var_40, var_41);
        var_43 = wp::vec_t<3, wp::float32>(var_36, var_39, var_42);
        // pos_i = particle_q[i]                                                                  <L 76>
        var_44 = wp::address(var_particle_q, var_0);
        var_46 = wp::load(var_44);
        var_45 = wp::copy(var_46);
        // n = particle_neighbors[i, 0]  # -X                                                     <L 80>
        var_48 = wp::address(var_particle_neighbors, var_0, var_47);
        var_50 = wp::load(var_48);
        var_49 = wp::copy(var_50);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 81>
        var_53 = (var_49 >= var_52);
        var_51 = var_53;
        if (var_51) {
            var_54 = wp::address(var_particle_flags, var_49);
            var_57 = wp::int32(var_56);
            var_59 = wp::load(var_54);
            var_58 = wp::bit_and(var_59, var_57);
            var_61 = (var_58 != var_60);
            var_51 = var_51 && var_61;
        }
        if (var_51) {
            // orie_x = _update_axis(orie_x, -1.0, particle_q[n] - pos_i, fA)                     <L 82>
            var_63 = wp::address(var_particle_q, var_49);
            var_65 = wp::load(var_63);
            var_64 = wp::sub(var_65, var_45);
            var_66 = _update_axis_0(var_23, var_62, var_64, var_fA);
            // nb = orientation_in[n]                                                             <L 83>
            var_67 = wp::address(var_orientation_in, var_49);
            var_69 = wp::load(var_67);
            var_68 = wp::copy(var_69);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 84>
            var_72 = wp::extract(var_68, var_70, var_71);
            var_75 = wp::extract(var_68, var_73, var_74);
            var_78 = wp::extract(var_68, var_76, var_77);
            var_79 = wp::vec_t<3, wp::float32>(var_72, var_75, var_78);
            var_80 = wp::mul(var_79, var_fB);
            var_81 = wp::add(var_66, var_80);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 85>
            var_84 = wp::extract(var_68, var_82, var_83);
            var_87 = wp::extract(var_68, var_85, var_86);
            var_90 = wp::extract(var_68, var_88, var_89);
            var_91 = wp::vec_t<3, wp::float32>(var_84, var_87, var_90);
            var_92 = wp::mul(var_91, var_fB);
            var_93 = wp::add(var_33, var_92);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 86>
            var_96 = wp::extract(var_68, var_94, var_95);
            var_99 = wp::extract(var_68, var_97, var_98);
            var_102 = wp::extract(var_68, var_100, var_101);
            var_103 = wp::vec_t<3, wp::float32>(var_96, var_99, var_102);
            var_104 = wp::mul(var_103, var_fB);
            var_105 = wp::add(var_43, var_104);
        }
        var_106 = wp::where(var_51, var_81, var_23);
        var_107 = wp::where(var_51, var_93, var_33);
        var_108 = wp::where(var_51, var_105, var_43);
        // n = particle_neighbors[i, 1]  # +X                                                     <L 88>
        var_110 = wp::address(var_particle_neighbors, var_0, var_109);
        var_112 = wp::load(var_110);
        var_111 = wp::copy(var_112);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 89>
        var_115 = (var_111 >= var_114);
        var_113 = var_115;
        if (var_113) {
            var_116 = wp::address(var_particle_flags, var_111);
            var_119 = wp::int32(var_118);
            var_121 = wp::load(var_116);
            var_120 = wp::bit_and(var_121, var_119);
            var_123 = (var_120 != var_122);
            var_113 = var_113 && var_123;
        }
        if (var_113) {
            // orie_x = _update_axis(orie_x, 1.0, particle_q[n] - pos_i, fA)                      <L 90>
            var_125 = wp::address(var_particle_q, var_111);
            var_127 = wp::load(var_125);
            var_126 = wp::sub(var_127, var_45);
            var_128 = _update_axis_0(var_106, var_124, var_126, var_fA);
            // nb = orientation_in[n]                                                             <L 91>
            var_129 = wp::address(var_orientation_in, var_111);
            var_131 = wp::load(var_129);
            var_130 = wp::copy(var_131);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 92>
            var_134 = wp::extract(var_130, var_132, var_133);
            var_137 = wp::extract(var_130, var_135, var_136);
            var_140 = wp::extract(var_130, var_138, var_139);
            var_141 = wp::vec_t<3, wp::float32>(var_134, var_137, var_140);
            var_142 = wp::mul(var_141, var_fB);
            var_143 = wp::add(var_128, var_142);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 93>
            var_146 = wp::extract(var_130, var_144, var_145);
            var_149 = wp::extract(var_130, var_147, var_148);
            var_152 = wp::extract(var_130, var_150, var_151);
            var_153 = wp::vec_t<3, wp::float32>(var_146, var_149, var_152);
            var_154 = wp::mul(var_153, var_fB);
            var_155 = wp::add(var_107, var_154);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 94>
            var_158 = wp::extract(var_130, var_156, var_157);
            var_161 = wp::extract(var_130, var_159, var_160);
            var_164 = wp::extract(var_130, var_162, var_163);
            var_165 = wp::vec_t<3, wp::float32>(var_158, var_161, var_164);
            var_166 = wp::mul(var_165, var_fB);
            var_167 = wp::add(var_108, var_166);
        }
        var_168 = wp::where(var_113, var_143, var_106);
        var_169 = wp::where(var_113, var_155, var_107);
        var_170 = wp::where(var_113, var_167, var_108);
        var_171 = wp::where(var_113, var_130, var_68);
        // n = particle_neighbors[i, 2]  # -Y                                                     <L 96>
        var_173 = wp::address(var_particle_neighbors, var_0, var_172);
        var_175 = wp::load(var_173);
        var_174 = wp::copy(var_175);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 97>
        var_178 = (var_174 >= var_177);
        var_176 = var_178;
        if (var_176) {
            var_179 = wp::address(var_particle_flags, var_174);
            var_182 = wp::int32(var_181);
            var_184 = wp::load(var_179);
            var_183 = wp::bit_and(var_184, var_182);
            var_186 = (var_183 != var_185);
            var_176 = var_176 && var_186;
        }
        if (var_176) {
            // orie_y = _update_axis(orie_y, -1.0, particle_q[n] - pos_i, fA)                     <L 98>
            var_188 = wp::address(var_particle_q, var_174);
            var_190 = wp::load(var_188);
            var_189 = wp::sub(var_190, var_45);
            var_191 = _update_axis_0(var_169, var_187, var_189, var_fA);
            // nb = orientation_in[n]                                                             <L 99>
            var_192 = wp::address(var_orientation_in, var_174);
            var_194 = wp::load(var_192);
            var_193 = wp::copy(var_194);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 100>
            var_197 = wp::extract(var_193, var_195, var_196);
            var_200 = wp::extract(var_193, var_198, var_199);
            var_203 = wp::extract(var_193, var_201, var_202);
            var_204 = wp::vec_t<3, wp::float32>(var_197, var_200, var_203);
            var_205 = wp::mul(var_204, var_fB);
            var_206 = wp::add(var_168, var_205);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 101>
            var_209 = wp::extract(var_193, var_207, var_208);
            var_212 = wp::extract(var_193, var_210, var_211);
            var_215 = wp::extract(var_193, var_213, var_214);
            var_216 = wp::vec_t<3, wp::float32>(var_209, var_212, var_215);
            var_217 = wp::mul(var_216, var_fB);
            var_218 = wp::add(var_191, var_217);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 102>
            var_221 = wp::extract(var_193, var_219, var_220);
            var_224 = wp::extract(var_193, var_222, var_223);
            var_227 = wp::extract(var_193, var_225, var_226);
            var_228 = wp::vec_t<3, wp::float32>(var_221, var_224, var_227);
            var_229 = wp::mul(var_228, var_fB);
            var_230 = wp::add(var_170, var_229);
        }
        var_231 = wp::where(var_176, var_206, var_168);
        var_232 = wp::where(var_176, var_218, var_169);
        var_233 = wp::where(var_176, var_230, var_170);
        var_234 = wp::where(var_176, var_193, var_171);
        // n = particle_neighbors[i, 3]  # +Y                                                     <L 104>
        var_236 = wp::address(var_particle_neighbors, var_0, var_235);
        var_238 = wp::load(var_236);
        var_237 = wp::copy(var_238);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 105>
        var_241 = (var_237 >= var_240);
        var_239 = var_241;
        if (var_239) {
            var_242 = wp::address(var_particle_flags, var_237);
            var_245 = wp::int32(var_244);
            var_247 = wp::load(var_242);
            var_246 = wp::bit_and(var_247, var_245);
            var_249 = (var_246 != var_248);
            var_239 = var_239 && var_249;
        }
        if (var_239) {
            // orie_y = _update_axis(orie_y, 1.0, particle_q[n] - pos_i, fA)                      <L 106>
            var_251 = wp::address(var_particle_q, var_237);
            var_253 = wp::load(var_251);
            var_252 = wp::sub(var_253, var_45);
            var_254 = _update_axis_0(var_232, var_250, var_252, var_fA);
            // nb = orientation_in[n]                                                             <L 107>
            var_255 = wp::address(var_orientation_in, var_237);
            var_257 = wp::load(var_255);
            var_256 = wp::copy(var_257);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 108>
            var_260 = wp::extract(var_256, var_258, var_259);
            var_263 = wp::extract(var_256, var_261, var_262);
            var_266 = wp::extract(var_256, var_264, var_265);
            var_267 = wp::vec_t<3, wp::float32>(var_260, var_263, var_266);
            var_268 = wp::mul(var_267, var_fB);
            var_269 = wp::add(var_231, var_268);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 109>
            var_272 = wp::extract(var_256, var_270, var_271);
            var_275 = wp::extract(var_256, var_273, var_274);
            var_278 = wp::extract(var_256, var_276, var_277);
            var_279 = wp::vec_t<3, wp::float32>(var_272, var_275, var_278);
            var_280 = wp::mul(var_279, var_fB);
            var_281 = wp::add(var_254, var_280);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 110>
            var_284 = wp::extract(var_256, var_282, var_283);
            var_287 = wp::extract(var_256, var_285, var_286);
            var_290 = wp::extract(var_256, var_288, var_289);
            var_291 = wp::vec_t<3, wp::float32>(var_284, var_287, var_290);
            var_292 = wp::mul(var_291, var_fB);
            var_293 = wp::add(var_233, var_292);
        }
        var_294 = wp::where(var_239, var_269, var_231);
        var_295 = wp::where(var_239, var_281, var_232);
        var_296 = wp::where(var_239, var_293, var_233);
        var_297 = wp::where(var_239, var_256, var_234);
        // n = particle_neighbors[i, 4]  # -Z                                                     <L 112>
        var_299 = wp::address(var_particle_neighbors, var_0, var_298);
        var_301 = wp::load(var_299);
        var_300 = wp::copy(var_301);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 113>
        var_304 = (var_300 >= var_303);
        var_302 = var_304;
        if (var_302) {
            var_305 = wp::address(var_particle_flags, var_300);
            var_308 = wp::int32(var_307);
            var_310 = wp::load(var_305);
            var_309 = wp::bit_and(var_310, var_308);
            var_312 = (var_309 != var_311);
            var_302 = var_302 && var_312;
        }
        if (var_302) {
            // orie_z = _update_axis(orie_z, -1.0, particle_q[n] - pos_i, fA)                     <L 114>
            var_314 = wp::address(var_particle_q, var_300);
            var_316 = wp::load(var_314);
            var_315 = wp::sub(var_316, var_45);
            var_317 = _update_axis_0(var_296, var_313, var_315, var_fA);
            // nb = orientation_in[n]                                                             <L 115>
            var_318 = wp::address(var_orientation_in, var_300);
            var_320 = wp::load(var_318);
            var_319 = wp::copy(var_320);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 116>
            var_323 = wp::extract(var_319, var_321, var_322);
            var_326 = wp::extract(var_319, var_324, var_325);
            var_329 = wp::extract(var_319, var_327, var_328);
            var_330 = wp::vec_t<3, wp::float32>(var_323, var_326, var_329);
            var_331 = wp::mul(var_330, var_fB);
            var_332 = wp::add(var_294, var_331);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 117>
            var_335 = wp::extract(var_319, var_333, var_334);
            var_338 = wp::extract(var_319, var_336, var_337);
            var_341 = wp::extract(var_319, var_339, var_340);
            var_342 = wp::vec_t<3, wp::float32>(var_335, var_338, var_341);
            var_343 = wp::mul(var_342, var_fB);
            var_344 = wp::add(var_295, var_343);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 118>
            var_347 = wp::extract(var_319, var_345, var_346);
            var_350 = wp::extract(var_319, var_348, var_349);
            var_353 = wp::extract(var_319, var_351, var_352);
            var_354 = wp::vec_t<3, wp::float32>(var_347, var_350, var_353);
            var_355 = wp::mul(var_354, var_fB);
            var_356 = wp::add(var_317, var_355);
        }
        var_357 = wp::where(var_302, var_332, var_294);
        var_358 = wp::where(var_302, var_344, var_295);
        var_359 = wp::where(var_302, var_356, var_296);
        var_360 = wp::where(var_302, var_319, var_297);
        // n = particle_neighbors[i, 5]  # +Z                                                     <L 120>
        var_362 = wp::address(var_particle_neighbors, var_0, var_361);
        var_364 = wp::load(var_362);
        var_363 = wp::copy(var_364);
        // if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:               <L 121>
        var_367 = (var_363 >= var_366);
        var_365 = var_367;
        if (var_365) {
            var_368 = wp::address(var_particle_flags, var_363);
            var_371 = wp::int32(var_370);
            var_373 = wp::load(var_368);
            var_372 = wp::bit_and(var_373, var_371);
            var_375 = (var_372 != var_374);
            var_365 = var_365 && var_375;
        }
        if (var_365) {
            // orie_z = _update_axis(orie_z, 1.0, particle_q[n] - pos_i, fA)                      <L 122>
            var_377 = wp::address(var_particle_q, var_363);
            var_379 = wp::load(var_377);
            var_378 = wp::sub(var_379, var_45);
            var_380 = _update_axis_0(var_359, var_376, var_378, var_fA);
            // nb = orientation_in[n]                                                             <L 123>
            var_381 = wp::address(var_orientation_in, var_363);
            var_383 = wp::load(var_381);
            var_382 = wp::copy(var_383);
            // orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                       <L 124>
            var_386 = wp::extract(var_382, var_384, var_385);
            var_389 = wp::extract(var_382, var_387, var_388);
            var_392 = wp::extract(var_382, var_390, var_391);
            var_393 = wp::vec_t<3, wp::float32>(var_386, var_389, var_392);
            var_394 = wp::mul(var_393, var_fB);
            var_395 = wp::add(var_357, var_394);
            // orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                       <L 125>
            var_398 = wp::extract(var_382, var_396, var_397);
            var_401 = wp::extract(var_382, var_399, var_400);
            var_404 = wp::extract(var_382, var_402, var_403);
            var_405 = wp::vec_t<3, wp::float32>(var_398, var_401, var_404);
            var_406 = wp::mul(var_405, var_fB);
            var_407 = wp::add(var_358, var_406);
            // orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                       <L 126>
            var_410 = wp::extract(var_382, var_408, var_409);
            var_413 = wp::extract(var_382, var_411, var_412);
            var_416 = wp::extract(var_382, var_414, var_415);
            var_417 = wp::vec_t<3, wp::float32>(var_410, var_413, var_416);
            var_418 = wp::mul(var_417, var_fB);
            var_419 = wp::add(var_380, var_418);
        }
        var_420 = wp::where(var_365, var_395, var_357);
        var_421 = wp::where(var_365, var_407, var_358);
        var_422 = wp::where(var_365, var_419, var_359);
        var_423 = wp::where(var_365, var_382, var_360);
        // orie_x = _safe_normalize(orie_x)                                                       <L 128>
        var_424 = _safe_normalize_0(var_420);
        // orie_y = _safe_normalize(orie_y)                                                       <L 129>
        var_425 = _safe_normalize_0(var_421);
        // orie_z = _safe_normalize(orie_z)                                                       <L 130>
        var_426 = _safe_normalize_0(var_422);
        // orientation_out[i] = wp.mat33(                                                         <L 132>
        // orie_x[0], orie_x[1], orie_x[2],                                                       <L 133>
        var_428 = wp::extract(var_424, var_427);
        var_430 = wp::extract(var_424, var_429);
        var_432 = wp::extract(var_424, var_431);
        // orie_y[0], orie_y[1], orie_y[2],                                                       <L 134>
        var_434 = wp::extract(var_425, var_433);
        var_436 = wp::extract(var_425, var_435);
        var_438 = wp::extract(var_425, var_437);
        // orie_z[0], orie_z[1], orie_z[2],                                                       <L 135>
        var_440 = wp::extract(var_426, var_439);
        var_442 = wp::extract(var_426, var_441);
        var_444 = wp::extract(var_426, var_443);
        var_445 = wp::mat_t<3, 3, wp::float32>(var_428, var_430, var_432, var_434, var_436, var_438, var_440, var_442, var_444);
        // orientation_out[i] = wp.mat33(                                                         <L 132>
        // wp::array_store(var_orientation_out, var_0, var_445);
        //---------
        // reverse
        wp::adj_array_store(var_orientation_out, var_0, var_445, adj_orientation_out, adj_0, adj_445);
        // adj: orientation_out[i] = wp.mat33(                                                    <L 132>
        wp::adj_mat_t(var_428, var_430, var_432, var_434, var_436, var_438, var_440, var_442, var_444, adj_428, adj_430, adj_432, adj_434, adj_436, adj_438, adj_440, adj_442, adj_444, adj_445);
        wp::adj_extract(var_426, var_443, adj_426, adj_443, adj_444);
        wp::adj_extract(var_426, var_441, adj_426, adj_441, adj_442);
        wp::adj_extract(var_426, var_439, adj_426, adj_439, adj_440);
        // adj: orie_z[0], orie_z[1], orie_z[2],                                                  <L 135>
        wp::adj_extract(var_425, var_437, adj_425, adj_437, adj_438);
        wp::adj_extract(var_425, var_435, adj_425, adj_435, adj_436);
        wp::adj_extract(var_425, var_433, adj_425, adj_433, adj_434);
        // adj: orie_y[0], orie_y[1], orie_y[2],                                                  <L 134>
        wp::adj_extract(var_424, var_431, adj_424, adj_431, adj_432);
        wp::adj_extract(var_424, var_429, adj_424, adj_429, adj_430);
        wp::adj_extract(var_424, var_427, adj_424, adj_427, adj_428);
        // adj: orie_x[0], orie_x[1], orie_x[2],                                                  <L 133>
        // adj: orientation_out[i] = wp.mat33(                                                    <L 132>
        adj__safe_normalize_0(var_422, adj_422, adj_426);
        // adj: orie_z = _safe_normalize(orie_z)                                                  <L 130>
        adj__safe_normalize_0(var_421, adj_421, adj_425);
        // adj: orie_y = _safe_normalize(orie_y)                                                  <L 129>
        adj__safe_normalize_0(var_420, adj_420, adj_424);
        // adj: orie_x = _safe_normalize(orie_x)                                                  <L 128>
        wp::adj_where(var_365, var_382, var_360, adj_365, adj_382, adj_360, adj_423);
        wp::adj_where(var_365, var_419, var_359, adj_365, adj_419, adj_359, adj_422);
        wp::adj_where(var_365, var_407, var_358, adj_365, adj_407, adj_358, adj_421);
        wp::adj_where(var_365, var_395, var_357, adj_365, adj_395, adj_357, adj_420);
        if (var_365) {
            wp::adj_add(var_380, var_418, adj_380, adj_418, adj_419);
            wp::adj_mul(var_417, var_fB, adj_417, adj_fB, adj_418);
            wp::adj_vec_t(var_410, var_413, var_416, adj_410, adj_413, adj_416, adj_417);
            wp::adj_extract(var_382, var_414, var_415, adj_382, adj_414, adj_415, adj_416);
            wp::adj_extract(var_382, var_411, var_412, adj_382, adj_411, adj_412, adj_413);
            wp::adj_extract(var_382, var_408, var_409, adj_382, adj_408, adj_409, adj_410);
            // adj: orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                  <L 126>
            wp::adj_add(var_358, var_406, adj_358, adj_406, adj_407);
            wp::adj_mul(var_405, var_fB, adj_405, adj_fB, adj_406);
            wp::adj_vec_t(var_398, var_401, var_404, adj_398, adj_401, adj_404, adj_405);
            wp::adj_extract(var_382, var_402, var_403, adj_382, adj_402, adj_403, adj_404);
            wp::adj_extract(var_382, var_399, var_400, adj_382, adj_399, adj_400, adj_401);
            wp::adj_extract(var_382, var_396, var_397, adj_382, adj_396, adj_397, adj_398);
            // adj: orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                  <L 125>
            wp::adj_add(var_357, var_394, adj_357, adj_394, adj_395);
            wp::adj_mul(var_393, var_fB, adj_393, adj_fB, adj_394);
            wp::adj_vec_t(var_386, var_389, var_392, adj_386, adj_389, adj_392, adj_393);
            wp::adj_extract(var_382, var_390, var_391, adj_382, adj_390, adj_391, adj_392);
            wp::adj_extract(var_382, var_387, var_388, adj_382, adj_387, adj_388, adj_389);
            wp::adj_extract(var_382, var_384, var_385, adj_382, adj_384, adj_385, adj_386);
            // adj: orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                  <L 124>
            wp::adj_copy(var_383, adj_381, adj_382);
            wp::adj_address(var_orientation_in, var_363, adj_orientation_in, adj_363, adj_381);
            // adj: nb = orientation_in[n]                                                        <L 123>
            adj__update_axis_0(var_359, var_376, var_378, var_fA, adj_359, adj_376, adj_378, adj_fA, adj_380);
            wp::adj_sub(var_379, var_45, adj_377, adj_45, adj_378);
            wp::adj_address(var_particle_q, var_363, adj_particle_q, adj_363, adj_377);
            // adj: orie_z = _update_axis(orie_z, 1.0, particle_q[n] - pos_i, fA)                 <L 122>
        }
        if (var_365) {
            wp::adj_int32(var_370, adj_370, adj_371);
            wp::adj_address(var_particle_flags, var_363, adj_particle_flags, adj_363, adj_368);
        }
        // adj: if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:          <L 121>
        wp::adj_copy(var_364, adj_362, adj_363);
        wp::adj_address(var_particle_neighbors, var_0, var_361, adj_particle_neighbors, adj_0, adj_361, adj_362);
        // adj: n = particle_neighbors[i, 5]  # +Z                                                <L 120>
        wp::adj_where(var_302, var_319, var_297, adj_302, adj_319, adj_297, adj_360);
        wp::adj_where(var_302, var_356, var_296, adj_302, adj_356, adj_296, adj_359);
        wp::adj_where(var_302, var_344, var_295, adj_302, adj_344, adj_295, adj_358);
        wp::adj_where(var_302, var_332, var_294, adj_302, adj_332, adj_294, adj_357);
        if (var_302) {
            wp::adj_add(var_317, var_355, adj_317, adj_355, adj_356);
            wp::adj_mul(var_354, var_fB, adj_354, adj_fB, adj_355);
            wp::adj_vec_t(var_347, var_350, var_353, adj_347, adj_350, adj_353, adj_354);
            wp::adj_extract(var_319, var_351, var_352, adj_319, adj_351, adj_352, adj_353);
            wp::adj_extract(var_319, var_348, var_349, adj_319, adj_348, adj_349, adj_350);
            wp::adj_extract(var_319, var_345, var_346, adj_319, adj_345, adj_346, adj_347);
            // adj: orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                  <L 118>
            wp::adj_add(var_295, var_343, adj_295, adj_343, adj_344);
            wp::adj_mul(var_342, var_fB, adj_342, adj_fB, adj_343);
            wp::adj_vec_t(var_335, var_338, var_341, adj_335, adj_338, adj_341, adj_342);
            wp::adj_extract(var_319, var_339, var_340, adj_319, adj_339, adj_340, adj_341);
            wp::adj_extract(var_319, var_336, var_337, adj_319, adj_336, adj_337, adj_338);
            wp::adj_extract(var_319, var_333, var_334, adj_319, adj_333, adj_334, adj_335);
            // adj: orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                  <L 117>
            wp::adj_add(var_294, var_331, adj_294, adj_331, adj_332);
            wp::adj_mul(var_330, var_fB, adj_330, adj_fB, adj_331);
            wp::adj_vec_t(var_323, var_326, var_329, adj_323, adj_326, adj_329, adj_330);
            wp::adj_extract(var_319, var_327, var_328, adj_319, adj_327, adj_328, adj_329);
            wp::adj_extract(var_319, var_324, var_325, adj_319, adj_324, adj_325, adj_326);
            wp::adj_extract(var_319, var_321, var_322, adj_319, adj_321, adj_322, adj_323);
            // adj: orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                  <L 116>
            wp::adj_copy(var_320, adj_318, adj_319);
            wp::adj_address(var_orientation_in, var_300, adj_orientation_in, adj_300, adj_318);
            // adj: nb = orientation_in[n]                                                        <L 115>
            adj__update_axis_0(var_296, var_313, var_315, var_fA, adj_296, adj_313, adj_315, adj_fA, adj_317);
            wp::adj_sub(var_316, var_45, adj_314, adj_45, adj_315);
            wp::adj_address(var_particle_q, var_300, adj_particle_q, adj_300, adj_314);
            // adj: orie_z = _update_axis(orie_z, -1.0, particle_q[n] - pos_i, fA)                <L 114>
        }
        if (var_302) {
            wp::adj_int32(var_307, adj_307, adj_308);
            wp::adj_address(var_particle_flags, var_300, adj_particle_flags, adj_300, adj_305);
        }
        // adj: if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:          <L 113>
        wp::adj_copy(var_301, adj_299, adj_300);
        wp::adj_address(var_particle_neighbors, var_0, var_298, adj_particle_neighbors, adj_0, adj_298, adj_299);
        // adj: n = particle_neighbors[i, 4]  # -Z                                                <L 112>
        wp::adj_where(var_239, var_256, var_234, adj_239, adj_256, adj_234, adj_297);
        wp::adj_where(var_239, var_293, var_233, adj_239, adj_293, adj_233, adj_296);
        wp::adj_where(var_239, var_281, var_232, adj_239, adj_281, adj_232, adj_295);
        wp::adj_where(var_239, var_269, var_231, adj_239, adj_269, adj_231, adj_294);
        if (var_239) {
            wp::adj_add(var_233, var_292, adj_233, adj_292, adj_293);
            wp::adj_mul(var_291, var_fB, adj_291, adj_fB, adj_292);
            wp::adj_vec_t(var_284, var_287, var_290, adj_284, adj_287, adj_290, adj_291);
            wp::adj_extract(var_256, var_288, var_289, adj_256, adj_288, adj_289, adj_290);
            wp::adj_extract(var_256, var_285, var_286, adj_256, adj_285, adj_286, adj_287);
            wp::adj_extract(var_256, var_282, var_283, adj_256, adj_282, adj_283, adj_284);
            // adj: orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                  <L 110>
            wp::adj_add(var_254, var_280, adj_254, adj_280, adj_281);
            wp::adj_mul(var_279, var_fB, adj_279, adj_fB, adj_280);
            wp::adj_vec_t(var_272, var_275, var_278, adj_272, adj_275, adj_278, adj_279);
            wp::adj_extract(var_256, var_276, var_277, adj_256, adj_276, adj_277, adj_278);
            wp::adj_extract(var_256, var_273, var_274, adj_256, adj_273, adj_274, adj_275);
            wp::adj_extract(var_256, var_270, var_271, adj_256, adj_270, adj_271, adj_272);
            // adj: orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                  <L 109>
            wp::adj_add(var_231, var_268, adj_231, adj_268, adj_269);
            wp::adj_mul(var_267, var_fB, adj_267, adj_fB, adj_268);
            wp::adj_vec_t(var_260, var_263, var_266, adj_260, adj_263, adj_266, adj_267);
            wp::adj_extract(var_256, var_264, var_265, adj_256, adj_264, adj_265, adj_266);
            wp::adj_extract(var_256, var_261, var_262, adj_256, adj_261, adj_262, adj_263);
            wp::adj_extract(var_256, var_258, var_259, adj_256, adj_258, adj_259, adj_260);
            // adj: orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                  <L 108>
            wp::adj_copy(var_257, adj_255, adj_256);
            wp::adj_address(var_orientation_in, var_237, adj_orientation_in, adj_237, adj_255);
            // adj: nb = orientation_in[n]                                                        <L 107>
            adj__update_axis_0(var_232, var_250, var_252, var_fA, adj_232, adj_250, adj_252, adj_fA, adj_254);
            wp::adj_sub(var_253, var_45, adj_251, adj_45, adj_252);
            wp::adj_address(var_particle_q, var_237, adj_particle_q, adj_237, adj_251);
            // adj: orie_y = _update_axis(orie_y, 1.0, particle_q[n] - pos_i, fA)                 <L 106>
        }
        if (var_239) {
            wp::adj_int32(var_244, adj_244, adj_245);
            wp::adj_address(var_particle_flags, var_237, adj_particle_flags, adj_237, adj_242);
        }
        // adj: if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:          <L 105>
        wp::adj_copy(var_238, adj_236, adj_237);
        wp::adj_address(var_particle_neighbors, var_0, var_235, adj_particle_neighbors, adj_0, adj_235, adj_236);
        // adj: n = particle_neighbors[i, 3]  # +Y                                                <L 104>
        wp::adj_where(var_176, var_193, var_171, adj_176, adj_193, adj_171, adj_234);
        wp::adj_where(var_176, var_230, var_170, adj_176, adj_230, adj_170, adj_233);
        wp::adj_where(var_176, var_218, var_169, adj_176, adj_218, adj_169, adj_232);
        wp::adj_where(var_176, var_206, var_168, adj_176, adj_206, adj_168, adj_231);
        if (var_176) {
            wp::adj_add(var_170, var_229, adj_170, adj_229, adj_230);
            wp::adj_mul(var_228, var_fB, adj_228, adj_fB, adj_229);
            wp::adj_vec_t(var_221, var_224, var_227, adj_221, adj_224, adj_227, adj_228);
            wp::adj_extract(var_193, var_225, var_226, adj_193, adj_225, adj_226, adj_227);
            wp::adj_extract(var_193, var_222, var_223, adj_193, adj_222, adj_223, adj_224);
            wp::adj_extract(var_193, var_219, var_220, adj_193, adj_219, adj_220, adj_221);
            // adj: orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                  <L 102>
            wp::adj_add(var_191, var_217, adj_191, adj_217, adj_218);
            wp::adj_mul(var_216, var_fB, adj_216, adj_fB, adj_217);
            wp::adj_vec_t(var_209, var_212, var_215, adj_209, adj_212, adj_215, adj_216);
            wp::adj_extract(var_193, var_213, var_214, adj_193, adj_213, adj_214, adj_215);
            wp::adj_extract(var_193, var_210, var_211, adj_193, adj_210, adj_211, adj_212);
            wp::adj_extract(var_193, var_207, var_208, adj_193, adj_207, adj_208, adj_209);
            // adj: orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                  <L 101>
            wp::adj_add(var_168, var_205, adj_168, adj_205, adj_206);
            wp::adj_mul(var_204, var_fB, adj_204, adj_fB, adj_205);
            wp::adj_vec_t(var_197, var_200, var_203, adj_197, adj_200, adj_203, adj_204);
            wp::adj_extract(var_193, var_201, var_202, adj_193, adj_201, adj_202, adj_203);
            wp::adj_extract(var_193, var_198, var_199, adj_193, adj_198, adj_199, adj_200);
            wp::adj_extract(var_193, var_195, var_196, adj_193, adj_195, adj_196, adj_197);
            // adj: orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                  <L 100>
            wp::adj_copy(var_194, adj_192, adj_193);
            wp::adj_address(var_orientation_in, var_174, adj_orientation_in, adj_174, adj_192);
            // adj: nb = orientation_in[n]                                                        <L 99>
            adj__update_axis_0(var_169, var_187, var_189, var_fA, adj_169, adj_187, adj_189, adj_fA, adj_191);
            wp::adj_sub(var_190, var_45, adj_188, adj_45, adj_189);
            wp::adj_address(var_particle_q, var_174, adj_particle_q, adj_174, adj_188);
            // adj: orie_y = _update_axis(orie_y, -1.0, particle_q[n] - pos_i, fA)                <L 98>
        }
        if (var_176) {
            wp::adj_int32(var_181, adj_181, adj_182);
            wp::adj_address(var_particle_flags, var_174, adj_particle_flags, adj_174, adj_179);
        }
        // adj: if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:          <L 97>
        wp::adj_copy(var_175, adj_173, adj_174);
        wp::adj_address(var_particle_neighbors, var_0, var_172, adj_particle_neighbors, adj_0, adj_172, adj_173);
        // adj: n = particle_neighbors[i, 2]  # -Y                                                <L 96>
        wp::adj_where(var_113, var_130, var_68, adj_113, adj_130, adj_68, adj_171);
        wp::adj_where(var_113, var_167, var_108, adj_113, adj_167, adj_108, adj_170);
        wp::adj_where(var_113, var_155, var_107, adj_113, adj_155, adj_107, adj_169);
        wp::adj_where(var_113, var_143, var_106, adj_113, adj_143, adj_106, adj_168);
        if (var_113) {
            wp::adj_add(var_108, var_166, adj_108, adj_166, adj_167);
            wp::adj_mul(var_165, var_fB, adj_165, adj_fB, adj_166);
            wp::adj_vec_t(var_158, var_161, var_164, adj_158, adj_161, adj_164, adj_165);
            wp::adj_extract(var_130, var_162, var_163, adj_130, adj_162, adj_163, adj_164);
            wp::adj_extract(var_130, var_159, var_160, adj_130, adj_159, adj_160, adj_161);
            wp::adj_extract(var_130, var_156, var_157, adj_130, adj_156, adj_157, adj_158);
            // adj: orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                  <L 94>
            wp::adj_add(var_107, var_154, adj_107, adj_154, adj_155);
            wp::adj_mul(var_153, var_fB, adj_153, adj_fB, adj_154);
            wp::adj_vec_t(var_146, var_149, var_152, adj_146, adj_149, adj_152, adj_153);
            wp::adj_extract(var_130, var_150, var_151, adj_130, adj_150, adj_151, adj_152);
            wp::adj_extract(var_130, var_147, var_148, adj_130, adj_147, adj_148, adj_149);
            wp::adj_extract(var_130, var_144, var_145, adj_130, adj_144, adj_145, adj_146);
            // adj: orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                  <L 93>
            wp::adj_add(var_128, var_142, adj_128, adj_142, adj_143);
            wp::adj_mul(var_141, var_fB, adj_141, adj_fB, adj_142);
            wp::adj_vec_t(var_134, var_137, var_140, adj_134, adj_137, adj_140, adj_141);
            wp::adj_extract(var_130, var_138, var_139, adj_130, adj_138, adj_139, adj_140);
            wp::adj_extract(var_130, var_135, var_136, adj_130, adj_135, adj_136, adj_137);
            wp::adj_extract(var_130, var_132, var_133, adj_130, adj_132, adj_133, adj_134);
            // adj: orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                  <L 92>
            wp::adj_copy(var_131, adj_129, adj_130);
            wp::adj_address(var_orientation_in, var_111, adj_orientation_in, adj_111, adj_129);
            // adj: nb = orientation_in[n]                                                        <L 91>
            adj__update_axis_0(var_106, var_124, var_126, var_fA, adj_106, adj_124, adj_126, adj_fA, adj_128);
            wp::adj_sub(var_127, var_45, adj_125, adj_45, adj_126);
            wp::adj_address(var_particle_q, var_111, adj_particle_q, adj_111, adj_125);
            // adj: orie_x = _update_axis(orie_x, 1.0, particle_q[n] - pos_i, fA)                 <L 90>
        }
        if (var_113) {
            wp::adj_int32(var_118, adj_118, adj_119);
            wp::adj_address(var_particle_flags, var_111, adj_particle_flags, adj_111, adj_116);
        }
        // adj: if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:          <L 89>
        wp::adj_copy(var_112, adj_110, adj_111);
        wp::adj_address(var_particle_neighbors, var_0, var_109, adj_particle_neighbors, adj_0, adj_109, adj_110);
        // adj: n = particle_neighbors[i, 1]  # +X                                                <L 88>
        wp::adj_where(var_51, var_105, var_43, adj_51, adj_105, adj_43, adj_108);
        wp::adj_where(var_51, var_93, var_33, adj_51, adj_93, adj_33, adj_107);
        wp::adj_where(var_51, var_81, var_23, adj_51, adj_81, adj_23, adj_106);
        if (var_51) {
            wp::adj_add(var_43, var_104, adj_43, adj_104, adj_105);
            wp::adj_mul(var_103, var_fB, adj_103, adj_fB, adj_104);
            wp::adj_vec_t(var_96, var_99, var_102, adj_96, adj_99, adj_102, adj_103);
            wp::adj_extract(var_68, var_100, var_101, adj_68, adj_100, adj_101, adj_102);
            wp::adj_extract(var_68, var_97, var_98, adj_68, adj_97, adj_98, adj_99);
            wp::adj_extract(var_68, var_94, var_95, adj_68, adj_94, adj_95, adj_96);
            // adj: orie_z = orie_z + wp.vec3(nb[2, 0], nb[2, 1], nb[2, 2]) * fB                  <L 86>
            wp::adj_add(var_33, var_92, adj_33, adj_92, adj_93);
            wp::adj_mul(var_91, var_fB, adj_91, adj_fB, adj_92);
            wp::adj_vec_t(var_84, var_87, var_90, adj_84, adj_87, adj_90, adj_91);
            wp::adj_extract(var_68, var_88, var_89, adj_68, adj_88, adj_89, adj_90);
            wp::adj_extract(var_68, var_85, var_86, adj_68, adj_85, adj_86, adj_87);
            wp::adj_extract(var_68, var_82, var_83, adj_68, adj_82, adj_83, adj_84);
            // adj: orie_y = orie_y + wp.vec3(nb[1, 0], nb[1, 1], nb[1, 2]) * fB                  <L 85>
            wp::adj_add(var_66, var_80, adj_66, adj_80, adj_81);
            wp::adj_mul(var_79, var_fB, adj_79, adj_fB, adj_80);
            wp::adj_vec_t(var_72, var_75, var_78, adj_72, adj_75, adj_78, adj_79);
            wp::adj_extract(var_68, var_76, var_77, adj_68, adj_76, adj_77, adj_78);
            wp::adj_extract(var_68, var_73, var_74, adj_68, adj_73, adj_74, adj_75);
            wp::adj_extract(var_68, var_70, var_71, adj_68, adj_70, adj_71, adj_72);
            // adj: orie_x = orie_x + wp.vec3(nb[0, 0], nb[0, 1], nb[0, 2]) * fB                  <L 84>
            wp::adj_copy(var_69, adj_67, adj_68);
            wp::adj_address(var_orientation_in, var_49, adj_orientation_in, adj_49, adj_67);
            // adj: nb = orientation_in[n]                                                        <L 83>
            adj__update_axis_0(var_23, var_62, var_64, var_fA, adj_23, adj_62, adj_64, adj_fA, adj_66);
            wp::adj_sub(var_65, var_45, adj_63, adj_45, adj_64);
            wp::adj_address(var_particle_q, var_49, adj_particle_q, adj_49, adj_63);
            // adj: orie_x = _update_axis(orie_x, -1.0, particle_q[n] - pos_i, fA)                <L 82>
        }
        if (var_51) {
            wp::adj_int32(var_56, adj_56, adj_57);
            wp::adj_address(var_particle_flags, var_49, adj_particle_flags, adj_49, adj_54);
        }
        // adj: if n >= 0 and (particle_flags[n] & wp.int32(ParticleFlags.ACTIVE)) != 0:          <L 81>
        wp::adj_copy(var_50, adj_48, adj_49);
        wp::adj_address(var_particle_neighbors, var_0, var_47, adj_particle_neighbors, adj_0, adj_47, adj_48);
        // adj: n = particle_neighbors[i, 0]  # -X                                                <L 80>
        wp::adj_copy(var_46, adj_44, adj_45);
        wp::adj_address(var_particle_q, var_0, adj_particle_q, adj_0, adj_44);
        // adj: pos_i = particle_q[i]                                                             <L 76>
        wp::adj_vec_t(var_36, var_39, var_42, adj_36, adj_39, adj_42, adj_43);
        wp::adj_extract(var_12, var_40, var_41, adj_12, adj_40, adj_41, adj_42);
        wp::adj_extract(var_12, var_37, var_38, adj_12, adj_37, adj_38, adj_39);
        wp::adj_extract(var_12, var_34, var_35, adj_12, adj_34, adj_35, adj_36);
        // adj: orie_z = wp.vec3(prev[2, 0], prev[2, 1], prev[2, 2])                              <L 75>
        wp::adj_vec_t(var_26, var_29, var_32, adj_26, adj_29, adj_32, adj_33);
        wp::adj_extract(var_12, var_30, var_31, adj_12, adj_30, adj_31, adj_32);
        wp::adj_extract(var_12, var_27, var_28, adj_12, adj_27, adj_28, adj_29);
        wp::adj_extract(var_12, var_24, var_25, adj_12, adj_24, adj_25, adj_26);
        // adj: orie_y = wp.vec3(prev[1, 0], prev[1, 1], prev[1, 2])                              <L 74>
        wp::adj_vec_t(var_16, var_19, var_22, adj_16, adj_19, adj_22, adj_23);
        wp::adj_extract(var_12, var_20, var_21, adj_12, adj_20, adj_21, adj_22);
        wp::adj_extract(var_12, var_17, var_18, adj_12, adj_17, adj_18, adj_19);
        wp::adj_extract(var_12, var_14, var_15, adj_12, adj_14, adj_15, adj_16);
        // adj: orie_x = wp.vec3(prev[0, 0], prev[0, 1], prev[0, 2])                              <L 73>
        wp::adj_copy(var_13, adj_11, adj_12);
        wp::adj_address(var_orientation_in, var_0, adj_orientation_in, adj_0, adj_11);
        // adj: prev = orientation_in[i]                                                          <L 72>
        if (var_8) {
            label0:;
            // adj: return                                                                        <L 70>
            wp::adj_array_store(var_orientation_out, var_0, var_10, adj_orientation_out, adj_0, adj_9);
            wp::adj_address(var_orientation_in, var_0, adj_orientation_in, adj_0, adj_9);
            // adj: orientation_out[i] = orientation_in[i]                                        <L 69>
        }
        wp::adj_int32(var_3, adj_3, adj_4);
        wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
        // adj: if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:                     <L 68>
        // adj: i = wp.tid()                                                                      <L 67>
        // adj: def update_orientation_kernel(                                                    <L 51>
        continue;
    }
}

