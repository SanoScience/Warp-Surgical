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

struct wp_args_validate_and_correct_inertia_kernel_0b4bb59b {
    wp::array_t<wp::float32> body_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> body_inertia;
    wp::array_t<wp::float32> body_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> body_inv_inertia;
    bool balance_inertia;
    wp::float32 bound_mass;
    wp::float32 bound_inertia;
    wp::array_t<wp::int32> correction_count;
};


void validate_and_correct_inertia_kernel_0b4bb59b_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_validate_and_correct_inertia_kernel_0b4bb59b *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_body_mass = _wp_args->body_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inertia = _wp_args->body_inertia;
    wp::array_t<wp::float32> var_body_inv_mass = _wp_args->body_inv_mass;
    wp::array_t<wp::mat_t<3, 3, wp::float32>> var_body_inv_inertia = _wp_args->body_inv_inertia;
    bool var_balance_inertia = _wp_args->balance_inertia;
    wp::float32 var_bound_mass = _wp_args->bound_mass;
    wp::float32 var_bound_inertia = _wp_args->bound_inertia;
    wp::array_t<wp::int32> var_correction_count = _wp_args->correction_count;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::float32* var_1;
    wp::float32 var_2;
    wp::float32 var_3;
    wp::mat_t<3, 3, wp::float32>* var_4;
    wp::mat_t<3, 3, wp::float32> var_5;
    wp::mat_t<3, 3, wp::float32> var_6;
    wp::mat_t<3, 3, wp::float32> var_7;
    const bool var_8 = false;
    bool var_9;
    bool var_10;
    bool var_11;
    const wp::int32 var_12 = 0;
    const wp::int32 var_13 = 0;
    wp::float32 var_14;
    bool var_15;
    bool var_16;
    const wp::int32 var_17 = 0;
    const wp::int32 var_18 = 1;
    wp::float32 var_19;
    bool var_20;
    bool var_21;
    const wp::int32 var_22 = 0;
    const wp::int32 var_23 = 2;
    wp::float32 var_24;
    bool var_25;
    bool var_26;
    const wp::int32 var_27 = 1;
    const wp::int32 var_28 = 0;
    wp::float32 var_29;
    bool var_30;
    bool var_31;
    const wp::int32 var_32 = 1;
    const wp::int32 var_33 = 1;
    wp::float32 var_34;
    bool var_35;
    bool var_36;
    const wp::int32 var_37 = 1;
    const wp::int32 var_38 = 2;
    wp::float32 var_39;
    bool var_40;
    bool var_41;
    const wp::int32 var_42 = 2;
    const wp::int32 var_43 = 0;
    wp::float32 var_44;
    bool var_45;
    bool var_46;
    const wp::int32 var_47 = 2;
    const wp::int32 var_48 = 1;
    wp::float32 var_49;
    bool var_50;
    bool var_51;
    const wp::int32 var_52 = 2;
    const wp::int32 var_53 = 2;
    wp::float32 var_54;
    bool var_55;
    bool var_56;
    const wp::float32 var_57 = 0.0;
    const wp::float32 var_58 = 0.0;
    const wp::float32 var_59 = 0.0;
    const wp::float32 var_60 = 0.0;
    const wp::float32 var_61 = 0.0;
    const wp::float32 var_62 = 0.0;
    const wp::float32 var_63 = 0.0;
    const wp::float32 var_64 = 0.0;
    const wp::float32 var_65 = 0.0;
    const wp::float32 var_66 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_67;
    const bool var_68 = true;
    wp::float32 var_69;
    wp::mat_t<3, 3, wp::float32> var_70;
    bool var_71;
    const wp::float32 var_72 = 0.0;
    bool var_73;
    const wp::float32 var_74 = 0.0;
    const bool var_75 = true;
    wp::float32 var_76;
    bool var_77;
    bool var_78;
    const wp::float32 var_79 = 0.0;
    bool var_80;
    bool var_81;
    const wp::float32 var_82 = 0.0;
    bool var_83;
    wp::float32 var_84;
    const bool var_85 = true;
    wp::float32 var_86;
    bool var_87;
    const wp::float32 var_88 = 0.0;
    bool var_89;
    bool var_90;
    wp::float32 var_91;
    const wp::float32 var_92 = 0.0;
    bool var_93;
    const wp::float32 var_94 = 0.0;
    const wp::float32 var_95 = 0.0;
    const wp::float32 var_96 = 0.0;
    const wp::float32 var_97 = 0.0;
    const wp::float32 var_98 = 0.0;
    const wp::float32 var_99 = 0.0;
    const wp::float32 var_100 = 0.0;
    const wp::float32 var_101 = 0.0;
    const wp::float32 var_102 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_103;
    wp::mat_t<3, 3, wp::float32> var_104;
    bool var_105;
    const wp::int32 var_106 = 0;
    const wp::int32 var_107 = 1;
    wp::float32 var_108;
    const wp::int32 var_109 = 1;
    const wp::int32 var_110 = 0;
    wp::float32 var_111;
    wp::float32 var_112;
    const wp::float32 var_113 = 0.5;
    wp::float32 var_114;
    const wp::int32 var_115 = 0;
    const wp::int32 var_116 = 2;
    wp::float32 var_117;
    const wp::int32 var_118 = 2;
    const wp::int32 var_119 = 0;
    wp::float32 var_120;
    wp::float32 var_121;
    const wp::float32 var_122 = 0.5;
    wp::float32 var_123;
    const wp::int32 var_124 = 1;
    const wp::int32 var_125 = 2;
    wp::float32 var_126;
    const wp::int32 var_127 = 2;
    const wp::int32 var_128 = 1;
    wp::float32 var_129;
    wp::float32 var_130;
    const wp::float32 var_131 = 0.5;
    wp::float32 var_132;
    const wp::int32 var_133 = 0;
    const wp::int32 var_134 = 0;
    wp::float32 var_135;
    const wp::int32 var_136 = 1;
    const wp::int32 var_137 = 1;
    wp::float32 var_138;
    const wp::int32 var_139 = 2;
    const wp::int32 var_140 = 2;
    wp::float32 var_141;
    wp::mat_t<3, 3, wp::float32> var_142;
    const wp::float32 var_143 = 1e-08;
    const wp::float32 var_144 = 1e-05;
    wp::float32 var_145;
    wp::float32 var_146;
    wp::float32 var_147;
    wp::float32 var_148;
    wp::float32 var_149;
    wp::float32 var_150;
    wp::float32 var_151;
    wp::float32 var_152;
    wp::float32 var_153;
    bool var_154;
    const wp::int32 var_155 = 0;
    const wp::int32 var_156 = 1;
    wp::float32 var_157;
    wp::float32 var_158;
    wp::float32 var_159;
    bool var_160;
    const wp::int32 var_161 = 1;
    const wp::int32 var_162 = 0;
    wp::float32 var_163;
    wp::float32 var_164;
    wp::float32 var_165;
    bool var_166;
    const wp::int32 var_167 = 0;
    const wp::int32 var_168 = 2;
    wp::float32 var_169;
    wp::float32 var_170;
    wp::float32 var_171;
    bool var_172;
    const wp::int32 var_173 = 2;
    const wp::int32 var_174 = 0;
    wp::float32 var_175;
    wp::float32 var_176;
    wp::float32 var_177;
    bool var_178;
    const wp::int32 var_179 = 1;
    const wp::int32 var_180 = 2;
    wp::float32 var_181;
    wp::float32 var_182;
    wp::float32 var_183;
    bool var_184;
    const wp::int32 var_185 = 2;
    const wp::int32 var_186 = 1;
    wp::float32 var_187;
    wp::float32 var_188;
    wp::float32 var_189;
    bool var_190;
    const bool var_191 = true;
    bool var_192;
    wp::mat_t<3, 3, wp::float32> var_193;
    wp::mat_t<3, 3, wp::float32> var_194;
    wp::vec_t<3, wp::float32> var_195;
    const wp::int32 var_196 = 0;
    wp::float32 var_197;
    const wp::int32 var_198 = 1;
    wp::float32 var_199;
    const wp::int32 var_200 = 2;
    wp::float32 var_201;
    bool var_202;
    wp::float32 var_203;
    wp::float32 var_204;
    bool var_205;
    bool var_206;
    wp::float32 var_207;
    wp::float32 var_208;
    wp::float32 var_209;
    wp::float32 var_210;
    wp::float32 var_211;
    const wp::float32 var_212 = 1e-06;
    wp::float32 var_213;
    const wp::float32 var_214 = 1e-10;
    wp::float32 var_215;
    bool var_216;
    wp::float32 var_217;
    const wp::float32 var_218 = 1e-06;
    wp::float32 var_219;
    wp::float32 var_220;
    wp::float32 var_221;
    wp::float32 var_222;
    const wp::float32 var_223 = 0.0;
    const wp::float32 var_224 = 0.0;
    const wp::float32 var_225 = 0.0;
    const wp::float32 var_226 = 0.0;
    const wp::float32 var_227 = 0.0;
    const wp::float32 var_228 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_229;
    wp::mat_t<3, 3, wp::float32> var_230;
    const bool var_231 = true;
    wp::mat_t<3, 3, wp::float32> var_232;
    bool var_233;
    wp::float32 var_234;
    wp::float32 var_235;
    wp::float32 var_236;
    bool var_237;
    const wp::float32 var_238 = 0.0;
    bool var_239;
    bool var_240;
    wp::float32 var_241;
    wp::float32 var_242;
    wp::float32 var_243;
    wp::float32 var_244;
    const wp::float32 var_245 = 0.0;
    const wp::float32 var_246 = 0.0;
    const wp::float32 var_247 = 0.0;
    const wp::float32 var_248 = 0.0;
    const wp::float32 var_249 = 0.0;
    const wp::float32 var_250 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_251;
    wp::mat_t<3, 3, wp::float32> var_252;
    const bool var_253 = true;
    wp::mat_t<3, 3, wp::float32> var_254;
    bool var_255;
    wp::float32 var_256;
    wp::float32 var_257;
    wp::float32 var_258;
    wp::float32 var_259;
    const wp::float32 var_260 = 1.1920929e-07;
    wp::float32 var_261;
    const wp::float32 var_262 = 1e-10;
    wp::float32 var_263;
    bool var_264;
    wp::float32 var_265;
    wp::float32 var_266;
    bool var_267;
    wp::float32 var_268;
    wp::float32 var_269;
    const wp::float32 var_270 = 1e-06;
    wp::float32 var_271;
    const wp::float32 var_272 = 0.0;
    const wp::float32 var_273 = 0.0;
    const wp::float32 var_274 = 0.0;
    const wp::float32 var_275 = 0.0;
    const wp::float32 var_276 = 0.0;
    const wp::float32 var_277 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_278;
    wp::mat_t<3, 3, wp::float32> var_279;
    const bool var_280 = true;
    wp::mat_t<3, 3, wp::float32> var_281;
    bool var_282;
    wp::float32 var_283;
    wp::mat_t<3, 3, wp::float32> var_284;
    bool var_285;
    wp::mat_t<3, 3, wp::float32> var_286;
    const wp::float32 var_287 = 0.0;
    bool var_288;
    const wp::float32 var_289 = 1.0;
    wp::float32 var_290;
    const wp::float32 var_291 = 0.0;
    const wp::float32 var_292 = 0.0;
    bool var_293;
    wp::mat_t<3, 3, wp::float32> var_294;
    const wp::float32 var_295 = 0.0;
    const wp::float32 var_296 = 0.0;
    const wp::float32 var_297 = 0.0;
    const wp::float32 var_298 = 0.0;
    const wp::float32 var_299 = 0.0;
    const wp::float32 var_300 = 0.0;
    const wp::float32 var_301 = 0.0;
    const wp::float32 var_302 = 0.0;
    const wp::float32 var_303 = 0.0;
    wp::mat_t<3, 3, wp::float32> var_304;
    const wp::int32 var_305 = 0;
    const wp::int32 var_306 = 1;
    wp::int32 var_307;
    //---------
    // forward
    // def validate_and_correct_inertia_kernel(                                               <L 806>
    // tid = wp.tid()                                                                         <L 822>
    var_0 = builtin_tid1d();
    // mass = body_mass[tid]                                                                  <L 824>
    var_1 = wp::address(var_body_mass, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // inertia = body_inertia[tid]                                                            <L 825>
    var_4 = wp::address(var_body_inertia, var_0);
    var_6 = wp::load(var_4);
    var_5 = wp::copy(var_6);
    // original_inertia = inertia                                                             <L 826>
    var_7 = wp::copy(var_5);
    // was_corrected = False                                                                  <L 827>
    // if (                                                                                   <L 830>
    // not wp.isfinite(mass)                                                                  <L 831>
    var_10 = wp::isfinite(var_2);
    var_11 = wp::unot(var_10);
    var_9 = var_11;
    if (!var_9) {
        // or not wp.isfinite(inertia[0, 0])                                                  <L 832>
        var_14 = wp::extract(var_5, var_12, var_13);
        var_15 = wp::isfinite(var_14);
        var_16 = wp::unot(var_15);
        var_9 = var_9 || var_16;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[0, 1])                                                  <L 833>
        var_19 = wp::extract(var_5, var_17, var_18);
        var_20 = wp::isfinite(var_19);
        var_21 = wp::unot(var_20);
        var_9 = var_9 || var_21;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[0, 2])                                                  <L 834>
        var_24 = wp::extract(var_5, var_22, var_23);
        var_25 = wp::isfinite(var_24);
        var_26 = wp::unot(var_25);
        var_9 = var_9 || var_26;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[1, 0])                                                  <L 835>
        var_29 = wp::extract(var_5, var_27, var_28);
        var_30 = wp::isfinite(var_29);
        var_31 = wp::unot(var_30);
        var_9 = var_9 || var_31;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[1, 1])                                                  <L 836>
        var_34 = wp::extract(var_5, var_32, var_33);
        var_35 = wp::isfinite(var_34);
        var_36 = wp::unot(var_35);
        var_9 = var_9 || var_36;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[1, 2])                                                  <L 837>
        var_39 = wp::extract(var_5, var_37, var_38);
        var_40 = wp::isfinite(var_39);
        var_41 = wp::unot(var_40);
        var_9 = var_9 || var_41;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[2, 0])                                                  <L 838>
        var_44 = wp::extract(var_5, var_42, var_43);
        var_45 = wp::isfinite(var_44);
        var_46 = wp::unot(var_45);
        var_9 = var_9 || var_46;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[2, 1])                                                  <L 839>
        var_49 = wp::extract(var_5, var_47, var_48);
        var_50 = wp::isfinite(var_49);
        var_51 = wp::unot(var_50);
        var_9 = var_9 || var_51;
    }
    if (!var_9) {
        // or not wp.isfinite(inertia[2, 2])                                                  <L 840>
        var_54 = wp::extract(var_5, var_52, var_53);
        var_55 = wp::isfinite(var_54);
        var_56 = wp::unot(var_55);
        var_9 = var_9 || var_56;
    }
    if (var_9) {
        // mass = 0.0                                                                         <L 842>
        // inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                    <L 843>
        var_67 = wp::mat_t<3, 3, wp::float32>(var_58, var_59, var_60, var_61, var_62, var_63, var_64, var_65, var_66);
        // was_corrected = True                                                               <L 844>
    }
    var_69 = wp::where(var_9, var_57, var_2);
    var_70 = wp::where(var_9, var_67, var_5);
    var_71 = wp::where(var_9, var_68, var_8);
    // if mass < 0.0:                                                                         <L 847>
    var_73 = (var_69 < var_72);
    if (var_73) {
        // mass = 0.0                                                                         <L 848>
        // was_corrected = True                                                               <L 849>
    }
    var_76 = wp::where(var_73, var_74, var_69);
    var_77 = wp::where(var_73, var_75, var_71);
    // if bound_mass > 0.0 and mass < bound_mass and mass > 0.0:                              <L 852>
    var_80 = (var_bound_mass > var_79);
    var_78 = var_80;
    if (var_78) {
        var_81 = (var_76 < var_bound_mass);
        var_78 = var_78 && var_81;
    }
    if (var_78) {
        var_83 = (var_76 > var_82);
        var_78 = var_78 && var_83;
    }
    if (var_78) {
        // mass = bound_mass                                                                  <L 853>
        var_84 = wp::copy(var_bound_mass);
        // was_corrected = True                                                               <L 854>
    }
    var_86 = wp::where(var_78, var_84, var_76);
    var_87 = wp::where(var_78, var_85, var_77);
    // if mass == 0.0:                                                                        <L 857>
    var_89 = (var_86 == var_88);
    if (var_89) {
        // was_corrected = was_corrected or (wp.ddot(inertia, inertia) > 0.0)                 <L 858>
        var_90 = var_87;
        if (!var_90) {
            var_91 = wp::ddot(var_70, var_70);
            var_93 = (var_91 > var_92);
            var_90 = var_90 || var_93;
        }
        // inertia = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)                    <L 859>
        var_103 = wp::mat_t<3, 3, wp::float32>(var_94, var_95, var_96, var_97, var_98, var_99, var_100, var_101, var_102);
    }
    var_104 = wp::where(var_89, var_103, var_70);
    var_105 = wp::where(var_89, var_90, var_87);
    if (!var_89) {
        // sym01 = (inertia[0, 1] + inertia[1, 0]) * 0.5                                      <L 862>
        var_108 = wp::extract(var_104, var_106, var_107);
        var_111 = wp::extract(var_104, var_109, var_110);
        var_112 = wp::add(var_108, var_111);
        var_114 = wp::mul(var_112, var_113);
        // sym02 = (inertia[0, 2] + inertia[2, 0]) * 0.5                                      <L 863>
        var_117 = wp::extract(var_104, var_115, var_116);
        var_120 = wp::extract(var_104, var_118, var_119);
        var_121 = wp::add(var_117, var_120);
        var_123 = wp::mul(var_121, var_122);
        // sym12 = (inertia[1, 2] + inertia[2, 1]) * 0.5                                      <L 864>
        var_126 = wp::extract(var_104, var_124, var_125);
        var_129 = wp::extract(var_104, var_127, var_128);
        var_130 = wp::add(var_126, var_129);
        var_132 = wp::mul(var_130, var_131);
        // sym = wp.mat33(                                                                    <L 865>
        // inertia[0, 0],                                                                     <L 866>
        var_135 = wp::extract(var_104, var_133, var_134);
        // sym01,                                                                             <L 867>
        // sym02,                                                                             <L 868>
        // sym01,                                                                             <L 869>
        // inertia[1, 1],                                                                     <L 870>
        var_138 = wp::extract(var_104, var_136, var_137);
        // sym12,                                                                             <L 871>
        // sym02,                                                                             <L 872>
        // sym12,                                                                             <L 873>
        // inertia[2, 2],                                                                     <L 874>
        var_141 = wp::extract(var_104, var_139, var_140);
        var_142 = wp::mat_t<3, 3, wp::float32>(var_135, var_114, var_123, var_114, var_138, var_132, var_123, var_132, var_141);
        // tol01 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym01)            <L 877>
        var_145 = wp::abs(var_114);
        var_146 = wp::mul(var_144, var_145);
        var_147 = wp::add(var_143, var_146);
        // tol02 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym02)            <L 878>
        var_148 = wp::abs(var_123);
        var_149 = wp::mul(var_144, var_148);
        var_150 = wp::add(var_143, var_149);
        // tol12 = _INERTIA_SYMMETRY_ATOL + _INERTIA_SYMMETRY_RTOL * wp.abs(sym12)            <L 879>
        var_151 = wp::abs(var_132);
        var_152 = wp::mul(var_144, var_151);
        var_153 = wp::add(var_143, var_152);
        // if (                                                                               <L 880>
        // wp.abs(inertia[0, 1] - sym01) > tol01                                              <L 881>
        var_157 = wp::extract(var_104, var_155, var_156);
        var_158 = wp::sub(var_157, var_114);
        var_159 = wp::abs(var_158);
        var_160 = (var_159 > var_147);
        var_154 = var_160;
        if (!var_154) {
            // or wp.abs(inertia[1, 0] - sym01) > tol01                                       <L 882>
            var_163 = wp::extract(var_104, var_161, var_162);
            var_164 = wp::sub(var_163, var_114);
            var_165 = wp::abs(var_164);
            var_166 = (var_165 > var_147);
            var_154 = var_154 || var_166;
        }
        if (!var_154) {
            // or wp.abs(inertia[0, 2] - sym02) > tol02                                       <L 883>
            var_169 = wp::extract(var_104, var_167, var_168);
            var_170 = wp::sub(var_169, var_123);
            var_171 = wp::abs(var_170);
            var_172 = (var_171 > var_150);
            var_154 = var_154 || var_172;
        }
        if (!var_154) {
            // or wp.abs(inertia[2, 0] - sym02) > tol02                                       <L 884>
            var_175 = wp::extract(var_104, var_173, var_174);
            var_176 = wp::sub(var_175, var_123);
            var_177 = wp::abs(var_176);
            var_178 = (var_177 > var_150);
            var_154 = var_154 || var_178;
        }
        if (!var_154) {
            // or wp.abs(inertia[1, 2] - sym12) > tol12                                       <L 885>
            var_181 = wp::extract(var_104, var_179, var_180);
            var_182 = wp::sub(var_181, var_132);
            var_183 = wp::abs(var_182);
            var_184 = (var_183 > var_153);
            var_154 = var_154 || var_184;
        }
        if (!var_154) {
            // or wp.abs(inertia[2, 1] - sym12) > tol12                                       <L 886>
            var_187 = wp::extract(var_104, var_185, var_186);
            var_188 = wp::sub(var_187, var_132);
            var_189 = wp::abs(var_188);
            var_190 = (var_189 > var_153);
            var_154 = var_154 || var_190;
        }
        if (var_154) {
            // was_corrected = True                                                           <L 888>
        }
        var_192 = wp::where(var_154, var_191, var_105);
        // inertia = sym                                                                      <L 889>
        var_193 = wp::copy(var_142);
        // _eigvecs, eigvals = wp.eig3(inertia)                                               <L 892>
        wp::eig3(var_193, var_194, var_195);
        // I1, I2, I3 = eigvals[0], eigvals[1], eigvals[2]                                    <L 895>
        var_197 = wp::extract(var_195, var_196);
        var_199 = wp::extract(var_195, var_198);
        var_201 = wp::extract(var_195, var_200);
        // if I1 > I2:                                                                        <L 896>
        var_202 = (var_197 > var_199);
        if (var_202) {
            // I1, I2 = I2, I1                                                                <L 897>
        }
        var_203 = wp::where(var_202, var_199, var_197);
        var_204 = wp::where(var_202, var_197, var_199);
        // if I2 > I3:                                                                        <L 898>
        var_205 = (var_204 > var_201);
        if (var_205) {
            // I2, I3 = I3, I2                                                                <L 899>
            // if I1 > I2:                                                                    <L 900>
            var_206 = (var_203 > var_201);
            if (var_206) {
                // I1, I2 = I2, I1                                                            <L 901>
            }
            var_207 = wp::where(var_206, var_201, var_203);
            var_208 = wp::where(var_206, var_203, var_201);
        }
        var_209 = wp::where(var_205, var_207, var_203);
        var_210 = wp::where(var_205, var_208, var_204);
        var_211 = wp::where(var_205, var_204, var_201);
        // eig_threshold = wp.max(1.0e-6 * I3, 1.0e-10)                                       <L 905>
        var_213 = wp::mul(var_212, var_211);
        var_215 = wp::max(var_213, var_214);
        // if I1 < eig_threshold:                                                             <L 906>
        var_216 = (var_209 < var_215);
        if (var_216) {
            // adjustment = eig_threshold - I1 + 1.0e-6                                       <L 907>
            var_217 = wp::sub(var_215, var_209);
            var_219 = wp::add(var_217, var_218);
            // I1 += adjustment                                                               <L 909>
            var_220 = wp::add(var_209, var_219);
            // I2 += adjustment                                                               <L 910>
            var_221 = wp::add(var_210, var_219);
            // I3 += adjustment                                                               <L 911>
            var_222 = wp::add(var_211, var_219);
            // inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)       <L 912>
            var_229 = wp::mat_t<3, 3, wp::float32>(var_219, var_223, var_224, var_225, var_219, var_226, var_227, var_228, var_219);
            var_230 = wp::add(var_193, var_229);
            // was_corrected = True                                                           <L 913>
        }
        var_232 = wp::where(var_216, var_230, var_193);
        var_233 = wp::where(var_216, var_231, var_192);
        var_234 = wp::where(var_216, var_220, var_209);
        var_235 = wp::where(var_216, var_221, var_210);
        var_236 = wp::where(var_216, var_222, var_211);
        // if bound_inertia > 0.0 and I1 < bound_inertia:                                     <L 916>
        var_239 = (var_bound_inertia > var_238);
        var_237 = var_239;
        if (var_237) {
            var_240 = (var_234 < var_bound_inertia);
            var_237 = var_237 && var_240;
        }
        if (var_237) {
            // adjustment = bound_inertia - I1                                                <L 917>
            var_241 = wp::sub(var_bound_inertia, var_234);
            // I1 += adjustment                                                               <L 918>
            var_242 = wp::add(var_234, var_241);
            // I2 += adjustment                                                               <L 919>
            var_243 = wp::add(var_235, var_241);
            // I3 += adjustment                                                               <L 920>
            var_244 = wp::add(var_236, var_241);
            // inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)       <L 921>
            var_251 = wp::mat_t<3, 3, wp::float32>(var_241, var_245, var_246, var_247, var_241, var_248, var_249, var_250, var_241);
            var_252 = wp::add(var_232, var_251);
            // was_corrected = True                                                           <L 922>
        }
        var_254 = wp::where(var_237, var_252, var_232);
        var_255 = wp::where(var_237, var_253, var_233);
        var_256 = wp::where(var_237, var_242, var_234);
        var_257 = wp::where(var_237, var_243, var_235);
        var_258 = wp::where(var_237, var_244, var_236);
        var_259 = wp::where(var_237, var_241, var_219);
        // tri_tol = wp.max(1.1920929e-7 * I3, 1.0e-10)  # float32 eps * I3                   <L 925>
        var_261 = wp::mul(var_260, var_258);
        var_263 = wp::max(var_261, var_262);
        // if balance_inertia and (I1 + I2 < I3 - tri_tol):                                   <L 926>
        var_264 = var_balance_inertia;
        if (var_264) {
            var_265 = wp::add(var_256, var_257);
            var_266 = wp::sub(var_258, var_263);
            var_267 = (var_265 < var_266);
            var_264 = var_264 && var_267;
        }
        if (var_264) {
            // deficit = I3 - I1 - I2                                                         <L 927>
            var_268 = wp::sub(var_258, var_256);
            var_269 = wp::sub(var_268, var_257);
            // adjustment = deficit + 1.0e-6                                                  <L 928>
            var_271 = wp::add(var_269, var_270);
            // inertia = inertia + wp.mat33(adjustment, 0.0, 0.0, 0.0, adjustment, 0.0, 0.0, 0.0, adjustment)       <L 930>
            var_278 = wp::mat_t<3, 3, wp::float32>(var_271, var_272, var_273, var_274, var_271, var_275, var_276, var_277, var_271);
            var_279 = wp::add(var_254, var_278);
            // was_corrected = True                                                           <L 931>
        }
        var_281 = wp::where(var_264, var_279, var_254);
        var_282 = wp::where(var_264, var_280, var_255);
        var_283 = wp::where(var_264, var_271, var_259);
    }
    var_284 = wp::where(var_89, var_104, var_281);
    var_285 = wp::where(var_89, var_105, var_282);
    // output_inertia = inertia if was_corrected else original_inertia                        <L 933>
    if (var_285) {
    }
    if (!var_285) {
    }
    var_286 = wp::where(var_285, var_284, var_7);
    // body_mass[tid] = mass                                                                  <L 936>
    wp::array_store(var_body_mass, var_0, var_86);
    // body_inertia[tid] = output_inertia                                                     <L 937>
    wp::array_store(var_body_inertia, var_0, var_286);
    // if mass > 0.0:                                                                         <L 940>
    var_288 = (var_86 > var_287);
    if (var_288) {
        // body_inv_mass[tid] = 1.0 / mass                                                    <L 941>
        var_290 = wp::div(var_289, var_86);
        wp::array_store(var_body_inv_mass, var_0, var_290);
    }
    if (!var_288) {
        // body_inv_mass[tid] = 0.0                                                           <L 943>
        wp::array_store(var_body_inv_mass, var_0, var_291);
    }
    // if mass > 0.0:                                                                         <L 946>
    var_293 = (var_86 > var_292);
    if (var_293) {
        // body_inv_inertia[tid] = wp.inverse(output_inertia)                                 <L 947>
        var_294 = wp::inverse(var_286);
        wp::array_store(var_body_inv_inertia, var_0, var_294);
    }
    if (!var_293) {
        // body_inv_inertia[tid] = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)       <L 949>
        var_304 = wp::mat_t<3, 3, wp::float32>(var_295, var_296, var_297, var_298, var_299, var_300, var_301, var_302, var_303);
        wp::array_store(var_body_inv_inertia, var_0, var_304);
    }
    // if was_corrected:                                                                      <L 951>
    if (var_285) {
        // wp.atomic_add(correction_count, 0, 1)                                              <L 952>
        var_307 = wp::atomic_add(var_correction_count, var_305, var_306);
    }
}



extern "C" {

// Python CPU entry points
WP_API void validate_and_correct_inertia_kernel_0b4bb59b_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_validate_and_correct_inertia_kernel_0b4bb59b *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        validate_and_correct_inertia_kernel_0b4bb59b_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C

