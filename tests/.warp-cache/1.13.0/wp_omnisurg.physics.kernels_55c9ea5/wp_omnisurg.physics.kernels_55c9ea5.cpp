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


struct TriPointsConnector_7f248e59
{
    wp::int32 particle_id;
    wp::float32 rest_dist;
    wp::vec_t<3, wp::int32> tri_ids;
    wp::vec_t<3, wp::float32> tri_bar;


    TriPointsConnector_7f248e59() = default;
    CUDA_CALLABLE TriPointsConnector_7f248e59(wp::int32 const& particle_id,
    wp::float32 const& rest_dist = {},
    wp::vec_t<3, wp::int32> const& tri_ids = {},
    wp::vec_t<3, wp::float32> const& tri_bar = {})
        : particle_id{particle_id}
        , rest_dist{rest_dist}
        , tri_ids{tri_ids}
        , tri_bar{tri_bar}

    {
    }

    CUDA_CALLABLE TriPointsConnector_7f248e59& operator += (const TriPointsConnector_7f248e59& rhs)
    {    particle_id += rhs.particle_id;
    rest_dist += rhs.rest_dist;
    tri_ids += rhs.tri_ids;
    tri_bar += rhs.tri_bar;

        return *this;}

};

static CUDA_CALLABLE void adj_TriPointsConnector_7f248e59(wp::int32 const&,
    wp::float32 const&,
    wp::vec_t<3, wp::int32> const&,
    wp::vec_t<3, wp::float32> const&,
    wp::int32 & adj_particle_id,
    wp::float32 & adj_rest_dist,
    wp::vec_t<3, wp::int32> & adj_tri_ids,
    wp::vec_t<3, wp::float32> & adj_tri_bar,
    TriPointsConnector_7f248e59 & adj_ret)
{
    adj_particle_id += adj_ret.particle_id;
    adj_rest_dist += adj_ret.rest_dist;
    adj_tri_ids += adj_ret.tri_ids;
    adj_tri_bar += adj_ret.tri_bar;
}

// Required when compiling adjoints.
CUDA_CALLABLE TriPointsConnector_7f248e59 add(const TriPointsConnector_7f248e59& a, const TriPointsConnector_7f248e59& b)
{
    return TriPointsConnector_7f248e59();
}

CUDA_CALLABLE void adj_atomic_add(TriPointsConnector_7f248e59* p, TriPointsConnector_7f248e59 t)
{
    wp::adj_atomic_add(&p->particle_id, t.particle_id);
    wp::adj_atomic_add(&p->rest_dist, t.rest_dist);
    wp::adj_atomic_add(&p->tri_ids, t.tri_ids);
    wp::adj_atomic_add(&p->tri_bar, t.tri_bar);
}



struct Tetrahedron_4d48766a
{
    wp::vec_t<4, wp::int32> ids;
    wp::float32 rest_volume;


    Tetrahedron_4d48766a() = default;
    CUDA_CALLABLE Tetrahedron_4d48766a(wp::vec_t<4, wp::int32> const& ids,
    wp::float32 const& rest_volume = {})
        : ids{ids}
        , rest_volume{rest_volume}

    {
    }

    CUDA_CALLABLE Tetrahedron_4d48766a& operator += (const Tetrahedron_4d48766a& rhs)
    {    ids += rhs.ids;
    rest_volume += rhs.rest_volume;

        return *this;}

};

static CUDA_CALLABLE void adj_Tetrahedron_4d48766a(wp::vec_t<4, wp::int32> const&,
    wp::float32 const&,
    wp::vec_t<4, wp::int32> & adj_ids,
    wp::float32 & adj_rest_volume,
    Tetrahedron_4d48766a & adj_ret)
{
    adj_ids += adj_ret.ids;
    adj_rest_volume += adj_ret.rest_volume;
}

// Required when compiling adjoints.
CUDA_CALLABLE Tetrahedron_4d48766a add(const Tetrahedron_4d48766a& a, const Tetrahedron_4d48766a& b)
{
    return Tetrahedron_4d48766a();
}

CUDA_CALLABLE void adj_atomic_add(Tetrahedron_4d48766a* p, Tetrahedron_4d48766a t)
{
    wp::adj_atomic_add(&p->ids, t.ids);
    wp::adj_atomic_add(&p->rest_volume, t.rest_volume);
}


struct wp_args_compute_position_deltas_cb116588 {
    wp::array_t<wp::vec_t<3, wp::float32>> positions_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> positions_current;
    wp::array_t<wp::vec_t<3, wp::float32>> deltas_out;
};


void compute_position_deltas_cb116588_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_compute_position_deltas_cb116588 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions_prev = _wp_args->positions_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions_current = _wp_args->positions_current;
    wp::array_t<wp::vec_t<3, wp::float32>> var_deltas_out = _wp_args->deltas_out;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // forward
    // def compute_position_deltas(                                                           <L 104>
    // tid = wp.tid()                                                                         <L 109>
    var_0 = builtin_tid1d();
    // deltas_out[tid] = positions_current[tid] - positions_prev[tid]                         <L 110>
    var_1 = wp::address(var_positions_current, var_0);
    var_2 = wp::address(var_positions_prev, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::sub(var_4, var_5);
    wp::array_store(var_deltas_out, var_0, var_3);
}



void compute_position_deltas_cb116588_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_compute_position_deltas_cb116588 *_wp_args,
    wp_args_compute_position_deltas_cb116588 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions_prev = _wp_args->positions_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions_current = _wp_args->positions_current;
    wp::array_t<wp::vec_t<3, wp::float32>> var_deltas_out = _wp_args->deltas_out;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions_prev = _wp_adj_args->positions_prev;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions_current = _wp_adj_args->positions_current;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_deltas_out = _wp_adj_args->deltas_out;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::vec_t<3, wp::float32>* var_1;
    wp::vec_t<3, wp::float32>* var_2;
    wp::vec_t<3, wp::float32> var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::vec_t<3, wp::float32> adj_1 = {};
    wp::vec_t<3, wp::float32> adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    //---------
    // forward
    // def compute_position_deltas(                                                           <L 104>
    // tid = wp.tid()                                                                         <L 109>
    var_0 = builtin_tid1d();
    // deltas_out[tid] = positions_current[tid] - positions_prev[tid]                         <L 110>
    var_1 = wp::address(var_positions_current, var_0);
    var_2 = wp::address(var_positions_prev, var_0);
    var_4 = wp::load(var_1);
    var_5 = wp::load(var_2);
    var_3 = wp::sub(var_4, var_5);
    // wp::array_store(var_deltas_out, var_0, var_3);
    //---------
    // reverse
    wp::adj_array_store(var_deltas_out, var_0, var_3, adj_deltas_out, adj_0, adj_3);
    wp::adj_sub(var_4, var_5, adj_1, adj_2, adj_3);
    wp::adj_address(var_positions_prev, var_0, adj_positions_prev, adj_0, adj_2);
    wp::adj_address(var_positions_current, var_0, adj_positions_current, adj_0, adj_1);
    // adj: deltas_out[tid] = positions_current[tid] - positions_prev[tid]                    <L 110>
    // adj: tid = wp.tid()                                                                    <L 109>
    // adj: def compute_position_deltas(                                                      <L 104>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void compute_position_deltas_cb116588_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_compute_position_deltas_cb116588 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        compute_position_deltas_cb116588_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void compute_position_deltas_cb116588_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_compute_position_deltas_cb116588 *_wp_args,
    wp_args_compute_position_deltas_cb116588 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        compute_position_deltas_cb116588_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_solve_distance_constraints_57e3c0d4 {
    wp::array_t<wp::vec_t<3, wp::float32>> x;
    wp::array_t<wp::vec_t<3, wp::float32>> v;
    wp::array_t<wp::float32> invmass;
    wp::array_t<wp::int32> spring_indices;
    wp::array_t<wp::float32> spring_rest_lengths;
    wp::array_t<wp::float32> spring_stiffness;
    wp::array_t<wp::float32> spring_damping;
    wp::float32 dt;
    wp::array_t<wp::float32> lambdas;
    wp::array_t<wp::vec_t<3, wp::float32>> delta;
    wp::array_t<wp::int32> delta_counter;
};


void solve_distance_constraints_57e3c0d4_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_solve_distance_constraints_57e3c0d4 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_x = _wp_args->x;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v = _wp_args->v;
    wp::array_t<wp::float32> var_invmass = _wp_args->invmass;
    wp::array_t<wp::int32> var_spring_indices = _wp_args->spring_indices;
    wp::array_t<wp::float32> var_spring_rest_lengths = _wp_args->spring_rest_lengths;
    wp::array_t<wp::float32> var_spring_stiffness = _wp_args->spring_stiffness;
    wp::array_t<wp::float32> var_spring_damping = _wp_args->spring_damping;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::float32> var_lambdas = _wp_args->lambdas;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta = _wp_args->delta;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 2;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    wp::int32* var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::float32* var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32* var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::float32* var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32* var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    bool var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 0.0;
    bool var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::float32 var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::float32 var_56 = -1.0;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::int32 var_67 = 1;
    wp::int32 var_68;
    const wp::int32 var_69 = 1;
    wp::int32 var_70;
    //---------
    // forward
    // def solve_distance_constraints(                                                        <L 8>
    // tid = wp.tid()                                                                         <L 21>
    var_0 = builtin_tid1d();
    // i = spring_indices[tid * 2 + 0]                                                        <L 23>
    var_2 = wp::mul(var_0, var_1);
    var_4 = wp::add(var_2, var_3);
    var_5 = wp::address(var_spring_indices, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // j = spring_indices[tid * 2 + 1]                                                        <L 24>
    var_9 = wp::mul(var_0, var_8);
    var_11 = wp::add(var_9, var_10);
    var_12 = wp::address(var_spring_indices, var_11);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // ke = spring_stiffness[tid]                                                             <L 26>
    var_15 = wp::address(var_spring_stiffness, var_0);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // kd = spring_damping[tid]                                                               <L 27>
    var_18 = wp::address(var_spring_damping, var_0);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // rest = spring_rest_lengths[tid]                                                        <L 28>
    var_21 = wp::address(var_spring_rest_lengths, var_0);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // xi = x[i]                                                                              <L 30>
    var_24 = wp::address(var_x, var_6);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // xj = x[j]                                                                              <L 31>
    var_27 = wp::address(var_x, var_13);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // vi = v[i]                                                                              <L 33>
    var_30 = wp::address(var_v, var_6);
    var_32 = wp::load(var_30);
    var_31 = wp::copy(var_32);
    // vj = v[j]                                                                              <L 34>
    var_33 = wp::address(var_v, var_13);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // wi = invmass[i]                                                                        <L 36>
    var_36 = wp::address(var_invmass, var_6);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    // wj = invmass[j]                                                                        <L 37>
    var_39 = wp::address(var_invmass, var_13);
    var_41 = wp::load(var_39);
    var_40 = wp::copy(var_41);
    // w = wi + wj                                                                            <L 39>
    var_42 = wp::add(var_37, var_40);
    // if w <= 0.0 or ke <= 0.0:                                                              <L 40>
    var_45 = (var_42 <= var_44);
    var_43 = var_45;
    if (!var_43) {
        var_47 = (var_16 <= var_46);
        var_43 = var_43 || var_47;
    }
    if (var_43) {
        // return                                                                             <L 41>
        return;
    }
    // xij = xi - xj                                                                          <L 43>
    var_48 = wp::sub(var_25, var_28);
    // vij = vi - vj                                                                          <L 44>
    var_49 = wp::sub(var_31, var_34);
    // l = wp.length(xij)                                                                     <L 46>
    var_50 = wp::length(var_48);
    // if l == 0.0:                                                                           <L 47>
    var_52 = (var_50 == var_51);
    if (var_52) {
        // return                                                                             <L 48>
        return;
    }
    // n = xij / l                                                                            <L 50>
    var_53 = wp::div(var_48, var_50);
    // c = l - rest                                                                           <L 51>
    var_54 = wp::sub(var_50, var_22);
    // grad = n                                                                               <L 52>
    var_55 = wp::copy(var_53);
    // dlambda = -1.0 * (c / w) * ke                                                          <L 53>
    var_57 = wp::div(var_54, var_42);
    var_58 = wp::mul(var_56, var_57);
    var_59 = wp::mul(var_58, var_16);
    // dxi = wi * dlambda * grad                                                              <L 55>
    var_60 = wp::mul(var_37, var_59);
    var_61 = wp::mul(var_60, var_55);
    // dxj = wj * dlambda * -grad                                                             <L 56>
    var_62 = wp::mul(var_40, var_59);
    var_63 = wp::neg(var_55);
    var_64 = wp::mul(var_62, var_63);
    // wp.atomic_add(delta, i, dxi)                                                           <L 58>
    var_65 = wp::atomic_add(var_delta, var_6, var_61);
    // wp.atomic_add(delta, j, dxj)                                                           <L 59>
    var_66 = wp::atomic_add(var_delta, var_13, var_64);
    // wp.atomic_add(delta_counter, i, 1)                                                     <L 60>
    var_68 = wp::atomic_add(var_delta_counter, var_6, var_67);
    // wp.atomic_add(delta_counter, j, 1)                                                     <L 61>
    var_70 = wp::atomic_add(var_delta_counter, var_13, var_69);
}



void solve_distance_constraints_57e3c0d4_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_solve_distance_constraints_57e3c0d4 *_wp_args,
    wp_args_solve_distance_constraints_57e3c0d4 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_x = _wp_args->x;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v = _wp_args->v;
    wp::array_t<wp::float32> var_invmass = _wp_args->invmass;
    wp::array_t<wp::int32> var_spring_indices = _wp_args->spring_indices;
    wp::array_t<wp::float32> var_spring_rest_lengths = _wp_args->spring_rest_lengths;
    wp::array_t<wp::float32> var_spring_stiffness = _wp_args->spring_stiffness;
    wp::array_t<wp::float32> var_spring_damping = _wp_args->spring_damping;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::float32> var_lambdas = _wp_args->lambdas;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta = _wp_args->delta;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_x = _wp_adj_args->x;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_v = _wp_adj_args->v;
    wp::array_t<wp::float32> adj_invmass = _wp_adj_args->invmass;
    wp::array_t<wp::int32> adj_spring_indices = _wp_adj_args->spring_indices;
    wp::array_t<wp::float32> adj_spring_rest_lengths = _wp_adj_args->spring_rest_lengths;
    wp::array_t<wp::float32> adj_spring_stiffness = _wp_adj_args->spring_stiffness;
    wp::array_t<wp::float32> adj_spring_damping = _wp_adj_args->spring_damping;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::array_t<wp::float32> adj_lambdas = _wp_adj_args->lambdas;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_delta = _wp_adj_args->delta;
    wp::array_t<wp::int32> adj_delta_counter = _wp_adj_args->delta_counter;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::int32 var_1 = 2;
    wp::int32 var_2;
    const wp::int32 var_3 = 0;
    wp::int32 var_4;
    wp::int32* var_5;
    wp::int32 var_6;
    wp::int32 var_7;
    const wp::int32 var_8 = 2;
    wp::int32 var_9;
    const wp::int32 var_10 = 1;
    wp::int32 var_11;
    wp::int32* var_12;
    wp::int32 var_13;
    wp::int32 var_14;
    wp::float32* var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32* var_18;
    wp::float32 var_19;
    wp::float32 var_20;
    wp::float32* var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::vec_t<3, wp::float32>* var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32>* var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32>* var_30;
    wp::vec_t<3, wp::float32> var_31;
    wp::vec_t<3, wp::float32> var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::vec_t<3, wp::float32> var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::float32* var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32* var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    bool var_43;
    const wp::float32 var_44 = 0.0;
    bool var_45;
    const wp::float32 var_46 = 0.0;
    bool var_47;
    wp::vec_t<3, wp::float32> var_48;
    wp::vec_t<3, wp::float32> var_49;
    wp::float32 var_50;
    const wp::float32 var_51 = 0.0;
    bool var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::float32 var_54;
    wp::vec_t<3, wp::float32> var_55;
    const wp::float32 var_56 = -1.0;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::float32 var_62;
    wp::vec_t<3, wp::float32> var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    const wp::int32 var_67 = 1;
    wp::int32 var_68;
    const wp::int32 var_69 = 1;
    wp::int32 var_70;
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
    wp::float32 adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    wp::float32 adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
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
    wp::vec_t<3, wp::float32> adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    bool adj_43 = {};
    wp::float32 adj_44 = {};
    bool adj_45 = {};
    wp::float32 adj_46 = {};
    bool adj_47 = {};
    wp::vec_t<3, wp::float32> adj_48 = {};
    wp::vec_t<3, wp::float32> adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    bool adj_52 = {};
    wp::vec_t<3, wp::float32> adj_53 = {};
    wp::float32 adj_54 = {};
    wp::vec_t<3, wp::float32> adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::float32 adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::vec_t<3, wp::float32> adj_64 = {};
    wp::vec_t<3, wp::float32> adj_65 = {};
    wp::vec_t<3, wp::float32> adj_66 = {};
    wp::int32 adj_67 = {};
    wp::int32 adj_68 = {};
    wp::int32 adj_69 = {};
    wp::int32 adj_70 = {};
    //---------
    // forward
    // def solve_distance_constraints(                                                        <L 8>
    // tid = wp.tid()                                                                         <L 21>
    var_0 = builtin_tid1d();
    // i = spring_indices[tid * 2 + 0]                                                        <L 23>
    var_2 = wp::mul(var_0, var_1);
    var_4 = wp::add(var_2, var_3);
    var_5 = wp::address(var_spring_indices, var_4);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // j = spring_indices[tid * 2 + 1]                                                        <L 24>
    var_9 = wp::mul(var_0, var_8);
    var_11 = wp::add(var_9, var_10);
    var_12 = wp::address(var_spring_indices, var_11);
    var_14 = wp::load(var_12);
    var_13 = wp::copy(var_14);
    // ke = spring_stiffness[tid]                                                             <L 26>
    var_15 = wp::address(var_spring_stiffness, var_0);
    var_17 = wp::load(var_15);
    var_16 = wp::copy(var_17);
    // kd = spring_damping[tid]                                                               <L 27>
    var_18 = wp::address(var_spring_damping, var_0);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // rest = spring_rest_lengths[tid]                                                        <L 28>
    var_21 = wp::address(var_spring_rest_lengths, var_0);
    var_23 = wp::load(var_21);
    var_22 = wp::copy(var_23);
    // xi = x[i]                                                                              <L 30>
    var_24 = wp::address(var_x, var_6);
    var_26 = wp::load(var_24);
    var_25 = wp::copy(var_26);
    // xj = x[j]                                                                              <L 31>
    var_27 = wp::address(var_x, var_13);
    var_29 = wp::load(var_27);
    var_28 = wp::copy(var_29);
    // vi = v[i]                                                                              <L 33>
    var_30 = wp::address(var_v, var_6);
    var_32 = wp::load(var_30);
    var_31 = wp::copy(var_32);
    // vj = v[j]                                                                              <L 34>
    var_33 = wp::address(var_v, var_13);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // wi = invmass[i]                                                                        <L 36>
    var_36 = wp::address(var_invmass, var_6);
    var_38 = wp::load(var_36);
    var_37 = wp::copy(var_38);
    // wj = invmass[j]                                                                        <L 37>
    var_39 = wp::address(var_invmass, var_13);
    var_41 = wp::load(var_39);
    var_40 = wp::copy(var_41);
    // w = wi + wj                                                                            <L 39>
    var_42 = wp::add(var_37, var_40);
    // if w <= 0.0 or ke <= 0.0:                                                              <L 40>
    var_45 = (var_42 <= var_44);
    var_43 = var_45;
    if (!var_43) {
        var_47 = (var_16 <= var_46);
        var_43 = var_43 || var_47;
    }
    if (var_43) {
        // return                                                                             <L 41>
        goto label0;
    }
    // xij = xi - xj                                                                          <L 43>
    var_48 = wp::sub(var_25, var_28);
    // vij = vi - vj                                                                          <L 44>
    var_49 = wp::sub(var_31, var_34);
    // l = wp.length(xij)                                                                     <L 46>
    var_50 = wp::length(var_48);
    // if l == 0.0:                                                                           <L 47>
    var_52 = (var_50 == var_51);
    if (var_52) {
        // return                                                                             <L 48>
        goto label1;
    }
    // n = xij / l                                                                            <L 50>
    var_53 = wp::div(var_48, var_50);
    // c = l - rest                                                                           <L 51>
    var_54 = wp::sub(var_50, var_22);
    // grad = n                                                                               <L 52>
    var_55 = wp::copy(var_53);
    // dlambda = -1.0 * (c / w) * ke                                                          <L 53>
    var_57 = wp::div(var_54, var_42);
    var_58 = wp::mul(var_56, var_57);
    var_59 = wp::mul(var_58, var_16);
    // dxi = wi * dlambda * grad                                                              <L 55>
    var_60 = wp::mul(var_37, var_59);
    var_61 = wp::mul(var_60, var_55);
    // dxj = wj * dlambda * -grad                                                             <L 56>
    var_62 = wp::mul(var_40, var_59);
    var_63 = wp::neg(var_55);
    var_64 = wp::mul(var_62, var_63);
    // wp.atomic_add(delta, i, dxi)                                                           <L 58>
    // var_65 = wp::atomic_add(var_delta, var_6, var_61);
    // wp.atomic_add(delta, j, dxj)                                                           <L 59>
    // var_66 = wp::atomic_add(var_delta, var_13, var_64);
    // wp.atomic_add(delta_counter, i, 1)                                                     <L 60>
    // var_68 = wp::atomic_add(var_delta_counter, var_6, var_67);
    // wp.atomic_add(delta_counter, j, 1)                                                     <L 61>
    // var_70 = wp::atomic_add(var_delta_counter, var_13, var_69);
    //---------
    // reverse
    wp::adj_atomic_add(var_delta_counter, var_13, var_69, adj_delta_counter, adj_13, adj_69, adj_70);
    // adj: wp.atomic_add(delta_counter, j, 1)                                                <L 61>
    wp::adj_atomic_add(var_delta_counter, var_6, var_67, adj_delta_counter, adj_6, adj_67, adj_68);
    // adj: wp.atomic_add(delta_counter, i, 1)                                                <L 60>
    wp::adj_atomic_add(var_delta, var_13, var_64, adj_delta, adj_13, adj_64, adj_66);
    // adj: wp.atomic_add(delta, j, dxj)                                                      <L 59>
    wp::adj_atomic_add(var_delta, var_6, var_61, adj_delta, adj_6, adj_61, adj_65);
    // adj: wp.atomic_add(delta, i, dxi)                                                      <L 58>
    wp::adj_mul(var_62, var_63, adj_62, adj_63, adj_64);
    wp::adj_neg(var_55, adj_55, adj_63);
    wp::adj_mul(var_40, var_59, adj_40, adj_59, adj_62);
    // adj: dxj = wj * dlambda * -grad                                                        <L 56>
    wp::adj_mul(var_60, var_55, adj_60, adj_55, adj_61);
    wp::adj_mul(var_37, var_59, adj_37, adj_59, adj_60);
    // adj: dxi = wi * dlambda * grad                                                         <L 55>
    wp::adj_mul(var_58, var_16, adj_58, adj_16, adj_59);
    wp::adj_mul(var_56, var_57, adj_56, adj_57, adj_58);
    wp::adj_div(var_54, var_42, var_57, adj_54, adj_42, adj_57);
    // adj: dlambda = -1.0 * (c / w) * ke                                                     <L 53>
    wp::adj_copy(var_53, adj_53, adj_55);
    // adj: grad = n                                                                          <L 52>
    wp::adj_sub(var_50, var_22, adj_50, adj_22, adj_54);
    // adj: c = l - rest                                                                      <L 51>
    wp::adj_div(var_48, var_50, adj_48, adj_50, adj_53);
    // adj: n = xij / l                                                                       <L 50>
    if (var_52) {
        label1:;
        // adj: return                                                                        <L 48>
    }
    // adj: if l == 0.0:                                                                      <L 47>
    wp::adj_length(var_48, var_50, adj_48, adj_50);
    // adj: l = wp.length(xij)                                                                <L 46>
    wp::adj_sub(var_31, var_34, adj_31, adj_34, adj_49);
    // adj: vij = vi - vj                                                                     <L 44>
    wp::adj_sub(var_25, var_28, adj_25, adj_28, adj_48);
    // adj: xij = xi - xj                                                                     <L 43>
    if (var_43) {
        label0:;
        // adj: return                                                                        <L 41>
    }
    if (!var_43) {
    }
    // adj: if w <= 0.0 or ke <= 0.0:                                                         <L 40>
    wp::adj_add(var_37, var_40, adj_37, adj_40, adj_42);
    // adj: w = wi + wj                                                                       <L 39>
    wp::adj_copy(var_41, adj_39, adj_40);
    wp::adj_address(var_invmass, var_13, adj_invmass, adj_13, adj_39);
    // adj: wj = invmass[j]                                                                   <L 37>
    wp::adj_copy(var_38, adj_36, adj_37);
    wp::adj_address(var_invmass, var_6, adj_invmass, adj_6, adj_36);
    // adj: wi = invmass[i]                                                                   <L 36>
    wp::adj_copy(var_35, adj_33, adj_34);
    wp::adj_address(var_v, var_13, adj_v, adj_13, adj_33);
    // adj: vj = v[j]                                                                         <L 34>
    wp::adj_copy(var_32, adj_30, adj_31);
    wp::adj_address(var_v, var_6, adj_v, adj_6, adj_30);
    // adj: vi = v[i]                                                                         <L 33>
    wp::adj_copy(var_29, adj_27, adj_28);
    wp::adj_address(var_x, var_13, adj_x, adj_13, adj_27);
    // adj: xj = x[j]                                                                         <L 31>
    wp::adj_copy(var_26, adj_24, adj_25);
    wp::adj_address(var_x, var_6, adj_x, adj_6, adj_24);
    // adj: xi = x[i]                                                                         <L 30>
    wp::adj_copy(var_23, adj_21, adj_22);
    wp::adj_address(var_spring_rest_lengths, var_0, adj_spring_rest_lengths, adj_0, adj_21);
    // adj: rest = spring_rest_lengths[tid]                                                   <L 28>
    wp::adj_copy(var_20, adj_18, adj_19);
    wp::adj_address(var_spring_damping, var_0, adj_spring_damping, adj_0, adj_18);
    // adj: kd = spring_damping[tid]                                                          <L 27>
    wp::adj_copy(var_17, adj_15, adj_16);
    wp::adj_address(var_spring_stiffness, var_0, adj_spring_stiffness, adj_0, adj_15);
    // adj: ke = spring_stiffness[tid]                                                        <L 26>
    wp::adj_copy(var_14, adj_12, adj_13);
    wp::adj_address(var_spring_indices, var_11, adj_spring_indices, adj_11, adj_12);
    wp::adj_add(var_9, var_10, adj_9, adj_10, adj_11);
    wp::adj_mul(var_0, var_8, adj_0, adj_8, adj_9);
    // adj: j = spring_indices[tid * 2 + 1]                                                   <L 24>
    wp::adj_copy(var_7, adj_5, adj_6);
    wp::adj_address(var_spring_indices, var_4, adj_spring_indices, adj_4, adj_5);
    wp::adj_add(var_2, var_3, adj_2, adj_3, adj_4);
    wp::adj_mul(var_0, var_1, adj_0, adj_1, adj_2);
    // adj: i = spring_indices[tid * 2 + 0]                                                   <L 23>
    // adj: tid = wp.tid()                                                                    <L 21>
    // adj: def solve_distance_constraints(                                                   <L 8>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void solve_distance_constraints_57e3c0d4_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_solve_distance_constraints_57e3c0d4 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        solve_distance_constraints_57e3c0d4_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void solve_distance_constraints_57e3c0d4_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_solve_distance_constraints_57e3c0d4 *_wp_args,
    wp_args_solve_distance_constraints_57e3c0d4 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        solve_distance_constraints_57e3c0d4_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_apply_deltas_and_zero_accumulators_df3e92c3 {
    wp::array_t<wp::vec_t<3, wp::float32>> delta;
    wp::array_t<wp::int32> delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> target;
};


void apply_deltas_and_zero_accumulators_df3e92c3_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_deltas_and_zero_accumulators_df3e92c3 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta = _wp_args->delta;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_target = _wp_args->target;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 0;
    bool var_3;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::int32* var_6;
    wp::float32 var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.0;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    //---------
    // forward
    // def apply_deltas_and_zero_accumulators(                                                <L 142>
    // tid = wp.tid()                                                                         <L 147>
    var_0 = builtin_tid1d();
    // if delta_counter[tid] > 0:                                                             <L 148>
    var_1 = wp::address(var_delta_counter, var_0);
    var_4 = wp::load(var_1);
    var_3 = (var_4 > var_2);
    if (var_3) {
        // target[tid] += delta[tid] / wp.float32(delta_counter[tid])                         <L 149>
        var_5 = wp::address(var_delta, var_0);
        var_6 = wp::address(var_delta_counter, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::float32(var_8);
        var_10 = wp::load(var_5);
        var_9 = wp::div(var_10, var_7);
        var_11 = wp::atomic_add(var_target, var_0, var_9);
    }
    // delta[tid] = wp.vec3(0.0, 0.0, 0.0)                                                    <L 151>
    var_15 = wp::vec_t<3, wp::float32>(var_12, var_13, var_14);
    wp::array_store(var_delta, var_0, var_15);
    // delta_counter[tid] = 0                                                                 <L 152>
    wp::array_store(var_delta_counter, var_0, var_16);
}



void apply_deltas_and_zero_accumulators_df3e92c3_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_deltas_and_zero_accumulators_df3e92c3 *_wp_args,
    wp_args_apply_deltas_and_zero_accumulators_df3e92c3 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta = _wp_args->delta;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> var_target = _wp_args->target;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_delta = _wp_adj_args->delta;
    wp::array_t<wp::int32> adj_delta_counter = _wp_adj_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_target = _wp_adj_args->target;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 0;
    bool var_3;
    wp::int32 var_4;
    wp::vec_t<3, wp::float32>* var_5;
    wp::int32* var_6;
    wp::float32 var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32> var_9;
    wp::vec_t<3, wp::float32> var_10;
    wp::vec_t<3, wp::float32> var_11;
    const wp::float32 var_12 = 0.0;
    const wp::float32 var_13 = 0.0;
    const wp::float32 var_14 = 0.0;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 0;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    bool adj_3 = {};
    wp::int32 adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::int32 adj_6 = {};
    wp::float32 adj_7 = {};
    wp::int32 adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::float32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::int32 adj_16 = {};
    //---------
    // forward
    // def apply_deltas_and_zero_accumulators(                                                <L 142>
    // tid = wp.tid()                                                                         <L 147>
    var_0 = builtin_tid1d();
    // if delta_counter[tid] > 0:                                                             <L 148>
    var_1 = wp::address(var_delta_counter, var_0);
    var_4 = wp::load(var_1);
    var_3 = (var_4 > var_2);
    if (var_3) {
        // target[tid] += delta[tid] / wp.float32(delta_counter[tid])                         <L 149>
        var_5 = wp::address(var_delta, var_0);
        var_6 = wp::address(var_delta_counter, var_0);
        var_8 = wp::load(var_6);
        var_7 = wp::float32(var_8);
        var_10 = wp::load(var_5);
        var_9 = wp::div(var_10, var_7);
        // var_11 = wp::atomic_add(var_target, var_0, var_9);
    }
    // delta[tid] = wp.vec3(0.0, 0.0, 0.0)                                                    <L 151>
    var_15 = wp::vec_t<3, wp::float32>(var_12, var_13, var_14);
    // wp::array_store(var_delta, var_0, var_15);
    // delta_counter[tid] = 0                                                                 <L 152>
    // wp::array_store(var_delta_counter, var_0, var_16);
    //---------
    // reverse
    wp::adj_array_store(var_delta_counter, var_0, var_16, adj_delta_counter, adj_0, adj_16);
    // adj: delta_counter[tid] = 0                                                            <L 152>
    wp::adj_array_store(var_delta, var_0, var_15, adj_delta, adj_0, adj_15);
    wp::adj_vec_t(var_12, var_13, var_14, adj_12, adj_13, adj_14, adj_15);
    // adj: delta[tid] = wp.vec3(0.0, 0.0, 0.0)                                               <L 151>
    if (var_3) {
        wp::adj_atomic_add(var_target, var_0, var_9, adj_target, adj_0, adj_9, adj_11);
        wp::adj_div(var_10, var_7, adj_5, adj_7, adj_9);
        wp::adj_float32(var_8, adj_6, adj_7);
        wp::adj_address(var_delta_counter, var_0, adj_delta_counter, adj_0, adj_6);
        wp::adj_address(var_delta, var_0, adj_delta, adj_0, adj_5);
        // adj: target[tid] += delta[tid] / wp.float32(delta_counter[tid])                    <L 149>
    }
    wp::adj_address(var_delta_counter, var_0, adj_delta_counter, adj_0, adj_1);
    // adj: if delta_counter[tid] > 0:                                                        <L 148>
    // adj: tid = wp.tid()                                                                    <L 147>
    // adj: def apply_deltas_and_zero_accumulators(                                           <L 142>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void apply_deltas_and_zero_accumulators_df3e92c3_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_apply_deltas_and_zero_accumulators_df3e92c3 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_deltas_and_zero_accumulators_df3e92c3_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void apply_deltas_and_zero_accumulators_df3e92c3_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_apply_deltas_and_zero_accumulators_df3e92c3 *_wp_args,
    wp_args_apply_deltas_and_zero_accumulators_df3e92c3 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_deltas_and_zero_accumulators_df3e92c3_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_apply_displacements_from_base_32f4ac82 {
    wp::array_t<wp::vec_t<3, wp::float32>> x_orig;
    wp::array_t<wp::int32> particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> displacement;
    wp::float32 dt;
    wp::float32 v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> x_out;
    wp::array_t<wp::vec_t<3, wp::float32>> v_out;
};


void apply_displacements_from_base_32f4ac82_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_displacements_from_base_32f4ac82 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_x_orig = _wp_args->x_orig;
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacement = _wp_args->displacement;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_v_max = _wp_args->v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> var_x_out = _wp_args->x_out;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v_out = _wp_args->v_out;
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
    //---------
    // forward
    // def apply_displacements_from_base(                                                     <L 114>
    // tid = wp.tid()                                                                         <L 123>
    var_0 = builtin_tid1d();
    // if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                                  <L 124>
    var_1 = wp::address(var_particle_flags, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::bit_and(var_4, var_2);
    var_6 = (var_3 == var_5);
    if (var_6) {
        // return                                                                             <L 125>
        return;
    }
    // x0 = x_orig[tid]                                                                       <L 127>
    var_7 = wp::address(var_x_orig, var_0);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // d = displacement[tid]                                                                  <L 128>
    var_10 = wp::address(var_displacement, var_0);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // x_new = x0 + d                                                                         <L 130>
    var_13 = wp::add(var_8, var_11);
    // v_new = d / dt                                                                         <L 131>
    var_14 = wp::div(var_11, var_dt);
    // v_new_mag = wp.length(v_new)                                                           <L 133>
    var_15 = wp::length(var_14);
    // if v_new_mag > v_max:                                                                  <L 134>
    var_16 = (var_15 > var_v_max);
    if (var_16) {
        // v_new *= v_max / v_new_mag                                                         <L 135>
        var_17 = wp::div(var_v_max, var_15);
        var_18 = wp::mul(var_14, var_17);
    }
    var_19 = wp::where(var_16, var_18, var_14);
    // x_out[tid] = x_new                                                                     <L 137>
    wp::array_store(var_x_out, var_0, var_13);
    // v_out[tid] = v_new                                                                     <L 138>
    wp::array_store(var_v_out, var_0, var_19);
}



void apply_displacements_from_base_32f4ac82_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_displacements_from_base_32f4ac82 *_wp_args,
    wp_args_apply_displacements_from_base_32f4ac82 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_x_orig = _wp_args->x_orig;
    wp::array_t<wp::int32> var_particle_flags = _wp_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> var_displacement = _wp_args->displacement;
    wp::float32 var_dt = _wp_args->dt;
    wp::float32 var_v_max = _wp_args->v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> var_x_out = _wp_args->x_out;
    wp::array_t<wp::vec_t<3, wp::float32>> var_v_out = _wp_args->v_out;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_x_orig = _wp_adj_args->x_orig;
    wp::array_t<wp::int32> adj_particle_flags = _wp_adj_args->particle_flags;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_displacement = _wp_adj_args->displacement;
    wp::float32 adj_dt = _wp_adj_args->dt;
    wp::float32 adj_v_max = _wp_adj_args->v_max;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_x_out = _wp_adj_args->x_out;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_v_out = _wp_adj_args->v_out;
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
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    wp::int32 adj_3 = {};
    wp::int32 adj_4 = {};
    wp::int32 adj_5 = {};
    bool adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::vec_t<3, wp::float32> adj_10 = {};
    wp::vec_t<3, wp::float32> adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::float32 adj_15 = {};
    bool adj_16 = {};
    wp::float32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    //---------
    // forward
    // def apply_displacements_from_base(                                                     <L 114>
    // tid = wp.tid()                                                                         <L 123>
    var_0 = builtin_tid1d();
    // if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                                  <L 124>
    var_1 = wp::address(var_particle_flags, var_0);
    var_4 = wp::load(var_1);
    var_3 = wp::bit_and(var_4, var_2);
    var_6 = (var_3 == var_5);
    if (var_6) {
        // return                                                                             <L 125>
        goto label0;
    }
    // x0 = x_orig[tid]                                                                       <L 127>
    var_7 = wp::address(var_x_orig, var_0);
    var_9 = wp::load(var_7);
    var_8 = wp::copy(var_9);
    // d = displacement[tid]                                                                  <L 128>
    var_10 = wp::address(var_displacement, var_0);
    var_12 = wp::load(var_10);
    var_11 = wp::copy(var_12);
    // x_new = x0 + d                                                                         <L 130>
    var_13 = wp::add(var_8, var_11);
    // v_new = d / dt                                                                         <L 131>
    var_14 = wp::div(var_11, var_dt);
    // v_new_mag = wp.length(v_new)                                                           <L 133>
    var_15 = wp::length(var_14);
    // if v_new_mag > v_max:                                                                  <L 134>
    var_16 = (var_15 > var_v_max);
    if (var_16) {
        // v_new *= v_max / v_new_mag                                                         <L 135>
        var_17 = wp::div(var_v_max, var_15);
        var_18 = wp::mul(var_14, var_17);
    }
    var_19 = wp::where(var_16, var_18, var_14);
    // x_out[tid] = x_new                                                                     <L 137>
    // wp::array_store(var_x_out, var_0, var_13);
    // v_out[tid] = v_new                                                                     <L 138>
    // wp::array_store(var_v_out, var_0, var_19);
    //---------
    // reverse
    wp::adj_array_store(var_v_out, var_0, var_19, adj_v_out, adj_0, adj_19);
    // adj: v_out[tid] = v_new                                                                <L 138>
    wp::adj_array_store(var_x_out, var_0, var_13, adj_x_out, adj_0, adj_13);
    // adj: x_out[tid] = x_new                                                                <L 137>
    wp::adj_where(var_16, var_18, var_14, adj_16, adj_18, adj_14, adj_19);
    if (var_16) {
        wp::adj_mul(var_14, var_17, adj_14, adj_17, adj_18);
        wp::adj_div(var_v_max, var_15, var_17, adj_v_max, adj_15, adj_17);
        // adj: v_new *= v_max / v_new_mag                                                    <L 135>
    }
    // adj: if v_new_mag > v_max:                                                             <L 134>
    wp::adj_length(var_14, var_15, adj_14, adj_15);
    // adj: v_new_mag = wp.length(v_new)                                                      <L 133>
    wp::adj_div(var_11, var_dt, adj_11, adj_dt, adj_14);
    // adj: v_new = d / dt                                                                    <L 131>
    wp::adj_add(var_8, var_11, adj_8, adj_11, adj_13);
    // adj: x_new = x0 + d                                                                    <L 130>
    wp::adj_copy(var_12, adj_10, adj_11);
    wp::adj_address(var_displacement, var_0, adj_displacement, adj_0, adj_10);
    // adj: d = displacement[tid]                                                             <L 128>
    wp::adj_copy(var_9, adj_7, adj_8);
    wp::adj_address(var_x_orig, var_0, adj_x_orig, adj_0, adj_7);
    // adj: x0 = x_orig[tid]                                                                  <L 127>
    if (var_6) {
        label0:;
        // adj: return                                                                        <L 125>
    }
    wp::adj_address(var_particle_flags, var_0, adj_particle_flags, adj_0, adj_1);
    // adj: if (particle_flags[tid] & ParticleFlags.ACTIVE) == 0:                             <L 124>
    // adj: tid = wp.tid()                                                                    <L 123>
    // adj: def apply_displacements_from_base(                                                <L 114>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void apply_displacements_from_base_32f4ac82_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_apply_displacements_from_base_32f4ac82 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_displacements_from_base_32f4ac82_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void apply_displacements_from_base_32f4ac82_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_apply_displacements_from_base_32f4ac82 *_wp_args,
    wp_args_apply_displacements_from_base_32f4ac82 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_displacements_from_base_32f4ac82_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_apply_fused_3_accumulators_9b0b2266 {
    wp::array_t<wp::vec_t<3, wp::float32>> da;
    wp::array_t<wp::int32> ca;
    wp::array_t<wp::vec_t<3, wp::float32>> db;
    wp::array_t<wp::int32> cb;
    wp::array_t<wp::vec_t<3, wp::float32>> dc;
    wp::array_t<wp::int32> cc;
    wp::array_t<wp::vec_t<3, wp::float32>> target;
};


void apply_fused_3_accumulators_9b0b2266_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_fused_3_accumulators_9b0b2266 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_da = _wp_args->da;
    wp::array_t<wp::int32> var_ca = _wp_args->ca;
    wp::array_t<wp::vec_t<3, wp::float32>> var_db = _wp_args->db;
    wp::array_t<wp::int32> var_cb = _wp_args->cb;
    wp::array_t<wp::vec_t<3, wp::float32>> var_dc = _wp_args->dc;
    wp::array_t<wp::int32> var_cc = _wp_args->cc;
    wp::array_t<wp::vec_t<3, wp::float32>> var_target = _wp_args->target;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    wp::int32* var_5;
    const wp::int32 var_6 = 0;
    bool var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::int32* var_10;
    wp::float32 var_11;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::int32* var_17;
    const wp::int32 var_18 = 0;
    bool var_19;
    wp::int32 var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::int32* var_22;
    wp::float32 var_23;
    wp::int32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32* var_29;
    const wp::int32 var_30 = 0;
    bool var_31;
    wp::int32 var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::int32* var_34;
    wp::float32 var_35;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::float32 var_44 = 0.0;
    const wp::float32 var_45 = 0.0;
    const wp::float32 var_46 = 0.0;
    wp::vec_t<3, wp::float32> var_47;
    const wp::int32 var_48 = 0;
    const wp::float32 var_49 = 0.0;
    const wp::float32 var_50 = 0.0;
    const wp::float32 var_51 = 0.0;
    wp::vec_t<3, wp::float32> var_52;
    const wp::int32 var_53 = 0;
    const wp::float32 var_54 = 0.0;
    const wp::float32 var_55 = 0.0;
    const wp::float32 var_56 = 0.0;
    wp::vec_t<3, wp::float32> var_57;
    const wp::int32 var_58 = 0;
    //---------
    // forward
    // def apply_fused_3_accumulators(                                                        <L 156>
    // tid = wp.tid()                                                                         <L 165>
    var_0 = builtin_tid1d();
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 166>
    var_4 = wp::vec_t<3, wp::float32>(var_1, var_2, var_3);
    // if ca[tid] > 0:                                                                        <L 167>
    var_5 = wp::address(var_ca, var_0);
    var_8 = wp::load(var_5);
    var_7 = (var_8 > var_6);
    if (var_7) {
        // result = result + da[tid] / wp.float32(ca[tid])                                    <L 168>
        var_9 = wp::address(var_da, var_0);
        var_10 = wp::address(var_ca, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::float32(var_12);
        var_14 = wp::load(var_9);
        var_13 = wp::div(var_14, var_11);
        var_15 = wp::add(var_4, var_13);
    }
    var_16 = wp::where(var_7, var_15, var_4);
    // if cb[tid] > 0:                                                                        <L 169>
    var_17 = wp::address(var_cb, var_0);
    var_20 = wp::load(var_17);
    var_19 = (var_20 > var_18);
    if (var_19) {
        // result = result + db[tid] / wp.float32(cb[tid])                                    <L 170>
        var_21 = wp::address(var_db, var_0);
        var_22 = wp::address(var_cb, var_0);
        var_24 = wp::load(var_22);
        var_23 = wp::float32(var_24);
        var_26 = wp::load(var_21);
        var_25 = wp::div(var_26, var_23);
        var_27 = wp::add(var_16, var_25);
    }
    var_28 = wp::where(var_19, var_27, var_16);
    // if cc[tid] > 0:                                                                        <L 171>
    var_29 = wp::address(var_cc, var_0);
    var_32 = wp::load(var_29);
    var_31 = (var_32 > var_30);
    if (var_31) {
        // result = result + dc[tid] / wp.float32(cc[tid])                                    <L 172>
        var_33 = wp::address(var_dc, var_0);
        var_34 = wp::address(var_cc, var_0);
        var_36 = wp::load(var_34);
        var_35 = wp::float32(var_36);
        var_38 = wp::load(var_33);
        var_37 = wp::div(var_38, var_35);
        var_39 = wp::add(var_28, var_37);
    }
    var_40 = wp::where(var_31, var_39, var_28);
    // target[tid] = target[tid] + result                                                     <L 174>
    var_41 = wp::address(var_target, var_0);
    var_43 = wp::load(var_41);
    var_42 = wp::add(var_43, var_40);
    wp::array_store(var_target, var_0, var_42);
    // da[tid] = wp.vec3(0.0, 0.0, 0.0)                                                       <L 176>
    var_47 = wp::vec_t<3, wp::float32>(var_44, var_45, var_46);
    wp::array_store(var_da, var_0, var_47);
    // ca[tid] = 0                                                                            <L 177>
    wp::array_store(var_ca, var_0, var_48);
    // db[tid] = wp.vec3(0.0, 0.0, 0.0)                                                       <L 178>
    var_52 = wp::vec_t<3, wp::float32>(var_49, var_50, var_51);
    wp::array_store(var_db, var_0, var_52);
    // cb[tid] = 0                                                                            <L 179>
    wp::array_store(var_cb, var_0, var_53);
    // dc[tid] = wp.vec3(0.0, 0.0, 0.0)                                                       <L 180>
    var_57 = wp::vec_t<3, wp::float32>(var_54, var_55, var_56);
    wp::array_store(var_dc, var_0, var_57);
    // cc[tid] = 0                                                                            <L 181>
    wp::array_store(var_cc, var_0, var_58);
}



void apply_fused_3_accumulators_9b0b2266_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_fused_3_accumulators_9b0b2266 *_wp_args,
    wp_args_apply_fused_3_accumulators_9b0b2266 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_da = _wp_args->da;
    wp::array_t<wp::int32> var_ca = _wp_args->ca;
    wp::array_t<wp::vec_t<3, wp::float32>> var_db = _wp_args->db;
    wp::array_t<wp::int32> var_cb = _wp_args->cb;
    wp::array_t<wp::vec_t<3, wp::float32>> var_dc = _wp_args->dc;
    wp::array_t<wp::int32> var_cc = _wp_args->cc;
    wp::array_t<wp::vec_t<3, wp::float32>> var_target = _wp_args->target;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_da = _wp_adj_args->da;
    wp::array_t<wp::int32> adj_ca = _wp_adj_args->ca;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_db = _wp_adj_args->db;
    wp::array_t<wp::int32> adj_cb = _wp_adj_args->cb;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_dc = _wp_adj_args->dc;
    wp::array_t<wp::int32> adj_cc = _wp_adj_args->cc;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_target = _wp_adj_args->target;
    //---------
    // primal vars
    wp::int32 var_0;
    const wp::float32 var_1 = 0.0;
    const wp::float32 var_2 = 0.0;
    const wp::float32 var_3 = 0.0;
    wp::vec_t<3, wp::float32> var_4;
    wp::int32* var_5;
    const wp::int32 var_6 = 0;
    bool var_7;
    wp::int32 var_8;
    wp::vec_t<3, wp::float32>* var_9;
    wp::int32* var_10;
    wp::float32 var_11;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    wp::vec_t<3, wp::float32> var_16;
    wp::int32* var_17;
    const wp::int32 var_18 = 0;
    bool var_19;
    wp::int32 var_20;
    wp::vec_t<3, wp::float32>* var_21;
    wp::int32* var_22;
    wp::float32 var_23;
    wp::int32 var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::float32> var_27;
    wp::vec_t<3, wp::float32> var_28;
    wp::int32* var_29;
    const wp::int32 var_30 = 0;
    bool var_31;
    wp::int32 var_32;
    wp::vec_t<3, wp::float32>* var_33;
    wp::int32* var_34;
    wp::float32 var_35;
    wp::int32 var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::vec_t<3, wp::float32> var_39;
    wp::vec_t<3, wp::float32> var_40;
    wp::vec_t<3, wp::float32>* var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    const wp::float32 var_44 = 0.0;
    const wp::float32 var_45 = 0.0;
    const wp::float32 var_46 = 0.0;
    wp::vec_t<3, wp::float32> var_47;
    const wp::int32 var_48 = 0;
    const wp::float32 var_49 = 0.0;
    const wp::float32 var_50 = 0.0;
    const wp::float32 var_51 = 0.0;
    wp::vec_t<3, wp::float32> var_52;
    const wp::int32 var_53 = 0;
    const wp::float32 var_54 = 0.0;
    const wp::float32 var_55 = 0.0;
    const wp::float32 var_56 = 0.0;
    wp::vec_t<3, wp::float32> var_57;
    const wp::int32 var_58 = 0;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::float32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    bool adj_7 = {};
    wp::int32 adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::vec_t<3, wp::float32> adj_16 = {};
    wp::int32 adj_17 = {};
    wp::int32 adj_18 = {};
    bool adj_19 = {};
    wp::int32 adj_20 = {};
    wp::vec_t<3, wp::float32> adj_21 = {};
    wp::int32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::int32 adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::vec_t<3, wp::float32> adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::int32 adj_29 = {};
    wp::int32 adj_30 = {};
    bool adj_31 = {};
    wp::int32 adj_32 = {};
    wp::vec_t<3, wp::float32> adj_33 = {};
    wp::int32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::vec_t<3, wp::float32> adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::vec_t<3, wp::float32> adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    wp::vec_t<3, wp::float32> adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::vec_t<3, wp::float32> adj_47 = {};
    wp::int32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::vec_t<3, wp::float32> adj_52 = {};
    wp::int32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::vec_t<3, wp::float32> adj_57 = {};
    wp::int32 adj_58 = {};
    //---------
    // forward
    // def apply_fused_3_accumulators(                                                        <L 156>
    // tid = wp.tid()                                                                         <L 165>
    var_0 = builtin_tid1d();
    // result = wp.vec3(0.0, 0.0, 0.0)                                                        <L 166>
    var_4 = wp::vec_t<3, wp::float32>(var_1, var_2, var_3);
    // if ca[tid] > 0:                                                                        <L 167>
    var_5 = wp::address(var_ca, var_0);
    var_8 = wp::load(var_5);
    var_7 = (var_8 > var_6);
    if (var_7) {
        // result = result + da[tid] / wp.float32(ca[tid])                                    <L 168>
        var_9 = wp::address(var_da, var_0);
        var_10 = wp::address(var_ca, var_0);
        var_12 = wp::load(var_10);
        var_11 = wp::float32(var_12);
        var_14 = wp::load(var_9);
        var_13 = wp::div(var_14, var_11);
        var_15 = wp::add(var_4, var_13);
    }
    var_16 = wp::where(var_7, var_15, var_4);
    // if cb[tid] > 0:                                                                        <L 169>
    var_17 = wp::address(var_cb, var_0);
    var_20 = wp::load(var_17);
    var_19 = (var_20 > var_18);
    if (var_19) {
        // result = result + db[tid] / wp.float32(cb[tid])                                    <L 170>
        var_21 = wp::address(var_db, var_0);
        var_22 = wp::address(var_cb, var_0);
        var_24 = wp::load(var_22);
        var_23 = wp::float32(var_24);
        var_26 = wp::load(var_21);
        var_25 = wp::div(var_26, var_23);
        var_27 = wp::add(var_16, var_25);
    }
    var_28 = wp::where(var_19, var_27, var_16);
    // if cc[tid] > 0:                                                                        <L 171>
    var_29 = wp::address(var_cc, var_0);
    var_32 = wp::load(var_29);
    var_31 = (var_32 > var_30);
    if (var_31) {
        // result = result + dc[tid] / wp.float32(cc[tid])                                    <L 172>
        var_33 = wp::address(var_dc, var_0);
        var_34 = wp::address(var_cc, var_0);
        var_36 = wp::load(var_34);
        var_35 = wp::float32(var_36);
        var_38 = wp::load(var_33);
        var_37 = wp::div(var_38, var_35);
        var_39 = wp::add(var_28, var_37);
    }
    var_40 = wp::where(var_31, var_39, var_28);
    // target[tid] = target[tid] + result                                                     <L 174>
    var_41 = wp::address(var_target, var_0);
    var_43 = wp::load(var_41);
    var_42 = wp::add(var_43, var_40);
    // wp::array_store(var_target, var_0, var_42);
    // da[tid] = wp.vec3(0.0, 0.0, 0.0)                                                       <L 176>
    var_47 = wp::vec_t<3, wp::float32>(var_44, var_45, var_46);
    // wp::array_store(var_da, var_0, var_47);
    // ca[tid] = 0                                                                            <L 177>
    // wp::array_store(var_ca, var_0, var_48);
    // db[tid] = wp.vec3(0.0, 0.0, 0.0)                                                       <L 178>
    var_52 = wp::vec_t<3, wp::float32>(var_49, var_50, var_51);
    // wp::array_store(var_db, var_0, var_52);
    // cb[tid] = 0                                                                            <L 179>
    // wp::array_store(var_cb, var_0, var_53);
    // dc[tid] = wp.vec3(0.0, 0.0, 0.0)                                                       <L 180>
    var_57 = wp::vec_t<3, wp::float32>(var_54, var_55, var_56);
    // wp::array_store(var_dc, var_0, var_57);
    // cc[tid] = 0                                                                            <L 181>
    // wp::array_store(var_cc, var_0, var_58);
    //---------
    // reverse
    wp::adj_array_store(var_cc, var_0, var_58, adj_cc, adj_0, adj_58);
    // adj: cc[tid] = 0                                                                       <L 181>
    wp::adj_array_store(var_dc, var_0, var_57, adj_dc, adj_0, adj_57);
    wp::adj_vec_t(var_54, var_55, var_56, adj_54, adj_55, adj_56, adj_57);
    // adj: dc[tid] = wp.vec3(0.0, 0.0, 0.0)                                                  <L 180>
    wp::adj_array_store(var_cb, var_0, var_53, adj_cb, adj_0, adj_53);
    // adj: cb[tid] = 0                                                                       <L 179>
    wp::adj_array_store(var_db, var_0, var_52, adj_db, adj_0, adj_52);
    wp::adj_vec_t(var_49, var_50, var_51, adj_49, adj_50, adj_51, adj_52);
    // adj: db[tid] = wp.vec3(0.0, 0.0, 0.0)                                                  <L 178>
    wp::adj_array_store(var_ca, var_0, var_48, adj_ca, adj_0, adj_48);
    // adj: ca[tid] = 0                                                                       <L 177>
    wp::adj_array_store(var_da, var_0, var_47, adj_da, adj_0, adj_47);
    wp::adj_vec_t(var_44, var_45, var_46, adj_44, adj_45, adj_46, adj_47);
    // adj: da[tid] = wp.vec3(0.0, 0.0, 0.0)                                                  <L 176>
    wp::adj_array_store(var_target, var_0, var_42, adj_target, adj_0, adj_42);
    wp::adj_add(var_43, var_40, adj_41, adj_40, adj_42);
    wp::adj_address(var_target, var_0, adj_target, adj_0, adj_41);
    // adj: target[tid] = target[tid] + result                                                <L 174>
    wp::adj_where(var_31, var_39, var_28, adj_31, adj_39, adj_28, adj_40);
    if (var_31) {
        wp::adj_add(var_28, var_37, adj_28, adj_37, adj_39);
        wp::adj_div(var_38, var_35, adj_33, adj_35, adj_37);
        wp::adj_float32(var_36, adj_34, adj_35);
        wp::adj_address(var_cc, var_0, adj_cc, adj_0, adj_34);
        wp::adj_address(var_dc, var_0, adj_dc, adj_0, adj_33);
        // adj: result = result + dc[tid] / wp.float32(cc[tid])                               <L 172>
    }
    wp::adj_address(var_cc, var_0, adj_cc, adj_0, adj_29);
    // adj: if cc[tid] > 0:                                                                   <L 171>
    wp::adj_where(var_19, var_27, var_16, adj_19, adj_27, adj_16, adj_28);
    if (var_19) {
        wp::adj_add(var_16, var_25, adj_16, adj_25, adj_27);
        wp::adj_div(var_26, var_23, adj_21, adj_23, adj_25);
        wp::adj_float32(var_24, adj_22, adj_23);
        wp::adj_address(var_cb, var_0, adj_cb, adj_0, adj_22);
        wp::adj_address(var_db, var_0, adj_db, adj_0, adj_21);
        // adj: result = result + db[tid] / wp.float32(cb[tid])                               <L 170>
    }
    wp::adj_address(var_cb, var_0, adj_cb, adj_0, adj_17);
    // adj: if cb[tid] > 0:                                                                   <L 169>
    wp::adj_where(var_7, var_15, var_4, adj_7, adj_15, adj_4, adj_16);
    if (var_7) {
        wp::adj_add(var_4, var_13, adj_4, adj_13, adj_15);
        wp::adj_div(var_14, var_11, adj_9, adj_11, adj_13);
        wp::adj_float32(var_12, adj_10, adj_11);
        wp::adj_address(var_ca, var_0, adj_ca, adj_0, adj_10);
        wp::adj_address(var_da, var_0, adj_da, adj_0, adj_9);
        // adj: result = result + da[tid] / wp.float32(ca[tid])                               <L 168>
    }
    wp::adj_address(var_ca, var_0, adj_ca, adj_0, adj_5);
    // adj: if ca[tid] > 0:                                                                   <L 167>
    wp::adj_vec_t(var_1, var_2, var_3, adj_1, adj_2, adj_3, adj_4);
    // adj: result = wp.vec3(0.0, 0.0, 0.0)                                                   <L 166>
    // adj: tid = wp.tid()                                                                    <L 165>
    // adj: def apply_fused_3_accumulators(                                                   <L 156>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void apply_fused_3_accumulators_9b0b2266_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_apply_fused_3_accumulators_9b0b2266 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_fused_3_accumulators_9b0b2266_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void apply_fused_3_accumulators_9b0b2266_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_apply_fused_3_accumulators_9b0b2266 *_wp_args,
    wp_args_apply_fused_3_accumulators_9b0b2266 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_fused_3_accumulators_9b0b2266_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_apply_tri_points_constraints_jacobian_1ebcce44 {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<TriPointsConnector_7f248e59> connectors;
    wp::array_t<wp::vec_t<3, wp::float32>> delta_accumulator;
};


void apply_tri_points_constraints_jacobian_1ebcce44_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_tri_points_constraints_jacobian_1ebcce44 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<TriPointsConnector_7f248e59> var_connectors = _wp_args->connectors;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta_accumulator = _wp_args->delta_accumulator;
    //---------
    // primal vars
    wp::int32 var_0;
    TriPointsConnector_7f248e59* var_1;
    TriPointsConnector_7f248e59 var_2;
    TriPointsConnector_7f248e59 var_3;
    wp::vec_t<3, wp::int32>* var_4;
    const wp::int32 var_5 = 0;
    wp::int32 var_6;
    wp::vec_t<3, wp::int32> var_7;
    wp::vec_t<3, wp::float32>* var_8;
    wp::vec_t<3, wp::float32>* var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::int32>* var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::vec_t<3, wp::int32> var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::vec_t<3, wp::float32>* var_20;
    const wp::int32 var_21 = 1;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::int32>* var_27;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    wp::vec_t<3, wp::int32> var_30;
    wp::vec_t<3, wp::float32>* var_31;
    wp::vec_t<3, wp::float32>* var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::int32* var_39;
    wp::vec_t<3, wp::float32>* var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 1e-07;
    bool var_46;
    const wp::float32 var_47 = 0.1;
    const wp::float32 var_48 = 0.75;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32* var_51;
    const wp::float32 var_52 = 0.1;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32>* var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32>* var_61;
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32>* var_67;
    const wp::int32 var_68 = 1;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32>* var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32>* var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::float32 var_82;
    wp::vec_t<3, wp::float32>* var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::int32* var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::int32 var_97;
    wp::vec_t<3, wp::int32>* var_98;
    const wp::int32 var_99 = 0;
    wp::int32 var_100;
    wp::vec_t<3, wp::int32> var_101;
    wp::vec_t<3, wp::float32>* var_102;
    const wp::int32 var_103 = 0;
    wp::float32 var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::int32>* var_109;
    const wp::int32 var_110 = 1;
    wp::int32 var_111;
    wp::vec_t<3, wp::int32> var_112;
    wp::vec_t<3, wp::float32>* var_113;
    const wp::int32 var_114 = 1;
    wp::float32 var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::int32>* var_120;
    const wp::int32 var_121 = 2;
    wp::int32 var_122;
    wp::vec_t<3, wp::int32> var_123;
    wp::vec_t<3, wp::float32>* var_124;
    const wp::int32 var_125 = 2;
    wp::float32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    //---------
    // forward
    // def apply_tri_points_constraints_jacobian(                                             <L 65>
    // tid = wp.tid()                                                                         <L 70>
    var_0 = builtin_tid1d();
    // conn = connectors[tid]                                                                 <L 71>
    var_1 = wp::address(var_connectors, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // tri_pos = (                                                                            <L 73>
    // positions[conn.tri_ids[0]] * conn.tri_bar[0]                                           <L 74>
    var_4 = &(var_2.tri_ids);
    var_7 = wp::load(var_4);
    var_6 = wp::extract(var_7, var_5);
    var_8 = wp::address(var_positions, var_6);
    var_9 = &(var_2.tri_bar);
    var_12 = wp::load(var_9);
    var_11 = wp::extract(var_12, var_10);
    var_14 = wp::load(var_8);
    var_13 = wp::mul(var_14, var_11);
    // + positions[conn.tri_ids[1]] * conn.tri_bar[1]                                         <L 75>
    var_15 = &(var_2.tri_ids);
    var_18 = wp::load(var_15);
    var_17 = wp::extract(var_18, var_16);
    var_19 = wp::address(var_positions, var_17);
    var_20 = &(var_2.tri_bar);
    var_23 = wp::load(var_20);
    var_22 = wp::extract(var_23, var_21);
    var_25 = wp::load(var_19);
    var_24 = wp::mul(var_25, var_22);
    var_26 = wp::add(var_13, var_24);
    // + positions[conn.tri_ids[2]] * conn.tri_bar[2]                                         <L 76>
    var_27 = &(var_2.tri_ids);
    var_30 = wp::load(var_27);
    var_29 = wp::extract(var_30, var_28);
    var_31 = wp::address(var_positions, var_29);
    var_32 = &(var_2.tri_bar);
    var_35 = wp::load(var_32);
    var_34 = wp::extract(var_35, var_33);
    var_37 = wp::load(var_31);
    var_36 = wp::mul(var_37, var_34);
    var_38 = wp::add(var_26, var_36);
    // direction = positions[conn.particle_id] - tri_pos                                      <L 79>
    var_39 = &(var_2.particle_id);
    var_41 = wp::load(var_39);
    var_40 = wp::address(var_positions, var_41);
    var_43 = wp::load(var_40);
    var_42 = wp::sub(var_43, var_38);
    // length = wp.length(direction)                                                          <L 80>
    var_44 = wp::length(var_42);
    // if length < 1e-7:                                                                      <L 81>
    var_46 = (var_44 < var_45);
    if (var_46) {
        // return                                                                             <L 82>
        return;
    }
    // stiffness = 0.1                                                                        <L 84>
    // inv_mass_point = 0.75                                                                  <L 85>
    // inv_mass_tri = 1.0 - inv_mass_point                                                    <L 86>
    var_50 = wp::sub(var_49, var_48);
    // c = length - conn.rest_dist * 0.1                                                      <L 88>
    var_51 = &(var_2.rest_dist);
    var_54 = wp::load(var_51);
    var_53 = wp::mul(var_54, var_52);
    var_55 = wp::sub(var_44, var_53);
    // denom = (                                                                              <L 89>
    // inv_mass_point                                                                         <L 90>
    // + inv_mass_tri * conn.tri_bar[0] * conn.tri_bar[0]                                     <L 91>
    var_56 = &(var_2.tri_bar);
    var_59 = wp::load(var_56);
    var_58 = wp::extract(var_59, var_57);
    var_60 = wp::mul(var_50, var_58);
    var_61 = &(var_2.tri_bar);
    var_64 = wp::load(var_61);
    var_63 = wp::extract(var_64, var_62);
    var_65 = wp::mul(var_60, var_63);
    var_66 = wp::add(var_48, var_65);
    // + inv_mass_tri * conn.tri_bar[1] * conn.tri_bar[1]                                     <L 92>
    var_67 = &(var_2.tri_bar);
    var_70 = wp::load(var_67);
    var_69 = wp::extract(var_70, var_68);
    var_71 = wp::mul(var_50, var_69);
    var_72 = &(var_2.tri_bar);
    var_75 = wp::load(var_72);
    var_74 = wp::extract(var_75, var_73);
    var_76 = wp::mul(var_71, var_74);
    var_77 = wp::add(var_66, var_76);
    // + inv_mass_tri * conn.tri_bar[2] * conn.tri_bar[2]                                     <L 93>
    var_78 = &(var_2.tri_bar);
    var_81 = wp::load(var_78);
    var_80 = wp::extract(var_81, var_79);
    var_82 = wp::mul(var_50, var_80);
    var_83 = &(var_2.tri_bar);
    var_86 = wp::load(var_83);
    var_85 = wp::extract(var_86, var_84);
    var_87 = wp::mul(var_82, var_85);
    var_88 = wp::add(var_77, var_87);
    // delta = (c / denom) * (direction / length) * stiffness                                 <L 95>
    var_89 = wp::div(var_55, var_88);
    var_90 = wp::div(var_42, var_44);
    var_91 = wp::mul(var_89, var_90);
    var_92 = wp::mul(var_91, var_47);
    // wp.atomic_add(delta_accumulator, conn.particle_id, -delta * inv_mass_point)            <L 97>
    var_93 = &(var_2.particle_id);
    var_94 = wp::neg(var_92);
    var_95 = wp::mul(var_94, var_48);
    var_97 = wp::load(var_93);
    var_96 = wp::atomic_add(var_delta_accumulator, var_97, var_95);
    // wp.atomic_add(delta_accumulator, conn.tri_ids[0], delta * conn.tri_bar[0] * inv_mass_tri)       <L 98>
    var_98 = &(var_2.tri_ids);
    var_101 = wp::load(var_98);
    var_100 = wp::extract(var_101, var_99);
    var_102 = &(var_2.tri_bar);
    var_105 = wp::load(var_102);
    var_104 = wp::extract(var_105, var_103);
    var_106 = wp::mul(var_92, var_104);
    var_107 = wp::mul(var_106, var_50);
    var_108 = wp::atomic_add(var_delta_accumulator, var_100, var_107);
    // wp.atomic_add(delta_accumulator, conn.tri_ids[1], delta * conn.tri_bar[1] * inv_mass_tri)       <L 99>
    var_109 = &(var_2.tri_ids);
    var_112 = wp::load(var_109);
    var_111 = wp::extract(var_112, var_110);
    var_113 = &(var_2.tri_bar);
    var_116 = wp::load(var_113);
    var_115 = wp::extract(var_116, var_114);
    var_117 = wp::mul(var_92, var_115);
    var_118 = wp::mul(var_117, var_50);
    var_119 = wp::atomic_add(var_delta_accumulator, var_111, var_118);
    // wp.atomic_add(delta_accumulator, conn.tri_ids[2], delta * conn.tri_bar[2] * inv_mass_tri)       <L 100>
    var_120 = &(var_2.tri_ids);
    var_123 = wp::load(var_120);
    var_122 = wp::extract(var_123, var_121);
    var_124 = &(var_2.tri_bar);
    var_127 = wp::load(var_124);
    var_126 = wp::extract(var_127, var_125);
    var_128 = wp::mul(var_92, var_126);
    var_129 = wp::mul(var_128, var_50);
    var_130 = wp::atomic_add(var_delta_accumulator, var_122, var_129);
}



void apply_tri_points_constraints_jacobian_1ebcce44_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_apply_tri_points_constraints_jacobian_1ebcce44 *_wp_args,
    wp_args_apply_tri_points_constraints_jacobian_1ebcce44 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<TriPointsConnector_7f248e59> var_connectors = _wp_args->connectors;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta_accumulator = _wp_args->delta_accumulator;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions = _wp_adj_args->positions;
    wp::array_t<TriPointsConnector_7f248e59> adj_connectors = _wp_adj_args->connectors;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_delta_accumulator = _wp_adj_args->delta_accumulator;
    //---------
    // primal vars
    wp::int32 var_0;
    TriPointsConnector_7f248e59* var_1;
    TriPointsConnector_7f248e59 var_2;
    TriPointsConnector_7f248e59 var_3;
    wp::vec_t<3, wp::int32>* var_4;
    const wp::int32 var_5 = 0;
    wp::int32 var_6;
    wp::vec_t<3, wp::int32> var_7;
    wp::vec_t<3, wp::float32>* var_8;
    wp::vec_t<3, wp::float32>* var_9;
    const wp::int32 var_10 = 0;
    wp::float32 var_11;
    wp::vec_t<3, wp::float32> var_12;
    wp::vec_t<3, wp::float32> var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::int32>* var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::vec_t<3, wp::int32> var_18;
    wp::vec_t<3, wp::float32>* var_19;
    wp::vec_t<3, wp::float32>* var_20;
    const wp::int32 var_21 = 1;
    wp::float32 var_22;
    wp::vec_t<3, wp::float32> var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    wp::vec_t<3, wp::float32> var_26;
    wp::vec_t<3, wp::int32>* var_27;
    const wp::int32 var_28 = 2;
    wp::int32 var_29;
    wp::vec_t<3, wp::int32> var_30;
    wp::vec_t<3, wp::float32>* var_31;
    wp::vec_t<3, wp::float32>* var_32;
    const wp::int32 var_33 = 2;
    wp::float32 var_34;
    wp::vec_t<3, wp::float32> var_35;
    wp::vec_t<3, wp::float32> var_36;
    wp::vec_t<3, wp::float32> var_37;
    wp::vec_t<3, wp::float32> var_38;
    wp::int32* var_39;
    wp::vec_t<3, wp::float32>* var_40;
    wp::int32 var_41;
    wp::vec_t<3, wp::float32> var_42;
    wp::vec_t<3, wp::float32> var_43;
    wp::float32 var_44;
    const wp::float32 var_45 = 1e-07;
    bool var_46;
    const wp::float32 var_47 = 0.1;
    const wp::float32 var_48 = 0.75;
    const wp::float32 var_49 = 1.0;
    wp::float32 var_50;
    wp::float32* var_51;
    const wp::float32 var_52 = 0.1;
    wp::float32 var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::vec_t<3, wp::float32>* var_56;
    const wp::int32 var_57 = 0;
    wp::float32 var_58;
    wp::vec_t<3, wp::float32> var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32>* var_61;
    const wp::int32 var_62 = 0;
    wp::float32 var_63;
    wp::vec_t<3, wp::float32> var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::vec_t<3, wp::float32>* var_67;
    const wp::int32 var_68 = 1;
    wp::float32 var_69;
    wp::vec_t<3, wp::float32> var_70;
    wp::float32 var_71;
    wp::vec_t<3, wp::float32>* var_72;
    const wp::int32 var_73 = 1;
    wp::float32 var_74;
    wp::vec_t<3, wp::float32> var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::vec_t<3, wp::float32>* var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::vec_t<3, wp::float32> var_81;
    wp::float32 var_82;
    wp::vec_t<3, wp::float32>* var_83;
    const wp::int32 var_84 = 2;
    wp::float32 var_85;
    wp::vec_t<3, wp::float32> var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::vec_t<3, wp::float32> var_90;
    wp::vec_t<3, wp::float32> var_91;
    wp::vec_t<3, wp::float32> var_92;
    wp::int32* var_93;
    wp::vec_t<3, wp::float32> var_94;
    wp::vec_t<3, wp::float32> var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::int32 var_97;
    wp::vec_t<3, wp::int32>* var_98;
    const wp::int32 var_99 = 0;
    wp::int32 var_100;
    wp::vec_t<3, wp::int32> var_101;
    wp::vec_t<3, wp::float32>* var_102;
    const wp::int32 var_103 = 0;
    wp::float32 var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    wp::vec_t<3, wp::float32> var_108;
    wp::vec_t<3, wp::int32>* var_109;
    const wp::int32 var_110 = 1;
    wp::int32 var_111;
    wp::vec_t<3, wp::int32> var_112;
    wp::vec_t<3, wp::float32>* var_113;
    const wp::int32 var_114 = 1;
    wp::float32 var_115;
    wp::vec_t<3, wp::float32> var_116;
    wp::vec_t<3, wp::float32> var_117;
    wp::vec_t<3, wp::float32> var_118;
    wp::vec_t<3, wp::float32> var_119;
    wp::vec_t<3, wp::int32>* var_120;
    const wp::int32 var_121 = 2;
    wp::int32 var_122;
    wp::vec_t<3, wp::int32> var_123;
    wp::vec_t<3, wp::float32>* var_124;
    const wp::int32 var_125 = 2;
    wp::float32 var_126;
    wp::vec_t<3, wp::float32> var_127;
    wp::vec_t<3, wp::float32> var_128;
    wp::vec_t<3, wp::float32> var_129;
    wp::vec_t<3, wp::float32> var_130;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    TriPointsConnector_7f248e59 adj_1 = {};
    TriPointsConnector_7f248e59 adj_2 = {};
    TriPointsConnector_7f248e59 adj_3 = {};
    wp::vec_t<3, wp::int32> adj_4 = {};
    wp::int32 adj_5 = {};
    wp::int32 adj_6 = {};
    wp::vec_t<3, wp::int32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::vec_t<3, wp::float32> adj_9 = {};
    wp::int32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::vec_t<3, wp::float32> adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::int32> adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::vec_t<3, wp::int32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::int32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::vec_t<3, wp::float32> adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    wp::vec_t<3, wp::float32> adj_26 = {};
    wp::vec_t<3, wp::int32> adj_27 = {};
    wp::int32 adj_28 = {};
    wp::int32 adj_29 = {};
    wp::vec_t<3, wp::int32> adj_30 = {};
    wp::vec_t<3, wp::float32> adj_31 = {};
    wp::vec_t<3, wp::float32> adj_32 = {};
    wp::int32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::vec_t<3, wp::float32> adj_35 = {};
    wp::vec_t<3, wp::float32> adj_36 = {};
    wp::vec_t<3, wp::float32> adj_37 = {};
    wp::vec_t<3, wp::float32> adj_38 = {};
    wp::int32 adj_39 = {};
    wp::vec_t<3, wp::float32> adj_40 = {};
    wp::int32 adj_41 = {};
    wp::vec_t<3, wp::float32> adj_42 = {};
    wp::vec_t<3, wp::float32> adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    bool adj_46 = {};
    wp::float32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    wp::float32 adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::vec_t<3, wp::float32> adj_56 = {};
    wp::int32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::vec_t<3, wp::float32> adj_59 = {};
    wp::float32 adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::int32 adj_62 = {};
    wp::float32 adj_63 = {};
    wp::vec_t<3, wp::float32> adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::vec_t<3, wp::float32> adj_67 = {};
    wp::int32 adj_68 = {};
    wp::float32 adj_69 = {};
    wp::vec_t<3, wp::float32> adj_70 = {};
    wp::float32 adj_71 = {};
    wp::vec_t<3, wp::float32> adj_72 = {};
    wp::int32 adj_73 = {};
    wp::float32 adj_74 = {};
    wp::vec_t<3, wp::float32> adj_75 = {};
    wp::float32 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::vec_t<3, wp::float32> adj_78 = {};
    wp::int32 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::vec_t<3, wp::float32> adj_81 = {};
    wp::float32 adj_82 = {};
    wp::vec_t<3, wp::float32> adj_83 = {};
    wp::int32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::vec_t<3, wp::float32> adj_86 = {};
    wp::float32 adj_87 = {};
    wp::float32 adj_88 = {};
    wp::float32 adj_89 = {};
    wp::vec_t<3, wp::float32> adj_90 = {};
    wp::vec_t<3, wp::float32> adj_91 = {};
    wp::vec_t<3, wp::float32> adj_92 = {};
    wp::int32 adj_93 = {};
    wp::vec_t<3, wp::float32> adj_94 = {};
    wp::vec_t<3, wp::float32> adj_95 = {};
    wp::vec_t<3, wp::float32> adj_96 = {};
    wp::int32 adj_97 = {};
    wp::vec_t<3, wp::int32> adj_98 = {};
    wp::int32 adj_99 = {};
    wp::int32 adj_100 = {};
    wp::vec_t<3, wp::int32> adj_101 = {};
    wp::vec_t<3, wp::float32> adj_102 = {};
    wp::int32 adj_103 = {};
    wp::float32 adj_104 = {};
    wp::vec_t<3, wp::float32> adj_105 = {};
    wp::vec_t<3, wp::float32> adj_106 = {};
    wp::vec_t<3, wp::float32> adj_107 = {};
    wp::vec_t<3, wp::float32> adj_108 = {};
    wp::vec_t<3, wp::int32> adj_109 = {};
    wp::int32 adj_110 = {};
    wp::int32 adj_111 = {};
    wp::vec_t<3, wp::int32> adj_112 = {};
    wp::vec_t<3, wp::float32> adj_113 = {};
    wp::int32 adj_114 = {};
    wp::float32 adj_115 = {};
    wp::vec_t<3, wp::float32> adj_116 = {};
    wp::vec_t<3, wp::float32> adj_117 = {};
    wp::vec_t<3, wp::float32> adj_118 = {};
    wp::vec_t<3, wp::float32> adj_119 = {};
    wp::vec_t<3, wp::int32> adj_120 = {};
    wp::int32 adj_121 = {};
    wp::int32 adj_122 = {};
    wp::vec_t<3, wp::int32> adj_123 = {};
    wp::vec_t<3, wp::float32> adj_124 = {};
    wp::int32 adj_125 = {};
    wp::float32 adj_126 = {};
    wp::vec_t<3, wp::float32> adj_127 = {};
    wp::vec_t<3, wp::float32> adj_128 = {};
    wp::vec_t<3, wp::float32> adj_129 = {};
    wp::vec_t<3, wp::float32> adj_130 = {};
    //---------
    // forward
    // def apply_tri_points_constraints_jacobian(                                             <L 65>
    // tid = wp.tid()                                                                         <L 70>
    var_0 = builtin_tid1d();
    // conn = connectors[tid]                                                                 <L 71>
    var_1 = wp::address(var_connectors, var_0);
    var_3 = wp::load(var_1);
    var_2 = wp::copy(var_3);
    // tri_pos = (                                                                            <L 73>
    // positions[conn.tri_ids[0]] * conn.tri_bar[0]                                           <L 74>
    var_4 = &(var_2.tri_ids);
    var_7 = wp::load(var_4);
    var_6 = wp::extract(var_7, var_5);
    var_8 = wp::address(var_positions, var_6);
    var_9 = &(var_2.tri_bar);
    var_12 = wp::load(var_9);
    var_11 = wp::extract(var_12, var_10);
    var_14 = wp::load(var_8);
    var_13 = wp::mul(var_14, var_11);
    // + positions[conn.tri_ids[1]] * conn.tri_bar[1]                                         <L 75>
    var_15 = &(var_2.tri_ids);
    var_18 = wp::load(var_15);
    var_17 = wp::extract(var_18, var_16);
    var_19 = wp::address(var_positions, var_17);
    var_20 = &(var_2.tri_bar);
    var_23 = wp::load(var_20);
    var_22 = wp::extract(var_23, var_21);
    var_25 = wp::load(var_19);
    var_24 = wp::mul(var_25, var_22);
    var_26 = wp::add(var_13, var_24);
    // + positions[conn.tri_ids[2]] * conn.tri_bar[2]                                         <L 76>
    var_27 = &(var_2.tri_ids);
    var_30 = wp::load(var_27);
    var_29 = wp::extract(var_30, var_28);
    var_31 = wp::address(var_positions, var_29);
    var_32 = &(var_2.tri_bar);
    var_35 = wp::load(var_32);
    var_34 = wp::extract(var_35, var_33);
    var_37 = wp::load(var_31);
    var_36 = wp::mul(var_37, var_34);
    var_38 = wp::add(var_26, var_36);
    // direction = positions[conn.particle_id] - tri_pos                                      <L 79>
    var_39 = &(var_2.particle_id);
    var_41 = wp::load(var_39);
    var_40 = wp::address(var_positions, var_41);
    var_43 = wp::load(var_40);
    var_42 = wp::sub(var_43, var_38);
    // length = wp.length(direction)                                                          <L 80>
    var_44 = wp::length(var_42);
    // if length < 1e-7:                                                                      <L 81>
    var_46 = (var_44 < var_45);
    if (var_46) {
        // return                                                                             <L 82>
        goto label0;
    }
    // stiffness = 0.1                                                                        <L 84>
    // inv_mass_point = 0.75                                                                  <L 85>
    // inv_mass_tri = 1.0 - inv_mass_point                                                    <L 86>
    var_50 = wp::sub(var_49, var_48);
    // c = length - conn.rest_dist * 0.1                                                      <L 88>
    var_51 = &(var_2.rest_dist);
    var_54 = wp::load(var_51);
    var_53 = wp::mul(var_54, var_52);
    var_55 = wp::sub(var_44, var_53);
    // denom = (                                                                              <L 89>
    // inv_mass_point                                                                         <L 90>
    // + inv_mass_tri * conn.tri_bar[0] * conn.tri_bar[0]                                     <L 91>
    var_56 = &(var_2.tri_bar);
    var_59 = wp::load(var_56);
    var_58 = wp::extract(var_59, var_57);
    var_60 = wp::mul(var_50, var_58);
    var_61 = &(var_2.tri_bar);
    var_64 = wp::load(var_61);
    var_63 = wp::extract(var_64, var_62);
    var_65 = wp::mul(var_60, var_63);
    var_66 = wp::add(var_48, var_65);
    // + inv_mass_tri * conn.tri_bar[1] * conn.tri_bar[1]                                     <L 92>
    var_67 = &(var_2.tri_bar);
    var_70 = wp::load(var_67);
    var_69 = wp::extract(var_70, var_68);
    var_71 = wp::mul(var_50, var_69);
    var_72 = &(var_2.tri_bar);
    var_75 = wp::load(var_72);
    var_74 = wp::extract(var_75, var_73);
    var_76 = wp::mul(var_71, var_74);
    var_77 = wp::add(var_66, var_76);
    // + inv_mass_tri * conn.tri_bar[2] * conn.tri_bar[2]                                     <L 93>
    var_78 = &(var_2.tri_bar);
    var_81 = wp::load(var_78);
    var_80 = wp::extract(var_81, var_79);
    var_82 = wp::mul(var_50, var_80);
    var_83 = &(var_2.tri_bar);
    var_86 = wp::load(var_83);
    var_85 = wp::extract(var_86, var_84);
    var_87 = wp::mul(var_82, var_85);
    var_88 = wp::add(var_77, var_87);
    // delta = (c / denom) * (direction / length) * stiffness                                 <L 95>
    var_89 = wp::div(var_55, var_88);
    var_90 = wp::div(var_42, var_44);
    var_91 = wp::mul(var_89, var_90);
    var_92 = wp::mul(var_91, var_47);
    // wp.atomic_add(delta_accumulator, conn.particle_id, -delta * inv_mass_point)            <L 97>
    var_93 = &(var_2.particle_id);
    var_94 = wp::neg(var_92);
    var_95 = wp::mul(var_94, var_48);
    var_97 = wp::load(var_93);
    // var_96 = wp::atomic_add(var_delta_accumulator, var_97, var_95);
    // wp.atomic_add(delta_accumulator, conn.tri_ids[0], delta * conn.tri_bar[0] * inv_mass_tri)       <L 98>
    var_98 = &(var_2.tri_ids);
    var_101 = wp::load(var_98);
    var_100 = wp::extract(var_101, var_99);
    var_102 = &(var_2.tri_bar);
    var_105 = wp::load(var_102);
    var_104 = wp::extract(var_105, var_103);
    var_106 = wp::mul(var_92, var_104);
    var_107 = wp::mul(var_106, var_50);
    // var_108 = wp::atomic_add(var_delta_accumulator, var_100, var_107);
    // wp.atomic_add(delta_accumulator, conn.tri_ids[1], delta * conn.tri_bar[1] * inv_mass_tri)       <L 99>
    var_109 = &(var_2.tri_ids);
    var_112 = wp::load(var_109);
    var_111 = wp::extract(var_112, var_110);
    var_113 = &(var_2.tri_bar);
    var_116 = wp::load(var_113);
    var_115 = wp::extract(var_116, var_114);
    var_117 = wp::mul(var_92, var_115);
    var_118 = wp::mul(var_117, var_50);
    // var_119 = wp::atomic_add(var_delta_accumulator, var_111, var_118);
    // wp.atomic_add(delta_accumulator, conn.tri_ids[2], delta * conn.tri_bar[2] * inv_mass_tri)       <L 100>
    var_120 = &(var_2.tri_ids);
    var_123 = wp::load(var_120);
    var_122 = wp::extract(var_123, var_121);
    var_124 = &(var_2.tri_bar);
    var_127 = wp::load(var_124);
    var_126 = wp::extract(var_127, var_125);
    var_128 = wp::mul(var_92, var_126);
    var_129 = wp::mul(var_128, var_50);
    // var_130 = wp::atomic_add(var_delta_accumulator, var_122, var_129);
    //---------
    // reverse
    wp::adj_atomic_add(var_delta_accumulator, var_122, var_129, adj_delta_accumulator, adj_122, adj_129, adj_130);
    wp::adj_mul(var_128, var_50, adj_128, adj_50, adj_129);
    wp::adj_mul(var_92, var_126, adj_92, adj_126, adj_128);
    wp::adj_extract(var_127, var_125, adj_124, adj_125, adj_126);
    adj_2.tri_bar += adj_124;
    wp::adj_extract(var_123, var_121, adj_120, adj_121, adj_122);
    adj_2.tri_ids = adj_120;
    // adj: wp.atomic_add(delta_accumulator, conn.tri_ids[2], delta * conn.tri_bar[2] * inv_mass_tri)  <L 100>
    wp::adj_atomic_add(var_delta_accumulator, var_111, var_118, adj_delta_accumulator, adj_111, adj_118, adj_119);
    wp::adj_mul(var_117, var_50, adj_117, adj_50, adj_118);
    wp::adj_mul(var_92, var_115, adj_92, adj_115, adj_117);
    wp::adj_extract(var_116, var_114, adj_113, adj_114, adj_115);
    adj_2.tri_bar += adj_113;
    wp::adj_extract(var_112, var_110, adj_109, adj_110, adj_111);
    adj_2.tri_ids = adj_109;
    // adj: wp.atomic_add(delta_accumulator, conn.tri_ids[1], delta * conn.tri_bar[1] * inv_mass_tri)  <L 99>
    wp::adj_atomic_add(var_delta_accumulator, var_100, var_107, adj_delta_accumulator, adj_100, adj_107, adj_108);
    wp::adj_mul(var_106, var_50, adj_106, adj_50, adj_107);
    wp::adj_mul(var_92, var_104, adj_92, adj_104, adj_106);
    wp::adj_extract(var_105, var_103, adj_102, adj_103, adj_104);
    adj_2.tri_bar += adj_102;
    wp::adj_extract(var_101, var_99, adj_98, adj_99, adj_100);
    adj_2.tri_ids = adj_98;
    // adj: wp.atomic_add(delta_accumulator, conn.tri_ids[0], delta * conn.tri_bar[0] * inv_mass_tri)  <L 98>
    wp::adj_atomic_add(var_delta_accumulator, var_97, var_95, adj_delta_accumulator, adj_93, adj_95, adj_96);
    wp::adj_mul(var_94, var_48, adj_94, adj_48, adj_95);
    wp::adj_neg(var_92, adj_92, adj_94);
    adj_2.particle_id = adj_93;
    // adj: wp.atomic_add(delta_accumulator, conn.particle_id, -delta * inv_mass_point)       <L 97>
    wp::adj_mul(var_91, var_47, adj_91, adj_47, adj_92);
    wp::adj_mul(var_89, var_90, adj_89, adj_90, adj_91);
    wp::adj_div(var_42, var_44, adj_42, adj_44, adj_90);
    wp::adj_div(var_55, var_88, var_89, adj_55, adj_88, adj_89);
    // adj: delta = (c / denom) * (direction / length) * stiffness                            <L 95>
    wp::adj_add(var_77, var_87, adj_77, adj_87, adj_88);
    wp::adj_mul(var_82, var_85, adj_82, adj_85, adj_87);
    wp::adj_extract(var_86, var_84, adj_83, adj_84, adj_85);
    adj_2.tri_bar += adj_83;
    wp::adj_mul(var_50, var_80, adj_50, adj_80, adj_82);
    wp::adj_extract(var_81, var_79, adj_78, adj_79, adj_80);
    adj_2.tri_bar += adj_78;
    // adj: + inv_mass_tri * conn.tri_bar[2] * conn.tri_bar[2]                                <L 93>
    wp::adj_add(var_66, var_76, adj_66, adj_76, adj_77);
    wp::adj_mul(var_71, var_74, adj_71, adj_74, adj_76);
    wp::adj_extract(var_75, var_73, adj_72, adj_73, adj_74);
    adj_2.tri_bar += adj_72;
    wp::adj_mul(var_50, var_69, adj_50, adj_69, adj_71);
    wp::adj_extract(var_70, var_68, adj_67, adj_68, adj_69);
    adj_2.tri_bar += adj_67;
    // adj: + inv_mass_tri * conn.tri_bar[1] * conn.tri_bar[1]                                <L 92>
    wp::adj_add(var_48, var_65, adj_48, adj_65, adj_66);
    wp::adj_mul(var_60, var_63, adj_60, adj_63, adj_65);
    wp::adj_extract(var_64, var_62, adj_61, adj_62, adj_63);
    adj_2.tri_bar += adj_61;
    wp::adj_mul(var_50, var_58, adj_50, adj_58, adj_60);
    wp::adj_extract(var_59, var_57, adj_56, adj_57, adj_58);
    adj_2.tri_bar += adj_56;
    // adj: + inv_mass_tri * conn.tri_bar[0] * conn.tri_bar[0]                                <L 91>
    // adj: inv_mass_point                                                                    <L 90>
    // adj: denom = (                                                                         <L 89>
    wp::adj_sub(var_44, var_53, adj_44, adj_53, adj_55);
    wp::adj_mul(var_54, var_52, adj_51, adj_52, adj_53);
    adj_2.rest_dist += adj_51;
    // adj: c = length - conn.rest_dist * 0.1                                                 <L 88>
    wp::adj_sub(var_49, var_48, adj_49, adj_48, adj_50);
    // adj: inv_mass_tri = 1.0 - inv_mass_point                                               <L 86>
    // adj: inv_mass_point = 0.75                                                             <L 85>
    // adj: stiffness = 0.1                                                                   <L 84>
    if (var_46) {
        label0:;
        // adj: return                                                                        <L 82>
    }
    // adj: if length < 1e-7:                                                                 <L 81>
    wp::adj_length(var_42, var_44, adj_42, adj_44);
    // adj: length = wp.length(direction)                                                     <L 80>
    wp::adj_sub(var_43, var_38, adj_40, adj_38, adj_42);
    wp::adj_address(var_positions, var_41, adj_positions, adj_39, adj_40);
    adj_2.particle_id = adj_39;
    // adj: direction = positions[conn.particle_id] - tri_pos                                 <L 79>
    wp::adj_add(var_26, var_36, adj_26, adj_36, adj_38);
    wp::adj_mul(var_37, var_34, adj_31, adj_34, adj_36);
    wp::adj_extract(var_35, var_33, adj_32, adj_33, adj_34);
    adj_2.tri_bar += adj_32;
    wp::adj_address(var_positions, var_29, adj_positions, adj_29, adj_31);
    wp::adj_extract(var_30, var_28, adj_27, adj_28, adj_29);
    adj_2.tri_ids = adj_27;
    // adj: + positions[conn.tri_ids[2]] * conn.tri_bar[2]                                    <L 76>
    wp::adj_add(var_13, var_24, adj_13, adj_24, adj_26);
    wp::adj_mul(var_25, var_22, adj_19, adj_22, adj_24);
    wp::adj_extract(var_23, var_21, adj_20, adj_21, adj_22);
    adj_2.tri_bar += adj_20;
    wp::adj_address(var_positions, var_17, adj_positions, adj_17, adj_19);
    wp::adj_extract(var_18, var_16, adj_15, adj_16, adj_17);
    adj_2.tri_ids = adj_15;
    // adj: + positions[conn.tri_ids[1]] * conn.tri_bar[1]                                    <L 75>
    wp::adj_mul(var_14, var_11, adj_8, adj_11, adj_13);
    wp::adj_extract(var_12, var_10, adj_9, adj_10, adj_11);
    adj_2.tri_bar += adj_9;
    wp::adj_address(var_positions, var_6, adj_positions, adj_6, adj_8);
    wp::adj_extract(var_7, var_5, adj_4, adj_5, adj_6);
    adj_2.tri_ids = adj_4;
    // adj: positions[conn.tri_ids[0]] * conn.tri_bar[0]                                      <L 74>
    // adj: tri_pos = (                                                                       <L 73>
    wp::adj_copy(var_3, adj_1, adj_2);
    wp::adj_address(var_connectors, var_0, adj_connectors, adj_0, adj_1);
    // adj: conn = connectors[tid]                                                            <L 71>
    // adj: tid = wp.tid()                                                                    <L 70>
    // adj: def apply_tri_points_constraints_jacobian(                                        <L 65>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void apply_tri_points_constraints_jacobian_1ebcce44_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_apply_tri_points_constraints_jacobian_1ebcce44 *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_tri_points_constraints_jacobian_1ebcce44_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void apply_tri_points_constraints_jacobian_1ebcce44_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_apply_tri_points_constraints_jacobian_1ebcce44 *_wp_args,
    wp_args_apply_tri_points_constraints_jacobian_1ebcce44 *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        apply_tri_points_constraints_jacobian_1ebcce44_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_solve_volume_constraints_e00c95df {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<wp::float32> invmass;
    wp::array_t<Tetrahedron_4d48766a> tetrahedra;
    wp::array_t<wp::int32> tetrahedra_active;
    wp::float32 stiffness;
    wp::array_t<wp::vec_t<3, wp::float32>> delta_accumulator;
    wp::array_t<wp::int32> delta_counter;
};


void solve_volume_constraints_e00c95df_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_solve_volume_constraints_e00c95df *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::float32> var_invmass = _wp_args->invmass;
    wp::array_t<Tetrahedron_4d48766a> var_tetrahedra = _wp_args->tetrahedra;
    wp::array_t<wp::int32> var_tetrahedra_active = _wp_args->tetrahedra_active;
    wp::float32 var_stiffness = _wp_args->stiffness;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta_accumulator = _wp_args->delta_accumulator;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 0;
    bool var_3;
    wp::int32 var_4;
    Tetrahedron_4d48766a* var_5;
    Tetrahedron_4d48766a var_6;
    Tetrahedron_4d48766a var_7;
    wp::vec_t<4, wp::int32>* var_8;
    wp::vec_t<4, wp::int32> var_9;
    wp::vec_t<4, wp::int32> var_10;
    const wp::int32 var_11 = 0;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 2;
    wp::int32 var_22;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    const wp::int32 var_26 = 3;
    wp::int32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::int32 var_32;
    wp::float32* var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    wp::float32* var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::int32 var_42;
    wp::float32* var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 3;
    wp::int32 var_47;
    wp::float32* var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 6.0;
    wp::float32 var_57;
    wp::float32* var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::float32 var_64 = 6.0;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = 6.0;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::float32 var_74 = 6.0;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    const wp::float32 var_79 = 6.0;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::float32 var_92 = 1e-08;
    bool var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    const wp::int32 var_108 = 0;
    wp::int32 var_109;
    wp::vec_t<3, wp::float32> var_110;
    const wp::int32 var_111 = 1;
    wp::int32 var_112;
    wp::vec_t<3, wp::float32> var_113;
    const wp::int32 var_114 = 2;
    wp::int32 var_115;
    wp::vec_t<3, wp::float32> var_116;
    const wp::int32 var_117 = 3;
    wp::int32 var_118;
    wp::vec_t<3, wp::float32> var_119;
    const wp::int32 var_120 = 0;
    wp::int32 var_121;
    const wp::int32 var_122 = 1;
    wp::int32 var_123;
    const wp::int32 var_124 = 1;
    wp::int32 var_125;
    const wp::int32 var_126 = 1;
    wp::int32 var_127;
    const wp::int32 var_128 = 2;
    wp::int32 var_129;
    const wp::int32 var_130 = 1;
    wp::int32 var_131;
    const wp::int32 var_132 = 3;
    wp::int32 var_133;
    const wp::int32 var_134 = 1;
    wp::int32 var_135;
    //---------
    // forward
    // def solve_volume_constraints(                                                          <L 185>
    // tid = wp.tid()                                                                         <L 194>
    var_0 = builtin_tid1d();
    // if tetrahedra_active[tid] == 0:                                                        <L 195>
    var_1 = wp::address(var_tetrahedra_active, var_0);
    var_4 = wp::load(var_1);
    var_3 = (var_4 == var_2);
    if (var_3) {
        // return                                                                             <L 196>
        return;
    }
    // tet = tetrahedra[tid]                                                                  <L 198>
    var_5 = wp::address(var_tetrahedra, var_0);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // ids = tet.ids                                                                          <L 199>
    var_8 = &(var_6.ids);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // p0 = positions[ids[0]]                                                                 <L 201>
    var_12 = wp::extract(var_9, var_11);
    var_13 = wp::address(var_positions, var_12);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // p1 = positions[ids[1]]                                                                 <L 202>
    var_17 = wp::extract(var_9, var_16);
    var_18 = wp::address(var_positions, var_17);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // p2 = positions[ids[2]]                                                                 <L 203>
    var_22 = wp::extract(var_9, var_21);
    var_23 = wp::address(var_positions, var_22);
    var_25 = wp::load(var_23);
    var_24 = wp::copy(var_25);
    // p3 = positions[ids[3]]                                                                 <L 204>
    var_27 = wp::extract(var_9, var_26);
    var_28 = wp::address(var_positions, var_27);
    var_30 = wp::load(var_28);
    var_29 = wp::copy(var_30);
    // w0 = invmass[ids[0]]                                                                   <L 206>
    var_32 = wp::extract(var_9, var_31);
    var_33 = wp::address(var_invmass, var_32);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // w1 = invmass[ids[1]]                                                                   <L 207>
    var_37 = wp::extract(var_9, var_36);
    var_38 = wp::address(var_invmass, var_37);
    var_40 = wp::load(var_38);
    var_39 = wp::copy(var_40);
    // w2 = invmass[ids[2]]                                                                   <L 208>
    var_42 = wp::extract(var_9, var_41);
    var_43 = wp::address(var_invmass, var_42);
    var_45 = wp::load(var_43);
    var_44 = wp::copy(var_45);
    // w3 = invmass[ids[3]]                                                                   <L 209>
    var_47 = wp::extract(var_9, var_46);
    var_48 = wp::address(var_invmass, var_47);
    var_50 = wp::load(var_48);
    var_49 = wp::copy(var_50);
    // v = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0                                  <L 211>
    var_51 = wp::sub(var_19, var_14);
    var_52 = wp::sub(var_24, var_14);
    var_53 = wp::cross(var_51, var_52);
    var_54 = wp::sub(var_29, var_14);
    var_55 = wp::dot(var_53, var_54);
    var_57 = wp::div(var_55, var_56);
    // c = v - tet.rest_volume                                                                <L 212>
    var_58 = &(var_6.rest_volume);
    var_60 = wp::load(var_58);
    var_59 = wp::sub(var_57, var_60);
    // grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0                                               <L 214>
    var_61 = wp::sub(var_19, var_24);
    var_62 = wp::sub(var_29, var_24);
    var_63 = wp::cross(var_61, var_62);
    var_65 = wp::div(var_63, var_64);
    // grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0                                               <L 215>
    var_66 = wp::sub(var_24, var_14);
    var_67 = wp::sub(var_29, var_14);
    var_68 = wp::cross(var_66, var_67);
    var_70 = wp::div(var_68, var_69);
    // grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0                                               <L 216>
    var_71 = wp::sub(var_14, var_19);
    var_72 = wp::sub(var_29, var_19);
    var_73 = wp::cross(var_71, var_72);
    var_75 = wp::div(var_73, var_74);
    // grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0                                               <L 217>
    var_76 = wp::sub(var_19, var_14);
    var_77 = wp::sub(var_24, var_14);
    var_78 = wp::cross(var_76, var_77);
    var_80 = wp::div(var_78, var_79);
    // sum_grad = (                                                                           <L 219>
    // w0 * wp.length_sq(grad0)                                                               <L 220>
    var_81 = wp::length_sq(var_65);
    var_82 = wp::mul(var_34, var_81);
    // + w1 * wp.length_sq(grad1)                                                             <L 221>
    var_83 = wp::length_sq(var_70);
    var_84 = wp::mul(var_39, var_83);
    var_85 = wp::add(var_82, var_84);
    // + w2 * wp.length_sq(grad2)                                                             <L 222>
    var_86 = wp::length_sq(var_75);
    var_87 = wp::mul(var_44, var_86);
    var_88 = wp::add(var_85, var_87);
    // + w3 * wp.length_sq(grad3)                                                             <L 223>
    var_89 = wp::length_sq(var_80);
    var_90 = wp::mul(var_49, var_89);
    var_91 = wp::add(var_88, var_90);
    // if sum_grad < 1e-8:                                                                    <L 225>
    var_93 = (var_91 < var_92);
    if (var_93) {
        // return                                                                             <L 226>
        return;
    }
    // scale = stiffness * c / sum_grad                                                       <L 228>
    var_94 = wp::mul(var_stiffness, var_59);
    var_95 = wp::div(var_94, var_91);
    // d0 = -grad0 * scale * w0                                                               <L 230>
    var_96 = wp::neg(var_65);
    var_97 = wp::mul(var_96, var_95);
    var_98 = wp::mul(var_97, var_34);
    // d1 = -grad1 * scale * w1                                                               <L 231>
    var_99 = wp::neg(var_70);
    var_100 = wp::mul(var_99, var_95);
    var_101 = wp::mul(var_100, var_39);
    // d2 = -grad2 * scale * w2                                                               <L 232>
    var_102 = wp::neg(var_75);
    var_103 = wp::mul(var_102, var_95);
    var_104 = wp::mul(var_103, var_44);
    // d3 = -grad3 * scale * w3                                                               <L 233>
    var_105 = wp::neg(var_80);
    var_106 = wp::mul(var_105, var_95);
    var_107 = wp::mul(var_106, var_49);
    // wp.atomic_add(delta_accumulator, ids[0], d0)                                           <L 235>
    var_109 = wp::extract(var_9, var_108);
    var_110 = wp::atomic_add(var_delta_accumulator, var_109, var_98);
    // wp.atomic_add(delta_accumulator, ids[1], d1)                                           <L 236>
    var_112 = wp::extract(var_9, var_111);
    var_113 = wp::atomic_add(var_delta_accumulator, var_112, var_101);
    // wp.atomic_add(delta_accumulator, ids[2], d2)                                           <L 237>
    var_115 = wp::extract(var_9, var_114);
    var_116 = wp::atomic_add(var_delta_accumulator, var_115, var_104);
    // wp.atomic_add(delta_accumulator, ids[3], d3)                                           <L 238>
    var_118 = wp::extract(var_9, var_117);
    var_119 = wp::atomic_add(var_delta_accumulator, var_118, var_107);
    // wp.atomic_add(delta_counter, ids[0], 1)                                                <L 240>
    var_121 = wp::extract(var_9, var_120);
    var_123 = wp::atomic_add(var_delta_counter, var_121, var_122);
    // wp.atomic_add(delta_counter, ids[1], 1)                                                <L 241>
    var_125 = wp::extract(var_9, var_124);
    var_127 = wp::atomic_add(var_delta_counter, var_125, var_126);
    // wp.atomic_add(delta_counter, ids[2], 1)                                                <L 242>
    var_129 = wp::extract(var_9, var_128);
    var_131 = wp::atomic_add(var_delta_counter, var_129, var_130);
    // wp.atomic_add(delta_counter, ids[3], 1)                                                <L 243>
    var_133 = wp::extract(var_9, var_132);
    var_135 = wp::atomic_add(var_delta_counter, var_133, var_134);
}



void solve_volume_constraints_e00c95df_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_solve_volume_constraints_e00c95df *_wp_args,
    wp_args_solve_volume_constraints_e00c95df *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::float32> var_invmass = _wp_args->invmass;
    wp::array_t<Tetrahedron_4d48766a> var_tetrahedra = _wp_args->tetrahedra;
    wp::array_t<wp::int32> var_tetrahedra_active = _wp_args->tetrahedra_active;
    wp::float32 var_stiffness = _wp_args->stiffness;
    wp::array_t<wp::vec_t<3, wp::float32>> var_delta_accumulator = _wp_args->delta_accumulator;
    wp::array_t<wp::int32> var_delta_counter = _wp_args->delta_counter;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions = _wp_adj_args->positions;
    wp::array_t<wp::float32> adj_invmass = _wp_adj_args->invmass;
    wp::array_t<Tetrahedron_4d48766a> adj_tetrahedra = _wp_adj_args->tetrahedra;
    wp::array_t<wp::int32> adj_tetrahedra_active = _wp_adj_args->tetrahedra_active;
    wp::float32 adj_stiffness = _wp_adj_args->stiffness;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_delta_accumulator = _wp_adj_args->delta_accumulator;
    wp::array_t<wp::int32> adj_delta_counter = _wp_adj_args->delta_counter;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32* var_1;
    const wp::int32 var_2 = 0;
    bool var_3;
    wp::int32 var_4;
    Tetrahedron_4d48766a* var_5;
    Tetrahedron_4d48766a var_6;
    Tetrahedron_4d48766a var_7;
    wp::vec_t<4, wp::int32>* var_8;
    wp::vec_t<4, wp::int32> var_9;
    wp::vec_t<4, wp::int32> var_10;
    const wp::int32 var_11 = 0;
    wp::int32 var_12;
    wp::vec_t<3, wp::float32>* var_13;
    wp::vec_t<3, wp::float32> var_14;
    wp::vec_t<3, wp::float32> var_15;
    const wp::int32 var_16 = 1;
    wp::int32 var_17;
    wp::vec_t<3, wp::float32>* var_18;
    wp::vec_t<3, wp::float32> var_19;
    wp::vec_t<3, wp::float32> var_20;
    const wp::int32 var_21 = 2;
    wp::int32 var_22;
    wp::vec_t<3, wp::float32>* var_23;
    wp::vec_t<3, wp::float32> var_24;
    wp::vec_t<3, wp::float32> var_25;
    const wp::int32 var_26 = 3;
    wp::int32 var_27;
    wp::vec_t<3, wp::float32>* var_28;
    wp::vec_t<3, wp::float32> var_29;
    wp::vec_t<3, wp::float32> var_30;
    const wp::int32 var_31 = 0;
    wp::int32 var_32;
    wp::float32* var_33;
    wp::float32 var_34;
    wp::float32 var_35;
    const wp::int32 var_36 = 1;
    wp::int32 var_37;
    wp::float32* var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    const wp::int32 var_41 = 2;
    wp::int32 var_42;
    wp::float32* var_43;
    wp::float32 var_44;
    wp::float32 var_45;
    const wp::int32 var_46 = 3;
    wp::int32 var_47;
    wp::float32* var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::vec_t<3, wp::float32> var_51;
    wp::vec_t<3, wp::float32> var_52;
    wp::vec_t<3, wp::float32> var_53;
    wp::vec_t<3, wp::float32> var_54;
    wp::float32 var_55;
    const wp::float32 var_56 = 6.0;
    wp::float32 var_57;
    wp::float32* var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::vec_t<3, wp::float32> var_61;
    wp::vec_t<3, wp::float32> var_62;
    wp::vec_t<3, wp::float32> var_63;
    const wp::float32 var_64 = 6.0;
    wp::vec_t<3, wp::float32> var_65;
    wp::vec_t<3, wp::float32> var_66;
    wp::vec_t<3, wp::float32> var_67;
    wp::vec_t<3, wp::float32> var_68;
    const wp::float32 var_69 = 6.0;
    wp::vec_t<3, wp::float32> var_70;
    wp::vec_t<3, wp::float32> var_71;
    wp::vec_t<3, wp::float32> var_72;
    wp::vec_t<3, wp::float32> var_73;
    const wp::float32 var_74 = 6.0;
    wp::vec_t<3, wp::float32> var_75;
    wp::vec_t<3, wp::float32> var_76;
    wp::vec_t<3, wp::float32> var_77;
    wp::vec_t<3, wp::float32> var_78;
    const wp::float32 var_79 = 6.0;
    wp::vec_t<3, wp::float32> var_80;
    wp::float32 var_81;
    wp::float32 var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    wp::float32 var_86;
    wp::float32 var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    const wp::float32 var_92 = 1e-08;
    bool var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::vec_t<3, wp::float32> var_96;
    wp::vec_t<3, wp::float32> var_97;
    wp::vec_t<3, wp::float32> var_98;
    wp::vec_t<3, wp::float32> var_99;
    wp::vec_t<3, wp::float32> var_100;
    wp::vec_t<3, wp::float32> var_101;
    wp::vec_t<3, wp::float32> var_102;
    wp::vec_t<3, wp::float32> var_103;
    wp::vec_t<3, wp::float32> var_104;
    wp::vec_t<3, wp::float32> var_105;
    wp::vec_t<3, wp::float32> var_106;
    wp::vec_t<3, wp::float32> var_107;
    const wp::int32 var_108 = 0;
    wp::int32 var_109;
    wp::vec_t<3, wp::float32> var_110;
    const wp::int32 var_111 = 1;
    wp::int32 var_112;
    wp::vec_t<3, wp::float32> var_113;
    const wp::int32 var_114 = 2;
    wp::int32 var_115;
    wp::vec_t<3, wp::float32> var_116;
    const wp::int32 var_117 = 3;
    wp::int32 var_118;
    wp::vec_t<3, wp::float32> var_119;
    const wp::int32 var_120 = 0;
    wp::int32 var_121;
    const wp::int32 var_122 = 1;
    wp::int32 var_123;
    const wp::int32 var_124 = 1;
    wp::int32 var_125;
    const wp::int32 var_126 = 1;
    wp::int32 var_127;
    const wp::int32 var_128 = 2;
    wp::int32 var_129;
    const wp::int32 var_130 = 1;
    wp::int32 var_131;
    const wp::int32 var_132 = 3;
    wp::int32 var_133;
    const wp::int32 var_134 = 1;
    wp::int32 var_135;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::int32 adj_2 = {};
    bool adj_3 = {};
    wp::int32 adj_4 = {};
    Tetrahedron_4d48766a adj_5 = {};
    Tetrahedron_4d48766a adj_6 = {};
    Tetrahedron_4d48766a adj_7 = {};
    wp::vec_t<4, wp::int32> adj_8 = {};
    wp::vec_t<4, wp::int32> adj_9 = {};
    wp::vec_t<4, wp::int32> adj_10 = {};
    wp::int32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::vec_t<3, wp::float32> adj_13 = {};
    wp::vec_t<3, wp::float32> adj_14 = {};
    wp::vec_t<3, wp::float32> adj_15 = {};
    wp::int32 adj_16 = {};
    wp::int32 adj_17 = {};
    wp::vec_t<3, wp::float32> adj_18 = {};
    wp::vec_t<3, wp::float32> adj_19 = {};
    wp::vec_t<3, wp::float32> adj_20 = {};
    wp::int32 adj_21 = {};
    wp::int32 adj_22 = {};
    wp::vec_t<3, wp::float32> adj_23 = {};
    wp::vec_t<3, wp::float32> adj_24 = {};
    wp::vec_t<3, wp::float32> adj_25 = {};
    wp::int32 adj_26 = {};
    wp::int32 adj_27 = {};
    wp::vec_t<3, wp::float32> adj_28 = {};
    wp::vec_t<3, wp::float32> adj_29 = {};
    wp::vec_t<3, wp::float32> adj_30 = {};
    wp::int32 adj_31 = {};
    wp::int32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    wp::float32 adj_35 = {};
    wp::int32 adj_36 = {};
    wp::int32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::int32 adj_41 = {};
    wp::int32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::float32 adj_45 = {};
    wp::int32 adj_46 = {};
    wp::int32 adj_47 = {};
    wp::float32 adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::vec_t<3, wp::float32> adj_51 = {};
    wp::vec_t<3, wp::float32> adj_52 = {};
    wp::vec_t<3, wp::float32> adj_53 = {};
    wp::vec_t<3, wp::float32> adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::vec_t<3, wp::float32> adj_61 = {};
    wp::vec_t<3, wp::float32> adj_62 = {};
    wp::vec_t<3, wp::float32> adj_63 = {};
    wp::float32 adj_64 = {};
    wp::vec_t<3, wp::float32> adj_65 = {};
    wp::vec_t<3, wp::float32> adj_66 = {};
    wp::vec_t<3, wp::float32> adj_67 = {};
    wp::vec_t<3, wp::float32> adj_68 = {};
    wp::float32 adj_69 = {};
    wp::vec_t<3, wp::float32> adj_70 = {};
    wp::vec_t<3, wp::float32> adj_71 = {};
    wp::vec_t<3, wp::float32> adj_72 = {};
    wp::vec_t<3, wp::float32> adj_73 = {};
    wp::float32 adj_74 = {};
    wp::vec_t<3, wp::float32> adj_75 = {};
    wp::vec_t<3, wp::float32> adj_76 = {};
    wp::vec_t<3, wp::float32> adj_77 = {};
    wp::vec_t<3, wp::float32> adj_78 = {};
    wp::float32 adj_79 = {};
    wp::vec_t<3, wp::float32> adj_80 = {};
    wp::float32 adj_81 = {};
    wp::float32 adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float32 adj_86 = {};
    wp::float32 adj_87 = {};
    wp::float32 adj_88 = {};
    wp::float32 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float32 adj_91 = {};
    wp::float32 adj_92 = {};
    bool adj_93 = {};
    wp::float32 adj_94 = {};
    wp::float32 adj_95 = {};
    wp::vec_t<3, wp::float32> adj_96 = {};
    wp::vec_t<3, wp::float32> adj_97 = {};
    wp::vec_t<3, wp::float32> adj_98 = {};
    wp::vec_t<3, wp::float32> adj_99 = {};
    wp::vec_t<3, wp::float32> adj_100 = {};
    wp::vec_t<3, wp::float32> adj_101 = {};
    wp::vec_t<3, wp::float32> adj_102 = {};
    wp::vec_t<3, wp::float32> adj_103 = {};
    wp::vec_t<3, wp::float32> adj_104 = {};
    wp::vec_t<3, wp::float32> adj_105 = {};
    wp::vec_t<3, wp::float32> adj_106 = {};
    wp::vec_t<3, wp::float32> adj_107 = {};
    wp::int32 adj_108 = {};
    wp::int32 adj_109 = {};
    wp::vec_t<3, wp::float32> adj_110 = {};
    wp::int32 adj_111 = {};
    wp::int32 adj_112 = {};
    wp::vec_t<3, wp::float32> adj_113 = {};
    wp::int32 adj_114 = {};
    wp::int32 adj_115 = {};
    wp::vec_t<3, wp::float32> adj_116 = {};
    wp::int32 adj_117 = {};
    wp::int32 adj_118 = {};
    wp::vec_t<3, wp::float32> adj_119 = {};
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
    wp::int32 adj_133 = {};
    wp::int32 adj_134 = {};
    wp::int32 adj_135 = {};
    //---------
    // forward
    // def solve_volume_constraints(                                                          <L 185>
    // tid = wp.tid()                                                                         <L 194>
    var_0 = builtin_tid1d();
    // if tetrahedra_active[tid] == 0:                                                        <L 195>
    var_1 = wp::address(var_tetrahedra_active, var_0);
    var_4 = wp::load(var_1);
    var_3 = (var_4 == var_2);
    if (var_3) {
        // return                                                                             <L 196>
        goto label0;
    }
    // tet = tetrahedra[tid]                                                                  <L 198>
    var_5 = wp::address(var_tetrahedra, var_0);
    var_7 = wp::load(var_5);
    var_6 = wp::copy(var_7);
    // ids = tet.ids                                                                          <L 199>
    var_8 = &(var_6.ids);
    var_10 = wp::load(var_8);
    var_9 = wp::copy(var_10);
    // p0 = positions[ids[0]]                                                                 <L 201>
    var_12 = wp::extract(var_9, var_11);
    var_13 = wp::address(var_positions, var_12);
    var_15 = wp::load(var_13);
    var_14 = wp::copy(var_15);
    // p1 = positions[ids[1]]                                                                 <L 202>
    var_17 = wp::extract(var_9, var_16);
    var_18 = wp::address(var_positions, var_17);
    var_20 = wp::load(var_18);
    var_19 = wp::copy(var_20);
    // p2 = positions[ids[2]]                                                                 <L 203>
    var_22 = wp::extract(var_9, var_21);
    var_23 = wp::address(var_positions, var_22);
    var_25 = wp::load(var_23);
    var_24 = wp::copy(var_25);
    // p3 = positions[ids[3]]                                                                 <L 204>
    var_27 = wp::extract(var_9, var_26);
    var_28 = wp::address(var_positions, var_27);
    var_30 = wp::load(var_28);
    var_29 = wp::copy(var_30);
    // w0 = invmass[ids[0]]                                                                   <L 206>
    var_32 = wp::extract(var_9, var_31);
    var_33 = wp::address(var_invmass, var_32);
    var_35 = wp::load(var_33);
    var_34 = wp::copy(var_35);
    // w1 = invmass[ids[1]]                                                                   <L 207>
    var_37 = wp::extract(var_9, var_36);
    var_38 = wp::address(var_invmass, var_37);
    var_40 = wp::load(var_38);
    var_39 = wp::copy(var_40);
    // w2 = invmass[ids[2]]                                                                   <L 208>
    var_42 = wp::extract(var_9, var_41);
    var_43 = wp::address(var_invmass, var_42);
    var_45 = wp::load(var_43);
    var_44 = wp::copy(var_45);
    // w3 = invmass[ids[3]]                                                                   <L 209>
    var_47 = wp::extract(var_9, var_46);
    var_48 = wp::address(var_invmass, var_47);
    var_50 = wp::load(var_48);
    var_49 = wp::copy(var_50);
    // v = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0                                  <L 211>
    var_51 = wp::sub(var_19, var_14);
    var_52 = wp::sub(var_24, var_14);
    var_53 = wp::cross(var_51, var_52);
    var_54 = wp::sub(var_29, var_14);
    var_55 = wp::dot(var_53, var_54);
    var_57 = wp::div(var_55, var_56);
    // c = v - tet.rest_volume                                                                <L 212>
    var_58 = &(var_6.rest_volume);
    var_60 = wp::load(var_58);
    var_59 = wp::sub(var_57, var_60);
    // grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0                                               <L 214>
    var_61 = wp::sub(var_19, var_24);
    var_62 = wp::sub(var_29, var_24);
    var_63 = wp::cross(var_61, var_62);
    var_65 = wp::div(var_63, var_64);
    // grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0                                               <L 215>
    var_66 = wp::sub(var_24, var_14);
    var_67 = wp::sub(var_29, var_14);
    var_68 = wp::cross(var_66, var_67);
    var_70 = wp::div(var_68, var_69);
    // grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0                                               <L 216>
    var_71 = wp::sub(var_14, var_19);
    var_72 = wp::sub(var_29, var_19);
    var_73 = wp::cross(var_71, var_72);
    var_75 = wp::div(var_73, var_74);
    // grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0                                               <L 217>
    var_76 = wp::sub(var_19, var_14);
    var_77 = wp::sub(var_24, var_14);
    var_78 = wp::cross(var_76, var_77);
    var_80 = wp::div(var_78, var_79);
    // sum_grad = (                                                                           <L 219>
    // w0 * wp.length_sq(grad0)                                                               <L 220>
    var_81 = wp::length_sq(var_65);
    var_82 = wp::mul(var_34, var_81);
    // + w1 * wp.length_sq(grad1)                                                             <L 221>
    var_83 = wp::length_sq(var_70);
    var_84 = wp::mul(var_39, var_83);
    var_85 = wp::add(var_82, var_84);
    // + w2 * wp.length_sq(grad2)                                                             <L 222>
    var_86 = wp::length_sq(var_75);
    var_87 = wp::mul(var_44, var_86);
    var_88 = wp::add(var_85, var_87);
    // + w3 * wp.length_sq(grad3)                                                             <L 223>
    var_89 = wp::length_sq(var_80);
    var_90 = wp::mul(var_49, var_89);
    var_91 = wp::add(var_88, var_90);
    // if sum_grad < 1e-8:                                                                    <L 225>
    var_93 = (var_91 < var_92);
    if (var_93) {
        // return                                                                             <L 226>
        goto label1;
    }
    // scale = stiffness * c / sum_grad                                                       <L 228>
    var_94 = wp::mul(var_stiffness, var_59);
    var_95 = wp::div(var_94, var_91);
    // d0 = -grad0 * scale * w0                                                               <L 230>
    var_96 = wp::neg(var_65);
    var_97 = wp::mul(var_96, var_95);
    var_98 = wp::mul(var_97, var_34);
    // d1 = -grad1 * scale * w1                                                               <L 231>
    var_99 = wp::neg(var_70);
    var_100 = wp::mul(var_99, var_95);
    var_101 = wp::mul(var_100, var_39);
    // d2 = -grad2 * scale * w2                                                               <L 232>
    var_102 = wp::neg(var_75);
    var_103 = wp::mul(var_102, var_95);
    var_104 = wp::mul(var_103, var_44);
    // d3 = -grad3 * scale * w3                                                               <L 233>
    var_105 = wp::neg(var_80);
    var_106 = wp::mul(var_105, var_95);
    var_107 = wp::mul(var_106, var_49);
    // wp.atomic_add(delta_accumulator, ids[0], d0)                                           <L 235>
    var_109 = wp::extract(var_9, var_108);
    // var_110 = wp::atomic_add(var_delta_accumulator, var_109, var_98);
    // wp.atomic_add(delta_accumulator, ids[1], d1)                                           <L 236>
    var_112 = wp::extract(var_9, var_111);
    // var_113 = wp::atomic_add(var_delta_accumulator, var_112, var_101);
    // wp.atomic_add(delta_accumulator, ids[2], d2)                                           <L 237>
    var_115 = wp::extract(var_9, var_114);
    // var_116 = wp::atomic_add(var_delta_accumulator, var_115, var_104);
    // wp.atomic_add(delta_accumulator, ids[3], d3)                                           <L 238>
    var_118 = wp::extract(var_9, var_117);
    // var_119 = wp::atomic_add(var_delta_accumulator, var_118, var_107);
    // wp.atomic_add(delta_counter, ids[0], 1)                                                <L 240>
    var_121 = wp::extract(var_9, var_120);
    // var_123 = wp::atomic_add(var_delta_counter, var_121, var_122);
    // wp.atomic_add(delta_counter, ids[1], 1)                                                <L 241>
    var_125 = wp::extract(var_9, var_124);
    // var_127 = wp::atomic_add(var_delta_counter, var_125, var_126);
    // wp.atomic_add(delta_counter, ids[2], 1)                                                <L 242>
    var_129 = wp::extract(var_9, var_128);
    // var_131 = wp::atomic_add(var_delta_counter, var_129, var_130);
    // wp.atomic_add(delta_counter, ids[3], 1)                                                <L 243>
    var_133 = wp::extract(var_9, var_132);
    // var_135 = wp::atomic_add(var_delta_counter, var_133, var_134);
    //---------
    // reverse
    wp::adj_atomic_add(var_delta_counter, var_133, var_134, adj_delta_counter, adj_133, adj_134, adj_135);
    wp::adj_extract(var_9, var_132, adj_9, adj_132, adj_133);
    // adj: wp.atomic_add(delta_counter, ids[3], 1)                                           <L 243>
    wp::adj_atomic_add(var_delta_counter, var_129, var_130, adj_delta_counter, adj_129, adj_130, adj_131);
    wp::adj_extract(var_9, var_128, adj_9, adj_128, adj_129);
    // adj: wp.atomic_add(delta_counter, ids[2], 1)                                           <L 242>
    wp::adj_atomic_add(var_delta_counter, var_125, var_126, adj_delta_counter, adj_125, adj_126, adj_127);
    wp::adj_extract(var_9, var_124, adj_9, adj_124, adj_125);
    // adj: wp.atomic_add(delta_counter, ids[1], 1)                                           <L 241>
    wp::adj_atomic_add(var_delta_counter, var_121, var_122, adj_delta_counter, adj_121, adj_122, adj_123);
    wp::adj_extract(var_9, var_120, adj_9, adj_120, adj_121);
    // adj: wp.atomic_add(delta_counter, ids[0], 1)                                           <L 240>
    wp::adj_atomic_add(var_delta_accumulator, var_118, var_107, adj_delta_accumulator, adj_118, adj_107, adj_119);
    wp::adj_extract(var_9, var_117, adj_9, adj_117, adj_118);
    // adj: wp.atomic_add(delta_accumulator, ids[3], d3)                                      <L 238>
    wp::adj_atomic_add(var_delta_accumulator, var_115, var_104, adj_delta_accumulator, adj_115, adj_104, adj_116);
    wp::adj_extract(var_9, var_114, adj_9, adj_114, adj_115);
    // adj: wp.atomic_add(delta_accumulator, ids[2], d2)                                      <L 237>
    wp::adj_atomic_add(var_delta_accumulator, var_112, var_101, adj_delta_accumulator, adj_112, adj_101, adj_113);
    wp::adj_extract(var_9, var_111, adj_9, adj_111, adj_112);
    // adj: wp.atomic_add(delta_accumulator, ids[1], d1)                                      <L 236>
    wp::adj_atomic_add(var_delta_accumulator, var_109, var_98, adj_delta_accumulator, adj_109, adj_98, adj_110);
    wp::adj_extract(var_9, var_108, adj_9, adj_108, adj_109);
    // adj: wp.atomic_add(delta_accumulator, ids[0], d0)                                      <L 235>
    wp::adj_mul(var_106, var_49, adj_106, adj_49, adj_107);
    wp::adj_mul(var_105, var_95, adj_105, adj_95, adj_106);
    wp::adj_neg(var_80, adj_80, adj_105);
    // adj: d3 = -grad3 * scale * w3                                                          <L 233>
    wp::adj_mul(var_103, var_44, adj_103, adj_44, adj_104);
    wp::adj_mul(var_102, var_95, adj_102, adj_95, adj_103);
    wp::adj_neg(var_75, adj_75, adj_102);
    // adj: d2 = -grad2 * scale * w2                                                          <L 232>
    wp::adj_mul(var_100, var_39, adj_100, adj_39, adj_101);
    wp::adj_mul(var_99, var_95, adj_99, adj_95, adj_100);
    wp::adj_neg(var_70, adj_70, adj_99);
    // adj: d1 = -grad1 * scale * w1                                                          <L 231>
    wp::adj_mul(var_97, var_34, adj_97, adj_34, adj_98);
    wp::adj_mul(var_96, var_95, adj_96, adj_95, adj_97);
    wp::adj_neg(var_65, adj_65, adj_96);
    // adj: d0 = -grad0 * scale * w0                                                          <L 230>
    wp::adj_div(var_94, var_91, var_95, adj_94, adj_91, adj_95);
    wp::adj_mul(var_stiffness, var_59, adj_stiffness, adj_59, adj_94);
    // adj: scale = stiffness * c / sum_grad                                                  <L 228>
    if (var_93) {
        label1:;
        // adj: return                                                                        <L 226>
    }
    // adj: if sum_grad < 1e-8:                                                               <L 225>
    wp::adj_add(var_88, var_90, adj_88, adj_90, adj_91);
    wp::adj_mul(var_49, var_89, adj_49, adj_89, adj_90);
    wp::adj_length_sq(var_80, adj_80, adj_89);
    // adj: + w3 * wp.length_sq(grad3)                                                        <L 223>
    wp::adj_add(var_85, var_87, adj_85, adj_87, adj_88);
    wp::adj_mul(var_44, var_86, adj_44, adj_86, adj_87);
    wp::adj_length_sq(var_75, adj_75, adj_86);
    // adj: + w2 * wp.length_sq(grad2)                                                        <L 222>
    wp::adj_add(var_82, var_84, adj_82, adj_84, adj_85);
    wp::adj_mul(var_39, var_83, adj_39, adj_83, adj_84);
    wp::adj_length_sq(var_70, adj_70, adj_83);
    // adj: + w1 * wp.length_sq(grad1)                                                        <L 221>
    wp::adj_mul(var_34, var_81, adj_34, adj_81, adj_82);
    wp::adj_length_sq(var_65, adj_65, adj_81);
    // adj: w0 * wp.length_sq(grad0)                                                          <L 220>
    // adj: sum_grad = (                                                                      <L 219>
    wp::adj_div(var_78, var_79, adj_78, adj_79, adj_80);
    wp::adj_cross(var_76, var_77, adj_76, adj_77, adj_78);
    wp::adj_sub(var_24, var_14, adj_24, adj_14, adj_77);
    wp::adj_sub(var_19, var_14, adj_19, adj_14, adj_76);
    // adj: grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0                                          <L 217>
    wp::adj_div(var_73, var_74, adj_73, adj_74, adj_75);
    wp::adj_cross(var_71, var_72, adj_71, adj_72, adj_73);
    wp::adj_sub(var_29, var_19, adj_29, adj_19, adj_72);
    wp::adj_sub(var_14, var_19, adj_14, adj_19, adj_71);
    // adj: grad2 = wp.cross(p0 - p1, p3 - p1) / 6.0                                          <L 216>
    wp::adj_div(var_68, var_69, adj_68, adj_69, adj_70);
    wp::adj_cross(var_66, var_67, adj_66, adj_67, adj_68);
    wp::adj_sub(var_29, var_14, adj_29, adj_14, adj_67);
    wp::adj_sub(var_24, var_14, adj_24, adj_14, adj_66);
    // adj: grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0                                          <L 215>
    wp::adj_div(var_63, var_64, adj_63, adj_64, adj_65);
    wp::adj_cross(var_61, var_62, adj_61, adj_62, adj_63);
    wp::adj_sub(var_29, var_24, adj_29, adj_24, adj_62);
    wp::adj_sub(var_19, var_24, adj_19, adj_24, adj_61);
    // adj: grad0 = wp.cross(p1 - p2, p3 - p2) / 6.0                                          <L 214>
    wp::adj_sub(var_57, var_60, adj_57, adj_58, adj_59);
    adj_6.rest_volume += adj_58;
    // adj: c = v - tet.rest_volume                                                           <L 212>
    wp::adj_div(var_55, var_56, var_57, adj_55, adj_56, adj_57);
    wp::adj_dot(var_53, var_54, adj_53, adj_54, adj_55);
    wp::adj_sub(var_29, var_14, adj_29, adj_14, adj_54);
    wp::adj_cross(var_51, var_52, adj_51, adj_52, adj_53);
    wp::adj_sub(var_24, var_14, adj_24, adj_14, adj_52);
    wp::adj_sub(var_19, var_14, adj_19, adj_14, adj_51);
    // adj: v = wp.dot(wp.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0                             <L 211>
    wp::adj_copy(var_50, adj_48, adj_49);
    wp::adj_address(var_invmass, var_47, adj_invmass, adj_47, adj_48);
    wp::adj_extract(var_9, var_46, adj_9, adj_46, adj_47);
    // adj: w3 = invmass[ids[3]]                                                              <L 209>
    wp::adj_copy(var_45, adj_43, adj_44);
    wp::adj_address(var_invmass, var_42, adj_invmass, adj_42, adj_43);
    wp::adj_extract(var_9, var_41, adj_9, adj_41, adj_42);
    // adj: w2 = invmass[ids[2]]                                                              <L 208>
    wp::adj_copy(var_40, adj_38, adj_39);
    wp::adj_address(var_invmass, var_37, adj_invmass, adj_37, adj_38);
    wp::adj_extract(var_9, var_36, adj_9, adj_36, adj_37);
    // adj: w1 = invmass[ids[1]]                                                              <L 207>
    wp::adj_copy(var_35, adj_33, adj_34);
    wp::adj_address(var_invmass, var_32, adj_invmass, adj_32, adj_33);
    wp::adj_extract(var_9, var_31, adj_9, adj_31, adj_32);
    // adj: w0 = invmass[ids[0]]                                                              <L 206>
    wp::adj_copy(var_30, adj_28, adj_29);
    wp::adj_address(var_positions, var_27, adj_positions, adj_27, adj_28);
    wp::adj_extract(var_9, var_26, adj_9, adj_26, adj_27);
    // adj: p3 = positions[ids[3]]                                                            <L 204>
    wp::adj_copy(var_25, adj_23, adj_24);
    wp::adj_address(var_positions, var_22, adj_positions, adj_22, adj_23);
    wp::adj_extract(var_9, var_21, adj_9, adj_21, adj_22);
    // adj: p2 = positions[ids[2]]                                                            <L 203>
    wp::adj_copy(var_20, adj_18, adj_19);
    wp::adj_address(var_positions, var_17, adj_positions, adj_17, adj_18);
    wp::adj_extract(var_9, var_16, adj_9, adj_16, adj_17);
    // adj: p1 = positions[ids[1]]                                                            <L 202>
    wp::adj_copy(var_15, adj_13, adj_14);
    wp::adj_address(var_positions, var_12, adj_positions, adj_12, adj_13);
    wp::adj_extract(var_9, var_11, adj_9, adj_11, adj_12);
    // adj: p0 = positions[ids[0]]                                                            <L 201>
    wp::adj_copy(var_10, adj_8, adj_9);
    adj_6.ids = adj_8;
    // adj: ids = tet.ids                                                                     <L 199>
    wp::adj_copy(var_7, adj_5, adj_6);
    wp::adj_address(var_tetrahedra, var_0, adj_tetrahedra, adj_0, adj_5);
    // adj: tet = tetrahedra[tid]                                                             <L 198>
    if (var_3) {
        label0:;
        // adj: return                                                                        <L 196>
    }
    wp::adj_address(var_tetrahedra_active, var_0, adj_tetrahedra_active, adj_0, adj_1);
    // adj: if tetrahedra_active[tid] == 0:                                                   <L 195>
    // adj: tid = wp.tid()                                                                    <L 194>
    // adj: def solve_volume_constraints(                                                     <L 185>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void solve_volume_constraints_e00c95df_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_solve_volume_constraints_e00c95df *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        solve_volume_constraints_e00c95df_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void solve_volume_constraints_e00c95df_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_solve_volume_constraints_e00c95df *_wp_args,
    wp_args_solve_volume_constraints_e00c95df *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        solve_volume_constraints_e00c95df_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

struct wp_args_bounds_collision_4556631c {
    wp::array_t<wp::vec_t<3, wp::float32>> positions;
    wp::array_t<wp::vec_t<3, wp::float32>> velocities;
    wp::array_t<wp::float32> inv_masses;
    wp::vec_t<3, wp::float32> bounds_min;
    wp::vec_t<3, wp::float32> bounds_max;
    wp::float32 restitution;
    wp::float32 friction;
    wp::float32 dt;
};


void bounds_collision_4556631c_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_bounds_collision_4556631c *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::float32> var_inv_masses = _wp_args->inv_masses;
    wp::vec_t<3, wp::float32> var_bounds_min = _wp_args->bounds_min;
    wp::vec_t<3, wp::float32> var_bounds_max = _wp_args->bounds_max;
    wp::float32 var_restitution = _wp_args->restitution;
    wp::float32 var_friction = _wp_args->friction;
    wp::float32 var_dt = _wp_args->dt;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    bool var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32* var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    wp::float32 var_14;
    bool var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    wp::float32 var_47;
    bool var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    bool var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::float32 var_81;
    bool var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::float32 var_86 = 0.0;
    bool var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    const wp::float32 var_102 = 0.0;
    bool var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    //---------
    // forward
    // def bounds_collision(                                                                  <L 247>
    // tid = wp.tid()                                                                         <L 257>
    var_0 = builtin_tid1d();
    // if tid >= len(positions):                                                              <L 258>
    var_1 = wp::len(var_positions);
    var_2 = (var_0 >= var_1);
    if (var_2) {
        // return                                                                             <L 259>
        return;
    }
    // pos = positions[tid]                                                                   <L 261>
    var_3 = wp::address(var_positions, var_0);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // vel = velocities[tid]                                                                  <L 262>
    var_6 = wp::address(var_velocities, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // inv_mass = inv_masses[tid]                                                             <L 263>
    var_9 = wp::address(var_inv_masses, var_0);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // for axis in range(3):                                                                  <L 265>
    // if pos[axis] < bounds_min[axis]:                                                       <L 266>
    var_13 = wp::extract(var_4, var_12);
    var_14 = wp::extract(var_bounds_min, var_12);
    var_15 = (var_13 < var_14);
    if (var_15) {
        // penetration = bounds_min[axis] - pos[axis]                                         <L 267>
        var_16 = wp::extract(var_bounds_min, var_12);
        var_17 = wp::extract(var_4, var_12);
        var_18 = wp::sub(var_16, var_17);
        // if inv_mass > 0.0:                                                                 <L 268>
        var_20 = (var_10 > var_19);
        if (var_20) {
            // vel[axis] = -vel[axis] * restitution                                           <L 269>
            var_21 = wp::extract(var_7, var_12);
            var_22 = wp::neg(var_21);
            var_23 = wp::mul(var_22, var_restitution);
            wp::assign_inplace(var_7, var_12, var_23);
            // pos[axis] = bounds_min[axis] + penetration                                     <L 270>
            var_24 = wp::extract(var_bounds_min, var_12);
            var_25 = wp::add(var_24, var_18);
            wp::assign_inplace(var_4, var_12, var_25);
        }
        if (!var_20) {
            // pos[axis] = bounds_min[axis] + penetration                                     <L 272>
            var_26 = wp::extract(var_bounds_min, var_12);
            var_27 = wp::add(var_26, var_18);
            wp::assign_inplace(var_4, var_12, var_27);
        }
    }
    if (!var_15) {
        // elif pos[axis] > bounds_max[axis]:                                                 <L 273>
        var_28 = wp::extract(var_4, var_12);
        var_29 = wp::extract(var_bounds_max, var_12);
        var_30 = (var_28 > var_29);
        if (var_30) {
            // penetration = pos[axis] - bounds_max[axis]                                     <L 274>
            var_31 = wp::extract(var_4, var_12);
            var_32 = wp::extract(var_bounds_max, var_12);
            var_33 = wp::sub(var_31, var_32);
            // if inv_mass > 0.0:                                                             <L 275>
            var_35 = (var_10 > var_34);
            if (var_35) {
                // vel[axis] = -vel[axis] * restitution                                       <L 276>
                var_36 = wp::extract(var_7, var_12);
                var_37 = wp::neg(var_36);
                var_38 = wp::mul(var_37, var_restitution);
                wp::assign_inplace(var_7, var_12, var_38);
                // pos[axis] = bounds_max[axis] - penetration                                 <L 277>
                var_39 = wp::extract(var_bounds_max, var_12);
                var_40 = wp::sub(var_39, var_33);
                wp::assign_inplace(var_4, var_12, var_40);
            }
            if (!var_35) {
                // pos[axis] = bounds_max[axis] - penetration                                 <L 279>
                var_41 = wp::extract(var_bounds_max, var_12);
                var_42 = wp::sub(var_41, var_33);
                wp::assign_inplace(var_4, var_12, var_42);
            }
        }
        var_43 = wp::where(var_30, var_33, var_18);
    }
    var_44 = wp::where(var_15, var_18, var_43);
    // if pos[axis] < bounds_min[axis]:                                                       <L 266>
    var_46 = wp::extract(var_4, var_45);
    var_47 = wp::extract(var_bounds_min, var_45);
    var_48 = (var_46 < var_47);
    if (var_48) {
        // penetration = bounds_min[axis] - pos[axis]                                         <L 267>
        var_49 = wp::extract(var_bounds_min, var_45);
        var_50 = wp::extract(var_4, var_45);
        var_51 = wp::sub(var_49, var_50);
        // if inv_mass > 0.0:                                                                 <L 268>
        var_53 = (var_10 > var_52);
        if (var_53) {
            // vel[axis] = -vel[axis] * restitution                                           <L 269>
            var_54 = wp::extract(var_7, var_45);
            var_55 = wp::neg(var_54);
            var_56 = wp::mul(var_55, var_restitution);
            wp::assign_inplace(var_7, var_45, var_56);
            // pos[axis] = bounds_min[axis] + penetration                                     <L 270>
            var_57 = wp::extract(var_bounds_min, var_45);
            var_58 = wp::add(var_57, var_51);
            wp::assign_inplace(var_4, var_45, var_58);
        }
        if (!var_53) {
            // pos[axis] = bounds_min[axis] + penetration                                     <L 272>
            var_59 = wp::extract(var_bounds_min, var_45);
            var_60 = wp::add(var_59, var_51);
            wp::assign_inplace(var_4, var_45, var_60);
        }
    }
    var_61 = wp::where(var_48, var_51, var_44);
    if (!var_48) {
        // elif pos[axis] > bounds_max[axis]:                                                 <L 273>
        var_62 = wp::extract(var_4, var_45);
        var_63 = wp::extract(var_bounds_max, var_45);
        var_64 = (var_62 > var_63);
        if (var_64) {
            // penetration = pos[axis] - bounds_max[axis]                                     <L 274>
            var_65 = wp::extract(var_4, var_45);
            var_66 = wp::extract(var_bounds_max, var_45);
            var_67 = wp::sub(var_65, var_66);
            // if inv_mass > 0.0:                                                             <L 275>
            var_69 = (var_10 > var_68);
            if (var_69) {
                // vel[axis] = -vel[axis] * restitution                                       <L 276>
                var_70 = wp::extract(var_7, var_45);
                var_71 = wp::neg(var_70);
                var_72 = wp::mul(var_71, var_restitution);
                wp::assign_inplace(var_7, var_45, var_72);
                // pos[axis] = bounds_max[axis] - penetration                                 <L 277>
                var_73 = wp::extract(var_bounds_max, var_45);
                var_74 = wp::sub(var_73, var_67);
                wp::assign_inplace(var_4, var_45, var_74);
            }
            if (!var_69) {
                // pos[axis] = bounds_max[axis] - penetration                                 <L 279>
                var_75 = wp::extract(var_bounds_max, var_45);
                var_76 = wp::sub(var_75, var_67);
                wp::assign_inplace(var_4, var_45, var_76);
            }
        }
        var_77 = wp::where(var_64, var_67, var_61);
    }
    var_78 = wp::where(var_48, var_61, var_77);
    // if pos[axis] < bounds_min[axis]:                                                       <L 266>
    var_80 = wp::extract(var_4, var_79);
    var_81 = wp::extract(var_bounds_min, var_79);
    var_82 = (var_80 < var_81);
    if (var_82) {
        // penetration = bounds_min[axis] - pos[axis]                                         <L 267>
        var_83 = wp::extract(var_bounds_min, var_79);
        var_84 = wp::extract(var_4, var_79);
        var_85 = wp::sub(var_83, var_84);
        // if inv_mass > 0.0:                                                                 <L 268>
        var_87 = (var_10 > var_86);
        if (var_87) {
            // vel[axis] = -vel[axis] * restitution                                           <L 269>
            var_88 = wp::extract(var_7, var_79);
            var_89 = wp::neg(var_88);
            var_90 = wp::mul(var_89, var_restitution);
            wp::assign_inplace(var_7, var_79, var_90);
            // pos[axis] = bounds_min[axis] + penetration                                     <L 270>
            var_91 = wp::extract(var_bounds_min, var_79);
            var_92 = wp::add(var_91, var_85);
            wp::assign_inplace(var_4, var_79, var_92);
        }
        if (!var_87) {
            // pos[axis] = bounds_min[axis] + penetration                                     <L 272>
            var_93 = wp::extract(var_bounds_min, var_79);
            var_94 = wp::add(var_93, var_85);
            wp::assign_inplace(var_4, var_79, var_94);
        }
    }
    var_95 = wp::where(var_82, var_85, var_78);
    if (!var_82) {
        // elif pos[axis] > bounds_max[axis]:                                                 <L 273>
        var_96 = wp::extract(var_4, var_79);
        var_97 = wp::extract(var_bounds_max, var_79);
        var_98 = (var_96 > var_97);
        if (var_98) {
            // penetration = pos[axis] - bounds_max[axis]                                     <L 274>
            var_99 = wp::extract(var_4, var_79);
            var_100 = wp::extract(var_bounds_max, var_79);
            var_101 = wp::sub(var_99, var_100);
            // if inv_mass > 0.0:                                                             <L 275>
            var_103 = (var_10 > var_102);
            if (var_103) {
                // vel[axis] = -vel[axis] * restitution                                       <L 276>
                var_104 = wp::extract(var_7, var_79);
                var_105 = wp::neg(var_104);
                var_106 = wp::mul(var_105, var_restitution);
                wp::assign_inplace(var_7, var_79, var_106);
                // pos[axis] = bounds_max[axis] - penetration                                 <L 277>
                var_107 = wp::extract(var_bounds_max, var_79);
                var_108 = wp::sub(var_107, var_101);
                wp::assign_inplace(var_4, var_79, var_108);
            }
            if (!var_103) {
                // pos[axis] = bounds_max[axis] - penetration                                 <L 279>
                var_109 = wp::extract(var_bounds_max, var_79);
                var_110 = wp::sub(var_109, var_101);
                wp::assign_inplace(var_4, var_79, var_110);
            }
        }
        var_111 = wp::where(var_98, var_101, var_95);
    }
    var_112 = wp::where(var_82, var_95, var_111);
    // positions[tid] = pos                                                                   <L 281>
    wp::array_store(var_positions, var_0, var_4);
    // velocities[tid] = vel                                                                  <L 282>
    wp::array_store(var_velocities, var_0, var_7);
}



void bounds_collision_4556631c_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_bounds_collision_4556631c *_wp_args,
    wp_args_bounds_collision_4556631c *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::vec_t<3, wp::float32>> var_positions = _wp_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> var_velocities = _wp_args->velocities;
    wp::array_t<wp::float32> var_inv_masses = _wp_args->inv_masses;
    wp::vec_t<3, wp::float32> var_bounds_min = _wp_args->bounds_min;
    wp::vec_t<3, wp::float32> var_bounds_max = _wp_args->bounds_max;
    wp::float32 var_restitution = _wp_args->restitution;
    wp::float32 var_friction = _wp_args->friction;
    wp::float32 var_dt = _wp_args->dt;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_positions = _wp_adj_args->positions;
    wp::array_t<wp::vec_t<3, wp::float32>> adj_velocities = _wp_adj_args->velocities;
    wp::array_t<wp::float32> adj_inv_masses = _wp_adj_args->inv_masses;
    wp::vec_t<3, wp::float32> adj_bounds_min = _wp_adj_args->bounds_min;
    wp::vec_t<3, wp::float32> adj_bounds_max = _wp_adj_args->bounds_max;
    wp::float32 adj_restitution = _wp_adj_args->restitution;
    wp::float32 adj_friction = _wp_adj_args->friction;
    wp::float32 adj_dt = _wp_adj_args->dt;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    bool var_2;
    wp::vec_t<3, wp::float32>* var_3;
    wp::vec_t<3, wp::float32> var_4;
    wp::vec_t<3, wp::float32> var_5;
    wp::vec_t<3, wp::float32>* var_6;
    wp::vec_t<3, wp::float32> var_7;
    wp::vec_t<3, wp::float32> var_8;
    wp::float32* var_9;
    wp::float32 var_10;
    wp::float32 var_11;
    const wp::int32 var_12 = 0;
    wp::float32 var_13;
    wp::float32 var_14;
    bool var_15;
    wp::float32 var_16;
    wp::float32 var_17;
    wp::float32 var_18;
    const wp::float32 var_19 = 0.0;
    bool var_20;
    wp::float32 var_21;
    wp::float32 var_22;
    wp::float32 var_23;
    wp::float32 var_24;
    wp::float32 var_25;
    wp::float32 var_26;
    wp::float32 var_27;
    wp::float32 var_28;
    wp::float32 var_29;
    bool var_30;
    wp::float32 var_31;
    wp::float32 var_32;
    wp::float32 var_33;
    const wp::float32 var_34 = 0.0;
    bool var_35;
    wp::float32 var_36;
    wp::float32 var_37;
    wp::float32 var_38;
    wp::float32 var_39;
    wp::float32 var_40;
    wp::float32 var_41;
    wp::float32 var_42;
    wp::float32 var_43;
    wp::float32 var_44;
    const wp::int32 var_45 = 1;
    wp::float32 var_46;
    wp::float32 var_47;
    bool var_48;
    wp::float32 var_49;
    wp::float32 var_50;
    wp::float32 var_51;
    const wp::float32 var_52 = 0.0;
    bool var_53;
    wp::float32 var_54;
    wp::float32 var_55;
    wp::float32 var_56;
    wp::float32 var_57;
    wp::float32 var_58;
    wp::float32 var_59;
    wp::float32 var_60;
    wp::float32 var_61;
    wp::float32 var_62;
    wp::float32 var_63;
    bool var_64;
    wp::float32 var_65;
    wp::float32 var_66;
    wp::float32 var_67;
    const wp::float32 var_68 = 0.0;
    bool var_69;
    wp::float32 var_70;
    wp::float32 var_71;
    wp::float32 var_72;
    wp::float32 var_73;
    wp::float32 var_74;
    wp::float32 var_75;
    wp::float32 var_76;
    wp::float32 var_77;
    wp::float32 var_78;
    const wp::int32 var_79 = 2;
    wp::float32 var_80;
    wp::float32 var_81;
    bool var_82;
    wp::float32 var_83;
    wp::float32 var_84;
    wp::float32 var_85;
    const wp::float32 var_86 = 0.0;
    bool var_87;
    wp::float32 var_88;
    wp::float32 var_89;
    wp::float32 var_90;
    wp::float32 var_91;
    wp::float32 var_92;
    wp::float32 var_93;
    wp::float32 var_94;
    wp::float32 var_95;
    wp::float32 var_96;
    wp::float32 var_97;
    bool var_98;
    wp::float32 var_99;
    wp::float32 var_100;
    wp::float32 var_101;
    const wp::float32 var_102 = 0.0;
    bool var_103;
    wp::float32 var_104;
    wp::float32 var_105;
    wp::float32 var_106;
    wp::float32 var_107;
    wp::float32 var_108;
    wp::float32 var_109;
    wp::float32 var_110;
    wp::float32 var_111;
    wp::float32 var_112;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    bool adj_2 = {};
    wp::vec_t<3, wp::float32> adj_3 = {};
    wp::vec_t<3, wp::float32> adj_4 = {};
    wp::vec_t<3, wp::float32> adj_5 = {};
    wp::vec_t<3, wp::float32> adj_6 = {};
    wp::vec_t<3, wp::float32> adj_7 = {};
    wp::vec_t<3, wp::float32> adj_8 = {};
    wp::float32 adj_9 = {};
    wp::float32 adj_10 = {};
    wp::float32 adj_11 = {};
    wp::int32 adj_12 = {};
    wp::float32 adj_13 = {};
    wp::float32 adj_14 = {};
    bool adj_15 = {};
    wp::float32 adj_16 = {};
    wp::float32 adj_17 = {};
    wp::float32 adj_18 = {};
    wp::float32 adj_19 = {};
    bool adj_20 = {};
    wp::float32 adj_21 = {};
    wp::float32 adj_22 = {};
    wp::float32 adj_23 = {};
    wp::float32 adj_24 = {};
    wp::float32 adj_25 = {};
    wp::float32 adj_26 = {};
    wp::float32 adj_27 = {};
    wp::float32 adj_28 = {};
    wp::float32 adj_29 = {};
    bool adj_30 = {};
    wp::float32 adj_31 = {};
    wp::float32 adj_32 = {};
    wp::float32 adj_33 = {};
    wp::float32 adj_34 = {};
    bool adj_35 = {};
    wp::float32 adj_36 = {};
    wp::float32 adj_37 = {};
    wp::float32 adj_38 = {};
    wp::float32 adj_39 = {};
    wp::float32 adj_40 = {};
    wp::float32 adj_41 = {};
    wp::float32 adj_42 = {};
    wp::float32 adj_43 = {};
    wp::float32 adj_44 = {};
    wp::int32 adj_45 = {};
    wp::float32 adj_46 = {};
    wp::float32 adj_47 = {};
    bool adj_48 = {};
    wp::float32 adj_49 = {};
    wp::float32 adj_50 = {};
    wp::float32 adj_51 = {};
    wp::float32 adj_52 = {};
    bool adj_53 = {};
    wp::float32 adj_54 = {};
    wp::float32 adj_55 = {};
    wp::float32 adj_56 = {};
    wp::float32 adj_57 = {};
    wp::float32 adj_58 = {};
    wp::float32 adj_59 = {};
    wp::float32 adj_60 = {};
    wp::float32 adj_61 = {};
    wp::float32 adj_62 = {};
    wp::float32 adj_63 = {};
    bool adj_64 = {};
    wp::float32 adj_65 = {};
    wp::float32 adj_66 = {};
    wp::float32 adj_67 = {};
    wp::float32 adj_68 = {};
    bool adj_69 = {};
    wp::float32 adj_70 = {};
    wp::float32 adj_71 = {};
    wp::float32 adj_72 = {};
    wp::float32 adj_73 = {};
    wp::float32 adj_74 = {};
    wp::float32 adj_75 = {};
    wp::float32 adj_76 = {};
    wp::float32 adj_77 = {};
    wp::float32 adj_78 = {};
    wp::int32 adj_79 = {};
    wp::float32 adj_80 = {};
    wp::float32 adj_81 = {};
    bool adj_82 = {};
    wp::float32 adj_83 = {};
    wp::float32 adj_84 = {};
    wp::float32 adj_85 = {};
    wp::float32 adj_86 = {};
    bool adj_87 = {};
    wp::float32 adj_88 = {};
    wp::float32 adj_89 = {};
    wp::float32 adj_90 = {};
    wp::float32 adj_91 = {};
    wp::float32 adj_92 = {};
    wp::float32 adj_93 = {};
    wp::float32 adj_94 = {};
    wp::float32 adj_95 = {};
    wp::float32 adj_96 = {};
    wp::float32 adj_97 = {};
    bool adj_98 = {};
    wp::float32 adj_99 = {};
    wp::float32 adj_100 = {};
    wp::float32 adj_101 = {};
    wp::float32 adj_102 = {};
    bool adj_103 = {};
    wp::float32 adj_104 = {};
    wp::float32 adj_105 = {};
    wp::float32 adj_106 = {};
    wp::float32 adj_107 = {};
    wp::float32 adj_108 = {};
    wp::float32 adj_109 = {};
    wp::float32 adj_110 = {};
    wp::float32 adj_111 = {};
    wp::float32 adj_112 = {};
    //---------
    // forward
    // def bounds_collision(                                                                  <L 247>
    // tid = wp.tid()                                                                         <L 257>
    var_0 = builtin_tid1d();
    // if tid >= len(positions):                                                              <L 258>
    var_1 = wp::len(var_positions);
    var_2 = (var_0 >= var_1);
    if (var_2) {
        // return                                                                             <L 259>
        goto label0;
    }
    // pos = positions[tid]                                                                   <L 261>
    var_3 = wp::address(var_positions, var_0);
    var_5 = wp::load(var_3);
    var_4 = wp::copy(var_5);
    // vel = velocities[tid]                                                                  <L 262>
    var_6 = wp::address(var_velocities, var_0);
    var_8 = wp::load(var_6);
    var_7 = wp::copy(var_8);
    // inv_mass = inv_masses[tid]                                                             <L 263>
    var_9 = wp::address(var_inv_masses, var_0);
    var_11 = wp::load(var_9);
    var_10 = wp::copy(var_11);
    // for axis in range(3):                                                                  <L 265>
    // if pos[axis] < bounds_min[axis]:                                                       <L 266>
    var_13 = wp::extract(var_4, var_12);
    var_14 = wp::extract(var_bounds_min, var_12);
    var_15 = (var_13 < var_14);
    if (var_15) {
        // penetration = bounds_min[axis] - pos[axis]                                         <L 267>
        var_16 = wp::extract(var_bounds_min, var_12);
        var_17 = wp::extract(var_4, var_12);
        var_18 = wp::sub(var_16, var_17);
        // if inv_mass > 0.0:                                                                 <L 268>
        var_20 = (var_10 > var_19);
        if (var_20) {
            // vel[axis] = -vel[axis] * restitution                                           <L 269>
            var_21 = wp::extract(var_7, var_12);
            var_22 = wp::neg(var_21);
            var_23 = wp::mul(var_22, var_restitution);
            wp::assign_inplace(var_7, var_12, var_23);
            // pos[axis] = bounds_min[axis] + penetration                                     <L 270>
            var_24 = wp::extract(var_bounds_min, var_12);
            var_25 = wp::add(var_24, var_18);
            wp::assign_inplace(var_4, var_12, var_25);
        }
        if (!var_20) {
            // pos[axis] = bounds_min[axis] + penetration                                     <L 272>
            var_26 = wp::extract(var_bounds_min, var_12);
            var_27 = wp::add(var_26, var_18);
            wp::assign_inplace(var_4, var_12, var_27);
        }
    }
    if (!var_15) {
        // elif pos[axis] > bounds_max[axis]:                                                 <L 273>
        var_28 = wp::extract(var_4, var_12);
        var_29 = wp::extract(var_bounds_max, var_12);
        var_30 = (var_28 > var_29);
        if (var_30) {
            // penetration = pos[axis] - bounds_max[axis]                                     <L 274>
            var_31 = wp::extract(var_4, var_12);
            var_32 = wp::extract(var_bounds_max, var_12);
            var_33 = wp::sub(var_31, var_32);
            // if inv_mass > 0.0:                                                             <L 275>
            var_35 = (var_10 > var_34);
            if (var_35) {
                // vel[axis] = -vel[axis] * restitution                                       <L 276>
                var_36 = wp::extract(var_7, var_12);
                var_37 = wp::neg(var_36);
                var_38 = wp::mul(var_37, var_restitution);
                wp::assign_inplace(var_7, var_12, var_38);
                // pos[axis] = bounds_max[axis] - penetration                                 <L 277>
                var_39 = wp::extract(var_bounds_max, var_12);
                var_40 = wp::sub(var_39, var_33);
                wp::assign_inplace(var_4, var_12, var_40);
            }
            if (!var_35) {
                // pos[axis] = bounds_max[axis] - penetration                                 <L 279>
                var_41 = wp::extract(var_bounds_max, var_12);
                var_42 = wp::sub(var_41, var_33);
                wp::assign_inplace(var_4, var_12, var_42);
            }
        }
        var_43 = wp::where(var_30, var_33, var_18);
    }
    var_44 = wp::where(var_15, var_18, var_43);
    // if pos[axis] < bounds_min[axis]:                                                       <L 266>
    var_46 = wp::extract(var_4, var_45);
    var_47 = wp::extract(var_bounds_min, var_45);
    var_48 = (var_46 < var_47);
    if (var_48) {
        // penetration = bounds_min[axis] - pos[axis]                                         <L 267>
        var_49 = wp::extract(var_bounds_min, var_45);
        var_50 = wp::extract(var_4, var_45);
        var_51 = wp::sub(var_49, var_50);
        // if inv_mass > 0.0:                                                                 <L 268>
        var_53 = (var_10 > var_52);
        if (var_53) {
            // vel[axis] = -vel[axis] * restitution                                           <L 269>
            var_54 = wp::extract(var_7, var_45);
            var_55 = wp::neg(var_54);
            var_56 = wp::mul(var_55, var_restitution);
            wp::assign_inplace(var_7, var_45, var_56);
            // pos[axis] = bounds_min[axis] + penetration                                     <L 270>
            var_57 = wp::extract(var_bounds_min, var_45);
            var_58 = wp::add(var_57, var_51);
            wp::assign_inplace(var_4, var_45, var_58);
        }
        if (!var_53) {
            // pos[axis] = bounds_min[axis] + penetration                                     <L 272>
            var_59 = wp::extract(var_bounds_min, var_45);
            var_60 = wp::add(var_59, var_51);
            wp::assign_inplace(var_4, var_45, var_60);
        }
    }
    var_61 = wp::where(var_48, var_51, var_44);
    if (!var_48) {
        // elif pos[axis] > bounds_max[axis]:                                                 <L 273>
        var_62 = wp::extract(var_4, var_45);
        var_63 = wp::extract(var_bounds_max, var_45);
        var_64 = (var_62 > var_63);
        if (var_64) {
            // penetration = pos[axis] - bounds_max[axis]                                     <L 274>
            var_65 = wp::extract(var_4, var_45);
            var_66 = wp::extract(var_bounds_max, var_45);
            var_67 = wp::sub(var_65, var_66);
            // if inv_mass > 0.0:                                                             <L 275>
            var_69 = (var_10 > var_68);
            if (var_69) {
                // vel[axis] = -vel[axis] * restitution                                       <L 276>
                var_70 = wp::extract(var_7, var_45);
                var_71 = wp::neg(var_70);
                var_72 = wp::mul(var_71, var_restitution);
                wp::assign_inplace(var_7, var_45, var_72);
                // pos[axis] = bounds_max[axis] - penetration                                 <L 277>
                var_73 = wp::extract(var_bounds_max, var_45);
                var_74 = wp::sub(var_73, var_67);
                wp::assign_inplace(var_4, var_45, var_74);
            }
            if (!var_69) {
                // pos[axis] = bounds_max[axis] - penetration                                 <L 279>
                var_75 = wp::extract(var_bounds_max, var_45);
                var_76 = wp::sub(var_75, var_67);
                wp::assign_inplace(var_4, var_45, var_76);
            }
        }
        var_77 = wp::where(var_64, var_67, var_61);
    }
    var_78 = wp::where(var_48, var_61, var_77);
    // if pos[axis] < bounds_min[axis]:                                                       <L 266>
    var_80 = wp::extract(var_4, var_79);
    var_81 = wp::extract(var_bounds_min, var_79);
    var_82 = (var_80 < var_81);
    if (var_82) {
        // penetration = bounds_min[axis] - pos[axis]                                         <L 267>
        var_83 = wp::extract(var_bounds_min, var_79);
        var_84 = wp::extract(var_4, var_79);
        var_85 = wp::sub(var_83, var_84);
        // if inv_mass > 0.0:                                                                 <L 268>
        var_87 = (var_10 > var_86);
        if (var_87) {
            // vel[axis] = -vel[axis] * restitution                                           <L 269>
            var_88 = wp::extract(var_7, var_79);
            var_89 = wp::neg(var_88);
            var_90 = wp::mul(var_89, var_restitution);
            wp::assign_inplace(var_7, var_79, var_90);
            // pos[axis] = bounds_min[axis] + penetration                                     <L 270>
            var_91 = wp::extract(var_bounds_min, var_79);
            var_92 = wp::add(var_91, var_85);
            wp::assign_inplace(var_4, var_79, var_92);
        }
        if (!var_87) {
            // pos[axis] = bounds_min[axis] + penetration                                     <L 272>
            var_93 = wp::extract(var_bounds_min, var_79);
            var_94 = wp::add(var_93, var_85);
            wp::assign_inplace(var_4, var_79, var_94);
        }
    }
    var_95 = wp::where(var_82, var_85, var_78);
    if (!var_82) {
        // elif pos[axis] > bounds_max[axis]:                                                 <L 273>
        var_96 = wp::extract(var_4, var_79);
        var_97 = wp::extract(var_bounds_max, var_79);
        var_98 = (var_96 > var_97);
        if (var_98) {
            // penetration = pos[axis] - bounds_max[axis]                                     <L 274>
            var_99 = wp::extract(var_4, var_79);
            var_100 = wp::extract(var_bounds_max, var_79);
            var_101 = wp::sub(var_99, var_100);
            // if inv_mass > 0.0:                                                             <L 275>
            var_103 = (var_10 > var_102);
            if (var_103) {
                // vel[axis] = -vel[axis] * restitution                                       <L 276>
                var_104 = wp::extract(var_7, var_79);
                var_105 = wp::neg(var_104);
                var_106 = wp::mul(var_105, var_restitution);
                wp::assign_inplace(var_7, var_79, var_106);
                // pos[axis] = bounds_max[axis] - penetration                                 <L 277>
                var_107 = wp::extract(var_bounds_max, var_79);
                var_108 = wp::sub(var_107, var_101);
                wp::assign_inplace(var_4, var_79, var_108);
            }
            if (!var_103) {
                // pos[axis] = bounds_max[axis] - penetration                                 <L 279>
                var_109 = wp::extract(var_bounds_max, var_79);
                var_110 = wp::sub(var_109, var_101);
                wp::assign_inplace(var_4, var_79, var_110);
            }
        }
        var_111 = wp::where(var_98, var_101, var_95);
    }
    var_112 = wp::where(var_82, var_95, var_111);
    // positions[tid] = pos                                                                   <L 281>
    // wp::array_store(var_positions, var_0, var_4);
    // velocities[tid] = vel                                                                  <L 282>
    // wp::array_store(var_velocities, var_0, var_7);
    //---------
    // reverse
    wp::adj_array_store(var_velocities, var_0, var_7, adj_velocities, adj_0, adj_7);
    // adj: velocities[tid] = vel                                                             <L 282>
    wp::adj_array_store(var_positions, var_0, var_4, adj_positions, adj_0, adj_4);
    // adj: positions[tid] = pos                                                              <L 281>
    wp::adj_where(var_82, var_95, var_111, adj_82, adj_95, adj_111, adj_112);
    if (!var_82) {
        wp::adj_where(var_98, var_101, var_95, adj_98, adj_101, adj_95, adj_111);
        if (var_98) {
            if (!var_103) {
                wp::adj_assign_inplace(var_4, var_79, var_110, adj_4, adj_79, adj_110);
                wp::adj_sub(var_109, var_101, adj_109, adj_101, adj_110);
                wp::adj_extract(var_bounds_max, var_79, adj_bounds_max, adj_79, adj_109);
                // adj: pos[axis] = bounds_max[axis] - penetration                            <L 279>
            }
            if (var_103) {
                wp::adj_assign_inplace(var_4, var_79, var_108, adj_4, adj_79, adj_108);
                wp::adj_sub(var_107, var_101, adj_107, adj_101, adj_108);
                wp::adj_extract(var_bounds_max, var_79, adj_bounds_max, adj_79, adj_107);
                // adj: pos[axis] = bounds_max[axis] - penetration                            <L 277>
                wp::adj_assign_inplace(var_7, var_79, var_106, adj_7, adj_79, adj_106);
                wp::adj_mul(var_105, var_restitution, adj_105, adj_restitution, adj_106);
                wp::adj_neg(var_104, adj_104, adj_105);
                wp::adj_extract(var_7, var_79, adj_7, adj_79, adj_104);
                // adj: vel[axis] = -vel[axis] * restitution                                  <L 276>
            }
            // adj: if inv_mass > 0.0:                                                        <L 275>
            wp::adj_sub(var_99, var_100, adj_99, adj_100, adj_101);
            wp::adj_extract(var_bounds_max, var_79, adj_bounds_max, adj_79, adj_100);
            wp::adj_extract(var_4, var_79, adj_4, adj_79, adj_99);
            // adj: penetration = pos[axis] - bounds_max[axis]                                <L 274>
        }
        wp::adj_extract(var_bounds_max, var_79, adj_bounds_max, adj_79, adj_97);
        wp::adj_extract(var_4, var_79, adj_4, adj_79, adj_96);
        // adj: elif pos[axis] > bounds_max[axis]:                                            <L 273>
    }
    wp::adj_where(var_82, var_85, var_78, adj_82, adj_85, adj_78, adj_95);
    if (var_82) {
        if (!var_87) {
            wp::adj_assign_inplace(var_4, var_79, var_94, adj_4, adj_79, adj_94);
            wp::adj_add(var_93, var_85, adj_93, adj_85, adj_94);
            wp::adj_extract(var_bounds_min, var_79, adj_bounds_min, adj_79, adj_93);
            // adj: pos[axis] = bounds_min[axis] + penetration                                <L 272>
        }
        if (var_87) {
            wp::adj_assign_inplace(var_4, var_79, var_92, adj_4, adj_79, adj_92);
            wp::adj_add(var_91, var_85, adj_91, adj_85, adj_92);
            wp::adj_extract(var_bounds_min, var_79, adj_bounds_min, adj_79, adj_91);
            // adj: pos[axis] = bounds_min[axis] + penetration                                <L 270>
            wp::adj_assign_inplace(var_7, var_79, var_90, adj_7, adj_79, adj_90);
            wp::adj_mul(var_89, var_restitution, adj_89, adj_restitution, adj_90);
            wp::adj_neg(var_88, adj_88, adj_89);
            wp::adj_extract(var_7, var_79, adj_7, adj_79, adj_88);
            // adj: vel[axis] = -vel[axis] * restitution                                      <L 269>
        }
        // adj: if inv_mass > 0.0:                                                            <L 268>
        wp::adj_sub(var_83, var_84, adj_83, adj_84, adj_85);
        wp::adj_extract(var_4, var_79, adj_4, adj_79, adj_84);
        wp::adj_extract(var_bounds_min, var_79, adj_bounds_min, adj_79, adj_83);
        // adj: penetration = bounds_min[axis] - pos[axis]                                    <L 267>
    }
    wp::adj_extract(var_bounds_min, var_79, adj_bounds_min, adj_79, adj_81);
    wp::adj_extract(var_4, var_79, adj_4, adj_79, adj_80);
    // adj: if pos[axis] < bounds_min[axis]:                                                  <L 266>
    wp::adj_where(var_48, var_61, var_77, adj_48, adj_61, adj_77, adj_78);
    if (!var_48) {
        wp::adj_where(var_64, var_67, var_61, adj_64, adj_67, adj_61, adj_77);
        if (var_64) {
            if (!var_69) {
                wp::adj_assign_inplace(var_4, var_45, var_76, adj_4, adj_45, adj_76);
                wp::adj_sub(var_75, var_67, adj_75, adj_67, adj_76);
                wp::adj_extract(var_bounds_max, var_45, adj_bounds_max, adj_45, adj_75);
                // adj: pos[axis] = bounds_max[axis] - penetration                            <L 279>
            }
            if (var_69) {
                wp::adj_assign_inplace(var_4, var_45, var_74, adj_4, adj_45, adj_74);
                wp::adj_sub(var_73, var_67, adj_73, adj_67, adj_74);
                wp::adj_extract(var_bounds_max, var_45, adj_bounds_max, adj_45, adj_73);
                // adj: pos[axis] = bounds_max[axis] - penetration                            <L 277>
                wp::adj_assign_inplace(var_7, var_45, var_72, adj_7, adj_45, adj_72);
                wp::adj_mul(var_71, var_restitution, adj_71, adj_restitution, adj_72);
                wp::adj_neg(var_70, adj_70, adj_71);
                wp::adj_extract(var_7, var_45, adj_7, adj_45, adj_70);
                // adj: vel[axis] = -vel[axis] * restitution                                  <L 276>
            }
            // adj: if inv_mass > 0.0:                                                        <L 275>
            wp::adj_sub(var_65, var_66, adj_65, adj_66, adj_67);
            wp::adj_extract(var_bounds_max, var_45, adj_bounds_max, adj_45, adj_66);
            wp::adj_extract(var_4, var_45, adj_4, adj_45, adj_65);
            // adj: penetration = pos[axis] - bounds_max[axis]                                <L 274>
        }
        wp::adj_extract(var_bounds_max, var_45, adj_bounds_max, adj_45, adj_63);
        wp::adj_extract(var_4, var_45, adj_4, adj_45, adj_62);
        // adj: elif pos[axis] > bounds_max[axis]:                                            <L 273>
    }
    wp::adj_where(var_48, var_51, var_44, adj_48, adj_51, adj_44, adj_61);
    if (var_48) {
        if (!var_53) {
            wp::adj_assign_inplace(var_4, var_45, var_60, adj_4, adj_45, adj_60);
            wp::adj_add(var_59, var_51, adj_59, adj_51, adj_60);
            wp::adj_extract(var_bounds_min, var_45, adj_bounds_min, adj_45, adj_59);
            // adj: pos[axis] = bounds_min[axis] + penetration                                <L 272>
        }
        if (var_53) {
            wp::adj_assign_inplace(var_4, var_45, var_58, adj_4, adj_45, adj_58);
            wp::adj_add(var_57, var_51, adj_57, adj_51, adj_58);
            wp::adj_extract(var_bounds_min, var_45, adj_bounds_min, adj_45, adj_57);
            // adj: pos[axis] = bounds_min[axis] + penetration                                <L 270>
            wp::adj_assign_inplace(var_7, var_45, var_56, adj_7, adj_45, adj_56);
            wp::adj_mul(var_55, var_restitution, adj_55, adj_restitution, adj_56);
            wp::adj_neg(var_54, adj_54, adj_55);
            wp::adj_extract(var_7, var_45, adj_7, adj_45, adj_54);
            // adj: vel[axis] = -vel[axis] * restitution                                      <L 269>
        }
        // adj: if inv_mass > 0.0:                                                            <L 268>
        wp::adj_sub(var_49, var_50, adj_49, adj_50, adj_51);
        wp::adj_extract(var_4, var_45, adj_4, adj_45, adj_50);
        wp::adj_extract(var_bounds_min, var_45, adj_bounds_min, adj_45, adj_49);
        // adj: penetration = bounds_min[axis] - pos[axis]                                    <L 267>
    }
    wp::adj_extract(var_bounds_min, var_45, adj_bounds_min, adj_45, adj_47);
    wp::adj_extract(var_4, var_45, adj_4, adj_45, adj_46);
    // adj: if pos[axis] < bounds_min[axis]:                                                  <L 266>
    wp::adj_where(var_15, var_18, var_43, adj_15, adj_18, adj_43, adj_44);
    if (!var_15) {
        wp::adj_where(var_30, var_33, var_18, adj_30, adj_33, adj_18, adj_43);
        if (var_30) {
            if (!var_35) {
                wp::adj_assign_inplace(var_4, var_12, var_42, adj_4, adj_12, adj_42);
                wp::adj_sub(var_41, var_33, adj_41, adj_33, adj_42);
                wp::adj_extract(var_bounds_max, var_12, adj_bounds_max, adj_12, adj_41);
                // adj: pos[axis] = bounds_max[axis] - penetration                            <L 279>
            }
            if (var_35) {
                wp::adj_assign_inplace(var_4, var_12, var_40, adj_4, adj_12, adj_40);
                wp::adj_sub(var_39, var_33, adj_39, adj_33, adj_40);
                wp::adj_extract(var_bounds_max, var_12, adj_bounds_max, adj_12, adj_39);
                // adj: pos[axis] = bounds_max[axis] - penetration                            <L 277>
                wp::adj_assign_inplace(var_7, var_12, var_38, adj_7, adj_12, adj_38);
                wp::adj_mul(var_37, var_restitution, adj_37, adj_restitution, adj_38);
                wp::adj_neg(var_36, adj_36, adj_37);
                wp::adj_extract(var_7, var_12, adj_7, adj_12, adj_36);
                // adj: vel[axis] = -vel[axis] * restitution                                  <L 276>
            }
            // adj: if inv_mass > 0.0:                                                        <L 275>
            wp::adj_sub(var_31, var_32, adj_31, adj_32, adj_33);
            wp::adj_extract(var_bounds_max, var_12, adj_bounds_max, adj_12, adj_32);
            wp::adj_extract(var_4, var_12, adj_4, adj_12, adj_31);
            // adj: penetration = pos[axis] - bounds_max[axis]                                <L 274>
        }
        wp::adj_extract(var_bounds_max, var_12, adj_bounds_max, adj_12, adj_29);
        wp::adj_extract(var_4, var_12, adj_4, adj_12, adj_28);
        // adj: elif pos[axis] > bounds_max[axis]:                                            <L 273>
    }
    if (var_15) {
        if (!var_20) {
            wp::adj_assign_inplace(var_4, var_12, var_27, adj_4, adj_12, adj_27);
            wp::adj_add(var_26, var_18, adj_26, adj_18, adj_27);
            wp::adj_extract(var_bounds_min, var_12, adj_bounds_min, adj_12, adj_26);
            // adj: pos[axis] = bounds_min[axis] + penetration                                <L 272>
        }
        if (var_20) {
            wp::adj_assign_inplace(var_4, var_12, var_25, adj_4, adj_12, adj_25);
            wp::adj_add(var_24, var_18, adj_24, adj_18, adj_25);
            wp::adj_extract(var_bounds_min, var_12, adj_bounds_min, adj_12, adj_24);
            // adj: pos[axis] = bounds_min[axis] + penetration                                <L 270>
            wp::adj_assign_inplace(var_7, var_12, var_23, adj_7, adj_12, adj_23);
            wp::adj_mul(var_22, var_restitution, adj_22, adj_restitution, adj_23);
            wp::adj_neg(var_21, adj_21, adj_22);
            wp::adj_extract(var_7, var_12, adj_7, adj_12, adj_21);
            // adj: vel[axis] = -vel[axis] * restitution                                      <L 269>
        }
        // adj: if inv_mass > 0.0:                                                            <L 268>
        wp::adj_sub(var_16, var_17, adj_16, adj_17, adj_18);
        wp::adj_extract(var_4, var_12, adj_4, adj_12, adj_17);
        wp::adj_extract(var_bounds_min, var_12, adj_bounds_min, adj_12, adj_16);
        // adj: penetration = bounds_min[axis] - pos[axis]                                    <L 267>
    }
    wp::adj_extract(var_bounds_min, var_12, adj_bounds_min, adj_12, adj_14);
    wp::adj_extract(var_4, var_12, adj_4, adj_12, adj_13);
    // adj: if pos[axis] < bounds_min[axis]:                                                  <L 266>
    // adj: for axis in range(3):                                                             <L 265>
    wp::adj_copy(var_11, adj_9, adj_10);
    wp::adj_address(var_inv_masses, var_0, adj_inv_masses, adj_0, adj_9);
    // adj: inv_mass = inv_masses[tid]                                                        <L 263>
    wp::adj_copy(var_8, adj_6, adj_7);
    wp::adj_address(var_velocities, var_0, adj_velocities, adj_0, adj_6);
    // adj: vel = velocities[tid]                                                             <L 262>
    wp::adj_copy(var_5, adj_3, adj_4);
    wp::adj_address(var_positions, var_0, adj_positions, adj_0, adj_3);
    // adj: pos = positions[tid]                                                              <L 261>
    if (var_2) {
        label0:;
        // adj: return                                                                        <L 259>
    }
    // adj: if tid >= len(positions):                                                         <L 258>
    // adj: tid = wp.tid()                                                                    <L 257>
    // adj: def bounds_collision(                                                             <L 247>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void bounds_collision_4556631c_cpu_forward(
    wp::launch_bounds_t *dim,
    wp_args_bounds_collision_4556631c *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        bounds_collision_4556631c_cpu_kernel_forward(*dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void bounds_collision_4556631c_cpu_backward(
    wp::launch_bounds_t *dim,
    wp_args_bounds_collision_4556631c *_wp_args,
    wp_args_bounds_collision_4556631c *_wp_adj_args)
{
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif

    for (size_t task_index = 0; task_index < dim->size; ++task_index)
    {
        bounds_collision_4556631c_cpu_kernel_backward(*dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

