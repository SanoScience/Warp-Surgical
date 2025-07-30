import warp as wp
from mesh_loader import Tetrahedron

@wp.kernel
def extract_surface_from_tets_old(
    tets: wp.array(dtype=Tetrahedron),           # [num_tets]
    tet_active: wp.array(dtype=wp.int32),        # [num_tets], 1=keep, 0=skip
    counter: wp.array(dtype=wp.int32),           # [1], atomic counter
    out_indices: wp.array(dtype=wp.int32, ndim=2), # [max_triangles, 3]
):
    tid = wp.tid()
    if tid >= tets.shape[0]:
        return

    if tet_active[tid] == 0:
        return

    tet = tets[tid]

    for f in range(4):
        # Get face indices
        if f == 0:
            i0, i1, i2 = tet.ids[0], tet.ids[1], tet.ids[2]
        elif f == 1:
            i0, i1, i2 = tet.ids[0], tet.ids[1], tet.ids[3]
        elif f == 2:
            i0, i1, i2 = tet.ids[0], tet.ids[2], tet.ids[3]
        else:
            i0, i1, i2 = tet.ids[1], tet.ids[2], tet.ids[3]

        # Sort
        a, b, c = i0, i1, i2
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a

        # Check if internal
        # TODO: Extremely inefficient
        shared = bool(False)
        for j in range(tets.shape[0]):
            if j == tid or tet_active[j] == 0:
                continue
            tet2 = tets[j]
            for f2 in range(4):
                # Get face indices for tet2
                if f2 == 0:
                    j0, j1, j2 = tet2.ids[0], tet2.ids[1], tet2.ids[2]
                elif f2 == 1:
                    j0, j1, j2 = tet2.ids[0], tet2.ids[1], tet2.ids[3]
                elif f2 == 2:
                    j0, j1, j2 = tet2.ids[0], tet2.ids[2], tet2.ids[3]
                else:
                    j0, j1, j2 = tet2.ids[1], tet2.ids[2], tet2.ids[3]

                # Sort
                aa, bb, cc = j0, j1, j2
                if aa > bb:
                    aa, bb = bb, aa
                if bb > cc:
                    bb, cc = cc, bb
                if aa > bb:
                    aa, bb = bb, aa

                # If the sorted face matches, it's shared
                if a == aa and b == bb and c == cc:
                    shared = True
                    break
            if shared:
                break

        # Only emit if not shared with any other active tet
        # This outputs the tet in the proper culling orientation
        if not shared:
            tri_idx = wp.atomic_add(counter, 0, 1)
            if f == 0:
                out_indices[tri_idx, 0] = tet.ids[0]
                out_indices[tri_idx, 1] = tet.ids[2]
                out_indices[tri_idx, 2] = tet.ids[1]
            elif f == 1:
                out_indices[tri_idx, 0] = tet.ids[0]
                out_indices[tri_idx, 1] = tet.ids[1]
                out_indices[tri_idx, 2] = tet.ids[3]
            elif f == 2:
                out_indices[tri_idx, 0] = tet.ids[0]
                out_indices[tri_idx, 1] = tet.ids[3]
                out_indices[tri_idx, 2] = tet.ids[2]
            else: 
                out_indices[tri_idx, 0] = tet.ids[1]
                out_indices[tri_idx, 1] = tet.ids[2]
                out_indices[tri_idx, 2] = tet.ids[3]

@wp.kernel
def surface_count_tri_references(
    tets: wp.array(dtype=Tetrahedron),           # [num_tets]
    tet_active: wp.array(dtype=wp.int32),        # [num_tets], 1=keep, 0=skip
    bucket_counters: wp.array(dtype=wp.int32),   # [num_buckets], atomic counters
    bucket_storage: wp.array(dtype=wp.int32, ndim=3), # [num_buckets, bucket_size, 3]
    num_buckets: int,                            # number of buckets
    bucket_size: int                             # max tris per bucket
):
    tid = wp.tid()
    if tid >= tets.shape[0]:
        return

    if tet_active[tid] == 0:
        return

    tet = tets[tid]

    # For each face of the tet
    for f in range(4):
        if f == 0:
            i0, i1, i2 = tet.ids[0], tet.ids[1], tet.ids[2]
        elif f == 1:
            i0, i1, i2 = tet.ids[0], tet.ids[1], tet.ids[3]
        elif f == 2:
            i0, i1, i2 = tet.ids[0], tet.ids[2], tet.ids[3]
        else:
            i0, i1, i2 = tet.ids[1], tet.ids[2], tet.ids[3]

        # Sort indices
        a, b, c = i0, i1, i2
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a

        # Hash the sorted triangle to a bucket
        h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
        bucket = wp.abs(h) % num_buckets

        # Atomically get an index in the bucket to write the triangle to
        idx = wp.atomic_add(bucket_counters, bucket, 1)
        if idx < bucket_size:
            bucket_storage[bucket, idx, 0] = a
            bucket_storage[bucket, idx, 1] = b
            bucket_storage[bucket, idx, 2] = c

@wp.kernel
def extract_surface_from_buckets(
    tets: wp.array(dtype=Tetrahedron),           # [num_tets]
    tet_active: wp.array(dtype=wp.int32),        # [num_tets], 1=keep, 0=skip
    bucket_storage: wp.array(dtype=wp.int32, ndim=3), # [num_buckets, bucket_size, 3]
    bucket_counters: wp.array(dtype=wp.int32),   # [num_buckets]
    num_buckets: int,
    counter: wp.array(dtype=wp.int32),           # [1], atomic counter
    out_indices: wp.array(dtype=wp.int32, ndim=2), # [max_triangles, 3]
):
    tid = wp.tid()
    if tid >= tets.shape[0]:
        return

    if tet_active[tid] == 0:
        return

    tet = tets[tid]

    for f in range(4):
        if f == 0:
            i0, i1, i2 = tet.ids[0], tet.ids[1], tet.ids[2]
        elif f == 1:
            i0, i1, i2 = tet.ids[0], tet.ids[1], tet.ids[3]
        elif f == 2:
            i0, i1, i2 = tet.ids[0], tet.ids[2], tet.ids[3]
        else:
            i0, i1, i2 = tet.ids[1], tet.ids[2], tet.ids[3]

        # Sort
        a, b, c = i0, i1, i2
        if a > b:
            a, b = b, a
        if b > c:
            b, c = c, b
        if a > b:
            a, b = b, a

        # Hash to bucket
        h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
        bucket = wp.abs(h) % num_buckets

        # Count references in this bucket
        ref_count = int(0)
        for idx in range(bucket_counters[bucket]):
            aa = bucket_storage[bucket, idx, 0]
            bb = bucket_storage[bucket, idx, 1]
            cc = bucket_storage[bucket, idx, 2]
            if a == aa and b == bb and c == cc:
                ref_count += 1
                if ref_count > 1:
                    break

        # Only emit if referenced at most once
        if ref_count == 1:
            tri_idx = wp.atomic_add(counter, 0, 1)
            
            # Output in proper culling orientation
            if f == 0:
                out_indices[tri_idx, 0] = tet.ids[0]
                out_indices[tri_idx, 1] = tet.ids[2]
                out_indices[tri_idx, 2] = tet.ids[1]
            elif f == 1:
                out_indices[tri_idx, 0] = tet.ids[0]
                out_indices[tri_idx, 1] = tet.ids[1]
                out_indices[tri_idx, 2] = tet.ids[3]
            elif f == 2:
                out_indices[tri_idx, 0] = tet.ids[0]
                out_indices[tri_idx, 1] = tet.ids[3]
                out_indices[tri_idx, 2] = tet.ids[2]
            else:
                out_indices[tri_idx, 0] = tet.ids[1]
                out_indices[tri_idx, 1] = tet.ids[2]
                out_indices[tri_idx, 2] = tet.ids[3]

def extract_surface_triangles_bucketed(
        mesh_tets,
        tet_active,
        out_indices,
        out_counter,
        bucket_counters,
        bucket_storage,
        num_buckets,
        bucket_size
    ):
        device = wp.get_device()
        wp.copy(bucket_counters, wp.zeros(num_buckets, dtype=wp.int32, device=device))
        wp.copy(bucket_storage, wp.zeros((num_buckets, bucket_size, 3), dtype=wp.int32, device=device))

        # Count tri references
        wp.launch(
            surface_count_tri_references,
            dim=mesh_tets.shape[0],
            inputs=[
                mesh_tets,
                tet_active,
                bucket_counters,
                bucket_storage,
                num_buckets,
                bucket_size
            ],
            device=device
        )

        # Extract surface
        wp.copy(out_counter, wp.zeros(1, dtype=wp.int32, device=device))
        wp.launch(
            extract_surface_from_buckets,
            dim=mesh_tets.shape[0],
            inputs=[
                mesh_tets,
                tet_active,
                bucket_storage,
                bucket_counters,
                num_buckets,
                out_counter,
                out_indices
            ],
            device=device
        )
