#!/usr/bin/env python3
"""
Test script to verify inverted tetrahedron detection and fixing.
"""
import warp as wp

# Initialize warp
wp.init()

def fix_inverted_tetrahedrons(vertices, indices):
    """
    Detect and fix inverted tetrahedrons by checking their signed volume.
    
    A tetrahedron is inverted if its signed volume is negative. The signed volume
    is computed as: V = (1/6) * dot(v1-v0, cross(v2-v0, v3-v0))
    
    Args:
        vertices: List of wp.vec3 vertices
        indices: List of tetrahedral indices (4 indices per tetrahedron)
    
    Returns:
        Fixed indices list with inverted tetrahedrons corrected
        Number of inverted tetrahedrons fixed
    """
    if indices is None or len(indices) == 0:
        return indices, 0
    
    if len(indices) % 4 != 0:
        # Not a tetrahedral mesh
        return indices, 0
    
    num_tets = len(indices) // 4
    fixed_indices = list(indices)
    num_inverted = 0
    
    for i in range(num_tets):
        idx = i * 4
        i0, i1, i2, i3 = indices[idx], indices[idx + 1], indices[idx + 2], indices[idx + 3]
        
        # Get vertices
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        v3 = vertices[i3]
        
        # Compute edge vectors
        e1 = wp.vec3(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
        e2 = wp.vec3(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
        e3 = wp.vec3(v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2])
        
        # Compute signed volume: V = (1/6) * dot(e1, cross(e2, e3))
        # We can skip the 1/6 factor since we only care about the sign
        cross_e2_e3 = wp.vec3(
            e2[1] * e3[2] - e2[2] * e3[1],
            e2[2] * e3[0] - e2[0] * e3[2],
            e2[0] * e3[1] - e2[1] * e3[0]
        )
        signed_volume = e1[0] * cross_e2_e3[0] + e1[1] * cross_e2_e3[1] + e1[2] * cross_e2_e3[2]
        
        # If negative, tetrahedron is inverted - fix by swapping two vertices
        if signed_volume < 0:
            # Swap indices 0 and 1 to flip orientation
            fixed_indices[idx], fixed_indices[idx + 1] = fixed_indices[idx + 1], fixed_indices[idx]
            num_inverted += 1
    
    return fixed_indices, num_inverted


def compute_tet_volume(vertices, indices, tet_idx):
    """Compute the signed volume of a tetrahedron."""
    idx = tet_idx * 4
    i0, i1, i2, i3 = indices[idx], indices[idx + 1], indices[idx + 2], indices[idx + 3]
    
    v0 = vertices[i0]
    v1 = vertices[i1]
    v2 = vertices[i2]
    v3 = vertices[i3]
    
    # Compute edge vectors
    e1 = wp.vec3(v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
    e2 = wp.vec3(v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
    e3 = wp.vec3(v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2])
    
    # Compute signed volume
    cross_e2_e3 = wp.vec3(
        e2[1] * e3[2] - e2[2] * e3[1],
        e2[2] * e3[0] - e2[0] * e3[2],
        e2[0] * e3[1] - e2[1] * e3[0]
    )
    signed_volume = e1[0] * cross_e2_e3[0] + e1[1] * cross_e2_e3[1] + e1[2] * cross_e2_e3[2]
    return signed_volume / 6.0


# Test 1: Normal tetrahedron (should not be inverted)
print("Test 1: Normal tetrahedron")
vertices = [
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, 0.0, 0.0),
    wp.vec3(0.0, 1.0, 0.0),
    wp.vec3(0.0, 0.0, 1.0),
]
indices = [0, 1, 2, 3]

volume_before = compute_tet_volume(vertices, indices, 0)
print(f"  Volume before: {volume_before:.6f}")

fixed_indices, num_inverted = fix_inverted_tetrahedrons(vertices, indices)
print(f"  Inverted tets found: {num_inverted}")
assert num_inverted == 0, "Expected 0 inverted tetrahedrons"
assert fixed_indices == indices, "Indices should not change"
print("  ✓ Test passed!")


# Test 2: Inverted tetrahedron (should be fixed)
print("\nTest 2: Inverted tetrahedron")
vertices = [
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, 0.0, 0.0),
    wp.vec3(0.0, 1.0, 0.0),
    wp.vec3(0.0, 0.0, 1.0),
]
# Swap first two vertices to create inverted tet
indices = [1, 0, 2, 3]

volume_before = compute_tet_volume(vertices, indices, 0)
print(f"  Volume before: {volume_before:.6f}")

fixed_indices, num_inverted = fix_inverted_tetrahedrons(vertices, indices)
print(f"  Inverted tets found: {num_inverted}")
assert num_inverted == 1, "Expected 1 inverted tetrahedron"

volume_after = compute_tet_volume(vertices, fixed_indices, 0)
print(f"  Volume after: {volume_after:.6f}")
assert volume_after > 0, "Volume should be positive after fixing"
print("  ✓ Test passed!")


# Test 3: Multiple tetrahedrons, some inverted
print("\nTest 3: Multiple tetrahedrons")
vertices = [
    wp.vec3(0.0, 0.0, 0.0),
    wp.vec3(1.0, 0.0, 0.0),
    wp.vec3(0.0, 1.0, 0.0),
    wp.vec3(0.0, 0.0, 1.0),
    wp.vec3(1.0, 1.0, 0.0),
    wp.vec3(1.0, 0.0, 1.0),
]
# Two tets: first normal, second inverted
indices = [0, 1, 2, 3, 5, 4, 2, 3]

volume_0_before = compute_tet_volume(vertices, indices, 0)
volume_1_before = compute_tet_volume(vertices, indices, 1)
print(f"  Tet 0 volume before: {volume_0_before:.6f}")
print(f"  Tet 1 volume before: {volume_1_before:.6f}")

fixed_indices, num_inverted = fix_inverted_tetrahedrons(vertices, indices)
print(f"  Inverted tets found: {num_inverted}")
assert num_inverted == 1, "Expected 1 inverted tetrahedron"

volume_0_after = compute_tet_volume(vertices, fixed_indices, 0)
volume_1_after = compute_tet_volume(vertices, fixed_indices, 1)
print(f"  Tet 0 volume after: {volume_0_after:.6f}")
print(f"  Tet 1 volume after: {volume_1_after:.6f}")
assert volume_0_after > 0, "Tet 0 volume should be positive"
assert volume_1_after > 0, "Tet 1 volume should be positive"
print("  ✓ Test passed!")


print("\n✓ All tests passed!")


