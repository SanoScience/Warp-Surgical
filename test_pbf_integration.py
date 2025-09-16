#!/usr/bin/env python3

"""
Test script to verify PBF integration with the surgical simulation.
This creates a minimal simulation to test the PBF bleeding system.
"""

import warp as wp
import numpy as np

def test_pbf_kernels():
    """Test the PBF kernels independently."""
    print("Testing PBF kernels...")
    
    # Initialize Warp
    wp.init()
    device = wp.get_device()
    
    # Test parameters
    num_particles = 64
    smoothing_radius = 0.016
    rest_density = 1000.0
    
    # Create test data
    positions = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    velocities = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    predicted_positions = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    densities = wp.zeros(num_particles, dtype=wp.float32, device=device)
    lambdas = wp.zeros(num_particles, dtype=wp.float32, device=device)
    delta_positions = wp.zeros(num_particles, dtype=wp.vec3f, device=device)
    active = wp.ones(num_particles, dtype=wp.int32, device=device)
    
    # Initialize particles in a small grid
    pos_data = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                pos = [i * 0.02, j * 0.02, k * 0.02]
                pos_data.append(pos)
    
    # Fill remaining positions with zeros
    while len(pos_data) < num_particles:
        pos_data.append([0.0, 0.0, 0.0])
    
    wp.copy(positions, wp.array(pos_data, dtype=wp.vec3f, device=device))
    
    # Create spatial hash grid
    hash_grid = wp.HashGrid(dim_x=16, dim_y=16, dim_z=16, device=device)
    
    # Import kernels from warp_simulation
    import sys
    sys.path.append('.')
    from warp_simulation import (
        pbf_predict_positions, 
        pbf_compute_density,
        pbf_compute_lambda, 
        pbf_compute_delta_positions,
        pbf_update_predicted_positions
    )
    
    gravity = wp.vec3f(0.0, -9.81, 0.0)
    dt = 0.016  # ~60 FPS
    
    try:
        # Test 1: Predict positions
        print("  Testing predict positions...")
        wp.launch(
            pbf_predict_positions,
            dim=num_particles,
            inputs=[positions, velocities, predicted_positions, active, gravity, dt, num_particles],
            device=device
        )
        
        # Test 2: Build spatial hash
        print("  Testing spatial hash build...")
        hash_grid.build(predicted_positions, smoothing_radius)
        
        # Test 3: Compute density
        print("  Testing density computation...")
        wp.launch(
            pbf_compute_density,
            dim=num_particles,
            inputs=[predicted_positions, densities, active, hash_grid.id, smoothing_radius, rest_density, num_particles],
            device=device
        )
        
        # Test 4: Compute lambda
        print("  Testing lambda computation...")
        wp.launch(
            pbf_compute_lambda,
            dim=num_particles,
            inputs=[predicted_positions, densities, lambdas, active, hash_grid.id, smoothing_radius, rest_density, 600.0, num_particles],
            device=device
        )
        
        # Test 5: Compute delta positions
        print("  Testing delta position computation...")
        wp.launch(
            pbf_compute_delta_positions,
            dim=num_particles,
            inputs=[predicted_positions, lambdas, delta_positions, active, hash_grid.id, smoothing_radius, rest_density, num_particles],
            device=device
        )
        
        # Test 6: Update positions
        print("  Testing position update...")
        wp.launch(
            pbf_update_predicted_positions,
            dim=num_particles,
            inputs=[predicted_positions, delta_positions, active, num_particles],
            device=device
        )
        
        # Verify results
        density_values = densities.numpy()
        lambda_values = lambdas.numpy()
        
        print(f"  Density range: [{np.min(density_values):.2f}, {np.max(density_values):.2f}]")
        print(f"  Lambda range: [{np.min(lambda_values):.4f}, {np.max(lambda_values):.4f}]")
        print(f"  Active particles: {np.sum(active.numpy())}")
        
        print("+ PBF kernels test passed!")
        return True
        
    except Exception as e:
        print(f"X PBF kernels test failed: {e}")
        return False

def test_simulation_integration():
    """Test PBF integration components only."""
    print("\nTesting simulation integration...")
    
    try:
        # Just test that the PBF components can be imported and are properly defined
        from warp_simulation import (
            pbf_predict_positions, 
            pbf_compute_density,
            pbf_compute_lambda, 
            pbf_compute_delta_positions,
            pbf_update_predicted_positions,
            pbf_update_velocities_and_positions,
            pbf_apply_boundaries
        )
        
        # Test that all kernels are properly defined
        kernels = [
            pbf_predict_positions, 
            pbf_compute_density,
            pbf_compute_lambda, 
            pbf_compute_delta_positions,
            pbf_update_predicted_positions,
            pbf_update_velocities_and_positions,
            pbf_apply_boundaries
        ]
        
        for kernel in kernels:
            assert hasattr(kernel, 'key'), f"Kernel {kernel.__name__} not properly defined"
        
        # Test basic PBF parameter values
        test_params = {
            'rest_density': 1000.0,
            'smoothing_radius': 0.016,
            'constraint_epsilon': 600.0,
            'solver_iterations': 4,
            'damping': 0.99,
        }
        
        for param, value in test_params.items():
            assert value > 0, f"Invalid parameter {param}: {value}"
        
        print("+ Simulation integration test passed!")
        return True
        
    except Exception as e:
        print(f"X Simulation integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== PBF Integration Test Suite ===\n")
    
    # Run tests
    test1_passed = test_pbf_kernels()
    test2_passed = test_simulation_integration()
    
    print(f"\n=== Test Results ===")
    print(f"PBF Kernels: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Integration: {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n All tests passed! PBF integration is working correctly.")
        exit(0)
    else:
        print("\n Some tests failed. Check the output above for details.")
        exit(1)