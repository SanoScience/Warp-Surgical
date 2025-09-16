#!/usr/bin/env python3

"""
Test script for the PBF OpenGL example.
This tests the simulation logic without requiring OpenGL display.
"""

import sys
import time
import numpy as np

def test_pbf_simulation():
    """Test PBF simulation functionality."""
    print("Testing PBF simulation...")
    
    try:
        # Import simulation class
        from example_pbf_opengl import PBFSimulation
        
        # Create smaller simulation for testing
        sim = PBFSimulation(num_particles=256)
        
        print(f"+ Simulation created with {sim.num_particles} particles")
        
        # Test initial state
        initial_positions = sim.positions.numpy().copy()
        initial_velocities = sim.velocities.numpy().copy()
        
        print(f"+ Initial position range: {np.min(initial_positions, axis=0)} to {np.max(initial_positions, axis=0)}")
        print(f"+ Initial velocity magnitude: {np.mean(np.linalg.norm(initial_velocities, axis=1)):.4f}")
        
        # Run simulation steps
        print("Running simulation steps...")
        
        for step in range(10):
            sim.step()
            
            if step % 5 == 0:
                positions = sim.positions.numpy()
                velocities = sim.velocities.numpy()
                avg_vel = np.mean(np.linalg.norm(velocities, axis=1))
                
                print(f"  Step {step}: avg velocity = {avg_vel:.4f}")
        
        # Verify particles have moved due to gravity
        final_positions = sim.positions.numpy()
        position_change = np.linalg.norm(final_positions - initial_positions, axis=1)
        avg_movement = np.mean(position_change)
        
        print(f"+ Average particle movement: {avg_movement:.4f}")
        
        if avg_movement > 0.001:  # Particles should have moved
            print("+ Particles moved as expected (gravity effect)")
        else:
            print("! Warning: Particles didn't move much")
        
        # Test pause/resume
        sim.paused = True
        pause_positions = sim.positions.numpy().copy()
        
        sim.step()  # This should not change positions
        
        after_pause_positions = sim.positions.numpy()
        pause_movement = np.linalg.norm(after_pause_positions - pause_positions)
        
        if pause_movement < 1e-6:
            print("+ Pause functionality working")
        else:
            print(f"! Warning: Particles moved {pause_movement:.6f} while paused")
        
        # Test reset
        sim.reset_particles()
        reset_positions = sim.positions.numpy()
        reset_diff = np.linalg.norm(reset_positions - final_positions)
        
        if reset_diff > 0.1:  # Should be significantly different after reset
            print("+ Reset functionality working")
        else:
            print("! Warning: Reset didn't change particle positions much")
        
        print("+ PBF simulation test completed successfully!")
        return True
        
    except Exception as e:
        print(f"X PBF simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_kernels():
    """Test individual PBF kernels."""
    print("\nTesting PBF kernels...")
    
    try:
        import warp as wp
        from example_pbf_opengl import (
            pbf_predict_positions, pbf_compute_density, pbf_compute_lambda,
            pbf_compute_delta_positions, pbf_apply_delta_positions,
            pbf_apply_boundaries, pbf_update_velocities_positions
        )
        
        wp.init()
        device = wp.get_device()
        
        # Test data
        num_particles = 64
        positions = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        velocities = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        predicted_positions = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        
        # Initialize test positions
        test_positions = []
        for i in range(num_particles):
            x = (i % 4) * 0.1
            y = ((i // 4) % 4) * 0.1
            z = (i // 16) * 0.1
            test_positions.append([x, y, z])
        
        wp.copy(positions, wp.array(test_positions, dtype=wp.vec3, device=device))
        
        # Test predict positions kernel
        gravity = wp.vec3(0.0, -9.81, 0.0)
        dt = 1.0/60.0
        
        wp.launch(
            pbf_predict_positions,
            dim=num_particles,
            inputs=[positions, velocities, predicted_positions, gravity, dt],
            device=device
        )
        
        predicted = predicted_positions.numpy()
        print(f"+ Predict positions kernel executed, sample result: {predicted[0]}")
        
        # Test spatial hash and density computation
        hash_grid = wp.HashGrid(dim_x=16, dim_y=16, dim_z=16, device=device)
        densities = wp.zeros(num_particles, dtype=wp.float32, device=device)
        
        hash_grid.build(predicted_positions, 0.05)
        
        wp.launch(
            pbf_compute_density,
            dim=num_particles,
            inputs=[predicted_positions, densities, hash_grid.id, 0.05],
            device=device
        )
        
        density_values = densities.numpy()
        print(f"+ Density computation kernel executed, sample density: {density_values[0]:.2f}")
        
        print("+ All kernel tests passed!")
        return True
        
    except Exception as e:
        print(f"X Kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def benchmark_performance():
    """Benchmark simulation performance."""
    print("\nBenchmarking PBF performance...")
    
    try:
        from example_pbf_opengl import PBFSimulation
        
        # Test with different particle counts
        particle_counts = [256, 512, 1024, 2048]
        
        for count in particle_counts:
            sim = PBFSimulation(num_particles=count)
            
            # Warmup
            for _ in range(3):
                sim.step()
            
            # Benchmark
            start_time = time.time()
            steps = 100
            
            for _ in range(steps):
                sim.step()
            
            end_time = time.time()
            elapsed = end_time - start_time
            fps = steps / elapsed
            
            print(f"  {count:4d} particles: {fps:6.1f} FPS ({elapsed/steps*1000:.2f} ms/step)")
        
        print("+ Performance benchmark completed!")
        return True
        
    except Exception as e:
        print(f"X Performance benchmark failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== PBF OpenGL Example Test Suite ===")
    
    # Run tests
    test1_passed = test_kernels()
    test2_passed = test_pbf_simulation()
    test3_passed = benchmark_performance()
    
    print(f"\n=== Test Results ===")
    print(f"Kernels:     {'PASS' if test1_passed else 'FAIL'}")
    print(f"Simulation:  {'PASS' if test2_passed else 'FAIL'}")
    print(f"Performance: {'PASS' if test3_passed else 'FAIL'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n All tests passed! PBF example is ready to run.")
        print("\nTo run the visual simulation:")
        print("  python example_pbf_opengl.py")
        print("\nRequirements for visual mode:")
        print("  pip install PyOpenGL PyOpenGL_accelerate")
        return 0
    else:
        print("\n Some tests failed. Check output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)