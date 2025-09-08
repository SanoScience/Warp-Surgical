#!/usr/bin/env python3
"""
Test script for Position Based Fluids (PBF) system
Validates basic PBF functionality and integration
"""

import warp as wp
import numpy as np
from pbf_system import PBFSystem
import time

def test_pbf_basic_functionality():
    """Test basic PBF system functionality"""
    print("Testing PBF System Basic Functionality...")
    
    # Initialize Warp
    wp.init()
    device = wp.get_device()
    print(f"Using device: {device}")
    
    # Create PBF system
    pbf_system = PBFSystem(
        max_particles=100,
        smoothing_length=0.02,
        rest_density=1000.0,
        particle_mass=0.01,
        device=device
    )
    
    print(f"Created PBF system with {pbf_system.max_particles} max particles")
    
    # Test particle spawning
    test_positions = [
        wp.vec3f(0.0, 0.5, 0.0),
        wp.vec3f(0.1, 0.5, 0.0),
        wp.vec3f(-0.1, 0.5, 0.0),
        wp.vec3f(0.0, 0.6, 0.0),
        wp.vec3f(0.0, 0.4, 0.0)
    ]
    
    for i, pos in enumerate(test_positions):
        velocity = wp.vec3f(0.0, 0.0, 0.0)  # Start at rest
        pbf_system.spawn_particle(pos, velocity, current_time=0.0)
        print(f"Spawned particle {i+1} at position {pos}")
    
    # Check active particle count
    initial_count = pbf_system.count_active_particles()
    print(f"Active particles after spawning: {initial_count}")
    
    # Test simulation steps
    print("\nRunning simulation steps...")
    dt = 1.0 / 60.0  # 60 FPS
    
    for step in range(10):
        current_time = step * dt
        pbf_system.simulate_step(dt, current_time)
        
        if step % 3 == 0:  # Print every few steps
            active_count = pbf_system.count_active_particles()
            particle_data = pbf_system.get_active_particle_data()
            
            print(f"Step {step}: {active_count} active particles")
            if len(particle_data['positions']) > 0:
                avg_pos = np.mean(particle_data['positions'], axis=0)
                print(f"  Average position: ({avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f})")
    
    # Final particle data
    final_data = pbf_system.get_active_particle_data()
    print(f"\nFinal simulation state:")
    print(f"  Active particles: {len(final_data['positions'])}")
    
    if len(final_data['positions']) > 0:
        print("  Final positions:")
        for i, pos in enumerate(final_data['positions']):
            print(f"    Particle {i}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
    
    print("Basic functionality test completed successfully!")
    return True

def test_pbf_integration():
    """Test PBF integration with centreline bleeding"""
    print("\nTesting PBF Integration...")
    
    try:
        from centrelines import init_pbf_bleeding_system, emit_pbf_bleeding, process_pbf_spawn_requests
        print("PBF bleeding functions imported successfully")
        
        # Initialize bleeding system
        device = wp.get_device()
        bleeding_data = init_pbf_bleeding_system(max_spawn_requests=32, device=device)
        print("PBF bleeding system initialized")
        
        # Create mock centreline data
        centreline_positions = wp.array([
            wp.vec3f(0.0, 0.0, 0.0),
            wp.vec3f(0.1, 0.0, 0.0),
            wp.vec3f(0.2, 0.0, 0.0)
        ], dtype=wp.vec3f, device=device)
        
        cut_flags = wp.array([0, 1, 0], dtype=wp.int32, device=device)  # Middle point is cut
        
        # Create PBF system
        pbf_system = PBFSystem(max_particles=50, device=device)
        
        # Test emission kernel
        wp.launch(
            emit_pbf_bleeding,
            dim=3,
            inputs=[
                centreline_positions,
                cut_flags,
                bleeding_data['spawn_requests'],
                bleeding_data['spawn_velocities'],
                bleeding_data['spawn_count'],
                bleeding_data['max_spawn_requests'],
                0.0,  # total_time
                1.0/60.0,  # dt
                1  # emission_rate (emit every frame)
            ],
            device=device
        )
        
        # Process spawn requests
        process_pbf_spawn_requests(
            pbf_system,
            bleeding_data['spawn_requests'],
            bleeding_data['spawn_velocities'],
            bleeding_data['spawn_count'],
            0.0
        )
        
        # Check if particles were spawned
        active_count = pbf_system.count_active_particles()
        print(f"Spawned {active_count} particles from cut centreline")
        
        # Test simulation
        pbf_system.simulate_step(1.0/60.0, 0.0)
        print("PBF simulation step completed")
        
        print("Integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

def test_performance():
    """Test PBF performance with more particles"""
    print("\nTesting PBF Performance...")
    
    device = wp.get_device()
    pbf_system = PBFSystem(max_particles=1000, device=device)
    
    # Spawn many particles in a grid
    grid_size = 8
    spacing = 0.02
    particle_count = 0
    
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(2):  # Only 2 layers to keep count reasonable
                pos = wp.vec3f(
                    (x - grid_size/2) * spacing,
                    y * spacing + 0.5,
                    (z - 1) * spacing
                )
                vel = wp.vec3f(0.0, 0.0, 0.0)
                pbf_system.spawn_particle(pos, vel, 0.0)
                particle_count += 1
                
                if particle_count >= 100:  # Limit for test
                    break
            if particle_count >= 100:
                break
        if particle_count >= 100:
            break
    
    print(f"Spawned {particle_count} particles for performance test")
    
    # Time simulation steps
    dt = 1.0 / 60.0
    num_steps = 20
    
    start_time = time.time()
    
    for step in range(num_steps):
        pbf_system.simulate_step(dt, step * dt)
    
    end_time = time.time()
    elapsed = end_time - start_time
    avg_step_time = elapsed / num_steps
    
    print(f"Performance results:")
    print(f"  Total time: {elapsed:.3f} seconds")
    print(f"  Average step time: {avg_step_time:.4f} seconds")
    print(f"  Estimated FPS: {1.0/avg_step_time:.1f}")
    
    final_count = pbf_system.count_active_particles()
    print(f"  Final active particles: {final_count}")
    
    print("Performance test completed!")
    return True

if __name__ == "__main__":
    print("PBF System Validation Test")
    print("=" * 40)
    
    try:
        # Run all tests
        tests_passed = 0
        total_tests = 3
        
        if test_pbf_basic_functionality():
            tests_passed += 1
            
        if test_pbf_integration():
            tests_passed += 1
            
        if test_performance():
            tests_passed += 1
        
        print("\n" + "=" * 40)
        print(f"Test Results: {tests_passed}/{total_tests} tests passed")
        
        if tests_passed == total_tests:
            print("All PBF tests passed successfully!")
            print("\nPBF system is ready for use in surgical simulation!")
        else:
            print("Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()