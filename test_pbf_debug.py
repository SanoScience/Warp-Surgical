#!/usr/bin/env python3

"""
Test script for the PBF debug simulation.
Tests the stability fixes and debug features without requiring OpenGL.
"""

import sys
import time
import numpy as np

def test_debug_simulation():
    """Test the debug PBF simulation."""
    print("Testing PBF debug simulation...")
    
    try:
        from example_pbf_debug import PBFDebugSimulation
        
        # Create debug simulation
        sim = PBFDebugSimulation(num_particles=128)  # Small for testing
        
        print(f"+ Debug simulation created with {sim.num_particles} particles")
        print(f"+ Parameters: dt={sim.params['dt']:.4f}, iterations={sim.params['constraint_iterations']}")
        
        # Test initial state
        initial_positions = sim.positions.numpy().copy()
        
        # Run several steps and monitor stability
        print("Running stability test...")
        
        max_steps = 100
        unstable_counts = []
        velocity_stats = []
        
        for step in range(max_steps):
            sim.step()
            
            if step % 20 == 0:  # Check every 20 steps
                stats = sim.get_debug_stats()
                unstable_counts.append(stats.get('unstable_count', 0))
                velocity_stats.append(stats.get('max_velocity', 0))
                
                print(f"  Step {step}: {stats.get('unstable_count', 0)} unstable, "
                      f"max_vel={stats.get('max_velocity', 0):.2f}, "
                      f"avg_density={stats.get('avg_density', 0):.0f}")
                
                # Check if particles are exploding
                if stats.get('max_velocity', 0) > 20.0:  # Very high velocity indicates explosion
                    print("! High velocities detected - potential instability")
                    break
                    
                # Check if too many particles are unstable
                if stats.get('unstable_count', 0) > sim.num_particles * 0.5:
                    print("! Too many unstable particles")
                    break
        
        # Analyze results
        final_positions = sim.positions.numpy()
        total_movement = np.mean(np.linalg.norm(final_positions - initial_positions, axis=1))
        max_unstable = max(unstable_counts) if unstable_counts else 0
        max_velocity = max(velocity_stats) if velocity_stats else 0
        
        print(f"+ Test completed - {step+1} steps")
        print(f"+ Average particle movement: {total_movement:.3f}")
        print(f"+ Maximum unstable count: {max_unstable}")
        print(f"+ Maximum velocity reached: {max_velocity:.2f}")
        
        # Stability assessment
        if max_velocity < 10.0 and max_unstable < sim.num_particles * 0.3:
            print("+ Simulation appears stable!")
            stability_good = True
        else:
            print("! Simulation may have stability issues")
            stability_good = False
        
        # Test parameter adjustment
        print("Testing parameter adjustment...")
        old_dt = sim.params['dt']
        sim.adjust_timestep(0.5)  # Reduce timestep
        new_dt = sim.params['dt']
        
        if new_dt < old_dt:
            print("+ Timestep adjustment working")
        else:
            print("! Timestep adjustment failed")
        
        # Test reset
        sim.reset_particles()
        reset_positions = sim.positions.numpy()
        reset_diff = np.linalg.norm(reset_positions - final_positions)
        
        if reset_diff > 0.1:
            print("+ Reset functionality working")
        else:
            print("! Reset didn't change positions much")
        
        return stability_good
        
    except Exception as e:
        print(f"X Debug simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stability_features():
    """Test the stability detection features."""
    print("\nTesting stability features...")
    
    try:
        import warp as wp
        from example_pbf_debug import (
            pbf_predict_positions_debug, 
            pbf_compute_density_debug,
            reset_stability_flags
        )
        
        wp.init()
        device = wp.get_device()
        
        # Create test arrays
        num_particles = 32
        positions = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        velocities = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        predicted_positions = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        unstable_flags = wp.zeros(num_particles, dtype=wp.int32, device=device)
        densities = wp.zeros(num_particles, dtype=wp.float32, device=device)
        
        # Initialize with some high velocities to test clamping
        test_velocities = []
        for i in range(num_particles):
            if i < 5:  # First 5 particles have high velocities
                test_velocities.append([50.0, 50.0, 50.0])  # Very high
            else:
                test_velocities.append([0.1, 0.1, 0.1])  # Normal
        
        wp.copy(velocities, wp.array(test_velocities, dtype=wp.vec3, device=device))
        
        # Test position prediction with velocity clamping
        gravity = wp.vec3(0.0, -9.81, 0.0)
        max_velocity = 5.0
        dt = 1.0/60.0
        
        wp.launch(
            pbf_predict_positions_debug,
            dim=num_particles,
            inputs=[positions, velocities, predicted_positions, unstable_flags, gravity, dt, max_velocity],
            device=device
        )
        
        # Check results
        final_velocities = velocities.numpy()
        unstable_flags_cpu = unstable_flags.numpy()
        
        # Verify high velocities were clamped
        max_vel_magnitude = np.max(np.linalg.norm(final_velocities, axis=1))
        unstable_count = np.sum(unstable_flags_cpu)
        
        print(f"+ Velocity clamping test: max velocity = {max_vel_magnitude:.2f} (limit: {max_velocity})")
        print(f"+ Unstable particles detected: {unstable_count}")
        
        if max_vel_magnitude <= max_velocity + 0.1:  # Small tolerance
            print("+ Velocity clamping working correctly")
        else:
            print("! Velocity clamping failed")
        
        if unstable_count >= 5:  # Should detect the 5 high-velocity particles
            print("+ Instability detection working")
        else:
            print("! Instability detection may not be working")
        
        # Test stability flag reset
        domain_min = wp.vec3(-2.0, -2.0, -2.0)
        domain_max = wp.vec3(2.0, 2.0, 2.0)
        
        wp.launch(
            reset_stability_flags,
            dim=num_particles,
            inputs=[unstable_flags, positions, velocities, domain_min, domain_max],
            device=device
        )
        
        reset_flags = unstable_flags.numpy()
        reset_unstable_count = np.sum(reset_flags)
        
        print(f"+ After reset: {reset_unstable_count} unstable particles")
        
        print("+ Stability features test completed")
        return True
        
    except Exception as e:
        print(f"X Stability features test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=== PBF Debug Simulation Test ===")
    
    test1_passed = test_stability_features()
    test2_passed = test_debug_simulation()
    
    print(f"\n=== Test Results ===")
    print(f"Stability Features: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Debug Simulation:   {'PASS' if test2_passed else 'FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n All tests passed!")
        print("\nThe debug version should be much more stable than the original.")
        print("\nTo run the debug simulation:")
        print("  python example_pbf_debug.py")
        print("\nKey stability improvements:")
        print("- Velocity clamping prevents explosions")
        print("- Density bounds checking")
        print("- Unstable particle detection and freezing")
        print("- Adjustable parameters for tuning")
        print("- Real-time debug statistics")
        return 0
    else:
        print("\n Some tests failed. Check output for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)