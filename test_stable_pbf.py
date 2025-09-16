#!/usr/bin/env python3

"""
Quick test for the stable PBF simulation.
"""

import sys
import time
import numpy as np

def test_stable_simulation():
    """Test the stable PBF simulation for explosions."""
    print("Testing stable PBF simulation...")
    
    try:
        from example_pbf_stable import StablePBFSimulation
        
        # Create stable simulation
        sim = StablePBFSimulation(num_particles=256)
        
        print(f"+ Stable simulation created with {sim.num_particles} particles")
        print(f"+ Parameters: dt={sim.params['dt']:.4f}s, iterations={sim.params['constraint_iterations']}")
        print(f"+ Max velocity limit: {sim.params['max_velocity']}")
        
        # Test initial state
        initial_positions = sim.positions.numpy().copy()
        
        # Run simulation and monitor for explosions
        print("Running explosion test...")
        
        max_steps = 200  # Run for ~1.7 seconds at 120 FPS
        max_velocities = []
        position_ranges = []
        
        for step in range(max_steps):
            sim.step()
            
            if step % 40 == 0:  # Check every 1/3 second
                positions = sim.positions.numpy()
                velocities = sim.velocities.numpy()
                
                vel_magnitudes = np.linalg.norm(velocities, axis=1)
                max_vel = np.max(vel_magnitudes)
                avg_vel = np.mean(vel_magnitudes)
                
                pos_range = np.ptp(positions, axis=0)  # Range in each dimension
                max_pos_range = np.max(pos_range)
                
                max_velocities.append(max_vel)
                position_ranges.append(max_pos_range)
                
                print(f"  Step {step}: max_vel={max_vel:.2f}, avg_vel={avg_vel:.2f}, pos_range={max_pos_range:.2f}")
                
                # Check for explosion indicators
                if max_vel > 50.0:  # Very high velocity
                    print("! Explosion detected - high velocities!")
                    break
                    
                if max_pos_range > 20.0:  # Particles spread very far
                    print("! Explosion detected - particles scattered!")
                    break
                    
                # Check for NaN or infinite values
                if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                    print("! NaN/Inf values detected!")
                    break
                    
                if np.any(np.isnan(velocities)) or np.any(np.isinf(velocities)):
                    print("! NaN/Inf velocities detected!")
                    break
        
        # Analyze results
        final_positions = sim.positions.numpy()
        final_velocities = sim.velocities.numpy()
        
        max_velocity_reached = max(max_velocities) if max_velocities else 0
        max_spread = max(position_ranges) if position_ranges else 0
        
        total_movement = np.mean(np.linalg.norm(final_positions - initial_positions, axis=1))
        final_max_vel = np.max(np.linalg.norm(final_velocities, axis=1))
        
        print(f"\n+ Test completed - {step+1} steps")
        print(f"+ Maximum velocity reached: {max_velocity_reached:.2f}")
        print(f"+ Maximum particle spread: {max_spread:.2f}")
        print(f"+ Final maximum velocity: {final_max_vel:.2f}")
        print(f"+ Average particle movement: {total_movement:.3f}")
        
        # Stability assessment
        if (max_velocity_reached < 15.0 and  # Reasonable velocity
            max_spread < 8.0 and             # Particles didn't scatter  
            final_max_vel < 10.0 and         # Final velocities reasonable
            not np.any(np.isnan(final_positions)) and  # No NaN values
            not np.any(np.isnan(final_velocities))):   # No NaN velocities
            
            print("+ SIMULATION IS STABLE! No explosion detected.")
            return True
        else:
            print("! Simulation may have stability issues")
            return False
        
    except Exception as e:
        print(f"X Stable simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=== Stable PBF Simulation Test ===")
    
    success = test_stable_simulation()
    
    if success:
        print("\n SUCCESS! The stable version should work without explosions.")
        print("\nTo run the stable simulation:")
        print("  python example_pbf_stable.py")
        print("\nKey improvements:")
        print("- Smaller timestep (1/120s)")
        print("- More constraint iterations (6)")
        print("- Velocity clamping (max 8.0 m/s)")
        print("- Better particle spacing")
        print("- Higher damping (0.96)")
        print("- Conservative parameters")
        return 0
    else:
        print("\n Still having issues. Try the debug version:")
        print("  python example_pbf_debug.py")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)