# Shape Recovery Guide for Soft Bodies

## Problem: Liver Not Recovering Full Shape After Collision

If your soft body (liver) is no longer completely flat but doesn't recover its full shape, you need to adjust the **elasticity and volume preservation** parameters.

## Key Parameters Updated (Latest)

### 1. Volume Constraint Stiffness ⭐ CRITICAL
```python
soft_body_relaxation=0.0001  # Reduced from 0.01 → 100x stiffer!
```
This is the most important parameter. Lower values = stiffer volume preservation.

**Guideline:**
- `0.9` (default) - Very soft, will flatten
- `0.01` - Stiff, prevents flattening but may not recover fully
- `0.001` - Very stiff, good recovery
- `0.0001` - Nearly rigid volume, excellent recovery ✅ (current)
- `0.00001` - Ultra-stiff (may cause instability)

### 2. Material Elasticity
```python
young_mod = 5.0e6  # Increased from 1.0e6
poisson_ratio = 0.48  # Increased from 0.45 (more incompressible)
```

**Poisson's Ratio Guide:**
- `0.0-0.2` - Compressible (rubber, cork)
- `0.3-0.4` - Moderately compressible (most materials)
- `0.45-0.49` - Nearly incompressible (soft tissue) ✅
- `0.5` - Fully incompressible (water) - Can cause numerical issues!

For liver-like behavior, stay in the 0.48-0.49 range.

### 3. Damping (Affects Recovery Speed)
```python
k_damp_override = 0.01  # Very low for elastic recovery
```

**Damping Guide:**
- `0.0` - No damping (may oscillate)
- `0.01` - Very low damping, fast recovery ✅ (current)
- `0.1` - Low damping, slower recovery
- `0.5` - Medium damping, will settle faster but less bouncy
- `1.0+` - High damping, may prevent full recovery

### 4. Solver Iterations
```python
iterations=16  # Good balance
```

More iterations = better constraint satisfaction, but slower.
- Minimum: 4
- Recommended: 8-16 ✅
- Maximum useful: 32 (diminishing returns after this)

## Complete Settings Summary

```python
# Material Parameters
young_mod = 5.0e6
poisson_ratio = 0.48
k_damp = 0.01

# Solver Settings
iterations = 16
soft_body_relaxation = 0.0001

# Contact Parameters
soft_contact_ke = 1.0e6
soft_contact_kd = 1.0e3

# Simulation
sim_substeps = 32
```

## Debugging: Monitor Volume Changes

Uncomment lines 229-236 in the example to print volume diagnostics:

```python
if substep == 0 and int(self.sim_time * self.fps) % 60 == 0:
    particles = self.state_0.particle_q.numpy()
    if len(particles) > 0:
        min_coords = particles.min(axis=0)
        max_coords = particles.max(axis=0)
        size = max_coords - min_coords
        volume_proxy = size[0] * size[1] * size[2]
        print(f"Time {self.sim_time:.2f}s: volume_proxy={volume_proxy:.6f}")
```

This will print the bounding box volume every second. Watch for:
- Initial volume (before collision)
- Minimum volume (during collision)
- Final volume (after recovery)

**Good recovery:** Final volume should be close to initial volume (within 5-10%)
**Poor recovery:** Final volume stays much smaller than initial

## Troubleshooting

### Issue: Still doesn't recover fully

**Try in order:**
1. Reduce `soft_body_relaxation` further:
   ```python
   soft_body_relaxation=0.00001
   ```

2. Increase Poisson's ratio (closer to incompressible):
   ```python
   poisson_ratio = 0.49
   ```

3. Reduce damping even more:
   ```python
   k_damp_override = 0.001
   ```

4. Increase Young's modulus:
   ```python
   young_mod = 1.0e7
   ```

5. Try more iterations:
   ```python
   iterations=24
   ```

### Issue: Simulation becomes unstable/jittery

**Try in order:**
1. Increase `soft_body_relaxation` slightly:
   ```python
   soft_body_relaxation=0.001
   ```

2. Add more damping:
   ```python
   k_damp_override = 0.05
   ```

3. Increase substeps:
   ```python
   sim_substeps = 48
   ```

4. Reduce contact stiffness:
   ```python
   soft_contact_ke = 5.0e5
   ```

### Issue: Too slow

**Try in order:**
1. Reduce iterations:
   ```python
   iterations=8
   ```

2. Reduce substeps:
   ```python
   sim_substeps = 24
   ```

3. Enable CUDA graph capture (uncomment line 132):
   ```python
   self.capture()  # Line 132
   ```

## Alternative: Try SemiImplicit Solver

If XPBD still doesn't give good results, try the SemiImplicit solver (line 119):

```python
# Comment out XPBD solver
# self.solver = newton.solvers.SolverXPBD(...)

# Use SemiImplicit instead
self.solver = newton.solvers.SolverSemiImplicit(self.model)
```

The SemiImplicit solver uses FEM directly and may handle volume preservation differently. It's generally slower but can be more physically accurate for some materials.

## Understanding the Physics

### Why Low Relaxation Works

In XPBD, relaxation controls constraint compliance:
```
compliance = relaxation / (dt² * volume)
```

Lower compliance → stiffer constraint → better volume preservation.

The volume constraint tries to maintain `det(F) = 1` where F is the deformation gradient. When relaxation is too high, the constraint is weak and allows volume loss.

### Why Poisson's Ratio Matters

Poisson's ratio (ν) controls how much a material contracts in perpendicular directions when stretched:
- Low ν: Material can compress easily
- High ν: Material is nearly incompressible

For soft tissue like liver, ν ≈ 0.48-0.49 because it's mostly water and nearly incompressible.

### Damping vs. Elasticity

- **Low damping** (0.01): Material recovers shape quickly, may oscillate
- **High damping** (1.0): Material settles quickly but may not fully recover

For elastic recovery, keep damping low!

## Real-World Liver Properties

For reference, actual liver tissue properties:
- Young's modulus: 2-3 kPa (very soft!)
- Poisson's ratio: 0.48-0.49
- Density: ~1060 kg/m³

We use much higher Young's modulus (MPa instead of kPa) for numerical stability in the simulation.

## Quick Test Checklist

After adjusting parameters, verify:
- [ ] Liver falls and hits ground
- [ ] Liver deforms on impact (some compression is ok)
- [ ] Liver doesn't flatten completely
- [ ] Liver recovers most of its shape within 1-2 seconds
- [ ] No jittering or instability
- [ ] Simulation runs at acceptable speed

If all checkboxes pass, your parameters are good! ✅






