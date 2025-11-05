# Fix for Liver Model Flattening Issue

## Problem
The liver model behaves correctly until it hits the ground, at which point it becomes flat due to excessive volume loss during collision.

## Root Cause
The issue was caused by **insufficient volume preservation** in the XPBD solver. Specifically:

1. **High compliance values**: The default `soft_body_relaxation = 0.9` is very compliant (soft), allowing significant constraint violation
2. **Weak material parameters**: Default `k_mu = 1000.0` and `k_lambda = 1000.0` from USD loading were too low
3. **Insufficient contact stiffness**: `soft_contact_ke = 1.0e4` allowed too much penetration
4. **Too few solver iterations**: Default 2 iterations insufficient for stiff constraints

In XPBD, compliance = 1/stiffness. Higher compliance → softer constraints → more volume loss.

## Solution Applied

### 1. Reduced Soft Body Relaxation (Most Important)
```python
self.solver = newton.solvers.SolverXPBD(
    self.model,
    iterations=4,              # Increased from 2
    soft_body_relaxation=0.01, # Reduced from 0.9 (much stiffer!)
)
```
- Lower relaxation = stiffer volume constraints = better volume preservation
- Changed from 0.9 → 0.01 (90x stiffer)

### 2. Increased Material Stiffness
```python
young_mod = 1.0e6  # Increased from ~1.0e3
poisson_ratio = 0.45  # Nearly incompressible (liver-like)
k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
```
- Higher Young's modulus → more resistant to deformation
- High Poisson's ratio (0.45) → nearly incompressible like soft tissue

### 3. Increased Contact Stiffness
```python
self.model.soft_contact_ke = 1.0e6  # Increased from 1.0e4
self.model.soft_contact_kd = 1.0e3  # Increased from 1.0e2
```
- Prevents excessive penetration into ground plane
- Higher damping for stability

### 4. More Substeps
```python
self.sim_substeps = 32  # Increased from 16
```
- Smaller timestep → more stable with stiff constraints

## How XPBD Volume Constraint Works

In `solve_tetrahedra` kernel (`newton/_src/solvers/xpbd/kernels.py`):

```python
# Two constraint terms:
for term in range(0, num_terms):
    if term == 0:
        # Deviatoric (shape preservation)
        C = tr - 3.0
        compliance = stretching_compliance
    elif term == 1:
        # Volume conservation
        C = wp.determinant(F) - 1.0
        compliance = volume_compliance
```

Both use the same `relaxation` parameter, so lowering it stiffens **both** shape and volume preservation.

## Tuning Guide

If the liver still deforms too much:
1. **Lower `soft_body_relaxation`** further (try 0.001)
2. **Increase `young_mod`** (try 1.0e7)
3. **Increase solver `iterations`** (try 6-8)
4. **Increase `sim_substeps`** (try 48-64)

If the simulation becomes unstable or too stiff:
1. **Increase `soft_body_relaxation`** slightly (try 0.05)
2. **Decrease `young_mod`** (try 5.0e5)
3. **Increase damping**: `k_damp = 0.5`

If there's still penetration into the ground:
1. **Increase `soft_contact_ke`** (try 1.0e7)
2. **Reduce `soft_contact_margin`** in `collide()` call

## Material Parameter Reference

For different tissue types, use these approximate values:

| Tissue Type | Young's Modulus | Poisson's Ratio |
|-------------|----------------|-----------------|
| Liver       | 0.5-3 kPa      | 0.45-0.49      |
| Heart       | 10-50 kPa      | 0.45-0.49      |
| Muscle      | 10-100 kPa     | 0.45-0.49      |
| Fat         | 0.5-2 kPa      | 0.48-0.49      |
| Bone        | 10-20 GPa      | 0.2-0.3        |

Note: Convert kPa to Pa by multiplying by 1000. In code:
```python
young_mod = 3000.0  # 3 kPa = 3000 Pa for liver
```

## Testing Your Changes

Run the example:
```bash
python -m newton.examples basic_soft_body_from_usd --asset /path/to/your/liver.usd
```

Watch for:
- ✅ Liver maintains volume after hitting ground
- ✅ No excessive bouncing
- ✅ Realistic deformation
- ❌ Liver doesn't become flat
- ❌ No jittering or instability

## Advanced: Custom Volume Constraint Stiffness

If you need **independent control** of stretching vs volume constraints, you would need to modify the XPBD solver kernel to accept separate compliance values. This would involve:

1. Adding `volume_relaxation` parameter to `SolverXPBD.__init__()`
2. Passing it separately to `solve_tetrahedra` kernel
3. Using different compliance values for each term

This is more advanced but would allow you to have soft stretching with stiff volume preservation.

## Files Modified

- `newton/examples/basic/example_basic_soft_body_from_usd.py`
  - Reduced `soft_body_relaxation` from 0.9 → 0.01
  - Increased solver `iterations` from 2 → 4
  - Increased `sim_substeps` from 16 → 32
  - Added material parameter override for stiffer liver properties
  - Increased contact stiffness parameters






