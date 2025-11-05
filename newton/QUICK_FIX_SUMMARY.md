# Quick Fix Summary: Liver Model Flattening

## The Problem
Liver model becomes flat when hitting the ground.

## The Root Cause
XPBD volume constraints were too soft (high compliance = low stiffness).

## The Solution (3 Key Changes)

### 1. Lower `soft_body_relaxation` ⭐ MOST IMPORTANT
```python
self.solver = newton.solvers.SolverXPBD(
    self.model,
    soft_body_relaxation=0.01,  # Changed from default 0.9
)
```

### 2. Stiffer Material Parameters
```python
young_mod = 1.0e6  # Increased from default ~1.0e3
poisson_ratio = 0.45  # Nearly incompressible (liver-like)
```

### 3. Stiffer Contact
```python
self.model.soft_contact_ke = 1.0e6  # Increased from 1.0e4
```

## Quick Test
```bash
python -m newton.examples basic_soft_body_from_usd --asset /path/to/liver.usd
```

## Tuning If Needed

**Still flattening?**
- Decrease `soft_body_relaxation` → try 0.001
- Increase `young_mod` → try 1.0e7
- Increase `iterations` → try 6-8

**Too stiff/unstable?**
- Increase `soft_body_relaxation` → try 0.05-0.1
- Decrease `young_mod` → try 5.0e5
- Add damping: `k_damp = 0.5`

## Understanding the Parameters

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `soft_body_relaxation` | Stiffer, preserves volume | Softer, allows deformation |
| `young_mod` | More elastic | More rigid |
| `iterations` | Faster, less accurate | Slower, more accurate |
| `soft_contact_ke` | More penetration | Less penetration |

## Files Changed
- ✅ `newton/examples/basic/example_basic_soft_body_from_usd.py`

See `LIVER_FLATTENING_FIX.md` for detailed explanation.






