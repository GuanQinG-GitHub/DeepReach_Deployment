# Quick Modification Checklist

## Files Copied ✅
All necessary files have been copied from `deepreach/` to `IH_DeepReach/`

## Files Requiring Modification

### 1. ⚠️ `dynamics/dynamics.py` - CRITICAL
**What:** Add new `Dubins3DDiscounted` class at end of file  
**Where:** After line ~800 (after all existing dynamics classes)  
**How:** Copy code from Implementation_Guide.md Section 1.2-1.3  
**Why:** Implements the discounted infinite-horizon dynamics with γ parameter  

**Key signature change:**
```python
def hamiltonian(self, state, dvds, value):  # NOTE: value is 3rd argument!
    # ... compute optimal u, d
    ham = p·f - self.gamma * value  # Include discount term
    return ham
```

---

### 2. ⚠️ `utils/losses.py` - CRITICAL  
**What:** Modify `init_brt_hjivi_loss` function  
**Where:** Line ~20-50 (the main loss function)  
**How:** Change hamiltonian call from 2 args to 3 args  

**Before:**
```python
ham = dynamics.hamiltonian(state, dvds)
```

**After:**
```python
ham = dynamics.hamiltonian(state, dvds, value.squeeze(-1))
```

**Also verify:** Residual = min{V_t + H, g - V}

---

### 3. ⚠️ `run_experiment.py` - CRITICAL
**What:** Add argument parsing for Dubins3DDiscounted parameters  
**Where:** After line ~106 (in the dynamics argument parsing section)  
**How:** Add this block:

```python
if orig_opt.dynamics_class == 'Dubins3DDiscounted':
    p.add_argument('--gamma', type=float, required=True, help='Discount factor')
    p.add_argument('--L', type=float, default=0.9, help='State bound')
    p.add_argument('--r', type=float, default=0.2, help='Obstacle radius')  
    p.add_argument('--Cx', type=float, default=0.0, help='Obstacle x')
    p.add_argument('--Cy', type=float, default=0.0, help='Obstacle y')
    p.add_argument('--u1_min', type=float, default=0.05)
    p.add_argument('--u1_max', type=float, default=1.0)
    p.add_argument('--u2_min', type=float, default=-1.0)
    p.add_argument('--u2_max', type=float, default=1.0)
    p.add_argument('--d1_min', type=float, default=-0.01)
    p.add_argument('--d1_max', type=float, default=0.01)
    p.add_argument('--d2_min', type=float, default=-0.01)
    p.add_argument('--d2_max', type=float, default=0.01)
    p.add_argument('--angle_alpha_factor', type=float, default=1.0)
```

---

### 4. ℹ️ `utils/dataio.py` - VERIFY ONLY
**What:** Check curriculum sampling exists  
**Action:** Open file and verify `ReachabilityDataset` has:
- `pretrain` parameter in `__init__`
- Time sampling that respects curriculum (backward from tMax)
- Counter for progressive time expansion

**If missing:** Needs modification (see guide)

---

## Testing After Modifications

### Quick syntax check:
```bash
python -c "from dynamics import dynamics; print('Import OK')"
```

### Pretrain test (short):
```bash
python run_experiment.py --mode train --dynamics_class Dubins3DDiscounted \
    --gamma 0.7 --tMin 0.0 --tMax 10.0 --pretrain --num_epochs 100 \
    --numpoints 1000 --minWith target --experiment_name test_pretrain
```

---

## Priority Order

1. **FIRST:** Add `Dubins3DDiscounted` to `dynamics/dynamics.py`  
2. **SECOND:** Modify `hamiltonian` call in `utils/losses.py`  
3. **THIRD:** Add parameters to `run_experiment.py`  
4. **FOURTH:** Test with small pretrain run  
5. **FIFTH:** Run full training

---

## Expected File Sizes After Modification

- `dynamics/dynamics.py`: ~1350 lines (was ~1176, add ~170)
- `utils/losses.py`: ~2500 lines (was ~2476, modify ~10)  
- `run_experiment.py`: ~220 lines (was ~205, add ~15)

---

## Common Issues & Fixes

**Issue:** `TypeError: hamiltonian() takes 3 positional arguments but 4 were given`  
**Fix:** Check that new `Dubins3DDiscounted.hamiltonian` has signature `(self, state, dvds, value)`

**Issue:** `ImportError: cannot import name 'Dubins3DDiscounted'`  
**Fix:** Check `dynamics/__init__.py` imports the class

**Issue:** `KeyError: 'gamma'`  
**Fix:** Add `--gamma` to run command

---

## Documentation Files

- 📘 `Implementation_Guide.md` - Full technical details
- 📋 `Setup_Summary.md` - What's been done  
- ✅ `Modification_Checklist.md` - This file (quick reference)
- 📝 `DeepReach_IH.tex` - Mathematical theory

**Start here:** Read Implementation_Guide.md Section 1.2 for the complete `Dubins3DDiscounted` code.
