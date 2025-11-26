# Setup Summary: IH_DeepReach Implementation

## Completed Steps

### ✅ 1. Created Implementation Guide
- **File:** `Notes/Implementation_Guide.md`
- **Content:** Comprehensive guide detailing all modifications needed
- Includes mathematical formulation, code modifications, training procedure, and troubleshooting

### ✅ 2. Created Directory Structure
```
IH_DeepReach/
├── dynamics/
│   ├── __init__.py          ✅ Created
│   └── dynamics.py          ✅ Copied (needs modification)
├── experiments/
│   ├── __init__.py          ✅ Created  
│   └── experiments.py       ✅ Copied (minor tweaks needed)
├── utils/
│   ├── __init__.py          ✅ Created
│   ├── dataio.py            ✅ Copied (verification needed)
│   ├── diff_operators.py    ✅ Copied (no changes)
│   ├── losses.py            ✅ Copied (needs modification)
│   ├── modules.py           ✅ Copied (no changes)
│   └── error_evaluators.py  ✅ Copied (optional)
├── run_experiment.py        ✅ Copied (needs modification)
└── Notes/
    ├── DeepReach_IH.tex     ✅ Already exists
    ├── Implementation_Guide.md  ✅ Created
    └── Setup_Summary.md     ✅ This file
```

## Next Steps (From Implementation Guide)

### 🔴 Critical Modifications Required (Must Do)

#### 1. **dynamics/dynamics.py** - Add `Dubins3DDiscounted` class

**Location:** End of file (after existing dynamics classes)

**Action:** Add the complete `Dubins3DDiscounted` class as specified in Implementation_Guide.md section 1.2-1.3

**Key points:**
- Inherits from `Dynamics` base class
- Constructor takes `gamma` parameter (discount factor)
- `hamiltonian(state, dvds, value)` method **must include value as 3rd argument**
- Implements optimal control/disturbance for Dubins car with disturbance bounds
- Returns Hamiltonian: `H = p·f - γV`

**Lines to add:** ~150 lines

---

#### 2. **utils/losses.py** - Update `init_brt_hjivi_loss` function

**Location:** Modify the existing `init_brt_hjivi_loss` function

**Action:** Pass `value` to `dynamics.hamiltonian()` call

**Current (wrong):**
```python
ham = dynamics.hamiltonian(state, dvds)
```

**Modified (correct):**
```python
ham = dynamics.hamiltonian(state, dvds, value.squeeze(-1))
```

**Also need to verify:**
- Residual computation: `min{V_t + H, g - V} = 0`
- Terminal loss at `t=T`
- Loss weighting

**Lines to modify:** ~5-10 lines in one function

---

#### 3. **run_experiment.py** - Add `gamma` parameter

**Location:** Argument parser section (around line 94-106)

**Action:** Add argument parsing for `Dubins3DDiscounted` dynamics class

**Code to add:**
```python
# After line ~106, add:
if 'Dubins3DDiscounted' in [cls[0] for cls in inspect.getmembers(dynamics, inspect.isclass)]:
    if orig_opt.dynamics_class == 'Dubins3DDiscounted':
        for param in inspect.signature(dynamics.Dubins3DDiscounted).parameters.keys():
            if param == 'self':
                continue
            param_type = inspect.signature(dynamics.Dubins3DDiscounted).parameters[param].annotation
            if param_type == inspect.Parameter.empty:
                param_type = float
            p.add_argument(f'--{param}', type=param_type, required=True, 
                         help=f'Dubins3DDiscounted parameter: {param}')
```

**Lines to add:** ~10 lines

---

### ⚠️ Verification Needed

#### 4. **utils/dataio.py** - Verify curriculum sampling

**Action:** Check that `ReachabilityDataset` class includes:
- `pretrain` flag for terminal-only sampling
- `counter` for progressive time expansion  
- Time sampling based on curriculum schedule

**If not present:** Need to modify `__getitem__` method to implement curriculum

---

#### 5. **experiments/experiments.py** - Add t=0 extraction

**Action:** In visualization/testing code, ensure we extract `V(x, t=0)` for final results

**Optional:** Can be done after training works

---

## Training Command Template

Once modifications are complete, use:

```bash
# Step 1: Pretrain terminal condition
python run_experiment.py \
    --mode train \
    --dynamics_class Dubins3DDiscounted \
    --gamma 0.7 \
    --L 0.9 \
    --r 0.2 \
    --Cx 0.0 \
    --Cy 0.0 \
    --u1_min 0.05 \
    --u1_max 1.0 \
    --u2_min -1.0 \
    --u2_max 1.0 \
    --d1_min -0.01 \
    --d1_max 0.01 \
    --d2_min -0.01 \
    --d2_max 0.01 \
    --angle_alpha_factor 1.0 \
    --tMin 0.0 \
    --tMax 10.0 \
    --pretrain \
    --pretrain_iters 2000 \
    --num_epochs 5000 \
    --numpoints 65000 \
    --minWith target \
    --experiment_name dubins_ih_pretrain
```

---

## Checklist

- [x] Create directory structure
- [x] Copy base files from original DeepReach
- [x] Create `__init__.py` files
- [x] Write comprehensive implementation guide
- [ ] **Add `Dubins3DDiscounted` class to dynamics.py**
- [ ] **Modify `init_brt_hjivi_loss` in losses.py**
- [ ] **Add gamma parameter to run_experiment.py**
- [ ] Verify curriculum in dataio.py
- [ ] Test pretrain phase
- [ ] Test full training
- [ ] Extract and visualize V(x, t=0)

---

## Key Concept Reminder

**Why time-dependent training?**

The stationary PDE approach (no time) lacks **causal structure** - the network must learn `V(x)` everywhere simultaneously with only the constraint `H(x, ∇V, V) = 0`. This is **numerically ill-conditioned**.

The time-dependent approach adds a **terminal boundary condition** `V(x, T) = g(x)` and the **temporal PDE** `V_t + H = 0`. Information propagates **backward in time** from a known terminal state, providing **stable training dynamics**.

After convergence with large `T`, the slice `V(x, t=0)` approximates the infinite-horizon discounted value function because the exponential discount renders future states less relevant.

**Mathematical justification:** See `Notes/DeepReach_IH.tex` equations (1)-(10).

---

## References

- **Theoretical foundation:** `Notes/DeepReach_IH.tex`
- **Implementation details:** `Notes/Implementation_Guide.md`
- **Original DeepReach code:** `c:\Users\Leixi\Desktop\DeepReach\deepreach\`
- **Stationary version (unstable):** `c:\Users\Leixi\Desktop\DeepReach\DeepReach_Model_Deployment\`

---

**Status:** Ready for code modifications. Start with step 1 (Dubins3DDiscounted class).
