# Implementation Guide: Infinite-Horizon Discounted Viability with Time-Dependent Training

## Overview

This guide details how to adapt the **original DeepReach framework** to learn the **infinite-horizon discounted viability kernel** for the 3D Dubins car, using the stable backward-in-time curriculum training approach.

### Problem Context

**Current situation:**
- The stationary DeepReach implementation in `DeepReach_Model_Deployment` (without time dependence) is theoretically correct but experiences **unstable training**.
- The original DeepReach uses time-dependent training with curriculum learning, which provides **stable propagation** of the terminal condition.

**Solution:**
- Reintroduce **time dependence** during training (input: `[t, x1, x2, x3]`)
- Keep the **discount factor γ** in the Hamiltonian
- Use **backward-in-time curriculum** from the original DeepReach
- After training, extract `V(x, t=0)` as the infinite-horizon approximation

### Mathematical Formulation

#### Time-Dependent Discounted HJ-VI

For `t ∈ [0, T]`, we solve:

```
min{ V_t + H(x, ∇_x V, V),  g(x) - V(x,t) } = 0
```

with terminal condition `V(x, T) = g(x)`.

#### Hamiltonian for 3D Dubins Car with Disturbance

```
H = min_u max_d [ p·f(x,u,d) - γV ]
```

where:
- `f(x,u,d) = [u₁cos(x₃) + d₁, u₁sin(x₃) + d₂, u₂]`
- `p = ∇_x V = [p₁, p₂, p₃]`
- `γ` is the discount factor
- Control bounds: `u₁ ∈ [0.05, 1]`, `u₂ ∈ [-1, 1]`
- Disturbance bounds: `d₁, d₂ ∈ [-0.01, 0.01]`

#### Boundary Function

```
g(x) = max{ |x₁| - L,  |x₂| - L,  r² - (x₁-Cₓ)² - (x₂-Cᵧ)² }
```

Safe region: `{x : g(x) ≤ 0}`

---

## Implementation Strategy

### Phase 1: Setup & File Organization

1. **Copy necessary files from original DeepReach** to `IH_DeepReach/` folder
2. **Modify** the copied files to support discounted infinite-horizon learning
3. **Keep** 3D Dubins car implementation only (remove other dynamics)

### Phase 2: Modifications Required

We'll modify the following components:

| Component | File | Modifications |
|-----------|------|---------------|
| Dynamics | `dynamics/dynamics.py` | Add discount factor γ; modify Hamiltonian to include `-γV` term |
| Loss Function | `utils/losses.py` | Update residual computation for time-dependent discounted HJ-VI |
| Dataset | `utils/dataio.py` | Time-dependent sampling with curriculum schedule |
| Training Script | `run_experiment.py` | Add γ parameter; configure curriculum learning |

---

## Detailed Modification Checklist

### 1. Dynamics File (`dynamics/dynamics.py`)

**File to create:** `IH_DeepReach/dynamics/dynamics.py`

**Base:** Copy from `deepreach/dynamics/dynamics.py`

**Modifications:**

#### 1.1 Base `Dynamics` Class
- **Keep** the base class structure (time-dependent input/output conversions)
- **Ensure** `input_dim = state_dim + 1` (to include time)

#### 1.2 Create `Dubins3DDiscounted` Class

Add a new class for the discounted 3D Dubins car:

```python
class Dubins3DDiscounted(Dynamics):
    def __init__(self, 
                 gamma: float,           # NEW: discount factor
                 L: float,               # state bound
                 r: float,               # obstacle radius
                 Cx: float,              # obstacle center x
                 Cy: float,              # obstacle center y
                 u1_min: float,          # control bounds
                 u1_max: float,
                 u2_min: float,
                 u2_max: float,
                 d1_min: float,          # disturbance bounds
                 d1_max: float,
                 d2_min: float,
                 d2_max: float,
                 angle_alpha_factor: float):
        
        self.gamma = gamma              # NEW: store discount factor
        self.L = L
        self.r = r
        self.Cx = Cx
        self.Cy = Cy
        self.u_min = torch.tensor([u1_min, u2_min])
        self.u_max = torch.tensor([u1_max, u2_max])
        self.d_min = torch.tensor([d1_min, d2_min])
        self.d_max = torch.tensor([d1_max, d2_max])
        
        super().__init__(
            loss_type='brt_hjivi',      # Use standard BRT loss type
            set_mode='avoid',           # Viability = avoid unsafe set
            state_dim=3,                # [x1, x2, x3]
            input_dim=4,                # [t, x1, x2, x3]
            control_dim=2,              # [u1, u2]
            disturbance_dim=2,          # [d1, d2]
            state_mean=[0, 0, 0],
            state_var=[L, L, angle_alpha_factor * math.pi],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepreach_model="exact"     # Use exact DeepReach model
        )
```

#### 1.3 Implement Required Methods

**a) `state_test_range()`**
```python
def state_test_range(self):
    return [
        [-self.L - 0.1, self.L + 0.1],  # x1
        [-self.L - 0.1, self.L + 0.1],  # x2
        [-math.pi, math.pi],            # x3 (theta)
    ]
```

**b) `boundary_fn(state)` - obstacle/constraints**
```python
def boundary_fn(self, state):
    """
    g(x) = max{|x1|-L, |x2|-L, r² - dist²}
    Safe region: g(x) ≤ 0
    """
    x1 = state[..., 0]
    x2 = state[..., 1]
    
    # Box constraints
    g1 = torch.abs(x1) - self.L
    g2 = torch.abs(x2) - self.L
    
    # Circular obstacle (inside is unsafe)
    dist_sq = (x1 - self.Cx)**2 + (x2 - self.Cy)**2
    g3 = self.r**2 - dist_sq
    
    return torch.maximum(torch.maximum(g1, g2), g3)
```

**c) `hamiltonian(state, dvds, value)` - **KEY MODIFICATION**
```python
def hamiltonian(self, state, dvds, value):
    """
    Computes H = min_u max_d [p·f(x,u,d) - γV]
    
    Args:
        state: [batch, 3] - (x1, x2, x3)
        dvds: [batch, 3] - spatial gradient (p1, p2, p3)
        value: [batch] or [batch, 1] - V(x,t)
    
    Returns:
        ham: [batch] - Hamiltonian value
    """
    x3 = state[..., 2]
    p1 = dvds[..., 0]
    p2 = dvds[..., 1]
    p3 = dvds[..., 2]
    
    # Optimal control (minimize)
    det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
    u1 = torch.where(det1 > 0, self.u_min[0], self.u_max[0])
    u2 = torch.where(p3 > 0, self.u_min[1], self.u_max[1])
    
    # Optimal disturbance (maximize)
    d1 = torch.where(p1 > 0, self.d_max[0], self.d_min[0])
    d2 = torch.where(p2 > 0, self.d_max[1], self.d_min[1])
    
    # Hamiltonian: p·f - γV
    # Ensure value is [batch] shape
    if value.dim() > 1:
        value = value.squeeze(-1)
    
    ham = p1 * (u1 * torch.cos(x3) + d1) + \
          p2 * (u1 * torch.sin(x3) + d2) + \
          p3 * u2 - \
          self.gamma * value  # NEW: discount term
          
    return ham
```

**d) Other required methods**
```python
def equivalent_wrapped_state(self, state):
    wrapped_state = torch.clone(state)
    wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
    return wrapped_state

def dsdt(self, state, control, disturbance):
    """Dynamics for testing/simulation"""
    dsdt = torch.zeros_like(state)
    dsdt[..., 0] = control[..., 0] * torch.cos(state[..., 2]) + disturbance[..., 0]
    dsdt[..., 1] = control[..., 0] * torch.sin(state[..., 2]) + disturbance[..., 1]
    dsdt[..., 2] = control[..., 1]
    return dsdt

def optimal_control(self, state, dvds):
    """Return optimal control for simulation"""
    x3 = state[..., 2]
    p1 = dvds[..., 0]
    p2 = dvds[..., 1]
    p3 = dvds[..., 2]
    
    det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
    u1 = torch.where(det1 > 0, self.u_min[0], self.u_max[0])
    u2 = torch.where(p3 > 0, self.u_min[1], self.u_max[1])
    
    return torch.stack([u1, u2], dim=-1)

def optimal_disturbance(self, state, dvds):
    """Return optimal disturbance for simulation"""
    p1 = dvds[..., 0]
    p2 = dvds[..., 1]
    
    d1 = torch.where(p1 > 0, self.d_max[0], self.d_min[0])
    d2 = torch.where(p2 > 0, self.d_max[1], self.d_min[1])
    
    return torch.stack([d1, d2], dim=-1)

def plot_config(self):
    return {
        'state_slices': [0.0, 0.0, 0.0],
        'state_labels': ['x', 'y', r'$\theta$'],
        'x_axis_idx': 0,
        'y_axis_idx': 1,
        'z_axis_idx': 2,
    }

def sample_target_state(self, num_samples):
    # Not needed for viability
    raise NotImplementedError

def cost_fn(self, state_traj):
    # For testing: min_t g(x(t))
    return torch.min(self.boundary_fn(state_traj), dim=-1).values
```

---

### 2. Loss File (`utils/losses.py`)

**File to create:** `IH_DeepReach/utils/losses.py`

**Base:** Copy from `deepreach/utils/losses.py`

**Modifications:**

#### 2.1 Update `init_brt_hjivi_loss` function

The key change is passing `value` to the Hamiltonian:

```python
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(model_results, gt):
        """
        Computes the BRT HJ-VI loss for discounted infinite-horizon.
        
        Residual: min{ V_t + H(x, ∇_x V, V),  g(x) - V(x,t) } = 0
        """
        # Extract from model_results
        coords = model_results['coords']           # [batch, 4] = [t, x1, x2, x3]
        value = model_results['model_out']         # [batch, 1]
        dvdt = model_results['dvdt']               # [batch, 1]
        dvds = model_results['dvds']               # [batch, 3]
        
        # Get state only (remove time)
        state = coords[..., 1:]                    # [batch, 3]
        
        # Compute Hamiltonian - PASS VALUE HERE
        ham = dynamics.hamiltonian(state, dvds, value.squeeze(-1))  # NEW: pass value
        
        # Boundary function
        g_x = dynamics.boundary_fn(state)          # [batch]
        
        # HJ-VI residual
        # pde_term = V_t + H
        pde_term = dvdt.squeeze(-1) + ham
        
        # boundary_term = g(x) - V
        boundary_term = g_x - value.squeeze(-1)
        
        # Residual = min{pde_term, boundary_term}
        residual = torch.minimum(pde_term, boundary_term)
        
        # Loss components
        # 1. HJ-VI residual loss (should be zero everywhere)
        hjivi_loss = torch.abs(residual).mean()
        
        # 2. Dirichlet boundary loss at t=T (terminal condition)
        # Filter coords where t is close to T (last time slice)
        time = coords[..., 0]
        T = time.max()
        terminal_mask = (time > T - 0.01)  # Points near t=T
        
        if terminal_mask.any():
            terminal_value = value[terminal_mask]
            terminal_state = state[terminal_mask]
            terminal_g = dynamics.boundary_fn(terminal_state)
            dirichlet_loss = torch.abs(terminal_value.squeeze(-1) - terminal_g).mean()
        else:
            dirichlet_loss = torch.tensor(0.0, device=value.device)
        
        # Total loss
        total_loss = hjivi_loss + dirichlet_loss / dirichlet_loss_divisor
        
        return {
            'loss': total_loss,
            'hjivi_loss': hjivi_loss,
            'dirichlet_loss': dirichlet_loss,
        }
    
    return brt_hjivi_loss
```

**Note:** The signature of `dynamics.hamiltonian()` now includes `value` as the third argument.

---

### 3. Dataset File (`utils/dataio.py`)

**File to create:** `IH_DeepReach/utils/dataio.py`

**Base:** Copy from `deepreach/utils/dataio.py`

**Modifications:**

The original DeepReach dataset `ReachabilityDataset` already implements curriculum learning with time-dependent sampling. We just need to ensure it's configured correctly.

#### 3.1 Verify `ReachabilityDataset` class

Check that it includes:
- `tMin`, `tMax` parameters for time range
- `counter` for curriculum progression
- `pretrain` flag for terminal-only training
- `num_src_samples` for number of spatial samples

The curriculum logic should look like:

```python
class ReachabilityDataset:
    def __init__(self, dynamics, tMin, tMax, num_src_samples, pretrain=False, ...):
        self.dynamics = dynamics
        self.tMin = tMin
        self.tMax = tMax
        self.num_src_samples = num_src_samples
        self.pretrain = pretrain
        self.counter = 0
        ...
    
    def __getitem__(self, idx):
        # Sample state space
        state = sample_state(self.dynamics, self.num_src_samples)
        
        # Sample time based on curriculum
        if self.pretrain:
            # Phase 1: Only terminal time
            time = torch.ones(self.num_src_samples, 1) * self.tMax
        else:
            # Phase 2+: Progressive expansion backward in time
            t_min_current = max(self.tMin, self.tMax - self.counter * dt)
            time = sample_time(t_min_current, self.tMax, self.num_src_samples)
        
        # Combine into coords
        coords = torch.cat([time, state], dim=-1)  # [batch, 4]
        
        return {'coords': coords}
```

If not present, the dataset needs to be modified to support this curriculum structure.

---

### 4. Training Script (`run_experiment.py`)

**File to create:** `IH_DeepReach/run_experiment.py`

**Base:** Copy from `deepreach/run_experiment.py`

**Modifications:**

#### 4.1 Add gamma parameter

In the argument parser section for `Dubins3DDiscounted`:

```python
if orig_opt.dynamics_class == 'Dubins3DDiscounted':
    p.add_argument('--gamma', type=float, required=True, help='Discount factor')
    p.add_argument('--L', type=float, default=0.9, help='State bound')
    p.add_argument('--r', type=float, default=0.2, help='Obstacle radius')
    p.add_argument('--Cx', type=float, default=0.0, help='Obstacle center x')
    p.add_argument('--Cy', type=float, default=0.0, help='Obstacle center y')
    p.add_argument('--u1_min', type=float, default=0.05, help='Min u1')
    p.add_argument('--u1_max', type=float, default=1.0, help='Max u1')
    p.add_argument('--u2_min', type=float, default=-1.0, help='Min u2')
    p.add_argument('--u2_max', type=float, default=1.0, help='Max u2')
    p.add_argument('--d1_min', type=float, default=-0.01, help='Min d1')
    p.add_argument('--d1_max', type=float, default=0.01, help='Max d1')
    p.add_argument('--d2_min', type=float, default=-0.01, help='Min d2')
    p.add_argument('--d2_max', type=float, default=0.01, help='Max d2')
    p.add_argument('--angle_alpha_factor', type=float, default=1.0, help='Angle scaling')
```

#### 4.2 Ensure time range is set

```python
p.add_argument('--tMin', type=float, default=0.0, help='Start time')
p.add_argument('--tMax', type=float, default=10.0, help='End time (large enough for convergence)')
```

#### 4.3 Curriculum parameters

```python
p.add_argument('--pretrain', action='store_true', help='Pretrain terminal condition')
p.add_argument('--pretrain_iters', type=int, default=2000, help='Iterations for pretraining')
p.add_argument('--numpoints', type=int, default=65000, help='Spatial samples per batch')
```

**No other major changes needed** - the original training loop should work as-is.

---

### 5. Experiments File (`experiments/experiments.py`)

**File to create:** `IH_DeepReach/experiments/experiments.py`

**Base:** Copy from `deepreach/experiments/experiments.py`

**Minimal changes needed** - the generic experiment class should work. Just ensure plotting/visualization extracts the `t=0` slice for final results.

---

### 6. Supporting Utilities

Copy the following files **as-is** (no modifications needed):

- `utils/diff_operators.py` - gradient computation utilities
- `utils/modules.py` - SIREN network architecture
- `utils/error_evaluators.py` - testing utilities (optional)

---

## File Structure Summary

```
IH_DeepReach/
├── dynamics/
│   ├── __init__.py          # Import Dubins3DDiscounted
│   └── dynamics.py          # MODIFIED: Base class + Dubins3DDiscounted with gamma
├── experiments/
│   ├── __init__.py
│   └── experiments.py       # COPIED (minor tweaks for t=0 extraction)
├── utils/
│   ├── __init__.py
│   ├── dataio.py            # VERIFIED: Curriculum dataset
│   ├── diff_operators.py    # COPIED as-is
│   ├── losses.py            # MODIFIED: Pass value to Hamiltonian
│   └── modules.py           # COPIED as-is
├── run_experiment.py        # MODIFIED: Add gamma parameter
└── Notes/
    ├── DeepReach_IH.tex
    └── Implementation_Guide.md  # This file
```

---

## Training Procedure

### Step 1: Pretrain Terminal Condition

Run with `--pretrain` flag for 2000-5000 iterations to learn `V(x, T) = g(x)`:

```bash
python run_experiment.py \
    --mode train \
    --dynamics_class Dubins3DDiscounted \
    --gamma 0.7 \
    --tMin 0.0 \
    --tMax 10.0 \
    --pretrain \
    --pretrain_iters 2000 \
    --num_epochs 5000 \
    --numpoints 65000
```

### Step 2: Backward-in-Time Training

Continue training without `--pretrain` to expand backward in time:

```bash
python run_experiment.py \
    --mode train \
    --dynamics_class Dubins3DDiscounted \
    --gamma 0.7 \
    --tMin 0.0 \
    --tMax 10.0 \
    --num_epochs 50000 \
    --numpoints 65000 \
    --checkpoint_toload 5000  # Load pretrained model
```

### Step 3: Extract Infinite-Horizon Solution

After training converges, extract `V(x, t=0)`:

```python
# In evaluation code
t = torch.zeros(num_samples, 1)  # t = 0
x = sample_states(num_samples, 3)
coords = torch.cat([t, x], dim=-1)

with torch.no_grad():
    V_inf = model(coords)  # Approximate infinite-horizon value
```

The viability kernel is: `{x : V(x, 0) ≤ 0}`

---

## Key Differences from Stationary Implementation

| Aspect | Stationary (unstable) | Time-Dependent (stable) |
|--------|----------------------|-------------------------|
| **Input** | `[x1, x2, x3]` | `[t, x1, x2, x3]` |
| **Output** | `V(x)` | `V(x, t)` |
| **Terminal condition** | None | `V(x, T) = g(x)` |
| **Curriculum** | All states at once | Backward from T to 0 |
| **Hamiltonian** | `H = p·f - γV` | Same, but with time derivative |
| **HJ-VI** | `H = 0` constraint | `V_t + H = 0` with `min{·, g-V}=0` |
| **Training stability** | Poor (no causality) | Good (temporal propagation) |
| **Final result** | `V(x)` directly | Extract `V(x, t=0)` |

---

## Testing & Validation

### 1. Monitor Training Metrics

- **Dirichlet loss** should go to ~0 during pretraining
- **HJ-VI residual** should decrease steadily during backward training
- **Value function range** should be reasonable (not exploding)

### 2. Visualize Slices

Plot `V(x1, x2, θ=0, t)` for different time slices:
- `t = T`: Should match `g(x)`
- `t = T/2`: Should show propagation
- `t = 0`: Final infinite-horizon approximation

### 3. Compare with DP Baseline

If available, compare `V(x, t=0)` with the DP solution from MATLAB's `HJIPDE_solve`.

---

## Expected Parameters

Based on the tex file:

```python
gamma = 0.7
T = 10.0
L = 0.9
r = 0.2
Cx = 0.0
Cy = 0.0
u1_range = [0.05, 1.0]
u2_range = [-1.0, 1.0]
d_range = [-0.01, 0.01]
```

---

## Next Steps

1. **Copy files** from `deepreach/` to `IH_DeepReach/` as outlined above
2. **Implement modifications** to `dynamics.py` (add `Dubins3DDiscounted` class)
3. **Update loss function** in `losses.py` (pass `value` to Hamiltonian)
4. **Verify dataset** has curriculum support in `dataio.py`
5. **Add parameters** to `run_experiment.py`
6. **Test** with pretraining first
7. **Run full training** and extract `V(x, 0)`

---

## Troubleshooting

### Issue: Training diverges
- Check that `value_normto` is small (e.g., 0.02)
- Reduce learning rate
- Increase pretrain iterations
- Check that gamma is not too large

### Issue: Terminal condition not satisfied
- Increase `dirichlet_loss_divisor` (weight terminal loss more)
- Extend pretraining phase
- Check boundary function implementation

### Issue: HJ-VI residual stuck
- Verify Hamiltonian computation (especially `-γV` term)
- Check that gradients are flowing (use `deepreach_model="exact"`)
- Ensure time sampling covers full range

---

## Code Modifications Summary

### Critical modifications:
1. ✅ **dynamics.py**: Add `Dubins3DDiscounted` with `hamiltonian(state, dvds, value)`
2. ✅ **losses.py**: Pass `value` to Hamiltonian in loss computation
3. ✅ **run_experiment.py**: Add `--gamma` parameter

### Verification needed:
4. ⚠️ **dataio.py**: Confirm curriculum time sampling exists
5. ⚠️ **experiments.py**: Add `t=0` slice extraction for final visualization

### Copy as-is:
6. ✅ **diff_operators.py**
7. ✅ **modules.py**

This completes the implementation guide. Follow the checklist systematically, starting with file copying, then modifications, then testing.
