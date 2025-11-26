# DeepReach: A Deep Learning Approach to High-Dimensional Reachability

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Setup: Backward Reachable Tubes](#problem-setup)
3. [Hamilton-Jacobi-Isaacs Variational Inequality (HJI-VI)](#hji-vi)
4. [DeepReach Method: Neural Network Representation](#deepreach-method)
5. [Self-Supervised Learning via HJI-VI](#self-supervised-learning)
6. [Network Architecture: Sinusoidal Activations](#network-architecture)
7. [Training Procedure](#training-procedure)
8. [Implementation Details](#implementation-details)
9. [Key Insights and Why It Works](#key-insights)

---

## Introduction

DeepReach solves high-dimensional reachability problems by representing the value function as a deep neural network (DNN) instead of discretizing the state space. The key innovation is using the HJI-VI itself as a source of self-supervision, eliminating the need for explicit supervision data that is difficult to generate for high-dimensional systems.

**Key Advantages:**
- Memory and computation scale with value function complexity, not grid resolution
- No explicit supervision required
- Handles nonlinear dynamics, disturbances, and constraints
- Provides safety controllers directly from value function gradients

---

## Problem Setup: Backward Reachable Tubes

### System Dynamics

Consider a dynamical system with state $x \in \mathbb{R}^n$, control $u \in \mathcal{U}$, and disturbance $d \in \mathcal{D}$:

$$\dot{x} = f(x, u, d)$$

**Code Location:** `deepreach/dynamics/dynamics.py`

The dynamics are implemented in the `dsdt` method of each dynamics class. For example, in `Air3D`:

```python
def dsdt(self, state, control, disturbance):
    dsdt = torch.zeros_like(state)
    dsdt[..., 0] = -self.velocity + self.velocity*torch.cos(state[..., 2]) + control[..., 0]*state[..., 1]
    dsdt[..., 1] = self.velocity*torch.sin(state[..., 2]) - control[..., 0]*state[..., 0]
    dsdt[..., 2] = disturbance[..., 0] - control[..., 0]
    return dsdt
```

### Backward Reachable Tube (BRT)

The BRT $V(t)$ is the set of initial states from which the system, acting optimally and under worst-case disturbances, will eventually reach the target set $\mathcal{L}$ within time horizon $[t, T]$:

$$V(t) = \{x : \forall u(\cdot), \exists d(\cdot), \exists \tau \in [t,T], \xi_{x,t}^{u,d}(\tau) \in \mathcal{L}\}$$

where $\xi_{x,t}^{u,d}(\tau)$ denotes the state trajectory starting from $(x,t)$ under control $u(\cdot)$ and disturbance $d(\cdot)$.

**Target Set Definition:** A target function $l(x)$ is defined such that $\mathcal{L} = \{x : l(x) \leq 0\}$. Typically, $l(x)$ is a signed distance function.

**Code Location:** `deepreach/dynamics/dynamics.py` - `boundary_fn` method

For example, in `Air3D`:
```python
def boundary_fn(self, state):
    return torch.norm(state[..., :2], dim=-1) - self.collisionR
```

### Backward Reach-Avoid Tube (BRAT)

For reach-avoid problems, we want to reach target set $\mathcal{L}$ while avoiding unsafe set $\mathcal{G}$:

$$V(t) = \{x : \forall d(\cdot), \exists u(\cdot), \forall s \in [t,T], \xi_{x,t}^{u,d}(s) \notin \mathcal{G}, \exists \tau \in [t,T], \xi_{x,t}^{u,d}(\tau) \in \mathcal{L}\}$$

---

## Hamilton-Jacobi-Isaacs Variational Inequality (HJI-VI)

### Value Function

The value function $V(x,t)$ represents the minimum distance to the target set over optimal trajectories:

$$J(x,t,u(\cdot),d(\cdot)) = \min_{\tau \in [t,T]} l(\xi_{x,t}^{u,d}(\tau))$$

$$V(x,t) = \inf_{d(\cdot)} \sup_{u(\cdot)} J(x,t,u(\cdot),d(\cdot))$$

### HJI Variational Inequality for BRT

Using dynamic programming, the value function satisfies the HJI-VI:

$$\min\{D_t V(x,t) + H(x,t), l(x) - V(x,t)\} = 0 \tag{1}$$

with terminal condition: $V(x,T) = l(x)$

where:
- $D_t V(x,t)$ is the time derivative of the value function
- $\nabla V(x,t)$ is the spatial gradient
- $H(x,t)$ is the **Hamiltonian**

### Hamiltonian

The Hamiltonian encodes the dynamics and optimal control/disturbance:

$$H(x,t) = \max_{u \in \mathcal{U}} \min_{d \in \mathcal{D}} \langle \nabla V(x,t), f(x,u,d) \rangle \tag{2}$$

**Code Location:** `deepreach/dynamics/dynamics.py` - `hamiltonian` method

For example, in `Air3D`:
```python
def hamiltonian(self, state, dvds):
    ham = self.omega_max * torch.abs(dvds[..., 0] * state[..., 1] - dvds[..., 1] * state[..., 0] - dvds[..., 2])
    ham = ham - self.omega_max * torch.abs(dvds[..., 2])
    ham = ham + (self.velocity * (torch.cos(state[..., 2]) - 1.0) * dvds[..., 0]) + (self.velocity * torch.sin(state[..., 2]) * dvds[..., 1])
    return ham
```

Here, `dvds` is $\nabla V(x,t)$ (the spatial gradients).

### HJI-VI for BRAT

For reach-avoid problems, the HJI-VI becomes:

$$\max\{\min\{D_t V(x,t) + H(x,t), l(x) - V(x,t)\}, g(x) - V(x,t)\} = 0$$

with $V(x,T) = \max\{l(x), g(x)\}$, where $g(x)$ defines the avoid set $\mathcal{G} = \{x : g(x) > 0\}$.

**Code Location:** `deepreach/utils/losses.py` - `init_brat_hjivi_loss` function

### Optimal Control

Once the value function is learned, the optimal safety controller is:

$$u^*(x,t) = \arg\max_{u \in \mathcal{U}} \min_{d \in \mathcal{D}} \langle \nabla V(x,t), f(x,u,d) \rangle$$

**Code Location:** `deepreach/dynamics/dynamics.py` - `optimal_control` method

---

## DeepReach Method: Neural Network Representation

### Key Idea

Instead of solving the HJI-VI on a discretized grid (which scales exponentially with dimension), DeepReach represents the value function as a DNN:

$$V_\theta(x,t) \approx V(x,t)$$

where $\theta$ are the neural network parameters.

**Code Location:** `deepreach/utils/modules.py` - `SingleBVPNet` class

```python
class SingleBVPNet(nn.Module):
    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.net = FCBlock(in_features=in_features, out_features=out_features, 
                          num_hidden_layers=num_hidden_layers,
                          hidden_features=hidden_features, 
                          outermost_linear=True, nonlinearity=type)
    
    def forward(self, model_input, params=None):
        coords_org = model_input['coords'].clone().detach().requires_grad_(True)
        output = self.net(coords_org)
        return {'model_in': coords_org, 'model_out': output}
```

**Key Points:**
- Input: `[t, x_1, x_2, ..., x_n]` (time and state coordinates)
- Output: $V_\theta(x,t)$ (scalar value)
- The input coordinates are set to `requires_grad_(True)` to enable gradient computation

### Value Function Conversion

The network output is normalized and converted to the actual value function:

**Code Location:** `deepreach/dynamics/dynamics.py` - `io_to_value` method

For the "exact" model (most common):
```python
def io_to_value(self, input, output):
    if self.deepreach_model == "exact":
        return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
    # ... other model types
```

The "exact" model uses the form: $V(x,t) = t \cdot \tilde{V}_\theta(x,t) + l(x)$, where $\tilde{V}_\theta$ is the network output. This ensures $V(x,T) = l(x)$ at terminal time.

---

## Self-Supervised Learning via HJI-VI

### Loss Function

The key insight is to use the HJI-VI itself as supervision. The loss function is:

$$h(x_i, t_i; \theta) = h_1(x_i, t_i; \theta) + \lambda h_2(x_i, t_i; \theta) \tag{3}$$

where:

**Terminal Condition Loss:**
$$h_1(x_i, t_i; \theta) = \|V_\theta(x_i, t_i) - l(x_i)\| \cdot \mathbf{1}(t_i = T) \tag{4}$$

**HJI-VI Residual Loss:**
$$h_2(x_i, t_i; \theta) = \left\|\min\{D_t V_\theta(x_i, t_i) + H(x_i, t_i), l(x_i) - V_\theta(x_i, t_i)\}\right\| \tag{5}$$

**Code Location:** `deepreach/utils/losses.py` - `init_brt_hjivi_loss` function

```python
def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
    if torch.all(dirichlet_mask):
        # pretraining loss - only terminal condition
        diff_constraint_hom = torch.Tensor([0])
    else:
        # Compute Hamiltonian
        ham = dynamics.hamiltonian(state, dvds)
        if minWith == 'zero':
            ham = torch.clamp(ham, max=0.0)
        
        # HJI-VI residual: min{D_t V + H, l(x) - V}
        diff_constraint_hom = dvdt - ham
        if minWith == 'target':
            diff_constraint_hom = torch.max(diff_constraint_hom, value - boundary_value)
    
    # Terminal condition loss
    dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
    
    return {
        'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
        'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()
    }
```

**Variables:**
- `value`: $V_\theta(x,t)$
- `dvdt`: $D_t V_\theta(x,t)$ (time derivative)
- `dvds`: $\nabla V_\theta(x,t)$ (spatial gradients)
- `boundary_value`: $l(x)$
- `dirichlet_mask`: Boolean mask for terminal time points ($t = T$)

### Gradient Computation

The time and spatial gradients are computed using automatic differentiation:

**Code Location:** `deepreach/dynamics/dynamics.py` - `io_to_dv` method

```python
def io_to_dv(self, input, output):
    # Compute Jacobian: gradients of output w.r.t. input
    dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)
    
    if self.deepreach_model == "exact":
        # Time derivative: dV/dt
        dvdt = (self.value_var / self.value_normto) * (input[..., 0]*dodi[..., 0] + output)
        
        # Spatial gradients: dV/dx
        dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
        state = self.input_to_coord(input)[..., 1:]
        dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
        dvds = dvds_term1 + dvds_term2
    
    return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)
```

**Code Location:** `deepreach/utils/diff_operators.py` - `jacobian` function

```python
def jacobian(y, x):
    ''' jacobian of y wrt x '''
    jac = torch.zeros(*y.shape, x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        y_flat = y[...,i].view(-1, 1)
        jac[..., i, :] = grad(y_flat, x, torch.ones_like(y_flat), create_graph=True)[0]
    return jac, status
```

### Training Loop

**Code Location:** `deepreach/experiments/experiments.py` - `train` method (lines 91-467)

```python
# Forward pass
model_results = self.model({'coords': model_input['model_coords']})

# Extract states, values, and gradients
states = self.dataset.dynamics.input_to_coord(model_results['model_in'].detach())[..., 1:]
values = self.dataset.dynamics.io_to_value(model_results['model_in'].detach(), model_results['model_out'].squeeze(dim=-1))
dvs = self.dataset.dynamics.io_to_dv(model_results['model_in'], model_results['model_out'].squeeze(dim=-1))

# Compute loss
losses = loss_fn(states, values, dvs[..., 0], dvs[..., 1:], boundary_values, dirichlet_masks, model_results['model_out'])

# Backward pass
train_loss = sum(loss.mean() for loss in losses.values())
train_loss.backward()
optim.step()
```

---

## Network Architecture: Sinusoidal Activations

### Why Sinusoidal Activations?

The loss function requires accurate computation of **gradients** of the value function:
1. For the HJI-VI residual: $D_t V$ and $\nabla V$
2. For the Hamiltonian: $\nabla V$
3. For optimal control: $\nabla V$

**Problem with ReLU networks:**
- ReLU networks are piecewise linear
- Their derivatives are piecewise constant
- They struggle to represent smooth gradients accurately

**Solution: Sinusoidal activations**
- Periodic functions can represent both the signal and its derivatives well
- Enable accurate gradient computation via automatic differentiation

**Code Location:** `deepreach/utils/modules.py` - `Sine` class

```python
class Sine(nn.Module):
    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
```

The factor of 30 is chosen to balance the frequency content of the network.

### Network Initialization

**Code Location:** `deepreach/utils/modules.py` - initialization functions

```python
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5
            m.weight.uniform_(-1 / num_input, 1 / num_input)
```

The first layer uses a different initialization scheme to prevent high-frequency components initially.

### Architecture Details

**Typical Configuration:**
- 3 hidden layers
- 512 hidden units per layer
- Sinusoidal activation functions
- Linear output layer

**Code Location:** `deepreach/utils/modules.py` - `FCBlock` class

```python
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='sine', ...):
        # First layer
        self.net.append(nn.Sequential(BatchLinear(in_features, hidden_features), Sine()))
        
        # Hidden layers
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(BatchLinear(hidden_features, hidden_features), Sine()))
        
        # Output layer (linear)
        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
```

---

## Training Procedure

### Curriculum Learning

The training uses a curriculum learning strategy:

1. **Pretraining Phase:** Learn terminal condition $V(x,T) = l(x)$
   - Only $h_1$ loss is used ($\lambda = 0$)
   - All samples are at terminal time $t = T$

2. **Curriculum Phase:** Gradually expand time horizon
   - Time $t$ is sampled from $[t_{\min}, t_{\min} + \alpha(t_{\max} - t_{\min})]$
   - $\alpha$ increases from 0 to 1 as training progresses
   - This allows the terminal condition to propagate backward in time

**Code Location:** `deepreach/utils/dataio.py` - `ReachabilityDataset.__getitem__`

```python
if self.pretrain:
    # Only sample at terminal time
    times = torch.full((self.numpoints, 1), self.tMin)
else:
    # Curriculum learning: gradually expand time range
    time_interval_length = (self.counter/self.counter_end)*(self.tMax-self.tMin)
    times = self.tMin + torch.zeros(self.numpoints, 1).uniform_(0, time_interval_length)
    # Always include some samples at initial time
    times[-self.num_src_samples:, 0] = self.tMin
```

**Code Location:** `deepreach/experiments/experiments.py` - training loop (line 135)

```python
time_interval_length = (self.dataset.counter/self.dataset.counter_end)*(self.dataset.tMax-self.dataset.tMin)
```

### Sampling Strategy

At each iteration:
- Sample $N$ states uniformly from state space (normalized to $[-1,1]$)
- Sample times according to curriculum schedule
- Compute loss and update network parameters

**Code Location:** `deepreach/utils/dataio.py` - `ReachabilityDataset.__getitem__`

```python
# Uniformly sample domain
model_states = torch.zeros(self.numpoints, self.dynamics.state_dim).uniform_(-1, 1)

# Optionally add target state samples
if self.num_target_samples > 0:
    target_state_samples = self.dynamics.sample_target_state(self.num_target_samples)
    model_states[-self.num_target_samples:] = ...
```

### Loss Weight Balancing

The relative magnitudes of $h_1$ and $h_2$ are balanced to ensure both terms contribute meaningfully:

**Code Location:** `deepreach/experiments/experiments.py` - lines 173-218

```python
if adjust_relative_grads:
    # Compute gradients w.r.t. PDE loss
    losses['diff_constraint_hom'].backward(retain_graph=True)
    grads_PDE = torch.cat([param.grad.view(-1) for param in params.values()])
    
    # Compute gradients w.r.t. boundary loss
    optim.zero_grad()
    losses['dirichlet'].backward(retain_graph=True)
    grads_dirichlet = torch.cat([param.grad.view(-1) for param in params.values()])
    
    # Adaptive weight scaling
    num = torch.mean(torch.abs(grads_PDE))
    den = torch.mean(torch.abs(grads_dirichlet))
    new_weight = 0.9*new_weight + 0.1*num/den
    losses['dirichlet'] = new_weight*losses['dirichlet']
```

---

## Implementation Details

### State Normalization

States are normalized to $[-1, 1]$ for training stability:

**Code Location:** `deepreach/dynamics/dynamics.py` - `coord_to_input` and `input_to_coord`

```python
def coord_to_input(self, coord):
    """Convert real coordinates to normalized model input"""
    input = coord.clone()
    input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
    return input

def input_to_coord(self, input):
    """Convert normalized model input back to real coordinates"""
    coord = input.clone()
    coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
    return coord
```

### Value Function Normalization

The value function is normalized using `value_mean`, `value_var`, and `value_normto`:

**Code Location:** `deepreach/dynamics/dynamics.py` - `io_to_value` (exact model)

```python
def io_to_value(self, input, output):
    if self.deepreach_model == "exact":
        # V(x,t) = t * (output * value_var / value_normto) + l(x)
        return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
```

This ensures:
- The network output is in a normalized range
- The value function satisfies $V(x,T) = l(x)$ at terminal time
- Gradients are properly scaled

### Dirichlet Masks

Dirichlet masks identify samples at terminal time where the boundary condition must be enforced:

**Code Location:** `deepreach/utils/dataio.py` - line 49

```python
if self.pretrain:
    dirichlet_masks = torch.ones(model_coords.shape[0]) > 0  # All samples
else:
    # Only enforce initial conditions at tMin
    dirichlet_masks = (model_coords[:, 0] == self.tMin)
```

**Code Location:** `deepreach/utils/losses.py` - line 18

```python
dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
```

---

## Key Insights and Why It Works

### 1. Self-Supervision via PDE

**Key Insight:** The HJI-VI itself provides supervision without needing explicit value function labels.

- Traditional methods require solving the PDE on a grid to get supervision
- DeepReach uses the PDE residual as the loss function
- This eliminates the exponential scaling with dimension

**Why it works:**
- If $V_\theta$ satisfies the HJI-VI, then $h_2 = 0$
- The terminal condition $h_1$ ensures we learn the correct solution (not degenerate ones)
- Together, they enforce both the PDE and boundary conditions

### 2. Sinusoidal Activations for Gradients

**Key Insight:** Accurate gradient computation is essential, and sinusoidal activations enable this.

- The Hamiltonian requires $\nabla V(x,t)$
- Optimal control requires $\nabla V(x,t)$
- ReLU networks have poor gradient representations
- Sinusoidal networks can represent both function and derivatives accurately

**Why it works:**
- Periodic functions have smooth, well-behaved derivatives
- The network can learn high-frequency details in both the value function and its gradients
- Automatic differentiation provides exact gradients during training

### 3. Curriculum Learning

**Key Insight:** Gradually expanding the time horizon helps the terminal condition propagate backward.

- Start with terminal condition at $t = T$
- Gradually include earlier time points
- This mimics the backward-in-time nature of the HJI-VI

**Why it works:**
- The HJI-VI propagates information backward from terminal time
- Curriculum learning aligns with this natural propagation direction
- Prevents the network from learning degenerate solutions early in training

### 4. Memory Efficiency

**Key Insight:** Memory scales with network size, not grid resolution.

- Traditional methods: Memory $\propto N^n$ where $N$ is grid points per dimension
- DeepReach: Memory $\propto$ network parameters (typically $O(10^5 - 10^6)$)

**Why it works:**
- Neural networks are function approximators
- They can represent complex functions with relatively few parameters
- The complexity scales with the "complexity" of the value function, not the state space dimension

### 5. Optimal Control from Gradients

**Key Insight:** Once the value function is learned, optimal control is computed directly from gradients.

**Code Location:** `deepreach/dynamics/dynamics.py` - `optimal_control`

```python
def optimal_control(self, state, dvds):
    # dvds is ∇V(x,t)
    det = dvds[..., 0]*state[..., 1] - dvds[..., 1]*state[..., 0] - dvds[..., 2]
    return (self.omega_max * torch.sign(det))[..., None]
```

This is computed analytically for control-affine systems, using the value function gradients.

---

## Summary

DeepReach combines several key innovations:

1. **Neural network representation** of the value function avoids grid discretization
2. **Self-supervised learning** using the HJI-VI as the loss function
3. **Sinusoidal activations** enable accurate gradient computation
4. **Curriculum learning** aligns with the backward propagation nature of the PDE
5. **Direct control synthesis** from value function gradients

The method scales to high-dimensional systems (9D, 10D demonstrated) while maintaining accuracy comparable to traditional grid-based methods, but with much better memory and computational efficiency.

---

## References

- Bansal, S., & Tomlin, C. J. (2021). DeepReach: A Deep Learning Approach to High-Dimensional Reachability. *2021 IEEE International Conference on Robotics and Automation (ICRA)*.
- Sitzmann, V., et al. (2020). Implicit Neural Representations with Periodic Activation Functions. *NeurIPS*.

