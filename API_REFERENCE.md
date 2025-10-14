# API Reference

## DeepReachModel Class

The main interface for using trained DeepReach models.

### Constructor
```python
DeepReachModel(model_path: str, device: str = 'cpu')
```
- `model_path`: Path to the trained model checkpoint (.pth file)
- `device`: Device to run on ('cpu' or 'cuda')

### Methods

#### `evaluate_value(time, state)`
Evaluate the value function V(t,x) at a given time and state.

**Parameters:**
- `time` (float): Time t
- `state` (list/np.ndarray/torch.Tensor): State vector [x, y, θ]

**Returns:**
- `float`: Value function V(t,x). V ≤ 0 indicates avoidable/reachable states.

**Example:**
```python
value = model.evaluate_value(0.0, [0.5, 0.0, 0.0])
```

#### `get_optimal_control(time, state)`
Compute optimal control at a given time and state.

**Parameters:**
- `time` (float): Time t
- `state` (list/np.ndarray/torch.Tensor): State vector [x, y, θ]

**Returns:**
- `float`: Optimal control u* (angular velocity)

**Example:**
```python
control = model.get_optimal_control(0.0, [0.5, 0.0, 0.0])
```

#### `is_safe(time, state)`
Check if a state is safe (avoidable) at a given time.

**Parameters:**
- `time` (float): Time t
- `state` (list/np.ndarray/torch.Tensor): State vector [x, y, θ]

**Returns:**
- `bool`: True if state is safe (avoidable), False otherwise

**Example:**
```python
safe = model.is_safe(0.0, [0.5, 0.0, 0.0])
```

#### `generate_reachability_plot(time, theta, resolution, save_path)`
Generate a 2D reachability plot at fixed time and heading.

**Parameters:**
- `time` (float): Time slice to visualize
- `theta` (float): Heading angle to visualize
- `resolution` (int): Grid resolution
- `save_path` (str, optional): Path to save the plot

**Example:**
```python
model.generate_reachability_plot(time=0.0, theta=0.0, resolution=200, save_path="plot.png")
```

#### `analyze_trajectory(states, times)`
Analyze a trajectory for safety and optimal controls.

**Parameters:**
- `states` (np.ndarray): Array of states [N, 3] with columns [x, y, θ]
- `times` (np.ndarray): Array of times [N]

**Returns:**
- `dict`: Dictionary with analysis results containing:
  - `values`: Array of value functions along trajectory
  - `controls`: Array of optimal controls along trajectory
  - `safe_states`: Boolean array indicating safe states
  - `min_value`, `max_value`: Value function bounds
  - `safety_ratio`: Fraction of safe states

**Example:**
```python
states = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
times = np.array([0.0, 0.5])
analysis = model.analyze_trajectory(states, times)
```

## Utility Functions

### `load_deepreach_model(model_path, device)`
Convenience function to load a trained DeepReach model.

**Parameters:**
- `model_path` (str): Path to the trained model checkpoint
- `device` (str): Device to run on ('cpu' or 'cuda')

**Returns:**
- `DeepReachModel`: Loaded model instance

**Example:**
```python
from deepreach_deployment import load_deepreach_model
model = load_deepreach_model("model_current.pth", device='cpu')
```

## System Parameters

The model is configured with the following parameters (matching training):

- **Goal radius**: 0.25
- **Vehicle velocity**: 0.6
- **Maximum angular velocity**: 1.1
- **Angle normalization factor**: 1.2
- **Set mode**: 'avoid' (avoiding the target region)
- **State dimensions**: [x, y, θ]
- **Control dimension**: 1 (angular velocity)

## Value Function Interpretation

- **V(t,x) ≤ 0**: State is avoidable (can be avoided from this state)
- **V(t,x) > 0**: State is safe (cannot be avoided, must be reached)
- **V(t,x) = 0**: Boundary of the avoidable set

## Control Interpretation

- **u* > 0**: Turn left (positive angular velocity)
- **u* < 0**: Turn right (negative angular velocity)
- **|u*| = 1.1**: Maximum turn rate
- **u* = 0**: No turning (rare, only at specific conditions)
