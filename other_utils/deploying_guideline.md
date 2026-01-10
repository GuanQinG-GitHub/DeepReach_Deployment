# Deployment Guideline for Dubins3DDiscounted Model

This document provides detailed instructions on how to load and use the trained neural network for the `Dubins3DDiscounted` dynamics.

## 1. Model Overview

The trained model is a **SIREN (Sinusoidal Representation Network)**, which is a Multi-Layer Perceptron (MLP) with sine activation functions. It approximates the value function $V(t, x)$ for the Hamilton-Jacobi reachability problem.

### Architecture
- **Type**: `SingleBVPNet` (wrapping a Fully Connected Block)
- **Input Dimension**: 4 (`[time, x, y, theta]`)
- **Output Dimension**: 1 (Value $V$)
- **Hidden Layers**: 3 (default)
- **Neurons per Layer**: 512 (default)
- **Activation**: Sine (SIREN)

### Inputs and Outputs
- **Input**: A tensor of shape `(batch_size, 4)` representing the state and time:
  - `input[..., 0]`: Time $t$
  - `input[..., 1]`: $x$ position
  - `input[..., 2]`: $y$ position
  - `input[..., 3]`: $\theta$ heading angle
- **Output**: A tensor of shape `(batch_size, 1)` representing the value $V$.
  - **Sign Convention**:
    - $V(x) < 0$: **Unsafe** (inside obstacle or failure set).
    - $V(x) \ge 0$: **Safe**.

## 2. Prerequisites

Ensure you have the `deepreach` codebase in your Python path. You will need the following dependencies:
- `torch`
- `numpy`
- `matplotlib` (for visualization)

## 3. Step-by-Step Deployment

### Step 1: Import Modules

```python
import torch
import sys
import os
import math

# Add deepreach to path if running from outside
sys.path.append(os.path.abspath("path/to/deepreach"))

from dynamics import dynamics
from utils import modules
```

### Step 2: Initialize Dynamics and Model

You must instantiate the dynamics and model classes with the **exact same parameters** used during training.

```python
# 1. Initialize Dynamics
# Ensure these match your training config (check config.txt in the run folder)
gamma = 0.7
angle_alpha_factor = 1.0
set_mode = 'avoid'

dyn = dynamics.Dubins3DDiscounted(
    gamma=gamma, 
    angle_alpha_factor=angle_alpha_factor, 
    set_mode=set_mode
)

# 2. Initialize Model
# Check config.txt for num_hl (hidden layers) and num_nl (neurons per layer)
model = modules.SingleBVPNet(
    in_features=dyn.input_dim, 
    out_features=1, 
    type='sine', 
    mode='mlp', 
    hidden_features=512, 
    num_hidden_layers=3
)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

### Step 3: Load Trained Weights

Load the checkpoint file (`model_final.pth` or a specific epoch).

```python
checkpoint_path = "path/to/deepreach/runs/dubins3dDiscounted_trial_2/training/checkpoints/model_final.pth"

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

# Load state dictionary
# Note: The checkpoint dictionary keys might vary. 
# Usually it is checkpoint['model'] if saved by the training script.
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
else:
    model.load_state_dict(checkpoint)

model.eval() # Set to evaluation mode
print("Model loaded successfully.")
```

### Step 4: Inference (Querying Value)

To get the safety value for a batch of states:

```python
# Example: Query value at t=0 for a batch of states
# State: [x, y, theta]
batch_size = 5
t = torch.zeros(batch_size, 1)
x = torch.zeros(batch_size, 1) # x = 0
y = torch.zeros(batch_size, 1) # y = 0
theta = torch.zeros(batch_size, 1) # theta = 0

# Create coordinate tensor [t, x, y, theta]
coords = torch.cat((t, x, y, theta), dim=1).to(device)

# 1. Normalize input (Real coords -> Model input)
model_input = dyn.coord_to_input(coords)

# 2. Forward pass
with torch.no_grad():
    model_results = model({'coords': model_input})
    
    # 3. Denormalize output (Model output -> Real value)
    values = dyn.io_to_value(model_results['model_in'], model_results['model_out'])

print("Values:", values)
# Interpretation: Negative = Unsafe, Positive = Safe
```

### Step 5: Optimal Control (Safety Controller)

To get the optimal safety control $u^*$ (which tries to keep the system safe):

```python
# We need gradients to compute control, so enable grad tracking
coords.requires_grad_(True)

# 1. Normalize input
model_input = dyn.coord_to_input(coords)

# 2. Forward pass
model_results = model({'coords': model_input})

# 3. Compute gradients (dv/ds)
# io_to_dv computes the spatial gradients needed for the Hamiltonian
dvs = dyn.io_to_dv(model_results['model_in'], model_results['model_out'])

# 4. Compute Optimal Control
# u* = argmin H(x, p) for reach / argmax H(x, p) for avoid
# The dynamics class handles the logic based on set_mode
opt_control = dyn.optimal_control(coords[..., 1:], dvs[..., 1:])

print("Optimal Control (u1, u2):", opt_control)
```

## 4. Summary of Key Files

- **`deepreach/dynamics/dynamics.py`**: Contains the `Dubins3DDiscounted` class definition.
- **`deepreach/utils/modules.py`**: Contains the `SingleBVPNet` neural network definition.
- **`deepreach/run_experiment.py`**: The training script (reference for hyperparameters).

## 5. Troubleshooting

- **Dimension Mismatch**: Ensure `in_features` matches `dyn.input_dim` (should be 4).
- **Device Errors**: Ensure both the model and input tensors are on the same device (`cpu` or `cuda`).
- **Value Scaling**: The model outputs normalized values. Always use `dyn.io_to_value` to convert back to real units.

## 6. Using MATLAB

To use the trained model in MATLAB, the recommended approach is to export the weights and biases to a `.mat` file and reconstruct the simple MLP forward pass in MATLAB.

### Step 1: Export Weights to .mat (Python)

Run this Python script to save the model parameters.

```python
import scipy.io
import torch
import numpy as np

# ... (Load model as shown in Step 3 above) ...

# Extract weights and biases
# Note: PyTorch weights are [out_features, in_features], MATLAB usually expects [in_features, out_features] for x*W, 
# or we can keep it as is and do W*x. Let's keep it as is (W*x).

weights = {}
i = 1
for name, param in model.named_parameters():
    if 'weight' in name:
        weights[f'W{i}'] = param.detach().cpu().numpy()
    elif 'bias' in name:
        weights[f'b{i}'] = param.detach().cpu().numpy()[:, None] # Make column vector
        i += 1

# Also export normalization parameters
weights['state_mean'] = dyn.state_mean.cpu().numpy()[:, None]
weights['state_var'] = dyn.state_var.cpu().numpy()[:, None]
weights['value_mean'] = dyn.value_mean.cpu().numpy()
weights['value_var'] = dyn.value_var.cpu().numpy()

# Save to .mat
scipy.io.savemat('deepreach_model.mat', weights)
print("Saved model to deepreach_model.mat")
```

### Step 2: Inference in MATLAB

```matlab
% Load the model
data = load('deepreach_model.mat');

% Define Input (e.g., [t; x; y; theta])
% NOTE: Input must be a column vector [4 x 1]
state = [0; 0.5; 0.5; 0]; 

% 1. Normalize Input
% input = (coord - mean) ./ var
% Note: The first dimension (time) is not normalized in the python code logic 
% (input[..., 1:] = ...), but let's check input_to_coord logic.
% Actually, coord_to_input does:
% input = coord.clone();
% input[..., 1:] = (coord[..., 1:] - state_mean) / state_var;
% So time (index 1) is untouched.

model_in = state;
model_in(2:end) = (state(2:end) - data.state_mean) ./ data.state_var;

% 2. Forward Pass
% Architecture: Linear -> Sin(30x) -> ... -> Linear
% Based on export: 5 Layers (4 Hidden + 1 Output)

% Layer 1
z = data.W1 * model_in + data.b1;
z = sin(30 * z);

% Layer 2
z = data.W2 * z + data.b2;
z = sin(30 * z);

% Layer 3
z = data.W3 * z + data.b3;
z = sin(30 * z);

% Layer 4
z = data.W4 * z + data.b4;
z = sin(30 * z);

% Layer 5 (Output Layer, no activation)
z = data.W5 * z + data.b5;

model_out = z;

% 3. Denormalize Output
% value = model_out * value_var + value_mean
value = model_out * data.value_var + data.value_mean;

disp(['Value: ', num2str(value)]);
```

**Note**: The script above assumes a 5-layer network (Standard for `num_hl=3` in DeepReach: 1 input + 3 hidden + 1 output = 5 layers). If your model differs, add/remove layer blocks accordingly.

