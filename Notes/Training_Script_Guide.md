# DeepReach Training Script Guide

This guide explains the structure and functionality of the DeepReach training scripts for the 3D Dubins car example.

## 1. Overview

The training process solves a Hamilton-Jacobi (HJ) Partial Differential Equation (PDE) using a neural network. The goal is to learn the value function $V(x)$ which represents the safety of the system.

- **Objective**: Find $V(x)$ such that $V(x) \le 0$ inside the safe region (viability kernel).
- **Method**: Stationary Discounted Hamilton-Jacobi Reachability.
- **Loss Function**: Enforces the HJ PDE constraints: $\max(g(x) - V(x), \nabla V \cdot f(x, u, d) - \gamma V(x)) = 0$.

## 2. Code Structure

The codebase is organized into modular components:

### Entry Point
- **`run_experiment.py`**: The main script.
    - Parses command-line arguments (hyperparameters).
    - Initializes the **Dynamics** (physics).
    - Initializes the **Dataset** (sampling points).
    - Initializes the **Model** (neural network).
    - Initializes the **Loss Function** (PDE constraint).
    - Initializes the **Experiment** (training loop).
    - Starts the training.

### Components

1.  **`dynamics/dynamics.py`** (`Dubins3D` class)
    - Defines the system physics ($f(x, u, d)$).
    - **`hamiltonian`**: Computes $\min_u \max_d \nabla V \cdot f(x, u, d)$. This is the core of the optimal control logic.
    - **`boundary_fn`**: Defines the target/obstacle set $g(x)$. Here, it represents the obstacle (cylinder) and state bounds.

2.  **`utils/modules.py`** (`SingleBVPNet`)
    - Defines the Neural Network architecture.
    - Uses a Multi-Layer Perceptron (MLP) with **Sine** activation functions (SIREN), which are good for representing derivatives.
    - **`forward`**: Computes $V(x)$ given state $x$. It also returns the input coordinates with `requires_grad=True` to allow computing $\nabla V$.

3.  **`utils/dataio.py`** (`StationaryReachabilityDataset`)
    - Generates training data.
    - Unlike traditional supervised learning, we don't have "labels".
    - We sample random points $x$ in the state space. The "label" comes from the PDE loss function itself.

4.  **`utils/losses.py`** (`init_stationary_discounted_loss`)
    - Defines the loss function.
    - Calculates the residual of the PDE: `residual = max(g(x) - V(x), Hamiltonian - gamma * V(x))`.
    - Minimizes the mean absolute residual.

5.  **`experiments/experiments.py`** (`StationaryExperiment`)
    - Manages the training loop.
    - **`train`**: Iterates through epochs, computes loss, updates weights.
    - **`validate`**: Periodically plots the 0-level set of $V(x)$ to visualize the learned safe region.
    - Handles logging to TensorBoard and WandB.

## 3. How to Learn This Script

To understand the code deeply, follow this path:

1.  **Start at `run_experiment.py`**:
    - Look at the `main()` function.
    - See how the components are instantiated and connected.
    - Notice the flow: `Dynamics` -> `Dataset` -> `Model` -> `Loss` -> `Experiment`.

2.  **Dive into `dynamics/dynamics.py`**:
    - Understand the `Dubins3D` class.
    - Focus on `hamiltonian`: This is where the control logic lives. It calculates the optimal control input $u$ and disturbance $d$ based on the gradient of the value function ($\nabla V$).
    - Check `boundary_fn`: This defines what "safe" and "unsafe" mean (the obstacle).

3.  **Examine `utils/losses.py`**:
    - See how the math connects.
    - `ham_dot` comes from `dynamics.hamiltonian`.
    - `g_x` comes from `dynamics.boundary_fn`.
    - The loss forces the network to satisfy the HJ PDE.

4.  **Check `experiments/experiments.py`**:
    - Look at the `train` loop.
    - Note how `dvdx` (gradient of V w.r.t x) is computed using `torch.autograd.grad`. This is crucial for the Hamiltonian.
    - Look at `validate`: It evaluates the model on a grid to create the visualization plots.

## 4. Key Concepts

- **SIREN (Sine Activation)**: We use sine activations because we need accurate derivatives ($\nabla V$) for the Hamiltonian. Standard ReLUs have discontinuous derivatives.
- **Physics-Informed Loss**: The network is trained without ground truth data. It is trained to satisfy a physical equation (the PDE) at random points.
- **Curriculum**: In more complex setups (not this simple one), we might use a curriculum (e.g., training on time). Here, we solve the stationary problem directly.
