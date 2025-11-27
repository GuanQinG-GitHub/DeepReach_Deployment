# DeepReach Training Algorithm & Curriculum Learning Detail

This document details the training flow and algorithm used in the DeepReach implementation for the infinite horizon discounted viability kernel problem.

## 1. Entry Point: `run_experiment.py`

The training process begins in `run_experiment.py`. Its primary responsibilities are:

1. **Argument Parsing**: Uses `configargparse` to handle command-line arguments and configuration files. It dynamically loads arguments for the chosen `dynamics_class` (e.g., `Dubins3DDiscounted`).
2. **Initialization**:
    * **Dynamics**: Instantiates the system dynamics (e.g., `Dubins3DDiscounted`).
    * **Dataset**: Instantiates `ReachabilityDataset` (from `utils/dataio.py`), which handles data sampling and curriculum logic.
    * **Model**: Instantiates `SingleBVPNet` (from `utils/modules.py`), which is the neural network (typically a SIREN - Sinusoidal Representation Network).
    * **Experiment**: Instantiates `DeepReach` (from `experiments/experiments.py`), which manages the training loop.
    * **Loss Function**: Selects the appropriate loss function (e.g., `init_brt_hjivi_loss` from `utils/losses.py`).
3. **Execution**: Calls `experiment.train()` to start the training loop.

## 2. The Training Loop: `experiments/experiments.py`

The core training logic resides in the `train` method of the `Experiment` class.

### High-Level Flow

For each epoch (from 0 to `num_epochs`):

1. **Data Sampling**: The `DataLoader` requests a batch from `dataset`. Since `ReachabilityDataset` generates data on-the-fly, every batch consists of fresh, randomly sampled points in the state-time space.
2. **Forward Pass**: The batch of coordinates `(t, x)` is passed through the `model` to get predicted values $V(x, t)$.
3. **Loss Computation**:
    * The model output and inputs are converted to physical units.
    * Gradients $\nabla_x V$ and time derivative $V_t$ are computed using automatic differentiation.
    * The Hamiltonian $H(x, \nabla V, V)$ is computed using the dynamics class.
    * The PDE residual is calculated: $\mathcal{L}_{PDE} = |V_t + H|$.
    * The boundary condition residual is calculated: $\mathcal{L}_{BC} = |V(x, t_{min}) - l(x)|$.
    * The total loss is a weighted sum of these residuals.
4. **Optimization**: Backpropagation computes gradients of the loss w.r.t. network weights, and the optimizer (Adam) updates the weights.
5. **Logging**: Loss values are logged to TensorBoard/WandB. Checkpoints are saved periodically.

## 3. Curriculum Learning Detail: `utils/dataio.py`

DeepReach uses a specific curriculum learning strategy to ensure stable training of the value function, propagating information from the known boundary condition backward in time.

### Phase 1: Pretraining (Terminal Condition)

* **Trigger**: Enabled by `--pretrain` flag.
* **Duration**: Runs for `pretrain_iters` iterations.
* **Sampling Logic**:
  * All time samples are forced to be at the "start" time `tMin` (which physically represents the terminal time $T$ or time 0 in backward reachability).
  * `times = tMin`
  * `dirichlet_masks = True` everywhere.
* **Goal**: The network learns to perfectly approximate the boundary function $l(x)$ (the obstacle/target set) at $t=t_{min}$ before trying to solve the PDE. This provides a solid "initial condition" for the PDE solver.

### Phase 2: Progressive Time Expansion

* **Trigger**: After pretraining finishes (or immediately if pretraining is disabled).
* **Mechanism**: The time horizon for sampling grows linearly with the training progress.
* **Sampling Logic**:
  * A `counter` variable increments at every step until it reaches `counter_end` (usually `num_epochs`).
  * The effective time horizon is calculated as:
        $$ T_{current} = t_{min} + (t_{max} - t_{min}) \times \frac{counter}{counter_{end}} $$
  * Time samples $t$ are drawn uniformly from $[t_{min}, T_{current}]$.
* **Effect**:
  * Initially, the network only sees points very close to $t_{min}$. Since it already knows the solution at $t_{min}$ (from pretraining), it's easy to learn the solution at $t_{min} + \epsilon$.
  * As training proceeds, the horizon expands, allowing the solution to "grow" or propagate backward in time naturally.
  * This mimics the physical causality of the system and prevents the "vanishing gradient" or "cold start" problems often seen when trying to solve PDEs over the entire domain at once.

### Phase 3: Boundary Maintenance

* Even during the progressive expansion phase, a small fraction of samples (`num_src_samples`) are always forced to be exactly at `tMin`.
* This ensures the network doesn't "forget" the boundary condition while trying to satisfy the PDE elsewhere.

## 4. Summary of Key Parameters

* `--pretrain`: Enables Phase 1.
* `--pretrain_iters`: Duration of Phase 1.
* `--tMin`: The time where the boundary condition is known (usually 0.0).
* `--tMax`: The final time horizon to solve for.
* `--counter_end`: The epoch at which the curriculum fully covers `[tMin, tMax]`. usually set to `num_epochs`.
