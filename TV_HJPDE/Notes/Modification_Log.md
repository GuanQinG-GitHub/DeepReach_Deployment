# Modification Log: Infinite Horizon Discounted Viability Kernel

This document details the modifications made to the DeepReach scripts in `DeepReach_Deployment/TV_HJPDE` to support the infinite horizon discounted viability kernel problem.

## 1. Dynamics Configuration (`dynamics/dynamics.py`)

**Modification:**

- Removed all extraneous dynamics classes to keep the file clean and focused.
- Added a new class `Dubins3DDiscounted` that inherits from the base `Dynamics` class.

**Reasoning:**

- The original DeepReach repository contains many example systems that are not relevant to this specific problem.
- We need a specific system definition that supports the discount factor $\gamma$ and the specific Hamiltonian for the infinite horizon problem.
- **Key Change:** The `hamiltonian` method in `Dubins3DDiscounted` now accepts `value` as a third argument and includes the discount term:
  $$H(x, \nabla V, V) = \min_u \max_d [ \nabla V \cdot f(x,u,d) - \gamma V(x) ]$$
  This $-\gamma V$ term is essential for the discounted infinite horizon formulation.

## 2. Loss Function (`utils/losses.py`)

**Modification:**

- Updated `init_brt_hjivi_loss` and `init_brat_hjivi_loss` functions.
- Changed the call to `dynamics.hamiltonian` to pass the `value` (squeezed to match dimensions) as the third argument.

**Reasoning:**

- The standard HJI-VI for finite horizon problems does not depend on $V$ itself in the Hamiltonian, only on $\nabla V$.
- For the discounted case, the PDE becomes $V_t + H(x, \nabla V, V) = 0$.
- Passing `value` to the Hamiltonian allows the dynamics class to compute the $-\gamma V$ term correctly.

## 3. Experiment Runner (`run_experiment.py`)

**Status:** No modifications were necessary.

**Reasoning:**

- The script uses `configargparse` and dynamic inspection to parse arguments for the chosen dynamics class.
- Since we added `gamma` and other parameters to the `__init__` method of `Dubins3DDiscounted` with type hints, `run_experiment.py` automatically handles the command-line arguments (e.g., `--gamma`).

## Summary of Approach

We are solving the infinite horizon problem by treating it as a finite horizon problem with a large time horizon $T$. We use the time-dependent training machinery of DeepReach to propagate the value function backward from a terminal condition $V(x, T) = g(x)$. The discount factor $\gamma$ is incorporated into the Hamiltonian, which drives the value function towards the infinite horizon solution $V(x)$ as $T \to \infty$ (or effectively, at $t=0$ after a long backward horizon).
