# Verification Plan for DeepReach Deployment

This document outlines the steps to verify the implementation of the stationary discounted HJ-VI training for the 3D Dubins car.

## 1. Environment Setup
Ensure you have the necessary dependencies installed. The environment should be the same as the standard DeepReach environment.

```bash
pip install torch numpy matplotlib configargparse wandb tqdm tensorboard
```

## 2. Running the Training Script
To train the model for the 3D Dubins car with a discount factor of $\gamma=0.1$, run the following command from the `DeepReach` root directory:

```bash
python DeepReach_Deployment/run_experiment.py \
    --mode train \
    --experiment_class StationaryExperiment \
    --dynamics_class Dubins3D \
    --experiment_name dubins3d_gamma0.1 \
    --gamma 0.1 \
    --numpoints 65000 \
    --num_epochs 10000 \
    --epochs_til_ckpt 1000 \
    --lr 2e-5 \
    --device cuda:0  # Or cpu if no GPU available
```

## 3. Expected Output
- **Console Output**: You should see a progress bar (tqdm) and periodic loss updates. The loss should generally decrease over time.
- **WandB**: If you enabled `--use_wandb`, you should see loss curves and validation plots in your WandB project.
- **Directories**:
    - `runs/dubins3d_gamma0.1/training/checkpoints/`: Should contain `model_epoch_XXXX.pth` and `val_plot_epoch_XXXX.png`.
    - `runs/dubins3d_gamma0.1/training/summaries/`: Should contain TensorBoard logs.

## 4. Verifying Results
- **Loss Convergence**: Check `runs/dubins3d_gamma0.1/training/summaries/` using TensorBoard or check the console output. The loss should converge to a low value (e.g., < 1e-3).
- **Validation Plots**: Open the generated PNG files in the checkpoints directory. You should see the 0-level set of the value function.
    - The safe region (inside the viability kernel) is where $V(x) \le 0$.
    - The plots show slices of the state space (e.g., x-y plane for a fixed $\theta$).
    - You should see the "safe" region (blue/white) and "unsafe" region (red).
    - As training progresses, the boundary should sharpen and converge to the true viability kernel.

## 5. Troubleshooting
- **Loss not decreasing**: Try adjusting the learning rate (`--lr`) or the number of points (`--numpoints`).
- **CUDA errors**: Ensure your PyTorch installation matches your CUDA version. Use `--device cpu` to debug.
- **Import errors**: Ensure you are running the command from the `DeepReach` root directory so that python can find the `DeepReach_Deployment` package.
