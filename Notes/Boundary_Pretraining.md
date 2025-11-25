# Boundary Function Pretraining

## Overview

The training script now supports initializing the neural network to approximate the boundary function `g(x)` before the main PDE training. This can significantly improve convergence speed.

## How It Works

Before the main training loop, the network is trained for a specified number of iterations to minimize:

```
Loss = MSE(V(x), g(x))
```

where:
- `V(x)` is the network output
- `g(x)` is the boundary function (obstacle definition)

This gives the network a "head start" by initializing it close to the correct boundary values.

## Usage

Add the `--pretrain_boundary` flag to your training command:

```bash
python run_experiment.py \
    --mode train \
    --experiment_class StationaryExperiment \
    --dynamics_class Dubins3D \
    --experiment_name dubins3d_pretrained \
    --gamma 0.1 \
    --numpoints 65000 \
    --num_epochs 100000 \
    --epochs_til_ckpt 1000 \
    --lr 2e-5 \
    --device cuda:0 \
    --pretrain_boundary \
    --pretrain_iters 2000
```

## Parameters

- `--pretrain_boundary`: Enable pretraining (flag, no value needed)
- `--pretrain_iters`: Number of pretraining iterations (default: 2000)

## Expected Behavior

With pretraining enabled:
1. The script will first run 2000 iterations of boundary function fitting
2. You'll see output like: `Pretrain iter 0/2000, Loss: 0.XXXXX`
3. After pretraining completes, the main PDE training begins
4. The initial validation plots should already show a reasonable approximation of the obstacle boundary

## Benefits

- **Faster convergence**: The network starts closer to the solution
- **Better initial plots**: Epoch 0 plots will show meaningful structure
- **More stable training**: Reduces early training instability
