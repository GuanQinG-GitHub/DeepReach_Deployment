# DeepReach Model Deployment

This repository contains a deployment-ready implementation for using trained DeepReach models, specifically for the Dubins3D system trained in the original DeepReach repository.

## Overview

This package provides:
- **Model Loading**: Load trained DeepReach neural network models
- **Value Function Evaluation**: Compute reachable/avoidable sets
- **Optimal Control**: Extract optimal control policies
- **Visualization**: Generate reachability plots and analysis
- **Integration**: Easy-to-use API for robotics applications

## Files

- `deepreach_deployment.py` - Main deployment script with all functionality
- `requirements.txt` - Python dependencies
- `example_usage.py` - Example usage and demonstrations
- `README.md` - This documentation

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy your trained model checkpoint to this directory:
   ```bash
   cp ../deepreach/runs/dubins3d_gpu_final/training/checkpoints/model_current.pth ./
   ```

3. Run the example:
   ```bash
   python example_usage.py
   ```

## Model Information

- **System**: Dubins3D (2D vehicle with heading control)
- **Task**: Avoid circular target region (radius=0.25)
- **State**: [x, y, θ] (position and heading)
- **Control**: Angular velocity u ∈ [-1.1, 1.1]
- **Value Function**: V(t,x) ≤ 0 indicates avoidable states

## Reference

Based on the original DeepReach repository: https://github.com/smlbansal/deepreach
Paper: "DeepReach: A Deep Learning Approach to High-Dimensional Reachability" (ICRA 2021)
