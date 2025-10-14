# Installation Guide

## Quick Setup

1. **Copy your trained model**:
   ```bash
   # From the DeepReach training directory
   cp ../deepreach/runs/dubins3d_gpu_final/training/checkpoints/model_current.pth ./
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the example**:
   ```bash
   python example_usage.py
   ```

## Detailed Setup

### Prerequisites
- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy, Matplotlib, SciPy

### Step 1: Get Your Trained Model
After training DeepReach, copy the model checkpoint:
```bash
# Navigate to your DeepReach training results
cd path/to/deepreach/runs/dubins3d_gpu_final/training/checkpoints/

# Copy the final model
cp model_current.pth /path/to/DeepReach_Model_Deployment/
```

### Step 2: Install Dependencies
```bash
cd DeepReach_Model_Deployment
pip install -r requirements.txt
```

### Step 3: Test Installation
```bash
python example_usage.py
```

This should generate:
- Reachability plots (PNG files)
- Trajectory analysis plots
- Console output showing value functions and controls

## Troubleshooting

### Model File Not Found
If you get "Model file not found", ensure you've copied the checkpoint:
```bash
ls -la model_current.pth  # Should show the file
```

### CUDA Issues
If you have CUDA available but want to use CPU:
```python
model = load_deepreach_model("model_current.pth", device='cpu')
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install torch numpy matplotlib scipy tqdm
```

## Usage in Your Code

```python
from deepreach_deployment import load_deepreach_model

# Load model
model = load_deepreach_model("model_current.pth", device='cpu')

# Check if state is safe
is_safe = model.is_safe(time=0.0, state=[0.5, 0.0, 0.0])

# Get optimal control
control = model.get_optimal_control(time=0.0, state=[0.5, 0.0, 0.0])

# Evaluate value function
value = model.evaluate_value(time=0.0, state=[0.5, 0.0, 0.0])
```
