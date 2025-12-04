import torch
import scipy.io
import numpy as np
import sys
import os
import pickle
import inspect

# Add deepreach root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
deepreach_root = os.path.dirname(current_dir)
sys.path.append(deepreach_root)

from dynamics import dynamics
from utils import modules

def export_to_matlab(experiment_dir, output_mat_file='deepreach_model_epoch_600000_3.mat'):
    print(f"Loading experiment from: {experiment_dir}")
    
    # 1. Load original options
    opt_path = os.path.join(experiment_dir, 'orig_opt.pickle')
    if not os.path.exists(opt_path):
        print(f"Error: Could not find {opt_path}")
        return

    with open(opt_path, 'rb') as f:
        opt = pickle.load(f)

    # 2. Initialize Dynamics
    # We use the same logic as run_experiment.py
    dynamics_class = getattr(dynamics, opt.dynamics_class)
    dynamics_params = {argname: getattr(opt, argname) for argname in inspect.signature(dynamics_class).parameters.keys() if argname != 'self'}
    dyn = dynamics_class(**dynamics_params)
    dyn.deepreach_model = opt.deepreach_model

    print(f"Initialized Dynamics: {opt.dynamics_class}")

    # 3. Initialize Model
    model = modules.SingleBVPNet(
        in_features=dyn.input_dim, 
        out_features=1, 
        type=opt.model, 
        mode=opt.model_mode,
        final_layer_factor=1., 
        hidden_features=opt.num_nl, 
        num_hidden_layers=opt.num_hl
    )
    
    # Move to CPU for export
    device = torch.device('cpu')
    model.to(device)
    
    print(f"Initialized Model: SingleBVPNet (Hidden Layers: {opt.num_hl}, Neurons: {opt.num_nl})")

    # 4. Load Checkpoint
    checkpoint_path = os.path.join(experiment_dir, 'training', 'checkpoints', 'model_epoch_600000.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("Loaded trained weights.")

    # 5. Extract Weights and Biases
    export_dict = {}
    
    # Export normalization parameters
    def to_numpy(val):
        if torch.is_tensor(val):
            return val.detach().cpu().numpy()
        elif isinstance(val, (int, float)):
            return np.array([val])
        else:
            return np.array(val)

    export_dict['state_mean'] = to_numpy(dyn.state_mean)[:, None] if to_numpy(dyn.state_mean).ndim == 1 else to_numpy(dyn.state_mean)
    export_dict['state_var'] = to_numpy(dyn.state_var)[:, None] if to_numpy(dyn.state_var).ndim == 1 else to_numpy(dyn.state_var)
    export_dict['value_mean'] = to_numpy(dyn.value_mean)
    export_dict['value_var'] = to_numpy(dyn.value_var)
    export_dict['value_normto'] = to_numpy(dyn.value_normto)
    export_dict['deepreach_model'] = dyn.deepreach_model
    
    # Export Network Weights
    # We iterate through the named parameters to find W and b for each layer
    # The structure in SingleBVPNet -> FCBlock is:
    # net[0]: Linear -> Activation
    # net[1]: Linear -> Activation
    # ...
    # net[k]: Linear (Outermost)
    
    # We can just iterate named_parameters and assign indices based on order
    # Or rely on the naming convention "net.0.0.weight", "net.1.0.weight", etc.
    
    layer_count = 0
    
    # Helper to parse layer index from name "net.X.0.weight"
    # The FCBlock structure is sequential.
    # net[0] is layer 1
    # net[1] is layer 2
    # ...
    
    for name, param in model.net.named_parameters():
        # name format example: "net.0.0.weight" or "net.0.0.bias"
        # The first number is the layer index in the Sequential block
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == 'net':
            layer_idx = int(parts[1]) + 1 # 1-based index for MATLAB
            param_type = parts[-1] # 'weight' or 'bias'
            
            key = ''
            if param_type == 'weight':
                key = f'W{layer_idx}'
                export_dict[key] = param.detach().cpu().numpy()
            elif param_type == 'bias':
                key = f'b{layer_idx}'
                export_dict[key] = param.detach().cpu().numpy()[:, None] # Column vector
            
            if layer_idx > layer_count:
                layer_count = layer_idx

    export_dict['num_layers'] = np.array([[layer_count]])
    
    # 6. Save to .mat
    scipy.io.savemat(output_mat_file, export_dict)
    print(f"Successfully exported model to {output_mat_file}")
    print("-" * 30)
    print("MATLAB Usage Info:")
    print(f"  - Number of Layers: {layer_count}")
    print(f"  - Weight Matrices: W1 to W{layer_count}")
    print(f"  - Bias Vectors: b1 to b{layer_count}")
    print("  - Activation: sin(30 * x) for all hidden layers")
    print("  - Output Layer: Linear (no activation)")
    print("-" * 30)

if __name__ == '__main__':
    # Default path based on user's workspace
    # Adjust this path if needed
    default_exp_dir = os.path.join(deepreach_root, 'runs', 'dubins3dDiscounted_trial_4')
    
    if len(sys.argv) > 1:
        exp_dir = sys.argv[1]
    else:
        exp_dir = default_exp_dir
        
    export_to_matlab(exp_dir)
