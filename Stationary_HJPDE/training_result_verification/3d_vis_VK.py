import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import sys

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import modules
from dynamics import dynamics

def visualize_3d(model_path, dynamics_class_name='Dubins3D', hidden_features=512, num_hidden_layers=3, resolution=200):
    # 1. Load Dynamics
    dyn_class = getattr(dynamics, dynamics_class_name)
    # Assuming default params for now, or we could load from a config if available
    dyn = dyn_class() 
    
    # 2. Load Model
    # We need to reconstruct the model architecture exactly as it was trained
    model = modules.SingleBVPNet(in_features=dyn.input_dim, out_features=1, type='sine', mode='mlp',
                                 hidden_features=hidden_features, num_hidden_layers=num_hidden_layers)
    
    # Load weights
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    # Handle both full checkpoint dict and just state_dict
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    model.requires_grad_(False)
    
    # 3. Setup Plotting Grid
    plot_config = dyn.plot_config()
    state_range = dyn.state_test_range()
    
    x_idx = plot_config['x_axis_idx']
    y_idx = plot_config['y_axis_idx']
    z_idx = plot_config['z_axis_idx']
    
    x_min, x_max = state_range[x_idx]
    y_min, y_max = state_range[y_idx]
    z_min, z_max = state_range[z_idx]
    
    # Define theta slices
    z_values = torch.linspace(z_min, z_max, 5)
    
    xs = torch.linspace(x_min, x_max, resolution)
    ys = torch.linspace(y_min, y_max, resolution)
    xys = torch.cartesian_prod(xs, ys)
    
    # Create meshgrid for 3D plotting
    X, Y = torch.meshgrid(xs, ys, indexing='xy')
    
    # 4. Generate Plots
    # Create a figure with 2 rows: Top = 2D, Bottom = 3D
    fig = plt.figure(figsize=(5*len(z_values), 10))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print("Generating 2D and 3D plots...")
    for i, z_val in enumerate(z_values):
        coords = torch.zeros(resolution*resolution, dyn.state_dim)
        coords[:, x_idx] = xys[:, 0]
        coords[:, y_idx] = xys[:, 1]
        coords[:, z_idx] = z_val 
        
        with torch.no_grad():
            model_results = model({'coords': coords.to(device)})
            values = model_results['model_out'].squeeze(dim=-1).detach().cpu()
        
        # Reshape 
        # values_grid for 2D: [x, y] -> transpose to [y, x] for imshow
        values_grid = values.reshape(resolution, resolution).T
        # Z for 3D: same shape
        Z = values_grid
        
        # --- 2D Plot (Top Row) ---
        ax_2d = fig.add_subplot(2, len(z_values), i+1)
        ax_2d.set_title(f'2D: {plot_config["state_labels"][z_idx]} = {z_val:.2f}')
        # Plot raw values instead of binary mask
        # Use same colormap as 3D plot for consistency
        s = ax_2d.imshow(values_grid, cmap='viridis', origin='lower', extent=(x_min, x_max, y_min, y_max))
        fig.colorbar(s, ax=ax_2d, fraction=0.046, pad=0.04)
        ax_2d.set_xlabel(plot_config["state_labels"][x_idx])
        ax_2d.set_ylabel(plot_config["state_labels"][y_idx])

        # --- 3D Plot (Bottom Row) ---
        ax_3d = fig.add_subplot(2, len(z_values), len(z_values) + i+1, projection='3d')
        ax_3d.set_title(f'3D: {plot_config["state_labels"][z_idx]} = {z_val:.2f}')
        
        surf = ax_3d.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', edgecolor='none')
        fig.colorbar(surf, ax=ax_3d, fraction=0.046, pad=0.04)
        ax_3d.set_xlabel(plot_config["state_labels"][x_idx])
        ax_3d.set_ylabel(plot_config["state_labels"][y_idx])
        ax_3d.set_zlabel('V(x)')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to model .pth file')
    parser.add_argument('--dynamics', type=str, default='Dubins3D', help='Dynamics class name')
    args = parser.parse_args()
    
    visualize_3d(args.model_path, dynamics_class_name=args.dynamics)
