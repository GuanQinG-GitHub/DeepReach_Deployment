import wandb
import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from utils import losses

class StationaryExperiment:
    """
    Manages the training process for stationary reachability problems.
    """
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb

    def init_special(self):
        pass

    def validate(self, device, epoch, save_path, resolution=200):
        """
        Validates the model by plotting the 0-level set of the value function.
        The 0-level set represents the boundary of the safe region (viability kernel).
        """
        was_training = self.model.training
        self.model.eval()
        self.model.requires_grad_(False)

        plot_config = self.dataset.dynamics.plot_config()
        state_range = self.dataset.dynamics.state_test_range()
        
        x_idx = plot_config['x_axis_idx']
        y_idx = plot_config['y_axis_idx']
        z_idx = plot_config['z_axis_idx']
        
        x_min, x_max = state_range[x_idx]
        y_min, y_max = state_range[y_idx]
        
        # Define z-values (thetas) to plot
        # Plot 5 slices from min to max of the z-dimension
        z_min, z_max = state_range[z_idx]
        z_values = torch.linspace(z_min, z_max, 5)

        xs = torch.linspace(x_min, x_max, resolution)
        ys = torch.linspace(y_min, y_max, resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        # --- 2D Plots (Level Sets) ---
        fig_2d = plt.figure(figsize=(5*len(z_values), 5))
        
        for i, z_val in enumerate(z_values):
            coords = torch.zeros(resolution*resolution, self.dataset.dynamics.state_dim)
            coords[:, x_idx] = xys[:, 0]
            coords[:, y_idx] = xys[:, 1]
            coords[:, z_idx] = z_val 
            
            with torch.no_grad():
                model_results = self.model({'coords': coords.to(device)})
                values = model_results['model_out'].squeeze(dim=-1).detach().cpu()
            
            values_grid = values.reshape(resolution, resolution).T
            
            ax = fig_2d.add_subplot(1, len(z_values), i+1)
            ax.set_title(f'Epoch {epoch}, {plot_config["state_labels"][z_idx]} = {z_val:.2f}')
            
            # Plot 0-level set
            # Red (1) = Unsafe (V > 0, inside obstacle)
            # Blue (0) = Safe (V <= 0, outside obstacle)
            s = ax.imshow(1*(values_grid > 0), cmap='bwr', origin='lower', extent=(x_min, x_max, y_min, y_max))
            fig_2d.colorbar(s)
        
        fig_2d.savefig(save_path)
        
        # --- 3D Plots (Surface) ---
        fig_3d = plt.figure(figsize=(5*len(z_values), 5))
        
        # Create meshgrid for 3D plotting
        X, Y = torch.meshgrid(xs, ys, indexing='xy')
        
        for i, z_val in enumerate(z_values):
            coords = torch.zeros(resolution*resolution, self.dataset.dynamics.state_dim)
            coords[:, x_idx] = xys[:, 0]
            coords[:, y_idx] = xys[:, 1]
            coords[:, z_idx] = z_val 
            
            with torch.no_grad():
                model_results = self.model({'coords': coords.to(device)})
                values = model_results['model_out'].squeeze(dim=-1).detach().cpu()
            
            # Reshape for surface plot
            Z = values.reshape(resolution, resolution).T
            
            ax = fig_3d.add_subplot(1, len(z_values), i+1, projection='3d')
            ax.set_title(f'Epoch {epoch}, {plot_config["state_labels"][z_idx]} = {z_val:.2f}')
            
            surf = ax.plot_surface(X.numpy(), Y.numpy(), Z.numpy(), cmap='viridis', edgecolor='none')
            fig_3d.colorbar(surf)
            ax.set_xlabel(plot_config["state_labels"][x_idx])
            ax.set_ylabel(plot_config["state_labels"][y_idx])
            ax.set_zlabel('V(x)')
            
        save_path_3d = save_path.replace('.png', '_3d.png')
        fig_3d.savefig(save_path_3d)

        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot_2d': wandb.Image(fig_2d),
                'val_plot_3d': wandb.Image(fig_3d),
            })
        plt.close(fig_2d)
        plt.close(fig_3d)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def pretrain_to_boundary(self, device, iters=2000, lr=1e-3, numpoints=10000):
        """
        Pretrain the network to match the boundary function g(x).
        This provides a good initialization for the value function.
        """
        print("Pretraining network to boundary function g(x)...")
        self.model.train()
        self.model.requires_grad_(True)
        
        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())
        
        state_range = self.dataset.dynamics.state_test_range()
        low = torch.tensor([r[0] for r in state_range])
        high = torch.tensor([r[1] for r in state_range])
        
        for i in range(iters):
            # Sample random points
            state = torch.rand(numpoints, self.dataset.dynamics.state_dim) * (high - low) + low
            state = state.to(device)
            
            # Get boundary function values
            g_x = self.dataset.dynamics.boundary_fn(state)
            
            # Forward pass
            model_results = self.model({'coords': state})
            value = model_results['model_out'].squeeze(-1)
            
            # MSE loss between V(x) and g(x)
            loss = torch.mean((value - g_x)**2)
            
            # Backprop
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if i % 100 == 0:
                print(f"Pretrain iter {i}/{iters}, Loss: {loss.item():.6f}")
        
        print("Pretraining complete!")

    def train(self, device, batch_size, epochs, lr, steps_til_summary, epochs_til_checkpoint, loss_fn, 
              pretrain_boundary=False, pretrain_iters=2000):
        """
        Main training loop.
        Iterates through epochs, samples data, computes loss, and updates model weights.
        """
        # Pretrain to boundary function if requested
        if pretrain_boundary:
            self.pretrain_to_boundary(device, iters=pretrain_iters)
        
        self.model.train()
        self.model.requires_grad_(True)
        
        train_dataloader = DataLoader(self.dataset, shuffle=True, batch_size=batch_size, num_workers=0)
        optim = torch.optim.Adam(lr=lr, params=self.model.parameters())
        
        training_dir = os.path.join(self.experiment_dir, 'training')
        summaries_dir = os.path.join(training_dir, 'summaries')
        checkpoints_dir = os.path.join(training_dir, 'checkpoints')
        os.makedirs(summaries_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        writer = SummaryWriter(summaries_dir)
        total_steps = 0
        
        with tqdm(total=epochs) as pbar:
            for epoch in range(epochs):
                for step, (model_input, _) in enumerate(train_dataloader):
                    start_time = time.time()
                    
                    model_input = {key: value.to(device) for key, value in model_input.items()}
                    
                    # Forward pass
                    # The model outputs the value function V(x)
                    model_results = self.model(model_input)
                    
                    # Compute gradients
                    # model_in has requires_grad=True
                    state = model_results['model_in']
                    value = model_results['model_out']
                    
                    # Compute dvdx
                    # We need the gradient of V w.r.t state x to compute the Hamiltonian.
                    # This is done using torch.autograd.grad.
                    dvdx = self.dataset.dynamics.io_to_dv(state, value.squeeze(-1))
                    
                    # Compute loss
                    # The loss function enforces the PDE constraint.
                    loss_dict = loss_fn(state, value, dvdx)
                    loss = loss_dict['loss']
                    
                    # Backprop
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    
                    # Logging
                    if not total_steps % steps_til_summary:
                        writer.add_scalar('loss', loss.item(), total_steps)
                        tqdm.write(f"Epoch {epoch}, Loss {loss.item():.6f}")
                        
                        if self.use_wandb:
                            wandb.log({
                                'step': total_steps,
                                'loss': loss.item(),
                            })
                            
                    total_steps += 1
                
                pbar.update(1)
                
                # Checkpointing and Validation
                if not (epoch + 1) % epochs_til_checkpoint:
                    torch.save(self.model.state_dict(), os.path.join(checkpoints_dir, f'model_epoch_{epoch+1:04d}.pth'))
                    self.validate(device, epoch+1, os.path.join(checkpoints_dir, f'val_plot_epoch_{epoch+1:04d}.png'))
                    
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(checkpoints_dir, 'model_final.pth'))
