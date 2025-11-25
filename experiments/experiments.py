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
    def __init__(self, model, dataset, experiment_dir, use_wandb):
        self.model = model
        self.dataset = dataset
        self.experiment_dir = experiment_dir
        self.use_wandb = use_wandb

    def init_special(self):
        pass

    def validate(self, device, epoch, save_path, resolution=200):
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
        z_val = plot_config['state_slices'][z_idx] # Fixed Z value for 2D slice

        xs = torch.linspace(x_min, x_max, resolution)
        ys = torch.linspace(y_min, y_max, resolution)
        xys = torch.cartesian_prod(xs, ys)
        
        coords = torch.zeros(resolution*resolution, self.dataset.dynamics.state_dim)
        coords[:, x_idx] = xys[:, 0]
        coords[:, y_idx] = xys[:, 1]
        coords[:, z_idx] = z_val # Fix Z
        
        # If there are other dimensions, they should be fixed too. 
        # For Dubins3D, we have 3 dims, so this is sufficient.
        
        with torch.no_grad():
            model_results = self.model({'coords': coords.to(device)})
            values = model_results['model_out'].squeeze(dim=-1).detach().cpu()
            
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(f'Epoch {epoch}, {plot_config["state_labels"][z_idx]} = {z_val}')
        
        # Reshape for imshow (x corresponds to columns, y to rows)
        # imshow origin='lower' expects [rows, cols] -> [y, x]
        # values are from cartesian_prod(xs, ys), so first dim is x, second is y.
        # We need to reshape to [len(xs), len(ys)] and then transpose to [len(ys), len(xs)]
        values_grid = values.reshape(resolution, resolution).T
        
        # Plot 0-level set
        s = ax.imshow(1*(values_grid <= 0), cmap='bwr', origin='lower', extent=(x_min, x_max, y_min, y_max))
        fig.colorbar(s)
        
        fig.savefig(save_path)
        
        if self.use_wandb:
            wandb.log({
                'step': epoch,
                'val_plot': wandb.Image(fig),
            })
        plt.close(fig)

        if was_training:
            self.model.train()
            self.model.requires_grad_(True)

    def train(self, device, batch_size, epochs, lr, steps_til_summary, epochs_til_checkpoint, loss_fn):
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
                    model_results = self.model(model_input)
                    
                    # Compute gradients
                    # model_in has requires_grad=True
                    state = model_results['model_in']
                    value = model_results['model_out']
                    
                    # Compute dvdx
                    dvdx = self.dataset.dynamics.io_to_dv(state, value.squeeze(-1))
                    
                    # Compute loss
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
