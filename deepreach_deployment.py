"""
DeepReach Model Deployment Script

This script provides a complete interface for loading and using trained DeepReach models.
It includes all necessary components extracted from the original DeepReach repository.

Based on: https://github.com/smlbansal/deepreach
Paper: "DeepReach: A Deep Learning Approach to High-Dimensional Reachability" (ICRA 2021)

Author: Generated from DeepReach training results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import math


class Sine(nn.Module):
    """
    Sine activation function used in DeepReach networks.
    Reference: deepreach/utils/modules.py
    """
    def forward(self, input):
        return torch.sin(30 * input)


class BatchLinear(nn.Module):
    """
    Batch-compatible linear layer.
    Reference: deepreach/utils/modules.py
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None

    def forward(self, input):
        output = torch.matmul(input, self.weight.T)
        if self.bias is not None:
            output += self.bias.unsqueeze(-2)
        return output


class FCBlock(nn.Module):
    """
    Fully connected block with sine activation.
    Reference: deepreach/utils/modules.py
    """
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features):
        super().__init__()
        self.net = nn.Sequential()
        
        # Input layer
        self.net.add_module('0', nn.Sequential(
            BatchLinear(in_features, hidden_features, bias=True),
            Sine()
        ))
        
        # Hidden layers
        for i in range(num_hidden_layers):
            self.net.add_module(str(i+1), nn.Sequential(
                BatchLinear(hidden_features, hidden_features, bias=True),
                Sine()
            ))
        
        # Output layer
        self.net.add_module(str(num_hidden_layers+1), 
                           BatchLinear(hidden_features, out_features, bias=True))

    def forward(self, coords):
        return self.net(coords)


class SingleBVPNet(nn.Module):
    """
    Single Boundary Value Problem Network - the main DeepReach architecture.
    Reference: deepreach/utils/modules.py
    """
    def __init__(self, in_features, out_features, type='sine', mode='mlp',
                 final_layer_factor=1., hidden_features=512, num_hidden_layers=3):
        super().__init__()
        self.net = FCBlock(in_features, out_features, num_hidden_layers, hidden_features)

    def forward(self, model_input):
        coords = model_input['coords']
        output = self.net(coords)
        return {'model_in': coords, 'model_out': output}


class Dubins3DDynamics:
    """
    Dubins3D system dynamics and value function computations.
    Reference: deepreach/dynamics/dynamics.py (lines 270-346)
    """
    def __init__(self, goalR=0.25, velocity=0.6, omega_max=1.1, 
                 angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False):
        self.goalR = goalR
        self.velocity = velocity
        self.omega_max = omega_max
        self.angle_alpha_factor = angle_alpha_factor
        self.set_mode = set_mode
        self.freeze_model = freeze_model
        
        # System dimensions
        self.state_dim = 3  # [x, y, θ]
        self.input_dim = 4  # [t, x, y, θ]
        self.control_dim = 1  # angular velocity
        self.disturbance_dim = 0
        
        # Normalization parameters (from training)
        self.state_mean = torch.tensor([0, 0, 0])
        self.state_var = torch.tensor([1, 1, self.angle_alpha_factor * math.pi])
        self.value_mean = 0.25
        self.value_var = 0.5
        self.value_normto = 0.02
        self.deepreach_model = "exact"

    def coord_to_input(self, coord):
        """
        Convert real coordinates [t, x, y, θ] to model input.
        Reference: deepreach/dynamics/dynamics.py line 49-52
        """
        input_coord = coord.clone()
        input_coord[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input_coord

    def input_to_coord(self, input_coord):
        """
        Convert model input to real coordinates.
        Reference: deepreach/dynamics/dynamics.py line 43-46
        """
        coord = input_coord.clone()
        coord[..., 1:] = (input_coord[..., 1:] * self.state_var.to(device=input_coord.device)) + self.state_mean.to(device=input_coord.device)
        return coord

    def io_to_value(self, input_coord, output):
        """
        Convert model input/output to real value function V(t,x).
        Reference: deepreach/dynamics/dynamics.py line 55-61
        """
        if self.deepreach_model == "exact":
            return (output * input_coord[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input_coord)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    def boundary_fn(self, state):
        """
        Boundary function: distance to circular target.
        Reference: deepreach/dynamics/dynamics.py line 313-314
        """
        return torch.norm(state[..., :2], dim=-1) - self.goalR

    def io_to_dv(self, input_coord, output):
        """
        Compute gradients [∂V/∂t, ∂V/∂x] from model output.
        Reference: deepreach/dynamics/dynamics.py line 64-88
        """
        # Compute Jacobian using autograd
        dodi = torch.autograd.grad(output.unsqueeze(dim=-1), input_coord, 
                                  torch.ones_like(output.unsqueeze(dim=-1)), 
                                  create_graph=True)[0].squeeze(dim=-2)

        if self.deepreach_model == "exact":
            dvdt = (self.value_var / self.value_normto) * (input_coord[..., 0] * dodi[..., 0] + output)
            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input_coord[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input_coord)[..., 1:]
            dvds_term2 = torch.autograd.grad(self.boundary_fn(state).unsqueeze(dim=-1), state, 
                                           torch.ones_like(self.boundary_fn(state).unsqueeze(dim=-1)), 
                                           create_graph=True)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    def optimal_control(self, state, dvds):
        """
        Compute optimal control from value function gradients.
        Reference: deepreach/dynamics/dynamics.py line 330-334
        """
        if self.set_mode == 'reach':
            return (-self.omega_max * torch.sign(dvds[..., 2]))[..., None]
        elif self.set_mode == 'avoid':
            return (self.omega_max * torch.sign(dvds[..., 2]))[..., None]

    def hamiltonian(self, state, dvds):
        """
        Compute Hamiltonian for the system.
        Reference: deepreach/dynamics/dynamics.py line 322-328
        """
        if self.freeze_model:
            raise NotImplementedError
        if self.set_mode == 'reach':
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) - self.omega_max * torch.abs(dvds[..., 2])
        elif self.set_mode == 'avoid':
            return self.velocity * (torch.cos(state[..., 2]) * dvds[..., 0] + torch.sin(state[..., 2]) * dvds[..., 1]) + self.omega_max * torch.abs(dvds[..., 2])


class DeepReachModel:
    """
    Main interface for loading and using trained DeepReach models.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the DeepReach model.
        
        Args:
            model_path: Path to the trained model checkpoint (.pth file)
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        
        # Initialize dynamics (matching training parameters)
        self.dynamics = Dubins3DDynamics(
            goalR=0.25, velocity=0.6, omega_max=1.1, 
            angle_alpha_factor=1.2, set_mode='avoid', freeze_model=False
        )
        
        # Initialize model architecture (matching training)
        self.model = SingleBVPNet(
            in_features=self.dynamics.input_dim, 
            out_features=1, 
            type='sine', 
            mode='mlp',
            final_layer_factor=1., 
            hidden_features=512, 
            num_hidden_layers=3
        )
        
        # Load trained weights
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path: str):
        """Load trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")

    def evaluate_value(self, time: float, state: Union[list, np.ndarray, torch.Tensor]) -> float:
        """
        Evaluate the value function V(t,x) at a given time and state.
        
        Args:
            time: Time t
            state: State vector [x, y, θ]
            
        Returns:
            Value function V(t,x). V ≤ 0 indicates avoidable/reachable states.
        """
        if isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Create coordinate tensor [t, x, y, θ]
        coord = torch.cat([torch.tensor([[time]], dtype=torch.float32), 
                          state.unsqueeze(0)], dim=-1)
        coord = coord.to(self.device)
        
        # Convert to model input
        model_input = self.dynamics.coord_to_input(coord)
        
        # Forward pass
        with torch.no_grad():
            output = self.model({'coords': model_input})['model_out'].squeeze(-1)
            value = self.dynamics.io_to_value(model_input, output)
        
        return float(value)

    def get_optimal_control(self, time: float, state: Union[list, np.ndarray, torch.Tensor]) -> float:
        """
        Compute optimal control at a given time and state.
        
        Args:
            time: Time t
            state: State vector [x, y, θ]
            
        Returns:
            Optimal control u* (angular velocity)
        """
        if isinstance(state, (list, np.ndarray)):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Create coordinate tensor [t, x, y, θ]
        coord = torch.cat([torch.tensor([[time]], dtype=torch.float32), 
                          state.unsqueeze(0)], dim=-1)
        coord = coord.to(self.device)
        
        # Convert to model input and enable gradients
        model_input = self.dynamics.coord_to_input(coord)
        model_input.requires_grad_(True)
        
        # Forward pass
        output = self.model({'coords': model_input})['model_out'].squeeze(-1)
        
        # Compute gradients
        dv = self.dynamics.io_to_dv(model_input, output)
        dvds = dv[..., 1:]  # ∂V/∂x
        real_state = coord[..., 1:]  # real state coordinates
        
        # Get optimal control
        u_star = self.dynamics.optimal_control(real_state, dvds)
        
        return float(u_star)

    def is_safe(self, time: float, state: Union[list, np.ndarray, torch.Tensor]) -> bool:
        """
        Check if a state is safe (avoidable) at a given time.
        
        Args:
            time: Time t
            state: State vector [x, y, θ]
            
        Returns:
            True if state is safe (avoidable), False otherwise
        """
        value = self.evaluate_value(time, state)
        return value > 0  # Safe if V > 0

    def generate_reachability_plot(self, time: float = 0.0, theta: float = 0.0, 
                                 resolution: int = 200, save_path: Optional[str] = None):
        """
        Generate a 2D reachability plot at fixed time and heading.
        
        Args:
            time: Time slice to visualize
            theta: Heading angle to visualize
            resolution: Grid resolution
            save_path: Path to save the plot (optional)
        """
        # Create grid
        x_range = np.linspace(-1, 1, resolution)
        y_range = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Flatten for batch processing
        coords = torch.zeros(resolution * resolution, 4)
        coords[:, 0] = time  # t
        coords[:, 1] = torch.from_numpy(X.flatten()).float()  # x
        coords[:, 2] = torch.from_numpy(Y.flatten()).float()  # y
        coords[:, 3] = theta  # θ
        coords = coords.to(self.device)
        
        # Convert to model input
        model_input = self.dynamics.coord_to_input(coords)
        
        # Evaluate value function
        with torch.no_grad():
            output = self.model({'coords': model_input})['model_out'].squeeze(-1)
            values = self.dynamics.io_to_value(model_input, output)
        
        # Reshape for plotting
        V = values.cpu().numpy().reshape(resolution, resolution)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, V, levels=50, cmap='RdYlBu_r')
        plt.colorbar(label='Value Function V(t,x)')
        plt.contour(X, Y, V, levels=[0], colors='black', linewidths=2, label='V=0 boundary')
        
        # Add target circle
        circle = plt.Circle((0, 0), self.dynamics.goalR, color='red', alpha=0.3, label='Target')
        plt.gca().add_patch(circle)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Reachability Set at t={time:.2f}, θ={theta:.2f}\nRed: V≤0 (avoidable), Blue: V>0 (safe)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

    def analyze_trajectory(self, states: np.ndarray, times: np.ndarray) -> dict:
        """
        Analyze a trajectory for safety and optimal controls.
        
        Args:
            states: Array of states [N, 3] with columns [x, y, θ]
            times: Array of times [N]
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'values': [],
            'controls': [],
            'safe_states': [],
            'min_value': float('inf'),
            'max_value': float('-inf')
        }
        
        for i, (t, state) in enumerate(zip(times, states)):
            value = self.evaluate_value(t, state)
            control = self.get_optimal_control(t, state)
            is_safe = self.is_safe(t, state)
            
            results['values'].append(value)
            results['controls'].append(control)
            results['safe_states'].append(is_safe)
            results['min_value'] = min(results['min_value'], value)
            results['max_value'] = max(results['max_value'], value)
        
        results['values'] = np.array(results['values'])
        results['controls'] = np.array(results['controls'])
        results['safe_states'] = np.array(results['safe_states'])
        results['safety_ratio'] = np.mean(results['safe_states'])
        
        return results


def load_deepreach_model(model_path: str, device: str = 'cpu') -> DeepReachModel:
    """
    Convenience function to load a trained DeepReach model.
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        Loaded DeepReachModel instance
    """
    return DeepReachModel(model_path, device)


if __name__ == "__main__":
    # Example usage
    print("DeepReach Model Deployment Script")
    print("=================================")
    print("This script provides a complete interface for using trained DeepReach models.")
    print("See example_usage.py for demonstration code.")
