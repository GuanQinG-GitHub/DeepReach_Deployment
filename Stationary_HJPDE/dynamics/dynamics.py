import torch
import numpy as np
from abc import ABC, abstractmethod

# StationaryDynamics is an abstract base class (inherits from abc.ABC)
class StationaryDynamics(ABC):
# This class inherits from `ABC`, making it an Abstract Base Class.
# Methods marked with `@abstractmethod` must be overridden by any concrete subclass.
# Instantiating a subclass without implementing all abstract methods raises a TypeError.
    def __init__(self, state_dim, input_dim, control_dim, disturbance_dim, set_mode='avoid'):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.set_mode = set_mode

    @abstractmethod
    def hamiltonian(self, state, dvdx):
        raise NotImplementedError

    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError
    
    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    def coord_to_input(self, coord):
        return coord

    def input_to_coord(self, input):
        return input

    def io_to_value(self, input, output):
        return output

    def io_to_dv(self, input, output):
        return torch.autograd.grad(output, input, grad_outputs=torch.ones_like(output), create_graph=True)[0]

class Dubins3D(StationaryDynamics):
    """
    Dynamics for a 3D Dubins Car.
    State: [x, y, theta]
    Control: [v, w] (velocity, angular velocity)
    Disturbance: [d1, d2] (additive disturbance on x, y)
    """
    def __init__(self, gamma=0.0):
        super().__init__(state_dim=3, input_dim=3, control_dim=2, disturbance_dim=2, set_mode='avoid')
        self.gamma = gamma
        
        # Problem parameters from Toy_example.md
        self.L = 0.9 # Assuming L=1.0
        self.r = 0.3 # Assuming r=0.3
        self.Cx = 0.0 # Assuming Cx=0.0
        self.Cy = 0.0 # Assuming Cy=0.0
        
        # Control bounds
        self.uMin = torch.tensor([0.05, -1.0])
        self.uMax = torch.tensor([1.0, 1.0])
        
        # Disturbance bounds
        self.dMin = torch.tensor([-0.01, -0.01])
        self.dMax = torch.tensor([0.01, 0.01])

    def hamiltonian(self, state, dvdx):
        """
        Computes the Hamiltonian: H(x, p) = min_u max_d (p \cdot f(x, u, d))
        where p = dvdx (gradient of value function).
        
        This function implements the optimal control logic.
        It analytically finds the optimal u and d that minimize/maximize the dot product.
        """
        # input arguments state and dvdx should be torch tensors
        # state: [batch, 3] (x1, x2, x3)
        # dvdx: [batch, 3] (p1, p2, p3)
        
        x3 = state[..., 2]   # take the 3rd component (index 2) of the state vector for every sample
        p1 = dvdx[..., 0]    # take the 1st gradient component for every sample
        p2 = dvdx[..., 1]    # take the 2nd gradient component for every sample
        p3 = dvdx[..., 2]    # take the 3rd gradient component for every sample
        
        # Optimal Control (minimize Hamiltonian)
        # term1 = p1*u1*cos(x3) + p2*u1*sin(x3) = u1 * (p1*cos(x3) + p2*sin(x3))
        # if coeff > 0, choose uMin1, else uMax1
        det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
        u1 = torch.where(det1 > 0, self.uMin[0], self.uMax[0]) # vectorized conditional
        
        # term2 = p3*u2
        # if p3 > 0, choose uMin2, else uMax2
        u2 = torch.where(p3 > 0, self.uMin[1], self.uMax[1]) # vectorized conditional
        
        # Optimal Disturbance (maximize Hamiltonian)
        # term3 = p1*d1
        # if p1 > 0, choose dMax1, else dMin1
        d1 = torch.where(p1 > 0, self.dMax[0], self.dMin[0])
        
        # term4 = p2*d2
        # if p2 > 0, choose dMax2, else dMin2
        d2 = torch.where(p2 > 0, self.dMax[1], self.dMin[1])
        
        # Hamiltonian
        # H = p1(u1_opt*cos(x3)+d1_opt) + p2(u1_opt*sin(x3)+d2_opt) + p3*u2_opt
        # Note that -gamma*V is handled in the loss function
        
        ham = p1 * (u1 * torch.cos(x3) + d1) + \
              p2 * (u1 * torch.sin(x3) + d2) + \
              p3 * u2
              
        return ham

    def boundary_fn(self, state):
        """
        Defines the target/obstacle set g(x).
        The safe region is defined by V(x) <= 0.
        Here, g(x) represents the obstacle (cylinder) and the state bounds.
        If g(x) <= 0, the state is safe (outside obstacle, inside bounds).
        """
        # input argument, state, is a torch tensor
        # g(x) = max(|x1|-L, |x2|-L, r^2 - (x1-Cx)^2 - (x2-Cy)^2)
        x1 = state[..., 0]
        x2 = state[..., 1]
        
        # Box constraints
        g1 = torch.abs(x1) - self.L
        g2 = torch.abs(x2) - self.L
        
        # Obstacle constraint (inside circle is unsafe)
        # g(x) <= 0 is safe.
        dist_sq = (x1 - self.Cx)**2 + (x2 - self.Cy)**2
        g3 = self.r**2 - dist_sq
        
        return torch.maximum(torch.maximum(g1, g2), g3)

    def plot_config(self):
        return {
            'state_slices': [0.0, 0.0, 0.0], # Default slice
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
            'state_labels': ['x', 'y', 'theta']
        }

    def state_test_range(self):
        return [
            [-self.L-0.1, self.L+0.1],
            [-self.L-0.1, self.L+0.1],
            [-np.pi, np.pi]
        ]
