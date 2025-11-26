from abc import ABC, abstractmethod
from utils import diff_operators

import math
import torch

# during training, states will be sampled uniformly by each state dimension from the model-unit -1 to 1 range (for training stability),
# which may or may not correspond to proper test ranges
# note that coord refers to [time, *state], and input refers to whatever is fed directly to the model (often [time, *state, params])
# in the future, code will need to be fixed to correctly handle parameterized models
class Dynamics(ABC):
    def __init__(self, 
    loss_type:str, set_mode:str, 
    state_dim:int, input_dim:int, 
    control_dim:int, disturbance_dim:int, 
    state_mean:list, state_var:list, 
    value_mean:float, value_var:float, value_normto:float, 
    deepreach_model:str):
        self.loss_type = loss_type
        self.set_mode = set_mode
        self.state_dim = state_dim 
        self.input_dim = input_dim
        self.control_dim = control_dim
        self.disturbance_dim = disturbance_dim
        self.state_mean = torch.tensor(state_mean) 
        self.state_var = torch.tensor(state_var)
        self.value_mean = value_mean
        self.value_var = value_var
        self.value_normto = value_normto
        self.deepreach_model = deepreach_model
        assert self.loss_type in ['brt_hjivi', 'brat_hjivi'], f'loss type {self.loss_type} not recognized'
        if self.loss_type == 'brat_hjivi':
            assert callable(self.reach_fn) and callable(self.avoid_fn)
        assert self.set_mode in ['reach', 'avoid'], f'set mode {self.set_mode} not recognized'
        for state_descriptor in [self.state_mean, self.state_var]:
            assert len(state_descriptor) == self.state_dim, 'state descriptor dimension does not equal state dimension, ' + str(len(state_descriptor)) + ' != ' + str(self.state_dim)
    
    # ALL METHODS ARE BATCH COMPATIBLE

    # MODEL-UNIT CONVERSIONS (TODO: refactor into separate model-unit conversion class?)

    # convert model input to real coord
    def input_to_coord(self, input):
        coord = input.clone()
        coord[..., 1:] = (input[..., 1:] * self.state_var.to(device=input.device)) + self.state_mean.to(device=input.device)
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    def io_to_value(self, input, output):
        if self.deepreach_model=="diff":
            return (output * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        elif self.deepreach_model=="exact":
            return (output * input[..., 0] * self.value_var / self.value_normto) + self.boundary_fn(self.input_to_coord(input)[..., 1:])
        else:
            return (output * self.value_var / self.value_normto) + self.value_mean

    # convert model io to real dv
    def io_to_dv(self, input, output):
        dodi = diff_operators.jacobian(output.unsqueeze(dim=-1), input)[0].squeeze(dim=-2)

        if self.deepreach_model=="diff":
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]

            dvds_term1 = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        elif self.deepreach_model=="exact":
            dvdt = (self.value_var / self.value_normto) * \
                (input[..., 0]*dodi[..., 0] + output)

            dvds_term1 = (self.value_var / self.value_normto /
                          self.state_var.to(device=dodi.device)) * dodi[..., 1:] * input[..., 0].unsqueeze(-1)
            state = self.input_to_coord(input)[..., 1:]
            dvds_term2 = diff_operators.jacobian(self.boundary_fn(
                state).unsqueeze(dim=-1), state)[0].squeeze(dim=-2)
            dvds = dvds_term1 + dvds_term2
        else:
            dvdt = (self.value_var / self.value_normto) * dodi[..., 0]
            dvds = (self.value_var / self.value_normto / self.state_var.to(device=dodi.device)) * dodi[..., 1:]
        
        return torch.cat((dvdt.unsqueeze(dim=-1), dvds), dim=-1)

    # ALL FOLLOWING METHODS USE REAL UNITS

    @abstractmethod
    def state_test_range(self):
        raise NotImplementedError

    @abstractmethod
    def equivalent_wrapped_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def dsdt(self, state, control, disturbance):
        raise NotImplementedError
    
    @abstractmethod
    def boundary_fn(self, state):
        raise NotImplementedError

    @abstractmethod
    def sample_target_state(self, num_samples):
        raise NotImplementedError

    @abstractmethod
    def cost_fn(self, state_traj):
        raise NotImplementedError

    @abstractmethod
    def hamiltonian(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_control(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def optimal_disturbance(self, state, dvds):
        raise NotImplementedError

    @abstractmethod
    def plot_config(self):
        raise NotImplementedError

class Dubins3DDiscounted(Dynamics):
    def __init__(self, 
                 gamma: float,           # NEW: discount factor
                 L: float,               # state bound
                 r: float,               # obstacle radius
                 Cx: float,              # obstacle center x
                 Cy: float,              # obstacle center y
                 u1_min: float,          # control bounds
                 u1_max: float,
                 u2_min: float,
                 u2_max: float,
                 d1_min: float,          # disturbance bounds
                 d1_max: float,
                 d2_min: float,
                 d2_max: float,
                 angle_alpha_factor: float):
        
        self.gamma = gamma              # NEW: store discount factor
        self.L = L
        self.r = r
        self.Cx = Cx
        self.Cy = Cy
        self.u_min = torch.tensor([u1_min, u2_min])
        self.u_max = torch.tensor([u1_max, u2_max])
        self.d_min = torch.tensor([d1_min, d2_min])
        self.d_max = torch.tensor([d1_max, d2_max])
        
        super().__init__(
            loss_type='brt_hjivi',      # Use standard BRT loss type
            set_mode='avoid',           # Viability = avoid unsafe set
            state_dim=3,                # [x1, x2, x3]
            input_dim=4,                # [t, x1, x2, x3]
            control_dim=2,              # [u1, u2]
            disturbance_dim=2,          # [d1, d2]
            state_mean=[0, 0, 0],
            state_var=[L, L, angle_alpha_factor * math.pi],
            value_mean=0.0,
            value_var=1.0,
            value_normto=0.02,
            deepreach_model="exact"     # Use exact DeepReach model
        )

    def state_test_range(self):
        return [
            [-self.L - 0.1, self.L + 0.1],  # x1
            [-self.L - 0.1, self.L + 0.1],  # x2
            [-math.pi, math.pi],            # x3 (theta)
        ]

    def boundary_fn(self, state):
        """
        g(x) = max{|x1|-L, |x2|-L, r² - dist²}
        Safe region: g(x) ≤ 0
        """
        x1 = state[..., 0]
        x2 = state[..., 1]
        
        # Box constraints
        g1 = torch.abs(x1) - self.L
        g2 = torch.abs(x2) - self.L
        
        # Circular obstacle (inside is unsafe)
        dist_sq = (x1 - self.Cx)**2 + (x2 - self.Cy)**2
        g3 = self.r**2 - dist_sq
        
        return torch.maximum(torch.maximum(g1, g2), g3)

    def hamiltonian(self, state, dvds, value):
        """
        Computes H = min_u max_d [p·f(x,u,d) - γV]
        
        Args:
            state: [batch, 3] - (x1, x2, x3)
            dvds: [batch, 3] - spatial gradient (p1, p2, p3)
            value: [batch] or [batch, 1] - V(x,t)
        
        Returns:
            ham: [batch] - Hamiltonian value
        """
        x3 = state[..., 2]
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        p3 = dvds[..., 2]
        
        # Optimal control (minimize)
        det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
        u1 = torch.where(det1 > 0, self.u_min[0].to(state.device), self.u_max[0].to(state.device))
        u2 = torch.where(p3 > 0, self.u_min[1].to(state.device), self.u_max[1].to(state.device))
        
        # Optimal disturbance (maximize)
        d1 = torch.where(p1 > 0, self.d_max[0].to(state.device), self.d_min[0].to(state.device))
        d2 = torch.where(p2 > 0, self.d_max[1].to(state.device), self.d_min[1].to(state.device))
        
        # Hamiltonian: p·f - γV
        # Ensure value is [batch] shape
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        ham = p1 * (u1 * torch.cos(x3) + d1) + \
              p2 * (u1 * torch.sin(x3) + d2) + \
              p3 * u2 - \
              self.gamma * value  # NEW: discount term
              
        return ham

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        """Dynamics for testing/simulation"""
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = control[..., 0] * torch.cos(state[..., 2]) + disturbance[..., 0]
        dsdt[..., 1] = control[..., 0] * torch.sin(state[..., 2]) + disturbance[..., 1]
        dsdt[..., 2] = control[..., 1]
        return dsdt

    def optimal_control(self, state, dvds):
        """Return optimal control for simulation"""
        x3 = state[..., 2]
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        p3 = dvds[..., 2]
        
        det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
        u1 = torch.where(det1 > 0, self.u_min[0].to(state.device), self.u_max[0].to(state.device))
        u2 = torch.where(p3 > 0, self.u_min[1].to(state.device), self.u_max[1].to(state.device))
        
        return torch.stack([u1, u2], dim=-1)

    def optimal_disturbance(self, state, dvds):
        """Return optimal disturbance for simulation"""
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        
        d1 = torch.where(p1 > 0, self.d_max[0].to(state.device), self.d_min[0].to(state.device))
        d2 = torch.where(p2 > 0, self.d_max[1].to(state.device), self.d_min[1].to(state.device))
        
        return torch.stack([d1, d2], dim=-1)

    def plot_config(self):
        return {
            'state_slices': [0.0, 0.0, 0.0],
            'state_labels': ['x', 'y', r'$\theta$'],
            'x_axis_idx': 0,
            'y_axis_idx': 1,
            'z_axis_idx': 2,
        }

    def sample_target_state(self, num_samples):
        # Not needed for viability
        raise NotImplementedError

    def cost_fn(self, state_traj):
        # For testing: min_t g(x(t))
        return torch.min(self.boundary_fn(state_traj), dim=-1).values
