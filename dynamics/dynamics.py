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
        # question: what does 'input' tensor look like?
        return coord

    # convert real coord to model input
    def coord_to_input(self, coord):
        input = coord.clone()
        input[..., 1:] = (coord[..., 1:] - self.state_mean.to(device=coord.device)) / self.state_var.to(device=coord.device)
        return input

    # convert model io to real value
    # Note: need to read again
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
    """
    Dynamics for a 3D Dubins Car with a discount factor.
    State: [x, y, theta]
    Control: [u1, u2] (velocity, angular velocity)
    Disturbance: [d1, d2] (additive disturbance on x, y)
    """
    def __init__(self, gamma:float, angle_alpha_factor:float, set_mode:str, value_normto:float):
        # Store discount factor
        self.gamma = gamma
        
        # Problem parameters
        self.L = 0.9           # State bound
        self.r = 0.3           # Obstacle radius
        self.Cx = 0.0          # Obstacle center x
        self.Cy = 0.0          # Obstacle center y
        
        # Control bounds
        self.u_min = torch.tensor([0.05, -1.0])   # [u1_min, u2_min]
        self.u_max = torch.tensor([1.0, 1.0])     # [u1_max, u2_max]
        
        # Disturbance bounds
        self.d_min = torch.tensor([-0.01, -0.01]) # [d1_min, d2_min]
        self.d_max = torch.tensor([0.01, 0.01])   # [d1_max, d2_max]
        
        # Angle scaling factor
        # question: what is the angle scaling factor? Why we have this parameter?
        angle_alpha_factor = angle_alpha_factor
        
        super().__init__(
            loss_type='brt_hjivi',      # Use standard BRT loss type
            set_mode=set_mode,           # Viability = avoid unsafe set
            state_dim=3,                # [x1, x2, x3]
            input_dim=4,                # [t, x1, x2, x3]
            control_dim=2,              # [u1, u2]
            disturbance_dim=2,          # [d1, d2]
            state_mean=[0, 0, 0],
            state_var=[self.L + 0.1, self.L + 0.1, angle_alpha_factor * math.pi], # question: how to set the state variance?
            value_mean=0.0,
            value_var=0.12,
            value_normto=value_normto,
            deepreach_model="vanilla"     # Use exact DeepReach model
        )

    def state_test_range(self):
        # use a slightly larger range than the state bounds
        return [
            [-self.L - 0.1, self.L + 0.1],  # x1
            [-self.L - 0.1, self.L + 0.1],  # x2
            [-math.pi, math.pi]            # x3 (theta)
        ]

    def boundary_fn(self, state):
        """
        g(x) = min{L-|x1|, L-|x2|, dist² - r²}
        Safe region: g(x) >= 0
        """
        x1 = state[..., 0]
        x2 = state[..., 1]
        
        # Box constraints
        g1 = self.L - torch.abs(x1)
        g2 = self.L - torch.abs(x2)
        
        # Circular obstacle (inside is unsafe)
        dist_sq = (x1 - self.Cx)**2 + (x2 - self.Cy)**2
        g3 = dist_sq - self.r**2
        
        return torch.minimum(torch.minimum(g1, g2), g3)

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
        
        # Optimal control (maximize)
        det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
        u1 = torch.where(det1 < 0, self.u_min[0].to(state.device), self.u_max[0].to(state.device))
        u2 = torch.where(p3 < 0, self.u_min[1].to(state.device), self.u_max[1].to(state.device))
        
        # Optimal disturbance (minimize)
        d1 = torch.where(p1 < 0, self.d_max[0].to(state.device), self.d_min[0].to(state.device))
        d2 = torch.where(p2 < 0, self.d_max[1].to(state.device), self.d_min[1].to(state.device))
        
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
        u1 = torch.where(det1 < 0, self.u_min[0].to(state.device), self.u_max[0].to(state.device))
        u2 = torch.where(p3 < 0, self.u_min[1].to(state.device), self.u_max[1].to(state.device))
        
        return torch.stack([u1, u2], dim=-1)

    def optimal_disturbance(self, state, dvds):
        """Return optimal disturbance for simulation"""
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        
        d1 = torch.where(p1 < 0, self.d_max[0].to(state.device), self.d_min[0].to(state.device))
        d2 = torch.where(p2 < 0, self.d_max[1].to(state.device), self.d_min[1].to(state.device))
        
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

class TwoDubins3DLeaderFollower(Dynamics):
    """
    Dynamics for 2 3D Dubins Car leader-follower system
    Infinite Horizon with a discount factor.
    State: [x1, y1, theta1, x2, y2, theta2] for follower and leader
    Control: [u1, u2] (velocity, angular velocity for follower)
    Disturbance: [d1, d2] (velocity and angular velocity for leader)
    """
    def __init__(self, gamma:float, angle_alpha_factor:float, set_mode:str, value_normto:float):
        # Store discount factor
        self.gamma = gamma
        
        # Problem parameters
        self.L = 5.0           # State bound
        self.r = 2.0           # Safe distance between leader and follower
        # Control bounds
        self.u_min = torch.tensor([0, -0.4])   # [u1_min, u2_min]
        self.u_max = torch.tensor([1.0, 0.4])     # [u1_max, u2_max]
        
        # Disturbance bounds
        self.d_min = torch.tensor([0, -0.4]) # [d1_min, d2_min]
        self.d_max = torch.tensor([0.5, 0.4])   # [d1_max, d2_max]
        
        # Angle scaling factor
        # question: what is the angle scaling factor? Why we have this parameter?
        angle_alpha_factor = angle_alpha_factor
        
        super().__init__(
            loss_type='brt_hjivi',      # Use standard BRT loss type
            set_mode=set_mode,           # Viability = avoid unsafe set
            state_dim=6,                # [x1, y1, theta1, x2, y2, theta2]
            input_dim=7,                # Neural network input:[t, x1, y1, theta1, x2, y2, theta2]
            control_dim=2,              # [u1, u2]
            disturbance_dim=2,          # [d1, d2]
            state_mean=[0, 0, 0, 0, 0, 0],
            state_var=[self.L, self.L, angle_alpha_factor * math.pi, self.L, self.L, angle_alpha_factor * math.pi],
            value_mean=0.0,
            value_var=4.0,
            value_normto=value_normto,
            deepreach_model="vanilla"     # Use exact DeepReach model
        )

    def state_test_range(self):
        # use a slightly larger range than the state bounds
        return [
            [-self.L - 0.1, self.L + 0.1],  # x1
            [-self.L - 0.1, self.L + 0.1],  # x2
            [-math.pi, math.pi],            # x3 (theta)
            [-self.L - 0.1, self.L + 0.1],  # x4
            [-self.L - 0.1, self.L + 0.1],  # x5
            [-math.pi, math.pi],            # x6 (theta)
        ]

    def boundary_fn(self, state):
        """
        g(x) = min{L-|x1|, L-|x2|, L-|x4|, L-|x5|, dist² - r²}
        Safe region: g(x) >= 0
        """
        x1 = state[..., 0]
        x2 = state[..., 1]
        x4 = state[..., 3]
        x5 = state[..., 4]
        
        # Box constraints
        g1 = self.L - torch.abs(x1)
        g2 = self.L - torch.abs(x2)
        g3 = self.L - torch.abs(x4)
        g4 = self.L - torch.abs(x5)
        
        # Circular obstacle (inside is unsafe)
        dist_sq = (x1 - x4)**2 + (x2 - x5)**2
        g5 = self.r**2 - dist_sq
        
        return torch.minimum(torch.minimum(torch.minimum(torch.minimum(g1, g2), g3), g4), g5)

    def hamiltonian(self, state, dvds, value):
        """
        Computes H = min_u max_d [p·f(x,u,d) - γV]
        
        Args:
            state: [batch, 6] - (x1, y1, theta1, x2, y2, theta2)
            dvds: [batch, 6] - spatial gradient (p1, p2, p3, p4, p5, p6)
            value: [batch] or [batch, 1] - V(x,t)
        
        Returns:
            ham: [batch] - Hamiltonian value
        """
        x3 = state[..., 2]
        x6 = state[..., 5]
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        p3 = dvds[..., 2]
        p4 = dvds[..., 3]
        p5 = dvds[..., 4]
        p6 = dvds[..., 5]

        # Optimal control (maximize)
        det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
        u1 = torch.where(det1 < 0, self.u_min[0].to(state.device), self.u_max[0].to(state.device))
        u2 = torch.where(p3 < 0, self.u_min[1].to(state.device), self.u_max[1].to(state.device))
        
        # Optimal disturbance (minimize)
        det2 = p4 * torch.cos(x6) + p5 * torch.sin(x6)
        d1 = torch.where(det2 < 0, self.d_max[0].to(state.device), self.d_min[0].to(state.device))
        d2 = torch.where(p6 < 0, self.d_max[1].to(state.device), self.d_min[1].to(state.device))
        
        # Hamiltonian: p·f - γV
        # Ensure value is [batch] shape
        if value.dim() > 1:
            value = value.squeeze(-1)
        
        ham = p1 * u1 * torch.cos(x3) + \
              p2 * u1 * torch.sin(x3) + \
              p3 * u2 + \
              p4 * d1 * torch.cos(x6) + \
              p5 * d1 * torch.sin(x6) + \
              p6 * d2 - \
              self.gamma * value  # NEW: discount term
              
        return ham

    def equivalent_wrapped_state(self, state):
        wrapped_state = torch.clone(state)
        wrapped_state[..., 2] = (wrapped_state[..., 2] + math.pi) % (2*math.pi) - math.pi
        wrapped_state[..., 5] = (wrapped_state[..., 5] + math.pi) % (2*math.pi) - math.pi
        return wrapped_state

    def dsdt(self, state, control, disturbance):
        """Dynamics for testing/simulation"""
        dsdt = torch.zeros_like(state)
        dsdt[..., 0] = control[..., 0] * torch.cos(state[..., 2])
        dsdt[..., 1] = control[..., 0] * torch.sin(state[..., 2])
        dsdt[..., 2] = control[..., 1]
        dsdt[..., 3] = disturbance[..., 0] * torch.cos(state[..., 5])
        dsdt[..., 4] = disturbance[..., 0] * torch.sin(state[..., 5])
        dsdt[..., 5] = disturbance[..., 1]
        return dsdt

    def optimal_control(self, state, dvds):
        """Return optimal control for simulation"""
        x3 = state[..., 2]
        p1 = dvds[..., 0]
        p2 = dvds[..., 1]
        p3 = dvds[..., 2]
        
        det1 = p1 * torch.cos(x3) + p2 * torch.sin(x3)
        u1 = torch.where(det1 < 0, self.u_min[0].to(state.device), self.u_max[0].to(state.device))
        u2 = torch.where(p3 < 0, self.u_min[1].to(state.device), self.u_max[1].to(state.device))
        
        return torch.stack([u1, u2], dim=-1)

    def optimal_disturbance(self, state, dvds):
        """Return optimal disturbance for simulation"""
        x6 = state[..., 5]
        p4 = dvds[..., 3]
        p5 = dvds[..., 4]
        p6 = dvds[..., 5]
        
        det2 = p4 * torch.cos(x6) + p5 * torch.sin(x6)
        d1 = torch.where(det2 < 0, self.d_max[0].to(state.device), self.d_min[0].to(state.device))
        d2 = torch.where(p6 < 0, self.d_max[1].to(state.device), self.d_min[1].to(state.device))
        
        return torch.stack([d1, d2], dim=-1)

    def plot_config(self):
        return {
            'state_slices': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'state_labels': [r'x1', r'y1', r'$\theta1$', r'x2', r'y2', r'$\theta2$'],
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
