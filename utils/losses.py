import torch

def init_stationary_discounted_loss(dynamics):
    def stationary_discounted_loss(state, value, dvdx):
        # state: [batch, state_dim]
        # value: [batch, 1]
        # dvdx: [batch, state_dim]
        
        # Compute Hamiltonian part: \nabla V \cdot f
        ham_dot = dynamics.hamiltonian(state, dvdx)
        
        # Full Hamiltonian: \nabla V \cdot f - \gamma V
        # Note: dynamics.gamma should be available
        ham = ham_dot - dynamics.gamma * value.squeeze(-1)
        
        # Constraint: g(x)
        g_x = dynamics.boundary_fn(state)
        
        # Residual: max(g(x) - V(x), H(x))
        # value is [batch, 1], g_x is [batch]
        residual = torch.maximum(g_x - value.squeeze(-1), ham)
        
        # Loss is mean absolute residual
        loss = torch.abs(residual).mean()
        
        return {'loss': loss, 'residual': residual}

    return stationary_discounted_loss
