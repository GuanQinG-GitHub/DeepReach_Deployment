import torch

def init_stationary_discounted_loss(dynamics):
    def stationary_discounted_loss(state, value, dvdx):
        """
        Computes the loss for the stationary discounted Hamilton-Jacobi PDE.
        The PDE is: max(g(x) - V(x), H(x, \nabla V) - \gamma V(x)) = 0
        """
        # state: [batch, state_dim]
        # value: [batch, 1]
        # dvdx: [batch, state_dim]
        
        # Compute Hamiltonian part: \nabla V \cdot f
        ham_dot = dynamics.hamiltonian(state, dvdx)
        
        # Full Hamiltonian: \nabla V \cdot f - \gamma V
        # Note: dynamics.gamma should be available
        # squeeze(-1) is used to remove the last dimension of the value tensor for dimension matching
        ham = ham_dot - dynamics.gamma * value.squeeze(-1)
        
        # Constraint: g(x)
        # g(x) defines the obstacle/target set.
        g_x = dynamics.boundary_fn(state)
        
        # Residual: max(g(x) - V(x), H(x) - gamma*V(x))
        # We want this residual to be 0.
        # If V(x) satisfies the PDE, then either:
        # 1. V(x) = g(x) (on the boundary)
        # 2. H(x) - gamma*V(x) = 0 (optimal control holds)
        # value is [batch, 1], g_x is [batch]
        residual = torch.maximum(g_x - value.squeeze(-1), ham)
        
        # Loss is sum absolute residual
        loss = torch.abs(residual).sum()
        
        return {'loss': loss, 'residual': residual}

    return stationary_discounted_loss
