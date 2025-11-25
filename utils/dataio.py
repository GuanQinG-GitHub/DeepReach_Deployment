import torch
from torch.utils.data import Dataset

class StationaryReachabilityDataset(Dataset):
    """
    Dataset for stationary reachability.
    Samples uniform random points in the state space.
    """
    def __init__(self, dynamics, numpoints, **kwargs):
        self.dynamics = dynamics
        self.numpoints = numpoints
        
        # Get state bounds from dynamics
        self.state_range = self.dynamics.state_test_range()
        self.low = torch.tensor([r[0] for r in self.state_range])
        self.high = torch.tensor([r[1] for r in self.state_range])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Uniformly sample state space
        # state: [numpoints, state_dim]
        # We sample 'numpoints' points at every iteration.
        # This means every batch is a fresh set of random points.
        state = torch.rand(self.numpoints, self.dynamics.state_dim) * (self.high - self.low) + self.low
        
        # Return dictionary compatible with model input
        # The model expects 'coords'
        return {'coords': state}, {}
