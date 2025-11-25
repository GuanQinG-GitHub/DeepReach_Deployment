import configargparse
import os
import torch
import shutil
import random
import numpy as np
import wandb
from datetime import datetime

from dynamics import dynamics
from experiments import experiments
from utils import modules, dataio, losses

# This script is the main entry point for training.
# It handles:
# 1. Argument parsing (hyperparameters)
# 2. Initialization of all components (Dynamics, Dataset, Model, Loss, Experiment)
# 3. Execution of the training loop

def main():
    p = configargparse.ArgumentParser()
    p.add_argument('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
    
    # Experiment setup
    p.add_argument('--mode', type=str, required=True, choices=['train'], help="Experiment mode.")
    p.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment.')
    p.add_argument('--experiments_dir', type=str, default='./runs', help='Directory to save results.')
    p.add_argument('--use_wandb', action='store_true', default=False, help='Use WandB for logging.')
    p.add_argument('--seed', type=int, default=0, help='Random seed.')
    p.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0, cpu).')
    
    # WandB
    p.add_argument('--wandb_project', type=str, default='DeepReach-Deployment', help='WandB project name.')
    p.add_argument('--wandb_entity', type=str, default=None, help='WandB entity.')
    
    # Dynamics
    p.add_argument('--dynamics_class', type=str, default='Dubins3D', help='Dynamics class to use.')
    p.add_argument('--gamma', type=float, default=0.0, help='Discount factor.')
    
    # Model
    p.add_argument('--model_type', type=str, default='sine', choices=['sine', 'relu', 'tanh'], help='Model activation type.')
    p.add_argument('--num_hl', type=int, default=3, help='Number of hidden layers.')
    p.add_argument('--num_nl', type=int, default=512, help='Number of neurons per layer.')
    
    # Training
    p.add_argument('--num_epochs', type=int, default=10000, help='Number of epochs.')
    p.add_argument('--batch_size', type=int, default=1, help='Batch size (number of dataset items per step).')
    p.add_argument('--numpoints', type=int, default=65000, help='Number of points per batch.')
    p.add_argument('--lr', type=float, default=2e-5, help='Learning rate.')
    p.add_argument('--steps_til_summary', type=int, default=100, help='Steps until summary.')
    p.add_argument('--epochs_til_ckpt', type=int, default=1000, help='Epochs until checkpoint.')
    p.add_argument('--pretrain_boundary', action='store_true', default=False, help='Pretrain network to match boundary function g(x).')
    p.add_argument('--pretrain_iters', type=int, default=2000, help='Number of pretraining iterations.')
    
    # Experiment Class
    p.add_argument('--experiment_class', type=str, default='StationaryExperiment', help='Experiment class to use.')

    # Parse arguments. configargparse allows providing arguments via config file or command line.
    opt = p.parse_args()

    # Set seeds
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # Setup WandB
    if opt.use_wandb:
        wandb.init(project=opt.wandb_project, entity=opt.wandb_entity, name=opt.experiment_name, config=opt)

    # Setup directories
    experiment_dir = os.path.join(opt.experiments_dir, opt.experiment_name)
    if os.path.exists(experiment_dir):
        print(f"Warning: Experiment directory {experiment_dir} already exists.")
        # In a real script we might ask, but for automation we'll proceed or overwrite if needed.
        # For now, let's just use it.
    else:
        os.makedirs(experiment_dir)

    # Initialize Dynamics
    # The dynamics class defines the physics of the system (f(x,u,d)) and the Hamiltonian.
    dynamics_class = getattr(dynamics, opt.dynamics_class)
    # Pass gamma if it's in the constructor
    # We assume Dubins3D takes gamma
    dyn = dynamics_class(gamma=opt.gamma)

    # Initialize Dataset
    # The dataset samples random points in the state space.
    # Unlike standard supervised learning, we don't have fixed labels.
    # The "label" is the physics constraint enforced by the loss function.
    dataset = dataio.StationaryReachabilityDataset(dynamics=dyn, numpoints=opt.numpoints)

    # Initialize Model
    # We use a SingleBVPNet (Multi-Layer Perceptron) to approximate the value function V(x).
    # It uses Sine activations (SIREN) to allow for accurate derivative computation.
    model = modules.SingleBVPNet(in_features=dyn.state_dim, out_features=1, type=opt.model_type, 
                                 hidden_features=opt.num_nl, num_hidden_layers=opt.num_hl)
    model.to(opt.device)

    # Initialize Loss
    # The loss function enforces the Hamilton-Jacobi PDE.
    # loss = |max(g(x) - V(x), Hamiltonian - gamma*V(x))|
    loss_fn = losses.init_stationary_discounted_loss(dyn)

    # Initialize Experiment
    # The experiment class manages the training loop, logging, and validation.
    experiment_class = getattr(experiments, opt.experiment_class)
    experiment = experiment_class(model=model, dataset=dataset, experiment_dir=experiment_dir, use_wandb=opt.use_wandb)

    # Train
    experiment.train(device=opt.device, batch_size=opt.batch_size, epochs=opt.num_epochs, lr=opt.lr,
                     steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
                     loss_fn=loss_fn, pretrain_boundary=opt.pretrain_boundary, pretrain_iters=opt.pretrain_iters)

if __name__ == '__main__':
    main()
