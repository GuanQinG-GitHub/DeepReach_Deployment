Training Command for the IH
```python
  python run_experiment.py --mode train --experiment_class DeepReach --dynamics_class Dubins3DDiscounted --gamma 0.7 --experiment_name dubins3dDiscounted_trial_4 --minWith target --angle_alpha_factor 1.2 --set_mode avoid --num_epochs 300000  --tMax 5 --use_wandb       
```

Note:
- Start to use this log from Dec. 3rd, in HPC
- tested, it doesn't save time if commenting out the plotting part
## Log
