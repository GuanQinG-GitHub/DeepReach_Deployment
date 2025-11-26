## Test 1
- Nov. 25, 2025
- https://wandb.ai/xzhan245-nc-state-university/DeepReach-Deployment/runs/7pcl3ezc?nw=nwuserxzhan245
- Result:
    - In most epochs, we just see the obstacle region, not the viability kernel
    - In some epoch, we see nothing from the plot
- Issue:
    - why the loss from 0.1 and the converge process is slow, but seems to converge
    - starting from random values, maybe we can start from the boundary value 
- Need to address:
    - add a 3D plot to examine
    - start from the obstacle boundary


## Test 2
- Nov. 25, 2025
- https://wandb.ai/xzhan245-nc-state-university/DeepReach-Deployment/runs/nnfscwm0?nw=nwuserxzhan245
- Summary:
    - use the terminal condition to warmup the network, with pretrain steps 10k
    - added the 3D plot to examine the viability kernel
- Result:
    - the first iterations cannot learn the terminal condition
    - it learns nothing basically
- Issue:
    - the loss is the mean residual, which is different from the original script which uses the sum of the residual
- Next move:
    - remove the pretrain part, since it's not helpful at this stage
    - switch to the sum of the residual
    - increase the iterations and later tuning the learning rate

- Next
    - Examine the loss, why each point has such a small loss?
    - Try to add the time dependency, following Somil's work