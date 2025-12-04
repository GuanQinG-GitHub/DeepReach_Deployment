import wandb

wandb.login()  # If not logged in, will ask for your API key
wandb.init(project="test_hpc")

for i in range(10):
    wandb.log({"step": i, "loss": 1.0 / (i + 1)})

print("WandB test finished")

