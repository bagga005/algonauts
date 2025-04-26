import utils
import wandb

print('hello world 2')

wandb_config = {
    'epochs': 10
}
project_name, model_name = utils.get_wandb_config()
wandb.init(
    id=model_name,
    project=project_name,
    name=model_name,
    config=wandb_config,
    resume="allow",
)
steps =1
logs = {
    'train/loss': 1,
    'train/num_steps': steps,
    'test/loss': 1,
    'test/num_steps': steps
}
wandb.log(logs)
steps = 2
logs = {
    'train/loss': 2,
    'train/num_steps': steps,
    'test/loss': 2,
    'test/num_steps': steps
}
wandb.log(logs)