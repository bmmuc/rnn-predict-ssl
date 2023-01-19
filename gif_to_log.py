import wandb
import os

indexes = [3,5,8,10,20]
run = wandb.init(name='batch:128-hidden_layer_size: 256-weight_decay:0.0-epochs:10-hidden_is_with_noise:False n_steps_future')

for j in indexes:
    i = 1
    for file in os.listdir(f'./gifs/predict_n_steps/{j}/'):
        run.log({f"gif-steps_future-{j}-n_gif-{i}": wandb.Video(f'./gifs/predict_n_steps/{j}/{file}', fps=30, format="gif")})
        i+= 1
