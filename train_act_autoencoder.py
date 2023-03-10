from act_autoencoder import ActAutoEncoder
from src.concat_data_set_autoencoder import ConcatDataSetAutoencoder
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import torch
import wandb
import ipdb
from tqdm import tqdm
from datetime import datetime
import pytz

BATCH_SIZE = 256
HIDDEN_SIZE = 256
WINDOW_SIZE = 10
INPUT_SIZE = 38
EPOCHS = 100
NUM_WORKERS = 15
LR = 1e-4
HORIZON_SIZE = 1
dataset = ConcatDataSetAutoencoder(
    root_dir='../all_data/data-3v3-v2',
    num_of_data_sets=100,
    window=WINDOW_SIZE,
    is_pos=False,
    should_test_the_new_data_set=True,
    type_of_data='train',
    horizon=HORIZON_SIZE
)


train_loader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


dataset = ConcatDataSetAutoencoder(
    root_dir='../all_data/data-3v3-v2',
    num_of_data_sets=100,
    window=WINDOW_SIZE,
    is_pos=False,
    should_test_the_new_data_set=True,
    type_of_data='val',
    horizon=HORIZON_SIZE
)

val_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)
# raise Exception('stop')
model = ActAutoEncoder(
    window=WINDOW_SIZE,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=INPUT_SIZE,
    lr=LR,
)
today = datetime.now(pytz.timezone('America/Sao_Paulo')
                     ).strftime("%Y-%m-%d_%H:%M:%S")

wandb.init(project="ssl_env_acts", entity="breno-cavalcanti", name=f"act_autoencoder_{today}",
           config={
               "batch_size": BATCH_SIZE,
               "hidden_size": HIDDEN_SIZE,
               "window_size": WINDOW_SIZE,
               "input_size": INPUT_SIZE,
               "epochs": EPOCHS,
               "lr": LR,
               "horizon": HORIZON_SIZE
           })

wandb.define_metric("epoch")


wandb.watch(model)

steps = 0
epochs = 0
for epoch in tqdm(range(EPOCHS)):
    wandb.log({'epoch': epochs})
    general_val_loss = 0

    for x, y in tqdm(val_loader):
        x = x.to(model.device)
        y = y.to(model.device)
        loss = model.validation_step(x, y)

        general_val_loss += loss.item()

    general_loss = 0
    for x, y in tqdm(train_loader):
        x = x.to(model.device)
        y = y.to(model.device)
        loss, ave_grads, norm_grad, total_norm = model.training_step(x, y)
        steps += 1
        loss_dict = {
            'global_step': steps,
            'loss_act_train/step': loss.item(),
        }

        wandb.log(loss_dict)

        general_loss += loss.item()

    log_dict = {
        'loss_act_train/epoch': general_loss / len(train_loader),
        'loss_act_val/epoch': general_val_loss / len(val_loader),
    }

    wandb.log(log_dict)
    epochs += 1


torch.save(model.state_dict(), './model_with_ball_v.pth')
