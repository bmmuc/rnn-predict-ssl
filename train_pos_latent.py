from pos_latent import PosLatent
from src.concat_data_set import ConcatDataSet
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import torch
import wandb
import ipdb
from tqdm import tqdm
from datetime import datetime
import pytz
import os

BATCH_SIZE = 32
HIDDEN_SIZE = 256
POS_HIDDEN_SIZE = 256
WINDOW_SIZE = 10
INPUT_SIZE = 74
EPOCHS = 100
LR = 1e-3
NUM_WORKERS = 20
WEIGHTS = [0.9, 0.1]

ACT_PATH = './model_act_10.pth'
POS_PATH = './model_pos_10.pth'

dataset = ConcatDataSet(
    type_of_data='train',
)


train_loader = DataLoader(
    dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


dataset_val = ConcatDataSet(
    type_of_data='val',
)

val_loader = DataLoader(
    dataset_val,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

model = PosLatent(
    window=WINDOW_SIZE,
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=INPUT_SIZE,
    weights=WEIGHTS,
    act_path=ACT_PATH,
    pos_hidden_size=POS_HIDDEN_SIZE,
    pos_path=POS_PATH,
    lr=LR,
)

# wandb.watch(model)
today = datetime.now(pytz.timezone('America/Sao_Paulo')
                     ).strftime("%Y-%m-%d_%H:%M:%S")

os.makedirs(f'./results/{today}', exist_ok=True)
files = os.listdir(f'./results/{today}')

os.makedirs(f'./results/{today}/{len(files)}/', exist_ok=True)

wandb.init(project="ssl_env_pred", entity="breno-cavalcanti", name=f"new_ds_reduced_model_w_grad_clip_more_lr_less_bt{today}",
           config={
               "batch_size": BATCH_SIZE,
               "hidden_size": HIDDEN_SIZE,
               "window_size": WINDOW_SIZE,
               "input_size": INPUT_SIZE,
               "epochs": EPOCHS,
               "lr": LR,
               "weights": WEIGHTS,
           })

step = 0
val_step = 0

steps = 0
epochs = 0

for epoch in tqdm(range(EPOCHS)):
    wandb.log({'epoch': epochs})
    general_val_loss = 0
    pos_val_loss = 0
    vel_val_loss = 0
    # print(f'Epoch: {epoch}, loss: {general_loss / total}')
    for x, y in tqdm(val_loader):
        x = x.to(model.device)
        y = y.to(model.device)
        loss = model.validation_step(x, y)

        general_val_loss += loss['val/loss/general_loss'].item()
        pos_val_loss += loss['val/loss/loss_pos'].item()
        vel_val_loss += loss['val/loss/loss_act'].item()

    general_loss = 0
    pos_val_loss = 0
    vel_val_loss = 0

    for x, y in tqdm(train_loader):
        x = x.to(model.device)
        y = y.to(model.device)
        loss = model.training_step(x, y)
        loss['step'] = steps

        wandb.log(loss)

        general_loss += loss['train/loss/general_loss'].item()
        pos_val_loss += loss['train/loss/loss_pos'].item()
        vel_val_loss += loss['train/loss/loss_act'].item()

    log_dict = {
        'epoch': epochs,
        'train/loss/general_loss': general_loss / len(train_loader),
        'train/loss/loss_pos': pos_val_loss / len(train_loader),
        'train/loss/loss_act': vel_val_loss / len(train_loader),
        'val/loss/general_loss': general_val_loss / len(val_loader),
        'val/loss/loss_pos': pos_val_loss / len(val_loader),
        'val/loss/loss_act': vel_val_loss / len(val_loader),
    }

    wandb.log(log_dict)
    epochs += 1
    if epoch % 10 == 0:
        torch.save(model.state_dict(),
                   f'./results/{today}/{len(files)}/checkpoint_{epoch}.pth')


torch.save(model.state_dict(),
           f'./results/{today}/{len(files)}/checkpoint_final.pth')
