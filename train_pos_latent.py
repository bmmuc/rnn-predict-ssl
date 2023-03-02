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

BATCH_SIZE = 256
HIDDEN_SIZE = 512
POS_HIDDEN_SIZE = 512
WINDOW_SIZE = 10
INPUT_SIZE = 74
EPOCHS = 100
LR = 1e-5
NUM_WORKERS = 15
WEIGHTS = [0.9, 0.1]

ACT_PATH = './model_act.pth'
POS_PATH = './model_pos.pth'

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


dataset = ConcatDataSet(
    type_of_data='val',
)

val_loader = DataLoader(
    dataset,
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

wandb.init(project="ssl_env_pred", entity="breno-cavalcanti", name=f"pred_with_grad_clip_and_more_linears{today}",
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

torch.save(model.state_dict(), './model_pred.pth')
