from pos_autoencoder import PositionAutoEncoder
from src.concat_data_set_autoencoder import ConcatDataSetAutoencoder
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
import torch
import wandb
import ipdb

BATCH_SIZE = 64
HIDDEN_SIZE = 256
WINDOW_SIZE = 50
INPUT_SIZE = 38
EPOCHS = 2

wandb.init(project="ssl_env", entity="breno-cavalcanti", name="pos_autoencoder_v2")

dataset = ConcatDataSetAutoencoder(
    root_dir='../all_data/data-3v3-v2',
    num_of_data_sets=100,
    window=WINDOW_SIZE,
    is_pos=True,
    should_test_the_new_data_set=True,
    type_of_data='train',
    horizon=1
)


train_loader = DataLoader(
            dataset,
            shuffle= True,
            batch_size=BATCH_SIZE,
            num_workers=6,
            pin_memory=True
)


dataset = ConcatDataSetAutoencoder(
    root_dir='../all_data/data-3v3-v2',
    num_of_data_sets=100,
    window=WINDOW_SIZE,
    is_pos=True,
    should_test_the_new_data_set=True,
    type_of_data='val',
    horizon=1
)

val_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=1,
            pin_memory=True
)

model = PositionAutoEncoder(
    window= WINDOW_SIZE,
    input_size=INPUT_SIZE,
    hidden_size= HIDDEN_SIZE,
    output_size=INPUT_SIZE,
    lr=1e-3,
)



step = 0
val_step = 0

for epoch in range(EPOCHS):
    model.train()
    general_loss = 0
    total = 0
    total_val = 0 
    general_val_loss = 0

    for i, (x, y) in enumerate(val_loader):
        x = x.to(model.device)
        y = y.to(model.device)
        loss = model.validation_step(x, y)
        loss_dict = {
            'loss_pos_val/step': loss,
        }
        wandb.log(loss_dict)
        val_step += 1
        total_val += 1
        general_val_loss += loss

    loss_dict = {
        'loss_pos_val/epoch': general_val_loss / total_val,
    }

    wandb.log(loss_dict)

    for i, (x, y) in enumerate(train_loader):
        x = x.to(model.device)
        y = y.to(model.device)
        loss = model.training_step(x, y)
        loss_dict = {
            'loss_pos_train/step': loss,
        }
        wandb.log(loss_dict)
        step += 1
        general_loss += loss
        total += 1
    loss_dict = {
        'loss_pos_train/epoch': general_loss / total,
    }
    wandb.log(loss_dict)

    print(f'Epoch: {epoch}, loss: {general_loss / total}')


    # writer.add_scalars('loss_pos_train_vs_val', {'val': general_val_loss / total_val,
    #                                         'train': general_loss / total})


    print(f'Epoch: {epoch}, loss_pos_val: {general_val_loss / total_val}')

torch.save(model.state_dict(), './model_pos.pth')