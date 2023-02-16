# from act_autoencoder import ActAutoEncoder
from pos_autoencoder import PositionAutoEncoder
from src.ssl_dataset import ConcatDataSetSsl
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import tqdm

BATCH_SIZE = 128
HIDDEN_SIZE = 256
WINDOW_SIZE = 50
INPUT_SIZE = 24
EPOCHS = 144

dataset = ConcatDataSetSsl(
    type_of_data='train',
    is_autoencoder=True
)

train_loader = DataLoader(
            dataset,
            shuffle= True,
            batch_size=BATCH_SIZE,
            num_workers=10,
            pin_memory=True
)


dataset = ConcatDataSetSsl(
    type_of_data='val',
    is_autoencoder=True
)

val_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=10,
            pin_memory=True
)

model = PositionAutoEncoder(
    window= WINDOW_SIZE,
    input_size=INPUT_SIZE,
    hidden_size= HIDDEN_SIZE,
    output_size=INPUT_SIZE,
    lr=1e-3,
)

writer = SummaryWriter(log_dir='./runs/v2')


step = 0
val_step = 0

for epoch in range(EPOCHS):
    model.train()
    general_loss = 0
    total = 0
    total_val = 0 
    general_val_loss = 0
    # creates a for loop using tqdm
    for i, (x, y) in enumerate(tqdm.tqdm(train_loader, desc=f'Epoch {epoch}')):
        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                for i, (x, y) in enumerate(tqdm.tqdm(val_loader, desc=f'Epoch {epoch}')):
                    x = x.to(model.device)
                    y = y.to(model.device)
                    loss = model.validation_step(x, y)
                    writer.add_scalar('loss_val/step', loss, val_step)
                    val_step += 1
                    total_val += 1
                    general_val_loss += loss
            writer.add_scalar('loss_val/epoch', general_val_loss / total_val, epoch)
            model.train()

        x = x.to(model.device)
        y = y.to(model.device)
        loss = model.training_step(x, y)
        writer.add_scalar('loss_train/step', loss, step)
        step += 1
        general_loss += loss
        total += 1
    writer.add_scalar('loss_train/epoch', general_loss / total, epoch)


    print(f'Epoch: {epoch}, loss: {general_loss / total}')


    # plot train loss vs val loss
    # plot train loss vs val loss

    writer.add_scalars('train_vs_val', {'train_loss': general_loss / total, 'val_loss': general_val_loss / total_val}, epoch)

torch.save(model.state_dict(), '/tmp/bmmuc/model.pth')