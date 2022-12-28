import time
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from data import DriveDataset
from UNet_model import build_unet
from monai.losses import DiceCELoss
from utils import seeding, create_dir, train_time
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/mans4021/Desktop/REFUGE_4YP/Board/')

def train(model, loader, optimizer, loss_fn, device):
    iteration_loss = 0.0
    model.train()
    for x, y in loader:
        y = torch.nn.functional.one_hot(y, 3).squeeze()
        y = y.permute(0,3,1,2)
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        iteration_loss += loss.item()

    iteration_loss = iteration_loss/len(loader)
    return iteration_loss

def evaluate(model, loader, loss_fn, device):
    val_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            y = torch.nn.functional.one_hot(y, 3).squeeze()
            y = y.permute(0, 3, 1, 2)
            x = x.to(device, dtype=torch.float32)
            y = y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()

        val_loss = val_loss/len(loader)
    return val_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)
    """ Directories """
    create_dir("files")
    """ Load dataset """
    train_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/image/*"))
    train_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/train/mask/*"))
    valid_x = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/image/*"))
    valid_y = sorted(glob("/home/mans4021/Desktop/new_data/REFUGE2/test/mask/*"))
    '''check size of datasets'''
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 512
    W = 512
    size = (H, W)
    epoch = 5
    iteration = 80
    batch_size = 5
    lr = 1e-4 # 0.0001
    checkpoint_path = "/home/mans4021/Desktop/checkpoint/checkpoint_refuge_unet.pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceCELoss(softmax=True)

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch_n in tqdm(range(epoch)):
        for iteration_n in tqdm(range(iteration)):
            start_time = time.time()
            train_loss = train(model, train_loader, optimizer, loss_fn, device)
            valid_loss = evaluate(model, valid_loader, loss_fn, device)

            writer.add_scalar('Training Loss', train_loss, iteration + 1 + 80*epoch_n)
            writer.add_scalar('Validation Loss', valid_loss, iteration + 1 + 80 * epoch_n)

            """ Saving the model """
            if valid_loss < best_valid_loss:
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
                print(data_str)
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), checkpoint_path)

            end_time = time.time()
            iteration_mins, iteration_secs = train_time(start_time, end_time)

            data_str = f'Epoch: {epoch_n+1:02} | Iteration Time: {iteration_mins}m {iteration_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            print(data_str)