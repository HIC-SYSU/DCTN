import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import time
from tqdm import tqdm
from sklearn.model_selection import KFold

import DCTN
import Loss

device_ids = [0,1]

args = {
    'epoch_num': 200,
    'batch_size': 16,
    'val_batch_size': 16,
    'last_epoch': 0,
    'lr': 1e-2,
    'weight_decay': 5e-3,
    'momentum': 0.9,
    'optimizer': 'Adam',
}


train_image_path = '/data/MyAorta/NPY/train/image'
train_label_path = '/data/MyAorta/NPY/train/label'
val_image_path = '/data/MyAorta/NPY/val/image'
val_label_path = '/data/MyAorta/NPY/val/label'

train_data = MyDataset(train_image_path, train_label_path, train=True, transform=None)
val_data = MyDataset(val_image_path, val_label_path, train=False, transform=None)
dataloaders = {
        'train': DataLoader(train_data, batch_size=args['train_batch_size'], shuffle=True, num_workers=4),
        'val': DataLoader(val_data, batch_size=args['val_batch_size'], shuffle=False, num_workers=4, drop_last=True)}

def training(loader, model, optimizer, loss_all, scaler):
    loop = tqdm(loader)

    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device_ids)
        targets = targets.long().to(device=device_ids)

        predictions = model(data)
        loss = loss_all(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.DataParallel(DCTN, device_ids=device_ids)

    path='path'

    loss_all = Loss()

    optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    train_loader, val_loader = DataLoader['train'],DataLoader['val']

    scaler = torch.cuda.amp.GradScaler()

    kf = KFold(n_splits=10, shuffle=True)


    for epoch in range(args['epoch_num']):

        train_loss = 0.0

        for train_idx, val_idx in kf.split(train_loader.dataset):
            train_subdataset = torch.utils.data.Subset(train_loader.dataset, train_idx)
            val_subdataset = torch.utils.data.Subset(train_loader.dataset, val_idx)

            train_subset_loader = DataLoader(train_subdataset, batch_size=args['batch_size'], shuffle=True)
            val_subset_loader = DataLoader(val_subdataset, batch_size=args['batch_size'], shuffle=True)

        # Training
        net.train()
        train_loss += training(train_subset_loader, net, optimizer, loss_all, scaler)

        # Validation
        net.eval()
        with torch.no_grad():
            val_loss = training(val_subset_loader, net, optimizer, loss_all, scaler)

        if val_loss < best_avg_val_loss:
            best_avg_val_loss = val_loss
            no_improvement_count = 0
            checkpoint = {
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        else:
            no_improvement_count += 1

        # Early stopping
        if no_improvement_count == 10:
            print("No improvement in validation loss for 10 epochs. Stopping training.")
            break

        # save model
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint,path)



