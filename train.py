# importing useful libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter

from dataset.CityDataset import CityDataset
from utils.hyper_param import parse_args
from utils.checkpoints import save_checkpoint, load_checkpoint
from model.UNet import UNet

import os
import time

# Setting GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Setting Hyperparameters
args = parse_args()
HEIGHT = 144
WIDTH = 288

# Defining transform
inputTransform = transforms.Compose([
        transforms.Resize((HEIGHT,WIDTH) , interpolation=transforms.InterpolationMode.NEAREST), #interpolation mode necessary because label pixels are coded into each class id
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

# Training dataset (Dataset class is custom)
train_ds = CityDataset(args.ds_path, split='train', mode='fine', target_type='semantic', transform=inputTransform)
# Validation dataset
val_ds = CityDataset(args.ds_path, split='val', mode='fine', target_type='semantic', transform=inputTransform)

# Dataloaders
train_dl = DataLoader(train_ds, batch_size=args.batch_size)
val_dl = DataLoader(val_ds, batch_size=1)

# Model
model = UNet().to(device)

# Choose the loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

IoU = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(device)

# Creating Tensorboard folder
runs_folder = args.runs_folder + '/' + time.strftime("%Y%m%d-%H%M")

if not os.path.exists(runs_folder):
    os.makedirs(runs_folder)

writer = SummaryWriter(runs_folder)

# Start training
print('=======> Start training')
print('Parameters: epochs= {}, bs= {}, lr= {}'.format(args.epochs, args.batch_size, args.lr))

# Load checkpoint if requested
if args.resume:
    load_checkpoint(model, args)

# losses and accuracy
train_loss = []
train_acc = []
val_loss = []
val_acc = []

# train loop
for epoch in range(args.epochs):
    # losses and accuracy for each epoch
    trainloss = 0
    valloss = 0
    trainaccuracy = 0
    valaccuracy = 0

    model.train()

    for i, data in enumerate(train_dl, 0):
        inputs, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs) # outputs [B, N_Classes, H, W]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trainloss += loss.item()

        metric = IoU(outputs, labels)
        trainaccuracy += metric.item()

        writer.add_scalar('Loss/train', trainloss / (i+1), epoch * len(train_dl) + i)
        writer.add_scalar('Accuracy/train', trainaccuracy / (i+1), epoch * len(train_dl) + i)

        if i % 200 == 199:
            print("[it: {}] loss: {} accuracy: {}".format(i+1, trainloss / (i+1), trainaccuracy / (i+1)))
            # uncomment for faster debug
            # break

    print("Epoch : {} finished train, starting eval".format(epoch))
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        save_checkpoint(model, args.save_weights, epoch)
    train_loss.append(trainloss/len(train_dl))
    train_acc.append(trainaccuracy/len(val_dl))

    # eval but without saving images (eval with saving images is on the file eval.py)
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dl, 0):
            inputs, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            valloss += loss.item()

            metric = IoU(outputs, labels)
            valaccuracy += metric.item()

            writer.add_scalar('Loss/val', valloss / (i+1), epoch * len(val_dl) + i)
            writer.add_scalar('Accuracy/val', valaccuracy / (i+1), epoch * len(val_dl) + i)

            if i % 100 == 99:
                print("[it: {}] loss: {} accuracy: {}".format(i+1, valloss / (i+1), valaccuracy / (i+1)))

        val_loss.append(valloss/len(val_dl))
        val_acc.append(valaccuracy/len(val_dl))

    print("Epoch : {} , train loss : {} , valid loss : {} , train accuracy: {} , val accuracy: {} ".format(epoch, train_loss[-1], val_loss[-1], train_acc[-1], val_acc[-1]))

writer.close()
print("Finished training")