# importing useful libraries
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset.CityDataset import CityDataset
from utils.hyper_param import parse_args
from utils.checkpoints import save_checkpoint, load_checkpoint
from model.UNet import UNet

# Setting GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Setting Hyperparameters
args = parse_args()
HEIGHT = 144
WIDTH = 288

# Defining transform
inputTransform = transforms.Compose([
        transforms.Resize((HEIGHT,WIDTH) , interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

# Training dataset
train_ds = CityDataset(args.ds_path, split='train', mode='fine', target_type='semantic', transform=inputTransform)
# Validation dataset
val_ds = CityDataset(args.ds_path, split='val', mode='fine', target_type='semantic', transform=inputTransform)

# Dataloaders
train_dl = DataLoader(train_ds, batch_size=args.batch_size)
val_dl = DataLoader(val_ds, batch_size=1)

# Model
model = UNet().to(device)

# Choose the loss function and optimizer
criterion = nn.MSELoss()
# criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

print('Start training')
print('Parameters: epochs= {}, bs= {}, lr= {}'.format(args.epochs, args.batch_size, args.lr))

if args.resume:
    load_checkpoint(model, args)

# losses
train_loss = []
val_loss = []

# train loop
for epoch in range(args.epochs):
    # losses for each batch
    trainloss = 0
    valloss = 0

    model.train()

    for i, data in enumerate(train_dl, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trainloss += loss.item()

        if i % 200 == 199:
            print("[it: {}] loss: {}".format(i+1, trainloss / 200))
            # uncomment for faster debug
            break

    # save checkpoint every 5 epochs
    if epoch % 5 == 0:
        save_checkpoint(model, args.save_weights, epoch)
    train_loss.append(trainloss/len(train_dl))

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_dl, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valloss += loss.item()

        val_loss.append(valloss/len(val_dl))

    print("epoch : {} , train loss : {} , valid loss : {} ".format(epoch, train_loss[-1], val_loss[-1]))