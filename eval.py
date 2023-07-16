# importing useful libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from dataset.CityDataset import CityDataset
from utils.hyper_param import parse_args
from utils.checkpoints import load_checkpoint
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
        transforms.Resize((HEIGHT,WIDTH) , interpolation=transforms.InterpolationMode.NEAREST),
        # transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

# Validation dataset
val_ds = CityDataset(args.ds_path, split='val', mode='fine', target_type='semantic', transform=inputTransform)
val_dl = DataLoader(val_ds, batch_size=1)

# Model
model = UNet().to(device)

IoU = MulticlassJaccardIndex(num_classes=19, ignore_index=255).to(device)

if args.resume:
    load_checkpoint(model, args)
else:
    print("Put some weights for evaluation using --resume")
    exit()

print('=======> Start evaluation')

out_folder = args.output_folder + '/' + time.strftime("%Y%m%d-%H%M")

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

totaccuracy = 0

model.eval()
with torch.no_grad():
    for i, data in enumerate(val_dl, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)

        metric = IoU(outputs, labels)
        totaccuracy += metric.item()
        
        pred_labels = torch.argmax(outputs, dim=1)

        filename = "mask_" + str(i) + ".png"
        pred_labels = transforms.ToPILImage()(pred_labels.byte())
        pred_labels.save(os.path.join(out_folder, filename))

        if i % 100 == 99:
                print("[it: {}] accuracy: {}".format(i+1, totaccuracy / i))

print("Evaluation accuracy: {} ".format(totaccuracy/len(val_dl)))

