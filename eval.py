# importing useful libraries
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from dataset.CityDataset import CityDataset
from utils.hyper_param import parse_args
from utils.checkpoints import load_checkpoint
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

# Validation dataset
val_ds = CityDataset(args.ds_path, split='val', mode='fine', target_type='semantic', transform=inputTransform)
val_dl = DataLoader(val_ds, batch_size=1)

# Model
model = UNet().to(device)

if args.resume:
    load_checkpoint(model, args)
else:
    print("Put some weights for evaluation using --resume")
    exit()

print('=======> Start evaluation')

model.eval()
with torch.no_grad():
    for i, data in enumerate(val_dl, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)
        # Da continuare
