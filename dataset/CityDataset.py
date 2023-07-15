import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

# Defining Custom Dataset
# Supposing following valid values for the dataset (as in Kaggle and semantic as requested)
# split valid values: train, test, val
# mode valid value: fine
# target_type valid value: semantic
class CityDataset(Dataset):
    def __init__(self, root, split = "train", mode = "fine", target_type = "semantic", transform = None):
        self.root = root
        self.split = split
        self.mode = "gtFine" if mode == "fine" else "gtCoarse" # however I did not provide any support for gtCoarse data loading
        self.images_dir = os.path.join(self.root, "leftImg8bit", self.split)
        self.targets_dir = os.path.join(self.root, self.mode, self.split)
        self.target_type = target_type
        self.images = []
        self.targets = []
        self.transform = transform

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], f"{self.mode}_labelTrainIds.png" # This new label wrt to labelIds are with the correct mapping after executing a script from the original documentation of the dataset
                )

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])

        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)

        image = transforms.ToTensor()(image)
        target = np.array(target)
        target = torch.from_numpy(target)
        
        target = target.type(torch.LongTensor)

        return image, target
