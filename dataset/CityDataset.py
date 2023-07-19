import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os

# Defining Custom Dataset
# created similar to the source https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
# but with some changes.
# Supposing following valid values for the dataset (as in Kaggle and target type 'semantic' as requested)
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
        self.rgb_targets = []
        self.transform = transform

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(img_dir):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], f"{self.mode}_labelTrainIds.png" # This new label wrt to labelIds are with the correct mapping after executing a script from the original documentation of the dataset
                )
                rgb_target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], f"{self.mode}_color.png" # This new label wrt to labelIds are with the correct mapping after executing a script from the original documentation of the dataset
                )

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(os.path.join(target_dir, target_name))
                self.rgb_targets.append(os.path.join(target_dir, rgb_target_name))


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.targets[index])
        rgb_target = Image.open(self.rgb_targets[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
            target = self.transform(target)
            rgb_target = self.transform(rgb_target)

        image = transforms.ToTensor()(image)
        target = np.array(target)
        target = torch.from_numpy(target)
        target = target.type(torch.LongTensor)
        rgb_target = transforms.ToTensor()(rgb_target)

        return image, target, rgb_target
