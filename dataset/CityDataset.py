import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from utils.classes import class_mapping

# Defining Custom Dataset
class CityDataset(Dataset):
    def __init__(self, images_paths, transform_img=None, transform_mask=None):
        self.images_paths = images_paths
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.mapping = class_mapping

    def __len__(self):
        return len(self.images_paths)
    
    def mask_to_class(self, mask):
        mask = torch.from_numpy(np.array(mask))

        class_mask = mask
        h, w = class_mask.shape[1], class_mask.shape[2]

        mask_out = torch.empty(h, w, dtype=torch.long)

        for k in self.mapping:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))         
            validx = (idx.sum(0) == 3)          
            mask_out[validx] = torch.tensor(self.mapping[k], dtype=torch.long)
       
        return mask_out

    def __getitem__(self, index):
        img = Image.open(self.images_paths[index])
        image = img.crop((0, 0, 256, 256))
        mask = img.crop((256, 0, 512, 256))

        if self.transform_img:
            image = self.transform_img(image)
            
        if self.transform_mask:
            mask = self.transform_mask(mask)

        #mask = self.mask_to_class(mask)
        #mask = mask.long()

        return image, mask