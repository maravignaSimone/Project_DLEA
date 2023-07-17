import torch
import torchvision.transforms as transforms
from PIL import Image
import os

def save_images(inputs, pred_labels, rgbmask, mapping, out_folder, i):
    # squeeze output in order to remove batch_size (in eval bs = 1 so squeeze is enough)
    pred = torch.squeeze(pred_labels)
    colored_pred = torch.zeros(3, pred.size(0), pred.size(1), dtype=torch.uint8)
    for k in mapping:
        colored_pred[:, pred==k] = torch.tensor(mapping[k]).byte().view(3, 1)

    image = torch.squeeze(inputs)
    rgbmask = torch.squeeze(rgbmask)
    image = transforms.ToPILImage()(image)
    colored_pred = transforms.ToPILImage()(colored_pred.byte())
    rgbmask = transforms.ToPILImage()(rgbmask)
    new_image = Image.new('RGB',((image.size[0], 3*image.size[1])))
    new_image.paste(image,(0,0))
    new_image.paste(colored_pred,(0, image.size[1]))
    new_image.paste(rgbmask, (0, 2*image.size[1]))

    filename = "mask_" + str(i) + ".png"
    new_image.save(os.path.join(out_folder, filename))