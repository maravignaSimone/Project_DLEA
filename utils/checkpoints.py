import torch
import os
import time

# functions to save and load the model with a custom name got by actual time and number of current epoch
def save_checkpoint(model, folder, epoch=0):
    if not os.path.exists(folder):
        os.makedirs(folder)
    print("Saving checkpoints")
    filename = "cp_" + time.strftime("%Y%m%d-%H%M%S") + "_e" + str(epoch) + ".pth"
    save_path = os.path.join(folder, filename)
    torch.save(model.state_dict(), save_path)

def load_checkpoint(model, args):
    if os.path.isfile(args.resume):
        print("Resuming from {}".format(args.resume))
        model.load_state_dict(torch.load(args.resume))