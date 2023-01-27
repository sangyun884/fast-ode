from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as TF
import argparse 
import torch
import os
parser = argparse.ArgumentParser(description='Configs')

parser.add_argument('--dir', type=str, help='image dir')
parser.add_argument('--save_dir', type=str, default = "grid.png", help='image save dir')
parser.add_argument('--size', type=int, default = 8, help='grid size')
parser.add_argument('--shuffle', action='store_true', help='shuffle images')


arg = parser.parse_args()

from PIL import Image
import glob 

files = glob.glob(os.path.join(arg.dir, "*.png")) + glob.glob(os.path.join(arg.dir, "*.jpg"))
files = sorted(files)[:arg.size**2]
if arg.shuffle:
    import random
    random.shuffle(files)
img_list = []
for file in files:
    img = Image.open(file)
    img = TF.to_tensor(img)
    img_list.append(img)
imgs = torch.stack(img_list)
grid = make_grid(imgs, nrow=arg.size, padding = 0)
save_image(grid, arg.save_dir)