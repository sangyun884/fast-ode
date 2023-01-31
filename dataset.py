import os
from PIL import Image
from torch.utils.data import Dataset
import glob
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch

class CelebAHQImgDataset(Dataset):
    def __init__(self, size, im_dir, transform):
        super().__init__()
        self.size = size
        self.im_dir = im_dir
        self.im_names  = glob.glob(os.path.join(im_dir, "*.jpg")) + glob.glob(os.path.join(im_dir, "*.png"))
        self.transform = transform
        print(f"len(self.im_names) = {len(self.im_names)}")

    def __getitem__(self, i):
        im_name = self.im_names[i]
        img = Image.open(im_name)
        img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.im_names)

class DatasetWithLatent(Dataset):
    def __init__(self, im_dir, latent_dir, input_nc):
        super().__init__()
        self.input_nc = input_nc
        img_list = glob.glob(os.path.join(im_dir, '*.png')) + glob.glob(os.path.join(im_dir, '*.jpg'))
        img_list.sort()
        self.img_list = img_list
        self.latent_names = []
        for im_name in img_list:
            num = im_name.split('\\')[-1].split('_')[-1].split('.')[0]
            latent_name = os.path.join(latent_dir, f'{num}.npy')
            self.latent_names.append(latent_name)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) if input_nc == 3 else transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        if self.input_nc == 1:
            img = img.convert('L')
        img = np.array(img)
        img = self.transforms(img)
        latent_name = self.latent_names[idx]
        latent = np.load(latent_name)
        latent = torch.tensor(latent, dtype=torch.float32)
        return img, latent
