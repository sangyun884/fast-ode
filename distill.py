# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
from flows import RectifiedFlow
import torch.nn as nn
import tensorboardX
import os
from models import UNetAE
from guided_diffusion.unet import UNetModel
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.utils import save_image
from dataset import DatasetWithLatent
import argparse
from network_edm import SongUNet
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Configs')
from tqdm import tqdm
from train_reverse_img_ddp import parse_config
torch.manual_seed(0)



parser.add_argument('--gpu', type=str, help='gpu num')
parser.add_argument('--dir', type=str, help='Saving directory name')
parser.add_argument('--im_dir', type=str, help='Image dir')
parser.add_argument('--im_dir_test', type=str, help='Image test dir')
parser.add_argument('--z_dir', type=str, help='zs dir')
parser.add_argument('--z_dir_test', type=str, help='zs test dir')
parser.add_argument('--iterations', type=int, default = 100000, help='Number of iterations')
parser.add_argument('--batchsize', type=int, default = 8, help='Batch size')
parser.add_argument('--learning_rate', type=float, default = 3e-5, help='Learning rate')
parser.add_argument('--ckpt', type=str, default = None, help='Pretrained ODE checkpoint')
parser.add_argument('--input_nc', type=int, default = 3, help='Image channels')
parser.add_argument('--res', type=int, default = 64, help='Image resolution')
parser.add_argument('--config_en', type=str, default = None, help='Encoder config path, must be .json file')
parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')


arg = parser.parse_args()
device = torch.device(f"cuda:{arg.gpu}")



def distill(flow_model, train_loader, test_loader, iterations, optimizer, data_shape, writer):
    z_fixed = torch.randn(data_shape, device=device)
    for i in tqdm(range(iterations+1)):
        optimizer.zero_grad()
        try:
            x, z = train_iter.next()
        except:
            train_iter = iter(train_loader)
            x, z = train_iter.next()
        x = x.to(device)
        z = z.to(device)
        
        
        # Learn student model
        pred_v = flow_model(z, torch.ones(z.shape[0], device=device))
        pred = z - pred_v

        loss = torch.mean((pred - x)**2)

        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Iteration {i}: loss {loss.item()}")
            writer.add_scalar("loss", loss.item(), i)

        if i % 1000 == 0:
            flow_model.eval()
            with torch.no_grad():
                pred_v = flow_model(z_fixed, torch.ones(z.shape[0], device=device))
                pred = z_fixed - pred_v
                save_image(pred * 0.5 + 0.5, os.path.join(arg.dir, f"pred_{i}.jpg"))
                # test
                loss_test_list = []
                for x_test, z_test in test_loader:
                    x_test = x_test.to(device)
                    z_test = z_test.to(device)
                    pred_v = flow_model(z_test, torch.ones(z_test.shape[0], device=device))
                    pred = z_test - pred_v
                    loss = torch.mean((pred - x_test)**2)
                    loss_test_list.append(loss.item())
                writer.add_scalar("test_loss", np.mean(loss_test_list), i)
                print(f"Iteration {i}: test loss {np.mean(loss_test_list)}")
                
            flow_model.train()
        if i % 10000 == 0:
            torch.save(flow_model.state_dict(), os.path.join(arg.dir, f"flow_model_distilled_{i}.pth"))




def main():
    writer = tensorboardX.SummaryWriter(log_dir=arg.dir)
    sample_data = np.load(os.path.join(arg.z_dir, "05000.npy"))
    res = sample_data.shape[1]
    input_nc = sample_data.shape[0]
    config_de = parse_config(arg.config_de)
    if config_de['unet_type'] == 'adm':
        model_class = UNetModel
    elif config_de['unet_type'] == 'songunet':
        model_class = SongUNet
    flow_model = model_class(**config_de)
    # Print the number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
    # Convert to M
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total number of parameters: {pytorch_total_params}M")

    if arg.ckpt is not None:
        flow_model.load_state_dict(torch.load(arg.ckpt))
    else:
        raise NotImplementedError("Teacher flow ckpt should be provided.")
    flow_model = flow_model.to(device)

    flow_model.load_state_dict(torch.load(arg.ckpt))

    optimizer = torch.optim.Adam(flow_model.parameters(), lr=arg.learning_rate, betas = (0.9, 0.9999), eps=1e-8)

    train_dataset = DatasetWithLatent(arg.im_dir, arg.z_dir, input_nc = input_nc)
    test_dataset = DatasetWithLatent(arg.im_dir_test, arg.z_dir_test, input_nc = input_nc)  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=arg.batchsize, shuffle=True, num_workers=6)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=arg.batchsize, shuffle=False)
    data_shape = (arg.batchsize, input_nc, res, res)
    distill(flow_model, train_loader, test_loader, arg.iterations, optimizer, data_shape, writer)

if __name__ == "__main__":
    main()