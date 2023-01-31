# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
from flows import RectifiedFlow
import torch.nn as nn
import tensorboardX
import os
from models import UNetEncoder
from guided_diffusion.unet import UNetModel
import torchvision.datasets as dsets
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from utils import straightness
from dataset import CelebAHQImgDataset
import argparse
from tqdm import tqdm
from network_edm import SongUNet
from torch.nn import DataParallel
import json
from train_reverse_img_ddp import parse_config

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--dir', type=str, help='Saving directory name')
    parser.add_argument('--ckpt', type=str, default = None, help='Flow network checkpoint')
    parser.add_argument('--batchsize', type=int, default = 4, help='Batch size')
    parser.add_argument('--res', type=int, default = 64, help='Image resolution')
    parser.add_argument('--input_nc', type=int, default = 3, help='Unet num_channels')
    parser.add_argument('--N', type=int, default = 20, help='Number of sampling steps')
    parser.add_argument('--num_samples', type=int, default = 64, help='Number of samples to generate')
    parser.add_argument('--encoder', type=str, default = None, help='Encoder ckpt')
    parser.add_argument('--dataset', type=str, help='cifar10 / mnist / celebahq')
    parser.add_argument('--no_scale', action='store_true', help='Store true if the model is trained on [0,1] scale')    
    parser.add_argument('--save_traj', action='store_true', help='Save the trajectories')    
    parser.add_argument('--save_z', action='store_true', help='Save zs for distillation')    
    parser.add_argument('--save_data', action='store_true', help='Save data')    
    parser.add_argument('--solver', type=str, default = 'euler', help='ODE solvers')
    parser.add_argument('--config_de', type=str, default = None, help='Decoder config path, must be .json file')
    parser.add_argument('--config_en', type=str, default = None, help='Encoder config path, must be .json file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--rtol', type=float, default=1e-5, help='rtol for RK45 solver')
    parser.add_argument('--atol', type=float, default=1e-5, help='atol for RK45 solver')
    


    arg = parser.parse_args()
    return arg

def main(arg):

    if not os.path.exists(arg.dir):
        os.makedirs(arg.dir)
    assert arg.config_de is not None
    config = parse_config(arg.config_de)


    if not os.path.exists(os.path.join(arg.dir, "samples")):
        os.makedirs(os.path.join(arg.dir, "samples"))
    if not os.path.exists(os.path.join(arg.dir, "zs")):
        os.makedirs(os.path.join(arg.dir, "zs"))
    if not os.path.exists(os.path.join(arg.dir, "trajs")):
        os.makedirs(os.path.join(arg.dir, "trajs"))
    if not os.path.exists(os.path.join(arg.dir, "data")):
        os.makedirs(os.path.join(arg.dir, "data"))
    


    if config['unet_type'] == 'adm':
        model_class = UNetModel
    elif config['unet_type'] == 'songunet':
        model_class = SongUNet
    # Pass the arguments in the config file to the model
    flow_model = model_class(**config)

    device_ids = arg.gpu.split(',')
    if len(device_ids) > 1:
        device = torch.device(f"cuda")
        print(f"Using {device_ids} GPUs!")
        flow_model = DataParallel(flow_model)
    else:
        device = torch.device(f"cuda:{arg.gpu}")
        print(f"Using GPU {arg.gpu}!")
    # Print the number of parameters in the model
    pytorch_total_params = sum(p.numel() for p in flow_model.parameters())
    # Convert to M
    pytorch_total_params = pytorch_total_params / 1000000
    print(f"Total number of parameters: {pytorch_total_params}M")

    if arg.ckpt is not None:
        flow_model.load_state_dict(torch.load(arg.ckpt, map_location = "cpu"))
    else:
        raise NotImplementedError("Model ckpt should be provided.")
    flow_model = flow_model.to(device)


    rectified_flow = RectifiedFlow(device, flow_model, num_steps = arg.N)

    rectified_flow.model.eval()

    if arg.encoder is not None:
        from train_reverse_img_ddp import get_loader
        config_en = parse_config(arg.config_en)
        if config_en['unet_type'] == 'adm':
            encoder_class = UNetModel
        elif config_en['unet_type'] == 'songunet':
            encoder_class = SongUNet
        # Pass the arguments in the config file to the model
        encoder = encoder_class(**config_en)
        # encoder = SongUNet(img_resolution = arg.res, in_channels = arg.input_nc, out_channels = arg.input_nc * 2, channel_mult = [2,2,2], dropout = 0.13, num_blocks = 2, model_channels = 32)
        
        forward_model = UNetEncoder(encoder = encoder, input_nc = arg.input_nc)
        forward_model.load_state_dict(torch.load(arg.encoder, map_location = "cpu"), strict = True)
      
        
        forward_model = forward_model.to(device).eval()
        data_loader, _, _, _ = get_loader(arg.dataset, arg.batchsize, 1, 0)
        # dataset_train = CelebAHQImgDataset(arg.res, im_dir = 'D:\datasets\CelebAMask-HQ\CelebA-HQ-img-train-64')
        # dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=arg.batchsize)
        train_iter = iter(data_loader)
    # Save configs as json file
    config_dict = vars(arg)
    with open(os.path.join(arg.dir, 'config_sampling.json'), 'w') as f:
        json.dump(config_dict, f, indent = 4)
    with torch.no_grad():
        i = 0
        epoch = arg.num_samples // arg.batchsize + 1
        x0_list = []
        straightness_list = []
        nfes = []
        z_norm_list = []
        for ep in tqdm(range(epoch)):
            noise = torch.randn(arg.batchsize, arg.input_nc, arg.res, arg.res).to(device)
            save_image(noise, "debug1.jpg")
            if arg.encoder is not None:
                x, _= next(train_iter)
                x = x.to(device)
                # x = x[1].repeat(arg.batchsize, 1, 1, 1)
                noise = noise[1].repeat(arg.batchsize, 1, 1, 1)
                z, _, _ = forward_model(x, noise = noise)
            else:
                z = noise
            # Compute the norm of z
            z_norm = torch.sum(z ** 2, dim = [1,2,3]).sqrt()
            z_norm_list.append(z_norm)
            save_image(z, "debug2.jpg")
            if arg.solver in ['euler', 'heun']:
                traj_uncond, traj_uncond_x0 = rectified_flow.sample_ode_generative(z1=z, N=arg.N, use_tqdm = False, solver = arg.solver)
                x0 = traj_uncond[-1]
                uncond_straightness = straightness(traj_uncond)
                straightness_list.append(uncond_straightness.item())
            else:
                x0, nfe = rectified_flow.sample_ode_generative_bbox(z1=z, N=arg.N, use_tqdm = False, solver = arg.solver, atol = arg.atol, rtol = arg.rtol)
                nfes.append(nfe)
                # print(f"nfe: {nfe}")

            if arg.save_traj:
                if len(traj_uncond_x0) > 10:
                    interval = len(traj_uncond_x0) // 5
                    grid = torch.cat(traj_uncond_x0[::interval], dim=3)
                else:
                    grid = torch.cat(traj_uncond_x0, dim=3) # grid.shape: (bsize, channel, H, W * N)
                if len(traj_uncond_x0) == 100:
                    idx = [0, 5, 10, 15, 20, 35, 50, 70, 99] # For visualization, currently hard-coded
                    grid = torch.cat([traj_uncond_x0[i] for i in idx], dim=3)
                # (batch_size, channel, H, W * N) -> (channel, H * bsize, W * N)
                grid = grid.permute(1, 0, 2, 3).contiguous().view(grid.shape[1], -1, grid.shape[3])
                save_image(grid * 0.5 + 0.5 if not arg.no_scale else grid, os.path.join(arg.dir, "trajs", f"{ep:05d}_traj.png"))

            for idx in range(len(x0)):
                save_image(x0[idx] * 0.5 + 0.5 if not arg.no_scale else x0[idx], os.path.join(arg.dir, "samples", f"{i:05d}.png"))
                # Save z as npy file
                if arg.save_z:
                    np.save(os.path.join(arg.dir, "zs", f"{i:05d}.npy"), z[idx].cpu().numpy())
                if arg.save_data:
                    save_image(x[idx] * 0.5 + 0.5 if not arg.no_scale else x[idx], os.path.join(arg.dir, "data", f"{i:05d}.png"))
                i+=1
                if i >= arg.num_samples:
                    break

            x0_list.append(x0)
        straightness_mean = np.mean(straightness_list)
        print(f"straightness_mean: {straightness_mean}")
        nfes_mean = np.mean(nfes) if len(nfes) > 0 else arg.N
        print(f"nfes_mean: {nfes_mean}")
        z_norms = torch.stack(z_norm_list).view(-1)
        result_dict = {"straightness_mean": straightness_mean, "z_norms": z_norms.tolist(), "nfes_mean": nfes_mean}
        with open(os.path.join(arg.dir, 'result_sampling.json'), 'w') as f:
            json.dump(result_dict, f, indent = 4)
        
        

if __name__ == "__main__":
    arg = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpu
    torch.manual_seed(arg.seed)
    print(f"seed: {arg.seed}")
    main(arg)