# From https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing#scrollTo=yn1KM6WQ_7Em

import torch
import numpy as np
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
from flows import RectifiedFlow, NonlinearFlow
import torch.nn as nn
from einops import rearrange
import tensorboardX
import os
from models import AE, FourierMLP
from utils import get_train_data, draw_plot, get_train_data_two_gaussian, get_kl_2d, alpha
from PIL import Image

device = torch.device("cuda:0")
torch.manual_seed(0)
dir = f"runs/reverse/tree-learned-beta1"
writer = tensorboardX.SummaryWriter(log_dir=dir)
iterations = 1000000
batchsize = 4096
input_dim = 2
weight_prior = 1
learning_rate = 1e-3
independent = False
ddpm = False
N = 64

def train_rectified_flow(rectified_flow, forward_model, optimizer, data, batchsize, iterations):
    loss_curve = []
    for i in range(iterations+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(data))[:batchsize]
        x = data[indices]

        
        if independent:
            z = torch.randn_like(x)
        else:
            out = forward_model(x, torch.ones(batchsize, 1, device = device))
            mu = out[:, :2]
            logvar = out[:, 2:]
            z = mu + torch.randn_like(mu) * torch.exp(logvar/2)
        if not ddpm:
            z_t, t, target = rectified_flow.get_train_tuple(z0=x, z1=z)
        else:
            z_t, t, target = rectified_flow.get_train_tuple_ddpm(z0=x, z1=z)
        
        # Learn reverse model
        pred = rectified_flow.model(z_t, t)
        
        loss_fm = F.mse_loss(pred, target)

        loss_prior = get_kl_2d(mu, logvar, wide_prior=False) if not independent else 0

        loss = loss_fm + weight_prior * loss_prior

        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch {i}: loss {loss.item()}, loss_fm {loss_fm.item()}, loss_prior {loss_prior}")
            writer.add_scalar("loss", loss.item(), i)
            writer.add_scalar("loss_fm", loss_fm.item(), i)
            writer.add_scalar("loss_prior", loss_prior, i)

            loss_curve.append(loss.item())
            if i % 1000 == 0:
                with torch.no_grad():
                    
                    z = torch.randn(4096, 2).to(device)
                    traj_uncond, _ = rectified_flow.sample_ode_generative(z1=z, N=N)
                    samples_uncond = traj_uncond[-1]
                    samples_uncond = samples_uncond.cpu().numpy()

                    plt.figure(figsize=(4,4))
                    plt.xlim(0,1)
                    plt.ylim(0,1)
                    plt.xticks([])
                    plt.yticks([])
                    
                    # plt.title('Transport Trajectory')
                    plt.tight_layout()
                    plt.scatter(samples_uncond[:, 0], samples_uncond[:, 1], alpha=0.15, color='blue', s = DOT_SIZE)
                    plt.savefig(os.path.join(dir, f'iter_{i}.jpg'), dpi=300)

                    plt.close()
            if i % 100000 == 0:
                torch.save(rectified_flow.model.state_dict(), os.path.join(dir, f"model_{i}.pt"))


    return rectified_flow

def cosine_similarity(x1, x2):
    x1 = x1.view(x1.shape[0], -1)
    x2 = x2.view(x2.shape[0], -1)
    x1_norm = x1 / x1.norm(dim=1, keepdim=True)
    x2_norm = x2 / x2.norm(dim=1, keepdim=True)
    return torch.sum(x1_norm * x2_norm, dim=1)


DOT_SIZE = 3
# # Load 2d data from image
data = Image.open("./tree.png")
data = np.array(data)[:,:,:3]
data = data.sum(axis=-1) > 0
# Get the coordinates of the pixels where the value is False
data = np.argwhere(data == False)
# Normalize the coordinates
data = data / data.max(axis=0)
# Change x and y coordinates
data = data[:, [1, 0]]
data[:, 1] = 1 - data[:, 1]

# Checkerboard 2d data: 500 x 500 grid, total 64 squares
# data = np.zeros((500, 500))
# for i in range(500):
#     for j in range(500):
#         if (i//125 + j//125) % 2 == 0:
#             data[i, j] = 1
# data = np.argwhere(data == 1)
# data = data / data.max(axis=0)
# data = data[:, [1, 0]]
# data[:, 1] = 1 - data[:, 1]



# shuffle
np.random.shuffle(data)

# Get random samples and plot them
plt.figure(figsize=(4,4))
plt.xlim(0,1)
plt.ylim(0,1)
plt.xticks([])
plt.yticks([])
plt.scatter(data[:4096, 0], data[:4096, 1], alpha=0.15, color='blue', s = DOT_SIZE)
plt.tight_layout()
plt.savefig(os.path.join(dir, f'data.jpg'), dpi=300)
print(data.shape)






forward_model = FourierMLP(num_layers=3, output_dim = 4).to(device)
flow_model = FourierMLP(num_layers=3).to(device)
flow_model.load_state_dict(torch.load(os.path.join(dir, f"model_200000.pt")))
optimizer = torch.optim.Adam(list(forward_model.parameters()) + list(flow_model.parameters()), lr=learning_rate)

data = torch.from_numpy(data).float().to(device)
rectified_flow = RectifiedFlow(device, flow_model)
rectified_flow = train_rectified_flow(rectified_flow, forward_model, optimizer, data, batchsize, iterations)
