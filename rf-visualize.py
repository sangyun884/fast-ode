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
from models import AE, MLP
from utils import get_train_data, draw_plot, get_train_data_two_gaussian, get_kl_2d, alpha
import argparse

torch.manual_seed(0)

def train_rectified_flow(rectified_flow, forward_model, optimizer, train_data1, train_data2, batchsize, iterations, paring):
    mu_z_1 = torch.tensor([6., 6.], device=device)
    mu_z_2 = torch.tensor([6., -6.], device=device)
    loss_curve = []
    for i in range(iterations+1):
        optimizer.zero_grad()
        indices = torch.randperm(len(train_data1))[:batchsize//2]
        x2 = train_data1[indices]
        indices = torch.randperm(len(train_data2))[:batchsize//2]
        x1 = train_data2[indices]
        x = torch.cat([x1, x2], dim=0)

        if not paring:
            x = x[torch.randperm(len(x))]
        if independent:
            z1 = torch.randn(batchsize // 2, 2, device=device) + mu_z_1
            z2 = torch.randn(batchsize // 2, 2, device=device) + mu_z_2
            z = torch.cat([z1, z2], dim=0)
            
            if wide_prior:
                z[:, 0] = z[:, 0] * 6
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

        loss_prior = get_kl_2d(mu, logvar) if not independent else 0

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
                    x1 = train_data1[:batchsize]
                    x2 = train_data2[:batchsize]
                    if independent:
                        z1 = torch.randn_like(x1) + mu_z_1
                        z2 = torch.randn_like(x2) + mu_z_2
                        if wide_prior:
                            z1[:, 0] = z1[:, 0] * 6
                            z2[:, 0] = z2[:, 0] * 6
                    else:
                        out1 = forward_model(x1, torch.ones(batchsize, 1, device = device))
                        out2 = forward_model(x2, torch.ones(batchsize, 1, device = device))

                        mu1 = out1[:, :2]
                        logvar1 = out1[:, 2:]
                        mu2 = out2[:, :2]
                        logvar2 = out2[:, 2:]

                        z1 = mu1 + torch.randn_like(mu1) * torch.exp(logvar1/2)
                        z2 = mu2 + torch.randn_like(mu2) * torch.exp(logvar2/2)
                    traj_reverse1, _ = rectified_flow.sample_ode_generative(z1=z1, N=N)
                    traj_reverse2, _ = rectified_flow.sample_ode_generative(z1=z2, N=N)

                    x_recon1 = traj_reverse1[-1]
                    x_recon2 = traj_reverse2[-1]
                    draw_plot(x1, x2, x_recon1, x_recon2, z1, z2, i, DOT_SIZE, M, dir, np.array([0, 0]),np.array([36, 1]) )
                    z1 = torch.randn(512, 2, device=device) + mu_z_1
                    z2 = torch.randn(512, 2, device=device) + mu_z_2
                    z = torch.cat([z1, z2], dim=0)
                    z = z[torch.randperm(len(z))]
                    
                    if wide_prior:
                        z[:, 0] = z[:, 0] * 6
                    traj_uncond_z1, _ = rectified_flow.sample_ode_generative(z1=z1, N=N)
                    traj_uncond_z2, _ = rectified_flow.sample_ode_generative(z1=z2, N=N)
                    traj_forward_x1 = rectified_flow.sample_ode(z0=x1)
                    traj_forward_x2 = rectified_flow.sample_ode(z0=x2)
                    

                    traj_particles_uncond_z1 = torch.stack(traj_uncond_z1).cpu().numpy()
                    traj_particles_uncond_z2 = torch.stack(traj_uncond_z2).cpu().numpy()

                    traj_particles1 = torch.stack(traj_reverse1).cpu().numpy()
                    traj_particles2 = torch.stack(traj_reverse2).cpu().numpy()
                    traj_particles_x1 = torch.stack(traj_forward_x1).cpu().numpy()
                    traj_particles_x2 = torch.stack(traj_forward_x2).cpu().numpy()


                    plt.figure(figsize=figsize)
                    plt.xlim(*xlim)
                    plt.ylim(*ylim)
                    plt.xticks([])
                    plt.yticks([])
                    for j in range(60):
                        plt.plot(traj_particles_uncond_z1[:, j, 0], traj_particles_uncond_z1[:, j, 1], color='orange',  linewidth=2, alpha=0.3, zorder=1)
                        plt.plot(traj_particles_uncond_z2[:, j, 0], traj_particles_uncond_z2[:, j, 1], color='green',  linewidth=2, alpha=0.3, zorder=1)
                     
                    plt.scatter(traj_uncond_z1[-1].cpu().numpy()[:, 0], traj_uncond_z1[-1].cpu().numpy()[:, 1], alpha=0.6, color='blue', s = DOT_SIZE, zorder=2)
                    plt.scatter(traj_uncond_z2[-1].cpu().numpy()[:, 0], traj_uncond_z2[-1].cpu().numpy()[:, 1], alpha=0.6, color='blue', s = DOT_SIZE, zorder=2)

                    plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), alpha=0.6, color='red', s = DOT_SIZE, zorder=2)
                    plt.scatter(z2[:, 0].cpu().numpy(), z2[:, 1].cpu().numpy(), alpha=0.6, color='red', s = DOT_SIZE, zorder=2)
                    # plt.title('Transport Trajectory')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f'traj_uncond{i}.jpg'), dpi=300)

                    # Draw reconstruction trajectory
                    plt.figure(figsize=figsize)
                    plt.xlim(*xlim)
                    plt.ylim(*ylim)
                    plt.xticks([])
                    plt.yticks([])
                    for j in range(30):
                        plt.plot(traj_particles1[:, j, 0], traj_particles1[:, j, 1], color='red')
                        plt.plot(traj_particles2[:, j, 0], traj_particles2[:, j, 1], color='green')
                    plt.scatter(traj_reverse1[-1].cpu().numpy()[:, 0], traj_reverse1[-1].cpu().numpy()[:, 1], alpha=0.15, color="red", s = DOT_SIZE)
                    plt.scatter(traj_reverse2[-1].cpu().numpy()[:, 0], traj_reverse2[-1].cpu().numpy()[:, 1], alpha=0.15, color="green", s = DOT_SIZE)
                    # plt.title('Reconstruction Trajectory')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f'traj_recon_{i}.jpg'), dpi=300)

                    # Draw forward trajectory
                    plt.figure(figsize=figsize)
                    plt.xlim(*xlim)
                    plt.ylim(*ylim)
                    plt.xticks([])
                    plt.yticks([])
                    for j in range(30):
                        plt.plot(traj_particles_x1[:, j, 0], traj_particles_x1[:, j, 1], color='red')
                        plt.plot(traj_particles_x2[:, j, 0], traj_particles_x2[:, j, 1], color='green')
                    plt.scatter(traj_forward_x1[-1].cpu().numpy()[:, 0], traj_forward_x1[-1].cpu().numpy()[:, 1], alpha=0.15, color="red", s = DOT_SIZE)
                    plt.scatter(traj_forward_x2[-1].cpu().numpy()[:, 0], traj_forward_x2[-1].cpu().numpy()[:, 1], alpha=0.15, color="green", s = DOT_SIZE)
                    # plt.title('Forward Trajectory')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f'traj_forward_{i}.jpg'), dpi=300)

                    # Draw forward model trajectory
                    plt.figure(figsize=figsize)
                    plt.xlim(*xlim)
                    plt.ylim(*ylim)
                    # plt.xtics = np.arange(-M, M, 5)
                    # plt.ytics = np.arange(-M//2, M, 5)
                    # no tics
                    plt.xticks([])
                    plt.yticks([])
                    # plt.axis('equal')
                    if not ddpm:
                        # Trajectory from x1 to z1, linear interpolation
                        batch_data = torch.cat([x2, x1], dim=0)
                        if not paring:
                            batch_data = batch_data[torch.randperm(len(batch_data))]

                        traj_x_z1 = torch.stack([batch_data[:batchsize] + (z1-batch_data[:batchsize])*t for t in np.linspace(0, 1, 30)]).cpu().numpy()
                        traj_x_z2 = torch.stack([batch_data[batchsize:] + (z2-batch_data[batchsize:])*t for t in np.linspace(0, 1, 30)]).cpu().numpy()
                        # traj_x1_z1 = torch.stack([x1 + (z1-x1)*t for t in np.linspace(0, 1, 30)]).cpu().numpy()

                        # Trajectory from x2 to z2, linear interpolation
                        # traj_x2_z2 = torch.stack([x2 + (z2-x2)*t for t in np.linspace(0, 1, 30)]).cpu().numpy()
                    else:
                        t = np.linspace(0, 1, 100)
                        alpha_ts = alpha(t)
                        coeffs = (1-alpha_ts) ** 0.5
                        traj_x_z1 = torch.stack([x1 * alpha_t + z1 * coeff for alpha_t, coeff in zip(alpha_ts, coeffs)]).cpu().numpy()
                        traj_x_z2 = torch.stack([x2 * alpha_t + z2 * coeff for alpha_t, coeff in zip(alpha_ts, coeffs)]).cpu().numpy()
                        
                    # plot
                    for j in range(100):
                        plt.plot(traj_x_z1[:, j, 0], traj_x_z1[:, j, 1], color='orange', linestyle='--',  linewidth=2, alpha=0.3, zorder=1)
                        plt.plot(traj_x_z2[:, j, 0], traj_x_z2[:, j, 1], color='green', linestyle='--',  linewidth=2, alpha=0.3, zorder=1)
                    plt.scatter(traj_x_z1[-1, :, 0], traj_x_z1[-1, :, 1], alpha=0.6, color="red", s = DOT_SIZE, zorder=2)
                    plt.scatter(traj_x_z2[-1, :, 0], traj_x_z2[-1, :, 1], alpha=0.6, color="red", s = DOT_SIZE, zorder=2)
                    
                    plt.scatter(samples_1[:512, 0].cpu().numpy(), samples_1[:512, 1].cpu().numpy(), alpha=0.6, color='blue', s = DOT_SIZE, zorder=2)
                    plt.scatter(samples_2[:512, 0].cpu().numpy(), samples_2[:512, 1].cpu().numpy(), alpha=0.6, color='blue', s = DOT_SIZE, zorder=2)
                    # plt.title('Forward Model Trajectory')
                    plt.tight_layout()
                    plt.savefig(os.path.join(dir, f'traj_forward_model_{i}.jpg'), dpi=300)

                    plt.close()

            if i % 10000 == 0:
                torch.save(rectified_flow.model.state_dict(), os.path.join(dir, f"flow_model_{i}.pth"))
                rectified_flow.model.eval()
                if not independent:
                    forward_model.eval()
                loss_fm_test_list = []
                with torch.no_grad():
                    x_test = torch.cat([train_data1[:10000], train_data2[:10000]], dim=0)
                    if independent:
                        z1 = torch.randn(10000, 2, device=device) + mu_z_1
                        z2 = torch.randn(10000, 2, device=device) + mu_z_2
                        z = torch.cat([z1, z2], dim=0)
                        if wide_prior:
                            z[:, 0] = z[:, 0] * 6
                    else:
                        out = forward_model(x_test, torch.ones(x_test.shape[0], 1, device = device))
                        mu = out[:, :2]
                        logvar = out[:, 2:]
                        z = mu + torch.randn_like(mu) * torch.exp(logvar/2)
                    for j in range(10):
                        if not ddpm:
                            z_t, t, target = rectified_flow.get_train_tuple(z0=x_test, z1=z)
                        else:
                            z_t, t, target = rectified_flow.get_train_tuple_ddpm(z0=x_test, z1=z)
                    
                        # Learn reverse model
                        pred = rectified_flow.model(z_t, t)
                        
                        loss_fm_test = F.mse_loss(pred, target)
                        loss_fm_test_list.append(loss_fm_test.item())
                loss_fm_test_mean = np.mean(loss_fm_test_list)
                writer.add_scalar('loss_fm_test', loss_fm_test_mean, i)
                rectified_flow.model.train()
                if not independent:
                    forward_model.train()

    return rectified_flow


def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='Configs')
    parser.add_argument('--gpu', type=str, help='gpu index')
    parser.add_argument('--N', type=int, default = 20, help='NFEs')
    parser.add_argument('--lr', type=float, default = 1e-3, help='learning rate')
    parser.add_argument('--weight_prior', type=float, default = 10, help='weight prior')
    parser.add_argument('--batchsize', type=int, default = 512, help='batchsize')
    parser.add_argument('--iterations', type=int, default = 100000, help='iterations')
    parser.add_argument('--dir', type=str, default = 'results', help='dir')
    parser.add_argument('--pairing', action='store_true', help='True for visualize 2-rectified flow, false for 1-rectified flow (independent coupling)')
    
    args = parser.parse_args()
    args.independent = True
    args.ddpm = False
    return args

arg = get_args()
dir = arg.dir

writer = tensorboardX.SummaryWriter(log_dir=dir)
figsize = (4, 4)

D = 10.
M = 14
VAR = 0.3
DOT_SIZE = 4
COMP = 3
iterations = arg.iterations
batchsize = arg.batchsize
input_dim = 2
weight_prior = arg.weight_prior
learning_rate = arg.lr
independent = arg.independent
wide_prior = false
ddpm = arg.ddpm
N = arg.N
xlim = (-M, M)
ylim = (-M, M)
device = torch.device('cuda:'+arg.gpu)

mu_val = -6.
cov_val = 0.
mu1 = torch.tensor([mu_val, mu_val])
mu2 = torch.tensor([mu_val, -mu_val])
cov1 = torch.tensor([[1, cov_val], [cov_val, 1]])
cov2 = torch.tensor([[1, -cov_val], [-cov_val, 1]])



    




samples_1, samples_2 = get_train_data_two_gaussian(mu1, mu2, cov1, cov2)
samples_1, samples_2 = samples_1.to(device), samples_2.to(device)

print('Shape of the samples:', samples_1.shape, samples_1.shape)

# Print mean and variance for each dimension (i.e. x-axis and y-axis)
print(f"x-axis: mean {samples_1[:, 0].mean()}, var {samples_1[:, 0].var()}")
print(f"y-axis: mean {samples_1[:, 1].mean()}, var {samples_1[:, 1].var()}")

plt.figure(figsize=figsize)
plt.xlim(*xlim)
plt.ylim(*ylim)
plt.xticks([])
plt.yticks([])
plt.scatter(samples_1[:2048, 0].cpu().numpy(), samples_1[:2048, 1].cpu().numpy(), alpha=0.15, label=r'$sample_1$', color='red', s = DOT_SIZE)
plt.scatter(samples_2[:2048, 0].cpu().numpy(), samples_2[:2048, 1].cpu().numpy(), alpha=0.15, label=r'$sample_2$', color='green', s = DOT_SIZE)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(dir, 'samples.jpg'), dpi=300)


# draw prior distribution, shape of z is same as z_history
z = np.random.randn(2048, 2)
if wide_prior:
    z[:, 0] = z[:, 0] * 6

plt.figure(figsize=figsize)
plt.xlim(*xlim)
plt.ylim(*ylim)
plt.xticks([])
plt.yticks([])
plt.scatter(z[:, 0], z[:, 1], alpha=0.15, color='blue', s = DOT_SIZE)
plt.tight_layout()
plt.savefig(os.path.join(dir, f"prior.jpg"))


train_data1 = samples_1.detach().clone()[torch.randperm(len(samples_1))]

train_data2 = samples_2.detach().clone()[torch.randperm(len(samples_2))]





forward_model = MLP(output_dim = 4).to(device)
flow_model = MLP().to(device)
optimizer = torch.optim.Adam(list(forward_model.parameters()) + list(flow_model.parameters()), lr=learning_rate)


rectified_flow = RectifiedFlow(device, flow_model)
rectified_flow = train_rectified_flow(rectified_flow, forward_model, optimizer, train_data1, train_data2, batchsize, iterations, arg.pairing)
