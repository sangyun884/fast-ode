import torch
import numpy as np
from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os


def get_train_data(COMP, D, VAR):
    initial_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
    initial_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., D / 2.], [-D * np.sqrt(3) / 2., D / 2.], [0.0, - D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
    initial_model = MixtureSameFamily(initial_mix, initial_comp)
    samples_1 = initial_model.sample([1000000])

    target_mix = Categorical(torch.tensor([1/COMP for i in range(COMP)]))
    target_comp = MultivariateNormal(torch.tensor([[D * np.sqrt(3) / 2., - D / 2.], [-D * np.sqrt(3) / 2., - D / 2.], [0.0, D * np.sqrt(3) / 2.]]).float(), VAR * torch.stack([torch.eye(2) for i in range(COMP)]))
    target_model = MixtureSameFamily(target_mix, target_comp)
    samples_2 = target_model.sample([100000])

    return samples_1, samples_2

def get_train_data_two_gaussian(mu1, mu2, cov1, cov2):
    # mu_val = 8.
    # cov_val = 0.
    # mu1 = torch.tensor([-mu_val, mu_val])
    # mu2 = torch.tensor([mu_val, mu_val])
    # cov1 = torch.tensor([[3, cov_val], [cov_val, .1]])
    # cov2 = torch.tensor([[3, -cov_val], [-cov_val, .1]])

    dist1 = MultivariateNormal(mu1, cov1)
    dist2 = MultivariateNormal(mu2, cov2)

    samples_1 = dist1.sample([100000])
    samples_2 = dist2.sample([100000])

    return samples_1, samples_2


@torch.no_grad()
def draw_plot(x1, x2, x_recon1, x_recon2, z1, z2, i, DOT_SIZE, M, dir, mu_prior, var_prior):
    z = np.random.normal(size=z1.shape)
    z = mu_prior + var_prior ** 0.5 * z
    x1 = x1.detach().cpu().numpy()
    x2 = x2.detach().cpu().numpy()
    x_recon1 = x_recon1.detach().cpu().numpy()
    x_recon2 = x_recon2.detach().cpu().numpy()
    z1 = z1.detach().cpu().numpy()
    z2 = z2.detach().cpu().numpy()

    # Draw x1, x2, x_recon1, x_recon2 with labels
    plt.figure(figsize=(4, 4))
    plt.scatter(x1[:, 0], x1[:, 1], alpha=0.15, color="red", s = DOT_SIZE*3)
    plt.scatter(x2[:, 0], x2[:, 1], alpha=0.15, color="orange", s = DOT_SIZE*3)
    plt.scatter(x_recon1[:, 0], x_recon1[:, 1], alpha=0.15, color="blue", s = DOT_SIZE*3)
    plt.scatter(x_recon2[:, 0], x_recon2[:, 1], alpha=0.15, color="green", s = DOT_SIZE*3)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.legend(["x1", "x2", "x_recon1", "x_recon2"])
    plt.title("x1, x2, x_recon1, x_recon2")
    plt.savefig(os.path.join(dir, f"recon_{i}.jpg"))

    # Draw z, z1, z2 with labels
    plt.figure(figsize=(4, 4))
    plt.scatter(z[:, 0], z[:, 1], alpha=0.15, color="black", s = DOT_SIZE)
    plt.scatter(z1[:, 0], z1[:, 1], alpha=0.15, color="red", s = DOT_SIZE)
    plt.scatter(z2[:, 0], z2[:, 1], alpha=0.15, color="orange", s = DOT_SIZE)
    plt.xlim(-M, M)
    plt.ylim(-M, M)
    plt.legend(["z", "z1", "z2"])
    plt.title("z, z1, z2")
    plt.savefig(os.path.join(dir, f"z_{i}.jpg"))

    # close all figures
    plt.close('all')
def cosine_similarity(x1, x2):
    x1 = x1.view(x1.shape[0], -1)
    x2 = x2.view(x2.shape[0], -1)
    x1_norm = x1 / x1.norm(dim=1, keepdim=True)
    x2_norm = x2 / x2.norm(dim=1, keepdim=True)
    return torch.sum(x1_norm * x2_norm, dim=1)
def straightness(traj):
    N = len(traj) - 1
    dt = 1 / N
    base = traj[0] - traj[-1]
    mse = []
    for i in range(1, len(traj)):
        v = (traj[i-1] - traj[i]) / dt
        mse.append(torch.mean((v - base) ** 2))
    return torch.mean(torch.stack(mse))
def get_kl(mu, logvar):
    # Return KL divergence between N(mu, var) and N(0, 1), divided by data dimension.
    kl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[1,2,3])
    loss_prior = torch.mean(kl) / (mu.shape[1]*mu.shape[2]*mu.shape[3])
    return loss_prior
def get_kl_2d(mu, logvar, wide_prior = True):
    if wide_prior:
        kl = 0.5 * torch.sum(-1 + np.log(36) - logvar + mu.pow(2) / torch.tensor([36,1], device=mu.device) + logvar.exp() / torch.tensor([36,1], device=mu.device), dim=1)
    else:
        # Return KL divergence between N(mu, var) and N(0, 1), divided by data dimension.
        kl = 0.5 * torch.sum(-1 - logvar + mu.pow(2) + logvar.exp(), dim=[1])
    loss_prior = torch.mean(kl) / 2
    return loss_prior
def get_kl_2d_gen(mu1, logvar1, mu2, var2):
    # Generalized KL divergence between N(mu1, var1) and N(mu2, var2), divided by data dimension.
    mu2, var2 = mu2.unsqueeze(0), var2.unsqueeze(0)
    # Return KL divergence between N(mu1, var1) and N(mu2, var2), divided by data dimension.
    kl = 0.5 * (torch.sum(torch.log(var2), dim = 1) - torch.sum(logvar1, dim = 1) - 2 + torch.sum((mu1 - mu2) ** 2 / var2, dim = 1) + torch.sum(logvar1.exp() / var2, dim = 1))
    # kl = 0.5 * torch.sum(-1 + np.log(36) - logvar1 + mu1.pow(2) / torch.tensor([36,1], device=mu1.device) + logvar1.exp() / torch.tensor([36,1], device=mu1.device), dim=1)

    loss_prior = torch.mean(kl) / 2
    return loss_prior

def alpha(t):
    # DDPM defines x_t(x, z) = alpha(t)x + sqrt(1 - alpha(t)^2)z
    a = 19.9
    b = 0.1
    exp = torch.exp if isinstance(t, torch.Tensor) else np.exp
    return exp(-0.25 * a * t ** 2 - 0.5 * b * t)
def dalpha_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return (-0.5 * a * t - 0.5 * b) * alpha_t
def d_1_minus_alpha_sq_dt(t):
    a = 19.9
    b = 0.1
    alpha_t = alpha(t)
    return 0.5 * (1 - alpha_t ** 2) ** (-0.5) * (-2 * alpha_t) * dalpha_dt(t)