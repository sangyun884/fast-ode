from torch import nn
import torch
import torch.nn.functional as F
from guided_diffusion.unet import UNetModel
from einops import rearrange
import numpy as np
class AE(nn.Module):
    def __init__(self, input_dim, z_dim, num_layers=2, channels=128):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        en_layers = []
        de_layers = []
        en_layers.append(nn.Linear(input_dim, channels))
        for i in range(num_layers-1):
            en_layers.append(nn.Linear(channels, channels))
            en_layers.append(nn.ReLU())
        en_layers.append(nn.Linear(channels, z_dim*2))
        self.en_layers = nn.Sequential(*en_layers)

        de_layers.append(nn.Linear(z_dim, channels))
        for i in range(num_layers-1):
            de_layers.append(nn.Linear(channels, channels))
            de_layers.append(nn.ReLU())
        de_layers.append(nn.Linear(channels, input_dim))
        self.de_layers = nn.Sequential(*de_layers)
    def forward(self, x):
        mu_logvar = self.en_layers(x)
        mu, logvar = mu_logvar[:, :self.z_dim], mu_logvar[:, self.z_dim:]
        z = mu + torch.randn_like(mu)*torch.exp(logvar/2)
        x_recon = self.de_layers(z)
        return x_recon, z, mu, logvar
    def encode(self, x):
        mu_logvar = self.en_layers(x)
        mu, logvar = mu_logvar[:, :self.z_dim], mu_logvar[:, self.z_dim:]
        z = mu + torch.randn_like(mu)*torch.exp(logvar/2)
        return z


class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim = 2, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, output_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
    
    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x

class UNet(nn.Module):
    def __init__(self, input_nc=1, output_nc = 1, ngf=64, norm_layer='bn', use_dropout=False):
        super().__init__()
        self.conv_en1 = nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(ngf)
        self.conv_en2 = nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(ngf*2)
        self.conv_bottleneck = nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_de1 = nn.Conv2d(ngf*2, ngf, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(ngf)
        self.conv_de2 = nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.ReLU()
        
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        # Bilinear up/down sample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.AvgPool2d(2, stride=2)
    def forward(self, x, t=None):
        x_en1 = self.conv_en1(x)
        x_en1 = self.act(x_en1)
        x_en1 = self.norm1(x_en1)
        if self.use_dropout:
            x_en1 = self.dropout(x_en1)
        x_en1 = self.downsample(x_en1)
        x_en2 = self.conv_en2(x_en1)
        x_en2 = self.act(x_en2)
        x_en2 = self.norm2(x_en2)
        if self.use_dropout:
            x_en2 = self.dropout(x_en2)
        x_en2 = self.downsample(x_en2)

        
        x_bottleneck = self.conv_bottleneck(x_en2)
        x_bottleneck = self.act(x_bottleneck)
        
        x_de1 = self.upsample(x_bottleneck)
        x_de1 = self.conv_de1(x_de1)
        x_de1 = self.act(x_de1)
        x_de1 = self.norm3(x_de1)
        if self.use_dropout:
            x_de1 = self.dropout(x_de1)
        x_de1 = x_de1 + x_en1

        x_de2 = self.upsample(x_de1)
        x_de2 = self.conv_de2(x_de2)

        x_de2 = x_de2

        return x_de2

class UNetAE(nn.Module):
    def __init__(self, res, input_nc=1, output_nc = 2, ngf=64, norm_layer=nn.BatchNorm2d, large = False, use_dropout=False, encoder_only = False):
        super().__init__()
        self.input_nc = input_nc
        if not large:
            self.encoder = UNet(input_nc, output_nc, ngf, norm_layer, use_dropout)
            if not encoder_only:
                self.decoder = UNet(input_nc, input_nc, ngf, norm_layer, use_dropout)
        else:
            self.encoder = UNetModel(res, input_nc, 32, output_nc, no_time = True)
            if not encoder_only:
                self.decoder = UNetModel(res, input_nc, 32, input_nc, no_time = True)


    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z, mu, logvar
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar[:, :self.input_nc], mu_logvar[:, self.input_nc:]
        z = mu + torch.randn_like(mu)*torch.exp(logvar/2)
        return z, mu, logvar
    def decode(self, z):
        x_recon = self.decoder(z)
        return x_recon
class UNetEncoder(nn.Module):
    def __init__(self, encoder, input_nc=3):
        super().__init__()
        if encoder is None:
            raise NotImplementedError
        self.input_nc = input_nc
        self.encoder = encoder
        

    def forward(self, x, t = None, noise = None):
        if t is None:
            t = torch.ones((x.shape[0]), device=x.device)
        if noise is None:
            noise = torch.randn_like(x)
        z, mu, logvar = self.encode(x, t, noise)
        return z, mu, logvar
    def encode(self, x, t, noise):
        mu_logvar = self.encoder(x, t)
        mu, logvar = mu_logvar[:, :self.input_nc], mu_logvar[:, self.input_nc:]
        z = mu + noise*torch.exp(logvar/2)
        return z, mu, logvar



class FourierMLP(nn.Module):
    def __init__(self, input_dim=2, output_dim = 2, num_layers=2, channels=128):
        super().__init__()
        self.data_shape = [input_dim]
        self.output_dim = output_dim

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(input_dim)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(output_dim))),
        )

    def forward(self, inputs, cond):
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(inputs.shape[0], self.output_dim)