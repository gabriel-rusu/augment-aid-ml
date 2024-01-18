import torch
from torch import nn


class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
        )

        self.mean_layer = nn.Linear(latent_dim, 2)
        self.log_var_layer = nn.Linear(latent_dim, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.mean_layer(x), self.log_var_layer(x)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var, device=self.device)
        z = mean + var * epsilon
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, var = self.encode(x)
        z = self.reparameterization(mean, var)
        return self.decoder(z), mean, var
