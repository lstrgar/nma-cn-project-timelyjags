import torch
from torch import nn


class ReferenceVAE(nn.Module):
    """
    Implementation of the reference VAE model.
    Described by Han et. Al. in https://www.biorxiv.org/content/10.1101/214247v1.full.pdf
    """

    def __init__(self):
        super(ReferenceVAE, self).__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 128, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 1024, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(1024 * 4 * 4, 1024)
        self.enc_logvar = nn.Linear(1024 * 4 * 4, 1024)
        self.dec_fc = nn.Linear(1024, 1024 * 4 * 4)
        self.dec_transconv = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 512, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        x = self.enc_conv(x)
        x = x.view(x.size(0), 1024 * 4 * 4)
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)
        z = self.reparameterize(mu, logvar)
        y = self.dec_fc(z)
        y = y.view(y.size(0), 1024, 4, 4)
        y = self.dec_transconv(y)
        return y, z, mu, logvar
