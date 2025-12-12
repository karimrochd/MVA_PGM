# src/models_mnist.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreUNet(nn.Module):
    """
    Simple U-Net style score network for MNIST.
    Conditioning: concat a sigma map as an extra channel.
    Output convention: returns score(x, sigma) (scaled as out / sigma).
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.conv_latent = nn.Conv2d(128, 128, 3, padding=1)

        self.tconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)

    def _sigma_to_b1111(self, x, sigma):
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
        else:
            sigma = sigma.to(device=x.device, dtype=x.dtype)

        B = x.size(0)

        if sigma.numel() == 1:
            sigma_b = sigma.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif sigma.dim() == 1 and sigma.size(0) == B:
            sigma_b = sigma.view(B, 1, 1, 1)
        elif sigma.dim() == 2 and sigma.size(0) == B and sigma.size(1) == 1:
            sigma_b = sigma.view(B, 1, 1, 1)
        else:
            raise ValueError(f"Bad sigma shape {tuple(sigma.shape)} for batch size {B}")

        return sigma_b

    def forward(self, x, sigma):
        sigma_b = self._sigma_to_b1111(x, sigma)
        sigma_map = sigma_b.expand(x.size(0), 1, x.size(2), x.size(3))

        h = torch.cat([x, sigma_map], dim=1)

        h1 = F.relu(self.conv1(h))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))

        latent = F.relu(self.conv_latent(h3))

        up1 = F.relu(self.tconv1(latent))
        cat1 = torch.cat([up1, h2], dim=1)

        up2 = F.relu(self.tconv2(cat1))
        cat2 = torch.cat([up2, h1], dim=1)

        out = self.conv_out(cat2)

        # output scaling convention (kept consistent with your current code)
        return out / sigma_b.clamp_min(1e-4)
