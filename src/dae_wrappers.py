


# src/dae_wrappers.py
import torch
import torch.nn as nn


class DAEWrapper(nn.Module):
    """
    Wraps a denoiser D(x_tilde, sigma) (trained with reconstruction MSE)
    into a score model via Vincent's relation:
        score(x_tilde) = (D(x_tilde) - x_tilde) / sigma^2
    """
    def __init__(self, denoiser: nn.Module, sigma_scalar: float):
        super().__init__()
        self.denoiser = denoiser
        self.sigma_scalar = float(sigma_scalar)

    def forward(self, x_tilde, sigma=None):
        s = torch.tensor(self.sigma_scalar, device=x_tilde.device, dtype=x_tilde.dtype)
        x_hat = self.denoiser(x_tilde, s)
        return (x_hat - x_tilde) / (s * s + 1e-8)


def dae_reconstruction_loss(denoiser: nn.Module, x, sigma_scalar: float):
    """
    Vincent-style DAE training objective (squared reconstruction error).
    """
    noise = torch.randn_like(x)
    s = torch.tensor(float(sigma_scalar), device=x.device, dtype=x.dtype)
    x_tilde = x + s * noise
    x_hat = denoiser(x_tilde, s)
    return 0.5 * ((x_hat - x) ** 2).sum(dim=(1, 2, 3)).mean()
