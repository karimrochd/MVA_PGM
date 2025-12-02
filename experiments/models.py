# ==============================================================================
# MODELS
# Defines the Neural Network architecture for the Score Function approximation.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Class: ScoreNet
# A simple 4-layer MLP that estimates the score vector s(x, sigma).
#
# Architecture:
#   Input : concatenation of x (2D) and log_sigma (1D) → total dim = 3
#   Hidden: 3 layers of 128 units with Softplus activation
#   Output: 2D vector (estimated gradient of log-density)
# ------------------------------------------------------------------------------
class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.lin4 = nn.Linear(128, 2)

    def _prepare_sigma(self, x, sigma):
        """
        Convert sigma to a tensor of shape (B, 1) on the same device/dtype as x.

        Accepts:
          - scalar python float
          - scalar / 0D tensor
          - (1,) tensor
          - (B,) tensor
          - (B,1) tensor

        Returns:
          sigma_batched: tensor of shape (B, 1)
        """
        # Ensure sigma is a tensor on the same device/dtype as x
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
        else:
            sigma = sigma.to(dtype=x.dtype, device=x.device)

        B = x.size(0)

        # Treat any single-element tensor as a scalar and broadcast
        if sigma.numel() == 1:
            sigma = sigma.view(1, 1).expand(B, 1)

        # Batch of sigmas, one per sample
        elif sigma.dim() == 1 and sigma.size(0) == B:
            sigma = sigma.view(B, 1)

        # Already in (B,1) form
        elif sigma.dim() == 2 and sigma.size(0) == B and sigma.size(1) == 1:
            pass

        else:
            raise ValueError(f"Incompatible sigma shape {sigma.shape} for batch size {B}")

        return sigma

    def forward(self, x, sigma):
        """
        Args:
          x     : (B, 2) data points
          sigma : noise level(s), any of the accepted shapes in _prepare_sigma

        Returns:
          score : (B, 2) estimated score vectors
        """
        sigma = self._prepare_sigma(x, sigma)
        # Use log sigma as conditioning input (more stable across scales)
        log_sigma = torch.log(sigma + 1e-8)

        # Concatenate x and log_sigma → (B, 3)
        h = torch.cat([x, log_sigma], dim=1)

        # MLP with Softplus activations
        h = F.softplus(self.lin1(h))
        h = F.softplus(self.lin2(h))
        h = F.softplus(self.lin3(h))
        out = self.lin4(h)

        return out
