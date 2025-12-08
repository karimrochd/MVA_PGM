import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(3, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 128)
        self.lin4 = nn.Linear(128, 2)

    def _prepare_sigma(self, x, sigma):
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
        else:
            sigma = sigma.to(dtype=x.dtype, device=x.device)

        B = x.size(0)

        if sigma.numel() == 1:
            sigma = sigma.view(1, 1).expand(B, 1)

        elif sigma.dim() == 1 and sigma.size(0) == B:
            sigma = sigma.view(B, 1)

        elif sigma.dim() == 2 and sigma.size(0) == B and sigma.size(1) == 1:
            pass

        else:
            raise ValueError(f"Incompatible sigma shape {sigma.shape} for batch size {B}")

        return sigma

    def forward(self, x, sigma):
        sigma = self._prepare_sigma(x, sigma)
        log_sigma = torch.log(sigma + 1e-8)

        h = torch.cat([x, log_sigma], dim=1)

        h = F.softplus(self.lin1(h))
        h = F.softplus(self.lin2(h))
        h = F.softplus(self.lin3(h))
        out = self.lin4(h)

        return out