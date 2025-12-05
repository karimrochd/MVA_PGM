

import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input: 1 channel (MNIST) + 1 channel (sigma map) = 2 input channels
        
        # Encoder
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) 
        
        # Latent
        self.conv_latent = nn.Conv2d(128, 128, 3, padding=1)
        
        # Decoder
        self.tconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, x, sigma):
        # x: (B, 1, 28, 28)
        # sigma: Tensor of shape (B,) or (B, 1) or scalar
        
        # 1. Expand sigma to match image spatial dims for concatenation
        if not torch.is_tensor(sigma):
             sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)

        # Reshape scale for broadcasting later
        if sigma.dim() == 0:
            sigma_b = sigma.view(1, 1, 1, 1).expand(x.size(0), 1, 1, 1)
        elif sigma.dim() == 1:
            sigma_b = sigma.view(-1, 1, 1, 1)
            
        # Create the sigma map for the network input
        sigma_map = sigma_b.expand(x.size(0), 1, x.size(2), x.size(3))
        
        # 2. Concatenate input image and sigma map
        h = torch.cat([x, sigma_map], dim=1)
        
        # Encoder
        h1 = F.relu(self.conv1(h))       # 32 channels
        h2 = F.relu(self.conv2(h1))      # 64 channels
        h3 = F.relu(self.conv3(h2))      # 128 channels
        
        # Latent
        latent = F.relu(self.conv_latent(h3))
        
        # Decoder (with Skip Connections)
        up1 = F.relu(self.tconv1(latent))
        cat1 = torch.cat([up1, h2], dim=1)
        
        up2 = F.relu(self.tconv2(cat1))
        cat2 = torch.cat([up2, h1], dim=1)
        
        # Raw output from network (Estimates -noise)
        out = self.conv_out(cat2)
        
        # --- CRITICAL FIX ---
        # Normalize the output by sigma. 
        # The network learns to predict the unit-variance score, 
        # and we scale it up effectively here.
        return out / sigma_b
        