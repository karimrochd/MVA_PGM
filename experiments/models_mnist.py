



import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # We inject sigma by concatenating it as an extra channel
        # Input: 1 channel (MNIST) + 1 channel (sigma map) = 2 input channels
        
        # Encoder
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # Downsample
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1) # Downsample
        
        # Latent
        self.conv_latent = nn.Conv2d(128, 128, 3, padding=1)
        
        # Decoder
        self.tconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1) # Concat skip
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1) # Concat skip
        
    def forward(self, x, sigma):
        # x: (B, 1, 28, 28)
        # sigma: (B, 1) or scalar
        
        # 1. Expand sigma to match image spatial dims: (B, 1, 28, 28)
        if sigma.dim() == 0:
            sigma = sigma.view(1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        elif sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
            
        # 2. Concatenate input image and sigma map
        # h shape: (B, 2, 28, 28)
        h = torch.cat([x, sigma], dim=1)
        
        # Encoder
        h1 = F.relu(self.conv1(h))       # 32 channels
        h2 = F.relu(self.conv2(h1))      # 64 channels
        h3 = F.relu(self.conv3(h2))      # 128 channels
        
        # Latent
        latent = F.relu(self.conv_latent(h3))
        
        # Decoder (with Skip Connections)
        # up1 shape: (B, 64, 14, 14)
        up1 = F.relu(self.tconv1(latent))
        
        # Concatenate with h2 (skip connection)
        # cat1 shape: (B, 64+64=128, 14, 14)
        cat1 = torch.cat([up1, h2], dim=1)
        
        # up2 shape: (B, 32, 28, 28)
        up2 = F.relu(self.tconv2(cat1))
        
        # Concatenate with h1
        # cat2 shape: (B, 32+32=64, 28, 28)
        cat2 = torch.cat([up2, h1], dim=1)
        
        # Output score
        out = self.conv_out(cat2)
        return out