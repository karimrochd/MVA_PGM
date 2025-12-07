import matplotlib
# CRITICAL: Use 'Agg' backend for headless cluster
matplotlib.use('Agg')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

def DEBUG_PRINT(message):
    """Prints a debugging message with a timestamp to track execution flow."""
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {message}", flush=True)

DEBUG_PRINT("01. Imports complete. Starting PyTorch setup.")

# ==============================================================================
# 1. MODEL DEFINITION
# ==============================================================================
class ScoreUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_latent = nn.Conv2d(128, 128, 3, padding=1)
        self.tconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, x, sigma):
        if sigma.dim() == 0:
            sigma = sigma.view(1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        elif sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        h = torch.cat([x, sigma], dim=1)
        h1 = F.relu(self.conv1(h))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        latent = F.relu(self.conv_latent(h3))
        up1 = F.relu(self.tconv1(latent))
        cat1 = torch.cat([up1, h2], dim=1)
        up2 = F.relu(self.tconv2(cat1))
        cat2 = torch.cat([up2, h1], dim=1)
        out = self.conv_out(cat2)
        return out

# ==============================================================================
# 2. INPAINTING SAMPLING FUNCTION
# ==============================================================================
def inpainting_langevin(score_model, x_init, mask, data_original, sigmas, n_steps_each=100, epsilon=2e-5):
    """
    Performs conditional generation using Annealed Langevin Dynamics.
    """
    x = x_init.clone().detach()
    sigma_L = sigmas[-1] # Smallest sigma
    
    for sigma in sigmas:
        sigma = sigma.to(x.device)
        alpha = epsilon * (sigma / sigma_L) ** 2
        
        for _ in range(n_steps_each):
            # 1. Standard Langevin Update
            z = torch.randn_like(x)
            with torch.no_grad():
                score = score_model(x, sigma)
            x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z
            
            # 2. Data Consistency Step
            # Add noise to known pixels to match current sampling noise level
            noise = torch.randn_like(x) * sigma
            known_part = data_original + noise
            
            # Combine: Known (Mask=1) + Generated (Mask=0)
            x = x * (1 - mask) + known_part * mask
            
    return x

# ==============================================================================
# 3. CONFIG & HELPERS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4             # Low LR for stability
EPOCHS = 30           # 30 Epochs for good convergence
BATCH_SIZE = 64
SIGMA_BEGIN = 1.0
SIGMA_END = 0.01
L = 10                # 10 Levels is optimal

DEBUG_PRINT(f"02. Device set to {DEVICE}. Starting data loading check.")

def get_mnist_data(batch_size=64):
    DEBUG_PRINT("03. Attempting to load MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
    DEBUG_PRINT("04. MNIST loaded.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def dsm_loss(model, x, sigma):
    noise = torch.randn_like(x)
    sigma_reshaped = sigma.view(-1, 1, 1, 1) if sigma.dim() == 1 else sigma
    x_tilde = x + noise * sigma_reshaped
    score_pred = model(x_tilde, sigma)
    target = -noise / sigma_reshaped
    loss = 0.5 * ((score_pred - target) ** 2).sum(dim=(1,2,3)) * (sigma ** 2)
    return loss.mean()

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================
def main():
    DEBUG_PRINT("05. Entering main(). Starting process.")
    
    loader = get_mnist_data(batch_size=BATCH_SIZE)
    DEBUG_PRINT("06. Data loading complete. Starting training.")

    # 1. Train the Model
    # --------------------------------------------------------------------------
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(SIGMA_BEGIN), np.log(SIGMA_END), L)),
        device=DEVICE
    ).float()
    
    model = ScoreUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    
    print(f"\nTraining Inpainting Model (L={L})...", flush=True)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_t = time.time()
        for data, _ in loader:
            data = data.to(DEVICE)
            idx = torch.randint(0, len(sigmas), (data.shape[0],), device=DEVICE)
            sigma_batch = sigmas[idx]
            
            optimizer.zero_grad()
            loss = dsm_loss(model, data, sigma_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(loader):.4f} | Time: {time.time()-start_t:.1f}s", flush=True)

    # 2. Setup Inpainting Task (SIMPLIFIED: RANDOM DROPOUT)
    # --------------------------------------------------------------------------
    print("\nRunning Inpainting Task...", flush=True)
    
    data_batch, _ = next(iter(loader))
    data_batch = data_batch[:16].to(DEVICE) 
    
    # --- TASK: RANDOM PIXEL RESTORATION (Salt & Pepper) ---
    # Randomly keep only 40% of pixels. The model must hallucinate the other 60%.
    # This is often easier for CNNs than large holes because context is local.
    probability_keep = 0.4
    mask = torch.bernoulli(torch.full_like(data_batch, probability_keep)).to(DEVICE)
    
    x_init = torch.randn_like(data_batch)
    
    model.eval()
    # Using 200 steps for higher quality
    x_inpainted = inpainting_langevin(
        model, 
        x_init, 
        mask, 
        data_batch, 
        sigmas, 
        n_steps_each=200,
        epsilon=1e-5
    )

    # 3. Plotting
    # --------------------------------------------------------------------------
    print("Plotting Inpainting Results...", flush=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    
    def show(t, ax, tit):
        # Un-normalize for display [-1, 1] -> [0, 1]
        t = (t * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(t, nrow=4, padding=2)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax.set_title(tit)
        ax.axis('off')

    # Visual for "Occluded" input
    # Where mask is 0, show black (-1.0). Where mask is 1, show data.
    occluded_display = data_batch * mask + (1 - mask) * -1.0
    
    show(data_batch, axs[0], "Ground Truth")
    show(occluded_display, axs[1], "Input (60% Pixels Dropped)")
    show(x_inpainted, axs[2], "Restored Result")
    
    plt.tight_layout()
    plt.savefig('exp4_inpainting.png', dpi=200)
    print("Done! Saved 'exp4_inpainting.png'", flush=True)

if __name__ == "__main__":
    main()