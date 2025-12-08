import matplotlib
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

# --- CONFIGURATION FOR BETTER GENERATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 50          # INCREASED: 10 -> 50 for convergence
BATCH_SIZE = 64
SIGMA_BEGIN = 10.0   # INCREASED: 1.0 -> 10.0 for better initial mixing
SIGMA_END = 0.01
L_VALUES = [1, 10, 50]

def DEBUG_PRINT(message):
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {message}", flush=True)

DEBUG_PRINT("01. Imports complete. Starting PyTorch setup.")

class ScoreUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard U-Net like architecture for score estimation
        self.conv1 = nn.Conv2d(2, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv_latent = nn.Conv2d(128, 128, 3, padding=1)
        self.tconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.tconv2 = nn.ConvTranspose2d(128, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self, x, sigma):
        # Conditioning on sigma
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

def annealed_langevin_dynamics(score_model, x_init, sigmas, n_steps_each=100, epsilon=2e-5):
    """
    Performs Annealed Langevin Dynamics sampling.
    """
    x = x_init.clone().detach()
    sigma_L = sigmas[-1] # Smallest sigma
    
    for i, sigma in enumerate(sigmas):
        sigma = sigma.to(x.device)
        # Step size calculation from Song & Ermon (2019)
        alpha = epsilon * (sigma / sigma_L) ** 2
        
        for _ in range(n_steps_each):
            z = torch.randn_like(x)
            with torch.no_grad():
                score = score_model(x, sigma)
            # Langevin update step
            x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z
            
    return x

DEBUG_PRINT(f"02. Device set to {DEVICE}. Starting data loading check.")

def get_mnist_data(batch_size=64):
    DEBUG_PRINT("03. Downloading/Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize to [-1, 1] usually preferred for GANs/EBMs, 
        # but [0,1] or standard normalization works if consistent.
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    try:
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    except:
        # Fallback if download fails (e.g. Kaggle/Colab internet issues)
        print("Download failed, attempting local load...")
        dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
        
    DEBUG_PRINT("04. MNIST loaded.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def dsm_loss(model, x, sigma):
    """
    Denoising Score Matching Loss
    """
    noise = torch.randn_like(x)
    sigma_reshaped = sigma.view(-1, 1, 1, 1) if sigma.dim() == 1 else sigma
    
    # Perturb data
    x_tilde = x + noise * sigma_reshaped
    
    # Estimate score
    score_pred = model(x_tilde, sigma)
    
    # Target score: -noise / sigma
    target = -noise / sigma_reshaped
    
    # Loss: 1/2 * || score - target ||^2 * sigma^2 (weighting)
    loss = 0.5 * ((score_pred - target) ** 2).sum(dim=(1,2,3)) * (sigma ** 2)
    return loss.mean()

def train_model_for_L(L, loader):
    print(f"\nTraining Model with L = {L} Noise Levels", flush=True)
    
    # Define sigma schedule
    if L == 1:
        # Baseline: Fixed small noise
        sigmas = torch.tensor([SIGMA_END], device=DEVICE).float()
    else:
        # Geometric sequence from high to low
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(SIGMA_BEGIN), np.log(SIGMA_END), L)),
            device=DEVICE
        ).float()
        
    model = ScoreUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_t = time.time()
        
        for data, _ in loader:
            data = data.to(DEVICE)
            
            # Sample random sigma for each image in batch
            idx = torch.randint(0, len(sigmas), (data.shape[0],), device=DEVICE)
            sigma_batch = sigmas[idx]
            
            optimizer.zero_grad()
            loss = dsm_loss(model, data, sigma_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"  L={L} | Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {time.time()-start_t:.1f}s", flush=True)
        
    return model, sigmas

def main():
    DEBUG_PRINT("05. Entering main(). Starting process.")
    
    loader = get_mnist_data(batch_size=BATCH_SIZE)
    DEBUG_PRINT("06. Data loading complete. Starting loops.")

    results = []
    
    for L in L_VALUES:
        model, sigmas = train_model_for_L(L, loader)
        print(f"Generating samples for L={L}...", flush=True)
        
        # Sampling
        x_init = torch.randn(16, 1, 28, 28, device=DEVICE) # Batch of 16 for grid
        
        # INCREASED: More steps for better refinement
        n_steps = 1000 if L == 1 else 200 
        
        model.eval()
        x_sample = annealed_langevin_dynamics(
            model, x_init, sigmas, n_steps_each=n_steps, epsilon=2e-5
        )
        results.append((L, x_sample))

    print("Plotting Ablation Results...", flush=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    
    for i, (L, sample) in enumerate(results):
        # Denormalize
        sample = (sample * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(sample, nrow=4, padding=2)
        
        axs[i].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        axs[i].set_title(f"Noise Levels L={L}\n({L} scales)")
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig('exp3_ablation_study_improved.png')
    print("Done! Saved 'exp3_ablation_study_improved.png'", flush=True)

if __name__ == "__main__":
    main()