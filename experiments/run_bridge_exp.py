# ==============================================================================
# BRIDGE EXPERIMENT: MNIST
# Single-Scale DAE vs. Multi-Scale NCSN
# ==============================================================================

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time

# Imports
from data_utils import get_mnist_data
from models_mnist import ScoreUNet
from sampling import langevin_dynamics, annealed_langevin_dynamics

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 5      # 5 is enough for a quick demo; use 10+ for better quality
BATCH_SIZE = 64
SIGMA_BEGIN = 1.0
SIGMA_END = 0.01
NUM_SIGMAS = 10

# --- Loss Function for Images ---
def dsm_loss(model, x, sigma):
    # x: (B, 1, 28, 28)
    
    # 1. Perturb data
    noise = torch.randn_like(x)
    
    # Handle sigma reshaping for broadcasting
    # If sigma is (B,), view as (B, 1, 1, 1)
    if sigma.dim() == 1:
        sigma_reshaped = sigma.view(-1, 1, 1, 1)
    else:
        # If sigma is scalar, simple broadcasting works
        sigma_reshaped = sigma
        
    x_tilde = x + noise * sigma_reshaped
    
    # 2. Predict Score
    score_pred = model(x_tilde, sigma)
    
    # 3. Target Score: -noise / sigma
    target = -noise / sigma_reshaped
    
    # 4. Weighted MSE Loss
    # We weight by sigma^2 to stabilize training across scales
    # Loss = 0.5 * || s - target ||^2 * sigma^2
    # This simplifies algebraically to: 0.5 * || s*sigma + noise ||^2
    # We use the explicit form here for clarity:
    loss = 0.5 * ((score_pred - target) ** 2).sum(dim=(1,2,3)) * (sigma ** 2)
    return loss.mean()

def train_model(model, loader, sigmas, mode='ncsn'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    
    print(f"Training {mode.upper()} Model on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(DEVICE)
            
            # Select Sigma
            if mode == 'dae':
                # Baseline: Train on a single fixed moderate noise level (e.g., 0.1)
                # We expand it to match batch size for consistency
                sigma = torch.tensor([0.1], device=DEVICE).expand(data.shape[0])
            else:
                # Ours: Randomly sample one sigma per image from the schedule
                idx = torch.randint(0, len(sigmas), (data.shape[0],), device=DEVICE)
                sigma = sigmas[idx]
            
            optimizer.zero_grad()
            loss = dsm_loss(model, data, sigma)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # LOGGING: Print every 100 batches
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(loader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1} Complete ({elapsed:.1f}s) | Avg Loss: {avg_loss:.4f}")
        
    return model

def main():
    # 1. Setup Data & Noise Schedule
    print("Loading MNIST...")
    loader = get_mnist_data(batch_size=BATCH_SIZE)
    
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(SIGMA_BEGIN), np.log(SIGMA_END), NUM_SIGMAS)),
        dtype=torch.float32, device=DEVICE
    )
    
    # ==========================================
    # EXPERIMENT A: Baseline (Vincent DAE)
    # Train on SINGLE sigma, Sample with STANDARD Langevin
    # ==========================================
    print("\n" + "="*40)
    print("1. Running Baseline (Vincent DAE)")
    print("Goal: Show that single-scale models produce noisy/blurry samples.")
    print("="*40)
    
    model_dae = ScoreUNet().to(DEVICE)
    model_dae = train_model(model_dae, loader, sigmas, mode='dae')
    
    print("Generating DAE samples...")
    # Generate 16 digits
    x_init = torch.rand(16, 1, 28, 28, device=DEVICE)
    # Sample using the same fixed sigma used in training
    x_dae = langevin_dynamics(model_dae, x_init, sigma=0.1, n_steps=200, step_size=2e-5) 
    
    # ==========================================
    # EXPERIMENT B: Ours (Song NCSN)
    # Train on MULTIPLE sigmas, Sample with ANNEALED Langevin
    # ==========================================
    print("\n" + "="*40)
    print("2. Running Ours (Song NCSN)")
    print("Goal: Show that annealing cleans up samples.")
    print("="*40)
    
    model_ncsn = ScoreUNet().to(DEVICE)
    model_ncsn = train_model(model_ncsn, loader, sigmas, mode='ncsn')
    
    print("Generating NCSN samples...")
    x_init = torch.rand(16, 1, 28, 28, device=DEVICE)
    # Sample using annealing (High noise -> Low noise)
    x_ncsn = annealed_langevin_dynamics(model_ncsn, x_init, sigmas, n_steps_each=20, epsilon=2e-5)
    
    # ==========================================
    # PLOT RESULTS
    # ==========================================
    print("\nPlotting results...")
    
    def show(img_tensor, ax, title):
        # Clamp between 0 and 1 for valid image display
        img_tensor = img_tensor.detach().cpu().clamp(0, 1)
        grid = make_grid(img_tensor, nrow=4, padding=2)
        # Permute (C, H, W) -> (H, W, C) for Matplotlib
        ax.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    show(x_dae, axs[0], 
         "Baseline (Vincent DAE)\nSingle scale training ($\sigma=0.1$)\nStandard Langevin")
    
    show(x_ncsn, axs[1], 
         "Ours (Song NCSN)\nMulti-scale training ($\sigma_1 \\to \sigma_L$)\nAnnealed Langevin")
    
    plt.tight_layout()
    plt.savefig('bridge_experiment_mnist.png')
    plt.show()
    print("Done! Saved plot to 'bridge_experiment_mnist.png'")

if __name__ == "__main__":
    main()