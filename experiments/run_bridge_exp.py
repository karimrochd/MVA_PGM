# ==============================================================================
# BRIDGE EXPERIMENT: MNIST
# Single-Scale DAE vs. Multi-Scale NCSN
# ==============================================================================

import matplotlib
# CRITICAL FIX FOR CLUSTER: Use 'Agg' backend to save files without a screen
matplotlib.use('Agg') 

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Imports from your local files
from models_mnist import ScoreUNet
from sampling import langevin_dynamics, annealed_langevin_dynamics

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 10           # Reduced to 10 for faster iteration
BATCH_SIZE = 64
SIGMA_BEGIN = 1.0
SIGMA_END = 0.01
NUM_SIGMAS = 10

# ------------------------------------------------------------------------------
# Data Loader
# ------------------------------------------------------------------------------
def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalize inputs to roughly [-1, 1] for better Score Matching stability
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    # Download to a local folder './data'
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# ------------------------------------------------------------------------------
# Loss Function (Denoising Score Matching)
# ------------------------------------------------------------------------------
def dsm_loss(model, x, sigma):
    noise = torch.randn_like(x)
    
    # Handle sigma reshaping for broadcasting
    if sigma.dim() == 1:
        sigma_reshaped = sigma.view(-1, 1, 1, 1)
    else:
        sigma_reshaped = sigma
        
    x_tilde = x + noise * sigma_reshaped
    
    # Predict Score
    score_pred = model(x_tilde, sigma)
    
    # Target Score: -noise / sigma
    target = -noise / sigma_reshaped
    
    # Weighted MSE Loss
    # We weight by sigma^2 to stabilize the objective across scales
    loss = 0.5 * ((score_pred - target) ** 2).sum(dim=(1,2,3)) * (sigma ** 2)
    return loss.mean()

# ------------------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------------------
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
                # Baseline: Train on a single fixed noise level
                sigma = torch.tensor([0.1], device=DEVICE).expand(data.shape[0])
            else:
                # Ours: Randomly sample one sigma per image
                idx = torch.randint(0, len(sigmas), (data.shape[0],), device=DEVICE)
                sigma = sigmas[idx]
            
            optimizer.zero_grad()
            loss = dsm_loss(model, data, sigma)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        # Log once per epoch
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        
    return model

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    print("Loading MNIST...")
    loader = get_mnist_data(batch_size=BATCH_SIZE)
    
    # Geometric sequence of sigmas
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(SIGMA_BEGIN), np.log(SIGMA_END), NUM_SIGMAS)),
        dtype=torch.float32, device=DEVICE
    )
    
    # ==========================================
    # EXPERIMENT A: Baseline (Vincent DAE)
    # ==========================================
    print("\n1. Running Baseline (Vincent DAE)")
    model_dae = ScoreUNet().to(DEVICE)
    model_dae = train_model(model_dae, loader, sigmas, mode='dae')
    
    print("Generating DAE samples...")
    # Initialize with Gaussian Noise
    x_init = torch.randn(16, 1, 28, 28, device=DEVICE)
    
    x_dae = langevin_dynamics(
        model_dae, 
        x_init, 
        sigma=0.1, 
        n_steps=1000, 
        step_size=1e-4 
    ) 
    
    # ==========================================
    # EXPERIMENT B: Ours (Song NCSN)
    # ==========================================
    print("\n2. Running Ours (Song NCSN)")
    model_ncsn = ScoreUNet().to(DEVICE)
    model_ncsn = train_model(model_ncsn, loader, sigmas, mode='ncsn')
    
    print("Generating NCSN samples...")
    x_init = torch.randn(16, 1, 28, 28, device=DEVICE)
    
    x_ncsn = annealed_langevin_dynamics(
        model_ncsn, 
        x_init, 
        sigmas, 
        n_steps_each=100, 
        epsilon=2e-5
    )
    
    # ==========================================
    # PLOT RESULTS
    # ==========================================
    print("\nPlotting results...")
    
    def show(img_tensor, ax, title):
        # We trained on [-1, 1], so we un-normalize to [0, 1] for display
        img_tensor = (img_tensor * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(img_tensor, nrow=4, padding=2)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Using raw strings r"..." to fix the \sigma escape warning
    show(x_dae, axs[0], 
         r"Baseline (DAE)" + "\n" + r"Fixed $\sigma=0.1$")
    
    show(x_ncsn, axs[1], 
         r"Ours (NCSN)" + "\n" + r"Annealing $\sigma_1 \to \sigma_L$")
    
    plt.tight_layout()
    
    # CLUSTER FIX: Save instead of show
    output_filename = 'bridge_experiment_mnist.png'
    plt.savefig(output_filename)
    print(f"Done! Saved plot to '{output_filename}'")
    
    # plt.show() # Commented out to prevent crash

if __name__ == "__main__":
    main()