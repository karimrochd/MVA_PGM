# ==============================================================================
# MAIN EXPERIMENT: 2D TOY DATA (Visual Proof & Mixing Analysis)
# Orchestrates training and plotting for Sections 5.1 and 5.2 of the report.
# ==============================================================================

import matplotlib
# CRITICAL FIX: Use 'Agg' backend to prevent crash on headless cluster
matplotlib.use('Agg') 

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Assumes these files exist in the same directory
from data_utils import get_toy_data
from models import ScoreNet
from sampling import langevin_dynamics, annealed_langevin_dynamics

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
N_EPOCHS = 10000
BATCH_SIZE = 128
LR = 1e-3
SIGMA_BEGIN = 5.0
SIGMA_END = 0.01
NUM_SIGMAS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

# ------------------------------------------------------------------------------
# Denoising Score Matching Loss (Vincent 2011 / Song 2019)
# ------------------------------------------------------------------------------
def dsm_loss(model, x, sigma):
    # x: (B,2)
    B = x.size(0)

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, dtype=x.dtype, device=x.device)
    else:
        sigma = sigma.to(dtype=x.dtype, device=x.device)

    # Broadcast sigma to shape (B,1)
    if sigma.dim() == 0:
        sigma = sigma.view(1, 1).expand(B, 1)
    elif sigma.dim() == 1 and sigma.size(0) == B:
        sigma = sigma.view(B, 1)
    elif sigma.dim() == 2 and sigma.size(0) == B and sigma.size(1) == 1:
        pass
    else:
        raise ValueError(f"Incompatible sigma shape {sigma.shape} for batch size {B}")

    # 1. Perturb data
    noise = torch.randn_like(x) * sigma
    x_tilde = x + noise

    # 2. Predict score
    score_pred = model(x_tilde, sigma)  # sigma already (B,1)

    # 3. Target score (Gradient of log q_sigma)
    target = -noise / (sigma ** 2)

    # 4. Per-sample weighted loss
    sq_err = ((score_pred - target) ** 2).sum(dim=1)  # (B,)
    weights = (sigma.squeeze(1) ** 2)                 # (B,)
    loss = 0.5 * (sq_err * weights).mean()

    return loss


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    print(f"Running on {DEVICE}")

    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Dataset
    # ---------------------------------------------------------
    data = get_toy_data(n_samples=5000, seed=SEED).to(DEVICE)

    # 2. Define Noise Schedule (Geometric Sequence)
    # ---------------------------------------------------------
    sigmas = torch.tensor(
        np.exp(np.linspace(np.log(SIGMA_BEGIN), np.log(SIGMA_END), NUM_SIGMAS)),
        dtype=torch.float32,
        device=DEVICE,
    )

    # 3. Model & Optimizer
    # ---------------------------------------------------------
    model = ScoreNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 4. Training Loop
    # ---------------------------------------------------------
    print("Starting Training...")
    for epoch in range(N_EPOCHS):
        # Randomly sample a batch of data
        indices = torch.randint(0, data.shape[0], (BATCH_SIZE,), device=DEVICE)
        batch = data[indices]

        # Sample a sigma independently for each sample in the batch
        sigma_idx = torch.randint(0, len(sigmas), (BATCH_SIZE,), device=DEVICE)
        sigma_batch = sigmas[sigma_idx]  # (BATCH_SIZE,)

        optimizer.zero_grad()
        loss = dsm_loss(model, batch, sigma_batch)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0: # Reduced print frequency to keep logs clean
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    print("Training Complete.")

    # ---------------------------------------------------------
    # EXPERIMENT PART 1: VISUAL PROOF (VECTOR FIELD)
    # ---------------------------------------------------------
    print("Generating Vector Field Plot...")

    # Define grid
    x_lin = np.linspace(-8, 8, 25)
    y_lin = np.linspace(-8, 8, 25)
    xx, yy = np.meshgrid(x_lin, y_lin)
    grid_points = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=DEVICE)

    # Choose small sigma for visualization (local score behaviour)
    vis_sigma = sigmas[-1]  # smallest sigma

    model.eval()
    with torch.no_grad():
        scores = model(grid_tensor, vis_sigma).cpu().numpy()

    plt.figure(figsize=(8, 8))
    plt.scatter(
        data.cpu()[:, 0],
        data.cpu()[:, 1],
        s=1,
        alpha=0.15,
        color="gray",
        label="Data",
    )

    # Use true magnitudes for arrows (no full normalization)
    plt.quiver(
        grid_points[:, 0],
        grid_points[:, 1],
        scores[:, 0],
        scores[:, 1],
        angles="xy",
        scale_units="xy",
        color="red",
        alpha=0.8,
        label="Learned Score",
    )

    plt.title(f"Visual Proof: Learned Score Field (sigma={vis_sigma.item():.3f})")
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    # Save and Print (No plt.show)
    plt.savefig("exp1_vector_field.png", dpi=200)
    print("Saved 'exp1_vector_field.png'")
    plt.close() # Good practice to close figure

    # ---------------------------------------------------------
    # EXPERIMENT PART 2: THE MIXING PROBLEM
    # Compare Standard vs. Annealed Langevin
    # ---------------------------------------------------------
    print("Running Mixing Comparison...")

    # FIX: Initialize with Gaussian Noise scaled by largest sigma (theoretical standard)
    # This covers the whole space better than uniform noise
    x_init = torch.randn(1000, 2, device=DEVICE) * SIGMA_BEGIN

    # A. Standard Langevin (Fixed small sigma)
    fixed_sigma = sigmas[-1]  # Smallest sigma
    x_standard = langevin_dynamics(
        model,
        x_init,
        fixed_sigma,
        n_steps=1000,
        step_size=1e-2,
    )

    # B. Annealed Langevin (Sequence of sigmas)
    x_annealed = annealed_langevin_dynamics(
        model,
        x_init,
        sigmas,
        n_steps_each=100,
        epsilon=2e-6,
    )

    # Plot Comparison
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Standard Langevin
    ax[0].scatter(
        data.cpu()[:, 0], data.cpu()[:, 1], s=1, alpha=0.1, color="gray"
    )
    ax[0].scatter(
        x_standard.cpu()[:, 0], x_standard.cpu()[:, 1], s=2, alpha=0.8, color="blue"
    )
    ax[0].set_title(
        r"Standard Langevin (Fixed $\sigma_{\text{small}}$)"
        "\nProblem: Samples stuck, poor mixing"
    )
    ax[0].set_xlim(-8, 8)
    ax[0].set_ylim(-8, 8)
    ax[0].grid(True, alpha=0.3)

    # Annealed Langevin
    ax[1].scatter(
        data.cpu()[:, 0], data.cpu()[:, 1], s=1, alpha=0.1, color="gray"
    )
    ax[1].scatter(
        x_annealed.cpu()[:, 0], x_annealed.cpu()[:, 1], s=2, alpha=0.8, color="green"
    )
    ax[1].set_title(
        r"Annealed Langevin (Sequence $\sigma_1 \to \sigma_L$)"
        "\nSolution: Samples cover all modes"
    )
    ax[1].set_xlim(-8, 8)
    ax[1].set_ylim(-8, 8)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save and Print (No plt.show)
    plt.savefig("exp1_mixing_comparison.png", dpi=200)
    print("Saved 'exp1_mixing_comparison.png'")
    plt.close()

if __name__ == "__main__":
    main()