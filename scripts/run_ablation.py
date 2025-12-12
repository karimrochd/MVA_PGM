import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from src.run_utils import set_seed, make_run_dir, save_json, env_info
from src.models_mnist import ScoreUNet
from src.sampling import annealed_langevin_dynamics


# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 64
SIGMA_BEGIN = 10.0
SIGMA_END = 0.01
L_VALUES = [1, 10, 50]

SEED = 0

CONFIG = {
    "script": "scripts/run_ablation.py",
    "seed": SEED,
    "device": str(DEVICE),
    "lr": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "sigma_begin": SIGMA_BEGIN,
    "sigma_end": SIGMA_END,
    "L_values": L_VALUES,
    "sampler": {
        "type": "annealed_langevin_dynamics",
        "epsilon": 2e-5,
        "n_steps_each_L1": 1000,
        "n_steps_each_else": 200,
    },
    "model": {
        "type": "ScoreUNet (src/models_mnist.py)",
        "sigma_conditioning": "concat sigma as extra channel (sigma map)",
        "output_scaling": "out / sigma",
    },
    "loss": {
        "type": "DSM",
        "weighting": "sigma^2",
    },
}


def DEBUG_PRINT(message: str):
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {message}", flush=True)


def get_mnist_data(batch_size=64):
    DEBUG_PRINT("Loading MNIST...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Your repo already has data/MNIST/raw; download can be False.
    # Keep try/except to be robust on fresh machines.
    try:
        dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    except Exception:
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    DEBUG_PRINT("MNIST loaded.")
    return loader


def dsm_loss(model, x, sigma):
    """
    Denoising Score Matching loss for Gaussian perturbations.
    Target: -noise / sigma
    Weighted by sigma^2 (Song & Ermon 2019 style equalization).
    """
    noise = torch.randn_like(x)

    if sigma.dim() == 1:
        sigma_reshaped = sigma.view(-1, 1, 1, 1)
    else:
        sigma_reshaped = sigma  # scalar case

    x_tilde = x + noise * sigma_reshaped
    score_pred = model(x_tilde, sigma)
    target = -noise / sigma_reshaped

    loss = 0.5 * ((score_pred - target) ** 2).sum(dim=(1, 2, 3)) * (sigma ** 2)
    return loss.mean()


def sigma_schedule(L: int):
    if L == 1:
        return torch.tensor([SIGMA_END], device=DEVICE).float()

    return torch.tensor(
        np.exp(np.linspace(np.log(SIGMA_BEGIN), np.log(SIGMA_END), L)),
        device=DEVICE
    ).float()


def train_model_for_L(L, loader):
    print(f"\nTraining model with L={L} noise levels", flush=True)

    sigmas = sigma_schedule(L)

    model = ScoreUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()

    log = []

    for epoch in range(EPOCHS):
        total_loss = 0.0
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

        avg_loss = total_loss / len(loader)
        sec = time.time() - start_t

        print(f"  L={L} | Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {sec:.1f}s", flush=True)
        log.append({"epoch": epoch + 1, "avg_loss": float(avg_loss), "sec": float(sec)})

    return model, sigmas, log


def main():
    DEBUG_PRINT("Starting run.")

    out_dir = make_run_dir(root="results/ablation", name="mnist_ablation")
    set_seed(SEED)
    save_json(os.path.join(out_dir, "config.json"), CONFIG)
    save_json(os.path.join(out_dir, "env.json"), env_info())
    DEBUG_PRINT(f"Run directory: {out_dir}")

    loader = get_mnist_data(batch_size=BATCH_SIZE)

    results = []

    for L in L_VALUES:
        model, sigmas, train_log = train_model_for_L(L, loader)
        save_json(os.path.join(out_dir, f"train_log_L{L}.json"), train_log)

        print(f"Generating samples for L={L}...", flush=True)

        x_init = torch.randn(16, 1, 28, 28, device=DEVICE)
        n_steps_each = 1000 if L == 1 else 200

        model.eval()
        x_sample = annealed_langevin_dynamics(
            model, x_init, sigmas, n_steps_each=n_steps_each, epsilon=2e-5
        )

        torch.save(
            {"L": L, "sigmas": sigmas.detach().cpu(), "samples": x_sample.detach().cpu()},
            os.path.join(out_dir, f"samples_L{L}.pt"),
        )

        results.append((L, x_sample))

    print("Plotting ablation results...", flush=True)

    fig, axs = plt.subplots(1, len(results), figsize=(5 * len(results), 6))

    if len(results) == 1:
        axs = [axs]

    for i, (L, sample) in enumerate(results):
        sample = (sample * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(sample, nrow=4, padding=2)

        axs[i].imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        axs[i].set_title(f"Noise Levels L={L}\n({L} scales)")
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_ablation_study_improved.png"), dpi=200)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
