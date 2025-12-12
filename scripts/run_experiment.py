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

from src.models_mnist import ScoreUNet
from src.sampling import langevin_dynamics, annealed_langevin_dynamics
from src.run_utils import set_seed, make_run_dir, save_json, env_info
from src.dae_wrappers import DAEWrapper, dae_reconstruction_loss


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Quick run settings
LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 64

SIGMA_BEGIN = 1.0
SIGMA_END = 0.01
NUM_SIGMAS = 10

# Vincent baseline sigma (fixed)
BASELINE_SIGMA = 0.1

# Sampling
LD_N_STEPS = 1000
LD_STEP_SIZE = 1e-4
ALD_N_STEPS_EACH = 100
ALD_EPSILON = 2e-5

SEED = 0

CONFIG = {
    "script": "scripts/run_experiment.py",
    "seed": SEED,
    "device": str(DEVICE),
    "baseline": {
        "type": "Vincent DAE (reconstruction → score)",
        "sigma": BASELINE_SIGMA,
        "training": "reconstruction MSE",
        "sampling": {
            "type": "Langevin",
            "n_steps": LD_N_STEPS,
            "step_size": LD_STEP_SIZE,
        },
    },
    "ncsn": {
        "type": "Song & Ermon (2019) quick",
        "sigmas": {
            "begin": SIGMA_BEGIN,
            "end": SIGMA_END,
            "num": NUM_SIGMAS,
            "schedule": "geometric",
        },
        "training": "DSM (sigma^2 weighted)",
        "sampling": {
            "type": "Annealed Langevin",
            "n_steps_each": ALD_N_STEPS_EACH,
            "epsilon": ALD_EPSILON,
        },
    },
    "model": {
        "type": "ScoreUNet (src/models_mnist.py)",
        "conditioning": "concat sigma map channel",
        "output_scaling": "out / sigma",
    },
    "train": {
        "lr": LR,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
    },
}


def debug(msg: str):
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_mnist_loader(batch_size: int):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    try:
        ds = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
    except Exception:
        ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def geometric_sigmas(sigma_begin: float, sigma_end: float, L: int, device):
    sigmas = np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), L))
    return torch.tensor(sigmas, dtype=torch.float32, device=device)


def dsm_loss(model, x, sigma):
    noise = torch.randn_like(x)

    if sigma.dim() == 1:
        sigma_r = sigma.view(-1, 1, 1, 1)
    else:
        sigma_r = sigma

    x_tilde = x + noise * sigma_r
    score_pred = model(x_tilde, sigma)
    target = -noise / sigma_r

    loss = 0.5 * ((score_pred - target) ** 2).sum(dim=(1, 2, 3)) * (sigma ** 2)
    return loss.mean()


def train_ncsn(model, loader, sigmas):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.train()
    log = []

    for epoch in range(EPOCHS):
        total = 0.0
        start = time.time()

        for x, _ in loader:
            x = x.to(DEVICE)
            idx = torch.randint(0, len(sigmas), (x.size(0),), device=DEVICE)
            sigma = sigmas[idx]

            optimizer.zero_grad()
            loss = dsm_loss(model, x, sigma)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        sec = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS} | NCSN | loss={avg:.4f} | {sec:.1f}s", flush=True)
        log.append({"epoch": epoch + 1, "avg_loss": float(avg), "sec": float(sec)})

    return model, log


def train_dae(denoiser, loader, sigma_scalar: float):
    optimizer = optim.Adam(denoiser.parameters(), lr=LR)
    denoiser.train()
    log = []

    for epoch in range(EPOCHS):
        total = 0.0
        start = time.time()

        for x, _ in loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = dae_reconstruction_loss(denoiser, x, sigma_scalar=sigma_scalar)
            loss.backward()
            optimizer.step()
            total += loss.item()

        avg = total / len(loader)
        sec = time.time() - start
        print(f"Epoch {epoch+1}/{EPOCHS} | DAE | recon={avg:.4f} | {sec:.1f}s", flush=True)
        log.append({"epoch": epoch + 1, "avg_loss": float(avg), "sec": float(sec)})

    return denoiser, log


def main():
    out_dir = make_run_dir(root="results/bridge", name="mnist_bridge_quick_vincent")
    set_seed(SEED)
    save_json(os.path.join(out_dir, "config.json"), CONFIG)
    save_json(os.path.join(out_dir, "env.json"), env_info())
    debug(f"Run directory: {out_dir}")

    loader = get_mnist_loader(BATCH_SIZE)
    sigmas = geometric_sigmas(SIGMA_BEGIN, SIGMA_END, NUM_SIGMAS, DEVICE)

    # -------------------------------------------------------------------------
    # 1) Vincent DAE baseline
    # -------------------------------------------------------------------------
    print("\n1) Training Vincent DAE baseline (reconstruction)", flush=True)
    denoiser = ScoreUNet().to(DEVICE)
    denoiser, dae_log = train_dae(denoiser, loader, sigma_scalar=BASELINE_SIGMA)
    save_json(os.path.join(out_dir, "train_log_dae.json"), dae_log)

    print("Sampling DAE baseline (score-from-denoiser + Langevin)", flush=True)
    score_dae = DAEWrapper(denoiser, sigma_scalar=BASELINE_SIGMA).to(DEVICE).eval()

    x_init = torch.randn(16, 1, 28, 28, device=DEVICE)
    x_dae = langevin_dynamics(
        score_dae,
        x_init,
        sigma=BASELINE_SIGMA,
        n_steps=LD_N_STEPS,
        step_size=LD_STEP_SIZE,
    )
    torch.save({"samples": x_dae.detach().cpu()}, os.path.join(out_dir, "samples_dae.pt"))

    # -------------------------------------------------------------------------
    # 2) NCSN multi-scale
    # -------------------------------------------------------------------------
    print("\n2) Training NCSN (multi-scale DSM)", flush=True)
    ncsn = ScoreUNet().to(DEVICE)
    ncsn, ncsn_log = train_ncsn(ncsn, loader, sigmas)
    save_json(os.path.join(out_dir, "train_log_ncsn.json"), ncsn_log)

    print("Sampling NCSN (Annealed Langevin)", flush=True)
    x_init = torch.randn(16, 1, 28, 28, device=DEVICE)
    x_ncsn = annealed_langevin_dynamics(
        ncsn,
        x_init,
        sigmas,
        n_steps_each=ALD_N_STEPS_EACH,
        epsilon=ALD_EPSILON,
    )
    torch.save(
        {"sigmas": sigmas.detach().cpu(), "samples": x_ncsn.detach().cpu()},
        os.path.join(out_dir, "samples_ncsn.pt"),
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    print("Plotting", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    def show(ax, t, title):
        t = (t * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(t, nrow=4, padding=2)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    show(axs[0], x_dae, f"Vincent DAE\n(score-from-denoiser, σ={BASELINE_SIGMA})")
    show(axs[1], x_ncsn, "Song & Ermon NCSN\n(annealed)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bridge_experiment_mnist.png"), dpi=200)
    plt.close()

    print(f"Done. Saved outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
