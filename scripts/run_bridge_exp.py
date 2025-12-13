import os
import sys
import time
import argparse

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

# Training
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 64

# NCSN sigma ladder
SIGMA_BEGIN = 5.0
SIGMA_END = 0.01
NUM_SIGMAS = 10

# Vincent DAE baseline
BASELINE_SIGMA = 0.1

# Sampling
LD_N_STEPS = 1000
LD_STEP_SIZE = 1e-4
ALD_N_STEPS_EACH = 200
ALD_EPSILON = 2e-5

SEED = 0


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


def geometric_sigmas(sigma_begin, sigma_end, L, device):
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


def save_ckpt(path: str, model: torch.nn.Module):
    torch.save({"state_dict": model.state_dict()}, path)


def load_ckpt(path: str, model: torch.nn.Module):
    obj = torch.load(path, map_location="cpu")
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(state)
    return model


def auto_nrow(n: int) -> int:
    if n <= 16:
        return 4
    if n <= 64:
        return 8
    if n <= 256:
        return 16
    return 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=256)
    ap.add_argument("--only_sample", action="store_true", help="Skip training, load checkpoints from out_dir")
    ap.add_argument("--out_dir", type=str, default=None, help="Use an existing run directory (for --only_sample)")
    args = ap.parse_args()

    set_seed(SEED)

    if args.out_dir is None:
        out_dir = make_run_dir(root="results/bridge", name="mnist_bridge")
        CONFIG = {
            "script": "scripts/run_bridge_exp.py",
            "seed": SEED,
            "device": str(DEVICE),
            "num_samples_default": args.num_samples,
            "baseline": {
                "type": "Vincent DAE (reconstruction → score)",
                "sigma": BASELINE_SIGMA,
                "training": "reconstruction MSE",
                "sampling": {"type": "Langevin", "n_steps": LD_N_STEPS, "step_size": LD_STEP_SIZE},
            },
            "ncsn": {
                "type": "Song & Ermon (2019)",
                "sigmas": {"begin": SIGMA_BEGIN, "end": SIGMA_END, "num": NUM_SIGMAS, "schedule": "geometric"},
                "training": "DSM (sigma^2 weighted)",
                "sampling": {"type": "Annealed Langevin", "n_steps_each": ALD_N_STEPS_EACH, "epsilon": ALD_EPSILON},
            },
            "model": {"arch": "ScoreUNet", "conditioning": "concat sigma map", "output_scaling": "out / sigma"},
        }
        save_json(os.path.join(out_dir, "config.json"), CONFIG)
        save_json(os.path.join(out_dir, "env.json"), env_info())
    else:
        out_dir = args.out_dir

    debug(f"Run directory: {out_dir}")

    loader = None if args.only_sample else get_mnist_loader(BATCH_SIZE)
    sigmas = geometric_sigmas(SIGMA_BEGIN, SIGMA_END, NUM_SIGMAS, DEVICE)

    dae_ckpt = os.path.join(out_dir, "ckpt_dae_denoiser.pt")
    ncsn_ckpt = os.path.join(out_dir, "ckpt_ncsn.pt")

    # -------------------------------------------------------------------------
    # 1) DAE BASELINE
    # -------------------------------------------------------------------------
    if args.only_sample:
        if not os.path.exists(dae_ckpt):
            raise FileNotFoundError(f"Missing DAE checkpoint: {dae_ckpt}")
        denoiser = ScoreUNet().to(DEVICE)
        load_ckpt(dae_ckpt, denoiser)
    else:
        print("\n1) Training Vincent DAE baseline (reconstruction MSE)", flush=True)
        denoiser = ScoreUNet().to(DEVICE)
        opt = optim.Adam(denoiser.parameters(), lr=LR)
        denoiser.train()
        dae_log = []
        for epoch in range(EPOCHS):
            total = 0.0
            start = time.time()
            for x, _ in loader:
                x = x.to(DEVICE)
                opt.zero_grad()
                loss = dae_reconstruction_loss(denoiser, x, BASELINE_SIGMA)
                loss.backward()
                opt.step()
                total += loss.item()
            avg = total / len(loader)
            sec = time.time() - start
            print(f"Epoch {epoch+1}/{EPOCHS} | DAE | recon={avg:.4f} | {sec:.1f}s", flush=True)
            dae_log.append({"epoch": epoch + 1, "avg_loss": float(avg), "sec": float(sec)})

        save_json(os.path.join(out_dir, "train_log_dae.json"), dae_log)
        save_ckpt(dae_ckpt, denoiser)

    print(f"Sampling Vincent DAE (N={args.num_samples})", flush=True)
    score_dae = DAEWrapper(denoiser, BASELINE_SIGMA).to(DEVICE).eval()
    x_init = torch.randn(args.num_samples, 1, 28, 28, device=DEVICE)
    x_dae = langevin_dynamics(
        score_dae, x_init, sigma=BASELINE_SIGMA, n_steps=LD_N_STEPS, step_size=LD_STEP_SIZE
    )
    torch.save({"samples": x_dae.cpu()}, os.path.join(out_dir, f"samples_dae_N{args.num_samples}.pt"))

    # -------------------------------------------------------------------------
    # 2) NCSN
    # -------------------------------------------------------------------------
    if args.only_sample:
        if not os.path.exists(ncsn_ckpt):
            raise FileNotFoundError(f"Missing NCSN checkpoint: {ncsn_ckpt}")
        ncsn = ScoreUNet().to(DEVICE)
        load_ckpt(ncsn_ckpt, ncsn)
    else:
        print("\n2) Training NCSN (multi-scale DSM)", flush=True)
        ncsn = ScoreUNet().to(DEVICE)
        ncsn, ncsn_log = train_ncsn(ncsn, loader, sigmas)
        save_json(os.path.join(out_dir, "train_log_ncsn.json"), ncsn_log)
        save_ckpt(ncsn_ckpt, ncsn)

    print(f"Sampling NCSN (N={args.num_samples})", flush=True)
    x_init = torch.randn(args.num_samples, 1, 28, 28, device=DEVICE)
    x_ncsn = annealed_langevin_dynamics(
        ncsn, x_init, sigmas, n_steps_each=ALD_N_STEPS_EACH, epsilon=ALD_EPSILON
    )
    torch.save(
        {"sigmas": sigmas.cpu(), "samples": x_ncsn.cpu()},
        os.path.join(out_dir, f"samples_ncsn_N{args.num_samples}.pt"),
    )

    # -------------------------------------------------------------------------
    # PLOT
    # -------------------------------------------------------------------------
    print("Plotting comparison", flush=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    nrow = auto_nrow(args.num_samples)

    def show(ax, t, title):
        t = (t * 0.5 + 0.5).clamp(0, 1)
        grid = make_grid(t, nrow=nrow, padding=2)
        ax.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    show(axs[0], x_dae, f"Vincent DAE | N={args.num_samples}")
    show(axs[1], x_ncsn, f"NCSN | N={args.num_samples}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"bridge_experiment_mnist_N{args.num_samples}.png"), dpi=200)
    plt.close()

    print(f"Done. Results saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
