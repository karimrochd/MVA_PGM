import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _load_samples(obj: Any) -> torch.Tensor:
    """
    Accepts:
      - Tensor
      - Dict with keys like 'samples', 'x', 'generated', 'data'
      - List/tuple of tensors (takes first)
    Returns float tensor of shape (N, C, H, W) or (N, D)
    """
    if torch.is_tensor(obj):
        x = obj
    elif isinstance(obj, (list, tuple)) and len(obj) > 0 and torch.is_tensor(obj[0]):
        x = obj[0]
    elif isinstance(obj, dict):
        for k in ["samples", "x", "generated", "data"]:
            if k in obj and torch.is_tensor(obj[k]):
                x = obj[k]
                break
        else:
            raise ValueError(f"Could not find tensor samples in dict keys={list(obj.keys())}")
    else:
        raise ValueError(f"Unsupported .pt content type: {type(obj)}")

    x = x.detach().cpu()
    if not torch.is_floating_point(x):
        x = x.float()

    # Normalize shapes a bit for image-like tensors
    # Common cases: (N, 1, 28, 28), (N, 28, 28), (N, 784)
    if x.dim() == 3:  # (N, H, W)
        x = x.unsqueeze(1)
    return x


def _total_variation(x: torch.Tensor) -> float:
    """
    TV for image tensors in [-1,1], returns mean TV per sample.
    If not image-shaped, returns NaN.
    """
    if x.dim() != 4:
        return float("nan")
    # (N,C,H,W)
    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    tv = dh.mean(dim=(1, 2, 3)) + dw.mean(dim=(1, 2, 3))
    return tv.mean().item()


def _pairwise_l2(x: torch.Tensor, max_n: int = 512, seed: int = 0) -> Tuple[float, float]:
    """
    Mean and std of pairwise L2 on a subset.
    Uses efficient sampling of pairs instead of full O(n^2).
    """
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    m = min(n, max_n)
    idx = rng.choice(n, size=m, replace=False)
    y = x[idx].reshape(m, -1).float()

    # sample pairs
    num_pairs = min(20000, m * (m - 1) // 2)
    a = rng.integers(0, m, size=num_pairs)
    b = rng.integers(0, m, size=num_pairs)
    mask = a != b
    a = a[mask]
    b = b[mask]
    y1 = y[a]
    y2 = y[b]
    d = torch.norm(y1 - y2, dim=1)
    return d.mean().item(), d.std(unbiased=False).item()


def _pixel_var_mean(x: torch.Tensor) -> float:
    """
    Mean variance across pixels/features.
    """
    y = x.reshape(x.shape[0], -1).float()
    v = y.var(dim=0, unbiased=False)
    return v.mean().item()


@dataclass
class Metrics:
    name: str
    n: int
    pct_finite: float
    pct_in_range: float
    pct_saturated: float
    mean_abs: float
    std: float
    tv: float
    pairwise_l2_mean: float
    pairwise_l2_std: float
    pixel_var_mean: float


def compute_metrics(path: str, in_min: float, in_max: float, sat_thr: float, max_pair_n: int) -> Metrics:
    obj = torch.load(path, map_location="cpu")
    x = _load_samples(obj)
    name = os.path.basename(path)

    finite_mask = torch.isfinite(x)
    pct_finite = finite_mask.float().mean().item() * 100.0

    # for range stats, ignore non-finite
    xf = x.clone()
    xf[~finite_mask] = 0.0

    in_range = (xf >= in_min) & (xf <= in_max)
    pct_in_range = (in_range.float().mean().item() * 100.0)

    saturated = xf.abs() >= sat_thr
    pct_saturated = saturated.float().mean().item() * 100.0

    mean_abs = xf.abs().mean().item()
    std = xf.std(unbiased=False).item()

    tv = _total_variation(xf)

    pw_mean, pw_std = _pairwise_l2(xf, max_n=max_pair_n)
    pv_mean = _pixel_var_mean(xf)

    return Metrics(
        name=name,
        n=int(x.shape[0]),
        pct_finite=pct_finite,
        pct_in_range=pct_in_range,
        pct_saturated=pct_saturated,
        mean_abs=mean_abs,
        std=std,
        tv=tv,
        pairwise_l2_mean=pw_mean,
        pairwise_l2_std=pw_std,
        pixel_var_mean=pv_mean,
    )


def to_csv(rows: List[Metrics]) -> str:
    header = [
        "name", "n",
        "pct_finite", "pct_in_range", "pct_saturated",
        "mean_abs", "std", "tv",
        "pairwise_l2_mean", "pairwise_l2_std",
        "pixel_var_mean",
    ]
    lines = [",".join(header)]
    for r in rows:
        vals = [
            r.name, str(r.n),
            f"{r.pct_finite:.3f}", f"{r.pct_in_range:.3f}", f"{r.pct_saturated:.3f}",
            f"{r.mean_abs:.6f}", f"{r.std:.6f}", f"{r.tv:.6f}",
            f"{r.pairwise_l2_mean:.6f}", f"{r.pairwise_l2_std:.6f}",
            f"{r.pixel_var_mean:.6f}",
        ]
        lines.append(",".join(vals))
    return "\n".join(lines) + "\n"


def to_latex(rows: List[Metrics], caption: str, label: str) -> str:
    def fmt(r: Metrics) -> List[str]:
        return [
            r.name.replace("_", r"\_"),
            str(r.n),
            f"{r.pct_finite:.1f}",
            f"{r.pct_in_range:.1f}",
            f"{r.pct_saturated:.1f}",
            f"{r.mean_abs:.3f}",
            f"{r.std:.3f}",
            f"{r.tv:.3f}" if math.isfinite(r.tv) else "NA",
            f"{r.pairwise_l2_mean:.2f}",
            f"{r.pixel_var_mean:.4f}",
        ]

    cols = ["Run", "N", r"\% finite", r"\% in [-1,1]", r"\% sat", r"$\mathbb{E}|x|$", r"$\mathrm{Std}(x)$", "TV", "Pair L2", "PixVar"]
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{@{}lrrrrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(" & ".join(cols) + r" \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(" & ".join(fmt(r)) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{\textbf{{{caption}}} Simple diagnostics computed from saved sample tensors. \% sat uses $|x|\ge 0.99$.}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out_csv", default="results/sample_metrics.csv")
    ap.add_argument("--out_latex", default="results/sample_metrics_table.tex")
    ap.add_argument("--in_min", type=float, default=-1.0)
    ap.add_argument("--in_max", type=float, default=1.0)
    ap.add_argument("--sat_thr", type=float, default=0.99)
    ap.add_argument("--max_pair_n", type=int, default=512)
    ap.add_argument("--caption", default="MNIST sampling diagnostics")
    ap.add_argument("--label", default="tab:mnist_sample_diagnostics")
    args = ap.parse_args()

    rows = [compute_metrics(p, args.in_min, args.in_max, args.sat_thr, args.max_pair_n) for p in args.inputs]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write(to_csv(rows))

    with open(args.out_latex, "w") as f:
        f.write(to_latex(rows, args.caption, args.label))

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote LaTeX: {args.out_latex}")


if __name__ == "__main__":
    main()
