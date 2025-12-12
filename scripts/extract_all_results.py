
import json
import torch
from pathlib import Path

RESULTS_ROOT = Path("results")


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def summarize_train_log(log):
    losses = [e["avg_loss"] for e in log]
    times = [e.get("sec", 0.0) for e in log]

    return {
        "epochs": len(losses),
        "loss_start": losses[0],
        "loss_end": losses[-1],
        "loss_min": min(losses),
        "loss_max": max(losses),
        "mean_epoch_time_sec": sum(times) / max(len(times), 1),
    }


def summarize_samples(pt_path):
    data = torch.load(pt_path, map_location="cpu")

    samples = data.get("samples", data)
    samples = samples.float()

    return {
        "num_samples": samples.shape[0],
        "mean_pixel": samples.mean().item(),
        "std_pixel": samples.std().item(),
        "min_pixel": samples.min().item(),
        "max_pixel": samples.max().item(),
        "mean_l2_norm": samples.view(samples.shape[0], -1).norm(dim=1).mean().item(),
        "has_nan": torch.isnan(samples).any().item(),
    }


def extract():
    summary = {}

    for exp_type in RESULTS_ROOT.iterdir():
        if not exp_type.is_dir():
            continue

        summary[exp_type.name] = {}

        for run in exp_type.iterdir():
            if not run.is_dir():
                continue

            run_data = {}
            run_data["config"] = load_json(run / "config.json")
            run_data["env"] = load_json(run / "env.json")

            # training logs
            train_logs = {}
            for p in run.glob("train_log_*.json"):
                log = load_json(p)
                train_logs[p.stem] = summarize_train_log(log)

            run_data["training"] = train_logs

            # samples
            samples = {}
            for p in run.glob("samples_*.pt"):
                samples[p.stem] = summarize_samples(p)

            run_data["samples"] = samples

            summary[exp_type.name][run.name] = run_data

    return summary


if __name__ == "__main__":
    out = extract()
    with open("results/summary_all_experiments.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved results/summary_all_experiments.json")
