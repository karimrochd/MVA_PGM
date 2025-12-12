

# src/run_utils.py
import os
import json
import time
import random
import platform
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_run_dir(root: str, name: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(root, f"{ts}_{name}")
    os.makedirs(out, exist_ok=True)
    return out


def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def env_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
