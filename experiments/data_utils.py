import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



def get_toy_data(n_samples=10000, seed=None, return_centers=False):
    modes = 8
    radius = 5.0
    std = 0.5

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    thetas = np.linspace(0, 2 * np.pi, modes, endpoint=False)
    centers = []
    for theta in thetas:
        centers.append([radius * np.cos(theta), radius * np.sin(theta)])
    centers = torch.tensor(centers, dtype=torch.float32)

    mode_indices = torch.randint(0, modes, (n_samples,))
    sample_centers = centers[mode_indices]

    noise = torch.randn(n_samples, 2) * std
    dataset = sample_centers + noise

    if return_centers:
        return dataset, centers
    return dataset


def plot_scatter(data, title="Dataset"):
    data = data.detach().cpu()
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], s=2, alpha=0.6)
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def get_mnist_data(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader