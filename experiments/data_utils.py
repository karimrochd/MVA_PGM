# ==============================================================================
# DATA UTILS
# This file handles the generation of the 2D toy dataset (Mixture of Gaussians).
# ==============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



# ------------------------------------------------------------------------------
# Function: get_toy_data
# Generates a batch of data points from a Mixture of Gaussians on a circle.
#
# Args:
#   n_samples: Number of samples to generate.
#   seed: Optional int, for reproducibility.
#   return_centers: If True, also returns the Gaussian centers.
#
# Returns:
#   dataset: A torch tensor of shape (n_samples, 2).
#   centers (optional): Tensor of Gaussian centers of shape (modes, 2).
# ------------------------------------------------------------------------------
def get_toy_data(n_samples=10000, seed=None, return_centers=False):
    modes = 8
    radius = 5.0
    std = 0.5

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Define centers of the Gaussians arranged in a circle
    thetas = np.linspace(0, 2 * np.pi, modes, endpoint=False)
    centers = []
    for theta in thetas:
        centers.append([radius * np.cos(theta), radius * np.sin(theta)])
    centers = torch.tensor(centers, dtype=torch.float32)

    # Randomly select a mode for each sample
    mode_indices = torch.randint(0, modes, (n_samples,))
    sample_centers = centers[mode_indices]

    # Add noise to create the blobs
    noise = torch.randn(n_samples, 2) * std
    dataset = sample_centers + noise

    if return_centers:
        return dataset, centers
    return dataset


# ------------------------------------------------------------------------------
# Function: plot_scatter
# Helper to visualize the dataset.
# ------------------------------------------------------------------------------
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
        transforms.ToTensor(), # Scales to [0, 1]
    ])
    
    # Download to a local folder './data'
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Create loader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader