import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)


def make_manifold(num_points: int = 400) -> torch.Tensor:
    """Parametric 1D curve that mimics the classic DAE manifold."""
    t = torch.linspace(-3.0, 3.0, num_points)
    x = t
    y = torch.sin(t) * 1.2 * torch.exp(-0.1 * t**2)
    return torch.stack([x, y], dim=1)


def score_field(grid_range=(-4.0, 4.0), steps=14):
    """Compute score vectors pointing each grid point to the nearest manifold point."""
    curve = make_manifold()
    gx = torch.linspace(grid_range[0], grid_range[1], steps)
    gy = torch.linspace(grid_range[0] * 0.6, grid_range[1] * 0.6, steps)
    grid_x, grid_y = torch.meshgrid(gx, gy, indexing="xy")
    grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    diff = grid_points[:, None, :] - curve[None, :, :]
    dists = torch.sum(diff**2, dim=2)
    nearest_idx = torch.argmin(dists, dim=1)
    nearest_points = curve[nearest_idx]
    scores = nearest_points - grid_points

    scores = scores * 0.2
    return grid_points, scores, curve


def sample_noisy_data(num_samples=350, noise_std=0.25):
    """Sample points on the manifold plus Gaussian noise to show denoising."""
    curve = make_manifold(num_samples)
    noise = torch.randn_like(curve) * noise_std
    return curve + noise


def add_gaussian_circles(ax, curve, radius=0.25, count=12):
    """Overlay circles illustrating the isotropic noise around manifold points."""
    idx = torch.linspace(0, curve.shape[0] - 1, count).long()
    circle_handles = []
    for center in curve[idx]:
        circle = Circle(
            (center[0].item(), center[1].item()),
            radius=radius,
            facecolor="none",
            edgecolor="#2e8b57",
            linewidth=2.3,
            linestyle="--",
            alpha=0.9,
        )
        ax.add_patch(circle)
        circle_handles.append(circle)
    return circle_handles[0] if circle_handles else None


def main():
    grid_pts, scores, curve = score_field()
    noisy = sample_noisy_data()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.quiver(
        grid_pts[:, 0],
        grid_pts[:, 1],
        scores[:, 0],
        scores[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.8,
        color="#1f77b4",
        width=0.0028,
        alpha=0.8,
    )

    ax.plot(curve[:, 0], curve[:, 1], color="#b22222", linewidth=3, label="1D Manifold")
    scatter = ax.scatter(
        noisy[:, 0],
        noisy[:, 1],
        s=8,
        color="#7f7f7f",
        alpha=0.6,
        label="Noisy samples",
    )
    circle_handle = add_gaussian_circles(ax, curve)

    ax.set_title("Score Field of a DAE Projecting onto a 1D Manifold")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    handles = [ax.lines[0], scatter]
    labels = ["1D Manifold", "Noisy samples"]
    if circle_handle is not None:
        handles.append(circle_handle)
        labels.append("Gaussian noise")
    ax.legend(handles, labels, loc="upper right", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    plt.savefig("dae_score_field.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
