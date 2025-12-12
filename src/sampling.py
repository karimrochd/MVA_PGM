import torch


def langevin_dynamics(score_model, x_init, sigma, n_steps=100, step_size=1e-2):
    x = x_init.clone().detach()

    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
    else:
        sigma = sigma.to(device=x.device, dtype=x.dtype)

    for _ in range(n_steps):
        z = torch.randn_like(x)
        with torch.no_grad():
            score = score_model(x, sigma)
        x = x + 0.5 * step_size * score + torch.sqrt(torch.tensor(step_size, device=x.device, dtype=x.dtype)) * z

    return x


def annealed_langevin_dynamics(score_model, x_init, sigmas, n_steps_each=100, epsilon=2e-5, return_stats=False):
    x = x_init.clone().detach()
    sigma_L = sigmas[-1].to(x.device)

    stats = []

    for i, sigma in enumerate(sigmas):
        sigma = sigma.to(x.device)
        alpha = epsilon * (sigma / sigma_L) ** 2

        for t in range(n_steps_each):
            z = torch.randn_like(x)
            with torch.no_grad():
                score = score_model(x, sigma)

            x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z

            if return_stats and (t % 10 == 0):
                xf = x.view(x.size(0), -1)
                sf = score.view(score.size(0), -1)
                stats.append({
                    "i": int(i),
                    "t": int(t),
                    "sigma": float(sigma.item()) if sigma.numel() == 1 else float(sigma.mean().item()),
                    "alpha": float(alpha.item()) if alpha.numel() == 1 else float(alpha.mean().item()),
                    "x_norm_mean": float(xf.norm(dim=1).mean().item()),
                    "score_norm_mean": float(sf.norm(dim=1).mean().item()),
                    "nan_frac": float(torch.isnan(x).float().mean().item()),
                })

    return (x, stats) if return_stats else x
