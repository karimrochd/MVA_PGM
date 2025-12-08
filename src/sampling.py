import torch
import numpy as np

def langevin_dynamics(score_model, x_init, sigma, n_steps=100, step_size=1e-2):
    x = x_init.clone().detach()
    
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=x.device)
    sigma = sigma.to(x.device)
    
    for _ in range(n_steps):
        z = torch.randn_like(x)
        with torch.no_grad():
            score = score_model(x, sigma)
        
        x = x + 0.5 * step_size * score + np.sqrt(step_size) * z
    return x

def annealed_langevin_dynamics(score_model, x_init, sigmas, n_steps_each=100, epsilon=2e-5):
    x = x_init.clone().detach()
    sigma_L = sigmas[-1]
    
    for sigma in sigmas:
        sigma = sigma.to(x.device)
        
        alpha = epsilon * (sigma / sigma_L) ** 2
        
        for _ in range(n_steps_each):
            z = torch.randn_like(x)
            with torch.no_grad():
                score = score_model(x, sigma)
            
            x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z
            
    return x