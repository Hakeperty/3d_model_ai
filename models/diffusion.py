"""
Diffusion model components for 3D generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class DiffusionSchedule:
    """Noise schedule for diffusion process"""
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_steps = num_steps
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_steps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_beta_schedule(num_steps)
        elif schedule_type == "quadratic":
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def to(self, device):
        """Move tensors to device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self


class DiffusionModel(nn.Module):
    """Diffusion model wrapper for 3D generation"""
    
    def __init__(
        self,
        model: nn.Module,
        schedule: DiffusionSchedule
    ):
        super().__init__()
        self.model = model
        self.schedule = schedule
        self.num_steps = schedule.num_steps
    
    def forward_diffusion(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add noise to clean data x0 at timestep t
        
        Args:
            x0: Clean data [B, N, 3]
            t: Timestep [B]
            noise: Optional pre-generated noise
        
        Returns:
            Noisy data xt
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Ensure tensors are on the same device as x0
        device = x0.device
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod.to(device)[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod.to(device)[t]
        
        # Reshape for broadcasting [B, 1, 1]
        sqrt_alpha = sqrt_alpha.view(-1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1, 1)
        
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
    
    def predict_noise(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise at timestep t
        
        Args:
            xt: Noisy data [B, N, 3]
            t: Timestep [B]
            condition: Optional conditioning (e.g., text embeddings)
        
        Returns:
            Predicted noise
        """
        return self.model(xt, t, condition)
    
    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: torch.device,
        condition: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate samples using DDPM sampling
        
        Args:
            shape: Shape of output [B, N, 3]
            device: Device to generate on
            condition: Optional conditioning
            num_steps: Number of denoising steps (uses fewer for faster sampling)
        
        Returns:
            Generated point clouds
        """
        if num_steps is None:
            num_steps = self.num_steps
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Determine which timesteps to use
        if num_steps < self.num_steps:
            timesteps = torch.linspace(self.num_steps - 1, 0, num_steps).long()
        else:
            timesteps = torch.arange(self.num_steps - 1, -1, -1)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.predict_noise(x, t_batch, condition)
            
            # Get schedule parameters (ensure on correct device)
            alpha = self.schedule.alphas.to(device)[t]
            alpha_cumprod = self.schedule.alphas_cumprod.to(device)[t]
            beta = self.schedule.betas.to(device)[t]
            
            # Compute previous sample mean
            alpha_cumprod_prev = self.schedule.alphas_cumprod_prev.to(device)[t]
            
            # Predict x0
            pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha_cumprod)
            
            # Compute direction pointing to xt
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
            
            # Compute x_{t-1}
            x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt
            
            # Add noise (except for final step)
            if t > 0:
                noise = torch.randn_like(x)
                variance = self.schedule.posterior_variance.to(device)[t]
                x = x + torch.sqrt(variance) * noise
        
        return x
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute diffusion training loss
        
        Args:
            x0: Clean data [B, N, 3]
            condition: Optional conditioning
        
        Returns:
            MSE loss between predicted and actual noise
        """
        batch_size = x0.shape[0]
        device = x0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x0)
        
        # Forward diffusion
        xt = self.forward_diffusion(x0, t, noise)
        
        # Predict noise
        predicted_noise = self.predict_noise(xt, t, condition)
        
        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)
        
        return loss
