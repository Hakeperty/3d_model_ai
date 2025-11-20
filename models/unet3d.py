"""
3D U-Net architecture for point cloud processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps [B]
        Returns:
            Time embeddings [B, dim]
        """
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PointNetSetAbstraction(nn.Module):
    """PointNet++ set abstraction layer"""
    
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel // 2, 1)
        self.conv2 = nn.Conv1d(out_channel // 2, out_channel, 1)
        self.bn1 = nn.BatchNorm1d(out_channel // 2)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, N]
        Returns:
            Output features [B, C', N]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class PointNetFeaturePropagation(nn.Module):
    """PointNet++ feature propagation layer"""
    
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, N]
        Returns:
            Output features [B, C', N]
        """
        x = self.relu(self.bn1(self.conv1(x)))
        return x


class UNet3D(nn.Module):
    """
    3D U-Net for point cloud denoising in diffusion models
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        time_dim: int = 256,
        num_layers: int = 4,
        condition_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.num_layers = num_layers
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.ReLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Condition embedding (if provided)
        self.use_condition = condition_dim is not None
        if self.use_condition:
            self.condition_mlp = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # Initial projection
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        
        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        in_ch = hidden_dim
        for i in range(num_layers):
            out_ch = hidden_dim * (2 ** i)
            self.encoders.append(PointNetSetAbstraction(in_ch + time_dim, out_ch))
            in_ch = out_ch
        
        # Bottleneck
        self.bottleneck = PointNetSetAbstraction(in_ch + time_dim, in_ch)
        
        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        for i in range(num_layers - 1, -1, -1):
            out_ch = hidden_dim * (2 ** i) if i > 0 else hidden_dim
            # Skip connection doubles the input channels
            self.decoders.append(PointNetFeaturePropagation(in_ch * 2 + time_dim, out_ch))
            in_ch = out_ch
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, input_dim, 1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Point cloud [B, N, 3]
            t: Timestep [B]
            condition: Optional condition [B, condition_dim]
        
        Returns:
            Predicted noise [B, N, 3]
        """
        batch_size, num_points, _ = x.shape
        
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_dim]
        t_emb = t_emb[:, :, None].expand(-1, -1, num_points)  # [B, time_dim, N]
        
        # Condition embedding (if provided)
        if self.use_condition and condition is not None:
            cond_emb = self.condition_mlp(condition)  # [B, hidden_dim]
            cond_emb = cond_emb[:, :, None].expand(-1, -1, num_points)  # [B, hidden_dim, N]
        
        # Initial projection [B, N, 3] -> [B, 3, N] -> [B, hidden_dim, N]
        x = x.transpose(1, 2)  # [B, 3, N]
        x = self.input_proj(x)  # [B, hidden_dim, N]
        
        # Add condition to initial features
        if self.use_condition and condition is not None:
            x = x + cond_emb
        
        # Encoder with skip connections
        skip_connections = []
        for encoder in self.encoders:
            skip_connections.append(x)
            x = torch.cat([x, t_emb], dim=1)  # Concatenate time embedding
            x = encoder(x)
        
        # Bottleneck
        x = torch.cat([x, t_emb], dim=1)
        x = self.bottleneck(x)
        
        # Decoder with skip connections
        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = torch.cat([x, skip, t_emb], dim=1)  # Skip connection + time
            x = decoder(x)
        
        # Output projection
        x = self.output_proj(x)  # [B, 3, N]
        x = x.transpose(1, 2)  # [B, N, 3]
        
        return x
