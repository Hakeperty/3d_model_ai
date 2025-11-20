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
    Simplified 3D U-Net for point cloud denoising in diffusion models
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
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Condition embedding (if provided)
        self.use_condition = condition_dim is not None
        if self.use_condition:
            self.condition_mlp = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.ReLU()
            )
        
        # Initial projection: input + time
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim + time_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Encoder layers
        self.encoder1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU()
        )
        
        self.encoder2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, 1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 1),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder2 = nn.Sequential(
            nn.Conv1d(hidden_dim * 4 + hidden_dim * 2, hidden_dim * 2, 1),  # 4x + skip(2x)
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU()
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv1d(hidden_dim * 2 + hidden_dim, hidden_dim, 1),  # 2x + skip(1x)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Output projection
        self.output_proj = nn.Conv1d(hidden_dim, input_dim, 1)
    
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
        
        # Transpose for conv1d: [B, N, 3] -> [B, 3, N]
        x = x.transpose(1, 2)  # [B, 3, N]
        
        # Concatenate input with time
        x = torch.cat([x, t_emb], dim=1)  # [B, 3+time_dim, N]
        
        # Initial projection
        x = self.input_proj(x)  # [B, hidden_dim, N]
        
        # Add condition if provided
        if self.use_condition and condition is not None:
            cond_emb = self.condition_mlp(condition)  # [B, hidden_dim]
            cond_emb = cond_emb[:, :, None].expand(-1, -1, num_points)  # [B, hidden_dim, N]
            x = x + cond_emb
        
        # Encoder with skip connections
        skip1 = x
        x = self.encoder1(x)  # [B, hidden_dim*2, N]
        
        skip2 = x
        x = self.encoder2(x)  # [B, hidden_dim*4, N]
        
        # Bottleneck
        x = self.bottleneck(x)  # [B, hidden_dim*4, N]
        
        # Decoder with skip connections
        x = torch.cat([x, skip2], dim=1)  # Concatenate skip connection
        x = self.decoder2(x)  # [B, hidden_dim*2, N]
        
        x = torch.cat([x, skip1], dim=1)  # Concatenate skip connection
        x = self.decoder1(x)  # [B, hidden_dim, N]
        
        # Output projection
        x = self.output_proj(x)  # [B, 3, N]
        
        # Transpose back: [B, 3, N] -> [B, N, 3]
        x = x.transpose(1, 2)
        
        return x

