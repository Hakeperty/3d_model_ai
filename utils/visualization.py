"""
Visualization utilities for 3D point clouds
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Union, Optional
import torch


def visualize_point_cloud(
    points: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Point Cloud",
    color: str = 'blue',
    show: bool = False,
    elev: float = 30,
    azim: float = 45
):
    """
    Visualize a single point cloud
    
    Args:
        points: Point cloud [N, 3]
        save_path: Path to save figure
        title: Plot title
        color: Point color
        show: Whether to show plot
        elev: Elevation angle
        azim: Azimuth angle
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=color,
        marker='.',
        s=1,
        alpha=0.6
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.view_init(elev=elev, azim=azim)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_batch(
    points: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Generated Point Clouds",
    max_samples: int = 16
):
    """
    Visualize a batch of point clouds in a grid
    
    Args:
        points: Batch of point clouds [B, N, 3]
        save_path: Path to save figure
        title: Figure title
        max_samples: Maximum number of samples to show
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    batch_size = min(len(points), max_samples)
    n_cols = min(4, batch_size)
    n_rows = (batch_size + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))
    fig.suptitle(title, fontsize=16)
    
    for i in range(batch_size):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        
        pc = points[i]
        ax.scatter(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            c='blue',
            marker='.',
            s=1,
            alpha=0.6
        )
        
        ax.set_title(f'Sample {i+1}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set equal aspect ratio
        max_range = np.array([
            pc[:, 0].max() - pc[:, 0].min(),
            pc[:, 1].max() - pc[:, 1].min(),
            pc[:, 2].max() - pc[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (pc[:, 0].max() + pc[:, 0].min()) * 0.5
        mid_y = (pc[:, 1].max() + pc[:, 1].min()) * 0.5
        mid_z = (pc[:, 2].max() + pc[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def visualize_comparison(
    original: Union[np.ndarray, torch.Tensor],
    generated: Union[np.ndarray, torch.Tensor],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Original vs Generated"
):
    """
    Visualize original and generated point clouds side by side
    
    Args:
        original: Original point cloud [N, 3]
        generated: Generated point cloud [N, 3]
        save_path: Path to save figure
        title: Figure title
    """
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(title, fontsize=16)
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(
        original[:, 0],
        original[:, 1],
        original[:, 2],
        c='green',
        marker='.',
        s=1,
        alpha=0.6
    )
    ax1.set_title('Original')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Generated
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        generated[:, 0],
        generated[:, 1],
        generated[:, 2],
        c='blue',
        marker='.',
        s=1,
        alpha=0.6
    )
    ax2.set_title('Generated')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Set same scale for both
    all_points = np.vstack([original, generated])
    max_range = np.array([
        all_points[:, 0].max() - all_points[:, 0].min(),
        all_points[:, 1].max() - all_points[:, 1].min(),
        all_points[:, 2].max() - all_points[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()
