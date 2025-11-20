"""
Export utilities for 3D objects
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Union, Optional
import torch


def point_cloud_to_mesh(
    points: np.ndarray,
    method: str = "ball_pivoting",
    radius: float = 0.05
) -> trimesh.Trimesh:
    """
    Convert point cloud to mesh
    
    Args:
        points: Point cloud [N, 3]
        method: Reconstruction method ('ball_pivoting' or 'alpha_shape')
        radius: Radius for reconstruction
    
    Returns:
        Reconstructed mesh
    """
    # Use trimesh's built-in convex hull for simple mesh reconstruction
    # This works without Open3D
    try:
        # Try convex hull first (simple and fast)
        cloud = trimesh.PointCloud(points)
        mesh = cloud.convex_hull
        return mesh
    except Exception as e:
        print(f"Warning: Mesh conversion failed ({e}). Returning point cloud as-is.")
        # Return a simple point cloud mesh if conversion fails
        return trimesh.PointCloud(points)


def export_to_obj(
    points: Union[np.ndarray, torch.Tensor],
    filename: Union[str, Path],
    convert_to_mesh: bool = True
):
    """
    Export point cloud to OBJ file
    
    Args:
        points: Point cloud [N, 3]
        filename: Output filename
        convert_to_mesh: Whether to convert to mesh first
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    if convert_to_mesh:
        try:
            mesh = point_cloud_to_mesh(points)
            mesh.export(str(filename))
        except Exception as e:
            print(f"Mesh conversion failed: {e}. Saving as point cloud.")
            _save_point_cloud_obj(points, filename)
    else:
        _save_point_cloud_obj(points, filename)


def _save_point_cloud_obj(points: np.ndarray, filename: Path):
    """Save point cloud as OBJ vertices only"""
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")


def export_to_ply(
    points: Union[np.ndarray, torch.Tensor],
    filename: Union[str, Path],
    normals: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None
):
    """
    Export point cloud to PLY file
    
    Args:
        points: Point cloud [N, 3]
        filename: Output filename
        normals: Optional normals [N, 3]
        colors: Optional colors [N, 3] or [N, 4]
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Create point cloud
    cloud = trimesh.PointCloud(vertices=points)
    
    # Add normals if provided
    if normals is not None:
        cloud.vertices_normal = normals
    
    # Add colors if provided
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        cloud.colors = colors
    
    # Export
    cloud.export(str(filename))


def export_to_stl(
    points: Union[np.ndarray, torch.Tensor],
    filename: Union[str, Path],
    method: str = "ball_pivoting"
):
    """
    Export point cloud to STL file (requires mesh conversion)
    
    Args:
        points: Point cloud [N, 3]
        filename: Output filename
        method: Mesh reconstruction method
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to mesh
    mesh = point_cloud_to_mesh(points, method=method)
    
    # Export
    mesh.export(str(filename))


def export_to_glb(
    points: Union[np.ndarray, torch.Tensor],
    filename: Union[str, Path],
    method: str = "ball_pivoting"
):
    """
    Export point cloud to GLB file (requires mesh conversion)
    
    Args:
        points: Point cloud [N, 3]
        filename: Output filename
        method: Mesh reconstruction method
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to mesh
    mesh = point_cloud_to_mesh(points, method=method)
    
    # Export as GLB
    mesh.export(str(filename))


def export_batch(
    points: Union[np.ndarray, torch.Tensor],
    output_dir: Union[str, Path],
    format: str = "obj",
    prefix: str = "generated"
):
    """
    Export a batch of point clouds
    
    Args:
        points: Batch of point clouds [B, N, 3]
        output_dir: Output directory
        format: Export format
        prefix: Filename prefix
    """
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    export_functions = {
        'obj': export_to_obj,
        'ply': export_to_ply,
        'stl': export_to_stl,
        'glb': export_to_glb
    }
    
    if format not in export_functions:
        raise ValueError(f"Unsupported format: {format}")
    
    export_func = export_functions[format]
    
    for i, point_cloud in enumerate(points):
        filename = output_dir / f"{prefix}_{i:04d}.{format}"
        export_func(point_cloud, filename)
        print(f"Exported {filename}")
