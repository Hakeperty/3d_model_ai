"""
Evaluation metrics for 3D generation
"""

import numpy as np
import torch
from typing import Union
from scipy.spatial.distance import cdist


def chamfer_distance(
    pc1: Union[np.ndarray, torch.Tensor],
    pc2: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Chamfer Distance between two point clouds
    
    Args:
        pc1: First point cloud [N, 3]
        pc2: Second point cloud [M, 3]
    
    Returns:
        Chamfer distance
    """
    if isinstance(pc1, torch.Tensor):
        pc1 = pc1.cpu().numpy()
    if isinstance(pc2, torch.Tensor):
        pc2 = pc2.cpu().numpy()
    
    # Compute pairwise distances
    dist_matrix = cdist(pc1, pc2, metric='euclidean')
    
    # Minimum distances from pc1 to pc2
    min_dist_1_to_2 = np.min(dist_matrix, axis=1)
    
    # Minimum distances from pc2 to pc1
    min_dist_2_to_1 = np.min(dist_matrix, axis=0)
    
    # Chamfer distance
    cd = np.mean(min_dist_1_to_2) + np.mean(min_dist_2_to_1)
    
    return float(cd)


def earth_movers_distance(
    pc1: Union[np.ndarray, torch.Tensor],
    pc2: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute Earth Mover's Distance (approximation)
    
    Args:
        pc1: First point cloud [N, 3]
        pc2: Second point cloud [N, 3]
    
    Returns:
        EMD (approximation using Hungarian matching)
    """
    if isinstance(pc1, torch.Tensor):
        pc1 = pc1.cpu().numpy()
    if isinstance(pc2, torch.Tensor):
        pc2 = pc2.cpu().numpy()
    
    from scipy.optimize import linear_sum_assignment
    
    # Compute cost matrix
    cost_matrix = cdist(pc1, pc2, metric='euclidean')
    
    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # EMD is the sum of matched distances
    emd = cost_matrix[row_ind, col_ind].sum() / len(pc1)
    
    return float(emd)


def coverage_score(
    generated: Union[np.ndarray, torch.Tensor],
    reference: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.01
) -> float:
    """
    Compute coverage: percentage of reference points close to generated
    
    Args:
        generated: Generated point clouds [B, N, 3]
        reference: Reference point clouds [M, N, 3]
        threshold: Distance threshold
    
    Returns:
        Coverage score (0-1)
    """
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    
    covered = 0
    total = len(reference)
    
    for ref_pc in reference:
        min_cd = float('inf')
        for gen_pc in generated:
            cd = chamfer_distance(gen_pc, ref_pc)
            min_cd = min(min_cd, cd)
        
        if min_cd < threshold:
            covered += 1
    
    return covered / total


def minimum_matching_distance(
    generated: Union[np.ndarray, torch.Tensor],
    reference: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Compute MMD: average minimum Chamfer distance
    
    Args:
        generated: Generated point clouds [B, N, 3]
        reference: Reference point clouds [M, N, 3]
    
    Returns:
        MMD score
    """
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    if isinstance(reference, torch.Tensor):
        reference = reference.cpu().numpy()
    
    total_distance = 0
    
    for gen_pc in generated:
        min_cd = float('inf')
        for ref_pc in reference:
            cd = chamfer_distance(gen_pc, ref_pc)
            min_cd = min(min_cd, cd)
        total_distance += min_cd
    
    return total_distance / len(generated)
