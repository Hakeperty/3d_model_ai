"""
Evaluation script
"""

import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from config import Config
from models import Generator3D
from data import PointCloudDataset, SyntheticDataset, create_dataloader
from utils.metrics import (
    chamfer_distance,
    minimum_matching_distance,
    coverage_score
)


def evaluate_model(
    model: Generator3D,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 100
):
    """Evaluate model on dataset"""
    
    model.eval()
    
    print("\nGenerating samples for evaluation...")
    
    # Generate samples
    generated_samples = []
    with torch.no_grad():
        batch_size = 16
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches)):
            batch_size_current = min(batch_size, num_samples - i * batch_size)
            samples = model.generate(
                batch_size=batch_size_current,
                device=device,
                num_steps=50
            )
            generated_samples.append(samples)
    
    generated_samples = torch.cat(generated_samples, dim=0).cpu().numpy()
    
    print(f"Generated {len(generated_samples)} samples")
    
    # Get reference samples from dataset
    print("\nLoading reference samples...")
    reference_samples = []
    for i, (point_clouds, _) in enumerate(dataloader):
        reference_samples.append(point_clouds)
        if len(reference_samples) * point_clouds.shape[0] >= num_samples:
            break
    
    reference_samples = torch.cat(reference_samples, dim=0)[:num_samples].numpy()
    
    print(f"Loaded {len(reference_samples)} reference samples")
    
    # Compute metrics
    print("\nComputing metrics...")
    
    # Average Chamfer Distance
    print("Computing Chamfer Distance...")
    cd_scores = []
    for gen_pc, ref_pc in zip(generated_samples, reference_samples):
        cd = chamfer_distance(gen_pc, ref_pc)
        cd_scores.append(cd)
    avg_cd = np.mean(cd_scores)
    
    # Minimum Matching Distance
    print("Computing MMD...")
    mmd = minimum_matching_distance(generated_samples, reference_samples)
    
    # Coverage
    print("Computing Coverage...")
    coverage = coverage_score(generated_samples, reference_samples, threshold=0.01)
    
    # Results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Number of samples: {num_samples}")
    print(f"Average Chamfer Distance: {avg_cd:.6f}")
    print(f"Minimum Matching Distance: {mmd:.6f}")
    print(f"Coverage Score: {coverage:.4f}")
    print("=" * 50)
    
    return {
        'chamfer_distance': avg_cd,
        'mmd': mmd,
        'coverage': coverage
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D Generator')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data_path', type=str, default='./data/shapenet',
                        help='Path to evaluation data')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    args = parser.parse_args()
    
    # Set device
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    model = Generator3D(
        num_points=Config.NUM_POINTS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load dataset
    if args.use_synthetic:
        dataset = SyntheticDataset(
            num_samples=args.num_samples,
            num_points=Config.NUM_POINTS
        )
    else:
        dataset = PointCloudDataset(
            data_dir=args.data_path,
            num_points=Config.NUM_POINTS,
            normalize=True,
            augment=False
        )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS
    )
    
    # Evaluate
    results = evaluate_model(model, dataloader, device, args.num_samples)


if __name__ == '__main__':
    main()
