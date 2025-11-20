"""
Generation script - Generate 3D models from trained model
"""

import torch
import argparse
from pathlib import Path
import time

from config import Config
from models import Generator3D
from utils import export_batch, visualize_batch


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Generator3D:
    """Load model from checkpoint"""
    model = Generator3D(
        num_points=Config.NUM_POINTS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Model loss: {checkpoint['loss']:.6f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Generate 3D Models')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='./outputs/generated',
                        help='Output directory')
    parser.add_argument('--format', type=str, default='obj',
                        choices=['obj', 'ply', 'stl', 'glb'],
                        help='Export format')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Generation batch size')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Load model
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint, device)
    else:
        print("No checkpoint provided. Using randomly initialized model.")
        print("Note: For good results, you need to train the model first!")
        model = Generator3D(
            num_points=Config.NUM_POINTS,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS
        ).to(device)
    
    model.eval()
    
    print(f"\nGenerating {args.num_samples} samples...")
    print(f"Export format: {args.format}")
    print(f"Denoising steps: {args.steps}")
    
    # Generate in batches
    all_samples = []
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(num_batches):
            batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)
            
            print(f"Generating batch {i+1}/{num_batches}...")
            
            samples = model.generate(
                batch_size=batch_size,
                device=device,
                num_steps=args.steps
            )
            
            all_samples.append(samples)
    
    # Concatenate all samples
    all_samples = torch.cat(all_samples, dim=0)
    
    elapsed = time.time() - start_time
    print(f"\nGeneration completed in {elapsed:.2f} seconds")
    print(f"Average time per sample: {elapsed/args.num_samples:.2f} seconds")
    
    # Export samples
    print(f"\nExporting to {args.format} format...")
    export_batch(
        all_samples,
        output_dir,
        format=args.format,
        prefix='generated'
    )
    
    print(f"\nExported {args.num_samples} samples to {output_dir}")
    
    # Create visualization
    if args.visualize:
        print("\nCreating visualization...")
        vis_path = output_dir / 'visualization.png'
        visualize_batch(
            all_samples,
            save_path=vis_path,
            title=f'Generated 3D Models ({args.num_samples} samples)'
        )
        print(f"Saved visualization to {vis_path}")


if __name__ == '__main__':
    main()
