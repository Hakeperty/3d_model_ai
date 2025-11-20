"""
Demo script showing basic usage
"""

import torch
from pathlib import Path

from config import Config
from models import Generator3D
from utils import export_to_obj, visualize_point_cloud


def demo_generation():
    """Demo: Generate random 3D objects"""
    print("=" * 50)
    print("Demo: Generate Random 3D Objects")
    print("=" * 50)
    
    # Create directories
    Config.create_dirs()
    
    # Set device
    device = Config.get_device()
    print(f"\nUsing device: {device}")
    
    # Create model
    print("\nCreating Generator3D model...")
    model = Generator3D(
        num_points=Config.NUM_POINTS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Generate samples
    print("\nGenerating 3D objects...")
    model.eval()
    
    with torch.no_grad():
        # Generate 4 samples
        samples = model.generate(
            batch_size=4,
            device=device,
            num_steps=50  # Fast generation
        )
    
    print(f"Generated {len(samples)} point clouds")
    print(f"Shape: {samples.shape}")  # Should be [4, 2048, 3]
    
    # Save samples
    output_dir = Config.OUTPUT_DIR / "demo"
    output_dir.mkdir(exist_ok=True)
    
    for i, point_cloud in enumerate(samples):
        # Export to OBJ
        obj_path = output_dir / f"sample_{i}.obj"
        export_to_obj(point_cloud, obj_path, convert_to_mesh=False)
        print(f"Saved {obj_path}")
        
        # Create visualization
        vis_path = output_dir / f"sample_{i}.png"
        visualize_point_cloud(
            point_cloud,
            save_path=vis_path,
            title=f"Generated Sample {i}"
        )
    
    print(f"\n✓ Demo completed! Check {output_dir} for outputs.")
    print("\nNote: This model is randomly initialized.")
    print("For good results, train it first using train.py or train_multigpu.py")


def demo_synthetic_training():
    """Demo: Train on synthetic data"""
    print("\n" + "=" * 50)
    print("Demo: Quick Training on Synthetic Data")
    print("=" * 50)
    
    from data import SyntheticDataset, create_dataloader
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm
    
    # Create directories
    Config.create_dirs()
    
    # Set device
    device = Config.get_device()
    print(f"\nUsing device: {device}")
    
    # Create synthetic dataset
    print("\nCreating synthetic dataset...")
    dataset = SyntheticDataset(
        num_samples=100,
        num_points=Config.NUM_POINTS,
        shape_types=['sphere', 'cube', 'cylinder']
    )
    
    dataloader = create_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = Generator3D(
        num_points=Config.NUM_POINTS,
        hidden_dim=128,  # Smaller for demo
        num_layers=3
    ).to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4
    )
    
    # Train for a few epochs
    print("\nTraining for 5 epochs (demo)...")
    model.train()
    
    for epoch in range(1, 6):
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for point_clouds, _ in pbar:
            point_clouds = point_clouds.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(point_clouds)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")
    
    # Generate samples from trained model
    print("\nGenerating samples from trained model...")
    model.eval()
    
    with torch.no_grad():
        samples = model.generate(batch_size=3, device=device, num_steps=50)
    
    # Save samples
    output_dir = Config.OUTPUT_DIR / "demo_trained"
    output_dir.mkdir(exist_ok=True)
    
    for i, point_cloud in enumerate(samples):
        obj_path = output_dir / f"trained_sample_{i}.obj"
        export_to_obj(point_cloud, obj_path, convert_to_mesh=False)
        
        vis_path = output_dir / f"trained_sample_{i}.png"
        visualize_point_cloud(
            point_cloud,
            save_path=vis_path,
            title=f"Trained Model Sample {i}"
        )
    
    print(f"\n✓ Training demo completed! Check {output_dir} for outputs.")


if __name__ == '__main__':
    import sys
    
    print("\n3D Model AI - Demo Script")
    print("=" * 50)
    print("Choose a demo:")
    print("1. Generate random samples (untrained model)")
    print("2. Quick training on synthetic data")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        demo_generation()
    elif choice == '2':
        demo_synthetic_training()
    else:
        print("\nRunning both demos...\n")
        demo_generation()
        demo_synthetic_training()
    
    print("\n" + "=" * 50)
    print("All demos completed!")
    print("=" * 50)
