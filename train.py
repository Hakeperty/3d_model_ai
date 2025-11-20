"""
Single GPU training script
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from config import Config
from models import create_generator
from data import PointCloudDataset, SyntheticDataset, create_dataloader
from utils.visualization import visualize_point_cloud


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter = None
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (point_clouds, _) in enumerate(pbar):
        point_clouds = point_clouds.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.compute_loss(point_clouds)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if Config.GRADIENT_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                Config.GRADIENT_CLIP
            )
        
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Logging
        pbar.set_postfix({'loss': loss.item()})
        
        if writer and batch_idx % Config.LOG_INTERVAL == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train 3D Generator')
    parser.add_argument('--data_path', type=str, default='./data/shapenet',
                        help='Path to training data')
    parser.add_argument('--category', type=str, default=None,
                        help='Category to train on (e.g., car)')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--run_name', type=str, default='3d_generation',
                        help='Name for this training run')
    args = parser.parse_args()
    
    # Create directories
    Config.create_dirs()
    (Config.PROJECT_ROOT / 'runs').mkdir(exist_ok=True)
    
    # Set device
    device = Config.get_device()
    print(f"Using device: {device}")
    
    # Create model
    model = create_generator(
        num_points=Config.NUM_POINTS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Load dataset
    if args.use_synthetic:
        train_dataset = SyntheticDataset(
            num_samples=1000,
            num_points=Config.NUM_POINTS
        )
    else:
        train_dataset = PointCloudDataset(
            data_dir=args.data_path,
            num_points=Config.NUM_POINTS,
            normalize=True,
            augment=True,
            category=args.category
        )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Create dataloader
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=Config.PROJECT_ROOT / 'runs' / args.run_name)
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    print("\nStarting training...\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        
        # Step scheduler
        scheduler.step()
        
        # Logging
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        writer.add_scalar('train/epoch_loss', train_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        if epoch % Config.SAVE_INTERVAL == 0 or train_loss < best_loss:
            if train_loss < best_loss:
                best_loss = train_loss
            
            checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Generate samples for visualization
        if epoch % Config.VISUALIZE_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                samples = model.generate(batch_size=4, device=device, num_steps=50)
                # Save first sample
                save_path = Config.OUTPUT_DIR / f'sample_epoch_{epoch}.png'
                visualize_point_cloud(samples[0].cpu().numpy(), save_path)
            model.train()
    
    # Print training time
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/3600:.2f} hours")
    writer.close()


if __name__ == '__main__':
    main()
