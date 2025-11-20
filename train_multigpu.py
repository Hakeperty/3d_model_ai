"""
Multi-GPU training script using PyTorch DDP
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import time

from config import Config
from models import create_generator
from data import PointCloudDataset, SyntheticDataset, create_dataloader
from utils.visualization import visualize_point_cloud


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed training"""
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend=Config.DISTRIBUTED_BACKEND,
        rank=rank,
        world_size=world_size
    )


def cleanup_ddp():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rank: int,
    writer: SummaryWriter = None
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    for batch_idx, (point_clouds, _) in enumerate(pbar):
        point_clouds = point_clouds.to(rank)
        
        # Forward pass
        optimizer.zero_grad()
        loss = model.module.compute_loss(point_clouds)
        
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
        
        # Logging (only rank 0)
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
            
            if writer and batch_idx % Config.LOG_INTERVAL == 0:
                global_step = epoch * len(dataloader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    rank: int
):
    """Validate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for point_clouds, _ in dataloader:
            point_clouds = point_clouds.to(rank)
            loss = model.module.compute_loss(point_clouds)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_worker(rank: int, world_size: int, args):
    """Training worker for each GPU"""
    
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Create model
    model = create_generator(
        num_points=Config.NUM_POINTS,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS
    ).to(device)
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=Config.FIND_UNUSED_PARAMETERS
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.NUM_EPOCHS,
        eta_min=Config.LEARNING_RATE * 0.01
    )
    
    # Load datasets
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
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True
    )
    
    # Create tensorboard writer (only rank 0)
    writer = None
    if rank == 0:
        writer = SummaryWriter(log_dir=Config.PROJECT_ROOT / 'runs' / args.run_name)
        print(f"\nStarting training on {world_size} GPUs")
        print(f"Model parameters: {model.module.get_num_parameters():,}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Batch size per GPU: {Config.BATCH_SIZE}")
        print(f"Effective batch size: {Config.BATCH_SIZE * world_size}\n")
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, rank, writer)
        
        # Step scheduler
        scheduler.step()
        
        # Logging (only rank 0)
        if rank == 0:
            print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            if writer:
                writer.add_scalar('train/epoch_loss', train_loss, epoch)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
            
            # Save checkpoint
            if epoch % Config.SAVE_INTERVAL == 0 or train_loss < best_loss:
                if train_loss < best_loss:
                    best_loss = train_loss
                
                checkpoint_path = Config.CHECKPOINT_DIR / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Generate samples for visualization
            if epoch % Config.VISUALIZE_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    samples = model.module.generate(batch_size=4, device=device, num_steps=50)
                    # Save first sample
                    save_path = Config.OUTPUT_DIR / f'sample_epoch_{epoch}.png'
                    visualize_point_cloud(samples[0].cpu().numpy(), save_path)
                model.train()
    
    # Print training time
    if rank == 0:
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/3600:.2f} hours")
        if writer:
            writer.close()
    
    # Cleanup
    cleanup_ddp()


def main():
    parser = argparse.ArgumentParser(description='Train 3D Generator with Multi-GPU')
    parser.add_argument('--data_path', type=str, default='./data/shapenet',
                        help='Path to training data')
    parser.add_argument('--category', type=str, default=None,
                        help='Category to train on (e.g., car)')
    parser.add_argument('--use_synthetic', action='store_true',
                        help='Use synthetic data for testing')
    parser.add_argument('--gpus', type=int, default=Config.NUM_GPUS,
                        help='Number of GPUs to use')
    parser.add_argument('--run_name', type=str, default='3d_generation',
                        help='Name for this training run')
    args = parser.parse_args()
    
    # Create directories
    Config.create_dirs()
    (Config.PROJECT_ROOT / 'runs').mkdir(exist_ok=True)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script requires GPUs.")
    
    if torch.cuda.device_count() < args.gpus:
        print(f"Warning: Requested {args.gpus} GPUs but only {torch.cuda.device_count()} available")
        args.gpus = torch.cuda.device_count()
    
    print(f"Using {args.gpus} GPUs for training")
    
    # Launch training on multiple GPUs
    world_size = args.gpus
    mp.spawn(
        train_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    main()
