"""
Configuration file for 3D Model AI Generator
"""

import torch
from pathlib import Path

class Config:
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
    
    # Model architecture
    LATENT_DIM = 512
    POINT_DIM = 3  # x, y, z coordinates
    NUM_POINTS = 2048  # Number of points per object
    HIDDEN_DIM = 256
    NUM_LAYERS = 6
    NUM_HEADS = 8
    
    # Diffusion parameters
    DIFFUSION_STEPS = 1000
    BETA_START = 0.0001
    BETA_END = 0.02
    NOISE_SCHEDULE = "linear"  # linear, cosine, or quadratic
    
    # Training
    BATCH_SIZE = 16  # Per GPU
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0
    
    # Multi-GPU settings
    NUM_GPUS = 2
    DISTRIBUTED_BACKEND = "nccl"  # nccl for NVIDIA GPUs
    FIND_UNUSED_PARAMETERS = False
    
    # Data
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    PIN_MEMORY = True
    
    # Generation
    GENERATION_STEPS = 50  # Fewer steps for faster inference
    GUIDANCE_SCALE = 7.5
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    VISUALIZE_INTERVAL = 50
    USE_WANDB = False  # Set to True if using Weights & Biases
    
    # Export
    EXPORT_FORMAT = "obj"  # obj, ply, stl, or glb
    MESH_RESOLUTION = 64
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @classmethod
    def print_config(cls):
        """Print configuration"""
        print("=" * 50)
        print("Configuration")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith("_") and not callable(value):
                print(f"{key}: {value}")
        print("=" * 50)
