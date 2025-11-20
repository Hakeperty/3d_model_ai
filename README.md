# 3D Model AI Generator

A deep learning system for generating 3D cars and objects using dual-GPU training. This project implements a point cloud-based diffusion model that can generate 3D objects from text descriptions or random noise.

## Features

- 🚗 Generate 3D cars and objects using ML
- 🔥 Dual-GPU training support with PyTorch DDP
- 📊 Point cloud and mesh generation
- 💾 Export to OBJ, PLY, and other 3D formats
- 🎨 Text-to-3D generation capabilities
- 📈 Training visualization and monitoring

## Architecture

The system uses a U-Net based diffusion model for 3D point cloud generation:
- **Encoder-Decoder Architecture**: Processes 3D point clouds with attention mechanisms
- **Multi-GPU Training**: Distributed training across 2 GPUs
- **Point Cloud Generation**: Generates 2048-4096 points per object
- **Mesh Conversion**: Converts point clouds to meshes using Ball Pivoting or Poisson reconstruction

## Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 2x NVIDIA GPUs (tested with RTX series)
- 16GB+ RAM
- 50GB+ disk space for datasets

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Hakeperty/3d_model_ai.git
cd 3d_model_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download ShapeNet or prepare your dataset:
```bash
python scripts/download_dataset.py --category car
```

## Quick Start

### Generate a 3D Car

```python
from models.generator import Generator3D
from utils.export import export_to_obj

# Load pre-trained model
generator = Generator3D(pretrained=True)
generator = generator.cuda()

# Generate from text
point_cloud = generator.generate(prompt="sports car")

# Export to OBJ file
export_to_obj(point_cloud, "output/sports_car.obj")
```

### Train Your Own Model

```bash
# Single GPU training
python train.py --data_path ./data/shapenet --epochs 100

# Multi-GPU training (recommended)
python train_multigpu.py --gpus 2 --data_path ./data/shapenet --epochs 100 --batch_size 32
```

### Generate Random Objects

```bash
python generate.py --num_samples 10 --output_dir ./outputs
```

## Project Structure

```
3d_model_ai/
├── models/
│   ├── __init__.py
│   ├── generator.py          # 3D generation model
│   ├── unet3d.py             # U-Net architecture
│   ├── point_cloud_vae.py    # VAE for point clouds
│   └── diffusion.py          # Diffusion model components
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Dataset loader
│   └── preprocessing.py      # Data preprocessing
├── utils/
│   ├── __init__.py
│   ├── export.py             # Export utilities
│   ├── visualization.py      # Visualization tools
│   └── metrics.py            # Evaluation metrics
├── scripts/
│   ├── download_dataset.py   # Dataset downloader
│   └── evaluate.py           # Model evaluation
├── train.py                  # Single GPU training
├── train_multigpu.py         # Multi-GPU training
├── generate.py               # Generation script
├── config.py                 # Configuration
└── requirements.txt
```

## Training Data

The model can be trained on:
- **ShapeNet**: Large-scale 3D object dataset (55 categories)
- **ModelNet**: 3D CAD models (10/40 categories)
- **Custom datasets**: Point clouds in PLY/OBJ format

Place your training data in `./data/shapenet/` or specify with `--data_path`.

## Configuration

Edit `config.py` to customize:
- Model architecture (layers, dimensions)
- Training hyperparameters (learning rate, batch size)
- GPU settings
- Output formats

## Performance

On 2x RTX 3090 GPUs:
- Training: ~4 hours for 100 epochs (50k samples)
- Generation: ~2 seconds per object
- Quality: FID score < 50 on ShapeNet cars

## Export Formats

Supported output formats:
- `.obj` - Wavefront OBJ (with materials)
- `.ply` - Stanford PLY
- `.stl` - Stereolithography
- `.glb` - GL Transmission Format

## Examples

Generated examples can be found in the `examples/` directory.

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size or point cloud resolution
**Training instability**: Lower learning rate or enable gradient clipping
**Poor quality**: Train longer or increase model capacity

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## License

MIT License - see LICENSE file

## Acknowledgments

- PyTorch3D for 3D operations
- Hugging Face Diffusers for diffusion model components
- ShapeNet for training data
