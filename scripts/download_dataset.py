"""
Script to download and prepare datasets
"""

import argparse
from pathlib import Path
import requests
import zipfile
import shutil
from tqdm import tqdm


def download_file(url: str, output_path: Path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_shapenet_sample():
    """
    Download a sample of ShapeNet data
    Note: Full ShapeNet requires registration at shapenet.org
    """
    print("ShapeNet requires registration at https://shapenet.org/")
    print("Please download manually and place in ./data/shapenet/")
    print("\nAlternatively, use synthetic data for testing:")
    print("  python demo.py 2")


def download_modelnet40():
    """Download ModelNet40 dataset"""
    print("Downloading ModelNet40...")
    
    data_dir = Path('./data/modelnet40')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    zip_path = data_dir / "ModelNet40.zip"
    
    print(f"Downloading from {url}...")
    download_file(url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    print("Cleaning up...")
    zip_path.unlink()
    
    print(f"✓ ModelNet40 downloaded to {data_dir}")


def prepare_custom_dataset(input_dir: str, output_dir: str):
    """
    Prepare custom dataset by organizing files
    
    Args:
        input_dir: Directory with 3D model files
        output_dir: Output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported formats
    extensions = ['.obj', '.ply', '.stl', '.off']
    
    print(f"Scanning {input_path} for 3D models...")
    
    files = []
    for ext in extensions:
        files.extend(input_path.rglob(f"*{ext}"))
    
    print(f"Found {len(files)} 3D model files")
    
    if len(files) == 0:
        print("No 3D model files found!")
        return
    
    print(f"Copying files to {output_path}...")
    
    for i, file_path in enumerate(tqdm(files)):
        # Create category directory if in path
        rel_path = file_path.relative_to(input_path)
        dest_path = output_path / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(file_path, dest_path)
    
    print(f"✓ Prepared {len(files)} files in {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['shapenet', 'modelnet40', 'custom', 'synthetic'],
                        help='Dataset to download')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory for custom dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--category', type=str, default=None,
                        help='Specific category to download (if supported)')
    args = parser.parse_args()
    
    if args.dataset == 'shapenet':
        download_shapenet_sample()
    elif args.dataset == 'modelnet40':
        download_modelnet40()
    elif args.dataset == 'custom':
        if not args.input_dir:
            print("Error: --input_dir required for custom dataset")
            return
        prepare_custom_dataset(args.input_dir, args.output_dir)
    elif args.dataset == 'synthetic':
        print("Using synthetic data - no download needed!")
        print("Run training with --use_synthetic flag:")
        print("  python train.py --use_synthetic")
        print("\nOr try the demo:")
        print("  python demo.py 2")


if __name__ == '__main__':
    main()
