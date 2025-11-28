"""
Setup and initialization script
"""

import os
import subprocess
import sys


def install_dependencies():
    """Install required packages"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Dependencies installed!")


def create_directories():
    """Create necessary directories"""
    dirs = [
        'datasets',
        'checkpoints',
        'models',
        'logs'
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")


def download_datasets():
    """Download public datasets"""
    from dataset import create_synthetic_dataset
    
    print("\nCreating synthetic dataset for testing...")
    create_synthetic_dataset(num_samples=1000)
    print("Synthetic dataset created!")
    
    print("\nTo download real datasets:")
    print("1. FreiHAND: python -c 'from dataset import download_freihand_dataset; download_freihand_dataset()'")
    print("2. Or manually download from: https://lmb.informatik.uni-freiburg.de/projects/freihand/")


def main():
    """Main setup function"""
    print("="*60)
    print("Hand Landmark Model - Setup")
    print("="*60)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    try:
        install_dependencies()
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        print("Please install manually: pip install -r requirements.txt")
    
    # Download datasets
    print("\n3. Setting up datasets...")
    try:
        download_datasets()
    except Exception as e:
        print(f"Error setting up datasets: {e}")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train model: python train.py")
    print("2. Export model: python export.py")
    print("3. Run inference: python inference.py")
    print("="*60)


if __name__ == "__main__":
    main()
