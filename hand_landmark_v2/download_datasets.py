"""
Dataset download and preparation utilities
"""

import os
import urllib.request
import zipfile
import json
import numpy as np


def download_file(url, output_path):
    """
    Download file with progress bar
    
    Args:
        url: URL to download from
        output_path: Path to save file
    """
    print(f"Downloading {url}...")
    
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\r  Progress: {percent}%", end='')
    
    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print("\n✓ Download complete")


def download_freihand(output_dir='data/freihand'):
    """
    Download FreiHAND dataset
    
    FreiHAND is a dataset for hand pose estimation with 130K training images
    and 3960 test images with 3D hand pose annotations.
    
    Note: This is a placeholder. The actual FreiHAND dataset requires
    registration and manual download from:
    https://lmb.informatik.uni-freiburg.de/projects/freihand/
    
    Args:
        output_dir: Directory to save dataset
    """
    print("="*70)
    print("FreiHAND Dataset Download")
    print("="*70)
    print("\nNote: FreiHAND dataset requires manual download.")
    print("Please visit: https://lmb.informatik.uni-freiburg.de/projects/freihand/")
    print("\nSteps:")
    print("1. Register and download the dataset")
    print("2. Extract to:", output_dir)
    print("3. Run conversion script to create annotations.json")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create placeholder README
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("FreiHAND Dataset\n")
        f.write("="*50 + "\n\n")
        f.write("Download from: https://lmb.informatik.uni-freiburg.de/projects/freihand/\n\n")
        f.write("Expected structure:\n")
        f.write("  freihand/\n")
        f.write("    training/\n")
        f.write("      rgb/\n")
        f.write("      mask/\n")
        f.write("    evaluation/\n")
        f.write("      rgb/\n")
        f.write("    training_xyz.json\n")
        f.write("    training_K.json\n")
    
    print(f"\n✓ Created placeholder at {output_dir}")


def download_cmu_hand(output_dir='data/cmu_hand'):
    """
    Download CMU Hand dataset
    
    CMU Panoptic Hand dataset contains multi-view hand pose data.
    
    Note: This is a placeholder. The actual CMU dataset requires
    manual download from:
    http://domedb.perception.cs.cmu.edu/handdb.html
    
    Args:
        output_dir: Directory to save dataset
    """
    print("="*70)
    print("CMU Hand Dataset Download")
    print("="*70)
    print("\nNote: CMU Hand dataset requires manual download.")
    print("Please visit: http://domedb.perception.cs.cmu.edu/handdb.html")
    print("\nSteps:")
    print("1. Download the dataset")
    print("2. Extract to:", output_dir)
    print("3. Run conversion script to create annotations.json")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create placeholder README
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("CMU Hand Dataset\n")
        f.write("="*50 + "\n\n")
        f.write("Download from: http://domedb.perception.cs.cmu.edu/handdb.html\n\n")
        f.write("Expected structure:\n")
        f.write("  cmu_hand/\n")
        f.write("    images/\n")
        f.write("    annotations/\n")
    
    print(f"\n✓ Created placeholder at {output_dir}")


def convert_freihand_format(freihand_dir, output_dir):
    """
    Convert FreiHAND format to our annotation format
    
    Args:
        freihand_dir: FreiHAND dataset directory
        output_dir: Output directory for converted annotations
    """
    print("Converting FreiHAND format...")
    
    # Load FreiHAND annotations
    xyz_path = os.path.join(freihand_dir, 'training_xyz.json')
    k_path = os.path.join(freihand_dir, 'training_K.json')
    
    if not os.path.exists(xyz_path):
        print(f"✗ Error: {xyz_path} not found")
        return
    
    with open(xyz_path, 'r') as f:
        xyz_data = json.load(f)
    
    with open(k_path, 'r') as f:
        k_data = json.load(f)
    
    # Convert to our format
    annotations = {}
    
    for i, (xyz, K) in enumerate(zip(xyz_data, k_data)):
        img_name = f'{i:08d}.jpg'
        
        # Convert 3D coordinates to 2D + depth
        xyz_array = np.array(xyz).reshape(21, 3)
        K_array = np.array(K).reshape(3, 3)
        
        # Project to 2D
        xyz_homo = np.hstack([xyz_array, np.ones((21, 1))])
        uv_homo = (K_array @ xyz_array.T).T
        uv = uv_homo[:, :2] / uv_homo[:, 2:3]
        
        # Combine with depth
        landmarks = np.hstack([uv, xyz_array[:, 2:3]])
        
        annotations[img_name] = {
            'landmarks': landmarks.tolist()
        }
    
    # Save annotations
    os.makedirs(output_dir, exist_ok=True)
    ann_path = os.path.join(output_dir, 'train_annotations.json')
    
    with open(ann_path, 'w') as f:
        json.dump(annotations, f)
    
    print(f"✓ Converted {len(annotations)} annotations")
    print(f"  Saved to: {ann_path}")


def prepare_custom_dataset(images_dir, output_dir):
    """
    Prepare custom dataset from images directory
    
    Creates a template annotations file that needs to be filled manually
    or with a labeling tool.
    
    Args:
        images_dir: Directory containing images
        output_dir: Output directory for dataset
    """
    print("Preparing custom dataset...")
    
    # List all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images")
    
    # Create template annotations
    annotations = {}
    for img_file in image_files:
        annotations[img_file] = {
            'landmarks': [[0.0, 0.0, 0.0] for _ in range(21)],
            'bbox': [0, 0, 0, 0],
            'annotated': False
        }
    
    # Save template
    os.makedirs(output_dir, exist_ok=True)
    ann_path = os.path.join(output_dir, 'annotations_template.json')
    
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Created annotation template: {ann_path}")
    print("\nNext steps:")
    print("1. Use a labeling tool to annotate landmarks")
    print("2. Fill in the landmark coordinates")
    print("3. Rename to 'annotations.json' when complete")


def verify_dataset(data_dir):
    """
    Verify dataset structure and annotations
    
    Args:
        data_dir: Dataset directory
    """
    print("="*70)
    print("Verifying Dataset")
    print("="*70)
    
    # Check directory structure
    images_dir = os.path.join(data_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"✗ Error: images directory not found: {images_dir}")
        return False
    
    # Check annotations
    ann_file = os.path.join(data_dir, 'annotations.json')
    if not os.path.exists(ann_file):
        print(f"✗ Error: annotations file not found: {ann_file}")
        return False
    
    # Load annotations
    with open(ann_file, 'r') as f:
        annotations = json.load(f)
    
    print(f"✓ Found {len(annotations)} annotations")
    
    # Verify each annotation
    errors = []
    for img_name, ann in annotations.items():
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            errors.append(f"Image not found: {img_name}")
            continue
        
        if 'landmarks' not in ann:
            errors.append(f"Missing landmarks: {img_name}")
            continue
        
        landmarks = ann['landmarks']
        if len(landmarks) != 21:
            errors.append(f"Invalid landmark count: {img_name} ({len(landmarks)})")
            continue
        
        for i, lm in enumerate(landmarks):
            if len(lm) != 3:
                errors.append(f"Invalid landmark dimension: {img_name} landmark {i}")
                break
    
    if errors:
        print(f"\n✗ Found {len(errors)} errors:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        return False
    
    print("✓ Dataset verification passed")
    print("="*70)
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python download_datasets.py <command>")
        print("\nCommands:")
        print("  freihand - Download FreiHAND dataset")
        print("  cmu - Download CMU Hand dataset")
        print("  verify <data_dir> - Verify dataset structure")
        print("  prepare <images_dir> <output_dir> - Prepare custom dataset")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'freihand':
        download_freihand()
    elif command == 'cmu':
        download_cmu_hand()
    elif command == 'verify':
        if len(sys.argv) < 3:
            print("Error: Please specify data directory")
            sys.exit(1)
        verify_dataset(sys.argv[2])
    elif command == 'prepare':
        if len(sys.argv) < 4:
            print("Error: Please specify images_dir and output_dir")
            sys.exit(1)
        prepare_custom_dataset(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
