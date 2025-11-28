"""
Automated FreiHAND dataset preparation script
"""

import os
import json
import numpy as np
from tqdm import tqdm

print("="*70)
print("FreiHAND Dataset Preparation")
print("="*70)

# Check if FreiHAND data exists
freihand_dir = 'datasets/freihand'
output_dir = 'data/freihand_converted'

if not os.path.exists(freihand_dir):
    print("\n✗ FreiHAND dataset not found at:", freihand_dir)
    print("\nPlease download FreiHAND dataset first:")
    print("1. Visit: https://lmb.informatik.uni-freiburg.de/projects/freihand/")
    print("2. Register and download:")
    print("   - training_rgb.zip")
    print("   - training_xyz.json")
    print("   - training_K.json")
    print("3. Extract to:", freihand_dir)
    print("\nExpected structure:")
    print("  data/freihand/")
    print("    training/")
    print("      rgb/")
    print("        00000000.jpg")
    print("        00000001.jpg")
    print("        ...")
    print("    training_xyz.json")
    print("    training_K.json")
    exit(1)

# Check required files
xyz_file = os.path.join(freihand_dir, 'training_xyz.json')
k_file = os.path.join(freihand_dir, 'training_K.json')
rgb_dir = os.path.join(freihand_dir, 'training', 'rgb')

if not os.path.exists(xyz_file):
    print(f"\n✗ Missing: {xyz_file}")
    exit(1)
if not os.path.exists(k_file):
    print(f"\n✗ Missing: {k_file}")
    exit(1)
if not os.path.exists(rgb_dir):
    print(f"\n✗ Missing: {rgb_dir}")
    exit(1)

print("\n✓ All required files found")
print("\nLoading annotations...")

# Load annotations
with open(xyz_file, 'r') as f:
    xyz_data = json.load(f)

with open(k_file, 'r') as f:
    k_data = json.load(f)

print(f"✓ Loaded {len(xyz_data)} samples")

# Create output directory
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

print("\nConverting annotations...")

# Convert annotations
annotations = {}
train_annotations = {}
val_annotations = {}

for i in tqdm(range(len(xyz_data)), desc="Processing"):
    # Get 3D coordinates and camera matrix
    xyz = np.array(xyz_data[i]).reshape(21, 3)
    K = np.array(k_data[i]).reshape(3, 3)
    
    # Project to 2D
    xyz_homo = np.hstack([xyz, np.ones((21, 1))])
    uv_homo = (K @ xyz.T).T
    uv = uv_homo[:, :2] / uv_homo[:, 2:3]
    
    # Combine with depth
    landmarks = np.hstack([uv, xyz[:, 2:3]])
    
    # Image filename
    img_name = f'{i:08d}.jpg'
    src_path = os.path.join(rgb_dir, img_name)
    dst_path = os.path.join(output_dir, 'images', img_name)
    
    # Copy image (or create symlink)
    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            try:
                os.symlink(src_path, dst_path)
            except:
                # If symlink fails, copy file
                import shutil
                shutil.copy2(src_path, dst_path)
    
    # Store annotation
    annotations[img_name] = {
        'landmarks': landmarks.tolist()
    }

print(f"\n✓ Converted {len(annotations)} annotations")

# Split into train/val (90/10)
print("\nSplitting into train/val...")

import random
random.seed(42)

items = list(annotations.items())
random.shuffle(items)

split_idx = int(len(items) * 0.9)
train_items = items[:split_idx]
val_items = items[split_idx:]

train_annotations = dict(train_items)
val_annotations = dict(val_items)

print(f"✓ Train: {len(train_annotations)} samples")
print(f"✓ Val: {len(val_annotations)} samples")

# Save annotations
print("\nSaving annotations...")

with open(os.path.join(output_dir, 'train_annotations.json'), 'w') as f:
    json.dump(train_annotations, f)

with open(os.path.join(output_dir, 'val_annotations.json'), 'w') as f:
    json.dump(val_annotations, f)

with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
    json.dump(annotations, f)

print("✓ Annotations saved")

# Create README
readme_content = f"""# FreiHAND Dataset (Converted)

## Statistics
- Total samples: {len(annotations)}
- Training samples: {len(train_annotations)}
- Validation samples: {len(val_annotations)}

## Format
- Images: {output_dir}/images/
- Annotations: JSON files with 21 landmarks per hand
- Landmark format: [[x1, y1, z1], [x2, y2, z2], ...]

## Usage
```bash
python hand_landmark_v2/train.py --data_dir {output_dir} --epochs 200
```

## Original Dataset
FreiHAND: https://lmb.informatik.uni-freiburg.de/projects/freihand/
"""

with open(os.path.join(output_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

print("\n" + "="*70)
print("FreiHAND Dataset Preparation Complete!")
print("="*70)
print(f"\nDataset location: {output_dir}")
print(f"Total samples: {len(annotations)}")
print(f"Train samples: {len(train_annotations)}")
print(f"Val samples: {len(val_annotations)}")
print("\nNext steps:")
print("1. Verify dataset:")
print(f"   python hand_landmark_v2/download_datasets.py verify {output_dir}")
print("\n2. Start training:")
print(f"   python hand_landmark_v2/train.py --data_dir {output_dir} --epochs 200")
print("\n3. Monitor training:")
print("   tensorboard --logdir hand_landmark_v2/logs")
print("="*70)
