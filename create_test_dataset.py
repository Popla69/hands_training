"""
Create a small test dataset from FreiHAND for quick testing
"""

import os
import json
import shutil
from tqdm import tqdm

print("="*70)
print("Creating Test Dataset (10 images)")
print("="*70)

# Source and destination
source_dir = 'data/freihand_converted'
test_dir = 'data/freihand_test'

# Create directories
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)

# Load full annotations
with open(os.path.join(source_dir, 'train_annotations.json'), 'r') as f:
    full_annotations = json.load(f)

# Select first 10 images
test_images = list(full_annotations.keys())[:10]

print(f"\nSelected {len(test_images)} images for testing")

# Copy images and create annotations
test_annotations = {}
train_annotations = {}
val_annotations = {}

for i, img_name in enumerate(tqdm(test_images, desc="Copying images")):
    # Get annotation
    test_annotations[img_name] = full_annotations[img_name]
    
    # Split: 7 train, 2 val, 1 test
    if i < 7:
        train_annotations[img_name] = full_annotations[img_name]
    elif i < 9:
        val_annotations[img_name] = full_annotations[img_name]
    
    # Copy image from original FreiHAND location
    src_path = os.path.join('datasets/freihand/training/rgb', img_name)
    dst_path = os.path.join(test_dir, 'images', img_name)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
    else:
        print(f"Warning: {src_path} not found")

# Save annotations
with open(os.path.join(test_dir, 'annotations.json'), 'w') as f:
    json.dump(test_annotations, f)

with open(os.path.join(test_dir, 'train_annotations.json'), 'w') as f:
    json.dump(train_annotations, f)

with open(os.path.join(test_dir, 'val_annotations.json'), 'w') as f:
    json.dump(val_annotations, f)

print("\n" + "="*70)
print("Test Dataset Created!")
print("="*70)
print(f"Location: {test_dir}")
print(f"Total images: {len(test_annotations)}")
print(f"Train: {len(train_annotations)}")
print(f"Val: {len(val_annotations)}")
print("\nTest training with:")
print(f"  python hand_landmark_v2/train.py --data_dir {test_dir} --epochs 5 --batch_size 4")
print("="*70)
