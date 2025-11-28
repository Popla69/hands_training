"""
Fix dataset by copying actual images instead of symlinks
"""

import os
import json
import shutil
from tqdm import tqdm

print("="*70)
print("Fixing Dataset Images")
print("="*70)

source_rgb_dir = 'datasets/freihand/training/rgb'
dest_dir = 'data/freihand_converted/images'

# Load annotations to know which images we need
with open('data/freihand_converted/train_annotations.json', 'r') as f:
    train_ann = json.load(f)

with open('data/freihand_converted/val_annotations.json', 'r') as f:
    val_ann = json.load(f)

all_images = set(list(train_ann.keys()) + list(val_ann.keys()))

print(f"\nNeed to copy {len(all_images)} images")
print(f"From: {source_rgb_dir}")
print(f"To: {dest_dir}")

# Copy images
copied = 0
skipped = 0
errors = 0

for img_name in tqdm(all_images, desc="Copying"):
    src_path = os.path.join(source_rgb_dir, img_name)
    dst_path = os.path.join(dest_dir, img_name)
    
    # Skip if already exists and is a real file
    if os.path.exists(dst_path) and os.path.isfile(dst_path) and os.path.getsize(dst_path) > 0:
        skipped += 1
        continue
    
    # Remove broken symlink if exists
    if os.path.exists(dst_path) or os.path.islink(dst_path):
        try:
            os.remove(dst_path)
        except:
            pass
    
    # Copy file
    if os.path.exists(src_path):
        try:
            shutil.copy2(src_path, dst_path)
            copied += 1
        except Exception as e:
            print(f"\nError copying {img_name}: {e}")
            errors += 1
    else:
        print(f"\nSource not found: {src_path}")
        errors += 1

print("\n" + "="*70)
print("Dataset Fix Complete!")
print("="*70)
print(f"Copied: {copied}")
print(f"Skipped (already exist): {skipped}")
print(f"Errors: {errors}")
print("\nDataset is ready for training!")
print("="*70)
