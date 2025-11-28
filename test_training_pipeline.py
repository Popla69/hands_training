"""
Test training pipeline with 10 images
"""

import os
import json
import shutil
import sys

print("="*70)
print("Testing Training Pipeline with 10 Images")
print("="*70)

# Create test dataset directory
test_dir = 'data/test_10_images'
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=True)

# Copy 10 images from FreiHAND
source_dir = 'data/freihand_converted'
source_images = os.path.join(source_dir, 'images')
source_annotations = os.path.join(source_dir, 'train_annotations.json')

if not os.path.exists(source_annotations):
    print("✗ FreiHAND dataset not prepared yet")
    print("Run: python prepare_freihand.py")
    sys.exit(1)

# Load annotations
with open(source_annotations, 'r') as f:
    all_annotations = json.load(f)

# Take first 10 images
test_annotations = {}
train_annotations = {}
val_annotations = {}

image_files = list(all_annotations.keys())[:10]

print(f"\nCopying 10 test images...")
for i, img_name in enumerate(image_files):
    src = os.path.join(source_images, img_name)
    dst = os.path.join(test_dir, 'images', img_name)
    
    if os.path.exists(src):
        shutil.copy2(src, dst)
        test_annotations[img_name] = all_annotations[img_name]
        
        # Split: 7 train, 3 val
        if i < 7:
            train_annotations[img_name] = all_annotations[img_name]
        else:
            val_annotations[img_name] = all_annotations[img_name]
        
        print(f"  ✓ Copied {img_name}")

# Save annotations
with open(os.path.join(test_dir, 'annotations.json'), 'w') as f:
    json.dump(test_annotations, f)

with open(os.path.join(test_dir, 'train_annotations.json'), 'w') as f:
    json.dump(train_annotations, f)

with open(os.path.join(test_dir, 'val_annotations.json'), 'w') as f:
    json.dump(val_annotations, f)

print(f"\n✓ Test dataset created:")
print(f"  Total: {len(test_annotations)} images")
print(f"  Train: {len(train_annotations)} images")
print(f"  Val: {len(val_annotations)} images")
print(f"  Location: {test_dir}")

print("\n" + "="*70)
print("Starting Test Training (5 epochs)")
print("="*70)

# Run training
import subprocess
result = subprocess.run([
    sys.executable,
    'hand_landmark_v2/train.py',
    '--data_dir', test_dir,
    '--epochs', '5',
    '--batch_size', '2',
    '--lr', '0.001'
], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("Errors/Warnings:")
    print(result.stderr)

if result.returncode == 0:
    print("\n" + "="*70)
    print("✓ Training Pipeline Test PASSED!")
    print("="*70)
    print("\nThe training code works correctly!")
    print("You can now train on the full dataset:")
    print("  python hand_landmark_v2/train.py --data_dir data/freihand_converted --epochs 200")
else:
    print("\n" + "="*70)
    print("✗ Training Pipeline Test FAILED")
    print("="*70)
    print(f"Exit code: {result.returncode}")
    sys.exit(1)
