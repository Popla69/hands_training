"""
Test class X accuracy with 1000 random images
Compares original model vs fine-tuned model
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
from datetime import datetime
from collections import Counter
import time

print("="*70)
print("CLASS X ACCURACY TEST - 1000 RANDOM IMAGES")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
ORIGINAL_MODEL = 'models_tf2/checkpoint_resume.h5'
FINETUNED_MODEL = 'models_tf2/finetuned_X_model.h5'
LABELS_PATH = 'models_tf2/labels.txt'
DATASET_DIR = 'dataset'
NUM_TEST_IMAGES = 1000
IMG_SIZE = 224

# Load labels
print("Loading labels...")
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✓ Loaded {len(labels)} classes")

# Check which models exist
use_finetuned = os.path.exists(FINETUNED_MODEL)

# Load models
print(f"\nLoading models...")
print(f"1. Original model: {ORIGINAL_MODEL}")
original_model = load_model(ORIGINAL_MODEL)
print("   ✓ Loaded")

if use_finetuned:
    print(f"2. Fine-tuned model: {FINETUNED_MODEL}")
    finetuned_model = load_model(FINETUNED_MODEL)
    print("   ✓ Loaded")
else:
    print(f"2. Fine-tuned model not found - testing original only")
    finetuned_model = None

# Collect all images
print(f"\nScanning dataset: {DATASET_DIR}")
all_images = []

for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append({
                'path': os.path.join(class_path, img_file),
                'true_label': class_name,
                'filename': img_file
            })

print(f"✓ Found {len(all_images)} total images")

# Sample 1000 random images
random.seed(42)
test_images = random.sample(all_images, min(NUM_TEST_IMAGES, len(all_images)))
print(f"✓ Selected {len(test_images)} images for testing")

# Count X images
x_count = sum(1 for img in test_images if img['true_label'] == 'X')
print(f"✓ Class X images in sample: {x_count}")

# Test
print("\n" + "="*70)
print("TESTING IN PROGRESS...")
print("="*70)

original_results = {'correct': 0, 'total': 0, 'class_stats': {}}
finetuned_results = {'correct': 0, 'total': 0, 'class_stats': {}}

for label in labels:
    original_results['class_stats'][label] = {'correct': 0, 'total': 0, 'confidences': []}
    finetuned_results['class_stats'][label] = {'correct': 0, 'total': 0, 'confidences': []}

start_time = time.time()

for idx, img_data in enumerate(test_images, 1):
    try:
        # Load image
        img = image.load_img(img_data['path'], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        true_label = img_data['true_label']
        
        # Test original model
        orig_pred = original_model.predict(img_array, verbose=0)
        orig_idx = np.argmax(orig_pred[0])
        orig_label = labels[orig_idx]
        orig_conf = orig_pred[0][orig_idx]
        
        original_results['total'] += 1
        original_results['class_stats'][true_label]['total'] += 1
        original_results['class_stats'][true_label]['confidences'].append(orig_conf)
        
        if orig_label == true_label:
            original_results['correct'] += 1
            original_results['class_stats'][true_label]['correct'] += 1
        
        # Test fine-tuned model if available
        if finetuned_model:
            fine_pred = finetuned_model.predict(img_array, verbose=0)
            fine_idx = np.argmax(fine_pred[0])
            fine_label = labels[fine_idx]
            fine_conf = fine_pred[0][fine_idx]
            
            finetuned_results['total'] += 1
            finetuned_results['class_stats'][true_label]['total'] += 1
            finetuned_results['class_stats'][true_label]['confidences'].append(fine_conf)
            
            if fine_label == true_label:
                finetuned_results['correct'] += 1
                finetuned_results['class_stats'][true_label]['correct'] += 1
        
        # Progress
        if idx % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{idx}/{len(test_images)}] Elapsed: {elapsed:.1f}s")
    
    except Exception as e:
        print(f"  Error: {img_data['filename']}: {e}")
        continue

total_time = time.time() - start_time

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

orig_acc = (original_results['correct'] / original_results['total']) * 100
print(f"\nOriginal Model:")
print(f"  Total tested: {original_results['total']}")
print(f"  Correct: {original_results['correct']}")
print(f"  Accuracy: {orig_acc:.2f}%")

if finetuned_model:
    fine_acc = (finetuned_results['correct'] / finetuned_results['total']) * 100
    print(f"\nFine-tuned Model:")
    print(f"  Total tested: {finetuned_results['total']}")
    print(f"  Correct: {finetuned_results['correct']}")
    print(f"  Accuracy: {fine_acc:.2f}%")
    print(f"  Change: {fine_acc - orig_acc:+.2f}%")

print(f"\nTest Duration: {total_time:.1f} seconds")
print(f"Speed: {len(test_images)/total_time:.1f} images/second")

# Focus on Class X
print(f"\n{'='*70}")
print("CLASS X DETAILED ANALYSIS")
print(f"{'='*70}")

x_orig = original_results['class_stats']['X']
if x_orig['total'] > 0:
    x_orig_acc = (x_orig['correct'] / x_orig['total']) * 100
    x_orig_conf = np.mean(x_orig['confidences'])
    
    print(f"\nOriginal Model - Class X:")
    print(f"  Tested: {x_orig['total']} images")
    print(f"  Correct: {x_orig['correct']}")
    print(f"  Accuracy: {x_orig_acc:.2f}%")
    print(f"  Avg Confidence: {x_orig_conf:.2%}")
    
    if finetuned_model:
        x_fine = finetuned_results['class_stats']['X']
        x_fine_acc = (x_fine['correct'] / x_fine['total']) * 100
        x_fine_conf = np.mean(x_fine['confidences'])
        x_improvement = x_fine_acc - x_orig_acc
        
        print(f"\nFine-tuned Model - Class X:")
        print(f"  Tested: {x_fine['total']} images")
        print(f"  Correct: {x_fine['correct']}")
        print(f"  Accuracy: {x_fine_acc:.2f}%")
        print(f"  Avg Confidence: {x_fine_conf:.2%}")
        print(f"\n  Improvement: {x_improvement:+.2f}%")
        
        if x_improvement > 5:
            print(f"  ✓✓ EXCELLENT IMPROVEMENT!")
        elif x_improvement > 2:
            print(f"  ✓ GOOD IMPROVEMENT")
        elif x_improvement > 0:
            print(f"  ✓ Slight improvement")
        else:
            print(f"  ✗ No improvement")

# Per-class comparison
print(f"\n{'='*70}")
print("ALL CLASSES COMPARISON")
print(f"{'='*70}")

if finetuned_model:
    print(f"\n{'Class':<10} {'Original':<12} {'Fine-tuned':<12} {'Change':<10}")
    print("-" * 50)
    
    for label in sorted(labels):
        orig_stats = original_results['class_stats'][label]
        fine_stats = finetuned_results['class_stats'][label]
        
        if orig_stats['total'] > 0:
            orig_class_acc = (orig_stats['correct'] / orig_stats['total']) * 100
            fine_class_acc = (fine_stats['correct'] / fine_stats['total']) * 100
            change = fine_class_acc - orig_class_acc
            
            marker = ""
            if label == 'X':
                marker = " ← TARGET"
            elif change < -2:
                marker = " ↓"
            elif change > 2:
                marker = " ↑"
            
            print(f"{label:<10} {orig_class_acc:>6.2f}%     {fine_class_acc:>6.2f}%     "
                  f"{change:>+6.2f}%{marker}")
else:
    print(f"\n{'Class':<10} {'Tested':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-" * 50)
    
    for label in sorted(labels):
        stats = original_results['class_stats'][label]
        if stats['total'] > 0:
            class_acc = (stats['correct'] / stats['total']) * 100
            marker = " ← X" if label == 'X' else ""
            print(f"{label:<10} {stats['total']:<10} {stats['correct']:<10} {class_acc:>6.2f}%{marker}")

# Save results
results_file = 'test_results/class_X_1000_test.txt'
os.makedirs('test_results', exist_ok=True)

with open(results_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("CLASS X TEST - 1000 RANDOM IMAGES\n")
    f.write("="*70 + "\n")
    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write(f"Original Model Accuracy: {orig_acc:.2f}%\n")
    if finetuned_model:
        f.write(f"Fine-tuned Model Accuracy: {fine_acc:.2f}%\n")
        f.write(f"Overall Change: {fine_acc - orig_acc:+.2f}%\n")
    
    f.write(f"\nClass X Results:\n")
    f.write(f"  Original: {x_orig_acc:.2f}%\n")
    if finetuned_model:
        f.write(f"  Fine-tuned: {x_fine_acc:.2f}%\n")
        f.write(f"  Improvement: {x_improvement:+.2f}%\n")

print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*70)
print("TEST COMPLETE!")
print("="*70)

