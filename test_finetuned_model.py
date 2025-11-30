"""
Test fine-tuned model to see if X improved without hurting other classes
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
from collections import Counter

print("="*70)
print("TESTING FINE-TUNED MODEL")
print("="*70)

# Configuration
ORIGINAL_MODEL = 'models_tf2/checkpoint_resume.h5'
FINETUNED_MODEL = 'models_tf2/finetuned_X_model.h5'
LABELS_PATH = 'models_tf2/labels.txt'
DATASET_DIR = 'dataset'
NUM_TEST = 5000  # Test on 5000 images
IMG_SIZE = 224

# Load labels
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

print(f"\nLoading models...")
print("1. Original model...")
original_model = load_model(ORIGINAL_MODEL)
print("2. Fine-tuned model...")
finetuned_model = load_model(FINETUNED_MODEL)
print("✓ Both models loaded")

# Collect images
print(f"\nCollecting test images...")
all_images = []
for class_name in os.listdir(DATASET_DIR):
    class_path = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(class_path):
        continue
    for img_file in os.listdir(class_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            all_images.append({
                'path': os.path.join(class_path, img_file),
                'true_label': class_name
            })

random.seed(42)
test_images = random.sample(all_images, min(NUM_TEST, len(all_images)))
print(f"✓ Testing on {len(test_images)} images")

# Test both models
print("\nTesting...")
original_results = {'correct': 0, 'total': 0, 'class_stats': {}}
finetuned_results = {'correct': 0, 'total': 0, 'class_stats': {}}

for label in labels:
    original_results['class_stats'][label] = {'correct': 0, 'total': 0}
    finetuned_results['class_stats'][label] = {'correct': 0, 'total': 0}

for idx, img_data in enumerate(test_images):
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(test_images)}...")
    
    try:
        img = image.load_img(img_data['path'], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        true_label = img_data['true_label']
        
        # Original model
        orig_pred = original_model.predict(img_array, verbose=0)
        orig_label = labels[np.argmax(orig_pred[0])]
        original_results['total'] += 1
        original_results['class_stats'][true_label]['total'] += 1
        if orig_label == true_label:
            original_results['correct'] += 1
            original_results['class_stats'][true_label]['correct'] += 1
        
        # Fine-tuned model
        fine_pred = finetuned_model.predict(img_array, verbose=0)
        fine_label = labels[np.argmax(fine_pred[0])]
        finetuned_results['total'] += 1
        finetuned_results['class_stats'][true_label]['total'] += 1
        if fine_label == true_label:
            finetuned_results['correct'] += 1
            finetuned_results['class_stats'][true_label]['correct'] += 1
    
    except:
        continue

# Results
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

orig_acc = (original_results['correct'] / original_results['total']) * 100
fine_acc = (finetuned_results['correct'] / finetuned_results['total']) * 100

print(f"\nOverall Accuracy:")
print(f"  Original:   {orig_acc:.2f}%")
print(f"  Fine-tuned: {fine_acc:.2f}%")
print(f"  Change:     {fine_acc - orig_acc:+.2f}%")

# Per-class comparison
print(f"\n{'='*70}")
print("PER-CLASS COMPARISON")
print(f"{'='*70}")
print(f"\n{'Class':<10} {'Original':<12} {'Fine-tuned':<12} {'Change':<10}")
print("-" * 50)

improvements = []
degradations = []

for label in sorted(labels):
    orig_stats = original_results['class_stats'][label]
    fine_stats = finetuned_results['class_stats'][label]
    
    if orig_stats['total'] > 0:
        orig_class_acc = (orig_stats['correct'] / orig_stats['total']) * 100
        fine_class_acc = (fine_stats['correct'] / fine_stats['total']) * 100
        change = fine_class_acc - orig_class_acc
        
        status = ""
        if change > 1:
            status = "↑"
            improvements.append((label, change))
        elif change < -1:
            status = "↓"
            degradations.append((label, change))
        
        print(f"{label:<10} {orig_class_acc:>6.2f}%     {fine_class_acc:>6.2f}%     "
              f"{change:>+6.2f}% {status}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

# Focus on X
x_orig = original_results['class_stats']['X']
x_fine = finetuned_results['class_stats']['X']
if x_orig['total'] > 0:
    x_orig_acc = (x_orig['correct'] / x_orig['total']) * 100
    x_fine_acc = (x_fine['correct'] / x_fine['total']) * 100
    x_improvement = x_fine_acc - x_orig_acc
    
    print(f"\nClass X (Target):")
    print(f"  Original:   {x_orig_acc:.2f}%")
    print(f"  Fine-tuned: {x_fine_acc:.2f}%")
    print(f"  Improvement: {x_improvement:+.2f}%")
    
    if x_improvement > 2:
        print(f"  ✓ SIGNIFICANT IMPROVEMENT!")
    elif x_improvement > 0:
        print(f"  ✓ Slight improvement")
    else:
        print(f"  ✗ No improvement")

if improvements:
    print(f"\nImproved classes ({len(improvements)}):")
    for label, change in sorted(improvements, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {label}: +{change:.2f}%")

if degradations:
    print(f"\nDegraded classes ({len(degradations)}):")
    for label, change in sorted(degradations, key=lambda x: x[1])[:5]:
        print(f"  {label}: {change:.2f}%")

# Recommendation
print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")

if x_improvement > 2 and len(degradations) < 3:
    print("\n✓ ACCEPT FINE-TUNED MODEL")
    print("  - X improved significantly")
    print("  - Minimal impact on other classes")
    print(f"\n  To use: Replace checkpoint_resume.h5 with finetuned_X_model.h5")
elif x_improvement > 0 and len(degradations) == 0:
    print("\n✓ ACCEPT FINE-TUNED MODEL")
    print("  - X improved")
    print("  - No degradation in other classes")
else:
    print("\n✗ KEEP ORIGINAL MODEL")
    print("  - Insufficient improvement in X")
    print("  - Or too much degradation in other classes")

print()
