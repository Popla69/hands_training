"""
Test trained model on 1000 random images from dataset
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
print("COMPREHENSIVE MODEL TEST - 1000 RANDOM IMAGES")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
MODEL_PATH = 'models_tf2/checkpoint_resume.h5'
LABELS_PATH = 'models_tf2/labels.txt'
DATASET_DIR = 'dataset'
NUM_TEST_IMAGES = 1000
IMG_SIZE = 224

# Load labels
print("Loading labels...")
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✓ Loaded {len(labels)} classes")

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✓ Model loaded successfully")

# Collect all images from dataset
print(f"\nScanning dataset directory: {DATASET_DIR}")
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

print(f"✓ Found {len(all_images)} total images in dataset")

# Randomly sample images (with replacement to allow duplicates)
print(f"\nRandomly sampling {NUM_TEST_IMAGES} images...")
test_images = random.choices(all_images, k=NUM_TEST_IMAGES)
print(f"✓ Selected {NUM_TEST_IMAGES} images for testing")

# Count distribution
true_label_dist = Counter([img['true_label'] for img in test_images])
print(f"\nSample distribution across classes:")
for label, count in sorted(true_label_dist.items()):
    print(f"  {label}: {count} images")

# Test images
print("\n" + "="*70)
print("TESTING IN PROGRESS...")
print("="*70)

results = []
correct = 0
total = 0
start_time = time.time()

# Progress tracking
print_interval = 50  # Print progress every 50 images

for idx, img_data in enumerate(test_images, 1):
    try:
        # Load and preprocess image
        img = image.load_img(img_data['path'], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        
        # Predict
        predictions = model.predict(img_array, verbose=0)
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        predicted_label = labels[predicted_idx]
        
        # Check if correct
        true_label = img_data['true_label']
        is_correct = (predicted_label == true_label)
        
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'filename': img_data['filename'],
            'true_label': true_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'correct': is_correct
        })
        
        # Progress update
        if idx % print_interval == 0:
            elapsed = time.time() - start_time
            images_per_sec = idx / elapsed
            eta_seconds = (NUM_TEST_IMAGES - idx) / images_per_sec
            current_acc = (correct / total) * 100
            print(f"[{idx}/{NUM_TEST_IMAGES}] Accuracy: {current_acc:.2f}% | "
                  f"Speed: {images_per_sec:.1f} img/s | ETA: {eta_seconds:.0f}s")
    
    except Exception as e:
        print(f"  Error processing {img_data['filename']}: {e}")
        continue

total_time = time.time() - start_time

# Calculate metrics
print("\n" + "="*70)
print("RESULTS")
print("="*70)

accuracy = (correct / total) * 100
print(f"\nTotal images tested: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {total - correct}")
print(f"Overall Accuracy: {accuracy:.2f}%")
print(f"\nTotal time: {total_time:.1f} seconds")
print(f"Average time per image: {total_time/total*1000:.1f} ms")
print(f"Images per second: {total/total_time:.1f}")

# Per-class accuracy
print(f"\n{'='*70}")
print("PER-CLASS ACCURACY")
print(f"{'='*70}")

class_stats = {}
for r in results:
    label = r['true_label']
    if label not in class_stats:
        class_stats[label] = {'correct': 0, 'total': 0}
    class_stats[label]['total'] += 1
    if r['correct']:
        class_stats[label]['correct'] += 1

print(f"\n{'Class':<15} {'Tested':<10} {'Correct':<10} {'Accuracy':<10}")
print("-" * 50)
for label in sorted(class_stats.keys()):
    stats = class_stats[label]
    class_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    print(f"{label:<15} {stats['total']:<10} {stats['correct']:<10} {class_acc:>6.2f}%")

# Confidence distribution
confidences = [r['confidence'] for r in results]
avg_confidence = np.mean(confidences)
correct_confidences = [r['confidence'] for r in results if r['correct']]
wrong_confidences = [r['confidence'] for r in results if not r['correct']]

print(f"\n{'='*70}")
print("CONFIDENCE ANALYSIS")
print(f"{'='*70}")

print(f"\nOverall:")
print(f"  Average confidence: {avg_confidence:.2%}")
print(f"  Min confidence: {min(confidences):.2%}")
print(f"  Max confidence: {max(confidences):.2%}")

if correct_confidences:
    print(f"\nCorrect predictions:")
    print(f"  Average confidence: {np.mean(correct_confidences):.2%}")
    print(f"  Min confidence: {min(correct_confidences):.2%}")

if wrong_confidences:
    print(f"\nWrong predictions:")
    print(f"  Average confidence: {np.mean(wrong_confidences):.2%}")
    print(f"  Max confidence: {max(wrong_confidences):.2%}")

# Confidence buckets
high_conf = sum(1 for c in confidences if c >= 0.9)
med_conf = sum(1 for c in confidences if 0.7 <= c < 0.9)
low_conf = sum(1 for c in confidences if c < 0.7)

print(f"\nConfidence Distribution:")
print(f"  High (≥90%): {high_conf} images ({high_conf/len(results)*100:.1f}%)")
print(f"  Medium (70-90%): {med_conf} images ({med_conf/len(results)*100:.1f}%)")
print(f"  Low (<70%): {low_conf} images ({low_conf/len(results)*100:.1f}%)")

# Most confused classes
print(f"\n{'='*70}")
print("CONFUSION ANALYSIS")
print(f"{'='*70}")

wrong_predictions = [r for r in results if not r['correct']]
if wrong_predictions:
    confusion_pairs = Counter([(r['true_label'], r['predicted']) for r in wrong_predictions])
    print(f"\nMost common misclassifications:")
    for (true_label, pred_label), count in confusion_pairs.most_common(10):
        print(f"  {true_label} → {pred_label}: {count} times")
else:
    print("\n✓ No misclassifications! Perfect accuracy!")

# Save detailed results
results_file = 'test_results/1000_images_test_results.txt'
os.makedirs('test_results', exist_ok=True)

print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

with open(results_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("1000 RANDOM IMAGES TEST RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Total images tested: {total}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"Test duration: {total_time:.1f} seconds\n\n")
    
    f.write("="*70 + "\n")
    f.write("PER-CLASS ACCURACY\n")
    f.write("="*70 + "\n")
    for label in sorted(class_stats.keys()):
        stats = class_stats[label]
        class_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        f.write(f"{label}: {stats['correct']}/{stats['total']} = {class_acc:.2f}%\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("WRONG PREDICTIONS\n")
    f.write("="*70 + "\n")
    for r in results:
        if not r['correct']:
            f.write(f"\n{r['filename']}\n")
            f.write(f"  True: {r['true_label']}\n")
            f.write(f"  Predicted: {r['predicted']} ({r['confidence']:.2%})\n")

print(f"✓ Detailed results saved to: {results_file}")

# Save CSV for analysis
csv_file = 'test_results/1000_images_test_results.csv'
with open(csv_file, 'w', encoding='utf-8') as f:
    f.write("filename,true_label,predicted_label,confidence,correct\n")
    for r in results:
        f.write(f"{r['filename']},{r['true_label']},{r['predicted']},{r['confidence']:.4f},{r['correct']}\n")

print(f"✓ CSV results saved to: {csv_file}")

print("\n" + "="*70)
print("TESTING COMPLETE!")
print("="*70)
print(f"\n🎯 Final Accuracy: {accuracy:.2f}%")
print(f"⏱️  Total Time: {total_time:.1f} seconds")
print(f"🚀 Speed: {total/total_time:.1f} images/second")
