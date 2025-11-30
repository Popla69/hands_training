"""
Test trained model on 10,000 random images from dataset
Large-scale comprehensive accuracy test
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
print("LARGE-SCALE MODEL TEST - 10,000 RANDOM IMAGES")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("This will take approximately 20-25 minutes...")
print()

# Configuration
MODEL_PATH = 'models_tf2/checkpoint_resume.h5'
LABELS_PATH = 'models_tf2/labels.txt'
DATASET_DIR = 'dataset'
NUM_TEST_IMAGES = 10000
IMG_SIZE = 224
BATCH_SIZE = 32  # Process in batches for speed

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

# Randomly sample images
print(f"\nRandomly sampling {NUM_TEST_IMAGES} images...")
random.seed(42)  # For reproducibility
test_images = random.choices(all_images, k=NUM_TEST_IMAGES)
print(f"✓ Selected {NUM_TEST_IMAGES} images for testing")

# Count distribution
true_label_dist = Counter([img['true_label'] for img in test_images])
print(f"\nSample distribution across {len(true_label_dist)} classes")
print(f"Average per class: {NUM_TEST_IMAGES / len(true_label_dist):.0f} images")

# Test images with batch processing
print("\n" + "="*70)
print("TESTING IN PROGRESS...")
print("="*70)
print("Progress updates every 500 images\n")

results = []
correct = 0
total = 0
start_time = time.time()

# Process in batches for efficiency
for batch_start in range(0, len(test_images), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(test_images))
    batch = test_images[batch_start:batch_end]
    
    # Load batch images
    batch_images = []
    batch_labels = []
    batch_filenames = []
    
    for img_data in batch:
        try:
            img = image.load_img(img_data['path'], target_size=(IMG_SIZE, IMG_SIZE))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            batch_images.append(img_array)
            batch_labels.append(img_data['true_label'])
            batch_filenames.append(img_data['filename'])
        except Exception as e:
            print(f"  Error loading {img_data['filename']}: {e}")
            continue
    
    if not batch_images:
        continue
    
    # Predict batch
    batch_array = np.array(batch_images)
    predictions = model.predict(batch_array, verbose=0)
    
    # Process results
    for i, pred in enumerate(predictions):
        predicted_idx = np.argmax(pred)
        confidence = pred[predicted_idx]
        predicted_label = labels[predicted_idx]
        true_label = batch_labels[i]
        
        is_correct = (predicted_label == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        results.append({
            'filename': batch_filenames[i],
            'true_label': true_label,
            'predicted': predicted_label,
            'confidence': confidence,
            'correct': is_correct
        })
    
    # Progress update every 500 images
    if total % 500 == 0:
        elapsed = time.time() - start_time
        images_per_sec = total / elapsed
        eta_seconds = (NUM_TEST_IMAGES - total) / images_per_sec
        current_acc = (correct / total) * 100
        
        print(f"[{total:5d}/{NUM_TEST_IMAGES}] "
              f"Accuracy: {current_acc:6.2f}% | "
              f"Speed: {images_per_sec:5.1f} img/s | "
              f"ETA: {eta_seconds/60:4.1f} min")

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
print(f"\nTotal time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
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
        class_stats[label] = {'correct': 0, 'total': 0, 'confidences': []}
    class_stats[label]['total'] += 1
    class_stats[label]['confidences'].append(r['confidence'])
    if r['correct']:
        class_stats[label]['correct'] += 1

print(f"\n{'Class':<15} {'Tested':<10} {'Correct':<10} {'Accuracy':<12} {'Avg Conf':<10}")
print("-" * 65)
for label in sorted(class_stats.keys()):
    stats = class_stats[label]
    class_acc = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
    avg_conf = np.mean(stats['confidences']) * 100
    print(f"{label:<15} {stats['total']:<10} {stats['correct']:<10} "
          f"{class_acc:>6.2f}% {'':5} {avg_conf:>6.2f}%")

# Summary statistics
print(f"\n{'='*70}")
print("SUMMARY STATISTICS")
print(f"{'='*70}")

# Accuracy tiers
perfect_classes = [l for l, s in class_stats.items() if s['correct'] == s['total']]
high_acc_classes = [l for l, s in class_stats.items() if 95 <= (s['correct']/s['total']*100) < 100]
good_acc_classes = [l for l, s in class_stats.items() if 90 <= (s['correct']/s['total']*100) < 95]
low_acc_classes = [l for l, s in class_stats.items() if (s['correct']/s['total']*100) < 90]

print(f"\nAccuracy Distribution:")
print(f"  Perfect (100%): {len(perfect_classes)} classes")
print(f"  High (95-99%): {len(high_acc_classes)} classes")
print(f"  Good (90-94%): {len(good_acc_classes)} classes")
print(f"  Needs work (<90%): {len(low_acc_classes)} classes")

if low_acc_classes:
    print(f"\n  Classes needing attention:")
    for label in low_acc_classes:
        stats = class_stats[label]
        acc = (stats['correct'] / stats['total']) * 100
        print(f"    {label}: {acc:.2f}%")

# Confidence analysis
confidences = [r['confidence'] for r in results]
correct_confidences = [r['confidence'] for r in results if r['correct']]
wrong_confidences = [r['confidence'] for r in results if not r['correct']]

print(f"\n{'='*70}")
print("CONFIDENCE ANALYSIS")
print(f"{'='*70}")

print(f"\nOverall:")
print(f"  Average confidence: {np.mean(confidences):.2%}")
print(f"  Median confidence: {np.median(confidences):.2%}")
print(f"  Std deviation: {np.std(confidences):.2%}")

if correct_confidences:
    print(f"\nCorrect predictions ({len(correct_confidences)}):")
    print(f"  Average confidence: {np.mean(correct_confidences):.2%}")
    print(f"  Median confidence: {np.median(correct_confidences):.2%}")

if wrong_confidences:
    print(f"\nWrong predictions ({len(wrong_confidences)}):")
    print(f"  Average confidence: {np.mean(wrong_confidences):.2%}")
    print(f"  Median confidence: {np.median(wrong_confidences):.2%}")

# Confidence buckets
very_high = sum(1 for c in confidences if c >= 0.95)
high_conf = sum(1 for c in confidences if 0.90 <= c < 0.95)
med_conf = sum(1 for c in confidences if 0.70 <= c < 0.90)
low_conf = sum(1 for c in confidences if c < 0.70)

print(f"\nConfidence Distribution:")
print(f"  Very High (≥95%): {very_high:5d} ({very_high/len(results)*100:5.1f}%)")
print(f"  High (90-95%):    {high_conf:5d} ({high_conf/len(results)*100:5.1f}%)")
print(f"  Medium (70-90%):  {med_conf:5d} ({med_conf/len(results)*100:5.1f}%)")
print(f"  Low (<70%):       {low_conf:5d} ({low_conf/len(results)*100:5.1f}%)")

# Confusion analysis
print(f"\n{'='*70}")
print("CONFUSION ANALYSIS")
print(f"{'='*70}")

wrong_predictions = [r for r in results if not r['correct']]
if wrong_predictions:
    confusion_pairs = Counter([(r['true_label'], r['predicted']) for r in wrong_predictions])
    print(f"\nTop 20 most common misclassifications:")
    for i, ((true_label, pred_label), count) in enumerate(confusion_pairs.most_common(20), 1):
        pct = count / len(wrong_predictions) * 100
        print(f"  {i:2d}. {true_label:>10} → {pred_label:<10}: {count:4d} times ({pct:4.1f}%)")
else:
    print("\n✓ No misclassifications! Perfect accuracy!")

# Save detailed results
results_dir = 'test_results'
os.makedirs(results_dir, exist_ok=True)

print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

# Save summary
summary_file = f'{results_dir}/10000_images_test_summary.txt'
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("10,000 RANDOM IMAGES TEST RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Total images tested: {total}\n")
    f.write(f"Accuracy: {accuracy:.2f}%\n")
    f.write(f"Test duration: {total_time/60:.1f} minutes\n\n")
    
    f.write("="*70 + "\n")
    f.write("PER-CLASS ACCURACY\n")
    f.write("="*70 + "\n")
    for label in sorted(class_stats.keys()):
        stats = class_stats[label]
        class_acc = (stats['correct'] / stats['total']) * 100
        avg_conf = np.mean(stats['confidences']) * 100
        f.write(f"{label}: {stats['correct']}/{stats['total']} = {class_acc:.2f}% "
                f"(avg conf: {avg_conf:.1f}%)\n")
    
    if wrong_predictions:
        f.write("\n" + "="*70 + "\n")
        f.write("TOP CONFUSIONS\n")
        f.write("="*70 + "\n")
        for (true_label, pred_label), count in confusion_pairs.most_common(20):
            f.write(f"{true_label} → {pred_label}: {count} times\n")

print(f"✓ Summary saved to: {summary_file}")

# Save CSV
csv_file = f'{results_dir}/10000_images_test_results.csv'
with open(csv_file, 'w', encoding='utf-8') as f:
    f.write("filename,true_label,predicted_label,confidence,correct\n")
    for r in results:
        f.write(f"{r['filename']},{r['true_label']},{r['predicted']},"
                f"{r['confidence']:.4f},{r['correct']}\n")

print(f"✓ CSV results saved to: {csv_file}")

# Save wrong predictions only
if wrong_predictions:
    errors_file = f'{results_dir}/10000_images_errors_only.txt'
    with open(errors_file, 'w', encoding='utf-8') as f:
        f.write(f"WRONG PREDICTIONS ({len(wrong_predictions)} total)\n")
        f.write("="*70 + "\n\n")
        for r in wrong_predictions:
            f.write(f"{r['filename']}\n")
            f.write(f"  True: {r['true_label']}\n")
            f.write(f"  Predicted: {r['predicted']} ({r['confidence']:.2%})\n\n")
    print(f"✓ Errors saved to: {errors_file}")

print("\n" + "="*70)
print("TESTING COMPLETE!")
print("="*70)
print(f"\n🎯 Final Accuracy: {accuracy:.2f}%")
print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
print(f"🚀 Speed: {total/total_time:.1f} images/second")
print(f"📊 Error Rate: {(total-correct)/total*100:.2f}%")
print(f"✅ Correct: {correct:,} / {total:,}")
