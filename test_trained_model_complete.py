"""
Test the trained model against Test folder images
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
from datetime import datetime

print("="*70)
print("TESTING TRAINED MODEL")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
MODEL_PATH = 'models_tf2/checkpoint_resume.h5'  # Best checkpoint
LABELS_PATH = 'models_tf2/labels.txt'
TEST_DIR = 'Test'
IMG_SIZE = 224

# Load labels
print("Loading labels...")
with open(LABELS_PATH, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✓ Loaded {len(labels)} classes: {', '.join(labels[:5])}...")

# Load model
print(f"\nLoading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✓ Model loaded successfully")
print(f"  Input shape: {model.input_shape}")
print(f"  Output shape: {model.output_shape}")

# Get test images
print(f"\nScanning test directory: {TEST_DIR}")
test_images = [f for f in os.listdir(TEST_DIR) 
               if f.lower().endswith(('.jpg', '.jpeg', '.png')) 
               and not f.endswith('_result.jpg')]
test_images.sort()
print(f"✓ Found {len(test_images)} test images")

# Test each image
print("\n" + "="*70)
print("TESTING IMAGES")
print("="*70)

results = []
correct = 0
total = 0

for idx, img_name in enumerate(test_images, 1):
    img_path = os.path.join(TEST_DIR, img_name)
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]
    predicted_label = labels[predicted_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3 = [(labels[i], predictions[0][i]) for i in top_3_idx]
    
    # Try to extract true label from filename (if available)
    true_label = None
    # Common patterns: "A_1.jpg", "letter_A.jpg", etc.
    for label in labels:
        if label.lower() in img_name.lower():
            true_label = label
            break
    
    # Display result
    print(f"\n[{idx}/{len(test_images)}] {img_name}")
    print(f"  Prediction: {predicted_label} ({confidence:.2%})")
    print(f"  Top 3:")
    for i, (lbl, conf) in enumerate(top_3, 1):
        print(f"    {i}. {lbl}: {conf:.2%}")
    
    if true_label:
        is_correct = (predicted_label == true_label)
        if is_correct:
            print(f"  ✓ CORRECT (True label: {true_label})")
            correct += 1
        else:
            print(f"  ✗ WRONG (True label: {true_label})")
        total += 1
    
    results.append({
        'image': img_name,
        'predicted': predicted_label,
        'confidence': confidence,
        'true_label': true_label,
        'top_3': top_3
    })

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTotal images tested: {len(test_images)}")

if total > 0:
    accuracy = (correct / total) * 100
    print(f"Images with known labels: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

# Confidence distribution
confidences = [r['confidence'] for r in results]
avg_confidence = np.mean(confidences)
high_conf = sum(1 for c in confidences if c >= 0.9)
med_conf = sum(1 for c in confidences if 0.7 <= c < 0.9)
low_conf = sum(1 for c in confidences if c < 0.7)

print(f"\nConfidence Distribution:")
print(f"  High (≥90%): {high_conf} images ({high_conf/len(results)*100:.1f}%)")
print(f"  Medium (70-90%): {med_conf} images ({med_conf/len(results)*100:.1f}%)")
print(f"  Low (<70%): {low_conf} images ({low_conf/len(results)*100:.1f}%)")
print(f"  Average confidence: {avg_confidence:.2%}")

# Prediction distribution
from collections import Counter
pred_counts = Counter([r['predicted'] for r in results])
print(f"\nMost predicted classes:")
for label, count in pred_counts.most_common(5):
    print(f"  {label}: {count} times")

# Save results
results_file = 'test_results/trained_model_results.txt'
os.makedirs('test_results', exist_ok=True)

with open(results_file, 'w') as f:
    f.write("="*70 + "\n")
    f.write("TRAINED MODEL TEST RESULTS\n")
    f.write("="*70 + "\n")
    f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Total images: {len(test_images)}\n\n")
    
    for r in results:
        f.write(f"\n{r['image']}\n")
        f.write(f"  Predicted: {r['predicted']} ({r['confidence']:.2%})\n")
        if r['true_label']:
            status = "✓" if r['predicted'] == r['true_label'] else "✗"
            f.write(f"  True label: {r['true_label']} {status}\n")
        f.write(f"  Top 3:\n")
        for i, (lbl, conf) in enumerate(r['top_3'], 1):
            f.write(f"    {i}. {lbl}: {conf:.2%}\n")
    
    f.write(f"\n{'='*70}\n")
    f.write(f"SUMMARY\n")
    f.write(f"{'='*70}\n")
    if total > 0:
        f.write(f"Accuracy: {correct}/{total} = {accuracy:.2f}%\n")
    f.write(f"Average confidence: {avg_confidence:.2%}\n")

print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*70)
print("TESTING COMPLETE!")
print("="*70)
