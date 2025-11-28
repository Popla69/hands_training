"""
Test all images in Test folder and show accuracy metrics only
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, 'hand_landmark_v2')

print("="*70)
print("Hand Landmark Detection - Accuracy Test")
print("="*70)

# Load model
print("\nLoading model...")
try:
    from inference import HandLandmarkInference
    
    detector = HandLandmarkInference(
        'hand_landmark_v2/checkpoints/best_model.pth',
        backend='pytorch',
        use_kalman=False,
        use_gpu=False
    )
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Find all test images
test_dir = 'Test'
if not os.path.exists(test_dir):
    print(f"\n✗ Test folder not found: {test_dir}")
    sys.exit(1)

test_images = []
for file in os.listdir(test_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.endswith('_result.jpg'):
        test_images.append(os.path.join(test_dir, file))

test_images.sort()

print(f"\n✓ Found {len(test_images)} test images")
print("\n" + "="*70)
print("Testing...")
print("="*70)

# Test metrics
total_images = len(test_images)
hands_detected = 0
hands_not_detected = 0
confidences = []
fps_values = []

# Test each image
for i, img_path in enumerate(test_images, 1):
    try:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"{i:2d}. {os.path.basename(img_path):30s} - ✗ Load failed")
            hands_not_detected += 1
            continue
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        landmarks, confidence, fps = detector.predict(rgb_image)
        
        if landmarks is not None:
            avg_conf = np.mean(confidence)
            confidences.append(avg_conf)
            fps_values.append(fps)
            hands_detected += 1
            
            # Show result
            status = "✓"
            conf_str = f"{avg_conf*100:5.1f}%"
            fps_str = f"{fps:5.1f} FPS"
            print(f"{i:2d}. {os.path.basename(img_path):30s} - {status} Detected | Conf: {conf_str} | {fps_str}")
        else:
            hands_not_detected += 1
            print(f"{i:2d}. {os.path.basename(img_path):30s} - ✗ No hand detected")
            
    except Exception as e:
        hands_not_detected += 1
        print(f"{i:2d}. {os.path.basename(img_path):30s} - ✗ Error: {str(e)[:30]}")

# Calculate metrics
detection_rate = (hands_detected / total_images * 100) if total_images > 0 else 0
avg_confidence = np.mean(confidences) * 100 if confidences else 0
min_confidence = np.min(confidences) * 100 if confidences else 0
max_confidence = np.max(confidences) * 100 if confidences else 0
avg_fps = np.mean(fps_values) if fps_values else 0
min_fps = np.min(fps_values) if fps_values else 0
max_fps = np.max(fps_values) if fps_values else 0

# Display summary
print("\n" + "="*70)
print("ACCURACY METRICS")
print("="*70)

print(f"\nDetection Performance:")
print(f"  Total images tested:     {total_images}")
print(f"  Hands detected:          {hands_detected}")
print(f"  Hands not detected:      {hands_not_detected}")
print(f"  Detection rate:          {detection_rate:.1f}%")

if confidences:
    print(f"\nConfidence Scores:")
    print(f"  Average confidence:      {avg_confidence:.2f}%")
    print(f"  Minimum confidence:      {min_confidence:.2f}%")
    print(f"  Maximum confidence:      {max_confidence:.2f}%")
    print(f"  Std deviation:           {np.std(confidences)*100:.2f}%")

if fps_values:
    print(f"\nInference Speed:")
    print(f"  Average FPS:             {avg_fps:.1f}")
    print(f"  Minimum FPS:             {min_fps:.1f}")
    print(f"  Maximum FPS:             {max_fps:.1f}")
    print(f"  Average latency:         {1000/avg_fps:.1f} ms")

# Overall assessment
print(f"\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

if detection_rate >= 95:
    grade = "EXCELLENT ⭐⭐⭐⭐⭐"
elif detection_rate >= 90:
    grade = "VERY GOOD ⭐⭐⭐⭐"
elif detection_rate >= 80:
    grade = "GOOD ⭐⭐⭐"
elif detection_rate >= 70:
    grade = "FAIR ⭐⭐"
else:
    grade = "NEEDS IMPROVEMENT ⭐"

print(f"\nDetection Rate: {detection_rate:.1f}% - {grade}")

if avg_confidence >= 95:
    print(f"Confidence: {avg_confidence:.1f}% - EXCELLENT")
elif avg_confidence >= 90:
    print(f"Confidence: {avg_confidence:.1f}% - VERY GOOD")
elif avg_confidence >= 80:
    print(f"Confidence: {avg_confidence:.1f}% - GOOD")
else:
    print(f"Confidence: {avg_confidence:.1f}% - FAIR")

if avg_fps >= 30:
    print(f"Speed: {avg_fps:.1f} FPS - REAL-TIME CAPABLE ✓")
elif avg_fps >= 20:
    print(f"Speed: {avg_fps:.1f} FPS - ACCEPTABLE")
else:
    print(f"Speed: {avg_fps:.1f} FPS - SLOW")

print("\n" + "="*70)
