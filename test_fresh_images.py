"""
Test trained model with fresh images
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, 'hand_landmark_v2')

print("="*70)
print("Testing Trained Model with Fresh Images")
print("="*70)

# Load model
print("\nLoading trained model...")
try:
    from inference import HandLandmarkInference
    
    detector = HandLandmarkInference(
        'hand_landmark_v2/checkpoints/best_model.pth',
        backend='pytorch',
        use_kalman=False,  # No filtering for static images
        use_gpu=False
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Find test images
test_dirs = ['Test', 'dataset', 'datasets/freihand/evaluation/rgb']
test_images = []

for test_dir in test_dirs:
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_dir, file))
                if len(test_images) >= 10:  # Test with 10 images
                    break
    if len(test_images) >= 10:
        break

if not test_images:
    print("\n✗ No test images found")
    print("Please add some hand images to the 'Test' folder")
    sys.exit(1)

print(f"\n✓ Found {len(test_images)} test images")

# Create output directory
output_dir = 'test_results'
os.makedirs(output_dir, exist_ok=True)

# Test each image
print("\nTesting images...")
print("-"*70)

successful = 0
failed = 0
total_fps = []

for i, img_path in enumerate(test_images, 1):
    print(f"\n{i}. Testing: {os.path.basename(img_path)}")
    
    try:
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"   ✗ Could not load image")
            failed += 1
            continue
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        landmarks, confidence, fps = detector.predict(rgb_image)
        
        if landmarks is not None:
            # Draw landmarks
            result = detector.draw_landmarks(
                image.copy(), landmarks, confidence,
                draw_connections=True, dotted=False
            )
            
            # Get bounding box
            bbox = detector.extract_hand_bbox(landmarks, padding=60)
            h, w = image.shape[:2]
            x_min = int(bbox[0] * w)
            y_min = int(bbox[1] * h)
            x_max = int(bbox[2] * w)
            y_max = int(bbox[3] * h)
            
            # Draw bounding box
            cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
            
            # Add info
            avg_conf = np.mean(confidence)
            cv2.putText(result, f"Confidence: {avg_conf*100:.1f}%", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save result
            output_path = os.path.join(output_dir, f'result_{i}_{os.path.basename(img_path)}')
            cv2.imwrite(output_path, result)
            
            print(f"   ✓ Hand detected")
            print(f"   ✓ Confidence: {avg_conf*100:.1f}%")
            print(f"   ✓ FPS: {fps:.1f}")
            print(f"   ✓ Saved to: {output_path}")
            
            successful += 1
            total_fps.append(fps)
        else:
            print(f"   ✗ No hand detected")
            failed += 1
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        failed += 1

# Summary
print("\n" + "="*70)
print("Test Summary")
print("="*70)
print(f"\nTotal images tested: {len(test_images)}")
print(f"Successful detections: {successful}")
print(f"Failed detections: {failed}")
print(f"Success rate: {(successful/len(test_images)*100):.1f}%")

if total_fps:
    print(f"\nPerformance:")
    print(f"  Average FPS: {np.mean(total_fps):.1f}")
    print(f"  Min FPS: {np.min(total_fps):.1f}")
    print(f"  Max FPS: {np.max(total_fps):.1f}")

print(f"\nResults saved to: {output_dir}/")

print("\n" + "="*70)
print("Next Steps:")
print("="*70)
print("\n1. Check results in 'test_results' folder")
print("\n2. Test with webcam:")
print("   python classify_webcam_v2.py")
print("\n3. Test sign language recognition:")
print("   python classify_v2.py Test/IMG-20251111-WA0011.jpg --save-viz")
print("="*70)
