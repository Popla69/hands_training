"""
Test ULTIMATE system with test images
"""

import os
import cv2
import numpy as np
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("="*70)
print("Testing ULTIMATE System with Images")
print("="*70)

# Load model
print("\nLoading model...")
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print(f"✓ Model loaded ({len(label_lines)} classes)")

sess = tf.Session()
softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

# Find test images
test_dirs = ['Test', 'dataset', 'data']
test_images = []

for test_dir in test_dirs:
    if os.path.exists(test_dir):
        patterns = [
            os.path.join(test_dir, '**', '*.jpg'),
            os.path.join(test_dir, '**', '*.jpeg'),
            os.path.join(test_dir, '**', '*.png'),
            os.path.join(test_dir, '*.jpg'),
            os.path.join(test_dir, '*.jpeg'),
            os.path.join(test_dir, '*.png'),
        ]
        for pattern in patterns:
            test_images.extend(glob.glob(pattern, recursive=True))

if not test_images:
    print("\n✗ No test images found!")
    print("Looking in: Test/, dataset/, data/")
    sess.close()
    exit(1)

print(f"\n✓ Found {len(test_images)} test images")
print("\nTesting (press any key for next image, ESC to exit)...\n")

correct = 0
total = 0

for img_path in test_images[:50]:  # Test first 50 images
    # Get expected label from path
    expected_label = None
    path_parts = img_path.replace('\\', '/').split('/')
    for part in path_parts:
        if part.lower() in [l.lower() for l in label_lines]:
            expected_label = part.lower()
            break
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        continue
    
    # Resize to 299x299
    img_resized = cv2.resize(img, (299, 299))
    
    # Encode as JPEG
    image_data = cv2.imencode('.jpg', img_resized, 
                             [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    
    # Predict
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-5:][::-1]
    
    # Get top 5
    results = []
    for node_id in top_k:
        label = label_lines[node_id]
        score = predictions[0][node_id]
        results.append((label, score))
    
    top_label, top_score = results[0]
    
    # Check if correct
    is_correct = (expected_label and top_label.lower() == expected_label.lower())
    if is_correct:
        correct += 1
    total += 1
    
    # Display
    display_img = cv2.resize(img, (600, 600))
    
    # Add predictions
    y_off = 40
    for i, (label, score) in enumerate(results):
        color = (0, 255, 0) if i == 0 else (150, 150, 150)
        if i == 0 and is_correct:
            color = (0, 255, 0)
        elif i == 0:
            color = (0, 0, 255)
        
        text = f"{i+1}. {label.upper()}: {score*100:.1f}%"
        cv2.putText(display_img, text, (10, y_off), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_off += 35
    
    # Expected label
    if expected_label:
        status = "CORRECT" if is_correct else "WRONG"
        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
        cv2.putText(display_img, f"Expected: {expected_label.upper()}", 
                   (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_img, status, (10, 590), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    # Accuracy
    accuracy = (correct / total * 100) if total > 0 else 0
    cv2.putText(display_img, f"Accuracy: {accuracy:.1f}% ({correct}/{total})", 
               (10, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
    
    cv2.imshow('Testing ULTIMATE System', display_img)
    
    print(f"[{total}] {os.path.basename(img_path)}: {top_label.upper()} "
          f"({top_score*100:.1f}%) - {'✓' if is_correct else '✗'}")
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
sess.close()

print("\n" + "="*70)
print("Test Results")
print("="*70)
print(f"Total tested: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {(correct/total*100):.1f}%")
print("="*70)
