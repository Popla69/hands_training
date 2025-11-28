"""
Simple test of ULTIMATE system with Test folder images
"""

import os
import cv2
import numpy as np
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("="*70)
print("Testing ULTIMATE System with Test Folder")
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

# Get test images from Test folder only
test_images = glob.glob('Test/*.jpg')
test_images = [img for img in test_images if 'result' not in img]

print(f"\n✓ Found {len(test_images)} test images in Test folder")
print("\nTesting (press any key for next, ESC to exit)...\n")

for i, img_path in enumerate(test_images):
    print(f"[{i+1}/{len(test_images)}] Testing: {os.path.basename(img_path)}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"  ✗ Failed to load image")
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
    
    print(f"  Top prediction: {top_label.upper()} ({top_score*100:.1f}%)")
    
    # Display
    display_img = img.copy()
    h, w = display_img.shape[:2]
    
    # Resize for display if too large
    if w > 800:
        scale = 800 / w
        display_img = cv2.resize(display_img, None, fx=scale, fy=scale)
        h, w = display_img.shape[:2]
    
    # Add black bar at bottom for text
    display_img = cv2.copyMakeBorder(display_img, 0, 250, 0, 0, 
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Add predictions
    y_off = h + 40
    for j, (label, score) in enumerate(results):
        color = (0, 255, 0) if j == 0 else (150, 150, 150)
        text = f"{j+1}. {label.upper()}: {score*100:.1f}%"
        cv2.putText(display_img, text, (10, y_off), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_off += 35
    
    # Image name
    cv2.putText(display_img, os.path.basename(img_path), (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Testing ULTIMATE System', display_img)
    
    key = cv2.waitKey(0) & 0xFF
    if key == 27:  # ESC
        print("\nTest stopped by user")
        break

cv2.destroyAllWindows()
sess.close()

print("\n" + "="*70)
print("Test Complete")
print("="*70)
print(f"Tested {min(i+1, len(test_images))} images")
print("="*70)
