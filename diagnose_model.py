"""
Diagnose the trained model
Check if it's working properly
"""

import os
import sys
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("="*70)
print("MODEL DIAGNOSTIC")
print("="*70)

# Check files exist
print("\n1. Checking files...")
if not os.path.exists("logs/trained_graph.pb"):
    print("✗ Model file not found: logs/trained_graph.pb")
    sys.exit(1)
print("✓ Model file exists (87.7 MB)")

if not os.path.exists("logs/trained_labels.txt"):
    print("✗ Labels file not found")
    sys.exit(1)
print("✓ Labels file exists")

# Load labels
print("\n2. Loading labels...")
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]
print(f"✓ Loaded {len(label_lines)} classes:")
print(f"   {', '.join(label_lines)}")

# Load model
print("\n3. Loading model...")
try:
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Test with camera
print("\n4. Testing with camera...")
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("✗ Cannot open camera")
        sys.exit(1)
    
    print("✓ Camera opened")
    print("\n5. Testing predictions...")
    print("   Capturing 10 frames and showing predictions...")
    print("   (Put your hand in view)\n")
    
    # Warm up
    for _ in range(5):
        cap.read()
    
    all_predictions = []
    
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Center crop
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        hand_img = frame[y1:y2, x1:x2]
        hand_resized = cv2.resize(hand_img, (299, 299))
        
        image_data = cv2.imencode('.jpg', hand_resized, 
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        top_label = label_lines[top_k[0]]
        top_score = predictions[0][top_k[0]]
        
        all_predictions.append(top_label)
        
        print(f"   Frame {i+1}: {top_label.upper()} ({top_score*100:.1f}%)")
        
        # Show top 3
        for j in range(3):
            label = label_lines[top_k[j]]
            score = predictions[0][top_k[j]]
            print(f"      {j+1}. {label.upper()}: {score*100:.1f}%")
        print()
    
    cap.release()
    
    # Analysis
    print("\n6. Analysis:")
    from collections import Counter
    counter = Counter(all_predictions)
    most_common = counter.most_common(3)
    
    print(f"   Most common predictions:")
    for label, count in most_common:
        print(f"      {label.upper()}: {count}/10 frames ({count*10}%)")
    
    if most_common[0][1] >= 8:
        print(f"\n   ✓ Model is STABLE (predicting {most_common[0][0].upper()} consistently)")
    elif most_common[0][1] >= 5:
        print(f"\n   ⚠ Model is SOMEWHAT STABLE (predicting {most_common[0][0].upper()} {most_common[0][1]}/10 times)")
    else:
        print(f"\n   ✗ Model is UNSTABLE (predictions vary too much)")
        print(f"      This could mean:")
        print(f"      - Hand not in view")
        print(f"      - Poor lighting")
        print(f"      - Model needs retraining")
        print(f"      - Hand position not matching training data")

print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
