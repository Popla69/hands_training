"""
Test what model predicts for problem letters K, R, S, V, Y
"""

import os
import sys
import cv2
import numpy as np
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("="*70)
print("Testing Problem Letters: K, R, S, V, Y")
print("="*70)

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("✓ Model loaded\n")

PROBLEM_LETTERS = ['K', 'R', 'S', 'V', 'Y']

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    for letter in PROBLEM_LETTERS:
        print(f"\n{'='*70}")
        print(f"Testing letter: {letter}")
        print(f"{'='*70}")
        
        # Get 20 random images from dataset
        dataset_path = f"dataset/{letter}"
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset not found: {dataset_path}")
            continue
        
        all_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(all_files) == 0:
            print(f"✗ No images found")
            continue
        
        print(f"Found {len(all_files)} images")
        
        # Test 20 random images
        import random
        test_files = random.sample(all_files, min(20, len(all_files)))
        
        predictions = []
        confidences = []
        
        for img_file in test_files:
            img_path = os.path.join(dataset_path, img_file)
            
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize to 299x299
            img_resized = cv2.resize(img, (299, 299))
            
            # Encode as JPEG
            image_data = cv2.imencode('.jpg', img_resized, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            
            # Predict
            preds = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            top_k = preds[0].argsort()[-len(preds[0]):][::-1]
            
            # Get top prediction
            top_label = label_lines[top_k[0]]
            top_conf = preds[0][top_k[0]]
            
            predictions.append(top_label)
            confidences.append(top_conf)
        
        # Analyze results
        counter = Counter(predictions)
        most_common = counter.most_common(5)
        
        print(f"\nResults for {letter}:")
        print(f"  Tested: {len(predictions)} images")
        print(f"  Average confidence: {np.mean(confidences)*100:.1f}%")
        print(f"\n  Top predictions:")
        
        correct_count = 0
        for pred_label, count in most_common:
            percentage = (count / len(predictions)) * 100
            is_correct = pred_label.upper() == letter.upper()
            if is_correct:
                correct_count = count
            marker = "✓" if is_correct else "✗"
            print(f"    {marker} {pred_label.upper()}: {count}/{len(predictions)} ({percentage:.1f}%)")
        
        accuracy = (correct_count / len(predictions)) * 100
        print(f"\n  Accuracy: {accuracy:.1f}%")
        
        if accuracy < 50:
            print(f"  ⚠️  PROBLEM: Model confuses {letter} with {most_common[0][0].upper()}")

print("\n" + "="*70)
print("Test Complete")
print("="*70)
