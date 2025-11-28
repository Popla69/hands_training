"""
Console-only test of ULTIMATE system
"""

import os
import cv2
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("="*70)
print("Testing ULTIMATE System - Console Output")
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

# Get test images
test_images = glob.glob('Test/*.jpg')
test_images = [img for img in test_images if 'result' not in img]

print(f"\n✓ Found {len(test_images)} test images")
print("\n" + "="*70)
print("Testing...")
print("="*70 + "\n")

for i, img_path in enumerate(test_images):
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
    
    # Print results
    print(f"[{i+1}/{len(test_images)}] {os.path.basename(img_path)}")
    for j, (label, score) in enumerate(results):
        marker = "✓" if j == 0 else " "
        print(f"  {marker} {j+1}. {label.upper()}: {score*100:.1f}%")
    print()

sess.close()

print("="*70)
print("Test Complete!")
print("="*70)
