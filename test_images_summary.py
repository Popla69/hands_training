import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import sys
import os
import glob

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Loads label file
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

# Load graph
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def predict_image(image_path, sess, softmax_tensor):
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:3]:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        results.append((human_string, score))
    
    return results

# Get all images
test_images = glob.glob("Test/*.jpg")
test_images.sort()

print(f"\n{'='*100}")
print(f"TESTING {len(test_images)} IMAGES FROM TEST FOLDER")
print(f"{'='*100}\n")

results_summary = []

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    for idx, image_path in enumerate(test_images, 1):
        filename = os.path.basename(image_path)
        results = predict_image(image_path, sess, softmax_tensor)
        
        top_pred = results[0][0].upper()
        confidence = results[0][1] * 100
        
        results_summary.append({
            'filename': filename,
            'prediction': top_pred,
            'confidence': confidence,
            'top3': results
        })
        
        # Print inline progress
        status = "✓" if confidence > 70 else "?"
        print(f"{status} [{idx:2d}/{len(test_images)}] {filename:30s} → {top_pred:10s} ({confidence:5.1f}%)")

print(f"\n{'='*100}")
print("SUMMARY")
print(f"{'='*100}\n")

# Group by prediction
from collections import defaultdict
by_prediction = defaultdict(list)
for result in results_summary:
    by_prediction[result['prediction']].append(result)

print(f"Total images tested: {len(test_images)}")
print(f"Unique predictions: {len(by_prediction)}")
print(f"\nPredictions breakdown:")
for pred in sorted(by_prediction.keys()):
    count = len(by_prediction[pred])
    avg_conf = sum(r['confidence'] for r in by_prediction[pred]) / count
    print(f"  {pred:10s}: {count:2d} images (avg confidence: {avg_conf:.1f}%)")

# High confidence predictions
high_conf = [r for r in results_summary if r['confidence'] >= 90]
print(f"\nHigh confidence predictions (≥90%): {len(high_conf)}")

# Low confidence predictions
low_conf = [r for r in results_summary if r['confidence'] < 70]
if low_conf:
    print(f"\nLow confidence predictions (<70%):")
    for r in low_conf:
        print(f"  {r['filename']:30s} → {r['prediction']:10s} ({r['confidence']:.1f}%)")
        print(f"    Top 3: {', '.join([f'{p[0].upper()}({p[1]*100:.1f}%)' for p in r['top3']])}")

print(f"\n{'='*100}\n")
