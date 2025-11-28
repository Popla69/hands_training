"""
Quick test to verify HACKATHON script can load model
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("Testing model loading...")

# Load labels
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]
print(f"✓ Labels loaded: {len(label_lines)} classes")

# Load graph
with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("✓ Graph loaded successfully")

# Test session
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    print("✓ Session created successfully")
    print(f"✓ Softmax tensor shape: {softmax_tensor.shape}")

print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
print("The HACKATHON script should work!")
print("\nIf camera fails, check:")
print("  1. Camera is connected")
print("  2. No other app is using camera")
print("  3. Try camera index 0 or 1")
