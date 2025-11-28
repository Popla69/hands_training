"""Test production script startup"""

import sys
import os

print("Starting test...")
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")
    sys.exit(1)

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")
    sys.exit(1)

try:
    import tensorflow.compat.v1 as tf_v1
    print(f"✓ TensorFlow: {tf_v1.__version__}")
except Exception as e:
    print(f"✗ TensorFlow: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print(f"✓ MediaPipe: {mp.__version__}")
except Exception as e:
    print(f"✗ MediaPipe: {e}")
    sys.exit(1)

print("\nAll imports successful!")
print("\nNow testing model loading...")

try:
    label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
    print(f"✓ Labels loaded: {len(label_lines)} classes")
    print(f"  Classes: {', '.join(label_lines)}")
except Exception as e:
    print(f"✗ Labels: {e}")
    sys.exit(1)

try:
    with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf_v1.import_graph_def(graph_def, name='')
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Model: {e}")
    sys.exit(1)

print("\n✓ Everything loaded successfully!")
print("\nThe script should work. Try running classify_webcam_production.py")
