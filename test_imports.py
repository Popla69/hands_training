"""Test imports for classify_webcam_production.py"""

print("Testing imports...")

try:
    import sys
    print("✓ sys")
except Exception as e:
    print(f"✗ sys: {e}")

try:
    import os
    print("✓ os")
except Exception as e:
    print(f"✗ os: {e}")

try:
    import cv2
    print(f"✓ cv2 (version: {cv2.__version__})")
except Exception as e:
    print(f"✗ cv2: {e}")

try:
    import numpy as np
    print(f"✓ numpy (version: {np.__version__})")
except Exception as e:
    print(f"✗ numpy: {e}")

try:
    import time
    print("✓ time")
except Exception as e:
    print(f"✗ time: {e}")

try:
    from collections import deque, Counter
    print("✓ collections")
except Exception as e:
    print(f"✗ collections: {e}")

try:
    import tensorflow.compat.v1 as tf_v1
    print(f"✓ tensorflow (version: {tf_v1.__version__})")
except Exception as e:
    print(f"✗ tensorflow: {e}")

try:
    import mediapipe as mp
    print(f"✓ mediapipe (version: {mp.__version__})")
except Exception as e:
    print(f"✗ mediapipe: {e}")

print("\nAll imports tested!")
