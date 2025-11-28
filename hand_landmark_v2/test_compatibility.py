"""
Test compatibility between all dependencies
"""

import sys
import os

print("="*70)
print("Dependency Compatibility Test")
print("="*70)

# Test 1: PyTorch
print("\n1. Testing PyTorch...")
try:
    import torch
    import torchvision
    print(f"   ✓ PyTorch {torch.__version__}")
    print(f"   ✓ TorchVision {torchvision.__version__}")
    print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"   ✗ PyTorch import failed: {e}")

# Test 2: TensorFlow
print("\n2. Testing TensorFlow...")
try:
    import tensorflow as tf
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_v2_behavior()
    print(f"   ✓ TensorFlow {tf.__version__}")
    print(f"   ✓ TF v1 compatibility mode enabled")
except ImportError as e:
    print(f"   ✗ TensorFlow import failed: {e}")

# Test 3: OpenCV
print("\n3. Testing OpenCV...")
try:
    import cv2
    print(f"   ✓ OpenCV {cv2.__version__}")
    
    # Test GUI support
    test_img = cv2.imread('README.md')  # Will fail but tests import
    print(f"   ✓ OpenCV imread available")
    
    # Test VideoCapture
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print(f"   ✓ VideoCapture available")
        cap.release()
    else:
        print(f"   ⚠ VideoCapture not available (no camera?)")
except ImportError as e:
    print(f"   ✗ OpenCV import failed: {e}")
except Exception as e:
    print(f"   ⚠ OpenCV partial functionality: {e}")

# Test 4: NumPy
print("\n4. Testing NumPy...")
try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"   ✗ NumPy import failed: {e}")

# Test 5: MediaPipe
print("\n5. Testing MediaPipe...")
try:
    import mediapipe as mp
    print(f"   ✓ MediaPipe {mp.__version__}")
except ImportError as e:
    print(f"   ⚠ MediaPipe not available: {e}")
    print(f"      (Optional - install with: pip install mediapipe)")

# Test 6: Protobuf
print("\n6. Testing Protobuf...")
try:
    import google.protobuf
    print(f"   ✓ Protobuf {google.protobuf.__version__}")
except ImportError as e:
    print(f"   ✗ Protobuf import failed: {e}")

# Test 7: SciPy
print("\n7. Testing SciPy...")
try:
    import scipy
    print(f"   ✓ SciPy {scipy.__version__}")
except ImportError as e:
    print(f"   ⚠ SciPy not available: {e}")

# Test 8: TensorBoard
print("\n8. Testing TensorBoard...")
try:
    from torch.utils.tensorboard import SummaryWriter
    print(f"   ✓ TensorBoard available")
except ImportError as e:
    print(f"   ⚠ TensorBoard not available: {e}")

# Test 9: ONNX Runtime (optional)
print("\n9. Testing ONNX Runtime (optional)...")
try:
    import onnxruntime as ort
    print(f"   ✓ ONNX Runtime {ort.__version__}")
    print(f"   ✓ Available providers: {ort.get_available_providers()}")
except ImportError as e:
    print(f"   ⚠ ONNX Runtime not available: {e}")
    print(f"      (Optional - install with: pip install onnxruntime)")

# Test 10: Coexistence test
print("\n10. Testing PyTorch + TensorFlow coexistence...")
try:
    import torch
    import tensorflow as tf
    
    # Create simple tensors
    torch_tensor = torch.randn(2, 3)
    tf_tensor = tf.random.normal([2, 3])
    
    print(f"   ✓ PyTorch tensor created: {torch_tensor.shape}")
    print(f"   ✓ TensorFlow tensor created: {tf_tensor.shape}")
    print(f"   ✓ Both frameworks coexist successfully")
except Exception as e:
    print(f"   ✗ Coexistence test failed: {e}")

# Test 11: Hand landmark model
print("\n11. Testing hand landmark model...")
try:
    sys.path.insert(0, 'hand_landmark_v2')
    from model import create_model
    
    model = create_model(pretrained=False)
    print(f"   ✓ Hand landmark model created")
    
    # Test forward pass
    import torch
    dummy_input = torch.randn(1, 3, 224, 224)
    landmarks, confidence = model(dummy_input)
    print(f"   ✓ Forward pass successful")
    print(f"      Landmarks: {landmarks.shape}")
    print(f"      Confidence: {confidence.shape}")
except Exception as e:
    print(f"   ✗ Hand landmark model test failed: {e}")

# Test 12: Sign language classifier
print("\n12. Testing sign language classifier...")
try:
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_v2_behavior()
    
    if os.path.exists("logs/trained_graph.pb"):
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print(f"   ✓ Sign language classifier loaded")
    else:
        print(f"   ⚠ Sign language classifier not found")
        print(f"      (Expected at: logs/trained_graph.pb)")
except Exception as e:
    print(f"   ✗ Sign language classifier test failed: {e}")

# Summary
print("\n" + "="*70)
print("Compatibility Test Summary")
print("="*70)

required_packages = [
    'torch', 'torchvision', 'tensorflow', 'cv2', 'numpy'
]

optional_packages = [
    'mediapipe', 'onnxruntime', 'scipy'
]

print("\nRequired packages:")
all_required_ok = True
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} - MISSING")
        all_required_ok = False

print("\nOptional packages:")
for pkg in optional_packages:
    try:
        __import__(pkg)
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  - {pkg} - not installed")

print("\n" + "="*70)
if all_required_ok:
    print("✓ All required dependencies are compatible!")
    print("\nYou can now:")
    print("  1. Train the model: python hand_landmark_v2/train.py")
    print("  2. Run webcam demo: python classify_webcam_v2.py")
    print("  3. Classify images: python classify_v2.py <image_path>")
else:
    print("✗ Some required dependencies are missing")
    print("\nInstall missing dependencies:")
    print("  pip install torch torchvision opencv-python numpy tensorflow scipy")

print("="*70)
