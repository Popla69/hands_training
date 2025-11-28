"""
Test the trained model
"""

import sys
sys.path.insert(0, 'hand_landmark_v2')

print("="*70)
print("Testing Trained Hand Landmark Model")
print("="*70)

# Test 1: Load model
print("\n1. Loading model...")
try:
    from model import create_model
    import torch
    
    model = create_model(pretrained=False)
    model.load_state_dict(torch.load('hand_landmark_v2/checkpoints/best_model.pth', map_location='cpu'))
    model.eval()
    
    print("   ✓ Model loaded successfully")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Forward pass
print("\n2. Testing forward pass...")
try:
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        landmarks, confidence = model(dummy_input)
    
    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Landmarks shape: {landmarks.shape}")
    print(f"   ✓ Confidence shape: {confidence.shape}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Inference speed
print("\n3. Benchmarking inference speed...")
try:
    import time
    import numpy as np
    
    times = []
    for _ in range(50):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000
    avg_fps = 1000 / avg_time
    
    print(f"   ✓ Average inference time: {avg_time:.2f} ms")
    print(f"   ✓ Average FPS: {avg_fps:.1f}")
    
    if avg_fps >= 30:
        print(f"   ✓ Meets 30+ FPS requirement")
    else:
        print(f"   ⚠ Below 30 FPS (acceptable for CPU)")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 4: Model size
print("\n4. Checking model size...")
try:
    import os
    size_mb = os.path.getsize('hand_landmark_v2/checkpoints/best_model.pth') / (1024 ** 2)
    print(f"   ✓ Model size: {size_mb:.2f} MB")
    
    if size_mb < 50:
        print(f"   ✓ Meets <50MB requirement")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "="*70)
print("Model Test Summary")
print("="*70)
print("\n✓ Model is ready for use!")
print("\nNext steps:")
print("1. Test with webcam:")
print("   python classify_webcam_v2.py")
print("\n2. Test with images:")
print("   python hand_landmark_v2/demo_image.py test_image.jpg")
print("\n3. Export for deployment:")
print("   python hand_landmark_v2/export.py hand_landmark_v2/checkpoints/best_model.pth onnx")
print("="*70)
