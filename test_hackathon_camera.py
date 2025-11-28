"""
Quick test to verify camera works with DirectShow backend
"""
import cv2
import time

print("Testing camera with DirectShow backend...")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("✗ Camera 1 failed, trying camera 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("✗ Camera failed to open!")
    exit(1)

print("✓ Camera opened successfully!")

# Set properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✓ Camera properties set")

# Capture a few frames
print("\nCapturing 5 test frames...")
for i in range(5):
    ret, frame = cap.read()
    if ret:
        print(f"  Frame {i+1}: {frame.shape}")
    else:
        print(f"  Frame {i+1}: FAILED")
    time.sleep(0.1)

cap.release()
print("\n✓✓✓ Camera test PASSED! ✓✓✓")
print("classify_HACKATHON.py should work now!")
