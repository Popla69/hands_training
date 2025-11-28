"""
Test version WITHOUT ML - just camera feed
"""

import cv2
import numpy as np
import time

print("="*70)
print("Camera Test (No ML)")
print("="*70)
print("\nStarting camera...")

# Try camera with DirectShow backend
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera 1 failed, trying camera 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    input("Press ENTER to exit...")
    exit(1)

# Optimize camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("✓ Camera opened!")

# Create windows
cv2.namedWindow('Camera Test', cv2.WINDOW_NORMAL)

# Test read
ret, test_frame = cap.read()
if not ret:
    print("ERROR: Cannot read from camera!")
    cap.release()
    input("Press ENTER to exit...")
    exit(1)

print(f"✓ Camera working! Frame size: {test_frame.shape}")
print("\nShowing camera feed...")
print("Press ESC to exit")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Draw a green box in the center
    box_size = 400
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    # Add text
    cv2.putText(frame, "Camera is working!", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    
    # FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    cv2.putText(frame, f"FPS: {fps:.1f}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.putText(frame, "Press ESC to exit", (50, h - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    cv2.imshow('Camera Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTest complete! Processed {frame_count} frames")
print(f"Average FPS: {fps:.1f}")
