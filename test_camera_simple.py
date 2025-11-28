"""Simple camera test"""
import cv2
import time

print("Testing camera...")
print("Trying camera 1 with DirectShow...")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera 1 failed, trying camera 0...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    input("Press ENTER to exit...")
    exit(1)

print("✓ Camera opened successfully!")
print("Reading frames...")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {i}")
        break
    print(f"Frame {i}: {frame.shape}")
    
    # Show the frame
    cv2.imshow('Camera Test', frame)
    cv2.waitKey(100)

print("\nNow showing live feed for 10 seconds...")
print("Press ESC to exit early")

start_time = time.time()
while time.time() - start_time < 10:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    
    cv2.putText(frame, "Camera is working!", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera Test', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Test complete!")
