"""
Quick test to see which hand detection actually works
"""
import cv2
import numpy as np
import time

print("Testing hand detection methods...")
print("="*70)

# Test 1: MediaPipe
print("\n1. Testing MediaPipe...")
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.3,  # Lower threshold
        min_tracking_confidence=0.3
    )
    print("✓ MediaPipe loaded successfully")
    MEDIAPIPE_OK = True
except Exception as e:
    print(f"✗ MediaPipe failed: {e}")
    MEDIAPIPE_OK = False

# Test 2: Custom model
print("\n2. Testing Custom Hand Landmark Model...")
try:
    import sys
    sys.path.insert(0, 'hand_landmark_v2')
    from hand_landmark_v2.inference import HandLandmarkInference
    
    detector = HandLandmarkInference(
        'hand_landmark_v2/checkpoints/best_model.pth',
        backend='pytorch',
        use_kalman=False,  # Disable for speed
        use_gpu=False
    )
    print("✓ Custom model loaded successfully")
    CUSTOM_OK = True
except Exception as e:
    print(f"✗ Custom model failed: {e}")
    CUSTOM_OK = False

if not MEDIAPIPE_OK and not CUSTOM_OK:
    print("\n✗ NO HAND DETECTION AVAILABLE!")
    exit(1)

print("\n" + "="*70)
print("Starting camera test...")
print("Controls: ESC to exit, 1=MediaPipe, 2=Custom Model")
print("="*70)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Cannot open camera!")
    exit(1)

print("✓ Camera opened")

mode = 1 if MEDIAPIPE_OK else 2
frame_count = 0
detection_count = 0
fps_list = []
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Calculate FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    fps_list.append(fps)
    if len(fps_list) > 30:
        fps_list.pop(0)
    avg_fps = sum(fps_list) / len(fps_list)
    prev_time = current_time
    
    frame_count += 1
    hand_detected = False
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect based on mode
    if mode == 1 and MEDIAPIPE_OK:
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_detected = True
            detection_count += 1
            
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Draw bounding box
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords)) - 20
                x_max = int(max(x_coords)) + 20
                y_min = int(min(y_coords)) - 20
                y_max = int(max(y_coords)) + 20
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        method_name = "MediaPipe"
        method_color = (0, 255, 255)
    
    elif mode == 2 and CUSTOM_OK:
        try:
            landmarks, confidence, _ = detector.predict(rgb_frame)
            
            if landmarks is not None:
                hand_detected = True
                detection_count += 1
                
                # Draw landmarks
                frame = detector.draw_landmarks(frame, landmarks, confidence, 
                                               draw_connections=True, dotted=False)
                
                # Draw bounding box
                landmarks_px = landmarks.copy()
                landmarks_px[:, 0] *= w
                landmarks_px[:, 1] *= h
                
                x_min = int(np.min(landmarks_px[:, 0])) - 20
                x_max = int(np.max(landmarks_px[:, 0])) + 20
                y_min = int(np.min(landmarks_px[:, 1])) - 20
                y_max = int(np.max(landmarks_px[:, 1])) + 20
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        except Exception as e:
            pass
        
        method_name = "Custom Model"
        method_color = (255, 0, 255)
    
    # Draw info overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Status
    if hand_detected:
        status = "HAND DETECTED"
        status_color = (0, 255, 0)
    else:
        status = "NO HAND"
        status_color = (0, 0, 255)
    
    cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
    
    # Method
    cv2.putText(frame, f"Method: {method_name}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, method_color, 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Detection rate
    detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
    cv2.putText(frame, f"Detection: {detection_rate:.1f}%", (10, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 1=MediaPipe, 2=Custom, ESC=Exit", (10, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    cv2.imshow('Hand Detection Test', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord('1') and MEDIAPIPE_OK:
        mode = 1
        print("Switched to MediaPipe")
    elif key == ord('2') and CUSTOM_OK:
        mode = 2
        print("Switched to Custom Model")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("Test Results:")
print(f"Total frames: {frame_count}")
print(f"Detections: {detection_count}")
print(f"Detection rate: {detection_rate:.1f}%")
print(f"Average FPS: {avg_fps:.1f}")
print("="*70)
