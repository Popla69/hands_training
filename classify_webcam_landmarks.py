"""
Enhanced Sign Language Recognition with Hand Landmark Detection
Uses MediaPipe for robust hand tracking with 21 landmarks per hand
Integrates with trained InceptionV3 sign language classifier
"""

import sys
import os
import cv2
import numpy as np
from collections import deque

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import TensorFlow first to avoid protobuf conflicts
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

def predict(image_data, sess, softmax_tensor):
    """Predict sign language letter from image"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

class KalmanFilter:
    """Simple Kalman filter for landmark smoothing"""
    def __init__(self):
        self.x = None  # State
        self.P = None  # Covariance
        self.Q = 0.001  # Process noise
        self.R = 0.01   # Measurement noise
        
    def update(self, measurement):
        if self.x is None:
            self.x = measurement
            self.P = 1.0
            return measurement
        
        # Prediction
        x_pred = self.x
        P_pred = self.P + self.Q
        
        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred
        
        return self.x

class HandLandmarkTracker:
    """Track and smooth hand landmarks"""
    def __init__(self, num_landmarks=21):
        self.filters_x = [KalmanFilter() for _ in range(num_landmarks)]
        self.filters_y = [KalmanFilter() for _ in range(num_landmarks)]
        self.filters_z = [KalmanFilter() for _ in range(num_landmarks)]
        
    def update(self, landmarks):
        """Update with new landmarks and return smoothed version"""
        smoothed = []
        for i, lm in enumerate(landmarks):
            x = self.filters_x[i].update(lm.x)
            y = self.filters_y[i].update(lm.y)
            z = self.filters_z[i].update(lm.z)
            smoothed.append((x, y, z))
        return smoothed

def extract_hand_region_from_landmarks(image, landmarks, padding=50):
    """Extract hand region to match training data format - tight crop with minimal padding"""
    h, w, _ = image.shape
    
    # Get bounding box from landmarks
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    
    # Tight bounding box with minimal padding
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    # Make square - training data is square
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - size // 2)
    x_max = min(w, center_x + size // 2)
    y_min = max(0, center_y - size // 2)
    y_max = min(h, center_y + size // 2)
    
    # Extract region
    hand_region = image[y_min:y_max, x_min:x_max].copy()
    
    # Ensure we have a valid region
    if hand_region.shape[0] < 50 or hand_region.shape[1] < 50:
        return None, None
    
    return hand_region, (x_min, y_min, x_max, y_max)

def draw_landmarks_detailed(image, landmarks, connections, bbox=None):
    """Draw hand landmarks with detailed visualization"""
    h, w, _ = image.shape
    
    # Draw bounding box
    if bbox:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    # Draw connections
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        start_x = int(start_point.x * w)
        start_y = int(start_point.y * h)
        end_x = int(end_point.x * w)
        end_y = int(end_point.y * h)
        
        cv2.line(image, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Draw landmarks
    for idx, landmark in enumerate(landmarks):
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        
        # Color code: thumb=red, index=blue, middle=green, ring=yellow, pinky=magenta
        if idx in [1, 2, 3, 4]:  # Thumb
            color = (0, 0, 255)
        elif idx in [5, 6, 7, 8]:  # Index
            color = (255, 0, 0)
        elif idx in [9, 10, 11, 12]:  # Middle
            color = (0, 255, 0)
        elif idx in [13, 14, 15, 16]:  # Ring
            color = (0, 255, 255)
        elif idx in [17, 18, 19, 20]:  # Pinky
            color = (255, 0, 255)
        else:  # Wrist
            color = (255, 255, 255)
        
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.circle(image, (x, y), 6, (255, 255, 255), 1)

# Load model
print("Loading sign language model...")
label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]

with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf_v1.import_graph_def(graph_def, name='')

print("\nStarting Enhanced Sign Language Recognition...")
print("=" * 70)
print("Features:")
print("  ✓ 21-point hand landmark tracking per hand")
print("  ✓ Kalman filtering for smooth tracking")
print("  ✓ Automatic hand detection and cropping")
print("  ✓ Color-coded finger visualization")
print("  ✓ Real-time sign language recognition")
print("\nInstructions:")
print("  - Show your hand(s) to the camera")
print("  - System tracks all 21 landmarks automatically")
print("  - Hold a sign for 2-3 seconds to add to sequence")
print("  - Press ESC to exit")
print("=" * 70)

# Try to import MediaPipe
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    print("\n✓ MediaPipe loaded successfully - using advanced hand tracking")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("\n✗ MediaPipe not available - falling back to basic detection")
    print("  Install with: pip install mediapipe")

if not MEDIAPIPE_AVAILABLE:
    print("\nCannot run landmark-based detection without MediaPipe.")
    print("Please install: pip install mediapipe")
    sys.exit(1)

with tf_v1.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    
    # Initialize hand tracker
    tracker = HandLandmarkTracker()
    
    # FPS calculation
    fps_queue = deque(maxlen=30)
    import time
    prev_time = time.time()
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            fps_queue.append(fps)
            avg_fps = sum(fps_queue) / len(fps_queue)
            prev_time = curr_time
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hand_detected = False
            num_hands = 0
            
            if results.multi_hand_landmarks:
                hand_detected = True
                num_hands = len(results.multi_hand_landmarks)
                
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand label (Left/Right)
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    
                    # Smooth landmarks
                    smoothed_landmarks = tracker.update(hand_landmarks.landmark)
                    
                    # Draw detailed landmarks
                    draw_landmarks_detailed(
                        frame, 
                        hand_landmarks.landmark,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Extract hand region for classification - tight crop like training data
                    hand_img, bbox = extract_hand_region_from_landmarks(
                        frame, hand_landmarks.landmark, padding=50
                    )
                    
                    if hand_img is not None and hand_idx == 0:  # Only classify first hand
                        # Draw bbox
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                    (255, 255, 0), 3)
                        
                        # Show extracted hand in debug window
                        debug_hand = cv2.resize(hand_img, (299, 299))
                        cv2.imshow('Extracted Hand (299x299 - sent to classifier)', debug_hand)
                        
                        # Predict every 5 frames
                        if i == 4:
                            try:
                                # Resize to model input size
                                hand_resized = cv2.resize(hand_img, (299, 299))
                                
                                # Create a clean background (like training data)
                                # Option 1: White background
                                clean_img = np.ones((299, 299, 3), dtype=np.uint8) * 240
                                
                                # Option 2: Or use the hand with background blur
                                # Blur the background heavily
                                blurred = cv2.GaussianBlur(hand_resized, (51, 51), 0)
                                
                                # Create mask from hand landmarks
                                mask = np.zeros((299, 299), dtype=np.uint8)
                                # Scale landmarks to 299x299
                                scale_x = 299 / (bbox[2] - bbox[0])
                                scale_y = 299 / (bbox[3] - bbox[1])
                                
                                # Draw filled polygon around hand
                                pts = []
                                for lm in hand_landmarks.landmark:
                                    x = int((lm.x * w - bbox[0]) * scale_x)
                                    y = int((lm.y * h - bbox[1]) * scale_y)
                                    pts.append([x, y])
                                
                                if len(pts) > 0:
                                    hull = cv2.convexHull(np.array(pts))
                                    cv2.fillConvexPoly(mask, hull, 255)
                                    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
                                    mask = cv2.GaussianBlur(mask, (15, 15), 0)
                                    
                                    # Blend: sharp hand on blurred background
                                    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
                                    hand_resized = (hand_resized * mask_3ch + blurred * (1 - mask_3ch)).astype(np.uint8)
                                
                                # Apply histogram equalization
                                yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
                                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                                hand_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                                
                                # Encode as JPEG
                                image_data = cv2.imencode('.jpg', hand_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                                res_tmp, score = predict(image_data, sess, softmax_tensor)
                                res = res_tmp
                                i = 0
                                
                                # Sequence logic
                                if mem == res:
                                    consecutive += 1
                                else:
                                    consecutive = 0
                                
                                if consecutive == 2 and res not in ['nothing']:
                                    if res == 'space':
                                        sequence += ' '
                                    elif res == 'del':
                                        sequence = sequence[:-1]
                                    else:
                                        sequence += res
                                    consecutive = 0
                                
                                mem = res
                            except Exception as e:
                                print(f"Prediction error: {e}")
                        
                        i += 1
                    
                    # Display hand info
                    wrist = hand_landmarks.landmark[0]
                    wrist_x = int(wrist.x * w)
                    wrist_y = int(wrist.y * h)
                    cv2.putText(frame, f"{handedness}", (wrist_x - 30, wrist_y - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (400, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Status
            status_text = f"Hands: {num_hands}" if hand_detected else "No hands detected"
            status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Prediction
            if res:
                cv2.putText(frame, f"Sign: {res.upper()}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Conf: {score*100:.1f}%", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Legend
            cv2.putText(frame, "Thumb", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "Index", (80, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(frame, "Middle", (150, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "Ring", (230, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, "Pinky", (290, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            cv2.imshow('Sign Language - Landmark Tracking', frame)
            
            # Sequence window
            img_sequence = np.zeros((400, 1200, 3), np.uint8)
            cv2.putText(img_sequence, "Recognized Sequence:", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            # Word wrapping
            max_chars_per_line = 50
            lines = []
            current_line = ""
            
            for char in sequence.upper():
                if len(current_line) >= max_chars_per_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            y_offset = 80
            for line in lines[-8:]:
                cv2.putText(img_sequence, line, (30, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 45
            
            cv2.imshow('Sequence', img_sequence)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

print("\nSession ended. Final sequence:")
print(sequence.upper())
