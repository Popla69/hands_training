"""
Integrated Sign Language Recognition with Custom Hand Landmark Model
Uses the trained lightweight MobileNetV3 hand landmark detector
"""

import sys
import os
import cv2
import numpy as np
from collections import deque
import time

# Add hand landmark model to path
sys.path.insert(0, 'hand_landmark_model')

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import TensorFlow for sign language classifier
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

# Import hand landmark model
from inference import HandLandmarkInference
from config import HAND_CONNECTIONS


def predict_sign(image_data, sess, softmax_tensor, label_lines):
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


def extract_hand_region_from_landmarks(image, landmarks, padding=60):
    """Extract hand region based on landmarks"""
    h, w = image.shape[:2]
    
    # Denormalize landmarks
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= w
    landmarks_px[:, 1] *= h
    
    # Get bounding box
    x_coords = landmarks_px[:, 0]
    y_coords = landmarks_px[:, 1]
    
    x_min = max(0, int(np.min(x_coords)) - padding)
    x_max = min(w, int(np.max(x_coords)) + padding)
    y_min = max(0, int(np.min(y_coords)) - padding)
    y_max = min(h, int(np.max(y_coords)) + padding)
    
    # Make square
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
    hand_region = image[y_min:y_max, x_min:x_max]
    
    if hand_region.size == 0:
        return None, None
    
    return hand_region, (x_min, y_min, x_max, y_max)


def draw_landmarks_custom(image, landmarks, confidence=None):
    """Draw hand landmarks with custom styling"""
    h, w = image.shape[:2]
    img_draw = image.copy()
    
    # Denormalize landmarks
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= w
    landmarks_px[:, 1] *= h
    
    # Draw connections
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = tuple(landmarks_px[start_idx, :2].astype(int))
        end_point = tuple(landmarks_px[end_idx, :2].astype(int))
        cv2.line(img_draw, start_point, end_point, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Draw landmarks
    for i, (x, y, z) in enumerate(landmarks_px):
        x, y = int(x), int(y)
        
        # Color based on finger
        if i in [1, 2, 3, 4]:  # Thumb
            color = (0, 0, 255)
        elif i in [5, 6, 7, 8]:  # Index
            color = (255, 0, 0)
        elif i in [9, 10, 11, 12]:  # Middle
            color = (0, 255, 0)
        elif i in [13, 14, 15, 16]:  # Ring
            color = (0, 255, 255)
        elif i in [17, 18, 19, 20]:  # Pinky
            color = (255, 0, 255)
        else:  # Wrist
            color = (255, 255, 255)
        
        # Draw point
        cv2.circle(img_draw, (x, y), 4, color, -1)
        cv2.circle(img_draw, (x, y), 5, (255, 255, 255), 1)
    
    return img_draw


def main():
    """Main function"""
    
    print("="*70)
    print("Integrated Sign Language Recognition System")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Custom lightweight hand landmark detection (5.65 MB)")
    print("  ✓ MobileNetV3 backbone with Kalman filtering")
    print("  ✓ 21-point hand tracking")
    print("  ✓ Real-time sign language recognition")
    print("  ✓ Automatic hand detection and cropping")
    print("\nLoading models...")
    
    # Load hand landmark model
    try:
        hand_detector = HandLandmarkInference(
            'hand_landmark_model/models/hand_landmark.pth',
            backend='pytorch',
            use_kalman=True,
            use_gpu=False
        )
        print("✓ Hand landmark model loaded")
    except Exception as e:
        print(f"✗ Error loading hand landmark model: {e}")
        print("  Make sure the model is exported: python export_model.py")
        return
    
    # Load sign language classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Sign language classifier loaded")
    except Exception as e:
        print(f"✗ Error loading sign language classifier: {e}")
        return
    
    print("\n" + "="*70)
    print("Ready to start!")
    print("="*70)
    
    # Ask user before starting camera
    response = input("\nStart camera? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\nStarting camera...")
    print("\nControls:")
    print("  - ESC: Exit")
    print("  - Hold sign for 2-3 seconds to add to sequence")
    print("="*70)
    
    # Start inference
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # State variables
        res, score = '', 0.0
        i = 0
        mem = ''
        consecutive = 0
        sequence = ''
        
        # FPS tracking
        fps_queue = deque(maxlen=30)
        prev_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert to RGB for hand detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            try:
                landmarks, confidence, hand_fps = hand_detector.predict(rgb_frame)
                hand_detected = landmarks is not None and len(landmarks) > 0
            except Exception as e:
                hand_detected = False
                landmarks = None
                if i % 30 == 0:  # Print error every 30 frames
                    print(f"Hand detection error: {e}")
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            prev_time = current_time
            
            if hand_detected and landmarks is not None:
                # Draw landmarks
                frame = draw_landmarks_custom(frame, landmarks, confidence)
                
                # Extract hand region
                hand_img, bbox = extract_hand_region_from_landmarks(frame, landmarks, padding=60)
                
                if hand_img is not None:
                    # Draw bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                                (0, 255, 255), 3)
                    
                    # Show cropped hand in corner
                    try:
                        hand_preview = cv2.resize(hand_img, (150, 150))
                        frame[h-160:h-10, w-160:w-10] = hand_preview
                        cv2.rectangle(frame, (w-160, h-160), (w-10, h-10), (0, 255, 255), 2)
                        cv2.putText(frame, "Classifier Input", (w-155, h-170), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    except:
                        pass
                    
                    # Predict sign every 3 frames for better responsiveness
                    if i >= 2:
                        try:
                            # Preprocess for sign classifier
                            hand_resized = cv2.resize(hand_img, (299, 299))
                            
                            # Apply histogram equalization
                            yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
                            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                            hand_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                            
                            # Encode and predict
                            image_data = cv2.imencode('.jpg', hand_resized, 
                                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                            res_tmp, score = predict_sign(image_data, sess, softmax_tensor, label_lines)
                            
                            # Only update if confidence is reasonable
                            if score > 0.3:
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
            
            # Display info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 220), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Status
            if hand_detected:
                status_text = "Hand Detected"
                status_color = (0, 255, 0)
            else:
                status_text = "No Hand - Show your hand"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Model info
            cv2.putText(frame, "Model: MobileNetV3 (5.65MB)", (10, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Prediction
            if res:
                cv2.putText(frame, f"Sign: {res.upper()}", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Confidence: {score*100:.1f}%", (10, 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Legend
            cv2.putText(frame, "Thumb", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(frame, "Index", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(frame, "Middle", (130, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, "Ring", (200, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, "Pinky", (250, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            cv2.imshow('Integrated Sign Language Recognition', frame)
            
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
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nSession ended. Final sequence:")
    print(sequence.upper())


if __name__ == "__main__":
    main()
