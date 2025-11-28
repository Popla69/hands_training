"""
Sign Language Recognition with Hand Landmark Detection V2
Integrates new hand detection model with existing sign classifier
"""

import sys
import os
import cv2
import numpy as np
from collections import deque
import time

# Add hand_landmark_v2 to path
sys.path.insert(0, 'hand_landmark_v2')

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Import TensorFlow for sign language classifier
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

# Try to import hand landmark model (fallback to MediaPipe if not available)
USE_CUSTOM_MODEL = False
try:
    from hand_landmark_v2.inference import HandLandmarkInference
    if os.path.exists('hand_landmark_v2/checkpoints/best_model.pth'):
        USE_CUSTOM_MODEL = True
        print("[OK] Using custom hand landmark model")
    else:
        print("[WARNING] Custom model not found, falling back to MediaPipe")
except ImportError:
    print("[WARNING] hand_landmark_v2 not available, falling back to MediaPipe")

if not USE_CUSTOM_MODEL:
    import mediapipe as mp


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


def extract_hand_region_custom(image, landmarks, padding=60):
    """Extract hand region using custom model landmarks"""
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


def extract_hand_region_mediapipe(image, hand_landmarks, padding=60):
    """Extract hand region using MediaPipe landmarks"""
    h, w = image.shape[:2]
    
    # Get all landmark coordinates
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    # Get bounding box
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
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


def main():
    """Main function"""
    
    print("="*70)
    print("Sign Language Recognition V2")
    print("="*70)
    print(f"\nHand Detection: {'Custom Model' if USE_CUSTOM_MODEL else 'MediaPipe'}")
    print("Sign Classifier: InceptionV3")
    print("\nLoading models...")
    
    # Initialize hand detector
    if USE_CUSTOM_MODEL:
        hand_detector = HandLandmarkInference(
            'hand_landmark_v2/checkpoints/best_model.pth',
            backend='pytorch',
            use_kalman=True,
            filter_type='one_euro',
            use_gpu=False
        )
        print("[OK] Custom hand detector loaded")
    else:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        hand_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("[OK] MediaPipe initialized")
    
    # Load sign language classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("[OK] Sign language classifier loaded")
    except Exception as e:
        print(f"[ERROR] Error loading sign language classifier: {e}")
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
        
        print("\n[OK] Camera started successfully!")
        print("Show your hand to the camera...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_queue.append(fps)
            avg_fps = np.mean(fps_queue)
            prev_time = current_time
            
            hand_detected = False
            hand_img = None
            bbox = None
            
            # Detect hand
            if USE_CUSTOM_MODEL:
                try:
                    landmarks, confidence, hand_fps = hand_detector.predict(rgb_frame)
                    hand_detected = landmarks is not None
                    
                    if hand_detected:
                        # Draw landmarks
                        frame = hand_detector.draw_landmarks(frame, landmarks, confidence, 
                                                            draw_connections=True, dotted=False)
                        
                        # Extract hand region
                        hand_img, bbox = extract_hand_region_custom(frame, landmarks, padding=60)
                except Exception as e:
                    if i % 30 == 0:
                        print(f"Hand detection error: {e}")
            else:
                # MediaPipe detection
                results = hand_detector.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    hand_detected = True
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Extract hand region
                        hand_img, bbox = extract_hand_region_mediapipe(frame, hand_landmarks, padding=60)
            
            # Predict sign if hand detected
            if hand_detected and hand_img is not None:
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
                
                # Predict sign every 3 frames
                # NOTE: TensorFlow prediction disabled - causes freeze with TF 2.10 + old model
                # Need to retrain model with TensorFlow 2.10 for predictions to work
                if i >= 2 and False:
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
                        if i % 30 == 0:
                            print(f"Prediction error: {e}")
                
                i += 1
            
            # Display info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 200), (0, 0, 0), -1)
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
            model_name = "Custom V2" if USE_CUSTOM_MODEL else "MediaPipe"
            cv2.putText(frame, f"Detection: {model_name}", (10, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Prediction
            if res:
                cv2.putText(frame, f"Sign: {res.upper()}", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Confidence: {score*100:.1f}%", (10, 165), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "Prediction: DISABLED", (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                cv2.putText(frame, "Train new model to enable", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            
            cv2.imshow('Sign Language Recognition V2', frame)
            
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
        if not USE_CUSTOM_MODEL:
            hand_detector.close()
    
    print("\nSession ended. Final sequence:")
    print(sequence.upper())


if __name__ == "__main__":
    main()
