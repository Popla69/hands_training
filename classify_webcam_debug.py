"""
DEBUG VERSION - Shows what the classifier sees
This will help diagnose why it keeps predicting 'W'
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
        print("✓ Using custom hand landmark model")
    else:
        print("⚠ Custom model not found, falling back to MediaPipe")
except ImportError:
    print("⚠ hand_landmark_v2 not available, falling back to MediaPipe")

if not USE_CUSTOM_MODEL:
    import mediapipe as mp


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign language letter from image"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    # Get top 5 predictions
    results = []
    for node_id in top_k[:5]:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        results.append((human_string, score))
    
    return results


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
    print("Sign Language Recognition - DEBUG MODE")
    print("="*70)
    print(f"\nHand Detection: {'Custom Model' if USE_CUSTOM_MODEL else 'MediaPipe'}")
    print("Sign Classifier: InceptionV3")
    print("\nThis version shows TOP 5 predictions to help debug")
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
        print("✓ Custom hand detector loaded")
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
        print("✓ MediaPipe initialized")
    
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
    print("  - SPACE: Save current hand image for inspection")
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
        top_predictions = []
        i = 0
        
        # FPS tracking
        fps_queue = deque(maxlen=30)
        prev_time = time.time()
        
        # For saving debug images
        save_counter = 0
        
        print("\n✓ Camera started successfully!")
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
            processed_hand = None
            
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
                
                # Predict sign every 3 frames
                if i >= 2:
                    try:
                        # Preprocess for sign classifier
                        hand_resized = cv2.resize(hand_img, (299, 299))
                        
                        # Apply histogram equalization
                        yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
                        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                        processed_hand = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                        
                        # Encode and predict
                        image_data = cv2.imencode('.jpg', processed_hand, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                        top_predictions = predict_sign(image_data, sess, softmax_tensor, label_lines)
                        
                        i = 0
                    except Exception as e:
                        if i % 30 == 0:
                            print(f"Prediction error: {e}")
                
                i += 1
            
            # Create display with 3 panels
            display = np.zeros((h, w + 600, 3), dtype=np.uint8)
            display[:h, :w] = frame
            
            # Right panel - show processed hand and predictions
            panel_x = w
            
            # Show original cropped hand
            if hand_img is not None:
                try:
                    hand_preview = cv2.resize(hand_img, (250, 250))
                    display[20:270, panel_x+20:panel_x+270] = hand_preview
                    cv2.putText(display, "Cropped Hand", (panel_x+30, 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except:
                    pass
            
            # Show processed hand (what classifier sees)
            if processed_hand is not None:
                try:
                    processed_preview = cv2.resize(processed_hand, (250, 250))
                    display[290:540, panel_x+20:panel_x+270] = processed_preview
                    cv2.putText(display, "After Processing", (panel_x+30, 285), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                except:
                    pass
            
            # Show top 5 predictions
            cv2.putText(display, "TOP 5 PREDICTIONS:", (panel_x+30, 570), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            y_offset = 610
            for idx, (label, score) in enumerate(top_predictions[:5]):
                color = (0, 255, 0) if idx == 0 else (200, 200, 200)
                text = f"{idx+1}. {label.upper()}: {score*100:.1f}%"
                cv2.putText(display, text, (panel_x+30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 35
            
            # Status overlay on main frame
            overlay = display[:200, :500].copy()
            cv2.rectangle(overlay, (0, 0), (500, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display[:200, :500], 0.3, 0, display[:200, :500])
            
            # Status
            if hand_detected:
                status_text = "Hand Detected"
                status_color = (0, 255, 0)
            else:
                status_text = "No Hand - Show your hand"
                status_color = (0, 0, 255)
            
            cv2.putText(display, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            # FPS
            cv2.putText(display, f"FPS: {avg_fps:.1f}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Model info
            model_name = "Custom V2" if USE_CUSTOM_MODEL else "MediaPipe"
            cv2.putText(display, f"Detection: {model_name}", (10, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Instructions
            cv2.putText(display, "SPACE: Save image | ESC: Exit", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow('Sign Language Recognition - DEBUG', display)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            # Exit on ESC
            if key == 27:
                break
            
            # Save on SPACE
            if key == 32 and processed_hand is not None:
                filename = f"debug_hand_{save_counter}.jpg"
                cv2.imwrite(filename, processed_hand)
                print(f"Saved: {filename}")
                if top_predictions:
                    print(f"  Top prediction: {top_predictions[0][0]} ({top_predictions[0][1]*100:.1f}%)")
                save_counter += 1
        
        cap.release()
        cv2.destroyAllWindows()
        if not USE_CUSTOM_MODEL:
            hand_detector.close()
    
    print("\nDebug session ended.")


if __name__ == "__main__":
    main()
