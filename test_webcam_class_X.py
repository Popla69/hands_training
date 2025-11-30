"""
Real-time webcam test for Class X
Uses VideoCapture(1) for external camera
Shows live predictions and tracks X accuracy
"""

import sys
import os
import cv2
import numpy as np
import time
from collections import deque, Counter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

import mediapipe as mp


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign - returns top 5"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def extract_hand_region(frame, hand_landmarks, padding=60):
    """Extract hand region with padding"""
    h, w = frame.shape[:2]
    
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
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
    
    hand_region = frame[y_min:y_max, x_min:x_max]
    
    return hand_region, (x_min, y_min, x_max, y_max)


def main():
    """Main function"""
    
    print("="*70)
    print("CLASS X WEBCAM TEST")
    print("="*70)
    print("\nThis will test Class X recognition in real-time")
    print("Using external camera (VideoCapture(1))")
    print("\nLoading models...")
    
    # Load MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✓ Hand tracking loaded")
    
    # Load classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Sign classifier loaded")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*70)
    print("Ready!")
    print("="*70)
    print("\nStarting external camera (VideoCapture(1))...")
    print("\nHow to test Class X:")
    print("  1. Make the X sign with your hand")
    print("  2. Hold it steady in front of the camera")
    print("  3. Watch the predictions in real-time")
    print("  4. Press SPACE to mark when you're showing X")
    print("  5. System will track accuracy")
    print("\nControls:")
    print("  - SPACE: Mark 'I am showing X now'")
    print("  - R: Reset statistics")
    print("  - ESC: Exit")
    print("="*70)
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        # Use VideoCapture(1) for external camera
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("✗ Error: Could not open external camera (VideoCapture(1))")
            print("Trying default camera (VideoCapture(0))...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("✗ Error: Could not open any camera")
                return
        
        # Stats
        x_test_mode = False
        x_predictions = []
        x_correct = 0
        x_total = 0
        
        # Prediction buffer for smoothing
        prediction_buffer = deque(maxlen=15)
        
        # FPS
        fps_list = []
        prev_time = time.time()
        
        print("\n✓ Camera started!")
        print("Show your hand to start...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            prev_time = current_time
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand
            results = hands.process(rgb_frame)
            
            hand_detected = False
            current_prediction = None
            current_confidence = 0.0
            bbox = None
            
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
                    hand_img, bbox = extract_hand_region(frame, hand_landmarks, padding=60)
                    
                    # Predict
                    if hand_img is not None and hand_img.size > 0:
                        try:
                            # Preprocess
                            hand_resized = cv2.resize(hand_img, (299, 299))
                            yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
                            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                            hand_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                            
                            # Predict
                            image_data = cv2.imencode('.jpg', hand_resized, 
                                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                            top_predictions = predict_sign(image_data, sess, softmax_tensor, label_lines)
                            
                            current_prediction = top_predictions[0][0]
                            current_confidence = top_predictions[0][1]
                            
                            # Add to buffer
                            prediction_buffer.append((current_prediction, current_confidence))
                            
                            # If in X test mode, track predictions
                            if x_test_mode:
                                x_predictions.append(current_prediction)
                                x_total += 1
                                if current_prediction == 'X':
                                    x_correct += 1
                        
                        except Exception as e:
                            pass
            
            # Get smoothed prediction
            smoothed_pred = None
            smoothed_conf = 0.0
            
            if len(prediction_buffer) >= 5:
                pred_counter = Counter([p[0] for p in prediction_buffer])
                most_common = pred_counter.most_common(1)[0]
                smoothed_pred = most_common[0]
                
                # Average confidence for this prediction
                pred_confs = [p[1] for p in prediction_buffer if p[0] == smoothed_pred]
                smoothed_conf = np.mean(pred_confs) if pred_confs else 0.0
            
            # Draw bounding box
            if hand_detected and bbox:
                # Color based on prediction
                if smoothed_pred == 'X':
                    box_color = (0, 255, 0)  # Green for X
                elif x_test_mode:
                    box_color = (0, 0, 255)  # Red when testing X but not detecting X
                else:
                    box_color = (255, 255, 0)  # Cyan for other signs
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 3)
            
            # Info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (800, 400), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Camera info
            cv2.putText(frame, "Camera: VideoCapture(1)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
            # Current prediction
            if smoothed_pred:
                pred_color = (0, 255, 0) if smoothed_pred == 'X' else (255, 255, 255)
                cv2.putText(frame, f"Prediction: {smoothed_pred.upper()}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
                cv2.putText(frame, f"Confidence: {smoothed_conf*100:.1f}%", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
            elif hand_detected:
                cv2.putText(frame, "Analyzing...", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
            else:
                cv2.putText(frame, "Show your hand", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            
            # Top 5 predictions
            if len(prediction_buffer) > 0:
                cv2.putText(frame, "Recent predictions:", (10, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                recent_counter = Counter([p[0] for p in prediction_buffer])
                top5 = recent_counter.most_common(5)
                
                y_off = 230
                for idx, (label, count) in enumerate(top5):
                    pct = (count / len(prediction_buffer)) * 100
                    color = (0, 255, 0) if label == 'X' else (150, 150, 150)
                    text = f"{idx+1}. {label.upper()}: {pct:.0f}%"
                    cv2.putText(frame, text, (10, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_off += 30
            
            # X test mode indicator
            if x_test_mode:
                cv2.rectangle(frame, (w-350, 10), (w-10, 120), (0, 255, 0), 3)
                cv2.putText(frame, "X TEST MODE", (w-330, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                x_accuracy = (x_correct / x_total * 100) if x_total > 0 else 0.0
                cv2.putText(frame, f"Frames: {x_total}", (w-330, 75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"X detected: {x_correct} ({x_accuracy:.1f}%)", (w-330, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "SPACE=Start/Stop X test | R=Reset | ESC=Exit", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            cv2.imshow('Class X Webcam Test', frame)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                x_test_mode = not x_test_mode
                if x_test_mode:
                    print("\n✓ X TEST MODE STARTED")
                    print("  Make the X sign and hold it steady")
                    print("  System will track how often it detects X")
                    x_predictions = []
                    x_correct = 0
                    x_total = 0
                else:
                    print("\n✓ X TEST MODE STOPPED")
                    if x_total > 0:
                        x_accuracy = (x_correct / x_total) * 100
                        print(f"\n  Results:")
                        print(f"    Total frames: {x_total}")
                        print(f"    X detected: {x_correct}")
                        print(f"    Accuracy: {x_accuracy:.2f}%")
                        
                        # Show confusion
                        if x_predictions:
                            pred_counter = Counter(x_predictions)
                            print(f"\n  Prediction breakdown:")
                            for label, count in pred_counter.most_common(5):
                                pct = (count / len(x_predictions)) * 100
                                print(f"    {label}: {count} ({pct:.1f}%)")
            
            elif key == ord('r') or key == ord('R'):
                x_predictions = []
                x_correct = 0
                x_total = 0
                prediction_buffer.clear()
                print("\n✓ Statistics reset")
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    
    print("\n" + "="*70)
    print("Test ended")
    print("="*70)
    
    if x_total > 0:
        x_accuracy = (x_correct / x_total) * 100
        print(f"\nFinal X Test Results:")
        print(f"  Total frames tested: {x_total}")
        print(f"  X correctly detected: {x_correct}")
        print(f"  Accuracy: {x_accuracy:.2f}%")


if __name__ == "__main__":
    main()
