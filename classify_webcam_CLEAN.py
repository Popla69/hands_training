"""
Clean webcam classifier - No MediaPipe, just simple frame capture
Captures 5 frames per second and analyzes them
"""

import sys
import os
import cv2
import numpy as np
import time
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()


def detect_hand_in_roi(roi):
    """
    Detect if there's a hand in the ROI using skin detection
    Returns True if hand detected, False otherwise
    """
    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Calculate percentage of skin pixels
    skin_pixels = cv2.countNonZero(mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    skin_percentage = (skin_pixels / total_pixels) * 100
    
    # Hand detected if at least 15% of ROI is skin
    return skin_percentage >= 15, skin_percentage, mask


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign from image"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def main():
    print("="*70)
    print("CLEAN WEBCAM CLASSIFIER")
    print("="*70)
    print("\nLoading model...")
    
    # Load classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Model loaded")
        print(f"✓ {len(label_lines)} classes loaded")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    print("\n" + "="*70)
    print("Starting camera (VideoCapture 1)...")
    print("="*70)
    print("\nHow to use:")
    print("  1. Position your hand in the GREEN BOX")
    print("  2. Make a sign and hold it steady")
    print("  3. Press SPACE to capture and add to sequence")
    print("  4. System captures 5 frames and votes on the sign")
    print("\nControls:")
    print("  - SPACE: Capture sign")
    print("  - C: Clear sequence")
    print("  - ESC: Exit")
    print("="*70)
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        # Open camera
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("External camera not found, trying default...")
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Error: Could not open camera")
            return
        
        print("\n✓ Camera started!")
        
        # State
        sequence = ''
        capturing = False
        capture_frames = []
        capture_start_time = None
        
        # ROI (Region of Interest) - centered box
        roi_size = 400
        
        # Stats
        total_captured = 0
        
        # FPS
        fps_list = []
        prev_time = time.time()
        
        # Live prediction (not for capture, just display)
        last_prediction = None
        last_confidence = 0.0
        last_pred_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Calculate ROI position (centered)
            roi_x = (w - roi_size) // 2
            roi_y = (h - roi_size) // 2
            
            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            prev_time = current_time
            
            # Extract ROI (ONLY content inside the box)
            roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size].copy()
            
            # Detect hand in ROI
            hand_detected, skin_pct, skin_mask = detect_hand_in_roi(roi)
            
            # Live prediction (every 0.5 seconds, just for display, ONLY if hand detected)
            if current_time - last_pred_time > 0.5 and not capturing:
                if hand_detected:
                    try:
                        # Preprocess ROI
                        roi_resized = cv2.resize(roi, (299, 299))
                        
                        # Predict
                        image_data = cv2.imencode('.jpg', roi_resized, 
                                                 [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                        predictions = predict_sign(image_data, sess, softmax_tensor, label_lines)
                        
                        last_prediction = predictions[0][0]
                        last_confidence = predictions[0][1]
                        last_pred_time = current_time
                    except:
                        pass
                else:
                    # No hand detected
                    last_prediction = None
                    last_confidence = 0.0
                    last_pred_time = current_time
            
            # Capture mode
            if capturing:
                elapsed = current_time - capture_start_time
                
                # Check if hand is still in ROI
                if not hand_detected:
                    print("  ✗ Hand lost! Capture cancelled.")
                    capturing = False
                    capture_frames = []
                else:
                    # Capture 5 frames over 1 second (0.2 second intervals)
                    if len(capture_frames) < 5:
                        if len(capture_frames) == 0 or elapsed >= len(capture_frames) * 0.2:
                            try:
                                # Preprocess ROI (ONLY the box content)
                                roi_resized = cv2.resize(roi, (299, 299))
                                
                                # Predict
                                image_data = cv2.imencode('.jpg', roi_resized, 
                                                         [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                                predictions = predict_sign(image_data, sess, softmax_tensor, label_lines)
                                
                                capture_frames.append(predictions[0])
                                print(f"  Frame {len(capture_frames)}/5: {predictions[0][0].upper()} ({predictions[0][1]*100:.1f}%)")
                            except Exception as e:
                                print(f"  Error capturing frame: {e}")
                
                    # Done capturing
                    if len(capture_frames) >= 5:
                        # Vote on the sign
                        pred_labels = [p[0] for p in capture_frames]
                        pred_counter = Counter(pred_labels)
                        
                        most_common = pred_counter.most_common(1)[0]
                        final_pred = most_common[0]
                        vote_count = most_common[1]
                        
                        # Average confidence for the winning prediction
                        winning_confs = [p[1] for p in capture_frames if p[0] == final_pred]
                        avg_conf = np.mean(winning_confs)
                        
                        # Require at least 3/5 votes and minimum confidence
                        if vote_count >= 3 and avg_conf >= 0.3:
                            print(f"\n✓ Result: {final_pred.upper()} ({vote_count}/5 votes, {avg_conf*100:.1f}% confidence)")
                            
                            # Add to sequence
                            if final_pred == 'space':
                                sequence += ' '
                                print("  Added: SPACE")
                            elif final_pred == 'del':
                                sequence = sequence[:-1]
                                print("  Deleted last character")
                            elif final_pred != 'nothing':
                                sequence += final_pred
                                print(f"  Added: {final_pred.upper()}")
                            else:
                                print("  Ignored: nothing")
                            
                            total_captured += 1
                        else:
                            print(f"\n✗ Rejected: {final_pred.upper()} (only {vote_count}/5 votes or low confidence)")
                        
                        # Reset
                        capturing = False
                        capture_frames = []
                        last_pred_time = 0  # Force immediate live prediction update
            
            # Draw ROI box with hand detection status
            if capturing:
                box_color = (0, 255, 255)  # Yellow when capturing
                box_thickness = 5
            elif hand_detected:
                box_color = (0, 255, 0)  # Green when hand detected
                box_thickness = 3
            else:
                box_color = (0, 0, 255)  # Red when no hand
                box_thickness = 3
            
            cv2.rectangle(frame, (roi_x, roi_y), 
                         (roi_x + roi_size, roi_y + roi_size), 
                         box_color, box_thickness)
            
            # Show skin detection mask in corner
            if skin_mask is not None:
                mask_small = cv2.resize(skin_mask, (150, 150))
                mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[10:160, w-160:w-10] = mask_colored
            
            # Info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (600, 290), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Camera info
            cv2.putText(frame, "Camera: VideoCapture(1)", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
            # Hand detection status
            if hand_detected:
                hand_status = f"Hand: YES ({skin_pct:.1f}% skin)"
                hand_color = (0, 255, 0)
            else:
                hand_status = f"Hand: NO ({skin_pct:.1f}% skin)"
                hand_color = (0, 0, 255)
            
            cv2.putText(frame, hand_status, (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
            
            # Status
            if capturing:
                status_text = f"CAPTURING... {len(capture_frames)}/5 frames"
                status_color = (0, 255, 255)
            elif hand_detected:
                status_text = "Ready - Press SPACE to capture"
                status_color = (0, 255, 0)
            else:
                status_text = "Show hand in box"
                status_color = (0, 0, 255)
            
            cv2.putText(frame, status_text, (10, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Live prediction (when not capturing and hand detected)
            if not capturing and hand_detected and last_prediction:
                cv2.putText(frame, f"Live: {last_prediction.upper()}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.putText(frame, f"Conf: {last_confidence*100:.1f}%", (10, 215), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Instructions
            cv2.putText(frame, "SPACE=Capture | C=Clear | ESC=Exit", (10, 260), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            # Stats
            cv2.putText(frame, f"Captured: {total_captured}", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            cv2.imshow('Sign Language Classifier', frame)
            
            # Sequence window
            img_sequence = np.zeros((400, 1200, 3), np.uint8)
            cv2.putText(img_sequence, "Sequence:", (30, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            # Word wrap
            max_chars = 50
            lines = []
            current_line = ""
            
            for char in sequence.upper():
                if len(current_line) >= max_chars:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            y_offset = 90
            for line in lines[-6:]:
                cv2.putText(img_sequence, line, (30, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                y_offset += 50
            
            cv2.imshow('Sequence', img_sequence)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE
                if not capturing:
                    if hand_detected:
                        print("\n" + "="*50)
                        print("CAPTURING 5 FRAMES...")
                        print("="*50)
                        capturing = True
                        capture_frames = []
                        capture_start_time = current_time
                    else:
                        print("\n✗ No hand detected! Show your hand in the box first.")
            elif key == ord('c') or key == ord('C'):
                sequence = ''
                print("\n✓ Sequence cleared")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Session ended")
    print("="*70)
    print(f"\nFinal sequence: {sequence.upper()}")
    print(f"Total signs captured: {total_captured}")


if __name__ == "__main__":
    main()
