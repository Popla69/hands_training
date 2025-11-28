"""
FIXED Sign Language Recognition System
- Better accuracy with improved preprocessing
- Cooldown period after detection (no immediate restart)
- Higher confidence thresholds
- Optimized hand extraction
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


# Confusion handling
CONFUSION_PAIRS = {
    'M': {'N': 0.65},
    'V': {'W': 0.30},
    'E': {'B': 0.29},
    'J': {'I': 0.23},
    'Z': {'X': 0.25},
}


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign - returns top 5"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def calculate_hand_motion(landmarks_history):
    """Calculate hand movement"""
    if len(landmarks_history) < 2:
        return 0.0
    
    prev_landmarks = landmarks_history[-2]
    curr_landmarks = landmarks_history[-1]
    
    movements = []
    for i in range(len(prev_landmarks)):
        dx = curr_landmarks[i].x - prev_landmarks[i].x
        dy = curr_landmarks[i].y - prev_landmarks[i].y
        movement = np.sqrt(dx*dx + dy*dy)
        movements.append(movement)
    
    avg_movement = np.mean(movements)
    motion_score = min(avg_movement / 0.03, 1.0)  # More sensitive
    
    return motion_score


def analyze_predictions(prediction_buffer, min_agreement=0.60, min_confidence=0.40):
    """Analyze predictions with stricter thresholds"""
    
    if len(prediction_buffer) < 20:  # Need more samples
        return None, 0.0, "Not enough samples"
    
    all_predictions = [p[0] for p in prediction_buffer]
    counter = Counter(all_predictions)
    
    most_common = counter.most_common(3)
    
    if not most_common:
        return None, 0.0, "No predictions"
    
    top_pred, top_count = most_common[0]
    agreement_rate = top_count / len(prediction_buffer)
    
    # Average confidence for top prediction
    top_pred_scores = [p[1] for p in prediction_buffer if p[0] == top_pred]
    avg_confidence = np.mean(top_pred_scores) if top_pred_scores else 0.0
    
    # Stricter requirements
    if agreement_rate < min_agreement:
        return None, avg_confidence, f"Low agreement ({agreement_rate*100:.0f}%)"
    
    if avg_confidence < min_confidence:
        return None, avg_confidence, f"Low confidence ({avg_confidence*100:.0f}%)"
    
    # Handle confusion pairs
    if len(most_common) >= 2:
        second_pred, second_count = most_common[1]
        
        if top_pred in CONFUSION_PAIRS and second_pred in CONFUSION_PAIRS.get(top_pred, {}):
            if second_count / top_count > 0.6:
                second_scores = [p[1] for p in prediction_buffer if p[0] == second_pred]
                second_avg = np.mean(second_scores) if second_scores else 0.0
                
                if second_avg > avg_confidence * 1.1:  # 10% better
                    return second_pred, second_avg, f"Resolved {top_pred}/{second_pred}"
    
    return top_pred, avg_confidence, f"Agreement: {agreement_rate*100:.0f}%"


def extract_hand_region(frame, hand_landmarks, padding=80):
    """Extract hand region with better padding"""
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


def preprocess_hand_image(hand_img):
    """Better preprocessing for accuracy"""
    # Resize
    hand_resized = cv2.resize(hand_img, (299, 299))
    
    # Convert to grayscale for histogram equalization
    gray = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR
    hand_resized = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    # Slight blur to reduce noise
    hand_resized = cv2.GaussianBlur(hand_resized, (3, 3), 0)
    
    return hand_resized


def draw_timer_arc(frame, center, radius, progress, color):
    """Draw circular progress timer"""
    cv2.circle(frame, center, radius, (50, 50, 50), 3)
    
    if progress > 0:
        angle = int(360 * progress)
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, 4)
    
    cv2.circle(frame, center, 5, color, -1)


def main():
    """Main function"""
    
    print("="*70)
    print("FIXED Sign Language Recognition System")
    print("="*70)
    print("\nImprovements:")
    print("  ✓ Better preprocessing for accuracy")
    print("  ✓ Stricter confidence thresholds")
    print("  ✓ Cooldown period after detection")
    print("  ✓ Optimized hand extraction")
    print("\nLoading models...")
    
    # Load MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,  # Higher threshold
        min_tracking_confidence=0.6
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
    print("\nStarting camera...")
    print("\nHow it works:")
    print("  1. Show your hand clearly")
    print("  2. Make a sign and hold VERY STEADY for 3 seconds")
    print("  3. After detection, wait 2 seconds before next sign")
    print("  4. Keep hand in good lighting")
    print("\nControls:")
    print("  - ESC: Exit")
    print("  - C: Clear sequence")
    print("  - R: Reset current hold")
    print("="*70)
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # State
        sequence = ''
        
        # Hold timer
        HOLD_TIME = 3.0
        COOLDOWN_TIME = 2.0  # Wait 2 seconds after detection
        MOTION_THRESHOLD = 0.25  # Stricter motion threshold
        STABILITY_WINDOW = 0.7  # Longer stability requirement
        
        hold_start_time = None
        is_holding = False
        prediction_buffer = []
        
        # Cooldown
        cooldown_until = 0
        
        # Motion tracking
        landmarks_history = deque(maxlen=10)
        motion_scores = deque(maxlen=30)
        stable_time = None
        
        # FPS
        fps_list = []
        prev_time = time.time()
        
        # Stats
        total_added = 0
        resets = 0
        
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
            
            # Check cooldown
            in_cooldown = current_time < cooldown_until
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand
            results = hands.process(rgb_frame)
            
            hand_detected = False
            hand_img = None
            bbox = None
            motion_score = 0.0
            
            if results.multi_hand_landmarks and not in_cooldown:
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
                    
                    # Track landmarks for motion detection
                    landmarks_history.append(hand_landmarks.landmark)
                    
                    # Calculate motion
                    if len(landmarks_history) >= 2:
                        motion_score = calculate_hand_motion(landmarks_history)
                        motion_scores.append(motion_score)
                    
                    # Extract hand region
                    hand_img, bbox = extract_hand_region(frame, hand_landmarks, padding=80)
                    
                    # Check if hand is stable enough
                    avg_motion = np.mean(list(motion_scores)[-10:]) if motion_scores else 1.0
                    is_stable = avg_motion < MOTION_THRESHOLD
                    
                    # Stability tracking
                    if is_stable:
                        if stable_time is None:
                            stable_time = current_time
                        
                        stability_duration = current_time - stable_time
                        
                        # Start hold timer after stability window
                        if stability_duration >= STABILITY_WINDOW and not is_holding:
                            is_holding = True
                            hold_start_time = current_time
                            prediction_buffer = []
                            print(f"\n⏱ Hold timer started (stable for {STABILITY_WINDOW}s)")
                    else:
                        # Hand moving too much
                        if is_holding:
                            print(f"✗ Hold reset: hand moving (motion: {avg_motion:.2f})")
                            resets += 1
                        
                        is_holding = False
                        hold_start_time = None
                        stable_time = None
                        prediction_buffer = []
                    
                    # Predict if holding
                    if is_holding and hand_img is not None and hand_img.size > 0:
                        try:
                            # Better preprocessing
                            hand_processed = preprocess_hand_image(hand_img)
                            
                            # Predict
                            image_data = cv2.imencode('.jpg', hand_processed, 
                                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                            top_predictions = predict_sign(image_data, sess, softmax_tensor, label_lines)
                            
                            pred_label = top_predictions[0][0]
                            pred_score = top_predictions[0][1]
                            
                            # Higher confidence threshold
                            if pred_score > 0.35:
                                prediction_buffer.append((pred_label, pred_score))
                            
                            # Check if hold completed
                            hold_duration = current_time - hold_start_time
                            
                            if hold_duration >= HOLD_TIME:
                                # Analyze predictions with stricter thresholds
                                final_pred, final_conf, reason = analyze_predictions(
                                    prediction_buffer, min_agreement=0.60, min_confidence=0.40
                                )
                                
                                if final_pred and final_pred not in ['nothing']:
                                    if final_pred == 'space':
                                        sequence += ' '
                                        print(f"✓ Added: SPACE (conf: {final_conf*100:.1f}%)")
                                    elif final_pred == 'del':
                                        sequence = sequence[:-1]
                                        print(f"✓ Deleted last character")
                                    else:
                                        sequence += final_pred
                                        print(f"✓ Added: {final_pred.upper()} (conf: {final_conf*100:.1f}%, {reason})")
                                    
                                    total_added += 1
                                    
                                    # Start cooldown
                                    cooldown_until = current_time + COOLDOWN_TIME
                                    print(f"⏸ Cooldown: {COOLDOWN_TIME}s")
                                else:
                                    print(f"✗ Rejected: {reason}")
                                
                                # Reset
                                is_holding = False
                                hold_start_time = None
                                stable_time = None
                                prediction_buffer = []
                        
                        except Exception as e:
                            pass
            elif in_cooldown:
                # During cooldown, clear everything
                is_holding = False
                hold_start_time = None
                stable_time = None
                prediction_buffer = []
                landmarks_history.clear()
                motion_scores.clear()
            else:
                # No hand detected
                if is_holding:
                    print(f"✗ Hold reset: hand lost")
                    resets += 1
                
                is_holding = False
                hold_start_time = None
                stable_time = None
                prediction_buffer = []
                landmarks_history.clear()
                motion_scores.clear()
            
            # Calculate progress
            hold_progress = 0.0
            if is_holding and hold_start_time:
                hold_duration = current_time - hold_start_time
                hold_progress = min(hold_duration / HOLD_TIME, 1.0)
            
            # Draw bounding box if hand detected
            if hand_detected and bbox:
                # Color based on state
                if is_holding:
                    if hold_progress < 0.33:
                        box_color = (0, 255, 255)  # Yellow
                    elif hold_progress < 0.66:
                        box_color = (0, 165, 255)  # Orange
                    else:
                        box_color = (0, 255, 0)  # Green
                elif stable_time:
                    box_color = (255, 255, 0)  # Cyan - stabilizing
                else:
                    box_color = (0, 0, 255)  # Red - moving
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 3)
                
                # Draw timer if holding
                if is_holding:
                    timer_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    timer_radius = 50
                    draw_timer_arc(frame, timer_center, timer_radius, hold_progress, box_color)
                    
                    remaining = HOLD_TIME - (current_time - hold_start_time)
                    countdown_text = f"{remaining:.1f}s"
                    text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
                    text_x = timer_center[0] - text_size[0] // 2
                    text_y = timer_center[1] + text_size[1] // 2
                    cv2.putText(frame, countdown_text, (text_x, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, box_color, 2)
            
            # Info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (700, 350), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Cooldown indicator
            if in_cooldown:
                cooldown_remaining = cooldown_until - current_time
                cv2.putText(frame, f"COOLDOWN: {cooldown_remaining:.1f}s", (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            elif hand_detected:
                # Motion indicator
                avg_motion = np.mean(list(motion_scores)[-10:]) if motion_scores else 0.0
                motion_pct = min(avg_motion / MOTION_THRESHOLD, 1.0) * 100
                
                if avg_motion < MOTION_THRESHOLD:
                    motion_color = (0, 255, 0)
                    motion_text = f"Motion: {motion_pct:.0f}% - STABLE"
                else:
                    motion_color = (0, 0, 255)
                    motion_text = f"Motion: {motion_pct:.0f}% - MOVING"
                
                cv2.putText(frame, motion_text, (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
            
            # Status
            if in_cooldown:
                status_text = "Waiting... (cooldown)"
                status_color = (0, 165, 255)
            elif is_holding:
                status_text = f"HOLDING... ({len(prediction_buffer)} frames)"
                status_color = (0, 255, 255)
            elif stable_time:
                stab_duration = current_time - stable_time
                status_text = f"Stabilizing... {stab_duration:.1f}s"
                status_color = (255, 255, 0)
            elif hand_detected:
                status_text = "Hold hand VERY steady to start"
                status_color = (150, 150, 150)
            else:
                status_text = "Show your hand"
                status_color = (100, 100, 100)
            
            cv2.putText(frame, status_text, (10, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Current prediction
            if is_holding and prediction_buffer:
                temp_counter = Counter([p[0] for p in prediction_buffer])
                temp_most_common = temp_counter.most_common(1)[0]
                temp_pred = temp_most_common[0]
                temp_count = temp_most_common[1]
                temp_agreement = (temp_count / len(prediction_buffer)) * 100
                
                # Average confidence
                temp_scores = [p[1] for p in prediction_buffer if p[0] == temp_pred]
                temp_conf = np.mean(temp_scores) * 100 if temp_scores else 0
                
                pred_color = (0, 255, 255)
                
                cv2.putText(frame, f"Analyzing: {temp_pred.upper()}", (10, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
                cv2.putText(frame, f"Agreement: {temp_agreement:.0f}% | Conf: {temp_conf:.0f}%", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
                
                # Top 3
                cv2.putText(frame, "Top 3:", (10, 215), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                temp_top3 = temp_counter.most_common(3)
                y_off = 240
                for idx, (label, count) in enumerate(temp_top3):
                    pct = (count / len(prediction_buffer)) * 100
                    label_scores = [p[1] for p in prediction_buffer if p[0] == label]
                    label_conf = np.mean(label_scores) * 100 if label_scores else 0
                    color = (0, 255, 0) if idx == 0 else (150, 150, 150)
                    text = f"{idx+1}. {label.upper()}: {pct:.0f}% ({label_conf:.0f}%)"
                    cv2.putText(frame, text, (10, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_off += 25
            
            # Stats
            cv2.putText(frame, f"Added: {total_added} | Resets: {resets}", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            cv2.imshow('Sign Language Recognition - FIXED', frame)
            
            # Sequence window
            img_sequence = np.zeros((500, 1200, 3), np.uint8)
            cv2.putText(img_sequence, "Recognized Sequence:", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            cv2.putText(img_sequence, "C=Clear | R=Reset Hold | ESC=Exit", 
                       (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
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
            
            y_offset = 80
            for line in lines[-8:]:
                cv2.putText(img_sequence, line, (30, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 45
            
            cv2.imshow('Sequence', img_sequence)
            
            # Keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('c') or key == ord('C'):
                sequence = ''
                print("\nSequence cleared")
            elif key == ord('r') or key == ord('R'):
                if is_holding:
                    print("\nHold reset manually")
                    resets += 1
                is_holding = False
                hold_start_time = None
                stable_time = None
                prediction_buffer = []
                cooldown_until = 0
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
    
    print("\n" + "="*70)
    print("Session ended")
    print("="*70)
    print(f"\nFinal sequence: {sequence.upper()}")
    print(f"Total signs added: {total_added}")
    print(f"Timer resets: {resets}")
    if total_added > 0:
        print(f"Success rate: {(total_added / (total_added + resets)) * 100:.1f}%")


if __name__ == "__main__":
    main()
