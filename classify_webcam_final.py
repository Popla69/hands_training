"""
FINAL Sign Language Recognition System
- 3-second hold requirement before prediction
- Analyzes 30-60 frames during hold period
- Uses majority voting for accuracy
- Visual countdown timer
- Smart confusion handling
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


# Confusion pairs from analysis
CONFUSION_PAIRS = {
    'M': {'N': 0.65},
    'V': {'W': 0.30, 'U': 0.10},
    'E': {'B': 0.29},
    'J': {'I': 0.23},
    'Z': {'X': 0.25},
    'Y': {'A': 0.19, 'T': 0.17},
    'R': {'U': 0.19},
    'K': {'I': 0.10},
}

MOTION_SIGNS = ['J', 'Z']


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign - returns top 5"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def analyze_predictions(prediction_buffer, min_agreement=0.5):
    """
    Analyze all predictions collected during hold period
    Returns the most consistent prediction with confidence
    """
    
    if len(prediction_buffer) < 10:
        return None, 0.0, "Not enough samples"
    
    # Count all predictions
    all_predictions = [p[0] for p in prediction_buffer]
    counter = Counter(all_predictions)
    
    # Get most common prediction
    most_common = counter.most_common(3)
    
    if not most_common:
        return None, 0.0, "No predictions"
    
    top_pred, top_count = most_common[0]
    agreement_rate = top_count / len(prediction_buffer)
    
    # Calculate average confidence for the top prediction
    top_pred_scores = [p[1] for p in prediction_buffer if p[0] == top_pred]
    avg_confidence = np.mean(top_pred_scores) if top_pred_scores else 0.0
    
    # Require minimum agreement
    if agreement_rate < min_agreement:
        # Check if it's a confused pair
        if len(most_common) >= 2:
            second_pred, second_count = most_common[1]
            
            # If top two are confused pairs and close in count
            if top_pred in CONFUSION_PAIRS and second_pred in CONFUSION_PAIRS.get(top_pred, {}):
                if second_count / top_count > 0.7:  # Very close
                    # Use the one with higher average confidence
                    second_scores = [p[1] for p in prediction_buffer if p[0] == second_pred]
                    second_avg = np.mean(second_scores) if second_scores else 0.0
                    
                    if second_avg > avg_confidence:
                        return second_pred, second_avg, f"Resolved {top_pred}/{second_pred} confusion"
        
        return None, avg_confidence, f"Low agreement ({agreement_rate*100:.0f}%)"
    
    # Special handling for motion signs
    if top_pred in MOTION_SIGNS:
        if avg_confidence < 0.5 or agreement_rate < 0.6:
            return None, avg_confidence, f"Motion sign needs higher confidence"
    
    reason = f"Agreement: {agreement_rate*100:.0f}%"
    if len(most_common) >= 2:
        second_pred = most_common[1][0]
        reason += f" (vs {second_pred})"
    
    return top_pred, avg_confidence, reason


def draw_timer_arc(frame, center, radius, progress, color):
    """Draw a circular progress timer"""
    # Draw background circle
    cv2.circle(frame, center, radius, (50, 50, 50), 3)
    
    # Draw progress arc
    if progress > 0:
        angle = int(360 * progress)
        # OpenCV ellipse: start from top (-90°), go clockwise
        cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, color, 4)
    
    # Draw center dot
    cv2.circle(frame, center, 5, color, -1)


def main():
    """Main function"""
    
    print("="*70)
    print("FINAL Sign Language Recognition System")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ 3-second hold requirement")
    print("  ✓ Analyzes 30-60 frames per sign")
    print("  ✓ Majority voting for accuracy")
    print("  ✓ Visual countdown timer")
    print("  ✓ Smart confusion handling")
    print("  ✓ High accuracy mode")
    print("\nLoading model...")
    
    # Load classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Classifier loaded")
        print(f"✓ {len(label_lines)} classes")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    print("\n" + "="*70)
    print("Ready!")
    print("="*70)
    
    response = input("\nStart camera? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\nStarting camera...")
    print("\nHow it works:")
    print("  1. Put your hand in the GREEN BOX")
    print("  2. Hold the sign STEADY for 3 seconds")
    print("  3. Watch the circular timer fill up")
    print("  4. System analyzes 30-60 frames and picks best prediction")
    print("  5. Letter is added to sequence")
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
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # State
        sequence = ''
        
        # Hold timer state
        HOLD_TIME = 3.0  # seconds
        hold_start_time = None
        is_holding = False
        prediction_buffer = []  # Store all predictions during hold
        
        # Current prediction
        current_pred = None
        current_confidence = 0.0
        
        # FPS
        fps_list = []
        prev_time = time.time()
        
        # Box
        box_size = 400
        box_x = 100
        box_y = 100
        
        # Stats
        total_added = 0
        frame_count = 0
        
        print("\n✓ Camera started!")
        print("Put your hand in the GREEN BOX and hold for 3 seconds")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_count += 1
            
            # FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            prev_time = current_time
            
            # Extract hand
            x1, y1 = box_x, box_y
            x2, y2 = box_x + box_size, box_y + box_size
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            hand_img = frame[y1:y2, x1:x2]
            
            # Predict every frame during hold
            if hand_img.size > 0:
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
                    
                    pred_label = top_predictions[0][0]
                    pred_score = top_predictions[0][1]
                    
                    # Only consider predictions with reasonable confidence
                    if pred_score > 0.25:
                        # Start hold timer if not already holding
                        if not is_holding and pred_label not in ['nothing']:
                            is_holding = True
                            hold_start_time = current_time
                            prediction_buffer = []
                            print(f"\n⏱ Hold timer started for: {pred_label.upper()}")
                        
                        # If holding, collect predictions
                        if is_holding:
                            prediction_buffer.append((pred_label, pred_score))
                            
                            # Calculate hold progress
                            hold_duration = current_time - hold_start_time
                            hold_progress = min(hold_duration / HOLD_TIME, 1.0)
                            
                            # Check if hold time completed
                            if hold_duration >= HOLD_TIME:
                                # Analyze all collected predictions
                                final_pred, final_conf, reason = analyze_predictions(
                                    prediction_buffer, min_agreement=0.5
                                )
                                
                                if final_pred and final_pred not in ['nothing']:
                                    # Add to sequence
                                    if final_pred == 'space':
                                        sequence += ' '
                                        print(f"✓ Added: SPACE")
                                    elif final_pred == 'del':
                                        sequence = sequence[:-1]
                                        print(f"✓ Deleted last character")
                                    else:
                                        sequence += final_pred
                                        print(f"✓ Added: {final_pred.upper()} (confidence: {final_conf*100:.1f}%, {reason})")
                                    
                                    total_added += 1
                                else:
                                    print(f"✗ Rejected: {reason}")
                                
                                # Reset hold
                                is_holding = False
                                hold_start_time = None
                                prediction_buffer = []
                    else:
                        # Low confidence - reset hold
                        if is_holding:
                            print(f"✗ Hold reset: low confidence")
                        is_holding = False
                        hold_start_time = None
                        prediction_buffer = []
                
                except Exception as e:
                    pass
            else:
                # No hand - reset hold
                if is_holding:
                    print(f"✗ Hold reset: no hand detected")
                is_holding = False
                hold_start_time = None
                prediction_buffer = []
            
            # Calculate hold progress
            hold_progress = 0.0
            if is_holding and hold_start_time:
                hold_duration = current_time - hold_start_time
                hold_progress = min(hold_duration / HOLD_TIME, 1.0)
            
            # Draw box
            if is_holding:
                # Color changes as timer progresses
                if hold_progress < 0.33:
                    box_color = (0, 255, 255)  # Yellow
                elif hold_progress < 0.66:
                    box_color = (0, 165, 255)  # Orange
                else:
                    box_color = (0, 255, 0)  # Green
            else:
                box_color = (0, 255, 0)  # Green
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
            
            # Draw timer arc if holding
            if is_holding:
                timer_center = (x1 + box_size // 2, y1 + box_size // 2)
                timer_radius = 60
                draw_timer_arc(frame, timer_center, timer_radius, hold_progress, box_color)
                
                # Show countdown
                remaining = HOLD_TIME - (current_time - hold_start_time)
                countdown_text = f"{remaining:.1f}s"
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = timer_center[0] - text_size[0] // 2
                text_y = timer_center[1] + text_size[1] // 2
                cv2.putText(frame, countdown_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, box_color, 3)
            
            # Preview
            try:
                preview_size = 200
                preview = cv2.resize(hand_img, (preview_size, preview_size))
                yuv = cv2.cvtColor(preview, cv2.COLOR_BGR2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                preview = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                frame[h-preview_size-10:h-10, w-preview_size-10:w-10] = preview
                cv2.rectangle(frame, (w-preview_size-10, h-preview_size-10), 
                            (w-10, h-10), (0, 255, 255), 2)
            except:
                pass
            
            # Info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (700, 280), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Status
            if is_holding:
                status_text = f"HOLDING... ({len(prediction_buffer)} frames)"
                status_color = (0, 255, 255)
            else:
                status_text = "Show sign and hold for 3 seconds"
                status_color = (150, 150, 150)
            
            cv2.putText(frame, status_text, (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Current prediction during hold
            if is_holding and prediction_buffer:
                # Show most common prediction so far
                temp_counter = Counter([p[0] for p in prediction_buffer])
                temp_most_common = temp_counter.most_common(1)[0]
                temp_pred = temp_most_common[0]
                temp_count = temp_most_common[1]
                temp_agreement = (temp_count / len(prediction_buffer)) * 100
                
                pred_color = (0, 255, 255)
                if temp_pred in CONFUSION_PAIRS:
                    pred_color = (0, 165, 255)
                elif temp_pred in MOTION_SIGNS:
                    pred_color = (255, 0, 255)
                
                cv2.putText(frame, f"Analyzing: {temp_pred.upper()}", (10, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 2)
                cv2.putText(frame, f"Agreement: {temp_agreement:.0f}%", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
                
                # Show top 3 current predictions
                cv2.putText(frame, "Current top 3:", (10, 175), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                temp_top3 = temp_counter.most_common(3)
                y_off = 200
                for idx, (label, count) in enumerate(temp_top3):
                    pct = (count / len(prediction_buffer)) * 100
                    color = (0, 255, 0) if idx == 0 else (150, 150, 150)
                    text = f"{idx+1}. {label.upper()}: {pct:.0f}%"
                    cv2.putText(frame, text, (10, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_off += 25
            
            # Stats
            cv2.putText(frame, f"Added: {total_added} signs", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            # Instructions
            cv2.putText(frame, "HOLD SIGN FOR 3 SECONDS", (box_x, box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            cv2.imshow('Sign Language Recognition - FINAL', frame)
            
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
                is_holding = False
                hold_start_time = None
                prediction_buffer = []
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Session ended")
    print("="*70)
    print(f"\nFinal sequence: {sequence.upper()}")
    print(f"Total signs added: {total_added}")
    print(f"Total frames processed: {frame_count}")
    if total_added > 0:
        print(f"Average frames per sign: {frame_count / total_added:.0f}")


if __name__ == "__main__":
    main()
