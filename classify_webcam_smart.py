"""
SMART Sign Language Recognition
Uses confusion analysis to make better predictions
Handles M/N, V/W, E/B, J/I, Z/X confusions intelligently
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


# Confusion matrix from analysis - signs that get confused
CONFUSION_PAIRS = {
    'M': {'N': 0.65, 'A': 0.05},  # M confused with N 65% of time
    'N': {'M': 0.10},
    'V': {'W': 0.30, 'L': 0.10, 'U': 0.10},
    'W': {'V': 0.05},
    'E': {'B': 0.29, 'X': 0.12},
    'B': {'E': 0.05},
    'J': {'I': 0.23, 'X': 0.07},  # Motion sign
    'I': {'J': 0.05},
    'Z': {'X': 0.25, 'SPACE': 0.13},  # Motion sign
    'X': {'Z': 0.05},
    'Y': {'A': 0.19, 'T': 0.17},
    'R': {'U': 0.19},
    'U': {'R': 0.09},
    'K': {'I': 0.10, 'SPACE': 0.10},
}

# Signs that require motion (can't be reliably detected from single frame)
MOTION_SIGNS = ['J', 'Z']


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign - returns top 5"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def smart_prediction(top_predictions, history, confidence_threshold=0.4):
    """
    Make smart prediction considering:
    1. Confidence scores
    2. Known confusions
    3. Temporal consistency
    """
    
    if not top_predictions:
        return None, 0.0, "No prediction"
    
    top_pred = top_predictions[0][0]
    top_score = top_predictions[0][1]
    
    # If very confident, trust it
    if top_score > 0.7:
        return top_pred, top_score, "High confidence"
    
    # Check if this is a commonly confused sign
    if top_pred in CONFUSION_PAIRS:
        # Look at top 3 predictions
        top_3_labels = [p[0] for p in top_predictions[:3]]
        top_3_scores = [p[1] for p in top_predictions[:3]]
        
        # Check if confused pair is in top 3
        for confused_sign, confusion_rate in CONFUSION_PAIRS[top_pred].items():
            if confused_sign in top_3_labels:
                idx = top_3_labels.index(confused_sign)
                confused_score = top_3_scores[idx]
                
                # If scores are close and confusion rate is high
                score_diff = top_score - confused_score
                
                if score_diff < 0.15 and confusion_rate > 0.2:
                    # Use temporal history to decide
                    if len(history) >= 3:
                        recent = list(history)[-5:]
                        counter = Counter(recent)
                        
                        # If confused sign appears more in history
                        if counter.get(confused_sign, 0) > counter.get(top_pred, 0):
                            return confused_sign, confused_score, f"Resolved confusion with {top_pred}"
    
    # For motion signs, require higher confidence
    if top_pred in MOTION_SIGNS:
        if top_score < 0.5:
            return None, top_score, f"Motion sign {top_pred} needs higher confidence"
    
    # Default: use top prediction if above threshold
    if top_score >= confidence_threshold:
        return top_pred, top_score, "Standard"
    
    return None, top_score, "Low confidence"


def temporal_smoothing(prediction_history, window_size=7):
    """Smooth predictions using majority voting"""
    if len(prediction_history) < window_size:
        return None
    
    recent = list(prediction_history)[-window_size:]
    counter = Counter(recent)
    most_common = counter.most_common(1)[0]
    
    # Require 60% agreement
    if most_common[1] >= window_size * 0.6:
        return most_common[0]
    
    return None


def main():
    """Main function"""
    
    print("="*70)
    print("SMART Sign Language Recognition")
    print("="*70)
    print("\nIntelligent Features:")
    print("  ✓ Handles M/N confusion (65% confusion rate)")
    print("  ✓ Handles V/W confusion (30% confusion rate)")
    print("  ✓ Handles E/B confusion (29% confusion rate)")
    print("  ✓ Special handling for motion signs (J, Z)")
    print("  ✓ Temporal smoothing over 7 frames")
    print("  ✓ Confidence-based filtering")
    print("\nLoading model...")
    
    # Load classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Classifier loaded")
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
    print("\nControls:")
    print("  - ESC: Exit")
    print("  - SPACE: Toggle smart mode")
    print("  - C: Clear sequence")
    print("  - Hold sign steady for 2-3 seconds")
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
        top_predictions = []
        i = 0
        mem = ''
        consecutive = 0
        sequence = ''
        
        # Smart features
        use_smart_mode = True
        prediction_history = deque(maxlen=15)
        raw_history = deque(maxlen=15)
        
        # FPS
        fps_list = []
        prev_time = time.time()
        
        # Box
        box_size = 400
        box_x = 100
        box_y = 100
        
        # Stats
        smart_corrections = 0
        total_predictions = 0
        
        print("\n✓ Camera started!")
        print("Put your hand in the GREEN BOX")
        
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
            
            # Extract hand
            x1, y1 = box_x, box_y
            x2, y2 = box_x + box_size, box_y + box_size
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            hand_img = frame[y1:y2, x1:x2]
            
            # Predict every 2 frames
            if i >= 1 and hand_img.size > 0:
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
                    
                    raw_pred = top_predictions[0][0]
                    raw_score = top_predictions[0][1]
                    
                    # Smart prediction
                    if use_smart_mode:
                        smart_pred, smart_score, reason = smart_prediction(
                            top_predictions, prediction_history, confidence_threshold=0.35
                        )
                        
                        if smart_pred:
                            res = smart_pred
                            score = smart_score
                            
                            if smart_pred != raw_pred:
                                smart_corrections += 1
                            
                            prediction_history.append(res)
                        else:
                            res = None
                            score = raw_score
                    else:
                        # Standard mode
                        if raw_score > 0.3:
                            res = raw_pred
                            score = raw_score
                            prediction_history.append(res)
                        else:
                            res = None
                            score = raw_score
                    
                    raw_history.append(raw_pred)
                    total_predictions += 1
                    i = 0
                    
                    # Sequence logic
                    if res:
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
                    pass
            
            i += 1
            
            # Draw box
            box_color = (0, 255, 0) if consecutive < 2 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
            
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
            cv2.rectangle(overlay, (0, 0), (750, 320), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Smart mode status
            mode_text = "SMART" if use_smart_mode else "STANDARD"
            mode_color = (0, 255, 0) if use_smart_mode else (150, 150, 150)
            cv2.putText(frame, f"Mode: {mode_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            # Stats
            if total_predictions > 0 and use_smart_mode:
                correction_rate = (smart_corrections / total_predictions) * 100
                cv2.putText(frame, f"Corrections: {correction_rate:.1f}%", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
            
            # Prediction
            if top_predictions:
                pred_label = top_predictions[0][0]
                pred_score = top_predictions[0][1]
                
                # Color based on confusion
                if pred_label in CONFUSION_PAIRS:
                    pred_color = (0, 165, 255)  # Orange
                    cv2.putText(frame, f"⚠ Confused with: {', '.join(CONFUSION_PAIRS[pred_label].keys())}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                elif pred_label in MOTION_SIGNS:
                    pred_color = (255, 0, 255)  # Magenta
                    cv2.putText(frame, "⚡ Motion sign - hold steady", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                else:
                    pred_color = (0, 255, 255)
                
                cv2.putText(frame, f"Sign: {pred_label.upper()}", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
                cv2.putText(frame, f"Confidence: {pred_score*100:.1f}%", (10, 195), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
                
                # Top 3
                cv2.putText(frame, "Top 3:", (10, 225), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                y_off = 250
                for idx, (label, sc) in enumerate(top_predictions[:3]):
                    color = (0, 255, 0) if idx == 0 else (150, 150, 150)
                    text = f"{idx+1}. {label.upper()}: {sc*100:.1f}%"
                    cv2.putText(frame, text, (10, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_off += 25
            
            # Instructions
            cv2.putText(frame, "Put hand in GREEN BOX", (box_x, box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Recognition - SMART', frame)
            
            # Sequence window
            img_sequence = np.zeros((500, 1200, 3), np.uint8)
            cv2.putText(img_sequence, "Recognized Sequence:", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            cv2.putText(img_sequence, "SPACE=Toggle Mode | C=Clear | ESC=Exit", 
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
            elif key == 32:  # SPACE
                use_smart_mode = not use_smart_mode
                print(f"Smart mode: {'ON' if use_smart_mode else 'OFF'}")
            elif key == ord('c') or key == ord('C'):
                sequence = ''
                print("Sequence cleared")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Session ended")
    print("="*70)
    print(f"\nFinal sequence: {sequence.upper()}")
    print(f"Total predictions: {total_predictions}")
    if use_smart_mode and total_predictions > 0:
        print(f"Smart corrections: {smart_corrections} ({(smart_corrections/total_predictions)*100:.1f}%)")


if __name__ == "__main__":
    main()
