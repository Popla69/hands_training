"""
Improved Sign Language Recognition
- Shows top 3 predictions to see confusion
- Temporal smoothing over multiple frames
- Manual correction mode
- Better handling of similar signs
"""

import sys
import os
import cv2
import numpy as np
import time
from collections import deque, Counter

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()


# Signs that are commonly confused (similar hand shapes)
SIMILAR_SIGNS = {
    'u': ['v', 'n'],
    'v': ['u', 'n'],
    'r': ['u', 'v'],
    's': ['a', 't'],
    'k': ['p', 'v'],
    'j': ['i'],  # J requires motion
    'z': ['nothing'],  # Z requires motion
    'q': ['g'],
    'y': ['i']
}


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign language letter from image - returns top 5"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        results.append((human_string, score))
    
    return results


def temporal_smoothing(prediction_history, window_size=5):
    """Smooth predictions over time using voting"""
    if len(prediction_history) < window_size:
        return None
    
    # Get last N predictions
    recent = list(prediction_history)[-window_size:]
    
    # Count occurrences
    counter = Counter(recent)
    
    # Return most common if it appears at least 60% of the time
    most_common = counter.most_common(1)[0]
    if most_common[1] >= window_size * 0.6:
        return most_common[0]
    
    return None


def main():
    """Main function"""
    
    print("="*70)
    print("IMPROVED Sign Language Recognition")
    print("="*70)
    print("\nFeatures:")
    print("  - Shows top 3 predictions")
    print("  - Temporal smoothing for stability")
    print("  - Manual correction mode")
    print("  - Highlights confused signs")
    print("\nLoading model...")
    
    # Load sign language classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Sign language classifier loaded")
        print(f"✓ {len(label_lines)} classes: {', '.join(label_lines[:10])}...")
    except Exception as e:
        print(f"✗ Error loading classifier: {e}")
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
    print("  - SPACE: Toggle temporal smoothing")
    print("  - S: Save current frame for debugging")
    print("  - C: Clear sequence")
    print("  - M: Manual correction mode (type correct letter)")
    print("  - Hold sign steady for 2-3 seconds to add to sequence")
    print("="*70)
    
    # Start inference
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
        
        # State variables
        top_predictions = []
        i = 0
        mem = ''
        consecutive = 0
        sequence = ''
        
        # Temporal smoothing
        use_smoothing = True
        prediction_history = deque(maxlen=10)
        smoothed_result = None
        
        # FPS tracking
        fps_list = []
        prev_time = time.time()
        
        # Box configuration
        box_size = 400
        box_x = 100
        box_y = 100
        
        # Statistics
        confusion_log = []
        save_counter = 0
        
        print("\n✓ Camera started!")
        print("Put your hand in the GREEN BOX")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            prev_time = current_time
            
            # Extract hand region from fixed box
            x1, y1 = box_x, box_y
            x2, y2 = box_x + box_size, box_y + box_size
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            hand_img = frame[y1:y2, x1:x2]
            
            # Predict every 2 frames
            if i >= 1 and hand_img.size > 0:
                try:
                    # Resize and preprocess
                    hand_resized = cv2.resize(hand_img, (299, 299))
                    
                    # Histogram equalization
                    yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    hand_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    
                    # Encode and predict
                    image_data = cv2.imencode('.jpg', hand_resized, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                    top_predictions = predict_sign(image_data, sess, softmax_tensor, label_lines)
                    
                    # Get raw prediction
                    raw_res = top_predictions[0][0]
                    raw_score = top_predictions[0][1]
                    
                    # Add to history for smoothing
                    if raw_score > 0.2:  # Only add confident predictions
                        prediction_history.append(raw_res)
                    
                    # Use smoothed or raw result
                    if use_smoothing:
                        smoothed = temporal_smoothing(prediction_history, window_size=5)
                        if smoothed:
                            res = smoothed
                            score = raw_score  # Keep original score for display
                        else:
                            res = raw_res
                            score = raw_score
                    else:
                        res = raw_res
                        score = raw_score
                    
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
                        
                        # Log for confusion analysis
                        confusion_log.append({
                            'predicted': res,
                            'top_3': [p[0] for p in top_predictions[:3]],
                            'scores': [p[1] for p in top_predictions[:3]]
                        })
                        
                        consecutive = 0
                    
                    mem = res
                except Exception as e:
                    pass
            
            i += 1
            
            # Draw the capture box
            box_color = (0, 255, 0) if consecutive < 2 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
            
            # Show preview
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
            
            # Smoothing status
            smooth_text = "ON" if use_smoothing else "OFF"
            smooth_color = (0, 255, 0) if use_smoothing else (0, 0, 255)
            cv2.putText(frame, f"Smoothing: {smooth_text}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, smooth_color, 2)
            
            # Top prediction
            if top_predictions:
                res_display = top_predictions[0][0]
                score_display = top_predictions[0][1]
                
                # Highlight if it's a commonly confused sign
                if res_display in SIMILAR_SIGNS:
                    pred_color = (0, 165, 255)  # Orange for confused signs
                    cv2.putText(frame, "⚠ Similar signs exist", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                else:
                    pred_color = (0, 255, 255)
                
                cv2.putText(frame, f"Sign: {res_display.upper()}", (10, 125), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 3)
                cv2.putText(frame, f"Confidence: {score_display*100:.1f}%", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
                
                # Show top 3 predictions
                cv2.putText(frame, "Top 3:", (10, 190), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                y_off = 215
                for idx, (label, sc) in enumerate(top_predictions[:3]):
                    color = (0, 255, 0) if idx == 0 else (150, 150, 150)
                    text = f"{idx+1}. {label.upper()}: {sc*100:.1f}%"
                    cv2.putText(frame, text, (10, y_off), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_off += 25
            
            # Instructions
            cv2.putText(frame, "Put hand in GREEN BOX", (box_x, box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Recognition - IMPROVED', frame)
            
            # Sequence window
            img_sequence = np.zeros((500, 1200, 3), np.uint8)
            cv2.putText(img_sequence, "Recognized Sequence:", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            # Instructions
            cv2.putText(img_sequence, "SPACE=Toggle Smoothing | S=Save | C=Clear | M=Manual | ESC=Exit", 
                       (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
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
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                use_smoothing = not use_smoothing
                print(f"Temporal smoothing: {'ON' if use_smoothing else 'OFF'}")
            elif key == ord('s') or key == ord('S'):
                # Save current frame
                filename = f"debug_frame_{save_counter}.jpg"
                cv2.imwrite(filename, hand_img)
                print(f"Saved: {filename}")
                if top_predictions:
                    print(f"  Predictions: {', '.join([f'{p[0]}({p[1]*100:.1f}%)' for p in top_predictions[:3]])}")
                save_counter += 1
            elif key == ord('c') or key == ord('C'):
                # Clear sequence
                sequence = ''
                print("Sequence cleared")
            elif key == ord('m') or key == ord('M'):
                # Manual correction
                print("\nManual correction mode:")
                print("Current sequence:", sequence.upper())
                correction = input("Enter correct letter (or ENTER to cancel): ").strip().lower()
                if correction and len(correction) == 1:
                    sequence += correction
                    print(f"Added '{correction.upper()}' to sequence")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Session ended")
    print("="*70)
    print(f"\nFinal sequence: {sequence.upper()}")
    print(f"Total characters: {len(sequence)}")
    
    # Show confusion statistics
    if confusion_log:
        print(f"\nConfusion Analysis ({len(confusion_log)} predictions):")
        confused_signs = {}
        for entry in confusion_log:
            pred = entry['predicted']
            if pred in SIMILAR_SIGNS:
                if pred not in confused_signs:
                    confused_signs[pred] = []
                confused_signs[pred].append(entry['top_3'])
        
        if confused_signs:
            print("\nSigns with similar alternatives detected:")
            for sign, alternatives in confused_signs.items():
                print(f"  {sign.upper()}: commonly confused with {SIMILAR_SIGNS[sign]}")


if __name__ == "__main__":
    main()
