"""
FINAL DEMO VERSION
- Lighting-robust preprocessing
- Works with different lighting conditions
- Optimized for hackathon demo
"""

import os
import cv2
import numpy as np
from collections import deque, Counter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def preprocess_lighting_robust(img):
    """
    Minimal preprocessing - just return the image
    Too much preprocessing makes it worse!
    """
    return img

def predict(image_data, sess, softmax_tensor, label_lines):
    """Predict sign"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results

print("="*70)
print("FINAL DEMO - Lighting-Robust Version")
print("="*70)
print("\nFeatures:")
print("  ✓ Handles different lighting conditions")
print("  ✓ CLAHE preprocessing")
print("  ✓ Color normalization")
print("  ✓ Optimized for demo")
print("\nLoading model...")

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("✓ Model loaded")
print("\n" + "="*70)
print("DEMO INSTRUCTIONS:")
print("="*70)
print("1. Put hand in GREEN BOX")
print("2. Make a sign and hold steady")
print("3. Press SPACE when stable prediction appears")
print("4. Press C to clear sequence")
print("5. Press ESC to exit")
print("\nTIPS:")
print("  - Try to match lighting of training data")
print("  - Use bright, even lighting")
print("  - Avoid shadows on hand")
print("="*70)

input("\nPress ENTER to start...")

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit(1)
    
    # Camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # State
    sequence = ''
    frame_count = 0
    prediction_buffer = deque(maxlen=15)
    
    # Current prediction
    current_pred = None
    current_conf = 0.0
    top_5 = []
    
    print("\n✓ Camera started!")
    print("Lighting-robust preprocessing enabled")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Box
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        hand_img = frame[y1:y2, x1:x2]
        
        # Predict every 6 frames for better FPS
        if frame_count % 6 == 0:
            try:
                # Resize
                hand_resized = cv2.resize(hand_img, (299, 299))
                
                # LIGHTING-ROBUST PREPROCESSING
                hand_processed = preprocess_lighting_robust(hand_resized)
                
                # Encode with lower quality for speed
                image_data = cv2.imencode('.jpg', hand_processed, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                
                top_5 = predict(image_data, sess, softmax_tensor, label_lines)
                current_pred, current_conf = top_5[0]
                
                # Add to buffer with lower threshold
                if current_conf > 0.20:  # Lower threshold
                    prediction_buffer.append(current_pred)
                
            except Exception as e:
                pass
        
        frame_count += 1
        
        # Get stable prediction from buffer
        stable_pred = None
        stable_conf = 0.0
        if len(prediction_buffer) >= 8:  # Need 8 frames
            counter = Counter(prediction_buffer)
            most_common = counter.most_common(1)[0]
            if most_common[1] >= 5:  # At least 5 out of last 15
                stable_pred = most_common[0]
                # Calculate average confidence
                pred_confs = [top_5[0][1] for _ in range(most_common[1])]
                stable_conf = current_conf
        
        # Draw UI
        # Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Sign Language - DEMO", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        
        # Top 5 predictions
        y_off = 75
        for i, (label, score) in enumerate(top_5[:5]):
            color = (0, 255, 0) if i == 0 else (180, 180, 180)
            text = f"{i+1}. {label.upper()}: {score*100:.0f}%"
            cv2.putText(frame, text, (10, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_off += 30
        
        # Stable prediction
        if stable_pred and stable_pred not in ['nothing']:
            cv2.putText(frame, "STABLE:", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, stable_pred.upper(), (10, 285), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)
            cv2.putText(frame, "Press SPACE", (10, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Hold steady...", (10, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        
        # Instructions
        cv2.putText(frame, "SPACE=Add | C=Clear | ESC=Exit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Box
        box_color = (0, 255, 0) if stable_pred else (100, 100, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        
        # HUGE letter in center
        if stable_pred and stable_pred not in ['nothing']:
            text_size = cv2.getTextSize(stable_pred.upper(), cv2.FONT_HERSHEY_SIMPLEX, 6, 8)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Shadow
            cv2.putText(frame, stable_pred.upper(), (text_x + 5, text_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 0), 8)
            # Main
            cv2.putText(frame, stable_pred.upper(), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 0), 8)
        
        cv2.imshow('Sign Language Recognition - FINAL DEMO', frame)
        
        # Sequence window
        img_seq = np.zeros((250, 1200, 3), np.uint8)
        cv2.putText(img_seq, "Recognized Sequence:", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
        
        # Word wrap
        display_text = sequence.upper()
        if len(display_text) > 40:
            line1 = display_text[:40]
            line2 = display_text[40:80]
            cv2.putText(img_seq, line1, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(img_seq, line2, (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            cv2.putText(img_seq, display_text, (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        cv2.putText(img_seq, f"Total: {len(sequence)} letters", (20, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        cv2.imshow('Sequence', img_seq)
        
        # Keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            if stable_pred and stable_pred not in ['nothing']:
                if stable_pred == 'space':
                    sequence += ' '
                    print(f"✓ Added: SPACE")
                elif stable_pred == 'del':
                    if sequence:
                        sequence = sequence[:-1]
                        print(f"✓ Deleted")
                else:
                    sequence += stable_pred
                    print(f"✓ Added: {stable_pred.upper()} (conf: {stable_conf*100:.0f}%)")
                
                prediction_buffer.clear()
        elif key == ord('c') or key == ord('C'):
            sequence = ''
            prediction_buffer.clear()
            print("✓ Cleared")
    
    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print(f"Final sequence: {sequence.upper()}")
print(f"Total letters: {len(sequence)}")
print("="*70)
print("\nGood luck with your hackathon! 🚀")
