"""
Clean webcam classifier - Based on WORKING version
Manual capture with SPACE key for better control
"""

import os
import cv2
import numpy as np
from collections import deque, Counter
import threading
import time

# Global variables for model
model_loaded = False
sess = None
softmax_tensor = None
label_lines = []

def load_model_background():
    """Load TensorFlow model in background thread"""
    global model_loaded, sess, softmax_tensor, label_lines
    
    try:
        print("  [1/4] Loading TensorFlow...")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        print("  ✓ TensorFlow imported")
        
        print("  [2/4] Loading labels...")
        label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]
        print(f"  ✓ Loaded {len(label_lines)} labels")
        
        print("  [3/4] Loading graph...")
        with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
        print("  ✓ Graph loaded")
        
        print("  [4/4] Starting session...")
        sess = tf.Session()
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        print("  ✓ Session started")
        
        model_loaded = True
        print("\n  ✓✓✓ MODEL READY! ✓✓✓\n")
        
    except Exception as e:
        print(f"\n  ✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()

def predict_sign(image_data):
    """Predict sign from image"""
    if not model_loaded:
        return []
    
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-5:][::-1]
    
    results = []
    for node_id in top_k:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results

print("="*70)
print("CLEAN Webcam Classifier - Manual Capture Mode")
print("="*70)
print("\n[1/2] Starting camera...")

# Open camera FIRST
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("✓ Camera started!")

print("\n[2/2] Loading model in background...")
model_thread = threading.Thread(target=load_model_background, daemon=True)
model_thread.start()

print("\n" + "="*70)
print("Camera is running! Model loading in background...")
print("="*70)
print("\nHow to use:")
print("  1. Position your hand in the box")
print("  2. Make a sign and hold steady")
print("  3. Press SPACE to capture and predict")
print("  4. System captures 3 frames and votes")
print("\nControls:")
print("  - SPACE: Capture and predict")
print("  - C: Clear sequence")
print("  - ESC: Exit")
print("="*70)

sequence = ''
frame_count = 0
capturing = False
capture_frames = []
first_prediction_done = False

# FPS
fps_list = deque(maxlen=30)
prev_time = time.time()

read_failures = 0
max_failures = 30

while True:
    ret, frame = cap.read()
    if not ret:
        read_failures += 1
        if read_failures > max_failures:
            print("\n✗ Camera read failed too many times")
            break
        time.sleep(0.1)
        continue
    
    read_failures = 0
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0
    fps_list.append(fps)
    avg_fps = np.mean(fps_list)
    prev_time = current_time
    
    # Box
    box_size = 400
    x1 = (w - box_size) // 2
    y1 = (h - box_size) // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    
    hand_img = frame[y1:y2, x1:x2]
    
    # Capture mode - capture 3 frames when SPACE is pressed
    if capturing and model_loaded:
        if len(capture_frames) < 3:
            try:
                if not first_prediction_done:
                    print("  Making first prediction (this will be slow)...")
                
                hand_resized = cv2.resize(hand_img, (299, 299))
                image_data = cv2.imencode('.jpg', hand_resized, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
                
                predictions = predict_sign(image_data)
                
                if not first_prediction_done and predictions:
                    first_prediction_done = True
                    print("  ✓ First prediction complete!\n")
                
                if predictions:
                    capture_frames.append(predictions[0])
                    print(f"  Frame {len(capture_frames)}/3: {predictions[0][0].upper()} ({predictions[0][1]*100:.1f}%)")
            except Exception as e:
                print(f"  Error: {e}")
        
        # Done capturing
        if len(capture_frames) >= 3:
            # Vote on result
            pred_labels = [p[0] for p in capture_frames]
            pred_counter = Counter(pred_labels)
            
            most_common = pred_counter.most_common(1)[0]
            final_pred = most_common[0]
            vote_count = most_common[1]
            
            # Average confidence
            winning_confs = [p[1] for p in capture_frames if p[0] == final_pred]
            avg_conf = np.mean(winning_confs)
            
            # Require 2/3 votes
            if vote_count >= 2:
                print(f"\n✓ Result: {final_pred.upper()} ({vote_count}/3 votes, {avg_conf*100:.1f}% conf)")
                
                # Add to sequence
                if final_pred == 'space':
                    sequence += ' '
                    print("  Added: SPACE")
                elif final_pred == 'del':
                    sequence = sequence[:-1] if sequence else ''
                    print("  Deleted last character")
                elif final_pred != 'nothing':
                    sequence += final_pred
                    print(f"  Added: {final_pred.upper()}")
                else:
                    print("  Ignored: nothing")
            else:
                print(f"\n✗ Rejected: {final_pred.upper()} (only {vote_count}/3 votes)")
            
            # Reset
            capturing = False
            capture_frames = []
    
    frame_count += 1
    
    # Draw UI
    # Status bar
    status_color = (0, 255, 0) if model_loaded else (0, 165, 255)
    status_text = "READY - Press SPACE to capture" if model_loaded else "Loading model..."
    
    if capturing:
        status_text = f"CAPTURING... {len(capture_frames)}/3"
        status_color = (0, 255, 255)
    
    cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, f"FPS: {avg_fps:.0f}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Box
    box_color = (0, 255, 255) if capturing else (0, 255, 0) if model_loaded else (100, 100, 100)
    box_thickness = 5 if capturing else 3
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
    
    # Instructions
    cv2.putText(frame, "SPACE=Capture | C=Clear | ESC=Exit", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    cv2.imshow('Sign Language - CLEAN', frame)
    
    # Sequence window
    img_seq = np.zeros((150, 1000, 3), np.uint8)
    cv2.putText(img_seq, "Sequence:", (10, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
    display_text = sequence.upper() if sequence else "(empty)"
    cv2.putText(img_seq, display_text, (10, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.imshow('Sequence', img_seq)
    
    # Keys
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # SPACE
        if model_loaded and not capturing:
            print("\n" + "="*50)
            print("CAPTURING 3 FRAMES...")
            print("="*50)
            capturing = True
            capture_frames = []
    elif key == ord('c') or key == ord('C'):
        sequence = ''
        print("✓ Cleared")

cap.release()
cv2.destroyAllWindows()
if sess:
    sess.close()

print("\n" + "="*70)
print(f"Final sequence: {sequence.upper()}")
print("="*70)
