"""
WORKING Sign Language Classifier
Shows camera immediately, loads model in background
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
        print("\n  ✓✓✓ MODEL READY! ✓✓✓")
        print("  Note: First prediction will be slow (10-30 sec)\n")
        
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

class Smoother:
    def __init__(self):
        self.buffer = deque(maxlen=10)
    
    def add(self, label, conf):
        if conf > 0.20:
            self.buffer.append((label, conf))
    
    def get_stable(self):
        if len(self.buffer) < 6:
            return None, 0.0
        
        labels = [p[0] for p in self.buffer]
        counter = Counter(labels)
        top_label, count = counter.most_common(1)[0]
        
        if count < 4:
            return None, 0.0
        
        confs = [p[1] for p in self.buffer if p[0] == top_label]
        avg_conf = np.mean(confs)
        
        if avg_conf < 0.35:
            return None, avg_conf
        
        return top_label, avg_conf
    
    def clear(self):
        self.buffer.clear()

print("="*70)
print("WORKING Sign Language Classifier")
print("="*70)
print("\n[1/2] Starting camera...")

# Open camera FIRST
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✓ Camera started!")

print("\n[2/2] Loading model in background...")
# Start loading model in background thread
model_thread = threading.Thread(target=load_model_background, daemon=True)
model_thread.start()

print("\n" + "="*70)
print("Camera is running! Model loading in background...")
print("="*70)

smoother = Smoother()
sequence = ''
frame_count = 0
top_5 = []
stable_label = None
stable_conf = 0.0
first_prediction_done = False

# FPS
fps_list = deque(maxlen=30)
prev_time = time.time()

read_failures = 0
max_failures = 30  # Allow 30 failed reads before giving up

while True:
    ret, frame = cap.read()
    if not ret:
        read_failures += 1
        if read_failures > max_failures:
            print("\n✗ Camera read failed too many times")
            break
        time.sleep(0.1)
        continue
    
    read_failures = 0  # Reset on successful read
    
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
    
    # Predict if model is loaded
    if model_loaded and frame_count % 2 == 0:
        try:
            if not first_prediction_done:
                print("  Making first prediction (this will be slow)...")
            
            hand_resized = cv2.resize(hand_img, (299, 299))
            image_data = cv2.imencode('.jpg', hand_resized, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            
            top_5 = predict_sign(image_data)
            
            if not first_prediction_done and top_5:
                first_prediction_done = True
                print("  ✓ First prediction complete! Now running smoothly.\n")
            
            if top_5:
                top_label, top_conf = top_5[0]
                smoother.add(top_label, top_conf)
                stable_label, stable_conf = smoother.get_stable()
        except Exception as e:
            if not first_prediction_done:
                print(f"  ✗ Prediction error: {e}")
            pass
    
    frame_count += 1
    
    # Draw UI
    # Status bar
    status_color = (0, 255, 0) if model_loaded else (0, 165, 255)
    status_text = "READY" if model_loaded else "Loading model... (check console)"
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.putText(frame, f"Status: {status_text}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"FPS: {avg_fps:.0f}", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Model loading indicator
    if not model_loaded:
        elapsed = int(time.time() - prev_time) % 4
        dots = "." * elapsed
        cv2.putText(frame, f"Please wait{dots}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Top 5
    if top_5:
        y_off = 100
        for i, (label, score) in enumerate(top_5):
            color = (0, 255, 0) if i == 0 else (150, 150, 150)
            text = f"{label.upper()}: {score*100:.0f}%"
            cv2.putText(frame, text, (10, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_off += 35
    
    # Stable prediction
    if stable_label and stable_label not in ['nothing']:
        text_size = cv2.getTextSize(stable_label.upper(), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 8, 10)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(frame, stable_label.upper(), (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 10)
        
        cv2.putText(frame, f"STABLE: {stable_conf*100:.0f}%", (10, h - 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, "Press SPACE to add", (10, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Box
    box_color = (0, 255, 0) if stable_label else (100, 100, 100)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
    
    # Instructions
    cv2.putText(frame, "SPACE=Add | C=Clear | ESC=Exit", (w - 450, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.imshow('Sign Language - WORKING', frame)
    
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
        if stable_label and stable_label not in ['nothing']:
            if stable_label == 'space':
                sequence += ' '
            elif stable_label == 'del':
                sequence = sequence[:-1] if sequence else ''
            else:
                sequence += stable_label
            print(f"✓ Added: {stable_label.upper()}")
            smoother.clear()
    elif key == ord('c') or key == ord('C'):
        sequence = ''
        smoother.clear()
        print("✓ Cleared")

cap.release()
cv2.destroyAllWindows()
if sess:
    sess.close()

print("\n" + "="*70)
print(f"Final sequence: {sequence.upper()}")
print("="*70)
