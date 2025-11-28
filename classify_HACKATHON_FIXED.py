"""
HACKATHON Sign Language Classifier - FIXED VERSION
Based on working SIMPLE version
"""

import sys
import os
import cv2
import numpy as np
from collections import deque, Counter

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class TemporalSmoother:
    """Smooth predictions over time"""
    def __init__(self, window_size=12, min_confidence=0.30):
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.buffer = deque(maxlen=window_size)
    
    def add(self, label, confidence):
        if confidence > 0.15:
            self.buffer.append((label, confidence))
    
    def get_stable(self):
        if len(self.buffer) < 8:
            return None, 0.0
        
        labels = [p[0] for p in self.buffer]
        counter = Counter(labels)
        top_label, count = counter.most_common(1)[0]
        
        agreement = count / len(self.buffer)
        if agreement < 0.50:
            return None, 0.0
        
        confidences = [p[1] for p in self.buffer if p[0] == top_label]
        avg_conf = np.mean(confidences)
        
        if avg_conf < self.min_confidence:
            return None, avg_conf
        
        return top_label, avg_conf
    
    def clear(self):
        self.buffer.clear()

def predict(image_data, sess, softmax_tensor, label_lines):
    """Get prediction from model"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results

print("="*70)
print("HACKATHON Sign Language Classifier")
print("="*70)
print("\nLoading model...")

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("✓ Model loaded")
print(f"✓ {len(label_lines)} classes")

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("1. Put your hand in the GREEN BOX")
print("2. Make a sign and hold steady")
print("3. Wait for STABLE prediction (green text)")
print("4. Press SPACE to add the letter")
print("5. Press C to clear, ESC to exit")
print("="*70)

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    # Open camera 0
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    sequence = ''
    smoother = TemporalSmoother()
    frame_count = 0
    
    # Cache for display
    stable_label = None
    stable_conf = 0.0
    top_5 = []
    
    print("\n✓ Camera started! Show your hand...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Box in center
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        hand_img = frame[y1:y2, x1:x2]
        
        # Predict every 2 frames
        if frame_count % 2 == 0:
            try:
                hand_resized = cv2.resize(hand_img, (299, 299))
                image_data = cv2.imencode('.jpg', hand_resized, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                
                top_5 = predict(image_data, sess, softmax_tensor, label_lines)
                top_label, top_conf = top_5[0]
                
                smoother.add(top_label, top_conf)
                stable_label, stable_conf = smoother.get_stable()
                
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_count += 1
        
        # Draw UI
        # Top 5 predictions (small)
        if top_5:
            y_off = 30
            for i, (label, score) in enumerate(top_5):
                color = (100, 200, 255) if i == 0 else (150, 150, 150)
                text = f"{label.upper()}: {score*100:.0f}%"
                cv2.putText(frame, text, (20, y_off), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_off += 30
        
        # Stable prediction (BIG)
        if stable_label and stable_label not in ['nothing']:
            # Big letter in center
            text_size = cv2.getTextSize(stable_label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 8, 10)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            cv2.putText(frame, stable_label.upper(), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 10)
            
            # Confidence
            cv2.putText(frame, f"STABLE: {stable_conf*100:.0f}%", (20, h - 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(frame, "Press SPACE to add", (20, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Hold steady...", (20, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2)
        
        # Box
        box_color = (0, 255, 0) if stable_label and stable_label not in ['nothing'] else (100, 100, 100)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        
        # Instructions
        cv2.putText(frame, "SPACE=Add | C=Clear | ESC=Exit", (w - 450, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow('HACKATHON - Sign Language', frame)
        
        # Sequence window
        img_seq = np.zeros((200, 1200, 3), np.uint8)
        cv2.putText(img_seq, "Recognized Sequence:", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
        
        display_text = sequence.upper() if sequence else "(empty)"
        cv2.putText(img_seq, display_text, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.imshow('Sequence', img_seq)
        
        # Keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE
            if stable_label and stable_label not in ['nothing']:
                if stable_label == 'space':
                    sequence += ' '
                    print(f"✓ Added: SPACE")
                elif stable_label == 'del':
                    if sequence:
                        sequence = sequence[:-1]
                        print(f"✓ Deleted last character")
                else:
                    sequence += stable_label
                    print(f"✓ Added: {stable_label.upper()} ({stable_conf*100:.0f}%)")
                
                smoother.clear()
            else:
                print("✗ No stable prediction")
        elif key == ord('c') or key == ord('C'):
            sequence = ''
            smoother.clear()
            print("✓ Cleared")
    
    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print("Demo ended")
print("="*70)
print(f"Final sequence: {sequence.upper()}")
print("Good luck! 🚀")
