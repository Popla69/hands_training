"""
ULTIMATE Sign Language Recognition System
- Hand detection with green box
- 5-frame accumulation for stable predictions
- Shows most confident sign for 5 seconds
- 20+ FPS performance
- Camera index 1
"""

import os
import cv2
import numpy as np
from collections import deque, Counter
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class SignRecognizer:
    """Handles sign language recognition with temporal smoothing"""
    
    def __init__(self, sess, softmax_tensor, label_lines):
        self.sess = sess
        self.softmax_tensor = softmax_tensor
        self.label_lines = label_lines
        self.frame_buffer = deque(maxlen=5)  # 5 frames
        self.current_sign = None
        self.current_confidence = 0.0
        self.display_until = 0
        self.display_duration = 5.0  # 5 seconds
    
    def predict_single(self, image_data):
        """Get prediction for single frame"""
        predictions = self.sess.run(self.softmax_tensor, 
                                    {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        results = []
        for node_id in top_k[:5]:
            results.append((self.label_lines[node_id], predictions[0][node_id]))
        
        return results
    
    def add_frame(self, hand_img):
        """Add frame to buffer and get prediction"""
        try:
            # Resize to model input size
            hand_resized = cv2.resize(hand_img, (299, 299), 
                                     interpolation=cv2.INTER_LINEAR)
            
            # Encode as JPEG
            image_data = cv2.imencode('.jpg', hand_resized, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            
            # Get predictions
            predictions = self.predict_single(image_data)
            top_label, top_conf = predictions[0]
            
            # Add to buffer
            if top_conf > 0.20 and top_label not in ['nothing']:
                self.frame_buffer.append((top_label, top_conf))
            
            return predictions
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return []
    
    def get_stable_prediction(self):
        """Get stable prediction from 5 frames"""
        current_time = time.time()
        
        # If we're still displaying a previous sign
        if current_time < self.display_until:
            return self.current_sign, self.current_confidence, True
        
        # Need at least 5 frames
        if len(self.frame_buffer) < 5:
            return None, 0.0, False
        
        # Count occurrences
        labels = [p[0] for p in self.frame_buffer]
        counter = Counter(labels)
        
        # Get most common
        most_common = counter.most_common(1)
        if not most_common:
            return None, 0.0, False
        
        top_label, count = most_common[0]
        
        # Need at least 3 out of 5 frames agreeing
        if count < 3:
            return None, 0.0, False
        
        # Calculate average confidence
        confidences = [p[1] for p in self.frame_buffer if p[0] == top_label]
        avg_conf = np.mean(confidences)
        
        # Need minimum confidence
        if avg_conf < 0.35:
            return None, avg_conf, False
        
        # New stable prediction found!
        self.current_sign = top_label
        self.current_confidence = avg_conf
        self.display_until = current_time + self.display_duration
        self.frame_buffer.clear()  # Reset for next detection
        
        print(f"✓ Detected: {top_label.upper()} ({avg_conf*100:.0f}%)")
        
        return top_label, avg_conf, True
    
    def reset(self):
        """Reset the recognizer"""
        self.frame_buffer.clear()
        self.current_sign = None
        self.current_confidence = 0.0
        self.display_until = 0


def main():
    print("="*70)
    print("ULTIMATE Sign Language Recognition System")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ 5-frame accumulation for accuracy")
    print("  ✓ 5-second display of detected sign")
    print("  ✓ 20+ FPS performance")
    print("  ✓ Green box hand detection")
    print("\n[1/2] Loading sign language model...")
    
    # Load model
    label_lines = [line.rstrip() for line in 
                   tf.gfile.GFile("logs/trained_labels.txt")]
    
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    print(f"✓ Model loaded ({len(label_lines)} classes)")
    
    print("\n[2/2] Starting camera...")
    
    sess = tf.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    # Initialize recognizer
    recognizer = SignRecognizer(sess, softmax_tensor, label_lines)
    
    # Open camera 1
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera 1 failed, trying camera 0...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        sess.close()
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("✓ Camera started!")
    
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("="*70)
    print("1. Put your hand in the GREEN BOX")
    print("2. Make a sign and hold steady")
    print("3. System will detect after 5 frames")
    print("4. Sign displays for 5 seconds")
    print("5. Press R to reset, ESC to exit")
    print("="*70 + "\n")
    
    # FPS tracking
    fps_list = deque(maxlen=30)
    prev_time = time.time()
    
    # Prediction tracking
    frame_count = 0
    PREDICT_EVERY = 2  # Predict every 2 frames for 15 FPS predictions
    
    # Display cache
    top_5 = []
    stable_sign = None
    stable_conf = 0.0
    is_displaying = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0
        fps_list.append(fps)
        avg_fps = np.mean(fps_list)
        prev_time = current_time
        
        # Green box in center
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Extract hand region
        hand_img = frame[y1:y2, x1:x2]
        
        # Run prediction every N frames
        if frame_count % PREDICT_EVERY == 0:
            top_5 = recognizer.add_frame(hand_img)
            stable_sign, stable_conf, is_displaying = recognizer.get_stable_prediction()
        
        frame_count += 1
        
        # Draw UI
        # Semi-transparent overlay for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Sign Language Recognition", (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {avg_fps:.0f}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Buffer status
        buffer_size = len(recognizer.frame_buffer)
        buffer_text = f"Frames: {buffer_size}/5"
        cv2.putText(frame, buffer_text, (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Top 5 predictions (small, on right)
        if top_5:
            y_off = 30
            for i, (label, score) in enumerate(top_5):
                color = (0, 255, 0) if i == 0 else (180, 180, 180)
                text = f"{label.upper()}: {score*100:.0f}%"
                cv2.putText(frame, text, (w - 280, y_off), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_off += 30
        
        # Display stable sign (BIG)
        if is_displaying and stable_sign:
            # Calculate time remaining
            time_left = recognizer.display_until - time.time()
            
            # Huge letter in center
            text_size = cv2.getTextSize(stable_sign.upper(), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 10, 15)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Shadow
            cv2.putText(frame, stable_sign.upper(), (text_x + 5, text_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 15)
            # Main text
            cv2.putText(frame, stable_sign.upper(), (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 255, 0), 15)
            
            # Confidence and timer
            info_text = f"{stable_conf*100:.0f}% - {time_left:.1f}s"
            cv2.putText(frame, info_text, (10, h - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            # Waiting message
            if buffer_size > 0:
                cv2.putText(frame, "Analyzing...", (10, h - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            else:
                cv2.putText(frame, "Show hand in green box", (10, h - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
        
        # Draw green box
        box_color = (0, 255, 0) if is_displaying else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
        
        # Instructions
        cv2.putText(frame, "R=Reset | ESC=Exit", (w - 300, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        cv2.imshow('ULTIMATE Sign Language Recognition', frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('r') or key == ord('R'):
            recognizer.reset()
            print("✓ Reset")
    
    cap.release()
    cv2.destroyAllWindows()
    sess.close()
    
    print("\n" + "="*70)
    print("System stopped")
    print("="*70)


if __name__ == "__main__":
    main()
