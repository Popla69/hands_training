"""
HACKATHON AUTO Version
Automatically adds letters when stable (no SPACE key needed)
Better for smooth demo flow
"""

import os
import cv2
import numpy as np
from collections import deque, Counter
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

SIMILAR_SIGNS = {
    'a': ['e', 's', 'm', 'n', 't'],
    'e': ['a', 's'],
    's': ['a', 'e', 't'],
    'k': ['v', 'p'],
    'v': ['k', 'u', 'n'],
}

MOTION_SIGNS = ['j', 'z']

class AutoAdder:
    """Automatically add letters when stable"""
    
    def __init__(self, hold_time=2.0, cooldown=1.5):
        self.hold_time = hold_time
        self.cooldown = cooldown
        self.prediction_buffer = deque(maxlen=20)
        self.last_added_time = 0
        self.current_stable = None
        self.stable_start_time = None
    
    def add_prediction(self, label, confidence):
        """Add prediction"""
        if confidence > 0.25:
            self.prediction_buffer.append((label, confidence))
    
    def should_add(self, current_time):
        """Check if should auto-add"""
        # Check cooldown
        if current_time - self.last_added_time < self.cooldown:
            return None, 0.0, f"Cooldown ({self.cooldown - (current_time - self.last_added_time):.1f}s)"
        
        if len(self.prediction_buffer) < 15:
            return None, 0.0, "Analyzing..."
        
        # Get most common
        labels = [p[0] for p in self.prediction_buffer]
        counter = Counter(labels)
        most_common = counter.most_common(1)[0]
        top_label, top_count = most_common
        
        agreement = top_count / len(self.prediction_buffer)
        
        # Calculate confidence
        top_confidences = [p[1] for p in self.prediction_buffer if p[0] == top_label]
        avg_confidence = np.mean(top_confidences)
        
        # Check if stable enough
        if agreement < 0.65 or avg_confidence < 0.40:
            self.current_stable = None
            self.stable_start_time = None
            return None, avg_confidence, f"Not stable ({agreement*100:.0f}%)"
        
        # Check if this is a new stable prediction
        if self.current_stable != top_label:
            self.current_stable = top_label
            self.stable_start_time = current_time
            return None, avg_confidence, f"Hold steady... ({self.hold_time:.1f}s)"
        
        # Check if held long enough
        hold_duration = current_time - self.stable_start_time
        if hold_duration < self.hold_time:
            remaining = self.hold_time - hold_duration
            return None, avg_confidence, f"Hold steady... ({remaining:.1f}s)"
        
        # Ready to add!
        return top_label, avg_confidence, "Adding..."
    
    def mark_added(self, current_time):
        """Mark that letter was added"""
        self.last_added_time = current_time
        self.prediction_buffer.clear()
        self.current_stable = None
        self.stable_start_time = None


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Get predictions"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def draw_ui(frame, pred_to_add, confidence, status, top_5, sequence, fps, hold_progress):
    """Draw UI"""
    h, w, _ = frame.shape
    
    # Overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 280), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "Sign Language Recognition", (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
    cv2.putText(frame, "AUTO MODE", (10, 65), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Status
    status_color = (0, 255, 0) if pred_to_add else (150, 150, 150)
    cv2.putText(frame, status, (10, 105), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    
    # Current prediction
    if pred_to_add and pred_to_add not in ['nothing']:
        cv2.putText(frame, f"Letter: {pred_to_add.upper()}", (10, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence*100:.0f}%", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress bar
        if hold_progress > 0:
            bar_width = 400
            bar_height = 20
            filled_width = int(bar_width * hold_progress)
            cv2.rectangle(frame, (10, 200), (10 + bar_width, 200 + bar_height), (100, 100, 100), 2)
            cv2.rectangle(frame, (10, 200), (10 + filled_width, 200 + bar_height), (0, 255, 0), -1)
    
    # Top 5 (small)
    if top_5:
        y_off = 30
        for i, (label, score) in enumerate(top_5):
            color = (0, 255, 0) if i == 0 else (180, 180, 180)
            text = f"{label.upper()}: {score*100:.0f}%"
            cv2.putText(frame, text, (w - 250, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_off += 25
    
    # Instructions
    cv2.putText(frame, "C=Clear | ESC=Exit", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # HUGE letter
    if pred_to_add and pred_to_add not in ['nothing']:
        text_size = cv2.getTextSize(pred_to_add.upper(), cv2.FONT_HERSHEY_SIMPLEX, 8, 10)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        cv2.putText(frame, pred_to_add.upper(), (text_x + 5, text_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 0), 10)
        cv2.putText(frame, pred_to_add.upper(), (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 10)


def main():
    print("="*70)
    print("HACKATHON AUTO MODE - Sign Language Classifier")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Automatic letter addition (no SPACE key)")
    print("  ✓ Hold sign for 2 seconds to add")
    print("  ✓ 1.5 second cooldown between letters")
    print("  ✓ Stable predictions")
    print("\nLoading model...")
    
    label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]
    
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    print("✓ Model loaded")
    
    print("\n" + "="*70)
    print("INSTRUCTIONS:")
    print("="*70)
    print("1. Put hand in green box")
    print("2. Make a sign and hold STEADY for 2 seconds")
    print("3. Letter adds automatically")
    print("4. Wait 1.5 seconds before next letter")
    print("5. Press C to clear sequence")
    print("="*70)
    
    input("\nPress ENTER to start...")
    
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        if not cap.isOpened():
            print("ERROR: Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        sequence = ''
        auto_adder = AutoAdder(hold_time=2.0, cooldown=1.5)
        
        fps_list = deque(maxlen=30)
        prev_time = time.time()
        total_added = 0
        
        print("\n✓ Camera started!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
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
            
            try:
                hand_resized = cv2.resize(hand_img, (299, 299))
                image_data = cv2.imencode('.jpg', hand_resized, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                top_5 = predict_sign(image_data, sess, softmax_tensor, label_lines)
                
                top_label, top_conf = top_5[0]
                auto_adder.add_prediction(top_label, top_conf)
                
                # Check if should add
                pred_to_add, confidence, status = auto_adder.should_add(current_time)
                
                # Calculate hold progress
                hold_progress = 0.0
                if auto_adder.stable_start_time:
                    elapsed = current_time - auto_adder.stable_start_time
                    hold_progress = min(elapsed / auto_adder.hold_time, 1.0)
                
                # Auto-add if ready
                if pred_to_add and pred_to_add not in ['nothing', 'j', 'z']:
                    if pred_to_add == 'space':
                        sequence += ' '
                        print(f"✓ Added: SPACE")
                    elif pred_to_add == 'del':
                        if sequence:
                            sequence = sequence[:-1]
                            print(f"✓ Deleted")
                    else:
                        sequence += pred_to_add
                        print(f"✓ Added: {pred_to_add.upper()} ({confidence*100:.0f}%)")
                        total_added += 1
                    
                    auto_adder.mark_added(current_time)
                
                draw_ui(frame, pred_to_add, confidence, status, top_5, 
                       sequence, avg_fps, hold_progress)
                
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pred_to_add = None
                hold_progress = 0
            
            # Box
            box_color = (0, 255, 0) if pred_to_add else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            cv2.imshow('Sign Language - AUTO MODE', frame)
            
            # Sequence
            img_seq = np.zeros((250, 1200, 3), np.uint8)
            cv2.putText(img_seq, "Sequence:", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
            
            display_text = sequence.upper()
            if len(display_text) > 40:
                line1 = display_text[:40]
                line2 = display_text[40:80]
                cv2.putText(img_seq, line1, (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(img_seq, line2, (20, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            else:
                cv2.putText(img_seq, display_text, (20, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            cv2.putText(img_seq, f"Total: {total_added} letters", (20, 220), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            
            cv2.imshow('Sequence', img_seq)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == ord('c') or key == ord('C'):
                sequence = ''
                auto_adder.prediction_buffer.clear()
                auto_adder.current_stable = None
                print("✓ Cleared")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print(f"Final: {sequence.upper()}")
    print(f"Total: {total_added} letters")
    print("Good luck! 🚀")


if __name__ == "__main__":
    main()
