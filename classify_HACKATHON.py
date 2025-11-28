"""
HACKATHON-READY Sign Language Classifier
- Stable predictions (no blinking)
- Temporal smoothing over multiple frames
- High confidence threshold
- Manual confirmation with SPACE key
- Professional UI for demo
"""

import os
import cv2
import numpy as np
from collections import deque, Counter
import time

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Similar signs that need careful handling
SIMILAR_SIGNS = {
    'a': ['e', 's', 'm', 'n', 't'],
    'e': ['a', 's'],
    's': ['a', 'e', 't'],
    'k': ['v', 'p'],
    'v': ['k', 'u', 'n'],
    'q': ['g'],
    'c': ['o'],
    'm': ['n', 'a'],
    'n': ['m', 'a'],
}

# Motion signs (can't detect reliably)
MOTION_SIGNS = ['j', 'z']

class TemporalSmoother:
    """Smooth predictions over time to prevent blinking"""
    
    def __init__(self, window_size=15, min_agreement=0.60, min_confidence=0.35):
        self.window_size = window_size
        self.min_agreement = min_agreement
        self.min_confidence = min_confidence
        self.prediction_buffer = deque(maxlen=window_size)
    
    def add_prediction(self, label, confidence):
        """Add a prediction to the buffer"""
        if confidence > 0.20:  # Only add if somewhat confident
            self.prediction_buffer.append((label, confidence))
    
    def get_stable_prediction(self):
        """Get stable prediction from buffer"""
        if len(self.prediction_buffer) < 10:  # Need at least 10 frames
            return None, 0.0, "Analyzing..."
        
        # Count occurrences
        labels = [p[0] for p in self.prediction_buffer]
        counter = Counter(labels)
        most_common = counter.most_common(3)
        
        if not most_common:
            return None, 0.0, "No prediction"
        
        top_label, top_count = most_common[0]
        agreement = top_count / len(self.prediction_buffer)
        
        # Calculate average confidence for top prediction
        top_confidences = [p[1] for p in self.prediction_buffer if p[0] == top_label]
        avg_confidence = np.mean(top_confidences)
        
        # Check thresholds
        if agreement < self.min_agreement:
            return None, avg_confidence, f"Low agreement ({agreement*100:.0f}%)"
        
        if avg_confidence < self.min_confidence:
            return None, avg_confidence, f"Low confidence ({avg_confidence*100:.0f}%)"
        
        # Handle similar signs - require higher confidence
        if top_label in SIMILAR_SIGNS:
            if len(most_common) >= 2:
                second_label = most_common[1][0]
                if second_label in SIMILAR_SIGNS.get(top_label, []):
                    # Similar sign detected, need higher confidence
                    if avg_confidence < 0.45:
                        return None, avg_confidence, f"Similar to {second_label.upper()}"
        
        return top_label, avg_confidence, f"Stable ({agreement*100:.0f}%)"
    
    def clear(self):
        """Clear the buffer"""
        self.prediction_buffer.clear()


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Get predictions from model"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results


def draw_professional_ui(frame, stable_pred, stable_conf, status, top_5, sequence, fps):
    """Draw professional UI for demo"""
    h, w, _ = frame.shape
    
    # Semi-transparent overlay for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (500, 250), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Title
    cv2.putText(frame, "Sign Language Recognition", (10, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Current stable prediction
    if stable_pred and stable_pred not in ['nothing']:
        pred_color = (0, 255, 0) if stable_conf > 0.5 else (0, 255, 255)
        cv2.putText(frame, f"Detected: {stable_pred.upper()}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, pred_color, 2)
        cv2.putText(frame, f"Confidence: {stable_conf*100:.0f}%", (10, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
        
        # Instruction
        cv2.putText(frame, "Press SPACE to add", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        cv2.putText(frame, status, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Top 5 predictions (small, on the right)
    if top_5:
        y_off = 30
        for i, (label, score) in enumerate(top_5):
            color = (0, 255, 0) if i == 0 else (180, 180, 180)
            text = f"{label.upper()}: {score*100:.0f}%"
            cv2.putText(frame, text, (w - 250, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_off += 25
    
    # Instructions at bottom
    cv2.putText(frame, "SPACE=Add | C=Clear | ESC=Exit", (10, h - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # HUGE display of current stable letter
    if stable_pred and stable_pred not in ['nothing']:
        text_size = cv2.getTextSize(stable_pred.upper(), cv2.FONT_HERSHEY_SIMPLEX, 8, 10)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2
        
        # Shadow
        cv2.putText(frame, stable_pred.upper(), (text_x + 5, text_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 0, 0), 10)
        # Main text
        cv2.putText(frame, stable_pred.upper(), (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 8, (0, 255, 0), 10)


def main():
    print("="*70)
    print("HACKATHON-READY Sign Language Classifier")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Stable predictions (no blinking)")
    print("  ✓ Temporal smoothing")
    print("  ✓ High confidence filtering")
    print("  ✓ Manual confirmation (SPACE key)")
    print("  ✓ Professional UI")
    print("\n[1/3] Loading model files...")
    
    # Load model
    label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]
    
    with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    print(f"✓ Model loaded ({len(label_lines)} classes)")
    
    print("\n[2/3] Starting TensorFlow session...")
    sess = tf.Session()
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    print("✓ Session started")
    
    print("\n" + "="*70)
    print("INSTRUCTIONS FOR DEMO:")
    print("="*70)
    print("1. Put your hand in the GREEN BOX")
    print("2. Make a sign and hold steady")
    print("3. Wait for stable prediction (green text)")
    print("4. Press SPACE to add the letter")
    print("5. Repeat for next letter")
    print("\nTips:")
    print("  - Good lighting helps")
    print("  - Plain background helps")
    print("  - Hold hand steady for 1-2 seconds")
    print("  - J and Z require motion (skip for demo)")
    print("="*70)
    
    print("\n[3/3] Opening camera...")
    try:
        # Try camera with DirectShow backend (Windows) to avoid zlibwapi.dll issue
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Camera 1 with DirectShow failed, trying camera 0...")
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("DirectShow failed, trying default backend...")
            cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("\n" + "="*70)
            print("ERROR: Cannot open camera!")
            print("="*70)
            print("\nTroubleshooting:")
            print("  1. Make sure camera is connected")
            print("  2. Close other apps using camera (Zoom, Teams, etc.)")
            print("  3. Check camera permissions")
            print("  4. Try unplugging and replugging camera")
            print("="*70)
            sess.close()
            input("\nPress ENTER to exit...")
            return
        
        # Optimize camera settings for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
        print("✓ Camera opened successfully!")
        
        # State
        sequence = ''
        smoother = TemporalSmoother(window_size=15, min_agreement=0.60, min_confidence=0.35)
        
        # FPS tracking
        fps_list = deque(maxlen=30)
        prev_time = time.time()
        
        # Stats
        total_added = 0
        frame_count = 0
        first_prediction_done = False
        
        # Optimization: Predict every N frames for smooth video
        PREDICT_EVERY = 2  # Predict every 2 frames = 15 FPS predictions, 30 FPS video
        
        # Cache last prediction results
        stable_pred = None
        stable_conf = 0
        status = "Loading model (first prediction is slow)..."
        top_5 = []
        
        print("\n" + "="*70)
        print("✓ ALL SYSTEMS READY!")
        print("="*70)
        print("Optimized for smooth 30 FPS video with 15 FPS predictions")
        print("Show your hand in the green box...")
        print("="*70 + "\n")
        
        # Create windows explicitly
        cv2.namedWindow('Sign Language Recognition - HACKATHON DEMO', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Sequence', cv2.WINDOW_NORMAL)
        
        # Test read
        ret, test_frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from camera!")
            cap.release()
            input("Press ENTER to exit...")
            return
        print(f"✓ Camera working! Frame size: {test_frame.shape}")
        
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
            avg_fps = np.mean(fps_list)
            prev_time = current_time
            
            # Hand detection box (center)
            box_size = 400
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size
            
            # Extract hand region
            hand_img = frame[y1:y2, x1:x2]
            
            # Run prediction every N frames for smooth video
            if frame_count % PREDICT_EVERY == 0:
                try:
                    # Preprocess - optimized
                    hand_resized = cv2.resize(hand_img, (299, 299), interpolation=cv2.INTER_LINEAR)
                    
                    # Predict - use lower JPEG quality for speed
                    image_data = cv2.imencode('.jpg', hand_resized, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                    top_5 = predict_sign(image_data, sess, softmax_tensor, label_lines)
                    
                    if not first_prediction_done:
                        first_prediction_done = True
                        print("✓ First prediction complete! Model is ready.")
                        status = "Analyzing..."
                    
                    # Add to smoother
                    top_label, top_conf = top_5[0]
                    smoother.add_prediction(top_label, top_conf)
                    
                    # Get stable prediction
                    stable_pred, stable_conf, status = smoother.get_stable_prediction()
                    
                except Exception as e:
                    stable_pred = None
                    stable_conf = 0
                    status = f"Error: {str(e)}"
                    top_5 = []
                    print(f"Prediction error: {e}")
            
            frame_count += 1
            
            # Draw UI every frame for smooth display
            try:
                draw_professional_ui(frame, stable_pred, stable_conf, status, 
                                    top_5, sequence, avg_fps)
            except Exception as e:
                cv2.putText(frame, f"UI Error: {str(e)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw box
            box_color = (0, 255, 0) if stable_pred and stable_pred not in ['nothing'] else (100, 100, 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
            
            try:
                cv2.imshow('Sign Language Recognition - HACKATHON DEMO', frame)
            except Exception as e:
                print(f"Error showing main window: {e}")
                break
            
            # Sequence window
            img_seq = np.zeros((250, 1200, 3), np.uint8)
            cv2.putText(img_seq, "Recognized Sequence:", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 255), 2)
            
            # Display sequence with word wrap
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
            
            try:
                cv2.imshow('Sequence', img_seq)
            except Exception as e:
                print(f"Error showing sequence window: {e}")
                break
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - add letter
                if stable_pred and stable_pred not in ['nothing']:
                    if stable_pred == 'space':
                        sequence += ' '
                        print(f"✓ Added: SPACE")
                    elif stable_pred == 'del':
                        if sequence:
                            sequence = sequence[:-1]
                            print(f"✓ Deleted last character")
                    elif stable_pred in MOTION_SIGNS:
                        print(f"⚠ {stable_pred.upper()} requires motion - skipping")
                    else:
                        sequence += stable_pred
                        print(f"✓ Added: {stable_pred.upper()} (confidence: {stable_conf*100:.0f}%)")
                        total_added += 1
                    
                    # Clear smoother after adding
                    smoother.clear()
                else:
                    print("✗ No stable prediction to add")
            
            elif key == ord('c') or key == ord('C'):
                sequence = ''
                smoother.clear()
                print("✓ Cleared sequence")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sess.close()
        print("\n✓ Session closed")
    
    print("\n" + "="*70)
    print("Demo ended")
    print("="*70)
    print(f"\nFinal sequence: {sequence.upper()}")
    print(f"Total letters added: {total_added}")
    print("\nGood luck with your hackathon! 🚀")


if __name__ == "__main__":
    main()
