"""
ULTRA SIMPLE Sign Language Classifier
NO motion detection, NO complicated logic
Just shows what the model predicts in real-time
Press SPACE to add the current letter
"""

import sys
import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def predict(image_data, sess, softmax_tensor, label_lines):
    """Get prediction from model"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    results = []
    for node_id in top_k[:5]:
        results.append((label_lines[node_id], predictions[0][node_id]))
    
    return results

print("="*70)
print("ULTRA SIMPLE Sign Language Classifier")
print("="*70)
print("\nLoading model...")

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("✓ Model loaded")
print(f"✓ {len(label_lines)} classes: {', '.join(label_lines)}")

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("1. Put your hand in the GREEN BOX")
print("2. Make a sign")
print("3. Press SPACE to add the letter you see")
print("4. Press C to clear")
print("5. Press ESC to exit")
print("="*70)

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    sequence = ''
    frame_count = 0
    
    print("\n✓ Camera started!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Fixed box in center
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Extract hand region
        hand_img = frame[y1:y2, x1:x2]
        
        # Predict every 3 frames
        if frame_count % 3 == 0:
            try:
                # Resize to 299x299 (model input size)
                hand_resized = cv2.resize(hand_img, (299, 299))
                
                # Encode as JPEG
                image_data = cv2.imencode('.jpg', hand_resized, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                
                # Predict
                predictions = predict(image_data, sess, softmax_tensor, label_lines)
                
                # Get top prediction
                top_label, top_score = predictions[0]
                
                # Draw predictions
                y_offset = 50
                for i, (label, score) in enumerate(predictions[:5]):
                    color = (0, 255, 0) if i == 0 else (150, 150, 150)
                    text = f"{i+1}. {label.upper()}: {score*100:.1f}%"
                    cv2.putText(frame, text, (20, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 40
                
                # Big display of top prediction
                cv2.putText(frame, top_label.upper(), (20, h - 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)
                cv2.putText(frame, f"{top_score*100:.1f}%", (20, h - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
            except Exception as e:
                cv2.putText(frame, f"Error: {str(e)}", (20, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_count += 1
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Instructions
        cv2.putText(frame, "Press SPACE to add letter", (w - 400, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press C to clear", (w - 400, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Simple Classifier', frame)
        
        # Sequence window
        img_seq = np.zeros((200, 1200, 3), np.uint8)
        cv2.putText(img_seq, "Sequence:", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 255), 2)
        cv2.putText(img_seq, sequence.upper(), (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow('Sequence', img_seq)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE - add current letter
            if 'top_label' in locals() and top_label not in ['nothing']:
                if top_label == 'space':
                    sequence += ' '
                elif top_label == 'del':
                    sequence = sequence[:-1]
                else:
                    sequence += top_label
                print(f"Added: {top_label.upper()}")
        elif key == ord('c') or key == ord('C'):
            sequence = ''
            print("Cleared")
    
    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print(f"Final sequence: {sequence.upper()}")
print("="*70)
