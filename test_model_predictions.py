"""
Test what the model is actually predicting
Shows raw predictions for each frame
"""

import sys
import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("Loading model...")

label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print(f"✓ Model loaded with {len(label_lines)} classes")
print(f"Classes: {', '.join(label_lines)}")

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        sys.exit(1)
    
    print("\n✓ Camera started")
    print("\nShowing RAW predictions from model...")
    print("Put your hand in the box and see what it predicts")
    print("Press ESC to exit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Fixed box
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        hand_img = frame[y1:y2, x1:x2]
        
        # Predict every frame
        try:
            hand_resized = cv2.resize(hand_img, (299, 299))
            image_data = cv2.imencode('.jpg', hand_resized, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            # Print top 5 every 30 frames
            if frame_count % 30 == 0:
                print(f"\n--- Frame {frame_count} ---")
                for i in range(5):
                    node_id = top_k[i]
                    label = label_lines[node_id]
                    score = predictions[0][node_id]
                    print(f"{i+1}. {label.upper()}: {score*100:.2f}%")
            
            # Draw on frame
            y_off = 50
            for i in range(10):  # Show top 10
                node_id = top_k[i]
                label = label_lines[node_id]
                score = predictions[0][node_id]
                
                color = (0, 255, 0) if i == 0 else (200, 200, 200) if i < 3 else (100, 100, 100)
                text = f"{i+1}. {label.upper()}: {score*100:.1f}%"
                cv2.putText(frame, text, (20, y_off), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_off += 30
            
            # Big top prediction
            top_label = label_lines[top_k[0]]
            top_score = predictions[0][top_k[0]]
            cv2.putText(frame, top_label.upper(), (20, h - 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
            cv2.putText(frame, f"{top_score*100:.1f}%", (20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
        except Exception as e:
            cv2.putText(frame, f"Error: {str(e)}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        frame_count += 1
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow('Model Predictions Test', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

print("\nTest complete")
