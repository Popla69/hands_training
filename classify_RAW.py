"""
RAW UNFILTERED Classifier
Shows EXACTLY what the model predicts
NO filtering, NO logic, just RAW output
"""

import os
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("Model loaded. Starting camera...")
print("This shows RAW predictions - what the model ACTUALLY sees")
print("Press ESC to exit\n")

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
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
        hand_resized = cv2.resize(hand_img, (299, 299))
        
        # Predict
        image_data = cv2.imencode('.jpg', hand_resized)[1].tobytes()
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        # Show ALL predictions
        y_off = 30
        for i in range(len(label_lines)):
            node_id = top_k[i]
            label = label_lines[node_id]
            score = predictions[0][node_id]
            
            if i < 5:
                color = (0, 255, 0) if i == 0 else (200, 200, 200) if i < 3 else (150, 150, 150)
                size = 0.8 if i == 0 else 0.6
                thick = 2 if i == 0 else 1
                text = f"{i+1}. {label.upper()}: {score*100:.1f}%"
                cv2.putText(frame, text, (10, y_off), 
                           cv2.FONT_HERSHEY_SIMPLEX, size, color, thick)
                y_off += 35 if i == 0 else 25
        
        # HUGE top prediction
        top_label = label_lines[top_k[0]]
        top_score = predictions[0][top_k[0]]
        
        cv2.putText(frame, top_label.upper(), (w//2 - 100, h - 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
        cv2.putText(frame, f"{top_score*100:.0f}%", (w//2 - 80, h - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        cv2.imshow('RAW Model Output', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
