import sys
import os

# Disable tensorflow compilation warnings BEFORE importing
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import cv2
import numpy as np
import mediapipe as mp

# Import tensorflow after mediapipe
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def predict(image_data):
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    max_score = 0.0
    res = ''
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:
            max_score = score
            res = human_string
    return res, max_score

def extract_hand_region(image, hand_landmarks, padding=50):
    """Extract hand region with padding based on landmarks"""
    h, w, _ = image.shape
    
    # Get all landmark coordinates
    x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
    
    # Find bounding box
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    # Make it square for better model performance
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)
    
    # Center the square
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - size // 2)
    x_max = min(w, center_x + size // 2)
    y_min = max(0, center_y - size // 2)
    y_max = min(h, center_y + size // 2)
    
    return image[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("Starting Sign Language Recognition with Hand Tracking...")
print("Instructions:")
print("- Show your hand to the camera")
print("- The system will automatically detect and track your hand")
print("- Hold a sign for 2-3 seconds to add it to the sequence")
print("- Press ESC to exit")
print("-" * 60)

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    res, score = '', 0.0
    i = 0
    mem = ''
    consecutive = 0
    sequence = ''
    hand_detected = False
    
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Extract hand region
                    hand_img, bbox = extract_hand_region(frame, hand_landmarks, padding=60)
                    
                    if hand_img.size > 0:
                        # Draw bounding box
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
                        
                        # Predict every 5 frames
                        if i == 4:
                            try:
                                # Resize to model input size
                                hand_resized = cv2.resize(hand_img, (299, 299))
                                image_data = cv2.imencode('.jpg', hand_resized)[1].tobytes()
                                res_tmp, score = predict(image_data)
                                res = res_tmp
                                i = 0
                                
                                # Add to sequence logic
                                if mem == res:
                                    consecutive += 1
                                else:
                                    consecutive = 0
                                
                                if consecutive == 2 and res not in ['nothing']:
                                    if res == 'space':
                                        sequence += ' '
                                    elif res == 'del':
                                        sequence = sequence[:-1]
                                    else:
                                        sequence += res
                                    consecutive = 0
                                
                                mem = res
                            except Exception as e:
                                print(f"Prediction error: {e}")
                        
                        i += 1
            
            # Display info
            if hand_detected:
                status_text = "Hand Detected"
                status_color = (0, 255, 0)
            else:
                status_text = "No Hand Detected"
                status_color = (0, 0, 255)
            
            # Status
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Prediction
            if res:
                cv2.putText(frame, f"Sign: {res.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {score*100:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            cv2.imshow('Sign Language Recognition - Hand Tracking', frame)
            
            # Sequence window with word wrapping
            img_sequence = np.zeros((400, 1200, 3), np.uint8)
            
            # Title
            cv2.putText(img_sequence, "Recognized Sequence:", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            # Word wrapping
            max_chars_per_line = 50
            lines = []
            current_line = ""
            
            for char in sequence.upper():
                if len(current_line) >= max_chars_per_line:
                    lines.append(current_line)
                    current_line = char
                else:
                    current_line += char
            
            if current_line:
                lines.append(current_line)
            
            # Display lines
            y_offset = 80
            for line in lines[-8:]:
                cv2.putText(img_sequence, line, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 45
            
            cv2.imshow('Sequence', img_sequence)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()

print("\nSession ended. Final sequence:")
print(sequence.upper())
