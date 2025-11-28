import sys
import os
import cv2
import numpy as np

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

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

def detect_hand_skin(frame):
    """Detect hand using skin color detection"""
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (assumed to be hand)
        max_contour = max(contours, key=cv2.contourArea)
        
        if cv2.contourArea(max_contour) > 5000:  # Minimum area threshold
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # Add padding
            padding = 40
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2*padding)
            h = min(frame.shape[0] - y, h + 2*padding)
            
            # Make it square
            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            
            x = max(0, center_x - size // 2)
            y = max(0, center_y - size // 2)
            w = min(frame.shape[1] - x, size)
            h = min(frame.shape[0] - y, size)
            
            return (x, y, w, h), max_contour, mask
    
    return None, None, mask

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("Starting Sign Language Recognition with Automatic Hand Detection...")
print("Instructions:")
print("- Show your hand to the camera (ensure good lighting)")
print("- The system will automatically detect your hand using skin color")
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Detect hand
        hand_bbox, contour, mask = detect_hand_skin(frame)
        
        hand_detected = hand_bbox is not None
        
        if hand_detected:
            try:
                x, y, w, h = hand_bbox
                
                # Draw contour
                if contour is not None:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Extract hand region with bounds checking
                y_end = min(y+h, frame.shape[0])
                x_end = min(x+w, frame.shape[1])
                hand_img = frame[y:y_end, x:x_end]
                
                if hand_img.size > 0 and hand_img.shape[0] > 10 and hand_img.shape[1] > 10:
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
            except Exception as e:
                print(f"Hand extraction error: {e}")
                hand_detected = False
        
        # Display info
        if hand_detected:
            status_text = "Hand Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No Hand Detected - Show your hand"
            status_color = (0, 0, 255)
        
        # Status
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Prediction
        if res:
            cv2.putText(frame, f"Sign: {res.upper()}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)
            cv2.putText(frame, f"Confidence: {score*100:.1f}%", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Show mask in corner (only if frame is large enough)
        try:
            if w > 220 and h > 170:
                mask_small = cv2.resize(mask, (200, 150))
                mask_colored = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
                frame[10:160, w-210:w-10] = mask_colored
                cv2.rectangle(frame, (w-210, 10), (w-10, 160), (255, 255, 255), 2)
                cv2.putText(frame, "Skin Mask", (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            pass  # Skip mask display if there's an issue
        
        cv2.imshow('Sign Language Recognition - Auto Hand Detection', frame)
        
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
