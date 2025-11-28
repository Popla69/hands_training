"""
Simple Sign Language Recognition - FIXED BOX METHOD
This is the fastest and most reliable method - just put your hand in the box!
"""

import sys
import os
import cv2
import numpy as np
import time

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()


def predict_sign(image_data, sess, softmax_tensor, label_lines):
    """Predict sign language letter from image"""
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    # Get top 3 predictions
    results = []
    for node_id in top_k[:3]:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        results.append((human_string, score))
    
    return results[0][0], results[0][1], results


def main():
    """Main function"""
    
    print("="*70)
    print("Sign Language Recognition - SIMPLE & FAST")
    print("="*70)
    print("\nUsing: Fixed Box Method (Original)")
    print("Just put your hand in the green box!")
    print("\nLoading model...")
    
    # Load sign language classifier
    try:
        label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
        
        with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
            graph_def = tf_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf_v1.import_graph_def(graph_def, name='')
        
        print("✓ Sign language classifier loaded")
    except Exception as e:
        print(f"✗ Error loading classifier: {e}")
        return
    
    print("\n" + "="*70)
    print("Ready!")
    print("="*70)
    
    response = input("\nStart camera? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\nStarting camera...")
    print("\nControls:")
    print("  - ESC: Exit")
    print("  - SPACE: Toggle preprocessing")
    print("  - Hold sign for 2-3 seconds to add to sequence")
    print("="*70)
    
    # Start inference
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # State variables
        res, score = '', 0.0
        i = 0
        mem = ''
        consecutive = 0
        sequence = ''
        
        # FPS tracking
        fps_list = []
        prev_time = time.time()
        
        # Preprocessing toggle
        use_preprocessing = True
        
        # Box size - adjustable
        box_size = 400
        box_x = 100
        box_y = 100
        
        print("\n✓ Camera started!")
        print("Put your hand in the GREEN BOX")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = sum(fps_list) / len(fps_list)
            prev_time = current_time
            
            # Extract hand region from fixed box
            x1, y1 = box_x, box_y
            x2, y2 = box_x + box_size, box_y + box_size
            
            # Make sure box is within frame
            x2 = min(x2, w)
            y2 = min(y2, h)
            
            hand_img = frame[y1:y2, x1:x2]
            
            # Predict every 3 frames for speed
            if i >= 2 and hand_img.size > 0:
                try:
                    # Resize to 299x299 for InceptionV3
                    hand_resized = cv2.resize(hand_img, (299, 299))
                    
                    # Optional preprocessing
                    if use_preprocessing:
                        # Histogram equalization
                        yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
                        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                        hand_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                    
                    # Encode and predict
                    image_data = cv2.imencode('.jpg', hand_resized, 
                                             [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                    res_tmp, score, top_3 = predict_sign(image_data, sess, softmax_tensor, label_lines)
                    
                    # Only update if confidence is reasonable
                    if score > 0.2:
                        res = res_tmp
                    
                    i = 0
                    
                    # Sequence logic
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
                    pass
            
            i += 1
            
            # Draw the capture box
            box_color = (0, 255, 0) if consecutive < 2 else (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 4)
            
            # Show preview of what classifier sees
            try:
                preview_size = 200
                preview = cv2.resize(hand_img, (preview_size, preview_size))
                
                # Apply same preprocessing for preview
                if use_preprocessing:
                    yuv = cv2.cvtColor(preview, cv2.COLOR_BGR2YUV)
                    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                    preview = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                
                frame[h-preview_size-10:h-10, w-preview_size-10:w-10] = preview
                cv2.rectangle(frame, (w-preview_size-10, h-preview_size-10), 
                            (w-10, h-10), (0, 255, 255), 2)
                cv2.putText(frame, "Classifier Input", (w-preview_size-5, h-preview_size-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            except:
                pass
            
            # Info overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (600, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # FPS
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Preprocessing status
            preproc_text = "ON" if use_preprocessing else "OFF"
            preproc_color = (0, 255, 0) if use_preprocessing else (0, 0, 255)
            cv2.putText(frame, f"Preprocessing: {preproc_text}", (10, 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, preproc_color, 2)
            
            # Prediction
            if res:
                cv2.putText(frame, f"Sign: {res.upper()}", (10, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, f"Confidence: {score*100:.1f}%", (10, 145), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame, "Put hand in GREEN BOX", (box_x, box_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('Sign Language Recognition - SIMPLE', frame)
            
            # Sequence window
            img_sequence = np.zeros((400, 1200, 3), np.uint8)
            cv2.putText(img_sequence, "Recognized Sequence:", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
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
            
            y_offset = 80
            for line in lines[-8:]:
                cv2.putText(img_sequence, line, (30, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y_offset += 45
            
            cv2.imshow('Sequence', img_sequence)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 32:  # SPACE
                use_preprocessing = not use_preprocessing
                print(f"Preprocessing: {'ON' if use_preprocessing else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nSession ended. Final sequence:")
    print(sequence.upper())


if __name__ == "__main__":
    main()
