"""
Compare your hand with training data
Shows side-by-side comparison
"""

import os
import cv2
import numpy as np
import random

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print("="*70)
print("Compare Your Hand with Training Data")
print("="*70)
print("\nThis will help you see why K, R, S, V, Y don't work")
print("You'll see training images vs your hand")
print("\nLoading model...")

# Load model
label_lines = [line.rstrip() for line in tf.gfile.GFile("logs/trained_labels.txt")]

with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

print("✓ Model loaded")

PROBLEM_LETTERS = ['K', 'R', 'S', 'V', 'Y']

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("="*70)
print("1. Camera will start")
print("2. Press K, R, S, V, or Y to test that letter")
print("3. You'll see:")
print("   - LEFT: Training data examples")
print("   - RIGHT: Your hand")
print("   - BOTTOM: What model predicts")
print("4. Compare your hand position with training data")
print("5. Press ESC to exit")
print("="*70)

input("\nPress ENTER to start...")

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_letter = 'V'  # Start with V
    training_images = []
    
    # Load training images for current letter
    def load_training_images(letter):
        dataset_path = f"dataset/{letter}"
        all_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        sample_files = random.sample(all_files, min(4, len(all_files)))
        
        images = []
        for img_file in sample_files:
            img_path = os.path.join(dataset_path, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                images.append(img)
        
        return images
    
    training_images = load_training_images(current_letter)
    
    print(f"\n✓ Camera started!")
    print(f"Current letter: {current_letter}")
    print("Press K, R, S, V, or Y to switch letters")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Extract hand region
        box_size = 400
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        hand_img = frame[y1:y2, x1:x2]
        
        # Predict
        try:
            hand_resized = cv2.resize(hand_img, (299, 299))
            image_data = cv2.imencode('.jpg', hand_resized, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            
            # Get top 5
            top_5 = []
            for i in range(5):
                node_id = top_k[i]
                label = label_lines[node_id]
                score = predictions[0][node_id]
                top_5.append((label, score))
            
        except:
            top_5 = []
        
        # Create comparison display
        display = np.zeros((800, 1200, 3), np.uint8)
        
        # Title
        cv2.putText(display, f"Testing Letter: {current_letter}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 200, 255), 2)
        
        # Training images (left side)
        cv2.putText(display, "TRAINING DATA:", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_off = 100
        for i, train_img in enumerate(training_images[:4]):
            if i < 2:
                x_off = 20
                y_pos = y_off
            else:
                x_off = 240
                y_pos = y_off if i == 2 else y_off + 220
            
            if i >= 2:
                y_pos = y_off + 220
            
            display[y_pos:y_pos+200, x_off:x_off+200] = train_img
            
            if i == 1:
                y_off += 220
        
        # Your hand (right side)
        cv2.putText(display, "YOUR HAND:", (600, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        your_hand_display = cv2.resize(hand_img, (400, 400))
        display[100:500, 600:1000] = your_hand_display
        
        # Predictions
        cv2.putText(display, "MODEL PREDICTS:", (20, 580), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_off = 620
        for i, (label, score) in enumerate(top_5):
            is_correct = label.upper() == current_letter.upper()
            color = (0, 255, 0) if is_correct else (100, 100, 100)
            
            if is_correct:
                text = f"{i+1}. {label.upper()}: {score*100:.1f}% ✓ CORRECT"
            else:
                text = f"{i+1}. {label.upper()}: {score*100:.1f}%"
            
            cv2.putText(display, text, (20, y_off), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            y_off += 35
        
        # Instructions
        cv2.putText(display, "Press K, R, S, V, Y to switch | ESC to exit", (20, 770), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.imshow('Compare Your Hand', display)
        
        # Keys
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key in [ord('k'), ord('K')]:
            current_letter = 'K'
            training_images = load_training_images(current_letter)
            print(f"\nSwitched to: {current_letter}")
        elif key in [ord('r'), ord('R')]:
            current_letter = 'R'
            training_images = load_training_images(current_letter)
            print(f"\nSwitched to: {current_letter}")
        elif key in [ord('s'), ord('S')]:
            current_letter = 'S'
            training_images = load_training_images(current_letter)
            print(f"\nSwitched to: {current_letter}")
        elif key in [ord('v'), ord('V')]:
            current_letter = 'V'
            training_images = load_training_images(current_letter)
            print(f"\nSwitched to: {current_letter}")
        elif key in [ord('y'), ord('Y')]:
            current_letter = 'Y'
            training_images = load_training_images(current_letter)
            print(f"\nSwitched to: {current_letter}")
    
    cap.release()
    cv2.destroyAllWindows()

print("\n" + "="*70)
print("Comparison complete")
print("="*70)
print("\nDid you notice the difference?")
print("Your hand position/angle might be different from training data!")
