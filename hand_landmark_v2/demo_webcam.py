"""
Webcam demo for hand landmark detection
"""

import sys
import os
import cv2
import numpy as np
from collections import deque
import time
import argparse

sys.path.insert(0, '.')

from inference import HandLandmarkInference


def main():
    parser = argparse.ArgumentParser(description='Hand landmark detection webcam demo')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model file')
    parser.add_argument('--backend', type=str, default='pytorch',
                       choices=['pytorch', 'onnx', 'tflite'],
                       help='Inference backend')
    parser.add_argument('--filter', type=str, default='one_euro',
                       choices=['none', 'kalman', 'one_euro'],
                       help='Temporal filter type')
    parser.add_argument('--dotted', action='store_true',
                       help='Use dotted line visualization')
    parser.add_argument('--camera', type=int, default=1,
                       help='Camera index')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU if available')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hand Landmark Detection - Webcam Demo")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Backend: {args.backend}")
    print(f"Filter: {args.filter}")
    print(f"Visualization: {'Dotted' if args.dotted else 'Solid'}")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\n✗ Error: Model not found: {args.model}")
        print("\nPlease train the model first:")
        print("  python hand_landmark_v2/train.py")
        return
    
    # Initialize inference engine
    print("\nInitializing inference engine...")
    try:
        engine = HandLandmarkInference(
            args.model,
            backend=args.backend,
            use_kalman=(args.filter != 'none'),
            filter_type=args.filter if args.filter != 'none' else 'kalman',
            use_gpu=args.gpu
        )
        print("✓ Inference engine ready")
    except Exception as e:
        print(f"✗ Error initializing inference engine: {e}")
        return
    
    # Open camera
    print(f"\nOpening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open camera {args.camera}")
        return
    
    print("✓ Camera opened")
    print("\n" + "="*70)
    print("Controls:")
    print("  ESC - Exit")
    print("  D - Toggle dotted/solid lines")
    print("  F - Cycle through filters (None/Kalman/One Euro)")
    print("  R - Reset filter")
    print("="*70)
    
    # State
    use_dotted = args.dotted
    filter_modes = ['none', 'kalman', 'one_euro']
    current_filter_idx = filter_modes.index(args.filter)
    
    # FPS tracking
    fps_queue = deque(maxlen=30)
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        try:
            landmarks, confidence, model_fps = engine.predict(rgb_frame)
            hand_detected = True
        except Exception as e:
            hand_detected = False
            landmarks = None
        
        # Calculate overall FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        fps_queue.append(fps)
        avg_fps = np.mean(fps_queue)
        prev_time = current_time
        
        # Draw landmarks
        if hand_detected and landmarks is not None:
            frame = engine.draw_landmarks(
                frame, landmarks, confidence,
                draw_connections=True,
                dotted=use_dotted
            )
            
            # Extract bounding box
            bbox_norm = engine.extract_hand_bbox(landmarks, padding=60)
            x_min = int(bbox_norm[0] * w)
            y_min = int(bbox_norm[1] * h)
            x_max = int(bbox_norm[2] * w)
            y_max = int(bbox_norm[3] * h)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # Display info overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status
        if hand_detected:
            status_text = "Hand Detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No Hand"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Backend
        cv2.putText(frame, f"Backend: {args.backend.upper()}", (10, 95),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Filter
        filter_name = filter_modes[current_filter_idx].replace('_', ' ').title()
        cv2.putText(frame, f"Filter: {filter_name}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Visualization mode
        viz_mode = "Dotted" if use_dotted else "Solid"
        cv2.putText(frame, f"Lines: {viz_mode}", (10, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Confidence (if hand detected)
        if hand_detected and confidence is not None:
            avg_conf = np.mean(confidence)
            cv2.putText(frame, f"Confidence: {avg_conf*100:.1f}%", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Hand Landmark Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            break
        elif key == ord('d') or key == ord('D'):
            use_dotted = not use_dotted
            print(f"Visualization: {'Dotted' if use_dotted else 'Solid'}")
        elif key == ord('f') or key == ord('F'):
            current_filter_idx = (current_filter_idx + 1) % len(filter_modes)
            new_filter = filter_modes[current_filter_idx]
            
            # Reinitialize engine with new filter
            use_kalman = (new_filter != 'none')
            filter_type = new_filter if new_filter != 'none' else 'kalman'
            
            engine = HandLandmarkInference(
                args.model,
                backend=args.backend,
                use_kalman=use_kalman,
                filter_type=filter_type,
                use_gpu=args.gpu
            )
            print(f"Filter: {new_filter.replace('_', ' ').title()}")
        elif key == ord('r') or key == ord('R'):
            engine.reset_filter()
            print("Filter reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*70)
    print("Demo ended")
    print(f"Average FPS: {avg_fps:.1f}")
    print("="*70)


if __name__ == "__main__":
    main()
