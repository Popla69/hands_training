"""
Video processing demo for hand landmark detection
"""

import sys
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

sys.path.insert(0, '.')

from inference import HandLandmarkInference


def process_video(input_path, output_path, model_path, backend='pytorch',
                 use_filter=True, filter_type='one_euro', dotted=False):
    """
    Process video file with hand landmark detection
    
    Args:
        input_path: Input video path
        output_path: Output video path
        model_path: Model file path
        backend: Inference backend
        use_filter: Whether to use temporal filtering
        filter_type: Filter type ('kalman' or 'one_euro')
        dotted: Use dotted line visualization
    """
    print("="*70)
    print("Hand Landmark Detection - Video Processing")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {model_path}")
    print(f"Backend: {backend}")
    print(f"Filter: {filter_type if use_filter else 'None'}")
    print("="*70)
    
    # Initialize inference engine
    print("\nInitializing inference engine...")
    engine = HandLandmarkInference(
        model_path,
        backend=backend,
        use_kalman=use_filter,
        filter_type=filter_type,
        use_gpu=False
    )
    print("✓ Inference engine ready")
    
    # Open input video
    print(f"\nOpening input video...")
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"✗ Error: Could not open video: {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video opened")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"✗ Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    print(f"✓ Output video created")
    
    # Process frames
    print(f"\nProcessing frames...")
    frame_count = 0
    hands_detected = 0
    
    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect landmarks
            try:
                landmarks, confidence, model_fps = engine.predict(rgb_frame)
                hand_detected = True
                hands_detected += 1
            except Exception as e:
                hand_detected = False
                landmarks = None
            
            # Draw landmarks
            if hand_detected and landmarks is not None:
                frame = engine.draw_landmarks(
                    frame, landmarks, confidence,
                    draw_connections=True,
                    dotted=dotted
                )
                
                # Extract and draw bounding box
                bbox_norm = engine.extract_hand_bbox(landmarks, padding=60)
                x_min = int(bbox_norm[0] * width)
                y_min = int(bbox_norm[1] * height)
                x_max = int(bbox_norm[2] * width)
                y_max = int(bbox_norm[3] * height)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Add status text
                cv2.putText(frame, "Hand Detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Write frame
            out.write(frame)
            
            pbar.update(1)
    
    # Cleanup
    cap.release()
    out.release()
    
    # Summary
    detection_rate = (hands_detected / frame_count) * 100 if frame_count > 0 else 0
    
    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"Frames processed: {frame_count}")
    print(f"Hands detected: {hands_detected} ({detection_rate:.1f}%)")
    print(f"Output saved to: {output_path}")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Process video with hand landmark detection')
    parser.add_argument('input', type=str, help='Input video path')
    parser.add_argument('output', type=str, help='Output video path')
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
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found: {args.model}")
        print("\nPlease train the model first:")
        print("  python hand_landmark_v2/train.py")
        return
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"✗ Error: Input video not found: {args.input}")
        return
    
    # Process video
    success = process_video(
        args.input,
        args.output,
        args.model,
        backend=args.backend,
        use_filter=(args.filter != 'none'),
        filter_type=args.filter if args.filter != 'none' else 'kalman',
        dotted=args.dotted
    )
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
