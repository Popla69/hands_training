"""
Static image demo for hand landmark detection
"""

import sys
import os
import cv2
import numpy as np
import argparse
import glob

sys.path.insert(0, '.')

from inference import HandLandmarkInference


def process_image(image_path, output_path, model_path, backend='pytorch', dotted=False):
    """
    Process single image with hand landmark detection
    
    Args:
        image_path: Input image path
        output_path: Output image path
        model_path: Model file path
        backend: Inference backend
        dotted: Use dotted line visualization
        
    Returns:
        success: Whether processing was successful
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Error: Could not load image: {image_path}")
        return False
    
    h, w = image.shape[:2]
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize inference engine (without filtering for static images)
    engine = HandLandmarkInference(
        model_path,
        backend=backend,
        use_kalman=False,
        use_gpu=False
    )
    
    # Detect landmarks
    try:
        landmarks, confidence, fps = engine.predict(rgb_image)
        hand_detected = True
    except Exception as e:
        print(f"✗ Hand detection failed: {e}")
        hand_detected = False
        landmarks = None
    
    if not hand_detected or landmarks is None:
        print(f"✗ No hand detected in: {image_path}")
        return False
    
    # Draw landmarks
    image = engine.draw_landmarks(
        image, landmarks, confidence,
        draw_connections=True,
        dotted=dotted
    )
    
    # Extract and draw bounding box
    bbox_norm = engine.extract_hand_bbox(landmarks, padding=60)
    x_min = int(bbox_norm[0] * w)
    y_min = int(bbox_norm[1] * h)
    x_max = int(bbox_norm[2] * w)
    y_max = int(bbox_norm[3] * h)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
    
    # Add info text
    avg_conf = np.mean(confidence)
    cv2.putText(image, f"Hand Detected", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(image, f"Confidence: {avg_conf*100:.1f}%", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save output
    cv2.imwrite(output_path, image)
    print(f"✓ Processed: {image_path} -> {output_path}")
    
    return True


def process_directory(input_dir, output_dir, model_path, backend='pytorch', dotted=False):
    """
    Process all images in directory
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        model_path: Model file path
        backend: Inference backend
        dotted: Use dotted line visualization
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    if not image_files:
        print(f"✗ No images found in: {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print("="*70)
    
    # Process each image
    successful = 0
    failed = 0
    
    for image_path in image_files:
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        if process_image(image_path, output_path, model_path, backend, dotted):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("="*70)
    print(f"Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Process images with hand landmark detection')
    parser.add_argument('input', type=str, help='Input image or directory path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output image or directory path (default: input_result.jpg)')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model file')
    parser.add_argument('--backend', type=str, default='pytorch',
                       choices=['pytorch', 'onnx', 'tflite'],
                       help='Inference backend')
    parser.add_argument('--dotted', action='store_true',
                       help='Use dotted line visualization')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Hand Landmark Detection - Image Processing")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found: {args.model}")
        print("\nPlease train the model first:")
        print("  python hand_landmark_v2/train.py")
        return
    
    # Check if input exists
    if not os.path.exists(args.input):
        print(f"✗ Error: Input not found: {args.input}")
        return
    
    # Determine if input is file or directory
    if os.path.isfile(args.input):
        # Process single image
        if args.output is None:
            base, ext = os.path.splitext(args.input)
            args.output = f"{base}_result{ext}"
        
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Model: {args.model}")
        print(f"Backend: {args.backend}")
        print("="*70)
        
        success = process_image(args.input, args.output, args.model, args.backend, args.dotted)
        
        if not success:
            sys.exit(1)
    
    elif os.path.isdir(args.input):
        # Process directory
        if args.output is None:
            args.output = args.input + '_results'
        
        print(f"Input directory: {args.input}")
        print(f"Output directory: {args.output}")
        print(f"Model: {args.model}")
        print(f"Backend: {args.backend}")
        print("="*70)
        
        process_directory(args.input, args.output, args.model, args.backend, args.dotted)
    
    else:
        print(f"✗ Error: Invalid input: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
