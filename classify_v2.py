"""
Static image classification with hand detection V2
Integrates hand detection with sign language classification
"""

import sys
import os
import cv2
import numpy as np

# Add hand_landmark_v2 to path
sys.path.insert(0, 'hand_landmark_v2')

# Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

# Try to import hand landmark model (fallback to MediaPipe if not available)
USE_CUSTOM_MODEL = False
try:
    from hand_landmark_v2.inference import HandLandmarkInference
    if os.path.exists('hand_landmark_v2/checkpoints/best_model.pth'):
        USE_CUSTOM_MODEL = True
except ImportError:
    pass

if not USE_CUSTOM_MODEL:
    try:
        import mediapipe as mp
    except ImportError:
        print("Error: Neither custom model nor MediaPipe available")
        print("Install MediaPipe: pip install mediapipe")
        sys.exit(1)


def extract_hand_region_custom(image, landmarks, padding=60):
    """Extract hand region using custom model landmarks"""
    h, w = image.shape[:2]
    
    # Denormalize landmarks
    landmarks_px = landmarks.copy()
    landmarks_px[:, 0] *= w
    landmarks_px[:, 1] *= h
    
    # Get bounding box
    x_coords = landmarks_px[:, 0]
    y_coords = landmarks_px[:, 1]
    
    x_min = max(0, int(np.min(x_coords)) - padding)
    x_max = min(w, int(np.max(x_coords)) + padding)
    y_min = max(0, int(np.min(y_coords)) - padding)
    y_max = min(h, int(np.max(y_coords)) + padding)
    
    # Make square
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - size // 2)
    x_max = min(w, center_x + size // 2)
    y_min = max(0, center_y - size // 2)
    y_max = min(h, center_y + size // 2)
    
    # Extract region
    hand_region = image[y_min:y_max, x_min:x_max]
    
    if hand_region.size == 0:
        return None, None
    
    return hand_region, (x_min, y_min, x_max, y_max)


def extract_hand_region_mediapipe(image, hand_landmarks, padding=60):
    """Extract hand region using MediaPipe landmarks"""
    h, w = image.shape[:2]
    
    # Get all landmark coordinates
    x_coords = [lm.x * w for lm in hand_landmarks.landmark]
    y_coords = [lm.y * h for lm in hand_landmarks.landmark]
    
    # Get bounding box
    x_min = max(0, int(min(x_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_min = max(0, int(min(y_coords)) - padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    # Make square
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height)
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - size // 2)
    x_max = min(w, center_x + size // 2)
    y_min = max(0, center_y - size // 2)
    y_max = min(h, center_y + size // 2)
    
    # Extract region
    hand_region = image[y_min:y_max, x_min:x_max]
    
    if hand_region.size == 0:
        return None, None
    
    return hand_region, (x_min, y_min, x_max, y_max)


def classify_image(image_path, save_visualization=False):
    """
    Classify sign language from image
    
    Args:
        image_path: Path to input image
        save_visualization: Whether to save visualization
        
    Returns:
        results: Dictionary with classification results
    """
    print("="*70)
    print("Sign Language Classification V2")
    print("="*70)
    print(f"Image: {image_path}")
    print(f"Hand Detection: {'Custom Model' if USE_CUSTOM_MODEL else 'MediaPipe'}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize hand detector
    if USE_CUSTOM_MODEL:
        print("Loading custom hand detector...")
        hand_detector = HandLandmarkInference(
            'hand_landmark_v2/checkpoints/best_model.pth',
            backend='pytorch',
            use_kalman=False,
            use_gpu=False
        )
    else:
        print("Initializing MediaPipe...")
        mp_hands = mp.solutions.hands
        hand_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    
    # Detect hand
    print("Detecting hand...")
    hand_detected = False
    hand_img = None
    bbox = None
    
    if USE_CUSTOM_MODEL:
        try:
            landmarks, confidence, _ = hand_detector.predict(rgb_image)
            hand_detected = landmarks is not None
            
            if hand_detected:
                hand_img, bbox = extract_hand_region_custom(image, landmarks, padding=60)
                
                # Draw landmarks for visualization
                if save_visualization:
                    image = hand_detector.draw_landmarks(image, landmarks, confidence)
        except Exception as e:
            print(f"Hand detection error: {e}")
    else:
        results = hand_detector.process(rgb_image)
        
        if results.multi_hand_landmarks:
            hand_detected = True
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_img, bbox = extract_hand_region_mediapipe(image, hand_landmarks, padding=60)
            
            # Draw landmarks for visualization
            if save_visualization:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        
        hand_detector.close()
    
    if not hand_detected or hand_img is None:
        print("✗ No hand detected in image")
        return None
    
    print("✓ Hand detected")
    
    # Draw bounding box for visualization
    if save_visualization and bbox is not None:
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
    
    # Classify sign
    print("Classifying sign...")
    
    # Preprocess hand image
    hand_resized = cv2.resize(hand_img, (299, 299))
    
    # Apply histogram equalization
    yuv = cv2.cvtColor(hand_resized, cv2.COLOR_BGR2YUV)
    yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
    hand_resized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    # Encode image
    image_data = cv2.imencode('.jpg', hand_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    
    # Load sign classifier
    label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
    
    with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf_v1.import_graph_def(graph_def, name='')
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        
        # Get top predictions
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        
        results = []
        for node_id in top_k[:5]:  # Top 5 predictions
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            results.append({
                'label': human_string,
                'score': float(score)
            })
    
    # Print results
    print("\n" + "="*70)
    print("Classification Results:")
    print("="*70)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['label'].upper()}: {result['score']*100:.2f}%")
    print("="*70)
    
    # Save visualization
    if save_visualization:
        output_path = image_path.replace('.', '_result.')
        
        # Add prediction text
        top_result = results[0]
        cv2.putText(image, f"Prediction: {top_result['label'].upper()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Confidence: {top_result['score']*100:.1f}%", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image)
        print(f"\n✓ Visualization saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_v2.py <image_path> [--save-viz]")
        print("\nExample:")
        print("  python classify_v2.py test_image.jpg")
        print("  python classify_v2.py test_image.jpg --save-viz")
        sys.exit(1)
    
    image_path = sys.argv[1]
    save_viz = '--save-viz' in sys.argv
    
    results = classify_image(image_path, save_visualization=save_viz)
    
    if results is None:
        sys.exit(1)
