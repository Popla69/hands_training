"""
Inference engine with Kalman filtering and multi-backend support
"""

import numpy as np
import cv2
import time
import torch

from model import create_model
from kalman_filter import LandmarkKalmanFilter, LandmarkOneEuroFilter
from config import *


class HandLandmarkInference:
    """
    High-performance inference engine for hand landmark detection
    Supports PyTorch, ONNX, and TFLite backends
    """
    
    def __init__(self, model_path, backend='pytorch', use_kalman=True, 
                 filter_type='kalman', use_gpu=False):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to model file
            backend: 'pytorch', 'onnx', or 'tflite'
            use_kalman: Whether to use filtering
            filter_type: 'kalman' or 'one_euro'
            use_gpu: Whether to use GPU
        """
        self.backend = backend
        self.use_kalman = use_kalman
        self.filter_type = filter_type
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing inference engine...")
        print(f"  Backend: {backend}")
        print(f"  Device: {self.device}")
        print(f"  Filtering: {filter_type if use_kalman else 'None'}")
        
        # Load model
        if backend == 'pytorch':
            self.model = self._load_pytorch(model_path)
        elif backend == 'onnx':
            self.model = self._load_onnx(model_path)
        elif backend == 'tflite':
            self.model = self._load_tflite(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Initialize filter
        if use_kalman:
            if filter_type == 'kalman':
                self.filter = LandmarkKalmanFilter()
            elif filter_type == 'one_euro':
                self.filter = LandmarkOneEuroFilter()
            else:
                raise ValueError(f"Unknown filter type: {filter_type}")
        else:
            self.filter = None
        
        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()
        
    def _load_pytorch(self, model_path):
        """Load PyTorch model"""
        model = create_model(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"✓ Loaded PyTorch model from {model_path}")
        return model
    
    def _load_onnx(self, model_path):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"✓ Loaded ONNX model from {model_path}")
        print(f"  Providers: {session.get_providers()}")
        return session
    
    def _load_tflite(self, model_path):
        """Load TFLite model"""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("tensorflow not installed. Install with: pip install tensorflow")
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"✓ Loaded TFLite model from {model_path}")
        return interpreter
    
    def preprocess(self, image):
        """
        Preprocess image for model input
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            preprocessed: (1, 3, 224, 224) tensor
        """
        # Resize
        img = cv2.resize(image, INPUT_SIZE)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        
        # Transpose to CHW
        img = img.transpose(2, 0, 1)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict_pytorch(self, image):
        """Predict using PyTorch"""
        img_tensor = torch.from_numpy(image).float().to(self.device)
        
        with torch.no_grad():
            landmarks, confidence = self.model(img_tensor)
        
        landmarks = landmarks.cpu().numpy()[0]
        confidence = confidence.cpu().numpy()[0]
        
        return landmarks, confidence
    
    def predict_onnx(self, image):
        """Predict using ONNX"""
        input_name = self.model.get_inputs()[0].name
        outputs = self.model.run(None, {input_name: image.astype(np.float32)})
        
        landmarks = outputs[0][0]
        confidence = outputs[1][0]
        
        return landmarks, confidence
    
    def predict_tflite(self, image):
        """Predict using TFLite"""
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        
        self.model.set_tensor(input_details[0]['index'], image.astype(np.float32))
        self.model.invoke()
        
        landmarks = self.model.get_tensor(output_details[0]['index'])[0]
        confidence = self.model.get_tensor(output_details[1]['index'])[0]
        
        return landmarks, confidence
    
    def predict(self, image):
        """
        Predict hand landmarks from image
        
        Args:
            image: RGB image (H, W, 3)
            
        Returns:
            landmarks: (21, 3) array of normalized x, y, z coordinates
            confidence: (21,) array of per-landmark confidence
            fps: current FPS
        """
        # Preprocess
        img_preprocessed = self.preprocess(image)
        
        # Predict based on backend
        if self.backend == 'pytorch':
            landmarks, confidence = self.predict_pytorch(img_preprocessed)
        elif self.backend == 'onnx':
            landmarks, confidence = self.predict_onnx(img_preprocessed)
        elif self.backend == 'tflite':
            landmarks, confidence = self.predict_tflite(img_preprocessed)
        
        # Apply filtering
        if self.filter is not None:
            landmarks = self.filter.update(landmarks)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        avg_fps = np.mean(self.fps_history)
        
        return landmarks, confidence, avg_fps
    
    def draw_landmarks(self, image, landmarks, confidence=None, draw_connections=True, dotted=False):
        """
        Draw landmarks on image
        
        Args:
            image: RGB image
            landmarks: (21, 3) normalized landmarks
            confidence: (21,) confidence scores
            draw_connections: whether to draw connections
            dotted: whether to use dotted line style
        """
        h, w = image.shape[:2]
        img_draw = image.copy()
        
        # Denormalize landmarks
        landmarks_px = landmarks.copy()
        landmarks_px[:, 0] *= w
        landmarks_px[:, 1] *= h
        
        # Draw connections
        if draw_connections:
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = tuple(landmarks_px[start_idx, :2].astype(int))
                end_point = tuple(landmarks_px[end_idx, :2].astype(int))
                
                if dotted:
                    # Draw dotted line
                    self._draw_dotted_line(img_draw, start_point, end_point, (0, 255, 0), 2)
                else:
                    cv2.line(img_draw, start_point, end_point, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks_px):
            x, y = int(x), int(y)
            
            # Get color based on finger group
            color = self._get_landmark_color(i)
            
            # Draw landmark point
            cv2.circle(img_draw, (x, y), 4, color, -1)
            cv2.circle(img_draw, (x, y), 5, (255, 255, 255), 1)
            
            # Draw confidence if provided
            if confidence is not None:
                conf = confidence[i]
                if conf < CONFIDENCE_THRESHOLD:
                    # Draw warning for low confidence
                    cv2.circle(img_draw, (x, y), 8, (0, 0, 255), 2)
        
        return img_draw
    
    def _draw_dotted_line(self, img, pt1, pt2, color, thickness):
        """Draw dotted line"""
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        pts = []
        for i in np.arange(0, dist, 5):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
            pts.append((x, y))
        
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    
    def _get_landmark_color(self, landmark_idx):
        """Get color for landmark based on finger group"""
        for finger, indices in FINGER_GROUPS.items():
            if landmark_idx in indices:
                return FINGER_COLORS[finger]
        return (255, 255, 255)
    
    def extract_hand_bbox(self, landmarks, padding=60):
        """
        Extract bounding box from landmarks
        
        Args:
            landmarks: (21, 3) normalized landmarks
            padding: Padding around hand in pixels
            
        Returns:
            bbox: (x_min, y_min, x_max, y_max) in normalized coordinates
        """
        # Get min/max coordinates
        x_min = np.min(landmarks[:, 0])
        x_max = np.max(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        y_max = np.max(landmarks[:, 1])
        
        # Add padding (in normalized space, assuming 640x480 image)
        padding_norm_x = padding / 640.0
        padding_norm_y = padding / 480.0
        
        x_min = max(0.0, x_min - padding_norm_x)
        x_max = min(1.0, x_max + padding_norm_x)
        y_min = max(0.0, y_min - padding_norm_y)
        y_max = min(1.0, y_max + padding_norm_y)
        
        # Make square
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)
        
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        x_min = max(0.0, center_x - size / 2)
        x_max = min(1.0, center_x + size / 2)
        y_min = max(0.0, center_y - size / 2)
        y_max = min(1.0, center_y + size / 2)
        
        return (x_min, y_min, x_max, y_max)
    
    def reset_filter(self):
        """Reset temporal filter"""
        if self.filter is not None:
            self.filter.reset()


def benchmark_model(model_path, backend='pytorch', num_iterations=100):
    """
    Benchmark model performance
    
    Args:
        model_path: Path to model
        backend: Backend to use
        num_iterations: Number of iterations
    """
    print(f"\nBenchmarking {backend} backend...")
    print("="*50)
    
    # Create inference engine
    engine = HandLandmarkInference(model_path, backend=backend, use_kalman=False)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        engine.predict(dummy_image)
    
    # Benchmark
    print(f"Running {num_iterations} iterations...")
    times = []
    for _ in range(num_iterations):
        start = time.time()
        landmarks, confidence, _ = engine.predict(dummy_image)
        times.append(time.time() - start)
    
    # Results
    avg_time = np.mean(times) * 1000  # ms
    avg_fps = 1000 / avg_time
    std_time = np.std(times) * 1000
    
    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Average FPS: {avg_fps:.1f}")
    print(f"Min time: {np.min(times)*1000:.2f} ms")
    print(f"Max time: {np.max(times)*1000:.2f} ms")
    print("="*50)
    
    return avg_fps


if __name__ == "__main__":
    print("Testing inference engine...")
    
    # Note: This requires a trained model
    # For testing, we'll create a dummy model
    print("\nCreating dummy model for testing...")
    from model import create_model
    model = create_model(pretrained=False)
    
    # Save dummy model
    os.makedirs('hand_landmark_v2/models', exist_ok=True)
    torch.save(model.state_dict(), 'hand_landmark_v2/models/test_model.pth')
    
    # Test inference
    engine = HandLandmarkInference(
        'hand_landmark_v2/models/test_model.pth',
        backend='pytorch',
        use_kalman=True,
        filter_type='kalman'
    )
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Predict
    landmarks, confidence, fps = engine.predict(test_image)
    
    print(f"\n✓ Inference test successful")
    print(f"  Landmarks shape: {landmarks.shape}")
    print(f"  Confidence shape: {confidence.shape}")
    print(f"  FPS: {fps:.1f}")
    
    # Test visualization
    img_with_landmarks = engine.draw_landmarks(test_image, landmarks, confidence)
    print(f"  Visualization shape: {img_with_landmarks.shape}")
    
    # Benchmark
    benchmark_model('hand_landmark_v2/models/test_model.pth', backend='pytorch', num_iterations=50)
