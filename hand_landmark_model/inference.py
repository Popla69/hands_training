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
    
    def __init__(self, model_path, backend='pytorch', use_kalman=True, use_gpu=False):
        self.backend = backend
        self.use_kalman = use_kalman
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Load model
        if backend == 'pytorch':
            self.model = self._load_pytorch(model_path)
        elif backend == 'onnx':
            self.model = self._load_onnx(model_path)
        elif backend == 'tflite':
            self.model = self._load_tflite(model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Initialize Kalman filter
        if use_kalman:
            self.kalman_filter = LandmarkOneEuroFilter(num_landmarks=NUM_LANDMARKS)
        else:
            self.kalman_filter = None
        
        # Performance tracking
        self.fps_history = []
        self.last_time = time.time()
        
    def _load_pytorch(self, model_path):
        """Load PyTorch model"""
        model = create_model(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        print(f"Loaded PyTorch model from {model_path}")
        return model
    
    def _load_onnx(self, model_path):
        """Load ONNX model"""
        import onnxruntime as ort
        
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda':
            providers.insert(0, 'CUDAExecutionProvider')
        
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"Loaded ONNX model from {model_path}")
        print(f"Providers: {session.get_providers()}")
        return session
    
    def _load_tflite(self, model_path):
        """Load TFLite model"""
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"Loaded TFLite model from {model_path}")
        return interpreter
    
    def preprocess(self, image):
        """Preprocess image for model input"""
        # Resize
        img = cv2.resize(image, INPUT_SIZE)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
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
            landmarks: (21, 3) array of x, y, z coordinates
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
        
        # Apply Kalman filtering
        if self.kalman_filter is not None:
            landmarks = self.kalman_filter.update(landmarks)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        
        avg_fps = np.mean(self.fps_history)
        
        return landmarks, confidence, avg_fps
    
    def draw_landmarks(self, image, landmarks, confidence=None, draw_connections=True):
        """
        Draw landmarks on image with dotted overlay style
        
        Args:
            image: RGB image
            landmarks: (21, 3) normalized landmarks
            confidence: (21,) confidence scores
            draw_connections: whether to draw connections
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
                
                # Dotted line effect
                cv2.line(img_draw, start_point, end_point, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Draw landmarks
        for i, (x, y, z) in enumerate(landmarks_px):
            x, y = int(x), int(y)
            
            # Color based on confidence
            if confidence is not None:
                conf = confidence[i]
                color = (0, int(255 * conf), int(255 * (1 - conf)))
            else:
                color = (0, 255, 0)
            
            # Draw landmark point
            cv2.circle(img_draw, (x, y), 5, color, -1)
            cv2.circle(img_draw, (x, y), 6, (255, 255, 255), 1)
            
            # Draw landmark index
            cv2.putText(img_draw, str(i), (x+8, y+8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return img_draw


def benchmark_model(model_path, backend='pytorch', num_iterations=100):
    """Benchmark model performance"""
    
    print(f"\nBenchmarking {backend} backend...")
    print("="*50)
    
    # Create inference engine
    engine = HandLandmarkInference(model_path, backend=backend, use_kalman=False)
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Warmup
    for _ in range(10):
        engine.predict(dummy_image)
    
    # Benchmark
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
    # Test inference
    print("Testing inference engine...")
    
    # Create dummy model for testing
    model = create_model(pretrained=False)
    torch.save(model.state_dict(), 'models/test_model.pth')
    
    # Benchmark
    benchmark_model('models/test_model.pth', backend='pytorch', num_iterations=50)
