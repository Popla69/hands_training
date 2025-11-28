# API Documentation

Complete API reference for Hand Landmark Detection V2.

## Core Classes

### HandLandmarkModel

Neural network model for hand landmark detection.

```python
from hand_landmark_v2.model import HandLandmarkModel, create_model

# Create model
model = create_model(pretrained=True)

# Forward pass
landmarks, confidence = model(images)  # images: (B, 3, 224, 224)
# Returns:
#   landmarks: (B, 21, 3) - normalized x, y, z coordinates
#   confidence: (B, 21) - per-landmark confidence scores
```

**Methods:**
- `forward(x)`: Forward pass through the network

**Parameters:**
- `pretrained` (bool): Use ImageNet pretrained weights

---

### HandLandmarkInference

High-level inference engine with filtering and multi-backend support.

```python
from hand_landmark_v2.inference import HandLandmarkInference

# Initialize
engine = HandLandmarkInference(
    model_path='checkpoints/best_model.pth',
    backend='pytorch',  # 'pytorch', 'onnx', or 'tflite'
    use_kalman=True,
    filter_type='one_euro',  # 'kalman' or 'one_euro'
    use_gpu=False
)

# Predict
landmarks, confidence, fps = engine.predict(rgb_image)
# Returns:
#   landmarks: (21, 3) numpy array
#   confidence: (21,) numpy array
#   fps: float

# Draw landmarks
img_with_landmarks = engine.draw_landmarks(
    image, landmarks, confidence,
    draw_connections=True,
    dotted=False
)

# Extract bounding box
bbox = engine.extract_hand_bbox(landmarks, padding=60)
# Returns: (x_min, y_min, x_max, y_max) in normalized coordinates

# Reset filter
engine.reset_filter()
```

**Constructor Parameters:**
- `model_path` (str): Path to model file
- `backend` (str): Inference backend ('pytorch', 'onnx', 'tflite')
- `use_kalman` (bool): Enable temporal filtering
- `filter_type` (str): Filter type ('kalman' or 'one_euro')
- `use_gpu` (bool): Use GPU if available

**Methods:**
- `predict(image)`: Detect landmarks in image
- `draw_landmarks(image, landmarks, confidence, draw_connections, dotted)`: Visualize landmarks
- `extract_hand_bbox(landmarks, padding)`: Get bounding box
- `reset_filter()`: Reset temporal filter

---

### LandmarkKalmanFilter

Standard Kalman filter for landmark smoothing.

```python
from hand_landmark_v2.kalman_filter import LandmarkKalmanFilter

# Initialize
kalman = LandmarkKalmanFilter(
    num_landmarks=21,
    process_noise=0.01,
    measurement_noise=0.1
)

# Filter landmarks
filtered_landmarks = kalman.update(landmarks)  # (21, 3) array

# Reset
kalman.reset()
```

**Constructor Parameters:**
- `num_landmarks` (int): Number of landmarks (default: 21)
- `process_noise` (float): Process noise covariance
- `measurement_noise` (float): Measurement noise covariance

**Methods:**
- `update(landmarks)`: Filter landmarks
- `reset()`: Reset filter state

---

### LandmarkOneEuroFilter

One Euro filter for low-latency smoothing.

```python
from hand_landmark_v2.kalman_filter import LandmarkOneEuroFilter

# Initialize
one_euro = LandmarkOneEuroFilter(
    num_landmarks=21,
    min_cutoff=1.0,
    beta=0.007,
    d_cutoff=1.0
)

# Filter landmarks
filtered_landmarks = one_euro.update(landmarks)

# Reset
one_euro.reset()
```

**Constructor Parameters:**
- `num_landmarks` (int): Number of landmarks
- `min_cutoff` (float): Minimum cutoff frequency
- `beta` (float): Speed coefficient
- `d_cutoff` (float): Cutoff frequency for derivative

---

### HandLandmarkDataset

PyTorch dataset for training.

```python
from hand_landmark_v2.dataset import HandLandmarkDataset

# Create dataset
dataset = HandLandmarkDataset(
    data_dir='data/my_dataset',
    split='train',  # 'train', 'val', or 'test'
    augment=True
)

# Get item
image, landmarks, confidence = dataset[0]
# Returns:
#   image: (3, 224, 224) tensor
#   landmarks: (21, 3) tensor
#   confidence: (21,) tensor
```

**Constructor Parameters:**
- `data_dir` (str): Dataset directory
- `split` (str): Data split ('train', 'val', 'test')
- `augment` (bool): Apply data augmentation

---

### HandLandmarkTrainer

Training pipeline with validation and checkpointing.

```python
from hand_landmark_v2.train import HandLandmarkTrainer

# Create trainer
trainer = HandLandmarkTrainer(config={
    'data_dir': 'data/my_dataset',
    'output_dir': 'checkpoints',
    'num_epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.001
})

# Train
trainer.train()

# Resume from checkpoint
trainer.load_checkpoint('checkpoints/latest_checkpoint.pth')
trainer.train()
```

**Config Parameters:**
- `data_dir` (str): Dataset directory
- `output_dir` (str): Checkpoint output directory
- `log_dir` (str): TensorBoard log directory
- `num_epochs` (int): Number of training epochs
- `batch_size` (int): Batch size
- `learning_rate` (float): Learning rate
- `weight_decay` (float): Weight decay

**Methods:**
- `train(num_epochs)`: Run training loop
- `save_checkpoint(is_best)`: Save checkpoint
- `load_checkpoint(path)`: Load checkpoint

---

## Loss Functions

### WingLoss

Wing Loss for landmark localization.

```python
from hand_landmark_v2.losses import WingLoss

loss_fn = WingLoss(omega=10.0, epsilon=2.0)
loss = loss_fn(pred_landmarks, target_landmarks)
```

### HandLandmarkLoss

Combined loss for landmarks and confidence.

```python
from hand_landmark_v2.losses import HandLandmarkLoss

loss_fn = HandLandmarkLoss(
    landmark_weight=1.0,
    confidence_weight=0.5,
    use_wing_loss=True
)

total_loss, loss_dict = loss_fn(
    pred_landmarks, pred_confidence,
    target_landmarks, target_confidence
)
```

---

## Utility Functions

### Model Export

```python
from hand_landmark_v2.export import export_to_onnx, export_to_tflite

# Export to ONNX
export_to_onnx(
    model_path='checkpoints/best_model.pth',
    output_path='models/model.onnx',
    opset_version=12
)

# Export to TFLite
export_to_tflite(
    model_path='checkpoints/best_model.pth',
    output_path='models/model.tflite',
    quantize=False  # True for INT8 quantization
)
```

### Metrics

```python
from hand_landmark_v2.losses import compute_pck, compute_mean_error

# Percentage of Correct Keypoints
pck = compute_pck(pred_landmarks, target_landmarks, threshold=0.2)

# Mean Euclidean distance error
mean_error = compute_mean_error(pred_landmarks, target_landmarks)
```

### Benchmarking

```python
from hand_landmark_v2.inference import benchmark_model

# Benchmark performance
avg_fps = benchmark_model(
    model_path='checkpoints/best_model.pth',
    backend='pytorch',
    num_iterations=100
)
```

---

## Configuration

All configuration parameters are in `config.py`:

```python
from hand_landmark_v2 import config

# Model parameters
INPUT_SIZE = config.INPUT_SIZE  # (224, 224)
NUM_LANDMARKS = config.NUM_LANDMARKS  # 21
LANDMARK_DIM = config.LANDMARK_DIM  # 3

# Training parameters
BATCH_SIZE = config.BATCH_SIZE  # 32
LEARNING_RATE = config.LEARNING_RATE  # 0.001
NUM_EPOCHS = config.NUM_EPOCHS  # 100

# Landmark indices
WRIST = config.WRIST  # 0
THUMB_TIP = config.THUMB_TIP  # 4
INDEX_FINGER_TIP = config.INDEX_FINGER_TIP  # 8
# ... etc

# Hand connections for visualization
HAND_CONNECTIONS = config.HAND_CONNECTIONS

# Colors for visualization
FINGER_COLORS = config.FINGER_COLORS
```

---

## Examples

### Basic Inference

```python
import cv2
from hand_landmark_v2.inference import HandLandmarkInference

# Load model
engine = HandLandmarkInference('checkpoints/best_model.pth')

# Load image
image = cv2.imread('hand.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect landmarks
landmarks, confidence, fps = engine.predict(rgb_image)

# Visualize
result = engine.draw_landmarks(image, landmarks, confidence)
cv2.imshow('Result', result)
cv2.waitKey(0)
```

### Training Custom Model

```python
from hand_landmark_v2.train import HandLandmarkTrainer

# Configure training
config = {
    'data_dir': 'data/my_hands',
    'output_dir': 'my_checkpoints',
    'num_epochs': 50,
    'batch_size': 16,
    'learning_rate': 0.0005
}

# Train
trainer = HandLandmarkTrainer(config)
trainer.train()
```

### Video Processing

```python
import cv2
from hand_landmark_v2.inference import HandLandmarkInference

engine = HandLandmarkInference('checkpoints/best_model.pth')

cap = cv2.VideoCapture('video.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks, confidence, fps = engine.predict(rgb_frame)
    
    if landmarks is not None:
        frame = engine.draw_landmarks(frame, landmarks, confidence)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Error Handling

All functions raise standard Python exceptions:

- `FileNotFoundError`: Model or data file not found
- `ValueError`: Invalid parameter values
- `RuntimeError`: Inference or training errors
- `ImportError`: Missing dependencies

Example:

```python
try:
    engine = HandLandmarkInference('model.pth')
    landmarks, confidence, fps = engine.predict(image)
except FileNotFoundError:
    print("Model file not found")
except RuntimeError as e:
    print(f"Inference failed: {e}")
```

---

## Performance Tips

1. **Use GPU for training**: Set `use_gpu=True` if CUDA available
2. **Batch processing**: Process multiple images together
3. **ONNX for deployment**: Export to ONNX for faster inference
4. **Disable filtering for static images**: Set `use_kalman=False`
5. **Adjust filter parameters**: Tune for your specific use case

---

## Version Compatibility

- Python: 3.8, 3.9, 3.10
- PyTorch: 1.13.x
- OpenCV: 4.10.x
- NumPy: 1.26.x

See `requirements.txt` for complete dependency list.
