"""
Configuration for lightweight hand landmark detection model
"""

# Model Architecture
MODEL_NAME = "MobileNetV3_HandLandmark"
BACKBONE = "mobilenetv3_small"  # mobilenetv3_small or mobilenetv3_large
INPUT_SIZE = (224, 224)  # Optimized for speed
NUM_LANDMARKS = 21
OUTPUT_CHANNELS = NUM_LANDMARKS * 3  # x, y, z for each landmark

# Training Configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
WEIGHT_DECAY = 1e-4

# Data Augmentation
AUG_ROTATION_RANGE = 30
AUG_SCALE_RANGE = (0.8, 1.2)
AUG_BRIGHTNESS_RANGE = (0.7, 1.3)
AUG_CONTRAST_RANGE = (0.8, 1.2)
AUG_NOISE_STD = 0.02

# Kalman Filter Configuration
KALMAN_PROCESS_NOISE = 0.001
KALMAN_MEASUREMENT_NOISE = 0.01
KALMAN_INITIAL_COVARIANCE = 1.0

# Model Export
EXPORT_TFLITE = True
EXPORT_ONNX = True
EXPORT_PYTORCH = True
TFLITE_QUANTIZE = True  # INT8 quantization for smaller size

# Performance Targets
TARGET_FPS = 30
MAX_MEMORY_MB = 2000
MAX_MODEL_SIZE_MB = 50

# Dataset Paths
DATASET_ROOT = "datasets/hand_landmarks"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Hand Landmark Indices
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

# Hand Connections for Visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]
