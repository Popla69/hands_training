"""
CPU-Only Training Script - No GPU, No Errors
Uses current working environment
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
import numpy as np
import glob

print("="*70)
print("CPU-ONLY Training - Sign Language Model")
print("="*70)

# Verify CPU only
print(f"\nTensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Devices: {tf.config.list_physical_devices()}")
print("GPU disabled - using CPU only")

# Config
IMG_SIZE = 224
BATCH_SIZE = 16  # Smaller for CPU
EPOCHS = 2  # Quick training
DATASET_DIR = 'dataset'

print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Device: CPU")

# Get classes
classes = sorted([d for d in os.listdir(DATASET_DIR) 
                 if os.path.isdir(os.path.join(DATASET_DIR, d))])

print(f"\n✓ Found {len(classes)} classes")

# Load images with PIL (no zlib issues)
def load_images(dataset_dir, classes, img_size, max_per_class=500):
    """Load images using PIL - CPU friendly"""
    all_images = []
    all_labels = []
    
    print("\nLoading images...")
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(dataset_dir, cls)
        images = glob.glob(os.path.join(cls_path, '*.jpg'))
        images += glob.glob(os.path.join(cls_path, '*.jpeg'))
        images += glob.glob(os.path.join(cls_path, '*.png'))
        
        # Limit images per class for faster training
        images = images[:max_per_class]
        
        print(f"  {cls}: {len(images)} images")
        
        for img_file in images:
            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize((img_size, img_size), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                all_images.append(img_array)
                all_labels.append(idx)
            except Exception as e:
                pass
    
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split 80/20
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, len(classes))
    y_val = keras.utils.to_categorical(y_val, len(classes))
    
    return (X_train, y_train), (X_val, y_val)

# Load data
(X_train, y_train), (X_val, y_val) = load_images(DATASET_DIR, classes, IMG_SIZE)

print(f"\n✓ Data loaded:")
print(f"  Training: {len(X_train)} images")
print(f"  Validation: {len(X_val)} images")

# Build model
print("\nBuilding model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built")
model.summary()

# Create output dir
os.makedirs('models_tf2', exist_ok=True)

# Train
print("\n" + "="*70)
print("STARTING TRAINING (CPU)")
print("="*70)
print("This will take 10-20 minutes on CPU...")

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    verbose=1
)

# Save model
model.save('models_tf2/sign_language_model.h5')
print("\n✓ Model saved to models_tf2/sign_language_model.h5")

# Save class names
with open('models_tf2/classes.txt', 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")
print("✓ Classes saved to models_tf2/classes.txt")

# Evaluate
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nModel saved successfully!")
print("Next: Update classify_webcam_v2.py to use this model")
