"""
Training script that avoids zlibwapi.dll issue
Uses Pillow instead of OpenCV for image loading
"""
import os
import sys
import numpy as np
from pathlib import Path
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

# Use PIL instead of OpenCV
from PIL import Image

print("="*70)
print("SIGN LANGUAGE TRAINING - No zlib Issues")
print("="*70)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU: {gpus[0].name}")
    except:
        pass

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # Start with 3 epochs
DATASET_DIR = 'dataset'

print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")

# Get classes
classes = sorted([d for d in os.listdir(DATASET_DIR) 
                 if os.path.isdir(os.path.join(DATASET_DIR, d))])

print(f"\n✓ Found {len(classes)} classes")

# Custom data generator using PIL
def load_images_pil(dataset_dir, classes, img_size, validation_split=0.2):
    """Load images using PIL to avoid zlibwapi.dll"""
    all_images = []
    all_labels = []
    
    print("\nLoading images with PIL...")
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(cls_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  {cls}: {len(images)} images")
        
        for img_file in images:
            try:
                img_path = os.path.join(cls_path, img_file)
                # Use PIL to load
                img = Image.open(img_path).convert('RGB')
                img = img.resize((img_size, img_size), Image.BILINEAR)
                img_array = np.array(img) / 255.0
                
                all_images.append(img_array)
                all_labels.append(idx)
            except Exception as e:
                print(f"    Skipped {img_file}: {e}")
    
    # Convert to numpy
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, len(classes))
    y_val = keras.utils.to_categorical(y_val, len(classes))
    
    return (X_train, y_train), (X_val, y_val)

# Load data
(X_train, y_train), (X_val, y_val) = load_images_pil(
    DATASET_DIR, classes, IMG_SIZE)

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
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
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

# Train
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models_tf2/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# Create output dir
os.makedirs('models_tf2', exist_ok=True)

# Train
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# Save final model
model.save('models_tf2/final_model.h5')
print("\n✓ Model saved to models_tf2/")

# Save class names
with open('models_tf2/classes.txt', 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")
print("✓ Classes saved")

# Evaluate
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nModel saved to: models_tf2/")
print("Next step: Update classifier scripts to use this model")
