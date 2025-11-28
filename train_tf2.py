"""
Modern Training Script for TensorFlow 2.10
Trains sign language recognition model with GPU support
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*70)
print("SIGN LANGUAGE MODEL TRAINING - TensorFlow 2.10")
print("="*70)

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}")
if gpus:
    print(f"GPU: {gpus[0].name}")
    print("Training will use GPU acceleration!")
else:
    print("Training will use CPU only")

# Configuration
IMG_SIZE = 224  # MobileNetV2 input size
BATCH_SIZE = 32
EPOCHS = 10  # Adjust based on time available
DATASET_DIR = 'dataset'

print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Dataset: {DATASET_DIR}")

# Check dataset
if not os.path.exists(DATASET_DIR):
    print(f"\n✗ Dataset not found: {DATASET_DIR}")
    sys.exit(1)

# Count classes and images
classes = [d for d in os.listdir(DATASET_DIR) 
           if os.path.isdir(os.path.join(DATASET_DIR, d))]
classes.sort()

print(f"\n✓ Found {len(classes)} classes:")
print(f"  {', '.join(classes)}")

total_images = 0
for cls in classes:
    cls_path = os.path.join(DATASET_DIR, cls)
    num_images = len([f for f in os.listdir(cls_path) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))])
    total_images += num_images

print(f"\n✓ Total images: {total_images:,}")
print(f"  Average per class: {total_images // len(classes):,}")

# Data augmentation
print("\nSetting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,  # Don't flip hands
    fill_mode='nearest',
    validation_split=0.2  # 80% train, 20% validation
)

# Load training data
print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
print("Loading validation data...")
validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"\n✓ Data loaded:")
print(f"  Training samples: {train_generator.samples}")
print(f"  Validation samples: {validation_generator.samples}")
print(f"  Classes: {num_classes}")

# Build model
print("\nBuilding model...")
print("Using MobileNetV2 (optimized for speed)")

# Base model (pre-trained on ImageNet)
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# Add custom layers
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n✓ Model built:")
model.summary()

# Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models_tf2/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

# Create output directory
os.makedirs('models_tf2', exist_ok=True)

# Save class labels
with open('models_tf2/labels.txt', 'w') as f:
    for label in sorted(train_generator.class_indices.keys()):
        f.write(f"{label}\n")

print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"\nThis will take approximately {(train_generator.samples // BATCH_SIZE) * EPOCHS // 60} minutes")
print("Training on GPU..." if gpus else "Training on CPU...")
print("\nPress Ctrl+C to stop training\n")

start_time = time.time()

try:
    # Train model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTraining time: {training_time/60:.1f} minutes")
    print(f"Final training accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    
    # Fine-tuning (optional - unfreeze some layers)
    print("\n" + "="*70)
    print("FINE-TUNING (Optional)")
    print("="*70)
    
    response = input("\nDo you want to fine-tune the model? (y/n): ").strip().lower()
    
    if response == 'y':
        print("\nUnfreezing top layers for fine-tuning...")
        base_model.trainable = True
        
        # Freeze all layers except the last 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Fine-tuning for 5 more epochs...")
        
        history_fine = model.fit(
            train_generator,
            epochs=5,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\nFinal accuracy after fine-tuning: {history_fine.history['val_accuracy'][-1]*100:.2f}%")
    
    # Save final model
    print("\nSaving models...")
    
    # Save in TensorFlow SavedModel format (for GPU)
    model.save('models_tf2/saved_model')
    print("✓ Saved: models_tf2/saved_model (GPU-compatible)")
    
    # Save in H5 format (universal)
    model.save('models_tf2/model.h5')
    print("✓ Saved: models_tf2/model.h5 (CPU/GPU compatible)")
    
    # Convert to TFLite (for mobile/edge devices)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('models_tf2/model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("✓ Saved: models_tf2/model.tflite (Mobile/Edge)")
    
    print("\n" + "="*70)
    print("ALL MODELS SAVED!")
    print("="*70)
    print("\nModel files:")
    print("  1. models_tf2/saved_model/ - TensorFlow SavedModel (GPU)")
    print("  2. models_tf2/model.h5 - Keras H5 (CPU/GPU)")
    print("  3. models_tf2/model.tflite - TensorFlow Lite (Mobile)")
    print("  4. models_tf2/labels.txt - Class labels")
    
    print("\nTo use the model:")
    print("  GPU: Load 'saved_model' or 'model.h5'")
    print("  CPU: Load 'model.h5'")
    print("  Mobile: Load 'model.tflite'")
    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    print("Saving current model...")
    model.save('models_tf2/model_interrupted.h5')
    print("✓ Saved: models_tf2/model_interrupted.h5")

except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
