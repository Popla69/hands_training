"""
SUPER QUICK test - CPU only, minimal model, 5 images per class
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU only for speed

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

print("TensorFlow version:", tf.__version__)
print("Running on CPU only for quick test")

# Paths
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'models_tf2/super_quick_test.h5'

# Super quick parameters
IMG_SIZE = 64  # Smaller images
BATCH_SIZE = 8
EPOCHS = 1  # Just 1 epoch
MAX_IMAGES = 5  # Only 5 images per class

os.makedirs('models_tf2', exist_ok=True)

# Count classes
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
num_classes = len(classes)
print(f"\nFound {num_classes} classes")

# Create tiny dataset
TEMP_DATASET = 'dataset_super_quick'
if os.path.exists(TEMP_DATASET):
    shutil.rmtree(TEMP_DATASET)

print(f"\nCreating tiny dataset with {MAX_IMAGES} images per class...")
total_images = 0
for class_name in classes:
    src_dir = os.path.join(DATASET_DIR, class_name)
    dst_dir = os.path.join(TEMP_DATASET, class_name)
    os.makedirs(dst_dir, exist_ok=True)
    
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES]
    
    for img in images:
        shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    
    total_images += len(images)

print(f"Total images: {total_images}")

# Simple data generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    TEMP_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    TEMP_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"Training: {train_gen.samples}, Validation: {val_gen.samples}")

# Simple CNN model
print("\nBuilding simple model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "="*60)
print("TRAINING (This should take < 1 minute)")
print("="*60 + "\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    verbose=1
)

# Save
print(f"\nSaving to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)

# Test loading
print("Testing load...")
loaded = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Test prediction
print("Testing prediction...")
import numpy as np
test_batch = next(iter(val_gen))
pred = loaded.predict(test_batch[0][:1], verbose=0)
print(f"Prediction shape: {pred.shape}")
print(f"Predicted class: {np.argmax(pred[0])}")

# Cleanup
shutil.rmtree(TEMP_DATASET)

print("\n" + "="*60)
print("✓ SUPER QUICK TEST PASSED!")
print("="*60)
print("\nEverything works! The pipeline is solid.")
print("You can now run full training with confidence.")
