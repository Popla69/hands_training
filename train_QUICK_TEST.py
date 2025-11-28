"""
Quick test training with minimal data to verify pipeline works
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Paths
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'models_tf2/test_model.h5'

# Quick test parameters
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 2  # Just 2 epochs to test
MAX_IMAGES_PER_CLASS = 10  # Only 10 images per class

# Create output directory
os.makedirs('models_tf2', exist_ok=True)

# Count classes
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
num_classes = len(classes)
print(f"\nFound {num_classes} classes: {classes}")

# Create a temporary small dataset
import shutil
TEMP_DATASET = 'dataset_test_small'
if os.path.exists(TEMP_DATASET):
    shutil.rmtree(TEMP_DATASET)

print(f"\nCreating small test dataset with max {MAX_IMAGES_PER_CLASS} images per class...")
for class_name in classes:
    src_dir = os.path.join(DATASET_DIR, class_name)
    dst_dir = os.path.join(TEMP_DATASET, class_name)
    os.makedirs(dst_dir, exist_ok=True)
    
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = images[:MAX_IMAGES_PER_CLASS]
    
    for img in images:
        shutil.copy2(os.path.join(src_dir, img), os.path.join(dst_dir, img))
    
    print(f"  {class_name}: {len(images)} images")

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

train_generator = datagen.flow_from_directory(
    TEMP_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    TEMP_DATASET,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Build model
print("\nBuilding model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base model for quick test

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel summary:")
model.summary()

# Train
print(f"\n{'='*60}")
print("STARTING QUICK TEST TRAINING")
print(f"{'='*60}\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    verbose=1
)

# Save model
print(f"\nSaving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")

# Test loading the model
print("\nTesting model loading...")
loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
print("Model loaded successfully!")

# Test prediction
print("\nTesting prediction on a sample...")
import numpy as np
test_batch = next(iter(val_generator))
test_images, test_labels = test_batch
prediction = loaded_model.predict(test_images[:1])
predicted_class = np.argmax(prediction[0])
actual_class = np.argmax(test_labels[0])
print(f"Predicted class index: {predicted_class}")
print(f"Actual class index: {actual_class}")
print(f"Prediction confidence: {prediction[0][predicted_class]:.2%}")

# Cleanup
print(f"\nCleaning up temporary dataset...")
shutil.rmtree(TEMP_DATASET)

print(f"\n{'='*60}")
print("✓ QUICK TEST COMPLETED SUCCESSFULLY!")
print(f"{'='*60}")
print("\nThe training pipeline works! You can now:")
print("1. Run the full training with all data")
print("2. Use the test model at:", MODEL_SAVE_PATH)
print("\nClass mapping:")
for idx, class_name in enumerate(train_generator.class_indices.keys()):
    print(f"  {idx}: {class_name}")
