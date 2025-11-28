"""
Memory-Efficient Training - Uses Data Generators
No memory errors, works on any system
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("="*70)
print("Memory-Efficient Training - Data Generators")
print("="*70)

print(f"\nTensorFlow: {tf.__version__}")
print("Device: CPU only")

# Config
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = 'dataset'

print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")

# Data generators - load images on-the-fly (memory efficient)
print("\nSetting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n✓ Data generators ready:")
print(f"  Training: {train_generator.samples} images")
print(f"  Validation: {val_generator.samples} images")
print(f"  Classes: {train_generator.num_classes}")

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
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built")

# Create output dir
os.makedirs('models_tf2', exist_ok=True)

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
        min_lr=0.00001,
        verbose=1
    )
]

# Train
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print("Using data generators - memory efficient!")
print("="*70 + "\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save
model.save('models_tf2/final_model.h5')
print("\n✓ Model saved to models_tf2/")

# Save class names
classes = list(train_generator.class_indices.keys())
with open('models_tf2/classes.txt', 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")
print("✓ Classes saved")

# Evaluate
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy = model.evaluate(val_generator, verbose=0)
print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"Final accuracy: {accuracy*100:.2f}%")
print("Model saved to: models_tf2/best_model.h5")
print("="*70)
