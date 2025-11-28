"""
Full training with all dataset - Clean version with no warnings
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime

print("="*70)
print("SIGN LANGUAGE RECOGNITION - FULL TRAINING")
print("="*70)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configuration
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'models_tf2/sign_language_model.h5'
CHECKPOINT_PATH = 'models_tf2/checkpoint_best.h5'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Create output directory
os.makedirs('models_tf2', exist_ok=True)

# Count classes and images
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
num_classes = len(classes)
total_images = sum([len([f for f in os.listdir(os.path.join(DATASET_DIR, c)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                   for c in classes])

print(f"\nDataset Info:")
print(f"  Classes: {num_classes}")
print(f"  Total images: {total_images}")
print(f"  Classes: {', '.join(sorted(classes))}")

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,  # Don't flip - sign language is not symmetric
    fill_mode='nearest'
)

# Only rescaling for validation
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

print("Loading validation data...")
val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\nTraining samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Build model
print("\nBuilding MobileNetV2 model...")
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model initially
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel architecture:")
print(f"  Total parameters: {model.count_params():,}")
print(f"  Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")

# Callbacks
callbacks = [
    ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
]

# Training
print("\n" + "="*70)
print("STARTING TRAINING")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

# Save final model
print(f"\nSaving final model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)

# Save class labels
labels_path = 'models_tf2/labels.txt'
with open(labels_path, 'w') as f:
    for class_name in sorted(train_generator.class_indices.keys()):
        f.write(f"{class_name}\n")
print(f"Saved class labels to {labels_path}")

# Summary
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"\nModel saved to: {MODEL_SAVE_PATH}")
print(f"Best checkpoint: {CHECKPOINT_PATH}")
print(f"Class labels: {labels_path}")
print("\nYou can now use this model for real-time classification!")
