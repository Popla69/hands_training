"""
Fine-tune model specifically for class X
Uses class weighting to focus on X without hurting other classes
"""
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import os
import numpy as np
from datetime import datetime

print("="*70)
print("FINE-TUNING FOR CLASS X")
print("="*70)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
BASE_MODEL_PATH = 'models_tf2/checkpoint_resume.h5'
FINETUNED_MODEL_PATH = 'models_tf2/finetuned_X_model.h5'
CHECKPOINT_PATH = 'models_tf2/finetune_X_checkpoint.h5'
LOG_FILE = 'models_tf2/finetune_X_log.csv'
DATASET_DIR = 'dataset'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10  # Short fine-tuning
LEARNING_RATE = 0.0001  # Lower learning rate for fine-tuning

print("Loading base model...")
model = load_model(BASE_MODEL_PATH)
print("✓ Base model loaded")

# Get class names
classes = sorted([d for d in os.listdir(DATASET_DIR) 
                 if os.path.isdir(os.path.join(DATASET_DIR, d))])
num_classes = len(classes)
print(f"✓ Found {num_classes} classes")

# Find X index
x_index = classes.index('X')
print(f"✓ Class X is at index {x_index}")

# Calculate class weights - give X much higher weight
class_weights = {}
for i in range(num_classes):
    if i == x_index:
        class_weights[i] = 5.0  # 5x weight for X
    else:
        class_weights[i] = 1.0

print(f"\nClass weights:")
print(f"  X: {class_weights[x_index]}")
print(f"  Others: 1.0")

# Data augmentation - more aggressive for X
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # More rotation
    width_shift_range=0.2,  # More shift
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    brightness_range=[0.8, 1.2],  # Add brightness variation
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

print("\nLoading data...")
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")

# Recompile with lower learning rate
print(f"\nRecompiling model with learning rate: {LEARNING_RATE}")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
checkpoint = ModelCheckpoint(
    CHECKPOINT_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

csv_logger = CSVLogger(LOG_FILE, append=False)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, csv_logger, early_stopping, reduce_lr]

# Fine-tune
print("\n" + "="*70)
print("STARTING FINE-TUNING")
print("="*70)
print(f"Epochs: {EPOCHS}")
print(f"Focus: Class X (5x weight)")
print(f"Strategy: Preserve other classes, improve X")
print()

try:
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        class_weight=class_weights,  # Key: weighted training
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print(f"\n[OK] Saving fine-tuned model to {FINETUNED_MODEL_PATH}...")
    model.save(FINETUNED_MODEL_PATH)
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE!")
    print("="*70)
    
    # Show improvement
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"\nFinal validation accuracy: {final_val_acc:.4f}")
    print(f"Fine-tuned model saved: {FINETUNED_MODEL_PATH}")
    print(f"Best checkpoint: {CHECKPOINT_PATH}")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Test the fine-tuned model:")
    print("   python test_finetuned_model.py")
    print("\n2. If X accuracy improved without hurting others:")
    print(f"   - Replace checkpoint_resume.h5 with finetuned_X_model.h5")
    print("\n3. If results are worse:")
    print("   - Keep original model (checkpoint_resume.h5)")

except KeyboardInterrupt:
    print("\n\nFine-tuning interrupted!")
    print("Partial progress may be saved in checkpoint.")

except Exception as e:
    print(f"\n\nError during fine-tuning: {e}")
    import traceback
    traceback.print_exc()
