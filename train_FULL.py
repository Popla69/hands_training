"""
FULL Training Script - All Images, High Accuracy
CPU-only, no errors, 95%+ target accuracy
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from PIL import Image
import numpy as np
import glob
import time

print("="*70)
print("FULL TRAINING - All Images, High Accuracy Target")
print("="*70)

# Verify setup
print(f"\nTensorFlow: {tf.__version__}")
print(f"NumPy: {np.__version__}")
print("Device: CPU only")

# Config for high accuracy
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10  # More epochs for better accuracy
DATASET_DIR = 'dataset'

print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Target: 95%+ accuracy")

# Get classes
classes = sorted([d for d in os.listdir(DATASET_DIR) 
                 if os.path.isdir(os.path.join(DATASET_DIR, d))])

print(f"\n✓ Found {len(classes)} classes:")
print(f"  {', '.join(classes)}")

# Load ALL images
def load_all_images(dataset_dir, classes, img_size):
    """Load ALL images - no limits"""
    all_images = []
    all_labels = []
    
    print("\nLoading ALL images...")
    total_images = 0
    
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(dataset_dir, cls)
        images = glob.glob(os.path.join(cls_path, '*.jpg'))
        images += glob.glob(os.path.join(cls_path, '*.jpeg'))
        images += glob.glob(os.path.join(cls_path, '*.png'))
        
        print(f"  {cls}: {len(images)} images", end='')
        
        loaded = 0
        for img_file in images:
            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize((img_size, img_size), Image.BILINEAR)
                img_array = np.array(img, dtype=np.float32) / 255.0
                
                all_images.append(img_array)
                all_labels.append(idx)
                loaded += 1
            except:
                pass
        
        print(f" -> {loaded} loaded")
        total_images += loaded
    
    print(f"\n✓ Total images loaded: {total_images:,}")
    
    X = np.array(all_images, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Shuffle
    print("Shuffling data...")
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
start_time = time.time()
(X_train, y_train), (X_val, y_val) = load_all_images(DATASET_DIR, classes, IMG_SIZE)
load_time = time.time() - start_time

print(f"\n✓ Data loaded in {load_time/60:.1f} minutes:")
print(f"  Training: {len(X_train):,} images")
print(f"  Validation: {len(X_val):,} images")

# Build model with more capacity for high accuracy
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

# Callbacks for better training
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
print("STARTING FULL TRAINING")
print("="*70)
print(f"Training on {len(X_train):,} images for {EPOCHS} epochs")
print("This will take 1-3 hours on CPU...")
print("="*70 + "\n")

start_time = time.time()

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

train_time = time.time() - start_time

# Save final model
model.save('models_tf2/final_model.h5')
print(f"\n✓ Model saved to models_tf2/")

# Save class names
with open('models_tf2/classes.txt', 'w') as f:
    for cls in classes:
        f.write(f"{cls}\n")
print("✓ Classes saved")

# Final evaluation
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

loss, accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"\nValidation Accuracy: {accuracy*100:.2f}%")
print(f"Validation Loss: {loss:.4f}")

# Per-class accuracy
print("\nPer-class accuracy:")
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

for idx, cls in enumerate(classes):
    mask = y_true_classes == idx
    if mask.sum() > 0:
        cls_acc = (y_pred_classes[mask] == idx).mean()
        print(f"  {cls}: {cls_acc*100:.1f}%")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print(f"\nTotal training time: {train_time/3600:.1f} hours")
print(f"Final accuracy: {accuracy*100:.2f}%")
print(f"\nModel saved to: models_tf2/best_model.h5")
print("="*70)
