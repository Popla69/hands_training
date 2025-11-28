"""
Resumable Training - Can continue from last checkpoint if interrupted
"""
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import os
import json
from datetime import datetime

print("="*70)
print("RESUMABLE TRAINING - Sign Language Recognition")
print("="*70)
print(f"\nTensorFlow version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Configuration
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'models_tf2/sign_language_model.h5'
CHECKPOINT_PATH = 'models_tf2/checkpoint_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.h5'
RESUME_CHECKPOINT = 'models_tf2/checkpoint_resume.h5'
TRAINING_STATE_FILE = 'models_tf2/training_state.json'
LOG_FILE = 'models_tf2/training_log.csv'

IMG_SIZE = 224
BATCH_SIZE = 32
TOTAL_EPOCHS = 50
LEARNING_RATE = 0.001

# Create output directory
os.makedirs('models_tf2', exist_ok=True)

# Check for existing training state
initial_epoch = 0
resume_from_checkpoint = False

if os.path.exists(TRAINING_STATE_FILE):
    print("\n" + "="*70)
    print("PREVIOUS TRAINING FOUND!")
    print("="*70)
    try:
        with open(TRAINING_STATE_FILE, 'r') as f:
            state = json.load(f)
        
        print(f"\nLast training session:")
        print(f"  - Completed epochs: {state['last_epoch']}")
        print(f"  - Best validation accuracy: {state['best_val_accuracy']:.4f}")
        print(f"  - Checkpoint: {state['checkpoint_path']}")
        
        if os.path.exists(state['checkpoint_path']):
            response = input("\nResume from this checkpoint? (y/n): ").strip().lower()
            if response == 'y':
                resume_from_checkpoint = True
                initial_epoch = state['last_epoch']
                RESUME_CHECKPOINT = state['checkpoint_path']
                print(f"\n[OK] Will resume from epoch {initial_epoch + 1}")
            else:
                print("\n[OK] Starting fresh training")
        else:
            print("\n[WARNING] Checkpoint file not found, starting fresh")
    except Exception as e:
        print(f"\n[WARNING] Could not load training state: {e}")
        print("Starting fresh training")

# Count classes and images
classes = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
num_classes = len(classes)
total_images = sum([len([f for f in os.listdir(os.path.join(DATASET_DIR, c)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) 
                   for c in classes])

print(f"\nDataset Info:")
print(f"  Classes: {num_classes}")
print(f"  Total images: {total_images}")

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=False,
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

# Build or load model
if resume_from_checkpoint:
    print(f"\n[OK] Loading model from checkpoint: {RESUME_CHECKPOINT}")
    model = load_model(RESUME_CHECKPOINT)
    print("[OK] Model loaded successfully")
else:
    print("\nBuilding new MobileNetV2 model...")
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Total parameters: {model.count_params():,}")

# Custom callback to save training state
class TrainingStateCallback(tf.keras.callbacks.Callback):
    def __init__(self, state_file, checkpoint_path):
        super().__init__()
        self.state_file = state_file
        self.checkpoint_path = checkpoint_path
        self.best_val_accuracy = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy', 0)
        
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
        
        # Save training state
        state = {
            'last_epoch': epoch + 1,
            'best_val_accuracy': float(self.best_val_accuracy),
            'checkpoint_path': self.checkpoint_path,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    RESUME_CHECKPOINT,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

state_callback = TrainingStateCallback(TRAINING_STATE_FILE, RESUME_CHECKPOINT)

csv_logger = CSVLogger(LOG_FILE, append=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [
    checkpoint_callback,
    state_callback,
    csv_logger,
    early_stopping,
    reduce_lr
]

# Training
print("\n" + "="*70)
if resume_from_checkpoint:
    print(f"RESUMING TRAINING FROM EPOCH {initial_epoch + 1}")
else:
    print("STARTING NEW TRAINING")
print("="*70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total epochs: {TOTAL_EPOCHS}")
print(f"Starting from epoch: {initial_epoch + 1}")
print()

try:
    history = model.fit(
        train_generator,
        initial_epoch=initial_epoch,
        epochs=TOTAL_EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print(f"\n[OK] Saving final model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    # Save class labels
    labels_path = 'models_tf2/labels.txt'
    with open(labels_path, 'w') as f:
        for class_name in sorted(train_generator.class_indices.keys()):
            f.write(f"{class_name}\n")
    print(f"[OK] Saved class labels to {labels_path}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final model: {MODEL_SAVE_PATH}")
    print(f"Best checkpoint: {RESUME_CHECKPOINT}")

except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("TRAINING INTERRUPTED!")
    print("="*70)
    print("\nDon't worry! Your progress has been saved.")
    print(f"Checkpoint saved at: {RESUME_CHECKPOINT}")
    print(f"Training state saved at: {TRAINING_STATE_FILE}")
    print("\nTo resume training, simply run this script again!")
    print("It will automatically detect and resume from the last checkpoint.")

except Exception as e:
    print(f"\n\n[ERROR] Training failed: {e}")
    print("\nCheckpoint and training state have been saved.")
    print("You can try resuming by running this script again.")
