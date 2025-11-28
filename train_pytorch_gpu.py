"""
PyTorch GPU Training Script
Trains sign language model on GPU using PyTorch
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

print("="*70)
print("PYTORCH GPU TRAINING - Sign Language Recognition")
print("="*70)

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("✓ Training will use GPU!")
else:
    print("✗ GPU not available, using CPU")
    sys.exit(1)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 64  # Larger batch for GPU
EPOCHS = 15
DATASET_DIR = 'dataset'
OUTPUT_DIR = 'models_pytorch_gpu'

print(f"\nConfiguration:")
print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Output: {OUTPUT_DIR}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom Dataset
class SignLanguageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all image paths
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_path, img_name), self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
print("\nLoading dataset...")
full_dataset = SignLanguageDataset(DATASET_DIR, transform=train_transform)

# Split train/val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

# Update val dataset transform
val_dataset.dataset.transform = val_transform

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

num_classes = len(full_dataset.classes)
print(f"✓ Dataset loaded:")
print(f"  Classes: {num_classes}")
print(f"  Total samples: {len(full_dataset)}")
print(f"  Training: {len(train_dataset)}")
print(f"  Validation: {len(val_dataset)}")

# Save class labels
with open(f'{OUTPUT_DIR}/labels.txt', 'w') as f:
    for cls in full_dataset.classes:
        f.write(f"{cls}\n")

# Build model
print("\nBuilding model...")
print("Using MobileNetV2 (optimized for speed)")

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

print("✓ Model built")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total

# Training loop
print("\n" + "="*70)
print("STARTING GPU TRAINING")
print("="*70)

best_acc = 0.0
start_time = time.time()

try:
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{OUTPUT_DIR}/best_model.pth')
            print(f"✓ Saved best model (acc: {best_acc:.2f}%)")
        
        scheduler.step(val_acc)
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {training_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), f'{OUTPUT_DIR}/final_model.pth')
    print(f"\n✓ Saved: {OUTPUT_DIR}/best_model.pth")
    print(f"✓ Saved: {OUTPUT_DIR}/final_model.pth")
    print(f"✓ Saved: {OUTPUT_DIR}/labels.txt")
    
    print("\nModel trained on GPU successfully!")
    print("Use this model for fast inference on GPU")

except KeyboardInterrupt:
    print("\n\nTraining interrupted")
    torch.save(model.state_dict(), f'{OUTPUT_DIR}/interrupted_model.pth')
    print(f"✓ Saved: {OUTPUT_DIR}/interrupted_model.pth")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
