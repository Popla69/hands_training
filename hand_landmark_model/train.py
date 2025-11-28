"""
Training script for hand landmark detection model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json

from model import create_model, WingLoss
from dataset import HandLandmarkDataset, get_augmentation_pipeline, create_synthetic_dataset
from config import *


class Trainer:
    """Model trainer"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss functions
        self.landmark_loss = WingLoss()
        self.confidence_loss = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            images = batch['image'].float().to(self.device)
            landmarks_gt = batch['landmarks'].float().to(self.device)
            confidence_gt = batch['confidence'].float().to(self.device)
            
            # Forward pass
            landmarks_pred, confidence_pred = self.model(images)
            
            # Calculate losses
            loss_landmarks = self.landmark_loss(landmarks_pred, landmarks_gt, confidence_gt)
            loss_confidence = self.confidence_loss(confidence_pred, confidence_gt)
            
            loss = loss_landmarks + 0.1 * loss_confidence
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['image'].float().to(self.device)
                landmarks_gt = batch['landmarks'].float().to(self.device)
                confidence_gt = batch['confidence'].float().to(self.device)
                
                # Forward pass
                landmarks_pred, confidence_pred = self.model(images)
                
                # Calculate losses
                loss_landmarks = self.landmark_loss(landmarks_pred, landmarks_gt, confidence_gt)
                loss_confidence = self.confidence_loss(confidence_pred, confidence_gt)
                
                loss = loss_landmarks + 0.1 * loss_confidence
                total_loss += loss.item()
                
                # Calculate accuracy (within 5% threshold)
                diff = torch.abs(landmarks_pred - landmarks_gt)
                accuracy = (diff < 0.05).float().mean()
                total_accuracy += accuracy.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_accuracy / len(self.val_loader)
        
        return avg_loss, avg_accuracy
    
    def train(self, num_epochs=NUM_EPOCHS, save_dir='checkpoints'):
        """Full training loop"""
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_accuracy = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy*100:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy
                }, checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save training history
        with open(os.path.join(save_dir, 'history.json'), 'w') as f:
            json.dump(self.history, f)
        
        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic dataset for testing
    print("Creating synthetic dataset...")
    dataset_dir = create_synthetic_dataset(num_samples=1000)
    
    # Create datasets
    train_dataset = HandLandmarkDataset(
        dataset_dir, 
        split='train',
        transform=get_augmentation_pipeline('train')
    )
    
    val_dataset = HandLandmarkDataset(
        dataset_dir,
        split='val',
        transform=None
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = create_model(pretrained=True)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, device=device)
    
    # Train
    trainer.train(num_epochs=NUM_EPOCHS, save_dir='checkpoints')


if __name__ == "__main__":
    main()
