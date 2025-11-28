"""
Training pipeline for hand landmark detection
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import create_model, count_parameters
from dataset import HandLandmarkDataset, create_synthetic_dataset
from losses import HandLandmarkLoss, compute_pck, compute_mean_error
from config import *


class HandLandmarkTrainer:
    """
    Training pipeline with augmentation and validation
    """
    
    def __init__(self, config=None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', BATCH_SIZE)
        self.learning_rate = self.config.get('learning_rate', LEARNING_RATE)
        self.num_epochs = self.config.get('num_epochs', NUM_EPOCHS)
        self.weight_decay = self.config.get('weight_decay', WEIGHT_DECAY)
        
        # Paths
        self.data_dir = self.config.get('data_dir', 'data/synthetic_hands')
        self.output_dir = self.config.get('output_dir', 'hand_landmark_v2/checkpoints')
        self.log_dir = self.config.get('log_dir', 'hand_landmark_v2/logs')
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model
        print("Creating model...")
        self.model = create_model(pretrained=True)
        self.model.to(self.device)
        print(f"Model parameters: {count_parameters(self.model):,}")
        
        # Loss
        self.criterion = HandLandmarkLoss(
            landmark_weight=LANDMARK_LOSS_WEIGHT,
            confidence_weight=CONFIDENCE_LOSS_WEIGHT,
            use_wing_loss=True
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_pck': [],
            'val_pck': [],
            'learning_rate': []
        }
    
    def setup_data(self):
        """Setup data loaders"""
        print("Setting up data loaders...")
        
        # Check if dataset exists
        if not os.path.exists(self.data_dir):
            print(f"Dataset not found at {self.data_dir}")
            print("Creating synthetic dataset for testing...")
            create_synthetic_dataset(self.data_dir, num_samples=1000)
        
        # Create datasets
        try:
            train_dataset = HandLandmarkDataset(
                self.data_dir,
                split='train',
                augment=True
            )
            val_dataset = HandLandmarkDataset(
                self.data_dir,
                split='val',
                augment=False
            )
        except:
            # If split files don't exist, use same data for train/val
            print("Split files not found, using same data for train/val")
            train_dataset = HandLandmarkDataset(
                self.data_dir,
                split='train',
                augment=True
            )
            val_dataset = HandLandmarkDataset(
                self.data_dir,
                split='train',
                augment=False
            )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_landmark_loss = 0.0
        epoch_confidence_loss = 0.0
        epoch_pck = 0.0
        epoch_mean_error = 0.0
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, landmarks, confidence) in enumerate(self.train_loader):
            # Move to device
            images = images.to(self.device)
            landmarks = landmarks.to(self.device)
            confidence = confidence.to(self.device)
            
            # Forward pass
            pred_landmarks, pred_confidence = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                pred_landmarks, pred_confidence,
                landmarks, confidence
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            pck = compute_pck(pred_landmarks, landmarks, threshold=0.2)
            mean_error = compute_mean_error(pred_landmarks, landmarks)
            
            # Accumulate
            epoch_loss += loss_dict['total']
            epoch_landmark_loss += loss_dict['landmark']
            epoch_confidence_loss += loss_dict['confidence']
            epoch_pck += pck
            epoch_mean_error += mean_error
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {loss_dict['total']:.4f} "
                      f"PCK: {pck:.2f}%")
        
        # Average metrics
        epoch_loss /= num_batches
        epoch_landmark_loss /= num_batches
        epoch_confidence_loss /= num_batches
        epoch_pck /= num_batches
        epoch_mean_error /= num_batches
        
        return {
            'loss': epoch_loss,
            'landmark_loss': epoch_landmark_loss,
            'confidence_loss': epoch_confidence_loss,
            'pck': epoch_pck,
            'mean_error': epoch_mean_error
        }
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        
        epoch_loss = 0.0
        epoch_landmark_loss = 0.0
        epoch_confidence_loss = 0.0
        epoch_pck = 0.0
        epoch_mean_error = 0.0
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for images, landmarks, confidence in self.val_loader:
                # Move to device
                images = images.to(self.device)
                landmarks = landmarks.to(self.device)
                confidence = confidence.to(self.device)
                
                # Forward pass
                pred_landmarks, pred_confidence = self.model(images)
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    pred_landmarks, pred_confidence,
                    landmarks, confidence
                )
                
                # Metrics
                pck = compute_pck(pred_landmarks, landmarks, threshold=0.2)
                mean_error = compute_mean_error(pred_landmarks, landmarks)
                
                # Accumulate
                epoch_loss += loss_dict['total']
                epoch_landmark_loss += loss_dict['landmark']
                epoch_confidence_loss += loss_dict['confidence']
                epoch_pck += pck
                epoch_mean_error += mean_error
        
        # Average metrics
        epoch_loss /= num_batches
        epoch_landmark_loss /= num_batches
        epoch_confidence_loss /= num_batches
        epoch_pck /= num_batches
        epoch_mean_error /= num_batches
        
        return {
            'loss': epoch_loss,
            'landmark_loss': epoch_landmark_loss,
            'confidence_loss': epoch_confidence_loss,
            'pck': epoch_pck,
            'mean_error': epoch_mean_error
        }
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(self.model.state_dict(), best_path)
            print(f"  Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs=None):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train (overrides config)
        """
        if num_epochs is not None:
            self.num_epochs = num_epochs
        
        print("="*70)
        print("Starting Training")
        print("="*70)
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Device: {self.device}")
        print("="*70)
        
        # Setup data
        self.setup_data()
        
        # Training loop
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Print metrics
            epoch_time = time.time() - epoch_start_time
            print(f"\n  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Train PCK: {train_metrics['pck']:.2f}% | "
                  f"Val PCK: {val_metrics['pck']:.2f}%")
            print(f"  Train Error: {train_metrics['mean_error']:.4f} | "
                  f"Val Error: {val_metrics['mean_error']:.4f}")
            print(f"  LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.writer.add_scalar('PCK/train', train_metrics['pck'], epoch)
            self.writer.add_scalar('PCK/val', val_metrics['pck'], epoch)
            self.writer.add_scalar('Error/train', train_metrics['mean_error'], epoch)
            self.writer.add_scalar('Error/val', val_metrics['mean_error'], epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_pck'].append(train_metrics['pck'])
            self.history['val_pck'].append(val_metrics['pck'])
            self.history['learning_rate'].append(current_lr)
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(is_best=is_best)
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final history
        history_path = os.path.join(self.output_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train hand landmark model')
    parser.add_argument('--data_dir', type=str, default='data/synthetic_hands',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='hand_landmark_v2/checkpoints',
                       help='Path to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create config
    config = {
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
    }
    
    # Create trainer
    trainer = HandLandmarkTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()
