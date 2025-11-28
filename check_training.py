"""
Check training progress
"""

import os
import json
import time

print("="*70)
print("Training Progress Monitor")
print("="*70)

# Check if training is running
checkpoint_dir = 'hand_landmark_v2/checkpoints'
log_dir = 'hand_landmark_v2/logs'
history_file = os.path.join(checkpoint_dir, 'history.json')
best_model = os.path.join(checkpoint_dir, 'best_model.pth')
latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

# Check for checkpoints
if os.path.exists(latest_checkpoint):
    print("\n✓ Training checkpoint found")
    
    # Load checkpoint
    import torch
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"  Current epoch: {epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Load history
    if 'history' in checkpoint:
        history = checkpoint['history']
        
        if history['train_loss']:
            print(f"\n  Recent training metrics:")
            print(f"    Train loss: {history['train_loss'][-1]:.4f}")
            print(f"    Val loss: {history['val_loss'][-1]:.4f}")
            print(f"    Train PCK: {history['train_pck'][-1]:.2f}%")
            print(f"    Val PCK: {history['val_pck'][-1]:.2f}%")
            print(f"    Learning rate: {history['learning_rate'][-1]:.6f}")
else:
    print("\n⚠ No checkpoint found yet")
    print("  Training may still be initializing...")

# Check for best model
if os.path.exists(best_model):
    print("\n✓ Best model saved")
    size_mb = os.path.getsize(best_model) / (1024 ** 2)
    print(f"  Model size: {size_mb:.2f} MB")
else:
    print("\n⚠ Best model not saved yet")

# Check logs
if os.path.exists(log_dir):
    log_files = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    if log_files:
        print(f"\n✓ TensorBoard logs found: {len(log_files)} run(s)")
        print(f"  View with: tensorboard --logdir {log_dir}")
    else:
        print(f"\n⚠ No TensorBoard logs yet")
else:
    print(f"\n⚠ Log directory not created yet")

# Instructions
print("\n" + "="*70)
print("Monitoring Options:")
print("="*70)
print("\n1. Check this script periodically:")
print("   python check_training.py")
print("\n2. View TensorBoard (real-time):")
print("   tensorboard --logdir hand_landmark_v2/logs")
print("   Then open: http://localhost:6006")
print("\n3. Check log file:")
print("   type hand_landmark_v2\\logs\\training.log")
print("\n" + "="*70)
print("\nTraining typically takes:")
print("  - CPU: 2-3 days for 200 epochs")
print("  - GPU: 12-24 hours for 200 epochs")
print("\nTarget: PCK@0.2 > 95%")
print("="*70)
