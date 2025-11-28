import sys
sys.path.insert(0, 'hand_landmark_model')
from model import create_model
import torch
import os

os.makedirs('hand_landmark_model/models', exist_ok=True)
model = create_model(False)
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
torch.save(model.state_dict(), 'hand_landmark_model/models/hand_landmark.pth')

size_mb = os.path.getsize('hand_landmark_model/models/hand_landmark.pth') / (1024 * 1024)
print(f"✓ Model exported: hand_landmark_model/models/hand_landmark.pth ({size_mb:.2f} MB)")
print(f"✓ Validation accuracy: {ckpt['val_accuracy']*100:.2f}%")
print(f"✓ Model parameters: {model.count_parameters():,}")
print("\nModel ready for inference!")
