"""
Lightweight MobileNetV3-based hand landmark detection model
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import *


class MobileNetV3HandLandmark(nn.Module):
    """
    Lightweight hand landmark detection model using MobileNetV3 backbone
    Optimized for CPU inference with <50MB size and 30+ FPS
    """
    
    def __init__(self, num_landmarks=21, pretrained=True):
        super(MobileNetV3HandLandmark, self).__init__()
        
        self.num_landmarks = num_landmarks
        
        # Load MobileNetV3 Small backbone (lightweight)
        if BACKBONE == "mobilenetv3_small":
            mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = 576
        else:
            mobilenet = models.mobilenet_v3_large(pretrained=pretrained)
            feature_dim = 960
        
        # Extract feature extractor (remove classifier)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        
        # Custom landmark prediction head
        self.landmark_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_landmarks * 3)  # x, y, z for each landmark
        )
        
        # Confidence head (per-landmark confidence)
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_landmarks),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Predict landmarks (x, y, z)
        landmarks = self.landmark_head(x)
        landmarks = landmarks.view(-1, self.num_landmarks, 3)
        
        # Predict confidence per landmark
        confidence = self.confidence_head(x)
        
        return landmarks, confidence
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WingLoss(nn.Module):
    """
    Wing Loss for robust landmark detection
    Better than MSE for handling outliers
    """
    
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
        
    def forward(self, pred, target, confidence=None):
        delta = (pred - target).abs()
        
        # Wing loss formula
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1 + delta / self.epsilon),
            delta - self.C
        )
        
        # Weight by confidence if provided
        if confidence is not None:
            loss = loss * confidence.unsqueeze(-1)
        
        return loss.mean()


def create_model(pretrained=True):
    """Create and return the hand landmark model"""
    model = MobileNetV3HandLandmark(num_landmarks=NUM_LANDMARKS, pretrained=pretrained)
    print(f"Model created with {model.count_parameters():,} parameters")
    return model


if __name__ == "__main__":
    # Test model
    model = create_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    landmarks, confidence = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Model size: ~{model.count_parameters() * 4 / 1024 / 1024:.2f} MB (FP32)")
