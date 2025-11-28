"""
Hand landmark detection model based on MobileNetV3
"""

import torch
import torch.nn as nn
import torchvision.models as models
try:
    from .config import NUM_LANDMARKS, LANDMARK_DIM
except ImportError:
    from config import NUM_LANDMARKS, LANDMARK_DIM


class HandLandmarkModel(nn.Module):
    """
    MobileNetV3-based hand landmark detection model
    
    Architecture:
    - Backbone: MobileNetV3-Small (pretrained on ImageNet)
    - Landmark Head: FC layers → 63 outputs (21 landmarks × 3 coords)
    - Confidence Head: FC layers → 21 outputs (per-landmark confidence)
    
    Input: (B, 3, 224, 224) RGB image
    Output: 
        - landmarks: (B, 21, 3) normalized coordinates
        - confidence: (B, 21) per-landmark confidence scores
    """
    
    def __init__(self, pretrained=True):
        super(HandLandmarkModel, self).__init__()
        
        # Load MobileNetV3-Small backbone
        mobilenet = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Extract feature extractor (remove classifier)
        self.backbone = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Get feature dimension
        # MobileNetV3-Small outputs 576 features after avgpool
        feature_dim = 576
        
        # Landmark regression head
        self.landmark_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_LANDMARKS * LANDMARK_DIM)
        )
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, NUM_LANDMARKS),
            nn.Sigmoid()  # Output confidence in [0, 1]
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (B, 3, 224, 224) input images
            
        Returns:
            landmarks: (B, 21, 3) normalized landmark coordinates
            confidence: (B, 21) per-landmark confidence scores
        """
        # Extract features
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Predict landmarks
        landmarks_flat = self.landmark_head(features)
        landmarks = landmarks_flat.view(-1, NUM_LANDMARKS, LANDMARK_DIM)
        
        # Predict confidence
        confidence = self.confidence_head(features)
        
        return landmarks, confidence


def create_model(pretrained=True):
    """
    Create hand landmark model
    
    Args:
        pretrained: Whether to use ImageNet pretrained weights
        
    Returns:
        model: HandLandmarkModel instance
    """
    model = HandLandmarkModel(pretrained=pretrained)
    return model


def count_parameters(model):
    """
    Count trainable parameters in model
    
    Args:
        model: PyTorch model
        
    Returns:
        total_params: Total number of trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def model_summary(model, input_size=(1, 3, 224, 224)):
    """
    Print model summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    print("="*70)
    print("Hand Landmark Model Summary")
    print("="*70)
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Estimate model size
    param_size = total_params * 4 / (1024 ** 2)  # Assuming float32
    print(f"Estimated model size: {param_size:.2f} MB")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    model.eval()
    with torch.no_grad():
        landmarks, confidence = model(dummy_input)
    
    print(f"\nInput shape: {tuple(dummy_input.shape)}")
    print(f"Landmarks output shape: {tuple(landmarks.shape)}")
    print(f"Confidence output shape: {tuple(confidence.shape)}")
    print("="*70)


if __name__ == "__main__":
    # Test model creation
    print("Creating hand landmark model...")
    model = create_model(pretrained=False)
    
    # Print summary
    model_summary(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 224, 224)
    landmarks, confidence = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Landmarks: {landmarks.shape}")
    print(f"  Confidence: {confidence.shape}")
