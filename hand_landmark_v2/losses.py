"""
Loss functions for hand landmark training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    """
    Wing Loss for landmark localization
    
    Wing Loss is designed for landmark detection and provides better
    performance than L1/L2 loss, especially for small errors.
    
    Paper: "Wing Loss for Robust Facial Landmark Localisation with
    Convolutional Neural Networks" (https://arxiv.org/abs/1711.06753)
    """
    
    def __init__(self, omega=10.0, epsilon=2.0):
        """
        Initialize Wing Loss
        
        Args:
            omega: Width parameter
            epsilon: Curvature parameter
        """
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * torch.log(torch.tensor(1.0 + self.omega / self.epsilon))
    
    def forward(self, pred, target):
        """
        Compute Wing Loss
        
        Args:
            pred: Predicted landmarks (B, N, 3)
            target: Target landmarks (B, N, 3)
            
        Returns:
            loss: Wing loss value
        """
        delta = (pred - target).abs()
        
        # Wing loss formula
        loss = torch.where(
            delta < self.omega,
            self.omega * torch.log(1.0 + delta / self.epsilon),
            delta - self.C
        )
        
        return loss.mean()


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss
    
    An improved version of Wing Loss that adapts to different landmark types.
    """
    
    def __init__(self, omega=14.0, theta=0.5, epsilon=1.0, alpha=2.1):
        """
        Initialize Adaptive Wing Loss
        
        Args:
            omega: Width parameter
            theta: Threshold parameter
            epsilon: Curvature parameter
            alpha: Power parameter
        """
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Compute Adaptive Wing Loss
        
        Args:
            pred: Predicted landmarks (B, N, 3)
            target: Target landmarks (B, N, 3)
            
        Returns:
            loss: Adaptive wing loss value
        """
        delta = (pred - target).abs()
        
        A = self.omega * (1.0 / (1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) * \
            (1.0 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1.0 + torch.pow(self.theta / self.epsilon, self.alpha - target))
        
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1.0 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return loss.mean()


class HandLandmarkLoss(nn.Module):
    """
    Combined loss for hand landmark detection
    
    Combines landmark regression loss and confidence prediction loss
    """
    
    def __init__(self, landmark_weight=1.0, confidence_weight=0.5, use_wing_loss=True):
        """
        Initialize combined loss
        
        Args:
            landmark_weight: Weight for landmark loss
            confidence_weight: Weight for confidence loss
            use_wing_loss: Whether to use Wing Loss (otherwise use MSE)
        """
        super(HandLandmarkLoss, self).__init__()
        self.landmark_weight = landmark_weight
        self.confidence_weight = confidence_weight
        
        if use_wing_loss:
            self.landmark_loss_fn = WingLoss()
        else:
            self.landmark_loss_fn = nn.MSELoss()
        
        self.confidence_loss_fn = nn.BCELoss()
    
    def forward(self, pred_landmarks, pred_confidence, target_landmarks, target_confidence):
        """
        Compute combined loss
        
        Args:
            pred_landmarks: Predicted landmarks (B, 21, 3)
            pred_confidence: Predicted confidence (B, 21)
            target_landmarks: Target landmarks (B, 21, 3)
            target_confidence: Target confidence (B, 21)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Landmark loss
        landmark_loss = self.landmark_loss_fn(pred_landmarks, target_landmarks)
        
        # Confidence loss
        confidence_loss = self.confidence_loss_fn(pred_confidence, target_confidence)
        
        # Combined loss
        total_loss = (self.landmark_weight * landmark_loss + 
                     self.confidence_weight * confidence_loss)
        
        loss_dict = {
            'total': total_loss.item(),
            'landmark': landmark_loss.item(),
            'confidence': confidence_loss.item(),
        }
        
        return total_loss, loss_dict


def compute_pck(pred_landmarks, target_landmarks, threshold=0.2):
    """
    Compute Percentage of Correct Keypoints (PCK)
    
    A landmark is considered correct if the Euclidean distance between
    prediction and ground truth is less than threshold.
    
    Args:
        pred_landmarks: Predicted landmarks (B, 21, 3)
        target_landmarks: Target landmarks (B, 21, 3)
        threshold: Distance threshold (normalized)
        
    Returns:
        pck: Percentage of correct keypoints
    """
    # Compute Euclidean distance
    distances = torch.sqrt(((pred_landmarks - target_landmarks) ** 2).sum(dim=-1))
    
    # Count correct predictions
    correct = (distances < threshold).float()
    pck = correct.mean().item() * 100
    
    return pck


def compute_mean_error(pred_landmarks, target_landmarks):
    """
    Compute mean Euclidean distance error
    
    Args:
        pred_landmarks: Predicted landmarks (B, 21, 3)
        target_landmarks: Target landmarks (B, 21, 3)
        
    Returns:
        mean_error: Mean Euclidean distance
    """
    distances = torch.sqrt(((pred_landmarks - target_landmarks) ** 2).sum(dim=-1))
    mean_error = distances.mean().item()
    return mean_error


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 4
    pred_landmarks = torch.randn(batch_size, 21, 3)
    target_landmarks = torch.randn(batch_size, 21, 3)
    pred_confidence = torch.sigmoid(torch.randn(batch_size, 21))
    target_confidence = torch.ones(batch_size, 21)
    
    # Test Wing Loss
    wing_loss = WingLoss()
    loss = wing_loss(pred_landmarks, target_landmarks)
    print(f"Wing Loss: {loss.item():.4f}")
    
    # Test combined loss
    combined_loss = HandLandmarkLoss()
    total_loss, loss_dict = combined_loss(pred_landmarks, pred_confidence,
                                          target_landmarks, target_confidence)
    print(f"Combined Loss: {loss_dict}")
    
    # Test metrics
    pck = compute_pck(pred_landmarks, target_landmarks, threshold=0.2)
    mean_error = compute_mean_error(pred_landmarks, target_landmarks)
    print(f"PCK@0.2: {pck:.2f}%")
    print(f"Mean Error: {mean_error:.4f}")
    
    print("\n✓ Loss function test complete")
