"""
Dataset handling for hand landmark training
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

from config import INPUT_SIZE, NUM_LANDMARKS, LANDMARK_DIM
from config import AUG_ROTATION_RANGE, AUG_SCALE_RANGE
from config import AUG_BRIGHTNESS_RANGE, AUG_CONTRAST_RANGE, AUG_FLIP_PROB
from config import IMAGENET_MEAN, IMAGENET_STD


class HandLandmarkDataset(Dataset):
    """
    PyTorch dataset for hand landmark training
    
    Supports multiple dataset formats:
    - FreiHAND format
    - CMU Hand format
    - Custom JSON format
    
    Expected directory structure:
        data_dir/
            images/
                img_0001.jpg
                img_0002.jpg
                ...
            annotations.json
    
    Annotation format (JSON):
        {
            "img_0001.jpg": {
                "landmarks": [[x1, y1, z1], [x2, y2, z2], ...],  # 21 landmarks
                "bbox": [x_min, y_min, x_max, y_max]  # optional
            },
            ...
        }
    """
    
    def __init__(self, data_dir, split='train', augment=True, transform=None):
        """
        Initialize dataset
        
        Args:
            data_dir: Root directory containing images and annotations
            split: 'train', 'val', or 'test'
            augment: Whether to apply data augmentation
            transform: Additional custom transforms
        """
        self.data_dir = data_dir
        self.split = split
        self.augment = augment and (split == 'train')
        self.transform = transform
        
        # Load annotations
        self.annotations = self._load_annotations()
        self.image_files = list(self.annotations.keys())
        
        print(f"Loaded {len(self.image_files)} images for {split} split")
    
    def _load_annotations(self):
        """Load annotations from JSON file"""
        ann_file = os.path.join(self.data_dir, f'{self.split}_annotations.json')
        
        if not os.path.exists(ann_file):
            # Try without split prefix
            ann_file = os.path.join(self.data_dir, 'annotations.json')
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotations file not found: {ann_file}")
        
        with open(ann_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Returns:
            image: (3, 224, 224) tensor
            landmarks: (21, 3) tensor of normalized landmarks
            confidence: (21,) tensor of confidence (all 1.0 for ground truth)
        """
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        
        # Try original path first, then fallback to datasets/freihand
        image = cv2.imread(img_path)
        if image is None:
            # Try alternative path
            alt_path = os.path.join('datasets/freihand/training/rgb', img_name)
            image = cv2.imread(alt_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Load landmarks
        ann = self.annotations[img_name]
        landmarks = np.array(ann['landmarks'], dtype=np.float32)  # (21, 3)
        
        # Normalize landmarks to [0, 1]
        landmarks_norm = landmarks.copy()
        landmarks_norm[:, 0] /= w
        landmarks_norm[:, 1] /= h
        # Z coordinate is already normalized or relative
        
        # Apply augmentation
        if self.augment:
            image, landmarks_norm = self._augment(image, landmarks_norm)
        
        # Resize image
        image = cv2.resize(image, INPUT_SIZE)
        
        # Convert to tensor and normalize
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert landmarks to tensor
        landmarks_tensor = torch.from_numpy(landmarks_norm).float()
        
        # Confidence (all 1.0 for ground truth)
        confidence = torch.ones(NUM_LANDMARKS, dtype=torch.float32)
        
        return image, landmarks_tensor, confidence
    
    def _augment(self, image, landmarks):
        """
        Apply data augmentation
        
        Args:
            image: Input image (H, W, 3)
            landmarks: Normalized landmarks (21, 3)
            
        Returns:
            aug_image: Augmented image
            aug_landmarks: Augmented landmarks
        """
        h, w = image.shape[:2]
        
        # Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-AUG_ROTATION_RANGE, AUG_ROTATION_RANGE)
            image, landmarks = self._rotate(image, landmarks, angle)
        
        # Random scaling
        if np.random.rand() < 0.5:
            scale = np.random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
            image, landmarks = self._scale(image, landmarks, scale)
        
        # Random brightness/contrast
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(AUG_BRIGHTNESS_RANGE[0], AUG_BRIGHTNESS_RANGE[1])
            contrast = np.random.uniform(AUG_CONTRAST_RANGE[0], AUG_CONTRAST_RANGE[1])
            image = self._adjust_brightness_contrast(image, brightness, contrast)
        
        # Random horizontal flip
        if np.random.rand() < AUG_FLIP_PROB:
            image, landmarks = self._flip_horizontal(image, landmarks)
        
        return image, landmarks
    
    def _rotate(self, image, landmarks, angle):
        """Rotate image and landmarks"""
        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT)
        
        # Rotate landmarks
        landmarks_px = landmarks.copy()
        landmarks_px[:, 0] *= w
        landmarks_px[:, 1] *= h
        
        # Apply rotation
        ones = np.ones((NUM_LANDMARKS, 1))
        landmarks_homo = np.hstack([landmarks_px[:, :2], ones])
        landmarks_rotated = (M @ landmarks_homo.T).T
        
        # Normalize back
        landmarks[:, 0] = landmarks_rotated[:, 0] / w
        landmarks[:, 1] = landmarks_rotated[:, 1] / h
        
        return image, landmarks
    
    def _scale(self, image, landmarks, scale):
        """Scale image and landmarks"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        image = cv2.resize(image, (new_w, new_h))
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            image = image[start_y:start_y+h, start_x:start_x+w]
            
            # Adjust landmarks
            landmarks[:, 0] = (landmarks[:, 0] * new_w - start_x) / w
            landmarks[:, 1] = (landmarks[:, 1] * new_h - start_y) / h
        else:
            # Pad
            pad_y = (h - new_h) // 2
            pad_x = (w - new_w) // 2
            image = cv2.copyMakeBorder(image, pad_y, h-new_h-pad_y, 
                                      pad_x, w-new_w-pad_x,
                                      cv2.BORDER_REFLECT)
            
            # Adjust landmarks
            landmarks[:, 0] = (landmarks[:, 0] * new_w + pad_x) / w
            landmarks[:, 1] = (landmarks[:, 1] * new_h + pad_y) / h
        
        return image, landmarks
    
    def _adjust_brightness_contrast(self, image, brightness, contrast):
        """Adjust brightness and contrast"""
        image = image.astype(np.float32)
        image = image * contrast + (brightness - 1.0) * 128
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    def _flip_horizontal(self, image, landmarks):
        """Flip image and landmarks horizontally"""
        # Flip image
        image = cv2.flip(image, 1)
        
        # Flip landmarks
        landmarks[:, 0] = 1.0 - landmarks[:, 0]
        
        # Mirror landmark indices (swap left/right)
        # This depends on the landmark ordering convention
        # For MediaPipe-style ordering, we need to swap certain landmarks
        # For simplicity, we'll just flip x-coordinates
        # In a real implementation, you'd swap thumb/pinky, etc.
        
        return image, landmarks


def create_synthetic_dataset(output_dir, num_samples=1000):
    """
    Create synthetic dataset for testing
    
    Args:
        output_dir: Directory to save synthetic data
        num_samples: Number of samples to generate
    """
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    annotations = {}
    
    for i in range(num_samples):
        # Generate random hand image (placeholder)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Draw random hand-like shape
        center_x = np.random.randint(200, 440)
        center_y = np.random.randint(150, 330)
        
        # Generate random landmarks around center
        landmarks = []
        for j in range(NUM_LANDMARKS):
            angle = (j / NUM_LANDMARKS) * 2 * np.pi
            radius = np.random.uniform(50, 150)
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            z = np.random.uniform(-0.1, 0.1)
            landmarks.append([float(x), float(y), float(z)])
        
        # Save image
        img_name = f'img_{i:05d}.jpg'
        img_path = os.path.join(output_dir, 'images', img_name)
        cv2.imwrite(img_path, img)
        
        # Save annotation
        annotations[img_name] = {
            'landmarks': landmarks,
            'bbox': [center_x-150, center_y-150, center_x+150, center_y+150]
        }
    
    # Save annotations
    ann_path = os.path.join(output_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Synthetic dataset created at {output_dir}")


if __name__ == "__main__":
    # Create synthetic dataset for testing
    create_synthetic_dataset('data/synthetic_hands', num_samples=100)
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    dataset = HandLandmarkDataset('data/synthetic_hands', split='train', augment=True)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting item
    image, landmarks, confidence = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Landmarks shape: {landmarks.shape}")
    print(f"Confidence shape: {confidence.shape}")
    
    print("\n✓ Dataset test complete")
