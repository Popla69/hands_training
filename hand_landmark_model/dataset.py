"""
Dataset loader and downloader for hand landmark training
Downloads FreiHAND, RHD, and other public datasets
"""

import os
import urllib.request
import zipfile
import json
import numpy as np
import cv2
from torch.utils.data import Dataset
import albumentations as A
from config import *


class HandLandmarkDataset(Dataset):
    """Hand landmark dataset with augmentation"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load annotations
        self.samples = self._load_annotations()
        
    def _load_annotations(self):
        """Load dataset annotations"""
        annotation_file = os.path.join(self.root_dir, f"{self.split}_annotations.json")
        
        if not os.path.exists(annotation_file):
            print(f"Warning: {annotation_file} not found. Creating empty dataset.")
            return []
        
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        return annotations
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.root_dir, sample['image'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load landmarks (21, 3)
        landmarks = np.array(sample['landmarks'], dtype=np.float32)
        
        # Apply augmentation
        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks[:, :2])
            image = transformed['image']
            landmarks[:, :2] = np.array(transformed['keypoints'])
        
        # Resize to model input size
        image = cv2.resize(image, INPUT_SIZE)
        
        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Transpose to CHW format
        image = image.transpose(2, 0, 1)
        
        # Normalize landmarks to [0, 1]
        landmarks[:, 0] /= INPUT_SIZE[0]
        landmarks[:, 1] /= INPUT_SIZE[1]
        
        return {
            'image': image,
            'landmarks': landmarks,
            'confidence': np.ones(NUM_LANDMARKS, dtype=np.float32)
        }


def get_augmentation_pipeline(split='train'):
    """Get augmentation pipeline"""
    
    if split == 'train':
        return A.Compose([
            A.Rotate(limit=AUG_ROTATION_RANGE, p=0.5),
            A.RandomScale(scale_limit=(AUG_SCALE_RANGE[0]-1, AUG_SCALE_RANGE[1]-1), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=(AUG_BRIGHTNESS_RANGE[0]-1, AUG_BRIGHTNESS_RANGE[1]-1),
                contrast_limit=(AUG_CONTRAST_RANGE[0]-1, AUG_CONTRAST_RANGE[1]-1),
                p=0.5
            ),
            A.GaussNoise(var_limit=(0, AUG_NOISE_STD**2), p=0.3),
            A.HorizontalFlip(p=0.5),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    else:
        return None


def download_freihand_dataset(output_dir="datasets/freihand"):
    """Download FreiHAND dataset"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    urls = {
        'training_images': 'https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2.zip',
        'training_annotations': 'https://lmb.informatik.uni-freiburg.de/data/freihand/FreiHAND_pub_v2_training_xyz.zip'
    }
    
    print("Downloading FreiHAND dataset...")
    print("Note: This is a large dataset (~11GB). Please be patient.")
    
    for name, url in urls.items():
        zip_path = os.path.join(output_dir, f"{name}.zip")
        
        if not os.path.exists(zip_path):
            print(f"Downloading {name}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
                print(f"Downloaded {name}")
            except Exception as e:
                print(f"Error downloading {name}: {e}")
                print("Please download manually from: https://lmb.informatik.uni-freiburg.de/projects/freihand/")
                continue
        
        # Extract
        print(f"Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    print("FreiHAND dataset ready!")
    return output_dir


def create_synthetic_dataset(output_dir="datasets/synthetic", num_samples=1000):
    """Create synthetic hand landmark dataset for testing"""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    annotations = []
    
    print(f"Creating {num_samples} synthetic samples...")
    
    for i in range(num_samples):
        # Create synthetic image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Draw synthetic hand
        center_x, center_y = 320, 240
        
        # Generate random landmarks
        landmarks = []
        for j in range(21):
            angle = (j / 21) * 2 * np.pi
            radius = 50 + np.random.rand() * 50
            x = center_x + radius * np.cos(angle) + np.random.randn() * 10
            y = center_y + radius * np.sin(angle) + np.random.randn() * 10
            z = np.random.rand() * 0.1
            landmarks.append([float(x), float(y), float(z)])
        
        # Save image
        img_name = f"synthetic_{i:06d}.jpg"
        img_path = os.path.join(output_dir, 'images', img_name)
        cv2.imwrite(img_path, img)
        
        # Add annotation
        annotations.append({
            'image': f'images/{img_name}',
            'landmarks': landmarks
        })
    
    # Split into train/val/test
    np.random.shuffle(annotations)
    n_train = int(len(annotations) * TRAIN_SPLIT)
    n_val = int(len(annotations) * VAL_SPLIT)
    
    train_ann = annotations[:n_train]
    val_ann = annotations[n_train:n_train+n_val]
    test_ann = annotations[n_train+n_val:]
    
    # Save annotations
    with open(os.path.join(output_dir, 'train_annotations.json'), 'w') as f:
        json.dump(train_ann, f)
    
    with open(os.path.join(output_dir, 'val_annotations.json'), 'w') as f:
        json.dump(val_ann, f)
    
    with open(os.path.join(output_dir, 'test_annotations.json'), 'w') as f:
        json.dump(test_ann, f)
    
    print(f"Created synthetic dataset: {len(train_ann)} train, {len(val_ann)} val, {len(test_ann)} test")
    return output_dir


if __name__ == "__main__":
    # Create synthetic dataset for testing
    dataset_dir = create_synthetic_dataset(num_samples=100)
    
    # Test dataset loading
    dataset = HandLandmarkDataset(dataset_dir, split='train', 
                                  transform=get_augmentation_pipeline('train'))
    
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Landmarks shape: {sample['landmarks'].shape}")
        print(f"Confidence shape: {sample['confidence'].shape}")
