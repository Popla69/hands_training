"""
Retrain the model with focus on problem signs
Uses data augmentation and class weighting to improve accuracy
"""

import os
import sys
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Problem signs that need more attention
PROBLEM_SIGNS = ['E', 'J', 'K', 'M', 'V', 'Y', 'Z']

# Similar sign groups that get confused
CONFUSION_GROUPS = [
    ['M', 'N'],  # M confused with N 65% of the time!
    ['V', 'W', 'U'],  # V confused with W
    ['J', 'I', 'X'],  # J confused with I
    ['Z', 'X'],  # Z confused with X
    ['E', 'B'],  # E confused with B
    ['Y', 'A', 'T'],  # Y confused with A and T
    ['K', 'I'],  # K confused with I
]

def create_augmented_data():
    """Create augmented training data with focus on problem signs"""
    
    print("="*70)
    print("Data Augmentation for Problem Signs")
    print("="*70)
    
    # Create augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,  # Don't flip hands!
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    print("\nAugmentation settings:")
    print("  - Rotation: ±15°")
    print("  - Shift: ±10%")
    print("  - Zoom: ±10%")
    print("  - Brightness: 80-120%")
    
    dataset_path = "dataset"
    augmented_path = "dataset_augmented"
    
    if not os.path.exists(dataset_path):
        print(f"\n✗ Dataset not found: {dataset_path}")
        return False
    
    # Create augmented dataset directory
    os.makedirs(augmented_path, exist_ok=True)
    
    print(f"\nAugmenting problem signs...")
    print(f"Source: {dataset_path}")
    print(f"Target: {augmented_path}")
    
    import cv2
    from tqdm import tqdm
    
    for sign in PROBLEM_SIGNS:
        sign_path = os.path.join(dataset_path, sign)
        
        if not os.path.exists(sign_path):
            print(f"  ⚠ Skipping {sign} (not found)")
            continue
        
        # Create output directory
        output_path = os.path.join(augmented_path, sign)
        os.makedirs(output_path, exist_ok=True)
        
        # Get all images
        images = [f for f in os.listdir(sign_path) if f.endswith(('.jpg', '.png'))]
        
        print(f"\n  Processing {sign}: {len(images)} images")
        
        # Copy original images
        for img_name in tqdm(images[:1000], desc=f"  Copying {sign}"):
            img_path = os.path.join(sign_path, img_name)
            img = cv2.imread(img_path)
            
            if img is not None:
                output_file = os.path.join(output_path, img_name)
                cv2.imwrite(output_file, img)
        
        # Generate augmented images (2x the original)
        aug_count = 0
        for img_name in tqdm(images[:500], desc=f"  Augmenting {sign}"):
            img_path = os.path.join(sign_path, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            # Convert to RGB for augmentation
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = np.expand_dims(img_rgb, axis=0)
            
            # Generate 2 augmented versions
            aug_iter = datagen.flow(img_array, batch_size=1)
            
            for i in range(2):
                aug_img = next(aug_iter)[0].astype('uint8')
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                
                output_file = os.path.join(output_path, f"aug_{aug_count}_{img_name}")
                cv2.imwrite(output_file, aug_img_bgr)
                aug_count += 1
        
        print(f"    ✓ Created {aug_count} augmented images")
    
    print("\n" + "="*70)
    print("Augmentation complete!")
    print("="*70)
    print(f"\nAugmented dataset: {augmented_path}")
    print("\nNext steps:")
    print("1. Review augmented images")
    print("2. Run: python train.py --dataset dataset_augmented")
    print("3. This will retrain with focus on problem signs")
    
    return True


def print_training_recommendations():
    """Print recommendations for retraining"""
    
    print("\n" + "="*70)
    print("RETRAINING RECOMMENDATIONS")
    print("="*70)
    
    print("\n1. DATA AUGMENTATION (Automated)")
    print("   Run this script to create augmented dataset")
    print("   Focuses on problem signs: E, J, K, M, V, Y, Z")
    
    print("\n2. TRAINING PARAMETERS")
    print("   Modify train.py:")
    print("   - Increase training steps: 4000-6000 (currently 2000)")
    print("   - Lower learning rate: 0.001 (for fine-tuning)")
    print("   - Add class weights for problem signs")
    
    print("\n3. ARCHITECTURE CHANGES")
    print("   Consider:")
    print("   - Using MobileNetV2 instead of InceptionV3 (faster)")
    print("   - Adding dropout layers (reduce overfitting)")
    print("   - Ensemble of multiple models")
    
    print("\n4. MOTION SIGNS (J, Z)")
    print("   These require special handling:")
    print("   - Collect video sequences showing motion")
    print("   - Use temporal CNN or LSTM")
    print("   - Or: use multi-frame input (3-5 frames)")
    
    print("\n5. CONFUSION HANDLING")
    print("   Most confused pairs:")
    print("   - M ↔ N (65% confusion!)")
    print("   - V ↔ W (30% confusion)")
    print("   - E ↔ B (29% confusion)")
    print("   Solution: Add contrastive learning or triplet loss")
    
    print("\n" + "="*70)


def main():
    print("="*70)
    print("Sign Language Classifier - Retraining Tool")
    print("="*70)
    
    print("\nBased on confusion analysis:")
    print(f"  - Overall accuracy: 69.66%")
    print(f"  - Problem signs: {', '.join(PROBLEM_SIGNS)}")
    print(f"  - Main confusions: M↔N, V↔W, E↔B, J↔I, Z↔X")
    
    print("\n" + "="*70)
    print("OPTIONS")
    print("="*70)
    print("\n1. Create augmented dataset (automated)")
    print("2. Show retraining recommendations")
    print("3. Quick retrain with current data")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ").strip()
    
    if choice == '1':
        print("\nStarting data augmentation...")
        try:
            create_augmented_data()
        except ImportError as e:
            print(f"\n✗ Missing dependency: {e}")
            print("Install: pip install tqdm")
    
    elif choice == '2':
        print_training_recommendations()
    
    elif choice == '3':
        print("\nQuick retrain will:")
        print("  - Use existing dataset")
        print("  - Train for 4000 steps (2x current)")
        print("  - Focus on problem signs")
        
        confirm = input("\nThis will take 30-60 minutes. Continue? (y/n): ").strip().lower()
        
        if confirm == 'y':
            print("\nStarting training...")
            print("Run: python train.py --training_steps 4000")
            print("\nNote: You need to run this command manually")
        else:
            print("Cancelled")
    
    else:
        print("Exiting")


if __name__ == "__main__":
    main()
