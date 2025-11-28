"""
Quick start training script
Creates synthetic data and starts training to test the pipeline
"""

import os
import sys

print("="*70)
print("Hand Landmark Model - Quick Start Training")
print("="*70)

# Check if real dataset exists
real_data_exists = os.path.exists('data/freihand_converted') or os.path.exists('data/cmu_hand')

if real_data_exists:
    print("\n✓ Real dataset found!")
    print("\nStarting training with real data...")
    print("This will take several hours to achieve 95%+ accuracy.")
    print("\nMonitor progress with: tensorboard --logdir hand_landmark_v2/logs")
    
    response = input("\nStart training? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Train with real data
    os.system('python hand_landmark_v2/train.py --epochs 200 --batch_size 64')
    
else:
    print("\n⚠ No real dataset found.")
    print("\nOptions:")
    print("1. Create synthetic data for testing (quick, low accuracy)")
    print("2. Download real dataset for production (slow, high accuracy)")
    print("3. Exit and follow TRAINING_GUIDE.md")
    
    choice = input("\nChoose option (1/2/3): ").strip()
    
    if choice == '1':
        print("\nCreating synthetic dataset...")
        print("Note: This is for TESTING ONLY. Accuracy will be low.")
        
        # Create synthetic data
        sys.path.insert(0, 'hand_landmark_v2')
        from dataset import create_synthetic_dataset
        
        create_synthetic_dataset('data/synthetic_hands', num_samples=1000)
        
        print("\n✓ Synthetic dataset created")
        print("\nStarting training (this will take 10-30 minutes)...")
        print("Monitor with: tensorboard --logdir hand_landmark_v2/logs")
        
        # Train with synthetic data
        os.system('python hand_landmark_v2/train.py --data_dir data/synthetic_hands --epochs 50 --batch_size 32')
        
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        print("\nNote: This model was trained on synthetic data.")
        print("For 95%+ accuracy, you need to:")
        print("1. Download FreiHAND dataset")
        print("2. Follow TRAINING_GUIDE.md")
        print("3. Train for 200+ epochs")
        print("\nTest the model:")
        print("  python classify_webcam_v2.py")
        
    elif choice == '2':
        print("\n" + "="*70)
        print("Download Real Dataset")
        print("="*70)
        print("\nFollow these steps:")
        print("\n1. FreiHAND Dataset (Recommended)")
        print("   - Visit: https://lmb.informatik.uni-freiburg.de/projects/freihand/")
        print("   - Register and download")
        print("   - Extract to data/freihand/")
        print("   - Convert: python hand_landmark_v2/download_datasets.py convert_freihand")
        print("\n2. CMU Hand Dataset")
        print("   - Visit: http://domedb.perception.cs.cmu.edu/handdb.html")
        print("   - Download and extract to data/cmu_hand/")
        print("\n3. After downloading, run this script again")
        print("\nSee TRAINING_GUIDE.md for detailed instructions.")
        
    else:
        print("\nExiting. See TRAINING_GUIDE.md for instructions.")
        sys.exit(0)
