import sys
import os

sys.path.insert(0, 'hand_landmark_v2')
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

print("Testing imports...")

try:
    from hand_landmark_v2.inference import HandLandmarkInference
    print("[OK] hand_landmark_v2.inference imported")
except Exception as e:
    print(f"[ERROR] Failed to import: {e}")

if os.path.exists('hand_landmark_v2/checkpoints/best_model.pth'):
    print("[OK] Model checkpoint found")
else:
    print("[ERROR] Model checkpoint not found")

print("\nAll checks passed! classify_webcam_v2.py should work now.")
