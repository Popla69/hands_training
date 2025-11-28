"""
Fix zlibwapi.dll - Alternative method
"""
import os
import sys
import subprocess

print("="*70)
print("Fixing zlibwapi.dll - Method 2")
print("="*70)

print("\n[1/2] Reinstalling OpenCV without contrib...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"])
subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python==4.8.1.78"])

print("\n[2/2] Installing Pillow as backup...")
subprocess.run([sys.executable, "-m", "pip", "install", "Pillow", "--upgrade"])

print("\n" + "="*70)
print("Alternative Fix Applied")
print("="*70)
print("\nThe issue is that OpenCV is trying to use PNG compression")
print("which requires zlibwapi.dll")
print("\nSolution: Use JPEG instead of PNG in training")
print("\nTry running train_tf2.py again")
