"""
Fix all dependencies for the sign language project
"""
import subprocess
import sys

print("="*70)
print("Fixing Dependencies for Sign Language Project")
print("="*70)

packages_to_uninstall = [
    "opencv-python",
    "opencv-contrib-python",
    "opencv-python-headless",
]

packages_to_install = [
    "opencv-python==4.8.1.78",  # Stable version compatible with numpy 1.26
    "numpy==1.26.4",  # Compatible with TensorFlow 2.10
    "tensorflow==2.10.0",  # Latest that works with Python 3.10
]

print("\n[1/2] Uninstalling conflicting packages...")
for package in packages_to_uninstall:
    print(f"  Uninstalling {package}...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package],
                   capture_output=True)

print("\n[2/2] Installing correct versions...")
for package in packages_to_install:
    print(f"  Installing {package}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ✗ Failed: {result.stderr}")
    else:
        print(f"    ✓ Installed")

print("\n" + "="*70)
print("Verifying installation...")
print("="*70)

result = subprocess.run([sys.executable, "-m", "pip", "list"],
                       capture_output=True, text=True)

for line in result.stdout.split('\n'):
    if any(pkg in line.lower() for pkg in ['numpy', 'tensorflow', 'opencv']):
        print(f"  {line}")

print("\n" + "="*70)
print("Dependencies fixed!")
print("="*70)
print("\nYou can now run:")
print("  python classify_HACKATHON_FIXED.py")
print("="*70)
