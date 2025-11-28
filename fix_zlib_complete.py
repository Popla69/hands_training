"""
Complete fix for zlibwapi.dll issue
Downloads the DLL and places it in the correct location
"""
import os
import sys
import urllib.request
import shutil
from pathlib import Path

print("="*70)
print("Fixing zlibwapi.dll Issue")
print("="*70)

# Find Python DLLs directory
python_dir = Path(sys.executable).parent
dlls_dir = python_dir / "DLLs"
lib_dir = python_dir / "Library" / "bin"

print(f"\nPython directory: {python_dir}")
print(f"DLLs directory: {dlls_dir}")

# Create directories if they don't exist
dlls_dir.mkdir(exist_ok=True)
if lib_dir.parent.exists():
    lib_dir.mkdir(parents=True, exist_ok=True)

# Download zlibwapi.dll
dll_url = "https://github.com/horta/zlib.wapi/raw/master/dll/zlibwapi.dll"
dll_path = dlls_dir / "zlibwapi.dll"

print(f"\n[1/3] Downloading zlibwapi.dll...")
try:
    urllib.request.urlretrieve(dll_url, dll_path)
    print(f"✓ Downloaded to: {dll_path}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("\nManual fix:")
    print(f"1. Download from: {dll_url}")
    print(f"2. Place in: {dlls_dir}")
    sys.exit(1)

# Copy to Library/bin if it exists
print(f"\n[2/3] Copying to additional locations...")
if lib_dir.exists():
    shutil.copy(dll_path, lib_dir / "zlibwapi.dll")
    print(f"✓ Copied to: {lib_dir}")

# Also copy to current directory
current_dll = Path("zlibwapi.dll")
shutil.copy(dll_path, current_dll)
print(f"✓ Copied to: {current_dll.absolute()}")

print(f"\n[3/3] Installing Pillow (alternative to OpenCV)...")
os.system(f"{sys.executable} -m pip install Pillow --upgrade")

print("\n" + "="*70)
print("Fix Complete!")
print("="*70)
print("\nNow try running train_tf2.py again")
