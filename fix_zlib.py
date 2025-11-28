"""
Fix zlibwapi.dll missing error for OpenCV
Downloads and places the DLL in the correct location
"""
import os
import sys
import urllib.request
import shutil

def fix_zlib():
    print("Fixing zlibwapi.dll issue...")
    
    # Get OpenCV location
    import cv2
    cv2_path = os.path.dirname(cv2.__file__)
    print(f"OpenCV location: {cv2_path}")
    
    # Check if DLL already exists
    dll_path = os.path.join(cv2_path, "zlibwapi.dll")
    if os.path.exists(dll_path):
        print(f"✓ zlibwapi.dll already exists at: {dll_path}")
        return True
    
    # Download URL for zlibwapi.dll (64-bit) - from winlibs
    dll_url = "https://github.com/mstorsjo/llvm-mingw/releases/download/20230614/zlibwapi.dll"
    
    print(f"\nDownloading zlibwapi.dll...")
    print(f"From: {dll_url}")
    print(f"To: {dll_path}")
    
    try:
        # Download the file
        urllib.request.urlretrieve(dll_url, dll_path)
        print("✓ Downloaded successfully!")
        
        # Verify it exists
        if os.path.exists(dll_path):
            size = os.path.getsize(dll_path)
            print(f"✓ File size: {size} bytes")
            print(f"✓ zlibwapi.dll installed at: {dll_path}")
            return True
        else:
            print("✗ Download failed - file not found")
            return False
            
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        print("\nManual fix:")
        print("1. Download zlibwapi.dll from: https://github.com/opencv/opencv/raw/4.x/3rdparty/zlib/zlibwapi.dll")
        print(f"2. Place it in: {cv2_path}")
        return False

if __name__ == "__main__":
    success = fix_zlib()
    
    if success:
        print("\n" + "="*70)
        print("✓✓✓ FIXED! ✓✓✓")
        print("="*70)
        print("You can now run classify_HACKATHON.py")
    else:
        print("\n" + "="*70)
        print("Manual action required")
        print("="*70)
    
    input("\nPress ENTER to exit...")
