# Installation Guide

Complete installation and dependency resolution guide for Hand Landmark Detection V2.

## System Requirements

- **OS**: Windows 10/11, Linux, macOS
- **Python**: 3.8, 3.9, or 3.10
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for dependencies + models

## Quick Installation

```bash
# Clone or navigate to project directory
cd sign-language-alphabet-recognizer-master

# Install dependencies
pip install -r hand_landmark_v2/requirements.txt

# Test installation
python hand_landmark_v2/test_compatibility.py
```

## Detailed Installation

### Step 1: Python Environment

**Option A: Use existing Python**
```bash
python --version  # Should be 3.8-3.10
```

**Option B: Create virtual environment (recommended)**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 2: Install Core Dependencies

```bash
# PyTorch (CPU version)
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# OpenCV with GUI support
pip install opencv-python==4.10.0.84

# NumPy and SciPy
pip install numpy==1.26.4 scipy==1.11.4

# TensorBoard for training visualization
pip install tensorboard==2.15.1

# Other utilities
pip install tqdm==4.66.1 pillow==10.1.0 pyyaml==6.0.1
```

### Step 3: Install Optional Dependencies

```bash
# MediaPipe (for fallback hand detection)
pip install mediapipe==0.10.9

# ONNX Runtime (for ONNX inference)
pip install onnxruntime==1.16.3

# TensorFlow (if not already installed for sign classifier)
pip install tensorflow==2.15.0
```

### Step 4: Verify Installation

```bash
python hand_landmark_v2/test_compatibility.py
```

## Common Issues and Solutions

### Issue 1: OpenCV GUI Not Working

**Symptoms:**
```
cv2.error: The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

**Solution:**
```bash
# Uninstall all OpenCV packages
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y

# Install correct version
pip install opencv-python==4.10.0.84
```

### Issue 2: PyTorch + TensorFlow Conflict

**Symptoms:**
- Import errors
- Version conflicts
- Protobuf errors

**Solution:**
```bash
# Install in this specific order:
pip install torch==1.13.1 torchvision==0.14.1
pip install tensorflow==2.15.0
pip install protobuf==3.20.3
```

### Issue 3: MediaPipe Protobuf Conflict

**Symptoms:**
```
TypeError: Descriptors cannot not be created directly
```

**Solution:**
```bash
pip install protobuf==3.20.3
```

### Issue 4: NumPy Version Conflict

**Symptoms:**
```
AttributeError: module 'numpy' has no attribute 'X'
```

**Solution:**
```bash
pip install numpy==1.26.4 --force-reinstall
```

### Issue 5: CUDA/GPU Issues

**Symptoms:**
- CUDA errors
- GPU not detected

**Solution:**
```bash
# For CPU-only (recommended for this project):
pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu

# For GPU support:
# Visit https://pytorch.org and select your CUDA version
```

### Issue 6: Windows-Specific Issues

**Long Path Names:**
```bash
# Enable long paths in Windows
# Run as Administrator:
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f
```

**Permission Errors:**
```bash
# Run command prompt as Administrator
# Or use --user flag:
pip install --user <package>
```

## Platform-Specific Instructions

### Windows

```bash
# Use Command Prompt or PowerShell
python -m pip install -r hand_landmark_v2/requirements.txt

# If pip is not found:
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```

### Linux (Ubuntu/Debian)

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-dev

# Install Python packages
pip3 install -r hand_landmark_v2/requirements.txt
```

### macOS

```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Install packages
pip3 install -r hand_landmark_v2/requirements.txt
```

## Dependency Version Matrix

| Package | Version | Required | Notes |
|---------|---------|----------|-------|
| torch | 1.13.1 | Yes | Core ML framework |
| torchvision | 0.14.1 | Yes | Image processing |
| opencv-python | 4.10.0.84 | Yes | Computer vision |
| numpy | 1.26.4 | Yes | Numerical computing |
| scipy | 1.11.4 | Yes | Scientific computing |
| tensorflow | 2.15.0 | Yes* | Sign classifier |
| mediapipe | 0.10.9 | No | Fallback detection |
| onnxruntime | 1.16.3 | No | ONNX inference |
| tensorboard | 2.15.1 | No | Training visualization |

*Required only if using sign language classifier

## Testing Your Installation

### Test 1: Import Test
```python
python -c "import torch; import cv2; import numpy; print('✓ All imports successful')"
```

### Test 2: Model Creation
```python
python -c "from hand_landmark_v2.model import create_model; m = create_model(); print('✓ Model created')"
```

### Test 3: Full Compatibility
```bash
python hand_landmark_v2/test_compatibility.py
```

## Troubleshooting Checklist

- [ ] Python version is 3.8-3.10
- [ ] All required packages installed
- [ ] OpenCV GUI support working
- [ ] No version conflicts
- [ ] test_compatibility.py passes
- [ ] Can import all modules

## Getting Help

If you encounter issues not covered here:

1. Check the error message carefully
2. Run `python hand_landmark_v2/test_compatibility.py`
3. Check package versions: `pip list`
4. Try creating a fresh virtual environment
5. Check GitHub issues for similar problems

## Next Steps

After successful installation:

1. **Test the system**: `python classify_webcam_mediapipe.py`
2. **Train the model**: `python hand_landmark_v2/train.py`
3. **Run demos**: See README.md for usage examples
