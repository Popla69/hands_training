# GPU Setup Guide for Sign Language Recognition

## Your System
- **GPU:** NVIDIA GeForce MX250
- **TensorFlow:** 2.13.0
- **Python:** 3.10
- **Required CUDA:** 11.8
- **Required cuDNN:** 8.6

---

## Step-by-Step Installation

### Step 1: Download CUDA 11.8 (10 minutes)

1. Go to: https://developer.nvidia.com/cuda-11-8-0-download-archive
2. Select:
   - Operating System: Windows
   - Architecture: x86_64
   - Version: 10 or 11 (your Windows version)
   - Installer Type: exe (local)
3. Download the installer (~3 GB)
4. **DO NOT INSTALL YET** - just download

### Step 2: Download cuDNN 8.6 (5 minutes)

1. Go to: https://developer.nvidia.com/cudnn
2. Click "Download cuDNN"
3. Create free NVIDIA account if needed
4. Select: "cuDNN v8.6.0 for CUDA 11.x"
5. Download "cuDNN Library for Windows (x86)"
6. **DO NOT EXTRACT YET** - just download

### Step 3: Install CUDA 11.8 (15 minutes)

1. Run the CUDA installer you downloaded
2. Choose "Express Installation"
3. Wait for installation (takes 10-15 minutes)
4. Restart computer if prompted
5. Verify installation:
   ```
   nvcc --version
   ```
   Should show: "Cuda compilation tools, release 11.8"

### Step 4: Install cuDNN 8.6 (5 minutes)

1. Extract the cuDNN zip file
2. You'll see folders: bin, include, lib
3. Copy files to CUDA installation:
   - Copy `bin\cudnn*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
   - Copy `include\cudnn*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include`
   - Copy `lib\x64\cudnn*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64`

### Step 5: Verify GPU Detection (2 minutes)

Run this command:
```
python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

Should show: `GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

---

## If Something Goes Wrong

### CUDA Installation Fails
- Make sure you have admin rights
- Close all programs before installing
- Try "Custom Installation" instead of Express

### GPU Still Not Detected
- Restart computer
- Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
- Check CUDA path in environment variables

### TensorFlow Errors
- Reinstall TensorFlow:
  ```
  pip uninstall tensorflow
  pip install tensorflow==2.13.0
  ```

---

## Expected Performance Improvement

### Before GPU:
- Prediction: ~100-200ms per frame
- FPS: 5-10 predictions/sec
- Video: 20-25 FPS (with frame skipping)

### After GPU:
- Prediction: ~10-30ms per frame
- FPS: 30+ predictions/sec
- Video: 30 FPS (no frame skipping needed)

**Result: 5-10x faster!**

---

## Total Time Required

- Download: 15 minutes
- Install CUDA: 15 minutes
- Install cuDNN: 5 minutes
- Verify: 2 minutes
- **Total: ~40 minutes**

---

## Ready to Start?

Follow the steps above in order. Don't skip any steps!

Good luck! 🚀
