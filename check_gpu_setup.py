"""
Comprehensive GPU Setup Check
"""

import sys
import os

print("="*70)
print("GPU SETUP VERIFICATION")
print("="*70)

# 1. Check NVIDIA Driver
print("\n1. NVIDIA Driver Check:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line:
                print(f"   ✓ {line.strip()}")
                break
    else:
        print("   ✗ nvidia-smi failed")
except Exception as e:
    print(f"   ✗ Error: {e}")

# 2. Check CUDA Installation
print("\n2. CUDA Toolkit Check:")
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"   ✓ {line.strip()}")
                break
    else:
        print("   ✗ CUDA not found (nvcc not in PATH)")
except Exception as e:
    print("   ✗ CUDA not installed or not in PATH")

# 3. Check CUDA Path
print("\n3. CUDA Path Check:")
cuda_path = os.environ.get('CUDA_PATH')
if cuda_path:
    print(f"   ✓ CUDA_PATH: {cuda_path}")
    
    # Check if cuDNN files exist
    cudnn_dll = os.path.join(cuda_path, 'bin', 'cudnn64_8.dll')
    if os.path.exists(cudnn_dll):
        print(f"   ✓ cuDNN found: {cudnn_dll}")
    else:
        # Try other versions
        import glob
        cudnn_files = glob.glob(os.path.join(cuda_path, 'bin', 'cudnn*.dll'))
        if cudnn_files:
            print(f"   ✓ cuDNN found: {cudnn_files[0]}")
        else:
            print(f"   ✗ cuDNN not found in {cuda_path}\\bin")
else:
    print("   ✗ CUDA_PATH not set")

# 4. Check TensorFlow
print("\n4. TensorFlow Check:")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow version: {tf.__version__}")
    
    # Check if built with CUDA
    print(f"   ✓ Built with CUDA: {tf.test.is_built_with_cuda()}")
    
except Exception as e:
    print(f"   ✗ TensorFlow error: {e}")

# 5. Check GPU Detection
print("\n5. GPU Detection:")
try:
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   ✓ GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"      GPU {i}: {gpu.name}")
            
        # Get GPU details
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        if gpu_details:
            print(f"      Compute Capability: {gpu_details.get('compute_capability', 'Unknown')}")
    else:
        print("   ✗ No GPUs detected by TensorFlow")
        print("\n   Possible issues:")
        print("      - cuDNN not installed correctly")
        print("      - CUDA version mismatch")
        print("      - TensorFlow not compatible with CUDA version")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# 6. Test GPU Performance
print("\n6. GPU Performance Test:")
try:
    import tensorflow as tf
    import time
    
    # Create a simple computation
    with tf.device('/CPU:0'):
        cpu_start = time.time()
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        cpu_time = time.time() - cpu_start
        print(f"   CPU time: {cpu_time*1000:.2f}ms")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        with tf.device('/GPU:0'):
            gpu_start = time.time()
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            gpu_time = time.time() - gpu_start
            print(f"   GPU time: {gpu_time*1000:.2f}ms")
            
            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time
                print(f"   ✓ GPU is {speedup:.1f}x faster!")
            else:
                print(f"   ⚠ GPU slower than CPU (might be overhead for small ops)")
    else:
        print("   ⚠ Skipped (no GPU detected)")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# 7. Check cuDNN Version
print("\n7. cuDNN Version Check:")
try:
    import tensorflow as tf
    
    # Try to get cuDNN version
    try:
        from tensorflow.python.platform import build_info
        print(f"   ✓ cuDNN version: {build_info.build_info['cudnn_version']}")
        print(f"   ✓ CUDA version: {build_info.build_info['cuda_version']}")
    except:
        print("   ⚠ Could not determine cuDNN version from TensorFlow")
        
except Exception as e:
    print(f"   ✗ Error: {e}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus and tf.test.is_built_with_cuda():
        print("\n✓ GPU SETUP COMPLETE!")
        print("  Your system is ready to use GPU acceleration")
        print("  Expected speedup: 5-10x faster inference")
    elif tf.test.is_built_with_cuda() and not gpus:
        print("\n⚠ PARTIAL SETUP")
        print("  TensorFlow has CUDA support but can't detect GPU")
        print("  Possible issues:")
        print("    - cuDNN not installed or wrong version")
        print("    - CUDA version mismatch")
        print("    - Need to restart computer")
    else:
        print("\n✗ GPU NOT READY")
        print("  TensorFlow is not using GPU")
        print("  You're currently using CPU only")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")

print("\n" + "="*70)
