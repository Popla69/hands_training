"""
Simple CPU vs GPU Benchmark
"""

import os
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print("="*70)
print("SIMPLE CPU vs GPU BENCHMARK")
print("="*70)

# Test 1: GPU
print("\n1. Testing GPU...")
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"   GPUs detected: {len(gpus)}")

if gpus:
    # Create test computation
    with tf.device('/GPU:0'):
        # Simulate model inference (matrix operations)
        times_gpu = []
        for i in range(100):
            start = time.time()
            
            # Simulate Inception v3 operations
            a = tf.random.normal([1, 299, 299, 3])
            b = tf.nn.conv2d(a, tf.random.normal([3, 3, 3, 32]), strides=1, padding='SAME')
            c = tf.nn.relu(b)
            d = tf.reduce_mean(c)
            
            # Force execution
            _ = d.numpy()
            
            end = time.time()
            times_gpu.append((end - start) * 1000)
        
        gpu_avg = np.mean(times_gpu[10:])  # Skip first 10 for warmup
        print(f"   Average time: {gpu_avg:.2f}ms")
        print(f"   FPS: {1000/gpu_avg:.1f}")

# Test 2: CPU
print("\n2. Testing CPU...")
with tf.device('/CPU:0'):
    times_cpu = []
    for i in range(100):
        start = time.time()
        
        # Same operations on CPU
        a = tf.random.normal([1, 299, 299, 3])
        b = tf.nn.conv2d(a, tf.random.normal([3, 3, 3, 32]), strides=1, padding='SAME')
        c = tf.nn.relu(b)
        d = tf.reduce_mean(c)
        
        _ = d.numpy()
        
        end = time.time()
        times_cpu.append((end - start) * 1000)
    
    cpu_avg = np.mean(times_cpu[10:])
    print(f"   Average time: {cpu_avg:.2f}ms")
    print(f"   FPS: {1000/cpu_avg:.1f}")

# Comparison
print("\n" + "="*70)
print("RESULTS")
print("="*70)

if gpus:
    speedup = cpu_avg / gpu_avg
    print(f"\nGPU is {speedup:.2f}x faster than CPU")
    print(f"Time saved: {cpu_avg - gpu_avg:.2f}ms per operation")
    
    print("\nFor your sign language model:")
    print(f"  CPU: ~{cpu_avg*3:.0f}ms per prediction (~{1000/(cpu_avg*3):.1f} FPS)")
    print(f"  GPU: ~{gpu_avg*3:.0f}ms per prediction (~{1000/(gpu_avg*3):.1f} FPS)")
    
    if speedup > 2:
        print("\n✓ GPU provides SIGNIFICANT speedup!")
    elif speedup > 1.5:
        print("\n✓ GPU provides GOOD speedup")
    else:
        print("\n⚠ GPU speedup is modest")
else:
    print("\n✗ No GPU detected")

print("="*70)
