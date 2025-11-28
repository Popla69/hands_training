"""
Benchmark CPU vs GPU Performance
Tests actual sign language model inference speed
"""

import os
import sys
import cv2
import numpy as np
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

print("="*70)
print("CPU vs GPU BENCHMARK - Sign Language Model")
print("="*70)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs available: {len(gpus)}")
if gpus:
    print(f"GPU: {gpus[0].name}")

# Load model
print("\nLoading model...")
try:
    import tensorflow.compat.v1 as tf_v1
    tf_v1.disable_v2_behavior()
    
    label_lines = [line.rstrip() for line in tf_v1.gfile.GFile("logs/trained_labels.txt")]
    
    with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
        graph_def = tf_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf_v1.import_graph_def(graph_def, name='')
    
    print(f"✓ Model loaded ({len(label_lines)} classes)")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Load test images
print("\nLoading test images...")
test_images = []
test_dir = 'dataset/A'  # Use A dataset for testing

if not os.path.exists(test_dir):
    print(f"✗ Test directory not found: {test_dir}")
    sys.exit(1)

all_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
sample_files = random.sample(all_files, min(50, len(all_files)))

for img_file in sample_files:
    img_path = os.path.join(test_dir, img_file)
    img = cv2.imread(img_path)
    if img is not None:
        img_resized = cv2.resize(img, (299, 299))
        image_data = cv2.imencode('.jpg', img_resized, 
                                 [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        test_images.append(image_data)

print(f"✓ Loaded {len(test_images)} test images")

# Benchmark function
def benchmark(device, num_iterations=50):
    """Run benchmark on specified device"""
    
    with tf_v1.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        
        # Warm up (first run is always slower)
        for i in range(5):
            _ = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': test_images[0]})
        
        # Actual benchmark
        times = []
        
        for i in range(num_iterations):
            img_data = test_images[i % len(test_images)]
            
            start = time.time()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': img_data})
            end = time.time()
            
            times.append((end - start) * 1000)  # Convert to ms
        
        return times

print("\n" + "="*70)
print("BENCHMARK RESULTS")
print("="*70)

# Test with GPU (default)
print("\n1. Testing with GPU...")
gpu_times = benchmark('GPU', num_iterations=50)
gpu_avg = np.mean(gpu_times)
gpu_std = np.std(gpu_times)
gpu_min = np.min(gpu_times)
gpu_max = np.max(gpu_times)

print(f"   Average: {gpu_avg:.2f}ms")
print(f"   Min: {gpu_min:.2f}ms")
print(f"   Max: {gpu_max:.2f}ms")
print(f"   Std Dev: {gpu_std:.2f}ms")
print(f"   FPS: {1000/gpu_avg:.1f}")

# Test with CPU
print("\n2. Testing with CPU...")
# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Reload TensorFlow to apply CPU-only mode
import importlib
importlib.reload(tf)
import tensorflow.compat.v1 as tf_v1
tf_v1.disable_v2_behavior()

# Reload model for CPU
with tf_v1.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
    graph_def = tf_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf_v1.import_graph_def(graph_def, name='')

cpu_times = benchmark('CPU', num_iterations=50)
cpu_avg = np.mean(cpu_times)
cpu_std = np.std(cpu_times)
cpu_min = np.min(cpu_times)
cpu_max = np.max(cpu_times)

print(f"   Average: {cpu_avg:.2f}ms")
print(f"   Min: {cpu_min:.2f}ms")
print(f"   Max: {cpu_max:.2f}ms")
print(f"   Std Dev: {cpu_std:.2f}ms")
print(f"   FPS: {1000/cpu_avg:.1f}")

# Comparison
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

speedup = cpu_avg / gpu_avg
print(f"\nGPU is {speedup:.2f}x faster than CPU")
print(f"Time saved per prediction: {cpu_avg - gpu_avg:.2f}ms")

# Real-world impact
print("\n" + "="*70)
print("REAL-WORLD IMPACT")
print("="*70)

print("\nFor webcam at 30 FPS:")
print(f"  CPU: Can predict every {int(cpu_avg/33.33)} frames ({30/(cpu_avg/33.33):.1f} predictions/sec)")
print(f"  GPU: Can predict every {int(gpu_avg/33.33)} frames ({30/(gpu_avg/33.33):.1f} predictions/sec)")

if gpu_avg < 33.33:
    print(f"\n✓ GPU can predict EVERY frame at 30 FPS!")
else:
    print(f"\n⚠ GPU needs to skip {int(gpu_avg/33.33)-1} frames for 30 FPS")

if cpu_avg < 33.33:
    print(f"✓ CPU can predict EVERY frame at 30 FPS!")
else:
    print(f"⚠ CPU needs to skip {int(cpu_avg/33.33)-1} frames for 30 FPS")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\n{'Metric':<20} {'CPU':<15} {'GPU':<15} {'Improvement':<15}")
print("-" * 70)
print(f"{'Avg Time':<20} {cpu_avg:.2f}ms{'':<8} {gpu_avg:.2f}ms{'':<8} {speedup:.2f}x faster")
print(f"{'FPS':<20} {1000/cpu_avg:.1f}{'':<12} {1000/gpu_avg:.1f}{'':<12} {(1000/gpu_avg)/(1000/cpu_avg):.2f}x more")
print(f"{'Time Saved':<20} {'-':<15} {cpu_avg-gpu_avg:.2f}ms{'':<8} {((cpu_avg-gpu_avg)/cpu_avg)*100:.1f}% faster")

print("\n" + "="*70)

if speedup > 2:
    print("✓ GPU provides SIGNIFICANT speedup!")
    print("  Your webcam classifier will be much smoother")
elif speedup > 1.5:
    print("✓ GPU provides GOOD speedup")
    print("  Noticeable improvement in performance")
else:
    print("⚠ GPU speedup is modest")
    print("  May be due to small model or overhead")

print("="*70)
