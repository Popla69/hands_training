# Training Guide - Achieving 95%+ Accuracy

This guide will help you train the hand landmark model to achieve 95%+ accuracy.

## Prerequisites

1. **Hardware Requirements**
   - CPU: Multi-core processor (8+ cores recommended)
   - RAM: 16GB minimum, 32GB recommended
   - GPU: NVIDIA GPU with 6GB+ VRAM (highly recommended)
   - Storage: 50GB+ free space for datasets

2. **Software Requirements**
   - Python 3.8-3.10
   - CUDA 11.x (if using GPU)
   - All dependencies from requirements.txt

## Step 1: Obtain Training Data

### Option A: FreiHAND Dataset (Recommended)

FreiHAND is the gold standard for hand pose estimation with 130K training images.

1. **Register and Download**
   ```
   Visit: https://lmb.informatik.uni-freiburg.de/projects/freihand/
   Register for access
   Download: training_rgb.zip, training_xyz.json, training_K.json
   ```

2. **Extract and Organize**
   ```bash
   mkdir -p data/freihand
   unzip training_rgb.zip -d data/freihand/
   mv training_xyz.json data/freihand/
   mv training_K.json data/freihand/
   ```

3. **Convert to Our Format**
   ```bash
   python hand_landmark_v2/download_datasets.py convert_freihand data/freihand data/freihand_converted
   ```

### Option B: CMU Hand Dataset

1. **Download**
   ```
   Visit: http://domedb.perception.cs.cmu.edu/handdb.html
   Download hand pose sequences
   ```

2. **Organize**
   ```bash
   mkdir -p data/cmu_hand
   # Extract and organize according to dataset structure
   ```

### Option C: Use Your Own Dataset

Create your own dataset with hand images and landmark annotations:

```bash
python hand_landmark_v2/download_datasets.py prepare images/ data/my_dataset
```

Then manually annotate using a tool like:
- LabelMe
- CVAT
- VGG Image Annotator (VIA)

## Step 2: Prepare Dataset

### Split Data

Create train/val/test splits (80/10/10):

```python
import json
import random

# Load annotations
with open('data/freihand_converted/annotations.json', 'r') as f:
    annotations = json.load(f)

# Shuffle
items = list(annotations.items())
random.shuffle(items)

# Split
n = len(items)
train_n = int(n * 0.8)
val_n = int(n * 0.1)

train_data = dict(items[:train_n])
val_data = dict(items[train_n:train_n+val_n])
test_data = dict(items[train_n+val_n:])

# Save splits
with open('data/freihand_converted/train_annotations.json', 'w') as f:
    json.dump(train_data, f)
with open('data/freihand_converted/val_annotations.json', 'w') as f:
    json.dump(val_data, f)
with open('data/freihand_converted/test_annotations.json', 'w') as f:
    json.dump(test_data, f)
```

### Verify Dataset

```bash
python hand_landmark_v2/download_datasets.py verify data/freihand_converted
```

## Step 3: Configure Training

Edit `hand_landmark_v2/config.py` for optimal settings:

```python
# Training hyperparameters
BATCH_SIZE = 64  # Increase if you have more GPU memory
LEARNING_RATE = 0.001
NUM_EPOCHS = 200  # More epochs for better accuracy
WEIGHT_DECAY = 1e-4

# Data augmentation (aggressive for robustness)
AUG_ROTATION_RANGE = 45  # Increased from 30
AUG_SCALE_RANGE = (0.7, 1.3)  # Wider range
AUG_BRIGHTNESS_RANGE = (0.6, 1.4)
AUG_CONTRAST_RANGE = (0.6, 1.4)
AUG_FLIP_PROB = 0.5
```

## Step 4: Train the Model

### Basic Training

```bash
python hand_landmark_v2/train.py \
  --data_dir data/freihand_converted \
  --output_dir hand_landmark_v2/checkpoints \
  --epochs 200 \
  --batch_size 64 \
  --lr 0.001
```

### Advanced Training (GPU)

```bash
# With GPU acceleration
CUDA_VISIBLE_DEVICES=0 python hand_landmark_v2/train.py \
  --data_dir data/freihand_converted \
  --output_dir hand_landmark_v2/checkpoints \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.001
```

### Resume Training

```bash
python hand_landmark_v2/train.py \
  --data_dir data/freihand_converted \
  --resume hand_landmark_v2/checkpoints/latest_checkpoint.pth
```

## Step 5: Monitor Training

### TensorBoard

```bash
tensorboard --logdir hand_landmark_v2/logs
```

Open browser to `http://localhost:6006`

### Watch Metrics

Monitor these metrics:
- **Train Loss**: Should decrease steadily
- **Val Loss**: Should decrease without overfitting
- **PCK@0.2**: Should reach >95%
- **Mean Error**: Should be <0.02 (normalized coordinates)

### Expected Timeline

- **Epoch 1-20**: Rapid improvement (PCK: 60-80%)
- **Epoch 20-50**: Steady improvement (PCK: 80-90%)
- **Epoch 50-100**: Fine-tuning (PCK: 90-95%)
- **Epoch 100-200**: Refinement (PCK: 95%+)

## Step 6: Evaluate Model

### Test Set Evaluation

```python
from hand_landmark_v2.train import HandLandmarkTrainer
from hand_landmark_v2.losses import compute_pck, compute_mean_error
import torch

# Load best model
trainer = HandLandmarkTrainer()
trainer.load_checkpoint('hand_landmark_v2/checkpoints/best_model.pth')

# Evaluate on test set
test_metrics = trainer.validate()  # Run on test split

print(f"Test PCK@0.2: {test_metrics['pck']:.2f}%")
print(f"Test Mean Error: {test_metrics['mean_error']:.4f}")
```

### Visual Inspection

```bash
# Test on sample images
python hand_landmark_v2/demo_image.py data/test_images/ --output results/
```

## Step 7: Optimize for Deployment

### Export to ONNX

```bash
python hand_landmark_v2/export.py \
  hand_landmark_v2/checkpoints/best_model.pth \
  onnx
```

### Export to TFLite

```bash
# FP32
python hand_landmark_v2/export.py \
  hand_landmark_v2/checkpoints/best_model.pth \
  tflite

# INT8 (smaller, faster)
python hand_landmark_v2/export.py \
  hand_landmark_v2/checkpoints/best_model.pth \
  tflite --quantize
```

### Validate Exports

```bash
python -c "
from hand_landmark_v2.export import validate_export
validate_export(
    'hand_landmark_v2/checkpoints/best_model.pth',
    onnx_path='hand_landmark_v2/models/hand_landmark.onnx',
    tflite_path='hand_landmark_v2/models/hand_landmark.tflite'
)
"
```

## Step 8: Fine-Tuning Tips

### If Accuracy is Below 95%

1. **More Data**
   - Combine multiple datasets
   - Add more augmentation
   - Collect domain-specific data

2. **Longer Training**
   - Increase epochs to 300-500
   - Use learning rate scheduling
   - Try different optimizers (AdamW, SGD with momentum)

3. **Model Architecture**
   - Try MobileNetV3-Large instead of Small
   - Add more layers to landmark head
   - Experiment with different loss weights

4. **Hyperparameter Tuning**
   ```python
   # Try these combinations
   learning_rates = [0.0001, 0.0005, 0.001, 0.005]
   batch_sizes = [32, 64, 128]
   weight_decays = [1e-5, 1e-4, 1e-3]
   ```

5. **Advanced Techniques**
   - Mixup augmentation
   - Cutout/CutMix
   - Label smoothing
   - Ensemble multiple models

### If Overfitting

1. **Regularization**
   - Increase weight decay
   - Add dropout (0.3-0.5)
   - Use batch normalization

2. **Data Augmentation**
   - More aggressive augmentation
   - Add noise augmentation
   - Random erasing

3. **Early Stopping**
   - Stop when val loss stops improving
   - Use patience of 20-30 epochs

## Step 9: Benchmark Performance

```bash
# Test FPS
python -c "
from hand_landmark_v2.inference import benchmark_model
benchmark_model('hand_landmark_v2/checkpoints/best_model.pth', backend='pytorch', num_iterations=100)
"

# Test with different backends
python -c "
from hand_landmark_v2.inference import benchmark_model
benchmark_model('hand_landmark_v2/models/hand_landmark.onnx', backend='onnx', num_iterations=100)
"
```

## Step 10: Deploy

Once you achieve 95%+ accuracy:

```bash
# Test integration
python classify_webcam_v2.py

# Run full integration tests
python hand_landmark_v2/test_integration.py

# Deploy
# See DEPLOYMENT.md for production deployment
```

## Troubleshooting

### Training is Slow

- **Use GPU**: Install CUDA and PyTorch with GPU support
- **Increase Batch Size**: If you have more GPU memory
- **Use Mixed Precision**: Add `torch.cuda.amp` for faster training
- **Reduce Image Size**: Use 192x192 instead of 224x224

### Not Converging

- **Lower Learning Rate**: Try 0.0001 or 0.0005
- **Check Data**: Verify annotations are correct
- **Simplify Model**: Start with fewer augmentations
- **Warm-up**: Use learning rate warm-up for first 5 epochs

### Out of Memory

- **Reduce Batch Size**: Try 32 or 16
- **Use Gradient Accumulation**: Accumulate over 2-4 steps
- **Clear Cache**: Add `torch.cuda.empty_cache()` in training loop
- **Use CPU**: Set `use_gpu=False` (slower but works)

## Expected Results

With FreiHAND dataset and proper training:

| Metric | Expected Value |
|--------|----------------|
| PCK@0.2 | 95-98% |
| Mean Error | 0.015-0.020 |
| Training Time | 12-24 hours (GPU) |
| Final Model Size | 5-6 MB |
| Inference FPS (CPU) | 30-40 |
| Inference FPS (GPU) | 100-150 |

## Next Steps After Training

1. **Integrate with Sign Classifier**
   ```bash
   python classify_webcam_v2.py
   ```

2. **Test on Real Data**
   - Test with your webcam
   - Test with various lighting conditions
   - Test with different hand sizes/skin tones

3. **Collect Feedback**
   - Monitor accuracy in production
   - Collect failure cases
   - Retrain with additional data

4. **Continuous Improvement**
   - Periodically retrain with new data
   - Fine-tune for specific use cases
   - Update model as needed

## Support

If you encounter issues during training:
1. Check TensorBoard for training curves
2. Verify dataset with `verify` command
3. Start with small dataset (1000 images) to test pipeline
4. Use synthetic data first to verify code works

Good luck with training! 🚀
