# SLR Alphabet Recognizer

This project is a sign language alphabet recognizer using Python, OpenCV and TensorFlow for training InceptionV3 model, a convolutional neural network model for classification.

The framework used for the CNN implementation can be found here:
[Simple transfer learning with an Inception V3 architecture model](https://github.com/xuetsing/image-classification-tensorflow) by xuetsing

## Demo

You can [find the original demo here](https://youtu.be/kBw-xGEIYhY)

[![Demo](http://img.youtube.com/vi/kBw-xGEIYhY/0.jpg)](http://www.youtube.com/watch?v=kBw-xGEIYhY)

## Model Performance

- **Training Steps**: 2000
- **Final Test Accuracy**: 86.7%
- **Dataset Size**: 174,000 images across 29 classes (A-Z, space, del, nothing)

## Requirements

This project uses Python 3.10+ and the following packages:
* tensorflow
* opencv-python
* matplotlib
* numpy

See requirements.txt for specific versions.

### Install using PIP
```bash
pip install -r requirements.txt
```

### Using Docker
```bash
docker build -t hands-classifier .
docker run -it hands-classifier bash
```

## Training

To train the model, use the following command:
```bash
python train.py \
  --bottleneck_dir=logs/bottlenecks \
  --how_many_training_steps=2000 \
  --model_dir=inception \
  --summaries_dir=logs/training_summaries/basic \
  --output_graph=logs/trained_graph.pb \
  --output_labels=logs/trained_labels.txt \
  --image_dir=./dataset
```

**Note**: Training may take up to 3 hours depending on your hardware. The bottleneck files are cached, so subsequent training runs will be faster.

## Using the Model

### 1. Classify a Single Image
```bash
python classify.py path/to/image.jpg
```

### 2. Real-time Webcam Classification
```bash
python classify_webcam.py
```

**Instructions for webcam mode:**
- Position your hand inside the blue rectangle
- Make sign language gestures for letters A-Z
- Hold the same sign for a few frames to add it to the sequence
- Use "space" gesture to add a space
- Use "del" gesture to delete the last character
- Press **ESC** to exit

### 3. Test Multiple Images

Test all images in a folder with visual display:
```bash
python test_images.py
```

Get a quick summary of predictions:
```bash
python test_images_summary.py
```

## Project Structure

```
.
├── dataset/              # Training images organized by letter
├── Test/                 # Test images for evaluation
├── logs/                 # Model outputs (excluded from git)
│   ├── trained_graph.pb
│   └── trained_labels.txt
├── train.py              # Training script
├── classify.py           # Single image classification
├── classify_webcam.py    # Real-time webcam classification
├── test_images.py        # Interactive batch testing
├── test_images_summary.py # Quick batch testing summary
├── SLR_Training.ipynb    # Google Colab training notebook
└── requirements.txt      # Python dependencies
```

## Training on Google Colab

If you want to train on Google Colab for faster training with GPU:

1. Upload `SLR_Training.ipynb` to Google Colab
2. Enable GPU: Runtime → Change runtime type → Hardware accelerator → GPU
3. Update the git clone URL in the notebook
4. Run all cells
5. Download the trained model back to your local machine

## Test Results

Testing on 28 custom images showed:
- **High confidence predictions (≥90%)**: 7 images
- **Average confidence**: ~70%
- **Best performing letters**: C (98%), F (98%), P (95%), B (94%)
- **Challenging letters**: R (23%), Z (27%), W (33%) - these gestures are visually similar

## TensorFlow Compatibility

This project has been updated to work with TensorFlow 2.x using compatibility mode. The original code was written for TensorFlow 1.x.

## License

See LICENSE file for details.

## Acknowledgments

- Original framework by [xuetsing](https://github.com/xuetsing/image-classification-tensorflow)
- InceptionV3 model by Google
