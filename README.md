# 👋 Sign Language Hand Gesture Recognition

A fun project that teaches computers to understand sign language! Point your webcam at your hand, make a sign language letter, and watch the computer recognize it in real-time!

## 🎯 What Does This Do?

This project can:
- **Recognize sign language letters** (A to Z) using your webcam
- **Train a computer** to learn new hand gestures
- **Detect hand landmarks** (the points on your fingers and palm)
- **Work in real-time** - see instant results!

## 🎥 See It In Action

[![Demo Video](http://img.youtube.com/vi/kBw-xGEIYhY/0.jpg)](http://www.youtube.com/watch?v=kBw-xGEIYhY)

## 🚀 Quick Start (3 Easy Steps!)

### Step 1: Install Python
You need Python 3.7 or newer on your computer.
- Download from: https://www.python.org/downloads/
- During installation, check the box that says "Add Python to PATH"

### Step 2: Download This Project
Click the green "Code" button above and select "Download ZIP", then unzip it.

Or if you know git:
```bash
git clone https://github.com/Popla69/hands_training.git
cd hands_training
```

### Step 3: Install Required Packages
Open Command Prompt (Windows) or Terminal (Mac/Linux) in the project folder and run:

```bash
pip install -r requirements.txt
```

**Note**: If you get errors, see the [SETUP_GUIDE.md](SETUP_GUIDE.md) file for detailed help!

## 🎮 How to Use

### Option 1: Try the Webcam Demo (Most Fun!)

```bash
python classify_webcam_production.py
```

**How to use it:**
1. Your webcam will turn on
2. Put your hand in the blue rectangle on screen
3. Make a sign language letter (like ✊ for 'A')
4. The computer will guess what letter you're making!
5. Press **ESC** to exit

### Option 2: Test a Single Picture

```bash
python classify.py path/to/your/image.jpg
```

### Option 3: Test Multiple Pictures at Once

```bash
python test_images.py
```

## 📁 What's Inside?

```
hands_training/
│
├── 📷 classify_webcam_production.py  ← Start here! (Webcam demo)
├── 🖼️  classify.py                   ← Test single images
├── 🎓 train.py                       ← Train your own model
├── 📋 requirements.txt               ← List of needed packages
├── 📖 SETUP_GUIDE.md                 ← Detailed setup help
│
├── 📂 dataset/                       ← Training images (download separately)
├── 📂 models_tf2/                    ← Pre-trained AI models
├── 📂 hand_landmark_v2/              ← Hand detection system
└── 📂 Test/                          ← Sample test images
```

## 🎓 Want to Train Your Own Model?

See [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md) for a step-by-step guide!

**Quick version:**
```bash
python train.py
```

**Note**: Training takes 1-3 hours and requires a dataset of hand gesture images.

## 📦 Download Training Dataset

The training images are too large for GitHub. Download them from:
- **ASL Alphabet Dataset**: [Kaggle Link](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **FreiHAND Dataset**: [Project Page](https://lmb.informatik.uni-freiburg.de/projects/freihand/)

Put the downloaded images in the `dataset/` folder.

## ❓ Having Problems?

Check these guides:
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed installation instructions
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common problems and fixes
- **[HOW_TO_TRAIN.md](HOW_TO_TRAIN.md)** - Training your own model

## 🎯 What Can You Build With This?

- 🗣️ Communication tool for deaf/hard-of-hearing people
- 🎮 Gesture-controlled games
- 📚 Sign language learning app
- 🤖 Smart home controls with hand gestures
- 🎨 Interactive art installations

## 🛠️ Technical Details (For Developers)

- **AI Model**: InceptionV3 (transfer learning)
- **Framework**: TensorFlow 2.x
- **Computer Vision**: OpenCV + MediaPipe
- **Languages**: Python 3.7+
- **Accuracy**: ~87% on test set
- **Classes**: 29 (A-Z, space, delete, nothing)

## 📊 Model Performance

- Training Steps: 2000
- Test Accuracy: 86.7%
- Dataset Size: 174,000 images
- Best Letters: C (98%), F (98%), P (95%)
- Challenging Letters: R, Z, W (similar gestures)

## 🤝 Contributing

Want to make this better? 
1. Fork this repository
2. Make your changes
3. Submit a pull request

Ideas for improvements:
- Add more sign language alphabets (BSL, ISL, etc.)
- Improve accuracy for similar-looking letters
- Add word recognition (not just letters)
- Mobile app version

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Credits

- Original framework by [xuetsing](https://github.com/xuetsing/image-classification-tensorflow)
- InceptionV3 model by Google
- MediaPipe by Google
- ASL Dataset from Kaggle

## 💬 Questions?

Open an issue on GitHub or check the documentation files!

---

**Made with ❤️ for making technology more accessible**
