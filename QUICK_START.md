# ⚡ Quick Start Guide (5 Minutes!)

Want to get started FAST? Follow these steps!

## 1️⃣ Install Python
Download from: https://www.python.org/downloads/
- **Windows**: Check "Add Python to PATH" during install
- **Mac**: Use Homebrew: `brew install python`
- **Linux**: `sudo apt install python3 python3-pip`

## 2️⃣ Download Project
```bash
git clone https://github.com/Popla69/hands_training.git
cd hands_training
```

Or download ZIP from GitHub and unzip it.

## 3️⃣ Install Packages
Open terminal/command prompt in the project folder:

```bash
pip install -r requirements.txt
```

Wait 5-10 minutes for everything to download.

## 4️⃣ Run It!

### Try the Webcam Demo:
```bash
python classify_webcam_production.py
```

### Test a Picture:
```bash
python classify.py Test/A/1.jpg
```

## 🎉 That's It!

You should see your webcam turn on and start recognizing hand gestures!

## 🐛 Problems?

**"pip not found"**
- Try: `python -m pip install -r requirements.txt`

**"No webcam detected"**
- Make sure your webcam is plugged in
- Close other apps using the camera

**"Module not found"**
- Make sure you ran the install command in step 3

**Still stuck?**
- See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed help
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 📖 Want More Details?

- **Full Setup**: [SETUP_GUIDE.md](SETUP_GUIDE.md)
- **Train Your Own Model**: [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md)
- **Fix Problems**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

**Happy coding! 🚀**
