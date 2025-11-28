# 🛠️ Complete Setup Guide (Step-by-Step)

This guide will help you set up the project from scratch, even if you're new to programming!

## 📋 What You'll Need

- A computer (Windows, Mac, or Linux)
- A webcam (built-in or USB)
- Internet connection
- About 30 minutes

## Step 1: Install Python 🐍

### Windows:
1. Go to https://www.python.org/downloads/
2. Click the big yellow "Download Python" button
3. Run the installer
4. **IMPORTANT**: Check the box "Add Python to PATH" at the bottom
5. Click "Install Now"
6. Wait for it to finish

### Mac:
1. Open Terminal (search for it in Spotlight)
2. Install Homebrew (if you don't have it):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Install Python:
   ```bash
   brew install python
   ```

### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### Check if Python is Installed:
Open Command Prompt (Windows) or Terminal (Mac/Linux) and type:
```bash
python --version
```
You should see something like "Python 3.10.0" or higher.

## Step 2: Download This Project 📥

### Option A: Download ZIP (Easiest)
1. Go to https://github.com/Popla69/hands_training
2. Click the green "Code" button
3. Click "Download ZIP"
4. Unzip the file to a folder you can find easily (like Desktop or Documents)

### Option B: Use Git (If You Know How)
```bash
git clone https://github.com/Popla69/hands_training.git
cd hands_training
```

## Step 3: Open Command Prompt/Terminal in Project Folder 💻

### Windows:
1. Open File Explorer
2. Navigate to the project folder
3. Click in the address bar at the top
4. Type `cmd` and press Enter

### Mac/Linux:
1. Open Terminal
2. Type `cd ` (with a space after cd)
3. Drag the project folder into the Terminal window
4. Press Enter

## Step 4: Create a Virtual Environment (Optional but Recommended) 🏠

This keeps the project's packages separate from your other Python projects.

```bash
python -m venv venv
```

### Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You'll see `(venv)` appear at the start of your command line.

## Step 5: Install Required Packages 📦

Now install all the packages the project needs:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This might take 5-10 minutes. You'll see lots of text scrolling by - that's normal!

### If You Get Errors:

**Error: "pip is not recognized"**
- Try using `python -m pip` instead of just `pip`

**Error: "Microsoft Visual C++ required"**
- Windows users: Download and install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

**Error: "Permission denied"**
- Mac/Linux: Add `sudo` before the command: `sudo pip install -r requirements.txt`

**Error: TensorFlow won't install**
- Try installing an older version:
  ```bash
  pip install tensorflow==2.10.0
  ```

## Step 6: Download Pre-trained Models (Optional) 🤖

The project includes some pre-trained models, but if you want the latest:

1. Check if `models_tf2/` folder exists
2. If it has files inside, you're good to go!
3. If not, you'll need to train your own (see HOW_TO_TRAIN.md)

## Step 7: Test Your Setup ✅

Let's make sure everything works!

### Test 1: Check Imports
```bash
python -c "import tensorflow; import cv2; print('Success!')"
```

If you see "Success!" - you're good! If you see errors, see the Troubleshooting section below.

### Test 2: Test Your Webcam
```bash
python test_camera_simple.py
```

Your webcam should turn on. Press ESC to close it.

### Test 3: Run the Main Program
```bash
python classify_webcam_production.py
```

If you see your webcam and a blue rectangle, congratulations! 🎉

## 🎮 You're Ready to Go!

Try these commands:

**Webcam demo:**
```bash
python classify_webcam_production.py
```

**Test a single image:**
```bash
python classify.py Test/A/1.jpg
```

**Test multiple images:**
```bash
python test_images.py
```

## 🐛 Troubleshooting Common Issues

### "No module named 'tensorflow'"
```bash
pip install tensorflow
```

### "No module named 'cv2'"
```bash
pip install opencv-python
```

### "Camera not found" or "Can't open webcam"
- Make sure no other program is using your webcam
- Try unplugging and replugging USB webcams
- Check your privacy settings (Windows/Mac might block camera access)

### "ImportError: DLL load failed" (Windows)
- Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Program is very slow
- Close other programs
- Make sure you're not running on battery saver mode
- Consider using a GPU version of TensorFlow (advanced)

### "Permission denied" errors
- Mac/Linux: Use `sudo` before commands
- Windows: Run Command Prompt as Administrator

## 🆘 Still Having Problems?

1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Make sure you have Python 3.7 or newer
3. Try creating a fresh virtual environment
4. Open an issue on GitHub with:
   - Your operating system
   - Python version (`python --version`)
   - The exact error message
   - What you were trying to do

## 🎓 Next Steps

Once everything is working:
- Read [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md) to train your own model
- Check out [DEMO_GUIDE.md](DEMO_GUIDE.md) for tips on using the webcam demo
- Explore the different Python files to see what they do

## 📚 Learning Resources

New to Python or AI? Check these out:
- [Python for Beginners](https://www.python.org/about/gettingstarted/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

---

**You've got this! 💪**
