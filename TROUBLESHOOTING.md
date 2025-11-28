# TROUBLESHOOTING GUIDE

## Problem: Model only predicts 'W' or predictions are terrible

### Step 1: Run Diagnostic
```
run_diagnose.bat
```
This will tell you if the model is working at all.

### Step 2: See RAW predictions
```
run_RAW.bat
```
This shows EXACTLY what the model predicts with NO filtering.
- If it shows different letters as you change signs → Model works, just needs better logic
- If it only shows 'W' or 'nothing' → Model is broken or not trained properly

### Step 3: Test with SIMPLE version
```
run_SIMPLE.bat
```
This is a simple version where YOU control when to add letters (press SPACE).
No automatic detection, no timers, just manual control.

## Common Issues

### Issue 1: Model only predicts one letter
**Cause:** Model wasn't trained properly or training data is bad
**Solution:** Need to retrain the model with better data

### Issue 2: Model predictions jump around
**Cause:** This is normal - single frames are noisy
**Solution:** Use temporal smoothing (multiple frames)

### Issue 3: Timer keeps restarting
**Cause:** Motion detection is too sensitive OR predictions keep changing
**Solution:** 
- Use the SIMPLE version (manual control with SPACE key)
- Or lower motion threshold
- Or increase confidence threshold

### Issue 4: Can't detect hand
**Cause:** MediaPipe hand detection failing
**Solution:**
- Better lighting
- Plain background
- Hand fully visible
- Try different camera angle

## Files Created

1. **classify_RAW.py** - Shows raw model output, no filtering
2. **classify_SIMPLE.py** - Manual control, press SPACE to add letter
3. **test_model_predictions.py** - Prints predictions to console
4. **diagnose_model.py** - Tests if model is working

## What to Check

1. **Is the model file correct?**
   - File: `logs/trained_graph.pb`
   - Size: Should be ~87 MB
   - Date: November 11, 2025

2. **Are labels correct?**
   - File: `logs/trained_labels.txt`
   - Should have 29 classes (a-z, space, del, nothing)

3. **Is camera working?**
   - Try camera index 0 and 1
   - Check if other apps can use camera

4. **Is hand visible?**
   - MediaPipe needs to see the full hand
   - Good lighting helps
   - Plain background helps

## Next Steps

If model is broken:
1. Check training data quality
2. Retrain model with `train.py`
3. Verify training completed successfully
4. Check if model file was created

If model works but predictions are bad:
1. Use SIMPLE version for now (manual control)
2. Collect more training data for confused letters
3. Retrain specific letters that are confused
