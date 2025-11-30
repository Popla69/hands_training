@echo off
echo ======================================================================
echo TESTING MODEL ON 1000 RANDOM IMAGES
echo ======================================================================
echo.
echo This will test your trained model on 1000 randomly selected images
echo from the dataset folder. This may take 5-10 minutes.
echo.
echo Press any key to start testing...
pause >nul

python test_1000_random_images.py

echo.
echo ======================================================================
echo Testing complete! Check test_results folder for detailed results.
echo ======================================================================
echo.
pause
