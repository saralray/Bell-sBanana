# 🍌 Banana Ripeness Color Analysis

A computer vision project using OpenCV and HSV color segmentation to
analyze banana ripeness based on color percentages.

## 🚀 Features

-   Automatic batch processing (unlimited images)
-   Background removal using HSV white detection
-   HSV-based color classification
-   Expanded yellow detection for accurate ripeness
-   Percentages normalized to 100%
-   High-resolution ripeness graph saved automatically
-   Natural filename sorting (1,2,3,10 correct order)

------------------------------------------------------------------------

## 📁 Project Structure

project/  
│ ├── main.py   
|     ├── img/  
|           ├── img-in  
|           ----├── 1.jpg  
|               ├── 2.jpg  
|               └── ...  
|           └── img-out  
|           ----├── 1_transparent.png  
|               ├── 2_transparent.png  
|               └── banana_color_plot.png  

------------------------------------------------------------------------

## 🎯 Color Detection

  Color    Meaning
  -------- ----------
  Green    Unripe
  Yellow   Ripe
  Brown    Overripe

All percentages are normalized:

Green + Yellow + Brown = 100%

------------------------------------------------------------------------

## ⚙️ Requirements

Install dependencies:

pip install opencv-python numpy matplotlib

------------------------------------------------------------------------

## ▶️ How To Run

1.  Put banana images inside:

img/img-in/

2.  Run:

python main.py

3.  Results will appear in:

img/img-out/

------------------------------------------------------------------------

## 📊 Output

For each image:

Green: 65.23% Yellow: 30.14% Brown: 4.63%

A ripeness progression graph will be saved as:

img/img-out/banana_color_plot.png

------------------------------------------------------------------------

## 🧠 How It Works

1.  Resize images with padding to 500x600
2.  Remove white background using HSV thresholds
3.  Detect Green, Yellow, and Brown using HSV segmentation
4.  Normalize percentages to 100%
5.  Generate and save ripeness graph

------------------------------------------------------------------------

## 🔥 Future Improvements

-   Ripeness score formula
-   Automatic stage classification
-   Real-time camera detection
-   CSV export
-   Machine learning classification

------------------------------------------------------------------------

Banana ripeness detection using OpenCV and HSV color segmentation.
