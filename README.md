# Vehicle Counting Using YOLOv8 and SORT Tracker

This project implements a real-time vehicle counting system using the YOLOv8 object detection model combined with the SORT tracking algorithm. The system detects vehicles such as cars, buses, and trucks from video footage and counts them as they cross a designated counting line.

---

## Features

- **Object Detection**: Uses the YOLOv8 medium (`yolov8m.pt`) model for accurate detection of vehicles.
- **Multi-Object Tracking**: Employs the SORT algorithm to track detected vehicles frame-by-frame.
- **Vehicle Counting**: Counts vehicles crossing a predefined counting line, differentiating between cars, buses, and trucks.
- **Masking Support**: Optional mask image to focus detection on a specific region of interest in the video.
- **Visual Feedback**: Bounding boxes, class names, confidence scores, and track IDs displayed on video frames.
- **Performance**: Real-time processing with optimized model and tracking parameters.

---

## Demo

![Vehicle Counting Demo](demo.gif)  
*(Add your demo gif or screenshot here)*

---

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- cvzone
- numpy
- ultralytics (YOLOv8)
- sort (Simple Online and Realtime Tracking)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/vehicle-counting-yolov8-sort.git
cd vehicle-counting-yolov8-sort
