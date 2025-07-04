# Face Detector Application

A simple and interactive face detection application using **OpenCV** and **PyQt5**. This desktop app captures real-time video from your webcam and detects faces using Haar Cascade Classifier, displaying them with bounding boxes on the GUI.

## 🚀 Features

- 🎥 Real-time face detection using your webcam  
- 🧠 Uses Haar Cascade (`haarcascade_frontalface_default.xml`)  
- 🖼️ PyQt5 GUI with live video feed  
- 🧵 Runs detection on a separate thread for smooth UI  
- 🛑 Start and stop detection with a button  
- ⚠️ Error handling with dialog messages

## 🛠️ Requirements

- Python 3.x
- OpenCV
- PyQt5

You can install the dependencies using pip:

```bash
pip install opencv-python PyQt5
```

## 📁 Project Structure
```
face_detector.py
haarcascade_frontalface_default.xml
```

## ▶️ Usage
1. Make sure your webcam is connected.
2. Run the application:
```
python face_detector.py
```
3. Click the Start Detection button to begin face detection.
4. Click Stop Detection to end the session.

## 🧠 How It Works
- Captures video from webcam using OpenCV (cv2.VideoCapture).
- Converts each frame to grayscale and runs face detection using Haar Cascade.
- Detected faces are marked with blue rectangles.
- Frames are converted to QImage and displayed in the PyQt5 GUI.
- All processing is handled in a background thread to keep the GUI responsive.

## 🧩 Notes
- The app uses cv2.data.haarcascades to load the XML model file automatically.
- Make sure no other apps are using the webcam while running this app.

## 📄 License
MIT License