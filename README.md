# Face Recognition Project

This is a Face Recognition project that uses machine learning to identify people in real-time from video feeds.

## Core Features
- **Real-Time Face Detection**: Detects faces in video streams using OpenCV.
- **Firebase Integration**: Updates the presence status of recognized individuals in a Firebase database.
- **Simple Display**: Shows recognized faces with names and confidence levels on the video feed.

## How It Works
1. **Initialization**: Sets up Firebase and loads known faces from a folder.
2. **Video Capture**: Uses the webcam or an RTSP stream to capture video.
3. **Face Recognition**: Processes each frame to detect faces and compare them with known faces.
4. **Status Update**: Updates the Firebase database with the presence status of recognized individuals.

## Getting Started
1. Clone the repository.
2. Install the required dependencies (OpenCV, Firebase Admin SDK, TensorFlow, etc.).
3. Create a `db.json` file with your Firebase credentials.
4. Run the script to start recognizing faces!
