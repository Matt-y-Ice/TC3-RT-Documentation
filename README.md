# Tourniquet Detection and Tracking System

This application provides a real-time computer vision system for detecting and tracking
tourniquets in video feeds. It uses YOLO object detection models to identify tourniquets
and implements a tracking system to monitor their stability. The system is designed to
detect when a tourniquet has been properly applied based on position stability.

## Key Features

- **Camera Agility**: Automatically detects and uses Intel RealSense cameras when available, with seamless fallback to standard webcams when RealSense is not detected
- Real-time video processing from camera input
- Multi-scale object detection using YOLO models
- Feature tracking and motion detection for improved reliability
- Tourniquet stability monitoring to detect proper application
- GUI interface with live video feed and detection information
- Debug logging and video recording capabilities
- Real-time audio processing and speech transcription
- Adaptive audio threshold for speech detection
- Audio transcription with timestamps and duration

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- Intel RealSense SDK (optional - for RealSense camera support)
- Tkinter (for GUI)
- NumPy
- SciPy
- Sounddevice, Transformers, Pydub (for audio features)

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. For RealSense support, install the Intel RealSense SDK:
   ```
   pip install pyrealsense2
   ```

## Usage

Run the application with:
```
python gui_integration.py
```

The application will:
1. Check for an Intel RealSense camera
2. If RealSense is detected, use it for video input
3. If RealSense is not available, automatically fall back to a standard webcam
4. Display the video feed with real-time tourniquet detection and tracking
5. Show detection information in the GUI
6. Record audio and transcribe speech if audio features are enabled

## Camera Troubleshooting

If you're experiencing camera issues:

### For RealSense cameras:
- Ensure the RealSense SDK is properly installed
- Check USB connections (USB 3.0 recommended)
- Verify the camera is recognized by the system

### For webcams:
- Check if the camera is properly connected
- Ensure no other applications are using the camera
- Try a different USB port
- On Linux, verify your user is in the video group: `groups $USER`
- On Windows, check Device Manager for camera status
- On macOS, verify camera permissions in System Settings

## License

[Your License Information] 