# Tourniquet Detection and Tracking System

A real-time computer vision system for detecting and tracking tourniquets in video feeds, designed for medical training and assistance scenarios. The system uses YOLO object detection models and implements a tracking system to monitor tourniquet stability, helping to detect when a tourniquet has been properly applied.

## Features

- **Real-time Video Processing**
  - Integration with Intel RealSense cameras
  - Multi-scale object detection using YOLO models
  - Feature tracking and motion detection for improved reliability
  - Tourniquet stability monitoring to detect proper application

- **Audio Processing**
  - Real-time speech transcription
  - Adaptive audio threshold for speech detection
  - Timestamped transcriptions with duration tracking
  - Hands-free operation support

- **User Interface**
  - Live video feed display
  - Real-time detection information
  - Live transcript display
  - Model detection logging
  - Start/Stop controls
  - Data export capabilities

- **Debug Features**
  - Comprehensive logging system
  - Debug video recording
  - Track history visualization
  - Motion detection visualization

## Prerequisites

- Python 3.8 or higher
- Intel RealSense camera (for video input)
- CUDA-capable GPU (recommended for optimal performance)
- Microphone (for audio features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/tourniquet-detection-system.git
cd tourniquet-detection-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the required YOLO models:
   - Place `yolo11n.pt` in the `models/` directory
   - Place `yolo11n-pose.pt` in the `models/` directory
   - Place `whisper-small-en-finetuned` in the `models/` directory

## Usage

1. Start the application:
```bash
python gui_integration.py
```

2. The application will open with:
   - Live video feed on the left
   - Transcript and detection information on the right
   - Control buttons at the bottom

3. Click "Start Demo" to begin:
   - Video processing will start
   - Audio transcription will begin (if microphone is available)
   - Detections will be displayed in real-time

4. Click "Stop Demo" to end the session

5. Use "Export Data" to save session information

## Configuration

The system can be configured by modifying the following parameters in `gui_integration.py`:

- Video parameters (resolution, FPS)
- Detection confidence thresholds
- Audio processing parameters
- Tracking parameters
- Stability thresholds

## Output

The system generates several types of output:

- Live video feed with detection overlays
- Real-time transcriptions
- Detection logs
- Debug videos (if enabled)
- Session data exports

All output files are saved in the `output/` directory, organized by date and time.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO model developers
- Intel RealSense team
- Whisper model developers
- OpenCV community 