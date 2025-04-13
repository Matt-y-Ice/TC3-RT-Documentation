# Tourniquet Detection and Tracking System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8-green)
![Whisper](https://img.shields.io/badge/Whisper-OpenAI-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-red)
![Audio](https://img.shields.io/badge/Audio-Real--Time-orange)
![NLP](https://img.shields.io/badge/NLP-Contextual%20Extraction-purple)
![Status](https://img.shields.io/badge/Status-Active-blue)

This application provides a real-time computer vision and audio processing system for detecting, tracking, and verifying the correct application of tourniquets in live video feeds. It integrates object detection, motion tracking, speech transcription, and contextual analysis.

## Table of Contents

- [Features](#features)
- [Video Processing](#video-processing)
- [Audio Processing](#audio-processing)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Camera Troubleshooting](#camera-troubleshooting)

## Features

- **Device Adaptability**: Automatically detects Intel RealSense cameras, with fallback to standard webcams.
- Integrated video and audio pipelines.
- Live graphical interface showing video and detection status.
- Real-time audio recording, transcription, and natural language processing.

---

## Video Processing

- Real-time video capture from supported cameras.
- Multi-scale object detection using YOLO models to identify tourniquets.
- Feature-based tracking to monitor tourniquet movement and position.
- Stability detection to assess proper application based on minimal motion.
- GUI displaying:
  - Live video feed
  - Bounding boxes and labels for detected tourniquets
  - Tracking stability information
- Optional debug logging and video recording.

**Example Output in GUI:**

![Video Output Example](images/gui_video_output.png)

The GUI highlights detected tourniquets with labeled bounding boxes and stability indicators. A color-coded overlay (e.g., green for stable, red for unstable) indicates whether the detected item is considered properly applied. The detection confidence and object ID are also shown.

---

## Audio Processing

- Real-time audio capture during video monitoring sessions.
- Speech detection using adaptive volume thresholds.
- Transcription performed using the Whisper model for high-accuracy speech-to-text conversion.
- Natural language processing (NLP) applied to extract contextual information from the transcribed text.
- Timestamping and duration logging for all detected speech segments.
- Optional saving of transcriptions and audio logs for later review.

**Example Output in GUI:**

![Audio Output Example](images/audio_transcription_output.png)

Transcribed speech is displayed in a side panel with associated timestamps and durations. Contextual tags or summaries are extracted using NLP and shown beneath each transcription block to help users understand key spoken content related to the scene (e.g., commands, status updates).

---

## Requirements

- [Python 3.8+](https://www.python.org/downloads/)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Intel RealSense SDK](https://www.intelrealsense.com/sdk-2/)
- [Tkinter (Python GUI)](https://docs.python.org/3/library/tkinter.html)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [Sounddevice](https://python-sounddevice.readthedocs.io/)
- [Pydub](https://github.com/jiaaro/pydub)
- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [spaCy (for NLP)](https://spacy.io/)
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Matt-y-Ice/TC3-RT-Documentation.git
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install RealSense support:
   ```bash
   pip install pyrealsense2
   ```

---

## Usage

Run the application with:
```bash
python gui_integration.py
```

The system will:
- Attempt to initialize a RealSense camera, or fallback to a standard webcam.
- Begin video detection and tracking of tourniquets in real time.
- If audio features are enabled:
  - Begin recording audio.
  - Transcribe speech using Whisper.
  - Analyze transcripts for context using NLP.
  - Display or save results for review.

---

## Camera Troubleshooting

**For RealSense cameras:**
- Ensure SDK installation (`pyrealsense2`) is complete.
- Use USB 3.0 ports for stable bandwidth.
- Verify device visibility in system tools.

**For standard webcams:**
- Check physical connections and close other applications using the camera.
- On Linux: add user to the `video` group (`groups $USER`)
- On Windows: use Device Manager for diagnostics.
- On macOS: allow camera permissions via System Settings.
