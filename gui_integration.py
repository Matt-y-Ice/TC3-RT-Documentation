import cv2
import tkinter as tk
from tkinter import Text, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import queue
import time
import re
import warnings
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import os
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================
# Configuration
# ==============================================
BATCH_SIZE = 4           
FRAME_WIDTH, FRAME_HEIGHT = 640, 640  # Changed to match YOLO input size
OUTPUT_FPS = 30          
OUTPUT_DIR = "output"  # Directory to save output files
DISPLAY_BUFFER_SIZE = 5  # Number of frames to keep in display buffer
RECORDING_BUFFER_SIZE = 1000  # Number of frames to keep in recording buffer
SAVE_DEBUG_VIDEO = True  # Flag to enable/disable debug video saving

# Detection parameters
DETECTION_CONFIDENCE = 0.01  # Even lower confidence threshold
MIN_BOX_AREA = 300  # Lower minimum area
MAX_BOX_AREA = 200000  # Increased maximum area
TEMPORAL_SMOOTHING = 5  # Reduced for faster response to lost detections
MOTION_THRESHOLD = 15  # Lower threshold for motion detection
TRACKING_HISTORY = 10  # Reduced tracking history
TRACKING_IOU_THRESHOLD = 0.3  # Increased threshold for tracking association
MOTION_CONFIDENCE_BOOST = 2.0  # Reduced confidence boost for motion
MIN_TRACK_LENGTH = 2  # Lower minimum track length
MAX_TRACK_GAP = 3  # Reduced maximum gap between detections
MAX_TRACK_AGE = 5  # Maximum age of a track without updates
MIN_TRACK_CONFIDENCE = 0.2  # Minimum confidence to maintain a track

# Multi-scale detection parameters
SCALE_FACTORS = [0.8, 1.0, 1.2]  # Different scales to try
SCALE_CONFIDENCE_WEIGHTS = [0.8, 1.0, 0.8]  # Weights for each scale

# Feature tracking parameters
FEATURE_MAX_CORNERS = 100
FEATURE_QUALITY_LEVEL = 0.3
FEATURE_MIN_DISTANCE = 7
FEATURE_BLOCK_SIZE = 7

# Configure RealSense pipeline
RS_WIDTH = 640   
RS_HEIGHT = 480  
RS_FPS = 30      

# Define the keypoint connections for pose visualization
POSE_CONNECTIONS = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],  # arms and shoulders
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],    # body and face
    [2, 4], [3, 5], [4, 6], [5, 7]                                        # legs
]

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSC-490 Live Demo")

        # Initialize pipeline components
        self.frame_queue = queue.Queue(maxsize=20)
        self.processed_queue = queue.Queue(maxsize=20)
        self.stop_event = threading.Event()
        
        # Initialize display buffer
        self.display_buffer = deque(maxlen=DISPLAY_BUFFER_SIZE)
        self.recording_buffer = deque(maxlen=RECORDING_BUFFER_SIZE)
        self.last_valid_frame = None
        
        # Initialize detection and tracking history
        self.detection_history = deque(maxlen=TEMPORAL_SMOOTHING)
        self.tracking_history = deque(maxlen=TRACKING_HISTORY)
        self.last_frame = None
        self.active_tracks = []
        self.track_counter = 0
        self.last_keypoints = None
        self.last_descriptors = None
        
        # Initialize video writer
        self.video_writer = None
        self.is_recording = False
        
        # Load YOLO models
        self.object_model = YOLO("best.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")
        
        # Initialize feature detector
        self.feature_detector = cv2.FastFeatureDetector_create()
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Create output directory if it doesn't exist
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # -----------------------------
        # 1) SETUP MAIN GUI COMPONENTS
        # -----------------------------
        self.main_frame = Frame(root, bg="#b0bec5", padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights to allow resizing
        self.main_frame.grid_columnconfigure(0, weight=3)  # Video column
        self.main_frame.grid_columnconfigure(1, weight=2)  # Text column
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Webcam feed
        self.video_label = Label(self.main_frame, bg="white")
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right panel
        self.right_panel = Frame(self.main_frame, bg="#b0bec5")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_panel.grid_columnconfigure(0, weight=1)

        # Text area for live transcript
        self.label1 = Label(self.right_panel, text="Live Transcript", font=("Arial", 12, "bold"), bg="white")
        self.label1.pack(fill=tk.X, pady=(0, 5))
        self.text_area1 = Text(self.right_panel, state='disabled')
        self.text_area1.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Text area for any other detections (optional)
        self.label2 = Label(self.right_panel, text="Model Detections", font=("Arial", 12, "bold"), bg="white")
        self.label2.pack(fill=tk.X, pady=(0, 5))
        self.text_area2 = Text(self.right_panel, state='disabled')
        self.text_area2.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Button panel
        self.button_panel = Frame(self.main_frame, bg="#b0bec5")
        self.button_panel.grid(row=1, column=1, pady=10, sticky="e")

        # Initially the button says "Start Demo"
        self.start_button = Button(self.button_panel, text="Start Demo", bg="#7749F8", fg="white",
                                   command=self.toggle_transcription)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.export_button = Button(self.button_panel, text="Export Data", bg="#7749F8", fg="white",
                                  command=self.export_data)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Initialize pipeline threads
        self.capture_thread = None
        self.pipeline_thread = None
        self.pipeline_running = False

        # On app close, release resources
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Set initial window size and allow resizing
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)

        # Create initial blank frame
        self.blank_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(self.blank_frame, "No camera feed", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.last_valid_frame = self.blank_frame.copy()

        # Start the video update loop immediately
        self.update_video()

    def capture_frames(self):
        """Continuously capture frames from the RealSense camera"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)

        try:
            pipeline.start(config)
            print("[Capture] RealSense camera started successfully")
            timeout_ms = 1000  # 1 second timeout

            while not self.stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms)
                    color_frame = frames.get_color_frame()
                    
                    if not color_frame:
                        print("[Capture] Empty frame received. Retrying...")
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    # Resize to match YOLO input size
                    frame_resized = cv2.resize(color_image, (FRAME_WIDTH, FRAME_HEIGHT))
                    
                    if not self.frame_queue.full():
                        self.frame_queue.put(frame_resized)

                except RuntimeError as e:
                    if "Frame didn't arrive" in str(e):
                        print("[Capture] Frame timeout, attempting to recover...")
                        try:
                            pipeline.stop()
                            time.sleep(1)
                            pipeline.start(config)
                        except Exception as restart_error:
                            print(f"[Capture] Failed to restart pipeline: {restart_error}")
                    else:
                        print(f"[Capture] Runtime error: {e}")
                    time.sleep(0.1)
                except Exception as e:
                    print(f"[Capture] Exception: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"[Capture] Failed to initialize RealSense camera: {e}")
        finally:
            try:
                pipeline.stop()
                print("[Capture] RealSense pipeline stopped")
            except Exception as e:
                print(f"[Capture] Error stopping pipeline: {e}")

    def detect_motion(self, current_frame, last_frame):
        """Detect motion between frames to help identify regions of interest"""
        if last_frame is None:
            return None
            
        # Convert frames to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(current_gray, last_gray)
        
        # Apply threshold to get motion mask
        _, motion_mask = cv2.threshold(frame_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Find contours of motion regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes of motion regions
        motion_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            motion_boxes.append((x, y, x + w, y + h))
            
        return motion_boxes

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / float(box1_area + box2_area - intersection)

    def update_tracks(self, detections, frame):
        """Update tracking information for detected objects"""
        # Age all existing tracks
        for track in self.active_tracks:
            track['age'] += 1
            # Reduce confidence over time more aggressively
            track['confidence'] *= 0.8  # Decay confidence by 20% per frame
        
        # Remove old or low confidence tracks
        self.active_tracks = [
            track for track in self.active_tracks 
            if track['age'] <= MAX_TRACK_AGE and track['confidence'] >= MIN_TRACK_CONFIDENCE
        ]

        if not detections:
            # When there are no detections, be more aggressive about removing tracks
            self.active_tracks = [
                track for track in self.active_tracks 
                if track['age'] <= 2  # Only keep tracks that are very recent
            ]
            return

        # Convert detections to numpy array for efficient processing
        detections = np.array(detections)
        
        # Initialize cost matrix for tracking
        if self.active_tracks:
            cost_matrix = np.zeros((len(self.active_tracks), len(detections)))
            
            # Calculate IoU between existing tracks and new detections
            for i, track in enumerate(self.active_tracks):
                for j, det in enumerate(detections):
                    cost_matrix[i, j] = self.calculate_iou(
                        track['bbox'],
                        det[:4]
                    )
            
            # Assign detections to tracks using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
            
            # Update existing tracks
            for i, j in zip(row_ind, col_ind):
                if cost_matrix[i, j] > TRACKING_IOU_THRESHOLD:
                    track = self.active_tracks[i]
                    det = detections[j]
                    
                    # Update track with new detection
                    track['bbox'] = det[:4]
                    track['confidence'] = max(det[4], track['confidence'])  # Keep highest confidence
                    track['age'] = 0
                    track['history'].append(det[:4])
                    if len(track['history']) > TRACKING_HISTORY:
                        track['history'].popleft()
                    
                    # Remove assigned detection
                    detections = np.delete(detections, j, axis=0)
            
            # Remove tracks that haven't been updated
            self.active_tracks = [t for t in self.active_tracks if t['age'] <= MAX_TRACK_GAP]
        
        # Create new tracks for unassigned detections
        for det in detections:
            if len(det) >= 6:  # Ensure det has all required elements
                new_track = {
                    'id': self.track_counter,
                    'bbox': det[:4],
                    'confidence': det[4],
                    'age': 0,
                    'history': deque(maxlen=TRACKING_HISTORY),
                    'class_id': int(det[5])
                }
                new_track['history'].append(det[:4])
                self.active_tracks.append(new_track)
                self.track_counter += 1

    def predict_track_positions(self):
        """Predict next positions of tracks using simple motion model"""
        for track in self.active_tracks:
            if len(track['history']) >= 2:
                # Calculate velocity from last two positions
                last_pos = track['history'][-1]
                prev_pos = track['history'][-2]
                velocity = np.array(last_pos) - np.array(prev_pos)
                
                # Predict next position
                predicted_pos = np.array(last_pos) + velocity
                track['predicted_bbox'] = predicted_pos.tolist()

    def preprocess_frame(self, frame):
        """Apply preprocessing to enhance features"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Convert back to BGR
        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    def detect_features(self, frame):
        """Detect and track features in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect keypoints
        keypoints = self.feature_detector.detect(gray, None)
        
        # Convert keypoints to numpy array
        keypoints = np.array([kp.pt for kp in keypoints])
        
        return keypoints

    def match_features(self, current_keypoints, last_keypoints):
        """Match features between frames"""
        if last_keypoints is None or len(last_keypoints) == 0:
            return None
            
        # Convert to float32 for matching
        current_kp = np.float32(current_keypoints).reshape(-1, 1, 2)
        last_kp = np.float32(last_keypoints).reshape(-1, 1, 2)
        
        # Calculate optical flow
        lk_params = dict(winSize=(15, 15),
                        maxLevel=2,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.last_gray, self.current_gray, last_kp, None, **lk_params)
        
        # Filter good matches
        good_new = next_points[status == 1]
        good_old = last_kp[status == 1]
        
        return good_old, good_new

    def process_frames(self):
        """Process frames through YOLO models and annotate them"""
        while not self.stop_event.is_set():
            try:
                # Collect frames to form a batch
                batch_frames = []
                for _ in range(BATCH_SIZE):
                    try:
                        frame = self.frame_queue.get(timeout=1.0)
                        batch_frames.append(frame)
                    except queue.Empty:
                        print("[Pipeline] No frames available, skipping batch")
                        continue

                if not batch_frames:
                    continue

                batch_for_detection = batch_frames[:]
                batch_for_pose = batch_frames[:]

                # Results containers
                detection_results = [None] * len(batch_frames)
                pose_results = [None] * len(batch_frames)

                def detection_worker():
                    # Get motion regions and features for the first frame
                    motion_boxes = None
                    feature_matches = None
                    
                    if self.last_frame is not None:
                        # Preprocess frames
                        current_frame = self.preprocess_frame(batch_for_detection[0])
                        last_frame = self.preprocess_frame(self.last_frame)
                        
                        # Detect motion
                        motion_boxes = self.detect_motion(current_frame, last_frame)
                        
                        # Detect and match features
                        self.current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                        self.last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
                        feature_matches = self.match_features(
                            self.detect_features(current_frame),
                            self.last_keypoints
                        )
                    
                    # Multi-scale detection
                    all_detections = []
                    for scale, weight in zip(SCALE_FACTORS, SCALE_CONFIDENCE_WEIGHTS):
                        # Resize frame
                        scaled_frame = cv2.resize(batch_for_detection[0], 
                                                (int(FRAME_WIDTH * scale), 
                                                 int(FRAME_HEIGHT * scale)))
                        
                        # Run detection
                        results = self.object_model(
                            scaled_frame,
                            conf=DETECTION_CONFIDENCE,
                            iou=0.45,
                            agnostic_nms=True
                        )
                        
                        # Scale detections back to original size
                        for result in results:
                            for box in result.boxes.data:
                                x1, y1, x2, y2, conf, cls_id = box.tolist()
                                # Scale coordinates back
                                x1, y1 = x1/scale, y1/scale
                                x2, y2 = x2/scale, y2/scale
                                # Apply scale weight to confidence
                                conf = conf * weight
                                all_detections.append((x1, y1, x2, y2, conf, int(cls_id)))
                    
                    # Filter and process detections
                    boxes = []
                    for det in all_detections:
                        if len(det) >= 6:  # Ensure det has all required elements
                            x1, y1, x2, y2, conf, cls_id = det
                            
                            # Calculate box area
                            box_area = (x2 - x1) * (y2 - y1)
                            
                            # Filter based on size
                            if MIN_BOX_AREA <= box_area <= MAX_BOX_AREA:
                                # Boost confidence if box overlaps with motion region
                                if motion_boxes:
                                    for motion_box in motion_boxes:
                                        iou = self.calculate_iou((x1, y1, x2, y2), motion_box)
                                        if iou > 0.1:
                                            conf = min(conf * MOTION_CONFIDENCE_BOOST, 1.0)
                                
                                # Boost confidence if box contains feature matches
                                if feature_matches:
                                    good_old, good_new = feature_matches
                                    box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                                    box_size = max(x2 - x1, y2 - y1)
                                    
                                    # Count features inside box
                                    features_in_box = 0
                                    for pt in good_new:
                                        if (x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2):
                                            features_in_box += 1
                                    
                                    if features_in_box > 0:
                                        conf = min(conf * (1 + features_in_box/10), 1.0)
                                
                                boxes.append((x1, y1, x2, y2, conf, int(cls_id)))
                    
                    detection_results[0] = boxes
                    
                    # Update detection history and tracking
                    if len(batch_frames) > 0:
                        self.detection_history.append(boxes)
                        self.last_frame = batch_for_detection[0].copy()
                        self.last_keypoints = self.detect_features(batch_for_detection[0])
                        
                        # Update tracks with new detections
                        if boxes:  # Only update tracks if we have detections
                            self.update_tracks(boxes, batch_for_detection[0])
                            self.predict_track_positions()

                def pose_worker():
                    results = self.pose_model(batch_for_pose)
                    for i, result in enumerate(results):
                        if result.keypoints is not None:
                            keypoints = result.keypoints.data[0].tolist()
                            pose_results[i] = [(x, y, conf) for x, y, conf in keypoints]
                        else:
                            pose_results[i] = []

                # Create and run threads
                t_detect = threading.Thread(target=detection_worker)
                t_pose = threading.Thread(target=pose_worker)
                t_detect.start()
                t_pose.start()
                t_detect.join()
                t_pose.join()

                # Apply temporal smoothing with tracking consideration
                if detection_results[0] and len(self.detection_history) == TEMPORAL_SMOOTHING:
                    valid_tracks = [t for t in self.active_tracks 
                                  if len(t['history']) >= MIN_TRACK_LENGTH]
                    
                    if valid_tracks:
                        tracked_boxes = []
                        for track in valid_tracks:
                            if 'predicted_bbox' in track:
                                x1, y1, x2, y2 = track['predicted_bbox']
                                tracked_boxes.append((x1, y1, x2, y2, track['confidence'], track['class_id']))
                        
                        detection_results[0] = tracked_boxes

                # Post-process and annotate frames
                for i in range(len(batch_frames)):
                    frame_anno = self.annotate_frame(
                        batch_frames[i],
                        detection_results[i],
                        pose_results[i]
                    )
                    self.processed_queue.put(frame_anno)
                    self.last_valid_frame = frame_anno.copy()

            except Exception as e:
                print("[Pipeline] Exception:", e)
                time.sleep(0.1)

    def annotate_frame(self, frame, detections, keypoints):
        """Draw bounding boxes and keypoints on the frame"""
        annotated = frame.copy()

        # Calculate scaling factors for coordinate conversion
        y_scale = RS_HEIGHT / FRAME_HEIGHT  # 480/640 = 0.75

        # Draw tracked objects
        for track in self.active_tracks:
            # Only show tracks that have been recently updated (age <= 2)
            if len(track['history']) >= MIN_TRACK_LENGTH and track['confidence'] >= MIN_TRACK_CONFIDENCE and track['age'] <= 2:
                x1, y1, x2, y2 = track['bbox']
                conf = track['confidence']
                
                # Scale y coordinates back to original aspect ratio
                y1 = y1 * y_scale
                y2 = y2 * y_scale
                
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, frame.shape[1]-1))
                y1 = max(0, min(y1, frame.shape[0]-1))
                x2 = max(0, min(x2, frame.shape[1]-1))
                y2 = max(0, min(y2, frame.shape[0]-1))
                
                # Draw bounding box in red for tourniquet
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"Tourniquet {conf:.2f} ID:{track['id']}"
                cv2.putText(annotated, label, (x1, max(y1-5, 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Draw track history
                if len(track['history']) > 1:
                    points = []
                    for hist_box in track['history']:
                        hx1, hy1, hx2, hy2 = hist_box
                        hy1 = hy1 * y_scale
                        hy2 = hy2 * y_scale
                        center_x = int((hx1 + hx2) / 2)
                        center_y = int((hy1 + hy2) / 2)
                        points.append((center_x, center_y))
                    
                    # Draw track path
                    for i in range(1, len(points)):
                        cv2.line(annotated, points[i-1], points[i], (0, 255, 0), 1)

        # Draw keypoints for people
        if keypoints and len(keypoints) >= 17:
            for kx, ky, kconf in keypoints:
                if kconf > 0.2:
                    ky = ky * y_scale
                    kx, ky = int(kx), int(ky)
                    kx = max(0, min(kx, frame.shape[1]-1))
                    ky = max(0, min(ky, frame.shape[0]-1))
                    cv2.circle(annotated, (kx, ky), 4, (255, 0, 0), -1)
                    cv2.circle(annotated, (kx, ky), 2, (255, 255, 255), -1)

        return annotated

    def start_recording(self):
        """Start recording video"""
        if not self.is_recording:
            # Clear recording buffer
            self.recording_buffer.clear()
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_file, fourcc, OUTPUT_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
            
            if not self.video_writer.isOpened():
                print(f"[Video] Failed to open video writer for {output_file}")
                return
            
            self.is_recording = True
            print(f"[Video] Started recording to {output_file}")

    def stop_recording(self):
        """Stop recording video"""
        if self.is_recording and self.video_writer:
            # Write all frames from recording buffer
            for frame in self.recording_buffer:
                self.video_writer.write(frame)
            
            self.video_writer.release()
            self.video_writer = None
            self.is_recording = False
            print("[Video] Stopped recording")

    def export_data(self):
        """Export data (excluding video)"""
        try:
            # TODO: Add other export functionality here (audio, transcript, etc.)
            print("[Export] Exporting data...")
        except Exception as e:
            print(f"[Export] Error during export: {e}")

    def save_debug_video(self):
        """Save debug video if enabled and frames are available"""
        if SAVE_DEBUG_VIDEO and self.recording_buffer:
            try:
                # Generate timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(OUTPUT_DIR, f"debug_video_{timestamp}.mp4")
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file, fourcc, OUTPUT_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                
                if video_writer.isOpened():
                    # Write all frames from the recording buffer
                    for frame in self.recording_buffer:
                        video_writer.write(frame)
                    video_writer.release()
                    print(f"[Debug] Saved debug video to {output_file}")
                else:
                    print(f"[Debug] Failed to open video writer for {output_file}")
            except Exception as e:
                print(f"[Debug] Error saving debug video: {e}")

    def update_video(self):
        """Update the video display with processed frames"""
        try:
            # Get the current size of the video label
            width = self.video_label.winfo_width()
            height = self.video_label.winfo_height()
            
            # Ensure minimum size
            if width < 1 or height < 1:
                width = 640
                height = 480

            # Get frame from processed queue if available
            if not self.processed_queue.empty():
                frame = self.processed_queue.get()
                self.display_buffer.append(frame)
                
                # Add frame to recording buffer if debug video saving is enabled
                if SAVE_DEBUG_VIDEO:
                    self.recording_buffer.append(frame.copy())
            else:
                # If no processed frame, use the last valid frame
                frame = self.last_valid_frame.copy()

            # Resize frame to match label size
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height))
            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        except Exception as e:
            print(f"[Display] Error updating video: {e}")
        
        # Schedule next update
        self.root.after(10, self.update_video)

    def start_pipeline(self):
        """Start the video pipeline threads"""
        self.stop_event.clear()
        self.pipeline_running = True
        
        # Clear queues and buffers
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.processed_queue.empty():
            try:
                self.processed_queue.get_nowait()
            except queue.Empty:
                break
        self.display_buffer.clear()
        if SAVE_DEBUG_VIDEO:
            self.recording_buffer.clear()  # Clear recording buffer too
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Start pipeline thread
        self.pipeline_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.pipeline_thread.start()
        
        print("Starting video pipeline...")

    def stop_pipeline(self):
        """Stop the video pipeline threads"""
        self.stop_event.set()
        self.pipeline_running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=3)
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=3)
        
        # Reset to blank frame
        self.last_valid_frame = self.blank_frame.copy()
        
        print("Stopping video pipeline...")

    def on_closing(self):
        """Clean up resources when closing the application"""
        if self.pipeline_running:
            self.stop_pipeline()
        
        # Save debug video if enabled
        if SAVE_DEBUG_VIDEO:
            self.save_debug_video()
        
        self.root.destroy()

    def toggle_transcription(self):
        if not self.pipeline_running:
            self.start_pipeline()
            self.start_button.config(text="Stop Demo")
        else:
            self.stop_pipeline()
            self.start_button.config(text="Start Demo")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()