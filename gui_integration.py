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
        self.recording_buffer = deque(maxlen=RECORDING_BUFFER_SIZE)  # Buffer for recording
        self.last_valid_frame = None
        
        # Initialize video writer
        self.video_writer = None
        self.is_recording = False
        
        # Load YOLO models
        self.object_model = YOLO("yolo11n.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")

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

    def process_frames(self):
        """Process frames through YOLO models and annotate them"""
        while not self.stop_event.is_set():
            try:
                # Collect frames to form a batch
                batch_frames = []
                for _ in range(BATCH_SIZE):
                    try:
                        frame = self.frame_queue.get(timeout=1.0)  # Add timeout to prevent blocking indefinitely
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
                    results = self.object_model(batch_for_detection)
                    for i, result in enumerate(results):
                        boxes = []
                        for box in result.boxes.data:
                            x1, y1, x2, y2, conf, cls_id = box.tolist()
                            boxes.append((x1, y1, x2, y2, conf, int(cls_id)))
                        detection_results[i] = boxes

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

                # Post-process and annotate frames
                for i in range(len(batch_frames)):
                    frame_anno = self.annotate_frame(
                        batch_frames[i],
                        detection_results[i],
                        pose_results[i]
                    )
                    self.processed_queue.put(frame_anno)
                    # Update last valid frame
                    self.last_valid_frame = frame_anno.copy()

            except Exception as e:
                print("[Pipeline] Exception:", e)
                time.sleep(0.1)

    def annotate_frame(self, frame, detections, keypoints):
        """Draw bounding boxes and keypoints on the frame"""
        annotated = frame.copy()

        # Draw bounding boxes
        if detections:
            for box in detections:
                x1, y1, x2, y2, conf, cls_id = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure coordinates are within frame bounds
                x1 = max(0, min(x1, frame.shape[1]-1))
                y1 = max(0, min(y1, frame.shape[0]-1))
                x2 = max(0, min(x2, frame.shape[1]-1))
                y2 = max(0, min(y2, frame.shape[0]-1))
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(annotated, label, (x1, max(y1-5, 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw keypoints
        if keypoints and len(keypoints) >= 17:
            for kx, ky, kconf in keypoints:
                if kconf > 0.2:
                    kx, ky = int(kx), int(ky)
                    
                    # Ensure coordinates are within frame bounds
                    kx = max(0, min(kx, frame.shape[1]-1))
                    ky = max(0, min(ky, frame.shape[0]-1))
                    
                    # Draw keypoint circle
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