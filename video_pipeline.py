import os
import math
import time
import logging
import traceback
from datetime import datetime
from collections import deque
import numpy as np
import tkinter as tk
from logging.handlers import RotatingFileHandler
import sys
import threading

# VIDEO CONFIGURATION CONSTANTS
OUTPUT_DIR = "output"
DEBUG_LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
DEBUG_LOG_MAX_SIZE = 1024 * 1024
DEBUG_LOG_BACKUP_COUNT = 5
TOURNIQUET_STABILITY_THRESHOLD = 50
TOURNIQUET_STABILITY_FRAMES = 30
TOURNIQUET_STABILITY_PERCENTAGE = 0.8

if not os.path.exists(DEBUG_LOG_DIR):
    os.makedirs(DEBUG_LOG_DIR, exist_ok=True)

class TourniquetObserver:
    """
    Observer class that monitors tourniquet tracks to detect when they've been applied.

    A tourniquet is considered "applied" when its center position remains stable
    (within TOURNIQUET_STABILITY_THRESHOLD pixels) for TOURNIQUET_STABILITY_FRAMES consecutive frames.

    This class runs in a separate thread and continuously monitors the active tracks
    to detect when a tourniquet has been properly applied based on position stability.
    """

    def __init__(self, app, observer_stop_event):
        """
        Initialize the TourniquetObserver.

        Args:
            app: The main application instance
            observer_stop_event: Threading event to signal when to stop the observer
        """
        self.app = app
        self.observer_stop_event = observer_stop_event
        self.track_centers = {}  # Dictionary to store center positions for each track
        self.track_stability = (
            {}
        )  # Dictionary to store stability counters for each track
        self.applied_tracks = set()  # Set of track IDs that have been marked as applied
        self.track_last_seen = {}  # Dictionary to store when each track was last seen
        self.track_history = (
            {}
        )  # Dictionary to store historical positions for each track

        # Setup logging
        self.setup_logging()

        # Start the observer thread
        self.observer_thread = threading.Thread(target=self.run, daemon=True)
        self.observer_thread.start()

    def setup_logging(self):
        """
        Setup logging to a rotating file with enhanced debugging information.

        Configures both file and console handlers with custom formatting to provide
        detailed logging information for debugging and monitoring.
        """
        log_file = os.path.join(DEBUG_LOG_DIR, "debug.log")

        try:
            # Get module-level logger
            logger = logging.getLogger(__name__)

            # Configure logging with a custom formatter that includes more details
            formatter = logging.Formatter(
                "%(asctime)s.%(msecs)03d - %(levelname)s - [%(threadName)s] - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            # Create a rotating file handler with size limit and backup count
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=DEBUG_LOG_MAX_SIZE,
                backupCount=DEBUG_LOG_BACKUP_COUNT,
                mode="a",  # Append mode
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)

            # Create a console handler for immediate feedback
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)

            # Remove any existing handlers to avoid duplicates
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add our handlers
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.setLevel(logging.DEBUG)

            # Log initialization information
            logger.info(
                "Tourniquet Observer initialized with enhanced logging\n"
                f"Debug log file: {log_file}\n"
                f"Max log size: {DEBUG_LOG_MAX_SIZE/1024/1024:.1f}MB\n"
                f"Backup count: {DEBUG_LOG_BACKUP_COUNT}\n"
                f"Stability threshold: {TOURNIQUET_STABILITY_THRESHOLD} pixels\n"
                f"Stability frames: {TOURNIQUET_STABILITY_FRAMES}\n"
                f"Stability percentage: {TOURNIQUET_STABILITY_PERCENTAGE*100}%"
            )

            # Log system information
            logger.debug(
                "System Information:\n"
                f"Python version: {sys.version}\n"
                f"OpenCV version: {cv2.__version__}\n"
                f"Working directory: {os.getcwd()}\n"
                f"Log directory: {DEBUG_LOG_DIR}\n"
                f"Log file permissions: {oct(os.stat(log_file).st_mode)[-3:]}"
            )

        except Exception as e:
            print(f"Error setting up logging: {e}")
            print(traceback.format_exc())

    def calculate_center(self, bbox):
        """
        Calculate the center point of a bounding box.

        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)

        Returns:
            tuple: (x, y) coordinates of the center point
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def calculate_distance(self, point1, point2):
        """
        Calculate Euclidean distance between two points.

        Args:
            point1: First point (x1, y1)
            point2: Second point (x2, y2)

        Returns:
            float: Euclidean distance between the points
        """
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def calculate_average_center(self, centers):
        """
        Calculate the average center position from a list of centers.

        Args:
            centers: List of (x, y) center points

        Returns:
            tuple: Average (x, y) center position or None if list is empty
        """
        if not centers:
            return None

        x_sum = sum(center[0] for center in centers)
        y_sum = sum(center[1] for center in centers)
        count = len(centers)

        return (x_sum / count, y_sum / count)

    def is_stable(self, track_id):
        """
        Check if a track is stable based on its center positions.

        A track is considered stable if its center position remains within
        TOURNIQUET_STABILITY_THRESHOLD pixels of the average center for at least
        TOURNIQUET_STABILITY_PERCENTAGE of the frames in the history.

        Args:
            track_id: ID of the track to check

        Returns:
            bool: True if the track is stable, False otherwise
        """
        logger = logging.getLogger(__name__)

        if track_id not in self.track_centers:
            return False

        centers = self.track_centers[track_id]
        if len(centers) < TOURNIQUET_STABILITY_FRAMES:
            return False

        # Calculate average center position
        avg_center = self.calculate_average_center(centers)
        if not avg_center:
            return False

        # Count how many centers are within the threshold distance of the average
        stable_count = 0
        for center in centers:
            if (
                self.calculate_distance(center, avg_center)
                <= TOURNIQUET_STABILITY_THRESHOLD
            ):
                stable_count += 1

        # Calculate the percentage of stable frames
        stability_percentage = stable_count / len(centers)

        # Log stability information for debugging
        logger.debug(
            f"Track {track_id} stability check:\n"
            f"  Total frames: {len(centers)}\n"
            f"  Stable frames: {stable_count}\n"
            f"  Stability percentage: {stability_percentage:.2%}\n"
            f"  Average center: {avg_center}\n"
            f"  Max deviation: {max(self.calculate_distance(c, avg_center) for c in centers):.2f}px"
        )

        return stability_percentage >= TOURNIQUET_STABILITY_PERCENTAGE

    def update_model_detections(self, message):
        """Update the model detections text area with a message"""
        logger = logging.getLogger(__name__)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Extract just the first line for GUI display (removes position, confidence, and bbox info)
        gui_message = message.split("\n")[0]
        formatted_message = f"VIDEO: {timestamp} {gui_message}\n"

        # Update the text area in the main thread
        # self.app.root.after(0, lambda: self._update_text_area(formatted_message))
        self.app.post_detection(formatted_message)

        # Log full detailed message to file
        logger.info(message)

    def _update_text_area(self, message):
        """Helper method to update the text area (must be called from main thread)"""
        logger = logging.getLogger(__name__)
        try:
            self.app.text_area2.config(state="normal")
            self.app.text_area2.insert(tk.END, message)
            self.app.text_area2.see(tk.END)  # Scroll to the end
            self.app.text_area2.config(state="disabled")
        except Exception as e:
            logger.error(f"Error updating text area: {e}")

    def log_track_info(self, track_id, center, stability_count):
        """Log detailed information about a track for debugging"""
        logger = logging.getLogger(__name__)

        if track_id not in self.track_centers:
            return

        centers = self.track_centers[track_id]
        if not centers:
            return

        avg_center = self.calculate_average_center(centers)
        if not avg_center:
            return

        # Calculate max deviation from average
        max_deviation = 0
        deviations = []
        for c in centers:
            dist = self.calculate_distance(c, avg_center)
            max_deviation = max(max_deviation, dist)
            deviations.append(dist)

        # Get the track from active_tracks
        track = next((t for t in self.app.active_tracks if t["id"] == track_id), None)
        if track:
            # Calculate stability metrics
            avg_deviation = sum(deviations) / len(deviations) if deviations else 0
            stable_frames = sum(
                1 for d in deviations if d <= TOURNIQUET_STABILITY_THRESHOLD
            )
            stability_percentage = stable_frames / len(deviations) if deviations else 0

            # Log detailed information including bbox and confidence
            logger.debug(
                f"Track {track_id} Details:\n"
                f"  Current Center: {center}\n"
                f"  Average Center: {avg_center}\n"
                f"  Current Deviation: {self.calculate_distance(center, avg_center):.2f}px\n"
                f"  Average Deviation: {avg_deviation:.2f}px\n"
                f"  Max Deviation: {max_deviation:.2f}px\n"
                f"  Stability Count: {stability_count}\n"
                f"  Stable Frames: {stable_frames}/{len(deviations)} ({stability_percentage:.1%})\n"
                f"  Centers Count: {len(centers)}\n"
                f"  Bounding Box: {track['bbox']}\n"
                f"  Confidence: {track['confidence']:.3f}\n"
                f"  Age: {track['age']}\n"
                f"  Class ID: {track['class_id']}\n"
                f"  Last Seen: {self.track_last_seen.get(track_id, 'Never')}\n"
                f"  History Length: {len(self.track_history.get(track_id, []))}\n"
                f"  Time since last update: {time.time() - self.track_last_seen.get(track_id, time.time()):.2f}s"
            )

    def run(self):
        """Main observer loop that runs in a background thread"""
        logger = logging.getLogger(__name__)
        frame_count = 0
        last_log_time = time.time()

        while not self.observer_stop_event.is_set():
            try:
                # Get a copy of the active tracks to avoid race conditions
                active_tracks = self.app.active_tracks.copy()
                current_time = time.time()

                # Process each track
                for track in active_tracks:
                    track_id = track["id"]

                    # Skip tracks that have already been marked as applied
                    if track_id in self.applied_tracks:
                        continue

                    # Initialize track data if not already present
                    if track_id not in self.track_centers:
                        self.track_centers[track_id] = deque(
                            maxlen=TOURNIQUET_STABILITY_FRAMES
                        )
                        self.track_stability[track_id] = 0
                        self.track_last_seen[track_id] = current_time
                        self.track_history[track_id] = []
                        logger.info(
                            f"New track detected: {track_id} with bbox {track['bbox']} and confidence {track['confidence']:.3f}"
                        )

                    # Calculate center of current bounding box
                    center = self.calculate_center(track["bbox"])

                    # Add center to track history
                    self.track_centers[track_id].append(center)
                    self.track_history[track_id].append((center, current_time))
                    self.track_last_seen[track_id] = current_time

                    # Log track info periodically
                    self.log_track_info(
                        track_id, center, self.track_stability[track_id]
                    )

                    # Check if track is stable
                    if self.is_stable(track_id):
                        # Mark track as applied
                        self.applied_tracks.add(track_id)

                        # Log the event with detailed information
                        self.update_model_detections(
                            f"Tourniquet {track_id} has been APPLIED\n"
                            f"  Final Position: {center}\n"
                            f"  Confidence: {track['confidence']:.3f}\n"
                            f"  Bounding Box: {track['bbox']}"
                        )
                        logger.info(
                            f"Tourniquet {track_id} has been APPLIED\n"
                            f"  Final Position: {center}\n"
                            f"  Confidence: {track['confidence']:.3f}\n"
                            f"  Bounding Box: {track['bbox']}"
                        )

                        # Stop only the observer thread, not the entire pipeline
                        self.observer_stop_event.set()
                        break

                # Check for temporarily lost tracks
                for track_id in list(self.track_centers.keys()):
                    if track_id not in {t["id"] for t in active_tracks}:
                        # Track is not in current active tracks
                        last_seen = self.track_last_seen.get(track_id, 0)
                        time_since_last_seen = current_time - last_seen

                        if time_since_last_seen < 2.0:  # Within 2 seconds
                            # Get last known position
                            if (
                                track_id in self.track_history
                                and self.track_history[track_id]
                            ):
                                last_center, _ = self.track_history[track_id][-1]

                                # Check if any current track is close to the last known position
                                for track in active_tracks:
                                    current_center = self.calculate_center(
                                        track["bbox"]
                                    )
                                    if (
                                        self.calculate_distance(
                                            current_center, last_center
                                        )
                                        <= TOURNIQUET_STABILITY_THRESHOLD
                                    ):
                                        # Found a matching track, preserve the original track ID and history
                                        if (
                                            track["id"] != track_id
                                        ):  # Only update if it's a different ID
                                            # Update the track's ID to match the original
                                            track["id"] = track_id
                                            logger.info(
                                                f"Track {track['id']} matched to existing track {track_id}"
                                            )
                                        break

                # Clean up tracks that are no longer active and haven't been seen for a while
                active_track_ids = {track["id"] for track in active_tracks}
                for track_id in list(self.track_centers.keys()):
                    if track_id not in active_track_ids:
                        last_seen = self.track_last_seen.get(track_id, 0)
                        if (
                            current_time - last_seen > 5.0
                        ):  # Remove after 5 seconds of no detection
                            del self.track_centers[track_id]
                            if track_id in self.track_stability:
                                del self.track_stability[track_id]
                            if track_id in self.track_last_seen:
                                del self.track_last_seen[track_id]
                            if track_id in self.track_history:
                                del self.track_history[track_id]
                            logger.info(f"Track {track_id} removed (no longer active)")

                # Increment frame counter
                frame_count += 1

                # Log summary statistics periodically (every 5 seconds)
                if current_time - last_log_time > 5.0:
                    logger.info(
                        f"Observer stats:\n"
                        f"  Active tracks: {len(active_tracks)}\n"
                        f"  Applied tourniquets: {len(self.applied_tracks)}\n"
                        f"  Frames processed: {frame_count}\n"
                        f"  Track details:"
                    )
                    for track in active_tracks:
                        logger.info(
                            f"    Track {track['id']}:\n"
                            f"      Position: {self.calculate_center(track['bbox'])}\n"
                            f"      Confidence: {track['confidence']:.3f}\n"
                            f"      Age: {track['age']}\n"
                            f"      Last Seen: {self.track_last_seen.get(track['id'], 'Never')}"
                        )
                    last_log_time = current_time

                # Sleep to avoid consuming too much CPU
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in tourniquet observer: {e}")
                logger.error(traceback.format_exc())
                time.sleep(0.5)  # Sleep longer on error
