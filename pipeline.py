import cv2
import threading
import queue
import time
import traceback
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# ==============================================
# Configuration
# ==============================================
BATCH_SIZE = 4           
FRAME_WIDTH, FRAME_HEIGHT = 320, 320
OUTPUT_FPS = 30          
OUTPUT_FILE = "output.mp4"

# Configure RealSense pipeline
RS_WIDTH = 640   
RS_HEIGHT = 480  
RS_FPS = 30      

# Load the actual YOLO models
object_model = YOLO("yolo11n.pt")  # Update with your model path
pose_model = YOLO("yolo11n-pose.pt")  # Update with your model path

# Define the keypoint connections for pose visualization
POSE_CONNECTIONS = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],  # arms and shoulders
    [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],    # body and face
    [2, 4], [3, 5], [4, 6], [5, 7]                                        # legs
]

# Define keypoint-joint dictionary
joint_names = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

# ==============================================
# Thread 1: Frame Capture
# ==============================================
def capture_frames(frame_queue, stop_event):
    """
    Continuously capture frames from the RealSense camera,
    resize them, and push them into frame_queue.
    """
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, RS_WIDTH, RS_HEIGHT, rs.format.bgr8, RS_FPS)

    try:
        # Start streaming
        pipeline.start(config)
        print("[Capture] RealSense camera started successfully")

        # Set a shorter timeout for frame waiting
        timeout_ms = 1000  # 1 second timeout

        while not stop_event.is_set():
            try:
                # Wait for frames with timeout
                frames = pipeline.wait_for_frames(timeout_ms)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    print("[Capture] Empty frame received. Retrying...")
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                
                # Resize frame to (320, 320) for faster YOLO inference
                frame_resized = cv2.resize(color_image, (FRAME_WIDTH, FRAME_HEIGHT))
                
                # Use try_put to avoid blocking indefinitely
                if not frame_queue.full():
                    frame_queue.put(frame_resized)

            except RuntimeError as e:
                if "Frame didn't arrive" in str(e):
                    print("[Capture] Frame timeout, attempting to recover...")
                    # Optional: try to restart the pipeline
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
                traceback.print_exc()
                time.sleep(0.1)

    except Exception as e:
        print(f"[Capture] Failed to initialize RealSense camera: {e}")
        traceback.print_exc()
    finally:
        try:
            pipeline.stop()
            print("[Capture] RealSense pipeline stopped")
        except Exception as e:
            print(f"[Capture] Error stopping pipeline: {e}")


# ==============================================
# Thread 2 (Main Pipeline in a loop): 
#   - Collect BATCH_SIZE frames
#   - Spawn detection thread
#   - Spawn pose thread
#   - Wait for both to complete
#   - Post-process and draw bounding boxes + keypoints
#   - Send annotated frames to output queue
# ==============================================
def main_pipeline(frame_queue, processed_queue, stop_event):
    while not stop_event.is_set():
        try:
            # Collect frames to form a batch
            batch_frames = []
            for _ in range(BATCH_SIZE):
                frame = frame_queue.get()
                batch_frames.append(frame)

            batch_for_detection = batch_frames[:]
            batch_for_pose = batch_frames[:]

            # Results containers
            detection_results = [None] * BATCH_SIZE
            pose_results = [None] * BATCH_SIZE
            localization_results = []
            def detection_worker():
                # Run YOLO detection
                results = object_model(batch_for_detection)
                for i, result in enumerate(results):
                    # Extract boxes in (x1, y1, x2, y2, conf, cls_id) format
                    boxes = []
                    for box in result.boxes.data:
                        x1, y1, x2, y2, conf, cls_id = box.tolist()
                        boxes.append((x1, y1, x2, y2, conf, int(cls_id)))
                    detection_results[i] = boxes

            def pose_worker():
                # Run YOLO pose estimation
                results = pose_model(batch_for_pose)
                for i, result in enumerate(results):
                    # Extract keypoints in (x, y, conf) format
                    if result.keypoints is not None:
                        keypoints = result.keypoints.data[0].tolist()  # Get first person's keypoints
                        pose_results[i] = [(x, y, conf) for x, y, conf in keypoints]
                    else:
                        pose_results[i] = []
                
            def observer_worker():
                for i in range(len(pose_results)):
                    local = get_closest_joint(detection_results[i], pose_results[i])
                    if local not in localization_results:
                        localization_results.append(local)
                print(localization_results)

            # Create and run threads
            t_detect = threading.Thread(target=detection_worker)
            t_pose = threading.Thread(target=pose_worker)
            t_observer = threading.Thread(target=observer_worker)
            t_detect.start()
            t_pose.start()
            t_detect.join()
            t_pose.join()
            t_observer.start()
            t_observer.join()

            # Post-process and annotate frames
            annotated_batch = []
            for i in range(BATCH_SIZE):
                frame_anno = annotate_frame(
                    batch_frames[i],
                    detection_results[i],
                    pose_results[i]
                )
                annotated_batch.append(frame_anno)

            # Push annotated frames to processed_queue
            for frame in annotated_batch:
                processed_queue.put(frame)

        except Exception as e:
            print("[Pipeline] Exception:", e)
            traceback.print_exc()
            time.sleep(0.1)


# ==============================================
# Annotation Function
# ==============================================
def annotate_frame(frame, detections, keypoints):
    """
    Draw bounding boxes and keypoints on the frame, including pose connections.
    """
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

    # Draw pose skeleton and keypoints
    if keypoints and len(keypoints) >= 17:  # YOLO pose has 17 keypoints
        # Draw connections first (skeleton)
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection[0] - 1, connection[1] - 1  # Adjust for 0-based indexing
            
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                # Only draw if both points have good confidence
                if start_point[2] > 0.2 and end_point[2] > 0.2:
                    start_pos = (int(start_point[0]), int(start_point[1]))
                    end_pos = (int(end_point[0]), int(end_point[1]))
                    
                    # Ensure coordinates are within bounds
                    start_pos = (
                        max(0, min(start_pos[0], frame.shape[1]-1)),
                        max(0, min(start_pos[1], frame.shape[0]-1))
                    )
                    end_pos = (
                        max(0, min(end_pos[0], frame.shape[1]-1)),
                        max(0, min(end_pos[1], frame.shape[0]-1))
                    )
                    
                    # Draw the connection line
                    cv2.line(annotated, start_pos, end_pos, (0, 255, 255), 2)

        # Draw keypoints on top
        for kx, ky, kconf in keypoints:
            if kconf > 0.2:  # Only draw keypoints with confidence > 0.2
                kx, ky = int(kx), int(ky)
                
                # Ensure coordinates are within frame bounds
                kx = max(0, min(kx, frame.shape[1]-1))
                ky = max(0, min(ky, frame.shape[0]-1))
                
                # Draw keypoint circle
                cv2.circle(annotated, (kx, ky), 4, (255, 0, 0), -1)
                cv2.circle(annotated, (kx, ky), 2, (255, 255, 255), -1)
                
                # Optionally draw confidence
                # label = f"{kconf:.2f}"
                # cv2.putText(annotated, label, (kx+5, ky),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    return annotated

# ==============================================
# Observer Function:
#   - Calculates Euclidean distance between bounding box and keypoints
#   - Returns nearest "joint" where medical device is detected
# ==============================================
def get_closest_joint(detections, keypoints):
    """
    Given a detection bounding box and a list of keypoints (tuples of at least (x, y)),
    return the joint that is closest to the box center.

    Parameters:
      detection_box: list or tuple of [x1, y1, x2, y2]
      keypoints: list of tuples, each tuple containing (x, y) coordinates. Additional elements in the tuple are ignored.

    Returns:
      A list [joint_index, (x, y), distance] for the joint closest to the detection box center.
    """
    # Unpack the detection box and compute its center
    for box in detections:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    min_distance = float("inf")
    closest_joint_index = None

    # Loop through each keypoint and compute the Euclidean distance to the box center
    for i, kp in enumerate(keypoints):
        # Each keypoint is a tuple; take the first two values as x and y.
        x, y = kp[:2]
        distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_joint_index = i


    return [cls_id, closest_joint_index]
# ==============================================
# Thread 3: Video Writer
#   - Continuously read annotated frames from processed_queue
#   - Write them to an mp4 file at specified FPS
# ==============================================
def video_writer(processed_queue, stop_event):
    """Video writer thread with improved error handling and cleanup"""
    out = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_FILE, fourcc, OUTPUT_FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        
        if not out.isOpened():
            raise RuntimeError("Failed to open video writer")

        frames_written = 0
        while not stop_event.is_set():
            try:
                frame = processed_queue.get(timeout=0.5)
                out.write(frame)
                frames_written += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VideoWriter] Error writing frame: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"[VideoWriter] Fatal error: {e}")
        traceback.print_exc()
    finally:
        if out is not None:
            try:
                out.release()
                if frames_written > 0:
                    print(f"[VideoWriter] Video saved successfully: {OUTPUT_FILE} ({frames_written} frames)")
                else:
                    print("[VideoWriter] Warning: No frames were written to the video")
            except Exception as e:
                print(f"[VideoWriter] Error during cleanup: {e}")


# ==============================================
# Main Entrypoint
# ==============================================
def main():
    # Prepare shared queues with smaller max size to prevent memory issues
    frame_queue = queue.Queue(maxsize=20)
    processed_queue = queue.Queue(maxsize=20)

    # Event to signal threads to stop
    stop_event = threading.Event()

    try:
        # Start capture thread with stop_event
        t_cap = threading.Thread(target=capture_frames, 
                               args=(frame_queue, stop_event), 
                               daemon=True)
        t_cap.start()

        # Start pipeline thread
        t_pipeline = threading.Thread(target=main_pipeline, 
                                    args=(frame_queue, processed_queue, stop_event), 
                                    daemon=True)
        t_pipeline.start()

        # Start video writer thread
        t_writer = threading.Thread(target=video_writer, 
                                  args=(processed_queue, stop_event), 
                                  daemon=True)
        t_writer.start()

        print("Press Ctrl+C or close terminal to stop...")

        while True:
            time.sleep(1)
            # Check if any thread has died unexpectedly
            if not t_cap.is_alive() or not t_pipeline.is_alive() or not t_writer.is_alive():
                print("[Main] A thread has died unexpectedly. Stopping...")
                break

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received. Stopping gracefully...")
    finally:
        # Signal threads to stop
        stop_event.set()
        
        # Wait for threads to finish with timeout
        print("[Main] Waiting for threads to finish...")
        t_cap.join(timeout=3)
        t_pipeline.join(timeout=3)
        t_writer.join(timeout=3)
        
        print("[Main] Finished.")


if __name__ == "__main__":
    main()
