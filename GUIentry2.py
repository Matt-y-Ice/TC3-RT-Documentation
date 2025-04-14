import cv2
import tkinter as tk
from tkinter import Text, Label, Button, Frame
from PIL import Image, ImageTk

import threading
import queue
import time
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline

from pydub import AudioSegment
import os

# Import the feature extractor's process_transcript function.
# Ensure that feature_extractor.py is in the same directory.
from context_comprehension import process_transcript

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("CSC-490 Live Demo")

        # -----------------------------
        # 1) SETUP MAIN GUI COMPONENTS
        # -----------------------------
        self.main_frame = Frame(root, bg="#b0bec5", padx=10, pady=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Webcam feed
        self.video_label = Label(self.main_frame, bg="white", width=800, height=600)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, rowspan=3)

        # Right panel
        self.right_panel = Frame(self.main_frame, bg="#b0bec5")
        self.right_panel.grid(row=0, column=1, padx=10, pady=10, sticky="n")

        # Text area for live transcript
        self.label1 = Label(self.right_panel, text="Live Transcript", font=("Arial", 12, "bold"), bg="white")
        self.label1.pack(fill=tk.X, pady=(0, 5))
        self.text_area1 = Text(self.right_panel, width=100, height=19, state='disabled')
        self.text_area1.pack(fill=tk.X, pady=(0, 10))

        # Text area for model detections (feature extraction output)
        self.label2 = Label(self.right_panel, text="Model Detections", font=("Arial", 12, "bold"), bg="white")
        self.label2.pack(fill=tk.X, pady=(0, 5))
        self.text_area2 = Text(self.right_panel, width=100, height=10, state='disabled')
        self.text_area2.pack(fill=tk.X, pady=(0, 10))

        # Button panel
        self.button_panel = Frame(self.main_frame, bg="#b0bec5")
        self.button_panel.grid(row=2, column=1, pady=10, sticky="s")

        # Start / Stop Demo Button
        self.start_button = Button(self.button_panel, text="Start Demo", bg="#7749F8", fg="white",
                                   command=self.toggle_transcription)
        self.start_button.pack(side=tk.LEFT, padx=5)

        # Export Data button will trigger feature extraction on the live transcript
        self.export_button = Button(self.button_panel, text="Export Data", bg="#7749F8", fg="white",
                                    command=self.context_extraction)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # Setup video capture
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.update_video()

        # -----------------------------
        # 2) SETUP AUDIO + TRANSCRIPTION
        # -----------------------------
        self.SAMPLE_RATE = 16000
        self.CHANNELS = 1

        # Parameters for speech segmentation
        self.AUDIO_CHUNK_DURATION = 0.5  # seconds per small chunk
        self.SILENCE_DURATION_THRESHOLD = 1.0  # seconds of silence to mark end of utterance
        self.MAX_UTTERANCE_DURATION = 10.0  # maximum seconds to accumulate before forcing transcription

        # Adaptive threshold parameters:
        self.default_threshold = -35.0    # lower bound for effective threshold
        self.ambient_noise_level = -60.0  # initial estimate of ambient noise (in dBFS)
        self.adaptive_margin = 5.0        # margin above ambient noise level
        self.ambient_alpha = 0.1          # smoothing factor for ambient noise update

        # Queues for audio + transcriptions
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()

        # File for logging transcript (YYYYMMDD_Transcript.txt)
        self.transcript_filename = time.strftime("%Y%m%d_%H%M%S_Transcript.txt")
        if not os.path.exists(self.transcript_filename):
            with open(self.transcript_filename, "w", encoding="utf-8") as f:
                f.write(f"Live Transcription for {time.strftime('%Y-%m-%d', time.localtime())}\n")
                f.write("-" * 50 + "\n")

        # Flags & threads
        self.stop_event = threading.Event()
        self.record_thread = None
        self.process_thread = None
        self.transcription_running = False  # Toggle flag

        # Load ASR pipeline once
        fine_tuned_path = r"C:\Users\joshl\Documents\tayloraudio2\CSC-490\CSC-490-main\whisper-small-en-finetuned"  # Adjust if needed
        device = 0 if torch.cuda.is_available() else -1
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=fine_tuned_path,
            device=device
        )

        # On app close, release resources
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Center the window and lock its size
        self.center_window_and_lock()

    def center_window_and_lock(self):
        self.root.update_idletasks()
        width  = self.root.winfo_reqwidth()
        height = self.root.winfo_reqheight()

        screen_width  = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        self.root.geometry(f"+{x}+{y}")
        self.root.resizable(False, False)

    # -----------------------------
    # 3) WEBCAM LOOP
    # -----------------------------
    def update_video(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 600))
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.root.after(10, self.update_video)

    # -----------------------------
    # 4) TEXT AREA UTILS
    # -----------------------------
    def add_text_to_text_area1(self, text):
        self.text_area1.config(state='normal')
        self.text_area1.insert(tk.END, text + "\n")
        self.text_area1.see(tk.END)
        self.text_area1.config(state='disabled')

    def add_text_to_text_area2(self, text):
        self.text_area2.config(state='normal')
        self.text_area2.insert(tk.END, text + "\n")
        self.text_area2.see(tk.END)
        self.text_area2.config(state='disabled')

    # -----------------------------
    # 5) BACKGROUND AUDIO THREADS
    # -----------------------------
    def record_audio(self):
        """ Continuously capture audio from mic and push frames to audio_queue. """
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, callback=self.audio_callback):
            while not self.stop_event.is_set():
                sd.sleep(100)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """
        Pull small audio chunks from the audio_queue, accumulate speech segments
        based on an adaptive threshold, and once silence or a max utterance duration
        is detected, send the segment to the ASR model for transcription.
        """
        frames_per_chunk = int(self.SAMPLE_RATE * self.AUDIO_CHUNK_DURATION)
        max_frames = int(self.SAMPLE_RATE * self.MAX_UTTERANCE_DURATION)

        current_segment = []  # holds numpy arrays for current speech
        silence_duration = 0.0  # accumulated silence duration

        while not self.stop_event.is_set():
            frames = []
            frames_collected = 0

            while frames_collected < frames_per_chunk and not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get(timeout=0.1)
                    frames.append(data)
                    frames_collected += data.shape[0]
                except queue.Empty:
                    continue

            if self.stop_event.is_set():
                break

            audio_chunk = np.concatenate(frames, axis=0).flatten()
            segment_pd = self.float32_to_pydub(audio_chunk, self.SAMPLE_RATE)
            dbfs_val = segment_pd.dBFS

            # Compute effective threshold: do not drop below the default.
            effective_threshold = max(self.default_threshold, self.ambient_noise_level + self.adaptive_margin)
            
            dbfs_val = round(dbfs_val, 2)
            effective_threshold = round(effective_threshold, 2)
            print("Microphone dBFS: {:<7} | Activation threshold: {:<7}".format(dbfs_val, effective_threshold))

            if dbfs_val > effective_threshold:
                # Chunk is considered speech.
                current_segment.append(audio_chunk)
                silence_duration = 0.0
            else:
                # Update ambient noise estimate when no speech is detected.
                self.ambient_noise_level = (self.ambient_alpha * dbfs_val) + ((1 - self.ambient_alpha) * self.ambient_noise_level)
                silence_duration += self.AUDIO_CHUNK_DURATION

            # Process accumulated speech if silence or maximum length is reached.
            if current_segment:
                total_frames = sum(len(chunk) for chunk in current_segment)
                if silence_duration >= self.SILENCE_DURATION_THRESHOLD or total_frames >= max_frames:
                    utterance_audio = np.concatenate(current_segment)
                    # Calculate duration in seconds.
                    duration = len(utterance_audio) / self.SAMPLE_RATE
                    chunk_time_str = time.strftime("%H:%M:%S", time.localtime())
                    try:
                        result = self.asr(utterance_audio)
                    except Exception as e:
                        print(f"Error during transcription: {e}")
                        current_segment = []
                        silence_duration = 0.0
                        continue

                    raw_text = result.get("text", "")
                    cleaned_text = self.sanitize_english_text(raw_text).strip()

                    if cleaned_text:
                        # Append duration to the transcript line.
                        transcript_line = f"[{chunk_time_str}   ~{duration:.2f}s] {cleaned_text}"
                        print("Transcript:", transcript_line)
                        with open(self.transcript_filename, "a", encoding="utf-8") as f:
                            f.write(transcript_line + "\n")
                        self.transcription_queue.put(transcript_line)
                    else:
                        print("Transcript: (empty after cleaning)")

                    current_segment = []
                    silence_duration = 0.0

    # -----------------------------
    # 6) HELPER: float32 -> pydub AudioSegment
    # -----------------------------
    def float32_to_pydub(self, float_array, sample_rate=16000):
        float_array = np.clip(float_array, -1.0, 1.0)
        int16_array = (float_array * 32767).astype(np.int16)
        segment = AudioSegment(
            data=int16_array.tobytes(),
            sample_width=2,  # 16 bits = 2 bytes
            frame_rate=sample_rate,
            channels=1
        )
        return segment

    # -----------------------------
    # 7) POLLING THE TRANSCRIPTION QUEUE
    # -----------------------------
    def poll_transcriptions(self):
        while not self.transcription_queue.empty():
            line = self.transcription_queue.get()
            self.add_text_to_text_area1(line)
            self.context_extraction()
        if self.transcription_running and not self.stop_event.is_set():
            self.root.after(200, self.poll_transcriptions)

    # -----------------------------
    # 8) TOGGLE / START / STOP TRANSCRIPTION
    # -----------------------------
    def toggle_transcription(self):
        if not self.transcription_running:
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
            with self.transcription_queue.mutex:
                self.transcription_queue.queue.clear()
            self.stop_event.clear()
            self.transcription_running = True
            self.start_transcription()
            self.start_button.config(text="Stop Demo")
        else:
            self.stop_transcription()
            self.start_button.config(text="Start Demo")
            self.transcription_running = False

    def start_transcription(self):
        self.text_area1.config(state='normal')
        self.text_area1.delete("1.0", tk.END)
        self.text_area1.config(state='disabled')

        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.record_thread.start()
        self.process_thread.start()
        self.poll_transcriptions()

    def stop_transcription(self):
        self.stop_event.set()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        if self.record_thread is not None:
            self.record_thread.join(timeout=2)
        if self.process_thread is not None:
            self.process_thread.join(timeout=2)

    def on_closing(self):
        if self.transcription_running:
            self.stop_transcription()
        self.cap.release()
        self.root.destroy()

    # -----------------------------
    # 9) UTIL: SANITIZE TEXT
    # -----------------------------
    def sanitize_english_text(self, text):
        return re.sub(r"[^A-Za-z0-9\s,.!?'\"]", "", text)

    # -----------------------------
    # 10) EXPORT DATA BUTTON HANDLER
    # -----------------------------
    def context_extraction(self):
        """
        Triggered by the "Export Data" button.
        Retrieves the full transcript from the Live Transcript area and processes it via process_transcript.
        For each injury detection, it prints a new line with a fresh timestamp, pairing the injury with the tourniquet
        at the same sequential index. If the tourniquet’s location data doesn't match the injury’s (laterality or extremity),
        the injury's location is used instead. Additionally, any "forearm" is replaced with "arm".
        """
        # Retrieve transcript text from the Live Transcript area.
        transcript_text = self.text_area1.get("1.0", tk.END).strip()

        # Process the transcript to extract injuries and tourniquet detections.
        extracted_results = process_transcript(transcript_text)

        # Clear the Model Detections text area.
        self.text_area2.config(state='normal')
        self.text_area2.delete("1.0", tk.END)

        # Retrieve the lists of detections.
        injuries = extracted_results.get("injuries", [])
        tourniquets = extracted_results.get("tourniquets", [])

        output_lines = []

        # For each injury, generate a new line with its own fresh timestamp.
        # The injury at index n is paired with the tourniquet at index n (if it exists).
        for idx in range(len(injuries)):
            injury = injuries[idx]
            # Generate a fresh timestamp for this injury detection.
            time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Replace 'forearm' with 'arm' in the injury's location phrase.
            injury_location = injury["location_phrase"].replace("forearm", "arm")
            line = f"AUDIO: {time_stamp} Injury #{idx + 1} detected at {injury_location}"

            # Check if there is a corresponding tourniquet for this injury.
            if idx < len(tourniquets):
                tq = tourniquets[idx]
                # Get tourniquet location and replace 'forearm' with 'arm'.
                tq_location = tq["location_phrase"].replace("forearm", "arm")
                # If the tourniquet's laterality/extremity don't match the injury, override with injury's location.
                if tq.get("laterality") != injury.get("laterality") or tq.get("extremity") != injury.get("extremity"):
                    tq_location = injury_location

                # Append the tourniquet information based on its status.
                if tq["status"] == "confirmed tourniquet placement":
                    line += f" ; Tourniquet #{idx + 1} has been APPLIED at {tq_location}"
                else:
                    line += f" ; Tourniquet #{idx + 1} is SUGGESTED at {tq_location}"

            output_lines.append(line)

        # If no injuries were detected, optionally note that.
        if not output_lines:
            output_lines.append("AUDIO: No new detections found.")

        # Insert each output line into the Model Detections text area.
        for line in output_lines:
            self.text_area2.insert(tk.END, line + "\n")

        self.text_area2.config(state='disabled')







if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
