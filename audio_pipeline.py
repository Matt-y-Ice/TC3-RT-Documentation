import queue
import threading
import numpy as np
import os
import re
import traceback
import tkinter as tk

import sounddevice as sd
import torch
from transformers import pipeline
from pydub import AudioSegment
from datetime import datetime

from context_comprehension import process_transcript

# AUDIO CONFIGURATION CONSTANTS
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_CHUNK_DURATION = 0.5
SILENCE_DURATION_THRESHOLD = 1.0
MAX_UTTERANCE_DURATION = 10.0
DEFAULT_THRESHOLD = -35.0
AMBIENT_NOISE_LEVEL = -60.0
ADAPTIVE_MARGIN = 5.0
AMBIENT_ALPHA = 0.1
OUTPUT_DIR = "output"
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, "transcripts")

if not os.path.exists(TRANSCRIPTS_DIR):
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)

AUDIO_AVAILABLE = True

class AudioPipeline:
    """
    Audio pipeline for capturing, processing, and transcribing speech.

    This class handles all audio-related functionality including:
    - Capturing audio from the microphone
    - Processing audio chunks to detect speech
    - Transcribing speech using a fine-tuned Whisper model
    - Writing transcriptions to the GUI and a log file
    """

    def __init__(self, gui, logger):
        """
        Initialize the audio pipeline.

        Args:
            gui: The main application instance
            logger: Logger instance for logging
        """
        self.gui = gui
        self.logger = logger

        # Audio parameters
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.audio_chunk_duration = AUDIO_CHUNK_DURATION
        self.silence_duration_threshold = SILENCE_DURATION_THRESHOLD
        self.max_utterance_duration = MAX_UTTERANCE_DURATION
        self.default_threshold = DEFAULT_THRESHOLD
        self.ambient_noise_level = AMBIENT_NOISE_LEVEL
        self.adaptive_margin = ADAPTIVE_MARGIN
        self.ambient_alpha = AMBIENT_ALPHA

        # Queues for audio and transcriptions
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()

        # File for logging transcript
        self.transcript_filename = os.path.join(
            TRANSCRIPTS_DIR,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_Transcript.txt",
        )

        # Create transcript file
        with open(self.transcript_filename, "w", encoding="utf-8") as f:
            f.write(f"Live Transcription for {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("-" * 50 + "\n")

        # Flags and threads
        self.audio_stop_event = threading.Event()
        self.record_thread = None
        self.process_thread = None
        self.audio_running = False

        # Load ASR pipeline if available
        self.asr = None
        if AUDIO_AVAILABLE:
            try:
                fine_tuned_path = "models/whisper-small-en-finetuned"
                device = 0 if torch.cuda.is_available() else -1
                self.asr = pipeline(
                    "automatic-speech-recognition", model=fine_tuned_path, device=device
                )
                self.logger.info(
                    f"ASR pipeline loaded successfully. Using device: {device}"
                )
            except Exception as e:
                self.logger.error(f"Failed to load ASR pipeline: {e}")
                self.logger.error(traceback.format_exc())

        self.posted_audio_lines = set()  # Set to track posted lines

    def start(self):
        """
        Start the audio pipeline.

        This method initializes and starts the audio recording and processing threads.
        """
        self.logger.info("Starting audio pipeline...")

        # Clear queues
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        with self.transcription_queue.mutex:
            self.transcription_queue.queue.clear()

        # Reset stop event
        self.audio_stop_event.clear()
        self.audio_running = True

        # Start threads
        self.record_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.process_thread = threading.Thread(target=self.process_audio, daemon=True)

        self.record_thread.start()
        self.process_thread.start()

        # Start polling transcriptions
        self.poll_transcriptions()

        self.logger.info("Audio pipeline started successfully")

    def stop(self):
        """
        Stop the audio pipeline.

        This method signals the audio threads to stop and waits for them to finish.
        """
        self.logger.info("Stopping audio pipeline...")

        # Set stop event
        self.audio_stop_event.set()

        # Clear queues
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        # Wait for threads to finish
        if self.record_thread is not None:
            self.record_thread.join(timeout=3)
        if self.process_thread is not None:
            self.process_thread.join(timeout=3)

        self.audio_running = False
        self.logger.info("Audio pipeline stopped successfully")

    def record_audio(self):
        """
        Continuously capture audio from microphone and push frames to audio_queue.

        This method runs in a separate thread and captures audio from the microphone,
        then adds the audio frames to the audio queue for processing.
        """
        self.logger.info("Starting audio recording thread")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=self.audio_callback,
            ):
                while not self.audio_stop_event.is_set():
                    sd.sleep(100)
        except Exception as e:
            self.logger.error(f"Error in audio recording thread: {e}")
            self.logger.error(traceback.format_exc())

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for the audio input stream.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status information
        """
        if status:
            self.logger.warning(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """
        Process audio chunks from the audio queue.

        This method pulls small audio chunks from the audio queue, accumulates speech segments
        based on an adaptive threshold, and once silence or a max utterance duration is detected,
        sends the segment to the ASR model for transcription.
        """
        self.logger.info("Starting audio processing thread")

        frames_per_chunk = int(self.sample_rate * self.audio_chunk_duration)
        max_frames = int(self.sample_rate * self.max_utterance_duration)

        current_segment = []  # holds numpy arrays for current speech
        silence_duration = 0.0  # accumulated silence duration

        try:
            while not self.audio_stop_event.is_set():
                frames = []
                frames_collected = 0

                # Collect frames for the current chunk
                while (
                    frames_collected < frames_per_chunk
                    and not self.audio_stop_event.is_set()
                ):
                    try:
                        data = self.audio_queue.get(timeout=0.1)
                        frames.append(data)
                        frames_collected += data.shape[0]
                    except queue.Empty:
                        continue

                if self.audio_stop_event.is_set():
                    break

                # Process the collected frames
                if frames:
                    audio_chunk = np.concatenate(frames, axis=0).flatten()
                    segment_pd = self.float32_to_pydub(audio_chunk, self.sample_rate)
                    dbfs_val = segment_pd.dBFS

                    # Compute effective threshold
                    effective_threshold = max(
                        self.default_threshold,
                        self.ambient_noise_level + self.adaptive_margin,
                    )

                    dbfs_val = dbfs_val.__round__(2)
                    effective_threshold = effective_threshold.__round__(2)
                    self.logger.debug(
                        f"Microphone dBFS: {dbfs_val} | Activation threshold: {effective_threshold}"
                    )

                    if dbfs_val > effective_threshold:
                        # Chunk is considered speech
                        current_segment.append(audio_chunk)
                        silence_duration = 0.0
                    else:
                        # Update ambient noise estimate when no speech is detected
                        self.ambient_noise_level = (self.ambient_alpha * dbfs_val) + (
                            (1 - self.ambient_alpha) * self.ambient_noise_level
                        )
                        silence_duration += self.audio_chunk_duration

                    # Process accumulated speech if silence or maximum length is reached
                    if current_segment:
                        total_frames = sum(len(chunk) for chunk in current_segment)
                        if (
                            silence_duration >= self.silence_duration_threshold
                            or total_frames >= max_frames
                        ):
                            utterance_audio = np.concatenate(current_segment)
                            # Calculate duration in seconds
                            duration = len(utterance_audio) / self.sample_rate
                            chunk_time_str = datetime.now().strftime("%H:%M:%S")

                            try:
                                if self.asr is not None:
                                    result = self.asr(utterance_audio)
                                    raw_text = result.get("text", "")
                                    cleaned_text = self.sanitize_english_text(
                                        raw_text
                                    ).strip()

                                    if cleaned_text:
                                        # Append duration to the transcript line
                                        transcript_line = f"[{chunk_time_str}   ~{duration:.2f}s] {cleaned_text}"
                                        self.logger.info(
                                            f"Transcript: {transcript_line}"
                                        )

                                        # Write to file
                                        with open(
                                            self.transcript_filename,
                                            "a",
                                            encoding="utf-8",
                                        ) as f:
                                            f.write(transcript_line + "\n")

                                        # Add to queue for GUI display
                                        self.transcription_queue.put(transcript_line)
                                    else:
                                        self.logger.debug(
                                            "Transcript: (empty after cleaning)"
                                        )
                                else:
                                    self.logger.warning(
                                        "ASR pipeline not available, skipping transcription"
                                    )
                            except Exception as e:
                                self.logger.error(f"Error during transcription: {e}")
                                self.logger.error(traceback.format_exc())

                            # Reset for next utterance
                            current_segment = []
                            silence_duration = 0.0
        except Exception as e:
            self.logger.error(f"Error in audio processing thread: {e}")
            self.logger.error(traceback.format_exc())

    def float32_to_pydub(self, float_array, sample_rate=16000):
        """
        Convert a float32 numpy array to a pydub AudioSegment.

        Args:
            float_array: Float32 numpy array of audio data
            sample_rate: Sample rate of the audio data

        Returns:
            pydub.AudioSegment: Audio segment for processing
        """
        float_array = np.clip(float_array, -1.0, 1.0)
        int16_array = (float_array * 32767).astype(np.int16)
        segment = AudioSegment(
            data=int16_array.tobytes(),
            sample_width=2,  # 16 bits = 2 bytes
            frame_rate=sample_rate,
            channels=1,
        )
        return segment

    def poll_transcriptions(self):
        """
        Move any completed utterances to the transcript box, then
        send the entire transcript through the NLP pipeline to
        refresh the Model Detections panel.
        """
        while not self.transcription_queue.empty():
            line = self.transcription_queue.get()
            self.gui.add_text_to_text_area1(line)
            self.context_data()
        if self.audio_running and not self.audio_stop_event.is_set():
            self.gui.root.after(200, self.poll_transcriptions)

    def sanitize_english_text(self, text):
        """
        Sanitize text by removing non-English characters.

        Args:
            text: Text to sanitize

        Returns:
            str: Sanitized text
        """
        return re.sub(r"[^A-Za-z0-9\s,.!?'\"]", "", text)

    def context_data(self):
        """
        Process the transcript text to extract injuries and tourniquet information.
        
        This method uses the context_comprehension module to analyze the transcript
        and extract relevant information about injuries and tourniquets.
        
        It updates the GUI with the extracted information and logs it for debugging.
        """
        transcript_text = self.gui.text_area1.get("1.0", tk.END).strip()
        extracted = process_transcript(transcript_text)
        injuries = extracted.get("injuries", [])
        tourniquets = extracted.get("tourniquets", [])

        output_lines = []

        if not injuries and not tourniquets:
            output_lines.append("AUDIO: No new detections found.")
        else:
            for idx, inj in enumerate(injuries, 1):
                inj_loc = inj["location_phrase"].replace("forearm", "arm")
                line = f"AUDIO: Injury #{idx} detected at {inj_loc}"
                if idx <= len(tourniquets):
                    tq = tourniquets[idx - 1]
                    tq_loc = tq["location_phrase"].replace("forearm", "arm")
                    status = (
                        "APPLIED"
                        if tq["status"] == "confirmed tourniquet placement"
                        else "SUGGESTED"
                    )
                    line += f" ; Tourniquet #{idx} {status} at {tq_loc}"
                output_lines.append(line)

        for line in output_lines:
            if line not in self.posted_audio_lines:  # new?
                self.gui.post_detection(line)  # show it
                self.posted_audio_lines.add(line)  # mark as seen
