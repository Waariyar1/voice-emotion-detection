import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pyaudio
import wave
import os
import threading
import time
from predict_emotion import predict_emotion, get_emotion_sample
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

class EmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Voice Emotion Detector")
        self.root.geometry("600x600")
        self.root.configure(bg="#f0f0f0")
        
        self.is_recording = False
        self.frames = []
        self.sample_rate = 44100
        self.channels = 1
        self.chunk = 1024
        self.record_seconds = 5
        self.temp_audio_file = "temp_recording.wav"
        self.selected_file = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Voice Emotion Detector", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)
        
        # Tabs
        tab_control = ttk.Notebook(main_frame)
        
        # Recording tab
        record_tab = ttk.Frame(tab_control)
        tab_control.add(record_tab, text="Record Audio")
        
        # File selection tab
        file_tab = ttk.Frame(tab_control)
        tab_control.add(file_tab, text="Select File")
        
        # Dataset tab
        dataset_tab = ttk.Frame(tab_control)
        tab_control.add(dataset_tab, text="Use Dataset")
        
        tab_control.pack(expand=1, fill="both")
        
        # Setup Record Tab
        self.setup_record_tab(record_tab)
        
        # Setup File Tab
        self.setup_file_tab(file_tab)
        
        # Setup Dataset Tab
        self.setup_dataset_tab(dataset_tab)
        
        # Result frame - common to all tabs
        result_frame = ttk.LabelFrame(main_frame, text="Result", padding=10)
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_var = tk.StringVar(value="No analysis performed yet")
        self.result_label = ttk.Label(result_frame, textvariable=self.result_var, font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)
        
    def setup_record_tab(self, parent):
        # Instructions
        instructions = ttk.Label(parent, text="Click the Record button and speak for 5 seconds")
        instructions.pack(pady=10)
        
        # Record button
        self.record_button = ttk.Button(parent, text="Start Recording (5 seconds)", command=self.toggle_recording)
        self.record_button.pack(pady=10)
        
        # Progress bar
        self.progress_frame = ttk.Frame(parent)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready...")
        self.status_label = ttk.Label(parent, textvariable=self.status_var, font=("Arial", 10))
        self.status_label.pack(pady=10)
        
        # Analyze button
        analyze_button = ttk.Button(parent, text="Analyze Recorded Audio", command=self.analyze_recorded_audio)
        analyze_button.pack(pady=10)
        
    def setup_file_tab(self, parent):
        # Instructions
        instructions = ttk.Label(parent, text="Select an audio file from your computer")
        instructions.pack(pady=10)
        
        # File path display
        file_frame = ttk.Frame(parent)
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_path_var = tk.StringVar(value="No file selected")
        file_path_label = ttk.Label(file_frame, textvariable=self.file_path_var, width=50)
        file_path_label.pack(side=tk.LEFT, padx=5)
        
        browse_button = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_button.pack(side=tk.RIGHT)
        
        # Analyze button
        analyze_file_button = ttk.Button(parent, text="Analyze Selected File", command=self.analyze_selected_file)
        analyze_file_button.pack(pady=10)
        
    def setup_dataset_tab(self, parent):
        # Instructions
        instructions = ttk.Label(parent, text="Select an emotion to test with RAVDESS dataset samples")
        instructions.pack(pady=10)
        
        # Emotion selection
        emotion_frame = ttk.Frame(parent)
        emotion_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(emotion_frame, text="Emotion:").pack(side=tk.LEFT, padx=5)
        
        self.emotion_var = tk.StringVar()
        emotions = [
            ("Neutral", "01"),
            ("Calm", "02"),
            ("Happy", "03"),
            ("Sad", "04"),
            ("Angry", "05"),
            ("Fearful", "06"),
            ("Disgust", "07"),
            ("Surprised", "08")
        ]
        
        emotion_combobox = ttk.Combobox(emotion_frame, textvariable=self.emotion_var, values=[e[0] for e in emotions], state="readonly")
        emotion_combobox.pack(side=tk.LEFT, padx=5)
        emotion_combobox.current(0)
        
        # Store emotion codes for later use
        self.emotion_codes = {e[0]: e[1] for e in emotions}
        
        # File info display
        self.dataset_file_var = tk.StringVar(value="No sample selected yet")
        dataset_file_label = ttk.Label(parent, textvariable=self.dataset_file_var)
        dataset_file_label.pack(pady=10)
        
        # Analyze button
        analyze_dataset_button = ttk.Button(parent, text="Use Dataset Sample", command=self.use_dataset_sample)
        analyze_dataset_button.pack(pady=10)
        
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="Stop Recording")
        self.status_var.set("Recording in progress...")
        self.result_var.set("Recording in progress...")
        self.frames = []
        
        # Start recording in a separate thread
        self.recording_thread = threading.Thread(target=self.record_audio)
        self.recording_thread.start()
        
        # Start progress bar
        self.update_progress_bar()
    
    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.config(text="Start Recording (5 seconds)")
            self.status_var.set("Recording finished")
    
    def record_audio(self):
        try:
            def audio_callback(indata, frames, time, status):
                if self.is_recording:
                    self.frames.append(indata.copy())
            
            with sd.InputStream(callback=audio_callback, channels=self.channels, samplerate=self.sample_rate):
                start_time = time.time()
                while self.is_recording and (time.time() - start_time) < self.record_seconds:
                    time.sleep(0.1)
                
                if self.is_recording:  # If we reached the time limit
                    self.stop_recording()
            
            # Save recording
            if self.frames:
                self.save_audio()
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            self.is_recording = False
            self.record_button.config(text="Start Recording (5 seconds)")
    
    def save_audio(self):
        if not self.frames:
            return
        
        # Convert frames to numpy array
        audio_data = np.concatenate(self.frames, axis=0)
        
        # Save as WAV file
        try:
            wav.write(self.temp_audio_file, self.sample_rate, audio_data)
            self.status_var.set("Audio saved")
        except Exception as e:
            self.status_var.set(f"Error saving audio: {str(e)}")
    
    def analyze_recorded_audio(self):
        if not os.path.exists(self.temp_audio_file):
            messagebox.showerror("Error", "No recorded audio found. Please record audio first.")
            return
        
        self.status_var.set("Analyzing recorded audio...")
        try:
            # Analyze the emotion
            result = predict_emotion(self.temp_audio_file)
            self.result_var.set(result)
            self.status_var.set("Analysis complete")
        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            self.result_var.set(error_msg)
            self.status_var.set("Error in analysis")
            messagebox.showerror("Analysis Error", error_msg)
    
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if file_path:
            self.selected_file = file_path
            self.file_path_var.set(os.path.basename(file_path))
    
    def analyze_selected_file(self):
        if not self.selected_file:
            messagebox.showerror("Error", "No file selected. Please select an audio file first.")
            return
            
        if not os.path.exists(self.selected_file):
            messagebox.showerror("Error", f"File not found: {self.selected_file}")
            return
            
        try:
            # Analyze the emotion
            result = predict_emotion(self.selected_file)
            self.result_var.set(result)
        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            self.result_var.set(error_msg)
            messagebox.showerror("Analysis Error", error_msg)
    
    def use_dataset_sample(self):
        selected_emotion = self.emotion_var.get()
        if not selected_emotion:
            messagebox.showerror("Error", "No emotion selected")
            return
            
        emotion_code = self.emotion_codes.get(selected_emotion)
        if not emotion_code:
            messagebox.showerror("Error", "Invalid emotion selection")
            return
            
        # Get sample file for the selected emotion
        sample_file = get_emotion_sample(emotion_code)
        if not sample_file:
            messagebox.showerror("Error", f"No sample found for emotion: {selected_emotion}")
            return
            
        self.dataset_file_var.set(f"Using sample: {os.path.basename(sample_file)}")
        
        try:
            # Analyze the emotion
            result = predict_emotion(sample_file)
            self.result_var.set(result)
        except Exception as e:
            error_msg = f"Error in analysis: {str(e)}"
            self.result_var.set(error_msg)
            messagebox.showerror("Analysis Error", error_msg)
    
    def update_progress_bar(self):
        if self.is_recording:
            elapsed_time = len(self.frames) * (self.chunk / self.sample_rate)
            progress = min(100, (elapsed_time / self.record_seconds) * 100)
            self.progress_var.set(progress)
            self.root.after(100, self.update_progress_bar)
        else:
            self.progress_var.set(100 if self.frames else 0)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root)
    root.mainloop() 