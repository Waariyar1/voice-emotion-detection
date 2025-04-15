import numpy as np
import scipy.io.wavfile as wav

# Create a simple sine wave audio for testing
sample_rate = 44100  # Standard sample rate
duration = 3  # 3 seconds
t = np.linspace(0, duration, int(sample_rate * duration))
frequency = 440  # A4 note
audio_data = np.sin(2 * np.pi * frequency * t) * 0.5  # Amplitude 0.5 to avoid clipping

# Convert to 16-bit PCM
audio_data_16bit = (audio_data * 32767).astype(np.int16)

# Save the audio file
test_file = "test_audio.wav"
wav.write(test_file, sample_rate, audio_data_16bit)

print(f"Test audio file '{test_file}' created for demonstration purposes.") 