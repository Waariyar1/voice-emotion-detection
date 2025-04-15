import librosa
import numpy as np
import os

def extract_features(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File does not exist: {file_path}")
            return None
            
        # Load audio file with more robust settings
        audio, sample_rate = librosa.load(file_path, sr=None, mono=True, res_type='kaiser_fast')
        
        # Ensure audio has data
        if len(audio) == 0:
            print(f"Empty audio file: {file_path}")
            return None
            
        # Extract features
        # 1. MFCC (Mel Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
        # 2. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        spec_cent_processed = np.mean(spectral_centroids)
        
        # 3. Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_processed = np.mean(zero_crossing_rate)
        
        # 4. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_processed = np.mean(chroma.T, axis=0)
        
        # 5. Spectral Roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        rolloff_processed = np.mean(spectral_rolloff)
        
        # Combine all features
        features = np.hstack([mfccs_processed, 
                             spec_cent_processed, 
                             zcr_processed, 
                             chroma_processed,
                             rolloff_processed])
        
        return features
        
    except Exception as e:
        print(f"Error encountered while parsing file {file_path}: {str(e)}")
        return None

def get_sample_audio():
    """Returns path to a sample audio file from the dataset"""
    dataset_dir = "Dataset/RAVDESS/Actor_01"
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return None
        
    for file_name in os.listdir(dataset_dir):
        if file_name.endswith(".wav"):
            return os.path.join(dataset_dir, file_name)
    
    return None
