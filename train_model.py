import os
import numpy as np
import joblib
from extract_features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import sys

def train_model_with_dataset():
    print("Starting training process with RAVDESS dataset...")
    
    # Define the main directory where the dataset is located
    dataset_dir = "Dataset/RAVDESS/"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found at {dataset_dir}")
        print("Please make sure the RAVDESS dataset is placed in the Dataset/RAVDESS/ directory.")
        return
    
    # Initialize an empty list for storing features and labels
    data = []
    actor_counts = {}
    emotion_counts = {}
    
    print("Extracting features from audio files...")
    start_time = time.time()
    
    # Loop through all the actors in RAVDESS
    for actor_folder in os.listdir(dataset_dir):
        actor_folder_path = os.path.join(dataset_dir, actor_folder)
        
        # Skip if not a directory or doesn't start with "Actor_"
        if not os.path.isdir(actor_folder_path) or not actor_folder.startswith("Actor_"):
            continue
            
        print(f"Processing {actor_folder}...")
        actor_file_count = 0
        
        # Loop through all the .wav files in the actor's folder
        for file_name in os.listdir(actor_folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(actor_folder_path, file_name)
                
                # Extract features and emotion
                features = extract_features(file_path)
                if features is not None:
                    # File format: 03-01-05-01-01-01-01.wav where 3rd component is emotion
                    try:
                        emotion = file_name.split('-')[2]  # Get emotion code
                        data.append([features, emotion])
                        
                        # Update counts
                        actor_counts[actor_folder] = actor_counts.get(actor_folder, 0) + 1
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        actor_file_count += 1
                    except IndexError:
                        print(f"Warning: File {file_name} does not follow the expected naming convention.")
        
        print(f"  - Processed {actor_file_count} files for {actor_folder}")
    
    # Check if we have enough data
    if len(data) == 0:
        print("Error: No valid audio files were processed. Please check the dataset.")
        return
    
    # Convert the data into an appropriate format (X: features, y: labels)
    X = np.array([item[0] for item in data])
    y = np.array([item[1] for item in data])
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Total samples loaded: {len(X)}")
    print("\nSamples per actor:")
    for actor, count in sorted(actor_counts.items()):
        print(f"  - {actor}: {count} files")
    
    emotion_dict = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    print("\nSamples per emotion:")
    for emotion, count in sorted(emotion_counts.items()):
        emotion_name = emotion_dict.get(emotion, emotion)
        print(f"  - {emotion_name} ({emotion}): {count} files")
    
    print(f"\nFeature extraction completed in {time.time() - start_time:.2f} seconds")
    
    # Now, proceed with train_test_split
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Create and train the model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, 
                                min_samples_leaf=1, max_features='sqrt', 
                                bootstrap=True, n_jobs=-1, random_state=42)
    
    train_start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - train_start_time
    print(f"Model training completed in {train_time:.2f} seconds")
    
    # Evaluate the model
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.2f}")
    
    # Print detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[emotion_dict.get(e, e) for e in sorted(set(y_test))]))
    
    # Create directory if it doesn't exist
    os.makedirs('Models', exist_ok=True)
    
    # Save the model
    model_path = 'Models/emotion_model.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Return the trained model and metrics
    return model, accuracy

def train_dummy_model():
    print("Creating a dummy model for demonstration purposes...")
    
    # Create dummy data for the model (40 MFCC features)
    X_train = np.random.rand(100, 40)
    
    # Labels: Emotion codes (01-08)
    emotions = ['01', '02', '03', '04', '05', '06', '07', '08']
    y_train = np.random.choice(emotions, size=100)
    
    # Create and train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create directory if it doesn't exist
    os.makedirs('Models', exist_ok=True)
    
    # Save the model
    model_path = 'Models/emotion_model.pkl'
    joblib.dump(model, model_path)
    print(f"Dummy model saved to {model_path}")
    print("Note: This is just a demonstration model and will give random predictions.")
    
    return model

if __name__ == "__main__":
    print("Voice Emotion Detection - Model Training")
    print("=======================================\n")
    
    try:
        # Try to train with the real dataset
        train_model_with_dataset()
    except Exception as e:
        print(f"\nError training model with dataset: {str(e)}")
        print("\nFalling back to dummy model...")
        train_dummy_model()