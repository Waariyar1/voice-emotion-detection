from extract_features import extract_features, get_sample_audio
import joblib
import numpy as np
import os
import random

def predict_emotion(file_path=None):
    try:
        # If no file path is provided, use a sample from the dataset
        if file_path is None or not os.path.exists(file_path):
            file_path = get_sample_audio()
            if file_path is None:
                return "Error: No audio file found. Please provide a valid file path or ensure the RAVDESS dataset is available."
            print(f"Using sample audio file: {file_path}")
        
        # Load saved model
        model_path = 'Models/emotion_model.pkl'
        if not os.path.exists(model_path):
            return "Error: Model file not found. Please train the model first."
        
        model = joblib.load(model_path)
        
        # Extract features
        features = extract_features(file_path)
        if features is None:
            return "Error: Could not extract features from the audio file."
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Predict emotion
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
        
        # Get probabilities for each emotion
        emotion_probs = model.predict_proba(features)[0]
        predicted_emotion = model.predict(features)[0]
        emotion_name = emotion_dict.get(predicted_emotion, predicted_emotion)
        
        # Sort probabilities to find top emotions
        top_indices = np.argsort(emotion_probs)[::-1]
        class_labels = model.classes_
        
        # Format the result with top 3 emotions and their probabilities
        result = f"Primary Emotion: {emotion_name}\n\n"
        result += "Top Three Emotions:\n"
        
        for i in range(min(3, len(top_indices))):
            idx = top_indices[i]
            emotion_code = class_labels[idx]
            emotion_name = emotion_dict.get(emotion_code, emotion_code)
            probability = emotion_probs[idx]
            result += f"{i+1}. {emotion_name}: {probability:.2f}\n"
        
        # For file from RAVDESS dataset, get the true emotion from filename
        if "Actor_" in file_path:
            # File format: 03-01-05-01-01-01-01.wav where 3rd component is emotion
            try:
                filename = os.path.basename(file_path)
                emotion_code = filename.split('-')[2]
                result += f"\nActual Emotion: {emotion_dict.get(emotion_code, 'unknown')}"
            except:
                pass
        
        return result
    
    except Exception as e:
        return f"Error: {str(e)}"

def get_emotion_sample(emotion_code):
    """Returns a sample audio file with the specified emotion code"""
    dataset_dir = "Dataset/RAVDESS"
    
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory not found: {dataset_dir}")
        return None
    
    # Search in all actor folders
    for actor_folder in os.listdir(dataset_dir):
        actor_path = os.path.join(dataset_dir, actor_folder)
        if os.path.isdir(actor_path):
            for file_name in os.listdir(actor_path):
                if file_name.endswith(".wav") and f"-{emotion_code}-" in file_name:
                    return os.path.join(actor_path, file_name)
    
    return None

if __name__ == "__main__":
    print("Welcome to Emotion Predictor!")
    print("1. Enter a file path")
    print("2. Use a sample from dataset")
    print("3. Test with specific emotion")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        file_path = input("Enter the path to your audio file: ")
        result = predict_emotion(file_path)
    elif choice == "2":
        result = predict_emotion(None)  # Use sample
    elif choice == "3":
        emotions = {
            "1": "01", # neutral
            "2": "02", # calm
            "3": "03", # happy
            "4": "04", # sad
            "5": "05", # angry
            "6": "06", # fearful
            "7": "07", # disgust
            "8": "08"  # surprised
        }
        print("Select emotion:")
        for key, value in emotions.items():
            print(f"{key}. {value}")
        
        emotion_choice = input("Enter emotion (1-8): ")
        if emotion_choice in emotions:
            sample_file = get_emotion_sample(emotions[emotion_choice])
            if sample_file:
                result = predict_emotion(sample_file)
            else:
                result = "No sample found for the selected emotion."
        else:
            result = "Invalid emotion selection."
    else:
        result = "Invalid choice."
    
    print(result)
